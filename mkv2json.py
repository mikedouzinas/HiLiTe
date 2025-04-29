#!/usr/bin/env python3
"""

terminal command:
python mkv2json.py \
  "/Users/adityadaga/Downloads/COMP646PROJECT/HiLitR/data/raw/mkvfiles/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv" \
  output/events.json

"""
import os
import sys
import argparse
import json
import shutil
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet import ResNet152, preprocess_input
from huggingface_hub import snapshot_download


def downscale(input_path, output_path):
    # uses ffmpeg under the hood
    cmd = (
        f'ffmpeg -y -i "{input_path}" '
        '-vf scale=-1:224,fps=2 -an '
        f'"{output_path}"'
    )
    if os.system(cmd) != 0:
        raise RuntimeError("ffmpeg failed")


def extract_raw_features(video_path, out_npy, fps=1, batch_size=64):
    resnet = ResNet152(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 1
    step = max(1, int(orig_fps / fps))

    frames = []
    idx = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        if idx % step == 0:
            h, w = frm.shape[:2]
            # center-crop
            if h > w:
                frm = frm[(h-w)//2:(h+w)//2, :]
            else:
                frm = frm[:, (w-h)//2:(w+h)//2]
            frm = cv2.resize(frm, (224, 224))[:, :, ::-1]
            frames.append(frm)
        idx += 1
    cap.release()

    X = preprocess_input(np.array(frames, dtype=np.float32))
    feats = []
    for i in range(0, len(X), batch_size):
        feats.append(resnet.predict(X[i:i+batch_size], verbose=0))
    feats = np.vstack(feats)
    np.save(out_npy, feats)
    print(f"✅ Saved raw features {feats.shape} → {out_npy}")


def download_model(repo_id, model_dir_name, local_root):
    allow = [
        f"models/{model_dir_name}/**",
        "models/resnet_normalizer.pkl",
    ]
    hf_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=local_root,
        allow_patterns=allow
    )
    # move variables shards into subfolder
    best = os.path.join(local_root, "models", model_dir_name, "best_model")
    var = os.path.join(best, "variables")
    os.makedirs(var, exist_ok=True)
    for fn in os.listdir(best):
        if fn.startswith("variables.") and fn.endswith(("index","of-00001")):
            shutil.move(os.path.join(best, fn), os.path.join(var, fn))
    return best


def sliding_window_inference(
    raw_npy, model_dir, out_json,
    window=224, half_window=None,
    batch_w=16, threshold=0.5, peak_window=5
):
    half_window = half_window or (window // 2)
    feats = np.load(raw_npy)           # (T,2048)
    T, F = feats.shape
    pad = np.pad(feats, ((half_window, half_window),(0,0)), mode="edge")

    # load normalizer
    with open(os.path.join(model_dir, "..", "..", "resnet_normalizer.pkl"), "rb") as f:
        scaler = pickle.load(f)

    sm = tf.saved_model.load(model_dir)
    infer = sm.signatures["serving_default"]
    in_key = list(infer.structured_input_signature[1].keys())[0]
    out_key = list(infer.structured_outputs.keys())[0]

    preds = []
    for st in range(0, T, batch_w):
        en = min(st+batch_w, T)
        B  = en - st
        batch = np.zeros((B, window, F, 1), dtype=np.float32)
        for i, t in enumerate(range(st, en)):
            batch[i, :, :, 0] = pad[t:t+window]
        flat = batch[...,0].reshape(-1, F)
        flatn = scaler.transform(flat)
        batch[...,0] = flatn.reshape(B, window, F)

        out = infer(**{in_key: tf.constant(batch)})
        log = out[out_key].numpy()[..., 0]  # (B, W, 17)
        preds.append(log[:, half_window, :])
    all_preds = np.vstack(preds)         # (T,17)
    print("✅ predictions shape:", all_preds.shape)

    # peak-pick
    labels = [
        "Penalty","Kick-off","Goal","Substitution","Offside",
        "Shot on target","Shot off target","Clearance","Ball out of play",
        "Throw-in","Foul","Indirect free-kick","Direct free-kick","Corner",
        "Yellow card","Red card","Yellow->red card"
    ]
    events = []
    for ci, lab in enumerate(labels):
        sc = all_preds[:, ci]
        for t, s in enumerate(sc):
            if s < threshold:
                continue
            lo, hi = max(0, t-peak_window), min(T, t+peak_window+1)
            if s >= sc[lo:hi].max():
                sec = float(t)  # 1 FPS → t=seconds
                half = 1 if sec < 45*60 else 2
                rel = sec if half==1 else sec - 45*60
                m, s2 = divmod(int(rel), 60)
                events.append({
                    "label":     lab,
                    "half":      half,
                    "gameTime":  f"{half} - {m:02d}:{s2:02d}",
                    "timestamp": round(sec, 1)
                })

    with open(out_json, "w") as f:
        json.dump({"events": events}, f, indent=2)
    print(f"✅ Wrote {len(events)} events → {out_json}")


def main():
    p = argparse.ArgumentParser(
        description="MKV → events.json via SoccerNet + HF model"
    )
    p.add_argument("input_mkv", help="Path to input .mkv")
    p.add_argument("output_json", help="Path to output events.json")
    args = p.parse_args()

    # prep paths
    temp_mp4 = "temp_224p.mp4"
    raw_npy  = "output/raw_features.npy"
    model_dir_name = (
        "spotting_challenge_validated_resnet_normalized_confidence_"
        "zoo_lr5e-4_dwd2e-4_sr0.02_mu0.0"
    )

    os.makedirs("output", exist_ok=True)
    os.makedirs("pretrained_model", exist_ok=True)

    print("1) Downscaling video…")
    downscale(args.input_mkv, temp_mp4)

    print("2) Extracting raw ResNet-152 features…")
    extract_raw_features(temp_mp4, raw_npy)

    print("3) Downloading HF model & normalizer…")
    best_dir = download_model(
        repo_id="yahoo-inc/spivak-action-spotting-soccernet",
        model_dir_name=model_dir_name,
        local_root="pretrained_model"
    )

    print("4) Running sliding-window inference + JSON export…")
    sliding_window_inference(raw_npy, best_dir, args.output_json)


if __name__ == "__main__":
    main()
