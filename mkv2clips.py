"""
mkdir -p clips_output

python3 mkv2clips.py \
  "/Users/adityadaga/Downloads/COMP646PROJECT/HiLitR/data/raw/mkvfiles/france_ligue-1/2014-2015/2015-04-05 - 22-00 Marseille 2 - 3 Paris SG/1_224p.mkv" \
  clips_output

"""


#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

# ─── per-event clip windows (seconds before, seconds after) ───────────
EVENT_WINDOWS = {
    "Kick-off":           (0,   10),
    "Goal":              (-10,  10),
    "Substitution":      (-15,   8),
    "Offside":           ( -8,   5),
    "Shot on target":    ( -8,   5),
    "Shot off target":   ( -8,   5),
    "Clearance":         ( -5,   5),
    "Ball out of play":  ( -5,   5),
    "Throw-in":          ( -5,  10),
    "Foul":              (-10,  10),
    "Indirect free-kick":(-5,   12),
    "Direct free-kick":  (-5,   12),
    "Corner":            ( -3,  12),
    "Yellow card":       (-10,   7),
    "Red card":          (-15,  10),
    "Yellow->red card":  (-15,  10),
}

def parse_game_time(game_time_str: str) -> float:
    """
    "1 - 14:40" → total seconds (14*60 + 40), ignores half.
    """
    _, t = game_time_str.split(" - ")
    m, s = map(int, t.split(":"))
    return m * 60 + s

def sanitize_filename(label: str, game_time_str: str) -> str:
    fn = label.replace(" ", "_")
    ts = game_time_str.replace(":", "m").replace(" ", "").replace("-", "_")
    return f"{fn}_{ts}.mp4"

def extract_clip(
    event_time: float,
    start_off: float,
    end_off: float,
    label: str,
    game_time_str: str,
    src_video: str,
    dst_dir: str
):
    """
    Runs: ffmpeg -ss (t+start_off) -i src_video -t (end_off-start_off) -c copy dst_path
    """
    clip_start = max(event_time + start_off, 0.0)
    duration   = end_off - start_off
    fname      = sanitize_filename(label, game_time_str)
    dst        = os.path.join(dst_dir, fname)

    cmd = [
      "ffmpeg",
      "-hide_banner", "-loglevel","error",
      "-ss", str(clip_start),
      "-i", src_video,
      "-t", str(duration),
      "-c", "copy",
      "-y",  # overwrite
      dst
    ]
    print(f"Extracting → {label}@{game_time_str}  ({clip_start:.1f}s→+{duration:.1f}s)")
    subprocess.run(cmd, check=True)

def main(input_mkv: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1) generate events.json via mkv2json.py
    events_json = os.path.join(output_dir, "events.json")
    print("⏳ running mkv2json…")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mkv2json_path = os.path.join(script_dir, "mkv2json.py")
    try:
        result = subprocess.run(
            ["python3", mkv2json_path, input_mkv, events_json],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running mkv2json.py:")
        print(e.stdout)
        print(e.stderr)
        raise

    # 2) load your events
    with open(events_json, "r") as f:
        data = json.load(f)
    events = data.get("events", [])

    # 3) clip each event
    for ev in events:
        label = ev["label"]
        if label not in EVENT_WINDOWS:
            # skip anything we don't have a window for
            continue
        start_off, end_off = EVENT_WINDOWS[label]
        t = parse_game_time(ev["gameTime"])
        extract_clip(
            event_time    = t,
            start_off     = start_off,
            end_off       = end_off,
            label         = label,
            game_time_str = ev["gameTime"],
            src_video     = input_mkv,
            dst_dir       = output_dir
        )

    print(f"✅ Done!  Clips in → {output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
      description="MKV → events.json + per-event clips"
    )
    p.add_argument("input_mkv", help="path to your .mkv file")
    p.add_argument("output_dir", help="directory to write JSON + clips")
    args = p.parse_args()
    main(args.input_mkv, args.output_dir)
