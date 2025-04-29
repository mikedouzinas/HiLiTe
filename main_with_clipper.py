#!/usr/bin/env python3
"""
HiLiTr - Soccer Highlight Generator
Main pipeline script to connect all components, now using mkv2clips
for both event detection and clip extraction.
"""

import os
import sys
import argparse
import yaml
import json
import logging
import time
from pathlib import Path

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("hilitr.log")],
)
logger = logging.getLogger(__name__)

# commentary generator (optional)
try:
    from src.caption.gpt_commentator import GPTCommentator
    COMMENTATOR_AVAILABLE = True
except ImportError:
    logger.warning("GPT Commentator unavailable – will use fallback.")
    COMMENTATOR_AVAILABLE = False

# import your mkv2clips module
import mkv2clips as m2c 

def load_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg


def create_output_dirs(cfg):
    out = Path(cfg.get("output_dir", "outputs"))
    for sub in ("clips", "audio", "final"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory {(out/sub)}")
    return out


def generate_commentary(clips, cfg):
    logger.info("Step 3: Commentary generation")
    method = cfg.get("commentary", {}).get("method", "gpt")
    if method == "gpt" and COMMENTATOR_AVAILABLE:
        commentator = GPTCommentator(cfg["commentary"]["gpt"])
        for clip in clips:
            if not Path(clip["clip_path"]).exists():
                clip["commentary"] = (
                    f"Exciting {clip['event_type']} at {clip['gameTime']}!"
                )
                continue
            meta = {
                "label": clip["event_type"],
                "gameTime": clip["gameTime"],
                "team":      clip.get("team", ""),
            }
            res = commentator.process_clip(clip["clip_path"], meta)
            clip["commentary"] = res.get("commentary", "") if res["success"] else (
                f"Exciting {clip['event_type']} at {clip['gameTime']}!"
            )
    else:
        logger.info("Using fallback commentary")
        import random
        templates = {
            "Goal": ["{team} scores at {gameTime}!"],
            # … add more …
        }
        for clip in clips:
            tpl = templates.get(clip["event_type"], ["Great play by {team} at {gameTime}!"])
            clip["commentary"] = random.choice(tpl).format(**clip)
    return clips


def generate_speech(clips, cfg):
    logger.info("Step 3.5: TTS generation (simulated)")
    audio_dir = Path(cfg["output_dir"]) / "audio"
    for i, clip in enumerate(clips, start=1):
        txt = clip["commentary"]
        out = audio_dir / f"commentary_{i}.txt"
        out.write_text(txt)
        clip["commentary_audio"] = str(out)
    return clips


def compile_highlights(clips, cfg):
    logger.info("Step 4: Highlight compilation (simulated)")
    final_dir = Path(cfg["output_dir"]) / "final"
    ts = int(time.time())
    manifest = final_dir / f"highlights_{ts}_manifest.json"
    # just write out a debug manifest
    with open(manifest, "w") as f:
        json.dump(clips, f, indent=2)
    logger.info(f"Wrote manifest → {manifest}")
    return str(final_dir / f"highlights_{ts}.mp4")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--video",   required=True, help="path to .mkv")
    p.add_argument("--output",  help="override output_dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.output:
        cfg["output_dir"] = args.output

    out_dir = create_output_dirs(cfg)
    clips_dir = out_dir / "clips"

    # ── Steps 1+2: Event detection & clip extraction ──
    logger.info("Step 1+2: Running mkv2clips")
    events_json = clips_dir / "events.json"
    # mkv2clips.main will:
    #  • extract raw features & detect events
    #  • write events.json into clips_dir
    #  • ffmpeg-cut each event into clips_dir/*.mp4
    m2c.main(str(events_json), args.video, str(clips_dir))
    logger.info(f"Clips + JSON written to {clips_dir}")

    # now read back the JSON to build our list of clip-metadata dicts
    data = json.loads(events_json.read_text())
    evs  = data.get("events", [])

    extracted_clips = []
    for ev in evs:
        fname = m2c.sanitize_filename(ev["label"], ev["gameTime"])
        clipp = clips_dir / fname
        ev["clip_path"] = str(clipp)
        # compute duration from your EVENT_WINDOWS
        start, end = m2c.EVENT_WINDOWS.get(ev["label"], (0,5))
        ev["duration"] = end - start
        extracted_clips.append(ev)
    logger.info(f"Loaded {len(extracted_clips)} clipped events")

    # ── Step 3: Commentary ──
    extracted_clips = generate_commentary(extracted_clips, cfg)

    # ── Step 3.5: TTS ──
    extracted_clips = generate_speech(extracted_clips, cfg)

    # ── Step 4: Compile ──
    final_video = compile_highlights(extracted_clips, cfg)

    logger.info("Pipeline complete!")
    print(f"\nFinal highlight video would be: {final_video}")


if __name__ == "__main__":
    main()
