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
import tensorflow as tf
import random
import subprocess
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
from mkv2clips import main as m2c_main, EVENT_WINDOWS, sanitize_filename


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
        # Batch process clips
        batch_size = 5  # Process 5 clips at a time
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(clips) + batch_size - 1)//batch_size}")
            for clip in batch:
                if not Path(clip["clip_path"]).exists():
                    clip["commentary"] = f"Exciting {clip['label']} at {clip['gameTime']}!"
                    continue
                meta = {
                    "label": clip["label"],
                    "gameTime": clip["gameTime"],
                    "team": clip.get("team", ""),
                }
                res = commentator.process_clip(clip["clip_path"], meta)
                clip["commentary"] = res.get("commentary", "") if res["success"] else (
                    f"Exciting {clip['label']} at {clip['gameTime']}!"
                )
    else:
        logger.info("Using fallback commentary")
        templates = {
            "Goal": [
                "{team} scores! What a fantastic finish at {timestamp}!",
                "GOOOAAAL for {team}! The crowd goes wild as they take the lead!",
                "A brilliant strike by {team} at {timestamp}! That's a goal that will be remembered!"
            ],
            "Shots off target": [
                "Close attempt by {team}! The shot goes just wide of the post.",
                "{team} with a powerful strike, but it's off target. Good effort at {timestamp}.",
                "Nearly a goal for {team}! The keeper looked worried as that shot flew past the post."
            ],
            "Shots on target": [
                "Great shot by {team}! The goalkeeper makes a fantastic save.",
                "{team} tests the keeper with a powerful shot at {timestamp}.",
                "Brilliant attempt from {team}! The goalkeeper had to be at his best to keep that out."
            ],
            "Yellow card": [
                "Yellow card shown to the {team} player after that challenge.",
                "The referee reaches for his pocket! {team} player receives a caution at {timestamp}.",
                "That's a booking for {team}. The referee had no choice after that foul."
            ]
        }
        for clip in clips:
            tpl = templates.get(clip["label"], [
                                "Great play by {team} at {gameTime}!"])
            clip["commentary"] = random.choice(tpl).format(**clip)
    return clips


def generate_speech(clips, cfg):
    logger.info("Step 3.5: TTS generation (simulated)")
    output_dir = cfg.get("output_dir", "outputs")
    audio_dir = Path(output_dir) / "audio"
    for i, clip in enumerate(clips, start=1):
        # If commentary is missing, generate a basic one
        if "commentary" not in clip:
            clip["commentary"] = f"Exciting {clip['label']} at {clip['gameTime']}!"
        txt = clip["commentary"]
        out = audio_dir / f"commentary_{i}.txt"
        out.write_text(txt)
        clip["commentary_audio"] = str(out)
    return clips


def compile_highlights(clips, cfg):
    logger.info("Step 4: Highlight compilation")
    output_dir = cfg.get("output_dir", "outputs")
    final_dir = Path(output_dir) / "final"
    ts = int(time.time())
    manifest = final_dir / f"highlights_{ts}_manifest.json"
    
    # Ensure clips are sorted by timestamp
    clips.sort(key=lambda x: x["timestamp"])
    
    # Write manifest
    with open(manifest, "w") as f:
        json.dump(clips, f, indent=2)
    logger.info(f"Wrote manifest → {manifest}")

    # Create a temporary file listing all clips
    concat_file = final_dir / f"concat_{ts}.txt"
    with open(concat_file, "w") as f:
        for clip in clips:
            clip_path = Path(clip["clip_path"])
            if clip_path.exists():
                # Use absolute path for the video file
                f.write(f"file '{clip_path.absolute()}'\n")
            else:
                logger.warning(f"Clip file not found: {clip_path}")
    
    # Compile final video using ffmpeg
    final_video = final_dir / f"highlights_{ts}.mp4"
    cmd = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(final_video)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully compiled highlights → {final_video}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compile highlights: {e}")
        return None
    
    # Clean up temporary file
    concat_file.unlink()
    
    return str(final_video)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--video",   required=True, help="path to .mkv")
    p.add_argument("--output",  help="override output_dir")
    p.add_argument("--skip-processing", action="store_true", help="skip video processing if clips already exist")
    p.add_argument("--skip-commentary", action="store_true", help="skip commentary generation if already done")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.output:
        cfg["output_dir"] = args.output

    out_dir = create_output_dirs(cfg)
    clips_dir = out_dir / "clips"

    # ── Steps 1+2: Event detection & clip extraction ──
    if not args.skip_processing:
        logger.info("Step 1+2: Running mkv2clips")
        events_json = clips_dir / "events.json"
        m2c_main(args.video, str(clips_dir))
        logger.info(f"Clips + JSON written to {clips_dir}")
    else:
        logger.info("Skipping video processing - using existing clips")
        events_json = clips_dir / "events.json"

    # now read back the JSON to build our list of clip-metadata dicts
    data = json.loads(events_json.read_text())
    evs = data.get("events", [])

    extracted_clips = []
    for ev in evs:
        fname = sanitize_filename(ev["label"], ev["gameTime"])
        clipp = clips_dir / fname
        ev["clip_path"] = str(clipp)
        # compute duration from your EVENT_WINDOWS
        start, end = EVENT_WINDOWS.get(ev["label"], (0, 5))
        ev["duration"] = end - start
        extracted_clips.append(ev)
    
    # Sort clips by timestamp to ensure chronological order
    extracted_clips.sort(key=lambda x: x["timestamp"])
    logger.info(f"Loaded {len(extracted_clips)} clipped events")

    # ── Step 3: Commentary ──
    if not args.skip_commentary:
        extracted_clips = generate_commentary(extracted_clips, cfg)
    else:
        logger.info("Skipping commentary generation - using fallback commentary")
        # Add basic commentary for each clip
        for clip in extracted_clips:
            clip["commentary"] = f"Exciting {clip['label']} at {clip['gameTime']}!"

    # ── Step 3.5: TTS ──
    extracted_clips = generate_speech(extracted_clips, cfg)

    # ── Step 4: Compile ──
    final_video = compile_highlights(extracted_clips, cfg)

    logger.info("Pipeline complete!")
    print(f"\nFinal highlight video would be: {final_video}")


if __name__ == "__main__":
    main()
