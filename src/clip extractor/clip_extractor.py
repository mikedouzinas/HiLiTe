import json
import os
import subprocess
from pathlib import Path

# === CONFIGURATION === #
VIDEO_PATH = "/mnt/c/Users/saahi/HiLitR/data/raw/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1_224p.mkv"
JSON_PATH = "/mnt/c/Users/saahi/HiLitR/data/raw/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/new_labels_for_clip_extractor.json"
OUTPUT_DIR = "/mnt/c/Users/saahi/HiLitR/clips"
FPS = 25  # Used only if you ever need to compute frame-based timestamps

# Time windows per event label (in seconds)
EVENT_WINDOWS = {
    "Kick-off":        (0, 10),
    "Red card":       (-15, 10),
    "Yellow card":     (-10, 7),
    "Goal":           (-10, 10),
    "Shots on target": (-8, 5),
    "Shots off target": (-8, 5),
    "Indirect free-kick": (-5, 12),
    "Direct free-kick":   (-5, 12),
    "Corner":           (-3, 12)
}

# === HELPER FUNCTIONS === #

def parse_game_time(game_time_str):
    # Example: "1 - 14:40" => half = 1, time = 14*60 + 40 = 880
    half, time_str = game_time_str.split(" - ")
    minutes, seconds = map(int, time_str.strip().split(":"))
    base_seconds = minutes * 60 + seconds
    return base_seconds  # ignore half for now (can be used for naming)

def sanitize_filename(label, game_time_str):
    return label.replace(" ", "_") + "_" + game_time_str.replace(":", "m").replace(" ", "").replace("-", "_") + ".mp4"

def extract_clip(event_time, start_offset, end_offset, label, game_time_str, video_path, output_dir):
    clip_start = max(event_time + start_offset, 0)
    clip_duration = end_offset - start_offset
    output_filename = sanitize_filename(label, game_time_str)
    output_path = os.path.join(output_dir, output_filename)

    cmd = [
        "ffmpeg",
        "-ss", str(clip_start),
        "-i", video_path,
        "-t", str(clip_duration),
        "-c", "copy",
        "-y",  # overwrite if exists
        output_path
    ]

    print(f"→ Extracting {label} at {game_time_str} → {output_path}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


# === MAIN SCRIPT === #

def main(json_path, video_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    events = data.get("annotations", [])

    kickoff_seen = False

    for event in events:
        label = event.get("label")
        visibility = event.get("visibility", "")
        game_time = event.get("gameTime", "")

        # Only visible events
        if visibility != "visible":
            continue

        # Special case: only the first kick-off
        if label == "Kick-off":
            if not game_time.endswith("00:00") or kickoff_seen:
                continue
            kickoff_seen = True

        if label not in EVENT_WINDOWS:
            continue

        start_offset, end_offset = EVENT_WINDOWS[label]
        event_time = parse_game_time(game_time)
        extract_clip(event_time, start_offset, end_offset, label, game_time, video_path, output_dir)


if __name__ == "__main__":
    main(JSON_PATH, VIDEO_PATH, OUTPUT_DIR)
