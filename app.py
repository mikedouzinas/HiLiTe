# app.py
import streamlit as st
import tempfile
import shutil
import os
import yaml
import json
from pathlib import Path

# Import your pipeline functions
from mkv2json import main as mkv2json_main
from mkv2clips import main as mkv2clips_main, EVENT_WINDOWS, sanitize_filename
from main_with_clipper import (
    generate_commentary,
    generate_speech,
    compile_highlights,
    load_config,
    create_output_dirs,
)

st.set_page_config(page_title="HiLitR Soccer Highlight Generator", layout="wide")

st.title("🏟️ HiLitR: Soccer Highlight Generator")

st.markdown(
    """
    **Upload** a full-match video (MKV/MP4), and HiLitR will:
    1. Spot key events (goals, cards, fouls…)
    2. Extract & clip each event
    3. Generate AI commentary & TTS
    4. Assemble a polished highlight reel
    """
)

uploaded = st.file_uploader("🎥 Upload your full-match video", type=["mkv", "mp4"])
if not uploaded:
    st.info("Please upload an MKV/MP4 file to begin.")
    st.stop()

# Write uploaded video to a temp file so ffmpeg can read it
with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tfile:
    tfile.write(uploaded.read())
    video_path = tfile.name

# ──────────────────── Configuration & outputs ────────────────────────
CONFIG_PATH = "config.yaml"
config = load_config(CONFIG_PATH)

# Persist outputs in ./outputs unless the config already specifies one
config.setdefault("output_dir", "outputs")
create_output_dirs(config)

# Prepare Streamlit placeholders
progress = st.progress(0)
status = st.empty()

# ──────────────────── Step 1+2: Event spotting & clipping ────────────
status.text("Step 1+2: Detecting events & extracting clips…")
try:
    clips_dir = os.path.join(config["output_dir"], "clips")
    mkv2clips_main(video_path, clips_dir)  # writes events.json + clips
    progress.progress(20)
except Exception as e:
    st.error(f"Failed at clip extraction: {e}")
    st.stop()

# Load events.json produced by mkv2clips
events_json = os.path.join(clips_dir, "events.json")
with open(events_json, "r") as f:
    events = json.load(f).get("events", [])

# Build clip metadata list
status.text("Building clip list…")
clips = []
for ev in events:
    fname = sanitize_filename(ev["label"], ev["gameTime"])
    clip_path = os.path.join(clips_dir, fname)
    ev["clip_path"] = clip_path
    ev["duration"] = EVENT_WINDOWS.get(ev["label"], (0, 5))[1] - EVENT_WINDOWS.get(ev["label"], (0, 5))[0]
    clips.append(ev)
progress.progress(30)

# ──────────────────── Step 3: Commentary ─────────────────────────────
status.text("Step 3: Generating commentary…")
try:
    clips = generate_commentary(clips, config)
    progress.progress(50)
except Exception as e:
    st.warning(f"Commentary generation failed, using fallback: {e}")
    progress.progress(50)

# ──────────────────── Step 3.5: Text‑to‑Speech ───────────────────────
status.text("Step 3.5: Synthesizing speech…")
try:
    clips = generate_speech(clips, config)
    progress.progress(70)
except Exception as e:
    st.warning(f"TTS generation failed, skipping audio: {e}")
    progress.progress(70)

# ──────────────────── Step 4: Highlight compilation ──────────────────
status.text("Step 4: Compiling final highlight reel…")
try:
    final_video = compile_highlights(clips, config)
    progress.progress(100)
    status.success(f"✅ Highlight reel ready! Files saved in {config['output_dir']}")
except Exception as e:
    st.error(f"Failed to compile highlights: {e}")
    st.stop()

# Display and offer download
st.video(final_video)
with open(final_video, "rb") as vid:
    st.download_button(
        label="⬇️ Download Highlight Reel",
        data=vid,
        file_name="highlights.mp4",
        mime="video/mp4",
    )

# Clean up the temporary upload file
os.unlink(video_path)
