# HiLite

An AI-powered system that automatically generates highlight reels from full-length soccer match videos with natural, engaging commentary.

## Overview

HiLite transforms lengthy soccer matches into concise, professionally narrated highlight reels by:
- Detecting key moments like goals, fouls, and cards
- Extracting relevant video clips
- Generating natural sports-style commentary
- Compiling everything into a polished highlight video

## Pipeline

### 1. Event Detection

- Identifies significant match events using SoccerNet's pre-trained TimeSformer model
- Fallback to lightweight I3D model available
- Outputs timestamps and metadata (player names, event types)

### 2. Clip Extraction

- Extracts video segments around each detected event
- Creates clips with appropriate context before and after key moments
- Prepares segments for commentary generation

### 3. Commentary Generation

- Analyzes video clips using BLIP-2 vision-language model
- Generates contextual, sports-style commentary
- Converts text to natural-sounding speech using Bark by Suno AI

### 4. Highlight Reel Compilation

- Arranges clips chronologically
- Adds professional transitions and text overlays
- Incorporates background music
- Produces a final polished highlight video

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HiLite.git
cd HiLite

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python main.py --video path/to/match.mkv

# Advanced options
python main.py --video path/to/match.mkv --config custom_config.yaml --output custom_output_dir --skip-processing --skip-commentary
```

### Command Line Arguments

- `--video`: Path to the input .mkv video file (required)
- `--config`: Path to configuration file (default: config.yaml)
- `--output`: Override the output directory
- `--skip-processing`: Skip video processing if clips already exist
- `--skip-commentary`: Skip commentary generation if already done

## Configuration

Create a `config.yaml` file in your project directory to customize the behavior of HiLite. Here's an example configuration:

```yaml
# SoccerNet configuration
soccernet_local_dir: "data/raw/"  # Local directory for SoccerNet data
soccernet_password: "your_password_here"  # Your SoccerNet password

# Commentary generation configuration
commentary:
  method: "gpt"  # Options: "template_based", "frame_based", "gpt", "deep_learning"
  
  # GPT-based configuration
  gpt:
    openai_api_key: "your_api_key_here"  # Your OpenAI API key
    model: "gpt-4"  # or other GPT model
    max_tokens: 200
    temperature: 0.7

  # SoccerNet analysis configuration
  soccernet:
    use_player_detection: true
    use_ball_tracking: true
    use_action_spotting: true
    temp_dir: "temp_frames"

  # Deep learning configuration (optional)
  deep_learning:
    image_model: "resnet50"
    video_model: null
    language_model: "gpt2"
    action_classifier_path: "models/soccer_action_classifier.pth"
    temp_dir: "temp_frames"
```

### Configuration Options Explained

- `soccernet_local_dir`: Directory where SoccerNet data will be stored
- `soccernet_password`: Your SoccerNet account password
- `commentary`: Settings for the commentary generation
  - `method`: Choose between different commentary generation methods:
    - `template_based`: Uses predefined templates
    - `frame_based`: Analyzes individual frames
    - `gpt`: Uses GPT models for commentary
    - `deep_learning`: Uses custom deep learning models
  - `gpt`: Settings for GPT-based commentary
    - `openai_api_key`: Your OpenAI API key
    - `model`: GPT model to use
    - `max_tokens`: Maximum length of generated commentary
    - `temperature`: Controls randomness in generation
  - `soccernet`: SoccerNet analysis settings
    - `use_player_detection`: Enable player detection
    - `use_ball_tracking`: Enable ball tracking
    - `use_action_spotting`: Enable action spotting
    - `temp_dir`: Directory for temporary files
  - `deep_learning`: Optional deep learning model settings
    - `image_model`: Model for image analysis
    - `video_model`: Model for video analysis
    - `language_model`: Model for text generation
    - `action_classifier_path`: Path to action classifier model
    - `temp_dir`: Directory for temporary files

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- FFmpeg
- CUDA-enabled GPU (recommended)

## License

MIT 
