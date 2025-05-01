"""
Test script for the enhanced GPT commentary system with SoccerNet analysis.
"""

import os
import sys
import yaml
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.caption.gpt_commentator import GPTCommentator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test GPT commentary system with SoccerNet analysis')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (if not set in environment)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='OpenAI model to use')
    parser.add_argument('--analyze-only', action='store_true', help='Only run SoccerNet analysis, skip GPT call')
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        # Fallback to default config
        config = {
            "commentary": {
                "gpt": {
                    "model": args.model,
                    "max_tokens": 200,
                    "temperature": 0.7
                },
                "soccernet": {
                    "use_player_detection": True,
                    "use_ball_tracking": True
                }
            }
        }
    
    # Get OpenAI API key from args, environment or config
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or config.get("commentary", {}).get("gpt", {}).get("openai_api_key")
    
    # Initialize the commentator
    print("Initializing GPTCommentator with SoccerNet analysis...")
    gpt_config = config.get("commentary", {}).get("gpt", {})
    gpt_config["openai_api_key"] = api_key
    gpt_config["model"] = args.model
    gpt_config["use_soccernet"] = True
    gpt_config["soccernet"] = config.get("commentary", {}).get("soccernet", {})
    
    commentator = GPTCommentator(gpt_config)
    
    # Test video path - find a suitable test video
    test_dir = os.path.join('data', 'test')
    video_files = [f for f in os.listdir(test_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No test videos found in {test_dir}. Please add some test videos.")
        return
    
    # Find the "Shots_off_target" video specifically
    shots_video = None
    for file in video_files:
        if "shot" in file.lower() or "off_target" in file.lower():
            shots_video = file
            break
    
    if not shots_video:
        shots_video = video_files[0]  # Fallback to first video
    
    video_path = os.path.join(test_dir, shots_video)
    print(f"Using test video: {video_path}")
    
    # Test with actual SoccerNet metadata
    soccernet_metadata = {
        "gameTime": "1 - 07:47",
        "label": "Shots off target",
        "position": "467037",
        "team": "Chelsea",
        "visibility": "visible"
    }
    
    print("\nProcessing video with SoccerNet analysis + GPT:")
    print("-" * 80)
    print(f"Event: {soccernet_metadata['label']}")
    print(f"Time: {soccernet_metadata['gameTime']}")
    print(f"Team: {soccernet_metadata['team']}")
    print(f"GPT Model: {args.model}")
    print("-" * 80)
    
    # Process the clip
    result = commentator.process_clip(video_path, soccernet_metadata)
    
    # Print analysis report
    if result["success"]:
        print("\nSOCCERNET ANALYSIS REPORT:")
        print("-" * 80)
        print(result["report"])
        print("-" * 80)
        
        # Print GPT prompt
        print("\nGPT PROMPT:")
        print("-" * 80)
        print(result["prompt"])
        print("-" * 80)
        
        # If not analyze-only, print the commentary
        if not args.analyze_only:
            if not api_key:
                print("\nNo OpenAI API key provided. Cannot generate commentary.")
                print("Please provide an API key with --api-key, set the OPENAI_API_KEY environment variable,")
                print("or add it to your config.yaml file under commentary.gpt.openai_api_key")
            else:
                print("\nGENERATED COMMENTARY:")
                print("-" * 80)
                print(result["commentary"])
                print("-" * 80)
        else:
            print("\nAnalysis-only mode enabled, skipping GPT commentary generation.")
    else:
        print(f"\nFailed to process video: {result.get('error', 'Unknown error')}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()