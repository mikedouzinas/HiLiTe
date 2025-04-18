"""
Test script for the frame-based commentary system with GPT-4V.
"""

import os
import sys
import yaml
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.caption.frame_commentator import FrameCommentator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test frame-based commentary system')
    parser.add_argument('--api-key', type=str, help='OpenAI API key (if not set in environment)')
    parser.add_argument('--num-frames', type=int, default=5, help='Number of frames to extract from video')
    parser.add_argument('--save-only', action='store_true', help='Only extract and save frames, skip API call')
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
                "frame_based": {
                    "num_frames": args.num_frames
                }
            }
        }
    
    # Get OpenAI API key from args, environment or config
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or config.get("commentary", {}).get("frame_based", {}).get("openai_api_key")
    
    # Initialize the commentator
    print("Initializing FrameCommentator...")
    frame_config = config.get("commentary", {}).get("frame_based", {})
    frame_config["openai_api_key"] = api_key
    frame_config["num_frames"] = args.num_frames
    commentator = FrameCommentator(frame_config)
    
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
    
    print("\nAnalyzing video with frame-based approach:")
    print("-" * 80)
    print(f"Event: {soccernet_metadata['label']}")
    print(f"Time: {soccernet_metadata['gameTime']}")
    print(f"Team: {soccernet_metadata['team']}")
    print(f"Extracting {args.num_frames} frames...")
    print("-" * 80)
    
    # Extract frames
    try:
        frames = commentator.extract_frames(video_path)
        frame_paths = commentator.save_frames(frames, os.path.basename(video_path))
        
        print(f"Successfully extracted and saved {len(frames)} frames:")
        for path in frame_paths:
            print(f"  - {path}")
            
        if args.save_only:
            print("\nFrames saved successfully. Skipping API call as requested.")
            return
            
        # Check if API key is available
        if not api_key:
            print("\nNo OpenAI API key provided. Cannot generate commentary.")
            print("Please provide an API key with --api-key, set the OPENAI_API_KEY environment variable,")
            print("or add it to your config.yaml file under commentary.frame_based.openai_api_key")
            return
            
        # Process the clip with GPT-4V
        print("\nSending frames to GPT-4V for analysis...")
        result = commentator.process_clip(video_path, soccernet_metadata)
        
        # Print results
        if result["success"]:
            print("\nGENERATED COMMENTARY:")
            print("-" * 80)
            print(result['commentary'])
            print("-" * 80)
            
            print("\nSaved frames for reference at:")
            for path in result['frame_paths']:
                print(f"  - {path}")
        else:
            print(f"\nFailed to generate commentary: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()