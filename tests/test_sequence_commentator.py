"""
Test script for the sequence-based detailed commentary system.
"""

import os
import sys
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.caption.sequence_commentator import SequenceCommentator

def main():
    # Load configuration
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading config: {e}")
        # Fallback to default config
        config = {
            "commentary": {
                "sequence": {}  # No special config needed
            }
        }
    
    # Initialize commentator
    print("Initializing SequenceCommentator...")
    commentator = SequenceCommentator(config.get("commentary", {}).get("sequence", {}))
    
    # Test video path - update with your test video
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
    
    print("\nGenerating detailed sequence commentary:")
    print("-" * 80)
    print(f"Event: {soccernet_metadata['label']}")
    print(f"Time: {soccernet_metadata['gameTime']}")
    print(f"Team: {soccernet_metadata['team']}")
    print("-" * 80)
    
    # Process the clip with SoccerNet metadata
    result = commentator.process_clip(video_path, soccernet_metadata)
    
    # Print results
    if result["success"]:
        print(f"DETAILED COMMENTARY:\n{result['commentary']}")
        print("\nParsed Metadata:")
        for key, value in result['metadata'].items():
            if key not in ['raw_label', 'raw_time']:  # Skip raw fields for cleaner output
                print(f"  {key}: {value}")
    else:
        print(f"Failed to generate commentary: {result.get('error', 'Unknown error')}")
    
    # Test with a few different event types
    print("\nTesting different event types for comparison:")
    
    test_events = [
        "Goal", 
        "Yellow card"
    ]
    
    for event in test_events:
        test_metadata = {
            "gameTime": "1 - 07:47",
            "label": event,
            "team": "Chelsea",
            "visibility": "visible"
        }
        
        print(f"\n{'-' * 40}")
        print(f"EVENT TYPE: {event}")
        print(f"{'-' * 40}")
        
        result = commentator.process_clip(video_path, test_metadata)
        if result["success"]:
            print(f"COMMENTARY:\n{result['commentary']}")
        else:
            print(f"Failed: {result.get('error', 'Unknown error')}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()