"""
FrameCommentator: Generate detailed commentary by analyzing frames from video clips using GPT-4V.
"""

import os
import cv2
import base64
import requests
import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FrameCommentator:
    """Generate soccer commentary by analyzing frames using GPT-4V."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the frame commentator.
        
        Args:
            config: Configuration dictionary with API keys and parameters
        """
        self.config = config
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = config.get("model", "gpt-4-vision-preview")
        self.max_tokens = config.get("max_tokens", 500)
        self.temperature = config.get("temperature", 0.7)
        self.num_frames = config.get("num_frames", 5)
        self.temp_dir = config.get("temp_dir", "temp_frames")
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Verify API key is available
        if not self.api_key:
            logger.warning("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide in config.")
        else:
            logger.info("FrameCommentator initialized successfully")
    
    def extract_frames(self, video_path: str, num_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Extract evenly spaced frames from a video.
        
        Args:
            video_path: Path to the video clip
            num_frames: Number of frames to extract (defaults to self.num_frames)
            
        Returns:
            List of numpy arrays containing the extracted frames
        """
        if num_frames is None:
            num_frames = self.num_frames
            
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        logger.info(f"Video info: {frame_count} frames, {fps} fps, {duration:.2f} seconds")
        
        # Calculate frame indices to extract
        if frame_count <= num_frames:
            # If we have fewer frames than requested, use all frames
            indices = list(range(frame_count))
        else:
            # Extract evenly spaced frames
            indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        # Extract the frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                logger.warning(f"Failed to read frame at index {idx}")
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video")
        return frames
    
    def frames_to_base64(self, frames: List[np.ndarray]) -> List[str]:
        """
        Convert frames to base64 strings for API transmission.
        
        Args:
            frames: List of numpy arrays containing frames
            
        Returns:
            List of base64-encoded strings
        """
        base64_frames = []
        for i, frame in enumerate(frames):
            # Convert to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to base64
            base64_str = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(base64_str)
        
        return base64_frames
    
    def save_frames(self, frames: List[np.ndarray], video_name: str) -> List[str]:
        """
        Save frames to disk and return their paths.
        
        Args:
            frames: List of numpy arrays containing frames
            video_name: Name of the video (for creating unique filenames)
            
        Returns:
            List of paths to saved frames
        """
        # Create a unique subdirectory
        timestamp = int(time.time())
        video_basename = Path(video_name).stem
        subdir = os.path.join(self.temp_dir, f"{video_basename}_{timestamp}")
        os.makedirs(subdir, exist_ok=True)
        
        # Save frames
        frame_paths = []
        for i, frame in enumerate(frames):
            path = os.path.join(subdir, f"frame_{i:03d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
        
        return frame_paths
    
    def parse_soccernet_metadata(self, metadata: Dict) -> Dict:
        """
        Parse SoccerNet-style metadata into a standardized format.
        
        Args:
            metadata: Dictionary with SoccerNet metadata
            
        Returns:
            Standardized metadata dictionary
        """
        standardized = {}
        
        # Parse event type
        if "label" in metadata:
            standardized["event_type"] = metadata["label"]
        else:
            standardized["event_type"] = "soccer event"
        
        # Parse team
        if "team" in metadata:
            team_value = metadata["team"]
            if team_value == "home":
                standardized["team"] = "the home team"
            elif team_value == "Chelsea":
                standardized["team"] = "the away team"
            else:
                # If it's an actual team name (like "Chelsea"), use it directly
                standardized["team"] = team_value
        else:
            standardized["team"] = "the team"
            
        # Parse game time
        if "gameTime" in metadata:
            standardized["timestamp"] = metadata["gameTime"]
        else:
            standardized["timestamp"] = ""
            
        return standardized
    
    def create_prompt(self, metadata: Dict, frame_count: int) -> str:
        """
        Create a prompt for GPT-4V based on metadata and frames.
        
        Args:
            metadata: Dictionary with event metadata
            frame_count: Number of frames being analyzed
            
        Returns:
            Prompt text for GPT-4V
        """
        # Parse metadata if in SoccerNet format
        if "label" in metadata and "gameTime" in metadata:
            parsed = self.parse_soccernet_metadata(metadata)
        else:
            parsed = metadata
        
        event_type = parsed.get("event_type", "soccer play")
        team = parsed.get("team", "the team")
        timestamp = parsed.get("timestamp", "")
        
        # Create the prompt
        prompt = (
            f"Here are {frame_count} sequential frames from a soccer clip that shows a {event_type}. "
            f"This event occurs at minute {timestamp} and involves {team}.\n\n"
            "Analyze these frames carefully and generate detailed, exciting sports commentary that "
            "describes the entire sequence of play. Include:\n"
            "1. The build-up play and how the attack develops\n"
            "2. The specific technical details visible in the frames (player positions, ball movement)\n"
            "3. The main action ({event_type})\n"
            "4. The immediate aftermath\n\n"
            "Use authentic, professional soccer announcer style with appropriate energy and excitement. "
            "Make the commentary flow naturally as if calling the action live. "
            "Your commentary should be 3-5 sentences long and capture the complete sequence shown in the frames."
        )
        
        return prompt
    
    def generate_commentary_with_gpt4v(self, frames: List[np.ndarray], metadata: Dict) -> str:
        """
        Generate commentary using GPT-4V by analyzing the frames.
        
        Args:
            frames: List of video frames as numpy arrays
            metadata: Dictionary with event metadata
            
        Returns:
            Generated commentary text
        """
        if not self.api_key:
            raise ValueError("No OpenAI API key available. Please set OPENAI_API_KEY or provide in config.")
            
        # Convert frames to base64
        frame_images = self.frames_to_base64(frames)
        
        # Create the prompt
        prompt = self.create_prompt(metadata, len(frames))
        
        # Prepare the messages for the API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Add frame images to the message
        for frame in frame_images:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        try:
            # Make the API request
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            commentary = result["choices"][0]["message"]["content"]
            
            return commentary
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing API response: {str(e)}")
            raise
    
    def process_clip(self, video_path: str, metadata: Dict) -> Dict:
        """
        Process a video clip and generate commentary using GPT-4V.
        
        Args:
            video_path: Path to the video clip
            metadata: Dictionary with event metadata
            
        Returns:
            Dictionary with commentary and other information
        """
        try:
            # Extract frames from the video
            frames = self.extract_frames(video_path)
            
            # Check if we have frames
            if not frames:
                raise ValueError(f"Failed to extract frames from {video_path}")
            
            # Generate commentary using GPT-4V
            commentary = self.generate_commentary_with_gpt4v(frames, metadata)
            
            # Save frames for reference
            frame_paths = self.save_frames(frames, os.path.basename(video_path))
            
            # Parse metadata for return if needed
            if "label" in metadata and "gameTime" in metadata:
                parsed_metadata = self.parse_soccernet_metadata(metadata)
            else:
                parsed_metadata = metadata
            
            return {
                "video_path": video_path,
                "metadata": parsed_metadata,
                "original_metadata": metadata,
                "commentary": commentary,
                "frame_paths": frame_paths,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing clip {video_path}: {str(e)}")
            
            return {
                "video_path": video_path,
                "metadata": metadata,
                "commentary": f"An exciting soccer moment!",  # Fallback
                "success": False,
                "error": str(e)
            }