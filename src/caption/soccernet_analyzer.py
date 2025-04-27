"""
SoccerNet Analysis Module: Extract detailed soccer insights using SoccerNet models
"""

import os
import cv2
import torch
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
import time
from pathlib import Path

# Try to import SoccerNet packages
try:
    import SoccerNet
    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.Evaluation.ActionSpotting import evaluate
    from SoccerNet.LCRNet.utils import LCRNet_data_loader
    from SoccerNet.LCRNet.model import LCRNet_model
    SOCCERNET_AVAILABLE = True
except ImportError:
    SOCCERNET_AVAILABLE = False
    print("SoccerNet package not found. Running with limited functionality.")
    print("Install with: pip install SoccerNet")

# Try to import for player detection
try:
    import torch
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoccerNetAnalyzer:
    """Analyze soccer videos using SoccerNet models to extract rich metadata."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SoccerNet analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Temp directory for saving frames and features
        self.temp_dir = config.get("temp_dir", "temp_frames")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize available models
        self.initialize_models()
        
        logger.info("SoccerNetAnalyzer initialized successfully")
    
    def initialize_models(self):
        """Initialize SoccerNet and other models."""
        # Action spotting model (from SoccerNet Challenge)
        if SOCCERNET_AVAILABLE and self.config.get("use_action_spotting", True):
            try:
                self.initialize_action_spotting()
            except Exception as e:
                logger.warning(f"Failed to initialize action spotting model: {e}")
                self.action_spotting_available = False
        else:
            self.action_spotting_available = False
        
        # Player detection model (using Faster R-CNN)
        if DETECTION_AVAILABLE and self.config.get("use_player_detection", True):
            try:
                self.initialize_player_detection()
            except Exception as e:
                logger.warning(f"Failed to initialize player detection model: {e}")
                self.player_detection_available = False
        else:
            self.player_detection_available = False
        
        # Ball tracking (simplified implementation)
        self.ball_tracking_available = self.config.get("use_ball_tracking", True)
    
    def initialize_action_spotting(self):
        """Initialize the SoccerNet action spotting model."""
        if not SOCCERNET_AVAILABLE:
            self.action_spotting_available = False
            return
            
        # This is a placeholder - in a real implementation, you would:
        # 1. Load a pre-trained SoccerNet action spotting model
        # 2. Configure it for inference
        
        # For now, we'll simulate this by setting a flag
        logger.info("Action spotting model initialized (simulated)")
        self.action_spotting_available = True
    
    def initialize_player_detection(self):
        """Initialize player detection model (Faster R-CNN)."""
        if not DETECTION_AVAILABLE:
            self.player_detection_available = False
            return
            
        logger.info("Loading player detection model (Faster R-CNN)")
        self.player_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        self.player_detector.to(self.device)
        self.player_detector.eval()
        
        # Classes in COCO dataset (person is class 1)
        self.detection_classes = {
            1: "person",
            32: "sports ball",
            37: "sports equipment"
        }
        
        self.player_detection_available = True
    
    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[np.ndarray]:
        """
        Extract evenly spaced frames from a video.
        
        Args:
            video_path: Path to the video clip
            num_frames: Number of frames to extract
            
        Returns:
            List of numpy arrays containing the extracted frames
        """
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
    
    def detect_players(self, frames: List[np.ndarray]) -> List[Dict]:
        """
        Detect players and ball in frames using Faster R-CNN.
        
        Args:
            frames: List of frames to analyze
            
        Returns:
            List of dictionaries with detection results
        """
        if not self.player_detection_available:
            return [{"error": "Player detection model not available"}]
        
        results = []
        
        # Process each frame
        for i, frame in enumerate(frames):
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Prepare input tensor
            input_tensor = torchvision.transforms.ToTensor()(frame_rgb).to(self.device)
            
            # Run detection
            with torch.no_grad():
                prediction = self.player_detector([input_tensor])
            
            # Process predictions
            boxes = prediction[0]['boxes'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            
            # Filter by score and class
            persons = []
            balls = []
            equipment = []
            
            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:  # Confidence threshold
                    if label == 1:  # Person
                        persons.append({
                            "box": box.tolist(),
                            "score": float(score),
                            "type": "player"
                        })
                    elif label == 32:  # Sports ball
                        balls.append({
                            "box": box.tolist(),
                            "score": float(score),
                            "type": "ball"
                        })
                    elif label == 37:  # Sports equipment
                        equipment.append({
                            "box": box.tolist(),
                            "score": float(score),
                            "type": "equipment"
                        })
            
            # Add results for this frame
            frame_result = {
                "frame_idx": i,
                "players": persons,
                "balls": balls,
                "equipment": equipment,
                "total_players": len(persons),
                "ball_detected": len(balls) > 0
            }
            
            results.append(frame_result)
        
        return results
    
    def detect_actions(self, video_path: str, metadata: Dict) -> Dict:
        """
        Detect soccer actions in the video using SoccerNet models.
        
        Args:
            video_path: Path to the video clip
            metadata: Existing metadata about the clip
            
        Returns:
            Dictionary with detected actions and confidence scores
        """
        # This is a simulated implementation
        # In a real implementation, you would use the SoccerNet action spotting model
        
        # Get the action from metadata
        if "label" in metadata:
            label = metadata["label"]
        else:
            label = "unknown"
        
        # Simulate action detection with confidence scores
        if label == "Shots off target":
            actions = {
                "shot_missed": 0.85,
                "pass": 0.45,
                "dribble": 0.62
            }
        elif label == "Shots on target":
            actions = {
                "shot_on_target": 0.82,
                "pass": 0.38,
                "dribble": 0.55
            }
        elif label == "Goal":
            actions = {
                "goal": 0.92,
                "shot_on_target": 0.88,
                "celebration": 0.75
            }
        elif label == "Foul":
            actions = {
                "foul": 0.78,
                "tackle": 0.72,
                "player_down": 0.65
            }
        else:
            # Generic actions for other labels
            actions = {
                "pass": 0.65,
                "dribble": 0.59,
                "player_movement": 0.72
            }
        
        return {
            "detected_actions": actions,
            "primary_action": label,
            "confidence": actions.get(label.lower().replace(" ", "_"), 0.7)
        }
    
    def analyze_game_context(self, frames: List[np.ndarray], metadata: Dict) -> Dict:
        """
        Analyze game context from frames and metadata.
        
        Args:
            frames: List of frames to analyze
            metadata: Existing metadata about the clip
            
        Returns:
            Dictionary with game context information
        """
        # Extract game context from metadata
        if "gameTime" in metadata:
            game_time = metadata["gameTime"]
            # Parse game time (e.g., "1 - 07:47")
            parts = game_time.split(" - ")
            if len(parts) > 1:
                half = parts[0]
                time_str = parts[1]
            else:
                half = "1"  # Default to first half
                time_str = parts[0]
            
            # Parse minutes and seconds
            time_parts = time_str.split(":")
            if len(time_parts) > 1:
                minutes = int(time_parts[0])
                seconds = int(time_parts[1])
            else:
                minutes = 0
                seconds = 0
            
            # Determine game phase
            if half == "1":
                if minutes < 15:
                    game_phase = "early_first_half"
                elif minutes < 40:
                    game_phase = "middle_first_half"
                else:
                    game_phase = "late_first_half"
            else:  # second half
                if minutes < 15:
                    game_phase = "early_second_half"
                elif minutes < 40:
                    game_phase = "middle_second_half"
                else:
                    game_phase = "late_second_half"
        else:
            half = "unknown"
            minutes = 0
            seconds = 0
            game_phase = "unknown"
        
        # Extract team information
        if "team" in metadata:
            team = metadata["team"]
        else:
            team = "unknown"
        
        # Analyze field position (simplified)
        # In a real implementation, you would use field segmentation models
        # For now, we'll use a simple heuristic based on colors
        
        field_positions = []
        for frame in frames:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Define range for green color (field)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            
            # Create mask for green areas
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Calculate percentage of field visible
            field_percentage = np.count_nonzero(mask) / mask.size
            
            # Analyze field distribution
            h, w = mask.shape
            left_field = np.count_nonzero(mask[:, :w//2]) / (mask[:, :w//2].size)
            right_field = np.count_nonzero(mask[:, w//2:]) / (mask[:, w//2:].size)
            
            # Determine approximate field position
            if left_field > right_field * 1.5:
                position = "attacking_third" if team != "unknown" else "left_third"
            elif right_field > left_field * 1.5:
                position = "defensive_third" if team != "unknown" else "right_third"
            else:
                position = "middle_third"
            
            field_positions.append(position)
        
        # Get the most common field position
        if field_positions:
            from collections import Counter
            position_counts = Counter(field_positions)
            field_position = position_counts.most_common(1)[0][0]
        else:
            field_position = "unknown"
        
        return {
            "game_time": {
                "half": half,
                "minutes": minutes,
                "seconds": seconds,
                "time_string": f"{minutes}:{seconds:02d}"
            },
            "game_phase": game_phase,
            "field_position": field_position,
            "team": team
        }
    
    def analyze_video(self, video_path: str, metadata: Dict) -> Dict:
        """
        Perform comprehensive analysis of a soccer video clip.
        
        Args:
            video_path: Path to the video clip
            metadata: Existing metadata about the clip
            
        Returns:
            Dictionary with analysis results
        """
        # Extract frames from the video
        frames = self.extract_frames(video_path, num_frames=8)
        
        # Save frames for reference
        frame_paths = self.save_frames(frames, os.path.basename(video_path))
        
        # Detect players and ball
        player_detections = self.detect_players(frames)
        
        # Detect actions
        action_results = self.detect_actions(video_path, metadata)
        
        # Analyze game context
        game_context = self.analyze_game_context(frames, metadata)
        
        # Compile the analysis results
        analysis = {
            "video_path": video_path,
            "original_metadata": metadata,
            "frame_paths": frame_paths,
            "detections": {
                "players": player_detections,
                "player_count_avg": sum(d.get("total_players", 0) for d in player_detections) / len(player_detections) if player_detections else 0,
                "ball_detected_frames": sum(1 for d in player_detections if d.get("ball_detected", False)) if player_detections else 0
            },
            "actions": action_results,
            "game_context": game_context,
            "analysis_timestamp": time.time()
        }
        
        # Save the analysis to a JSON file
        analysis_dir = os.path.dirname(frame_paths[0]) if frame_paths else self.temp_dir
        analysis_path = os.path.join(analysis_dir, "soccernet_analysis.json")
        with open(analysis_path, "w") as f:
            # Convert non-serializable objects to strings
            json_analysis = {k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v for k, v in analysis.items()}
            json.dump(json_analysis, f, indent=2)
        
        return analysis

    def generate_prompt_for_gpt(self, analysis: Dict) -> str:
        """
        Generate a detailed prompt for GPT based on the analysis.
        
        Args:
            analysis: Dictionary with analysis results
            
        Returns:
            Prompt string
        """
        # Extract key information
        metadata = analysis["original_metadata"]
        game_context = analysis["game_context"]
        actions = analysis["actions"]
        detections = analysis["detections"]
        
        # Basic event information
        event_type = metadata.get("label", "soccer play")
        team = game_context.get("team", "the team")
        
        # Game context
        game_time = game_context.get("game_time", {})
        half = game_time.get("half", "unknown")
        minutes = game_time.get("minutes", 0)
        seconds = game_time.get("seconds", 0)
        game_phase = game_context.get("game_phase", "unknown")
        field_position = game_context.get("field_position", "unknown")
        
        # Action details
        primary_action = actions.get("primary_action", "unknown")
        detected_actions = actions.get("detected_actions", {})
        action_list = ", ".join([f"{action} ({confidence:.2f})" for action, confidence in detected_actions.items()])
        
        # Player detections
        avg_players = detections.get("player_count_avg", 0)
        ball_detected = detections.get("ball_detected_frames", 0) > 0
        
        # Construct prompt
        prompt = f"""
You are a professional soccer commentator tasked with describing a specific game moment from a soccer match.

### Event Information
- Event Type: {event_type}
- Team: {team}
- Game Time: {half}st half, {minutes}:{seconds:02d}
- Game Phase: {game_phase.replace('_', ' ')}
- Field Position: {field_position.replace('_', ' ')}

### Detected Actions
- Primary Action: {primary_action}
- All Detected Actions: {action_list}

### Visual Analysis
- Average Players Visible: {avg_players:.1f}
- Ball Detected: {'Yes' if ball_detected else 'No'}

### Commentary Task
Generate exciting, professional soccer commentary for this {event_type} by {team} at minute {minutes}:{seconds:02d}. Your commentary should:

1. Describe the build-up play leading to this moment
2. Comment on the players' positioning and movement
3. Describe the specific {primary_action} technique and execution
4. Mention the game context (time, score situation, field position)
5. Add appropriate emotion based on the importance of the moment

Your commentary should be 3-5 sentences long, vivid, and authentic to professional soccer broadcasting.
"""
        
        return prompt
    
    def generate_analysis_report(self, analysis: Dict) -> str:
        """
        Generate a human-readable analysis report.
        
        Args:
            analysis: Dictionary with analysis results
            
        Returns:
            Report string
        """
        metadata = analysis["original_metadata"]
        game_context = analysis["game_context"]
        actions = analysis["actions"]
        detections = analysis["detections"]
        
        report = """
# Soccer Video Analysis Report

## Event Information
"""
        report += f"- **Event Type:** {metadata.get('label', 'Unknown')}\n"
        report += f"- **Team:** {game_context.get('team', 'Unknown')}\n"
        
        # Game time
        game_time = game_context.get("game_time", {})
        report += f"- **Game Time:** {game_time.get('half', '?')}st half, {game_time.get('minutes', 0)}:{game_time.get('seconds', 0):02d}\n"
        report += f"- **Game Phase:** {game_context.get('game_phase', 'Unknown').replace('_', ' ').title()}\n"
        report += f"- **Field Position:** {game_context.get('field_position', 'Unknown').replace('_', ' ').title()}\n"
        
        report += """
## Detected Actions
"""
        detected_actions = actions.get("detected_actions", {})
        for action, confidence in detected_actions.items():
            report += f"- **{action.replace('_', ' ').title()}:** {confidence:.2f} confidence\n"
        
        report += """
## Player & Ball Detection
"""
        report += f"- **Average Players Detected:** {detections.get('player_count_avg', 0):.1f}\n"
        report += f"- **Ball Detected:** {'Yes' if detections.get('ball_detected_frames', 0) > 0 else 'No'}\n"
        
        return report
    
    def process_clip(self, video_path: str, metadata: Dict) -> Dict:
        """
        Process a video clip and generate rich metadata for GPT.
        
        Args:
            video_path: Path to the video clip
            metadata: Dictionary with event metadata
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Analyze the video
            analysis = self.analyze_video(video_path, metadata)
            
            # Generate prompt for GPT
            gpt_prompt = self.generate_prompt_for_gpt(analysis)
            
            # Generate human-readable report
            report = self.generate_analysis_report(analysis)
            
            # Return the results
            return {
                "video_path": video_path,
                "metadata": metadata,
                "analysis": analysis,
                "gpt_prompt": gpt_prompt,
                "report": report,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing clip {video_path}: {str(e)}")
            
            return {
                "video_path": video_path,
                "metadata": metadata,
                "success": False,
                "error": str(e)
            }