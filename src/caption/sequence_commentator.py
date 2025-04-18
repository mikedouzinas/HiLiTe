"""
SequenceCommentator: Generate detailed commentary covering the entire action sequence.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Any
import logging
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map SoccerNet labels to our event types
SOCCERNET_LABEL_MAP = {
    "Shots off target": "shot_missed",
    "Shots on target": "shot_on_target",
    "Goal": "goal",
    "Yellow card": "yellow_card",
    "Red card": "red_card",
    "Penalty": "penalty",
    "Kick-off": "kickoff",
    "Substitution": "substitution",
    "Offside": "offside",
    "Corner": "corner",
    "Ball out of play": "ball_out",
    "Throw-in": "throw_in",
    "Foul": "foul",
    "Indirect free-kick": "free_kick",
    "Direct free-kick": "free_kick",
    "Clearance": "clearance",
    "Shots blocked": "shot_blocked",
    "Saves": "save"
}

# Detailed sequence commentary patterns
SEQUENCE_COMMENTARY = {
    "shot_missed": {
        "build_up": [
            "{team} building from the back now. Patient passing as they work their way forward.",
            "Nice combination play from {team} as they advance up the field.",
            "{team} moving the ball around well, looking for an opening.",
            "Good possession from {team}, controlling the tempo of the game.",
            "Quick counter-attack by {team}, catching the opposition off guard."
        ],
        "progression": [
            "The ball is moved out wide, looking for space on the flank.",
            "Clever through ball into the channel, creating some danger.",
            "They're working it into the final third now, good attacking position.",
            "Beautiful diagonal pass switches the play and opens up space.",
            "The midfielder drives forward, committing defenders."
        ],
        "pre_shot": [
            "There's a chance opening up here for {team}!",
            "The defender is beaten, and now there's an opportunity!",
            "Good movement off the ball creates space for the shot.",
            "They've worked it into a dangerous position now.",
            "The ball falls nicely, and this is a real chance!"
        ],
        "shot": [
            "The striker pulls the trigger... but it's just wide of the post!",
            "Shot taken! But it sails over the crossbar.",
            "Strike from distance! Close, but off target.",
            "He takes aim... but can't find the frame of the goal.",
            "Powerful effort! But it's dragged just wide."
        ],
        "aftermath": [
            "The goalkeeper looked worried for a moment there.",
            "That was a decent opportunity wasted for {team}.",
            "The fans behind the goal were ready to celebrate.",
            "A let-off for the defending team, that could have been dangerous.",
            "The manager on the sideline can't believe that wasn't on target."
        ]
    },
    "shot_on_target": {
        "build_up": [
            "{team} patiently building from the back, looking for openings.",
            "Good passing sequence from {team} as they work through the lines.",
            "{team} maintaining possession well, probing for weaknesses.",
            "Quick transition from defense to attack by {team}.",
            "Clever movement off the ball from {team}, creating space."
        ],
        "progression": [
            "They push forward with purpose, getting players into advanced positions.",
            "Nice interchange of passes in the midfield, breaking through the lines.",
            "Cutting inside from the wing, looking dangerous now.",
            "Threading passes through the middle, carving open the defense.",
            "The winger drives forward with pace, putting the defense on the back foot."
        ],
        "pre_shot": [
            "There's a shooting opportunity developing here!",
            "The defense is split open, this is a chance!",
            "They've worked it into a great position, looking to capitalize!",
            "The ball breaks kindly in the box, chance to shoot!",
            "Clever movement creates space for a strike on goal!"
        ],
        "shot": [
            "The shot is fired in low and hard! On target!",
            "Powerful strike heading towards the bottom corner!",
            "Rising effort aimed at the top of the net!",
            "Side-footed attempt curling towards goal!",
            "Drilled shot through a crowd of players!"
        ],
        "aftermath": [
            "The goalkeeper makes the save, but that was a good effort.",
            "That's another shot on target for {team}, building pressure.",
            "They're getting closer to breaking the deadlock here.",
            "The keeper had to be alert there, good positioning.",
            "That's positive play from {team}, testing the goalkeeper again."
        ]
    },
    "goal": {
        "build_up": [
            "Excellent build-up play from {team}, moving the ball with confidence.",
            "Patient possession from {team}, waiting for the right moment to strike.",
            "{team} flowing forward now, this looks promising.",
            "Incisive passing from {team}, cutting through the opposition.",
            "Brilliant counter-attacking play from {team}, turning defense into attack."
        ],
        "progression": [
            "They're carving open the defense with surgical precision!",
            "Beautiful one-touch passing, the defenders can't get near them!",
            "Slicing through the midfield like a hot knife through butter!",
            "The winger beats his man and delivers a dangerous ball!",
            "Threading the needle with that pass, opening up the defense completely!"
        ],
        "pre_shot": [
            "What an opportunity this is!",
            "They've opened up the defense, this is it!",
            "Clear sight of goal now, surely they must score!",
            "The goalkeeper is exposed, this is a golden chance!",
            "Magnificent setup, the goal is at their mercy!"
        ],
        "shot": [
            "SHOT... GOAL!!! What a finish!",
            "He strikes it... GOOOAAAL! Unstoppable!",
            "The ball is struck... AND IT'S IN! Wonderful goal!",
            "Fires it towards goal... YES! It's in the back of the net!",
            "Takes aim... SCORES! A fabulous finish!"
        ],
        "aftermath": [
            "The stadium erupts! What a moment for {team}!",
            "The players rush to celebrate! They've been rewarded for their brilliant play!",
            "That's exactly what they deserved after that magnificent approach work!",
            "The bench is on their feet! What a crucial goal that could be!",
            "The fans are delirious! A picture-perfect goal from start to finish!"
        ]
    },
    "foul": {
        "build_up": [
            "The play is developing in midfield, both teams battling for control.",
            "Possession changing hands as they contest for the ball.",
            "Physical play in the center of the park, neither team giving an inch.",
            "The pace of the game has intensified, getting a bit scrappy now.",
            "Players challenging hard for every ball, showing real commitment."
        ],
        "progression": [
            "The ball is moved forward, but there's pressure coming.",
            "Tight marking from the opposition, making it difficult to progress.",
            "Looking to break through but meeting strong resistance.",
            "The defender comes across to close down the space quickly.",
            "There's a race for the loose ball in a dangerous area."
        ],
        "pre_foul": [
            "The player is trying to shield the ball under pressure.",
            "Determination from both sides as they contest for possession.",
            "He's looking to turn past his marker in a tight space.",
            "The defender is getting tight, not giving any room.",
            "There's a challenge coming in as they battle for control."
        ],
        "foul": [
            "The tackle comes in... and the referee blows for a foul!",
            "Challenge from behind, and that's a clear infringement!",
            "He goes to ground and takes the player! Free kick awarded!",
            "Late contact, and the referee immediately signals for the foul!",
            "Mistimed tackle, and there's no argument about that decision!"
        ],
        "aftermath": [
            "The player stays down, looks like he took a knock there.",
            "A bit of frustration showing from both teams after that incident.",
            "The referee has a word with the player, keeping control of the game.",
            "That's going to be a free kick in a promising position for {team}.",
            "The manager is having a word with his player about being more careful."
        ]
    },
    "corner": {
        "build_up": [
            "{team} looking to attack down the flank, pushing players forward.",
            "Good pressure from {team}, forcing the opposition back.",
            "Probing attack from {team}, trying to find a way through the defense.",
            "Patient play from {team}, waiting for the right moment to penetrate.",
            "Direct approach from {team}, looking to get the ball into the box."
        ],
        "progression": [
            "The ball is worked wide to the winger in space.",
            "They're looking to stretch the defense, creating width.",
            "Cross comes in from the flank, aiming for the penalty area.",
            "Trying to deliver the ball into a dangerous area.",
            "The fullback overlaps, adding another attacking option."
        ],
        "pre_corner": [
            "The defender is under pressure near his own goal line.",
            "They're looking to turn this pressure into a scoring opportunity.",
            "The cross is aimed towards the six-yard box, causing problems.",
            "Good aggressive play, forcing the defense to make a decision.",
            "The attacker nearly gets to the byline, looking to cut it back."
        ],
        "corner": [
            "The defender has to put it behind! Corner kick to {team}!",
            "Deflected out by the defense, and {team} have a corner!",
            "The goalkeeper tips it over the bar! Corner awarded!",
            "Last-ditch defending results in a corner for {team}!",
            "That's good pressure, and they've won a corner!"
        ],
        "aftermath": [
            "The big center-backs are making their way forward for this set piece.",
            "{team} sending plenty of players into the box, looking to capitalize.",
            "This is a good opportunity to test the opposition's defensive organization.",
            "They've been dangerous from corners already in this match.",
            "The crowd urges them forward, sensing an opportunity here."
        ]
    },
    "yellow_card": {
        "build_up": [
            "The game is getting heated now, challenges flying in.",
            "Tensions rising on the pitch, both teams fully committed.",
            "Physical battle developing in the middle of the park.",
            "The pace of the game has intensified, no quarter being given.",
            "Players contesting every ball with increasing intensity."
        ],
        "progression": [
            "The ball breaks loose, and there's a race to claim possession.",
            "Quick counter attack developing, players rushing back to defend.",
            "The attacker has a head of steam, driving towards goal.",
            "Breaking at pace, caught the defense outnumbered here.",
            "Dangerous situation developing as the play opens up."
        ],
        "pre_foul": [
            "The player is trying to recover ground, but he's out of position.",
            "He's stretching to reach the ball, but he's second best here.",
            "Desperate to stop the attack, but he's on the wrong side.",
            "The attacker has the beating of him for pace, trouble brewing.",
            "He's committed himself to the challenge, this could be risky."
        ],
        "foul": [
            "Lunging tackle comes in late! That's a bad challenge!",
            "Goes straight through the back of the player! Dangerous play!",
            "Cynical foul to stop the attack! Referee's not happy with that!",
            "Reckless challenge with no attempt to play the ball!",
            "Two-footed tackle that's completely mistimed!"
        ],
        "card": [
            "The referee reaches for his pocket... YELLOW CARD shown!",
            "And he's going into the book! Yellow card for that challenge!",
            "That's a booking, no arguments! Yellow card issued!",
            "The referee produces the yellow card! He can have no complaints!",
            "Caution for the player! That's a deserved yellow card!"
        ],
        "aftermath": [
            "He needs to be careful now, walking a disciplinary tightrope.",
            "The manager might be thinking about a substitution to avoid further trouble.",
            "That's put his team under pressure, they'll need to be more disciplined.",
            "The fans are letting the referee know what they think, but it was the correct decision.",
            "A moment of frustration that could prove costly later in the match."
        ]
    },
    "default": {
        "build_up": [
            "{team} in possession now, looking to create an opening.",
            "Patient build-up play from {team}, moving the ball around.",
            "{team} starting to exert some pressure here.",
            "Good spell of possession for {team}, controlling the tempo.",
            "Quick transition play from {team} as they move forward."
        ],
        "progression": [
            "They advance up the field with purpose, good movement off the ball.",
            "Looking to exploit the space on the flanks, stretching the play.",
            "Working the ball through the middle, trying to penetrate.",
            "The midfielders link up well, finding gaps between the lines.",
            "Direct approach now, playing with more urgency."
        ],
        "climax": [
            "This is promising for {team}, getting into a good position!",
            "They've worked it well, creating a chance here!",
            "The defense is scrambling to reorganize as {team} push forward!",
            "Good opportunity developing for {team}!",
            "They've got numbers forward, looking to capitalize!"
        ],
        "resolution": [
            "The move continues to develop, keeping the pressure on.",
            "They maintain possession, patiently waiting for the right moment.",
            "Looking for that final ball to create the clear-cut chance.",
            "The opposition defense stands firm, but {team} keep coming.",
            "Building momentum now, sensing a breakthrough might be close."
        ],
        "aftermath": [
            "That's positive play from {team}, showing good intent.",
            "The manager will be pleased with that passage of play.",
            "They're growing into the game, looking more dangerous.",
            "The fans appreciate the effort, encouraging their team forward.",
            "They'll need to keep this intensity up if they want to get a result today."
        ]
    }
}

class SequenceCommentator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the sequence commentator.
        
        Args:
            config: Configuration dictionary with model paths and parameters
        """
        self.config = config
        print("SequenceCommentator initialized successfully")
    
    def extract_video_segments(self, video_path: str) -> Dict:
        """
        Extract basic information about video segments for sequence commentary.
        
        Args:
            video_path: Path to the video clip
            
        Returns:
            Dictionary with video segment information
        """
        info = {}
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            info["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info["fps"] = cap.get(cv2.CAP_PROP_FPS)
            info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
            
            # Calculate segment boundaries (beginning, middle, end)
            total_frames = info["frame_count"]
            info["segments"] = {
                "beginning": (0, total_frames // 3),
                "middle": (total_frames // 3, 2 * total_frames // 3),
                "end": (2 * total_frames // 3, total_frames)
            }
            
            # Sample frames from each segment for basic motion analysis
            segment_motion = {}
            
            for segment_name, (start, end) in info["segments"].items():
                # Sample 3 evenly spaced frames from this segment
                if end > start + 2:  # Ensure we have enough frames
                    sample_points = np.linspace(start, end-1, 3, dtype=int)
                    
                    # Extract frames
                    frames = []
                    for point in sample_points:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, point)
                        ret, frame = cap.read()
                        if ret:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frames.append(gray)
                    
                    # Calculate motion for this segment
                    if len(frames) > 1:
                        diffs = []
                        for i in range(len(frames) - 1):
                            diff = cv2.absdiff(frames[i], frames[i+1])
                            non_zero = np.count_nonzero(diff)
                            # Normalize by frame size
                            height, width = frames[i].shape
                            diffs.append(non_zero / (width * height))
                        
                        segment_motion[segment_name] = np.mean(diffs)
                    else:
                        segment_motion[segment_name] = 0
                else:
                    segment_motion[segment_name] = 0
            
            info["segment_motion"] = segment_motion
            
            # Classify which segment has the most motion (likely the key action)
            if segment_motion:
                info["peak_action_segment"] = max(segment_motion, key=segment_motion.get)
            else:
                info["peak_action_segment"] = "end"  # Default to end if we couldn't detect
                
            cap.release()
                
        except Exception as e:
            logger.error(f"Error extracting video segments: {str(e)}")
            info["error"] = str(e)
            
        return info
        
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
            label = metadata["label"]
            standardized["raw_label"] = label
            standardized["event_type"] = SOCCERNET_LABEL_MAP.get(label, "default")
        else:
            standardized["event_type"] = "default"
        
        # Parse team
        if "team" in metadata:
            team_side = metadata["team"]
            if team_side == "home":
                standardized["team"] = "the home team"
            elif team_side == "away":
                standardized["team"] = "the away team"
            else:
                standardized["team"] = "the team"
        else:
            standardized["team"] = "the team"
            
        # Parse player - not typically in SoccerNet metadata
        standardized["player"] = metadata.get("player", "the player")
        
        # Parse game time
        if "gameTime" in metadata:
            game_time = metadata["gameTime"]
            standardized["raw_time"] = game_time
            
            # Handle different formats like "1 - 07:47" or just "07:47"
            parts = game_time.split(" - ")
            if len(parts) > 1:
                standardized["half"] = parts[0]
                time_str = parts[1]
            else:
                standardized["half"] = "1"  # Default to first half
                time_str = parts[0]
            
            standardized["timestamp"] = time_str
            
            # Parse minutes for context selection
            try:
                minutes = int(time_str.split(":")[0])
                standardized["minutes"] = minutes
            except:
                standardized["minutes"] = 0
        else:
            standardized["timestamp"] = ""
            standardized["minutes"] = 0
            
        return standardized
        
    def generate_sequence_commentary(self, video_path: str, metadata: Dict) -> str:
        """
        Generate detailed sequence commentary covering the entire action.
        
        Args:
            video_path: Path to the video clip
            metadata: Dictionary with event metadata (can be SoccerNet format)
            
        Returns:
            Multi-part sequence commentary text
        """
        # Extract video segment information
        video_info = self.extract_video_segments(video_path)
        
        # Parse metadata - check if it's SoccerNet format or already standardized
        if "label" in metadata or "gameTime" in metadata:
            # This appears to be SoccerNet format
            parsed_metadata = self.parse_soccernet_metadata(metadata)
        else:
            # Assume it's already in our expected format
            parsed_metadata = metadata
        
        # Extract standardized metadata
        event_type = parsed_metadata.get("event_type", "default")
        team = parsed_metadata.get("team", "the team")
        
        # Get the appropriate sequence templates
        if event_type in SEQUENCE_COMMENTARY:
            sequence_templates = SEQUENCE_COMMENTARY[event_type]
        else:
            sequence_templates = SEQUENCE_COMMENTARY["default"]
        
        # Build multi-part commentary based on sequence segments
        commentary_parts = []
        
        # Part 1: Build-up phase
        build_up = random.choice(sequence_templates["build_up"]).format(team=team)
        commentary_parts.append(build_up)
        
        # Part 2: Progression phase
        progression = random.choice(sequence_templates["progression"]).format(team=team)
        commentary_parts.append(progression)
        
        # Part 3: Pre-climax phase (different name depending on event type)
        pre_climax_key = next((k for k in sequence_templates.keys() if k.startswith("pre_")), "climax")
        pre_climax = random.choice(sequence_templates[pre_climax_key]).format(team=team)
        commentary_parts.append(pre_climax)
        
        # Part 4: Climax phase (the actual event - shot, foul, etc.)
        climax_key = event_type.split("_")[-1] if "_" in event_type else event_type
        # Fallback if the specific key doesn't exist
        if climax_key not in sequence_templates:
            climax_key = next((k for k in sequence_templates.keys() 
                             if k == climax_key or k == event_type or "shot" in k or "goal" in k or "foul" in k), 
                            "resolution")
        
        climax = random.choice(sequence_templates[climax_key]).format(team=team)
        commentary_parts.append(climax)
        
        # Part 5: Aftermath
        aftermath = random.choice(sequence_templates["aftermath"]).format(team=team)
        commentary_parts.append(aftermath)
        
        # Combine all parts into a cohesive commentary
        full_commentary = " ".join(commentary_parts)
        
        return full_commentary

    def process_clip(self, video_path: str, metadata: Dict) -> Dict:
        """
        Process a video clip and return the detailed sequence commentary.
        
        Args:
            video_path: Path to the video clip
            metadata: Dictionary with event metadata (can be SoccerNet format)
            
        Returns:
            Dictionary with commentary and other information
        """
        try:
            commentary = self.generate_sequence_commentary(video_path, metadata)
            
            # Parse metadata for return if needed
            if "label" in metadata or "gameTime" in metadata:
                # This appears to be SoccerNet format
                parsed_metadata = self.parse_soccernet_metadata(metadata)
            else:
                # Assume it's already in our expected format
                parsed_metadata = metadata
            
            return {
                "video_path": video_path,
                "metadata": parsed_metadata,
                "original_metadata": metadata,
                "commentary": commentary,
                "success": True
            }
        except Exception as e:
            logger.error(f"Error processing clip {video_path}: {str(e)}")
            
            # Attempt to get event type for fallback message
            event_type = "soccer action"
            if "label" in metadata:
                event_type = metadata["label"]
            elif "event_type" in metadata:
                event_type = metadata["event_type"]
                
            return {
                "video_path": video_path,
                "metadata": metadata,
                "commentary": f"An exciting {event_type} moment in the match!",  # Fallback
                "success": False,
                "error": str(e)
            }