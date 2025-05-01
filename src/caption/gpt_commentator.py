"""
GPT Commentator: Generate detailed soccer commentary using SoccerNet analysis and GPT.
"""

import os
import time
import logging
import json
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPTCommentator:
    """
    Generate detailed soccer commentary using GPT with SoccerNet metadata.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GPT commentator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY"))
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.model = config.get("model", "gpt-4o-mini")
        self.max_tokens = config.get("max_tokens", 200)
        self.temperature = config.get("temperature", 0.7)
        
        # SoccerNet analyzer configuration
        self.use_soccernet = config.get("use_soccernet", True)
        
        # Initialize SoccerNet analyzer if available and requested
        if self.use_soccernet:
            try:
                from src.caption.soccernet_analyzer import SoccerNetAnalyzer
                self.analyzer = SoccerNetAnalyzer(config.get("soccernet", {}))
                self.analyzer_available = True
                logger.info("SoccerNet analyzer initialized successfully")
            except ImportError as e:
                logger.warning(f"SoccerNet analyzer import failed: {e}")
                self.analyzer_available = False
        else:
            self.analyzer_available = False
        
        # Verify API key is available
        if not self.api_key:
            logger.warning("No OpenAI API key found. Please set OPENAI_API_KEY environment variable or provide in config.")
        else:
            logger.info("GPTCommentator initialized successfully")
    
    def generate_commentary_with_gpt(self, prompt: str) -> str:
        """
        Generate commentary using GPT API.
        
        Args:
            prompt: Prompt for GPT
            
        Returns:
            Generated commentary text
        """
        # Check if API key is available
        if not self.api_key:
            logger.error("No OpenAI API key provided. Set OPENAI_API_KEY environment variable or provide in config.")
            return "No commentary available - API key is missing."
        
        # Prepare the API request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a professional soccer commentator with extensive knowledge of the sport. Your commentary is exciting, technically accurate, and captures the drama of the moment. Do not inlude unicode characters in your response. Also, 50% of the time you should not reference the time of the moment."},
            {"role": "user", "content": prompt}
        ]
        
        # Prepare the API request data
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        # Make the API request
        try:
            logger.info("Sending request to OpenAI API...")
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Validate response structure
            if not result or "choices" not in result or not result["choices"]:
                logger.error(f"Unexpected API response format: {result}")
                return "Unable to generate commentary - unexpected API response format."
                
            if "message" not in result["choices"][0] or "content" not in result["choices"][0]["message"]:
                logger.error(f"Missing message or content in API response: {result['choices'][0]}")
                return "Unable to generate commentary - missing content in API response."
                
            commentary = result["choices"][0]["message"]["content"]
            logger.info("Received commentary from GPT")
            
            return commentary
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error communicating with OpenAI API: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return f"Unable to generate commentary - API error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in API processing: {str(e)}")
            return f"Unable to generate commentary - error: {str(e)}"
    
    def process_clip(self, video_path: str, metadata: Dict) -> Dict:
        """
        Process a video clip and generate commentary using SoccerNet analysis and GPT.
        
        Args:
            video_path: Path to the video clip
            metadata: Dictionary with event metadata
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Analysis and prompt generation phase
            if self.analyzer_available:
                # Use SoccerNet analyzer to get rich metadata
                logger.info("Analyzing video with SoccerNet...")
                analysis_result = self.analyzer.process_clip(video_path, metadata)
                
                if analysis_result["success"]:
                    # Get the GPT prompt from the analysis
                    prompt = analysis_result["gpt_prompt"]
                    analysis = analysis_result["analysis"]
                    report = analysis_result["report"]
                    logger.info("SoccerNet analysis successful")
                else:
                    # Fallback to basic prompt
                    logger.warning("SoccerNet analysis failed, using basic prompt")
                    prompt = self.create_basic_prompt(metadata)
                    analysis = {"error": analysis_result.get("error", "Unknown error")}
                    report = f"SoccerNet analysis failed: {analysis_result.get('error', 'Unknown error')}"
            else:
                # Use basic prompt without SoccerNet
                logger.info("SoccerNet analyzer not available, using basic prompt")
                prompt = self.create_basic_prompt(metadata)
                analysis = {"warning": "SoccerNet analyzer not available"}
                report = "No SoccerNet analysis available"
            
            # Commentary generation phase
            logger.info("Generating commentary with GPT...")
            commentary = self.generate_commentary_with_gpt(prompt)
            
            # Return the results
            return {
                "video_path": video_path,
                "metadata": metadata,
                "commentary": commentary,
                "analysis": analysis,
                "report": report,
                "prompt": prompt,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing clip {video_path}: {str(e)}")
            
            # Attempt to build a fallback commentary from metadata
            fallback = "An exciting soccer moment!"
            if "label" in metadata:
                event_type = metadata["label"]
                if "team" in metadata:
                    team = metadata["team"]
                    fallback = f"An exciting {event_type.lower()} attempt by {team}!"
                else:
                    fallback = f"An exciting {event_type.lower()} moment in the match!"
            
            return {
                "video_path": video_path,
                "metadata": metadata,
                "commentary": fallback,
                "analysis": {"error": str(e)},
                "report": f"Error: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def create_basic_prompt(self, metadata: Dict) -> str:
        """
        Create a basic prompt for GPT without SoccerNet analysis.
        
        Args:
            metadata: Dictionary with event metadata
            
        Returns:
            Prompt string
        """
        # Extract basic information
        if "label" in metadata:
            event_type = metadata["label"]
        else:
            event_type = "soccer event"
        
        if "team" in metadata:
            team = metadata["team"]
        else:
            team = "the team"
        
        if "gameTime" in metadata:
            game_time = metadata["gameTime"]
        else:
            game_time = "during the match"
        
        # Create the prompt
        prompt = f"""
You are a professional soccer commentator tasked with describing a specific game moment from a soccer match.

### Event Information
- Event Type: {event_type}
- Team: {team}
- Game Time: {game_time}

### Commentary Task
Generate exciting, professional soccer commentary for this {event_type} by {team} at {game_time}. Your commentary should:

1. Describe the build-up play leading to this moment
2. Comment on the players' positioning and movement
3. Describe the specific technique and execution
4. Add appropriate emotion based on the importance of the moment

Your commentary should be 1-2 sentences vivid, and authentic to professional soccer broadcasting.
"""
        
        return prompt