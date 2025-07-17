import os
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple

# --- Dependency Checks and Imports ---
# This setup ensures the module can be imported even if optional dependencies are missing,
# providing clear guidance to the user.

try:
    from moviepy.editor import (
        VideoFileClip,
        AudioFileClip,
        ImageClip,
        concatenate_videoclips,
        TextClip,
        CompositeVideoClip,
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: 'moviepy' library not found. Video generation will be disabled. Install with 'pip install moviepy'.")

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: 'gtts' library not found. Narration generation will be disabled. Install with 'pip install gTTS'.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: 'requests' library not found. Visual search will be disabled. Install with 'pip install requests'.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)


# --- Mock/Placeholder for AI Agent ---
class MockAgent:
    """A mock AI agent to simulate research and script generation."""
    def run(self, task: str) -> str:
        logger.info(f"MockAgent received task: {task[:100]}...")
        if "generate a documentary script" in task:
            return json.dumps({
                "title": "The Rise of Artificial Intelligence",
                "scenes": [
                    {
                        "scene_number": 1,
                        "description": "Vintage black and white footage of early computers like ENIAC. Scientists in lab coats.",
                        "narration": "In the mid-20th century, the seeds of a revolution were sown. Machines, once simple calculators, began to show glimmers of a new potential."
                    },
                    {
                        "scene_number": 2,
                        "description": "A montage of AI in movies: HAL 9000, The Terminator, modern friendly robots. Quick cuts.",
                        "narration": "From science fiction fantasy to everyday reality, artificial intelligence has captured our imagination and transformed our world in ways previously thought impossible."
                    },
                    {
                        "scene_number": 3,
                        "description": "Modern data centers with blinking lights. Abstract animations of neural networks and data flowing.",
                        "narration": "Today, deep learning and neural networks, inspired by the human brain, power everything from search engines to medical diagnostics, processing vast amounts of data in the blink of an eye."
                    }
                ]
            })
        return ""


class DocumentaryGenerator:
    """
    A class to generate documentary-style videos from a text prompt.
    """

    def __init__(
        self,
        agent: Any = None,
        output_dir: str = "documentaries",
        temp_dir: str = "temp_assets"
    ):
        """
        Initializes the DocumentaryGenerator.

        Args:
            agent (Any): An AI agent instance capable of research and script writing.
                         If None, a mock agent is used for demonstration.
            output_dir (str): Directory to save the final video.
            temp_dir (str): Directory to store temporary assets like audio and visuals.
        """
        if not MOVIEPY_AVAILABLE or not GTTS_AVAILABLE or not REQUESTS_AVAILABLE:
            raise ImportError("One or more required libraries (moviepy, gtts, requests) are not installed.")

        self.agent = agent or MockAgent()
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        logger.info(f"DocumentaryGenerator initialized. Output will be saved to '{self.output_dir}'.")

    def _generate_script(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Researches a topic and generates a structured documentary script.

        Args:
            topic (str): The topic for the documentary.

        Returns:
            A dictionary representing the structured script, or None on failure.
        """
        logger.info(f"Generating script for topic: '{topic}'...")
        prompt = f"Generate a documentary script about '{topic}'. The script should have a title and a list of scenes. Each scene needs a 'description' for visuals and a 'narration' text. Return as a JSON object."
        try:
            response = self.agent.run(prompt)
            script = json.loads(response)
            if "title" in script and "scenes" in script:
                logger.info("Script generated successfully.")
                return script
            else:
                logger.error("Generated script is missing required 'title' or 'scenes' keys.")
                return None
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Failed to generate or parse script: {e}")
            return None

    def _generate_narration(self, text: str, filename: str) -> Optional[str]:
        """
        Generates an MP3 narration file from text using gTTS.

        Args:
            text (str): The text to be converted to speech.
            filename (str): The name of the output MP3 file.

        Returns:
            The path to the generated audio file, or None on failure.
        """
        logger.info(f"Generating narration for '{filename}'...")
        try:
            tts = gTTS(text=text, lang='en')
            filepath = os.path.join(self.temp_dir, filename)
            tts.save(filepath)
            logger.info(f"Narration saved to '{filepath}'.")
            return filepath
        except Exception as e:
            logger.error(f"Failed to generate narration: {e}")
            return None

    def _find_visuals_for_scene(self, description: str) -> str:
        """
        Finds a relevant image or video clip for a scene description.

        Note: This is a placeholder. A real implementation would use stock footage APIs
        (like Pexels, Pixabay) or a video generation model.

        Args:
            description (str): The description of the visuals needed for the scene.

        Returns:
            A path to a local visual asset (image or video).
        """
        logger.info(f"Finding visuals for description: '{description[:50]}...'")
        # --- Placeholder Logic ---
        # In a real app, you'd search an API based on keywords from the description.
        # For this demo, we'll use a single placeholder image.
        placeholder_image_path = os.path.join(self.temp_dir, "placeholder.jpg")
        if not os.path.exists(placeholder_image_path):
            try:
                # Download a generic placeholder image
                response = requests.get("https://images.pexels.com/photos/3408744/pexels-photo-3408744.jpeg", stream=True)
                response.raise_for_status()
                with open(placeholder_image_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                logger.error(f"Could not download placeholder image: {e}")
                # Fallback to creating a blank image
                from PIL import Image
                Image.new('RGB', (1280, 720), color = 'darkgrey').save(placeholder_image_path)

        return placeholder_image_path

    def generate_video(self, topic: str, output_filename: Optional[str] = None) -> Optional[str]:
        """
        Orchestrates the entire process of generating a documentary video.

        Args:
            topic (str): The topic of the documentary.
            output_filename (str): The desired filename for the output video.
                                   If None, it's generated from the topic.

        Returns:
            The path to the final generated video, or None on failure.
        """
        # 1. Generate Script
        script = self._generate_script(topic)
        if not script:
            return None

        # 2. Generate Full Narration
        full_narration_text = " ".join([scene['narration'] for scene in script['scenes']])
        narration_path = self._generate_narration(full_narration_text, "full_narration.mp3")
        if not narration_path:
            return None

        main_audio_clip = AudioFileClip(narration_path)
        
        # 3. Create Video Clips for Each Scene
        scene_clips = []
        current_narration_time = 0
        
        for scene in script['scenes']:
            # Estimate duration of this scene's narration
            # A more accurate method would be to generate audio per scene
            words_in_scene = len(scene['narration'].split())
            estimated_duration = max(1.0, words_in_scene / 2.5) # Approx. 2.5 words per second

            # Find visuals
            visual_path = self._find_visuals_for_scene(scene['description'])
            
            # Create video clip from the visual
            if visual_path.lower().endswith(('.mp4', '.mov')):
                clip = VideoFileClip(visual_path).set_duration(estimated_duration)
            else: # Assume image
                clip = ImageClip(visual_path).set_duration(estimated_duration)
            
            # Add a simple title overlay for context
            txt_clip = TextClip(
                script['title'],
                fontsize=40,
                color='white',
                font='Arial-Bold',
                bg_color='black',
                size=(clip.size[0] * 0.8, None)
            ).set_position(('center', 'top')).set_duration(min(3.0, estimated_duration))

            # Add scene description as subtitle
            subtitle_clip = TextClip(
                scene['narration'],
                fontsize=24,
                color='white',
                bg_color='rgba(0,0,0,0.5)',
                size=(clip.size[0] * 0.9, None),
                method='caption'
            ).set_position(('center', 'bottom')).set_duration(estimated_duration)

            scene_video = CompositeVideoClip([clip, txt_clip, subtitle_clip])
            scene_clips.append(scene_video.set_fps(24)) # Standardize FPS
            
            current_narration_time += estimated_duration

        if not scene_clips:
            logger.error("No video clips were created for the scenes.")
            return None

        # 4. Concatenate and Finalize
        final_video = concatenate_videoclips(scene_clips)
        
        # Ensure video duration matches audio duration
        final_video = final_video.set_duration(main_audio_clip.duration)
        final_video = final_video.set_audio(main_audio_clip)

        # 5. Write to File
        if not output_filename:
            sanitized_title = re.sub(r'\s+', '_', script['title'].lower())
            sanitized_title = re.sub(r'[^a-z0-9_]', '', sanitized_title)
            output_filename = f"{sanitized_title}.mp4"
            
        final_video_path = os.path.join(self.output_dir, output_filename)
        
        logger.info(f"Writing final video to '{final_video_path}'...")
        final_video.write_videofile(final_video_path, codec='libx264', audio_codec='aac')
        
        # 6. Cleanup
        try:
            os.remove(narration_path)
            logger.info("Cleaned up temporary audio file.")
        except OSError as e:
            logger.warning(f"Could not clean up temporary file {narration_path}: {e}")

        return final_video_path


if __name__ == '__main__':
    logger.info("--- DocumentaryGenerator Demonstration ---")
    
    try:
        # Initialize the generator
        doc_gen = DocumentaryGenerator()
        
        # Generate a documentary
        video_path = doc_gen.generate_video(topic="The Rise of Artificial Intelligence")
        
        if video_path:
            logger.info(f"✅ Documentary generation successful! Video saved at: {video_path}")
        else:
            logger.error("❌ Documentary generation failed.")
            
    except ImportError as e:
        logger.error(f"Failed to run demonstration due to missing dependencies: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the demonstration: {e}")

```
