import os
import sys
import cv2
import dlib
import numpy as np
import logging
from typing import List, Tuple, Optional, Any
from tqdm import tqdm

# --- Dependency Checks and Imports ---
# This setup ensures the module can be imported even if optional dependencies are missing,
# providing clear guidance to the user.

try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: 'moviepy' library not found. Audio extraction/injection will be disabled. Install with 'pip install moviepy'.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions for Video Processing ---

def _extract_frames(video_path: str, output_folder: str) -> bool:
    """
    Extracts all frames from a video file into a specified folder.

    Args:
        video_path (str): The path to the input video file.
        output_folder (str): The directory to save the extracted frames.

    Returns:
        bool: True if frames were extracted successfully, False otherwise.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found at '{video_path}'")
        return False

    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Extracting {frame_count} frames from '{os.path.basename(video_path)}'...")
    
    success, image = cap.read()
    count = 0
    while success:
        frame_path = os.path.join(output_folder, f"frame_{count:06d}.png")
        cv2.imwrite(frame_path, image)
        success, image = cap.read()
        count += 1
        
    cap.release()
    logger.info("Frame extraction complete.")
    return True

def _extract_audio(video_path: str, audio_output_path: str) -> Optional[str]:
    """
    Extracts the audio from a video file.

    Args:
        video_path (str): The path to the input video file.
        audio_output_path (str): The path to save the extracted audio file.

    Returns:
        Optional[str]: The path to the audio file if successful, otherwise None.
    """
    if not MOVIEPY_AVAILABLE:
        logger.warning("MoviePy not available, cannot extract audio.")
        return None
    try:
        logger.info(f"Extracting audio from '{os.path.basename(video_path)}'...")
        video_clip = VideoFileClip(video_path)
        if video_clip.audio:
            video_clip.audio.write_audiofile(audio_output_path, codec='mp3')
            logger.info(f"Audio extracted and saved to '{audio_output_path}'.")
            return audio_output_path
        else:
            logger.warning("Video has no audio track.")
            return None
    except Exception as e:
        logger.error(f"Failed to extract audio: {e}")
        return None

def _combine_frames_to_video(frames_folder: str, output_path: str, fps: float, audio_path: Optional[str] = None):
    """
    Combines a sequence of image frames into a video file.

    Args:
        frames_folder (str): The directory containing the image frames.
        output_path (str): The path for the output video file.
        fps (float): The frames per second for the output video.
        audio_path (Optional[str]): Path to an audio file to add to the video.
    """
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.png')])
    if not frame_files:
        logger.error("No frames found in the specified folder.")
        return

    # Get frame dimensions from the first frame
    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = first_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = os.path.join(os.path.dirname(output_path), "temp_video_no_audio.mp4")
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    logger.info(f"Combining {len(frame_files)} frames into video...")
    for frame_file in tqdm(frame_files, desc="Combining Frames"):
        frame_path = os.path.join(frames_folder, frame_file)
        video_writer.write(cv2.imread(frame_path))
    
    video_writer.release()

    if audio_path and MOVIEPY_AVAILABLE:
        logger.info("Adding audio track to the video...")
        video_clip = VideoFileClip(temp_video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
        os.remove(temp_video_path) # Clean up temp video
    else:
        os.rename(temp_video_path, output_path)
    
    logger.info(f"Final video saved to '{output_path}'.")


class VideoManipulator:
    """
    A class to encapsulate advanced video manipulation functionalities.
    """

    def __init__(
        self,
        dlib_predictor_path: str,
        upscaler_model_path: Optional[str] = None,
        colorizer_model_paths: Optional[Dict[str, str]] = None,
        temp_dir: str = "temp_video_processing"
    ):
        """
        Initializes the VideoManipulator.

        Args:
            dlib_predictor_path (str): Path to dlib's 68-point facial landmark predictor model file.
                                       (e.g., 'shape_predictor_68_face_landmarks.dat')
            upscaler_model_path (Optional[str]): Path to a pre-trained DNN upscaling model (e.g., 'EDSR_x4.pb').
            colorizer_model_paths (Optional[Dict[str, str]]): A dictionary with paths to colorization model files.
                                                              Expected keys: 'prototxt', 'model', 'points'.
            temp_dir (str): A directory for storing intermediate files like frames and audio.
        """
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        # --- Face Swap Dependencies ---
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(dlib_predictor_path)
        except Exception as e:
            logger.error(f"Failed to load dlib models from '{dlib_predictor_path}': {e}")
            raise

        # --- Upscaling Dependencies ---
        self.upscaler = None
        if upscaler_model_path:
            try:
                self.upscaler = cv2.dnn_superres.DnnSuperResImpl_create()
                self.upscaler.readModel(upscaler_model_path)
                # Assuming model name format like 'EDSR_x4.pb'
                model_name = os.path.basename(upscaler_model_path).split('_')[0].lower()
                scale = int(re.search(r'x(\d+)', os.path.basename(upscaler_model_path)).group(1))
                self.upscaler.setModel(model_name, scale)
            except Exception as e:
                logger.error(f"Failed to load upscaler model from '{upscaler_model_path}': {e}")

        # --- Colorization Dependencies ---
        self.colorizer = None
        if colorizer_model_paths:
            try:
                self.colorizer = cv2.dnn.readNetFromCaffe(
                    colorizer_model_paths['prototxt'], colorizer_model_paths['model']
                )
                pts = np.load(colorizer_model_paths['points'])
                class8 = self.colorizer.getLayer(self.colorizer.getLayerId('class8_ab')).blobs[0].reshape((1, 313))
                class8[:, :] = pts.transpose()
                self.colorizer.getLayer(self.colorizer.getLayerId('conv8_313_rh')).blobs[0][:, :, 0, 0] = 2.606
            except Exception as e:
                logger.error(f"Failed to load colorizer models: {e}")

    def _get_facial_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detects face and its landmarks in a given image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray)
        if not faces:
            return None
        # Assume the largest face is the target
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        landmarks = self.landmark_predictor(gray, face)
        return np.array([[p.x, p.y] for p in landmarks.parts()])

    def swap_face(self, source_image_path: str, target_video_path: str, output_path: str):
        """
        Swaps the face from a source image onto the largest face in a target video.
        """
        logger.info("--- Starting Face Swap Process ---")
        source_image = cv2.imread(source_image_path)
        if source_image is None:
            raise FileNotFoundError(f"Source image not found at '{source_image_path}'")

        source_landmarks = self._get_facial_landmarks(source_image)
        if source_landmarks is None:
            raise ValueError("No face detected in the source image.")
        
        source_hull = cv2.convexHull(source_landmarks)

        # Setup temporary directories
        frames_dir = os.path.join(self.temp_dir, "frames")
        processed_frames_dir = os.path.join(self.temp_dir, "processed_frames")
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        # Extract frames and audio
        if not _extract_frames(target_video_path, frames_dir):
            return
        audio_path = _extract_audio(target_video_path, os.path.join(self.temp_dir, "audio.mp3"))
        
        cap = cv2.VideoCapture(target_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        for frame_file in tqdm(frame_files, desc="Swapping Faces"):
            frame_path = os.path.join(frames_dir, frame_file)
            target_frame = cv2.imread(frame_path)
            
            target_landmarks = self._get_facial_landmarks(target_frame)
            if target_landmarks is None:
                # If no face is detected, just copy the original frame
                cv2.imwrite(os.path.join(processed_frames_dir, frame_file), target_frame)
                continue

            # Delaunay Triangulation
            rect = (0, 0, source_image.shape[1], source_image.shape[0])
            dt = cv2.Subdiv2D(rect)
            dt.insert(list(source_landmarks.astype(float)))
            
            output_frame = np.copy(target_frame)
            
            # Warp triangles
            for triangle in dt.getTriangleList():
                pts1 = np.array([triangle[0:2], triangle[2:4], triangle[4:6]], dtype=np.int32)
                
                indices = []
                for p in pts1:
                    for i, sp in enumerate(source_landmarks):
                        if sp[0] == p[0] and sp[1] == p[1]:
                            indices.append(i)
                            break
                if len(indices) != 3:
                    continue

                pts2 = target_landmarks[indices]

                # Affine transform
                M = cv2.getAffineTransform(np.float32(pts1), np.float32(pts2))
                
                # Warp the triangle
                r1 = cv2.boundingRect(np.float32([pts1]))
                r2 = cv2.boundingRect(np.float32([pts2]))
                
                t1_cropped = source_image[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
                
                # Offset points
                pts1_offset = pts1 - (r1[0], r1[1])
                pts2_offset = pts2 - (r2[0], r2[1])

                warped_triangle = cv2.warpAffine(t1_cropped, M, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                
                # Create mask for seamless cloning
                mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
                cv2.fillConvexPoly(mask, np.int32(pts2_offset), (1.0, 1.0, 1.0), 16, 0)
                
                # Apply the warped triangle
                output_frame[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = output_frame[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + warped_triangle * mask

            # Seamlessly clone the center of the face
            target_hull = cv2.convexHull(target_landmarks)
            center = tuple(np.mean(target_landmarks, axis=0).astype(int))
            output_frame = cv2.seamlessClone(output_frame, target_frame, cv2.fillConvexPoly(np.zeros_like(target_frame), target_hull, (255, 255, 255)), center, cv2.NORMAL_CLONE)
            
            cv2.imwrite(os.path.join(processed_frames_dir, frame_file), output_frame)

        # Combine frames back to video
        _combine_frames_to_video(processed_frames_dir, output_path, fps, audio_path)
        logger.info("--- Face Swap Process Finished ---")

    def upscale_video(self, video_path: str, output_path: str):
        """
        Upscales a video to a higher resolution using a pre-trained DNN model.
        """
        if not self.upscaler:
            raise RuntimeError("Upscaler model not initialized. Provide a valid model path during instantiation.")
        
        logger.info("--- Starting Video Upscaling Process ---")
        
        frames_dir = os.path.join(self.temp_dir, "frames_to_upscale")
        processed_frames_dir = os.path.join(self.temp_dir, "upscaled_frames")
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        if not _extract_frames(video_path, frames_dir):
            return
        audio_path = _extract_audio(video_path, os.path.join(self.temp_dir, "audio_upscale.mp3"))
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        for frame_file in tqdm(frame_files, desc="Upscaling Frames"):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            upscaled_frame = self.upscaler.upsample(frame)
            cv2.imwrite(os.path.join(processed_frames_dir, frame_file), upscaled_frame)

        _combine_frames_to_video(processed_frames_dir, output_path, fps, audio_path)
        logger.info("--- Video Upscaling Process Finished ---")

    def colorize_video(self, video_path: str, output_path: str):
        """
        Colorizes a black and white video using a pre-trained DNN model.
        """
        if not self.colorizer:
            raise RuntimeError("Colorizer model not initialized. Provide valid model paths during instantiation.")

        logger.info("--- Starting Video Colorization Process ---")
        
        frames_dir = os.path.join(self.temp_dir, "frames_to_colorize")
        processed_frames_dir = os.path.join(self.temp_dir, "colorized_frames")
        os.makedirs(processed_frames_dir, exist_ok=True)
        
        if not _extract_frames(video_path, frames_dir):
            return
        audio_path = _extract_audio(video_path, os.path.join(self.temp_dir, "audio_colorize.mp3"))
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

        for frame_file in tqdm(frame_files, desc="Colorizing Frames"):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            # Normalize and convert to Lab color space
            normalized = frame.astype("float32") / 255.0
            lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
            
            resized = cv2.resize(lab, (224, 224))
            L = cv2.split(resized)[0]
            L -= 50 # Mean subtraction
            
            # Predict 'a' and 'b' channels
            self.colorizer.setInput(cv2.dnn.blobFromImage(L))
            ab = self.colorizer.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))
            
            # Merge channels and convert back to BGR
            L = cv2.split(lab)[0]
            colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
            colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
            colorized = np.clip(colorized * 255, 0, 255).astype("uint8")
            
            cv2.imwrite(os.path.join(processed_frames_dir, frame_file), colorized)

        _combine_frames_to_video(processed_frames_dir, output_path, fps, audio_path)
        logger.info("--- Video Colorization Process Finished ---")


if __name__ == '__main__':
    # --- Demonstration ---
    # This block demonstrates the class's usage. It requires the user to provide
    # their own media files and pre-trained models.
    
    logger.info("--- VideoManipulator Demonstration ---")
    logger.warning("This script is a demonstration and requires user-provided assets and models to run.")

    # --- REQUIRED USER SETUP ---
    # 1. Download dlib's facial landmark predictor:
    #    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 (unzip it)
    # 2. Provide a source image with a clear face.
    # 3. Provide a target video with a clear face.
    # 4. (Optional) For upscaling, download a model like EDSR from OpenCV's model zoo.
    # 5. (Optional) For colorization, download the required Caffe models.

    DLIB_MODEL = "shape_predictor_68_face_landmarks.dat"
    SOURCE_IMAGE = "path/to/your/source_face.jpg"
    TARGET_VIDEO = "path/to/your/target_video.mp4"
    BW_VIDEO = "path/to/your/black_and_white_video.mp4"
    UPSCALE_MODEL = "path/to/your/EDSR_x4.pb" # Optional
    
    # Check if required files exist before running
    if not os.path.exists(DLIB_MODEL) or not os.path.exists(SOURCE_IMAGE) or not os.path.exists(TARGET_VIDEO):
        logger.error("Please set the paths for DLIB_MODEL, SOURCE_IMAGE, and TARGET_VIDEO to run the demo.")
        sys.exit(1)

    try:
        manipulator = VideoManipulator(dlib_predictor_path=DLIB_MODEL, upscaler_model_path=UPSCALE_MODEL)
        
        # --- Run Face Swap ---
        logger.info("\n--- Demonstrating Face Swap ---")
        manipulator.swap_face(
            source_image_path=SOURCE_IMAGE,
            target_video_path=TARGET_VIDEO,
            output_path="output_face_swapped.mp4"
        )
        
        # --- Run Upscaling (if model provided) ---
        if manipulator.upscaler:
            logger.info("\n--- Demonstrating Video Upscaling ---")
            manipulator.upscale_video(
                video_path=TARGET_VIDEO, # Using target video for demo
                output_path="output_upscaled.mp4"
            )
        else:
            logger.warning("\nSkipping upscale demo: Upscaler model not provided or failed to load.")

        # --- Run Colorization (if models provided and BW video exists) ---
        # Note: Colorizer setup is more complex and commented out by default.
        # colorizer_models = {
        #     'prototxt': 'path/to/colorization_deploy_v2.prototxt',
        #     'model': 'path/to/colorization_release_v2.caffemodel',
        #     'points': 'path/to/pts_in_hull.npy'
        # }
        # if all(os.path.exists(p) for p in colorizer_models.values()) and os.path.exists(BW_VIDEO):
        #     logger.info("\n--- Demonstrating Video Colorization ---")
        #     colorizer_manipulator = VideoManipulator(dlib_predictor_path=DLIB_MODEL, colorizer_model_paths=colorizer_models)
        #     colorizer_manipulator.colorize_video(
        #         video_path=BW_VIDEO,
        #         output_path="output_colorized.mp4"
        #     )
        # else:
        #     logger.warning("\nSkipping colorization demo: Colorizer models or B&W video not provided.")

    except Exception as e:
        logger.error(f"An error occurred during the demonstration: {e}")

```
