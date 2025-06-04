import os
import shutil
import time
from PySide6.QtCore import QObject, Signal, QThread, Slot

# Conceptual flags (as per document)
DEOLDIFY_INSTALLED_CONCEPTUALLY = False # Simulate as not installed
TORCH_AVAILABLE = False # Simulate as not installed
MOVIEPY_AVAILABLE = True # Assume moviepy will be installed

try:
    from moviepy.editor import ImageClip, concatenate_videoclips
    # Attempt to import Pillow to check its availability for the test script primarily
    from PIL import Image, ImageDraw
    PILLOW_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False # If moviepy fails, assume it's unavailable
    PILLOW_AVAILABLE = False # If Pillow import fails alongside

class VideoTaskWorker(QObject):
    '''
    Worker object to perform video tasks in a separate thread.
    '''
    progress_updated = Signal(int)
    task_completed = Signal(str, str)  # output_path, task_type
    task_error = Signal(str, str)      # error_message, task_type
    status_message = Signal(str)       # intermediate status messages

    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params
        self._is_running = True

    @Slot()
    def run(self):
        '''
        Main entry point for the worker's task.
        '''
        if not self._is_running:
            self.status_message.emit(f"Task {self.task_type} was cancelled before starting.")
            return

        self.status_message.emit(f"Starting task: {self.task_type}...")
        try:
            if self.task_type == "colorize_video":
                self._colorize_video_simulated()
            elif self.task_type == "images_to_video":
                self._images_to_video()
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
        except Exception as e:
            # Ensure task_error is emitted only if the task was not intentionally stopped
            if self._is_running:
                self.task_error.emit(str(e), self.task_type)
            else:
                self.status_message.emit(f"Task {self.task_type} was stopped, error emission suppressed: {str(e)}")
        finally:
            if self._is_running:
                self.status_message.emit(f"Task {self.task_type} finished processing phase.")
            else:
                self.status_message.emit(f"Task {self.task_type} was stopped during processing.")


    def _init_deoldify_simulation(self):
        self.status_message.emit("Checking DeOldify (simulated) requirements...")
        time.sleep(0.5)
        if not DEOLDIFY_INSTALLED_CONCEPTUALLY or not TORCH_AVAILABLE:
            self.status_message.emit("Simulated DeOldify: Full DeOldify components not found. Proceeding with simulation.")
            return False
        self.status_message.emit("Simulated DeOldify: Components conceptually available.")
        return True

    def _colorize_video_simulated(self):
        input_video_path = self.params.get("input_video_path")
        output_path = self.params.get("output_path")

        if not input_video_path or not output_path:
            raise ValueError("Missing input or output path for colorization.")

        if not os.path.exists(input_video_path):
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        self.status_message.emit(f"Simulating colorization for: {os.path.basename(input_video_path)}")
        self._init_deoldify_simulation()

        total_steps = 10
        for i in range(total_steps):
            if not self._is_running:
                self.status_message.emit("Colorization task cancelled during simulation.")
                return
            time.sleep(0.3)
            self.progress_updated.emit(int((i + 1) / total_steps * 100))

        if not self._is_running:
            self.status_message.emit("Colorization task cancelled before file copy.")
            return

        try:
            output_dir = os.path.dirname(output_path)
            if output_dir: # Ensure output directory exists only if it's not root
                 os.makedirs(output_dir, exist_ok=True)
            shutil.copy(input_video_path, output_path)
            self.status_message.emit(f"Simulated colorization complete. Output: {output_path}")
            self.task_completed.emit(output_path, self.task_type)
        except Exception as e:
            raise RuntimeError(f"Failed to copy video for simulation: {str(e)}")


    def _init_moviepy(self):
        self.status_message.emit("Checking MoviePy availability...")
        time.sleep(0.2)
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy library is not installed or accessible. Please install it to use this feature.")
        self.status_message.emit("MoviePy is available.")
        return True

    def _images_to_video(self):
        if not self._init_moviepy():
            return # Error should have been raised by _init_moviepy and caught by run()

        image_paths = self.params.get("image_paths")
        output_path = self.params.get("output_path")
        fps = self.params.get("fps", 24)
        duration_per_image = self.params.get("duration_per_image")

        if not image_paths or not output_path:
            raise ValueError("Missing image paths or output path for video creation.")
        if not all(os.path.exists(p) for p in image_paths):
            raise FileNotFoundError("One or more input images not found.")

        self.status_message.emit(f"Starting video creation from {len(image_paths)} images.")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        clips = []
        video = None # Define video variable outside try block for broader scope in finally
        total_images = len(image_paths)

        try:
            for i, img_path in enumerate(image_paths):
                if not self._is_running:
                    raise Exception("Task cancelled during image processing.")

                self.status_message.emit(f"Processing image {i+1}/{total_images}: {os.path.basename(img_path)}")
                clip = ImageClip(img_path)
                if duration_per_image and float(duration_per_image) > 0:
                    clip = clip.set_duration(float(duration_per_image))
                else:
                    clip = clip.set_duration(1) # Default to 1 second per image

                clips.append(clip)
                self.progress_updated.emit(int(((i + 0.5) / total_images) * 100))

            if not self._is_running:
                 raise Exception("Task cancelled after image processing, before concatenation.")

            if not clips:
                raise ValueError("No image clips were created. Cannot generate video.")

            self.status_message.emit("Concatenating video clips...")
            video = concatenate_videoclips(clips, method="compose")

            if not self._is_running:
                raise Exception("Task cancelled after concatenation, before writing video file.")

            self.status_message.emit(f"Writing video file to {output_path} with FPS={fps}...")
            video.write_videofile(output_path, fps=fps, codec="libx264", logger=None)

        except Exception as e:
            if "No such file or directory" in str(e) and "ffmpeg" in str(e).lower():
                 raise RuntimeError("FFMPEG not found. Please install ffmpeg and ensure it's in your system's PATH.")
            raise RuntimeError(f"Failed to create video: {str(e)}") # Re-raise to be caught by run()
        finally:
            for clip_obj in clips:
                try:
                    clip_obj.close()
                except Exception: pass
            if video is not None and hasattr(video, 'close'):
                try:
                    video.close()
                except Exception: pass

        if not self._is_running:
            # This will be caught by run's exception handler, which then checks _is_running again
            raise Exception("Task cancelled before final completion signal.")

        self.progress_updated.emit(100)
        self.task_completed.emit(output_path, self.task_type)

    def stop(self):
        self.status_message.emit(f"Attempting to stop task: {self.task_type}...")
        self._is_running = False


class VideoAgent(QObject):
    overall_task_progress = Signal(int)
    overall_task_finished = Signal(str, str, str)
    overall_task_error = Signal(str, str)
    overall_status_update = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thread = None
        self.worker = None

    def _start_task(self, task_type, params):
        if self.thread is not None and self.thread.isRunning():
            self.overall_task_error.emit("Another task is already in progress.", task_type)
            return

        self.thread = QThread(self)
        self.worker = VideoTaskWorker(task_type, params)
        self.worker.moveToThread(self.thread)

        self.worker.progress_updated.connect(self.overall_task_progress)
        self.worker.status_message.connect(self.overall_status_update)
        self.worker.task_completed.connect(self._on_task_completed)
        self.worker.task_error.connect(self._on_task_error)

        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self._on_thread_finished)

        self.overall_status_update.emit(f"Video task '{task_type}' initiated.")
        self.thread.start()

    @Slot(str, str)
    def _on_task_completed(self, output_path, task_type):
        abs_output_path = os.path.abspath(output_path) if output_path else 'N/A'
        message = f"{task_type.replace('_', ' ').capitalize()} completed successfully. Output: {abs_output_path}"
        self.overall_task_finished.emit(abs_output_path, task_type, message)
        if self.thread and self.thread.isRunning():
            self.thread.quit()

    @Slot(str, str)
    def _on_task_error(self, error_message, task_type):
        self.overall_task_error.emit(error_message, task_type)
        if self.thread and self.thread.isRunning():
            self.thread.quit()

    @Slot()
    def _on_thread_finished(self):
        self.overall_status_update.emit("Video processing thread has finished.")
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        if self.thread:
            self.thread.deleteLater()
            self.thread = None

    def colorize_video(self, input_video_path, output_path, render_factor_simulated=21):
        params = {"input_video_path": input_video_path, "output_path": output_path, "render_factor": render_factor_simulated}
        self._start_task("colorize_video", params)

    def images_to_video(self, image_paths, output_path, fps=24, duration_per_image=None):
        params = {"image_paths": image_paths, "output_path": output_path, "fps": fps, "duration_per_image": duration_per_image}
        self._start_task("images_to_video", params)

    def cancel_current_task(self):
        if self.worker and self.thread and self.thread.isRunning():
            self.overall_status_update.emit("Attempting to cancel current video task...")
            self.worker.stop()
            # self.thread.quit() # Thread will quit once worker's run method finishes due to stop flag
            # self.thread.wait(500) # Brief wait, but not strictly necessary as run() should exit promptly
        else:
            self.overall_status_update.emit("No active video task to cancel.")

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
    import sys

    app = QApplication(sys.argv)
    agent = VideoAgent()
    current_test_stage = 1

    def handle_progress(p): print(f"Agent Progress: {p}%")
    def handle_status(status_msg): print(f"Agent Status: {status_msg}")

    def handle_finish_or_error(item, task_type, message=None):
        global current_test_stage
        if message: # Success
            print(f"Agent Finish for {task_type}: {message} (Path: {item})")
        else: # Error
            print(f"Agent Error for {task_type}: {item}")

        if current_test_stage == 1:
            current_test_stage = 2
            print("\n--- Running Test Stage 2: Images to Video ---\n")
            run_images_to_video_test()
        else:
            print("\n--- All tests finished. Quitting. ---\n")
            QTimer.singleShot(100, app.quit)

    agent.overall_task_progress.connect(handle_progress)
    agent.overall_task_finished.connect(lambda path, task, msg: handle_finish_or_error(path, task, msg))
    agent.overall_task_error.connect(lambda err, task: handle_finish_or_error(err, task))
    agent.overall_status_update.connect(handle_status)

    temp_base_dir = "temp_video_processing_test_files"
    if os.path.exists(temp_base_dir):
        shutil.rmtree(temp_base_dir)
    os.makedirs(temp_base_dir, exist_ok=True)

    dummy_input_dir = os.path.join(temp_base_dir, "input")
    dummy_output_dir = os.path.join(temp_base_dir, "output")
    os.makedirs(dummy_input_dir, exist_ok=True)
    os.makedirs(dummy_output_dir, exist_ok=True)

    def run_colorize_test():
        print("\n--- Running Test Stage 1: Colorize Video (Simulated) ---\n")
        dummy_video_file = os.path.join(dummy_input_dir, "dummy_bw_video.mp4")
        with open(dummy_video_file, "w") as f:
            f.write("This is a dummy black and white video file content.")

        colorize_output_path = os.path.join(dummy_output_dir, "colorized_video.mp4")
        agent.colorize_video(dummy_video_file, colorize_output_path)

    def run_images_to_video_test():
        dummy_image_dir = os.path.join(dummy_input_dir, "images")
        os.makedirs(dummy_image_dir, exist_ok=True)
        dummy_images = []

        if PILLOW_AVAILABLE:
            try:
                for i in range(3):
                    pth = os.path.join(dummy_image_dir, f"img_{i+1}.png")
                    img = Image.new('RGB', (120, 80), color = (200, i*40, 100))
                    draw = ImageDraw.Draw(img)
                    draw.text((5,5), f"Image {i+1}", fill=(0,0,0))
                    img.save(pth)
                    dummy_images.append(pth)
                print(f"Created {len(dummy_images)} dummy PNG images using Pillow.")
            except Exception as e:
                print(f"Error creating dummy images with Pillow: {e}. Skipping images_to_video test.")
                handle_finish_or_error("Pillow image creation failed.", "images_to_video_setup")
                return
        else:
            print("Pillow not available. Cannot create dummy PNGs for images_to_video test.")
            handle_finish_or_error("Pillow not available.", "images_to_video_setup")
            return

        if not MOVIEPY_AVAILABLE:
            print("Skipping images_to_video test: MoviePy library not available.")
            handle_finish_or_error("MoviePy not available.", "images_to_video_setup")
            return

        if not dummy_images:
            print("Skipping images_to_video test: No dummy images were created.")
            handle_finish_or_error("No dummy images.", "images_to_video_setup")
            return

        slideshow_output_path = os.path.join(dummy_output_dir, "slideshow.mp4")
        agent.images_to_video(dummy_images, slideshow_output_path, fps=24, duration_per_image=0.5)

    run_colorize_test() # Start the first test
    sys.exit(app.exec())
