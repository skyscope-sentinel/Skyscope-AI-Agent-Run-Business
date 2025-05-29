import subprocess
import json
import threading
from PySide6.QtCore import QObject, Signal

class OllamaRunner(QObject):
    """
    Runs Ollama commands in a separate thread and emits signals with the results.
    """
    command_complete = Signal(str, str, bool) # command_name, output, success
    download_progress = Signal(str, str) # model_name, progress_line (raw output from ollama pull)

    def __init__(self, command_name, command_args):
        super().__init__()
        self.command_name = command_name
        self.command_args = command_args

    def run(self):
        try:
            print(f"Running Ollama command: {self.command_name} with args: {self.command_args}")
            process = subprocess.Popen(self.command_args,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True,
                                       bufsize=1, # Line buffered
                                       universal_newlines=True) # Ensure text mode

            output_lines = []
            error_lines = []
            
            # For pull, we want to stream progress
            if self.command_name == "pull":
                for line in process.stdout:
                    line = line.strip()
                    output_lines.append(line)
                    self.download_progress.emit(self.command_args[-1] if len(self.command_args) > 0 else "", line)
            else:
                stdout, stderr = process.communicate()
                output_lines = stdout.strip().splitlines() if stdout else []
                error_lines = stderr.strip().splitlines() if stderr else []

            process.wait() # Wait for the process to complete

            if process.returncode == 0:
                print(f"Command '{self.command_name}' successful.")
                self.command_complete.emit(self.command_name, "\n".join(output_lines), True)
            else:
                error_message = "\n".join(error_lines) if error_lines else "\n".join(output_lines)
                print(f"Command '{self.command_name}' failed with code {process.returncode}: {error_message}")
                self.command_complete.emit(self.command_name, error_message, False)

        except FileNotFoundError:
            print("Error: 'ollama' command not found. Is Ollama installed and in your PATH?")
            self.command_complete.emit(self.command_name, "Ollama command not found. Ensure it's installed and in PATH.", False)
        except Exception as e:
            print(f"An unexpected error occurred with command '{self.command_name}': {e}")
            self.command_complete.emit(self.command_name, f"An unexpected error occurred with command '{self.command_name}': {str(e)}", False)


class OllamaIntegration:
    def __init__(self):
        self.runner = None 
        self.thread = None 

    def _run_command_async(self, command_name, command_args, on_complete_slot, on_progress_slot=None):
        from PySide6.QtCore import QThread 

        self.thread = QThread()
        self.runner = OllamaRunner(command_name, command_args)
        self.runner.moveToThread(self.thread)

        self.runner.command_complete.connect(on_complete_slot)
        if on_progress_slot and command_name == "pull": 
            self.runner.download_progress.connect(on_progress_slot)
        
        self.thread.started.connect(self.runner.run)
        self.runner.command_complete.connect(self.thread.quit) 
        self.thread.finished.connect(self.thread.deleteLater) 
        self.thread.finished.connect(self.runner.deleteLater) # Ensure runner is also cleaned up


        self.thread.start()

    def list_models(self, on_complete_slot):
        """
        Lists locally available Ollama models asynchronously.
        Output on success is a newline-separated JSON string of the models.
        """
        self._run_command_async("list", ["ollama", "list", "--format=json"], on_complete_slot)

    def pull_model(self, model_name, on_complete_slot, on_progress_slot):
        """
        Pulls a model from the Ollama Hub asynchronously.
        """
        if not model_name or len(model_name.strip()) == 0:
            # Emit failure directly if model name is invalid before starting thread
            if on_complete_slot:
                on_complete_slot.emit("pull", "Model name cannot be empty.", False)
            return
        self._run_command_async("pull", ["ollama", "pull", model_name], on_complete_slot, on_progress_slot)

    def show_model_info(self, model_name, on_complete_slot):
        """
        Shows detailed information about a specific model asynchronously.
        Output on success is a JSON string of the model details.
        """
        if not model_name or len(model_name.strip()) == 0:
            if on_complete_slot:
                on_complete_slot.emit("show", "Model name cannot be empty for showing info.", False)
            return
        self._run_command_async("show", ["ollama", "show", model_name, "--format=json"], on_complete_slot)

    # --- Synchronous versions ---
    
    def list_models_sync(self):
        try:
            result = subprocess.run(["ollama", "list", "--format=json"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return None, f"Ollama CLI Error: {result.stderr.strip()}"
            
            models = []
            # Handles cases where output might be empty or have unexpected newlines
            for line in result.stdout.strip().split('\n'):
                if line.strip(): # Ensure line is not just whitespace
                    try:
                        models.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        # Log or handle individual line parse error if necessary
                        print(f"Warning: Skipping malformed JSON line in 'ollama list' output: {line} - Error: {e}")
                        continue # Skip this line and try others
            if not result.stdout.strip() and not models: # If stdout was empty and no models parsed
                 return [], None # Successfully returned no models
            return models, None
        except subprocess.CalledProcessError as e: # Should be caught by check=False + returncode check, but good practice
            return None, f"Error listing models (CalledProcessError): {e}\n{e.stderr}"
        except FileNotFoundError:
            return None, "Ollama command not found. Ensure Ollama is installed and in your system's PATH."
        except Exception as e: # Catch any other unexpected errors
            return None, f"An unexpected error occurred while listing models: {str(e)}"


    def show_model_info_sync(self, model_name: str):
        if not model_name or len(model_name.strip()) == 0:
            return None, "Model name cannot be empty."
        try:
            result = subprocess.run(["ollama", "show", model_name, "--format=json"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return None, f"Ollama CLI Error for '{model_name}': {result.stderr.strip()}"
            return json.loads(result.stdout), None
        except FileNotFoundError:
            return None, "Ollama command not found. Ensure Ollama is installed and in your system's PATH."
        except json.JSONDecodeError:
            return None, f"Error decoding JSON from 'ollama show {model_name}'. Output: {result.stdout[:200]}..." # Show partial output
        except Exception as e:
            return None, f"An unexpected error occurred while showing model info for '{model_name}': {str(e)}"

    def get_ollama_version_sync(self):
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return None, f"Ollama CLI Error (--version): {result.stderr.strip()}"
            return result.stdout.strip(), None
        except FileNotFoundError:
            return None, "Ollama command not found. Ensure Ollama is installed and in your system's PATH."
        except Exception as e:
            return None, f"An unexpected error occurred while getting Ollama version: {str(e)}"


if __name__ == '__main__':
    # This basic test suite can be run with `python -m skyscope_sentinel.ollama_integration`
    # from the project root directory.
    print("--- Starting OllamaIntegration Test Suite ---")
    ollama_api = OllamaIntegration()

    print("\n[TEST] Synchronous Get Ollama Version:")
    version, error = ollama_api.get_ollama_version_sync()
    if error:
        print(f"  ERROR: {error}")
    else:
        print(f"  SUCCESS: Ollama Version: {version}")

    print("\n[TEST] Synchronous List Models:")
    models, error = ollama_api.list_models_sync()
    if error:
        print(f"  ERROR: {error}")
    else:
        print(f"  SUCCESS: Found {len(models)} models.")
        if models:
            print(f"    Models: {[m.get('name') for m in models]}")
        else:
            print("    No local models found or Ollama service might not be running.")

    if models:
        print(f"\n[TEST] Synchronous Show Model Info (for first model: {models[0].get('name')}):")
        first_model_name = models[0].get('name')
        info, error = ollama_api.show_model_info_sync(first_model_name)
        if error:
            print(f"  ERROR: {error}")
        else:
            print(f"  SUCCESS: Info for {first_model_name} retrieved.")
            print(f"    Family: {info.get('details', {}).get('family')}, Size: {info.get('details', {}).get('parameter_size')}")
    else:
        print("\n[INFO] Skipping Show Model Info test as no local models were found.")

    print("\n--- Asynchronous Tests (Setup) ---")
    print("Note: Asynchronous tests require a running Qt event loop to fully execute.")
    print("This script will set them up, but they will only print initial messages unless run within a Qt app.")

    # Dummy Qt Application for event loop (if this script is run directly)
    try:
        from PySide6.QtCore import QCoreApplication, QTimer
        import sys
        
        app_exists = QCoreApplication.instance() is not None
        if not app_exists:
            app = QCoreApplication(sys.argv)
        else:
            app = QCoreApplication.instance()

        print("\n[ASYNC TEST] List Models Asynchronously:")
        def handle_async_list_complete(command, output, success):
            print(f"  [ASYNC LIST CALLBACK] Command: {command}, Success: {success}")
            if success:
                try:
                    listed_models = [json.loads(line) for line in output.strip().split('\n') if line]
                    print(f"    Parsed Models: {[m.get('name') for m in listed_models]}")
                except json.JSONDecodeError as e:
                    print(f"    ERROR parsing JSON from async list: {e}. Output: {output[:100]}...")
            else:
                print(f"    ERROR in async list: {output}")
            # QCoreApplication.instance().quit() # Quit after first async test if running standalone

        ollama_api.list_models(handle_async_list_complete)

        # Example for pull (optional, can be very slow)
        # print("\n[ASYNC TEST] Pull Model Asynchronously (e.g., 'orca-mini'):")
        # test_model_to_pull = "orca-mini" 
        # def handle_async_pull_progress(model_name, progress_line):
        #     print(f"  [ASYNC PULL PROGRESS - {model_name}] {progress_line}")
        # def handle_async_pull_complete(command, output, success):
        #     print(f"  [ASYNC PULL CALLBACK] Command: {command}, Success: {success}")
        #     if not success:
        #         print(f"    ERROR in async pull: {output}")
        #     else:
        #         print(f"    SUCCESS: {test_model_to_pull} pulled.")
        #     QCoreApplication.instance().quit()
        # ollama_api.pull_model(test_model_to_pull, handle_async_pull_complete, handle_async_pull_progress)

        # Quit after a short delay if no other async tests are running
        if not app_exists: # Only manage event loop if we created it
            QTimer.singleShot(5000, app.quit) # Quit after 5s to allow async ops to start/finish
            print("\nStarting dummy event loop for 5 seconds for async tests...")
            app.exec()
            print("Dummy event loop finished.")

    except ImportError:
        print("\n[INFO] PySide6 not found, skipping asynchronous test execution setup.")
    except Exception as e:
        print(f"\n[ERROR] Could not set up/run Qt event loop for async tests: {e}")
        
    print("\n--- OllamaIntegration Test Suite Finished ---")
