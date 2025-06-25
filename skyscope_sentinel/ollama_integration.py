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

    def generate_text_sync(self, model_name: str, prompt: str, system_prompt: str = None) -> tuple[str | None, str | None]:
        """
        Synchronously generates text using a specified Ollama model via CLI for streaming JSON.

        Args:
            model_name (str): The name of the model to use (e.g., "qwen2:0.5b").
            prompt (str): The user prompt for the model.
            system_prompt (str, optional): An optional system-level prompt to guide the model's behavior.

        Returns:
            tuple[str | None, str | None]: (generated_text, error_message)
                                           generated_text is the full response if successful, None otherwise.
                                           error_message is a string if an error occurred, None otherwise.
        """
        if not model_name or len(model_name.strip()) == 0:
            return None, "Model name cannot be empty for text generation."
        if not prompt or len(prompt.strip()) == 0:
            return None, "Prompt cannot be empty for text generation."

        command = ["ollama", "generate", model_name, "--format=json"]

        payload = {"prompt": prompt}
        if system_prompt:
            payload["system"] = system_prompt

        self._log(f"Executing Ollama command: {' '.join(command)} with prompt (first 50 chars): '{prompt[:50]}...'")

        try:
            process = subprocess.run(
                command,
                input=json.dumps(payload), # Send payload as JSON string via stdin
                capture_output=True,
                text=True,
                check=False # Don't raise exception for non-zero exit codes immediately
            )

            if process.returncode != 0:
                error_msg = f"Ollama CLI error (code {process.returncode}) for model '{model_name}': {process.stderr.strip()}"
                self._log(error_msg, level="error")
                return None, error_msg

            full_response_text = ""
            json_error_count = 0

            # Output from `ollama generate ... --format=json` is a stream of JSON objects, one per line.
            # Each JSON object represents a part of the response. Concatenate the 'response' fields.
            for line in process.stdout.strip().split('\n'):
                if not line.strip(): # Skip empty lines
                    continue
                try:
                    json_part = json.loads(line)
                    full_response_text += json_part.get("response", "")
                    # Check for error in the 'done' message which might contain final error details
                    if json_part.get("done", False) and json_part.get("error"):
                        error_msg = f"Ollama generation error in 'done' response for model '{model_name}': {json_part['error']}"
                        self._log(error_msg, level="error")
                        # If an error occurs in the "done" part, it might supersede any partial text
                        return None, error_msg
                except json.JSONDecodeError:
                    json_error_count += 1
                    self._log(f"Warning: JSONDecodeError for line: '{line}' during generation with '{model_name}'.", level="warning")

            if json_error_count > 0:
                 self._log(f"Warning: Encountered {json_error_count} JSON decoding errors processing stream from '{model_name}'.", level="warning")
                 # Depending on strictness, one might return an error here or proceed with potentially incomplete text.

            if not full_response_text.strip() and process.stderr.strip():
                # If no text and there was stderr not caught by returncode (less common for generate)
                error_msg = f"Ollama generation for '{model_name}' produced no text, stderr: {process.stderr.strip()}"
                self._log(error_msg, level="error")
                return None, error_msg

            if not full_response_text.strip() and not process.stderr.strip() and process.returncode == 0 :
                # This case can happen if the model generates nothing (e.g. empty prompt for some models)
                # or if the output stream was empty for other reasons.
                self._log(f"Warning: Ollama generation for '{model_name}' produced no text output, but no explicit CLI error.", level="warning")
                # Return empty string and no error, as the command itself succeeded.
                return "", None


            self._log(f"Successfully generated text with '{model_name}' (first 100 chars): '{full_response_text[:100]}...'")
            return full_response_text, None

        except FileNotFoundError:
            error_msg = "Ollama CLI not found. Please ensure it's installed and in your PATH."
            self._log(error_msg, level="error")
            return None, error_msg
        except Exception as e:
            error_msg = f"An unexpected error occurred during Ollama text generation with '{model_name}': {e}"
            self._log(error_msg, level="error")
            return None, error_msg

    def _log(self, message: str, level: str = "info"):
        """Internal simple logger for the class integration methods."""
        # In a real app, this might use the application's logging framework.
        print(f"[OllamaIntegration] [{level.upper()}] {message}")


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
    models_list_data, error = ollama_api.list_models_sync() # Renamed to avoid conflict
    if error:
        print(f"  ERROR: {error}")
    else:
        print(f"  SUCCESS: Found {len(models_list_data)} models.")
        if models_list_data:
            print(f"    Models: {[m.get('name') for m in models_list_data]}")
        else:
            print("    No local models found or Ollama service might not be running.")

    test_generation_model_name = None
    if models_list_data:
        first_model_name = models_list_data[0].get('name')
        print(f"\n[TEST] Synchronous Show Model Info (for first model: {first_model_name}):")
        info, error = ollama_api.show_model_info_sync(first_model_name)
        if error:
            print(f"  ERROR: {error}")
        else:
            print(f"  SUCCESS: Info for {first_model_name} retrieved.")
            print(f"    Family: {info.get('details', {}).get('family')}, Size: {info.get('details', {}).get('parameter_size')}")
        test_generation_model_name = first_model_name # Use this model for generation test
    else:
        print("\n[INFO] Skipping Show Model Info test as no local models were found.")
        # Try a common small model if no local models are found
        # User should ensure this model is available if they want the test to pass.
        print("\n[INFO] Attempting generation test with a default small model 'qwen2:0.5b'.")
        print("[INFO] If 'qwen2:0.5b' is not available, this test may fail or take time to download.")
        test_generation_model_name = "qwen2:0.5b"


    if test_generation_model_name:
        print(f"\n[TEST] Synchronous Text Generation (Model: {test_generation_model_name}):")
        prompt1 = "Explain the concept of a Large Language Model in one simple paragraph."
        system_prompt1 = "You are an AI assistant that explains complex topics simply."

        print(f"  Test 1: Prompt: '{prompt1[:50]}...', System Prompt: '{system_prompt1[:50]}...'")
        generated_text, gen_error = ollama_api.generate_text_sync(test_generation_model_name, prompt1, system_prompt1)
        if gen_error:
            print(f"    ERROR generating text: {gen_error}")
        else:
            print(f"    SUCCESS. Generated text (first 100 chars): {generated_text[:100]}...")

        print(f"\n  Test 2: Prompt only: 'What are the planets in our solar system?'")
        generated_text_no_sys, gen_error_no_sys = ollama_api.generate_text_sync(test_generation_model_name, "What are the planets in our solar system?")
        if gen_error_no_sys:
            print(f"    ERROR generating text (no system prompt): {gen_error_no_sys}")
        else:
            print(f"    SUCCESS. Generated text (no system prompt, first 100 chars): {generated_text_no_sys[:100]}...")

        print(f"\n  Test 3: Using a non-existent model 'thismodeldoesnotexist:latest'")
        _, gen_error_non_existent = ollama_api.generate_text_sync("thismodeldoesnotexist:latest", "Hello?")
        if gen_error_non_existent:
            print(f"    SUCCESS: Correctly caught error for non-existent model: {gen_error_non_existent[:100]}...")
        else:
            print(f"    WARNING: Text generation with non-existent model did not return an error as expected.")

        print(f"\n  Test 4: Empty prompt string")
        _, gen_error_empty_prompt = ollama_api.generate_text_sync(test_generation_model_name, "")
        if gen_error_empty_prompt:
            print(f"    SUCCESS: Correctly caught error for empty prompt: {gen_error_empty_prompt}")
        else:
            print(f"    WARNING: Text generation with empty prompt did not return an error as expected.")

    else:
        print("\n[INFO] Skipping Text Generation test as no model could be determined.")


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
