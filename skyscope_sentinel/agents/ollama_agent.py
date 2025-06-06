import sys
import os

# Add project root to Python path to allow direct execution of agent script for testing
# and to ensure skyscope_sentinel modules are found.
try:
    # Standard case: Running from project root or IDE that sets PYTHONPATH
    from skyscope_sentinel.agents.base_agent import BaseAgent
    from skyscope_sentinel.ollama_integration import OllamaIntegration
except ImportError:
    # Fallback for direct script execution (e.g., python skyscope_sentinel/agents/ollama_agent.py)
    # This adjusts path to find modules in `skyscope_sentinel` when script is inside `agents` subdir.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Retry imports after path adjustment
    from skyscope_sentinel.agents.base_agent import BaseAgent
    from skyscope_sentinel.ollama_integration import OllamaIntegration

class OllamaWorkerAgent(BaseAgent):
    """
    An AI agent that uses Ollama to generate text based on prompts.
    It extends BaseAgent and utilizes OllamaIntegration for interacting with Ollama.
    """

    def __init__(self, agent_id: str, model_name: str = "qwen2:0.5b", ollama_integration_instance: OllamaIntegration = None):
        """
        Initializes the OllamaWorkerAgent.

        Args:
            agent_id (str): The unique identifier for this agent.
            model_name (str, optional): The Ollama model to be used for text generation.
                                        Defaults to "qwen2:0.5b".
            ollama_integration_instance (OllamaIntegration, optional):
                                        An instance of OllamaIntegration. If None, a new one is created.
        """
        super().__init__(agent_id)
        self.model_name = model_name
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.log(f"Initialized with model '{self.model_name}'. Current status: {self.status}")

    def log(self, message: str, level: str = "info"):
        """
        Logs a message with the agent's ID and specified level.
        """
        print(f"[OllamaWorkerAgent {self.agent_id}] [{level.upper()}] {message}")

    def process_task(self, task_description: dict) -> tuple[str | None, str | None]:
        """
        Processes a task that requires text generation using Ollama.

        The task_description dictionary must contain a 'prompt' key.
        An optional 'system_prompt' key can also be included.

        Args:
            task_description (dict): A dictionary containing task details.
                                     Expected keys: "prompt" (str), "system_prompt" (str, optional).

        Returns:
            tuple[str | None, str | None]: (generated_text, error_message)
                                           generated_text is the response from Ollama if successful.
                                           error_message contains details if an error occurred.
        """
        # Call super().process_task() to log receipt and set initial status
        # It internally sets status to "processing" then "idle". We'll override status later.
        super().process_task(task_description)

        if not isinstance(task_description, dict) or 'prompt' not in task_description:
            error_msg = "Invalid task_description: must be a dictionary and contain a 'prompt' key."
            self.log(error_msg, level="error")
            self.status = "error_invalid_task"
            return None, error_msg

        prompt = task_description['prompt']
        system_prompt = task_description.get('system_prompt') # Optional

        self.log(f"Processing prompt for model '{self.model_name}': '{prompt[:70]}...'")
        self.status = "processing_llm" # More specific status

        generated_text, error = self.ollama_integration.generate_text_sync(
            model_name=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt
        )

        if error:
            self.log(f"Error generating text with model '{self.model_name}': {error}", level="error")
            self.status = "error_ollama"
            return None, error

        self.log(f"Successfully generated text using model '{self.model_name}'. (Output length: {len(generated_text)})")
        self.status = "task_complete"
        return generated_text, None

if __name__ == '__main__':
    print("\n--- OllamaWorkerAgent Test Suite ---")

    # 1. Initialize OllamaIntegration
    print("\n[Step 1] Initializing OllamaIntegration...")
    ollama_integration = OllamaIntegration()
    version, error = ollama_integration.get_ollama_version_sync()
    if error:
        print(f"  Error initializing OllamaIntegration or getting version: {error}")
        print("  Cannot proceed with OllamaWorkerAgent tests. Ensure Ollama is running and accessible.")
        sys.exit(1)
    print(f"  Ollama version: {version}. Integration initialized.")

    # 2. Select a model for testing (try a small, common one)
    test_model = "qwen2:0.5b" # User should have this model or change to one they have
    print(f"\n[Step 2] Checking availability of test model: '{test_model}'...")
    local_models, error = ollama_integration.list_models_sync()
    if error:
        print(f"  Error listing local models: {error}")
        # Proceeding with test_model, assuming it might be pulled by Ollama automatically or fail gracefully.
    elif local_models is not None and any(m.get('name') == test_model for m in local_models):
        print(f"  Model '{test_model}' found locally.")
    else:
        print(f"  Warning: Model '{test_model}' not found locally. Ollama might attempt to pull it, or generation may fail if it's not available.")
        print(f"  Consider running 'ollama pull {test_model}' before extensive testing.")

    # 3. Initialize OllamaWorkerAgent
    print(f"\n[Step 3] Initializing OllamaWorkerAgent with agent_id 'Worker001' and model '{test_model}'...")
    worker_agent = OllamaWorkerAgent(agent_id="Worker001", model_name=test_model, ollama_integration_instance=ollama_integration)
    print(f"  Agent initialized. Initial status: {worker_agent.get_status()}")

    # 4. Test with a valid task
    print("\n[Step 4] Testing with a valid task (simple prompt)...")
    valid_task = {
        "prompt": "What is the main purpose of a CPU in a computer? Respond concisely.",
        "system_prompt": "You are a helpful AI assistant that explains technical concepts clearly."
    }
    generated_text, error = worker_agent.process_task(valid_task)
    if error:
        print(f"  Error processing valid task: {error}")
    else:
        print(f"  Successfully processed valid task. Generated text (first 150 chars):")
        print(f"    '{generated_text[:150]}...'")
    print(f"  Agent status after task: {worker_agent.get_status()}")
    assert worker_agent.get_status() == "task_complete" or worker_agent.get_status() == "error_ollama" # error_ollama if model not found

    # 5. Test with a task having no system prompt
    print("\n[Step 5] Testing with a valid task (prompt only)...")
    valid_task_no_system = {
        "prompt": "List three common types of renewable energy sources."
    }
    generated_text_ns, error_ns = worker_agent.process_task(valid_task_no_system)
    if error_ns:
        print(f"  Error processing task (no system prompt): {error_ns}")
    else:
        print(f"  Successfully processed task (no system prompt). Generated text (first 150 chars):")
        print(f"    '{generated_text_ns[:150]}...'")
    print(f"  Agent status after task: {worker_agent.get_status()}")


    # 6. Test with an invalid task (missing 'prompt' key)
    print("\n[Step 6] Testing with an invalid task (missing 'prompt')...")
    invalid_task_missing_prompt = {
        "description": "This task is missing the prompt."
    }
    _, error_invalid = worker_agent.process_task(invalid_task_missing_prompt)
    if error_invalid:
        print(f"  Successfully caught error for invalid task: {error_invalid}")
    else:
        print("  Warning: Invalid task did not return an error as expected.")
    print(f"  Agent status after invalid task: {worker_agent.get_status()}")
    assert worker_agent.get_status() == "error_invalid_task"

    # 7. Test with an invalid task (not a dictionary)
    print("\n[Step 7] Testing with an invalid task (not a dictionary)...")
    invalid_task_not_dict = "This is just a string, not a task dictionary."
    _, error_invalid_type = worker_agent.process_task(invalid_task_not_dict)
    if error_invalid_type:
        print(f"  Successfully caught error for invalid task type: {error_invalid_type}")
    else:
        print("  Warning: Invalid task type did not return an error as expected.")
    print(f"  Agent status after invalid task type: {worker_agent.get_status()}")
    assert worker_agent.get_status() == "error_invalid_task"

    # 8. Test with a model that might not exist (Ollama's behavior might vary: error or attempt to pull)
    print("\n[Step 8] Testing with a potentially non-existent model 'nonexistent-ollama-model:latest'...")
    worker_agent_nonexistent_model = OllamaWorkerAgent(agent_id="Worker002-BadModel", model_name="nonexistent-ollama-model:latest", ollama_integration_instance=ollama_integration)
    task_for_bad_model = {"prompt": "Does this work?"}
    _, error_bad_model = worker_agent_nonexistent_model.process_task(task_for_bad_model)
    if error_bad_model:
        print(f"  Error (as expected) when using non-existent model: {error_bad_model[:200]}...") # Show first 200 chars of error
    else:
        print("  Warning: Processing with non-existent model did not result in an error. This might indicate Ollama is pulling it or has fallback behavior.")
    print(f"  Agent 'Worker002-BadModel' status: {worker_agent_nonexistent_model.get_status()}")
    assert worker_agent_nonexistent_model.get_status() == "error_ollama" or "not found" in error_bad_model.lower()


    print("\n--- OllamaWorkerAgent Test Suite Finished ---")
    print("NOTE: Some tests depend on Ollama service running and specific models being available or auto-pullable.")
