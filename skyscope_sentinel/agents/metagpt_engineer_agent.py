import sys
import os

# Add project root to Python path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json # To format the PRD for the prompt
from skyscope_sentinel.agents.base_agent import BaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration
# from skyscope_sentinel.agents.messaging import AgentMessageQueue # May need if sending message directly

class EngineerAgent(BaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b"): # Using a small model, can be configured
        super().__init__(agent_id)
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name # Model to use for code generation
        self.status = "idle_eng"
        self.log(f"initialized with model '{self.model_name}'.")

    def log(self, message: str):
        print(f"[EngineerAgent {self.agent_id}] {message}")

    def generate_code(self, prd_data: dict) -> str | None:
        self.log(f"Received PRD data for project '{prd_data.get('project_name', 'Unknown Project')}'.")
        self.status = "generating_code"
        self.log(f"Generating code using Ollama model '{self.model_name}'...")

        try:
            prd_data_json_string = json.dumps(prd_data, indent=2)
        except TypeError as e:
            self.log(f"Error: Could not serialize PRD data to JSON: {e}")
            self.status = "error_serializing_prd"
            return None

        prompt = (
            f"Based on the following Product Requirement Document (PRD) in JSON format:\n"
            f"```json\n{prd_data_json_string}\n```\n"
            f"Generate Python code to implement the specified features. "
            f"Aim for a single runnable script or a few simple functions. "
            f"The target platform is '{prd_data.get('target_platform', 'Command Line Interface')}'. "
            f"Only output the raw Python code itself, without any surrounding text, explanations, or markdown formatting like ```python ... ```."
        )

        system_prompt = "You are an expert Python programmer. Your task is to write clean, functional code based on the provided PRD. Output only the code."

        generated_code, error = self.ollama_integration.generate_text_sync(
            model_name=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt
        )

        if error:
            self.log(f"Error generating code from Ollama: {error}")
            self.status = "error_generating_code"
            return None

        if not generated_code:
            self.log("Ollama returned an empty response for code generation.")
            self.status = "error_generating_code"
            return None

        self.log(f"Raw code string from Ollama: {generated_code[:300]}...")

        clean_code = generated_code.strip()
        if clean_code.startswith("```python"):
            clean_code = clean_code.removeprefix("```python").strip()
        if clean_code.startswith("```"):
            clean_code = clean_code.removeprefix("```").strip()
        if clean_code.endswith("```"):
            clean_code = clean_code.removesuffix("```").strip()

        self.log(f"Successfully generated and cleaned code.")
        self.status = "code_generated"
        return clean_code

    def handle_prd_and_generate_code(self, prd_data: dict, message_queue, original_sender_id: str):
        self.log(f"Handling PRD to generate code. Will respond to '{original_sender_id}'.")

        generated_code = self.generate_code(prd_data)

        response_content = {}
        if generated_code:
            self.log(f"Code generation successful for PRD: {prd_data.get('project_name')}.")
            response_content = {
                "type": "code_generation_result",
                "status": "success",
                "project_name": prd_data.get('project_name'),
                "generated_code_snippet": generated_code[:200] + "..."
            }
        else:
            self.log(f"Code generation failed for PRD: {prd_data.get('project_name')}.")
            response_content = {
                "type": "code_generation_result",
                "status": "failure",
                "project_name": prd_data.get('project_name'),
                "error_message": "Failed to generate code (see engineer agent logs for details)."
            }

        message_queue.send_message(
            sender_id=self.agent_id,
            recipient_id=original_sender_id,
            content=response_content
        )
        self.log(f"Sent code generation result to '{original_sender_id}'.")
        return generated_code is not None

if __name__ == '__main__':
    print("--- EngineerAgent Test (with Ollama Integration) ---")

    oi = OllamaIntegration()
    eng_model = "qwen2:0.5b"

    models, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}. Aborting test.")
        sys.exit(1)

    available_models = [m.get('name') for m in models if m.get('name')]
    if eng_model not in available_models:
        print(f"ERROR: Test model '{eng_model}' for Engineer Agent not found.")
        print(f"Available models: {available_models}")
        print(f"Please pull the model first (e.g., `ollama pull {eng_model}`). Aborting test.")
        sys.exit(1)

    eng_agent = EngineerAgent(agent_id="Eng_Agent_001", ollama_integration_instance=oi, model_name=eng_model)
    print(f"Initial status: {eng_agent.get_status()}")

    prd_example_calculator = {
        "project_name": "CLI Calculator",
        "description": "A command-line calculator that performs addition, subtraction, multiplication, and division.",
        "features": [
            "Prompt user for two numbers.",
            "Prompt user for an operation (add, subtract, multiply, divide).",
            "Perform the calculation.",
            "Display the result.",
            "Handle division by zero error."
        ],
        "target_platform": "Command Line Interface"
    }

    print(f"\nTesting generate_code for: '{prd_example_calculator['project_name']}'")
    generated_code = eng_agent.generate_code(prd_example_calculator)
    if generated_code:
        print(f"Generated code for '{prd_example_calculator['project_name']}':\n---BEGIN CODE---\n{generated_code}\n---END CODE---")
    else:
        print(f"Failed to generate code for '{prd_example_calculator['project_name']}'.")
    print(f"Status after generate_code: {eng_agent.get_status()}")

    prd_example_greeting = {
        "project_name": "Greeter Script",
        "description": "A simple script that asks for user's name and prints a personalized greeting.",
        "features": [
            "Prompt user for their name.",
            "Print a personalized greeting message."
        ],
        "target_platform": "Command Line Interface"
    }
    print(f"\nTesting generate_code for: '{prd_example_greeting['project_name']}'")
    generated_code_greeting = eng_agent.generate_code(prd_example_greeting)
    if generated_code_greeting:
        print(f"Generated code for '{prd_example_greeting['project_name']}':\n---BEGIN CODE---\n{generated_code_greeting}\n---END CODE---")
    else:
        print(f"Failed to generate code for '{prd_example_greeting['project_name']}'.")
    print(f"Status after generate_code for greeter: {eng_agent.get_status()}")
