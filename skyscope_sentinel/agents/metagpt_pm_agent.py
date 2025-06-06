import sys
import os

# Add project root to Python path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
from skyscope_sentinel.agents.base_agent import BaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration
# from skyscope_sentinel.agents.messaging import AgentMessageQueue # May need if directly sending

class ProductManagerAgent(BaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b"):
        super().__init__(agent_id)
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name # Model to use for PRD generation
        self.status = "idle_pm"
        self.log(f"initialized with model '{self.model_name}'.")

    def log(self, message: str):
        print(f"[ProductManagerAgent {self.agent_id}] {message}")

    def generate_prd(self, user_requirement_text: str) -> dict | None:
        self.log(f"Received user requirement: '{user_requirement_text[:100]}...'")
        self.status = "generating_prd"
        self.log(f"Generating PRD using Ollama model '{self.model_name}'...")

        prompt = (
            f"Given the user requirement: '{user_requirement_text}', "
            "generate a Product Requirement Document (PRD) as a JSON string. "
            "The JSON string should represent an object with the following keys: "
            "'project_name' (string, a concise name for the project), "
            "'description' (string, a brief explanation of the project), "
            "'features' (list of strings, where each string is an actionable item for a developer), "
            "'target_platform' (string, e.g., 'Command Line Interface', 'Web Application', 'Mobile App'). "
            "Ensure the JSON is well-formed and complete. Only output the JSON string."
        )

        system_prompt = "You are a helpful AI assistant acting as a Product Manager. Your task is to create clear and concise Product Requirement Documents in JSON format."

        generated_json_string, error = self.ollama_integration.generate_text_sync(
            model_name=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt
        )

        if error:
            self.log(f"Error generating PRD from Ollama: {error}")
            self.status = "error_generating_prd"
            return None

        if not generated_json_string:
            self.log("Ollama returned an empty response for PRD generation.")
            self.status = "error_generating_prd"
            return None

        self.log(f"Raw PRD JSON string from Ollama: {generated_json_string[:300]}...")

        try:
            clean_json_string = generated_json_string.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            prd_data = json.loads(clean_json_string)

            required_keys = ['project_name', 'description', 'features', 'target_platform']
            if not all(key in prd_data for key in required_keys):
                self.log(f"Generated PRD JSON is missing one or more required keys. Got: {prd_data.keys()}")
                self.status = "error_parsing_prd"
                return None
            if not isinstance(prd_data['features'], list):
                self.log(f"Generated PRD 'features' is not a list. Got: {type(prd_data['features'])}")
                self.status = "error_parsing_prd"
                return None

            self.log(f"Successfully parsed PRD: {prd_data}")
            self.status = "prd_generated"
            return prd_data
        except json.JSONDecodeError as e:
            self.log(f"Failed to decode PRD JSON string from Ollama: {e}")
            self.log(f"Problematic JSON string was: {generated_json_string[:500]}...")
            self.status = "error_parsing_prd"
            return None

    def handle_requirement_and_send_prd(self, user_requirement_text: str, message_queue, engineer_agent_id: str):
        self.log(f"Handling requirement to generate PRD and send to engineer '{engineer_agent_id}'.")

        prd_data = self.generate_prd(user_requirement_text)

        if prd_data:
            message_content = {
                "type": "prd_document",
                "prd": prd_data
            }
            message_queue.send_message(
                sender_id=self.agent_id,
                recipient_id=engineer_agent_id,
                content=message_content
            )
            self.log(f"Sent PRD to engineer '{engineer_agent_id}'.")
            return True
        else:
            self.log(f"Failed to generate or parse PRD. Nothing sent to engineer '{engineer_agent_id}'.")
            return False

if __name__ == '__main__':
    print("--- ProductManagerAgent Test (with Ollama Integration) ---")

    oi = OllamaIntegration()
    pm_model = "qwen2:0.5b"

    models, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}. Aborting test.")
        sys.exit(1)

    available_models = [m.get('name') for m in models if m.get('name')]
    if pm_model not in available_models:
        print(f"ERROR: Test model '{pm_model}' for PM Agent not found.")
        print(f"Available models: {available_models}")
        print(f"Please pull the model first (e.g., `ollama pull {pm_model}`). Aborting test.")
        sys.exit(1)

    pm_agent = ProductManagerAgent(agent_id="PM_Agent_001", ollama_integration_instance=oi, model_name=pm_model)
    print(f"Initial status: {pm_agent.get_status()}")

    requirement1 = "Create a command-line interface (CLI) application that acts as a simple calculator. It should be able to perform addition, subtraction, multiplication, and division on two numbers provided by the user and display the result."

    print(f"\nTesting generate_prd for: '{requirement1}'")
    prd1 = pm_agent.generate_prd(requirement1)
    if prd1:
        print(f"Generated PRD for '{requirement1}':")
        print(json.dumps(prd1, indent=2))
    else:
        print(f"Failed to generate PRD for '{requirement1}'.")
    print(f"Status after generate_prd: {pm_agent.get_status()}")

    print("\nTesting with a more vague requirement:")
    requirement2 = "I need a program that helps me manage my daily tasks."
    prd2 = pm_agent.generate_prd(requirement2)
    if prd2:
        print(f"Generated PRD for '{requirement2}':")
        print(json.dumps(prd2, indent=2))
    else:
        print(f"Failed to generate PRD for '{requirement2}'.")
    print(f"Status after generate_prd for vague task: {pm_agent.get_status()}")
