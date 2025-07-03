import sys
import os

# Add project root to Python path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json # To format the PRD for the prompt
# from skyscope_sentinel.agents.base_agent import BaseAgent # Replaced by OwlBaseAgent
from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
 feat/foundational-agent-system
from skyscope_sentinel.agents.base_agent import BaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration
from skyscope_sentinel.agents.messaging import AgentMessageQueue # For type hinting

class EngineerAgent(OwlBaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b", owl_toolkits: list = None): # Using a small model, can be configured
        super().__init__(
            agent_id,
            department="Developers", # Assigning to Developers department
            role_description="An AI agent that generates Python code based on Product Requirement Documents (PRDs) and refines it based on reviews.",
            owl_toolkits=owl_toolkits
        )
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name # Model to use for code generation
        self.status = "idle_eng"
        self.log(f"EngineerAgent initialized with model '{self.model_name}'. Identity: {self.identity.get('first_name')} {self.identity.get('last_name')}, Title: {self.identity.get('employee_title')}")

    # Using self.log() from OwlBaseAgent for consistent logging format.
    # def log(self, message: str):
    #     print(f"[EngineerAgent {self.agent_id}] {message}")

# from skyscope_sentinel.agents.base_agent import BaseAgent # Replaced by OwlBaseAgent
from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration
from skyscope_sentinel.agents.messaging import AgentMessageQueue # For type hinting

class EngineerAgent(OwlBaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b", owl_toolkits: list = None): # Using a small model, can be configured
        super().__init__(
            agent_id,
            department="Developers", # Assigning to Developers department
            role_description="An AI agent that generates Python code based on Product Requirement Documents (PRDs) and refines it based on reviews.",
            owl_toolkits=owl_toolkits
        )
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name # Model to use for code generation
        self.status = "idle_eng"
        self.log(f"EngineerAgent initialized with model '{self.model_name}'. Identity: {self.identity.get('first_name')} {self.identity.get('last_name')}, Title: {self.identity.get('employee_title')}")

    # Using self.log() from OwlBaseAgent for consistent logging format.
    # def log(self, message: str):
    #     print(f"[EngineerAgent {self.agent_id}] {message}")
 main

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
        self.log(f"Sent code to ReviewerAgent '{reviewer_agent_id}' for review.")
        self.status = "waiting_code_review"
        return generated_code

    def handle_code_review_feedback(self,
                                    review_feedback_content: dict,
                                    original_code: str,
                                    prd_data: dict,
                                    message_queue: AgentMessageQueue,
                                    final_recipient_id: str):
        self.log(f"Received code review feedback: {str(review_feedback_content)[:150]}...")
        self.status = "processing_code_review"

        approved = review_feedback_content.get('approved', False)
        suggestions = review_feedback_content.get('suggestions', [])

        self.log(f"Reviewer Code Feedback - Approved: {approved}, Suggestions: {len(suggestions)}")

        final_code_to_send = original_code
        regeneration_attempted = False
        code_was_revised = False

        if not approved and suggestions: # If not approved AND there are suggestions, try to regenerate
            self.log(f"Code not approved or has suggestions. Attempting regeneration based on feedback (Suggestions: {suggestions}).")
            self.status = "regenerating_code"
            regeneration_attempted = True

            try:
                prd_data_json_string = json.dumps(prd_data, indent=2)
            except TypeError as e:
                self.log(f"Error: Could not serialize PRD data to JSON for code regeneration prompt: {e}")
                self.status = "error_regenerating_code" # Error in prep for regen
                # Proceed to send original code below (pass will let it fall through)
            else:
                refinement_prompt = (
                    f"The original Product Requirement Document (PRD) was:\n"
                    f"```json\n{prd_data_json_string}\n```\n"
                    f"The first attempt at generating Python code produced:\n"
                    f"```{prd_data.get('target_platform', 'python')}\n{original_code}\n```\n" # Assuming python, or use target_platform
                    f"A review of this code provided the following suggestions for improvement:\n"
                )
                for i, suggestion in enumerate(suggestions):
                    refinement_prompt += f"- {suggestion}\n"

                refinement_prompt += (
                    f"\nPlease generate a revised Python code snippet that addresses these suggestions and adheres to the PRD. "
                    f"Only output the raw Python code itself, without any surrounding text, explanations, or markdown formatting like ```python ... ```."
                )

                system_prompt_refine = "You are an expert Python programmer. You previously wrote a piece of code which received feedback. Your task is to revise the code based on the provided PRD and suggestions to improve its quality and correctness. Output only the code."

                self.log(f"Code Refinement prompt (first 200 chars): {refinement_prompt[:200]}...")

                revised_code_string, error = self.ollama_integration.generate_text_sync(
                    model_name=self.model_name,
                    prompt=refinement_prompt,
                    system_prompt=system_prompt_refine
                )

                if error:
                    self.log(f"Error during code regeneration from Ollama: {error}")
                    self.status = "error_regenerating_code"
                elif not revised_code_string:
                    self.log("Ollama returned an empty response for code regeneration.")
                    self.status = "error_regenerating_code"
                else:
                    self.log(f"Raw revised code string from Ollama: {revised_code_string[:300]}...")
                    clean_revised_code = revised_code_string.strip()
                    if clean_revised_code.startswith("```python"):
                        clean_revised_code = clean_revised_code.removeprefix("```python").strip()
                    if clean_revised_code.startswith("```"):
                        clean_revised_code = clean_revised_code.removeprefix("```").strip()
                    if clean_revised_code.endswith("```"):
                        clean_revised_code = clean_revised_code.removesuffix("```").strip()

                    if clean_revised_code:
                        self.log("Successfully generated and cleaned revised code.")
                        final_code_to_send = clean_revised_code
                        code_was_revised = True
                        self.status = "code_regenerated_and_approved"
                    else:
                        self.log("Revised code was empty after cleaning. Sticking with original.")
                        self.status = "error_regenerating_code"
        elif approved:
            self.log("Code was approved by reviewer, or no actionable suggestions provided. Sending original code.")
        else:
            self.log("Code was not approved, but no suggestions to act upon. Sending original code with a warning.")

        if not final_code_to_send:
            self.log("Critical Error: No code available to send (original or revised). This should not happen if original_code was valid.")
            self.status = "error_eng_logic_critical"
            error_msg_content = {"type": "eng_error", "detail": "Critical error, no code available after review process."}
            message_queue.send_message(self.agent_id, final_recipient_id, error_msg_content)
            return False

        code_delivery_content = {
            "type": "code_generation_result",
            "status": "success" if (approved or code_was_revised) else "success_with_unaddressed_review_suggestions",
            "project_name": prd_data.get('project_name', 'Unknown Project'),
            "generated_code_snippet": final_code_to_send[:250] + "...",
            "full_code": final_code_to_send,
            "review_was_conducted": True,
            "code_was_revised": code_was_revised
        }
        message_queue.send_message(
            sender_id=self.agent_id,
            recipient_id=final_recipient_id,
            content=code_delivery_content
        )
        if code_was_revised:
            self.log(f"Sent REVISED code to final recipient '{final_recipient_id}'.")
        else:
            self.log(f"Sent ORIGINAL code to final recipient '{final_recipient_id}' (Review approved, no suggestions, or regeneration failed/skipped).")

        self.status = "code_sent_to_user"
        return True

if __name__ == '__main__':
    print("--- EngineerAgent Test (with Ollama Integration) ---")

    oi = OllamaIntegration()
    eng_model = "qwen2:0.5b"

    models, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}. Aborting test.")
        sys.exit(1)

    available_models = []
    if models: # Ensure models is not None
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

    print("\nNote: Full workflow including code review request and handling feedback")
    print("is typically tested in a multi-agent demo script using AgentMessageQueue.")
    print("The methods 'handle_prd_and_generate_code' (now for initiating review) and 'handle_code_review_feedback'")
    print("orchestrate these interactions.")
    print("\nThe code refinement logic in 'handle_code_review_feedback'")
    print("involves receiving feedback and potentially re-generating code using Ollama.")
    print("This iterative process is best tested in the full multi-agent demo script,")
    print("where actual reviewer feedback can be provided to trigger regeneration.")
