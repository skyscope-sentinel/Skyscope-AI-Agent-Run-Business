import sys
import os
import json
# from skyscope_sentinel.agents.base_agent import BaseAgent # Replaced by OwlBaseAgent
from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
 feat/foundational-agent-system
from skyscope_sentinel.agents.base_agent import BaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration
from skyscope_sentinel.agents.messaging import AgentMessageQueue # For type hinting

class ProductManagerAgent(OwlBaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b", owl_toolkits: list = None):
        # PMs could be part of "Strategists" or a dedicated "Product" department.
        # For now, assigning to "Strategists" as it's one of the defined departments.
        super().__init__(
            agent_id,
            department="Strategists", # Assigning a department
            role_description="An AI agent that translates user requirements into detailed Product Requirement Documents (PRDs).",
            owl_toolkits=owl_toolkits
        )
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name
        # self.status is managed by OwlBaseAgent, initialized to "idle_owl"
        # We can set a more specific status if needed, e.g., self.status = "idle_pm"
        self.status = "idle_pm"
        self.log(f"ProductManagerAgent initialized with model '{self.model_name}'. Identity: {self.identity.get('first_name')} {self.identity.get('last_name')}, Title: {self.identity.get('employee_title')}")

    # OwlBaseAgent provides a log method: self.log(message)
    # which prefixes with agent_id and name. If a different format is needed, this can be overridden.
    # For consistency, we'll use the one from OwlBaseAgent.
    # def log(self, message: str):
    #     print(f"[ProductManagerAgent {self.agent_id}] {message}")


# from skyscope_sentinel.agents.base_agent import BaseAgent # Replaced by OwlBaseAgent
from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration
from skyscope_sentinel.agents.messaging import AgentMessageQueue # For type hinting

class ProductManagerAgent(OwlBaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b", owl_toolkits: list = None):
        # PMs could be part of "Strategists" or a dedicated "Product" department.
        # For now, assigning to "Strategists" as it's one of the defined departments.
        super().__init__(
            agent_id,
            department="Strategists", # Assigning a department
            role_description="An AI agent that translates user requirements into detailed Product Requirement Documents (PRDs).",
            owl_toolkits=owl_toolkits
        )
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name
        # self.status is managed by OwlBaseAgent, initialized to "idle_owl"
        # We can set a more specific status if needed, e.g., self.status = "idle_pm"
        self.status = "idle_pm"
        self.log(f"ProductManagerAgent initialized with model '{self.model_name}'. Identity: {self.identity.get('first_name')} {self.identity.get('last_name')}, Title: {self.identity.get('employee_title')}")

    # OwlBaseAgent provides a log method: self.log(message)
    # which prefixes with agent_id and name. If a different format is needed, this can be overridden.
    # For consistency, we'll use the one from OwlBaseAgent.
    # def log(self, message: str):
    #     print(f"[ProductManagerAgent {self.agent_id}] {message}")

 main

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

    def handle_requirement_and_send_prd(self,
                                        user_requirement_text: str,
                                        message_queue: AgentMessageQueue,
                                        engineer_agent_id: str,
                                        reviewer_agent_id: str,
                                        original_user_id: str) -> dict | None: # Modified return type
        self.log(f"Handling requirement to generate PRD, get reviewed by '{reviewer_agent_id}', then send to engineer '{engineer_agent_id}'. Original user: '{original_user_id}'.")

        prd_data = self.generate_prd(user_requirement_text)

        if not prd_data:
            self.log(f"Failed to generate initial PRD. Nothing to review or send to engineer.")
            error_msg_content = {
                "type": "prd_generation_failed",
                "requirement": user_requirement_text,
                "error": "PM failed to generate PRD."
            }
            if message_queue:
                message_queue.send_message(self.agent_id, original_user_id, error_msg_content)
            return None

        self.log(f"Initial PRD generated for '{prd_data.get('project_name', 'Unknown Project')}'.")

        review_request_content = {
            "type": "prd_review_request",
            "prd_data": prd_data,
            "respond_to_agent_id": self.agent_id
        }
        if message_queue:
            message_queue.send_message(
                sender_id=self.agent_id,
                recipient_id=reviewer_agent_id,
                content=review_request_content
            )
            self.log(f"Sent PRD to ReviewerAgent '{reviewer_agent_id}' for review.")
        else:
            self.log("Warning: Message queue not provided. Cannot send PRD for review.")

        self.status = "waiting_prd_review"
        return prd_data

    def handle_prd_review_feedback(self,
                                   review_feedback_content: dict,
                                   original_prd_data: dict,
                                   user_requirement_text: str, # Needed for regeneration prompt
                                   message_queue: AgentMessageQueue,
                                   engineer_agent_id: str,
                                   original_user_id: str):
        self.log(f"Received PRD review feedback: {str(review_feedback_content)[:150]}...")
        self.status = "processing_prd_review"

        approved = review_feedback_content.get('approved', False)
        suggestions = review_feedback_content.get('suggestions', [])
        # comments = review_feedback_content.get('comments', "") # Already logged if needed

        self.log(f"Reviewer Feedback - Approved: {approved}, Suggestions: {len(suggestions)}")

        final_prd_to_send = original_prd_data
        regeneration_attempted = False

        if not approved and suggestions: # If not approved AND there are suggestions, try to regenerate
            self.log(f"PRD not approved or has suggestions. Attempting regeneration based on feedback (Suggestions: {suggestions}).")
            self.status = "regenerating_prd"
            regeneration_attempted = True

            refinement_prompt = (
                f"The initial user requirement was: '{user_requirement_text}'.\n"
                f"The first generated Product Requirement Document (PRD) was:\n"
                f"```json\n{json.dumps(original_prd_data, indent=2)}\n```\n"
                f"A review of this PRD provided the following suggestions for improvement:\n"
            )
            for i, suggestion in enumerate(suggestions):
                refinement_prompt += f"- {suggestion}\n"

            refinement_prompt += (
                "\nPlease generate a revised PRD as a JSON string, addressing these suggestions. "
                "The revised PRD must follow the same JSON structure as before: "
                "an object with keys 'project_name', 'description', 'features' (list of strings), and 'target_platform'. "
                "Ensure the JSON is well-formed and complete. Only output the JSON string."
            )

            system_prompt_refine = "You are a Product Manager AI. You previously created a PRD which received feedback. Your task is to revise the PRD based on the provided suggestions to improve its quality, clarity, and completeness, maintaining the JSON format."

            self.log(f"Refinement prompt (first 200 chars): {refinement_prompt[:200]}...")

            revised_prd_json_string, error = self.ollama_integration.generate_text_sync(
                model_name=self.model_name,
                prompt=refinement_prompt,
                system_prompt=system_prompt_refine
            )

            if error:
                self.log(f"Error during PRD regeneration from Ollama: {error}")
                self.status = "error_regenerating_prd"
            elif not revised_prd_json_string:
                self.log("Ollama returned an empty response for PRD regeneration.")
                self.status = "error_regenerating_prd"
            else:
                self.log(f"Raw revised PRD JSON string from Ollama: {revised_prd_json_string[:300]}...")
                try:
                    clean_revised_json_string = revised_prd_json_string.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                    revised_prd_data = json.loads(clean_revised_json_string)

                    required_keys = ['project_name', 'description', 'features', 'target_platform']
                    if not all(key in revised_prd_data for key in required_keys) or not isinstance(revised_prd_data['features'], list):
                        self.log(f"Revised PRD JSON is missing keys or 'features' is not a list. Sticking with original. Revised data: {revised_prd_data}")
                    else:
                        self.log(f"Successfully parsed revised PRD: {revised_prd_data}")
                        final_prd_to_send = revised_prd_data
                        self.status = "prd_regenerated_and_approved"
                except json.JSONDecodeError as e:
                    self.log(f"Failed to decode revised PRD JSON string: {e}. Sticking with original PRD.")
                    self.log(f"Problematic revised JSON: {revised_prd_json_string[:500]}")
                    self.status = "error_parsing_revised_prd"
        elif approved:
            self.log("PRD was approved by reviewer, or no actionable suggestions provided. Sending original PRD.")
        else:
            self.log("PRD was not approved, but no suggestions to act upon. Sending original PRD with a warning.")

        if not final_prd_to_send:
            self.log("Critical Error: No PRD data available to send (original or revised).")
            self.status = "error_pm_logic_critical"
            error_msg_content = {"type": "pm_error", "detail": "Critical error, no PRD available after review process."}
            message_queue.send_message(self.agent_id, original_user_id, error_msg_content)
            return False

        engineer_message_content = {
            "type": "prd_document",
            "prd": final_prd_to_send,
            "respond_to_agent_id": original_user_id,
            "review_was_conducted": True,
            "prd_was_revised": regeneration_attempted and (final_prd_to_send != original_prd_data)
        }
        message_queue.send_message(
            sender_id=self.agent_id,
            recipient_id=engineer_agent_id,
            content=engineer_message_content
        )
        if regeneration_attempted and (final_prd_to_send != original_prd_data):
            self.log(f"Sent REVISED PRD to engineer '{engineer_agent_id}'.")
        else:
            self.log(f"Sent ORIGINAL PRD to engineer '{engineer_agent_id}' (Review approved, no suggestions, or regeneration failed/skipped).")

        self.status = "prd_sent_to_engineer"
        return True

if __name__ == '__main__':
    print("--- ProductManagerAgent Test (with Ollama Integration) ---")

    oi = OllamaIntegration()
    pm_model = "qwen2:0.5b"

    models, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}. Aborting test.")
        sys.exit(1)

    available_models = []
    if models: # Ensure models is not None
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

    print("\nNote: Full workflow including review request and handling feedback")
    print("is typically tested in a multi-agent demo script using AgentMessageQueue.")
    print("The methods 'handle_requirement_and_send_prd' and 'handle_prd_review_feedback'")
    print("orchestrate these interactions.")
    print("\nThe PRD refinement logic in 'handle_prd_review_feedback'")
    print("involves receiving feedback and potentially re-generating the PRD using Ollama.")
    print("This iterative process is best tested in the full multi-agent demo script,")
    print("where actual reviewer feedback (simulated or LLM-generated) can be provided.")
