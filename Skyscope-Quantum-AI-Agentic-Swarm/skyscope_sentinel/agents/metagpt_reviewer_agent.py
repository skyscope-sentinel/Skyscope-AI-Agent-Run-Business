import sys
import os
import json # For potentially loading/dumping complex data if needed

# Add project root to Python path for sibling imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from skyscope_sentinel.agents.base_agent import BaseAgent # Replaced by OwlBaseAgent
from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
 feat/foundational-agent-system
from skyscope_sentinel.agents.base_agent import BaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration

class ReviewerAgent(OwlBaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b", owl_toolkits: list = None): # Default model
        super().__init__(
            agent_id,
            department="Expert Panels", # Assigning to Expert Panels department
            role_description="An AI agent that reviews Product Requirement Documents (PRDs) and code for quality, clarity, and adherence to standards.",
            owl_toolkits=owl_toolkits
        )
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name
        self.status = "idle_reviewer"
        self.log(f"ReviewerAgent initialized with model '{self.model_name}'. Identity: {self.identity.get('first_name')} {self.identity.get('last_name')}, Title: {self.identity.get('employee_title')}")

    # Using self.log() from OwlBaseAgent for consistent logging format.
    # def log(self, message: str):
    #     print(f"[ReviewerAgent {self.agent_id}] {message}")

# from skyscope_sentinel.agents.base_agent import BaseAgent # Replaced by OwlBaseAgent
from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
from skyscope_sentinel.ollama_integration import OllamaIntegration

class ReviewerAgent(OwlBaseAgent):
    def __init__(self, agent_id: str, ollama_integration_instance: OllamaIntegration = None, model_name: str = "qwen2:0.5b", owl_toolkits: list = None): # Default model
        super().__init__(
            agent_id,
            department="Expert Panels", # Assigning to Expert Panels department
            role_description="An AI agent that reviews Product Requirement Documents (PRDs) and code for quality, clarity, and adherence to standards.",
            owl_toolkits=owl_toolkits
        )
        self.ollama_integration = ollama_integration_instance if ollama_integration_instance else OllamaIntegration()
        self.model_name = model_name
        self.status = "idle_reviewer"
        self.log(f"ReviewerAgent initialized with model '{self.model_name}'. Identity: {self.identity.get('first_name')} {self.identity.get('last_name')}, Title: {self.identity.get('employee_title')}")

    # Using self.log() from OwlBaseAgent for consistent logging format.
    # def log(self, message: str):
    #     print(f"[ReviewerAgent {self.agent_id}] {message}")
 main

    def review_prd(self, prd_data: dict) -> dict:
        self.log(f"Reviewing PRD for project: '{prd_data.get('project_name', 'Unknown Project')}' using Ollama model '{self.model_name}'.")
        self.status = "reviewing_prd_ollama"

        if not isinstance(prd_data, dict) or not prd_data:
            self.log("Invalid PRD data: not a dictionary or empty.")
            return {'approved': False, 'suggestions': ["PRD data must be a non-empty dictionary."], 'comments': "Invalid input."}

        try:
            prd_json_string = json.dumps(prd_data, indent=2)
        except TypeError as e:
            self.log(f"Could not serialize PRD to JSON for Ollama: {e}")
            return {'approved': False, 'suggestions': [f"PRD data could not be serialized: {e}"], 'comments': "Serialization error."}

        system_prompt = (
            "You are an expert QA Reviewer specializing in Product Requirement Documents (PRDs). "
            "Your task is to analyze the provided PRD (in JSON format) and provide feedback. "
            "Focus on: clarity, completeness, actionability of features, consistency, and overall coherence. "
            "Return your review as a single, well-formed JSON string object with three keys: "
            "'approved' (boolean, true if the PRD is good overall, false if significant issues exist), "
            "'suggestions' (a list of specific, actionable string suggestions for improvement, can be empty if approved and no suggestions), "
            "and 'comments' (a brief overall summary of your review as a string)."
            "Be critical but constructive."
        )
        prompt = (
            f"Please review the following Product Requirement Document (PRD):\n\n"
            f"```json\n{prd_json_string}\n```\n\n"
            "Provide your review ONLY as a JSON string object with 'approved', 'suggestions', and 'comments' keys."
        )

        llm_response_str, error = self.ollama_integration.generate_text_sync(
            model_name=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt
        )

        if error:
            self.log(f"Error during Ollama call for PRD review: {error}")
            self.status = "error_review_prd"
            return {'approved': False, 'suggestions': [f"LLM call failed: {error}"], 'comments': "Failed to get LLM review."}

        if not llm_response_str:
            self.log("Ollama returned an empty response for PRD review.")
            self.status = "error_review_prd"
            return {'approved': False, 'suggestions': ["LLM returned empty response."], 'comments': "Empty LLM response."}

        self.log(f"Raw PRD review from Ollama: {llm_response_str[:300]}...")

        try:
            # Clean potential markdown fences around the JSON
            clean_json_str = llm_response_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            review_result = json.loads(clean_json_str)

            # Validate structure of LLM response
            if not all(k in review_result for k in ['approved', 'suggestions', 'comments']):
                raise ValueError("LLM review JSON missing required keys.")
            if not isinstance(review_result['approved'], bool):
                raise ValueError("'approved' key must be a boolean.")
            if not isinstance(review_result['suggestions'], list):
                raise ValueError("'suggestions' key must be a list.")
            if not isinstance(review_result['comments'], str):
                raise ValueError("'comments' key must be a string.")

            self.log(f"PRD Review complete. Approved by LLM: {review_result['approved']}. Suggestions: {len(review_result['suggestions'])}")
            self.status = "prd_review_complete"
            return review_result
        except (json.JSONDecodeError, ValueError) as e:
            self.log(f"Failed to decode or validate LLM review JSON: {e}")
            self.log(f"Problematic LLM response for PRD review: {llm_response_str[:500]}")
            self.status = "error_parsing_review"
            return {'approved': False, 'suggestions': [f"Invalid LLM review format: {e}"], 'comments': "Could not parse LLM review."}

    def review_code(self, code_string: str, language: str = "python", prd_data: dict = None) -> dict: # Added prd_data for context
        self.log(f"Reviewing code (lang: {language}) using Ollama model '{self.model_name}'. Code snippet: '{code_string[:100]}...'")
        self.status = "reviewing_code_ollama"

        if not isinstance(code_string, str) or not code_string.strip():
            self.log("Invalid code string: empty or not a string.")
            return {'approved': False, 'suggestions': ["Code string must be non-empty."], 'comments': "Invalid input."}

        prd_context_str = ""
        if prd_data and isinstance(prd_data, dict):
            try:
                prd_context_str = f"The code should implement the following PRD:\n```json\n{json.dumps(prd_data, indent=2)}\n```\n"
            except TypeError:
                prd_context_str = "Note: PRD data was provided but could not be serialized for context.\n"

        system_prompt = (
            f"You are an expert QA Reviewer specializing in {language} code. "
            "Your task is to analyze the provided source code and provide feedback. "
            "Focus on: basic code smells (e.g., overly complex parts, magic numbers), "
            "conceptual adherence to the PRD if provided, presence of basic error handling, "
            "clarity, and readability (e.g., comments, naming). "
            "Return your review as a single, well-formed JSON string object with three keys: "
            "'approved' (boolean, true if the code is good overall, false if significant issues exist), "
            "'suggestions' (a list of specific, actionable string suggestions for improvement or issues found, can be empty), "
            "and 'comments' (a brief overall summary of your review as a string). "
            "Be critical but constructive. If the code is trivial (e.g. less than 5 lines), be less strict."
        )
        prompt = (
            f"{prd_context_str}"
            f"Please review the following {language} source code:\n\n"
            f"```{language}\n{code_string}\n```\n\n"
            "Provide your review ONLY as a JSON string object with 'approved', 'suggestions', and 'comments' keys."
        )

        llm_response_str, error = self.ollama_integration.generate_text_sync(
            model_name=self.model_name,
            prompt=prompt,
            system_prompt=system_prompt
        )

        if error:
            self.log(f"Error during Ollama call for code review: {error}")
            self.status = "error_review_code"
            return {'approved': False, 'suggestions': [f"LLM call failed: {error}"], 'comments': "Failed to get LLM review."}

        if not llm_response_str:
            self.log("Ollama returned an empty response for code review.")
            self.status = "error_review_code"
            return {'approved': False, 'suggestions': ["LLM returned empty response."], 'comments': "Empty LLM response."}

        self.log(f"Raw code review from Ollama: {llm_response_str[:300]}...")

        try:
            clean_json_str = llm_response_str.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            review_result = json.loads(clean_json_str)

            if not all(k in review_result for k in ['approved', 'suggestions', 'comments']):
                raise ValueError("LLM review JSON missing required keys.")
            # Further type validation as in review_prd can be added here.

            self.log(f"Code Review complete. Approved by LLM: {review_result['approved']}. Suggestions: {len(review_result['suggestions'])}")
            self.status = "code_review_complete"
            return review_result
        except (json.JSONDecodeError, ValueError) as e:
            self.log(f"Failed to decode or validate LLM review JSON: {e}")
            self.log(f"Problematic LLM response for code review: {llm_response_str[:500]}")
            self.status = "error_parsing_review"
            return {'approved': False, 'suggestions': [f"Invalid LLM review format: {e}"], 'comments': "Could not parse LLM review."}

if __name__ == '__main__':
    print("--- ReviewerAgent Test (Iteration 2: Ollama-Powered Review) ---")

    oi = OllamaIntegration()
    reviewer_model = "qwen2:0.5b" # Or "gemma:2b" or other suitable model

    models, error = oi.list_models_sync()
    if error:
        print(f"ERROR: Could not list Ollama models: {error}. Aborting test.")
        sys.exit(1)

    available_models = []
    if models: # Ensure models is not None
        available_models = [m.get('name') for m in models if m.get('name')]

    if reviewer_model not in available_models:
        print(f"ERROR: Test model '{reviewer_model}' for Reviewer Agent not found.")
        print(f"Available models: {available_models}")
        print(f"Please pull the model first (e.g., `ollama pull {reviewer_model}`). Aborting test.")
        sys.exit(1)

    reviewer_agent = ReviewerAgent(agent_id="Reviewer_001", ollama_integration_instance=oi, model_name=reviewer_model)
    print(f"Initial status: {reviewer_agent.get_status()}")

    # Test PRD Review with Ollama
    print("\n--- Testing PRD Review with Ollama ---")
    sample_prd = {
        "project_name": "Simple Task Logger",
        "description": "A very basic CLI tool to log tasks and view them. Not much detail given.",
        "features": ["Add new task with a description.", "View all tasks.", "Mark task as complete (maybe)."],
        "target_platform": "CLI"
    }
    print(f"Reviewing Sample PRD: {json.dumps(sample_prd, indent=2)}")
    prd_review_result = reviewer_agent.review_prd(sample_prd)
    print(f"Ollama PRD Review Result: {json.dumps(prd_review_result, indent=2)}")
    assert 'approved' in prd_review_result
    assert 'suggestions' in prd_review_result
    assert 'comments' in prd_review_result

    # Test Code Review with Ollama
    print("\n--- Testing Code Review with Ollama ---")
    sample_python_code = (
        "def add(x, y): return x+y\n"
        "def main():\n"
        "  res = add(5,3)\n"
        "  print(f'Result is {res}')\n"
        # "  # TODO: Add subtraction\n"
    )
    print(f"Reviewing Sample Python Code:\n{sample_python_code}")
    code_review_result = reviewer_agent.review_code(sample_python_code, language="python", prd_data=sample_prd)
    print(f"Ollama Code Review Result: {json.dumps(code_review_result, indent=2)}")
    assert 'approved' in code_review_result
    assert 'suggestions' in code_review_result
    assert 'comments' in code_review_result

    print("\n--- Testing with potentially problematic PRD for review ---")
    problem_prd = {
        "project_name": "",
        "description": "A tool.",
        "features": [123, True],
        "target_platform": "Unknown"
    }
    print(f"Reviewing Problematic PRD: {json.dumps(problem_prd, indent=2)}")
    problem_prd_review_result = reviewer_agent.review__prd(problem_prd)
    print(f"Ollama Problematic PRD Review Result: {json.dumps(problem_prd_review_result, indent=2)}")
    # Assertion depends on LLM's strictness; it might still pass but give many suggestions.
    # For this test, we'll be optimistic that it flags issues or our parsing of its response fails if it's confused.
    # A more robust test would be to check if 'suggestions' list is non-empty if 'approved' is True for problematic cases.
    if problem_prd_review_result['approved'] is True and not problem_prd_review_result['suggestions']:
        print("WARNING: Problematic PRD was approved by LLM without suggestions. This might indicate a lenient LLM or prompt issue.")
    assert problem_prd_review_result['approved'] is False or len(problem_prd_review_result['suggestions']) > 0


    print("\nAll Ollama ReviewerAgent tests completed for Iteration 2 (basic execution).")
