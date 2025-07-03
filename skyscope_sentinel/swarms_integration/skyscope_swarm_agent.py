import os
from dotenv import load_dotenv

# Ensure project root is in path for sibling imports
import sys
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from swarms import Agent as SwarmsBaseAgent # Alias to avoid confusion if we have other Agent classes
from swarms.models import Ollama # Import specific Ollama LLM wrapper from swarms

from skyscope_sentinel.agent_identity import generate_agent_identity
from skyscope_sentinel.config import Config # To get Ollama settings

# Load .env for local development if API keys are needed by underlying models/tools via swarms
load_dotenv()

# Initialize global config to get settings
# In a running app, this global_config instance would be updated by main.py from GUI settings
# For standalone testing of this module, it will use env vars or defaults.
# It's better if this module receives config rather than relying on a global that might not be updated yet.
# However, for now, let's follow the pattern of other agent files.
# A better approach: Pass config object or relevant settings during instantiation.

# global_skyscope_config = Config() # This might lead to issues if not updated by GUI.
# For more robust behavior, SkyscopeSwarmAgent should ideally receive LLM config directly.

class SkyscopeSwarmAgent(SwarmsBaseAgent):
    """
    A Skyscope Sentinel agent that leverages the kyegomez/swarms framework.
    It integrates Skyscope's identity system and is configured to use
    Ollama as its primary LLM backend based on Skyscope's global config.
    """
    def __init__(
        self,
        # Skyscope specific parameters for identity and context
        department_name: str, # Renamed from 'department' for clarity with specialized agents
        role_in_department: str = None, # Specific role for this agent instance
        agent_id: str = None, # Skyscope specific internal ID, auto-generated if None
        goal: str = "Achieve objectives as part of the Skyscope Sentinel enterprise.", # Generic goal

        # swarms.Agent native parameters:
        agent_name: str = None, # Name for the swarms.Agent, derived from identity if None
        agent_description: str = None, # Description for swarms.Agent, derived if None
        system_prompt: str = None, # System prompt, constructed from identity if None
        llm = None, # LLM instance, will be configured internally using Ollama if None
        max_loops: int = 5,
        autosave: bool = False,
        verbose: bool = True,
        tools: list = None,
        long_term_memory=None,
        # Removed system_prompt, agent_name, agent_description from here as they are now higher up
        **kwargs
    ):
        # Generate Skyscope identity using department_name and role_in_department
        self.skyscope_identity = generate_agent_identity(
            department=department_name,
            preferred_role=role_in_department # agent_identity can use this to select a title
        )

        # Handle Skyscope agent_id: use provided, or generate from name + part of UUID for uniqueness
        if agent_id:
            self.agent_id = agent_id
        else:
            import uuid
            unique_suffix = str(uuid.uuid4()).split('-')[0] # Short unique suffix
            self.agent_id = f"{self.skyscope_identity.get('first_name', 'Agent')}_{unique_suffix}"

        # Construct agent_name for swarms.Agent (can be more descriptive)
        # If agent_name is explicitly passed, use it, otherwise derive from Skyscope identity.
        _agent_name_for_swarms = agent_name or f"{self.skyscope_identity.get('first_name', 'SwarmAgent')}_{self.skyscope_identity.get('last_name', self.agent_id)}"

        # Construct agent_description for swarms.Agent
        _agent_description_for_swarms = agent_description or (
            f"An AI agent from the {self.skyscope_identity.get('department', 'General')} department, "
            f"working as a {self.skyscope_identity.get('employee_title', 'Specialist')} for Skyscope Sentinel Intelligence."
            f" Skyscope ID: {self.agent_id}."
        )

        # Construct system_prompt for swarms.Agent
        # Use role_in_department if provided, else use the generated employee_title
        effective_role = role_in_department or self.skyscope_identity.get('employee_title', 'Autonomous Specialist')
        _system_prompt_for_swarms = system_prompt or (
            f"You are {_agent_name_for_swarms}, a designated {effective_role} "
            f"within the {self.skyscope_identity.get('department', 'Unspecified')} department of Skyscope Sentinel Intelligence. "
            f"Your expertise includes: {', '.join(self.skyscope_identity.get('expertise', ['general problem solving']))}. "
            f"Your current overarching goal is: {goal}. "
            f"You are part of a larger AI-driven enterprise aiming for autonomous income generation. "
            f"Collaborate effectively, utilize your tools wisely, and focus on achieving tangible outcomes."
        )

        # Configure LLM if not provided (llm parameter)
        _llm = llm
        if _llm is None:
            # Fetch Ollama configuration from Skyscope's global config
            # This relies on global_config being appropriately initialized and updated by main.py
            # For direct testing of this module, ensure config.py loads from .env or has defaults.
            temp_config = Config() # Create a temporary config instance to get settings
            ollama_model_str = temp_config.get_ollama_model_name()
            ollama_base_url = temp_config.get_ollama_base_url()

            if ollama_model_str:
                # swarms.models.Ollama expects model name without "ollama/"
                cleaned_model_name = ollama_model_str.replace("ollama/", "")
                _llm = Ollama(model=cleaned_model_name, base_url=ollama_base_url)
                print(f"[{_agent_name_for_swarms}] SkyscopeSwarmAgent: Using Ollama LLM: {cleaned_model_name} at {ollama_base_url}")
            else:
                print(f"[{_agent_name_for_swarms}] SkyscopeSwarmAgent: Warning - Ollama model not configured in Skyscope Config. LLM might not be set.")
                # swarms.Agent might use its own default if _llm is None, or raise an error.

        super().__init__(
            agent_name=_agent_name_for_swarms,
            agent_description=_agent_description_for_swarms,
            system_prompt=_system_prompt_for_swarms,
            llm=_llm,
            max_loops=max_loops,
            autosave=autosave,
            verbose=verbose,
            tools=tools or [],
            long_term_memory=long_term_memory,
            # Pass through any other kwargs for SwarmsBaseAgent
            **kwargs
        )

        print(f"[SkyscopeSwarmAgent {_agent_name_for_swarms}] Initialized. Skyscope ID: {self.agent_id}, Role: {effective_role}, Department: {self.skyscope_identity.get('department')}")

    def get_skyscope_identity_summary(self) -> str:
        """Returns a short summary of the Skyscope agent's identity."""
        return (f"SkyscopeID: {self.agent_id}, Name: {self.skyscope_identity.get('first_name')} {self.skyscope_identity.get('last_name')}, "
                f"Title: {self.skyscope_identity.get('employee_title')}, Dept: {self.skyscope_identity.get('department')}, Swarms Agent Name: {self.agent_name}")

if __name__ == '__main__':
    # This import needs to be here for the __main__ block to run directly
    from skyscope_sentinel.agent_identity import initialize_identity_manager, set_founder_details

    # Setup for testing SkyscopeSwarmAgent
    temp_config_for_main_test = Config()
    set_founder_details(
        temp_config_for_main_test.founder_name,
        temp_config_for_main_test.founder_contact,
        temp_config_for_main_test.business_name
    )
    initialize_identity_manager()


    print("--- Testing SkyscopeSwarmAgent ---")

    # This test assumes Ollama is running and serving a model like 'mistral' (or whatever default is in config.py)
    # It also assumes that if SERPER_API_KEY is needed for a tool, it's in .env

    # Example: Create a simple search tool for testing
    from crewai_tools import DuckDuckGoSearchRun # swarms might have its own tool registry or base class
                                             # For now, using a known simple tool.
                                             # We'll need to adapt/wrap tools for swarms properly later.

    # Note: swarms.Agent expects tools to be callables or instances of its own BaseTool or similar.
    # We will use the callable functions we defined.

    # Import the tools
    from skyscope_sentinel.tools.search_tools import duckduckgo_search_function, serper_search_function
    from skyscope_sentinel.tools.browser_tools import browse_web_page_and_extract_text
    from skyscope_sentinel.tools.code_execution_tools import execute_python_code_in_e2b
    from skyscope_sentinel.tools.file_io_tools import write_file, read_file, list_files

    # Prepare the tools list
    # For testing, we'll use DuckDuckGo by default. Serper would require SERPER_API_KEY.
    # E2B tool would require E2B_API_KEY.

    available_tools = [duckduckgo_search_function, browse_web_page_and_extract_text, write_file, read_file, list_files]

    # Conditionally add Serper if key is available (from .env for this standalone test)
    if os.getenv("SERPER_API_KEY"):
        available_tools.insert(0, serper_search_function) # Prioritize if available
        print("[Test] SERPER_API_KEY found, adding Serper tool.")
    else:
        print("[Test] SERPER_API_KEY not found, Serper tool will not be available to test agent.")

    # Conditionally add E2B tool if key is available
    if os.getenv("E2B_API_KEY"):
        available_tools.append(execute_python_code_in_e2b)
        print("[Test] E2B_API_KEY found, adding E2B code execution tool.")
    else:
        print("[Test] E2B_API_KEY not found, E2B tool will not be available to test agent.")


    try:
        research_specialist_with_tools = SkyscopeSwarmAgent(
            # agent_id="SSA_ToolTester001", # Will be auto-generated
            department_name="Market Research & Analysis", # Use department_name
            role_in_department="Lead Tooling Specialist", # Use role_in_department
            goal="Utilize available tools to find information, process it, and store it.",
            tools=available_tools,
            verbose=True,
            # Crucially, for swarms.Agent to use tools effectively with many LLMs,
            # we might need to ensure the LLM supports function calling or that
            # swarms formats the tool descriptions in a way the LLM can generate structured calls.
            # The `Ollama` model class in swarms is built on LiteLLM, which can handle this.
        )
        print(research_specialist_with_tools.get_skyscope_identity_summary())
        # The underlying swarms.Agent has agent_name, agent_description, system_prompt attributes
        print(f"Swarms Agent Name: {research_specialist_with_tools.agent_name}")
        # print(f"Swarms Agent Description: {research_specialist_with_tools.agent_description}") # This might be long
        # print(f"Swarms System Prompt: {research_specialist_with_tools.system_prompt[:300]}...") # Print first 300 chars

        # Test a simple run that might use a search tool
        # The default swarms.Agent.run() takes a task string.
        # Ensure your Ollama server is running with the configured model.

        # Check if Ollama is actually available for a more informative test run
        ollama_model_to_test = temp_config_for_main_test.get_ollama_model_name()
        if ollama_model_to_test:
            print(f"\nAttempting task with agent (using Ollama model: {ollama_model_to_test})...")
            task_output_search = research_specialist_with_tools.run("Search for the current weather in Paris and write it to a file named 'paris_weather.txt'. Then read the file and tell me its content.")
            print(f"\nTask Output for 'Search, Write, Read Paris Weather':\n{task_output_search}")

            # Test a task that might use code execution (if E2B key is present and tool added)
            if execute_python_code_in_e2b in research_specialist_with_tools.tools:
                task_output_code = research_specialist_with_tools.run(
                    "Write a python script to calculate 2 + 2 and print the result. Then execute this script."
                )
                print(f"\nTask Output for 'Code Execution Test':\n{task_output_code}")
            else:
                print("\nSkipping code execution test as E2B tool is not available.")
        else:
            print("\n[Test] Ollama model not configured. Skipping agent run tests.")


    except Exception as e:
        print(f"Error during SkyscopeSwarmAgent test: {e}")
        print("Ensure Ollama is running and configured, and any necessary API keys for tools are set if tools are used.")

    print("\n--- SkyscopeSwarmAgent Test Complete ---")
