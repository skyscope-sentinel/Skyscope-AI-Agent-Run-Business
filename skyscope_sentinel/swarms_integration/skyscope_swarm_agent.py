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
        agent_id: str, # Skyscope specific internal ID
        department: str,
        role: str = None, # Specific role for the system prompt, derived from identity if None
        goal: str = "Achieve objectives as part of the Skyscope Sentinel enterprise.", # Generic goal
        # swarms.Agent specific parameters:
        llm = None, # Will be configured internally if None
        max_loops: int = 5,
        autosave: bool = False,
        verbose: bool = True,
        tools: list = None,
        long_term_memory=None,
        system_prompt: str = None, # Will be constructed if None
        agent_name: str = None, # Will be derived from identity if None
        agent_description: str = None, # Will be derived from identity if None
        **kwargs
    ):
        self.skyscope_identity = generate_agent_identity(department=department)
        self.agent_id = agent_id # Skyscope's internal agent_id

        # Construct agent_name, agent_description, and system_prompt for swarms.Agent
        # from the Skyscope identity if not provided.

        # Name for swarms.Agent (can be more descriptive than just agent_id)
        _agent_name = agent_name or f"{self.skyscope_identity.get('first_name', 'Agent')}_{self.skyscope_identity.get('last_name', self.agent_id)}"

        _agent_description = agent_description or (
            f"An AI agent from the {self.skyscope_identity.get('department', 'General')} department, "
            f"working as a {self.skyscope_identity.get('employee_title', 'Specialist')} for Skyscope Sentinel Intelligence."
        )

        _role = role or self.skyscope_identity.get('employee_title', 'Autonomous Agent')

        _system_prompt = system_prompt or (
            f"You are {_agent_name}, a {self.skyscope_identity.get('employee_title', 'Specialist')} "
            f"in the {self.skyscope_identity.get('department', 'General')} department of Skyscope Sentinel Intelligence. "
            f"Your expertise includes: {', '.join(self.skyscope_identity.get('expertise', []))}. "
            f"Your current goal is: {goal}. Strive for excellence and collaboration."
        )

        # Configure LLM if not provided
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
                print(f"[{_agent_name}] SkyscopeSwarmAgent: Using Ollama LLM: {cleaned_model_name} at {ollama_base_url}")
            else:
                print(f"[{_agent_name}] SkyscopeSwarmAgent: Warning - Ollama model not configured in Skyscope Config. LLM might not be set.")
                # Potentially fall back to a default if swarms.Agent doesn't have one
                # or raise an error. For now, swarms.Agent might use its own default if _llm is None.

        super().__init__(
            agent_name=_agent_name,
            agent_description=_agent_description,
            system_prompt=_system_prompt,
            llm=_llm,
            max_loops=max_loops,
            autosave=autosave,
            verbose=verbose,
            tools=tools or [],
            long_term_memory=long_term_memory,
            # Pass through any other kwargs for SwarmsBaseAgent
            **kwargs
        )

        print(f"[SkyscopeSwarmAgent {_agent_name}] Initialized. Skyscope ID: {self.agent_id}, Role: {_role}, Department: {self.skyscope_identity.get('department')}")

    def get_skyscope_identity_summary(self) -> str:
        """Returns a short summary of the Skyscope agent's identity."""
        return (f"SkyscopeID: {self.agent_id}, Name: {self.skyscope_identity.get('first_name')} {self.skyscope_identity.get('last_name')}, "
                f"Title: {self.skyscope_identity.get('employee_title')}, Dept: {self.skyscope_identity.get('department')}")

if __name__ == '__main__':
    print("--- Testing SkyscopeSwarmAgent ---")

    # This test assumes Ollama is running and serving a model like 'qwen2:0.5b' (or whatever default is in config.py)
    # It also assumes that if SERPER_API_KEY is needed for a tool, it's in .env

    # Example: Create a simple search tool for testing
    from crewai_tools import DuckDuckGoSearchRun # swarms might have its own tool registry or base class
                                             # For now, using a known simple tool.
                                             # We'll need to adapt/wrap tools for swarms properly later.feature/phase1-agent-gui-owl-setup
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
            agent_id="SSA_ToolTester001",
            department="Researchers",
            role="Autonomous Research and Task Execution Agent",
            goal="Utilize available tools to find information, process it, and store it.",
            tools=available_tools,
            verbose=True,
            # Crucially, for swarms.Agent to use tools effectively with many LLMs,
            # we might need to ensure the LLM supports function calling or that
            # swarms formats the tool descriptions in a way the LLM can generate structured calls.
            # The `Ollama` model class in swarms is built on LiteLLM, which can handle this.
        )
        print(research_specialist_with_tools.get_skyscope_identity_summary())
    # Note: swarms.Agent expects tools to be instances of its own BaseTool or similar.
    # For this basic test, we might not pass a tool or pass a very simple one.
    # Let's try without a complex tool first to test initialization and LLM.

    try:
        research_specialist = SkyscopeSwarmAgent(
            agent_id="SSA_Res001",
            department="Researchers",
            role="Information Retrieval Specialist", # Overrides title from identity for this specific role
            goal="Find the latest information on a given topic.",
            # tools=[DuckDuckGoSearchRun()] # Example tool, ensure it's compatible or wrapped for swarms
            verbose=True
        )
        print(research_specialist.get_skyscope_identity_summary())
main
        print(f"Swarm Agent Name: {research_specialist.agent_name}")
        print(f"Swarm Agent Description: {research_specialist.agent_description}")
        print(f"Swarm System Prompt: {research_specialist.system_prompt[:200]}...") # Print first 200 chars

 feature/phase1-agent-gui-owl-setup
        # Test a simple run that might use a search tool
        # The default swarms.Agent.run() takes a task string.
        # Ensure your Ollama server is running with the configured model.
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
        # Test a simple run
        # The default swarms.Agent.run() takes a task string.
        # Ensure your Ollama server is running with the configured model.
        task_output = research_specialist.run("What is the capital of France?")
        print(f"\nTask Output for 'Capital of France':\n{task_output}")

        task_output_tech = research_specialist.run("Explain quantum computing in simple terms.")
        print(f"\nTask Output for 'Quantum Computing':\n{task_output_tech}")
main


    except Exception as e:
        print(f"Error during SkyscopeSwarmAgent test: {e}")
        print("Ensure Ollama is running and configured, and any necessary API keys for tools are set if tools are used.")

    print("\n--- SkyscopeSwarmAgent Test Complete ---")
