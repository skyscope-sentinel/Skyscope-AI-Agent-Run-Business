import os
from dotenv import load_dotenv

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool # Assuming SerperDevTool is available via crewai[tools]

# Attempt to load environment variables from .env file for API keys
load_dotenv()

# Assuming skyscope_sentinel is in PYTHONPATH or installed
# For development, ensure the root of skyscope_sentinel project is in PYTHONPATH
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from skyscope_sentinel.owl_integration.owl_base_agent import OwlBaseAgent
from skyscope_sentinel.config import Config # We'll create this simple config module

# --- LLM Configuration ---
# In a real app, this might be more sophisticated, e.g., reading from a global config
# or the GUI settings. For now, we define it here or expect it from Skyscope's main config.
# If Config().get_ollama_model_name() and Config().get_ollama_base_url() are available:
# OLLAMA_MODEL = Config().get_ollama_model_name() if Config().get_ollama_model_name() else "ollama/qwen2:0.5b"
# OLLAMA_BASE_URL = Config().get_ollama_base_url() if Config().get_ollama_base_url() else "http://localhost:11434"

# Fallback if Config isn't fully implemented yet or values are missing
try:
    OLLAMA_MODEL = Config().get_ollama_model_name()
    OLLAMA_BASE_URL = Config().get_ollama_base_url()
    if not OLLAMA_MODEL or not OLLAMA_BASE_URL: # Handle case where methods exist but return None/empty
        print("Warning: Ollama config not fully loaded from Config object. Using defaults for ResearchCrew.")
        OLLAMA_MODEL = "ollama/qwen2:0.5b" # Default model for this crew
        OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama URL
except AttributeError: # If Config or methods don't exist yet
    print("Warning: Config object or methods not found. Using default Ollama settings for ResearchCrew.")
    OLLAMA_MODEL = "ollama/qwen2:0.5b" # Default model for this crew
    OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama URL


# It's good practice to ensure API keys are loaded, though CrewAI tools might also check this.
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
if not SERPER_API_KEY:
    print("Warning: SERPER_API_KEY not found in environment variables. SerperDevTool may not function.")

# Initialize tools
# Use SerperDevTool if API key is available, otherwise fallback to DuckDuckGoSearchTool
if SERPER_API_KEY:
    search_tool = SerperDevTool()
    print("[ResearchCrew] Using SerperDevTool for search.")
else:
    from skyscope_sentinel.tools.search_tools import DuckDuckGoSearchTool
    search_tool = DuckDuckGoSearchTool()
    print("[ResearchCrew] SERPER_API_KEY not found. Using DuckDuckGoSearchTool as fallback.")

# Define a shared LLM configuration for this crew's agents
# Recent CrewAI versions might use specific classes like `from crewai.llms import OllamaLLM`
# For now, using the generic approach from docs if specific class not confirmed for this version.
# This might need adjustment based on the exact CrewAI version and its Ollama integration specifics.
try:
    from crewai.llms import OllamaLLM # Try importing specific OllamaLLM
    ollama_llm_config = OllamaLLM(model=OLLAMA_MODEL.replace("ollama/", ""), base_url=OLLAMA_BASE_URL)
    print(f"Using OllamaLLM specific class for ResearchCrew with model: {OLLAMA_MODEL}, URL: {OLLAMA_BASE_URL}")
except ImportError:
    from crewai import LLM # Fallback to generic LLM class if OllamaLLM is not found
    ollama_llm_config = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"Warning: crewai.llms.OllamaLLM not found. Using generic LLM class for ResearchCrew with model: {OLLAMA_MODEL}, URL: {OLLAMA_BASE_URL}")


class CrewResearchAgent(OwlBaseAgent, Agent): # Multiple inheritance
    """
    Adapter class to make OwlBaseAgent compatible with CrewAI's Agent structure
    while retaining OwlBaseAgent's identity and logging.
    """
    def __init__(self, owl_agent_id: str, department: str,
                 role: str, goal: str, backstory: str,
                 llm: any = None, tools: list = None, # 'any' for llm to accept CrewAI's llm object
                 allow_delegation: bool = False, verbose: bool = True, **kwargs):
        # Initialize OwlBaseAgent part
        OwlBaseAgent.__init__(self, agent_id=owl_agent_id, department=department, role_description=backstory)

        # Initialize CrewAI Agent part
        # We need to pass the arguments CrewAI's Agent expects.
        # OwlBaseAgent's __init__ already prints its own init message.
        Agent.__init__(self, role=role, goal=goal, backstory=backstory, llm=llm,
                       tools=tools or [], allow_delegation=allow_delegation, verbose=verbose, **kwargs)
        self.log(f"CrewResearchAgent (as CrewAI Agent) '{self.role}' part initialized.")


# --- Define Agents for the Research Crew ---
researcher = CrewResearchAgent(
    owl_agent_id="ResCrew001",
    department="Researchers",
    role='Market Research Analyst',
    goal='Gather comprehensive and unbiased information on a given topic, focusing on market trends, opportunities, and key players.',
    backstory=(
        "You are a meticulous Market Research Analyst working for Skyscope Sentinel Intelligence. "
        "Your expertise lies in sifting through web data to find actionable insights. "
        "You are adept at using search engines and web browsing tools to uncover relevant information."
    ),
    llm=ollama_llm_config,
    tools=[search_tool], # Pass the selected search_tool (Serper or DuckDuckGo)
    allow_delegation=False,
    verbose=True
)

analyzer = CrewResearchAgent(
    owl_agent_id="ResCrew002",
    department="Strategists", # Or Researchers, depending on focus
    role='Strategic Insights Analyst',
    goal='Distill gathered research into concise summaries, identify key patterns, potential opportunities, and actionable recommendations.',
    backstory=(
        "You are a Strategic Insights Analyst at Skyscope Sentinel Intelligence. "
        "You have a talent for seeing the bigger picture in complex data, identifying emerging trends, "
        "and translating raw information into strategic advice. You report directly on potential income-generating avenues."
    ),
    llm=ollama_llm_config,
    allow_delegation=False,
    verbose=True
)

# --- Define Tasks for the Research Crew ---
research_task = Task(
    description=(
        "Conduct in-depth web research on the topic: '{topic}'. "
        "Identify current market size, major trends, key players, growth projections, "
        "and any untapped niches or emerging opportunities related to this topic. "
        "Focus on information relevant for identifying potential income-generating activities."
    ),
    expected_output=(
        "A comprehensive report detailing: "
        "1. Overview of the market/topic. "
        "2. Current market size and growth projections (if available). "
        "3. Key trends and drivers. "
        "4. Major players and their strategies. "
        "5. Identified opportunities, particularly for new income streams or freelance work. "
        "6. Potential risks or challenges. "
        "Provide sources for key data points where possible."
    ),
    agent=researcher,
    # human_input=False # Can be set to True if clarification is needed during the task
)

analysis_task = Task(
    description=(
        "Review the research report provided on '{topic}'. "
        "Synthesize the information to produce a concise executive summary. "
        "Highlight the top 3-5 most promising income-generating opportunities identified, "
        "along with a brief rationale for each. Assess the potential viability and "
        "ease of entry for Skyscope Sentinel Intelligence for these opportunities."
    ),
    expected_output=(
        "An executive summary covering: "
        "1. Brief overview of the market/topic based on the research. "
        "2. Top 3-5 income-generating opportunities identified. "
        "3. For each opportunity: Rationale, estimated potential (if possible), ease of entry for Skyscope. "
        "4. Overall recommendation on whether Skyscope should pursue further investigation into this topic."
    ),
    agent=analyzer,
    context=[research_task] # Depends on the output of the research_task
)

# --- Assemble the Crew ---
market_research_crew = Crew(
    agents=[researcher, analyzer],
    tasks=[research_task, analysis_task],
    process=Process.sequential,  # Start with a sequential process
    verbose=2  # 0 for no log, 1 for agent actions, 2 for detailed logs
    # memory=True # Optional: enable memory for the crew
    # manager_llm=ollama_llm_config # For hierarchical process, if used
)

if __name__ == "__main__":
    print("--- Market Research Crew Definition ---")
    print(f"Researcher Agent: {researcher.role}, Tools: {[tool.name for tool in researcher.tools if hasattr(tool, 'name')]}")
    print(f"Analyzer Agent: {analyzer.role}")
    print(f"Crew Process: {market_research_crew.process}")
    print("\nTo run this crew, you would call market_research_crew.kickoff(inputs={'topic': 'your research topic'})")

    # Example Kickoff (requires API keys and Ollama running)
    if SERPER_API_KEY:
        print("\n--- Attempting Sample Crew Kickoff (requires Ollama running & SERPER_API_KEY) ---")
        try:
            # topic_to_research = "AI-powered tools for freelance writers in 2025"
            topic_to_research = "passive income opportunities for AI agents starting with no capital"
            inputs = {'topic': topic_to_research}
            result = market_research_crew.kickoff(inputs=inputs)

            print("\n--- Crew Kickoff Completed ---")
            print(f"Topic Researched: {topic_to_research}")
            print("Final Result from Crew:")
            print(result)
        except Exception as e:
            print(f"Error during sample crew kickoff: {e}")
            print("Please ensure Ollama is running and SERPER_API_KEY is correctly set in your .env file.")
    else:
        print("\nSkipping sample crew kickoff because SERPER_API_KEY is not set.")

    print("\n--- Research Crew Setup Complete ---")

# --- Simple Config Module (to be created as skyscope_sentinel/config.py) ---
# This is a placeholder. In a real app, this would load from GUI settings or a more robust config file.
# For now, create skyscope_sentinel/config.py with:
# class Config:
#     def __init__(self):
#         # In a real app, load these from settings manager
#         self.ollama_model_name = os.getenv("OLLAMA_MODEL", "ollama/qwen2:0.5b")
#         self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#
#     def get_ollama_model_name(self):
#         return self.ollama_model_name
#
#     def get_ollama_base_url(self):
#         return self.ollama_base_url
#
# if __name__ == '__main__':
#    conf = Config()
#    print(f"Ollama Model: {conf.get_ollama_model_name()}")
#    print(f"Ollama URL: {conf.get_ollama_base_url()}")
# --- End of Placeholder Config Module ---
