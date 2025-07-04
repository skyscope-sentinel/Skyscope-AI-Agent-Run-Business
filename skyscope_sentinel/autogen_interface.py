import asyncio
import os
from dotenv import load_dotenv

from autogen import UserProxyAgent, AssistantAgent
from autogen.coding import LocalCommandLineCodeExecutor # Example, if needed later

# Assuming skyscope_sentinel is in PYTHONPATH or installed
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from skyscope_sentinel.config import Config
from skyscope_sentinel.crews.research_crew import market_research_crew # Import the crew

# Load environment variables (e.g., for API keys if any AutoGen agent needed them directly)
load_dotenv()
config = Config()

# --- LLM Configuration for AutoGen Agents ---
# AutoGen's OpenAIWrapper can be used with Ollama by setting api_base and api_key (can be dummy for local)
# This needs to be compatible with how autogen-ext[openai] sets up clients.
# The exact model name for Ollama should be prefixed, e.g., "ollama/qwen2:0.5b"
# Or, if OllamaChatCompletionClient from autogen_ext.models.ollama is used, it's more direct.
# For simplicity with autogen-ext[openai], we often configure an "llm_config" dictionary.

# Get Ollama settings from our Config object
ollama_model_name_for_autogen = config.get_ollama_model_name()
# AutoGen's OpenAIWrapper often expects model name without "ollama/" prefix if api_type is "openai"
# and base_url points to Ollama. Let's try with and without prefix based on common patterns.
# For direct Ollama client (if available and used), it might be different.
# The key is that the `model` field in `config_list` matches what the Ollama server expects.

# Common way to configure AutoGen for Ollama using its OpenAI-compatible API:
llm_config_ollama = {
    "config_list": [
        {
            "model": ollama_model_name_for_autogen.replace("ollama/", ""), # Model name as known by Ollama
            "api_base": config.get_ollama_base_url() + "/v1",  # Standard Ollama OpenAI-compatible endpoint
            "api_type": "open_ai", # AutoGen uses "open_ai" for OpenAI-compatible endpoints
            "api_key": "NULL",  # Can be anything for local Ollama
        }
    ],
    "temperature": 0.7,
    # "cache_seed": 42, # Enable caching for consistency in tests/dev
}

print(f"[AutoGenInterface] LLM Config for AutoGen: {llm_config_ollama}")


# --- Function to be called by OrchestratorAgent ---
def run_market_research_crew(topic: str) -> str:
    """
    Kicks off the market research crew with the given topic and returns the result.
    This function will be registered with the AutoGen OrchestratorAgent.
    """
    print(f"[AutoGenInterface] Orchestrator received topic: '{topic}'. Kicking off Market Research Crew...")
    try:
        # The CrewAI crew's kickoff should be blocking for this synchronous function call
        result = market_research_crew.kickoff(inputs={'topic': topic})
        print(f"[AutoGenInterface] Market Research Crew finished. Result snippet: {str(result)[:200]}...")
        return str(result) # Ensure it's a string
    except Exception as e:
        print(f"[AutoGenInterface] Error running Market Research Crew: {e}")
        return f"Error encountered during market research: {str(e)}"

# --- Define AutoGen Agents ---

# User Proxy Agent: Represents the human user, gets input.
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER", # Initially NEVER, will be TERMINATE or ALWAYS for real interaction.
                              # For programmatic triggering, can be NEVER if messages are sent directly.
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False, # No code execution for user proxy
    # llm_config=llm_config_ollama, # UserProxyAgent usually doesn't need an LLM if it's just relaying
    system_message="You are a user proxy. You will receive a research topic and forward it. Reply TERMINATE when done.",
)

# Orchestrator Agent: Receives topic, calls the CrewAI crew, returns result.
orchestrator_agent = AssistantAgent(
    name="SkyscopeOrchestrator",
    llm_config=llm_config_ollama,
    system_message=(
        "You are an Orchestrator agent for Skyscope Sentinel Intelligence. "
        "Your role is to receive a research topic, delegate it to the specialized Market Research Crew, "
        "and return their findings. You have a tool 'run_market_research_crew' to do this."
    )
)

# Register the function as a tool for the orchestrator_agent
# Newer AutoGen versions might prefer a different way to register functions/tools,
# e.g., via `register_for_llm` and `register_for_execution` on the agent.
# For now, using a common pattern that involves describing it in a prompt
# and having the LLM generate the function call.
# A more robust way is to use agent.register_function.

orchestrator_agent.register_function(
    function_map={
        "run_market_research_crew": run_market_research_crew
    }
)

# --- Main function to initiate the AutoGen interaction ---
async def initiate_research_via_autogen(research_topic: str) -> str:
    """
    Initiates a chat between UserProxyAgent and OrchestratorAgent to perform research.
    """
    print(f"[AutoGenInterface] Initiating research for topic: '{research_topic}'")

    # Construct the initial message to trigger the function call via the orchestrator
    # This prompt needs to guide the orchestrator_agent to use its registered function.
    initial_message = (
        f"Please conduct market research on the following topic: '{research_topic}'. "
        f"Use the 'run_market_research_crew' tool to get the detailed analysis."
    )

    chat_result = await user_proxy.a_initiate_chat(
        recipient=orchestrator_agent,
        message=initial_message,
        max_turns=2, # User sends topic, Orchestrator calls tool and responds.
        summary_method="last_msg" # Get the last message as the result
    )

    # The result might be in the last message or need parsing from chat history
    if chat_result.chat_history and len(chat_result.chat_history) > 0:
        # Typically the orchestrator's response (which includes the crew's output) will be the last message.
        final_response = chat_result.summary # chat_result.chat_history[-1].get("content", "")
        print(f"[AutoGenInterface] Research completed. Final response snippet: {final_response[:200]}...")
        return final_response
    else:
        print("[AutoGenInterface] No response or error in AutoGen chat.")
        return "Error: No response received from the research process."


# --- Test function ---
async def main_test():
    print("--- Testing AutoGen Interface with Research Crew ---")
    # Ensure Ollama is running and SERPER_API_KEY (if used by crew) is set in .env

    # Load SERPER_API_KEY for the test, as the crew might need it.
    # The crew itself handles the fallback if not present.
    serper_key = os.getenv("SERPER_API_KEY")
    if not serper_key:
        print("WARNING: SERPER_API_KEY not found. Research quality may be affected if SerperDevTool is primary.")
        print("The crew will attempt to use DuckDuckGoSearchTool as a fallback.")

    topic = "future of AI-powered autonomous businesses"
    # topic = "best cat toys for active kittens" # A simpler topic for faster testing

    final_result = await initiate_research_via_autogen(topic)

    print("\n--- AutoGen Test Main Function Finished ---")
    print("Full Final Result from AutoGen Orchestration:")
    print(final_result)

if __name__ == "__main__":
    # To run this test:
    # 1. Ensure you have run `pip install -r requirements_gui.txt` (which includes autogen & crewai)
    # 2. Make sure your Ollama server is running (e.g., `ollama serve`)
    # 3. (Optional) Have a .env file in the project root with SERPER_API_KEY if you want to test SerperDevTool.
    # 4. Run this script from the project root: `python -m skyscope_sentinel.autogen_interface`

    # For asyncio in scripts:
    if sys.platform == "win32" and sys.version_info >= (3, 8, 0):
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main_test())
