"""
Defines the core agents for the Opportunity Scouting Swarm.
These agents will be orchestrated by a Swarm structure (e.g., SequentialWorkflow or Swarm)
to identify and analyze potential income-generating opportunities.
"""
from skyscope_sentinel.swarms_integration.skyscope_swarm_agent import SkyscopeSwarmAgent
from skyscope_sentinel.utils.search_tools import duckduckgo_search_function, serper_search_function
from skyscope_sentinel.utils.browser_tools import browse_web_page_and_extract_text
from skyscope_sentinel.utils.file_io_tools import save_text_to_file_in_workspace
# from skyscope_sentinel.utils.code_execution_tools import execute_python_code_in_e2b # For later if needed by AnalysisAgent

# Placeholder for department, will be refined
DEPARTMENT_NAME = "Strategic Opportunity Identification"

class TopicGeneratorAgent(SkyscopeSwarmAgent):
    """
    Generates or fetches research topics for opportunity scouting.
    """
    def __init__(self, agent_name: str = "TopicGenerator", **kwargs):
        system_prompt = (
            "You are a Topic Generator for Skyscope Sentinel Intelligence, an AI-driven enterprise."
            " Your role is to identify and propose broad topics or specific areas that hold potential for"
            " new income-generating opportunities. These topics will be researched by other agents."
            " Focus on emerging technologies, market trends, underserved niches, and innovative business models."
            " Examples: 'AI in personalized education', 'Decentralized finance (DeFi) for micro-loans',"
            " 'Affiliate marketing for sustainable products', 'Remote work productivity tools'."
            " Provide one clear, concise topic per request."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=DEPARTMENT_NAME,
            role_in_department="Initial Opportunity Ideator",
            **kwargs
        )

class ResearchAgent(SkyscopeSwarmAgent):
    """
    Gathers information on a given topic using search and browsing tools.
    """
    def __init__(self, agent_name: str = "ResearchAgent", **kwargs):
        system_prompt = (
            "You are a Research Agent for Skyscope Sentinel Intelligence."
            " Your primary function is to conduct thorough research on a given topic."
            " Utilize the provided search and browsing tools to gather comprehensive information."
            " Focus on finding reliable sources, key players, market size, potential challenges, and existing solutions."
            " Extract relevant text and data. Your output will be used by an AnalysisAgent."
        )
        # Tools will be passed during swarm initialization
        # Expected tools: duckduckgo_search_function, serper_search_function, browse_web_page_and_extract_text
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=DEPARTMENT_NAME,
            role_in_department="Information Gatherer & Web Navigator",
            **kwargs
        )

class AnalysisAgent(SkyscopeSwarmAgent):
    """
    Analyzes gathered information to identify viable opportunities.
    """
    def __init__(self, agent_name: str = "AnalysisAgent", **kwargs):
        system_prompt = (
            "You are an Analysis Agent for Skyscope Sentinel Intelligence."
            " You receive research data on a specific topic and your task is to analyze it critically."
            " Identify potential income-generating opportunities within the researched area."
            " For each opportunity, assess its viability, potential revenue streams, target audience,"
            " required skills/resources, initial investment (aim for low/no cost), and potential risks."
            " Synthesize the information into a structured analysis. Be realistic and data-driven."
            " The goal is to find actionable, high-potential opportunities for an AI-driven enterprise starting with no funds."
        )
        # Potential tools for later: execute_python_code_in_e2b for complex data tasks
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=DEPARTMENT_NAME,
            role_in_department="Opportunity Analyst & Viability Assessor",
            **kwargs
        )

class ReportingAgent(SkyscopeSwarmAgent):
    """
    Consolidates analysis into a structured report.
    """
    def __init__(self, agent_name: str = "ReportingAgent", **kwargs):
        system_prompt = (
            "You are a Reporting Agent for Skyscope Sentinel Intelligence."
            " Your role is to take the structured analysis of potential opportunities and compile it into a"
            " clear, concise, and actionable report. The report should be well-organized, detailing each"
            " identified opportunity, its analysis (viability, revenue potential, risks, resources), and a"
            " summary recommendation. Use Markdown format for the report."
            " The report will be saved to the workspace using the provided file I/O tool."
        )
        # Expected tools: save_text_to_file_in_workspace
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=DEPARTMENT_NAME,
            role_in_department="Lead Report Compiler & Documenter",
            **kwargs
        )

if __name__ == "__main__":
    # Example of how agents might be initialized (actual orchestration will be in a swarm runner)
    # This is for basic testing of agent instantiation and identity generation.
    from skyscope_sentinel.config import global_config
    from skyscope_sentinel.agent_identity import initialize_identity_manager, set_founder_details

    # Initialize identity manager (if not already done elsewhere for testing)
    set_founder_details(global_config.founder_name, global_config.founder_contact, global_config.business_name)
    initialize_identity_manager()

    print("Attempting to initialize Opportunity Scouting Agents (test mode)...")

    try:
        topic_gen = TopicGeneratorAgent(max_loops=1)
        print(f"Initialized: {topic_gen.agent_name}, ID: {topic_gen.id}, Role: {topic_gen.role_in_department}")
        # print(f"System Prompt: {topic_gen.system_prompt}") # Too verbose for console
        print("-" * 20)

        researcher = ResearchAgent(max_loops=1) # Tools would be passed in real scenario
        print(f"Initialized: {researcher.agent_name}, ID: {researcher.id}, Role: {researcher.role_in_department}")
        print("-" * 20)

        analyst = AnalysisAgent(max_loops=1)
        print(f"Initialized: {analyst.agent_name}, ID: {analyst.id}, Role: {analyst.role_in_department}")
        print("-" * 20)

        reporter = ReportingAgent(max_loops=1) # Tools would be passed in real scenario
        print(f"Initialized: {reporter.agent_name}, ID: {reporter.id}, Role: {reporter.role_in_department}")
        print("-" * 20)

        print("All agents initialized successfully in test mode.")

    except Exception as e:
        print(f"Error during agent initialization test: {e}")
        import traceback
        traceback.print_exc()

    # To run a simple test of the TopicGeneratorAgent:
    # task_topic_gen = "Generate one topic for freelance opportunities in AI."
    # try:
    #     print(f"\nTesting TopicGeneratorAgent with task: '{task_topic_gen}'")
    #     # Note: SkyscopeSwarmAgent currently doesn't auto-run with a task on init.
    #     # This would require calling agent.run(task) which needs the full swarm context or direct LLM call setup.
    #     # For now, this __main__ block primarily tests instantiation.
    #     # To truly test .run(), it needs to be part of the swarm orchestration or a dedicated test script.
    #     topic_output = topic_gen.run(task_topic_gen) # This will try to use the llm
    #     print(f"TopicGeneratorAgent Output: {topic_output}")
    # except Exception as e:
    #     print(f"Error running TopicGeneratorAgent: {e}")
    #     print("Ensure Ollama is running and the model (e.g., mistral) is available.")

# To make tools available to agents, they are typically passed during the Agent's initialization
# within the swarm orchestration logic, or the agent's `run` method needs to be adapted
# to receive them or have them bound. `swarms.Agent` can take a `tools` list.
# Our `SkyscopeSwarmAgent` would need to be updated to accept `tools` and pass them to `super().__init__`
# or the swarm orchestration will handle this.
# For now, the system prompts indicate tool usage, and the orchestration step will manage tool passing.

# The import of tool functions at the top is for type hinting and conceptual clarity.
# They will be instantiated and passed into agents in the swarm runner script.
# For example, ResearchAgent might be initialized like:
# research_agent = ResearchAgent(tools=[duckduckgo_search_function, ...])
# This requires SkyscopeSwarmAgent to handle the 'tools' kwarg and pass it to swarms.Agent.
# Let's update SkyscopeSwarmAgent to accept tools.
# (This will be done in a separate step if required, for now, focusing on agent definitions)

# Added department name and roles for better identity integration.
# The __main__ block is for very basic instantiation testing.
# Actual tool usage and inter-agent communication will be tested in the swarm orchestration step.
# System prompts are designed to be clear for the LLM.
# Max_loops for these agents will likely be 1 for their specific task within a sequential flow,
# but can be adjusted by the orchestrator.
# Kwargs are included to allow passing additional parameters like `llm`, `max_loops`, `tools` etc.
# from the orchestrator to the underlying swarms.Agent.

import os
import datetime
from swarms import SequentialWorkflow
from skyscope_sentinel.config import Config # For workspace path
from skyscope_sentinel.agent_identity import initialize_identity_manager, set_founder_details # For __main__ test

# Load environment variables (e.g., for SERPER_API_KEY)
from dotenv import load_dotenv
load_dotenv()

def run_opportunity_scouting_swarm(initial_topic: str = None, verbose: bool = True):
    """
    Initializes and runs the Opportunity Scouting Swarm.

    Args:
        initial_topic (str, optional): The initial topic for the TopicGeneratorAgent.
                                       If None, TopicGenerator will try to generate one.
        verbose (bool, optional): Enables verbose output from agents. Defaults to True.

    Returns:
        str: Path to the generated report or a summary message.
    """
    print("Initializing Opportunity Scouting Swarm...")

    # Prepare tools
    research_tools = [duckduckgo_search_function, browse_web_page_and_extract_text]
    if os.getenv("SERPER_API_KEY"):
        research_tools.insert(0, serper_search_function) # Prioritize Serper if available
        print("Serper API key found. Adding Serper search tool.")
    else:
        print("Serper API key not found. Using DuckDuckGo search tool only.")

    reporting_tools = [save_text_to_file_in_workspace]

    # Initialize agents
    # Note: SkyscopeSwarmAgent uses its own agent_id generation if not provided.
    # We are letting it auto-generate based on department.
    # max_loops=1 is suitable for a sequential workflow where each agent performs one step.

    topic_generator = TopicGeneratorAgent(
        max_loops=1,
        verbose=verbose
    )

    research_agent = ResearchAgent(
        tools=research_tools,
        max_loops=1, # Max loops for the agent's own internal process, not for retries in sequence.
        verbose=verbose
    )

    analysis_agent = AnalysisAgent(
        max_loops=1,
        verbose=verbose
    )

    # Define a dynamic report filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize initial_topic for filename if it exists, or use a generic name
    topic_slug = "".join(c if c.isalnum() else "_" for c in initial_topic.split(" ", 2)[0:2]) if initial_topic else "general_topic" # first two words
    report_filename = f"opportunity_report_{topic_slug}_{timestamp}.md"

    # The ReportingAgent's save_text_to_file_in_workspace tool needs the filename as an argument.
    # We can pass it via the task to the ReportingAgent, or modify the tool/agent.
    # For SequentialWorkflow, the output of AnalysisAgent becomes the input task for ReportingAgent.
    # The ReportingAgent's system prompt tells it to use the tool to save.
    # The tool itself needs a filename. Let's make the ReportingAgent's task to be "Create a report based on this analysis: {analysis_output}. Save it as {report_filename}"
    # This means the save_text_to_file_in_workspace tool needs to parse its input for filename and content.
    # OR, we can have the ReportingAgent output the markdown, and this function saves it.
    # Let's try the latter first for simplicity: ReportingAgent generates markdown, this function saves it.

    reporting_agent = ReportingAgent(
        # tools=reporting_tools, # Reporting agent will output markdown, this function will save it.
        max_loops=1,
        verbose=verbose
    )

    # Define the workflow
    agents_list = [topic_generator, research_agent, analysis_agent, reporting_agent]
    workflow = SequentialWorkflow(
        agents=agents_list,
        verbose=verbose
    )

    print("Opportunity Scouting Swarm initialized. Starting workflow...")

    # Determine the initial task for the workflow
    # If initial_topic is provided, TopicGenerator's first task is to refine/use it.
    # If not, its task is to generate a new one.
    # The output of TopicGenerator will be the input for ResearchAgent.

    if initial_topic:
        current_task = f"Refine and confirm this topic for research: '{initial_topic}'. If suitable, output the topic. If not, generate a better one."
    else:
        current_task = "Generate one promising topic for identifying new income-generating opportunities."

    final_report_markdown = workflow.run(current_task)

    if final_report_markdown and isinstance(final_report_markdown, str):
        print("\nWorkflow completed. Final output (Report Markdown):")
        print("----------------------------------------------------")
        print(final_report_markdown[:1000] + "..." if len(final_report_markdown) > 1000 else final_report_markdown)
        print("----------------------------------------------------")

        # Save the report
        cfg = Config()
        report_subdir = "opportunity_reports"
        full_report_path = os.path.join(cfg.AGENT_WORKSPACE, report_subdir, report_filename)

        # Ensure the subdirectory exists
        os.makedirs(os.path.dirname(full_report_path), exist_ok=True)

        try:
            with open(full_report_path, "w", encoding="utf-8") as f:
                f.write(final_report_markdown)
            print(f"Report saved to: {full_report_path}")
            return full_report_path
        except Exception as e:
            print(f"Error saving report: {e}")
            return f"Workflow completed, but failed to save report. Report content: {final_report_markdown[:200]}..."
    else:
        print("\nWorkflow completed, but no final report content was generated or output was not a string.")
        return "Workflow completed, but no report content generated."


if __name__ == "__main__":
    # Setup for testing run_opportunity_scouting_swarm
    temp_config_for_test = Config()
    set_founder_details(
        temp_config_for_test.founder_name,
        temp_config_for_test.founder_contact,
        temp_config_for_test.business_name
    )
    initialize_identity_manager() # Ensures identities can be generated for agents

    print("\n--- Testing run_opportunity_scouting_swarm ---")
    print("This test will run the full swarm. Ensure Ollama is running and accessible.")
    print(f"Using Ollama model: {temp_config_for_test.get_ollama_model_name()} at {temp_config_for_test.get_ollama_base_url()}")

    # Example: Run with a specific topic
    test_topic = "AI-powered tools for small business marketing automation"
    print(f"\nStarting swarm with topic: '{test_topic}'")
    report_path_or_message = run_opportunity_scouting_swarm(initial_topic=test_topic, verbose=True)
    print(f"\nSwarm run result: {report_path_or_message}")

    # Example: Run without a specific topic (TopicGeneratorAgent will create one)
    # print("\nStarting swarm without a predefined topic (TopicGenerator will create one)...")
    # report_path_or_message_auto_topic = run_opportunity_scouting_swarm(verbose=True)
    # print(f"\nSwarm run result (auto-topic): {report_path_or_message_auto_topic}")

    print("\n--- Opportunity Scouting Swarm Test Complete ---")


print("Skyscope Sentinel Intelligence - Opportunity Scouting Agents & Swarm Orchestration Defined.")
print("File: skyscope_sentinel/swarms_integration/opportunity_scouting_swarm.py")
