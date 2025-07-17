"""
Defines the core agents for the Opportunity Scouting Swarm.
These agents will be orchestrated by a Swarm structure (e.g., SequentialWorkflow or Swarm)
to identify and analyze potential income-generating opportunities.
"""
from skyscope_sentinel.swarms_integration.skyscope_swarm_agent import SkyscopeSwarmAgent
from skyscope_sentinel.tools.search_tools import duckduckgo_search_function, serper_search_function # Corrected path
from skyscope_sentinel.tools.browser_tools import browse_web_page_and_extract_text # Corrected path
from skyscope_sentinel.tools.file_io_tools import save_text_to_file_in_workspace # Corrected path
# from skyscope_sentinel.tools.code_execution_tools import execute_python_code_in_e2b # Corrected path for future
from skyscope_sentinel.tools.vector_store_utils import add_report_to_collection, get_contextual_information_for_topic # For RAG
from skyscope_sentinel.tools.vector_store_utils import add_report_to_collection # For RAG
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
    def __init__(self, agent_name: str = "TopicGenerator", tools: list = None, **kwargs):
        system_prompt = (
            "You are an advanced Topic Generator for Skyscope Sentinel Intelligence, an AI-driven enterprise focused on autonomous income generation with minimal initial capital."
            " Your primary role is to brainstorm and propose UNIQUE and ACTIONABLE research topics that could lead to viable, low-barrier income streams."
            " Think outside the box. Consider leveraging AI for niche services, hyper-automation in overlooked areas, or novel applications of existing tech for quick monetization."
            " You should aim for topics that are specific enough for targeted research but broad enough for genuine opportunity discovery."
            "\nKey Focus Areas for Topics:"
            "\n- AI-driven services for niche markets (e.g., 'AI-powered automated accessibility testing for indie game developers')."
            "\n- Hyper-specific affiliate marketing angles (e.g., 'Promoting AI-driven tools for sustainable urban farming')."
            "\n- Micro-SaaS ideas solvable by AI agents (e.g., 'Automated generation of personalized learning paths for corporate upskilling')."
            "\n- Creative content generation strategies using AI (e.g., 'AI-assisted creation of interactive educational content for K-12 STEM subjects')."
            "\n- Data analysis and insight generation for underserved sectors."
            "\n\nINSTRUCTIONS:"
            "\n1. If you are given an existing topic to refine, critically assess its potential for uncovering low-capital, high-impact AI-driven opportunities. If it's good, refine it to be more actionable. If not, discard it and generate a better, more innovative one based on the principles above."
            "\n2. If you need inspiration for a completely new topic, you MAY use the provided 'search_tool' ONCE to look up current discussions on 'novel AI business ideas 2024/2025' or 'untapped AI niches'."
            "\n3. Your final output MUST be a single, clear, and concise research topic string. Do not add any other commentary before or after the topic itself."
            "\nExample Output: 'Developing AI-powered tools for automated translation and localization of indie video games for emerging markets.'"
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            tools=tools or [], # Pass tools to the base class
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
    Analyzes gathered information to identify viable opportunities, potentially using context from past reports.
    """
    def __init__(self, agent_name: str = "AnalysisAgent", tools: list = None, **kwargs):
        system_prompt = (
            "You are a highly critical and pragmatic Analysis Agent for Skyscope Sentinel Intelligence, an AI-driven enterprise focused on generating significant income **starting with zero initial monetary investment**."
            " Your primary task is to analyze research data on a given topic and identify 1-3 concrete, actionable, and low-barrier income-generating opportunities."
            "\n\nYOUR PROCESS:"
            "\n1.  **Receive Research Data:** This will be the primary input for your analysis, typically from a ResearchAgent."
            "\n2.  **Retrieve Context (Tool Use):** Use the `get_contextual_information_for_topic` tool with the core research topic to fetch relevant insights from past reports. This context might highlight synergies, past successes/failures, or related ideas."
            "\n3.  **Synthesize & Analyze:** Combine the fresh research data with any retrieved contextual information."
            "\n4.  **Identify Opportunities & Detail (8-Point Structure):** For each opportunity (1-3 max), provide the following:"
            "\n    1.  **Opportunity Title:** Clear and concise."
            "\n    2.  **Core Concept:** Brief explanation."
            "\n    3.  **Zero-Cost Startup Strategy:** How to start with NO financial outlay, leveraging AI and agent work-hours."
            "\n    4.  **Key AI-Leveraged Activities:** Specific tasks for Skyscope's AI agents."
            "\n    5.  **Potential Revenue Streams (Short-Term Focus):** 2-3 quick monetization ideas."
            "\n    6.  **Target Audience/Market:** Initial customers/users."
            "\n    7.  **Actionable First Steps (3 concrete actions):** Specific initial actions for AI agents."
            "\n    8.  **Potential Challenges & Mitigation with AI:** Main hurdles and AI-driven solutions."
            "\n\nCRITICAL CONSIDERATIONS:"
            "\n- **No Funds Constraint:** This is paramount. No upfront payment for tools (beyond free tiers/open source), ads, or inventory, unless the path to funding is part of the very first steps."
            "\n- **AI Skillset:** Assume AI agents for research, content creation, simple coding, data analysis, and automated outreach."
            "\n- **Realistic & Actionable:** Focus on what can be started NOW by AI agents."
            "\n- **Contextual Awareness:** Clearly state if and how the retrieved contextual information (from past reports) influenced your analysis or the shaping of the opportunities."
            "\n\nYour output should be a well-structured analysis, detailing each point for every identified opportunity, ready for the ReportingAgent. If no opportunities are found, clearly state that and explain why based on the data and context."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            tools=tools or [],
    Analyzes gathered information to identify viable opportunities.
    """
    def __init__(self, agent_name: str = "AnalysisAgent", **kwargs):
        system_prompt = (
            "You are a highly critical and pragmatic Analysis Agent for Skyscope Sentinel Intelligence, an AI-driven enterprise focused on generating significant income **starting with zero initial monetary investment**."
            " You will receive comprehensive research data on a specific topic. Your core task is to dissect this information to identify 1-3 concrete, actionable, and **immediately pursuable (low-barrier)** income-generating opportunities."
            "\n\nFOR EACH IDENTIFIED OPPORTUNITY, YOU MUST PROVIDE:"
            "\n1.  **Opportunity Title:** A clear, concise title."
            "\n2.  **Core Concept:** A brief explanation of the opportunity."
            "\n3.  **Zero-Cost Startup Strategy:** How can Skyscope (an AI entity with AI staff) begin pursuing this with NO financial outlay? Focus on leveraging AI capabilities, open-source tools, and sweat equity (agent work-hours)."
            "\n4.  **Key AI-Leveraged Activities:** What specific tasks can Skyscope's AI agents (e.g., content writers, code generators, researchers, social media bots) perform to build and monetize this opportunity?"
            "\n5.  **Potential Revenue Streams (Short-Term Focus):** Identify 2-3 ways this could start generating revenue quickly (e.g., freelance service, affiliate commission, micro-product)."
            "\n6.  **Target Audience/Market:** Who are the initial customers or users?"
            "\n7.  **Actionable First Steps (3 concrete actions):** What are the absolute first three steps Skyscope's AI agents should take to begin executing this opportunity? Be specific (e.g., 'Draft 5 variations of a service proposal for X using AI content writers', 'Identify 10 relevant subreddits for initial outreach', 'Use AI to generate a list of 20 potential affiliate products in Y niche')."
            "\n8.  **Potential Challenges & Mitigation with AI:** What are the main hurdles, and how can AI help overcome them?"
            "\n\nCRITICAL CONSIDERATIONS:"
            "\n- **No Funds Constraint:** This is paramount. Solutions requiring upfront payment for tools (beyond existing free tiers or open source), advertising, or inventory are NOT acceptable unless a clear path to acquiring those funds *through the opportunity itself* is outlined as a very first step."
            "\n- **AI Skillset:** Assume Skyscope has AI agents capable of research, content creation (text, basic images), simple code generation/scripting, data analysis, and automated outreach (e.g., social media posting, email drafting)."
            "\n- **Realistic & Actionable:** Avoid purely theoretical or overly complex ideas. Focus on what can be started *now* by AI agents."
            "\n\nYour output should be a well-structured analysis, ready to be formatted into a Markdown report by the ReportingAgent. Address each of the 8 points above for every opportunity you identify."
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
            "You are a meticulous Reporting Agent for Skyscope Sentinel Intelligence."
            " Your input will be a structured analysis of one or more potential income-generating opportunities, typically provided by an AnalysisAgent."
            " Your sole responsibility is to format this analysis into a professional, well-organized, and easily readable Markdown report."
            "\n\nMARKDOWN REPORT STRUCTURE REQUIREMENTS:"
            "\n1.  **Main Title:** Start with '# Skyscope Sentinel - Opportunity Analysis Report'."
            "\n2.  **Research Topic:** Add a line: '**Research Focus:** [Original research topic provided to the AnalysisAgent - this might be part of the input or you may need to infer it if not explicitly passed in the analysis data. If so, state 'General Analysis'.]'."
            "\n3.  **Date of Report:** Add a line: '**Report Date:** [Current Date - YYYY-MM-DD]' (You'll need to generate this)."
            "\n4.  **Overall Summary (Optional but Recommended):** If the analysis provides an overall summary, include it here under '## Executive Summary'."
            "\n5.  **Opportunity Details (For EACH opportunity identified in the analysis):**"
            "\n    *   Use a clear heading: '## Opportunity: [Opportunity Title from Analysis]'."
            "\n    *   Present each of the 8 analysis points (Core Concept, Zero-Cost Startup Strategy, etc.) as a sub-section with a bolded label followed by the information. For example:"
            "\n        *   '**Core Concept:** Details...'"
            "\n        *   '**Zero-Cost Startup Strategy:** Details...'"
            "\n        *   '**Key AI-Leveraged Activities:** (Use bullet points for lists if appropriate)'"
            "\n        *   '**Potential Revenue Streams (Short-Term Focus):** (Use bullet points)'"
            "\n        *   '**Target Audience/Market:** Details...'"
            "\n        *   '**Actionable First Steps (for AI Agents):** (Use numbered list for steps)'"
            "\n        *   '**Potential Challenges & Mitigation with AI:** Details...'"
            "\n6.  **Overall Conclusion/Recommendation (Optional):** If the analysis provides a concluding thought or ranks opportunities, include it under '## Overall Conclusion'."
            "\n\nIMPORTANT FORMATTING NOTES:"
            "\n- Use Markdown effectively: Headings (`#`, `##`, `###`), bold (`**text**`), italics (`*text*`), lists (`- item`, `1. item`)."
            "\n- Ensure clarity and readability. Use paragraphs and line breaks appropriately."
            "\n- Do NOT add any commentary, opinions, or information not present in the input analysis from the AnalysisAgent, other than generating the current date and structuring the report as specified."
            "\n- Your entire output should be a single string containing the complete Markdown document."
            "\n\nExample of an opportunity section:"
            "\n## Opportunity: AI-Powered Content Moderation for Niche Online Communities"
            "\n**Core Concept:** Provide automated content moderation services using AI to small, specialized online forums that cannot afford large human moderation teams."
            "\n**Zero-Cost Startup Strategy:** Develop a prototype using open-source NLP models and offer free trials to 2-3 communities to gather testimonials."
            "\n..."
        )
        # This agent does not directly use tools; it formats text.
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
import functools # For functools.partial

load_dotenv()

def run_opportunity_scouting_swarm(initial_topic: str = None, verbose: bool = True, max_search_results_override: int = 5):
load_dotenv()

def run_opportunity_scouting_swarm(initial_topic: str = None, verbose: bool = True):
    """
    Initializes and runs the Opportunity Scouting Swarm.

    Args:
        initial_topic (str, optional): The initial topic for the TopicGeneratorAgent.
                                       If None, TopicGenerator will try to generate one.
        verbose (bool, optional): Enables verbose output from agents. Defaults to True.
        max_search_results_override (int, optional): Max search results for DuckDuckGo. Defaults to 5.

    Returns:
        str: Path to the generated report or a summary message.
    """
    print(f"Initializing Opportunity Scouting Swarm (Max Search Results for DDG: {max_search_results_override})...")
    print("Initializing Opportunity Scouting Swarm...")

    # Prepare tools
    # TopicGenerator might use a search tool.
    # ResearchAgent uses search and browse.
    # ReportingAgent doesn't directly use a tool in this setup (markdown saved by runner).

    # Tools for TopicGenerator (optional search)
    topic_gen_tools = []
    # For now, let's use duckduckgo for TopicGenerator if it needs search, as it's free.
    # We could make this configurable or pass a specific search tool.
    # The prompt instructs it to use 'search_tool', so the function name matters if using function calling.
    # For now, we assume the LLM will understand 'search_tool' if one is provided.
    # Let's pass duckduckgo_search_function as a general search tool.
    # To make it explicit for the LLM if it supports named tools:
    # topic_gen_tools = [{"tool": duckduckgo_search_function, "name": "search_tool"}] # If swarms supports this
    # For now, just passing the callable.
    topic_gen_tools.append(duckduckgo_search_function)


    # Tools for ResearchAgent
    # Apply max_search_results_override to DuckDuckGo
    custom_ddg_search = functools.partial(duckduckgo_search_function, max_results=max_search_results_override)

    research_agent_tools = [custom_ddg_search, browse_web_page_and_extract_text]

    # SerperDevTool used by serper_search_function typically has a fixed number of results or
    # one set at tool instantiation, not easily changed per call by the agent.
    # So, max_search_results_override primarily affects DuckDuckGo here.
    if os.getenv("SERPER_API_KEY"):
        research_agent_tools.insert(0, serper_search_function) # Add Serper (will use its default/tool-set result count)
        print(f"Serper API key found. Adding Serper search tool (uses its own result count settings). DDG set to {max_search_results_override} results.")
    else:
        print(f"Serper API key not found. ResearchAgent using DuckDuckGo (set to {max_search_results_override} results) and browser tool only.")
    research_agent_tools = [duckduckgo_search_function, browse_web_page_and_extract_text]
    if os.getenv("SERPER_API_KEY"):
        research_agent_tools.insert(0, serper_search_function) # Prioritize Serper if available
        print("Serper API key found. Adding Serper search tool for ResearchAgent.")
    else:
        print("Serper API key not found. ResearchAgent using DuckDuckGo search tool only.")


    # Initialize agents
    topic_generator = TopicGeneratorAgent(
        tools=topic_gen_tools,
        tools=topic_gen_tools, # Provide search tool
        max_loops=1,
        verbose=verbose
    )

    research_agent = ResearchAgent(
        tools=research_agent_tools, # Pass the potentially customized DDG tool
        tools=research_agent_tools,
        max_loops=1,
        verbose=verbose
    )

    # Tools for AnalysisAgent (RAG tool)
    analysis_agent_tools = [get_contextual_information_for_topic]

    analysis_agent = AnalysisAgent(
        tools=analysis_agent_tools,
        # tools=research_tools, # Removed redundant/incorrect tools parameter
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
    # The output of ResearchAgent will be the input for AnalysisAgent.
    # The output of AnalysisAgent will be the input for ReportingAgent.

    if initial_topic:
        current_task_for_topic_generator = f"Refine and confirm this topic for research: '{initial_topic}'. If suitable, output the topic. If not, generate a better one based on your advanced criteria."
    else:
        current_task_for_topic_generator = "Generate one promising and unique topic for identifying new AI-driven income-generating opportunities with low initial capital, using your search tool if needed for inspiration."

    # To effectively log input to AnalysisAgent, we need to capture intermediate outputs or structure the workflow differently.
    # For SequentialWorkflow, the input to AnalysisAgent is the direct output of ResearchAgent.
    # We can't easily intercept and log it without a custom workflow or modifying SequentialWorkflow.
    # However, the `verbose=True` on the workflow and agents should provide substantial logging from the swarms library itself.
    # The AnalysisAgent's prompt instructs it to use the RAG tool with the "core research topic".
    # The "core research topic" comes from TopicGeneratorAgent's output, which becomes ResearchAgent's task.
    # ResearchAgent's output (research data) becomes AnalysisAgent's task. AnalysisAgent needs to infer the topic from this.

    print(f"\n[OPP_SWARM_RUNNER] Starting sequential workflow with initial task for TopicGenerator: '{current_task_for_topic_generator}'")
    # The AnalysisAgent will receive the output of ResearchAgent.
    # The prompt for AnalysisAgent is designed to make it use its RAG tool with the topic it infers from its input.

    final_report_markdown = workflow.run(current_task_for_topic_generator)

    print(f"\n[OPP_SWARM_RUNNER] Raw output from final agent (ReportingAgent) in workflow: \n{final_report_markdown[:300]}...\n")


    if final_report_markdown and isinstance(final_report_markdown, str):
        print("\n[OPP_SWARM_RUNNER] Workflow completed. Final output (Report Markdown):")

    if initial_topic:
        current_task = f"Refine and confirm this topic for research: '{initial_topic}'. If suitable, output the topic. If not, generate a better one based on your advanced criteria." # Updated task for TopicGenerator
    else:
        current_task = "Generate one promising and unique topic for identifying new AI-driven income-generating opportunities with low initial capital, using your search tool if needed for inspiration." # Updated task
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
        # Determine topic for metadata: use initial_topic if provided, else try to get it from TopicGenerator output
        # This is tricky as TopicGenerator's output is directly fed to ResearchAgent.
        # For now, if initial_topic is None, the report metadata topic will be "Auto-Generated"
        # A more advanced setup might involve a custom workflow to extract the generated topic.
        report_topic_for_rag = initial_topic if initial_topic else "Auto-Generated by TopicGenerator"


        # Ensure filename uses the potentially refined/generated topic if possible for uniqueness.
        # However, the `topic_slug` for filename was based on `initial_topic` only.
        # This is acceptable for now; RAG metadata will store the more accurate `report_topic_for_rag`.

        full_report_path = os.path.join(cfg.AGENT_WORKSPACE, report_subdir, report_filename)

        # Ensure the subdirectory exists
        os.makedirs(os.path.dirname(full_report_path), exist_ok=True)

        try:
            with open(full_report_path, "w", encoding="utf-8") as f:
                f.write(final_report_markdown)
            print(f"Report saved to: {full_report_path}")

            # Add report to RAG store
            try:
                print(f"Adding report '{report_filename}' to RAG vector store...")
                add_report_to_collection(
                    report_markdown=final_report_markdown,
                    report_filename=report_filename,
                    topic=report_topic_for_rag
                )
            except Exception as rag_e:
                print(f"Error adding report to RAG store: {rag_e}")

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
