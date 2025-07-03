"""
Defines the core agents and orchestration for the Freelance Task Simulation Swarm.
This swarm simulates identifying freelance tasks from opportunity reports and drafting proposals.
"""
import os
import datetime
from swarms import SequentialWorkflow

# Ensure project root is in path for sibling imports if running standalone
import sys
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from skyscope_sentinel.swarms_integration.skyscope_swarm_agent import SkyscopeSwarmAgent
from skyscope_sentinel.tools.search_tools import duckduckgo_search_function # For TaskProspectorAgent
from skyscope_sentinel.config import Config
from skyscope_sentinel.agent_identity import initialize_identity_manager, set_founder_details

# Department name for these agents
FREELANCE_DEPARTMENT_NAME = "Freelance Acquisition & Proposal"

class TaskProspectorAgent(SkyscopeSwarmAgent):
    """
    Identifies potential freelance tasks from opportunity reports and formulates mock task descriptions.
    """
    def __init__(self, agent_name: str = "TaskProspector", tools: list = None, **kwargs):
        system_prompt = (
            "You are a Task Prospector for Skyscope Sentinel Intelligence's Freelance Acquisition department. "
            "Your input will be content from an Opportunity Report, focusing on 'Actionable First Steps' and 'Key AI-Leveraged Activities'. "
            "Your responsibilities are to:"
            "\n1. Parse the input report to understand the core opportunity."
            "\n2. Identify 1-2 specific, simple freelance-style tasks that align with this opportunity and Skyscope's AI capabilities (content creation, data analysis, basic coding, research)."
            "\n3. (Optional) If needed for inspiration on how such tasks are typically worded, use your search tool to find examples of similar freelance task postings (e.g., search for 'freelance blog writing job description' or 'data entry project examples')."
            "\n4. Formulate these 1-2 tasks as concrete, mock 'freelance task descriptions' that an AI agent team could bid on. Each description should be clear and concise."
            "\nYour output MUST be a list of these mock freelance task descriptions (each as a string). If you identify two tasks, separate them with '---TASK_SEPARATOR---'. "
            "If no suitable freelance tasks can be derived, output 'No suitable freelance tasks identified from this report.'"
            "\nExample output for one task: 'Create a series of 5 engaging blog posts (approx. 700 words each) on the topic of [Specific Topic from Report], optimized for SEO with keywords [X, Y, Z].'"
            "\nExample output for two tasks: 'Task 1: Data entry and cleaning for a dataset of 1000 records related to [Topic]. ---TASK_SEPARATOR--- Task 2: Generate 3 social media announcement posts for a new [Product/Service from Report].'"
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            tools=tools or [],
            department_name=FREELANCE_DEPARTMENT_NAME,
            role_in_department="Freelance Opportunity Scout",
            **kwargs
        )

class BidStrategistAgent(SkyscopeSwarmAgent):
    """
    Analyzes a mock freelance task and devises a bid strategy.
    """
    def __init__(self, agent_name: str = "BidStrategist", **kwargs):
        system_prompt = (
            "You are a Bid Strategist for Skyscope Sentinel Intelligence. "
            "You will receive a mock freelance task description. Your tasks are to:"
            "\n1. Analyze the task description to determine core requirements and deliverables."
            "\n2. Define a strategy for how Skyscope's AI agents can fulfill this task efficiently and with high quality, leveraging their specific AI skills (content generation, data analysis, research, simple coding)."
            "\n3. Outline 2-3 key selling points for Skyscope, emphasizing AI capabilities (e.g., speed, data-driven insights, consistency, scalability, cost-effectiveness for initial projects)."
            "\n4. Conceptually, suggest a pricing approach (e.g., 'per unit', 'fixed project fee - low initial bid for portfolio building'). Since this is a simulation for a company with no initial funds, focus on strategies that build reputation or lead to quick, small wins."
            "\nYour output MUST be a structured bid strategy as a single string, clearly labeling each section: Requirements, Fulfillment Plan, Selling Points, Conceptual Pricing Strategy."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=FREELANCE_DEPARTMENT_NAME,
            role_in_department="AI Bid & Value Proposition Strategist",
            **kwargs
        )

class ProposalDraftingAgent(SkyscopeSwarmAgent):
    """
    Drafts a proposal for a mock freelance task based on the task description and bid strategy.
    """
    def __init__(self, agent_name: str = "ProposalDrafter", **kwargs):
        system_prompt = (
            "You are a Proposal Drafting Agent for Skyscope Sentinel Intelligence. "
            "You will receive a mock freelance task description and a corresponding bid strategy. "
            "Your task is to draft a compelling, professional, and persuasive proposal or expression of interest tailored to this specific mock task. "
            "The proposal should:"
            "\n1. Clearly state understanding of the task requirements (from the task description)."
            "\n2. Briefly outline Skyscope's proposed AI-driven approach to fulfilling the task (from the bid strategy)."
            "\n3. Highlight Skyscope's key selling points relevant to the task, emphasizing AI capabilities."
            "\n4. Incorporate the conceptual pricing strategy in a suitable manner (e.g., 'We propose an initial low-cost engagement to demonstrate our AI-driven efficiency...')."
            "\n5. Maintain a professional, confident, and client-focused tone."
            "\n6. Be concise and to the point."
            "\nYour output MUST be the complete drafted proposal text as a single string (e.g., in Markdown or plain text)."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=FREELANCE_DEPARTMENT_NAME,
            role_in_department="AI Proposal & Bid Writer",
            **kwargs
        )

def run_freelance_task_simulation_swarm(opportunity_report_content: str, verbose: bool = True) -> list[str]:
    """
    Initializes and runs the Freelance Task Simulation Swarm.

    Args:
        opportunity_report_content (str): Content of an opportunity report to analyze.
        verbose (bool, optional): Enables verbose output. Defaults to True.

    Returns:
        list[str]: A list of file paths to the generated proposal(s) or error messages.
    """
    print(f"Initializing Freelance Task Simulation Swarm...")
    cfg = Config() # For workspace path
    saved_proposal_paths = []

    prospector_tools = [duckduckgo_search_function]
    task_prospector = TaskProspectorAgent(
        tools=prospector_tools,
        max_loops=1, # Expect it to generate task descriptions in one go
        verbose=verbose
    )

    # Task for prospector: analyze the report content
    prospector_task = f"Analyze the following opportunity report content and identify potential freelance tasks:\n\n{opportunity_report_content}"

    print("Running TaskProspectorAgent...")
    mock_task_descriptions_output = task_prospector.run(prospector_task)

    if not mock_task_descriptions_output or "No suitable freelance tasks identified" in mock_task_descriptions_output:
        message = "TaskProspectorAgent did not identify any suitable freelance tasks."
        print(message)
        return [message]

    # Split tasks if multiple are provided, using the defined separator
    task_list = mock_task_descriptions_output.split("---TASK_SEPARATOR---")

    print(f"TaskProspectorAgent identified {len(task_list)} mock task(s).")

    for i, task_desc in enumerate(task_list):
        task_desc = task_desc.strip()
        if not task_desc:
            continue

        print(f"\nProcessing mock task {i+1}: {task_desc[:100]}...")

        bid_strategist = BidStrategistAgent(max_loops=1, verbose=verbose)
        proposal_drafter = ProposalDraftingAgent(max_loops=1, verbose=verbose)

        # Workflow for this single task: Strategist -> Drafter
        proposal_workflow = SequentialWorkflow(
            agents=[bid_strategist, proposal_drafter],
            verbose=verbose
        )

        # The task_desc is input to BidStrategist, its output is input to ProposalDrafter
        print(f"Running BidStrategist and ProposalDrafter for task {i+1}...")
        final_proposal_text = proposal_workflow.run(task_desc)

        if final_proposal_text and isinstance(final_proposal_text, str):
            print(f"Proposal drafted for task {i+1}:\n{final_proposal_text[:200]}...")

            # Save the generated proposal
            proposal_subdir = "simulated_proposals"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Create a slug from the task description for the filename
            task_slug = "".join(c if c.isalnum() else "_" for c in task_desc.split(" ", 4)[:4]).replace("__", "_") # first 4 words
            proposal_filename = f"proposal_{task_slug}_{timestamp}.md" # Save as markdown

            full_proposal_path = os.path.join(cfg.AGENT_WORKSPACE, proposal_subdir, proposal_filename)
            os.makedirs(os.path.dirname(full_proposal_path), exist_ok=True)

            try:
                with open(full_proposal_path, "w", encoding="utf-8") as f:
                    f.write(f"# Mock Freelance Task Description:\n\n{task_desc}\n\n---\n\n# Drafted Proposal:\n\n{final_proposal_text}")
                print(f"Simulated proposal saved to: {full_proposal_path}")
                saved_proposal_paths.append(full_proposal_path)
            except Exception as e:
                error_msg = f"Error saving simulated proposal for task '{task_desc[:50]}...': {e}"
                print(error_msg)
                saved_proposal_paths.append(error_msg)
        else:
            message = f"No proposal generated for mock task: {task_desc[:100]}..."
            print(message)
            saved_proposal_paths.append(message)

    if not saved_proposal_paths:
        return ["No proposals were generated or saved."]
    return saved_proposal_paths


if __name__ == "__main__":
    cfg = Config()
    set_founder_details(cfg.founder_name, cfg.founder_contact, cfg.business_name)
    initialize_identity_manager()

    print("--- Testing Freelance Simulation Agent Instantiation ---")
    try:
        prospector = TaskProspectorAgent(tools=[duckduckgo_search_function], max_loops=1)
        print(f"Initialized: {prospector.agent_name}, ID: {prospector.id}, Role: {prospector.role_in_department}")

        strategist = BidStrategistAgent(max_loops=1)
        print(f"Initialized: {strategist.agent_name}, ID: {strategist.id}, Role: {strategist.role_in_department}")

        drafter = ProposalDraftingAgent(max_loops=1)
        print(f"Initialized: {drafter.agent_name}, ID: {drafter.id}, Role: {drafter.role_in_department}")

        print("\nFreelance Simulation Agents initialized successfully.")
    except Exception as e:
        print(f"Error during Freelance Simulation Agent instantiation test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Freelance Simulation Agent Test Complete ---")
