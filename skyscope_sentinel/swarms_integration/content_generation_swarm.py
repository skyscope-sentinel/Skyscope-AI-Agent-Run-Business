"""
Defines the core agents and orchestration for the Content Generation Swarm.
This swarm autonomously creates textual content based on a given topic or brief.
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
from skyscope_sentinel.tools.search_tools import duckduckgo_search_function # For ContentStrategist
from skyscope_sentinel.config import Config
from skyscope_sentinel.agent_identity import initialize_identity_manager, set_founder_details


# Department name for these agents
CONTENT_DEPARTMENT_NAME = "Content Creation & Marketing"

class ContentStrategistAgent(SkyscopeSwarmAgent):
    """
    Defines the content strategy, outline, keywords, and brief.
    """
    def __init__(self, agent_name: str = "ContentStrategist", tools: list = None, **kwargs):
        system_prompt = (
            "You are a Content Strategist for Skyscope Sentinel Intelligence. "
            "Your input will be a main topic/keyword, target audience, desired content type (e.g., blog post), and desired tone/style. "
            "Your responsibilities are to:"
            "\n1. Refine the topic and identify key sub-topics or angles to cover."
            "\n2. Define the overall structure/outline of the content."
            "\n3. Identify relevant keywords for SEO (you may use your search tool for this)."
            "\n4. Specify the target length (e.g., 'approx. 800 words') and any specific formatting notes for the writer."
            "\nYour output MUST be a detailed content brief as a single string, including sections for: "
            "Outline, Keywords, Target Audience, Tone, Target Length, and any other specific instructions for the DraftWriterAgent."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            tools=tools or [],
            department_name=CONTENT_DEPARTMENT_NAME,
            role_in_department="Lead Content Strategist & SEO Planner",
            **kwargs
        )

class DraftWriterAgent(SkyscopeSwarmAgent):
    """
    Generates the first draft of the content based on the strategist's brief.
    """
    def __init__(self, agent_name: str = "DraftWriter", **kwargs):
        system_prompt = (
            "You are a Draft Writer for Skyscope Sentinel Intelligence. "
            "You will receive a detailed content brief from the ContentStrategistAgent. "
            "Your task is to generate a compelling and informative first draft of the content strictly adhering to this brief. "
            "Focus on factual accuracy (based on the brief, no external research unless explicitly stated in the brief and tools provided), "
            "maintaining the specified tone, style, and structure. Incorporate keywords naturally. "
            "Your output should be the complete raw first draft of the content as a single string."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=CONTENT_DEPARTMENT_NAME,
            role_in_department="Creative Content Drafter",
            **kwargs
        )

class ReviewEditorAgent(SkyscopeSwarmAgent):
    """
    Reviews and edits the draft for clarity, coherence, grammar, and adherence to the brief.
    """
    def __init__(self, agent_name: str = "ReviewEditor", **kwargs):
        system_prompt = (
            "You are a Review Editor for Skyscope Sentinel Intelligence. "
            "You will receive a raw draft from the DraftWriterAgent and the original content brief. "
            "Your responsibilities are to:"
            "\n1. Review the draft for clarity, coherence, grammar, spelling, and style."
            "\n2. Ensure the draft strictly adheres to the original content brief (outline, keywords, tone, length)."
            "\n3. Fact-check critical claims if the brief implies this is necessary and information was provided for it."
            "\n4. Make necessary edits to improve the content. You can rephrase, restructure sentences, or correct errors."
            "\nYour output should be the polished, edited version of the content as a single string. "
            "If major revisions are needed that are beyond simple editing, you should clearly state this and suggest what the writer needs to address."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            department_name=CONTENT_DEPARTMENT_NAME,
            role_in_department="Senior Editor & Quality Assurance",
            **kwargs
        )

def run_content_generation_swarm(
    topic: str,
    target_audience: str = "general public",
    content_type: str = "blog post",
    tone: str = "informative and engaging",
    verbose: bool = True
) -> str:
    """
    Initializes and runs the Content Generation Swarm.

    Args:
        topic (str): The main topic or keyword for content generation.
        target_audience (str, optional): The intended audience. Defaults to "general public".
        content_type (str, optional): Type of content (e.g., "blog post", "tweet thread"). Defaults to "blog post".
        tone (str, optional): Desired tone/style. Defaults to "informative and engaging".
        verbose (bool, optional): Enables verbose output. Defaults to True.

    Returns:
        str: Path to the generated content file or an error message.
    """
    print(f"Initializing Content Generation Swarm for topic: '{topic}'...")
    cfg = Config() # For workspace path

    # Prepare tools
    strategist_tools = [duckduckgo_search_function] # ContentStrategist uses search

    # Initialize agents
    strategist = ContentStrategistAgent(
        tools=strategist_tools,
        max_loops=1, # Strategist performs one main task: creating the brief
        verbose=verbose
    )
    writer = DraftWriterAgent(
        max_loops=1, # Writer drafts once based on brief
        verbose=verbose
    )
    editor = ReviewEditorAgent(
        max_loops=1, # Editor reviews and polishes once
        verbose=verbose
    )

    # Define the workflow
    workflow = SequentialWorkflow(
        agents=[strategist, writer, editor],
        verbose=verbose
    )

    # Construct initial task for the ContentStrategistAgent
    initial_task_for_strategist = (
        f"Create a content brief for a '{content_type}' about '{topic}'. "
        f"The target audience is '{target_audience}', and the desired tone is '{tone}'. "
        f"Include an outline, SEO keywords, target length, and any specific instructions for the writer."
    )

    print("Content Generation Swarm initialized. Starting workflow...")
    final_content_output = workflow.run(initial_task_for_strategist)

    if final_content_output and isinstance(final_content_output, str):
        print("\nContent Generation Workflow completed. Final output (Generated Content):")
        print("----------------------------------------------------")
        print(final_content_output[:1000] + "..." if len(final_content_output) > 1000 else final_content_output)
        print("----------------------------------------------------")

        # Save the generated content
        content_subdir = "generated_content"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize topic for filename
        topic_slug = "".join(c if c.isalnum() else "_" for c in topic.split(" ", 3)[:3]).replace("__", "_") # first three words
        content_filename = f"{content_type.replace(' ', '_')}_{topic_slug}_{timestamp}.md" # Assume markdown for now

        full_content_path = os.path.join(cfg.AGENT_WORKSPACE, content_subdir, content_filename)
        os.makedirs(os.path.dirname(full_content_path), exist_ok=True)

        try:
            with open(full_content_path, "w", encoding="utf-8") as f:
                f.write(final_content_output)
            print(f"Generated content saved to: {full_content_path}")
            return full_content_path
        except Exception as e:
            print(f"Error saving generated content: {e}")
            return f"Workflow completed, but failed to save content. Content snippet: {final_content_output[:200]}..."
    else:
        print("\nContent Generation Workflow completed, but no final content was generated or output was not a string.")
        return "Workflow completed, but no content generated."


if __name__ == "__main__":
    # Basic test for agent instantiation AND the swarm runner
    cfg = Config()
    set_founder_details(cfg.founder_name, cfg.founder_contact, cfg.business_name)
    initialize_identity_manager()

    print("--- Testing Content Generation Agent Instantiation ---")
    try:
        strategist = ContentStrategistAgent(tools=[duckduckgo_search_function], max_loops=1)
        print(f"Initialized: {strategist.agent_name}, ID: {strategist.id}, Role: {strategist.role_in_department}")

        writer = DraftWriterAgent(max_loops=1)
        print(f"Initialized: {writer.agent_name}, ID: {writer.id}, Role: {writer.role_in_department}")

        editor = ReviewEditorAgent(max_loops=1)
        print(f"Initialized: {editor.agent_name}, ID: {editor.id}, Role: {editor.role_in_department}")

        print("\nContent Generation Agents initialized successfully.")
    except Exception as e:
        print(f"Error during Content Generation Agent instantiation test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Content Generation Agent Test Complete ---")
