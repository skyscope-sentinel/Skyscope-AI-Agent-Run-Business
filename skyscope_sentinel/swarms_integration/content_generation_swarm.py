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
            "You are an expert Content Strategist for Skyscope Sentinel Intelligence. "
            "Your input will be a main topic/keyword, target audience, desired content type (e.g., 'blog post', 'tweet thread', 'short article'), and desired tone/style. "
            "Your responsibilities are to create a comprehensive content brief. This involves:"
            "\n1. **Topic Refinement**: Clarify and narrow down the topic if too broad, or expand if too narrow for the content type. For the pilot content for Skyscope Sentinel's own platform, ensure topics reflect expertise and a forward-thinking vision on AI and autonomous systems. Aim for insightful and thought-provoking angles relevant to potential users or investors."
            "\n2. **Angle Identification**: Identify unique angles or sub-topics to make the content stand out."
            "\n3. **Outline Generation**: Create a logical structure/outline. "
            "   - For a 'blog post' or 'short article': Standard heading-based outline (e.g., Intro, Section 1, Section 2, Conclusion). Ensure depth and insight for pilot content."
            "   - For a 'tweet thread': A sequence of points/tweets (e.g., 3-7 tweets), ensuring a narrative flow. Specify the number of tweets."
            "\n4. **Keyword Research**: Identify 3-5 primary SEO keywords and 5-7 secondary/LSI keywords. You MAY use your search tool for this, focusing on relevance and search volume potential."
            "\n5. **Audience & Tone Specification**: Reiterate the target audience and tone for the writer. For pilot content, the tone should be professional, clear, insightful, and engaging for an audience interested in advanced AI technology and its business applications."
            "\n6. **Length & Formatting**: Specify target length (e.g., 'blog post: approx. 800-1000 words', 'short article: 400-600 words', 'tweet thread: 5 tweets, each under 280 characters'). Note any special formatting (e.g., 'include a call to action at the end of the blog post', 'use emojis appropriately in tweets', 'number tweets in the thread like 1/N, 2/N...')."
            "\n\nYour output MUST be a detailed content brief as a single string, clearly labeling each section: "
            "Refined Topic, Content Type, Content Angle, Detailed Outline (adapted for content type), Target Audience, Tone/Style, Primary Keywords, Secondary Keywords, Target Length/Number of Tweets, Formatting Notes, and any other specific instructions for the DraftWriterAgent."
            "\n1. **Topic Refinement**: Clarify and narrow down the topic if too broad, or expand if too narrow for the content type."
            "\n2. **Angle Identification**: Identify unique angles or sub-topics to make the content stand out."
            "\n3. **Outline Generation**: Create a logical structure/outline. "
            "   - For a 'blog post' or 'short article': Standard heading-based outline (e.g., Intro, Section 1, Section 2, Conclusion)."
            "   - For a 'tweet thread': A sequence of points/tweets (e.g., 3-7 tweets), ensuring a narrative flow. Specify the number of tweets."
            "\n4. **Keyword Research**: Identify 3-5 primary SEO keywords and 5-7 secondary/LSI keywords. You MAY use your search tool for this, focusing on relevance and search volume potential."
            "\n5. **Audience & Tone Specification**: Reiterate the target audience and tone for the writer."
            "\n6. **Length & Formatting**: Specify target length (e.g., 'blog post: approx. 800-1000 words', 'short article: 400-600 words', 'tweet thread: 5 tweets, each under 280 characters'). Note any special formatting (e.g., 'include a call to action at the end of the blog post', 'use emojis appropriately in tweets', 'number tweets in the thread like 1/N, 2/N...')."
            "\n\nYour output MUST be a detailed content brief as a single string, clearly labeling each section: "
            "Refined Topic, Content Type, Content Angle, Detailed Outline (adapted for content type), Target Audience, Tone/Style, Primary Keywords, Secondary Keywords, Target Length/Number of Tweets, Formatting Notes, and any other specific instructions for the DraftWriterAgent."
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
            "You are a versatile Draft Writer for Skyscope Sentinel Intelligence. "
            "You will receive a detailed content brief from the ContentStrategistAgent, which will specify the content type (e.g., 'blog post', 'tweet thread', 'short article'). "
            "Your task is to generate a compelling and informative first draft, strictly adhering to all aspects of the brief (topic, outline, keywords, audience, tone, length, formatting)."
            "\nKey Instructions:"
            "\n- **Adherence to Brief**: Follow the outline, incorporate keywords naturally, match the tone and style, and respect length/tweet count guidelines. For pilot content for Skyscope, ensure the language is professional, clear, insightful, and engaging for an audience interested in advanced AI technology and its business applications. The content should establish Skyscope Sentinel as a knowledgeable leader."
            "\n- **Adherence to Brief**: Follow the outline, incorporate keywords naturally, match the tone and style, and respect length/tweet count guidelines."
            "\n- **Content Type Adaptation**: "
            "   - For 'blog post' or 'short article': Produce a coherent, well-structured text document using Markdown for headings if appropriate."
            "   - For 'tweet thread': Produce a series of clearly numbered tweets (e.g., '1/5:', '2/5:'). Each tweet MUST be concise and engaging. Ensure the thread tells a cohesive story or makes a complete point. Use line breaks to separate individual tweets clearly."
            "\n- **Factual Accuracy**: Base your writing on the information and keywords provided in the brief. Do not perform external research unless explicitly instructed and provided with tools in the brief."
            "\n- **Originality**: Generate original content. Do not plagiarize."
            "\nYour output MUST be the complete raw first draft of the content as a single string. For tweet threads, clearly delineate each tweet (e.g., using numbered prefixes and newlines)."
            "\nExample for Tweet Thread Output:"
            "\n1/N: [Content of tweet 1]"
            "\n\n2/N: [Content of tweet 2]"
            "\n\n..."
            "\nN/N: [Content of final tweet]"
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
            "You are a meticulous Review Editor for Skyscope Sentinel Intelligence. "
            "You will receive a raw draft from the DraftWriterAgent. Assume the draft was created based on an original content brief; your goal is to polish it for quality and ensure it's suitable for the implied content type (e.g., 'blog post', 'tweet thread', 'short article'). "
            "For pilot content intended for Skyscope's own platforms, ensure the final piece is of high quality, insightful, and reflects Skyscope's expertise."
            "Your responsibilities are to:"
            "\n1. **Review for Quality**: Check for clarity, coherence, grammar, spelling, punctuation, and overall style. Ensure the tone is consistent with the likely intent (especially if it's pilot content for Skyscope itself)."
            "Your responsibilities are to:"
            "\n1. **Review for Quality**: Check for clarity, coherence, grammar, spelling, punctuation, and overall style. Ensure the tone is consistent with the likely intent."
            "\n2. **Brief Adherence (Inferred)**: Ensure the draft aligns with the characteristics of its content type. For example, a blog post should be well-structured with paragraphs; a tweet thread should consist of concise, numbered tweets. Check for natural keyword integration if keywords are evident."
            "\n3. **Fact-Checking (If Applicable)**: If the content makes specific factual claims that seem questionable, note this. (You don't have tools to verify externally unless the draft itself contains the source and it's part of your task to check it)."
            "\n4. **Edit for Improvement**: Make necessary edits to improve flow, engagement, and correctness. This includes rephrasing, restructuring sentences, and correcting errors."
            "\n5. **Format Consistency**: Ensure consistent formatting, especially for tweet threads (e.g., numbering, conciseness per tweet, clear separation)."
            "\nYour output should be the polished, edited version of the content as a single string. "
            "If major revisions are needed that are beyond your scope of editing (e.g., content is completely off-topic or structure is fundamentally wrong for the content type), clearly state this at the beginning of your output, followed by 'MAJOR_REVISIONS_NEEDED:', and then detail the issues. Otherwise, just output the edited content."
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

class SeoOptimizerAgent(SkyscopeSwarmAgent):
    """
    Optimizes content for SEO, generates meta descriptions and titles.
    """
    def __init__(self, agent_name: str = "SeoOptimizer", tools: list = None, **kwargs):
        system_prompt = (
            "You are an SEO Optimizer for Skyscope Sentinel Intelligence. "
            "You will receive edited content. Your task is to optimize it for search engines and generate relevant metadata. Infer the content type (e.g., 'blog post', 'tweet thread', 'short article') from the content's structure and style."
            "\nYour tasks are:"
            "\n1. **Keyword Analysis**: Identify the main keywords and themes within the received content. If target keywords were implicitly part of the content's generation, ensure they are well-represented."
            "\n2. **On-Page SEO**: Review and refine the content for:"
            "   - Appropriate keyword density and natural placement (avoid stuffing)."
            "   - Effective use of headings (H1, H2, H3) for structure and keyword relevance (primarily for articles/posts)."
            "   - Readability and good sentence structure."
            "\n3. **Content Optimization**: Suggest and apply improvements. This might involve minor rephrasing, adding keywords where appropriate, or adjusting heading structures. Preserve the original meaning and tone."
            "\n4. **Metadata Generation**: "
            "   - For 'blog post' or 'short article': Generate a concise meta description (approx. 150-160 characters) and an SEO-friendly title suggestion (50-60 characters)."
            "   - For 'tweet thread': Suggest 3-5 relevant hashtags. A meta description and SEO title are generally not applicable here; you can output 'Meta Description: N/A for tweet thread' and 'SEO Title: N/A for tweet thread'."
            "\n\nYour output MUST be a single string, structured as follows:"
            "\n[SEO-OPTIMIZED CONTENT BODY]"
            "\n---META_DESCRIPTION_START---"
            "\n[Generated Meta Description OR 'N/A for tweet thread']"
            "\n---META_DESCRIPTION_END---"
            "\n---SEO_TITLE_START---"
            "\n[Suggested SEO Title OR 'N/A for tweet thread']"
            "\n---SEO_TITLE_END---"
            "\n---HASHTAGS_START--- (Only if content type is 'tweet thread')"
            "\n[#hashtag1 #hashtag2 #hashtag3]"
            "\n---HASHTAGS_END--- (Only if content type is 'tweet thread')"
            "\nIf the content type is not a tweet thread, omit the HASHTAGS sections entirely, including the markers."
        )
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            tools=tools or [],
            department_name=CONTENT_DEPARTMENT_NAME,
            role_in_department="SEO & Content Performance Analyst",
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
    """
    print(f"Initializing Content Generation Swarm for topic: '{topic}', type: '{content_type}'...")
    cfg = Config()

    strategist_tools = [duckduckgo_search_function]

    strategist = ContentStrategistAgent(tools=strategist_tools, max_loops=1, verbose=verbose)
    writer = DraftWriterAgent(max_loops=1, verbose=verbose)
    editor = ReviewEditorAgent(max_loops=1, verbose=verbose)
    seo_optimizer = SeoOptimizerAgent(max_loops=1, verbose=verbose)

    agents_list = [strategist, writer, editor, seo_optimizer]
    workflow = SequentialWorkflow(agents=agents_list, verbose=verbose)

    initial_task_for_strategist = (
        f"Create a content brief for a '{content_type}' about '{topic}'. "
        f"The target audience is '{target_audience}', and the desired tone is '{tone}'. "
        f"Include an outline, SEO keywords, target length/tweet count, formatting specifics, and any other instructions for the DraftWriterAgent, adapting all elements for the specified content type."
    )

    print("Content Generation Swarm initialized. Starting workflow...")
    raw_seo_output = workflow.run(initial_task_for_strategist)

    content_body = raw_seo_output
    meta_description = "N/A"
    seo_title = "N/A"
    hashtags = "N/A"

    try:
        content_body_part, meta_part_full = raw_seo_output.split("---META_DESCRIPTION_START---", 1)
        content_body = content_body_part.strip()

        meta_desc_part, title_part_full = meta_part_full.split("---META_DESCRIPTION_END---", 1)
        meta_description = meta_desc_part.strip()

        # SEO_TITLE_START might not be present if it's N/A for tweet thread
        if "---SEO_TITLE_START---" in title_part_full:
            seo_title_content, rest_after_title = title_part_full.split("---SEO_TITLE_END---", 1)
            seo_title = seo_title_content.replace("---SEO_TITLE_START---", "").strip()

            if "---HASHTAGS_START---" in rest_after_title: # Check for hashtags only after title
                hashtags_content, _ = rest_after_title.split("---HASHTAGS_END---", 1)
                hashtags = hashtags_content.replace("---HASHTAGS_START---", "").strip()
        else: # Fallback if SEO_TITLE_START is not found (e.g. N/A case for tweet thread)
            # This part of parsing could be more robust, but assumes SEO agent follows one of the two paths for title.
             pass


    except ValueError:
        print("[WARN] Could not parse full SEO metadata from agent output. Using raw output as content body.")
        content_body = raw_seo_output

    if content_body:
        print("\nContent Generation Workflow completed.")
        print("----------------------------------------------------")
        print(f"SEO Title: {seo_title}")
        print(f"Meta Description: {meta_description}")
        if hashtags != "N/A" and hashtags: # Print hashtags only if they exist
            print(f"Hashtags: {hashtags}")
        print("Content Body (first 1000 chars):")
        print(content_body[:1000] + "..." if len(content_body) > 1000 else content_body)
        print("----------------------------------------------------")

        content_subdir = "generated_content"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = "".join(c if c.isalnum() else "_" for c in topic.split(" ", 3)[:3]).replace("__", "_")
        content_filename = f"{content_type.replace(' ', '_')}_{topic_slug}_{timestamp}.md"

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
                if seo_title != "N/A":
                    f.write(f"# SEO Title: {seo_title}\n\n")
                if meta_description != "N/A":
                    f.write(f"## Meta Description:\n{meta_description}\n\n")
                if hashtags != "N/A" and hashtags:
                    f.write(f"**Hashtags:** {hashtags}\n\n")
                if seo_title != "N/A" or meta_description != "N/A" or (hashtags != "N/A" and hashtags): # Add separator if any metadata present
                    f.write("---\n\n")
                f.write(f"# Content: {topic}\n\n")
                f.write(content_body)
                f.write(final_content_output)
            print(f"Generated content saved to: {full_content_path}")
            return full_content_path
        except Exception as e:
            print(f"Error saving generated content: {e}")
            return f"Workflow completed, but failed to save content. Content snippet: {content_body[:200]}..."
    else:
        print("\nContent Generation Workflow completed, but no final content body was generated.")
        return "Workflow completed, but no content body generated."


if __name__ == "__main__":
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

        seo_agent = SeoOptimizerAgent(max_loops=1)
        print(f"Initialized: {seo_agent.agent_name}, ID: {seo_agent.id}, Role: {seo_agent.role_in_department}")

        print("\nContent Generation Agents initialized successfully.")
    except Exception as e:
        print(f"Error during Content Generation Agent instantiation test: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Content Generation Agent Instantiation Test Complete ---")
    print("\n--- Testing run_content_generation_swarm ---")
    print("This test will run the full content generation swarm. Ensure Ollama is running.")

    test_topic = "The Future of AI in Education"
    test_audience = "Educators and Students"
    test_content_type = "blog post" # Test with "tweet thread" or "short article" as well
    test_tone = "optimistic yet realistic"

    print(f"\nStarting content generation swarm with topic: '{test_topic}', type: '{test_content_type}'")
    content_path_or_message = run_content_generation_swarm(
        topic=test_topic,
        target_audience=test_audience,
        content_type=test_content_type,
        tone=test_tone,
        verbose=False # Set to True for detailed agent logs
    )
    print(f"\nContent Generation Swarm run result: {content_path_or_message}")

    # Example for tweet thread
    # print(f"\nStarting content generation swarm for a TWEET THREAD...")
    # content_path_or_message_tweet = run_content_generation_swarm(
    #     topic="AI productivity hacks",
    #     content_type="tweet thread",
    #     tone="punchy and actionable",
    #     verbose=False
    # )
    # print(f"\nTweet Thread Generation Swarm run result: {content_path_or_message_tweet}")

    print("\n--- Content Generation Swarm Test Complete ---")
    print("\n--- Content Generation Agent Test Complete ---")
