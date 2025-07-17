import yaml
import os
import random
from datetime import datetime
import google.generativeai as genai
import re

# Rich library imports
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress # For potential future use
from rich.status import Status

# Initialize Rich Console
console = Console()

CONFIG_PATH = "ai_marketing_agent/config/settings.yaml"
KB_DIR = "ai_marketing_agent/knowledge_base"
IDEAS_FILE = "ai_marketing_agent/generated_content/content_ideas.txt" # Fallback
NEXT_IDEA_FILE = "ai_marketing_agent/generated_content/next_article_to_generate.txt"
OUTPUT_DIR = "ai_marketing_agent/generated_content"

def load_config():
    '''Loads configuration from YAML file with enhanced error handling.'''
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if not config: # Handles empty file or basic YAML parsing that results in None
            console.print(f"[bold red]CRITICAL ERROR:[/bold red] Configuration file '{CONFIG_PATH}' is empty or malformed. Please check its content and YAML syntax. Exiting.")
            return None

        essential_keys = ['gemini_api_key_env_var', 'bybit_affiliate_link', 'audience_personas', 'compliance']
        for key in essential_keys:
            if key not in config:
                console.print(f"[yellow]CONFIG WARNING:[/yellow] Key '{key}' is missing in '{CONFIG_PATH}'. The script might not function as expected. Please ensure this key is defined.")

        return config

    except FileNotFoundError:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Configuration file not found at '{CONFIG_PATH}'. Please ensure the file exists. You might need to create it based on the template or run an initial setup if available. Exiting.")
        return None
    except yaml.YAMLError as e:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Configuration file '{CONFIG_PATH}' is malformed. YAML parsing error: {e}. Please check its content and YAML syntax. Exiting.")
        return None
    except Exception as e:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] An unexpected error occurred while loading configuration from '{CONFIG_PATH}': {e}. Exiting.")
        return None

def load_knowledge_base_file(filename):
    filepath = os.path.join(KB_DIR, filename)
    try:
        with open(filepath, 'r') as f: return f.read()
    except FileNotFoundError:
        console.print(f"[yellow]Warning:[/yellow] KB file {filepath} not found. Proceeding with empty content for this KB.")
        return ""
    except Exception as e:
        console.print(f"[red]Error loading KB file {filepath}:[/red] {e}")
        return ""

def load_content_ideas(): # Fallback
    try:
        with open(IDEAS_FILE, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("=") and not "Content Ideas Generated on" in line]
            lines = [line for line in lines if line]
            return [line[2:] if line.startswith("- ") else line for line in lines if line.strip()]
    except FileNotFoundError:
        console.print(f"[yellow]Warning:[/yellow] Fallback ideas file {IDEAS_FILE} not found.")
        return []
    except Exception as e:
        console.print(f"[red]Error loading fallback ideas from {IDEAS_FILE}:[/red] {e}")
        return []

def load_next_idea():
    try:
        os.makedirs(os.path.dirname(NEXT_IDEA_FILE), exist_ok=True)
        if not os.path.exists(NEXT_IDEA_FILE):
             console.print(f"[blue]Info:[/blue] Strategic choice file {NEXT_IDEA_FILE} not found. Will use fallback ideas list.")
             return None
        with open(NEXT_IDEA_FILE, 'r') as f:
            chosen_idea = f.read().strip()
            return chosen_idea if chosen_idea else None
    except Exception as e:
        console.print(f"[red]Error loading next idea from {NEXT_IDEA_FILE}:[/red] {e}")
        return None

def get_content_type(idea_text):
    if not idea_text: return "general_article"
    idea_lower = idea_text.lower()
    if "how to" in idea_lower or "guide" in idea_lower or "getting started" in idea_lower: return "how-to"
    if "vs" in idea_lower or "versus" in idea_lower or "compare" in idea_lower: return "comparison"
    if "what is" in idea_lower or "explaining" in idea_lower or "understanding" in idea_lower: return "explainer"
    if "review" in idea_lower: return "review"
    if "news" in idea_lower or "update" in idea_lower or "latest" in idea_lower: return "news_update"
    return "general_article"

def generate_llm_content(prompt_text, api_key, content_description="content"): # Added content_description for spinner
    with console.status(f"[b blue]Communicating with LLM for {content_description}...[/b blue]", spinner="dots") as status:
        try:
            genai.configure(api_key=api_key)
            model_name = "gemini-1.0-pro"
            model = genai.GenerativeModel(model_name)
            # console.print(f"\nAttempting LLM generation (model: {model_name})...") # Replaced by status
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            response = model.generate_content(prompt_text, safety_settings=safety_settings)

            if response.prompt_feedback and response.prompt_feedback.block_reason:
                return f"Error: Prompt for {content_description} blocked by API ({response.prompt_feedback.block_reason}). Review prompt or safety settings."
            if not response.candidates:
                 return f"Error: No candidates from LLM for {content_description}. Prompt may be too restrictive or issue with API. Response details: {response}"

            candidate = response.candidates[0]
            if candidate.finish_reason.name != "STOP":
                 finish_reason_message = f"Warning: LLM generation for {content_description} finished with reason: {candidate.finish_reason.name}."
                 if candidate.finish_reason.name == "SAFETY":
                      safety_info = " Safety details: "
                      if candidate.safety_ratings:
                          for rating in candidate.safety_ratings:
                              if rating.probability.name != "NEGLIGIBLE":
                                  safety_info += f" {rating.category.name} - {rating.probability.name};"
                      return f"Error: Generation of {content_description} stopped by safety filter.{safety_info if safety_info != ' Safety details: ' else ''}"

                 if candidate.content and candidate.content.parts:
                     console.print(f"[yellow]{finish_reason_message}[/yellow] Partial content might be returned for {content_description}.")
                     return "".join(part.text for part in candidate.content.parts)
                 return f"Error: Generation of {content_description} finished with reason '{candidate.finish_reason.name}' but no content. {finish_reason_message}"

            if candidate.content and candidate.content.parts:
                console.print(f"[green]LLM generation for {content_description} successful.[/green]")
                return "".join(part.text for part in candidate.content.parts)

            return f"Error: No valid content parts in LLM response for {content_description} despite 'STOP' reason."
        except Exception as e:
            console.print(f"[bold red]Exception during LLM call for {content_description}:[/bold red] {e}")
            return f"Error during LLM call for {content_description}: {e}" # Return error message for main block to handle

def construct_prompt_v3(idea, content_type, persona, config, kb_features_summary, kb_ethics_summary, kb_programs_summary, affiliate_link_override=None):
    target_keywords_list = config.get('target_keywords', [])
    default_primary_keyword = random.choice(target_keywords_list) if target_keywords_list else "Bybit trading"

    if persona and persona.get('keywords'):
        primary_keyword = random.choice(persona['keywords'])
    else:
        primary_keyword = default_primary_keyword

    final_affiliate_link = affiliate_link_override if affiliate_link_override else config.get('bybit_affiliate_link', 'YOUR_BYBIT_LINK')

    blog_disclosure = config.get('compliance', {}).get('disclosure_texts', {}).get('blog', '#Ad #BybitAffiliate')
    risk_disclaimer = config.get('compliance', {}).get('risk_disclaimer', 'Cryptocurrency investment is subject to high market risk.')

    persona_name = persona['name'] if persona else "General User"
    persona_desc = persona['description'] if persona else "A crypto enthusiast."
    persona_tone = persona['preferred_tone'] if persona and 'preferred_tone' in persona else "clear, informative, and engaging"

    prompt = f'''
You are an AI Marketing Assistant creating a blog post for Bybit, a cryptocurrency exchange.
Your target audience is '{persona_name}': {persona_desc}.
The tone of the article should be: {persona_tone}.

**Blog Post Title/Idea:** {idea}
**Content Type to Generate:** {content_type}
**Incorporate this Primary Keyword naturally:** {primary_keyword}

**Task:** Write a compelling and informative blog post of approximately 400-700 words. Use Markdown for formatting.

**Detailed Instructions based on Content Type '{content_type}':**
'''
    if content_type == "how-to":
        prompt += "- Provide clear, actionable, step-by-step instructions. Simplify complex steps for the target persona.\n"
        prompt += "- Focus on specific Bybit features or tools that make this 'how-to' easy or effective for the user.\n"
    elif content_type == "comparison":
        competitor_match = re.search(r"vs\.?\s*([\w\s]+)", idea, re.IGNORECASE)
        competitor_name = competitor_match.group(1).strip() if competitor_match else "another platform"
        prompt += f"- Objectively compare Bybit with '{competitor_name}' concerning aspects of '{idea}'.\n"
        prompt += f"- Highlight Bybit's advantages, tailoring points to what '{persona_name}' would value most.\n"
        prompt += f"- If specific, verifiable data for '{competitor_name}' isn't available, state this and focus on Bybit's offerings.\n"
    elif content_type == "explainer":
        prompt += f"- Explain the core concepts of '{idea}' in a way that is easily understandable for '{persona_name}'.\n"
        prompt += f"- Clearly connect the explanation to practical applications or benefits on the Bybit platform.\n"
    elif content_type == "review":
        prompt += f"- Write a balanced and honest review of the subject matter in '{idea}'.\n"
        prompt += f"- Discuss both pros and cons from the perspective of '{persona_name}'.\n"
    else: # general_article, news_update
        prompt += f"- Discuss the topic '{idea}' and its current relevance to '{persona_name}' in the crypto space.\n"
        prompt += f"- Naturally integrate Bybit's role, related platform features, or services where appropriate.\n"

    prompt += f'''
**Knowledge Base Context (Summaries - use these to inform your writing):**
*   Key Bybit Features (for reference): ...{kb_features_summary}...
*   Ethical Marketing Rules (Strictly Follow): ...{kb_ethics_summary}... (Crucial: No profit guarantees, be truthful, avoid hype)
*   Bybit Programs Overview (for background context): ...{kb_programs_summary}...

**Mandatory Compliance Requirements:**
1.  **Disclosure First:** The VERY FIRST line of the blog post MUST be: {blog_disclosure}
2.  **Risk Disclaimer Last:** The VERY LAST line of the blog post MUST be: {risk_disclaimer}
3.  **Call to Action:** Before the final risk disclaimer, include a relevant call to action. Examples: "Explore these features on Bybit: {final_affiliate_link}", "Ready to get started? Sign up at Bybit: {final_affiliate_link}"

Output ONLY the Markdown content for the blog post. Do not add any other text, commentary, or preambles before or after the Markdown content.
'''
    return prompt

def save_generated_content(idea, content_type, persona_name, content_body, content_desc="content"): # Added content_desc
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_tag = persona_name.replace(" ", "_").lower() if persona_name else "general"
    sanitized_idea = re.sub(r'[^a-zA-Z0-9_\-]', '_', idea[:30])
    filename = f"llm_draft_{content_type}_{persona_tag}_{sanitized_idea}_{timestamp}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)

    header = f"--- Generated Content (LLM) ---\n"
    header += f"Type: {content_type.capitalize()}\n"
    header += f"Idea: {idea}\n"
    header += f"Persona: {persona_name if persona_name else 'N/A'}\n"
    header += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"--- \n\n"

    try:
        with open(filepath, 'w') as f: f.write(header + content_body)
        console.print(f"[green]Successfully saved LLM-generated {content_desc} to {filepath}[/green]")
    except IOError as e:
        console.print(f"[red]Error saving {content_desc} to {filepath}:[/red] {e}")

if __name__ == "__main__":
    console.print(Panel("Enhanced LLM Content Generator (V3 - With Personas & Strategy Input)",
                      title="[bold magenta]Agent Script[/bold magenta]",
                      subtitle="[dim]Initializing...[/dim]"))

    config_data = load_config()
    if config_data is None:
        console.print("[bold red]Exiting script due to critical configuration loading error.[/bold red]")
        exit(1)

    simulated_qr_link = "https://www.bybit.com/invite?ref=SIMULATEDQR"
    console.print(f"[blue]MAIN_EXEC:[/blue] Using simulated QR link for testing: [link={simulated_qr_link}]{simulated_qr_link}[/link]")

    gemini_api_key_name = config_data.get('gemini_api_key_env_var', "GEMINI_API_KEY")
    api_key = os.environ.get(gemini_api_key_name)

    if not api_key:
        console.print(f"[bold red]Error:[/bold red] API Key '{gemini_api_key_name}' not found in environment variables.")
        console.print(f"[blue]Please run 'python ai_marketing_agent/scripts/setup_env.py' or set the variable manually.[/blue]")
    else:
        console.print(f"[green]API Key '{gemini_api_key_name}' found in environment.[/green]")

        selected_idea = load_next_idea()
        if not selected_idea or "Error:" in selected_idea:
            console.print(f"[yellow]Warning:[/yellow] No valid strategic idea from '{NEXT_IDEA_FILE}' (Content: '{selected_idea}'). Choosing randomly from general ideas list.")
            content_ideas = load_content_ideas()
            if not content_ideas:
                console.print("[bold red]CRITICAL:[/bold red] No content ideas available at all (from content_ideas.txt). Exiting.");
                exit(1)
            selected_idea = random.choice(content_ideas)
            if not selected_idea:
                 console.print("[bold red]CRITICAL:[/bold red] No idea could be selected even from fallback. Exiting."); exit(1)

        content_type_val = get_content_type(selected_idea) # Renamed to avoid conflict

        personas = config_data.get('audience_personas', {})
        chosen_persona_key = random.choice(list(personas.keys())) if personas else None
        chosen_persona = personas.get(chosen_persona_key) if chosen_persona_key else None
        persona_name_for_log = chosen_persona['name'] if chosen_persona else "General"

        console.print(Panel(f"Selected idea: '[b]{selected_idea}[/b]'\nType: [cyan]{content_type_val}[/cyan]\nPersona: [italic green]{persona_name_for_log}[/italic green]", title="[bold blue]Content Generation Task[/bold blue]"))

        kb_features_full = load_knowledge_base_file("kb_bybit_features.txt")
        kb_ethics_full = load_knowledge_base_file("kb_ethical_guidelines.txt")
        kb_programs_full = load_knowledge_base_file("kb_bybit_programs.txt")

        features_summary = "\n".join(kb_features_full.split('\n')[:20])
        ethics_summary = "\n".join(kb_ethics_full.split('\n')[:25])
        programs_summary = "\n".join(kb_programs_full.split('\n')[:15])

        prompt = construct_prompt_v3(selected_idea, content_type_val, chosen_persona, config_data,
                                     features_summary, ethics_summary, programs_summary,
                                     affiliate_link_override=simulated_qr_link)

        # Pass content_description to generate_llm_content
        generated_text_raw = generate_llm_content(prompt, api_key, content_description=f"{content_type_val} blog post")


        if "Error:" not in generated_text_raw: # Check if error message was returned
            final_text = generated_text_raw.strip()

            blog_disclosure = config_data.get('compliance', {}).get('disclosure_texts', {}).get('blog', '#Ad')
            risk_disclaimer = config_data.get('compliance', {}).get('risk_disclaimer', 'Trade crypto responsibly.')

            current_first_line = final_text.split('\n')[0].strip()
            if not current_first_line.startswith(blog_disclosure):
                console.print(f"[yellow]Warning:[/yellow] Disclosure '{blog_disclosure}' not at the very start. Prepending.")
                final_text = re.sub(r"^(#Ad|#BybitAffiliate|Disclosure:.*?)\s*\n*", "", final_text, flags=re.IGNORECASE | re.MULTILINE).strip()
                final_text = f"{blog_disclosure}\n\n{final_text}"

            if not final_text.strip().endswith(risk_disclaimer):
                 console.print(f"[yellow]Warning:[/yellow] Risk disclaimer '{risk_disclaimer}' not at the very end. Appending.")
                 final_text = final_text.strip()
                 if final_text.endswith("---"): final_text = final_text[:-3].strip()
                 final_text = f"{final_text}\n\n---\n{risk_disclaimer}"

            save_generated_content(selected_idea, content_type_val, persona_name_for_log, final_text, content_desc=f"{content_type_val} blog post")

            console.print("\n[blue]INFO:[/blue] Social media prompt generation skipped as 'construct_social_media_prompt_v1' was not found in this version of the script.")

        else:
            # Error string from generate_llm_content or previous checks
            console.print(f"[bold red]Content generation failed.[/bold red] LLM Output/Error: {generated_text_raw}")
            debug_filename = f"failed_prompt_and_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            debug_filepath = os.path.join(OUTPUT_DIR, debug_filename)
            os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure dir exists
            try:
                with open(debug_filepath, 'w') as df:
                    df.write("--- FAILED PROMPT ---\n")
                    df.write(prompt)
                    df.write("\n\n--- LLM ERROR/OUTPUT ---\n")
                    df.write(generated_text_raw)
                console.print(f"[yellow]Saved failed prompt and error to {debug_filepath}[/yellow]")
            except Exception as e_debug:
                console.print(f"[bold red]Error saving debug file:[/bold red] {e_debug}")

    console.print(Panel("[bold green]Enhanced Content Generator script (V3 with personas) finished.[/bold green]",padding=(1,2)))
