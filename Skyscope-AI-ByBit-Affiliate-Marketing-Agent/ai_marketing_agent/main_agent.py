import os
import yaml
import random
import importlib # To dynamically load our scripts as modules
import importlib.util # Required for spec_from_file_location

# Rich imports for main agent's CLI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.prompt import Confirm

console = Console()

# --- Configuration ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/settings.yaml') # Relative to this file's location

def load_main_config():
    """Loads the main configuration file for the agent."""
    if not os.path.exists(CONFIG_PATH):
        console.print(f"[bold red]FATAL AGENT ERROR:[/bold red] Main configuration file not found at '{CONFIG_PATH}'.")
        console.print("Please ensure 'settings.yaml' exists in the 'ai_marketing_agent/config/' directory.")
        return None
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            console.print(f"[bold red]FATAL AGENT ERROR:[/bold red] Main configuration file '{CONFIG_PATH}' is empty or malformed.")
            return None
        console.print("[green]Main configuration loaded successfully.[/green]")
        return config
    except Exception as e:
        console.print(f"[bold red]FATAL AGENT ERROR:[/bold red] Error loading main configuration from '{CONFIG_PATH}': {e}")
        return None

# --- Dynamically Load Agent Scripts as Modules ---
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')

def import_script_module(script_name):
    try:
        module_full_name = f"ai_marketing_agent.scripts.{script_name.replace('.py', '')}"

        # Check if already imported (e.g. by another import statement or previous call)
        if module_full_name in globals(): # More robust check would be sys.modules
             return globals()[module_full_name] # Or sys.modules[module_full_name]

        module_path = os.path.join(scripts_dir, script_name)
        spec = importlib.util.spec_from_file_location(module_full_name, module_path)

        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # Before execution, add to sys.modules to handle potential circular dependencies within scripts if any
            # sys.modules[module_full_name] = module
            spec.loader.exec_module(module)
            # Try to set the console instance for unified output
            if hasattr(module, 'console') and isinstance(getattr(module, 'console'), Console):
                 # If the module already has its own console, we might not want to overwrite it,
                 # or we ensure it's a compatible Rich console. For now, let's assume it's fine.
                 pass # It will use its own console instance.
            elif hasattr(module, 'console'): # If it has a console attribute but not instance of Rich Console
                 module.console = console # Assign the main agent's console
            return module
        else:
            console.print(f"[bold red]AGENT ERROR:[/bold red] Could not create spec for script module: {script_name}")
            return None
    except FileNotFoundError:
        console.print(f"[bold red]AGENT ERROR:[/bold red] Script file not found: {script_name} in {scripts_dir}")
        return None
    except Exception as e:
        console.print(f"[bold red]AGENT ERROR:[/bold red] Failed to import script module '{script_name}': {e}")
        return None

# --- Main Agent Workflow ---
def run_agent_workflow():
    console.print(Panel(" Autonomous AI Marketing Agent for ByBit ", title="[bold blue_violet]Welcome![/bold blue_violet]", style="bold bright_blue", expand=False))

    config = load_main_config()
    if not config:
        console.print("[bold red]Agent cannot start without valid configuration. Exiting.[/bold red]")
        return

    console.print(Rule("[b bright_cyan]Loading Agent Modules[/b bright_cyan]"))
    idea_gen_mod = import_script_module("content_idea_generator.py")
    strat_chooser_mod = import_script_module("strategic_content_chooser.py")
    qr_proc_mod = import_script_module("qr_processor.py")
    content_gen_mod = import_script_module("basic_content_generator.py")
    post_sched_mod = import_script_module("post_scheduler.py")
    opp_finder_mod = import_script_module("opportunity_finder.py")

    essential_modules = {
        "Idea Generator": idea_gen_mod, "Strategic Chooser": strat_chooser_mod,
        "QR Processor": qr_proc_mod, "Content Generator": content_gen_mod,
        "Post Scheduler": post_sched_mod
    }
    modules_ok = True
    for name, mod in essential_modules.items():
        if not mod:
            console.print(f"[bold red]Failed to load essential module: {name}. Agent cannot continue.[/bold red]")
            modules_ok = False
    if not modules_ok: return

    if opp_finder_mod is None and config.get('agent_workflow', {}).get('enable_opportunity_finder', False):
        console.print("[yellow]WARN:[/yellow] Opportunity finder module failed to load, but is enabled in config. This step will be skipped.")

    console.print("[green]All essential agent modules loaded.[/green]")

    console.print(Rule("[b bright_cyan]Step 1: Content Idea Generation[/b bright_cyan]"))
    try:
        if hasattr(idea_gen_mod, 'generate_content_ideas') and hasattr(idea_gen_mod, 'save_content_ideas'):
            console.print("[blue]INFO:[/blue] Generating content ideas...")
            ideas = idea_gen_mod.generate_content_ideas(config)
            if ideas:
                idea_gen_mod.save_content_ideas(ideas)
            else:
                console.print("[yellow]WARN:[/yellow] No content ideas were generated by idea_gen_mod.")
        else:
            console.print("[yellow]WARN:[/yellow] `generate_content_ideas` or `save_content_ideas` not found in idea_gen_mod. Skipping direct idea generation.")
    except Exception as e:
        console.print(f"[red]ERROR in Idea Generation step:[/red] {e}")

    console.print(Rule("[b bright_cyan]Step 2: Strategic Content Choice[/b bright_cyan]"))
    selected_idea_for_content = "Default: Explore ByBit Today"
    try:
        if hasattr(strat_chooser_mod, 'load_ideas') and hasattr(strat_chooser_mod, 'choose_next_article'):
            available_ideas = strat_chooser_mod.load_ideas()
            if available_ideas:
                perf_data = strat_chooser_mod.load_performance_data() if hasattr(strat_chooser_mod, 'load_performance_data') else {}
                trends = strat_chooser_mod.load_trending_topics() if hasattr(strat_chooser_mod, 'load_trending_topics') else {}
                chosen_idea_text = strat_chooser_mod.choose_next_article(available_ideas, perf_data, trends)
                if chosen_idea_text and "Error:" not in chosen_idea_text:
                    selected_idea_for_content = chosen_idea_text
                    console.print(f"[bold green]Strategically selected idea:[/bold green] [i]'{selected_idea_for_content}'[/i]")
                else:
                    console.print(f"[yellow]WARN:[/yellow] Could not strategically choose an idea (Result: {chosen_idea_text}). Using fallback.")
            else:
                console.print("[yellow]WARN:[/yellow] No ideas available for strategic chooser. Using fallback.")
        else:
            console.print("[yellow]WARN:[/yellow] Core functions not found in strat_chooser_mod. Using fallback idea.")
    except Exception as e:
        console.print(f"[red]ERROR in Strategic Choice step:[/red] {e}")
    console.print(f"[blue]INFO:[/blue] Idea for content generation: '[b]{selected_idea_for_content}[/b]'")

    console.print(Rule("[b bright_cyan]Step 3: Image & QR Code Processing[/b bright_cyan]"))
    affiliate_link_to_use = config.get('bybit_affiliate_link', 'YOUR_BYBIT_LINK_DEFAULT')
    selected_image_for_post = None # Full path to image

    image_source_dir_config = config.get('agent_workflow', {}).get('image_source_directory', '.')
    # Make image_source_dir relative to the project root (where main_agent.py is)
    image_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', image_source_dir_config))
    image_extensions = config.get('agent_workflow', {}).get('image_extensions_to_scan', ['.png', '.jpg', '.jpeg'])

    try:
        if not os.path.isdir(image_source_dir):
            console.print(f"[yellow]WARN:[/yellow] Image source directory '{image_source_dir}' not found. Skipping image selection.")
        else:
            candidate_images = [f for f in os.listdir(image_source_dir) if os.path.isfile(os.path.join(image_source_dir, f)) and any(f.lower().endswith(ext) for ext in image_extensions) and "bybit" in f.lower()]
            if not candidate_images:
                 candidate_images = [f for f in os.listdir(image_source_dir) if os.path.isfile(os.path.join(image_source_dir, f)) and any(f.lower().endswith(ext) for ext in image_extensions)]

            if candidate_images:
                selected_image_name = random.choice(candidate_images)
                selected_image_for_post = os.path.join(image_source_dir, selected_image_name)
                console.print(f"[blue]INFO:[/blue] Selected image for QR processing: '[b]{selected_image_for_post}[/b]'")

                qr_default_fallback_link = config.get('qr_code_processing', {}).get('fallback_link_if_no_qr', affiliate_link_to_use)
                extracted_link = qr_proc_mod.extract_qr_link_from_image(selected_image_for_post, default_if_not_found=qr_default_fallback_link)

                if extracted_link and extracted_link != qr_default_fallback_link:
                    affiliate_link_to_use = extracted_link
                    console.print(f"[bold green]SUCCESS:[/bold green] Extracted affiliate link from QR code: [link={affiliate_link_to_use}]{affiliate_link_to_use}[/link]")
                else:
                    affiliate_link_to_use = qr_default_fallback_link
                    console.print(f"[yellow]WARN:[/yellow] No QR code link extracted, or error. Using fallback link: [link={affiliate_link_to_use}]{affiliate_link_to_use}[/link]")
            else:
                console.print(f"[yellow]WARN:[/yellow] No images found in '{image_source_dir}' with extensions {image_extensions}. QR processing skipped. Using default affiliate link.")
    except Exception as e:
        console.print(f"[red]ERROR in Image/QR Processing step:[/red] {e}")
    console.print(f"[blue]INFO:[/blue] Affiliate link to be used in content: [link={affiliate_link_to_use}]{affiliate_link_to_use}[/link]")

    console.print(Rule("[b bright_cyan]Step 4: Content Generation[/b bright_cyan]"))
    generated_blog_content_md = None # Store the actual markdown content

    try:
        personas = config.get('audience_personas', {})
        chosen_persona_key = random.choice(list(personas.keys())) if personas else None
        chosen_persona = personas.get(chosen_persona_key) if chosen_persona_key else None
        persona_name_for_log = chosen_persona['name'] if chosen_persona else "General"
        console.print(f"[blue]INFO:[/blue] Using persona: '[b]{persona_name_for_log}[/b]'")

        blog_content_type = content_gen_mod.get_content_type(selected_idea_for_content) if hasattr(content_gen_mod, 'get_content_type') else "general_article"

        kb_features_summary = content_gen_mod.load_knowledge_base_file("kb_bybit_features.txt") if hasattr(content_gen_mod, 'load_knowledge_base_file') else ""
        kb_ethics_summary = content_gen_mod.load_knowledge_base_file("kb_ethical_guidelines.txt") if hasattr(content_gen_mod, 'load_knowledge_base_file') else ""
        kb_programs_summary = content_gen_mod.load_knowledge_base_file("kb_bybit_programs.txt") if hasattr(content_gen_mod, 'load_knowledge_base_file') else ""

        prompt_constructor_func_name = None
        if hasattr(content_gen_mod, 'construct_blog_prompt_v4'): prompt_constructor_func_name = 'construct_blog_prompt_v4'
        elif hasattr(content_gen_mod, 'construct_prompt_v3'): prompt_constructor_func_name = 'construct_prompt_v3'

        if prompt_constructor_func_name:
            prompt_constructor = getattr(content_gen_mod, prompt_constructor_func_name)
            blog_prompt = prompt_constructor(selected_idea_for_content, blog_content_type, chosen_persona, config, kb_features_summary, kb_ethics_summary, kb_programs_summary, affiliate_link_override=affiliate_link_to_use)

            api_key_env_var = config.get('gemini_api_key_env_var', "GEMINI_API_KEY")
            api_key = os.environ.get(api_key_env_var)

            if not api_key:
                console.print(f"[bold red]ERROR:[/bold red] Gemini API Key from env var '{api_key_env_var}' not found. Cannot generate LLM content.")
            else:
                raw_blog_text = content_gen_mod.generate_llm_content(blog_prompt, api_key, f"{blog_content_type} blog post")
                if raw_blog_text and "Error:" not in raw_blog_text:
                    generated_blog_content_md = raw_blog_text.strip() # Keep MD, scheduler might convert to HTML
                    # (Disclosure/disclaimer logic might be needed here if not handled by basic_content_generator)
                    content_gen_mod.save_generated_content(selected_idea_for_content, blog_content_type, persona_name_for_log, generated_blog_content_md, content_desc=f"{blog_content_type} blog post")
                    console.print(f"[green]SUCCESS:[/green] Blog content generated for '{selected_idea_for_content}'.")

                    # Social Media Snippets
                    # ... (Social media snippet generation logic would go here, similar to basic_content_generator) ...
                else:
                    console.print(f"[red]ERROR:[/red] Failed to generate blog content: {raw_blog_text}")
        else:
            console.print("[red]ERROR:[/red] No suitable blog prompt constructor found in content_gen_mod.")

    except Exception as e:
        console.print(f"[red]ERROR in Content Generation step:[/red] {e}")

    if config.get('agent_workflow', {}).get('enable_autonomous_posting', False):
        console.print(Rule("[b bright_cyan]Step 5: Autonomous Posting[/b bright_cyan]"))
        if generated_blog_content_md:
            try:
                # TODO: Convert Markdown to HTML if needed by posting platforms (e.g. Blogger)
                # For now, assume content is HTML or platform handles MD
                blog_html_content_for_post = generated_blog_content_md # Placeholder

                blogger_config = config.get('posting_platforms', {}).get('blogger', {})
                if blogger_config.get('enabled', False) and hasattr(post_sched_mod, 'get_blogger_service') and hasattr(post_sched_mod, 'post_to_blogger'):
                    console.print("[blue]INFO:[/blue] Attempting to post to Blogger...")
                    blogger_service = post_sched_mod.get_blogger_service(config)
                    if blogger_service:
                        post_sched_mod.post_to_blogger(blogger_service, config, title=selected_idea_for_content, content_html=blog_html_content_for_post, labels=[blog_content_type, persona_name_for_log], affiliate_link_override=affiliate_link_to_use, image_path_for_post=selected_image_for_post)

                wordpress_config = config.get('posting_platforms', {}).get('wordpress', {})
                if wordpress_config.get('enabled', False) and hasattr(post_sched_mod, 'post_to_wordpress'):
                    console.print("[blue]INFO:[/blue] Attempting to post to WordPress (placeholder)...")
                    post_sched_mod.post_to_wordpress(config, title=selected_idea_for_content, content_html=blog_html_content_for_post, affiliate_link_override=affiliate_link_to_use, image_path_for_post=selected_image_for_post)

                # ... (Social media posting call would go here) ...
            except Exception as e:
                console.print(f"[red]ERROR in Autonomous Posting step:[/red] {e}")
        else:
            console.print("[yellow]WARN:[/yellow] No generated blog content available to post.")
    else:
        console.print("[blue]INFO:[/blue] Autonomous posting is disabled in settings.")

    if config.get('agent_workflow', {}).get('enable_opportunity_finder', False) and opp_finder_mod:
        console.print(Rule("[b bright_cyan]Step 6: Opportunity Finding[/b bright_cyan]"))
        try:
            if hasattr(opp_finder_mod, 'search_google') and hasattr(opp_finder_mod, 'save_opportunities'):
                console.print("[blue]INFO:[/blue] Starting online search for posting opportunities...")
                target_keywords = config.get('target_keywords', ["crypto"])
                primary_keyword = target_keywords[0] if target_keywords else "cryptocurrency"
                queries_to_search = [f"{primary_keyword} blogs guest posts", f"{primary_keyword} forums", f"write for us {primary_keyword}"]
                all_ops_found = {}
                for q_idx, q_val in enumerate(queries_to_search):
                    if q_idx > 0:
                         if not Confirm.ask(f"Continue with next opportunity search query ('{q_val}')?", default=True, console=console):
                            console.print("[blue]INFO: Skipping remaining opportunity searches by user choice.[/blue]")
                            break
                    raw_urls = opp_finder_mod.search_google(q_val)
                    if raw_urls:
                         all_ops_found[q_val] = opp_finder_mod.filter_and_analyze_urls(raw_urls)
                    else:
                         all_ops_found[q_val] = []
                opp_finder_mod.save_opportunities(all_ops_found)
            else:
                console.print("[yellow]WARN:[/yellow] Core functions not found in opp_finder_mod. Skipping opportunity finding.")
        except Exception as e:
            console.print(f"[red]ERROR in Opportunity Finding step:[/red] {e}")
    else:
        console.print("[blue]INFO:[/blue] Online opportunity finding is disabled or module failed to load.")

    console.print(Panel(" Agent Workflow Completed ", style="bold bright_green", title="[bold blue_violet]Finished![/bold blue_violet]", expand=False))

if __name__ == "__main__":
    run_agent_workflow()
