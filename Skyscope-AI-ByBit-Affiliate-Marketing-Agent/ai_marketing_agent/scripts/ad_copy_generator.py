import yaml
import os
import random
from datetime import datetime
import google.generativeai as genai # Kept for consistency, though not used in this version's ad gen
import re

# Rich library imports
from rich.console import Console
from rich.panel import Panel
from rich.text import Text # If needed for complex text
from rich.status import Status

# Initialize Rich Console
console = Console()

CONFIG_PATH = "ai_marketing_agent/config/settings.yaml"
OUTPUT_DIR = "ai_marketing_agent/generated_content"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "ad_copy_examples.txt")
IDEAS_FILE = os.path.join(OUTPUT_DIR, "content_ideas.txt") # Optional: to pick an idea

def load_config():
    '''Loads configuration from YAML file with enhanced error handling.'''
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            console.print(f"[bold red]CRITICAL ERROR:[/bold red] Configuration file '{CONFIG_PATH}' is empty or malformed. Please check its content and YAML syntax. Exiting.")
            return None

        essential_keys = {
            'gemini_api_key_env_var': "Gemini API key environment variable name is needed (though current ad gen is template-based).",
            'compliance': "Compliance settings are crucial for ad copy."
        }
        for key, message in essential_keys.items():
            if key not in config:
                console.print(f"[yellow]CONFIG WARNING:[/yellow] Key '[b]{key}[/b]' is missing in '{CONFIG_PATH}'. {message} Ad copy generation might not function as expected or adhere to compliance. Please ensure this key is defined.")

        if 'compliance' in config and 'restricted_keywords' not in config.get('compliance', {}):
            console.print(f"[yellow]CONFIG WARNING:[/yellow] Sub-key '[b]restricted_keywords[/b]' is missing under 'compliance' in '{CONFIG_PATH}'. This is important for ad copy compliance. Please ensure this key is defined.")

        return config

    except FileNotFoundError:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Configuration file not found at '{CONFIG_PATH}'. Please ensure the file exists. You might need to create it based on the template or run an initial setup if available. Exiting.")
        return None
    except yaml.YAMLError as e:
        console.print(f"[bold red]CRITICAL ERROR:[/bold red] Configuration file '{CONFIG_PATH}' is malformed. YAML parsing error: {e}. Please check its content and YAML syntax. Exiting.")
        return None
    except Exception as e:
        console.print(f"[red]Error loading config {CONFIG_PATH}:[/red] {e}")
        return None

def load_content_ideas():
    """Loads content ideas from the ideas file (optional)."""
    try:
        with open(IDEAS_FILE, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("=")]
            if len(lines) > 1:
                if "Content Ideas Generated on" in lines[0] or all(c == '=' for c in lines[1]):
                    lines = lines[2:]
                lines = [line for line in lines if not all(c == '=' for c in line)]
                lines = [line for line in lines if line.strip()]
                return [line[2:] if line.startswith("- ") else line for line in lines if line.strip()]
        return []
    except FileNotFoundError:
        console.print(f"[blue]Info:[/blue] Content ideas file '{IDEAS_FILE}' not found. Proceeding without it.")
        return []
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Error loading content ideas from '{IDEAS_FILE}': {e}. Proceeding without them.")
        return []

def generate_ad_copy(idea_or_feature, config):
    """Generates simulated ad copy examples based on templates."""
    console.print(f"\n[cyan]Generating template-based ad copy for:[/cyan] [b]{idea_or_feature}[/b]")
    if not config:
        return {"error": "Config not loaded."} # Plain error string

    keywords = config.get('target_keywords', ['Bybit', 'crypto trading'])
    affiliate_link_placeholder = config.get('bybit_affiliate_link', 'YOUR_BYBIT_LINK_HERE')
    disclosure = config.get('compliance', {}).get('disclosure_texts', {}).get('social', '#Ad')
    restricted_keywords = config.get('compliance', {}).get('restricted_keywords', [])

    def check_restricted(text):
        for r_keyword in restricted_keywords:
            if r_keyword.lower() in text.lower():
                console.print(f"[yellow]Ad Compliance:[/yellow] Found restricted keyword '[b]{r_keyword}[/b]' in text: '{text}'")
                return True
        return False

    ad_examples = []
    base_display_url_part = "bybit.com"
    if "YOUR_BYBIT_AFFILIATE_LINK_HERE" not in affiliate_link_placeholder and "://" in affiliate_link_placeholder:
        try:
            base_display_url_part = affiliate_link_placeholder.split("://")[1].split("/")[0]
        except IndexError:
            pass

    headline1_generic = f"Trade Crypto Securely {disclosure}"
    headline2_generic = f"Low Fees, Top Platform"
    headline3_generic = f"Join Bybit Today!"
    description1_generic = f"Explore Bitcoin, Ethereum & more on Bybit. Advanced tools, 24/7 support. {random.choice(keywords)}."
    description2_generic = f"Trusted by millions. Your premier crypto exchange for digital assets."
    display_url_generic = base_display_url_part + "/Official"

    if not any(check_restricted(h) for h in [headline1_generic, headline2_generic, headline3_generic]) and \
       not any(check_restricted(d) for d in [description1_generic, description2_generic]):
        ad_examples.append({
            "campaign_name": "Bybit_Brand_Global",
            "ad_group": "General_Crypto_Traders",
            "headlines": [headline1_generic, headline2_generic, headline3_generic],
            "descriptions": [description1_generic, description2_generic],
            "final_url_placeholder": affiliate_link_placeholder,
            "display_url": display_url_generic,
            "notes": "Generic brand awareness ad, emphasizing security and platform strength."
        })

    input_keyword_base = idea_or_feature.replace("Bybit's", "").replace("Understanding", "").replace("Exploring", "").strip().split(":")[0]
    input_keyword_display = input_keyword_base.title()

    headline1_feat = f"{input_keyword_display} {disclosure}"
    headline2_feat = f"Explore Features on Bybit"
    headline3_feat = f"Learn & Trade Now"
    description1_feat = f"Discover {idea_or_feature} with Bybit's easy-to-use platform. Get started in minutes!"
    description2_feat = f"Access powerful tools, guides, and support. {random.choice(keywords)}."
    display_url_feat = base_display_url_part + "/" + input_keyword_base.replace(" ", "-").capitalize()[:15]

    if not any(check_restricted(h) for h in [headline1_feat, headline2_feat, headline3_feat]) and \
       not any(check_restricted(d) for d in [description1_feat, description2_feat]):
        ad_examples.append({
            "campaign_name": f"Bybit_Feature_{input_keyword_base.replace(' ', '_')[:15]}",
            "ad_group": f"{input_keyword_base.replace(' ', '_')[:20]}_Prospecting",
            "headlines": [headline1_feat, headline2_feat, headline3_feat],
            "descriptions": [description1_feat, description2_feat],
            "final_url_placeholder": affiliate_link_placeholder,
            "display_url": display_url_feat,
            "notes": f"Ad focusing on specific feature/topic: {idea_or_feature}"
        })

    if not ad_examples:
        console.print("[yellow]Warning:[/yellow] No ad examples were generated. This might be due to all attempts containing restricted keywords.")
        return {"error": "No valid ad examples generated, possibly due to compliance restrictions.", "ads": [], "api_simulation": ""}

    api_call_simulation = "\n    # --- SIMULATED GOOGLE ADS API CALL ---\n    # ... (simulation details) ...\n    # --- END SIMULATION ---\n"
    return {"ads": ad_examples, "api_simulation": api_call_simulation}

def save_ad_copy_examples(ad_data):
    """Saves the generated ad copy examples to a file."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"""--- Ad Copy Examples Generated on {timestamp} ---
This file contains simulated ad copy based on templates and configuration.
It includes placeholders for where real API calls to Google Ads would occur.
Compliance checks (disclosures, restricted keywords) are simulated based on config.
"""
    header += "=" * 50 + "\n\n"

    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write(header)
            # Error string from generate_ad_copy is plain, so it's fine for file logging
            if ad_data.get("error") and not ad_data.get("ads"):
                f.write(f"Error generating ad copy: {ad_data['error']}\n")
                return

            if not ad_data.get("ads"):
                f.write("No ad examples were generated or provided to save.\n")
                if ad_data.get("error"):
                     f.write(f"Additional info: {ad_data['error']}\n")
            else:
                for i, ad_group in enumerate(ad_data.get("ads", [])):
                    f.write(f"**Ad Example {i+1}**\n")
                    f.write(f"  Campaign Name (Simulated): {ad_group['campaign_name']}\n")
                    f.write(f"  Ad Group (Simulated): {ad_group['ad_group']}\n")
                    f.write(f"  Headlines:\n")
                    for h_idx, h_val in enumerate(ad_group.get('headlines', [])):
                        f.write(f"    HL{h_idx+1}: {h_val}\n")
                    f.write(f"  Descriptions:\n")
                    for d_idx, d_val in enumerate(ad_group.get('descriptions', [])):
                        f.write(f"    Desc{d_idx+1}: {d_val}\n")
                    f.write(f"  Final URL (Placeholder): {ad_group.get('final_url_placeholder', 'N/A')}\n")
                    f.write(f"  Display URL (Simulated): {ad_group.get('display_url', 'N/A')}\n")
                    f.write(f"  Notes: {ad_group.get('notes', 'N/A')}\n\n")

            f.write("\n--- Google Ads API Call Simulation (Placeholder) ---\n")
            f.write(ad_data.get("api_simulation", "No API simulation block generated."))
            f.write("\n" + "=" * 50 + "\n")

        console.print(f"[green]Successfully saved ad copy examples to {OUTPUT_FILE}[/green]")
    except IOError as e:
        console.print(f"[red]Error saving ad copy examples to {OUTPUT_FILE}:[/red] {e}")

if __name__ == "__main__":
    console.print(Panel("Ad Copy Generator", title="[bold magenta]Agent Script[/bold magenta]", subtitle="[dim]Initializing...[/dim]"))

    try:
        import yaml
    except ImportError:
        console.print("[bold red]PyYAML not found. Attempting to install...[/bold red]")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
            console.print("[green]PyYAML installed successfully.[/green]")
            import yaml
        except subprocess.CalledProcessError as e:
            console.print(f"[bold red]Failed to install PyYAML:[/bold red] {e}. Please install it manually.")
            sys.exit(1)
        except Exception as e:
            console.print(f"[bold red]An error occurred during PyYAML import after installation:[/bold red] {e}")
            sys.exit(1)

    config_data = load_config()

    if config_data is None:
        console.print("[bold red]Exiting script due to critical configuration loading error.[/bold red]")
        exit(1)
    else:
        console.print("[green]Configuration loaded successfully.[/green]")
        ideas = load_content_ideas()
        if ideas and random.choice([True, False]):
            selected_input = random.choice(ideas)
            console.print(f"[blue]Generating ad copy based on a random content idea:[/blue] '[b]{selected_input}[/b]'")
        else:
            default_features = ["Bybit's Advanced Charting Tools", "Bybit Earn Passive Income", "Secure Crypto Storage on Bybit", "Bybit Launchpad Access"]
            selected_input = random.choice(default_features)
            console.print(f"[blue]Generating ad copy based on a default Bybit feature:[/blue] '[b]{selected_input}[/b]'")

        # generate_ad_copy returns plain error strings
        ad_copy_data = generate_ad_copy(selected_input, config_data)

        if ad_copy_data.get("error"):
            console.print(f"[bold red]AD COPY GENERATION ERROR:[/bold red] {ad_copy_data['error']}")
            # raw_output is not a key returned by the template-based generate_ad_copy
        else:
            console.print("[green]Ad copy generated successfully (template-based).[/green]")

        save_ad_copy_examples(ad_copy_data) # save_ad_copy_examples uses console.print for its status

        # LLM-specific part from instructions - adapting for template-based
        # This section is for the case where LLM was intended.
        # Since current script is template-based, these specific messages might not all apply.
        # I'll keep the structure but adapt messages.
        gemini_api_key_name = config_data.get('gemini_api_key_env_var', "GEMINI_API_KEY")
        api_key = os.environ.get(gemini_api_key_name) # Check for key, even if not used by current ad generator

        if not api_key:
            console.print(f"[yellow]Info:[/yellow] Gemini API Key ([b]{gemini_api_key_name}[/b]) not found. LLM-based ad copy generation (if available in future) would be skipped.")
            console.print("[blue]To enable potential LLM features, run 'python ai_marketing_agent/scripts/setup_env.py' or set the env var manually.[/blue]")
        else:
            console.print(f"[green]Info:[/green] Gemini API Key ([b]{gemini_api_key_name}[/b]) found. (Note: Current ad generation is template-based).")


    console.print(Panel("[bold green]Ad Copy Generator script finished.[/bold green]",padding=(1,2)))
