import os
import csv
import random

# Rich library imports
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Rich Console
console = Console()

IDEAS_FILE = "ai_marketing_agent/generated_content/content_ideas.txt"
PERFORMANCE_DATA_FILE = "ai_marketing_agent/sim_data/simulated_performance_data.csv"
TRENDING_TOPICS_FILE = "ai_marketing_agent/sim_data/simulated_trending_topics.txt"
OUTPUT_CHOICE_FILE = "ai_marketing_agent/generated_content/next_article_to_generate.txt"

def load_ideas():
    try:
        with open(IDEAS_FILE, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if not lines: return []
            if "Content Ideas Generated on" in lines[0]:
                lines = lines[1:]
            lines = [line for line in lines if not all(c == '=' for c in line)]
            cleaned_ideas = []
            for line in lines:
                if line.startswith("- "):
                    cleaned_ideas.append(line[2:].strip())
                elif line:
                    cleaned_ideas.append(line)
            return [idea for idea in cleaned_ideas if idea]
    except FileNotFoundError:
        console.print(f"[yellow]Warning:[/yellow] Ideas file not found at {IDEAS_FILE}")
        return []
    except Exception as e:
        console.print(f"[red]Error loading ideas:[/red] {e}")
        return []

def load_performance_data():
    data = {}
    try:
        with open(PERFORMANCE_DATA_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[row['idea_title_exact_match']] = {
                    'views': int(row['sim_views']),
                    'ctr': float(row['sim_ctr']),
                    'conversions': int(row['sim_conversions'])
                }
    except FileNotFoundError:
        console.print(f"[yellow]Warning:[/yellow] Performance data file not found at {PERFORMANCE_DATA_FILE}")
        return {}
    except Exception as e:
        console.print(f"[red]Error loading performance data:[/red] {e}")
        return {}
    return data

def load_trending_topics():
    trends = {}
    try:
        with open(TRENDING_TOPICS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line: continue
                parts = line.split(',')
                if len(parts) == 2:
                    topic = parts[0].strip().replace('"', '')
                    try:
                        score = int(parts[1].strip())
                        trends[topic] = score
                    except ValueError:
                        console.print(f"[yellow]Warning:[/yellow] Could not parse trend score for: {line}")
                elif line:
                    console.print(f"[yellow]Warning:[/yellow] Malformed line in trending topics: {line}")
    except FileNotFoundError:
        console.print(f"[yellow]Warning:[/yellow] Trending topics file not found at {TRENDING_TOPICS_FILE}")
        return {}
    except Exception as e:
        console.print(f"[red]Error loading trending topics:[/red] {e}")
        return {}
    return trends

def choose_next_article(ideas, performance_data, trending_topics):
    if not ideas:
        return "Error: No content ideas available to choose from." # Plain error string

    scored_ideas = []
    for idea_text in ideas:
        score = 0
        perf = performance_data.get(idea_text)
        if perf:
            score += (30 - perf.get('conversions', 0)) * 0.5
            score += (10000 - perf.get('views', 0)) / 2000.0
        else:
            score += 15

        for topic, trend_score in trending_topics.items():
            if topic.lower() in idea_text.lower():
                score += trend_score * 2.5

        score += random.uniform(0, 8)
        scored_ideas.append((idea_text, round(score, 2)))

    scored_ideas.sort(key=lambda x: x[1], reverse=True)

    if scored_ideas:
        table = Table(title="[bold blue]Top Scored Content Ideas[/bold blue]", show_lines=True)
        table.add_column("Rank", style="dim cyan", no_wrap=True)
        table.add_column("Score", style="cyan", no_wrap=True)
        table.add_column("Idea", style="magenta")

        for i, (idea_text, idea_score) in enumerate(scored_ideas[:5]): # Show top 5
            table.add_row(str(i+1), f"{idea_score:.2f}", idea_text)
        console.print(table)
        return scored_ideas[0][0]

    console.print("[yellow]Warning:[/yellow] No scored ideas, picking randomly from available ideas.")
    return random.choice(ideas)

if __name__ == "__main__":
    console.print(Panel("Strategic Content Chooser", title="[bold magenta]Agent Script[/bold magenta]"))

    output_dir_for_choice = os.path.dirname(OUTPUT_CHOICE_FILE)
    if not os.path.exists(output_dir_for_choice):
        os.makedirs(output_dir_for_choice)
        console.print(f"[blue]Info:[/blue] Created directory: {output_dir_for_choice}")

    available_ideas = load_ideas()
    chosen_idea_to_write = ""

    if not available_ideas:
        console.print("[yellow]No content ideas found. Please generate ideas first (e.g., run content_idea_generator.py).[/yellow]")
        chosen_idea_to_write = "Generated Default Idea: What is Bybit and How to Use It?"
        console.print(f"[blue]Info:[/blue] Using dummy idea for {OUTPUT_CHOICE_FILE} as no ideas were found.")
    else:
        console.print(f"[blue]Info:[/blue] Loaded [b]{len(available_ideas)}[/b] ideas.")
        perf_data = load_performance_data()
        trends = load_trending_topics()

        console.print(f"[blue]Info:[/blue] Loaded [b]{len(perf_data)}[/b] performance records.")
        console.print(f"[blue]Info:[/blue] Loaded [b]{len(trends)}[/b] trending topics.")

        chosen_idea = choose_next_article(available_ideas, perf_data, trends)

        if "Error:" in chosen_idea: # Check for plain error string
            console.print(f"[bold red]{chosen_idea}[/bold red]")
            chosen_idea_to_write = "Error in choosing idea. Fallback: General Bybit Overview"
            console.print(f"[blue]Info:[/blue] Using fallback idea: {chosen_idea_to_write}")
        else:
            chosen_idea_to_write = chosen_idea
            console.print(f"[bold green]Strategic choice:[/bold green] '{chosen_idea_to_write}'")

    try:
        with open(OUTPUT_CHOICE_FILE, 'w') as f:
            f.write(chosen_idea_to_write)
        console.print(f"[blue]Info:[/blue] Chosen idea saved to {OUTPUT_CHOICE_FILE}")
    except IOError as e:
        console.print(f"[red]Error saving chosen idea to {OUTPUT_CHOICE_FILE}:[/red] {e}")
    console.print(Panel("Script Finished", title="[bold magenta]Agent Script[/bold magenta]",padding=(0,1)))
