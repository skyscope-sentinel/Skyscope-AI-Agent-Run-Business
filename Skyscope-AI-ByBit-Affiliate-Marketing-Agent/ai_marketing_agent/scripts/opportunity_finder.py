import os
import yaml
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from datetime import datetime

# Rich imports for CLI output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/settings.yaml')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../generated_content')
OPPORTUNITIES_FILE = os.path.join(OUTPUT_DIR, f"potential_posting_opportunities_{datetime.now().strftime('%Y%m%d')}.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_settings():
    """Loads settings from the YAML configuration file."""
    if not os.path.exists(CONFIG_PATH):
        console.print(f"[bold red]ERROR:[/bold red] Configuration file not found at {CONFIG_PATH}. Cannot proceed.")
        return None
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            console.print(f"[bold red]ERROR:[/bold red] Configuration file {CONFIG_PATH} is empty or malformed.")
            return None
        # Add a section for opportunity_finder_queries in settings.yaml if you want to make it configurable
        # For now, we'll use hardcoded queries or keywords from existing config.
        return config
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] Error loading configuration from {CONFIG_PATH}: {e}")
        return None

def search_google(query, num_results=10):
    """
    Performs a Google search and returns a list of URLs.
    Note: Web scraping Google is fragile and may be blocked.
          Using a proper API (e.g., Google Custom Search JSON API) is recommended for production.
          This is a simplified version for conceptual purposes.
    """
    console.print(f"[blue]INFO:[/blue] Searching Google for: '{query}' (first {num_results} results)")
    search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    }

    urls = []
    try:
        with console.status(f"[b blue]Fetching search results for '{query}'...[/b blue]", spinner="earth"):
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')

        link_tags = soup.find_all('a')
        found_count = 0
        for link_tag in link_tags:
            href = link_tag.get('href')
            if href and href.startswith("/url?q="):
                actual_url = href.split("/url?q=")[1].split("&sa=")[0]
                if actual_url.startswith("http") and "google.com" not in actual_url:
                    urls.append(actual_url)
                    found_count += 1
                    if found_count >= num_results:
                        break
            elif href and href.startswith("http") and "google.com" not in href and link_tag.h3:
                urls.append(href)
                found_count +=1
                if found_count >= num_results:
                    break

        if not urls:
            for item in soup.find_all('div', attrs={'class': 'g'}):
                link_tag = item.find('a', href=True)
                if link_tag and link_tag['href'].startswith("http") and "google.com" not in link_tag['href']:
                    urls.append(link_tag['href'])
                    if len(urls) >= num_results:
                        break

        console.print(f"[green]SUCCESS:[/green] Found {len(urls)} potential URLs for '{query}'.")
    except requests.exceptions.HTTPError as http_err:
        console.print(f"[bold red]HTTP ERROR:[/bold red] Could not fetch search results for '{query}': {http_err.response.status_code} - {http_err}")
        if http_err.response.status_code == 429:
            console.print("[yellow]WARN:[/yellow] Received a 429 (Too Many Requests) error. Google may be rate-limiting. Try again later or reduce search frequency.")
    except requests.exceptions.RequestException as e:
        console.print(f"[bold red]REQUEST ERROR:[/bold red] Could not fetch search results for '{query}': {e}")
    except Exception as e:
        console.print(f"[bold red]PARSING ERROR:[/bold red] Error parsing search results for '{query}': {e}")

    return list(set(urls)) # Return unique URLs

def filter_and_analyze_urls(urls, keywords_to_check=None):
    """
    Basic filter for URLs (e.g., looking for 'blog', 'forum' in URL or title - title fetching not done here for simplicity).
    This is a very basic filter. More advanced analysis would be needed.
    """
    if keywords_to_check is None:
        keywords_to_check = ["blog", "forum", "community", "guest-post", "write-for-us", "submit-article", "discussion"]

    filtered_urls = []
    for url in urls:
        if any(keyword in url.lower() for keyword in keywords_to_check):
            filtered_urls.append(url)

    console.print(f"[blue]INFO:[/blue] Filtered down to {len(filtered_urls)} URLs based on keywords: {', '.join(keywords_to_check)}")
    return filtered_urls

def save_opportunities(opportunities_map):
    """Saves the found opportunities to a text file."""
    try:
        with open(OPPORTUNITIES_FILE, 'w') as f:
            f.write(f"Potential Posting Opportunities Found on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            if not opportunities_map:
                f.write("No opportunities found in this run.\n")
            for query, urls in opportunities_map.items():
                f.write(f"Search Query: {query}\n")
                f.write("-" * 30 + "\n")
                if urls:
                    for url in urls:
                        f.write(f"- {url}\n")
                else:
                    f.write("  No relevant URLs found for this query.\n")
                f.write("\n")
        console.print(f"[green]SUCCESS:[/green] Saved potential opportunities to [link=file://{os.path.abspath(OPPORTUNITIES_FILE)}]{OPPORTUNITIES_FILE}[/link]")
    except IOError as e:
        console.print(f"[bold red]IO ERROR:[/bold red] Could not save opportunities file: {e}")

if __name__ == "__main__":
    console.print(Panel("Online Opportunity Finder (Conceptual)", title="[bold magenta]Agent Script[/bold magenta]", subtitle="[dim]Initializing...[/dim]"))
    settings = load_settings()

    if not settings:
        console.print("[bold red]FATAL:[/bold red] Exiting due to configuration loading failure.")
    else:
        console.print("[green]INFO:[/green] Configuration loaded successfully.")

        target_keywords = settings.get('target_keywords', ["crypto", "Bitcoin"])
        primary_keyword = target_keywords[0] if target_keywords else "cryptocurrency"

        search_queries = [
            f"{primary_keyword} blogs accepting guest posts",
            f"{primary_keyword} forums community",
            f"write for us {primary_keyword}",
            f"best {primary_keyword} blogs to read",
            "crypto news sites submit article"
        ]

        filter_keywords = ["blog", "forum", "community", "guest", "write", "submit", "article", "news", "discuss"]

        all_found_opportunities = {}

        console.print(f"[cyan]INFO:[/cyan] Starting online search for posting opportunities using {len(search_queries)} queries.")

        for query in search_queries:
            raw_urls = search_google(query, num_results=10)
            if raw_urls:
                filtered = filter_and_analyze_urls(raw_urls, filter_keywords)
                all_found_opportunities[query] = filtered
            else:
                all_found_opportunities[query] = []
            console.print("-" * 50)

        if all_found_opportunities and any(all_found_opportunities.values()): # Check if any query yielded results
            console.print(Panel("Search Results Summary", title="[bold blue]Opportunity Scan Complete[/bold blue]", expand=False))
            table = Table(title="Potential Posting Opportunities", show_lines=True)
            table.add_column("Search Query", style="cyan", overflow="fold", min_width=20)
            table.add_column("Found URLs", style="magenta", overflow="fold", min_width=50)

            for query, urls in all_found_opportunities.items():
                if urls:
                    table.add_row(query, "\n".join([f"- {url}" for url in urls]))
                else:
                    table.add_row(query, "[dim]No relevant URLs found.[/dim]")
            console.print(table)
            save_opportunities(all_found_opportunities)
        else:
            console.print("[yellow]WARN:[/yellow] No opportunities found across all search queries.")
            save_opportunities({})

    console.print(Panel("Opportunity Finder Script Finished", style="bold green"))
