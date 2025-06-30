from duckduckgo_search import DDGS
from crewai_tools import SerperDevTool # Keep for Serper function
import os

# --- DuckDuckGo Search Function for Swarms ---
def duckduckgo_search_function(query: str, max_results: int = 7) -> str:
    """
    Searches DuckDuckGo for the given query and returns a formatted string of up to max_results.
    Each result includes a title, href (link), and body (snippet).
    This tool is useful for general web research and information gathering.
    Args:
        query (str): The search query string.
        max_results (int): The maximum number of search results to return. Defaults to 7.
    Returns:
        str: A formatted string of search results, or an error message/no results message.
    """
    print(f"[Tool: DuckDuckGoSearch] Searching for: '{query}', max_results: {max_results}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found for your query using DuckDuckGo."

        formatted_results = []
        for i, res in enumerate(results):
            formatted_results.append(
                f"Result {i+1}:\n"
                f"  Title: {res.get('title')}\n"
                f"  Link: {res.get('href')}\n"
                f"  Snippet: {res.get('body')}\n---"
            )
        return "\n".join(formatted_results)
    except Exception as e:
        print(f"Error during DuckDuckGo search: {e}")
        return f"Error performing DuckDuckGo search: {str(e)}"

# --- Serper Search Function for Swarms ---
def serper_search_function(query: str) -> str:
    """
    Performs a web search using the Serper API for the given query.
    This tool is useful for targeted web research when a Serper API key is available.
    It typically provides concise and relevant search results.
    Args:
        query (str): The search query string.
    Returns:
        str: A string containing the search results, or an error message.
    """
    print(f"[Tool: SerperSearch] Searching for: '{query}'")
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not found in environment variables. Cannot use Serper search."
    try:
        # SerperDevTool from crewai_tools can be instantiated and its run method used.
        # The tool itself handles the API call.
        tool = SerperDevTool(api_key=api_key)
        return tool.run(query=query) # Pass query as a keyword argument
    except Exception as e:
        print(f"Error during Serper search: {e}")
        return f"Error performing Serper search: {str(e)}"

if __name__ == '__main__':
    print("--- Testing Search Tools for Swarms ---")

    print("\n--- Testing duckduckgo_search_function ---")
    ddg_query = "latest trends in renewable energy"
    ddg_results = duckduckgo_search_function(ddg_query, max_results=3)
    print(f"Results for '{ddg_query}':\n{ddg_results}")

    print("\n--- Testing serper_search_function ---")
    # To test serper_search_function, ensure SERPER_API_KEY is set in your .env file
    # and that you have run `from dotenv import load_dotenv; load_dotenv()` beforehand,
    # or set it directly as an environment variable.
    from dotenv import load_dotenv
    load_dotenv() # Load .env for this test script

    serper_query = "future of multi-agent AI systems"
    if os.getenv("SERPER_API_KEY"):
        serper_results = serper_search_function(serper_query)
        print(f"Results for '{serper_query}':\n{serper_results}")
    else:
        print(f"Skipping Serper test as SERPER_API_KEY is not set.")

    print("\n--- Search Tools Test Complete ---")
