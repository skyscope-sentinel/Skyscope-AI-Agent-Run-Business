from crewai_tools import BaseTool
from duckduckgo_search import DDGS

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "A tool that uses DuckDuckGo to search the internet for information on a given query. "
        "Provides a list of search results with snippets, titles, and links."
    )

    def _run(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """
        Executes a search query using DuckDuckGo.

        Args:
            query (str): The search query string.
            max_results (int): The maximum number of results to return. Defaults to 5.

        Returns:
            list[dict[str, str]]: A list of search results, where each result is a dictionary
                                 containing 'title', 'href', and 'body' (snippet).
                                 Returns an empty list if an error occurs or no results are found.
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            # Ensure results are in the format expected (e.g., by a summarizing agent)
            # DDGS().text returns a list of dicts with 'title', 'href', 'body'
            # which is already a good format.
            if not results:
                return "No results found for the query."
            return results
        except Exception as e:
            print(f"Error during DuckDuckGo search: {e}")
            return f"Error performing search: {str(e)}"

if __name__ == '__main__':
    # Test the tool
    ddg_search_tool = DuckDuckGoSearchTool()

    print("--- Testing DuckDuckGoSearchTool ---")

    search_query_1 = "latest advancements in AI agents"
    print(f"\nSearching for: '{search_query_1}'")
    results_1 = ddg_search_tool.run(search_query_1)
    if isinstance(results_1, str): # Error message
        print(results_1)
    else:
        for i, res in enumerate(results_1):
            print(f"Result {i+1}:")
            print(f"  Title: {res.get('title')}")
            print(f"  Link: {res.get('href')}")
            print(f"  Snippet: {res.get('body')[:150]}...") # Print first 150 chars of snippet

    search_query_2 = "who is the founder of Skyscope Sentinel Intelligence" # Should be a bit meta
    print(f"\nSearching for: '{search_query_2}' (max 2 results)")
    results_2 = ddg_search_tool.run(query=search_query_2, max_results=2)
    if isinstance(results_2, str):
        print(results_2)
    else:
        for i, res in enumerate(results_2):
            print(f"Result {i+1}:")
            print(f"  Title: {res.get('title')}")
            print(f"  Link: {res.get('href')}")
            print(f"  Snippet: {res.get('body')[:150]}...")

    search_query_3 = "nonexistent gibberish query for testing no results"
    print(f"\nSearching for: '{search_query_3}'")
    results_3 = ddg_search_tool.run(query=search_query_3)
    print(f"Results for gibberish query: {results_3}")

    print("\n--- DuckDuckGoSearchTool Test Complete ---")
