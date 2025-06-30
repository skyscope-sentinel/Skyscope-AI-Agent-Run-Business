from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def browse_web_page_and_extract_text(url: str, timeout_seconds: int = 60) -> str:
    """
    Navigates to a given URL using a headless browser (Playwright with Chromium)
    and extracts the visible text content from the body of the page.

    Args:
        url (str): The URL to browse.
        timeout_seconds (int): The timeout in seconds for page navigation. Defaults to 60.

    Returns:
        str: The extracted text content of the page (limited to the first 10,000 characters),
             or an error message if navigation or content extraction fails.
    """
    print(f"[Tool: SimpleBrowser] Attempting to navigate to: {url}")
    try:
        with sync_playwright() as p:
            # Launch browser - consider adding arguments for stealth if needed later
            # e.g., user_agent string
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"
            ) # Example user agent
            page = context.new_page()

            page.goto(url, timeout=timeout_seconds * 1000) # Playwright timeout is in ms

            # Attempt to extract text from the main content area if possible,
            # otherwise fall back to body. This is a naive approach.
            # More sophisticated extraction would involve looking for <article>, <main>, etc.
            main_content = page.query_selector('article') or page.query_selector('main')
            if main_content:
                content = main_content.inner_text()
            else:
                content = page.inner_text('body')

            browser.close()

            if not content:
                return f"[Tool: SimpleBrowser] No text content found at {url}"

            # Limit content length to avoid overwhelming LLM context or output
            max_length = 10000
            return content[:max_length] + ("..." if len(content) > max_length else "")

    except PlaywrightTimeoutError:
        print(f"[Tool: SimpleBrowser] Timeout error while navigating to {url}")
        return f"Error: Timeout occurred while trying to load {url}"
    except Exception as e:
        print(f"[Tool: SimpleBrowser] Error during browser operation for {url}: {e}")
        return f"Error browsing {url}: {str(e)}"

if __name__ == '__main__':
    print("--- Testing SimpleBrowser Tool ---")

    test_url_1 = "https://example.com"
    print(f"\nFetching content from: {test_url_1}")
    content_1 = browse_web_page_and_extract_text(test_url_1)
    print(f"Content (first 500 chars):\n{content_1[:500]}\n---")

    # Test with a potentially more complex site (use a known simple blog post or news article for reliability in tests)
    # For now, using a placeholder that might be dynamic.
    test_url_2 = "https://www.wikipedia.org/wiki/Artificial_intelligence" # A well-structured site
    print(f"\nFetching content from: {test_url_2}")
    content_2 = browse_web_page_and_extract_text(test_url_2)
    print(f"Content (first 500 chars):\n{content_2[:500]}\n---")

    test_url_3 = "https://nonexistentsite.thisshouldfail"
    print(f"\nFetching content from: {test_url_3}")
    content_3 = browse_web_page_and_extract_text(test_url_3)
    print(f"Content:\n{content_3}\n---")

    print("--- SimpleBrowser Tool Test Complete ---")
    print("Note: Playwright will download browser binaries on first run if not already present.")
    print("This might take a few moments.")
