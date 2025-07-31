import asyncio
import os
from typing import Any, Dict, List, Optional

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeoutError,
)

class BrowserAutomator:
    """
    A class to encapsulate browser automation functionality using Playwright.

    This class provides a high-level API for launching, controlling, and closing
    a web browser, as well as performing common actions like navigation,
    clicking, typing, and data extraction.
    """

    def __init__(self):
        """Initializes the BrowserAutomator instance."""
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def launch(
        self,
        headless: bool = False,
        user_data_dir: Optional[str] = None,
        browser_type: str = "chromium",
    ) -> None:
        """
        Launches the browser and creates a new page.

        Args:
            headless (bool): Whether to run the browser in headless mode. Defaults to False.
            user_data_dir (Optional[str]): Path to a directory to use for the browser user profile.
                                           This allows for persistent sessions (cookies, logins).
                                           If None, a temporary profile is used.
            browser_type (str): The type of browser to launch ('chromium', 'firefox', 'webkit').
                                Defaults to 'chromium'.
        """
        if self.is_running():
            print("Browser is already running.")
            return

        self.playwright = await async_playwright().start()
        browser_launcher = getattr(self.playwright, browser_type)

        if user_data_dir:
            self.context = await browser_launcher.launch_persistent_context(
                user_data_dir,
                headless=headless,
            )
            self.browser = self.context.browser
        else:
            self.browser = await browser_launcher.launch(headless=headless)
            self.context = await self.browser.new_context()

        self.page = await self.context.new_page()
        print(f"{browser_type.capitalize()} browser launched successfully.")

    def is_running(self) -> bool:
        """Checks if the browser is currently running and connected."""
        return self.browser is not None and self.browser.is_connected()

    async def close(self) -> None:
        """Closes the browser and cleans up resources."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        print("Browser closed and resources cleaned up.")

    async def navigate(self, url: str, timeout: int = 60000) -> None:
        """
        Navigates to a specified URL.

        Args:
            url (str): The URL to navigate to.
            timeout (int): Maximum time in milliseconds to wait for navigation.
        
        Raises:
            ConnectionError: If the browser is not running.
            PlaywrightTimeoutError: If the navigation times out.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        print(f"Navigating to {url}...")
        await self.page.goto(url, timeout=timeout)
        print("Navigation successful.")

    async def click(self, selector: str, timeout: int = 30000) -> None:
        """
        Clicks an element on the page identified by a selector.

        Args:
            selector (str): A CSS selector or text to identify the element.
            timeout (int): Maximum time in milliseconds to wait for the element.
        
        Raises:
            ConnectionError: If the browser is not running.
            PlaywrightTimeoutError: If the element is not found within the timeout.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        print(f"Attempting to click element with selector: '{selector}'")
        # Playwright can auto-detect if it's a CSS selector or text selector
        await self.page.click(selector, timeout=timeout)
        print("Click successful.")

    async def type(self, selector: str, text: str, delay: int = 50, timeout: int = 30000) -> None:
        """
        Types text into an input field identified by a selector.

        Args:
            selector (str): A CSS selector to identify the input field.
            text (str): The text to type into the field.
            delay (int): Delay between keystrokes in milliseconds.
            timeout (int): Maximum time in milliseconds to wait for the element.
        
        Raises:
            ConnectionError: If the browser is not running.
            PlaywrightTimeoutError: If the element is not found within the timeout.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        print(f"Typing text into element with selector: '{selector}'")
        await self.page.fill(selector, text, timeout=timeout)
        print("Typing successful.")

    async def get_text(self, selector: str, timeout: int = 30000) -> str:
        """
        Extracts the text content from an element.

        Args:
            selector (str): A CSS selector to identify the element.
            timeout (int): Maximum time in milliseconds to wait for the element.

        Returns:
            The text content of the element.
            
        Raises:
            ConnectionError: If the browser is not running.
            PlaywrightTimeoutError: If the element is not found within the timeout.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        element = await self.page.wait_for_selector(selector, timeout=timeout)
        return await element.inner_text()

    async def get_html(self, selector: str, timeout: int = 30000) -> str:
        """
        Extracts the inner HTML from an element.

        Args:
            selector (str): A CSS selector to identify the element.
            timeout (int): Maximum time in milliseconds to wait for the element.

        Returns:
            The inner HTML of the element.
            
        Raises:
            ConnectionError: If the browser is not running.
            PlaywrightTimeoutError: If the element is not found within the timeout.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        element = await self.page.wait_for_selector(selector, timeout=timeout)
        return await element.inner_html()

    async def get_page_content(self) -> str:
        """
        Gets the full HTML content of the current page.

        Returns:
            The HTML content of the page.
            
        Raises:
            ConnectionError: If the browser is not running.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        return await self.page.content()

    async def screenshot(self, path: str, full_page: bool = True) -> None:
        """
        Takes a screenshot of the current page.

        Args:
            path (str): The file path to save the screenshot to.
            full_page (bool): Whether to capture the full scrollable page.
        
        Raises:
            ConnectionError: If the browser is not running.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        await self.page.screenshot(path=path, full_page=full_page)
        print(f"Screenshot saved to {path}")

    async def wait_for_selector(self, selector: str, timeout: int = 30000) -> None:
        """
        Waits for an element to appear on the page.

        Args:
            selector (str): The CSS selector to wait for.
            timeout (int): Maximum time in milliseconds to wait.
            
        Raises:
            ConnectionError: If the browser is not running.
            PlaywrightTimeoutError: If the element does not appear within the timeout.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        print(f"Waiting for selector: '{selector}'")
        await self.page.wait_for_selector(selector, timeout=timeout)
        print("Selector found.")

    async def scroll(self, direction: str, pixels: int = 500) -> None:
        """
        Scrolls the page up or down.

        Args:
            direction (str): The direction to scroll ('up' or 'down').
            pixels (int): The number of pixels to scroll.
            
        Raises:
            ConnectionError: If the browser is not running.
            ValueError: If an invalid direction is provided.
        """
        if not self.page:
            raise ConnectionError("Browser is not running. Please call launch() first.")
        
        if direction not in ['up', 'down']:
            raise ValueError("Direction must be 'up' or 'down'.")
            
        scroll_y = pixels if direction == 'down' else -pixels
        await self.page.evaluate(f"window.scrollBy(0, {scroll_y})")
        print(f"Scrolled {direction} by {pixels} pixels.")


async def main():
    """Main function to demonstrate BrowserAutomator usage."""
    automator = BrowserAutomator()
    try:
        # Launch the browser
        await automator.launch()

        # Navigate to a website
        await automator.navigate("https://github.com/kyegomez/swarms")

        # Take a screenshot
        await automator.screenshot("swarms_github_page.png")

        # Get the text of the repository description
        description_selector = ".f4.my-3"
        await automator.wait_for_selector(description_selector)
        description = await automator.get_text(description_selector)
        print(f"\nRepository Description: {description.strip()}")

        # Type into the search bar
        search_selector = 'input[data-testid="search-input"]'
        await automator.click(search_selector)
        await automator.type(search_selector, "agent")

        # Take another screenshot after typing
        await automator.screenshot("swarms_search_typed.png")
        
        # Get page content for AI analysis
        # content = await automator.get_page_content()
        # print(f"\nPage content length: {len(content)} characters")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure the browser is closed
        await automator.close()


if __name__ == "__main__":
    asyncio.run(main())
