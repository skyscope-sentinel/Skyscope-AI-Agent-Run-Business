import os
import json
import logging
import time
import random
from enum import Enum
from typing import Dict, List, Optional, Any, Literal
from uuid import uuid4

# Configure logging for clear and informative output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enums and Data Structures ---

class AffiliateNetwork(Enum):
    """Enumeration of supported affiliate networks."""
    AMAZON_ASSOCIATES = "Amazon Associates"
    CLICKBANK = "ClickBank"
    SHAREASALE = "ShareASale"

class PlatformType(Enum):
    """Enumeration of platforms where links can be posted."""
    SOCIAL_MEDIA_POST = "Social Media Post"
    BLOG_ARTICLE = "Blog Article"
    FORUM_SIGNATURE = "Forum Signature"
    WEBSITE_SUBMISSION = "Website Directory Submission"

class ReferralLink:
    """A data class to represent and track an affiliate link."""
    def __init__(self, product_id: str, product_name: str, base_url: str, network: AffiliateNetwork):
        self.link_id: str = str(uuid4())
        self.product_id = product_id
        self.product_name = product_name
        self.base_url = base_url
        self.network = network
        self.campaigns: Dict[str, str] = {} # campaign_name -> full_tracking_url
        self.performance: Dict[str, Dict[str, Any]] = {} # campaign_name -> metrics

    def generate_tracking_url(self, campaign_name: str) -> str:
        """Generates a unique tracking URL for a campaign."""
        tracking_id = f"skyscope-{campaign_name}-{int(time.time())}"
        url = f"{self.base_url}?tag={tracking_id}"
        self.campaigns[campaign_name] = url
        self.performance[campaign_name] = {"clicks": 0, "conversions": 0, "revenue": 0.0}
        return url

# --- Mock/Placeholder Classes for Simulating External Interactions ---

class MockAgent:
    """A mock AI agent to simulate SEO and content generation."""
    def run(self, task: str) -> str:
        logger.info(f"MockAgent received task: {task[:100]}...")
        if "generate SEO keywords" in task:
            return json.dumps([
                "best budget drone", "drone photography", "4k drone review",
                "beginner drone", "quadcopter deals"
            ])
        if "generate a short, punchy social media post" in task:
            return "Looking for the best budget 4K drone? 🚁 The new Aero-X is a game-changer! Unbelievable stability and video quality for the price. Check it out! #drone #techtuesday #gadget"
        if "generate a detailed blog post" in task:
            return """
# The Aero-X Drone: The Only Budget 4K Drone You Should Consider in 2025

When it comes to value, the Aero-X drone is in a class of its own. We spent weeks testing its capabilities, from its 30-minute flight time to its crystal-clear 4K camera.

## Key Features
- **4K Video:** Stunning, stable footage.
- **Beginner Friendly:** Easy-to-use controls.

For anyone entering the world of drone photography, this is a must-buy.
"""
        return "Default mock content."

class MockAffiliateNetworkClient:
    """Simulates an API client for an affiliate network."""
    def __init__(self, network: AffiliateNetwork, api_key: str):
        if not api_key:
            raise ValueError("API key is required.")
        self.network = network
        logger.info(f"Mock client for {network.value} initialized.")

    def get_product_link(self, product_id: str) -> str:
        logger.info(f"Fetching product link for '{product_id}' from {self.network.value}.")
        return f"https://www.example-store.com/products/{product_id}"

    def get_revenue_data(self, tracking_id: str) -> Dict[str, Any]:
        """Simulates fetching performance data for a specific tracking ID."""
        clicks = random.randint(500, 2000)
        conversions = int(clicks * random.uniform(0.01, 0.05)) # 1-5% conversion rate
        revenue = conversions * random.uniform(5.0, 25.0) # $5-25 commission per conversion
        return {"clicks": clicks, "conversions": conversions, "revenue": round(revenue, 2)}

class MockPlatformApiClient:
    """Simulates an API client for a content platform (social media, blog, etc.)."""
    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        logger.info(f"Mock API client for '{platform_name}' initialized.")

    def post(self, content: str, link: str, keywords: List[str]) -> bool:
        logger.info(f"--- Posting to {self.platform_name} ---")
        logger.info(f"Content: {content[:80]}...")
        logger.info(f"Link: {link}")
        logger.info(f"SEO Keywords: {keywords}")
        logger.info("Post successful (simulated).")
        return True

# --- Main Affiliate Marketing Manager Class ---

class AffiliateMarketingManager:
    """
    Orchestrates autonomous affiliate marketing campaigns, from content creation to
    performance tracking and optimization.
    """

    def __init__(self, agent: Any, api_credentials: Dict[str, str]):
        self.agent = agent
        self.api_credentials = api_credentials
        self.links: Dict[str, ReferralLink] = {}
        self.network_clients: Dict[AffiliateNetwork, MockAffiliateNetworkClient] = {
            network: MockAffiliateNetworkClient(network, api_credentials.get(network.name, ""))
            for network in AffiliateNetwork
        }
        self.platform_clients = {
            "Twitter": MockPlatformApiClient("Twitter"),
            "TechBlog": MockPlatformApiClient("TechBlog.com"),
            "DroneForum": MockPlatformApiClient("DroneEnthusiastForums.com")
        }

    def register_affiliate_product(self, product_id: str, product_name: str, network: AffiliateNetwork) -> Optional[ReferralLink]:
        """
        Registers a new product to be marketed.

        Args:
            product_id (str): The product's unique ID from the affiliate network.
            product_name (str): The name of the product.
            network (AffiliateNetwork): The affiliate network the product belongs to.

        Returns:
            The created ReferralLink object, or None on failure.
        """
        logger.info(f"Registering new affiliate product: '{product_name}'")
        client = self.network_clients.get(network)
        if not client:
            logger.error(f"No client configured for network {network.value}.")
            return None
        
        base_url = client.get_product_link(product_id)
        if not base_url:
            logger.error(f"Could not retrieve base URL for product '{product_id}'.")
            return None
            
        link = ReferralLink(product_id, product_name, base_url, network)
        self.links[link.link_id] = link
        logger.info(f"Product '{product_name}' registered with Link ID: {link.link_id}")
        return link

    def generate_seo_keywords(self, product_name: str, count: int = 5) -> List[str]:
        """Generates a list of optimized SEO keywords for a product."""
        prompt = f"generate {count} SEO keywords for an affiliate marketing campaign about '{product_name}'."
        try:
            response = self.agent.run(prompt)
            return json.loads(response)
        except (json.JSONDecodeError, TypeError):
            logger.error("Failed to generate or parse SEO keywords.")
            return [product_name.lower().replace(" ", "-")]

    def create_promotional_content(self, product_name: str, keywords: List[str], platform_type: PlatformType) -> str:
        """Generates tailored promotional content for a specific platform type."""
        if platform_type == PlatformType.SOCIAL_MEDIA_POST:
            prompt = f"generate a short, punchy social media post about '{product_name}' using keywords like {keywords}."
        elif platform_type == PlatformType.BLOG_ARTICLE:
            prompt = f"generate a detailed blog post review for '{product_name}', naturally incorporating these keywords: {keywords}."
        else:
            prompt = f"generate a compelling promotional message for '{product_name}'."
        
        return self.agent.run(prompt)

    def run_campaign(self, link_id: str, campaign_name: str):
        """
        Runs a full, multi-platform marketing campaign for a registered product.
        """
        if link_id not in self.links:
            logger.error(f"Link ID '{link_id}' not found.")
            return
        
        link = self.links[link_id]
        logger.info(f"--- Launching Campaign '{campaign_name}' for '{link.product_name}' ---")

        # 1. Generate tracking URL
        tracking_url = link.generate_tracking_url(campaign_name)
        
        # 2. Generate Keywords
        keywords = self.generate_seo_keywords(link.product_name)
        
        # 3. Create and Post Content for each platform
        # Social Media Post
        social_content = self.create_promotional_content(link.product_name, keywords, PlatformType.SOCIAL_MEDIA_POST)
        self.platform_clients["Twitter"].post(social_content, tracking_url, keywords)
        
        # Blog Post
        blog_content = self.create_promotional_content(link.product_name, keywords, PlatformType.BLOG_ARTICLE)
        self.platform_clients["TechBlog"].post(blog_content, tracking_url, keywords)

        logger.info(f"Campaign '{campaign_name}' has been successfully executed across all platforms.")

    def update_all_performance_data(self):
        """
        Iterates through all campaigns and updates their performance metrics.
        """
        logger.info("--- Updating Performance Data for All Campaigns ---")
        for link in self.links.values():
            client = self.network_clients.get(link.network)
            if not client:
                continue
            
            for campaign_name in link.campaigns.keys():
                # In a real system, you'd use the tracking ID from the URL
                tracking_id = link.campaigns[campaign_name].split("tag=")[1]
                performance_data = client.get_revenue_data(tracking_id)
                link.performance[campaign_name] = performance_data
                logger.info(f"Updated performance for '{link.product_name}' -> '{campaign_name}': {performance_data}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Summarizes performance across all products and campaigns."""
        summary = {
            "total_revenue": 0.0,
            "total_clicks": 0,
            "total_conversions": 0,
            "best_performing_product": None,
            "product_breakdown": {}
        }
        max_revenue = -1

        for link in self.links.values():
            product_revenue = 0
            for perf in link.performance.values():
                product_revenue += perf.get("revenue", 0)
                summary["total_clicks"] += perf.get("clicks", 0)
                summary["total_conversions"] += perf.get("conversions", 0)
            
            summary["product_breakdown"][link.product_name] = product_revenue
            summary["total_revenue"] += product_revenue

            if product_revenue > max_revenue:
                max_revenue = product_revenue
                summary["best_performing_product"] = link.product_name
        
        return summary


if __name__ == '__main__':
    logger.info("--- AffiliateMarketingManager Demonstration ---")
    
    # 1. Setup
    mock_agent = MockAgent()
    api_keys = {
        "AMAZON_ASSOCIATES": "amazon_api_key_secret",
        "CLICKBANK": "clickbank_api_key_secret"
    }
    manager = AffiliateMarketingManager(agent=mock_agent, api_credentials=api_keys)
    
    # 2. Register a product
    drone_link = manager.register_affiliate_product(
        product_id="B09XYZ123",
        product_name="Aero-X 4K Drone",
        network=AffiliateNetwork.AMAZON_ASSOCIATES
    )
    
    if drone_link:
        # 3. Run an autonomous campaign
        manager.run_campaign(drone_link.link_id, "Q3_Summer_Sale")
        
        # 4. Simulate time passing and update performance
        logger.info("\n--- Simulating time passing before checking performance... ---")
        time.sleep(1)
        manager.update_all_performance_data()
        
        # 5. Get a final summary
        logger.info("\n--- Generating Final Performance Summary ---")
        final_summary = manager.get_performance_summary()
        print(json.dumps(final_summary, indent=2))
        
    logger.info("\n--- Demonstration Finished ---")
