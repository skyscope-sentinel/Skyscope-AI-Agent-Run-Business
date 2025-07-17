import os
import sys
import json
import time
import uuid
import logging
import datetime
import requests
import asyncio
import aiohttp
import pandas as pd
import numpy as np
import re
import random
import string
import hashlib
import base64
import smtplib
import schedule
import threading
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import internal modules
try:
    from agent_manager import AgentManager
    from database_manager import DatabaseManager
    from live_thinking_rag_system import LiveThinkingRAGSystem
    from performance_monitor import PerformanceMonitor
    from enhanced_security_compliance import EncryptionManager, ComplianceManager
except ImportError:
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/content_marketing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("content_generation_marketing_system")

# Constants
CONFIG_DIR = Path("config")
CONTENT_DIR = Path("content")
TEMPLATES_DIR = Path("templates")
CAMPAIGNS_DIR = Path("campaigns")
ANALYTICS_DIR = Path("analytics")
MEDIA_DIR = Path("media")
LOGS_DIR = Path("logs")

# Ensure directories exist
for directory in [CONFIG_DIR, CONTENT_DIR, TEMPLATES_DIR, CAMPAIGNS_DIR, ANALYTICS_DIR, MEDIA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "content_marketing_config.json"

class ContentType(Enum):
    """Content types."""
    BLOG_POST = "blog_post"
    SOCIAL_MEDIA = "social_media"
    EMAIL = "email"
    LANDING_PAGE = "landing_page"
    PRODUCT_DESCRIPTION = "product_description"
    AD_COPY = "ad_copy"
    VIDEO_SCRIPT = "video_script"
    INFOGRAPHIC = "infographic"
    EBOOK = "ebook"
    WHITEPAPER = "whitepaper"
    CASE_STUDY = "case_study"
    PRESS_RELEASE = "press_release"
    NEWSLETTER = "newsletter"
    PODCAST_SCRIPT = "podcast_script"
    CUSTOM = "custom"

class MarketingChannel(Enum):
    """Marketing channels."""
    EMAIL = "email"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    PINTEREST = "pinterest"
    GOOGLE = "google"
    BLOG = "blog"
    WEBSITE = "website"
    PODCAST = "podcast"
    WEBINAR = "webinar"
    SMS = "sms"
    DIRECT_MAIL = "direct_mail"
    CUSTOM = "custom"

class CampaignType(Enum):
    """Campaign types."""
    AWARENESS = "awareness"
    ACQUISITION = "acquisition"
    CONVERSION = "conversion"
    RETENTION = "retention"
    LOYALTY = "loyalty"
    PRODUCT_LAUNCH = "product_launch"
    EVENT_PROMOTION = "event_promotion"
    SEASONAL = "seasonal"
    REACTIVATION = "reactivation"
    REFERRAL = "referral"
    CUSTOM = "custom"

class CampaignStatus(Enum):
    """Campaign status."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class ContentTone(Enum):
    """Content tone."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    HUMOROUS = "humorous"
    FORMAL = "formal"
    INSPIRATIONAL = "inspirational"
    EDUCATIONAL = "educational"
    PERSUASIVE = "persuasive"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CUSTOM = "custom"

@dataclass
class MarketingConfig:
    """Configuration for the marketing system."""
    brand_name: str = "Skyscope Sentinel"
    brand_description: str = "AI-powered business automation and intelligence"
    brand_voice: ContentTone = ContentTone.PROFESSIONAL
    primary_audience: List[str] = field(default_factory=lambda: ["business owners", "entrepreneurs", "tech enthusiasts"])
    secondary_audience: List[str] = field(default_factory=lambda: ["investors", "developers", "marketing professionals"])
    target_markets: List[str] = field(default_factory=lambda: ["Australia", "United States", "United Kingdom", "Canada"])
    primary_channels: List[MarketingChannel] = field(default_factory=lambda: [
        MarketingChannel.EMAIL, MarketingChannel.LINKEDIN, MarketingChannel.BLOG
    ])
    secondary_channels: List[MarketingChannel] = field(default_factory=lambda: [
        MarketingChannel.TWITTER, MarketingChannel.YOUTUBE, MarketingChannel.PODCAST
    ])
    content_calendar: Dict[str, List[ContentType]] = field(default_factory=lambda: {
        "monday": [ContentType.BLOG_POST, ContentType.SOCIAL_MEDIA],
        "wednesday": [ContentType.EMAIL, ContentType.SOCIAL_MEDIA],
        "friday": [ContentType.SOCIAL_MEDIA, ContentType.NEWSLETTER]
    })
    posting_frequency: Dict[MarketingChannel, int] = field(default_factory=lambda: {
        "email": 1,  # per week
        "linkedin": 3,  # per week
        "blog": 1,  # per week
        "twitter": 5,  # per week
        "youtube": 1,  # per week
        "podcast": 1  # per week
    })
    email_settings: Dict[str, Any] = field(default_factory=lambda: {
        "smtp_server": "smtp.example.com",
        "smtp_port": 587,
        "smtp_username": "",
        "smtp_password": "",
        "from_email": "marketing@example.com",
        "from_name": "Skyscope Sentinel",
        "reply_to": "support@example.com"
    })
    social_media_settings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    seo_keywords: List[str] = field(default_factory=lambda: [
        "AI business automation", "business intelligence", "AI agents", 
        "automated marketing", "business optimization", "AI trading"
    ])
    content_templates: Dict[str, str] = field(default_factory=dict)
    analytics_tracking: Dict[str, str] = field(default_factory=lambda: {
        "google_analytics": "",
        "facebook_pixel": "",
        "linkedin_insight": "",
        "twitter_pixel": ""
    })
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "brand_name": self.brand_name,
            "brand_description": self.brand_description,
            "brand_voice": self.brand_voice.value,
            "primary_audience": self.primary_audience,
            "secondary_audience": self.secondary_audience,
            "target_markets": self.target_markets,
            "primary_channels": [channel.value for channel in self.primary_channels],
            "secondary_channels": [channel.value for channel in self.secondary_channels],
            "content_calendar": {day: [content_type.value for content_type in content_types] 
                               for day, content_types in self.content_calendar.items()},
            "posting_frequency": self.posting_frequency,
            "email_settings": {
                **{k: v for k, v in self.email_settings.items() if k != "smtp_password"},
                "smtp_password": "********" if self.email_settings.get("smtp_password") else ""
            },
            "social_media_settings": {
                platform: {**{k: v for k, v in settings.items() if not k.endswith("_secret") and not k.endswith("_password")},
                           **{k: "********" for k, v in settings.items() if k.endswith("_secret") or k.endswith("_password")}}
                for platform, settings in self.social_media_settings.items()
            },
            "seo_keywords": self.seo_keywords,
            "content_templates": self.content_templates,
            "analytics_tracking": self.analytics_tracking,
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketingConfig':
        """Create from dictionary."""
        config = cls(
            brand_name=data.get("brand_name", "Skyscope Sentinel"),
            brand_description=data.get("brand_description", "AI-powered business automation and intelligence"),
            brand_voice=ContentTone(data.get("brand_voice", ContentTone.PROFESSIONAL.value)),
            primary_audience=data.get("primary_audience", ["business owners", "entrepreneurs", "tech enthusiasts"]),
            secondary_audience=data.get("secondary_audience", ["investors", "developers", "marketing professionals"]),
            target_markets=data.get("target_markets", ["Australia", "United States", "United Kingdom", "Canada"]),
            primary_channels=[MarketingChannel(channel) for channel in data.get("primary_channels", 
                                                                              [MarketingChannel.EMAIL.value, 
                                                                               MarketingChannel.LINKEDIN.value,
                                                                               MarketingChannel.BLOG.value])],
            secondary_channels=[MarketingChannel(channel) for channel in data.get("secondary_channels", 
                                                                                [MarketingChannel.TWITTER.value,
                                                                                 MarketingChannel.YOUTUBE.value,
                                                                                 MarketingChannel.PODCAST.value])],
            seo_keywords=data.get("seo_keywords", [
                "AI business automation", "business intelligence", "AI agents", 
                "automated marketing", "business optimization", "AI trading"
            ]),
            custom_settings=data.get("custom_settings", {})
        )
        
        # Handle content calendar
        if "content_calendar" in data:
            config.content_calendar = {
                day: [ContentType(content_type) for content_type in content_types]
                for day, content_types in data["content_calendar"].items()
            }
        
        # Handle posting frequency
        if "posting_frequency" in data:
            config.posting_frequency = data["posting_frequency"]
        
        # Handle email settings
        if "email_settings" in data:
            config.email_settings = data["email_settings"]
            
            # Try to load sensitive information from environment variables or secure storage
            if config.email_settings.get("smtp_password") == "********":
                config.email_settings["smtp_password"] = os.environ.get("MARKETING_SMTP_PASSWORD", "")
        
        # Handle social media settings
        if "social_media_settings" in data:
            config.social_media_settings = data["social_media_settings"]
            
            # Try to load sensitive information from environment variables or secure storage
            for platform, settings in config.social_media_settings.items():
                for key, value in settings.items():
                    if value == "********":
                        env_var = f"MARKETING_{platform.upper()}_{key.upper()}"
                        settings[key] = os.environ.get(env_var, "")
        
        # Handle content templates
        if "content_templates" in data:
            config.content_templates = data["content_templates"]
        
        # Handle analytics tracking
        if "analytics_tracking" in data:
            config.analytics_tracking = data["analytics_tracking"]
        
        return config
    
    def save(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to file."""
        try:
            # Ensure sensitive data is not saved to disk
            config_dict = self.to_dict()
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Marketing configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving marketing configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = DEFAULT_CONFIG_PATH) -> 'MarketingConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Marketing configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading marketing configuration: {e}")
            return cls()

@dataclass
class ContentItem:
    """Content item."""
    id: str
    title: str
    content_type: ContentType
    content: str
    tags: List[str]
    created_at: int
    updated_at: int
    author: str = "Skyscope Sentinel AI"
    status: str = "draft"
    channels: List[MarketingChannel] = field(default_factory=list)
    scheduled_at: Optional[int] = None
    published_at: Optional[int] = None
    url: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content_type": self.content_type.value,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "author": self.author,
            "status": self.status,
            "channels": [channel.value for channel in self.channels],
            "scheduled_at": self.scheduled_at,
            "published_at": self.published_at,
            "url": self.url,
            "metrics": self.metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentItem':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            title=data["title"],
            content_type=ContentType(data["content_type"]),
            content=data["content"],
            tags=data["tags"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            author=data.get("author", "Skyscope Sentinel AI"),
            status=data.get("status", "draft"),
            channels=[MarketingChannel(channel) for channel in data.get("channels", [])],
            scheduled_at=data.get("scheduled_at"),
            published_at=data.get("published_at"),
            url=data.get("url"),
            metrics=data.get("metrics", {}),
            metadata=data.get("metadata", {})
        )
    
    def save(self) -> None:
        """Save content item to file."""
        content_path = CONTENT_DIR / f"{self.id}.json"
        try:
            with open(content_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Content item saved to {content_path}")
        except Exception as e:
            logger.error(f"Error saving content item: {e}")

@dataclass
class Campaign:
    """Marketing campaign."""
    id: str
    name: str
    campaign_type: CampaignType
    description: str
    start_date: int
    end_date: int
    status: CampaignStatus
    channels: List[MarketingChannel]
    target_audience: List[str]
    content_items: List[str]  # Content item IDs
    budget: Optional[float] = None
    goals: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "campaign_type": self.campaign_type.value,
            "description": self.description,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status.value,
            "channels": [channel.value for channel in self.channels],
            "target_audience": self.target_audience,
            "content_items": self.content_items,
            "budget": self.budget,
            "goals": self.goals,
            "metrics": self.metrics,
            "schedule": self.schedule,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Campaign':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            campaign_type=CampaignType(data["campaign_type"]),
            description=data["description"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            status=CampaignStatus(data["status"]),
            channels=[MarketingChannel(channel) for channel in data["channels"]],
            target_audience=data["target_audience"],
            content_items=data["content_items"],
            budget=data.get("budget"),
            goals=data.get("goals", {}),
            metrics=data.get("metrics", {}),
            schedule=data.get("schedule", {}),
            metadata=data.get("metadata", {})
        )
    
    def save(self) -> None:
        """Save campaign to file."""
        campaign_path = CAMPAIGNS_DIR / f"{self.id}.json"
        try:
            with open(campaign_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Campaign saved to {campaign_path}")
        except Exception as e:
            logger.error(f"Error saving campaign: {e}")

@dataclass
class ContentTemplate:
    """Content template."""
    id: str
    name: str
    content_type: ContentType
    template: str
    variables: List[str]
    description: str
    created_at: int
    updated_at: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "content_type": self.content_type.value,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentTemplate':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            content_type=ContentType(data["content_type"]),
            template=data["template"],
            variables=data["variables"],
            description=data["description"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
    
    def save(self) -> None:
        """Save template to file."""
        template_path = TEMPLATES_DIR / f"{self.id}.json"
        try:
            with open(template_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Template saved to {template_path}")
        except Exception as e:
            logger.error(f"Error saving template: {e}")

class ContentGenerator:
    """AI-powered content generator."""
    
    def __init__(self, agent_manager: Optional[AgentManager] = None, 
                 rag_system: Optional[LiveThinkingRAGSystem] = None):
        """Initialize the content generator."""
        self.agent_manager = agent_manager
        self.rag_system = rag_system
        self.config = MarketingConfig.load()
        self.templates = self._load_templates()
        
        # Try to initialize agent manager if not provided
        if self.agent_manager is None:
            try:
                from agent_manager import AgentManager
                self.agent_manager = AgentManager()
            except ImportError:
                logger.warning("AgentManager not available. Some features will be limited.")
        
        # Try to initialize RAG system if not provided
        if self.rag_system is None:
            try:
                from live_thinking_rag_system import LiveThinkingRAGSystem
                self.rag_system = LiveThinkingRAGSystem()
            except ImportError:
                logger.warning("LiveThinkingRAGSystem not available. Some features will be limited.")
    
    def _load_templates(self) -> Dict[str, ContentTemplate]:
        """Load content templates."""
        templates = {}
        try:
            for template_file in TEMPLATES_DIR.glob("*.json"):
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                template = ContentTemplate.from_dict(template_data)
                templates[template.id] = template
            
            logger.info(f"Loaded {len(templates)} content templates")
            return templates
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            return {}
    
    def generate_content(self, content_type: ContentType, title: str, 
                        keywords: List[str] = None, tone: ContentTone = None,
                        length: str = "medium", template_id: str = None,
                        additional_context: Dict[str, Any] = None) -> ContentItem:
        """Generate content using AI."""
        try:
            # Set defaults
            if keywords is None:
                keywords = self.config.seo_keywords
            
            if tone is None:
                tone = self.config.brand_voice
            
            # Prepare prompt based on content type
            prompt = self._create_content_prompt(content_type, title, keywords, tone, length, additional_context)
            
            # Use template if provided
            template_content = None
            if template_id and template_id in self.templates:
                template = self.templates[template_id]
                template_content = template.template
                
                # Replace variables in template
                variables = additional_context or {}
                variables.update({
                    "title": title,
                    "keywords": ", ".join(keywords),
                    "tone": tone.value,
                    "brand_name": self.config.brand_name,
                    "brand_description": self.config.brand_description
                })
                
                for var_name, var_value in variables.items():
                    placeholder = f"{{{var_name}}}"
                    if placeholder in template_content:
                        template_content = template_content.replace(placeholder, str(var_value))
            
            # Generate content using agent manager if available
            content = ""
            if self.agent_manager:
                # Create specialized agent for content generation
                agent_prompt = f"""
                You are a professional content creator for {self.config.brand_name}.
                Brand description: {self.config.brand_description}
                Brand voice: {tone.value}
                
                Please create {content_type.value} content with the following specifications:
                Title: {title}
                Keywords to include: {', '.join(keywords)}
                Length: {length}
                
                {f'Based on this template: {template_content}' if template_content else ''}
                
                Additional context:
                {json.dumps(additional_context) if additional_context else 'None'}
                
                The content should be well-structured, engaging, and optimized for SEO.
                Include appropriate headings, paragraphs, and formatting.
                """
                
                content = self.agent_manager.execute_task(
                    task_description=f"Generate {content_type.value} content: {title}",
                    prompt=agent_prompt,
                    context=additional_context,
                    agent_persona="marketing_specialist"
                )
            elif self.rag_system:
                # Use RAG system as fallback
                rag_query = f"Create {content_type.value} content titled '{title}' in a {tone.value} tone, including keywords: {', '.join(keywords)}"
                content = self.rag_system.query(rag_query, additional_context)
            else:
                # Fallback to template or basic content
                if template_content:
                    content = template_content
                else:
                    content = f"# {title}\n\n"
                    content += f"[This is auto-generated {content_type.value} content for {self.config.brand_name}]\n\n"
                    content += f"Keywords: {', '.join(keywords)}\n\n"
                    content += "Please replace this with actual content."
            
            # Create content item
            content_item = ContentItem(
                id=str(uuid.uuid4()),
                title=title,
                content_type=content_type,
                content=content,
                tags=keywords,
                created_at=int(time.time()),
                updated_at=int(time.time()),
                author="Skyscope Sentinel AI",
                status="draft"
            )
            
            # Save content item
            content_item.save()
            
            return content_item
        
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            logger.error(traceback.format_exc())
            
            # Create error content item
            error_content = f"# Error Generating Content\n\n"
            error_content += f"Failed to generate {content_type.value} content for title: {title}\n\n"
            error_content += f"Error: {str(e)}"
            
            content_item = ContentItem(
                id=str(uuid.uuid4()),
                title=f"[ERROR] {title}",
                content_type=content_type,
                content=error_content,
                tags=keywords or [],
                created_at=int(time.time()),
                updated_at=int(time.time()),
                author="System",
                status="error"
            )
            
            return content_item
    
    def _create_content_prompt(self, content_type: ContentType, title: str,
                              keywords: List[str], tone: ContentTone,
                              length: str, additional_context: Dict[str, Any] = None) -> str:
        """Create prompt for content generation."""
        prompt_parts = [
            f"# Content Generation Request",
            f"",
            f"## Brand Information",
            f"- Brand Name: {self.config.brand_name}",
            f"- Brand Description: {self.config.brand_description}",
            f"- Brand Voice: {tone.value}",
            f"",
            f"## Content Specifications",
            f"- Type: {content_type.value}",
            f"- Title: {title}",
            f"- Keywords: {', '.join(keywords)}",
            f"- Length: {length}",
            f"",
            f"## Target Audience",
            f"- Primary: {', '.join(self.config.primary_audience)}",
            f"- Secondary: {', '.join(self.config.secondary_audience)}",
            f""
        ]
        
        # Add content type specific instructions
        if content_type == ContentType.BLOG_POST:
            prompt_parts.extend([
                f"## Blog Post Guidelines",
                f"- Include an engaging introduction",
                f"- Use proper headings (H2, H3) for structure",
                f"- Include at least 3 main sections",
                f"- Add a conclusion with call-to-action",
                f"- Naturally incorporate keywords throughout the text",
                f"- Aim for {{'short': '800-1000', 'medium': '1200-1500', 'long': '2000-2500'}[length]} words"
            ])
        elif content_type == ContentType.SOCIAL_MEDIA:
            prompt_parts.extend([
                f"## Social Media Post Guidelines",
                f"- Create an attention-grabbing headline",
                f"- Keep it concise and engaging",
                f"- Include relevant hashtags",
                f"- Add a clear call-to-action",
                f"- Optimize for the platform (if specified in additional context)"
            ])
        elif content_type == ContentType.EMAIL:
            prompt_parts.extend([
                f"## Email Guidelines",
                f"- Create a compelling subject line",
                f"- Start with a personalized greeting",
                f"- Keep paragraphs short and scannable",
                f"- Include a clear call-to-action button/link",
                f"- Add a professional signature"
            ])
        elif content_type == ContentType.LANDING_PAGE:
            prompt_parts.extend([
                f"## Landing Page Guidelines",
                f"- Create an attention-grabbing headline",
                f"- Include a compelling subheading",
                f"- Highlight key benefits (3-5)",
                f"- Add persuasive copy explaining the offer",
                f"- Include testimonials or social proof sections",
                f"- Create a strong call-to-action",
                f"- Address potential objections"
            ])
        
        # Add additional context if provided
        if additional_context:
            prompt_parts.extend([
                f"",
                f"## Additional Context",
                f"{json.dumps(additional_context, indent=2)}"
            ])
        
        return "\n".join(prompt_parts)
    
    def optimize_for_seo(self, content: str, keywords: List[str]) -> str:
        """Optimize content for SEO."""
        try:
            if self.agent_manager:
                seo_prompt = f"""
                Please optimize the following content for SEO using these keywords: {', '.join(keywords)}
                
                Make sure to:
                1. Include keywords in headings (H1, H2, H3)
                2. Ensure keyword density is natural (2-3%)
                3. Add meta description suggestion
                4. Improve readability with shorter paragraphs if needed
                5. Add internal and external link suggestions
                6. Suggest image alt text where appropriate
                
                CONTENT TO OPTIMIZE:
                {content}
                
                Please return the fully optimized content.
                """
                
                optimized_content = self.agent_manager.execute_task(
                    task_description=f"SEO optimization for content",
                    prompt=seo_prompt,
                    agent_persona="seo_specialist"
                )
                
                return optimized_content
            else:
                # Basic SEO optimization if agent manager is not available
                # Add keyword in title if not present
                for keyword in keywords:
                    if keyword.lower() not in content.lower():
                        content += f"\n\nKeywords: {keyword}"
                
                return content
        except Exception as e:
            logger.error(f"Error optimizing content for SEO: {e}")
            return content
    
    def create_content_variations(self, content_item: ContentItem, 
                                 variations_count: int = 3) -> List[ContentItem]:
        """Create variations of existing content."""
        variations = []
        
        try:
            for i in range(variations_count):
                variation_prompt = f"""
                Create a variation of the following content.
                Variation #{i+1} of {variations_count}
                
                Original content:
                {content_item.content}
                
                Please create a unique variation that maintains the same message but with:
                - Different wording
                - Different structure
                - Different examples (if applicable)
                - Same keywords: {', '.join(content_item.tags)}
                
                The variation should be as unique as possible while conveying the same information.
                """
                
                variation_content = ""
                if self.agent_manager:
                    variation_content = self.agent_manager.execute_task(
                        task_description=f"Create content variation #{i+1}",
                        prompt=variation_prompt,
                        agent_persona="creative_writer"
                    )
                elif self.rag_system:
                    variation_content = self.rag_system.query(
                        f"Create a variation of content titled '{content_item.title}'",
                        {"original_content": content_item.content}
                    )
                else:
                    variation_content = content_item.content + f"\n\n[Variation #{i+1}]"
                
                # Create variation content item
                variation_item = ContentItem(
                    id=str(uuid.uuid4()),
                    title=f"{content_item.title} - Variation #{i+1}",
                    content_type=content_item.content_type,
                    content=variation_content,
                    tags=content_item.tags,
                    created_at=int(time.time()),
                    updated_at=int(time.time()),
                    author=content_item.author,
                    status="draft",
                    channels=content_item.channels,
                    metadata={
                        "original_content_id": content_item.id,
                        "variation_number": i+1
                    }
                )
                
                variation_item.save()
                variations.append(variation_item)
        
        except Exception as e:
            logger.error(f"Error creating content variations: {e}")
        
        return variations

class SocialMediaManager:
    """Social media management system."""
    
    def __init__(self, content_generator: Optional[ContentGenerator] = None):
        """Initialize the social media manager."""
        self.config = MarketingConfig.load()
        self.content_generator = content_generator or ContentGenerator()
        self.scheduled_posts = self._load_scheduled_posts()
        self.posting_thread = None
        self.running = False
    
    def _load_scheduled_posts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load scheduled posts."""
        scheduled_posts = {}
        
        try:
            schedule_file = CAMPAIGNS_DIR / "scheduled_social_posts.json"
            if schedule_file.exists():
                with open(schedule_file, 'r') as f:
                    scheduled_posts = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scheduled posts: {e}")
        
        return scheduled_posts
    
    def _save_scheduled_posts(self) -> None:
        """Save scheduled posts."""
        try:
            schedule_file = CAMPAIGNS_DIR / "scheduled_social_posts.json"
            with open(schedule_file, 'w') as f:
                json.dump(self.scheduled_posts, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scheduled posts: {e}")
    
    def create_social_post(self, channel: MarketingChannel, content: str, 
                          title: str = None, media_urls: List[str] = None,
                          schedule_time: Optional[int] = None) -> Dict[str, Any]:
        """Create a social media post."""
        post_id = str(uuid.uuid4())
        
        post = {
            "id": post_id,
            "channel": channel.value,
            "content": content,
            "title": title,
            "media_urls": media_urls or [],
            "created_at": int(time.time()),
            "status": "scheduled" if schedule_time else "draft",
            "schedule_time": schedule_time,
            "published_at": None,
            "metrics": {}
        }
        
        # Add to scheduled posts if a schedule time is provided
        if schedule_time:
            if channel.value not in self.scheduled_posts:
                self.scheduled_posts[channel.value] = []
            
            self.scheduled_posts[channel.value].append(post)
            self._save_scheduled_posts()
        
        return post
    
    def schedule_post(self, post_id: str, schedule_time: int) -> bool:
        """Schedule a post for publishing."""
        for channel, posts in self.scheduled_posts.items():
            for post in posts:
                if post["id"] == post_id:
                    post["schedule_time"] = schedule_time
                    post["status"] = "scheduled"
                    self._save_scheduled_posts()
                    return True
        
        # If post not found in scheduled posts, look for it in content items
        try:
            for content_file in CONTENT_DIR.glob("*.json"):
                with open(content_file, 'r') as f:
                    content_data = json.load(f)
                
                if content_data["id"] == post_id:
                    # Create a scheduled post from this content item
                    channel = MarketingChannel.SOCIAL_MEDIA
                    if content_data.get("channels") and len(content_data["channels"]) > 0:
                        channel = MarketingChannel(content_data["channels"][0])
                    
                    post = {
                        "id": post_id,
                        "channel": channel.value,
                        "content": content_data["content"],
                        "title": content_data["title"],
                        "media_urls": [],
                        "created_at": content_data["created_at"],
                        "status": "scheduled",
                        "schedule_time": schedule_time,
                        "published_at": None,
                        "metrics": {}
                    }
                    
                    if channel.value not in self.scheduled_posts:
                        self.scheduled_posts[channel.value] = []
                    
                    self.scheduled_posts[channel.value].append(post)
                    self._save_scheduled_posts()
                    return True
        except Exception as e:
            logger.error(f"Error scheduling post: {e}")
        
        return False
    
    def publish_post(self, post_id: str) -> bool:
        """Publish a post immediately."""
        for channel, posts in self.scheduled_posts.items():
            for post in posts:
                if post["id"] == post_id:
                    success = self._publish_to_channel(post, channel)
                    if success:
                        post["status"] = "published"
                        post["published_at"] = int(time.time())
                        self._save_scheduled_posts()
                    return success
        
        return False
    
    def _publish_to_channel(self, post: Dict[str, Any], channel_name: str) -> bool:
        """Publish a post to a specific channel."""
        try:
            channel = MarketingChannel(channel_name)
            
            # Get API credentials for the channel
            credentials = self.config.social_media_settings.get(channel_name, {})
            
            if not credentials:
                logger.warning(f"No API credentials found for {channel_name}")
                return False
            
            # Publish based on channel type
            if channel == MarketingChannel.TWITTER:
                return self._publish_to_twitter(post, credentials)
            elif channel == MarketingChannel.FACEBOOK:
                return self._publish_to_facebook(post, credentials)
            elif channel == MarketingChannel.LINKEDIN:
                return self._publish_to_linkedin(post, credentials)
            elif channel == MarketingChannel.INSTAGRAM:
                return self._publish_to_instagram(post, credentials)
            else:
                logger.warning(f"Publishing to {channel_name} not implemented")
                return False
        
        except Exception as e:
            logger.error(f"Error publishing to {channel_name}: {e}")
            return False
    
    def _publish_to_twitter(self, post: Dict[str, Any], credentials: Dict[str, str]) -> bool:
        """Publish to Twitter."""
        try:
            # This would use the Twitter API in a production environment
            logger.info(f"Simulating Twitter post: {post['content'][:50]}...")
            
            # In a real implementation, this would use the Twitter API
            # import tweepy
            # auth = tweepy.OAuthHandler(credentials['api_key'], credentials['api_secret'])
            # auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])
            # api = tweepy.API(auth)
            # tweet = api.update_status(post['content'])
            
            # For now, just log the post
            logger.info(f"Twitter post would be published: {post['content'][:100]}...")
            
            # Record metrics
            post["metrics"]["platform"] = "twitter"
            post["metrics"]["post_id"] = f"sim_{uuid.uuid4()}"
            post["metrics"]["timestamp"] = int(time.time())
            
            return True
        
        except Exception as e:
            logger.error(f"Error publishing to Twitter: {e}")
            return False
    
    def _publish_to_facebook(self, post: Dict[str, Any], credentials: Dict[str, str]) -> bool:
        """Publish to Facebook."""
        try:
            # This would use the Facebook Graph API in a production environment
            logger.info(f"Simulating Facebook post: {post['content'][:50]}...")
            
            # Record metrics
            post["metrics"]["platform"] = "facebook"
            post["metrics"]["post_id"] = f"sim_{uuid.uuid4()}"
            post["metrics"]["timestamp"] = int(time.time())
            
            return True
        
        except Exception as e:
            logger.error(f"Error publishing to Facebook: {e}")
            return False
    
    def _publish_to_linkedin(self, post: Dict[str, Any], credentials: Dict[str, str]) -> bool:
        """Publish to LinkedIn."""
        try:
            # This would use the LinkedIn API in a production environment
            logger.info(f"Simulating LinkedIn post: {post['content'][:50]}...")
            
            # Record metrics
            post["metrics"]["platform"] = "linkedin"
            post["metrics"]["post_id"] = f"sim_{uuid.uuid4()}"
            post["metrics"]["timestamp"] = int(time.time())
            
            return True
        
        except Exception as e:
            logger.error(f"Error publishing to LinkedIn: {e}")
            return False
    
    def _publish_to_instagram(self, post: Dict[str, Any], credentials: Dict[str, str]) -> bool:
        """Publish to Instagram."""
        try:
            # This would use the Instagram API in a production environment
            logger.info(f"Simulating Instagram post: {post['content'][:50]}...")
            
            # Record metrics
            post["metrics"]["platform"] = "instagram"
            post["metrics"]["post_id"] = f"sim_{uuid.uuid4()}"
            post["metrics"]["timestamp"] = int(time.time())
            
            return True
        
        except Exception as e:
            logger.error(f"Error publishing to Instagram: {e}")
            return False
    
    def start_scheduler(self) -> None:
        """Start the post scheduler."""
        if self.posting_thread and self.posting_thread.is_alive():
            return
        
        self.running = True
        
        def _scheduler_thread():
            while self.running:
                try:
                    current_time = int(time.time())
                    
                    # Check for posts to publish
                    for channel, posts in self.scheduled_posts.items():
                        for post in posts:
                            if post["status"] == "scheduled" and post["schedule_time"] <= current_time:
                                logger.info(f"Publishing scheduled post to {channel}: {post['id']}")
                                success = self._publish_to_channel(post, channel)
                                
                                if success:
                                    post["status"] = "published"
                                    post["published_at"] = current_time
                                    self._save_scheduled_posts()
                                else:
                                    post["status"] = "failed"
                                    post["failure_time"] = current_time
                                    self._save_scheduled_posts()
                    
                    # Sleep for a bit
                    time.sleep(60)  # Check every minute
                
                except Exception as e:
                    logger.error(f"Error in scheduler thread: {e}")
                    time.sleep(60)  # Wait a bit before retrying
        
        self.posting_thread = threading.Thread(target=_scheduler_thread)
        self.posting_thread.daemon = True
        self.posting_thread.start()
        
        logger.info("Social media post scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the post scheduler."""
        self.running = False
        logger.info("Social media post scheduler stopped")
    
    def generate_content_calendar(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a content calendar for the specified number of days."""
        calendar = {}
        current_time = time.time()
        day_seconds = 24 * 60 * 60
        
        try:
            # Get posting frequency from config
            posting_frequency = self.config.posting_frequency
            content_calendar = self.config.content_calendar
            
            # Generate posts for each day
            for day in range(days):
                day_timestamp = int(current_time + (day * day_seconds))
                day_date = datetime.datetime.fromtimestamp(day_timestamp).strftime("%Y-%m-%d")
                calendar[day_date] = []
                
                # Get day of week
                day_of_week = datetime.datetime.fromtimestamp(day_timestamp).strftime("%A").lower()
                
                # Check if we have specific content types for this day
                day_content_types = content_calendar.get(day_of_week, [])
                
                if not day_content_types:
                    continue
                
                # Generate content for each channel based on frequency
                for channel, freq in posting_frequency.items():
                    # Skip if we don't post on this day based on frequency
                    if day % (7 // freq) != 0:
                        continue
                    
                    # Determine content type
                    content_type = None
                    for ct in day_content_types:
                        if ct == ContentType.SOCIAL_MEDIA and channel in [
                            "facebook", "twitter", "instagram", "linkedin"
                        ]:
                            content_type = ContentType.SOCIAL_MEDIA
                            break
                        elif ct == ContentType.BLOG_POST and channel == "blog":
                            content_type = ContentType.BLOG_POST
                            break
                        elif ct == ContentType.EMAIL and channel == "email":
                            content_type = ContentType.EMAIL
                            break
                    
                    if not content_type:
                        continue
                    
                    # Generate placeholder for content
                    post_time = day_timestamp + random.randint(9, 17) * 3600  # Random time between 9 AM and 5 PM
                    
                    calendar[day_date].append({
                        "channel": channel,
                        "content_type": content_type.value,
                        "title": f"[Placeholder] {content_type.value.title()} for {channel} on {day_date}",
                        "schedule_time": post_time,
                        "status": "planned"
                    })
            
            # Save the calendar
            calendar_file = CAMPAIGNS_DIR / "content_calendar.json"
            with open(calendar_file, 'w') as f:
                json.dump(calendar, f, indent=2)
            
            return calendar
        
        except Exception as e:
            logger.error(f"Error generating content calendar: {e}")
            return {}

class EmailCampaignManager:
    """Email campaign management system."""
    
    def __init__(self, content_generator: Optional[ContentGenerator] = None):
        """Initialize the email campaign manager."""
        self.config = MarketingConfig.load()
        self.content_generator = content_generator or ContentGenerator()
        self.campaigns = self._load_campaigns()
        self.subscribers = self._load_subscribers()
        self.templates = self._load_email_templates()
        self.sending_thread = None
        self.running = False
    
    def _load_campaigns(self) -> Dict[str, Dict[str, Any]]:
        """Load email campaigns."""
        campaigns = {}
        
        try:
            campaigns_dir = CAMPAIGNS_DIR / "email"
            campaigns_dir.mkdir(parents=True, exist_ok=True)
            
            for campaign_file in campaigns_dir.glob("*.json"):
                with open(campaign_file, 'r') as f:
                    campaign_data = json.load(f)
                campaigns[campaign_data["id"]] = campaign_data
        
        except Exception as e:
            logger.error(f"Error loading email campaigns: {e}")
        
        return campaigns
    
    def _save_campaign(self, campaign_id: str, campaign_data: Dict[str, Any]) -> None:
        """Save email campaign."""
        try:
            campaigns_dir = CAMPAIGNS_DIR / "email"
            campaigns_dir.mkdir(parents=True, exist_ok=True)
            
            campaign_file = campaigns_dir / f"{campaign_id}.json"
            with open(campaign_file, 'w') as f:
                json.dump(campaign_data, f, indent=2)
            
            # Update in-memory campaigns
            self.campaigns[campaign_id] = campaign_data
        
        except Exception as e:
            logger.error(f"Error saving email campaign: {e}")
    
    def _load_subscribers(self) -> List[Dict[str, Any]]:
        """Load email subscribers."""
        subscribers = []
        
        try:
            subscribers_file = CAMPAIGNS_DIR / "email_subscribers.json"
            if subscribers_file.exists():
                with open(subscribers_file, 'r') as f:
                    subscribers = json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading email subscribers: {e}")
        
        return subscribers
    
    def _save_subscribers(self) -> None:
        """Save email subscribers."""
        try:
            subscribers_file = CAMPAIGNS_DIR / "email_subscribers.json"
            with open(subscribers_file, 'w') as f:
                json.dump(self.subscribers, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving email subscribers: {e}")
    
    def _load_email_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load email templates."""
        templates = {}
        
        try:
            templates_dir = TEMPLATES_DIR / "email"
            templates_dir.mkdir(parents=True, exist_ok=True)
            
            for template_file in templates_dir.glob("*.json"):
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                templates[template_data["id"]] = template_data
        
        except Exception as e:
            logger.error(f"Error loading email templates: {e}")
        
        return templates
    
    def create_campaign(self, name: str, subject: str, content: str, 
                       list_ids: List[str] = None, schedule_time: Optional[int] = None,
                       template_id: Optional[str] = None) -> str:
        """Create a new email campaign."""
        campaign_id = str(uuid.uuid4())
        
        # Apply template if provided
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            html_content = template["html"]
            
            # Replace placeholders in template
            html_content = html_content.replace("{{subject}}", subject)
            html_content = html_content.replace("{{content}}", content)
            html_content = html_content.replace("{{company_name}}", self.config.brand_name)
            
            content = html_content
        
        campaign = {
            "id": campaign_id,
            "name": name,
            "subject": subject,
            "content": content,
            "list_ids": list_ids or ["default"],
            "created_at": int(time.time()),
            "status": "draft",
            "schedule_time": schedule_time,
            "sent_at": None,
            "metrics": {
                "recipients": 0,
                "opens": 0,
                "clicks": 0,
                "bounces": 0,
                "unsubscribes": 0
            }
        }
        
        self._save_campaign(campaign_id, campaign)
        
        return campaign_id
    
    def schedule_campaign(self, campaign_id: str, schedule_time: int) -> bool:
        """Schedule an email campaign."""
        if campaign_id not in self.campaigns:
            logger.error(f"Campaign not found: {campaign_id}")
            return False
        
        campaign = self.campaigns[campaign_id]
        campaign["schedule_time"] = schedule_time
        campaign["status"] = "scheduled"
        
        self._save_campaign(campaign_id, campaign)
        return True
    
    def send_campaign(self, campaign_id: str) -> bool:
        """Send an email campaign immediately."""
        if campaign_id not in self.campaigns:
            logger.error(f"Campaign not found: {campaign_id}")
            return False
        
        campaign = self.campaigns[campaign_id]
        
        # Get subscribers for the campaign lists
        recipients = []
        for list_id in campaign["list_ids"]:
            for subscriber in self.subscribers:
                if list_id in subscriber.get("list_ids", []) and subscriber.get("status") == "active":
                    recipients.append(subscriber)
        
        # Send emails
        sent_count = 0
        for recipient in recipients:
            if self._send_email(
                to_email=recipient["email"],
                to_name=recipient.get("name", ""),
                subject=campaign["subject"],
                content=campaign["content"],
                campaign_id=campaign_id,
                subscriber_id=recipient["id"]
            ):
                sent_count += 1
        
        # Update campaign metrics
        campaign["status"] = "sent"
        campaign["sent_at"] = int(time.time())
        campaign["metrics"]["recipients"] = sent_count
        
        self._save_campaign(campaign_id, campaign)
        
        return sent_count > 0
    
    def _send_email(self, to_email: str, to_name: str, subject: str, 
                   content: str, campaign_id: str, subscriber_id: str) -> bool:
        """Send an individual email."""
        try:
            # Get email settings
            email_settings = self.config.email_settings
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{email_settings.get('from_name', 'Skyscope Sentinel')} <{email_settings.get('from_email', 'noreply@example.com')}>"
            msg['To'] = f"{to_name} <{to_email}>" if to_name else to_email
            
            # Add tracking pixel and click tracking
            tracking_pixel = f'<img src="https://example.com/track/open/{campaign_id}/{subscriber_id}" width="1" height="1" />'
            
            # Convert links to trackable links
            soup = BeautifulSoup(content, 'html.parser')
            for link in soup.find_all('a'):
                if 'href' in link.attrs:
                    original_url = link['href']
                    tracking_url = f"https://example.com/track/click/{campaign_id}/{subscriber_id}?url={original_url}"
                    link['href'] = tracking_url
            
            # Add tracking pixel to end of content
            tracked_content = str(soup) + tracking_pixel
            
            # Attach parts
            text_part = MIMEText(BeautifulSoup(content, 'html.parser').get_text(), 'plain')
            html_part = MIMEText(tracked_content, 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # In a production environment, this would actually send the email
            logger.info(f"Simulating email send to {to_email}: {subject}")
            
            # For demonstration purposes, just log the send
            # In production, we would use:
            # with smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port']) as server:
            #     server.starttls()
            #     server.login(email_settings['smtp_username'], email_settings['smtp_password'])
            #     server.send_message(msg)
            
            return True
        
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def add_subscriber(self, email: str, name: str = "", 
                      list_ids: List[str] = None, metadata: Dict[str, Any] = None) -> str:
        """Add a new subscriber."""
        # Check if subscriber already exists
        for subscriber in self.subscribers:
            if subscriber["email"].lower() == email.lower():
                # Update existing subscriber
                subscriber["name"] = name or subscriber["name"]
                subscriber["list_ids"] = list(set(subscriber.get("list_ids", []) + (list_ids or ["default"])))
                subscriber["updated_at"] = int(time.time())
                
                if metadata:
                    if "metadata" not in subscriber:
                        subscriber["metadata"] = {}
                    subscriber["metadata"].update(metadata)
                
                self._save_subscribers()
                return subscriber["id"]
        
        # Create new subscriber
        subscriber_id = str(uuid.uuid4())
        
        subscriber = {
            "id": subscriber_id,
            "email": email,
            "name": name,
            "list_ids": list_ids or ["default"],
            "status": "active",
            "created_at": int(time.time()),
            "updated_at": int(time.time()),
            "metadata": metadata or {}
        }
        
        self.subscribers.append(subscriber)
        self._save_subscribers()
        
        return subscriber_id
    
    def remove_subscriber(self, email: str) -> bool:
        """Remove a subscriber."""
        for i, subscriber in enumerate(self.subscribers):
            if subscriber["email"].lower() == email.lower():
                # Mark as unsubscribed instead of removing
                self.subscribers[i]["status"] = "unsubscribed"
                self.subscribers[i]["updated_at"] = int(time.time())
                self._save_subscribers()
                return True
        
        return False
    
    def create_email_template(self, name: str, html: str, description: str = "") -> str:
        """Create a new email template."""
        template_id = str(uuid.uuid4())
        
        template = {
            "id": template_id,
            "name": name,
            "html": html,
            "description": description,
            "created_at": int(time.time()),
            "updated_at": int(time.time())
        }
        
        # Save template
        templates_dir = TEMPLATES_DIR / "email"
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        template_file = templates_dir / f"{template_id}.json"
        with open(template_file, 'w') as f:
            json.dump(template, f, indent=2)
        
        # Update in-memory templates
        self.templates[template_id] = template
        
        return template_id
    
    def start_scheduler(self) -> None:
        """Start the email campaign scheduler."""
        if self.sending_thread and self.sending_thread.is_alive():
            return
        
        self.running = True
        
        def _scheduler_thread():
            while self.running:
                try:
                    current_time = int(time.time())
                    
                    # Check for campaigns to send
                    for campaign_id, campaign in self.campaigns.items():
                        if campaign["status"] == "scheduled" and campaign["schedule_time"] <= current_time:
                            logger.info(f"Sending scheduled email campaign: {campaign['name']}")
                            self.send_campaign(campaign_id)
                    
                    # Sleep for a bit
                    time.sleep(60)  # Check every minute
                
                except Exception as e:
                    logger.error(f"Error in email scheduler thread: {e}")
                    time.sleep(60)  # Wait a bit before retrying
        
        self.sending_thread = threading.Thread(target=_scheduler_thread)
        self.sending_thread.daemon = True
        self.sending_thread.start()
        
        logger.info("Email campaign scheduler started")
    
    def stop_scheduler(self) -> None:
        """Stop the email campaign scheduler."""
        self.running = False
        logger.info("Email campaign scheduler stopped")

class SEOManager:
    """SEO optimization and management system."""
    
    def __init__(self, content_generator: Optional[ContentGenerator] = None):
        """Initialize the SEO manager."""
        self.config = MarketingConfig.load()
        self.content_generator = content_generator or ContentGenerator()
        self.keyword_data = self._load_keyword_data()
    
    def _load_keyword_data(self) -> Dict[str, Dict[str, Any]]:
        """Load keyword data."""
        keyword_data = {}
        
        try:
            keyword_file = ANALYTICS_DIR / "keyword_data.json"
            if keyword_file.exists():
                with open(keyword_file, 'r') as f:
                    keyword_data = json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading keyword data: {e}")
        
        return keyword_data
    
    def _save_keyword_data(self) -> None:
        """Save keyword data."""
        try:
            keyword_file = ANALYTICS_DIR / "keyword_data.json"
            with open(keyword_file, 'w') as f:
                json.dump(self.keyword_data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving keyword data: {e}")
    
    def analyze_content(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Analyze content for SEO optimization."""
        analysis = {
            "keyword_density": {},
            "readability": {},
            "meta_data": {},
            "suggestions": []
        }
        
        try:
            # Extract text from HTML if needed
            if "<html" in content.lower() or "<body" in content.lower():
                text_content = BeautifulSoup(content, 'html.parser').get_text()
            else:
                text_content = content
            
            # Count words
            words = re.findall(r'\b\w+\b', text_content.lower())
            word_count = len(words)
            analysis["word_count"] = word_count
            
            # Calculate keyword density
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                keyword_count = text_content.lower().count(keyword_lower)
                if word_count > 0:
                    density = (keyword_count * len(keyword_lower.split())) / word_count * 100
                else:
                    density = 0
                analysis["keyword_density"][keyword] = {
                    "count": keyword_count,
                    "density": round(density, 2)
                }
            
            # Calculate readability (Flesch Reading Ease)
            sentences = len(re.split(r'[.!?]+', text_content))
            if sentences > 0:
                words_per_sentence = word_count / sentences
            else:
                words_per_sentence = 0
            
            syllables = 0
            for word in words:
                syllables += self._count_syllables(word)
            
            if word_count > 0:
                syllables_per_word = syllables / word_count
            else:
                syllables_per_word = 0
            
            if sentences > 0 and word_count > 0:
                flesch_score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
            else:
                flesch_score = 0
            
            analysis["readability"] = {
                "sentences": sentences,
                "words_per_sentence": round(words_per_sentence, 1),
                "syllables_per_word": round(syllables_per_word, 2),
                "flesch_reading_ease": round(flesch_score, 1),
                "readability_level": self._get_readability_level(flesch_score)
            }
            
            # Extract meta data if HTML
            if "<html" in content.lower():
                soup = BeautifulSoup(content, 'html.parser')
                
                # Title
                title_tag = soup.find('title')
                if title_tag:
                    analysis["meta_data"]["title"] = title_tag.text
                    analysis["meta_data"]["title_length"] = len(title_tag.text)
                
                # Meta description
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and 'content' in meta_desc.attrs:
                    analysis["meta_data"]["description"] = meta_desc['content']
                    analysis["meta_data"]["description_length"] = len(meta_desc['content'])
                
                # H1
                h1_tags = soup.find_all('h1')
                if h1_tags:
                    analysis["meta_data"]["h1"] = [tag.text for tag in h1_tags]
                    analysis["meta_data"]["h1_count"] = len(h1_tags)
                
                # H2
                h2_tags = soup.find_all('h2')
                if h2_tags:
                    analysis["meta_data"]["h2"] = [tag.text for tag in h2_tags]
                    analysis["meta_data"]["h2_count"] = len(h2_tags)
                
                # Images without alt text
                images = soup.find_all('img')
                images_without_alt = [img for img in images if 'alt' not in img.attrs or not img['alt']]
                analysis["meta_data"]["images_without_alt"] = len(images_without_alt)
            
            # Generate suggestions
            suggestions = []
            
            # Keyword density suggestions
            for keyword, data in analysis["keyword_density"].items():
                if data["density"] < 0.5:
                    suggestions.append(f"Increase '{keyword}' density (currently {data['density']}%)")
                elif data["density"] > 3.0:
                    suggestions.append(f"Decrease '{keyword}' density (currently {data['density']}%)")
            
            # Word count suggestions
            if word_count < 300:
                suggestions.append(f"Increase content length (currently {word_count} words)")
            
            # Readability suggestions
            if flesch_score < 30:
                suggestions.append("Content is too difficult to read. Simplify language.")
            elif flesch_score > 70:
                suggestions.append("Content may be too simple for professional audience.")
            
            # Meta suggestions
            if "title_length" in analysis.get("meta_data", {}) and analysis["meta_data"]["title_length"] > 60:
                suggestions.append(f"Title tag too long ({analysis['meta_data']['title_length']} chars). Keep under 60 chars.")
            
            if "description_length" in analysis.get("meta_data", {}) and analysis["meta_data"]["description_length"] > 160:
                suggestions.append(f"Meta description too long ({analysis['meta_data']['description_length']} chars). Keep under 160 chars.")
            
            if "h1_count" in analysis.get("meta_data", {}) and analysis["meta_data"]["h1_count"] > 1:
                suggestions.append(f"Too many H1 tags ({analysis['meta_data']['h1_count']}). Use only one H1.")
            
            if "images_without_alt" in analysis.get("meta_data", {}) and analysis["meta_data"]["images_without_alt"] > 0:
                suggestions.append(f"Add alt text to {analysis['meta_data']['images_without_alt']} images.")
            
            analysis["suggestions"] = suggestions
            
            return analysis
        
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {
                "error": str(e),
                "keyword_density": {},
                "readability": {},
                "meta_data": {},
                "suggestions": ["Error analyzing content"]
            }
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        
        # Remove non-alpha characters
        word = re.sub(r'[^a-z]', '', word)
        
        if not word:
            return 0
        
        # Count vowel groups
        count = len(re.findall(r'[aeiouy]+', word))
        
        # Adjust for special cases
        if word.endswith('e'):
            count -= 1
        
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
            count += 1
        
        if count == 0:
            count = 1
        
        return count
    
    def _get_readability_level(self, score: float) -> str:
        """Get readability level from Flesch score."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def optimize_content(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """Optimize content for SEO."""
        try:
            # First analyze the content
            analysis = self.analyze_content(content, target_keywords)
            
            # If there are no suggestions, content is already optimized
            if not analysis["suggestions"]:
                return {
                    "optimized_content": content,
                    "analysis": analysis,
                    "changes_made": []
                }
            
            # Use AI to optimize content if available
            optimized_content = content
            changes_made = []
            
            if self.content_generator and self.content_generator.agent_manager:
                optimization_prompt = f"""
                Please optimize the following content for SEO based on these suggestions:
                
                {json.dumps(analysis["suggestions"], indent=2)}
                
                Target keywords: {', '.join(target_keywords)}
                
                CONTENT TO OPTIMIZE:
                {content}
                
                Please return only the optimized content.
                """
                
                optimized_content = self.content_generator.agent_manager.execute_task(
                    task_description="SEO content optimization",
                    prompt=optimization_prompt,
                    agent_persona="seo_specialist"
                )
                
                # Record changes made
                changes_made = analysis["suggestions"]
            
            # Re-analyze the optimized content
            optimized_analysis = self.analyze_content(optimized_content, target_keywords)
            
            return {
                "optimized_content": optimized_content,
                "original_analysis": analysis,
                "optimized_analysis": optimized_analysis,
                "changes_made": changes_made
            }
        
        except Exception as e:
            logger.error(f"Error optimizing content: {e}")
            return {
                "error": str(e),
                "optimized_content": content,
                "analysis": {},
                "changes_made": []
            }
    
    def research_keywords(self, seed_keywords: List[str]) -> Dict[str, List[str]]:
        """Research related keywords."""
        related_keywords = {}
        
        try:
            for keyword in seed_keywords:
                # In a production environment, this would use an API like Google Keyword Planner,
                # SEMrush, Ahrefs, etc. For now, we'll simulate results.
                
                # Simulate related keywords
                related = []
                keyword_parts = keyword.split()
                
                # Add variations
                for i in range(len(keyword_parts)):
                    for prefix in ["best", "top", "affordable", "professional", "advanced"]:
                        new_parts = keyword_parts.copy()
                        new_parts.insert(i, prefix)
                        related.append(" ".join(new_parts))
                
                # Add questions
                for question in ["how to", "what is", "why use", "when to use"]:
                    related.append(f"{question} {keyword}")
                
                # Add location-based variations for target markets
                for market in self.config.target_markets:
                    related.append(f"{keyword} in {market}")
                    related.append(f"{market} {keyword}")
                
                related_keywords[keyword] = related
            
            return related_keywords
        
        except Exception as e:
            logger.error(f"Error researching keywords: {e}")
            return {keyword: [] for keyword in seed_keywords}
    
    def generate_seo_report(self, url: str = None, content: str = None) -> Dict[str, Any]:
        """Generate an SEO report for a URL or content."""
        report = {
            "timestamp": int(time.time()),
            "url": url,
            "overall_score": 0,
            "sections": {}
        }
        
        try:
            # If URL is provided, fetch content
            if url:
                try:
                    response = requests.get(url, timeout=10)
                    content = response.text
                except Exception as e:
                    logger.error(f"Error fetching URL {url}: {e}")
                    return {
                        "error": f"Could not fetch URL: {str(e)}",
                        "timestamp": int(time.time()),
                        "url": url
                    }
            
            if not content:
                return {
                    "error": "No content or URL provided",
                    "timestamp": int(time.time())
                }
            
            # Parse content
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text
            text_content = soup.get_text()
            
            # Technical SEO
            technical_seo = {}
            
            # Title
            title_tag = soup.find('title')
            technical_seo["title"] = {
                "content": title_tag.text if title_tag else None,
                "length": len(title_tag.text) if title_tag else 0,
                "score": 100 if title_tag and 30 <= len(title_tag.text) <= 60 else 
                        (50 if title_tag else 0)
            }
            
            # Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            technical_seo["meta_description"] = {
                "content": meta_desc['content'] if meta_desc and 'content' in meta_desc.attrs else None,
                "length": len(meta_desc['content']) if meta_desc and 'content' in meta_desc.attrs else 0,
                "score": 100 if meta_desc and 'content' in meta_desc.attrs and 
                        120 <= len(meta_desc['content']) <= 160 else 
                        (50 if meta_desc and 'content' in meta_desc.attrs else 0)
            }
            
            # Headings
            headings = {
                "h1": [h.text for h in soup.find_all('h1')],
                "h2": [h.text for h in soup.find_all('h2')],
                "h3": [h.text for h in soup.find_all('h3')]
            }
            
            heading_score = 100
            if not headings["h1"]:
                heading_score -= 50
            elif len(headings["h1"]) > 1:
                
