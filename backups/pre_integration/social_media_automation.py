#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Social Media Automation
==========================================================

This module provides comprehensive social media automation capabilities for the
Skyscope Sentinel Intelligence AI platform, enabling sophisticated content generation,
scheduling, posting, and engagement across multiple social media platforms.

Features:
- Multi-platform support (Twitter/X, Discord, Telegram)
- GPT-4o powered content generation via openai-unofficial
- Agent personality system for consistent brand voice
- Automated posting and engagement
- Trend detection and viral content analysis
- Community management and response automation
- Analytics and performance tracking
- Content scheduling and calendar management
- Multi-language support (20+ languages)
- Meme and visual content generation
- Sentiment analysis and brand monitoring
- Campaign management and A/B testing
- Crisis detection and management
- Compliance and content moderation

Dependencies:
- tweepy (for Twitter/X API)
- discord.py (for Discord API)
- python-telegram-bot (for Telegram API)
- openai-unofficial (for GPT-4o integration)
- pandas, numpy (for data analysis)
- pillow, matplotlib (for image generation)
- nltk, textblob (for sentiment analysis)
- schedule (for content scheduling)
- langdetect, googletrans (for language detection and translation)
"""

import os
import re
import json
import time
import uuid
import base64
import random
import logging
import asyncio
import threading
import traceback
import schedule
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/social_media.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("social_media")

# --- Twitter/X Integration ---
try:
    import tweepy
    TWITTER_AVAILABLE = True
    logger.info("Twitter/X integration available via tweepy")
except ImportError:
    TWITTER_AVAILABLE = False
    logger.warning("Twitter/X integration not available. Install tweepy for Twitter support.")

# --- Discord Integration ---
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
    logger.info("Discord integration available via discord.py")
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning("Discord integration not available. Install discord.py for Discord support.")

# --- Telegram Integration ---
try:
    from telegram import Update, Bot
    from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
    TELEGRAM_AVAILABLE = True
    logger.info("Telegram integration available via python-telegram-bot")
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("Telegram integration not available. Install python-telegram-bot for Telegram support.")

# --- OpenAI Integration ---
try:
    import openai_unofficial
    OPENAI_UNOFFICIAL_AVAILABLE = True
    logger.info("GPT-4o integration available via openai-unofficial")
except ImportError:
    OPENAI_UNOFFICIAL_AVAILABLE = False
    logger.warning("openai-unofficial not available. GPT-4o integration will be limited.")
    try:
        import openai
        OPENAI_AVAILABLE = True
        logger.info("Standard OpenAI integration available as fallback")
    except ImportError:
        OPENAI_AVAILABLE = False
        logger.warning("OpenAI integration not available. Content generation will be limited.")

# --- Image Generation and Processing ---
try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    import matplotlib.pyplot as plt
    IMAGE_PROCESSING_AVAILABLE = True
    logger.info("Image processing available via Pillow and Matplotlib")
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    logger.warning("Image processing libraries not available. Visual content generation will be limited.")

# --- NLP and Sentiment Analysis ---
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    nltk.download('vader_lexicon', quiet=True)
    NLP_AVAILABLE = True
    logger.info("NLP libraries loaded successfully")
except ImportError:
    NLP_AVAILABLE = False
    logger.warning("NLP libraries not available. Sentiment analysis will be limited.")

# --- Language Detection and Translation ---
try:
    from langdetect import detect
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
    logger.info("Language detection and translation available")
except ImportError:
    TRANSLATION_AVAILABLE = False
    logger.warning("Language detection and translation not available. Multi-language support will be limited.")

# --- Constants and Enumerations ---

class PlatformType(Enum):
    """Types of social media platforms supported."""
    TWITTER = auto()
    DISCORD = auto()
    TELEGRAM = auto()
    INSTAGRAM = auto()
    FACEBOOK = auto()
    LINKEDIN = auto()
    REDDIT = auto()
    TIKTOK = auto()
    YOUTUBE = auto()

class ContentType(Enum):
    """Types of content that can be created and posted."""
    TEXT = auto()
    IMAGE = auto()
    VIDEO = auto()
    POLL = auto()
    LINK = auto()
    THREAD = auto()
    STORY = auto()
    MEME = auto()
    INFOGRAPHIC = auto()
    CAROUSEL = auto()

class PostFrequency(Enum):
    """Posting frequency options."""
    HOURLY = auto()
    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    CUSTOM = auto()

class EngagementType(Enum):
    """Types of engagement actions."""
    LIKE = auto()
    COMMENT = auto()
    SHARE = auto()
    RETWEET = auto()
    REPLY = auto()
    FOLLOW = auto()
    DIRECT_MESSAGE = auto()

class LanguageCode(Enum):
    """Supported language codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE = "zh"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    TURKISH = "tr"
    SWEDISH = "sv"
    POLISH = "pl"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"
    GREEK = "el"

class PersonalityTrait(Enum):
    """Personality traits for agent personas."""
    PROFESSIONAL = auto()
    CASUAL = auto()
    HUMOROUS = auto()
    INFORMATIVE = auto()
    INSPIRATIONAL = auto()
    PROVOCATIVE = auto()
    SUPPORTIVE = auto()
    AUTHORITATIVE = auto()
    FRIENDLY = auto()
    TECHNICAL = auto()
    CREATIVE = auto()
    FORMAL = auto()

class TrendCategory(Enum):
    """Categories for trending topics."""
    CRYPTO = auto()
    TECHNOLOGY = auto()
    FINANCE = auto()
    POLITICS = auto()
    ENTERTAINMENT = auto()
    SPORTS = auto()
    HEALTH = auto()
    SCIENCE = auto()
    ENVIRONMENT = auto()
    BUSINESS = auto()
    MEMES = auto()

# --- Data Structures ---

@dataclass
class SocialMediaAccount:
    """Structure for a social media account."""
    platform: PlatformType
    username: str
    account_id: str
    auth_token: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    access_token: Optional[str] = None
    access_secret: Optional[str] = None
    webhook_url: Optional[str] = None
    profile_url: Optional[str] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    rate_limits: Dict[str, Any] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentPersonality:
    """Structure for an agent's personality."""
    name: str
    traits: List[PersonalityTrait]
    tone: str
    voice: str
    emoji_usage: float  # 0.0 to 1.0
    formality: float  # 0.0 to 1.0
    humor_level: float  # 0.0 to 1.0
    hashtag_usage: float  # 0.0 to 1.0
    emoji_set: List[str] = field(default_factory=list)
    hashtag_set: List[str] = field(default_factory=list)
    vocabulary: Dict[str, List[str]] = field(default_factory=dict)
    taboo_words: List[str] = field(default_factory=list)
    preferred_topics: List[str] = field(default_factory=list)
    avoided_topics: List[str] = field(default_factory=list)
    system_prompt: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentTemplate:
    """Structure for a content template."""
    template_id: str
    name: str
    content_type: ContentType
    template_text: str
    variables: List[str]
    platforms: List[PlatformType]
    character_limits: Dict[PlatformType, int] = field(default_factory=dict)
    hashtags: List[str] = field(default_factory=list)
    emoji: List[str] = field(default_factory=list)
    media_placeholders: List[str] = field(default_factory=list)
    is_active: bool = True
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScheduledPost:
    """Structure for a scheduled post."""
    post_id: str
    content: str
    platform: PlatformType
    account_id: str
    scheduled_time: datetime
    content_type: ContentType
    media_urls: List[str] = field(default_factory=list)
    status: str = "pending"
    is_recurring: bool = False
    recurrence_pattern: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    campaign_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PostAnalytics:
    """Structure for post analytics data."""
    post_id: str
    platform: PlatformType
    account_id: str
    post_url: str
    timestamp: datetime
    impressions: int = 0
    reach: int = 0
    engagement: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    clicks: int = 0
    profile_visits: int = 0
    follows: int = 0
    unfollows: int = 0
    sentiment_score: float = 0.0
    engagement_rate: float = 0.0
    demographic_data: Dict[str, Any] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendingTopic:
    """Structure for a trending topic."""
    topic: str
    category: TrendCategory
    volume: int
    sentiment: float
    momentum: float
    timestamp: datetime
    related_hashtags: List[str] = field(default_factory=list)
    related_keywords: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    geographic_data: Dict[str, Any] = field(default_factory=dict)
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CommunityMember:
    """Structure for a community member."""
    user_id: str
    platform: PlatformType
    username: str
    joined_date: datetime
    engagement_level: float = 0.0
    sentiment: float = 0.0
    influence_score: float = 0.0
    topics_of_interest: List[str] = field(default_factory=list)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Campaign:
    """Structure for a marketing campaign."""
    campaign_id: str
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    platforms: List[PlatformType]
    budget: float = 0.0
    target_audience: Dict[str, Any] = field(default_factory=dict)
    kpis: Dict[str, Any] = field(default_factory=dict)
    content_calendar: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "draft"
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemeTemplate:
    """Structure for a meme template."""
    template_id: str
    name: str
    image_url: str
    text_regions: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    popularity: float = 0.0
    example_texts: List[List[str]] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)

# --- Platform Integration Classes ---

class SocialMediaPlatform(ABC):
    """Abstract base class for social media platform integrations."""
    
    def __init__(self, account: SocialMediaAccount):
        """Initialize the platform integration.
        
        Args:
            account: Social media account information
        """
        self.account = account
        self.client = None
        self.connected = False
        self.rate_limit_remaining = {}
        self.rate_limit_reset = {}
        self.logger = logging.getLogger(f"social_media.{account.platform.name.lower()}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the platform API.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the platform API.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def post_content(self, content: str, content_type: ContentType, media_urls: List[str] = None) -> Dict[str, Any]:
        """Post content to the platform.
        
        Args:
            content: Content text to post
            content_type: Type of content
            media_urls: Optional list of media URLs to attach
            
        Returns:
            Dictionary with post information
        """
        pass
    
    @abstractmethod
    async def get_analytics(self, post_id: str) -> PostAnalytics:
        """Get analytics for a specific post.
        
        Args:
            post_id: ID of the post to get analytics for
            
        Returns:
            PostAnalytics object with analytics data
        """
        pass
    
    @abstractmethod
    async def get_trends(self) -> List[TrendingTopic]:
        """Get trending topics on the platform.
        
        Returns:
            List of trending topics
        """
        pass
    
    @abstractmethod
    async def engage(self, post_id: str, engagement_type: EngagementType, comment_text: str = None) -> bool:
        """Engage with a post.
        
        Args:
            post_id: ID of the post to engage with
            engagement_type: Type of engagement
            comment_text: Optional comment text for comment engagements
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_community_data(self) -> List[CommunityMember]:
        """Get data about the community/followers.
        
        Returns:
            List of community members
        """
        pass
    
    def check_rate_limits(self, endpoint: str) -> bool:
        """Check if we're within rate limits for an endpoint.
        
        Args:
            endpoint: API endpoint to check
            
        Returns:
            True if we can proceed, False if rate limited
        """
        if endpoint not in self.rate_limit_remaining or endpoint not in self.rate_limit_reset:
            return True
        
        if self.rate_limit_remaining[endpoint] <= 0:
            current_time = time.time()
            if current_time < self.rate_limit_reset[endpoint]:
                self.logger.warning(f"Rate limited for endpoint {endpoint}. Try again in {self.rate_limit_reset[endpoint] - current_time:.1f} seconds.")
                return False
        
        return True
    
    def update_rate_limits(self, endpoint: str, remaining: int, reset_time: int) -> None:
        """Update rate limit information for an endpoint.
        
        Args:
            endpoint: API endpoint
            remaining: Number of requests remaining
            reset_time: Timestamp when the rate limit resets
        """
        self.rate_limit_remaining[endpoint] = remaining
        self.rate_limit_reset[endpoint] = reset_time
        
        if remaining <= 5:
            self.logger.warning(f"Rate limit for {endpoint} is low: {remaining} requests remaining until {datetime.fromtimestamp(reset_time)}")

class TwitterPlatform(SocialMediaPlatform):
    """Twitter/X platform integration."""
    
    def __init__(self, account: SocialMediaAccount):
        """Initialize the Twitter platform integration.
        
        Args:
            account: Twitter account information
        """
        super().__init__(account)
        self.api = None
        self.client = None
    
    async def connect(self) -> bool:
        """Connect to the Twitter API.
        
        Returns:
            True if successful, False otherwise
        """
        if not TWITTER_AVAILABLE:
            self.logger.error("Twitter integration not available. Install tweepy package.")
            return False
        
        try:
            # Set up authentication
            auth = tweepy.OAuth1UserHandler(
                self.account.api_key,
                self.account.api_secret,
                self.account.access_token,
                self.account.access_secret
            )
            
            # Create API object
            self.api = tweepy.API(auth)
            
            # Create Client for v2 endpoints
            self.client = tweepy.Client(
                consumer_key=self.account.api_key,
                consumer_secret=self.account.api_secret,
                access_token=self.account.access_token,
                access_token_secret=self.account.access_secret
            )
            
            # Verify credentials
            self.api.verify_credentials()
            
            self.connected = True
            self.logger.info(f"Connected to Twitter as @{self.account.username}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error connecting to Twitter: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from the Twitter API.
        
        Returns:
            True if successful, False otherwise
        """
        self.api = None
        self.client = None
        self.connected = False
        self.logger.info("Disconnected from Twitter")
        return True
    
    async def post_content(self, content: str, content_type: ContentType, media_urls: List[str] = None) -> Dict[str, Any]:
        """Post content to Twitter.
        
        Args:
            content: Content text to post
            content_type: Type of content
            media_urls: Optional list of media URLs to attach
            
        Returns:
            Dictionary with post information
        """
        if not self.connected:
            self.logger.error("Not connected to Twitter")
            return {"error": "Not connected"}
        
        try:
            # Check rate limits
            if not self.check_rate_limits("statuses/update"):
                return {"error": "Rate limited"}
            
            # Handle different content types
            if content_type == ContentType.TEXT:
                # Simple text tweet
                status = self.api.update_status(content)
                
            elif content_type == ContentType.IMAGE and media_urls:
                # Tweet with media
                media_ids = []
                for media_url in media_urls[:4]:  # Twitter allows up to 4 images
                    if media_url.startswith(('http://', 'https://')):
                        # Download the image
                        response = requests.get(media_url)
                        if response.status_code == 200:
                            media = self.api.media_upload(filename="media.jpg", file=BytesIO(response.content))
                            media_ids.append(media.media_id)
                    else:
                        # Local file
                        media = self.api.media_upload(filename=media_url)
                        media_ids.append(media.media_id)
                
                status = self.api.update_status(content, media_ids=media_ids)
                
            elif content_type == ContentType.THREAD:
                # Create a thread of tweets
                tweets = content.split("\n---\n")
                previous_tweet_id = None
                statuses = []
                
                for tweet in tweets:
                    if previous_tweet_id:
                        status = self.api.update_status(
                            status=tweet,
                            in_reply_to_status_id=previous_tweet_id,
                            auto_populate_reply_metadata=True
                        )
                    else:
                        status = self.api.update_status(tweet)
                    
                    previous_tweet_id = status.id
                    statuses.append(status)
                
                # Return the last status
                status = statuses[-1]
                
            elif content_type == ContentType.POLL:
                # Create a poll (using v2 endpoint)
                options = content.split("\n")
                question = options.pop(0)
                
                if len(options) < 2 or len(options) > 4:
                    return {"error": "Polls require 2-4 options"}
                
                poll_duration = 1440  # 24 hours in minutes
                response = self.client.create_tweet(
                    text=question,
                    poll_options=options,
                    poll_duration_minutes=poll_duration
                )
                
                # Convert to status-like object
                status = type('obj', (object,), {
                    'id': response.data['id'],
                    'text': question,
                    'created_at': datetime.now()
                })
                
            else:
                return {"error": f"Unsupported content type: {content_type}"}
            
            # Update rate limits
            self.update_rate_limits(
                "statuses/update",
                self.api.rate_limit_status()['resources']['statuses']['/statuses/update']['remaining'],
                self.api.rate_limit_status()['resources']['statuses']['/statuses/update']['reset']
            )
            
            # Create result
            result = {
                "post_id": str(status.id),
                "platform": PlatformType.TWITTER.name,
                "account_id": self.account.account_id,
                "content": content,
                "content_type": content_type.name,
                "timestamp": datetime.now().isoformat(),
                "url": f"https://twitter.com/{self.account.username}/status/{status.id}"
            }
            
            self.logger.info(f"Posted to Twitter: {result['url']}")
            return result
        
        except Exception as e:
            self.logger.error(f"Error posting to Twitter: {str(e)}")
            return {"error": str(e)}
    
    async def get_analytics(self, post_id: str) -> PostAnalytics:
        """Get analytics for a specific tweet.
        
        Args:
            post_id: ID of the tweet to get analytics for
            
        Returns:
            PostAnalytics object with analytics data
        """
        if not self.connected:
            self.logger.error("Not connected to Twitter")
            return PostAnalytics(
                post_id=post_id,
                platform=PlatformType.TWITTER,
                account_id=self.account.account_id,
                post_url="",
                timestamp=datetime.now()
            )
        
        try:
            # Get tweet data
            tweet = self.api.get_status(post_id, tweet_mode="extended")
            
            # Get engagement metrics
            likes = tweet.favorite_count
            retweets = tweet.retweet_count
            
            # For detailed metrics, Twitter's API requires Premium or Enterprise access
            # This is a simplified version using available metrics
            
            # Create analytics object
            analytics = PostAnalytics(
                post_id=post_id,
                platform=PlatformType.TWITTER,
                account_id=self.account.account_id,
                post_url=f"https://twitter.com/{self.account.username}/status/{post_id}",
                timestamp=datetime.now(),
                likes=likes,
                shares=retweets,
                engagement=likes + retweets,
                engagement_rate=0.0  # Would need impression data to calculate
            )
            
            return analytics
        
        except Exception as e:
            self.logger.error(f"Error getting Twitter analytics: {str(e)}")
            return PostAnalytics(
                post_id=post_id,
                platform=PlatformType.TWITTER,
                account_id=self.account.account_id,
                post_url="",
                timestamp=datetime.now()
            )
    
    async def get_trends(self) -> List[TrendingTopic]:
        """Get trending topics on Twitter.
        
        Returns:
            List of trending topics
        """
        if not self.connected:
            self.logger.error("Not connected to Twitter")
            return []
        
        try:
            # Get worldwide trends (woeid 1 is worldwide)
            trends = self.api.get_place_trends(id=1)
            
            # Convert to TrendingTopic objects
            trending_topics = []
            for trend in trends[0]["trends"][:10]:  # Get top 10 trends
                # Determine category (simplified)
                category = TrendCategory.TECHNOLOGY
                if any(keyword in trend["name"].lower() for keyword in ["crypto", "bitcoin", "ethereum", "nft"]):
                    category = TrendCategory.CRYPTO
                elif any(keyword in trend["name"].lower() for keyword in ["finance", "stock", "market", "economy"]):
                    category = TrendCategory.FINANCE
                elif any(keyword in trend["name"].lower() for keyword in ["meme", "funny", "lol"]):
                    category = TrendCategory.MEMES
                
                # Create trending topic
                topic = TrendingTopic(
                    topic=trend["name"],
                    category=category,
                    volume=trend["tweet_volume"] if trend["tweet_volume"] else 0,
                    sentiment=0.0,  # Would need to analyze tweets to determine
                    momentum=0.0,  # Would need historical data to determine
                    timestamp=datetime.now(),
                    sources=["Twitter"]
                )
                
                trending_topics.append(topic)
            
            return trending_topics
        
        except Exception as e:
            self.logger.error(f"Error getting Twitter trends: {str(e)}")
            return []
    
    async def engage(self, post_id: str, engagement_type: EngagementType, comment_text: str = None) -> bool:
        """Engage with a tweet.
        
        Args:
            post_id: ID of the tweet to engage with
            engagement_type: Type of engagement
            comment_text: Optional comment text for replies
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            self.logger.error("Not connected to Twitter")
            return False
        
        try:
            if engagement_type == EngagementType.LIKE:
                self.api.create_favorite(post_id)
                self.logger.info(f"Liked tweet {post_id}")
                return True
                
            elif engagement_type == EngagementType.RETWEET:
                self.api.retweet(post_id)
                self.logger.info(f"Retweeted tweet {post_id}")
                return True
                
            elif engagement_type == EngagementType.REPLY and comment_text:
                status = self.api.update_status(
                    status=comment_text,
                    in_reply_to_status_id=post_id,
                    auto_populate_reply_metadata=True
                )
                self.logger.info(f"Replied to tweet {post_id}: {status.id}")
                return True
                
            elif engagement_type == EngagementType.DIRECT_MESSAGE and comment_text:
                # Get the user ID from the tweet
                tweet = self.api.get_status(post_id)
                user_id = tweet.user.id
                
                # Send direct message
                self.api.send_direct_message(user_id, comment_text)
                self.logger.info(f"Sent DM to user of tweet {post_id}")
                return True
                
            else:
                self.logger.warning(f"Unsupported engagement type: {engagement_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error engaging with Twitter post: {str(e)}")
            return False
    
    async def get_community_data(self) -> List[CommunityMember]:
        """Get data about Twitter followers.
        
        Returns:
            List of community members
        """
        if not self.connected:
            self.logger.error("Not connected to Twitter")
            return []
        
        try:
            # Get follower IDs (limited by API)
            follower_ids = self.api.get_follower_ids(screen_name=self.account.username)
            
            # Get detailed user info for each follower (limited to avoid rate limits)
            community_members = []
            for user_id in follower_ids[:100]:  # Limit to 100 to avoid rate limits
                try:
                    user = self.api.get_user(user_id=user_id)
                    
                    # Calculate influence score (simplified)
                    followers = user.followers_count
                    following = user.friends_count
                    influence = min(1.0, followers / (following + 1) / 100) if following > 0 else 0.0
                    
                    # Create community member
                    member = CommunityMember(
                        user_id=str(user.id),
                        platform=PlatformType.TWITTER,
                        username=user.screen_name,
                        joined_date=user.created_at,
                        influence_score=influence,
                        topics_of_interest=[],  # Would need to analyze tweets to determine
                        tags=[]
                    )
                    
                    community_members.append(member)
                except Exception as e:
                    self.logger.warning(f"Error getting user data for {user_id}: {str(e)}")
            
            return community_members
        
        except Exception as e:
            self.logger.error(f"Error getting Twitter community data: {str(e)}")
            return []

class DiscordPlatform(SocialMediaPlatform):
    """Discord platform integration."""
    
    def __init__(self, account: SocialMediaAccount):
        """Initialize the Discord platform integration.
        
        Args:
            account: Discord account information
        """
        super().__init__(account)
        self.bot = None
        self.channels = {}
    
    async def connect(self) -> bool:
        """Connect to the Discord API.
        
        Returns:
            True if successful, False otherwise
        """
        if not DISCORD_AVAILABLE:
            self.logger.error("Discord integration not available. Install discord.py package.")
            return False
        
        try:
            # Set up intents
            intents = discord.Intents.default()
            intents.message_content = True
            intents.members = True
            
            # Create bot
            self.bot = commands.Bot(command_prefix="!", intents=intents)
            
            # Set up event handlers
            @self.bot.event
            async def on_ready():
                self.logger.info(f"Connected to Discord as {self.bot.user}")
                self.connected = True
                
                # Cache channels
                for guild in self.bot.guilds:
                    for channel in guild.text_channels:
                        self.channels[channel.name] = channel.id
            
            # Start bot in a separate thread
            threading.Thread(target=self._run_bot, args=(self.account.auth_token,), daemon=True).start()
            
            # Wait for bot to connect
            start_time = time.time()
            while not self.connected and time.time() - start_time < 30:
                await asyncio.sleep(1)
            
            return self.connected
        
        except Exception as e:
            self.logger.error(f"Error connecting to Discord: {str(e)}")
            self.connected = False
            return False
    
    def _run_bot(self, token: str) -> None:
        """Run the Discord bot in a separate thread.
        
        Args:
            token: Discord bot token
        """
        try:
            asyncio.run(self.bot.start(token))
        except Exception as e:
            self.logger.error(f"Error running Discord bot: {str(e)}")
    
    async def disconnect(self) -> bool:
        """Disconnect from the Discord API.
        
        Returns:
            True if successful, False otherwise
        """
        if self.bot:
            await self.bot.close()
        
        self.bot = None
        self.connected = False
        self.logger.info("Disconnected from Discord")
        return True
    
    async def post_content(self, content: str, content_type: ContentType, media_urls: List[str] = None, channel_name: str = None) -> Dict[str, Any]:
        """Post content to Discord.
        
        Args:
            content: Content text to post
            content_type: Type of content
            media_urls: Optional list of media URLs to attach
            channel_name: Optional channel name to post to (default: first available)
            
        Returns:
            Dictionary with post information
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Discord")
            return {"error": "Not connected"}
        
        try:
            # Get channel
            channel_id = None
            if channel_name and channel_name in self.channels:
                channel_id = self.channels[channel_name]
            else:
                # Use first available channel
                if self.channels:
                    channel_id = next(iter(self.channels.values()))
            
            if not channel_id:
                self.logger.error("No Discord channel available")
                return {"error": "No channel available"}
            
            channel = self.bot.get_channel(channel_id)
            if not channel:
                self.logger.error(f"Discord channel {channel_id} not found")
                return {"error": "Channel not found"}
            
            # Handle different content types
            if content_type == ContentType.TEXT:
                # Simple text message
                message = await channel.send(content)
                
            elif content_type == ContentType.IMAGE and media_urls:
                # Message with media
                files = []
                for media_url in media_urls:
                    if media_url.startswith(('http://', 'https://')):
                        # Download the image
                        response = requests.get(media_url)
                        if response.status_code == 200:
                            file = discord.File(BytesIO(response.content), filename="media.jpg")
                            files.append(file)
                    else:
                        # Local file
                        file = discord.File(media_url)
                        files.append(file)
                
                message = await channel.send(content=content, files=files)
                
            elif content_type == ContentType.EMBED:
                # Create embed
                embed = discord.Embed(
                    title="Embedded Content",
                    description=content,
                    color=discord.Color.blue()
                )
                
                # Add image if available
                if media_urls and len(media_urls) > 0:
                    embed.set_image(url=media_urls[0])
                
                message = await channel.send(embed=embed)
                
            else:
                return {"error": f"Unsupported content type: {content_type}"}
            
            # Create result
            result = {
                "post_id": str(message.id),
                "platform": PlatformType.DISCORD.name,
                "account_id": self.account.account_id,
                "content": content,
                "content_type": content_type.name,
                "timestamp": datetime.now().isoformat(),
                "channel_id": str(channel_id),
                "channel_name": channel.name
            }
            
            self.logger.info(f"Posted to Discord channel {channel.name}")
            return result
        
        except Exception as e:
            self.logger.error(f"Error posting to Discord: {str(e)}")
            return {"error": str(e)}
    
    async def get_analytics(self, post_id: str, channel_id: str = None) -> PostAnalytics:
        """Get analytics for a specific Discord message.
        
        Args:
            post_id: ID of the message to get analytics for
            channel_id: Optional channel ID where the message was posted
            
        Returns:
            PostAnalytics object with analytics data
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Discord")
            return PostAnalytics(
                post_id=post_id,
                platform=PlatformType.DISCORD,
                account_id=self.account.account_id,
                post_url="",
                timestamp=datetime.now()
            )
        
        try:
            # Find the message
            message = None
            
            if channel_id:
                channel = self.bot.get_channel(int(channel_id))
                if channel:
                    try:
                        message = await channel.fetch_message(int(post_id))
                    except:
                        pass
            
            if not message:
                # Try all channels
                for channel_id in self.channels.values():
                    channel = self.bot.get_channel(channel_id)
                    if channel:
                        try:
                            message = await channel.fetch_message(int(post_id))
                            if message:
                                break
                        except:
                            continue
            
            if not message:
                self.logger.error(f"Discord message {post_id} not found")
                return PostAnalytics(
                    post_id=post_id,
                    platform=PlatformType.DISCORD,
                    account_id=self.account.account_id,
                    post_url="",
                    timestamp=datetime.now()
                )
            
            # Get reactions (Discord's equivalent to likes)
            reactions = sum(reaction.count for reaction in message.reactions)
            
            # Create analytics object
            analytics = PostAnalytics(
                post_id=post_id,
                platform=PlatformType.DISCORD,
                account_id=self.account.account_id,
                post_url="",  # Discord doesn't have public URLs for messages
                timestamp=datetime.now(),
                likes=reactions,
                engagement=reactions
            )
            
            return analytics
        
        except Exception as e:
            self.logger.error(f"Error getting Discord analytics: {str(e)}")
            return PostAnalytics(
                post_id=post_id,
                platform=PlatformType.DISCORD,
                account_id=self.account.account_id,
                post_url="",
                timestamp=datetime.now()
            )
    
    async def get_trends(self) -> List[TrendingTopic]:
        """Get trending topics on Discord.
        
        Returns:
            List of trending topics
        """
        # Discord doesn't have a concept of trending topics like Twitter
        # This is a simplified implementation that analyzes recent messages
        
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Discord")
            return []
        
        try:
            trending_topics = []
            word_counts = defaultdict(int)
            
            # Analyze recent messages in all channels
            for channel_id in self.channels.values():
                channel = self.bot.get_channel(channel_id)
                if not channel:
                    continue
                
                try:
                    # Get recent messages
                    messages = [message async for message in channel.history(limit=100)]
                    
                    # Extract words and count occurrences
                    for message in messages:
                        if message.author == self.bot.user:
                            continue
                        
                        # Split into words and normalize
                        words = message.content.lower().split()
                        
                        # Count words (excluding common words)
                        common_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "of", "in", "for", "with", "on", "at", "by", "about", "like", "through", "over", "before", "between", "after", "since", "without", "under", "within", "along", "following", "across", "behind", "beyond", "plus", "except", "but", "up", "out", "around", "down", "off", "above", "near"}
                        for word in words:
                            if len(word) > 3 and word not in common_words:
                                word_counts[word] += 1
                except Exception as e:
                    self.logger.warning(f"Error analyzing messages in channel {channel.name}: {str(e)}")
            
            # Get top words
            top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create trending topics
            for word, count in top_words:
                # Determine category (simplified)
                category = TrendCategory.TECHNOLOGY
                if any(keyword in word.lower() for keyword in ["crypto", "bitcoin", "ethereum", "nft"]):
                    category = TrendCategory.CRYPTO
                elif any(keyword in word.lower() for keyword in ["finance", "stock", "market", "economy"]):
                    category = TrendCategory.FINANCE
                elif any(keyword in word.lower() for keyword in ["meme", "funny", "lol"]):
                    category = TrendCategory.MEMES
                
                # Create trending topic
                topic = TrendingTopic(
                    topic=word,
                    category=category,
                    volume=count,
                    sentiment=0.0,
                    momentum=0.0,
                    timestamp=datetime.now(),
                    sources=["Discord"]
                )
                
                trending_topics.append(topic)
            
            return trending_topics
        
        except Exception as e:
            self.logger.error(f"Error getting Discord trends: {str(e)}")
            return []
    
    async def engage(self, post_id: str, engagement_type: EngagementType, comment_text: str = None, channel_id: str = None) -> bool:
        """Engage with a Discord message.
        
        Args:
            post_id: ID of the message to engage with
            engagement_type: Type of engagement
            comment_text: Optional comment text for replies
            channel_id: Optional channel ID where the message was posted
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Discord")
            return False
        
        try:
            # Find the message
            message = None
            
            if channel_id:
                channel = self.bot.get_channel(int(channel_id))
                if channel:
                    try:
                        message = await channel.fetch_message(int(post_id))
                    except:
                        pass
            
            if not message:
                # Try all channels
                for channel_id in self.channels.values():
                    channel = self.bot.get_channel(channel_id)
                    if channel:
                        try:
                            message = await channel.fetch_message(int(post_id))
                            if message:
                                break
                        except:
                            continue
            
            if not message:
                self.logger.error(f"Discord message {post_id} not found")
                return False
            
            if engagement_type == EngagementType.LIKE:
                # Add reaction (thumbs up)
                await message.add_reaction("👍")
                self.logger.info(f"Added reaction to Discord message {post_id}")
                return True
                
            elif engagement_type == EngagementType.REPLY and comment_text:
                # Reply to message
                await message.reply(comment_text)
                self.logger.info(f"Replied to Discord message {post_id}")
                return True
                
            else:
                self.logger.warning(f"Unsupported engagement type: {engagement_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error engaging with Discord message: {str(e)}")
            return False
    
    async def get_community_data(self) -> List[CommunityMember]:
        """Get data about Discord server members.
        
        Returns:
            List of community members
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Discord")
            return []
        
        try:
            community_members = []
            
            # Process each guild (server)
            for guild in self.bot.guilds:
                # Get members
                for member in guild.members:
                    if member.bot:
                        continue
                    
                    # Create community member
                    community_member = CommunityMember(
                        user_id=str(member.id),
                        platform=PlatformType.DISCORD,
                        username=f"{member.name}#{member.discriminator}",
                        joined_date=member.joined_at if member.joined_at else datetime.now(),
                        influence_score=0.0,  # Discord doesn't have a direct influence metric
                        tags=[role.name for role in member.roles if role.name != "@everyone"]
                    )
                    
                    community_members.append(community_member)
            
            return community_members
        
        except Exception as e:
            self.logger.error(f"Error getting Discord community data: {str(e)}")
            return []

class TelegramPlatform(SocialMediaPlatform):
    """Telegram platform integration."""
    
    def __init__(self, account: SocialMediaAccount):
        """Initialize the Telegram platform integration.
        
        Args:
            account: Telegram account information
        """
        super().__init__(account)
        self.bot = None
        self.application = None
        self.chat_ids = {}
    
    async def connect(self) -> bool:
        """Connect to the Telegram API.
        
        Returns:
            True if successful, False otherwise
        """
        if not TELEGRAM_AVAILABLE:
            self.logger.error("Telegram integration not available. Install python-telegram-bot package.")
            return False
        
        try:
            # Create bot
            self.bot = Bot(token=self.account.auth_token)
            
            # Get bot info
            bot_info = await self.bot.get_me()
            self.account.username = bot_info.username
            
            # Create application
            self.application = ApplicationBuilder().token(self.account.auth_token).build()
            
            # Set up handlers
            self.application.add_handler(CommandHandler("start", self._start_command))
            self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._message_handler))
            
            # Start application in a separate thread
            threading.Thread(target=self._run_application, daemon=True).start()
            
            self.connected = True
            self.logger.info(f"Connected to Telegram as @{self.account.username}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error connecting to Telegram: {str(e)}")
            self.connected = False
            return False
    
    def _run_application(self) -> None:
        """Run the Telegram application in a separate thread."""
        try:
            asyncio.run(self.application.run_polling())
        except Exception as e:
            self.logger.error(f"Error running Telegram application: {str(e)}")
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        chat_id = update.effective_chat.id
        chat_title = update.effective_chat.title if update.effective_chat.title else update.effective_chat.first_name
        
        # Store chat ID
        self.chat_ids[chat_title] = chat_id
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Hello! I'm {self.account.username}, your AI assistant. How can I help you today?"
        )
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages."""
        chat_id = update.effective_chat.id
        chat_title = update.effective_chat.title if update.effective_chat.title else update.effective_chat.first_name
        
        # Store chat ID
        self.chat_ids[chat_title] = chat_id
    
    async def disconnect(self) -> bool:
        """Disconnect from the Telegram API.
        
        Returns:
            True if successful, False otherwise
        """
        if self.application:
            await self.application.stop()
        
        self.bot = None
        self.application = None
        self.connected = False
        self.logger.info("Disconnected from Telegram")
        return True
    
    async def post_content(self, content: str, content_type: ContentType, media_urls: List[str] = None, chat_name: str = None) -> Dict[str, Any]:
        """Post content to Telegram.
        
        Args:
            content: Content text to post
            content_type: Type of content
            media_urls: Optional list of media URLs to attach
            chat_name: Optional chat name to post to (default: first available)
            
        Returns:
            Dictionary with post information
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Telegram")
            return {"error": "Not connected"}
        
        try:
            # Get chat ID
            chat_id = None
            if chat_name and chat_name in self.chat_ids:
                chat_id = self.chat_ids[chat_name]
            else:
                # Use first available chat
                if self.chat_ids:
                    chat_id = next(iter(self.chat_ids.values()))
            
            if not chat_id:
                self.logger.error("No Telegram chat available")
                return {"error": "No chat available"}
            
            # Handle different content types
            if content_type == ContentType.TEXT:
                # Simple text message
                message = await self.bot.send_message(
                    chat_id=chat_id,
                    text=content,
                    parse_mode="Markdown"
                )
                
            elif content_type == ContentType.IMAGE and media_urls and len(media_urls) > 0:
                # Message with photo
                media_url = media_urls[0]
                
                if media_url.startswith(('http://', 'https://')):
                    # Send photo from URL
                    message = await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=media_url,
                        caption=content,
                        parse_mode="Markdown"
                    )
                else:
                    # Send photo from file
                    with open(media_url, "rb") as photo:
                        message = await self.bot.send_photo(
                            chat_id=chat_id,
                            photo=photo,
                            caption=content,
                            parse_mode="Markdown"
                        )
                
            elif content_type == ContentType.VIDEO and media_urls and len(media_urls) > 0:
                # Message with video
                media_url = media_urls[0]
                
                if media_url.startswith(('http://', 'https://')):
                    # Send video from URL
                    message = await self.bot.send_video(
                        chat_id=chat_id,
                        video=media_url,
                        caption=content,
                        parse_mode="Markdown"
                    )
                else:
                    # Send video from file
                    with open(media_url, "rb") as video:
                        message = await self.bot.send_video(
                            chat_id=chat_id,
                            video=video,
                            caption=content,
                            parse_mode="Markdown"
                        )
                
            elif content_type == ContentType.POLL:
                # Create a poll
                options = content.split("\n")
                question = options.pop(0)
                
                if len(options) < 2 or len(options) > 10:
                    return {"error": "Polls require 2-10 options"}
                
                message = await self.bot.send_poll(
                    chat_id=chat_id,
                    question=question,
                    options=options,
                    is_anonymous=False
                )
                
            else:
                return {"error": f"Unsupported content type: {content_type}"}
            
            # Create result
            result = {
                "post_id": str(message.message_id),
                "platform": PlatformType.TELEGRAM.name,
                "account_id": self.account.account_id,
                "content": content,
                "content_type": content_type.name,
                "timestamp": datetime.now().isoformat(),
                "chat_id": str(chat_id)
            }
            
            self.logger.info(f"Posted to Telegram chat {chat_id}")
            return result
        
        except Exception as e:
            self.logger.error(f"Error posting to Telegram: {str(e)}")
            return {"error": str(e)}
    
    async def get_analytics(self, post_id: str, chat_id: str = None) -> PostAnalytics:
        """Get analytics for a specific Telegram message.
        
        Args:
            post_id: ID of the message to get analytics for
            chat_id: Optional chat ID where the message was posted
            
        Returns:
            PostAnalytics object with analytics data
        """
        # Telegram doesn't provide detailed analytics for messages
        # This is a placeholder implementation
        
        return PostAnalytics(
            post_id=post_id,
            platform=PlatformType.TELEGRAM,
            account_id=self.account.account_id,
            post_url="",
            timestamp=datetime.now()
        )
    
    async def get_trends(self) -> List[TrendingTopic]:
        """Get trending topics on Telegram.
        
        Returns:
            List of trending topics
        """
        # Telegram doesn't have a concept of trending topics
        # This is a placeholder implementation
        
        return []
    
    async def engage(self, post_id: str, engagement_type: EngagementType, comment_text: str = None, chat_id: str = None) -> bool:
        """Engage with a Telegram message.
        
        Args:
            post_id: ID of the message to engage with
            engagement_type: Type of engagement
            comment_text: Optional comment text for replies
            chat_id: Optional chat ID where the message was posted
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Telegram")
            return False
        
        try:
            # Need chat_id to engage with messages
            if not chat_id:
                self.logger.error("Chat ID required for Telegram engagement")
                return False
            
            if engagement_type == EngagementType.REPLY and comment_text:
                # Reply to message
                await self.bot.send_message(
                    chat_id=int(chat_id),
                    text=comment_text,
                    reply_to_message_id=int(post_id),
                    parse_mode="Markdown"
                )
                self.logger.info(f"Replied to Telegram message {post_id}")
                return True
                
            else:
                self.logger.warning(f"Unsupported engagement type: {engagement_type}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error engaging with Telegram message: {str(e)}")
            return False
    
    async def get_community_data(self) -> List[CommunityMember]:
        """Get data about Telegram chat members.
        
        Returns:
            List of community members
        """
        if not self.connected or not self.bot:
            self.logger.error("Not connected to Telegram")
            return []
        
        try:
            community_members = []
            
            # Process each chat
            for chat_name, chat_id in self.chat_ids.items():
                try:
                    # Get chat member count (basic info)
                    chat = await self.bot.get_chat(chat_id)
                    member_count = await self.bot.get_chat_member_count(chat_id)
                    
                    # Create a generic community member for the chat
                    community_member = CommunityMember(
                        user_id=str(chat_id),
                        platform=PlatformType.TELEGRAM,
                        username=chat_name,
                        joined_date=datetime.now(),
                        influence_score=0.0,
                        tags=["chat"],
                        additional_data={"member_count": member_count}
                    )
                    
                    community_members.append(community_member)
                except Exception as e:
                    self.logger.warning(f"Error getting members for chat {chat_name}: {str(e)}")
            
            return community_members
        
        except Exception as e:
            self.logger.error(f"Error getting Telegram community data: {str(e)}")
            return []

# --- Content Generation ---

class ContentGenerator:
    """Content generator for social media posts."""
    
    def __init__(self):
        """Initialize the content generator."""
        self.templates = {}
        self.personalities = {}
        self.meme_templates = {}
        self.sia = None
        
        # Initialize sentiment analyzer if available
        if NLP_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
            except Exception as e:
                logger.error(f"Error initializing sentiment analyzer: {str(e)}")
    
    def add_template(self, template: ContentTemplate) -> None:
        """Add a content template.
        
        Args:
            template: Content template to add
        """
        self.templates[template.template_id] = template
        logger.info(f"Added content template: {template.name}")
    
    def add_personality(self, personality: AgentPersonality) -> None:
        """Add an agent personality.
        
        Args:
            personality: Agent personality to add
        """
        self.personalities[personality.name] = personality
        logger.info(f"Added agent personality: {personality.name}")
    
    def add_meme_template(self, template: MemeTemplate) -> None:
        """Add a meme template.
        
        Args:
            template: Meme template to add
        """
        self.meme_templates[template.template_id] = template
        logger.info(f"Added meme template: {template.name}")
    
    async def generate_content(self, prompt: str, personality_name: str = None, platform: PlatformType = None, content_type: ContentType = ContentType.TEXT, max_length: int = 280) -> str:
        """Generate content based on a prompt and personality.
        
        Args:
            prompt: Content generation prompt
            personality_name: Optional personality to use
            platform: Optional platform to generate content for
            content_type: Type of content to generate
            max_length: Maximum content length
            
        Returns:
            Generated content
        """
        try:
            # Get personality
            personality = None
            if personality_name and personality_name in self.personalities:
                personality = self.personalities[personality_name]
            
            # Build system prompt
            system_prompt = "You are a social media content creator. Create engaging content that is concise and compelling."
            
            if personality:
                system_prompt += f"\n\nPersonality: {personality.name}"
                system_prompt += f"\nTone: {personality.tone}"
                system_prompt += f"\nVoice: {personality.voice}"
                
                if personality.system_prompt:
                    system_prompt += f"\n\n{personality.system_prompt}"
                
                if personality.emoji_usage > 0:
                    system_prompt += f"\n\nUse emojis at a frequency of {personality.emoji_usage * 100}%."
                    if personality.emoji_set:
                        system_prompt += f" Preferred emojis: {', '.join(personality.emoji_set)}"
                
                if personality.hashtag_usage > 0:
                    system_prompt += f"\n\nUse hashtags at a frequency of {personality.hashtag_usage * 100}%."
                    if personality.hashtag_set:
                        system_prompt += f" Preferred hashtags: {', '.join(personality.hashtag_set)}"
                
                if personality.taboo_words:
                    system_prompt += f"\n\nAvoid using these words: {', '.join(personality.taboo_words)}"
                
                if personality.preferred_topics:
                    system_prompt += f"\n\nPreferred topics: {', '.join(personality.preferred_topics)}"
                
                if personality.avoided_topics:
                    system_prompt += f"\n\nAvoided topics: {', '.join(personality.avoided_topics)}"
            
            if platform:
                system_prompt += f"\n\nPlatform: {platform.name}"
                
                # Add platform-specific instructions
                if platform == PlatformType.TWITTER:
                    system_prompt += f"\nKeep content under {max_length} characters for Twitter."
                elif platform == PlatformType.DISCORD:
                    system_prompt += "\nFormat content appropriately for Discord, using markdown formatting if needed."
                elif platform == PlatformType.TELEGRAM:
                    system_prompt += "\nFormat content appropriately for Telegram, using markdown formatting if needed."
            
            if content_type:
                system_prompt += f"\n\nContent type: {content_type.name}"
                
                # Add content type-specific instructions
                if content_type == ContentType.THREAD:
                    system_prompt += "\nCreate a thread of tweets separated by '---' on new lines."
                elif content_type == ContentType.POLL:
                    system_prompt += "\nCreate a poll question followed by 2-4 options, each on a new line."
                elif content_type == ContentType.MEME:
                    system_prompt += "\nCreate a funny and engaging meme caption."
            
            # Generate content using GPT-4o if available
            if OPENAI_UNOFFICIAL_AVAILABLE:
                response = await openai_unofficial.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()
            
            elif OPENAI_AVAILABLE:
                response = await openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                content = response.choices[0].message.content.strip()
            
            else:
                # Fallback to template-based generation
                content = self._generate_from_template(prompt, personality, platform, content_type)
            
            # Apply personality-specific modifications
            if personality:
                content = self._apply_personality(content, personality)
            
            # Ensure content meets platform requirements
            if platform == PlatformType.TWITTER and len(content) > max_length:
                content = content[:max_length - 3] + "..."
            
            return content
        
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return f"Error generating content: {str(e)}"
    
    def _generate_from_template(self, prompt: str, personality: AgentPersonality = None, platform: PlatformType = None, content_type: ContentType = ContentType.TEXT) -> str:
        """Generate content using templates when AI generation is not available.
        
        Args:
            prompt: Content generation prompt
            personality: Optional personality to use
            platform: Optional platform to generate content for
            content_type: Type of content to generate
            
        Returns:
            Generated content
        """
        # Find suitable templates
        suitable_templates = []
        for template in self.templates.values():
            if not template.is_active:
                continue
            
            if content_type and template.content_type != content_type:
                continue
            
            if platform and platform not in template.platforms:
                continue
            
            suitable_templates.append(template)
        
        if not suitable_templates:
            return f"No suitable template found for {content_type.name} on {platform.name if platform else 'any platform'}"
        
        # Select a random template
        template = random.choice(suitable_templates)
        
        # Fill in variables
        content = template.template_text
        
        # Replace variables with prompt keywords
        words = prompt.split()
        for variable in template.variables:
            if words:
                content = content.replace(f"{{{variable}}}", random.choice(words))
            else:
                content = content.replace(f"{{{variable}}}", variable)
        
        # Add hashtags
        if template.hashtags:
            hashtags = " ".join(random.sample(template.hashtags, min(3, len(template.hashtags))))
            content = f"{content}\n\n{hashtags}"
        
        # Add emoji
        if template.emoji:
            emoji = random.choice(template.emoji)
            content = f"{emoji} {content}"
        
        return content
    
    def _apply_personality(self, content: str, personality: AgentPersonality) -> str:
        """Apply personality traits to content.
        
        Args:
            content: Content to modify
            personality: Personality to apply
            
        Returns:
            Modified content
        """
        # Add emojis based on personality
        if personality.emoji_usage > 0 and personality.emoji_set:
            words = content.split()
            for i in range(len(words)):
                if random.random() < personality.emoji_usage:
                    words[i] = f"{words[i]} {random.choice(personality.emoji_set)}"
            content = " ".join(words)
        
        # Add hashtags based on personality
        if personality.hashtag_usage > 0 and personality.hashtag_set:
            hashtag_count = max(1, int(len(content.split()) * personality.hashtag_usage / 5))
            hashtags = random.sample(personality.hashtag_set, min(hashtag_count, len(personality.hashtag_set)))
            content = f"{content}\n\n{' '.join(hashtags)}"
        
        # Replace words based on vocabulary
        if personality.vocabulary:
            for category, words in personality.vocabulary.items():
                for word in words:
                    if word.lower() in content.lower():
                        replacement = random.choice(words)
                        content = re.sub(r'\b' + re.escape(word) + r'\b', replacement, content, flags=re.IGNORECASE)
        
        return content
    
    async def generate_meme(self, prompt: str, template_id: str = None) -> Tuple[str, str]:
        """Generate a meme based on a prompt.
        
        Args:
            prompt: Meme generation prompt
            template_id: Optional meme template ID
            
        Returns:
            Tuple of (caption, image_path)
        """
        if not IMAGE_PROCESSING_AVAILABLE:
            logger.error("Image processing not available. Cannot generate meme.")
            return "Image processing not available", None
        
        try:
            # Get template
            template = None
            if template_id and template_id in self.meme_templates:
                template = self.meme_templates[template_id]
            else:
                # Select a random template
                if self.meme_templates:
                    template = random.choice(list(self.meme_templates.values()))
            
            if not template:
                logger.error("No meme template available")
                return "No meme template available", None
            
            # Generate caption using GPT-4o
            caption = await self.generate_content(
                prompt=f"Create a funny meme caption for the template '{template.name}'. Prompt: {prompt}",
                content_type=ContentType.MEME
            )
            
            # Generate text for each region
            texts = []
            if template.example_texts:
                # Use example texts as inspiration
                example = random.choice(template.example_texts)
                
                # Generate text for each region
                for i, region in enumerate(template.text_regions):
                    if i < len(example):
                        texts.append(example[i])
                    else:
                        texts.append("")
            else:
                # Generate text for each region
                for i, region in enumerate(template.text_regions):
                    if i == 0:
                        texts.append(caption.split('\n')[0] if '\n' in caption else caption)
                    elif i == 1 and '\n' in caption:
                        texts.append(caption.split('\n')[1])
                    else:
                        texts.append("")
            
            # Create meme
            image_path = self._create_meme_image(template.image_url, texts, template.text_regions)
            
            return caption, image_path
        
        except Exception as e:
            logger.error(f"Error generating meme: {str(e)}")
            return f"Error generating meme: {str(e)}", None
    
    def _create_meme_image(self, image_url: str, texts: List[str], regions: List[Dict[str, Any]]) -> str:
        """Create a meme image.
        
        Args:
            image_url: URL or path to the base image
            texts: List of text strings for each region
            regions: List of region definitions
            
        Returns:
            Path to the generated meme image
        """
        try:
            # Load image
            if image_url.startswith(('http://', 'https://')):
                response = requests.get(image_url)
                if response.status_code != 200:
                    raise ValueError(f"Failed to download image: {response.status_code}")
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_url)
            
            # Create draw object
            draw = ImageDraw.Draw(image)
            
            # Add text to each region
            for i, (text, region) in enumerate(zip(texts, regions)):
                if not text:
                    continue
                
                # Get region parameters
                x = region.get("x", 0)
                y = region.get("y", 0)
                width = region.get("width", image.width)
                height = region.get("height", 100)
                font_size = region.get("font_size", 36)
                color = region.get("color", "white")
                stroke_width = region.get("stroke_width", 2)
                stroke_fill = region.get("stroke_fill", "black")
                
                # Load font
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Draw text with stroke
                draw.text((x, y), text, fill=color, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)
            
            # Save image
            output_path = f"temp/meme_{int(time.time())}_{uuid.uuid4().hex[:8]}.jpg"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating meme image: {str(e)}")
            return None
    
    async def translate_content(self, content: str, target_language: LanguageCode) -> str:
        """Translate content to a target language.
        
        Args:
            content: Content to translate
            target_language: Target language code
            
        Returns:
            Translated content
        """
        if not TRANSLATION_AVAILABLE:
            logger.error("Translation not available. Cannot translate content.")
            return content
        
        try:
            # Detect source language
            source_language = detect(content)
            
            # Skip if already in target language
            if source_language == target_language.value:
                return content
            
            # Translate using googletrans
            translator = Translator()
            translation = translator.translate(content, dest=target_language.value)
            
            return translation.text
        
        except Exception as e:
            logger.error(f"Error translating content: {str(e)}")
            return content
    
    def analyze_sentiment(self, content: str) -> float:
        """Analyze sentiment of content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not self.sia:
            logger.error("Sentiment analyzer not available")
            return 0.0
        
        try:
            sentiment = self.sia.polarity_scores(content)
            return sentiment["compound"]
        
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0

# --- Content Scheduler ---

class ContentScheduler:
    """Content scheduler for social media posts."""
    
    def __init__(self):
        """Initialize the content scheduler."""
        self.scheduled_posts = {}
        self.running = False
        self.thread = None
    
    def add_post(self, post: ScheduledPost) -> str:
        """Add a scheduled post.
        
        Args:
            post: Scheduled post to add
            
        Returns:
            Post ID
        """
        self.scheduled_posts[post.post_id] = post
        logger.info(f"Scheduled post {post.post_id} for {post.scheduled_time}")
        return post.post_id
    
    def remove_post(self, post_id: str) -> bool:
        """Remove a scheduled post.
        
        Args:
            post_id: ID of the post to remove
            
        Returns:
            True if successful, False otherwise
        """
        if post_id in self.scheduled_posts:
            del self.scheduled_posts[post_id]
            logger.info(f"Removed scheduled post {post_id}")
            return True
        else:
            logger.warning(f"Scheduled post {post_id} not found")
            return False
    
    def update_post(self, post_id: str, updates: Dict[str, Any]) -> bool:
        """Update a scheduled post.
        
        Args:
            post_id: ID of the post to update
            updates: Dictionary of updates
            
        Returns:
            True if successful, False otherwise
        """
        if post_id not in self.scheduled_posts:
            logger.warning(f"Scheduled post {post_id} not found")
            return False
        
        post = self.scheduled_posts[post_id]
        
        # Update fields
        for key, value in updates.items():
            if hasattr(post, key):
                setattr(post, key, value)
        
        logger.info(f"Updated scheduled post {post_id}")
        return True
    
    def get_post(self, post_id: str) -> Optional[ScheduledPost]: