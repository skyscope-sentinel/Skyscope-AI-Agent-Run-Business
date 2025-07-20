#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified AI Client for Skyscope Agent Swarm
==========================================

This module provides a unified interface to the openai-unofficial library,
enabling all 10,000 agents to access AI capabilities efficiently.
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("skyscope_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SkyscopeAI")

# Import openai-unofficial as the primary AI provider
try:
    import openai_unofficial
    from openai_unofficial import OpenAIUnofficial
    OPENAI_UNOFFICIAL_AVAILABLE = True
    logger.info("openai-unofficial package loaded successfully")
except ImportError:
    OPENAI_UNOFFICIAL_AVAILABLE = False
    logger.error("openai-unofficial package not available. Please install it with: pip install openai-unofficial>=1.0.0")
    sys.exit(1)  # Exit if openai-unofficial is not available as it's required

class AIClient:
    """
    Unified AI client using openai-unofficial for all agent operations.
    This is a singleton class to ensure efficient resource usage across all agents.
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AIClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI client with the openai-unofficial library.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
        """
        if self._initialized:
            return
            
        # Initialize the client
        self.client = OpenAIUnofficial()
        self.models_cache = {}
        self.token_usage = {"total": 0, "by_model": {}}
        self._initialized = True
        logger.info("AIClient initialized successfully")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models from openai-unofficial."""
        try:
            response = self.client.list_models()
            self.models_cache = {model["id"]: model for model in response["data"]}
            return response["data"]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def chat_completion(self, 
                       messages: List[Dict[str, Any]], 
                       model: str = "gpt-4o", 
                       temperature: float = 0.7,
                       max_tokens: Optional[int] = None,
                       stream: bool = False,
                       tools: Optional[List[Dict[str, Any]]] = None,
                       **kwargs) -> Any:
        """
        Generate a chat completion using openai-unofficial.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (default: gpt-4o)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: List of tools/functions to use
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The API response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                **kwargs
            )
            
            # Track token usage for non-streaming responses
            if not stream and hasattr(response, 'usage'):
                usage = response.usage
                self.token_usage["total"] += usage.total_tokens
                if model not in self.token_usage["by_model"]:
                    self.token_usage["by_model"][model] = 0
                self.token_usage["by_model"][model] += usage.total_tokens
                
            return response
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
    
    def streaming_chat(self, 
                      messages: List[Dict[str, Any]], 
                      model: str = "gpt-4o-realtime-preview",
                      callback: Optional[Callable[[str], None]] = None,
                      **kwargs) -> str:
        """
        Generate a streaming chat completion and process it with a callback.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (default: gpt-4o-realtime-preview)
            callback: Function to call with each chunk of text
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The complete generated text
        """
        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            
            full_response = ""
            for chunk in stream:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        if callback:
                            callback(content)
            
            return full_response
        except Exception as e:
            logger.error(f"Error in streaming chat: {e}")
            raise
    
    def audio_chat(self, 
                  messages: List[Dict[str, Any]], 
                  voice: str = "alloy",
                  audio_format: str = "mp3") -> Dict[str, Any]:
        """
        Generate both text and audio responses using gpt-4o-audio-preview.
        
        Args:
            messages: List of message dictionaries
            voice: Voice to use for audio
            audio_format: Audio format (mp3 or wav)
            
        Returns:
            Dict with text and audio data
        """
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o-audio-preview",
                modalities=["text", "audio"],
                audio={"voice": voice, "format": audio_format}
            )
            
            return {
                "text": response.choices[0].message.content,
                "audio": response.choices[0].message.audio
            }
        except Exception as e:
            logger.error(f"Error in audio chat: {e}")
            raise
    
    def generate_image(self, 
                      prompt: str, 
                      model: str = "dall-e-3", 
                      size: str = "1024x1024",
                      quality: str = "standard",
                      n: int = 1) -> Any:
        """
        Generate images from a prompt.
        
        Args:
            prompt: Text prompt for image generation
            model: Image model to use
            size: Image size
            quality: Image quality
            n: Number of images to generate
            
        Returns:
            The image generation response
        """
        try:
            response = self.client.image.create(
                prompt=prompt,
                model=model,
                size=size,
                quality=quality,
                n=n
            )
            return response
        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            raise
    
    def transcribe_audio(self, audio_file, model: str = "whisper-1") -> Any:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_file: Audio file to transcribe
            model: Model to use
            
        Returns:
            The transcription response
        """
        try:
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=model
            )
            return transcription
        except Exception as e:
            logger.error(f"Error in audio transcription: {e}")
            raise
    
    def text_to_speech(self, 
                      text: str, 
                      model: str = "tts-1", 
                      voice: str = "alloy") -> Any:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice to use
            
        Returns:
            The audio response
        """
        try:
            response = self.client.audio.create(
                input_text=text,
                model=model,
                voice=voice
            )
            return response
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            raise
    
    def get_token_usage(self) -> Dict[str, Any]:
        """Get the current token usage statistics."""
        return self.token_usage
    
    def reset_token_usage(self) -> None:
        """Reset the token usage statistics."""
        self.token_usage = {"total": 0, "by_model": {}}

# Create a singleton instance for easy import
ai_client = AIClient()

