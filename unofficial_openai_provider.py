#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unofficial OpenAI API Provider for Skyscope Enterprise Suite
===========================================================
This module provides an interface to the unofficial OpenAI API service
hosted at https://devsdocode-openai.hf.space. It supports chat completions,
audio generation, and image generation without requiring authentication.

The module is designed to work with all 10,000 agents in the Skyscope system
and includes comprehensive error handling and retry logic.
"""

import os
import json
import time
import base64
import logging
import requests
from typing import Dict, List, Any, Optional, Union, Generator, Tuple
from requests.exceptions import RequestException, Timeout, ConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.expanduser("~/SkyscopeEnterprise/logs/unofficial_openai.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
PRIMARY_API_URL = "https://devsdocode-openai.hf.space"
BACKUP_API_URL = "https://openai-devsdocode.up.railway.app"

# Available models
CHAT_MODELS = [
    # GPT-4 Series
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo",
    # GPT-3.5 Series
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-instruct-0914",
    "gpt-3.5-turbo-instruct"
]

AUDIO_MODELS = [
    "tts-1-hd-1106",
    "tts-1-hd",
    "tts-1-1106",
    "tts-1"
]

IMAGE_MODELS = [
    "dall-e-3",
    "dall-e-2"
]

AUDIO_CAPABLE_MODELS = [
    "gpt-4o-audio-preview-2024-10-01",
    "gpt-4o-audio-preview"
]

REALTIME_MODELS = [
    "gpt-4o-realtime-preview-2024-10-01",
    "gpt-4o-realtime-preview"
]

EMBEDDING_MODELS = [
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002"
]

AVAILABLE_VOICES = ["nova", "echo", "fable", "onyx", "shimmer", "alloy"]

class UnofficialOpenAIProvider:
    """
    Provider for accessing the unofficial OpenAI API service.
    
    This class provides methods for generating text completions,
    audio, and images using the unofficial OpenAI API.
    """
    
    def __init__(self, use_primary_api: bool = True, max_retries: int = 3, 
                 retry_delay: float = 1.0, timeout: float = 60.0):
        """
        Initialize the unofficial OpenAI API provider.
        
        Args:
            use_primary_api: Whether to use the primary API URL (Hugging Face)
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
        """
        self.api_url = PRIMARY_API_URL if use_primary_api else BACKUP_API_URL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        logger.info(f"Initialized Unofficial OpenAI Provider with API URL: {self.api_url}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from the API.
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = self._make_request("GET", "/models")
            if response and "data" in response:
                return response["data"]
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def get_api_info(self) -> Dict[str, Any]:
        """
        Get information about the API service.
        
        Returns:
            Dictionary containing API information
        """
        try:
            return self._make_request("GET", "/about")
        except Exception as e:
            logger.error(f"Failed to get API info: {e}")
            return {}
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini",
                       temperature: float = 0.7, top_p: float = 1.0,
                       presence_penalty: float = 0.0, frequency_penalty: float = 0.0,
                       stream: bool = False) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Generate a chat completion using the specified model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model ID to use for completion
            temperature: Sampling temperature (0.0 to 2.0)
            top_p: Nucleus sampling parameter (0.0 to 1.0)
            presence_penalty: Penalty for new topics (-2.0 to 2.0)
            frequency_penalty: Penalty for repetition (-2.0 to 2.0)
            stream: Whether to stream the response
            
        Returns:
            If stream=False: Dictionary containing the completion
            If stream=True: Generator yielding completion chunks
        """
        # Validate model
        if model not in CHAT_MODELS:
            logger.warning(f"Model {model} not in known chat models, but attempting anyway")
        
        # Prepare request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": stream
        }
        
        endpoint = "/chat/completions"
        
        if stream:
            return self._stream_request("POST", endpoint, payload)
        else:
            return self._make_request("POST", endpoint, payload)
    
    def generate_audio(self, text: str, model: str = "tts-1-hd", 
                      voice: str = "nova", format: str = "mp3") -> bytes:
        """
        Generate audio from text using the specified TTS model.
        
        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice to use (nova, echo, fable, onyx, shimmer, alloy)
            format: Audio format (mp3, wav)
            
        Returns:
            Audio data as bytes
        """
        # Validate model
        if model not in AUDIO_MODELS:
            logger.warning(f"Model {model} not in known audio models, but attempting anyway")
        
        # Validate voice
        if voice not in AVAILABLE_VOICES:
            logger.warning(f"Voice {voice} not in known voices, defaulting to 'nova'")
            voice = "nova"
        
        # Prepare request payload
        payload = {
            "model": model,
            "input": text,
            "voice": voice
        }
        
        endpoint = "/audio/speech"
        
        try:
            return self._make_audio_request("POST", endpoint, payload)
        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            raise
    
    def generate_image(self, prompt: str, model: str = "dall-e-3", 
                      n: int = 1, size: str = "1024x1024", 
                      quality: str = "hd", response_format: str = "url") -> List[str]:
        """
        Generate images from a text prompt.
        
        Args:
            prompt: Text prompt describing the desired image
            model: Image generation model to use
            n: Number of images to generate
            size: Image size (256x256, 512x512, 1024x1024)
            quality: Image quality (hd)
            response_format: Response format (url, b64_json)
            
        Returns:
            List of image URLs or base64-encoded JSON strings
        """
        # Validate model
        if model not in IMAGE_MODELS:
            logger.warning(f"Model {model} not in known image models, but attempting anyway")
        
        # Prepare request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
            "response_format": response_format
        }
        
        endpoint = "/images/generations"
        
        try:
            response = self._make_request("POST", endpoint, payload)
            
            if "data" in response:
                if response_format == "url":
                    return [item["url"] for item in response["data"]]
                else:  # b64_json
                    return [item["b64_json"] for item in response["data"]]
            else:
                logger.error(f"Unexpected response format: {response}")
                return []
        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return []
    
    def audio_chat_completion(self, messages: List[Dict[str, str]], 
                             model: str = "gpt-4o-audio-preview-2024-10-01",
                             voice: str = "nova", format: str = "wav",
                             temperature: float = 0.7) -> Tuple[str, bytes]:
        """
        Generate a chat completion with audio response.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Audio-capable model ID
            voice: Voice to use for audio
            format: Audio format (wav)
            temperature: Sampling temperature
            
        Returns:
            Tuple of (text_response, audio_bytes)
        """
        # Validate model
        if model not in AUDIO_CAPABLE_MODELS:
            logger.warning(f"Model {model} not in known audio-capable models, defaulting to gpt-4o-audio-preview")
            model = "gpt-4o-audio-preview"
        
        # Prepare request payload
        payload = {
            "messages": messages,
            "model": model,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": format},
            "temperature": temperature,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "top_p": 1
        }
        
        endpoint = "/chat/completions"
        
        try:
            response = self._make_request("POST", endpoint, payload)
            
            if "choices" in response and response["choices"]:
                message = response["choices"][0].get("message", {})
                
                text_response = message.get("content", "")
                
                if "audio" in message and "data" in message["audio"]:
                    audio_bytes = base64.b64decode(message["audio"]["data"])
                    return text_response, audio_bytes
                else:
                    logger.error("No audio data found in the response")
                    return text_response, b""
            else:
                logger.error(f"Unexpected response format: {response}")
                return "", b""
        except Exception as e:
            logger.error(f"Failed to generate audio chat completion: {e}")
            return "", b""
    
    def _make_request(self, method: str, endpoint: str, 
                     payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the API with retry logic.
        
        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.api_url}{endpoint}"
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                if method == "GET":
                    response = requests.get(url, timeout=self.timeout)
                elif method == "POST":
                    response = requests.post(url, json=payload, timeout=self.timeout)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    last_error = ValueError(error_msg)
            except (RequestException, Timeout, ConnectionError) as e:
                logger.warning(f"Request failed (attempt {retries+1}/{self.max_retries}): {e}")
                last_error = e
            
            # Exponential backoff
            sleep_time = self.retry_delay * (2 ** retries)
            time.sleep(sleep_time)
            retries += 1
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise Exception("Request failed for unknown reason")
    
    def _stream_request(self, method: str, endpoint: str, 
                       payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Make a streaming HTTP request to the API.
        
        Args:
            method: HTTP method (POST)
            endpoint: API endpoint
            payload: Request payload
            
        Yields:
            Response data chunks
        """
        url = f"{self.api_url}{endpoint}"
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
                    if response.status_code == 200:
                        for line in response.iter_lines():
                            if line:
                                line_text = line.decode('utf-8')
                                if line_text.startswith('data: '):
                                    data = line_text[6:]  # Remove 'data: ' prefix
                                    if data == "[DONE]":
                                        break
                                    try:
                                        yield json.loads(data)
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON: {e}")
                        return
                    else:
                        error_msg = f"API request failed with status {response.status_code}: {response.text}"
                        logger.error(error_msg)
                        last_error = ValueError(error_msg)
            except (RequestException, Timeout, ConnectionError) as e:
                logger.warning(f"Stream request failed (attempt {retries+1}/{self.max_retries}): {e}")
                last_error = e
            
            # Exponential backoff
            sleep_time = self.retry_delay * (2 ** retries)
            time.sleep(sleep_time)
            retries += 1
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise Exception("Stream request failed for unknown reason")
    
    def _make_audio_request(self, method: str, endpoint: str, 
                           payload: Dict[str, Any]) -> bytes:
        """
        Make an HTTP request for audio data.
        
        Args:
            method: HTTP method (POST)
            endpoint: API endpoint
            payload: Request payload
            
        Returns:
            Audio data as bytes
        """
        url = f"{self.api_url}{endpoint}"
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                response = requests.post(url, json=payload, stream=True, timeout=self.timeout)
                
                if response.status_code == 200:
                    audio_data = b''
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            audio_data += chunk
                    return audio_data
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    last_error = ValueError(error_msg)
            except (RequestException, Timeout, ConnectionError) as e:
                logger.warning(f"Audio request failed (attempt {retries+1}/{self.max_retries}): {e}")
                last_error = e
            
            # Exponential backoff
            sleep_time = self.retry_delay * (2 ** retries)
            time.sleep(sleep_time)
            retries += 1
        
        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise Exception("Audio request failed for unknown reason")
    
    def is_available(self) -> bool:
        """
        Check if the API is available.
        
        Returns:
            True if the API is available, False otherwise
        """
        try:
            response = requests.get(f"{self.api_url}/about", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API availability check failed: {e}")
            return False
    
    def switch_api_url(self):
        """Switch between primary and backup API URLs."""
        if self.api_url == PRIMARY_API_URL:
            self.api_url = BACKUP_API_URL
            logger.info(f"Switched to backup API URL: {self.api_url}")
        else:
            self.api_url = PRIMARY_API_URL
            logger.info(f"Switched to primary API URL: {self.api_url}")


class AgentAIProvider:
    """
    AI Provider specifically designed for agent interactions.
    
    This class extends the UnofficialOpenAIProvider with agent-specific
    functionality for the 10,000 agents in the Skyscope system.
    """
    
    def __init__(self, agent_id: Optional[str] = None, 
                use_primary_api: bool = True, max_retries: int = 3):
        """
        Initialize the agent AI provider.
        
        Args:
            agent_id: ID of the agent using this provider
            use_primary_api: Whether to use the primary API URL
            max_retries: Maximum number of retries for failed requests
        """
        self.agent_id = agent_id
        self.provider = UnofficialOpenAIProvider(
            use_primary_api=use_primary_api,
            max_retries=max_retries
        )
        logger.info(f"Initialized Agent AI Provider for agent {agent_id}")
    
    def agent_chat(self, messages: List[Dict[str, str]], 
                  model: str = "gpt-4o-mini", stream: bool = False) -> Dict[str, Any]:
        """
        Generate a chat completion for an agent.
        
        Args:
            messages: List of message dictionaries
            model: Model ID to use
            stream: Whether to stream the response
            
        Returns:
            Response data or generator
        """
        try:
            # Add agent context to messages if agent_id is provided
            if self.agent_id:
                agent_context = self._get_agent_context()
                if agent_context:
                    # Insert agent context as the first system message
                    system_msg = {
                        "role": "system",
                        "content": agent_context
                    }
                    messages.insert(0, system_msg)
            
            return self.provider.chat_completion(
                messages=messages,
                model=model,
                stream=stream
            )
        except Exception as e:
            logger.error(f"Agent chat failed for agent {self.agent_id}: {e}")
            # Return a fallback response
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "I apologize, but I'm currently unable to process your request due to a technical issue. Please try again later."
                    },
                    "finish_reason": "error"
                }]
            }
    
    def agent_think(self, task: str, context: Optional[str] = None, 
                   model: str = "gpt-4o-mini") -> str:
        """
        Generate a thinking process for an agent task.
        
        Args:
            task: Task description
            context: Additional context
            model: Model ID to use
            
        Returns:
            Thinking process as text
        """
        prompt = f"Task: {task}\n\n"
        if context:
            prompt += f"Context: {context}\n\n"
        
        prompt += "Think step by step about how to approach this task. Consider different strategies, potential challenges, and the best way to achieve the desired outcome."
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.agent_chat(messages, model=model)
            if "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"]
            return ""
        except Exception as e:
            logger.error(f"Agent thinking failed for agent {self.agent_id}: {e}")
            return "Unable to generate thinking process due to a technical issue."
    
    def agent_execute(self, plan: str, model: str = "gpt-4o-mini") -> Dict[str, Any]:
        """
        Execute a plan for an agent.
        
        Args:
            plan: Plan to execute
            model: Model ID to use
            
        Returns:
            Execution results
        """
        prompt = f"Execute the following plan step by step and provide the results:\n\n{plan}\n\nProvide your results in JSON format with the following structure:\n{{\"success\": true/false, \"results\": [...], \"explanation\": \"...\", \"next_steps\": \"...\"}}"
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.agent_chat(messages, model=model)
            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                
                # Try to extract JSON from the response
                try:
                    # Find JSON content (it might be wrapped in markdown code blocks)
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```|```(.*?)```|(\{.*\})', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1) or json_match.group(2) or json_match.group(3)
                        return json.loads(json_str)
                    else:
                        # Try to parse the entire content as JSON
                        return json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    # If JSON parsing fails, return a structured response
                    return {
                        "success": False,
                        "results": [],
                        "explanation": "Failed to parse execution results as JSON",
                        "next_steps": "Try a different approach or format"
                    }
            
            return {
                "success": False,
                "results": [],
                "explanation": "Failed to execute plan",
                "next_steps": "Try a different approach"
            }
        except Exception as e:
            logger.error(f"Agent execution failed for agent {self.agent_id}: {e}")
            return {
                "success": False,
                "results": [],
                "explanation": f"Execution failed due to a technical issue: {str(e)}",
                "next_steps": "Try again later or with a different approach"
            }
    
    def _get_agent_context(self) -> str:
        """
        Get context information for the agent.
        
        Returns:
            Agent context as text
        """
        if not self.agent_id:
            return ""
        
        try:
            # Load agent data from file
            agent_file = os.path.expanduser(f"~/SkyscopeEnterprise/agents/{self.agent_id}.json")
            if not os.path.exists(agent_file):
                logger.warning(f"Agent file not found: {agent_file}")
                return ""
            
            with open(agent_file, 'r') as f:
                agent_data = json.load(f)
            
            # Create a context string from agent data
            context = f"You are {agent_data.get('name', 'an AI agent')}, "
            context += f"a {agent_data.get('background', {}).get('experience_level', 'experienced')} specialist in {agent_data.get('expertise', {}).get('primary', 'various fields')}. "
            
            # Add personality
            personality = agent_data.get('personality', {})
            if personality:
                archetype = personality.get('archetype', '')
                if archetype:
                    context += f"You embody the {archetype} archetype. "
                
                comm_style = personality.get('communication_style', '')
                if comm_style:
                    context += f"Your communication style is {comm_style}. "
            
            # Add bio if available
            if 'bio' in agent_data:
                context += f"\n\nBackground: {agent_data['bio']}"
            
            # Add expertise areas
            expertise = agent_data.get('expertise', {})
            if expertise:
                context += f"\n\nYour primary expertise is in {expertise.get('primary', '')}. "
                if 'secondary' in expertise:
                    context += f"You also have knowledge in {expertise.get('secondary', '')}. "
                if 'tertiary' in expertise:
                    context += f"Additionally, you have some experience with {expertise.get('tertiary', '')}."
            
            # Add income strategies
            strategies = agent_data.get('income_strategies', {})
            if strategies:
                context += f"\n\nYour primary income strategy is {strategies.get('primary', '')}. "
                if 'secondary' in strategies:
                    context += f"Your secondary strategy is {strategies.get('secondary', '')}."
            
            return context
        except Exception as e:
            logger.error(f"Failed to get agent context for {self.agent_id}: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    # Initialize provider
    provider = UnofficialOpenAIProvider()
    
    # Check if API is available
    if provider.is_available():
        print("API is available!")
        
        # List models
        models = provider.list_models()
        print(f"Available models: {len(models)}")
        
        # Get API info
        api_info = provider.get_api_info()
        print(f"API version: {api_info.get('version', 'unknown')}")
        
        # Test chat completion
        response = provider.chat_completion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="gpt-4o-mini"
        )
        if "choices" in response and response["choices"]:
            print(f"Response: {response['choices'][0]['message']['content']}")
    else:
        print("API is not available.")
