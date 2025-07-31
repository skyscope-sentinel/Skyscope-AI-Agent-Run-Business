
"""
ml_deployment_management_part2.py - Continuation of ML Deployment and Management System

This module continues the implementation of the ML deployment and management system,
including OpenAI integration, Kubernetes deployment, monitoring, compliance, and API endpoints.

Part of Skyscope Sentinel Intelligence AI - ITERATION 11
"""

import asyncio
import base64
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, BinaryIO, Generator

try:
    import aiohttp
    import kubernetes
    import numpy as np
    import pandas as pd
    import psutil
    import pydantic
    import requests
    import torch
    import websockets
    import yaml
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, WebSocket, WebSocketDisconnect, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
    from kubernetes import client, config, watch
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    from pydantic import BaseModel, Field, validator
    from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, DateTime, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker
    from sqlalchemy.sql import func
    
    # Import openai-unofficial if available, otherwise prepare for fallback
    try:
        import openai_unofficial
        from openai_unofficial import OpenAI
        from openai_unofficial.types.audio import Speech
        from openai_unofficial.types.chat import ChatCompletion
        OPENAI_UNOFFICIAL_AVAILABLE = True
    except ImportError:
        OPENAI_UNOFFICIAL_AVAILABLE = False
        warnings.warn("openai-unofficial package not found. Using standard OpenAI package with fallback to Ollama.")
        try:
            import openai
        except ImportError:
            warnings.warn("OpenAI package not found. Only Ollama will be available.")
    
    # Try to import Ollama for fallback
    try:
        import ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
        if not OPENAI_UNOFFICIAL_AVAILABLE:
            warnings.warn("Neither openai-unofficial nor Ollama are available. Limited functionality.")

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "aiohttp", "kubernetes", "numpy", "pandas", 
                          "psutil", "pydantic", "requests", "torch", 
                          "websockets", "pyyaml", "fastapi", "uvicorn", 
                          "prometheus-client", "sqlalchemy", "python-multipart",
                          "python-jose[cryptography]", "passlib[bcrypt]"])
    print("Please restart the application.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_deployment_part2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_EMPLOYEES = 10000
DEFAULT_MODEL = "gpt-4o-2024-05-13"
AUDIO_PREVIEW_MODEL = "gpt-4o-audio-preview-2025-06-03"
REALTIME_PREVIEW_MODEL = "gpt-4o-realtime-preview-2024-10-01"
DEFAULT_OLLAMA_MODEL = "llama3"
KUBERNETES_NAMESPACE = "skyscope-ml"

#######################################################
# OpenAI Unofficial Integration (Continued)
#######################################################

class OpenAIUnofficialManager:
    """
    Manages integration with openai-unofficial package for GPT-4o models
    including audio-preview & realtime-preview variants.
    """
    
    def __init__(self):
        self.client = None
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.token_usage = defaultdict(int)
        self.fallback_to_ollama = True
        self.ollama_client = None
        self.active_websockets = {}
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients."""
        try:
            if OPENAI_UNOFFICIAL_AVAILABLE:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Unofficial client initialized")
            elif "openai" in sys.modules:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("Standard OpenAI client initialized as fallback")
            
            if OLLAMA_AVAILABLE and self.fallback_to_ollama:
                self.ollama_client = ollama
                logger.info("Ollama client initialized for fallback")
        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
    
    def set_api_key(self, api_key: str) -> bool:
        """Set the OpenAI API key."""
        try:
            self.api_key = api_key
            
            if OPENAI_UNOFFICIAL_AVAILABLE:
                self.client = OpenAI(api_key=api_key)
            elif "openai" in sys.modules:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            
            return True
        except Exception as e:
            logger.error(f"Error setting API key: {e}")
            return False
    
    def chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a chat completion using GPT-4o models."""
        try:
            if not self.client:
                return {"error": "No API client available"}
            
            # Check if audio output is requested
            audio_output = request.get("audio_output", False)
            
            # Select appropriate model based on audio output
            model = request.get("model", DEFAULT_MODEL)
            if audio_output and not model.endswith("audio-preview"):
                model = AUDIO_PREVIEW_MODEL
                logger.info(f"Switching to audio-preview model: {model}")
            
            # Prepare messages
            messages = request.get("messages", [])
            
            # Set up response format for audio if needed
            response_format = None
            if audio_output:
                response_format = {
                    "type": "text_and_audio",
                    "audio": {
                        "format": "mp3"
                    }
                }
            
            # Try with openai-unofficial first
            try:
                if OPENAI_UNOFFICIAL_AVAILABLE:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=request.get("stream", False),
                        response_format=response_format
                    )
                    
                    # Track token usage
                    if hasattr(response, "usage") and response.usage:
                        self.token_usage[model] += response.usage.total_tokens
                    
                    # Extract audio if present
                    audio_data = None
                    if audio_output and hasattr(response.choices[0].message, "audio"):
                        audio_data = response.choices[0].message.audio
                    
                    return {
                        "id": response.id,
                        "model": response.model,
                        "content": response.choices[0].message.content,
                        "audio": audio_data,
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": asdict(response.usage) if hasattr(response, "usage") else None
                    }
                elif "openai" in sys.modules:
                    # Fallback to standard OpenAI package
                    import openai
                    response = self.client.chat.completions.create(
                        model=model if "gpt-4o" in model else "gpt-4o",  # Fallback to standard model
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=request.get("stream", False)
                    )
                    
                    return {
                        "id": response.id,
                        "model": response.model,
                        "content": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason,
                        "usage": asdict(response.usage) if hasattr(response, "usage") else None
                    }
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}. Trying Ollama fallback.")
                
                # Fallback to Ollama if available
                if OLLAMA_AVAILABLE and self.fallback_to_ollama:
                    # Convert messages to Ollama format
                    ollama_messages = []
                    for msg in messages:
                        ollama_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    
                    response = self.ollama_client.chat(
                        model=DEFAULT_OLLAMA_MODEL,
                        messages=ollama_messages,
                        stream=False
                    )
                    
                    return {
                        "id": str(uuid.uuid4()),
                        "model": DEFAULT_OLLAMA_MODEL,
                        "content": response["message"]["content"],
                        "finish_reason": "stop",
                        "usage": None,
                        "using_fallback": True
                    }
                else:
                    raise
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return {"error": str(e)}
    
    async def streaming_chat_completion(self, request: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """Generate a streaming chat completion."""
        try:
            if not self.client:
                yield {"error": "No API client available"}
                return
            
            # Prepare messages
            messages = request.get("messages", [])
            model = request.get("model", DEFAULT_MODEL)
            
            # Try with openai-unofficial first
            try:
                if OPENAI_UNOFFICIAL_AVAILABLE:
                    stream = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield {
                                "id": chunk.id,
                                "model": chunk.model,
                                "content": chunk.choices[0].delta.content,
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                elif "openai" in sys.modules:
                    # Fallback to standard OpenAI package
                    import openai
                    stream = self.client.chat.completions.create(
                        model=model if "gpt-4o" in model else "gpt-4o",
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield {
                                "id": chunk.id,
                                "model": chunk.model,
                                "content": chunk.choices[0].delta.content,
                                "finish_reason": chunk.choices[0].finish_reason
                            }
            except Exception as e:
                logger.warning(f"OpenAI API streaming error: {e}. Trying Ollama fallback.")
                
                # Fallback to Ollama if available
                if OLLAMA_AVAILABLE and self.fallback_to_ollama:
                    # Convert messages to Ollama format
                    ollama_messages = []
                    for msg in messages:
                        ollama_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    
                    stream = self.ollama_client.chat(
                        model=DEFAULT_OLLAMA_MODEL,
                        messages=ollama_messages,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            yield {
                                "id": str(uuid.uuid4()),
                                "model": DEFAULT_OLLAMA_MODEL,
                                "content": chunk["message"]["content"],
                                "finish_reason": None,
                                "using_fallback": True
                            }
                else:
                    yield {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            yield {"error": str(e)}
    
    async def realtime_audio_session(self, websocket: WebSocket, model: str = REALTIME_PREVIEW_MODEL) -> None:
        """
        Handle a realtime audio session over WebSocket.
        This connects to the GPT-4o realtime-preview model for speech-to-speech interactions.
        """
        try:
            if not OPENAI_UNOFFICIAL_AVAILABLE:
                await websocket.send_json({"error": "openai-unofficial package required for realtime audio"})
                return
            
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            self.active_websockets[session_id] = websocket
            
            # Create WebSocket URL for OpenAI realtime API
            ws_url = f"wss://api.openai.com/v1/realtime?model={model}"
            
            # Headers for authentication
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Connect to OpenAI WebSocket
            async with websockets.connect(ws_url, extra_headers=headers) as openai_ws:
                # Send initial configuration
                await openai_ws.send(json.dumps({
                    "type": "config",
                    "data": {
                        "speech_recognition": {
                            "model": "whisper-1",
                            "language": "en"
                        },
                        "text_to_speech": {
                            "voice": "alloy"
                        },
                        "turn_detection": {
                            "mode": "auto"
                        }
                    }
                }))
                
                # Create bidirectional relay between client and OpenAI
                async def relay_to_openai():
                    try:
                        while True:
                            data = await websocket.receive_bytes()
                            await openai_ws.send(data)
                    except WebSocketDisconnect:
                        logger.info(f"Client disconnected from session {session_id}")
                        if session_id in self.active_websockets:
                            del self.active_websockets[session_id]
                    except Exception as e:
                        logger.error(f"Error in relay_to_openai: {e}")
                
                async def relay_from_openai():
                    try:
                        while True:
                            data = await openai_ws.recv()
                            if isinstance(data, str):
                                await websocket.send_text(data)
                            else:
                                await websocket.send_bytes(data)
                    except websockets.exceptions.ConnectionClosed:
                        logger.info(f"OpenAI WebSocket closed for session {session_id}")
                        await websocket.close()
                    except Exception as e:
                        logger.error(f"Error in relay_from_openai: {e}")
                
                # Run both relays concurrently
                await asyncio.gather(
                    relay_to_openai(),
                    relay_from_openai()
                )
        except Exception as e:
            logger.error(f"Error in realtime audio session: {e}")
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
    
    async def process_audio_input(self, audio_file: BinaryIO, prompt: str = None) -> Dict[str, Any]:
        """Process audio input using Whisper API."""
        try:
            if not self.client:
                return {"error": "No API client available"}
            
            if OPENAI_UNOFFICIAL_AVAILABLE:
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
                
                result = {"text": transcription.text}
                
                # If prompt is provided, send the transcription to chat completion
                if prompt:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": transcription.text}
                    ]
                    
                    chat_result = self.chat_completion({
                        "model": DEFAULT_MODEL,
                        "messages": messages
                    })
                    
                    result["response"] = chat_result
                
                return result
            elif "openai" in sys.modules:
                # Fallback to standard OpenAI package
                import openai
                transcription = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1"
                )
                
                result = {"text": transcription.text}
                
                # If prompt is provided, send the transcription to chat completion
                if prompt:
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": transcription.text}
                    ]
                    
                    chat_result = self.chat_completion({
                        "model": DEFAULT_MODEL,
                        "messages": messages
                    })
                    
                    result["response"] = chat_result
                
                return result
        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            return {"error": str(e)}
    
    async def process_image_input(self, image_file: BinaryIO, prompt: str) -> Dict[str, Any]:
        """Process image input using GPT-4o vision capabilities."""
        try:
            if not self.client:
                return {"error": "No API client available"}
            
            # Read image file and encode as base64
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Create message with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Process with GPT-4o
            if OPENAI_UNOFFICIAL_AVAILABLE:
                response = self.client.chat.completions.create(
                    model=DEFAULT_MODEL,  # GPT-4o supports vision by default
                    messages=messages,
                    max_tokens=1000
                )
                
                return {
                    "id": response.id,
                    "model": response.model,
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason
                }
            elif "openai" in sys.modules:
                # Fallback to standard OpenAI package
                import openai
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Standard package might not have the latest model names
                    messages=messages,
                    max_tokens=1000
                )
                
                return {
                    "id": response.id,
                    "model": response.model,
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason
                }
        except Exception as e:
            logger.error(f"Error processing image input: {e}")
            return {"error": str(e)}
    
    def text_to_speech(self, text: str, voice: str = "alloy", output_format: str = "mp3") -> Optional[bytes]:
        """Convert text to speech using OpenAI TTS."""
        try:
            if not self.client:
                return None
            
            if OPENAI_UNOFFICIAL_AVAILABLE:
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format=output_format
                )
                
                # Get audio data
                audio_data = response.content
                
                return audio_data
            elif "openai" in sys.modules:
                # Fallback to standard OpenAI package
                import openai
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format=output_format
                )
                
                # Get audio data
                audio_data = response.content
                
                return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            return None
    
    def get_token_usage_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return dict(self.token_usage)
    
    def reset_token_usage_stats(self) -> None:
        """Reset token usage statistics."""
        self.token_usage.clear()


#######################################################
# Kubernetes Deployment Manager
#######################################################

class KubernetesDeploymentManager:
    """
    Manages deployment of ML models to Kubernetes clusters.
    Supports auto-scaling, blue-green deployments, canary releases,
    and multi-region deployments.
    """
    
    def __init__(self, namespace: str = KUBERNETES_NAMESPACE):
        self.namespace = namespace
        self.active_deployments = {}
        self.deployment_history = {}
        self.metrics = {}
        
        # Initialize Kubernetes client
        try:
            config.load_kube_config()
            self.api_client = client.ApiClient()
            self.apps_v1 = client.AppsV1Api(self.api_client)
            self.core_v1 = client.CoreV1Api(self.api_client)
            self.autoscaling_v2 = client.AutoscalingV2Api(self.api_client)
            self.networking_v1 = client.NetworkingV1Api(self.api_client)
            
            # Check if namespace exists, create if not
            self._ensure_namespace()
            
            logger.info(f"Kubernetes client initialized with namespace '{namespace}'")
        except Exception as e:
            logger.error(f"Error initializing Kubernetes client: {e}")
            self.api_client = None
    
    def _ensure_namespace(self) -> None:
        """Ensure the namespace exists, create if not."""
        try:
            if self.api_client:
                namespaces = self.core_v1.list_namespace()
                namespace_exists = any(ns.metadata.name == self.namespace for ns in namespaces.items)
                
                if not namespace_exists:
                    namespace = client.V1Namespace(
                        metadata=client.V1ObjectMeta(name=self.namespace)
                    )
                    self.core_v1.create_namespace(namespace)
                    logger.info(f"Created namespace '{self.namespace}'")
        except Exception as e:
            logger.error(f"Error ensuring namespace: {e}")
    
    def deploy_model(self, model_id: int, model_path: str, 
                    deployment_name: str, replicas: int = 2,
                    resources: Dict[str, Any] = None,
                    deployment_strategy: str = "rolling") -> Dict[str, Any]:
        """
        Deploy a model to Kubernetes.
        
        Args:
            model_id: ID of the model to deploy
            model_path: Path to model artifacts
            deployment_name: Name for the deployment
            replicas: Number of replicas
            resources: Resource requirements
            deployment_strategy: One of "rolling", "blue-green", "canary"
        """
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Default resources if not provided
            if not resources:
                resources = {
                    "requests": {
                        "cpu": "500m",
                        "memory": "1Gi"
                    },
                    "limits": {
                        "cpu": "2",
                        "memory": "4Gi"
                    }
                }
            
            # Create ConfigMap for model metadata
            config_map_name = f"{deployment_name}-config"
            config_map_data = {
                "model_id": str(model_id),
                "model_path": model_path,
                "deployment_time": datetime.datetime.now().isoformat()
            }
            
            config_map = client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name=config_map_name),
                data={
                    "model_metadata.json": json.dumps(config_map_data)
                }
            )
            
            self.core_v1.create_namespaced_config_map(
                namespace=self.namespace,
                body=config_map
            )
            
            # Create deployment based on strategy
            if deployment_strategy == "rolling":
                return self._create_rolling_deployment(
                    deployment_name, model_id, model_path, replicas, resources
                )
            elif deployment_strategy == "blue-green":
                return self._create_blue_green_deployment(
                    deployment_name, model_id, model_path, replicas, resources
                )
            elif deployment_strategy == "canary":
                return self._create_canary_deployment(
                    deployment_name, model_id, model_path, replicas, resources
                )
            else:
                return {"status": "error", "message": f"Unknown deployment strategy: {deployment_strategy}"}
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_rolling_deployment(self, deployment_name: str, model_id: int, 
                                  model_path: str, replicas: int,
                                  resources: Dict[str, Any]) -> Dict[str, Any]:
        """Create a rolling update deployment."""
        try:
            # Create deployment
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=deployment_name,
                    labels={
                        "app": deployment_name,
                        "model_id": str(model_id)
                    }
                ),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={
                            "app": deployment_name
                        }
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={
                                "app": deployment_name,
                                "model_id": str(model_id)
                            }
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="model-server",
                                    image="skyscope/ml-model-server:latest",
                                    ports=[
                                        client.V1ContainerPort(container_port=8000)
                                    ],
                                    env=[
                                        client.V1EnvVar(
                                            name="MODEL_ID",
                                            value=str(model_id)
                                        ),
                                        client.V1EnvVar(
                                            name="MODEL_PATH",
                                            value=model_path
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=resources["requests"],
                                        limits=resources["limits"]
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/ready",
                                            port=8000
                                        ),
                                        initial_delay_seconds=15,
                                        period_seconds=5
                                    )
                                )
                            ]
                        )
                    ),
                    strategy=client.V1DeploymentStrategy(
                        type="RollingUpdate",
                        rolling_update=client.V1RollingUpdateDeployment(
                            max_surge="25%",
                            max_unavailable="25%"
                        )
                    )
                )
            )
            
            # Create deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            # Create service
            service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=deployment_name
                ),
                spec=client.V1ServiceSpec(
                    selector={
                        "app": deployment_name
                    },
                    ports=[
                        client.V1ServicePort(
                            port=80,
                            target_port=8000
                        )
                    ]
                )
            )
            
            self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=service
            )
            
            # Create HPA
            hpa = client.V2HorizontalPodAutoscaler(
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-hpa"
                ),
                spec=client.V2HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V2CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=replicas,
                    max_replicas=replicas * 3,
                    metrics=[
                        client.V2MetricSpec(
                            type="Resource",
                            resource=client.V2ResourceMetricSource(
                                name="cpu",
                                target=client.V2MetricTarget(
                                    type="Utilization",
                                    average_utilization=80
                                )
                            )
                        )
                    ]
                )
            )
            
            self.autoscaling_v2.create_namespaced_horizontal_pod_autoscaler(
                namespace=self.namespace,
                body=hpa
            )
            
            # Track deployment
            self.active_deployments[deployment_name] = {
                "model_id": model_id,
                "replicas": replicas,
                "created_at": datetime.datetime.now().isoformat(),
                "status": "active",
                "strategy": "rolling",
                "service": deployment_name
            }
            
            return {
                "status": "success",
                "message": f"Deployment '{deployment_name}' created",
                "deployment_name": deployment_name,
                "service_name": deployment_name,
                "endpoint": f"http://{deployment_name}.{self.namespace}.svc.cluster.local"
            }
        except Exception as e:
            logger.error(f"Error creating rolling deployment: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_blue_green_deployment(self, deployment_name: str, model_id: int, 
                                     model_path: str, replicas: int,
                                     resources: Dict[str, Any]) -> Dict[str, Any]:
        """Create a blue-green deployment."""
        try:
            # Check if deployment already exists
            existing_deployment = None
            try:
                existing_deployment = self.apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=self.namespace
                )
            except:
                pass
            
            # Determine blue/green labels
            if existing_deployment:
                current_color = "blue"
                if "color" in existing_deployment.metadata.labels and existing_deployment.metadata.labels["color"] == "blue":
                    current_color = "blue"
                    new_color = "green"
                else:
                    current_color = "green"
                    new_color = "blue"
            else:
                # First deployment is blue
                current_color = None
                new_color = "blue"
            
            # Create new deployment with color label
            new_deployment_name = f"{deployment_name}-{new_color}"
            
            deployment = client.V1Deployment(
                metadata=client.V1ObjectMeta(
                    name=new_deployment_name,
                    labels={
                        "app": deployment_name,
                        "model_id": str(model_id),
                        "color": new_color
                    }
                ),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(
                        match_labels={
                            "app": deployment_name,
                            "color": new_color
                        }
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={
                                "app": deployment_name,
                                "model_id": str(model_id),
                                "color": new_color
                            }
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name="model-server",
                                    image="skyscope/ml-model-server:latest",
                                    ports=[
                                        client.V1ContainerPort(container_port=8000)
                                    ],
                                    env=[
                                        client.V1EnvVar(
                                            name="MODEL_ID",
                                            value=str(model_id)
                                        ),
                                        client.V1EnvVar(
                                            name="MODEL_PATH",
                                            value=model_path
                                        ),
                                        client.V1EnvVar(
                                            name="DEPLOYMENT_COLOR",
                                            value=new_color
                                        )
                                    ],
                                    resources=client.V1ResourceRequirements(
                                        requests=resources["requests"],
                                        limits=resources["limits"]
                                    ),
                                    liveness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/health",
                                            port=8000
                                        ),
                                        initial_delay_seconds=30,
                                        period_seconds=10
                                    ),
                                    readiness_probe=client.V1Probe(
                                        http_get=client.V1HTTPGetAction(
                                            path="/ready",
                                            port=8000
                                        ),
                                        initial_delay_seconds=15,
                                        period_seconds=5
                                    )
                                )
                            ]
                        )
                    )
                )
            )
            
            # Create new deployment
            self.apps_v1.create_namespaced_deployment(
                namespace=self.namespace,
                body=deployment
            )
            
            # Create or update service to point to new deployment
            service_name = deployment_name
            
            # Create color-specific service for testing
            color_service = client.V1Service(
                metadata=client.V1ObjectMeta(
                    name=f"{deployment_name}-{new_color}"
                ),
                spec=client.V1ServiceSpec(
                    selector={
                        "app": deployment_name,
                        "color": new_color
                    },
                    ports=[
                        client.V1ServicePort(
                            port=80,
                            target_port=8000
                        )
                    ]
                )
            )
            
            self.core_v1.create_namespaced_service(
                namespace=self.namespace,
                body=color_service
            )
            
            # Create or update main service
            try:
                # Check if service exists
                self.core_v1.read_namespaced_service(
                    name=service_name,
                    namespace=self.namespace
                )
                
                # Service exists, update it
                service = client.V1Service(
                    metadata=client.V1ObjectMeta(
                        name=service_name
                    ),
                    spec=client.V1ServiceSpec(
                        selector={
                            "app": deployment_name,
                            "color": new_color
                        },
                        ports=[
                            client.V1ServicePort(
                                port=80,
                                target_port=8000
                            )
                        ]
                    )
                )
                
                self.core_v1.replace_namespaced_service(
                    name=service_name,
                    namespace=self.namespace,
                    body=service
                )
            except:
                # Service doesn't exist, create it
                service = client.V1Service(
                    metadata=client.V1ObjectMeta(
                        name=service_name
                    ),
                    spec=client.V1ServiceSpec(
                        selector={
                            "app": deployment_name,
                            "color": new_color
                        },
                        ports=[
                            client.V1ServicePort(
                                port=80,
                                target_port=8000
                            )
                        ]
                    )
                )
                
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace,
                    body=service
                )
            
            # Track deployment
            self.active_deployments[new_deployment_name] = {
                "model_id": model_id,
                "replicas": replicas,
                "created_at": datetime.datetime.now().isoformat(),
                "status": "active",
                "strategy": "blue-green",
                "color": new_color,
                "service": service_name
            }
            
            if current_color:
                # Keep track of old deployment for potential rollback
                old_deployment_name = f"{deployment_name}-{current_color}"
                self.active_deployments[old_deployment_name]["status"] = "inactive"
            
            return {
                "status": "success",
                "message": f"Blue-Green deployment created",
                "deployment_name": new_deployment_name,
                "color": new_color,
                "service_name": service_name,
                "test_endpoint": f"http://{new_deployment_name}.{self.namespace}.svc.cluster.local",
                "main_endpoint": f"http://{service_name}.{self.namespace}.svc.cluster.local"
            }
        except Exception as e:
            logger.error(f"Error creating blue-green deployment: {e}")
            return {"status": "error", "message": str(e)}
    
    def _create_canary_deployment(self, deployment_name: str, model_id: int, 
                                 model_path: str, replicas: int,
                                 resources: Dict[str, Any]) -> Dict[str, Any]:
        """Create a canary deployment."""
        try:
            # Check if stable deployment exists
            stable_deployment_name = f"{deployment_name}-stable"
            canary_deployment_name = f"{deployment_name}-canary"
            
            stable_exists = False
            try:
                self.apps_v1.read_namespaced_deployment(
                    name=stable_deployment_name,
                    namespace=self.namespace
                )
                stable_exists = True
            except:
                pass
            
            # If stable doesn't exist, create it as the first deployment
            if not stable_exists:
                # Create stable deployment
                stable_deployment = client.V1Deployment(
                    metadata=client.V1ObjectMeta(
                        name=stable_deployment_name,
                        labels={
                            "app": deployment_name,
                            "model_id": str(model_id),
                            "version": "stable"
                        }
                    ),
                    spec=client.V1DeploymentSpec(
                        replicas=replicas,
                        selector=client.V1LabelSelector(
                            match_labels={
                                "app": deployment_name,
                                "version": "stable"
                            }
                        ),
                        template=client.V1PodTemplateSpec(
                            metadata=client.V1ObjectMeta(
                                labels={
                                    "app": deployment_name,
                                    "model_id": str(model_id),
                                    "version": "stable"
                                }
                            ),
                            spec=client.V1PodSpec(
                                containers=[
                                    client.V1Container(
                                        name="model-server",
                                        image="skyscope/ml-model-server:latest",
                                        ports=[
                                            client.V1ContainerPort(container_port=8000)
                                        ],
                                        env=[
                                            client.V1EnvVar(
                                                name="MODEL_ID",
                                                value=str(model_id)
                                            ),
                                            client.V1EnvVar(
                                                name="MODEL_PATH",
                                                value=model_path
                                            ),
                                            client.V1EnvVar(
                                                name="DEPLOYMENT_VERSION",
                                                value="stable"
                                            )
                                        ],
                                        resources=client.V1ResourceRequirements(
                                            requests=resources["requests"],
                                            limits=resources["limits"]
                                        ),
                                        liveness_probe=client.V1Probe(
                                            http_get=client.V1HTTPGetAction(
                                                path="/health",
                                                port=8000
                                            ),
                                            initial_delay_seconds=30,
                                            period_seconds=10
                                        ),
                                        readiness_probe=client.V1Probe(
                                            http_get=client.V1HTTPGetAction(
                                                path="/ready",
                                                port=8000
                                            ),
                                            initial_delay_seconds=15,
                                            period_seconds=5
                                        )
                                    )
                                ]
                            )
                        )
                    )
                )
                
                # Create stable deployment
                self.apps_v1.create_namespaced_deployment(
                    namespace=self.namespace,
                    body=stable_deployment
                )
                
                # Create service
                service = client.V1Service(
                    metadata=client.V1ObjectMeta(
                        name=deployment_name
                    ),
                    spec=client.V1ServiceSpec(
                        selector={
                            "app": deployment_name
                        },
                        ports=[
                            client.V1ServicePort(
                                port=80,
                                target_port=8000
                            )
                        ]
                    )
                )
                
                self.core_v1.create_namespaced_service(
                    namespace=self.namespace,
                    body=service
                )
                
                # Track deployment
                self.active_deployments[stable_deployment_name] = {
                    "model_id": model_id,
                    "replicas": replicas,
                    "created_at": datetime.datetime.now().isoformat(),
                    "status": "active",
                    "strategy": "canary",
                    "version": "stable",
                    "service": deployment_name,
                    "traffic_percentage": 100
                }
                
                return {
                    "status": "success",
                    "message": f"Initial stable deployment created",
                    "deployment_name": stable_deployment_name,
                    "service_name": deployment_name,
                    "endpoint": f"http://{deployment_name}.{self.namespace}.svc.cluster.local",
                    "traffic_split": "100% stable"
                }
            else:
                # Create canary deployment with a small number of replicas
                canary_replicas = max(1, int(replicas * 0.2))  # 20% of stable replicas
                
                canary_deployment = client.V1Deployment(
                    metadata=client.V1ObjectMeta(
                        name=canary_deployment_name,
                        labels={
                            "app": deployment_name,
                            "model_id": str(model_id),
                            "version": "canary"
                        }
                    ),
                    spec=client.V1DeploymentSpec(
                        replicas=canary_replicas,
                        selector=client.V1LabelSelector(
                            match_labels={
                                "app": deployment_name,
                                "version": "canary"
                            }
                        ),
                        template=client.V1PodTemplateSpec(
                            metadata=client.V1ObjectMeta(
                                labels={
                                    "app": deployment_name,
                                    "model_id": str(model_id),
                                    "version": "canary"
                                }
                            ),
                            spec=client.V1PodSpec(
                                containers=[
                                    client.V1Container(
                                        name="model-server",
                                        image="skyscope/ml-model-server:latest",
                                        ports=[
                                            client.V1ContainerPort(container_port=8000)
                                        ],
                                        env=[
                                            client.V1EnvVar(
                                                name="MODEL_ID",
                                                value=str(model_id)
                                            ),
                                            client.V1EnvVar(
                                                name="MODEL_PATH",
                                                value=model_path
                                            ),
                                            client.V1EnvVar(
                                                name="DEPLOYMENT_VERSION",
                                                value="canary"
                                            )
                                        ],
                                        resources=client.V1ResourceRequirements(
                                            requests=resources["requests"],
                                            limits=resources["limits"]
                                        ),
                                        liveness_probe=client.V1Probe(
                                            http_get=client.V1HTTPGetAction(
                                                path="/health",
                                                port=8000
                                            ),
                                            initial_delay_seconds=30,
                                            period_seconds=10
                                        ),
                                        readiness_probe=client.V1Probe(
                                            http_get=client.V1HTTPGetAction(
                                                path="/ready",
                                                port=8000
                                            ),
                                            initial_delay_seconds=15,
                                            period_seconds=5
                                        )
                                    )
                                ]
                            )
                        )
                    )
                )
                
                # Create canary deployment
                self.apps_v1.create_namespaced_deployment(
                    namespace=self.namespace,
                    body=canary_deployment
                )
                
                # Track deployment
                self.active_deployments[canary_deployment_name] = {
                    "model_id": model_id,
                    "replicas": canary_replicas,
                    "created_at": datetime.datetime.now().isoformat(),
                    "status": "active",
                    "strategy": "canary",
                    "version": "canary",
                    "service": deployment_name,
                    "traffic_percentage": 20
                }
                
                # Update stable deployment traffic percentage
                self.active_deployments[stable_deployment_name]["traffic_percentage"] = 80
                
                return {
                    "status": "success",
                    "message": f"Canary deployment created",
                    "stable_deployment": stable_deployment_name,
                    "canary_deployment": canary_deployment_name,
                    "service_name": deployment_name,
                    "endpoint": f"http://{deployment_name}.{self.namespace}.svc.cluster.local",
                    "traffic_split": "80% stable, 20% canary"
                }
        except Exception as e:
            logger.error(f"Error creating canary deployment: {e}")
            return {"status": "error", "message": str(e)}
    
    def promote_canary_to_stable(self, deployment_name: str) -> Dict[str, Any]:
        """Promote a canary deployment to stable."""
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Check if canary deployment exists
            canary_deployment_name = f"{deployment_name}-canary"
            stable_deployment_name = f"{deployment_name}-stable"
            
            try:
                canary_deployment = self.apps_v1.read_namespaced_deployment(
                    name=canary_deployment_name,
                    namespace=self.namespace
                )
                
                stable_deployment = self.apps_v1.read_namespaced_deployment(
                    name=stable_deployment_name,
                    namespace=self.namespace
                )
            except Exception as e:
                return {"status": "error", "message": f"Deployment not found: {e}"}
            
            # Get canary deployment details
            canary_container = canary_deployment.spec.template.spec.containers[0]
            canary_image = canary_container.image
            canary_env = canary_container.env
            
            # Update stable deployment with canary details
            stable_deployment.spec.template.spec.containers[0].image = canary_image
            stable_deployment.spec.template.spec.containers[0].env = canary_env
            
            # Update stable deployment
            self.apps_v1.replace_namespaced_deployment(
                name=stable_deployment_name,
                namespace=self.namespace,
                body=stable_deployment
            )
            
            # Delete canary deployment
            self.apps_v1.delete_namespaced_deployment(
                name=canary_deployment_name,
                namespace=self.namespace
            )
            
            # Update deployment tracking
            if canary_deployment_name in self.active_deployments:
                # Archive deployment history
                self.deployment_history[f"{canary_deployment_name}-{int(time.time())}"] = self.active_deployments[canary_deployment_name]
                del self.active_deployments[canary_deployment_name]
            
            # Update stable deployment tracking
            if stable_deployment_name in self.active_deployments:
                self.active_deployments[stable_deployment_name]["traffic_percentage"] = 100
                self.active_deployments[stable_deployment_name]["updated_at"] = datetime.datetime.now().isoformat()
            
            return {
                "status": "success",
                "message": "Canary promoted to stable",
                "deployment_name": stable_deployment_name,
                "service_name": deployment_name
            }
        except Exception as e:
            logger.error(f"Error promoting canary to stable: {e}")
            return {"status": "error", "message": str(e)}
    
    def rollback_blue_green(self, deployment_name: str) -> Dict[str, Any]:
        """Rollback a blue-green deployment."""
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Find blue and green deployments
            blue_deployment_name = f"{deployment_name}-blue"
            green_deployment_name = f"{deployment_name}-green"
            
            try:
                blue_deployment = self.apps_v1.read_namespaced_deployment(
                    name=blue_deployment_name,
                    namespace=self.namespace
                )
                
                green_deployment = self.apps_v1.read_namespaced_deployment(
                    name=green_deployment_name,
                    namespace=self.namespace
                )
            except Exception as e:
                return {"status": "error", "message": f"Deployment not found: {e}"}
            
            # Get current service
            service = self.core_v1.read_namespaced_service(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Determine which deployment is active
            current_color = "blue"
            if "color" in service.spec.selector and service.spec.selector["color"] == "green":
                current_color = "green"
                new_color = "blue"
            else:
                current_color = "blue"
                new_color = "green"
            
            # Update service to point to the other deployment
            service.spec.selector["color"] = new_color
            
            self.core_v1.replace_namespaced_service(
                name=deployment_name,
                namespace=self.namespace,
                body=service
            )
            
            # Update deployment tracking
            if f"{deployment_name}-{current_color}" in self.active_deployments:
                self.active_deployments[f"{deployment_name}-{current_color}"]["status"] = "inactive"
            
            if f"{deployment_name}-{new_color}" in self.active_deployments:
                self.active_deployments[f"{deployment_name}-{new_color}"]["status"] = "active"
            
            return {
                "status": "success",
                "message": f"Rolled back to {new_color} deployment",
                "deployment_name": f"{deployment_name}-{new_color}",
                "service_name": deployment_name,
                "color": new_color
            }
        except Exception as e:
            logger.error(f"Error rolling back blue-green deployment: {e}")
            return {"status": "error", "message": str(e)}
    
    def scale_deployment(self, deployment_name: str, replicas: int) -> Dict[str, Any]:
        """Scale a deployment to a specific number of replicas."""
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Scale deployment
            self.apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=self.namespace,
                body={"spec": {"replicas": replicas}}
            )
            
            # Update deployment tracking
            if deployment_name in self.active_deployments:
                self.active_deployments[deployment_name]["replicas"] = replicas
            
            return {
                "status": "success",
                "message": f"Scaled deployment {deployment_name} to {replicas} replicas",
                "deployment_name": deployment_name,
                "replicas": replicas
            }
        except Exception as e:
            logger.error(f"Error scaling deployment: {e}")
            return {"status": "error", "message": str(e)}
    
    def delete_deployment(self, deployment_name: str) -> Dict[str, Any]:
        """Delete a deployment and associated resources."""
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Delete deployment
            self.apps_v1.delete_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Try to delete associated service
            try:
                self.core_v1.delete_namespaced_service(
                    name=deployment_name,
                    namespace=self.namespace
                )
            except:
                pass
            
            # Try to delete associated HPA
            try:
                self.autoscaling_v2.delete_namespaced_horizontal_pod_autoscaler(
                    name=f"{deployment_name}-hpa",
                    namespace=self.namespace
                )
            except:
                pass
            
            # Update deployment tracking
            if deployment_name in self.active_deployments:
                # Archive deployment history
                self.deployment_history[f"{deployment_name}-{int(time.time())}"] = self.active_deployments[deployment_name]
                del self.active_deployments[deployment_name]
            
            return {
                "status": "success",
                "message": f"Deleted deployment {deployment_name}",
                "deployment_name": deployment_name
            }
        except Exception as e:
            logger.error(f"Error deleting deployment: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_deployment_status(self, deployment_name: str) -> Dict[str, Any]:
        """Get the status of a deployment."""
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Get deployment
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Get deployment status
            status = {
                "name": deployment.metadata.name,
                "replicas": deployment.spec.replicas,
                "available_replicas": deployment.status.available_replicas,
                "ready_replicas": deployment.status.ready_replicas,
                "updated_replicas": deployment.status.updated_replicas,
                "conditions": [
                    {
                        "type": condition.type,
                        "status": condition.status,
                        "reason": condition.reason,
                        "message": condition.message,
                        "last_transition_time": condition.last_transition_time
                    }
                    for condition in deployment.status.conditions
                ] if deployment.status.conditions else []
            }
            
            return {
                "status": "success",
                "deployment_status": status
            }
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return {"status": "error", "message": str(e)}
    
    def list_deployments(self) -> Dict[str, Any]:
        """List all deployments in the namespace."""
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            # Get deployments
            deployments = self.apps_v1.list_namespaced_deployment(namespace=self.namespace)
            
            # Format deployment list
            deployment_list = []
            for deployment in deployments.items:
                deployment_list.append({
                    "name": deployment.metadata.name,
                    "replicas": deployment.spec.replicas,
                    "available_replicas": deployment.status.available_replicas,
                    "labels": deployment.metadata.labels
                })
            
            return {
                "status": "success",
                "deployments": deployment_list
            }
        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return {"status": "error", "message": str(e)}
    
    def deploy_to_multiple_regions(self, model_id: int, model_path: str,
                                  deployment_name: str, regions: List[str],
                                  replicas_per_region: int = 2) -> Dict[str, Any]:
        """Deploy a model to multiple regions."""
        # Note: In a real implementation, this would involve setting up
        # deployments across multiple Kubernetes clusters in different regions.
        # For this example, we'll simulate it with different namespaces.
        try:
            if not self.api_client:
                return {"status": "error", "message": "Kubernetes client not initialized"}
            
            results = {}
            
            for region in regions:
                # Create namespace for region if it doesn't exist
                region_namespace = f"{self.namespace}-{region}"
                try:
                    self.core_v1.read_namespace(name=region_namespace)
                except:
                    namespace = client.V1Namespace(
                        metadata=client.V1ObjectMeta(name=region_namespace)
                    )
                    self.core_v1.create_namespace(namespace)
                
                # Deploy to region
                region_deployment_name = f"{deployment_name}-{region}"
                
                # Create deployment in region
                deployment = client.V1Deployment(
                    metadata=client.V1ObjectMeta(
                        name=region_deployment_name,
                        labels={
                            "app": deployment_name,
                            "model_id": str(model_id),
                            "region": region
                        }
                    ),
                    spec=client.V1DeploymentSpec(
                        replicas=replicas_per_region,
                        selector=client.V1LabelSelector(
                            match_labels={
                                "app": deployment_name,
                                "region": region
                            }
                        ),
                        template=client.V1PodTemplateSpec(
                            metadata=client.V1ObjectMeta(
                                labels={
                                    "app": deployment_name,
                                    "model_id": str(model_id),
                                    "region": region
                                }
                            ),
                            spec=client.V1PodSpec(
                                containers=[
                                    client.V1Container(
                                        name="model-server",
                                        image="skyscope/ml-model-server:latest",
                                        ports=[
                                            client.V1ContainerPort(container_port=8000)
                                        ],
                                        env=[
                                            client.V1EnvVar(
                                                name="MODEL_ID",
                                                value=str(model_id)
                                            ),
                                            client.V1EnvVar(
                                                name="MODEL_PATH",
                                                value=model_path
                                            ),
                                            client.V1EnvVar(
                                                name="DEPLOYMENT_REGION",
                                                value=region
                                            )
                                        ],
                                        resources=client.V1ResourceRequirements(
                                            requests={
                                                "cpu": "500m",
                                                "memory": "1Gi"
                                            },
                                            limits={
                                                "cpu": "2",
                                                "memory": "4Gi"
                                            }
                                        )
                                    )
                                ]
                            )
                        )
                    )
                )
                
                # Create deployment
                self.apps_v1.create_namespaced_deployment(
                    namespace=region_namespace,
                    body=deployment
                )
                
                # Create service
                service = client.V1Service(
                    metadata=client.V1ObjectMeta(
                        name=region_deployment_name
                    ),
                    spec=client.V1ServiceSpec(
                        selector={
                            "app": deployment_name,
                            "region": region
                        },
                        ports=[
                            client.V1ServicePort(
                                port=80,
                                target_port=8000
                            )
                        ]
                    )
                )
                
                self.core_v1.create_namespaced_service(
                    namespace=region_namespace,
                    body=service
                )
                
                # Track deployment
                self.active_deployments[region_deployment_name] = {
                    "model_id": model_id,
                    "replicas": replicas_per_region,
                    "created_at": datetime.datetime.now().isoformat(),
                    "status": "active",
                    "region": region,
                    "namespace": region_namespace,
                    "service": region_deployment_name
                }
                
                results[region] = {
                    "deployment_name": region_deployment_name,
                    "service_name": region_deployment_name,
                    "namespace": region_namespace,
                    "endpoint": f"http://{region_deployment_name}.{region_namespace}.svc.cluster.local"
                }
            
            return {
                "status": "success",
                "message": f"Deployed to {len(regions)} regions",
                "deployments": results
            }
        except Exception as e:
            logger.error(f"Error deploying to multiple regions: {e}")
            return {"status": "error", "message": str(e)}


#######################################################
# Model Monitoring Manager
#######################################################

class ModelMonitoringManager:
    """
    Manages model monitoring, including performance tracking,
    drift detection, and anomaly detection.
    """
    
    def __init__(self, db_session=None):
        self.session = db_session or Session()
        self.monitored_models = {}
        self.drift_detectors = {}
        self.anomaly_detectors = {}
        self.alert_thresholds = {}
        self.alert_subscribers = defaultdict(list)
        
        # Initialize metrics
        self.model_performance_gauge = Gauge('model_performance', 'Model performance metrics', ['model_id', 'metric'])
        self.model_drift_gauge = Gauge('model_drift', 'Model drift metrics', ['model_id', 'feature'])
        self.model_latency_histogram = Histogram('model_inference_latency', 'Model inference latency', ['model_id'])
        self.model_prediction_counter = Counter('model_predictions_total', 'Total number of model predictions', ['model_id', 'result'])
    
    def register_model_for_monitoring(self, model_id: int, 
                                     reference_data_path: Optional[str] = None,
                                     performance_metrics: List[str] = None,
                                     drift_detection_features: List[str] = None,
                                     alert_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Register a model for monitoring."""
        try:
            # Get model
            model = self.session.query(MLModel).filter_by(id=model_id).first()
            if not model:
                return {"status": "error", "message": f"Model with ID {model_id} not found"}
            
            # Default metrics if not provided
            if not performance_metrics:
                performance_metrics = ["accuracy", "latency", "throughput"]
            
            # Default alert thresholds if not provided
            if not alert_thresholds:
                alert_thresholds = {
                    "accuracy_drop": 0.05,  # Alert if accuracy drops by 5%
                    "latency_increase": 0.2,  # Alert if latency increases by 20%
                    "drift_threshold": 0.1,  # Alert if drift score exceeds 0.1
                    "error_rate": 0.02  # Alert if error rate exceeds 2%
                }
            
            # Store monitoring configuration
            self.monitored_models[model_id] = {
                "model_name": model.name,
                "model_version": model.version,
                "reference_data_path": reference_data_path,
                "performance_metrics": performance_metrics,
                "drift_detection_features": drift_detection_features,
                "registered_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "status": "active"
            }
            
            # Store alert thresholds
            self.alert_thresholds[model_id] = alert_thresholds
            
            # Initialize drift detector if reference data provided
            if reference_data_path and drift_detection_features:
                self._initialize_drift_detector(model_id, reference_data_path, drift_detection_features)
            
            return {
                "status": "success",
                "message": f"Model {model.name} v{model.version} registered for monitoring",
                "model_id": model_id,
                "metrics": performance_metrics,
                "alert_thresholds": alert_thresholds
            }
        except Exception as e:
            logger.error(f"Error registering model for monitoring: {e}")
            return {"status": "error", "message": str(e)}
    
    def _initialize_drift_detector(self, model_id: int, reference_data_path: str,
                                  features: List[str]) -> None:
        """Initialize a drift detector for a model."""
        try:
            # Load reference data
            if reference_data_path.endswith('.csv'):
                reference_data = pd.read_csv(reference_data_path)
            elif reference_data_path.endswith('.parquet'):
                reference_data = pd.read_parquet(reference_data_path)
            else:
                raise ValueError(f"Unsupported reference data format: {reference_data_path}")
            
            # Validate features
            for feature in features:
                if feature not in reference_data.columns:
                    raise ValueError(f"Feature '{feature}' not found in reference data")
            
            # Store reference statistics
            feature_stats = {}
            for feature in features:
                feature_data = reference_data[feature]
                
                if pd.api.types.is_numeric_dtype(feature_data):
                    # Numeric feature
                    feature_stats[feature] = {
                        "mean": feature_data.mean(),
                        "std": feature_data.std(),
                        "min": feature_data.min(),
                        "max": feature_data.max(),
                        "median": feature_data.median(),
                        "type": "numeric"
                    }
                else:
                    # Categorical feature
                    value_counts = feature_data.value_counts(normalize=True).to_dict()
                    feature_stats[feature] = {
                        "distribution": value_counts,
                        "unique_values": feature_data.nunique(),
                        "type": "categorical"
                    }
            
            # Store drift detector
            self.drift_detectors[model_id] = {
                "reference_stats": feature_stats,
                "features": features,
                "last_updated": datetime.datetime.now().isoformat(),
                "drift_scores": {}
            }
            
            logger.info(f"Drift detector initialized for model {model_id} with {len(features)} features")
        except Exception as e:
            logger.error(f"Error initializing drift detector: {e}")
            raise
    
    def track_inference(self, model_id: int, input_data: Dict[str, Any],
                       prediction: Any, ground_truth: Optional[Any] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """Track a model inference for monitoring."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Record prediction
            timestamp = datetime.datetime.now().isoformat()
            
            # Check for feature drift
            drift_detected = False
            drift_features = []
            
            if model_id in self.drift_detectors:
                drift_results = self._check_feature_drift(model_id, input_data)
                drift_detected = drift_results["drift_detected"]
                drift_features = drift_results["drift_features"]
                
                # Update drift scores in detector
                self.drift_detectors[model_id]["drift_scores"] = drift_results["drift_scores"]
            
            # Record metrics
            if latency_ms is not None:
                self.model_latency_histogram.labels(model_id=str(model_id)).observe(latency_ms)
            
            # Increment prediction counter
            result = "unknown"
            if ground_truth is not None:
                if isinstance(prediction, (list, np.ndarray)) and isinstance(ground_truth, (list, np.ndarray)):
                    # For array predictions, check if they're equal
                    result = "correct" if np.array_equal(np.array(prediction), np.array(ground_truth)) else "incorrect"
                else:
                    # For scalar predictions
                    result = "correct" if prediction == ground_truth else "incorrect"
            
            self.model_prediction_counter.labels(model_id=str(model_id), result=result).inc()
            
            # Check for alerts
            alerts = []
            if drift_detected and model_id in self.alert_thresholds:
                drift_threshold = self.alert_thresholds[model_id].get("drift_threshold", 0.1)
                for feature, score in self.drift_detectors[model_id]["drift_scores"].items():
                    if score > drift_threshold:
                        alerts.append({
                            "type": "drift_alert",
                            "feature": feature,
                            "drift_score": score,
                            "threshold": drift_threshold,
                            "timestamp": timestamp
                        })
            
            # Send alerts if any
            if alerts:
                self._send_alerts(model_id, alerts)
            
            return {
                "status": "success",
                "model_id": model_id,
                "timestamp": timestamp,
                "drift_detected": drift_detected,
                "drift_features": drift_features,
                "alerts": alerts
            }
        except Exception as e:
            logger.error(f"Error tracking inference: {e}")
            return {"status": "error", "message": str(e)}
    
    def _check_feature_drift(self, model_id: int, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for feature drift in input data."""
        drift_detector = self.drift_detectors.get(model_id)
        if not drift_detector:
            return {"drift_detected": False, "drift_features": [], "drift_scores": {}}
        
        reference_stats = drift_detector["reference_stats"]
        features = drift_detector["features"]
        
        drift_scores = {}
        drift_features = []
        drift_detected = False
        
        for feature in features:
            if feature not in input_data:
                continue
            
            value = input_data[feature]
            ref_stats = reference_stats[feature]
            
            if ref_stats["type"] == "numeric":
                # Calculate z-score for numeric features
                z_score = abs((value - ref_stats["mean"]) / max(ref_stats["std"], 1e-10))
                drift_score = min(1.0, z_score / 10.0)  # Normalize to [0, 1]
            else:
                # For categorical features, check if value is in reference distribution
                if value in ref_stats["distribution"]:
                    # Use inverse of probability as drift score
                    drift_score = 1.0 - ref_stats["distribution"].get(value, 0)
                else:
                    # Value not seen in reference data
                    drift_score = 1.0
            
            drift_scores[feature] = drift_score
            
            # Check if drift exceeds threshold
            if drift_score > 0.5:  # Arbitrary threshold for demonstration
                drift_features.append(feature)
                drift_detected = True
            
            # Update drift gauge
            self.model_drift_gauge.labels(model_id=str(model_id), feature=feature).set(drift_score)
        
        return {
            "drift_detected": drift_detected,
            "drift_features": drift_features,
            "drift_scores": drift_scores
        }
    
    def update_model_performance(self, model_id: int, 
                               metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update performance metrics for a model."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Record timestamp
            timestamp = datetime.datetime.now().isoformat()
            
            # Update metrics
            for metric_name, metric_value in metrics.items():
                self.model_performance_gauge.labels(model_id=str(model_id), metric=metric_name).set(metric_value)
            
            # Store metrics in database
            for metric_name, metric_value in metrics.items():
                deployment_metric = DeploymentMetric(
                    deployment_id=model_id,  # Assuming model_id is the deployment_id
                    metric_name=metric_name,
                    metric_value=metric_value
                )
                self.session.add(deployment_metric)
            
            self.session.commit()
            