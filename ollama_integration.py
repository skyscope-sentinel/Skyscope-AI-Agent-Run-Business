import os
import sys
import json
import time
import logging
import requests
import subprocess
import threading
import platform
import psutil
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ollama_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ollama_integration")

# Constants
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_MAX_PIPELINES = 5
OLLAMA_MODELS_DIR = Path("models/ollama")
OLLAMA_CONFIG_PATH = Path("config/ollama_config.json")
OLLAMA_METRICS_PATH = Path("logs/ollama_metrics")

# Ensure directories exist
OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_METRICS_PATH.mkdir(parents=True, exist_ok=True)
OLLAMA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

class OllamaModelType(Enum):
    """Enumeration of Ollama model types."""
    COMPLETION = "completion"
    CHAT = "chat"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"

@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str
    size: int
    modified_at: str
    model_type: OllamaModelType
    quantization: str
    format: str
    parameter_size: str
    system_requirements: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OllamaModelInfo':
        """Create model info from dictionary."""
        model_type = OllamaModelType.CHAT  # Default
        
        # Determine model type based on capabilities or tags
        capabilities = data.get("capabilities", [])
        tags = data.get("tags", [])
        
        if "embedding" in capabilities or "embedding" in tags:
            model_type = OllamaModelType.EMBEDDING
        elif "vision" in capabilities or "multimodal" in tags:
            model_type = OllamaModelType.MULTIMODAL
        elif "completion" in capabilities and "chat" not in capabilities:
            model_type = OllamaModelType.COMPLETION
        
        return cls(
            name=data.get("name", ""),
            size=data.get("size", 0),
            modified_at=data.get("modified_at", ""),
            model_type=model_type,
            quantization=data.get("quantization", ""),
            format=data.get("format", ""),
            parameter_size=data.get("parameter_size", ""),
            system_requirements=data.get("system_requirements", {}),
            capabilities=capabilities,
            tags=tags
        )

@dataclass
class OllamaConfig:
    """Configuration for Ollama integration."""
    enabled: bool = True
    host: str = OLLAMA_DEFAULT_HOST
    max_pipelines: int = OLLAMA_MAX_PIPELINES
    preferred_models: Dict[str, str] = field(default_factory=dict)
    auto_download: bool = True
    auto_fallback_to_openai: bool = True
    timeout: int = 60
    keep_alive: bool = True
    metrics_enabled: bool = True
    metrics_interval: int = 60  # seconds
    
    @classmethod
    def load(cls) -> 'OllamaConfig':
        """Load configuration from file."""
        if not OLLAMA_CONFIG_PATH.exists():
            # Create default config
            config = cls()
            config.save()
            return config
        
        try:
            with open(OLLAMA_CONFIG_PATH, 'r') as f:
                data = json.load(f)
            
            return cls(
                enabled=data.get("enabled", True),
                host=data.get("host", OLLAMA_DEFAULT_HOST),
                max_pipelines=data.get("max_pipelines", OLLAMA_MAX_PIPELINES),
                preferred_models=data.get("preferred_models", {}),
                auto_download=data.get("auto_download", True),
                auto_fallback_to_openai=data.get("auto_fallback_to_openai", True),
                timeout=data.get("timeout", 60),
                keep_alive=data.get("keep_alive", True),
                metrics_enabled=data.get("metrics_enabled", True),
                metrics_interval=data.get("metrics_interval", 60)
            )
        except Exception as e:
            logger.error(f"Error loading Ollama config: {e}")
            return cls()
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(OLLAMA_CONFIG_PATH, 'w') as f:
                json.dump({
                    "enabled": self.enabled,
                    "host": self.host,
                    "max_pipelines": self.max_pipelines,
                    "preferred_models": self.preferred_models,
                    "auto_download": self.auto_download,
                    "auto_fallback_to_openai": self.auto_fallback_to_openai,
                    "timeout": self.timeout,
                    "keep_alive": self.keep_alive,
                    "metrics_enabled": self.metrics_enabled,
                    "metrics_interval": self.metrics_interval
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving Ollama config: {e}")

class OllamaStatus(Enum):
    """Enumeration of Ollama service statuses."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    ERROR = "error"
    UNKNOWN = "unknown"

class OllamaServiceManager:
    """Manager for the Ollama service."""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig.load()
        self._status = OllamaStatus.UNKNOWN
        self._process = None
        self._status_check_thread = None
        self._stop_status_check = threading.Event()
    
    @property
    def status(self) -> OllamaStatus:
        """Get the current status of the Ollama service."""
        return self._status
    
    def check_installation(self) -> bool:
        """Check if Ollama is installed."""
        system = platform.system().lower()
        
        if system == "windows":
            # Check Windows PATH
            return self._check_command_exists("ollama")
        elif system == "darwin" or system == "linux":
            # Check Unix PATH
            return self._check_command_exists("ollama")
        
        return False
    
    def _check_command_exists(self, command: str) -> bool:
        """Check if a command exists in the PATH."""
        try:
            if platform.system().lower() == "windows":
                result = subprocess.run(["where", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(["which", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except Exception:
            return False
    
    def install_instructions(self) -> str:
        """Get installation instructions for the current platform."""
        system = platform.system().lower()
        
        if system == "windows":
            return (
                "To install Ollama on Windows:\n"
                "1. Visit https://ollama.com/download/windows\n"
                "2. Download and run the installer\n"
                "3. Follow the installation prompts\n"
                "4. Restart your computer after installation"
            )
        elif system == "darwin":
            return (
                "To install Ollama on macOS:\n"
                "1. Visit https://ollama.com/download/mac\n"
                "2. Download and open the .dmg file\n"
                "3. Drag Ollama to your Applications folder\n"
                "4. Open Ollama from your Applications"
            )
        elif system == "linux":
            return (
                "To install Ollama on Linux:\n"
                "1. Run the following command in your terminal:\n"
                "   curl -fsSL https://ollama.com/install.sh | sh\n"
                "2. Start Ollama by running: ollama serve"
            )
        
        return "Please visit https://ollama.com/download for installation instructions."
    
    def check_status(self) -> OllamaStatus:
        """Check the status of the Ollama service."""
        try:
            response = requests.get(f"{self.config.host}/api/tags", timeout=2)
            if response.status_code == 200:
                self._status = OllamaStatus.RUNNING
                return self._status
        except Exception:
            pass
        
        # Check if process is running
        if self._process and self._process.poll() is None:
            self._status = OllamaStatus.STARTING
            return self._status
        
        # Check if ollama is running as a system service
        if platform.system().lower() != "windows":
            try:
                result = subprocess.run(
                    ["pgrep", "-f", "ollama serve"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._status = OllamaStatus.RUNNING
                    return self._status
            except Exception:
                pass
        
        self._status = OllamaStatus.STOPPED
        return self._status
    
    def start(self) -> bool:
        """Start the Ollama service."""
        if self.check_status() == OllamaStatus.RUNNING:
            return True
        
        if not self.check_installation():
            logger.error("Ollama is not installed.")
            return False
        
        try:
            system = platform.system().lower()
            
            if system == "windows":
                # Start Ollama on Windows
                self._process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            elif system == "darwin":
                # Start Ollama on macOS
                self._process = subprocess.Popen(
                    ["open", "-a", "Ollama"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            elif system == "linux":
                # Start Ollama on Linux
                self._process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            # Start status check thread
            self._stop_status_check.clear()
            self._status_check_thread = threading.Thread(target=self._status_check_loop)
            self._status_check_thread.daemon = True
            self._status_check_thread.start()
            
            # Wait for service to start
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 second timeout
                if self.check_status() == OllamaStatus.RUNNING:
                    return True
                time.sleep(1)
            
            return self.check_status() == OllamaStatus.RUNNING
        except Exception as e:
            logger.error(f"Error starting Ollama: {e}")
            self._status = OllamaStatus.ERROR
            return False
    
    def _status_check_loop(self) -> None:
        """Background thread to check Ollama status."""
        while not self._stop_status_check.is_set():
            self.check_status()
            time.sleep(5)
    
    def stop(self) -> bool:
        """Stop the Ollama service."""
        if self.check_status() != OllamaStatus.RUNNING:
            return True
        
        # Stop status check thread
        if self._status_check_thread:
            self._stop_status_check.set()
            self._status_check_thread.join(timeout=2)
            self._status_check_thread = None
        
        # Stop process if we started it
        if self._process:
            try:
                if platform.system().lower() == "windows":
                    # Windows requires different handling
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(self._process.pid)])
                else:
                    self._process.terminate()
                    self._process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping Ollama process: {e}")
                try:
                    self._process.kill()
                except Exception:
                    pass
            
            self._process = None
        
        # On Unix systems, try to kill any running ollama processes
        if platform.system().lower() != "windows":
            try:
                subprocess.run(["pkill", "-f", "ollama serve"], check=False)
            except Exception:
                pass
        
        # Verify it's stopped
        return self.check_status() != OllamaStatus.RUNNING
    
    def restart(self) -> bool:
        """Restart the Ollama service."""
        self.stop()
        time.sleep(2)
        return self.start()

class OllamaModelManager:
    """Manager for Ollama models."""
    
    def __init__(self, config: OllamaConfig = None, service_manager: OllamaServiceManager = None):
        self.config = config or OllamaConfig.load()
        self.service_manager = service_manager or OllamaServiceManager(self.config)
        self._models_cache = {}
        self._last_cache_update = 0
        self._cache_lock = threading.Lock()
    
    def list_models(self, force_refresh: bool = False) -> List[OllamaModelInfo]:
        """List available Ollama models."""
        # Check if cache is valid (less than 60 seconds old)
        current_time = time.time()
        with self._cache_lock:
            if not force_refresh and self._models_cache and current_time - self._last_cache_update < 60:
                return list(self._models_cache.values())
        
        if self.service_manager.status != OllamaStatus.RUNNING:
            if not self.service_manager.start():
                logger.error("Ollama service is not running and could not be started.")
                return []
        
        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                models = []
                model_data = response.json().get("models", [])
                
                for model in model_data:
                    # Get detailed model info
                    model_info = self._get_model_details(model.get("name", ""))
                    if model_info:
                        models.append(model_info)
                        with self._cache_lock:
                            self._models_cache[model_info.name] = model_info
                
                with self._cache_lock:
                    self._last_cache_update = current_time
                
                return models
            else:
                logger.error(f"Error listing Ollama models: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []
    
    def _get_model_details(self, model_name: str) -> Optional[OllamaModelInfo]:
        """Get detailed information about a model."""
        try:
            response = requests.get(
                f"{self.config.host}/api/show",
                params={"name": model_name},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                model_data = response.json()
                return OllamaModelInfo.from_dict({
                    "name": model_name,
                    "size": model_data.get("size", 0),
                    "modified_at": model_data.get("modified_at", ""),
                    "quantization": model_data.get("details", {}).get("quantization_level", ""),
                    "format": model_data.get("details", {}).get("format", ""),
                    "parameter_size": model_data.get("details", {}).get("parameter_size", ""),
                    "capabilities": model_data.get("details", {}).get("capabilities", []),
                    "tags": model_data.get("details", {}).get("tags", [])
                })
            else:
                logger.error(f"Error getting model details for {model_name}: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting model details for {model_name}: {e}")
            return None
    
    def get_model(self, model_name: str) -> Optional[OllamaModelInfo]:
        """Get information about a specific model."""
        # Check cache first
        with self._cache_lock:
            if model_name in self._models_cache:
                return self._models_cache[model_name]
        
        # Get all models (will update cache)
        self.list_models()
        
        # Check cache again
        with self._cache_lock:
            return self._models_cache.get(model_name)
    
    def download_model(self, model_name: str) -> bool:
        """Download an Ollama model."""
        if self.service_manager.status != OllamaStatus.RUNNING:
            if not self.service_manager.start():
                logger.error("Ollama service is not running and could not be started.")
                return False
        
        try:
            # Check if model already exists
            if self.get_model(model_name):
                logger.info(f"Model {model_name} is already downloaded.")
                return True
            
            logger.info(f"Downloading model {model_name}...")
            
            response = requests.post(
                f"{self.config.host}/api/pull",
                json={"name": model_name},
                timeout=None  # No timeout for downloads
            )
            
            if response.status_code == 200:
                # Invalidate cache
                with self._cache_lock:
                    self._last_cache_update = 0
                
                logger.info(f"Model {model_name} downloaded successfully.")
                return True
            else:
                logger.error(f"Error downloading model {model_name}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete an Ollama model."""
        if self.service_manager.status != OllamaStatus.RUNNING:
            if not self.service_manager.start():
                logger.error("Ollama service is not running and could not be started.")
                return False
        
        try:
            response = requests.delete(
                f"{self.config.host}/api/delete",
                json={"name": model_name},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                # Invalidate cache
                with self._cache_lock:
                    if model_name in self._models_cache:
                        del self._models_cache[model_name]
                    self._last_cache_update = 0
                
                logger.info(f"Model {model_name} deleted successfully.")
                return True
            else:
                logger.error(f"Error deleting model {model_name}: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            return False
    
    def get_recommended_models(self) -> Dict[str, str]:
        """Get recommended models for different tasks."""
        return {
            "chat": "llama3",
            "completion": "llama3",
            "embedding": "nomic-embed-text",
            "multimodal": "llava"
        }
    
    def get_model_for_task(self, task_type: str) -> Optional[str]:
        """Get the recommended model for a specific task."""
        # Check user preferred models first
        if task_type in self.config.preferred_models:
            model_name = self.config.preferred_models[task_type]
            if self.get_model(model_name) or (self.config.auto_download and self.download_model(model_name)):
                return model_name
        
        # Fall back to recommended models
        recommended = self.get_recommended_models()
        if task_type in recommended:
            model_name = recommended[task_type]
            if self.get_model(model_name) or (self.config.auto_download and self.download_model(model_name)):
                return model_name
        
        # If no specific model found, return any suitable model
        models = self.list_models()
        if models:
            return models[0].name
        
        return None

class OllamaPipelineManager:
    """Manager for Ollama pipelines."""
    
    def __init__(self, config: OllamaConfig = None, model_manager: OllamaModelManager = None):
        self.config = config or OllamaConfig.load()
        self.model_manager = model_manager or OllamaModelManager(self.config)
        self._active_pipelines = {}
        self._pipeline_lock = threading.Lock()
        self._metrics_thread = None
        self._stop_metrics = threading.Event()
        
        # Start metrics collection if enabled
        if self.config.metrics_enabled:
            self._start_metrics_collection()
    
    def _start_metrics_collection(self) -> None:
        """Start collecting metrics in a background thread."""
        if self._metrics_thread is not None:
            return
        
        self._stop_metrics.clear()
        self._metrics_thread = threading.Thread(target=self._metrics_loop)
        self._metrics_thread.daemon = True
        self._metrics_thread.start()
    
    def _metrics_loop(self) -> None:
        """Background thread to collect metrics."""
        while not self._stop_metrics.is_set():
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            # Sleep for the configured interval
            self._stop_metrics.wait(self.config.metrics_interval)
    
    def _collect_metrics(self) -> None:
        """Collect metrics about active pipelines."""
        timestamp = datetime.now().isoformat()
        metrics = {
            "timestamp": timestamp,
            "active_pipelines": len(self._active_pipelines),
            "pipelines": {},
            "system": self._get_system_metrics()
        }
        
        # Collect metrics for each pipeline
        with self._pipeline_lock:
            for pipeline_id, pipeline in self._active_pipelines.items():
                metrics["pipelines"][pipeline_id] = {
                    "model": pipeline.get("model", ""),
                    "start_time": pipeline.get("start_time", ""),
                    "requests": pipeline.get("requests", 0),
                    "tokens_in": pipeline.get("tokens_in", 0),
                    "tokens_out": pipeline.get("tokens_out", 0),
                    "last_activity": pipeline.get("last_activity", "")
                }
        
        # Save metrics to file
        metrics_file = OLLAMA_METRICS_PATH / f"metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        try:
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + "\n")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024 ** 3),
                "memory_total_gb": memory.total / (1024 ** 3)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {}
    
    def allocate_pipeline(self, model_name: Optional[str] = None, task_type: str = "chat") -> Optional[str]:
        """Allocate a pipeline for the specified model."""
        with self._pipeline_lock:
            # Check if we've reached the maximum number of pipelines
            if len(self._active_pipelines) >= self.config.max_pipelines:
                logger.warning(f"Maximum number of Ollama pipelines ({self.config.max_pipelines}) reached.")
                return None
        
        # Ensure Ollama service is running
        if self.model_manager.service_manager.status != OllamaStatus.RUNNING:
            if not self.model_manager.service_manager.start():
                logger.error("Ollama service is not running and could not be started.")
                return None
        
        # If no model specified, get recommended model for task
        if not model_name:
            model_name = self.model_manager.get_model_for_task(task_type)
            if not model_name:
                logger.error(f"No suitable model found for task type: {task_type}")
                return None
        
        # Check if model exists or needs to be downloaded
        model_info = self.model_manager.get_model(model_name)
        if not model_info:
            if self.config.auto_download:
                if not self.model_manager.download_model(model_name):
                    logger.error(f"Failed to download model: {model_name}")
                    return None
            else:
                logger.error(f"Model not found: {model_name}")
                return None
        
        # Generate a unique pipeline ID
        pipeline_id = f"ollama_pipeline_{int(time.time())}_{id(model_name)}"
        
        # Register the pipeline
        with self._pipeline_lock:
            self._active_pipelines[pipeline_id] = {
                "model": model_name,
                "start_time": datetime.now().isoformat(),
                "requests": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "last_activity": datetime.now().isoformat()
            }
        
        logger.info(f"Allocated Ollama pipeline {pipeline_id} for model {model_name}")
        return pipeline_id
    
    def release_pipeline(self, pipeline_id: str) -> bool:
        """Release a pipeline."""
        with self._pipeline_lock:
            if pipeline_id in self._active_pipelines:
                del self._active_pipelines[pipeline_id]
                logger.info(f"Released Ollama pipeline {pipeline_id}")
                return True
            else:
                logger.warning(f"Pipeline not found: {pipeline_id}")
                return False
    
    def get_pipeline_info(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a pipeline."""
        with self._pipeline_lock:
            return self._active_pipelines.get(pipeline_id)
    
    def update_pipeline_metrics(self, pipeline_id: str, tokens_in: int = 0, tokens_out: int = 0) -> bool:
        """Update metrics for a pipeline."""
        with self._pipeline_lock:
            if pipeline_id in self._active_pipelines:
                self._active_pipelines[pipeline_id]["requests"] += 1
                self._active_pipelines[pipeline_id]["tokens_in"] += tokens_in
                self._active_pipelines[pipeline_id]["tokens_out"] += tokens_out
                self._active_pipelines[pipeline_id]["last_activity"] = datetime.now().isoformat()
                return True
            else:
                logger.warning(f"Pipeline not found for metrics update: {pipeline_id}")
                return False
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get all active pipelines."""
        with self._pipeline_lock:
            return self._active_pipelines.copy()
    
    def get_pipeline_count(self) -> int:
        """Get the number of active pipelines."""
        with self._pipeline_lock:
            return len(self._active_pipelines)
    
    def cleanup_inactive_pipelines(self, max_idle_time: int = 300) -> int:
        """Clean up inactive pipelines."""
        now = datetime.now()
        to_release = []
        
        with self._pipeline_lock:
            for pipeline_id, info in self._active_pipelines.items():
                last_activity = datetime.fromisoformat(info["last_activity"])
                if (now - last_activity).total_seconds() > max_idle_time:
                    to_release.append(pipeline_id)
        
        # Release pipelines outside the lock to avoid deadlocks
        for pipeline_id in to_release:
            self.release_pipeline(pipeline_id)
        
        return len(to_release)

class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, config: OllamaConfig = None, pipeline_manager: OllamaPipelineManager = None):
        self.config = config or OllamaConfig.load()
        self.pipeline_manager = pipeline_manager or OllamaPipelineManager(self.config)
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Callable[[], Dict[str, Any]]]:
        """Create a chat completion."""
        # Allocate pipeline if not provided
        if not pipeline_id:
            pipeline_id = self.pipeline_manager.allocate_pipeline(model, "chat")
            if not pipeline_id and self.config.auto_fallback_to_openai:
                logger.info("Falling back to OpenAI for chat completion")
                return self._fallback_to_openai("chat", messages, model, temperature, max_tokens, stream, **kwargs)
            elif not pipeline_id:
                raise ValueError("Failed to allocate Ollama pipeline and auto_fallback_to_openai is disabled")
        
        # Get pipeline info
        pipeline_info = self.pipeline_manager.get_pipeline_info(pipeline_id)
        if not pipeline_info:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")
        
        # Use model from pipeline if not specified
        if not model:
            model = pipeline_info["model"]
        
        # Count input tokens (approximate)
        tokens_in = sum(len(m.get("content", "").split()) for m in messages)
        
        try:
            # Prepare request
            request_data = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {
                    "temperature": temperature
                }
            }
            
            if max_tokens:
                request_data["options"]["num_predict"] = max_tokens
            
            # Add any additional options
            for key, value in kwargs.items():
                if key not in request_data["options"]:
                    request_data["options"][key] = value
            
            # Make request
            if not stream:
                response = requests.post(
                    f"{self.config.host}/api/chat",
                    json=request_data,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Update metrics
                    tokens_out = len(result.get("message", {}).get("content", "").split())
                    self.pipeline_manager.update_pipeline_metrics(pipeline_id, tokens_in, tokens_out)
                    
                    # Format response like OpenAI
                    return self._format_chat_response(result, model)
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    if self.config.auto_fallback_to_openai:
                        logger.info("Falling back to OpenAI after Ollama error")
                        return self._fallback_to_openai("chat", messages, model, temperature, max_tokens, stream, **kwargs)
                    else:
                        raise ValueError(f"Ollama API error: {response.status_code} - {response.text}")
            else:
                # Streaming response
                return self._stream_chat_completion(
                    request_data, pipeline_id, tokens_in, model
                )
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            if self.config.auto_fallback_to_openai:
                logger.info("Falling back to OpenAI after exception")
                return self._fallback_to_openai("chat", messages, model, temperature, max_tokens, stream, **kwargs)
            else:
                raise
    
    def _stream_chat_completion(
        self, 
        request_data: Dict[str, Any], 
        pipeline_id: str,
        tokens_in: int,
        model: str
    ) -> Callable[[], Dict[str, Any]]:
        """Stream a chat completion."""
        # Start streaming request in a separate thread
        response_queue = []
        error_flag = [False]
        done_event = threading.Event()
        tokens_out = [0]
        
        def stream_worker():
            try:
                with requests.post(
                    f"{self.config.host}/api/chat",
                    json=request_data,
                    stream=True,
                    timeout=self.config.timeout
                ) as response:
                    if response.status_code != 200:
                        error_flag[0] = True
                        response_queue.append({
                            "error": f"Ollama API error: {response.status_code} - {response.text}"
                        })
                        done_event.set()
                        return
                    
                    for line in response.iter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                response_queue.append(chunk)
                                
                                # Count output tokens
                                if "message" in chunk and "content" in chunk["message"]:
                                    tokens_out[0] += len(chunk["message"]["content"].split())
                            except json.JSONDecodeError:
                                pass
                    
                    # Update metrics
                    self.pipeline_manager.update_pipeline_metrics(pipeline_id, tokens_in, tokens_out[0])
                    done_event.set()
            except Exception as e:
                error_flag[0] = True
                response_queue.append({"error": str(e)})
                done_event.set()
        
        # Start the worker thread
        worker_thread = threading.Thread(target=stream_worker)
        worker_thread.daemon = True
        worker_thread.start()
        
        # Return a generator function
        def generate_responses():
            last_idx = 0
            
            while not done_event.is_set() or last_idx < len(response_queue):
                # Check if there are new chunks
                if last_idx < len(response_queue):
                    chunk = response_queue[last_idx]
                    last_idx += 1
                    
                    # Check for error
                    if "error" in chunk:
                        if self.config.auto_fallback_to_openai:
                            logger.info("Falling back to OpenAI during streaming")
                            # Cannot actually fallback during streaming, so return error
                            yield {"error": chunk["error"], "fallback_failed": True}
                        else:
                            yield {"error": chunk["error"]}
                        return
                    
                    # Format response like OpenAI streaming
                    yield self._format_chat_stream_response(chunk, model)
                else:
                    # Wait for more chunks
                    time.sleep(0.01)
        
        return generate_responses
    
    def _format_chat_response(self, response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format Ollama response like OpenAI response."""
        message = response.get("message", {})
        
        return {
            "id": f"ollama-{int(time.time())}-{id(response)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", "")
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            },
            "provider": "ollama"
        }
    
    def _format_chat_stream_response(self, chunk: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Format Ollama streaming response like OpenAI streaming response."""
        message = chunk.get("message", {})
        
        return {
            "id": f"ollama-{int(time.time())}-{id(chunk)}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", "")
                    },
                    "finish_reason": "stop" if chunk.get("done", False) else None
                }
            ],
            "provider": "ollama"
        }
    
    def _fallback_to_openai(
        self, 
        api_type: str, 
        *args, 
        **kwargs
    ) -> Union[Dict[str, Any], Callable[[], Dict[str, Any]]]:
        """Fallback to OpenAI API."""
        try:
            # Import OpenAI client dynamically to avoid circular imports
            from openai_unofficial import OpenAI
            client = OpenAI()
            
            if api_type == "chat":
                messages, model, temperature, max_tokens, stream = args
                
                # Map Ollama model to OpenAI model
                openai_model = "gpt-3.5-turbo"  # Default fallback
                
                # Call OpenAI API
                if not stream:
                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens if max_tokens else None,
                        stream=False
                    )
                    # Add provider field to distinguish the source
                    response_dict = response.model_dump()
                    response_dict["provider"] = "openai"
                    return response_dict
                else:
                    # Return a generator function for streaming
                    stream_resp = client.chat.completions.create(
                        model=openai_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens if max_tokens else None,
                        stream=True
                    )
                    
                    def generate_openai_responses():
                        for chunk in stream_resp:
                            chunk_dict = chunk.model_dump()
                            chunk_dict["provider"] = "openai"
                            yield chunk_dict
                    
                    return generate_openai_responses
            else:
                raise ValueError(f"Unsupported API type for fallback: {api_type}")
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {e}")
            raise ValueError(f"Both Ollama and OpenAI fallback failed: {e}")
    
    def completion(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Callable[[], Dict[str, Any]]]:
        """Create a completion."""
        # Convert to chat format
        messages = [{"role": "user", "content": prompt}]
        
        # Use chat completion internally
        return self.chat_completion(
            messages=messages,
            model=model,
            pipeline_id=pipeline_id,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
    
    def embeddings(
        self, 
        input: Union[str, List[str]], 
        model: Optional[str] = None,
        pipeline_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create embeddings."""
        # Allocate pipeline if not provided
        if not pipeline_id:
            pipeline_id = self.pipeline_manager.allocate_pipeline(model, "embedding")
            if not pipeline_id and self.config.auto_fallback_to_openai:
                logger.info("Falling back to OpenAI for embeddings")
                return self._fallback_to_openai("embeddings", input, model, **kwargs)
            elif not pipeline_id:
                raise ValueError("Failed to allocate Ollama pipeline and auto_fallback_to_openai is disabled")
        
        # Get pipeline info
        pipeline_info = self.pipeline_manager.get_pipeline_info(pipeline_id)
        if not pipeline_info:
            raise ValueError(f"Invalid pipeline ID: {pipeline_id}")
        
        # Use model from pipeline if not specified
        if not model:
            model = pipeline_info["model"]
        
        # Handle single string or list of strings
        inputs = [input] if isinstance(input, str) else input
        
        # Count input tokens (approximate)
        tokens_in = sum(len(text.split()) for text in inputs)
        
        try:
            embeddings = []
            
            for text in inputs:
                response = requests.post(
                    f"{self.config.host}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embeddings.append(result.get("embedding", []))
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    if self.config.auto_fallback_to_openai:
                        logger.info("Falling back to OpenAI after Ollama error")
                        return self._fallback_to_openai("embeddings", input, model, **kwargs)
                    else:
                        raise ValueError(f"Ollama API error: {response.status_code} - {response.text}")
            
            # Update metrics (no output tokens for embeddings)
            self.pipeline_manager.update_pipeline_metrics(pipeline_id, tokens_in, 0)
            
            # Format response like OpenAI
            return {
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embedding,
                        "index": i
                    }
                    for i, embedding in enumerate(embeddings)
                ],
                "model": model,
                "usage": {
                    "prompt_tokens": tokens_in,
                    "total_tokens": tokens_in
                },
                "provider": "ollama"
            }
        except Exception as e:
            logger.error(f"Error in embeddings: {e}")
            if self.config.auto_fallback_to_openai:
                logger.info("Falling back to OpenAI after exception")
                return self._fallback_to_openai("embeddings", input, model, **kwargs)
            else:
                raise

class OllamaAdapter:
    """Adapter for using Ollama with the same interface as OpenAI."""
    
    def __init__(self):
        self.config = OllamaConfig.load()
        self.service_manager = OllamaServiceManager(self.config)
        self.model_manager = OllamaModelManager(self.config, self.service_manager)
        self.pipeline_manager = OllamaPipelineManager(self.config, self.model_manager)
        self.client = OllamaClient(self.config, self.pipeline_manager)
        
        # Initialize chat and embeddings as properties
        self.chat = OllamaChatAdapter(self.client)
        self.embeddings = OllamaEmbeddingsAdapter(self.client)
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        return self.service_manager.check_installation()
    
    def is_running(self) -> bool:
        """Check if Ollama service is running."""
        return self.service_manager.check_status() == OllamaStatus.RUNNING
    
    def start_service(self) -> bool:
        """Start the Ollama service."""
        return self.service_manager.start()
    
    def stop_service(self) -> bool:
        """Stop the Ollama service."""
        return self.service_manager.stop()
    
    def get_active_pipelines(self) -> Dict[str, Dict[str, Any]]:
        """Get all active pipelines."""
        return self.pipeline_manager.get_active_pipelines()
    
    def get_available_models(self) -> List[OllamaModelInfo]:
        """Get all available models."""
        return self.model_manager.list_models()
    
    def download_model(self, model_name: str) -> bool:
        """Download a model."""
        return self.model_manager.download_model(model_name)
    
    def get_installation_instructions(self) -> str:
        """Get installation instructions."""
        return self.service_manager.install_instructions()

class OllamaChatAdapter:
    """Adapter for Ollama chat completions."""
    
    def __init__(self, client: OllamaClient):
        self.client = client
        self.completions = self
    
    def create(self, *args, **kwargs):
        """Create a chat completion."""
        return self.client.chat_completion(*args, **kwargs)

class OllamaEmbeddingsAdapter:
    """Adapter for Ollama embeddings."""
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    def create(self, *args, **kwargs):
        """Create embeddings."""
        return self.client.embeddings(*args, **kwargs)

# Initialize the adapter as a singleton
_ollama_adapter = None

def get_ollama_adapter() -> OllamaAdapter:
    """Get the Ollama adapter singleton."""
    global _ollama_adapter
    if _ollama_adapter is None:
        _ollama_adapter = OllamaAdapter()
    return _ollama_adapter

# Utility functions for easy access
def is_ollama_available() -> bool:
    """Check if Ollama is available."""
    return get_ollama_adapter().is_available()

def is_ollama_running() -> bool:
    """Check if Ollama service is running."""
    return get_ollama_adapter().is_running()

def start_ollama_service() -> bool:
    """Start the Ollama service."""
    return get_ollama_adapter().start_service()

def get_ollama_models() -> List[OllamaModelInfo]:
    """Get all available Ollama models."""
    return get_ollama_adapter().get_available_models()

def get_ollama_installation_instructions() -> str:
    """Get Ollama installation instructions."""
    return get_ollama_adapter().get_installation_instructions()

def get_ollama_client() -> OllamaAdapter:
    """Get an Ollama client with OpenAI-compatible interface."""
    return get_ollama_adapter()

# Example usage
if __name__ == "__main__":
    # Initialize the Ollama adapter
    ollama = get_ollama_adapter()
    
    # Check if Ollama is available
    if not ollama.is_available():
        print("Ollama is not installed.")
        print(ollama.get_installation_instructions())
        sys.exit(1)
    
    # Start Ollama service if not running
    if not ollama.is_running():
        print("Starting Ollama service...")
        if not ollama.start_service():
            print("Failed to start Ollama service.")
            sys.exit(1)
    
    # List available models
    print("Available models:")
    for model in ollama.get_available_models():
        print(f"- {model.name} ({model.parameter_size}, {model.quantization})")
    
    # Create a chat completion
    print("\nTesting chat completion:")
    response = ollama.chat.completions.create(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="llama3"
    )
    print(f"Response: {response['choices'][0]['message']['content']}")
    
    # Show active pipelines
    print("\nActive pipelines:")
    for pipeline_id, info in ollama.get_active_pipelines().items():
        print(f"- {pipeline_id}: {info['model']}")
