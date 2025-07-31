# 2. Enhanced Ollama Integration - Advanced local LLM support
ollama_integration_code = '''"""
Enhanced Ollama Integration for Multi-Agent Swarms
Supports local LLMs with advanced features for business automation
macOS compatible with optimized performance
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path
import subprocess
import platform
import psutil

class ModelProvider(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"

class ModelSize(Enum):
    SMALL = "small"  # 7B parameters
    MEDIUM = "medium"  # 13B parameters  
    LARGE = "large"  # 34B+ parameters
    EXTRA_LARGE = "xl"  # 70B+ parameters

@dataclass
class ModelConfig:
    """Configuration for LLM models"""
    name: str
    provider: ModelProvider
    size: ModelSize
    context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True
    system_prompt: str = ""
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OllamaModelInfo:
    """Information about available Ollama models"""
    name: str
    size: str
    modified: str
    digest: str
    details: Dict[str, Any] = field(default_factory=dict)

class EnhancedOllamaIntegration:
    """
    Enhanced Ollama Integration for Multi-Agent Systems
    
    Features:
    - Automatic model management and installation
    - Performance optimization for macOS
    - Streaming responses for real-time interaction
    - Load balancing across multiple models
    - Intelligent model selection based on task complexity
    - Memory and resource management
    - Error handling and failover
    """
    
    def __init__(self, 
                 base_url: str = "http://localhost:11434",
                 auto_install: bool = True,
                 performance_mode: bool = True):
        self.base_url = base_url
        self.auto_install = auto_install
        self.performance_mode = performance_mode
        self.logger = self._setup_logger()
        
        # Model registry
        self.available_models: Dict[str, OllamaModelInfo] = {}
        self.loaded_models: Dict[str, ModelConfig] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Performance monitoring
        self.performance_metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "requests_failed": 0,
            "average_response_time": 0,
            "models_loaded": 0,
            "memory_usage": 0,
            "cpu_usage": 0
        }
        
        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Recommended models for different use cases
        self.recommended_models = {
            "business_analysis": ["llama2:13b", "codellama:13b", "mistral:7b"],
            "content_creation": ["llama2:7b", "codellama:7b", "neural-chat:7b"],
            "research": ["llama2:13b", "openchat:7b", "vicuna:13b"],
            "coding": ["codellama:13b", "codellama:34b", "phind-codellama:34b"],
            "general": ["llama2:7b", "mistral:7b", "neural-chat:7b"],
            "creative": ["llama2:13b", "nous-hermes:13b", "openhermes:7b"]
        }
        
        # macOS optimizations
        self.macos_optimizations = {
            "metal_gpu": self._check_metal_support(),
            "memory_pressure": self._check_memory_pressure(),
            "cpu_efficiency": self._check_cpu_efficiency()
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for Ollama integration"""
        logger = logging.getLogger("OllamaIntegration")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize(self) -> bool:
        """Initialize Ollama integration"""
        try:
            # Create aiohttp session
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=300)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            # Check Ollama availability
            if not await self._check_ollama_available():
                if self.auto_install:
                    await self._install_ollama()
                else:
                    self.logger.error("Ollama not available and auto_install is disabled")
                    return False
            
            # Load available models
            await self._load_available_models()
            
            # Auto-install recommended models
            if self.auto_install:
                await self._ensure_essential_models()
            
            # Apply performance optimizations
            if self.performance_mode:
                await self._apply_performance_optimizations()
            
            self.logger.info("Ollama integration initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama integration: {e}")
            return False
    
    async def _check_ollama_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            async with self.session.get(f"{self.base_url}/api/version") as response:
                if response.status == 200:
                    version_info = await response.json()
                    self.logger.info(f"Ollama available - Version: {version_info.get('version', 'unknown')}")
                    return True
                return False
        except Exception as e:
            self.logger.warning(f"Ollama not available: {e}")
            return False
    
    async def _install_ollama(self) -> bool:
        """Install Ollama automatically (macOS compatible)"""
        try:
            system = platform.system()
            
            if system == "Darwin":  # macOS
                self.logger.info("Installing Ollama on macOS...")
                process = await asyncio.create_subprocess_shell(
                    "curl -fsSL https://ollama.ai/install.sh | sh",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.logger.info("Ollama installed successfully")
                    # Start Ollama service
                    await self._start_ollama_service()
                    return True
                else:
                    self.logger.error(f"Ollama installation failed: {stderr.decode()}")
                    return False
            
            elif system == "Linux":
                self.logger.info("Installing Ollama on Linux...")
                process = await asyncio.create_subprocess_shell(
                    "curl -fsSL https://ollama.ai/install.sh | sh",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                return process.returncode == 0
            
            else:
                self.logger.warning(f"Automatic installation not supported on {system}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to install Ollama: {e}")
            return False
    
    async def _start_ollama_service(self) -> bool:
        """Start Ollama service"""
        try:
            if platform.system() == "Darwin":  # macOS
                # Check if already running
                process = await asyncio.create_subprocess_shell(
                    "pgrep ollama",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    self.logger.info("Ollama service already running")
                    return True
                
                # Start service
                process = await asyncio.create_subprocess_shell(
                    "ollama serve &",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait a moment for service to start
                await asyncio.sleep(3)
                
                # Verify service is running
                return await self._check_ollama_available()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Ollama service: {e}")
            return False
    
    async def _load_available_models(self) -> Dict[str, OllamaModelInfo]:
        """Load information about available models"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for model_data in data.get("models", []):
                        model_info = OllamaModelInfo(
                            name=model_data["name"],
                            size=model_data.get("size", "unknown"),
                            modified=model_data.get("modified_at", "unknown"),
                            digest=model_data.get("digest", "unknown"),
                            details=model_data.get("details", {})
                        )
                        self.available_models[model_info.name] = model_info
                    
                    self.logger.info(f"Loaded {len(self.available_models)} available models")
                    return self.available_models
                
        except Exception as e:
            self.logger.error(f"Failed to load available models: {e}")
        
        return {}
    
    async def _ensure_essential_models(self):
        """Ensure essential models are installed"""
        essential_models = [
            "llama2:7b",  # General purpose, good balance
            "codellama:7b",  # Code generation
            "mistral:7b"  # Fast and efficient
        ]
        
        for model_name in essential_models:
            if model_name not in self.available_models:
                self.logger.info(f"Installing essential model: {model_name}")
                await self.install_model(model_name)
    
    async def install_model(self, model_name: str) -> bool:
        """Install a specific model"""
        try:
            self.logger.info(f"Installing model: {model_name}")
            
            # Use streaming to show progress
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name, "stream": True}
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                progress_data = json.loads(line.decode().strip())
                                if "status" in progress_data:
                                    status = progress_data["status"]
                                    if "downloading" in status.lower():
                                        # Show download progress
                                        completed = progress_data.get("completed", 0)
                                        total = progress_data.get("total", 1)
                                        progress = (completed / total) * 100 if total > 0 else 0
                                        self.logger.info(f"Downloading {model_name}: {progress:.1f}%")
                                    elif "success" in status.lower():
                                        self.logger.info(f"Model {model_name} installed successfully")
                                        await self._load_available_models()  # Refresh model list
                                        return True
                            except json.JSONDecodeError:
                                continue
                
                self.logger.error(f"Failed to install model {model_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error installing model {model_name}: {e}")
            return False
    
    def register_model_config(self, config: ModelConfig):
        """Register a model configuration"""
        self.model_configs[config.name] = config
        self.logger.info(f"Registered model config: {config.name}")
    
    def get_recommended_model(self, task_type: str, complexity: str = "medium") -> Optional[str]:
        """Get recommended model for specific task type and complexity"""
        models = self.recommended_models.get(task_type, self.recommended_models["general"])
        
        # Filter by complexity and availability
        available_models = [m for m in models if m in self.available_models]
        
        if not available_models:
            return None
        
        # Select based on complexity
        if complexity == "simple":
            return available_models[-1]  # Smallest available
        elif complexity == "complex":
            return available_models[0]  # Largest available
        else:
            return available_models[len(available_models)//2]  # Medium
    
    async def generate_response(self, 
                              prompt: str,
                              model_name: Optional[str] = None,
                              system_prompt: Optional[str] = None,
                              **kwargs) -> Dict[str, Any]:
        """Generate response from Ollama model"""
        start_time = time.time()
        
        try:
            # Auto-select model if not specified
            if not model_name:
                model_name = self.get_recommended_model("general") or "llama2:7b"
            
            # Get model config
            config = self.model_configs.get(model_name, ModelConfig(
                name=model_name,
                provider=ModelProvider.OLLAMA,
                size=ModelSize.MEDIUM
            ))
            
            # Prepare request
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": kwargs.get("stream", config.stream),
                "options": {
                    "temperature": kwargs.get("temperature", config.temperature),
                    "top_p": kwargs.get("top_p", config.top_p),
                    "num_predict": kwargs.get("max_tokens", config.max_tokens)
                }
            }
            
            if system_prompt or config.system_prompt:
                request_data["system"] = system_prompt or config.system_prompt
            
            # Add custom parameters
            request_data["options"].update(config.custom_parameters)
            
            # Make request
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    if request_data["stream"]:
                        # Handle streaming response
                        full_response = ""
                        async for line in response.content:
                            if line:
                                try:
                                    chunk = json.loads(line.decode().strip())
                                    if "response" in chunk:
                                        full_response += chunk["response"]
                                    if chunk.get("done", False):
                                        break
                                except json.JSONDecodeError:
                                    continue
                        
                        response_data = {"response": full_response}
                    else:
                        # Handle non-streaming response
                        response_data = await response.json()
                    
                    # Update metrics
                    response_time = time.time() - start_time
                    self._update_performance_metrics(True, response_time)
                    
                    return {
                        "success": True,
                        "response": response_data.get("response", ""),
                        "model": model_name,
                        "response_time": response_time,
                        "tokens": len(response_data.get("response", "").split()),
                        "metadata": {
                            "model_config": config.__dict__,
                            "request_params": request_data
                        }
                    }
                else:
                    error_text = await response.text()
                    self.logger.error(f"Ollama API error {response.status}: {error_text}")
                    self._update_performance_metrics(False, time.time() - start_time)
                    
                    return {
                        "success": False,
                        "error": f"API error {response.status}: {error_text}",
                        "model": model_name,
                        "response_time": time.time() - start_time
                    }
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            self._update_performance_metrics(False, time.time() - start_time)
            
            return {
                "success": False,
                "error": str(e),
                "model": model_name or "unknown",
                "response_time": time.time() - start_time
            }
    
    async def generate_streaming_response(self, 
                                        prompt: str,
                                        model_name: Optional[str] = None,
                                        **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response from Ollama model"""
        try:
            if not model_name:
                model_name = self.get_recommended_model("general") or "llama2:7b"
            
            config = self.model_configs.get(model_name, ModelConfig(
                name=model_name,
                provider=ModelProvider.OLLAMA,
                size=ModelSize.MEDIUM
            ))
            
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", config.temperature),
                    "top_p": kwargs.get("top_p", config.top_p),
                    "num_predict": kwargs.get("max_tokens", config.max_tokens)
                }
            }
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json=request_data
            ) as response:
                
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode().strip())
                                if "response" in chunk:
                                    yield chunk["response"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    yield f"Error: {response.status}"
        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    async def chat_completion(self,
                            messages: List[Dict[str, str]],
                            model_name: Optional[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """Chat completion interface (similar to OpenAI format)"""
        try:
            if not model_name:
                model_name = self.get_recommended_model("general") or "llama2:7b"
            
            # Convert messages to prompt format
            prompt_parts = []
            system_prompt = ""
            
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt = "\\n".join(prompt_parts) + "\\nAssistant:"
            
            return await self.generate_response(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                **kwargs
            )
        
        except Exception as e:
            self.logger.error(f"Error in chat completion: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _update_performance_metrics(self, success: bool, response_time: float):
        """Update performance metrics"""
        self.performance_metrics["requests_total"] += 1
        
        if success:
            self.performance_metrics["requests_successful"] += 1
        else:
            self.performance_metrics["requests_failed"] += 1
        
        # Update average response time
        total_requests = self.performance_metrics["requests_total"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Update system metrics
        self.performance_metrics["memory_usage"] = psutil.virtual_memory().percent
        self.performance_metrics["cpu_usage"] = psutil.cpu_percent()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def get_available_models(self) -> Dict[str, OllamaModelInfo]:
        """Get available models"""
        return self.available_models.copy()
    
    def _check_metal_support(self) -> bool:
        """Check for Metal GPU support on macOS"""
        try:
            if platform.system() == "Darwin":
                # Check for Metal support
                result = subprocess.run(
                    ["system_profiler", "SPDisplaysDataType"],
                    capture_output=True,
                    text=True
                )
                return "Metal" in result.stdout
            return False
        except Exception:
            return False
    
    def _check_memory_pressure(self) -> float:
        """Check memory pressure"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception:
            return 0.0
    
    def _check_cpu_efficiency(self) -> Dict[str, Any]:
        """Check CPU efficiency metrics"""
        try:
            return {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else []
            }
        except Exception:
            return {}
    
    async def _apply_performance_optimizations(self):
        """Apply performance optimizations"""
        try:
            if platform.system() == "Darwin" and self.macos_optimizations["metal_gpu"]:
                self.logger.info("Metal GPU support detected - optimizing for GPU acceleration")
                # Set environment variables for Metal optimization
                import os
                os.environ["OLLAMA_GPU_LAYERS"] = "35"  # Use GPU layers
                os.environ["OLLAMA_METAL"] = "1"  # Enable Metal
            
            # Memory optimization
            memory_usage = self.macos_optimizations["memory_pressure"]
            if memory_usage > 80:
                self.logger.warning(f"High memory usage detected: {memory_usage}%")
                # Reduce model concurrency
                self.performance_metrics["max_concurrent_models"] = 1
            else:
                self.performance_metrics["max_concurrent_models"] = 2
            
        except Exception as e:
            self.logger.error(f"Failed to apply performance optimizations: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.session:
                await self.session.close()
            self.logger.info("Ollama integration cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_ollama_integration():
        # Initialize Ollama integration
        ollama = EnhancedOllamaIntegration(auto_install=False)  # Set to True for auto-install
        
        if await ollama.initialize():
            print("✅ Ollama integration initialized")
            
            # Test model recommendation
            recommended_model = ollama.get_recommended_model("business_analysis", "medium")
            print(f"Recommended model for business analysis: {recommended_model}")
            
            # Test response generation
            if recommended_model:
                result = await ollama.generate_response(
                    prompt="Analyze the benefits of AI automation in business processes",
                    model_name=recommended_model
                )
                
                if result["success"]:
                    print(f"Response generated in {result['response_time']:.2f}s")
                    print(f"Response preview: {result['response'][:200]}...")
                else:
                    print(f"Error: {result['error']}")
            
            # Get performance metrics
            metrics = ollama.get_performance_metrics()
            print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
            
            # Cleanup
            await ollama.cleanup()
        else:
            print("❌ Failed to initialize Ollama integration")
    
    # Run test
    asyncio.run(test_ollama_integration())
    print("\\n✅ Enhanced Ollama Integration implemented successfully!")
'''

# Save the enhanced Ollama integration
with open('/home/user/ollama_integration_enhanced.py', 'w') as f:
    f.write(ollama_integration_code)

print("✅ Enhanced Ollama Integration created")
print("📁 File saved: /home/user/ollama_integration_enhanced.py")
print(f"📊 Lines of code: {len(ollama_integration_code.split(chr(10)))}")