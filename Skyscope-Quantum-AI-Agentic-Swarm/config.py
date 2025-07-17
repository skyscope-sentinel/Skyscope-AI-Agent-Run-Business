import os
import json
import yaml
import logging
import base64
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# For encryption
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("skyscope_config")

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """
    Configuration management for Skyscope Sentinel Intelligence.
    Handles loading, saving, and encrypting configuration data.
    """
    
    # Default configurations
    DEFAULT_CONFIG = {
        "app": {
            "name": "Skyscope Sentinel Intelligence - AI Agentic Swarm",
            "version": "1.0.0",
            "data_dir": "data",
            "logs_dir": "logs",
            "temp_dir": "temp",
            "downloads_dir": str(Path.home() / "Downloads"),
            "debug_mode": False,
        },
        "ui": {
            "theme": "dark",
            "accent_color": "#4b5eff",
            "font_size": "medium",
            "layout": "wide",
            "sidebar_state": "expanded",
            "rounded_corners": True,
            "animations": True,
            "code_theme": "monokai",
        },
        "models": {
            "default_provider": "Local (Ollama)",
            "default_model": "llama3:latest",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "providers": {
                "Local (Ollama)": {
                    "base_url": "http://localhost:11434/api",
                    "models": [
                        "llama3:latest", 
                        "mistral:latest", 
                        "gemma:latest", 
                        "codellama:latest",
                        "phi3:latest",
                        "qwen:latest"
                    ]
                },
                "API": {
                    "models": [
                        "gpt-4o",
                        "claude-3-opus",
                        "claude-3-sonnet",
                        "gemini-pro",
                        "gemini-1.5-pro",
                        "anthropic.claude-3-haiku-20240307",
                        "meta.llama3-70b-instruct",
                        "meta.llama3-8b-instruct"
                    ]
                }
            }
        },
        "tools": {
            "web_search": {
                "enabled": False,
                "default_engine": "duckduckgo",
                "max_results": 5
            },
            "deep_research": {
                "enabled": False,
                "depth": "medium",
                "max_sources": 10
            },
            "deep_thinking": {
                "enabled": False,
                "reflection_rounds": 3
            },
            "browser_automation": {
                "enabled": False,
                "headless": False,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                "viewport": {"width": 1280, "height": 800}
            },
            "quantum_computing": {
                "enabled": False,
                "mode": "simulation",
                "default_qubits": 3,
                "default_shots": 1000
            },
            "filesystem_access": {
                "enabled": False,
                "allowed_directories": ["downloads", "documents"],
                "allowed_extensions": [".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".csv"]
            }
        },
        "prompts": {
            "system": """You are Skyscope Sentinel Intelligence, an advanced AI assistant with agentic capabilities. 
You help users with a wide range of tasks including research, coding, analysis, and creative work.
You have access to various tools including web search, browser automation, and file operations.
You can leverage quantum computing concepts for complex problem-solving when appropriate.
Always be helpful, accurate, and ethical in your responses.""",
            "quantum_system": """You are Skyscope Sentinel Intelligence with advanced quantum computing capabilities.
You can analyze problems through the lens of quantum algorithms and provide insights based on quantum principles.
When appropriate, use concepts like superposition, entanglement, and quantum interference to solve complex problems.
Translate classical problems into quantum frameworks when beneficial.""",
            "deep_thinking": """Take a deep breath and work through this problem step by step.
First, understand what is being asked.
Second, break down the problem into manageable components.
Third, analyze each component carefully.
Fourth, synthesize your findings into a comprehensive solution.
Finally, review your answer for accuracy and completeness."""
        },
        "security": {
            "encryption_enabled": True,
            "salt": None,  # Will be generated on first run
            "key_derivation_iterations": 100000,
        },
        "knowledge_stack": {
            "max_items": 100,
            "vector_db_path": "data/vector_db",
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "swarms": {
            "enabled": True,
            "max_agents": 10,
            "default_workflow": "sequential",
            "memory_enabled": True,
            "memory_path": "data/memory"
        }
    }
    
    def __init__(self, config_path: str = "config.json", encryption_password: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
            encryption_password: Password for encrypting sensitive data (if None, will generate a random one)
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.encryption_password = encryption_password or os.environ.get("SKYSCOPE_ENCRYPTION_PASSWORD") or str(uuid.uuid4())
        
        # Initialize encryption
        self._setup_encryption()
        
        # Create necessary directories
        self._create_directories()
        
        # Load configuration from file if it exists
        self.load_config()
        
        # Load API keys from environment variables
        self._load_api_keys_from_env()
        
        logger.info("Configuration initialized")
    
    def _setup_encryption(self):
        """Set up encryption for sensitive data"""
        # Generate or load salt
        if self.config["security"]["salt"] is None:
            self.config["security"]["salt"] = os.urandom(16)
        else:
            # Convert from base64 if stored as string
            if isinstance(self.config["security"]["salt"], str):
                self.config["security"]["salt"] = base64.b64decode(self.config["security"]["salt"])
        
        # Generate encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.config["security"]["salt"],
            iterations=self.config["security"]["key_derivation_iterations"],
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_password.encode()))
        self.cipher = Fernet(key)
        
        logger.debug("Encryption setup complete")
    
    def _create_directories(self):
        """Create necessary directories for the application"""
        for dir_key in ["data_dir", "logs_dir", "temp_dir"]:
            dir_path = Path(self.config["app"][dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
    
    def _load_api_keys_from_env(self):
        """Load API keys from environment variables"""
        # Create api_keys section if it doesn't exist
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        # Map of environment variable names to config keys
        env_to_config = {
            "OPENAI_API_KEY": "openai",
            "ANTHROPIC_API_KEY": "anthropic",
            "GOOGLE_API_KEY": "google",
            "HUGGINGFACE_API_KEY": "huggingface",
            "SERPER_API_KEY": "serper",
            "BROWSERLESS_API_KEY": "browserless"
        }
        
        # Load API keys from environment variables
        for env_var, config_key in env_to_config.items():
            api_key = os.environ.get(env_var)
            if api_key:
                self.set_api_key(config_key, api_key)
                logger.debug(f"Loaded API key for {config_key} from environment")
    
    def encrypt(self, data: str) -> str:
        """
        Encrypt sensitive data
        
        Args:
            data: The data to encrypt
            
        Returns:
            Encrypted data as a base64 string
        """
        if not self.config["security"]["encryption_enabled"]:
            return data
        
        encrypted = self.cipher.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt sensitive data
        
        Args:
            encrypted_data: The encrypted data as a base64 string
            
        Returns:
            Decrypted data
        """
        if not self.config["security"]["encryption_enabled"]:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            return ""
    
    def get_api_key(self, provider: str) -> str:
        """
        Get an API key for the specified provider
        
        Args:
            provider: The provider name (e.g., "openai", "anthropic")
            
        Returns:
            The API key or an empty string if not found
        """
        if "api_keys" not in self.config:
            return ""
        
        encrypted_key = self.config["api_keys"].get(provider, "")
        if not encrypted_key:
            return ""
        
        return self.decrypt(encrypted_key)
    
    def set_api_key(self, provider: str, api_key: str):
        """
        Set an API key for the specified provider
        
        Args:
            provider: The provider name (e.g., "openai", "anthropic")
            api_key: The API key to set
        """
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        self.config["api_keys"][provider] = self.encrypt(api_key)
        logger.debug(f"Set API key for {provider}")
    
    def load_config(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            True if configuration was loaded successfully, False otherwise
        """
        config_path = Path(self.config_path)
        
        if not config_path.exists():
            logger.info(f"Configuration file {config_path} not found, using defaults")
            return False
        
        try:
            if config_path.suffix == ".json":
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
            elif config_path.suffix in [".yaml", ".yml"]:
                with open(config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            # Update configuration with loaded values, preserving defaults for missing keys
            self._update_nested_dict(self.config, loaded_config)
            
            # Re-setup encryption with potentially new salt
            self._setup_encryption()
            
            logger.info(f"Configuration loaded from {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save the configuration file (defaults to self.config_path)
            
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        config_path = config_path or self.config_path
        config_path = Path(config_path)
        
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert salt to base64 for storage
            if isinstance(self.config["security"]["salt"], bytes):
                config_to_save = self.config.copy()
                config_to_save["security"]["salt"] = base64.b64encode(self.config["security"]["salt"]).decode()
            else:
                config_to_save = self.config
            
            if config_path.suffix == ".json":
                with open(config_path, "w") as f:
                    json.dump(config_to_save, f, indent=2)
            elif config_path.suffix in [".yaml", ".yml"]:
                with open(config_path, "w") as f:
                    yaml.dump(config_to_save, f)
            else:
                logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                return False
            
            logger.info(f"Configuration saved to {config_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """
        Update a nested dictionary with values from another dictionary
        
        Args:
            d: The dictionary to update
            u: The dictionary with update values
            
        Returns:
            The updated dictionary
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using a dot-separated path
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "models.default_model")
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default value if not found
        """
        keys = key_path.split(".")
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> bool:
        """
        Set a configuration value using a dot-separated path
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "models.default_model")
            value: The value to set
            
        Returns:
            True if the value was set successfully, False otherwise
        """
        keys = key_path.split(".")
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        logger.debug(f"Set configuration value: {key_path} = {value}")
        return True
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("Configuration reset to defaults")
    
    def get_all_api_keys(self) -> Dict[str, str]:
        """
        Get all API keys (decrypted)
        
        Returns:
            Dictionary of provider names to API keys
        """
        if "api_keys" not in self.config:
            return {}
        
        return {provider: self.decrypt(encrypted_key) 
                for provider, encrypted_key in self.config["api_keys"].items()}
    
    def get_enabled_tools(self) -> Dict[str, bool]:
        """
        Get the enabled status of all tools
        
        Returns:
            Dictionary of tool names to enabled status
        """
        return {tool: config.get("enabled", False) 
                for tool, config in self.config["tools"].items()}
    
    def set_tool_enabled(self, tool_name: str, enabled: bool) -> bool:
        """
        Enable or disable a tool
        
        Args:
            tool_name: The name of the tool
            enabled: Whether the tool should be enabled
            
        Returns:
            True if the tool was found and updated, False otherwise
        """
        if tool_name in self.config["tools"]:
            self.config["tools"][tool_name]["enabled"] = enabled
            logger.debug(f"Tool {tool_name} {'enabled' if enabled else 'disabled'}")
            return True
        return False
    
    def get_model_config(self, provider: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        
        Args:
            provider: The provider name (defaults to default_provider)
            model: The model name (defaults to default_model)
            
        Returns:
            Dictionary of model configuration
        """
        provider = provider or self.config["models"]["default_provider"]
        model = model or self.config["models"]["default_model"]
        
        model_config = {
            "provider": provider,
            "model": model,
            "temperature": self.config["models"]["temperature"],
            "max_tokens": self.config["models"]["max_tokens"],
            "top_p": self.config["models"]["top_p"],
            "frequency_penalty": self.config["models"]["frequency_penalty"],
            "presence_penalty": self.config["models"]["presence_penalty"],
        }
        
        # Add provider-specific configuration
        if provider in self.config["models"]["providers"]:
            for key, value in self.config["models"]["providers"][provider].items():
                if key != "models":
                    model_config[key] = value
        
        return model_config
    
    def get_available_models(self, provider: Optional[str] = None) -> List[str]:
        """
        Get available models for a provider
        
        Args:
            provider: The provider name (defaults to default_provider)
            
        Returns:
            List of available model names
        """
        provider = provider or self.config["models"]["default_provider"]
        
        if provider in self.config["models"]["providers"]:
            return self.config["models"]["providers"][provider].get("models", [])
        
        return []
    
    def get_ui_config(self) -> Dict[str, Any]:
        """
        Get UI configuration
        
        Returns:
            Dictionary of UI configuration
        """
        return self.config["ui"].copy()
    
    def get_system_prompt(self, prompt_type: str = "system") -> str:
        """
        Get a system prompt
        
        Args:
            prompt_type: The type of prompt to get (e.g., "system", "quantum_system")
            
        Returns:
            The system prompt
        """
        return self.config["prompts"].get(prompt_type, self.config["prompts"]["system"])
    
    def set_system_prompt(self, prompt: str, prompt_type: str = "system"):
        """
        Set a system prompt
        
        Args:
            prompt: The prompt text
            prompt_type: The type of prompt to set (e.g., "system", "quantum_system")
        """
        self.config["prompts"][prompt_type] = prompt
        logger.debug(f"Set {prompt_type} prompt")
    
    def get_app_paths(self) -> Dict[str, str]:
        """
        Get application paths
        
        Returns:
            Dictionary of path names to paths
        """
        return {
            key: value for key, value in self.config["app"].items()
            if key.endswith("_dir")
        }
    
    def get_knowledge_stack_config(self) -> Dict[str, Any]:
        """
        Get knowledge stack configuration
        
        Returns:
            Dictionary of knowledge stack configuration
        """
        return self.config["knowledge_stack"].copy()
    
    def get_swarms_config(self) -> Dict[str, Any]:
        """
        Get swarms configuration
        
        Returns:
            Dictionary of swarms configuration
        """
        return self.config["swarms"].copy()
    
    def export_config(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Export configuration as a dictionary
        
        Args:
            include_sensitive: Whether to include sensitive data like API keys
            
        Returns:
            Dictionary of configuration
        """
        config = self.config.copy()
        
        if not include_sensitive and "api_keys" in config:
            del config["api_keys"]
        
        return config
    
    def import_config(self, config: Dict[str, Any], overwrite: bool = False) -> bool:
        """
        Import configuration from a dictionary
        
        Args:
            config: The configuration dictionary
            overwrite: Whether to completely overwrite the current configuration
            
        Returns:
            True if configuration was imported successfully, False otherwise
        """
        try:
            if overwrite:
                self.config = config
            else:
                self._update_nested_dict(self.config, config)
            
            # Re-setup encryption with potentially new salt
            self._setup_encryption()
            
            logger.info("Configuration imported")
            return True
        
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False


# Create a global configuration instance
config = Config()

# Helper functions for easy access
def get_config() -> Config:
    """Get the global configuration instance"""
    return config

def get(key_path: str, default: Any = None) -> Any:
    """Get a configuration value using a dot-separated path"""
    return config.get(key_path, default)

def set(key_path: str, value: Any) -> bool:
    """Set a configuration value using a dot-separated path"""
    return config.set(key_path, value)

def save() -> bool:
    """Save the current configuration"""
    return config.save_config()

def load() -> bool:
    """Load configuration from file"""
    return config.load_config()

def get_api_key(provider: str) -> str:
    """Get an API key for the specified provider"""
    return config.get_api_key(provider)

def set_api_key(provider: str, api_key: str):
    """Set an API key for the specified provider"""
    config.set_api_key(provider, api_key)
