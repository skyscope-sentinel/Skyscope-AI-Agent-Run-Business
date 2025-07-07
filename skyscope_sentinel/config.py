import os

class Config:
    """
    Simple configuration class for Skyscope Sentinel.
    Initially loads settings from environment variables or provides defaults.
    Can be extended to integrate with the GUI's SettingsManager.
    """
    def __init__(self):
        # Ollama Settings
        # In a real app, these would eventually be loaded from SettingsManager
        # which reads from the GUI-configured settings.
        # For now, using environment variables as a common practice, with fallbacks.
        self.ollama_model_name_env = os.getenv("OLLAMA_MODEL")
        self.ollama_base_url_env = os.getenv("OLLAMA_BASE_URL")
        self.serper_api_key_env = os.getenv("SERPER_API_KEY")
        self.openai_api_key_env = os.getenv("OPENAI_API_KEY")
        self.e2b_api_key_env = os.getenv("E2B_API_KEY") # New E2B API Key

        # Default values if environment variables are not set
        self.default_ollama_model = "ollama/qwen2:0.5b" # A reasonable default
        self.default_ollama_url = "http://localhost:11434"

        # These will hold the final values after considering env and potentially GUI settings
        self.current_ollama_model = self.ollama_model_name_env or self.default_ollama_model
        self.current_ollama_url = self.ollama_base_url_env or self.default_ollama_url
        self.current_serper_api_key = self.serper_api_key_env
        self.current_openai_api_key = self.openai_api_key_env
        self.current_e2b_api_key = self.e2b_api_key_env

        # Financials - will be updated from SettingsManager
        self.current_btc_address = None # Default to None
        self.current_e2b_api_key = self.e2b_api_key_env # Initialize current value

        print(f"[Config] Initialized. OLLAMA_MODEL env: {self.ollama_model_name_env}, OLLAMA_BASE_URL env: {self.ollama_base_url_env}")
        print(f"[Config] Effective Ollama Model: {self.current_ollama_model}, URL: {self.current_ollama_url}")


    def update_from_settings_manager(self, settings_manager):
        """
        Updates configuration values from a SettingsManager instance (GUI settings).
        GUI settings will take precedence over environment variables or defaults if set.
        """
        gui_ollama_model = settings_manager.load_setting("ollama/model_name", None) # Key from settings_page.py
        gui_ollama_url = settings_manager.load_setting("ollama/service_url", None)
        gui_serper_key = settings_manager.load_setting("api_keys/serper_api_key", None)
        gui_openai_key = settings_manager.load_setting("api_keys/openai_api_key", None)
        gui_e2b_key = settings_manager.load_setting("api_keys/e2b_api_key", None)
        gui_btc_address = settings_manager.load_setting("financials/btc_address", None) # Key from settings_manager.py

        if gui_ollama_model:
            self.current_ollama_model = gui_ollama_model
        elif self.ollama_model_name_env: # Fallback to env if GUI not set
            self.current_ollama_model = self.ollama_model_name_env
        else: # Fallback to default
            self.current_ollama_model = self.default_ollama_model

        if gui_ollama_url:
            self.current_ollama_url = gui_ollama_url
        elif self.ollama_base_url_env:
            self.current_ollama_url = self.ollama_base_url_env
        else:
            self.current_ollama_url = self.default_ollama_url

        if gui_serper_key:
            self.current_serper_api_key = gui_serper_key
        elif self.serper_api_key_env:
             self.current_serper_api_key = self.serper_api_key_env
        # else it remains None if neither GUI nor env var is set

        if gui_openai_key:
            self.current_openai_api_key = gui_openai_key
        elif self.openai_api_key_env:
            self.current_openai_api_key = self.openai_api_key_env

        if gui_e2b_key:
            self.current_e2b_api_key = gui_e2b_key
        elif self.e2b_api_key_env:
            self.current_e2b_api_key = self.e2b_api_key_env

        if gui_btc_address:
            self.current_btc_address = gui_btc_address
        # No environment variable fallback for BTC address, only GUI or None
        gui_e2b_key = settings_manager.load_setting("api_keys/e2b_api_key", None) # Load E2B key

        if gui_ollama_model:
            self.current_ollama_model = gui_ollama_model
        if gui_ollama_url:
            self.current_ollama_url = gui_ollama_url
        if gui_serper_key:
            self.current_serper_api_key = gui_serper_key
        if gui_openai_key:
            self.current_openai_api_key = gui_openai_key
        if gui_e2b_key: # Update current E2B key
            self.current_e2b_api_key = gui_e2b_key

        # Optionally, update environment variables so other modules using os.getenv() directly also benefit
        # This can be useful but also has side effects. For now, just update internal state.
        # if self.current_serper_api_key: os.environ["SERPER_API_KEY"] = self.current_serper_api_key
        # if self.current_openai_api_key: os.environ["OPENAI_API_KEY"] = self.current_openai_api_key


        print(f"[Config] Updated from SettingsManager. Effective Ollama Model: {self.current_ollama_model}, URL: {self.current_ollama_url}")
        print(f"[Config] Effective SERPER_API_KEY: {'Set' if self.current_serper_api_key else 'Not Set'}")
        print(f"[Config] Effective OPENAI_API_KEY: {'Set' if self.current_openai_api_key else 'Not Set'}")
        print(f"[Config] Effective E2B_API_KEY: {'Set' if self.current_e2b_api_key else 'Not Set'}")
        print(f"[Config] Effective BTC Address: {'Set' if self.current_btc_address else 'Not Set'}")
        print(f"[Config] Effective E2B_API_KEY: {'Set' if self.current_e2b_api_key else 'Not Set'}") # Print status of E2B key

    def get_ollama_model_name(self) -> str:
        """Returns the configured Ollama model name, preferring env var, then default."""
        # CrewAI's OllamaLLM expects the model name without "ollama/" prefix.
        # However, ModelFactory in CAMEL might expect it.
        # For CrewAI directly, we'll ensure no prefix.
        # The research_crew.py handles adding "ollama/" if using generic LLM class
        # and removing it if using specific OllamaLLM class.
        # This getter should return the effective value.
        model_name = self.current_ollama_model
        # Ensure consistency for CrewAI's OllamaLLM which expects model name without "ollama/"
        # However, some parts might still add/remove it. For now, return as is from current_
        # and let the calling code (like research_crew.py) handle prefixing/stripping if necessary
        # based on the specific LLM client class being used.
        return model_name


    def get_ollama_base_url(self) -> str:
        """Returns the effective Ollama base URL."""
        return self.current_ollama_url

    def get_serper_api_key(self) -> str | None:
        """Returns the effective Serper API key if set, otherwise None."""
        return self.current_serper_api_key

    def get_openai_api_key(self) -> str | None:
        """Returns the effective OpenAI API key if set, otherwise None."""
        return self.current_openai_api_key

    def get_e2b_api_key(self) -> str | None:
        """Returns the effective E2B API key if set, otherwise None."""
        return self.current_e2b_api_key

    def get_btc_address(self) -> str | None:
        """Returns the effective BTC Address if set, otherwise None."""
        return self.current_btc_address

if __name__ == '__main__':
   from dotenv import load_dotenv
   load_dotenv() # Load .env file if present for testing this module directly

   # For testing Config with SettingsManager, we'd need to mock or instantiate SettingsManager
   # For now, direct instantiation test will show env vars or defaults
   conf = Config()
   print(f"Ollama Model Name: {conf.get_ollama_model_name()}")
   print(f"Ollama Base URL: {conf.get_ollama_base_url()}")
   print(f"Serper API Key: {'Set' if conf.get_serper_api_key() else 'Not Set'}")
   print(f"OpenAI API Key: {'Set' if conf.get_openai_api_key() else 'Not Set'}")
   print(f"E2B API Key: {'Set' if conf.get_e2b_api_key() else 'Not Set'}")
   print(f"BTC Address: {'Set' if conf.get_btc_address() else 'Not Set'}") # Test the new getter

   # Example of how it would work with SettingsManager
   # from skyscope_sentinel.settings_manager import SettingsManager # Assuming this path
   # settings_mgr = SettingsManager()
   # # Simulate saving a BTC address through GUI/SettingsManager
   # # settings_mgr.save_setting("financials/btc_address", "1YourTestBTCAddressHere")
   # conf.update_from_settings_manager(settings_mgr)
   # print(f"BTC Address after SettingsManager update: {'Set' if conf.get_btc_address() else 'Not Set'}")
   # print(f"Actual BTC Address from Config: {conf.get_btc_address()}")
