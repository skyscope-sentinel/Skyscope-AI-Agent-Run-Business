import json
import os

DEFAULT_CONFIG_PATH = "configs/config.dev.json"
EXAMPLE_CONFIG_PATH = "configs/config.json.example"

def load_config(config_path=None):
    """
    Loads configuration from a JSON file.

    Args:
        config_path (str, optional): Path to the config file.
                                     Defaults to None, which means it will try
                                     to load from `configs/config.dev.json` or
                                     use the example if it doesn't exist.

    Returns:
        dict: A dictionary containing the configuration settings.
              Returns an empty dict and prints an error if loading fails.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f"Warning: Configuration file '{config_path}' not found.")
        if os.path.exists(EXAMPLE_CONFIG_PATH):
            print(f"Please create your own configuration file by copying "
                  f"and renaming '{EXAMPLE_CONFIG_PATH}' to '{config_path}' "
                  f"and filling in your API keys and settings.")
            try:
                with open(EXAMPLE_CONFIG_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading example configuration '{EXAMPLE_CONFIG_PATH}': {e}")
                return {}
        else:
            print(f"Error: Example configuration '{EXAMPLE_CONFIG_PATH}' also not found. "
                  f"Please create '{config_path}' manually.")
            return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

            # --- API Key Loading with Environment Variable Fallback ---
            api_keys_to_check = {
                "bybit_api_key": "BYBIT_API_KEY", # JSON key: ENV_VAR_NAME
                "bybit_api_secret": "BYBIT_API_SECRET",
                "testnet_api_key_bybit": "TESTNET_API_KEY_BYBIT",
                "testnet_api_secret_bybit": "TESTNET_API_SECRET_BYBIT",
                # Assuming mainnet keys might also be used directly if specified in future
                "mainnet_api_key_bybit": "MAINNET_API_KEY_BYBIT",
                "mainnet_api_secret_bybit": "MAINNET_API_SECRET_BYBIT",
                "gemini_api_key": "GEMINI_API_KEY",
                # "gemini_api_secret": "GEMINI_API_SECRET", # If needed
            }

            placeholders = ["YOUR_", "PLACEHOLDER", "", None] # Common placeholder patterns

            for json_key, env_var_name in api_keys_to_check.items():
                loaded_from_env = False
                # Check if key is missing from JSON or is a placeholder
                if config.get(json_key) in placeholders:
                    env_value = os.getenv(env_var_name)
                    if env_value:
                        config[json_key] = env_value
                        loaded_from_env = True
                        print(f"Info: Loaded '{json_key}' from environment variable '{env_var_name}'.")
                    elif config.get(json_key) in placeholders: # Still a placeholder after trying env
                        # Avoid printing this for non-essential keys like testnet if mainnet is set,
                        # but for core keys, it's a useful warning.
                        if json_key in ["bybit_api_key", "gemini_api_key"]: # Example core keys
                             print(f"Warning: API key '{json_key}' is a placeholder in '{config_path}' and not found in environment variable '{env_var_name}'.")
                # else:
                    # print(f"Info: Loaded '{json_key}' from config file '{config_path}'.")

            # Basic validation for a few core required keys after attempting env load
            # This can be expanded based on which keys are truly essential for bot operation mode (testnet/mainnet)
            essential_keys_missing = []
            if config.get("use_bybit_testnet_connector", False): # Check if running in testnet mode
                if config.get("testnet_api_key_bybit") in placeholders:
                    essential_keys_missing.append("testnet_api_key_bybit")
                if config.get("testnet_api_secret_bybit") in placeholders:
                    essential_keys_missing.append("testnet_api_secret_bybit")
            else: # Assuming mainnet if not testnet
                if config.get("bybit_api_key") in placeholders: # or mainnet_api_key_bybit
                    essential_keys_missing.append("bybit_api_key (mainnet)")
                if config.get("bybit_api_secret") in placeholders:
                    essential_keys_missing.append("bybit_api_secret (mainnet)")

            if config.get("use_gemini_analyzer", False):
                 if config.get("gemini_api_key") in placeholders:
                    essential_keys_missing.append("gemini_api_key")

            if essential_keys_missing:
                print(f"Warning: Essential API keys are still missing or placeholders after checking config file and environment variables: {', '.join(essential_keys_missing)}")
                print("Please ensure these API keys are set correctly in your config file or environment.")

            return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{config_path}': {e}")
        return {}
    except Exception as e:
        print(f"Error loading configuration from '{config_path}': {e}")
        return {}

if __name__ == '__main__':
    # Example usage:
    # Create a dummy config.dev.json for testing if it doesn't exist
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        if not os.path.exists("configs"):
            os.makedirs("configs")
        dummy_config_data = {
            "bybit_api_key": "YOUR_BYBIT_API_KEY_DEV",
            "bybit_api_secret": "YOUR_BYBIT_API_SECRET_DEV",
            "gemini_api_key": "YOUR_GEMINI_API_KEY_DEV",
            "gemini_api_secret": "YOUR_GEMINI_API_SECRET_DEV",
            "testnet_api_key_bybit": "YOUR_BYBIT_TESTNET_API_KEY_DEV",
            "testnet_api_secret_bybit": "YOUR_BYBIT_TESTNET_API_SECRET_DEV",
            "mainnet_api_key_bybit": "YOUR_BYBIT_MAINNET_API_KEY_DEV",
            "mainnet_api_secret_bybit": "YOUR_BYBIT_MAINNET_API_SECRET_DEV",
            "trading_symbols": ["BTCUSD", "ETHUSD"],
            "risk_percentage": 0.01,
            "some_other_setting": "value"
        }
        with open(DEFAULT_CONFIG_PATH, 'w') as f:
            json.dump(dummy_config_data, f, indent=2)
        print(f"Created a dummy '{DEFAULT_CONFIG_PATH}' for example usage.")

    config = load_config()
    if config:
        print("\nConfiguration loaded successfully:")
        for key, value in config.items():
            if "key" in key.lower() or "secret" in key.lower():
                print(f"  {key}: {'*' * len(str(value))}")
            else:
                print(f"  {key}: {value}")
    else:
        print("\nFailed to load configuration or configuration is empty.")

    print(f"\nAttempting to load config that doesn't exist (expected warning):")
    config_non_existent = load_config("configs/non_existent_config.json")
    if not config_non_existent:
        print("Correctly handled non-existent config.")

    # To test the case where example is also missing, you would manually delete configs/config.json.example
    # and then run:
    # config_no_example = load_config("configs/config.dev.json") # Assuming config.dev.json also doesn't exist
    # print(config_no_example)
