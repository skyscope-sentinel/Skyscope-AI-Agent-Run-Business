import logging
import json
import google.generativeai as genai
# from google.generativeai import types # Not needed if using GenerativeModel directly for content
from google.api_core import exceptions as google_exceptions

class GeminiAnalyzer:
    """
    Analyzer using Google Gemini models (now using GenerativeModel due to environment issues with Client)
    to make trading decisions based on market data.
    """

    def __init__(self, api_key: str, model_name: str = 'models/gemma-2-9b-it', config_logging: bool = True):
        """
        Initializes the GeminiAnalyzer using genai.GenerativeModel.

        Args:
            api_key (str): The Google AI Studio API key for Gemini.
            model_name (str, optional): The name of the Gemini model to use (e.g., 'models/gemma-2-9b-it').
                                        Must be prefixed with "models/" for newer models.
            config_logging (bool, optional): Whether to configure basic logging if no handlers are set.
        """
        self.logger = logging.getLogger(__name__)
        if config_logging and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
            self.logger.info("Basic logging configured for GeminiAnalyzer standalone usage.")

        self.model_name = model_name
        self.model = None # Will hold the GenerativeModel instance

        if not api_key or "YOUR_" in api_key:
            self.logger.error("Gemini API key is missing or is a placeholder. GeminiAnalyzer will not function.")
            return

        try:
            # Configure API key (important for GenerativeModel)
            genai.configure(api_key=api_key)

            self.logger.info(f"Attempting to initialize GenerativeModel. genai module version: {genai.__version__ if hasattr(genai, '__version__') else 'unknown'}")
            self.logger.info(f"genai module path: {genai.__path__ if hasattr(genai, '__path__') else genai.__file__ if hasattr(genai, '__file__') else 'unknown'}")

            self.model = genai.GenerativeModel(self.model_name)
            self.logger.info(f"GeminiAnalyzer initialized with GenerativeModel for model: {self.model_name}. API Key ending with ...{api_key[-4:]}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini GenerativeModel: {e}", exc_info=True)
            self.model = None

    def analyze_market_data(self, market_data_prompt_str: str):
        """
        Sends a structured prompt to the specified Gemini model using genai.GenerativeModel
        and attempts to parse the JSON response.

        Args:
            market_data_prompt_str (str): The fully formed prompt string for the Gemini model,
                                          which should request a JSON formatted response.
        Returns:
            dict: A Python dictionary parsed from the Gemini model's JSON response.
                  Returns an error dictionary if API call fails, response is empty, or JSON parsing fails.
        """
        if self.model is None:
            self.logger.error("Gemini GenerativeModel not initialized. Cannot analyze market data.")
            return {"error": "ModelNotInitialized", "message": "Gemini GenerativeModel is None."}

        prompt_snippet = market_data_prompt_str[:300] + "..." if len(market_data_prompt_str) > 300 else market_data_prompt_str
        self.logger.info(f"Sending prompt to Gemini model '{self.model_name}' (snippet): {prompt_snippet}")

        # For GenerativeModel, content is passed directly.
        # Optional: Add generation_config if needed
        # generation_config = genai.types.GenerationConfig(temperature=0.7, max_output_tokens=2048)

        raw_response_text = ""
        try:
            # response = self.model.generate_content(market_data_prompt_str, generation_config=generation_config)
            response = self.model.generate_content(market_data_prompt_str)

            if hasattr(response, 'text') and response.text:
                raw_response_text = response.text
            # For GenerativeModel, response.parts might be more relevant if text is not directly populated.
            elif response.parts:
                 raw_response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                self.logger.warning(f"Gemini API returned a response, but no text content found in expected fields. Response: {response}")
                return {"error": "EmptyResponse", "message": "No text content in Gemini response.", "raw_response_obj": str(response)}

            self.logger.debug(f"Raw Gemini response text (snippet): {raw_response_text[:300] if raw_response_text else 'EMPTY_RESPONSE_TEXT'}")

            cleaned_response_text = raw_response_text.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:-3].strip()
            elif cleaned_response_text.startswith("```"):
                 cleaned_response_text = cleaned_response_text[3:-3].strip()

            if not cleaned_response_text:
                self.logger.warning("Gemini response was empty after stripping potential markdown.")
                return {"error": "EmptyCleanedResponse", "message": "Response empty after markdown stripping.", "raw_response_text": raw_response_text}

            parsed_json = json.loads(cleaned_response_text)
            self.logger.info("Successfully parsed JSON response from Gemini.")
            self.logger.debug(f"Parsed Gemini JSON: {json.dumps(parsed_json, indent=2)}")
            return parsed_json

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from Gemini response: {e}", exc_info=True)
            self.logger.error(f"Problematic raw response text was: '{raw_response_text}'")
            return {"error": "JSONDecodeError", "message": str(e), "raw_response": raw_response_text}
        except google_exceptions.GoogleAPIError as e:
            self.logger.error(f"Gemini API call failed (GoogleAPIError for model {self.model_name}): {e}", exc_info=True)
            return {"error": "GoogleAPIError", "message": str(e), "model_name": self.model_name}
        except AttributeError as e:
            self.logger.error(f"Attribute error processing Gemini response (unexpected structure from model {self.model_name}): {e}", exc_info=True)
            self.logger.error(f"Full response object was: {response if 'response' in locals() else 'Response object not available'}")
            return {"error": "AttributeErrorResponse", "message": str(e), "model_name": self.model_name}
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during Gemini API call (model {self.model_name}): {e}", exc_info=True)
            return {"error": "UnexpectedError", "message": str(e), "model_name": self.model_name}

def _create_sample_prompt(symbol, current_price, kline_data_snippet, order_book_snippet,
                          account_balance, open_positions_snippet, trading_strategy_rules):
    """
    Helper function to create a sample prompt string for Gemini, requesting JSON output.
    This is just an example; the actual prompt will be more complex and dynamically generated.
    """
    prompt = f"""
    Analyze the following market data for {symbol} and provide a trading decision.
    Current Price: {current_price}

    Recent K-line Data (OHLCV):
    {kline_data_snippet}

    Order Book Snippet (Top Bids/Asks):
    {order_book_snippet}

    Account Balance:
    {account_balance}

    Open Positions:
    {open_positions_snippet}

    Trading Strategy Rules:
    {trading_strategy_rules}

    Based on the data and strategy, provide your analysis in JSON format with the following keys:
    - "decision": String, one of ["BUY", "SELL", "HOLD", "NO_ACTION"]
    - "confidence_score": Float, between 0.0 and 1.0
    - "entry_price_target": Float or null (if HOLD or NO_ACTION)
    - "stop_loss_price": Float or null (if HOLD or NO_ACTION)
    - "take_profit_target": Float or null (if HOLD or NO_ACTION)
    - "reasoning": String, a brief explanation of the decision.

    Example JSON output:
    {{
      "decision": "BUY",
      "confidence_score": 0.75,
      "entry_price_target": 30050.0,
      "stop_loss_price": 29800.0,
      "take_profit_target": 30500.0,
      "reasoning": "Price broke above key resistance level with increasing volume, RSI shows upward momentum."
    }}

    Return ONLY the JSON object. Do not include any other text or explanations outside the JSON structure.
    Ensure the JSON is well-formed.
    """
    return prompt.strip()


if __name__ == '__main__':
    print("--- Running GeminiAnalyzer Standalone Example ---")

    import os
    import sys

    # Adjust sys.path to allow importing from the parent directory (project root)
    # This makes `from bybit_gemini_bot.config import load_config` work when script is run directly.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    try:
        from bybit_gemini_bot.config import load_config

        # For this specific test, we will directly attempt to load 'configs/config.dev.json'
        # as it's expected to be created with the actual API key for this subtask.

        dev_config_path = os.path.join(PROJECT_ROOT, "configs", "config.dev.json")

        if not os.path.exists(dev_config_path):
            print(f"CRITICAL ERROR: The required 'configs/config.dev.json' was not found at '{dev_config_path}'.")
            print("This test specifically relies on this file being present with the API key.")
            print("Please ensure the file creation step for this subtask was successful and the file is visible to the execution environment.")
            # Listing contents of configs directory for debugging:
            configs_dir_path = os.path.join(PROJECT_ROOT, "configs")
            if os.path.exists(configs_dir_path):
                print(f"Contents of '{configs_dir_path}': {os.listdir(configs_dir_path)}")
            else:
                print(f"Directory '{configs_dir_path}' does not exist.")
            exit()

        print(f"Attempting to load specific test configuration from: {dev_config_path}")
        app_config = load_config(dev_config_path)

    except ImportError as e:
        print(f"ImportError: Could not import 'load_config'. Ensure PYTHONPATH is set correctly or run from project root. Details: {e}")
        app_config = {} # Fallback
    except Exception as e:
        print(f"Error loading configuration: {e}")
        app_config = {}

    if not app_config: # Check if config loading actually failed or returned empty
        print("Critical: Configuration could not be loaded or is empty. Exiting standalone test.")
        exit()

    gemini_api_key = app_config.get("gemini_api_key")

    if not gemini_api_key or "YOUR_GEMINI_API_KEY" in gemini_api_key:
        print("Error: GEMINI_API_KEY not found in configuration or is a placeholder.")
        print("Please set it in your config file (e.g., configs/config.dev.json).")
        exit()

    # Instantiate the analyzer
    # Using a specific model for testing, you can change this.
    # Using a specific Gemma model as requested. Default is 'models/gemma-2-9b-it' in constructor.
    # For testing, we can override it here if needed, e.g. to test 'models/gemma-3-12b-it' or other Gemma variants.
    # The subtask specifies 'gemma-3-12b-it' as the target, but constructor defaults to gemma-2-9b-it for wider availability.
    # Let's use the subtask's primary target model here for the test.
    target_model_for_test = 'models/gemma-3-12b-it'
    # Fallback if the primary target fails, to ensure client logic is tested.
    fallback_model_for_test = 'models/gemma-2-9b-it'

    analyzer = GeminiAnalyzer(api_key=gemini_api_key, model_name=target_model_for_test, config_logging=True)

    if analyzer.model is None: # Check if GenerativeModel initialized
        print(f"Failed to initialize Gemini Analyzer with GenerativeModel for model {target_model_for_test}. Exiting.")
        exit()

    # Create a sample prompt using dummy data
    sample_kline = "[['1678886400000', '25000', '25500', '24800', '25200', '1000', '25200000'], ...]"
    sample_order_book = "{'bids': [['25190', '10'], ['25180', '5']], 'asks': [['25210', '8'], ['25220', '6']]}"
    sample_balance = "{'USDT': '10000', 'BTC': '0.5'}"
    sample_positions = "[]" # No open positions
    sample_strategy = "Trend following: Buy on breakout above 20-period SMA, Sell on break below. RSI confirmation."

    prompt = _create_sample_prompt(
        symbol="BTCUSDT",
        current_price="25200",
        kline_data_snippet=sample_kline,
        order_book_snippet=sample_order_book,
        account_balance=sample_balance,
        open_positions_snippet=sample_positions,
        trading_strategy_rules=sample_strategy
    )

    print(f"\n--- Sample Prompt (first 300 chars) for model {analyzer.model_name} ---\n{prompt[:300]}...")

    analysis_result = analyzer.analyze_market_data(prompt)

    print(f"\n--- Analysis Result from {analyzer.model_name} ---")
    if analysis_result:
        print(json.dumps(analysis_result, indent=2))
        if "error" in analysis_result:
            print(f"\nNote: An error occurred during analysis with {analyzer.model_name}: {analysis_result.get('message')}")

            # If primary model failed, try fallback (only if it's different from primary)
            if analyzer.model_name == target_model_for_test and target_model_for_test != fallback_model_for_test:
                print(f"\n--- Attempting fallback model: {fallback_model_for_test} ---")
                analyzer.model_name = fallback_model_for_test # Switch model
                print(f"Re-sending prompt to {analyzer.model_name} (snippet): {prompt[:300]}...")
                analysis_result_fallback = analyzer.analyze_market_data(prompt)

                print(f"\n--- Analysis Result from {analyzer.model_name} (fallback) ---")
                if analysis_result_fallback:
                    print(json.dumps(analysis_result_fallback, indent=2))
                    if "error" in analysis_result_fallback:
                        print(f"\nNote: An error also occurred with fallback model {analyzer.model_name}: {analysis_result_fallback.get('message')}")
                else:
                    print(f"No result from fallback model {analyzer.model_name} or an error occurred.")
    else:
        # This case should ideally be handled by the error dict from analyze_market_data
        print(f"No result from {analyzer.model_name} or a critical error occurred that prevented result dictionary generation.")

    print("\n--- GeminiAnalyzer Standalone Example Finished ---")
