import logging
import os

# Assuming logger_setup.py and config.py are in the same directory (bybit_gemini_bot)
from .logger_setup import setup_logging, DEFAULT_LOG_FILE
from .config import load_config, DEFAULT_CONFIG_PATH, EXAMPLE_CONFIG_PATH
from .bybit_connector import BybitConnector # Import BybitConnector

def main():
    """
    Main function to initialize and start the trading bot.
    """
    # 1. Setup Logging
    # You can customize log_level and log_file here if needed
    setup_logging(log_level=logging.INFO, log_file=DEFAULT_LOG_FILE)

    logger = logging.getLogger(__name__) # Get a logger for the main module

    logger.info("=============================================")
    logger.info("🚀 Trading Bot Starting Up...")
    logger.info("=============================================")

    # 2. Load Configuration
    # Determine the config file path.
    # Prioritize a specific dev/prod config, then example.
    config_file_to_load = DEFAULT_CONFIG_PATH
    if not os.path.exists(config_file_to_load):
        logger.warning(f"'{config_file_to_load}' not found. Looking for example config.")
        config_file_to_load = EXAMPLE_CONFIG_PATH
        if not os.path.exists(config_file_to_load):
            logger.error(f"Neither '{DEFAULT_CONFIG_PATH}' nor '{EXAMPLE_CONFIG_PATH}' found.")
            logger.error("Please ensure you have a configuration file.")
            logger.info("Bot shutting down due to missing configuration.")
            return # Exit if no config is found

    logger.info(f"Attempting to load configuration from: '{config_file_to_load}'")
    config = load_config(config_file_to_load)

    if not config:
        logger.error("Failed to load configuration or configuration is empty.")
        logger.info("Bot shutting down due to configuration load failure.")
        return

    logger.info("Configuration loaded successfully.")

    # Log loaded configuration (masking sensitive details)
    masked_config_str = "Loaded Configuration:\n"
    for key, value in config.items():
        if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower(): # Added "token" for general API tokens
            # Mask Gemini API Key as well
            if key == "gemini_api_key" and value:
                 masked_value = f"{str(value)[:5]}...{str(value)[-4:]}" if len(str(value)) > 9 else "*" * len(str(value))
            elif value: # Standard masking for other keys/secrets
                masked_value = f"{str(value)[:3]}...{str(value)[-3:]}" if len(str(value)) > 6 else "*" * len(str(value))
            else:
                masked_value = "NOT_SET"
            masked_config_str += f"  {key}: {masked_value}\n"
        elif key == "trading_symbols":
             masked_config_str += f"  {key}: {value}\n" # Show symbols
        elif key == "risk_percentage":
             masked_config_str += f"  {key}: {value}\n" # Show risk
        # Add other specific keys you want to see in logs here
        else:
            # For other keys, just show their presence without value for brevity, or show value if safe
            masked_config_str += f"  {key}: Present\n"

    logger.info(masked_config_str)

    # 3. Initialize Bybit Connector (Example Usage - conditional or commented out for production)
    # Ensure you have testnet_api_key_bybit and testnet_api_secret_bybit in your config
    # Also, ensure bybit_connector is imported: from .bybit_connector import BybitConnector
    use_bybit_testnet_connector = config.get("use_bybit_testnet_connector", False) # Default to False unless explicitly enabled

    if use_testnet_connector:
        logger.info("Attempting to initialize Bybit Testnet Connector...")
        bybit_testnet_api_key = config.get("testnet_api_key_bybit")
        bybit_testnet_api_secret = config.get("testnet_api_secret_bybit")

        if bybit_testnet_api_key and "YOUR_" not in bybit_testnet_api_key and \
           bybit_testnet_api_secret and "YOUR_" not in bybit_testnet_api_secret:

            bybit_conn = BybitConnector(
                api_key=bybit_testnet_api_key,
                api_secret=bybit_testnet_api_secret,
                testnet=True,
                config_logging=False # Main app already configured logging
            )

            if bybit_conn.session: # Check if session was initialized successfully
                logger.info("Bybit Testnet Connector initialized successfully.")

                # --- Test Bybit Connector Methods (Comment out or make conditional for production) ---
                logger.info("--- Testing Bybit Connector (Testnet) ---")

                # Get Server Time
                server_time_resp = bybit_conn.get_server_time()
                if server_time_resp and server_time_resp.get("retCode") == 0:
                    logger.info(f"Bybit Server Time (nano): {server_time_resp.get('result', {}).get('timeNano')}")
                else:
                    logger.warning(f"Failed to get Bybit server time or error in response: {server_time_resp}")

                # Get Wallet Balance (UNIFIED account)
                # Ensure your Testnet API key has "Account Information" or "Wallet" permissions
                wallet_balance_resp = bybit_conn.get_wallet_balance(account_type="UNIFIED")
                if wallet_balance_resp and wallet_balance_resp.get("retCode") == 0:
                    # The actual balance list might be empty if the account has no funds
                    balance_list = wallet_balance_resp.get('result', {}).get('list', [])
                    if balance_list:
                        logger.info(f"Bybit Wallet Balance (UNIFIED Account, Testnet): {balance_list[0]}")
                    else:
                        logger.info("Bybit Wallet Balance (UNIFIED Account, Testnet): No balance data found (list is empty).")
                else:
                    logger.warning(f"Failed to get Bybit wallet balance or error in response (check API key permissions): {wallet_balance_resp}")

                logger.info("--- Finished Bybit Connector Test ---")
            else:
                logger.error("Bybit Testnet Connector session FAILED to initialize. Check API keys and connector logs.")
        else:
            logger.warning("Bybit Testnet API key/secret not found in config or are placeholders. Skipping Bybit Connector test.")
    else:
        logger.info("Bybit Testnet connector usage is disabled in config ('use_bybit_testnet_connector': false).")


    # 4. Initialize Gemini Analyzer (Example Usage - conditional)
    # Ensure gemini_analyzer is imported: from .gemini_analyzer import GeminiAnalyzer
    # Ensure data_handler is imported for WS callbacks: from .data_handler import DataHandler

    data_handler = DataHandler(config_logging=False) # Initialize DataHandler

    # Prepare callback map for BybitConnector WebSockets
    data_handler_callback_map = {
        'kline': data_handler.handle_ws_kline,
        'orderbook': data_handler.handle_ws_orderbook,
        'public_trade': data_handler.handle_ws_public_trade,
        'orders': data_handler.handle_ws_order_update,       # For private WS
        'positions': data_handler.handle_ws_position_update  # For private WS
    }

    # Initialize Bybit Connector (Example Usage - conditional or commented out for production)
    if use_bybit_testnet_connector: # This flag is from the existing Bybit test block
        # ... (existing BybitConnector instantiation) ...

            bybit_conn = BybitConnector(
                api_key=bybit_testnet_api_key,
                api_secret=bybit_testnet_api_secret,
                testnet=True,
                config_logging=False
            )

            if bybit_conn.session: # Check if HTTP session was initialized successfully
                logger.info("Bybit Testnet Connector initialized successfully for REST.")

                # --- Conceptual WebSocket Setup (Public) ---
                logger.info("--- Conceptual Bybit Public WebSocket Setup ---")
                bybit_conn.connect_public_ws(data_handler_callback_map)
                bybit_conn.subscribe_to_kline_public(symbol="BTCUSDT", interval_minutes=1, category="linear")
                bybit_conn.subscribe_to_orderbook_public(symbol="BTCUSDT", depth=25, category="linear")
                # bybit_conn.subscribe_to_public_trades(symbol="BTCUSDT", category="linear") # Example
                bybit_conn.start_public_ws() # Logs conceptual start
                logger.info("--- Finished Conceptual Bybit Public WebSocket Setup ---")

                # --- Conceptual WebSocket Setup (Private) - if keys are valid ---
                # This assumes bybit_testnet_api_key and secret are valid for private WS too.
                logger.info("--- Conceptual Bybit Private WebSocket Setup ---")
                bybit_conn.connect_private_ws(data_handler_callback_map)
                if bybit_conn.ws_private: # Check if private WS client initialized
                    bybit_conn.subscribe_to_orders_private()
                    bybit_conn.subscribe_to_positions_private()
                    bybit_conn.start_private_ws() # Logs conceptual start
                else:
                    logger.warning("Skipping private WebSocket setup as client did not initialize (check API keys).")
                logger.info("--- Finished Conceptual Bybit Private WebSocket Setup ---")

                # ... (existing Bybit REST tests can remain or be moved/commented) ...
            else:
                logger.error("Bybit Testnet Connector session FAILED to initialize. Check API keys and connector logs.")
        else:
            logger.warning("Bybit Testnet API key/secret not found in config or are placeholders. Skipping Bybit Connector test and WebSocket setup.")
    else:
        logger.info("Bybit Testnet connector usage is disabled in config ('use_bybit_testnet_connector': false). Skipping REST and WebSocket tests.")


    # 4. Initialize Gemini Analyzer (Example Usage - conditional)
    use_gemini_analyzer_flag = config.get("use_gemini_analyzer", False) # Default to False

    if use_gemini_analyzer_flag:
        logger.info("Attempting to initialize Gemini Analyzer...")
        gemini_api_key = config.get("gemini_api_key")

        if gemini_api_key and "YOUR_" not in gemini_api_key:
            from .gemini_analyzer import GeminiAnalyzer # Import here to avoid issues if not used
            gemini_analyzer = GeminiAnalyzer(
                api_key=gemini_api_key,
                # model_name='gemini-1.5-flash-latest', # Optionally override model, default is 1.5-pro
                config_logging=False # Main app already configured logging
            )

            if gemini_analyzer.model: # Check if model was initialized successfully
                logger.info("Gemini Analyzer initialized successfully.")

                # --- Test Gemini Analyzer (Comment out or make conditional for production) ---
                logger.info("--- Testing Gemini Analyzer ---")
                # Create a very simple prompt for testing basic API call and JSON parsing.
                # The actual prompts will be constructed by the StrategyEngine.
                simple_test_prompt = """
                Please provide a JSON response with a field "status" set to "ok" and a field "test_message" set to "Gemini API is reachable".
                Example: {"status": "ok", "test_message": "Gemini API is reachable"}
                Return ONLY the JSON object.
                """

                analysis_response = gemini_analyzer.analyze_market_data(simple_test_prompt)

                if analysis_response:
                    logger.info(f"Gemini Analyzer test response: {analysis_response}")
                    if "error" in analysis_response:
                         logger.warning(f"Gemini Analyzer test returned an error: {analysis_response.get('message')}")
                else:
                    logger.error("Gemini Analyzer test call failed to return a response.")
                logger.info("--- Finished Gemini Analyzer Test ---")
            else:
                logger.error("Gemini Analyzer FAILED to initialize. Check API key and connector logs.")
        else:
            logger.warning("Gemini API key not found in config or is a placeholder. Skipping Gemini Analyzer test.")
    else:
        logger.info("Gemini Analyzer usage is disabled in config ('use_gemini_analyzer': false).")


    # Placeholder for future bot logic
    logger.info("Initialization complete. Bot is ready for strategy execution (not implemented yet).")

    try:
        # Main bot loop or further setup would go here
        pass

    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True) # exc_info=True logs stack trace
    finally:
        logger.info("=============================================")
        logger.info("👋 Trading Bot Shutting Down...")
        logger.info("=============================================")
        logging.shutdown() # Flushes and closes all handlers

if __name__ == "__main__":
    # Create dummy config.dev.json and example for direct execution testing if they don't exist
    # This is helpful if you run `python bybit_gemini_bot/main.py` directly
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Assuming this file is in bybit_gemini_bot/

    dev_config_path_for_main = os.path.join(project_root, DEFAULT_CONFIG_PATH)
    example_config_path_for_main = os.path.join(project_root, EXAMPLE_CONFIG_PATH)
    configs_dir_for_main = os.path.join(project_root, "configs")

    if not os.path.exists(configs_dir_for_main):
        os.makedirs(configs_dir_for_main)
        print(f"[Setup for main.py direct run] Created '{configs_dir_for_main}' directory.")

    if not os.path.exists(example_config_path_for_main):
        example_data = {
            "bybit_api_key": "YOUR_BYBIT_API_KEY_EXAMPLE",
            "bybit_api_secret": "YOUR_BYBIT_API_SECRET_EXAMPLE",
            "gemini_api_key": "YOUR_GEMINI_API_KEY_EXAMPLE",
            "gemini_api_secret": "YOUR_GEMINI_API_SECRET_EXAMPLE",
            "testnet_api_key_bybit": "YOUR_BYBIT_TESTNET_API_KEY_EXAMPLE", # Ensure this is present
            "testnet_api_secret_bybit": "YOUR_BYBIT_TESTNET_API_SECRET_EXAMPLE", # Ensure this is present
            "mainnet_api_key_bybit": "YOUR_BYBIT_MAINNET_API_KEY_EXAMPLE",
            "mainnet_api_secret_bybit": "YOUR_BYBIT_MAINNET_API_SECRET_EXAMPLE",
            "trading_symbols": ["BTCUSD", "ETHUSD"],
            "risk_percentage": 0.01,
            "some_other_setting": "example_value",
            "use_bybit_testnet_connector": False,
            "use_gemini_analyzer": False,
            "use_websockets": False # Add new flag for StrategyEngine WS usage
        }
        with open(example_config_path_for_main, 'w') as f:
            import json
            json.dump(example_data, f, indent=2)
        print(f"[Setup for main.py direct run] Created dummy '{example_config_path_for_main}'.")


    if not os.path.exists(dev_config_path_for_main):
        # If dev config doesn't exist, copy example to dev for the test run
        # In a real scenario, user would create this.
        print(f"[Setup for main.py direct run] '{dev_config_path_for_main}' not found.")
        if os.path.exists(example_config_path_for_main):
            import shutil
            shutil.copy(example_config_path_for_main, dev_config_path_for_main)
            print(f"[Setup for main.py direct run] Copied '{example_config_path_for_main}' to '{dev_config_path_for_main}'.")
            print(f"[Setup for main.py direct run] IMPORTANT: For actual use, please edit '{dev_config_path_for_main}' with your real API keys and settings.")
        else:
            print(f"[Setup for main.py direct run] CRITICAL: Example config '{example_config_path_for_main}' also missing. Cannot create dev config.")

    # 5. Initialize Strategy Engine (Conceptual)
    # This would require all components (BybitConnector, GeminiAnalyzer, DataHandler, RiskManager)
    # For this subtask, we'll just show conceptual instantiation if flags are set.
    if use_bybit_testnet_connector and use_gemini_analyzer_flag : # And if bybit_conn and gemini_analyzer were successfully created
        logger.info("--- Conceptual StrategyEngine Setup ---")
        # Assuming bybit_conn and gemini_analyzer are available from above blocks
        # Need to instantiate RiskManager too
        from .risk_manager import RiskManager
        risk_manager = RiskManager(config=config, config_logging=False)

        # Check if bybit_conn and gemini_analyzer were successfully instantiated
        # For simplicity in this conceptual setup, we'll assume they are if flags are true and keys are present.
        # A more robust check would verify the actual objects.

        if config.get("testnet_api_key_bybit") and not "YOUR_" in config.get("testnet_api_key_bybit", "") \
           and config.get("gemini_api_key") and not "YOUR_" in config.get("gemini_api_key", ""):

            # Re-check bybit_conn existence for safety, as it's conditional
            if not 'bybit_conn' in locals() or bybit_conn.session is None:
                 logger.warning("Bybit connector (bybit_conn) not properly initialized. Cannot fully init StrategyEngine.")
                 # Create a dummy/mock if needed for partial SE test, or skip
                 class DummyBybit: session = None; ws_public=None; ws_private=None; connect_public_ws=lambda x,y:None; subscribe_to_kline_public=lambda x,y,z,w:None; subscribe_to_orderbook_public=lambda x,y,z,w:None; start_public_ws=lambda x:None; connect_private_ws=lambda x,y:None; subscribe_to_orders_private=lambda x:None; subscribe_to_positions_private=lambda x:None; start_private_ws=lambda x:None
                 bybit_conn_for_se = DummyBybit() # type: ignore
            else:
                 bybit_conn_for_se = bybit_conn

            # Re-check gemini_analyzer existence
            if not 'gemini_analyzer' in locals() or gemini_analyzer.model is None :
                 logger.warning("Gemini analyzer not properly initialized. Cannot fully init StrategyEngine.")
                 class DummyGemini: model=None; analyze_market_data=lambda x,y:None
                 gemini_analyzer_for_se = DummyGemini() # type: ignore
            else:
                gemini_analyzer_for_se = gemini_analyzer


            from .strategy_engine import StrategyEngine
            strategy_engine = StrategyEngine(
                config=config,
                bybit_connector=bybit_conn_for_se, # Use the potentially dummy connector if real one failed
                gemini_analyzer=gemini_analyzer_for_se, # Use the potentially dummy analyzer
                data_handler=data_handler,
                risk_manager=risk_manager,
                config_logging=False
            )
            logger.info("StrategyEngine conceptually initialized.")

            # --- Main Trading Loop (Polling Example) ---
            if not config.get("use_websockets", False): # Only run polling loop if not using websockets
                logger.info("Starting polling-based trading loop...")
                trading_symbols_list = config.get("trading_symbols", [])
                if not trading_symbols_list:
                    logger.warning("No trading_symbols configured. Exiting main trading loop.")

                # Determine if it's a dry run (no actual orders) or live
                dry_run_mode = config.get("dry_run_mode", True) # Default to dry run
                if dry_run_mode:
                    logger.info("DRY RUN MODE ENABLED: No actual orders will be placed.")
                else:
                    logger.info("LIVE TRADING MODE ENABLED: Actual orders will be placed.")
                    # Potentially add a final warning/countdown here in a real bot

                # In a real bot, this loop would run indefinitely or on a schedule
                # For this subtask, we might run it once or a few times for testing.
                max_cycles_per_symbol = config.get("max_test_cycles_per_symbol", 1)

                for symbol in trading_symbols_list:
                    logger.info(f"Processing symbol: {symbol}")
                    # Ensure bybit_conn_for_se is not a dummy if we expect real interaction
                    if hasattr(bybit_conn_for_se, 'get_kline') and bybit_conn_for_se.session is not None:
                        for cycle_num in range(max_cycles_per_symbol):
                            logger.info(f"--- Starting cycle {cycle_num + 1}/{max_cycles_per_symbol} for {symbol} ---")
                            try:
                                strategy_engine.run_main_cycle(symbol)
                            except Exception as e_cycle:
                                logger.error(f"Exception during trading cycle for {symbol}: {e_cycle}", exc_info=True)

                            if max_cycles_per_symbol > 1: # Avoid sleep if only one test cycle
                                polling_interval = config.get('polling_interval_seconds', 180)
                                logger.info(f"Cycle finished for {symbol}. Sleeping for {polling_interval}s...")
                                time.sleep(polling_interval)
                        logger.info(f"Finished all configured cycles for symbol {symbol}.")
                    else:
                        logger.warning(f"Skipping trading cycle for {symbol} as Bybit connector is not fully initialized.")
                logger.info("All symbols processed for configured cycles.")
            else:
                logger.info("WebSocket mode is enabled in config. Polling loop in main.py is skipped.")
                logger.info("Ensure WebSocket connections are started and StrategyEngine.process_websocket_update is triggered by DataHandler/callbacks.")

        else:
            logger.warning("StrategyEngine not initialized as required API keys (Bybit Testnet, Gemini) are placeholders or missing in config, or connectors failed.")
        logger.info("--- Finished Conceptual StrategyEngine Setup & Main Loop ---")
    else:
        logger.info("StrategyEngine conceptual setup skipped as Bybit connector or Gemini analyzer is disabled or keys missing.")


    # Placeholder for future bot logic (e.g., starting the main application loop)
    logger.info("Main script initialization complete. Bot is ready for further development (e.g., main event loop).")

    try:
        # Main bot loop or further setup would go here in a real application
        # For example, if using WebSockets, this might be where you wait for threads or handle asyncio event loop.
        # time.sleep(10) # Keep alive for a short time to see WS messages if they were actually started (they are not in this subtask)
        pass

    except KeyboardInterrupt:
        logger.info("Bot shutdown requested by user (KeyboardInterrupt).")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True) # exc_info=True logs stack trace
    finally:
        logger.info("=============================================")
        logger.info("👋 Trading Bot Shutting Down (Main Script Finished)...")
        logger.info("=============================================")
        logging.shutdown() # Flushes and closes all handlers

if __name__ == "__main__":
    # Create dummy config.dev.json and example for direct execution testing if they don't exist
    # This is helpful if you run `python bybit_gemini_bot/main.py` directly
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir) # Assuming this file is in bybit_gemini_bot/

    dev_config_path_for_main = os.path.join(project_root, DEFAULT_CONFIG_PATH)
    example_config_path_for_main = os.path.join(project_root, EXAMPLE_CONFIG_PATH)
    configs_dir_for_main = os.path.join(project_root, "configs")

    if not os.path.exists(configs_dir_for_main):
        os.makedirs(configs_dir_for_main)
        print(f"[Setup for main.py direct run] Created '{configs_dir_for_main}' directory.")

    if not os.path.exists(example_config_path_for_main):
        example_data = {
            "bybit_api_key": "YOUR_BYBIT_API_KEY_EXAMPLE",
            "bybit_api_secret": "YOUR_BYBIT_API_SECRET_EXAMPLE",
            "gemini_api_key": "YOUR_GEMINI_API_KEY_EXAMPLE", # Ensure this is present
            "gemini_api_secret": "YOUR_GEMINI_API_SECRET_EXAMPLE", # Ensure this is present
            "testnet_api_key_bybit": "YOUR_BYBIT_TESTNET_API_KEY_EXAMPLE",
            "testnet_api_secret_bybit": "YOUR_BYBIT_TESTNET_API_SECRET_EXAMPLE",
            "mainnet_api_key_bybit": "YOUR_BYBIT_MAINNET_API_KEY_EXAMPLE",
            "mainnet_api_secret_bybit": "YOUR_BYBIT_MAINNET_API_SECRET_EXAMPLE",
            "trading_symbols": ["BTCUSDT", "ETHUSDT"],
            "risk_per_trade_percentage": 1.0, # Added for RiskManager
            "stop_loss_percentage": 0.02,     # Added for RiskManager/Backtester SL
            "tp_fee_multiplier": 3.0,         # Added for RiskManager TP calc
            "risk_reward_ratio": 1.5,         # Added for RiskManager fallback TP
            "taker_fee_rate": 0.00055,        # Added for fee calculations
            "some_other_setting": "example_value",
            "use_bybit_testnet_connector": False,
            "use_gemini_analyzer": False,
            "use_websockets": False,
            "ai_confidence_threshold": 0.6,
            "min_kline_data_points": 20,
            "polling_interval_seconds": 180,
            "max_test_cycles_per_symbol": 1, # For testing, run each symbol once
            "dry_run_mode": True # IMPORTANT: Default to dry run
        }
        with open(example_config_path_for_main, 'w') as f:
            import json
            json.dump(example_data, f, indent=2)
        print(f"[Setup for main.py direct run] Created dummy '{example_config_path_for_main}'.")


    if not os.path.exists(dev_config_path_for_main):
        # If dev config doesn't exist, copy example to dev for the test run
        # In a real scenario, user would create this.
        print(f"[Setup for main.py direct run] '{dev_config_path_for_main}' not found.")
        if os.path.exists(example_config_path_for_main):
            import shutil
            shutil.copy(example_config_path_for_main, dev_config_path_for_main)
            print(f"[Setup for main.py direct run] Copied '{example_config_path_for_main}' to '{dev_config_path_for_main}'.")
            print(f"[Setup for main.py direct run] IMPORTANT: For actual use, please edit '{dev_config_path_for_main}' with your real API keys and settings.")
        else:
            print(f"[Setup for main.py direct run] CRITICAL: Example config '{example_config_path_for_main}' also missing. Cannot create dev config.")

    main()
