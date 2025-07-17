import logging
import time
import hmac
import hashlib
import json
from pybit.unified_trading import HTTP

class BybitConnector:
    """
    Connector for Bybit V5 API using pybit library.
    Handles requests and responses for market data, account data, and order execution.
    """

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True, config_logging: bool = True):
        """
        Initializes the BybitConnector.

        Args:
            api_key (str): The API key.
            api_secret (str): The API secret.
            testnet (bool, optional): Whether to use the Testnet. Defaults to True.
            config_logging (bool, optional): Whether to configure basic logging if no handlers are set.
                                            Defaults to True. This is mainly for standalone testing.
                                            In the main app, logging should be configured by logger_setup.
        """
        self.logger = logging.getLogger(__name__)
        if config_logging and not logging.getLogger().hasHandlers(): # Configure basic logging if not already configured
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
            self.logger.info("Basic logging configured for BybitConnector standalone usage.")

        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.session = None
        self.ws_public = None
        self.ws_private = None

        # Determine WebSocket URLs
        if self.testnet:
            self.ws_url_public_linear = "wss://stream-testnet.bybit.com/v5/public/linear"
            self.ws_url_private = "wss://stream-testnet.bybit.com/v5/private"
            # Add other categories like spot, option if needed
            # self.ws_url_public_spot = "wss://stream-testnet.bybit.com/v5/public/spot"
        else:
            self.ws_url_public_linear = "wss://stream.bybit.com/v5/public/linear"
            self.ws_url_private = "wss://stream.bybit.com/v5/private"
            # self.ws_url_public_spot = "wss://stream.bybit.com/v5/public/spot"

        # Initialize HTTP session
        if self.api_key and self.api_secret: # HTTP session needs keys for most authenticated endpoints
            try:
                self.session = HTTP(
                    testnet=self.testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    log_requests=False # We will do custom logging in _make_request
                )
                self.logger.info(f"BybitConnector HTTP session initialized. Testnet: {self.testnet}. API Key ending with: ...{self.api_key[-4:]}")
            except Exception as e:
                self.logger.error(f"Failed to initialize pybit HTTP session: {e}", exc_info=True)
                self.session = None
        elif self.api_key: # Only API key might be provided for read-only operations or certain WS auth.
             self.logger.info(f"BybitConnector initialized with API Key only (no secret). Testnet: {self.testnet}. API Key ending with: ...{self.api_key[-4:]}")
        else:
            self.logger.warning("BybitConnector initialized without API key/secret. Only public unauthenticated endpoints will work.")
            # self.session can remain None, or a basic HTTP session for public GETs could be set up if pybit allows
            # For now, assuming self.session is used for authenticated calls or calls pybit handles.


    def _make_request(self, method_name: str, **kwargs):
        """
        Private helper method to make requests to the Bybit API via pybit session.

        Args:
            method_name (str): The name of the pybit session method to call (e.g., "get_kline").
            **kwargs: Arguments to pass to the pybit session method.

        Returns:
            dict: The JSON response from the API, or None if an error occurs.
        """
        if self.session is None:
            self.logger.error(f"Cannot make request '{method_name}'. Session not initialized (likely missing API key/secret or initialization failed).")
            return None

        try:
            method_to_call = getattr(self.session, method_name)

            # Log request (mask sensitive data if necessary - though pybit methods usually don't take secrets directly here)
            log_kwargs = {k: (v if k not in ['api_key', 'api_secret'] else '********') for k,v in kwargs.items()}
            self.logger.debug(f"Making Bybit API request. Method: {method_name}, Params: {log_kwargs}")

            response = method_to_call(**kwargs)

            # Log rate limit headers
            if hasattr(self.session, 'last_response_headers'): # pybit stores last response headers
                headers = self.session.last_response_headers
                rate_limit_status = headers.get('X-Bapi-Limit-Status')
                rate_limit = headers.get('X-Bapi-Limit')
                rate_limit_reset = headers.get('X-Bapi-Limit-Reset-Timestamp')
                if rate_limit_status:
                    self.logger.debug(f"Rate Limit Status: {rate_limit_status}, Limit: {rate_limit}, Reset: {rate_limit_reset}")

            # Check retCode
            if isinstance(response, dict):
                ret_code = response.get('retCode')
                if ret_code == 0:
                    self.logger.debug(f"Request '{method_name}' successful. retCode: {ret_code}")
                    return response
                else:
                    ret_msg = response.get('retMsg', 'No message')
                    ret_ext_info = response.get('retExtInfo', {})
                    self.logger.error(
                        f"Bybit API Error on '{method_name}'. retCode: {ret_code}, retMsg: '{ret_msg}', retExtInfo: {ret_ext_info}. Params: {log_kwargs}"
                    )
                    # For critical errors like auth failure, you might want to raise an exception
                    # if ret_code in [10003, 10004, 10005]: # Example: Invalid API key, signature errors
            #     self.logger.critical(f"Critical API error: {ret_code} - {ret_msg}. Consider raising an exception or stopping.")
            #     # raise BybitAPIException(ret_code, ret_msg, ret_ext_info) # If custom exception is defined
                    return response # Return error response for caller to handle
            else:
            self.logger.error(f"Unexpected response format from '{method_name}': {type(response)}. Expected dict. Response: {response}")
                return None

        except AttributeError:
            self.logger.error(f"Method '{method_name}' not found in pybit session.", exc_info=True)
            return None
        except Exception as e: # Catching generic Exception, pybit might raise specific ones
            self.logger.error(f"Exception during Bybit API request '{method_name}': {e}", exc_info=True)
            return None

    # --- Market Data Functions ---
    def get_kline(self, category: str, symbol: str, interval: str, limit: int = 200, start: int = None, end: int = None):
        params = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        return self._make_request("get_kline", **params)

    def get_instrument_info(self, category: str, symbol: str = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._make_request("get_instruments_info", **params)

    def get_tickers(self, category: str, symbol: str = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._make_request("get_tickers", **params)

    def get_orderbook(self, category: str, symbol: str, limit: int = 1): # Default limit to 1 for top bid/ask as per Bybit doc for V5
        # For category=linear or option, limit can be 1 to 50. For spot, 1 to 500.
        # pybit default is 25, but Bybit docs often show 1 for /v5/market/orderbook
        return self._make_request("get_orderbook", category=category, symbol=symbol, limit=limit)

    def get_server_time(self):
        # v5/market/time endpoint
        return self._make_request("get_server_time")


    # --- Account Data Functions ---
    def get_wallet_balance(self, account_type: str = "UNIFIED", coin: str = None):
        # accountType: UNIFIED, CONTRACT, SPOT
        params = {"accountType": account_type}
        if coin:
            params["coin"] = coin
        return self._make_request("get_wallet_balance", **params)

    def get_fee_rates(self, category: str, symbol: str = None):
        # category: spot, linear, inverse, option
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        return self._make_request("get_account_fee_rate", **params)


    # --- Order Execution Functions ---
    def place_order(self, category: str, symbol: str, side: str, order_type: str, qty: str, # qty should be string
                    price: str = None, time_in_force: str = "GTC", position_idx: int = 0,
                    take_profit: str = None, stop_loss: str = None, order_link_id: str = None,
                    reduce_only: bool = False, close_on_trigger: bool = False,
                    sl_trigger_by: str = None, tp_trigger_by: str = None,
                    trigger_price: str = None, trigger_direction: int = None,
                    is_leverage: int = None, # For spot margin
                    smp_type: str = 'None' # Standard market protection type
                    ):
        params = {
            "category": category, "symbol": symbol, "side": side, "orderType": order_type, "qty": str(qty),
            "timeInForce": time_in_force, "positionIdx": position_idx,
            "reduceOnly": reduce_only, "closeOnTrigger": close_on_trigger,
            "smpType": smp_type
        }
        if price: # Required for limit orders
            params["price"] = str(price)
        if take_profit:
            params["takeProfit"] = str(take_profit)
        if stop_loss:
            params["stopLoss"] = str(stop_loss)
        if order_link_id:
            params["orderLinkId"] = order_link_id
        if sl_trigger_by:
            params["slTriggerBy"] = sl_trigger_by
        if tp_trigger_by:
            params["tpTriggerBy"] = tp_trigger_by
        if trigger_price: # For conditional orders
             params["triggerPrice"] = str(trigger_price)
        if trigger_direction: # For conditional orders (1: Rise, 2: Fall)
             params["triggerDirection"] = trigger_direction
        if is_leverage is not None and category == 'spot': # For spot margin trading
            params["isLeverage"] = is_leverage

        return self._make_request("place_order", **params)

    def amend_order(self, category: str, symbol: str, order_id: str = None, order_link_id: str = None,
                    price: str = None, qty: str = None, take_profit: str = None, stop_loss: str = None,
                    sl_trigger_price: str = None, tp_trigger_price: str = None,
                    trigger_price: str = None): # For conditional orders
        params = {"category": category, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        if price:
            params["price"] = str(price)
        if qty:
            params["qty"] = str(qty)
        if take_profit: # For TP/SL on order
            params["takeProfit"] = str(take_profit)
        if stop_loss: # For TP/SL on order
            params["stopLoss"] = str(stop_loss)
        if sl_trigger_price: # For conditional orders, amending trigger price for SL
            params["slTriggerPrice"] = str(sl_trigger_price)
        if tp_trigger_price: # For conditional orders, amending trigger price for TP
            params["tpTriggerPrice"] = str(tp_trigger_price)
        if trigger_price: # For conditional orders
            params["triggerPrice"] = str(trigger_price)

        return self._make_request("amend_order", **params)

    def cancel_order(self, category: str, symbol: str, order_id: str = None, order_link_id: str = None):
        params = {"category": category, "symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        if order_link_id:
            params["orderLinkId"] = order_link_id
        return self._make_request("cancel_order", **params)

    def get_trade_history(self, category: str, symbol: str = None, order_id: str = None, limit: int = 50,
                          start_time: int = None, end_time: int = None):
        # Uses get_executions for V5 API
        params = {"category": category, "limit": limit}
        if symbol:
            params["symbol"] = symbol
        if order_id:
            params["orderId"] = order_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._make_request("get_executions", **params)

    # --- Position Management Functions ---
    def get_positions(self, category: str, symbol: str = None, base_coin: str = None, settle_coin: str = None):
        params = {"category": category}
        if symbol:
            params["symbol"] = symbol
        if base_coin: # For inverse contracts
            params["baseCoin"] = base_coin
        if settle_coin: # For linear contracts (USDT or USDC settled)
            params["settleCoin"] = settle_coin
        return self._make_request("get_positions", **params)

    def set_leverage(self, category: str, symbol: str, buy_leverage: str, sell_leverage: str): # Leverage as string e.g. "10"
        # For category=linear (except USDT margin) & inverse, buyLeverage and sellLeverage must be the same
        # For category=linear (USDT margin), buyLeverage and sellLeverage can be different
        params = {"category": category, "symbol": symbol, "buyLeverage": str(buy_leverage), "sellLeverage": str(sell_leverage)}
        return self._make_request("set_leverage", **params)

    def set_trading_stop(self, category: str, symbol: str, position_idx: int = 0,
                         take_profit: str = None, stop_loss: str = None,
                         tp_trigger_by: str = None, sl_trigger_by: str = None,
                         tpsl_mode: str = "Full", # Full or Partial
                         sl_order_type: str = "Market", tp_order_type: str = "Market",
                         active_price: str = None # For conditional TP/SL
                        ):
        # positionIdx: 0 for one-way mode, 1 for buy-side hedge, 2 for sell-side hedge
        params = {
            "category": category, "symbol": symbol, "positionIdx": position_idx,
            "tpslMode": tpsl_mode, "slOrderType": sl_order_type, "tpOrderType": tp_order_type
        }
        if take_profit: # Price for TP
            params["takeProfit"] = str(take_profit)
        if stop_loss: # Price for SL
            params["stopLoss"] = str(stop_loss)
        if tp_trigger_by: # MarkPrice, IndexPrice, LastPrice
            params["tpTriggerBy"] = tp_trigger_by
        if sl_trigger_by: # MarkPrice, IndexPrice, LastPrice
            params["slTriggerBy"] = sl_trigger_by
        if active_price: # For conditional TP/SL, price to activate the TP/SL order
            params["activePrice"] = str(active_price)

        return self._make_request("set_trading_stop", **params)

    # --- WebSocket Methods ---

    # Placeholder callback examples (these would typically be methods of DataHandler passed in)
    def _handle_dummy_kline_message(self, message):
        self.logger.info(f"WS KLINE (dummy handler): {message}")

    def _handle_dummy_orderbook_message(self, message):
        self.logger.info(f"WS ORDERBOOK (dummy handler): {message}")

    def _handle_dummy_public_trade_message(self, message):
        self.logger.info(f"WS PUBLIC_TRADE (dummy handler): {message}")

    def _handle_dummy_order_update_message(self, message):
        self.logger.info(f"WS ORDER_UPDATE (dummy handler): {message}")

    def _handle_dummy_position_update_message(self, message):
        self.logger.info(f"WS POSITION_UPDATE (dummy handler): {message}")


    def connect_public_ws(self, data_handler_callback_map: dict = None):
        """
        Initializes the public WebSocket client.
        Args:
            data_handler_callback_map (dict, optional): Maps topics to DataHandler methods.
                Example: {'kline': data_handler.handle_ws_kline, ...}
                         If None, internal dummy handlers will be used for logging.
        """
        if self.ws_public:
            self.logger.warning("Public WebSocket already initialized. Disconnect first or use existing.")
            return

        self.logger.info(f"Connecting to Public WebSocket: {self.ws_url_public_linear} (using linear channel type by default)")
        self.ws_public = HTTP( # Re-using HTTP for WebSocket setup as per pybit docs for v5
            testnet=self.testnet,
            # channel_type="linear" # channel_type is for the WebSocket class, not HTTP
        )
        # The actual WebSocket object is created when subscribing to a stream using the HTTP session instance.
        # This is a bit confusing in pybit's V5 structure. The HTTP session object is used to *initiate* WebSocket streams.
        # `pybit.unified_trading.WebSocket` is a separate class for managing the WS connection if used directly.
        # For simplicity and consistency with pybit examples, we'll use the HTTP session's stream methods.
        # `self.ws_public` will refer to the HTTP session configured for WS, not a direct WS client instance initially.
        # The actual `WebSocket` instance is managed internally by pybit when using `xxxx_stream` methods on the HTTP session.

        self.public_ws_handlers = data_handler_callback_map or {
            'kline': self._handle_dummy_kline_message,
            'orderbook': self._handle_dummy_orderbook_message,
            'public_trade': self._handle_dummy_public_trade_message,
        }
        self.logger.info("Public WebSocket 'connection' (via HTTP session for stream setup) ready.")


    def subscribe_to_kline_public(self, symbol: str, interval_minutes: int, category: str = "linear"):
        if not self.ws_public or not self.public_ws_handlers or 'kline' not in self.public_ws_handlers:
            self.logger.error("Public WebSocket or kline handler not initialized. Call connect_public_ws() first with a kline callback.")
            return
        topic = f"kline.{interval_minutes}.{symbol}" # Example topic format, adjust if pybit uses different
        self.logger.info(f"Subscribing to public K-line for {symbol}, interval {interval_minutes}min (Category: {category})")
        # pybit's HTTP session object is used to start streams.
        # The callback is directly passed.
        try:
            # This is how pybit docs show starting a kline stream with the HTTP session object
            self.session.kline_stream(
                symbol=symbol,
                interval=interval_minutes,
                callback=self.public_ws_handlers['kline'],
                category=category
            )
            self.logger.info(f"Successfully initiated K-line stream subscription for {symbol} via HTTP session's kline_stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to K-line stream for {symbol}: {e}", exc_info=True)


    def subscribe_to_orderbook_public(self, symbol: str, depth: int = 25, category: str = "linear"):
        if not self.ws_public or not self.public_ws_handlers or 'orderbook' not in self.public_ws_handlers: # check ws_public (HTTP session)
            self.logger.error("Public WebSocket or orderbook handler not initialized. Call connect_public_ws() first.")
            return
        self.logger.info(f"Subscribing to public Orderbook for {symbol}, depth {depth} (Category: {category})")
        try:
            self.session.orderbook_stream(
                symbol=symbol,
                depth=depth,
                callback=self.public_ws_handlers['orderbook'],
                category=category
            )
            self.logger.info(f"Successfully initiated Orderbook stream subscription for {symbol} via HTTP session's orderbook_stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to Orderbook stream for {symbol}: {e}", exc_info=True)

    def subscribe_to_public_trades(self, symbol: str, category: str = "linear"):
        if not self.ws_public or not self.public_ws_handlers or 'public_trade' not in self.public_ws_handlers:
            self.logger.error("Public WebSocket or public_trade handler not initialized. Call connect_public_ws() first.")
            return
        self.logger.info(f"Subscribing to public Trades for {symbol} (Category: {category})")
        try:
            self.session.public_trade_stream( # Assuming method name, pybit calls it `trade_stream`
                symbol=symbol,
                callback=self.public_ws_handlers['public_trade'],
                category=category
            )
            self.logger.info(f"Successfully initiated Public Trades stream subscription for {symbol} via HTTP session's trade_stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to Public Trades stream for {symbol}: {e}", exc_info=True)


    def start_public_ws(self):
        """
        Conceptually starts the public WebSocket connection.
        In pybit v5, when using streams via HTTP session, there isn't a separate run_forever() for public streams.
        The streams run in background threads started by pybit when a *_stream method is called.
        This method is more of a placeholder or could manage a list of active streams if needed.
        """
        if not self.session: # Check if HTTP session (which manages streams) is available
            self.logger.error("HTTP Session not available. Cannot start WebSocket streams.")
            return
        self.logger.info("Public WebSocket streams (kline, orderbook, etc.) are managed by pybit's HTTP session object and run in background threads once subscribed.")
        self.logger.info("No explicit 'run_forever()' needed for public streams when using HTTP session's stream methods.")
        # If using `pybit.unified_trading.WebSocket` directly, then `run_forever()` would be used.
        # For now, this method serves as a conceptual placeholder.

    def connect_private_ws(self, data_handler_callback_map: dict = None):
        """Initializes the private WebSocket client using pybit's WebSocket class."""
        if self.ws_private:
            self.logger.warning("Private WebSocket already initialized. Disconnect first or use existing.")
            return

        if not self.api_key or not self.api_secret:
            self.logger.error("API key and secret are required for Private WebSocket. Cannot connect.")
            return

        self.logger.info(f"Connecting to Private WebSocket: {self.ws_url_private}")
        try:
            self.ws_private = pybit.unified_trading.WebSocket( # Direct WebSocket class usage
                testnet=self.testnet,
                channel_type="private", # This is correct for this class
                api_key=self.api_key,
                api_secret=self.api_secret,
                # log_level=logging.DEBUG # Can enable for pybit WS internal logs
            )
            self.private_ws_handlers = data_handler_callback_map or {
                'orders': self._handle_dummy_order_update_message,
                'positions': self._handle_dummy_position_update_message,
            }
            self.logger.info("Private WebSocket client initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize pybit Private WebSocket client: {e}", exc_info=True)
            self.ws_private = None


    def subscribe_to_orders_private(self):
        if not self.ws_private or 'orders' not in self.private_ws_handlers:
            self.logger.error("Private WebSocket or order handler not initialized. Call connect_private_ws() first.")
            return
        self.logger.info("Subscribing to private Order updates.")
        try:
            self.ws_private.order_stream(callback=self.private_ws_handlers['orders'])
            self.logger.info("Successfully subscribed to private Order stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to private Order stream: {e}", exc_info=True)

    def subscribe_to_positions_private(self):
        if not self.ws_private or 'positions' not in self.private_ws_handlers:
            self.logger.error("Private WebSocket or position handler not initialized. Call connect_private_ws() first.")
            return
        self.logger.info("Subscribing to private Position updates.")
        try:
            self.ws_private.position_stream(callback=self.private_ws_handlers['positions'])
            self.logger.info("Successfully subscribed to private Position stream.")
        except Exception as e:
            self.logger.error(f"Error subscribing to private Position stream: {e}", exc_info=True)

    def start_private_ws(self):
        """
        Starts the private WebSocket connection loop in a separate thread.
        """
        if not self.ws_private:
            self.logger.error("Private WebSocket not initialized. Cannot start.")
            return

        self.logger.info("Starting private WebSocket connection (run_forever in a new thread)...")
        # self.ws_private.run_forever() # This would block.
        # In a real application, this needs to be run in a separate thread:
        # import threading
        # ws_thread = threading.Thread(target=self.ws_private.run_forever, daemon=True)
        # ws_thread.start()
        self.logger.info("Conceptual: Private WebSocket would be started in a separate thread using ws_private.run_forever().")
        self.logger.info("For this subtask, not actually starting the blocking loop.")


class BybitAPIException(Exception):
    """Custom exception for Bybit API errors."""
    def __init__(self, ret_code, message, extended_info=None):
        super().__init__(f"Bybit API Error: Code {ret_code} - {message}")
        self.ret_code = ret_code
        self.message = message
        self.extended_info = extended_info or {}

    def __str__(self):
        return f"BybitAPIException: [Code: {self.ret_code}] {self.message} (Details: {self.extended_info})"


if __name__ == '__main__':
    # This is for basic, standalone testing of the BybitConnector.
    # Ensure you have a `configs/config.dev.json` with valid TESTNET keys, or it will use the example.
    # DO NOT COMMIT ACTUAL KEYS.

    print("--- Running BybitConnector Standalone Example ---")

    # Attempt to load config to get API keys for testing
    # This replicates some logic from the main application's config loading.
    import os
    def get_test_config():
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir) # Assumes this file is in bybit_gemini_bot/

        dev_config = os.path.join(project_root, "configs", "config.dev.json")
        example_config = os.path.join(project_root, "configs", "config.json.example")

        config_to_load = dev_config if os.path.exists(dev_config) else example_config

        if not os.path.exists(config_to_load):
            print(f"Error: No config file found at {dev_config} or {example_config}. Cannot run standalone test.")
            return None

        try:
            with open(config_to_load, 'r') as f:
                cfg = json.load(f)
                # Prioritize testnet keys, then mainnet if testnet specific are not found
                api_key = cfg.get("testnet_api_key_bybit") or cfg.get("bybit_api_key")
                api_secret = cfg.get("testnet_api_secret_bybit") or cfg.get("bybit_api_secret")

                if not api_key or "YOUR_" in api_key or not api_secret or "YOUR_" in api_secret:
                    print("Warning: API keys in config are placeholders. Real API calls requiring auth will fail.")
                    print(f"Using Key: {api_key[:5]}...{api_key[-3:] if api_key else 'N/A'}")

                return api_key, api_secret
        except Exception as e:
            print(f"Error loading config for standalone test: {e}")
            return None

    config_keys = get_test_config()
    if not config_keys:
        print("Exiting standalone example due to config load failure.")
        exit()

    test_api_key, test_api_secret = config_keys

    # Initialize connector (using testnet=True)
    # Set config_logging=True to see basic log output if no handlers are configured elsewhere.
    connector = BybitConnector(api_key=test_api_key, api_secret=test_api_secret, testnet=True, config_logging=True)

    if connector.session is None:
        print("Failed to initialize Bybit session. Check API keys and network. Exiting.")
        exit()

    logger = connector.logger # Use the connector's logger for these test messages

    logger.info("--- Testing Market Data Endpoints (Testnet) ---")

    server_time = connector.get_server_time()
    if server_time and server_time.get("retCode") == 0:
        logger.info(f"Server Time: {server_time.get('result', {}).get('timeNano')}")
    else:
        logger.warning(f"Failed to get server time or error in response: {server_time}")

    # K-line for BTCUSDT (Testnet)
    kline_data = connector.get_kline(category="linear", symbol="BTCUSDT", interval="60", limit=5)
    if kline_data and kline_data.get("retCode") == 0:
        logger.info(f"K-line data for BTCUSDT (linear): {len(kline_data.get('result', {}).get('list', []))} candles")
        # logger.debug(json.dumps(kline_data, indent=2)) # Pretty print full response if needed
    else:
        logger.warning(f"Failed to get K-line data or error in response: {kline_data}")

    # Instrument Info for BTCUSDT (Testnet)
    instrument_info = connector.get_instrument_info(category="linear", symbol="BTCUSDT")
    if instrument_info and instrument_info.get("retCode") == 0:
        logger.info(f"Instrument Info for BTCUSDT (linear): Found {len(instrument_info.get('result', {}).get('list', []))} instrument(s).")
        # logger.debug(json.dumps(instrument_info, indent=2))
    else:
        logger.warning(f"Failed to get instrument info or error in response: {instrument_info}")

    # Tickers for SPOT (e.g. BTCUSDT spot - if available on testnet, might differ)
    # Spot symbols may not be on Bybit's V5 testnet for unified trading in the same way as linear.
    # Using linear for more reliable test.
    tickers_linear = connector.get_tickers(category="linear", symbol="BTCUSDT")
    if tickers_linear and tickers_linear.get("retCode") == 0:
        logger.info(f"Tickers for BTCUSDT (linear): {tickers_linear.get('result', {}).get('list', [])[0] if tickers_linear.get('result', {}).get('list') else 'No data'}")
    else:
        logger.warning(f"Failed to get linear tickers or error in response: {tickers_linear}")

    # Orderbook for BTCUSDT (Testnet)
    orderbook_data = connector.get_orderbook(category="linear", symbol="BTCUSDT", limit=5)
    if orderbook_data and orderbook_data.get("retCode") == 0:
        logger.info(f"Orderbook for BTCUSDT (linear): Ask: {orderbook_data.get('result', {}).get('a', [])[:2]}, Bid: {orderbook_data.get('result', {}).get('b', [])[:2]}")
        # logger.debug(json.dumps(orderbook_data, indent=2))
    else:
        logger.warning(f"Failed to get orderbook data or error in response: {orderbook_data}")


    logger.info("--- Testing Account Data Endpoints (Testnet - Requires Valid Testnet API Keys with permissions) ---")

    # Wallet Balance (UNIFIED account on Testnet)
    # Ensure your Testnet API key has "Account Information" permissions
    wallet_balance = connector.get_wallet_balance(account_type="UNIFIED") # UNIFIED or CONTRACT or SPOT
    if wallet_balance and wallet_balance.get("retCode") == 0:
        logger.info(f"Wallet Balance (UNIFIED): {json.dumps(wallet_balance.get('result', {}).get('list', [])[0] if wallet_balance.get('result', {}).get('list') else 'No balance data', indent=2)}")
    else:
        logger.warning(f"Failed to get wallet balance or error in response (check API key permissions): {wallet_balance}")

    # Fee Rates for linear BTCUSDT (Testnet)
    # Ensure your Testnet API key has "Account Information" or "Trading" permissions
    fee_rates = connector.get_fee_rates(category="linear", symbol="BTCUSDT")
    if fee_rates and fee_rates.get("retCode") == 0:
        logger.info(f"Fee Rates for BTCUSDT (linear): {fee_rates.get('result', {}).get('list', [])[0] if fee_rates.get('result', {}).get('list') else 'No fee rate data'}")
    else:
        logger.warning(f"Failed to get fee rates or error in response: {fee_rates}")

    logger.info("--- Example Order Placement & Management (Testnet - Requires Trading Permissions & Funds) ---")
    logger.warning("Order placement tests are commented out by default to prevent accidental orders.")
    logger.warning("Uncomment and ensure your Testnet account has USDT funds to test these.")

    # --- Example: Place a LIMIT order (ensure symbol, qty, price are valid for Testnet) ---
    # order_link_id_example = f"test_order_{int(time.time() * 1000)}"
    # place_order_response = connector.place_order(
    #     category="linear",
    #     symbol="BTCUSDT", # Ensure this is a valid symbol on testnet
    #     side="Buy",
    #     order_type="Limit",
    #     qty="0.001", # Min qty for BTCUSDT on testnet is usually 0.001
    #     price="20000", # Example price, adjust based on current market
    #     time_in_force="GTC",
    #     order_link_id=order_link_id_example
    # )
    # if place_order_response and place_order_response.get("retCode") == 0:
    #     order_id_to_manage = place_order_response.get("result", {}).get("orderId")
    #     logger.info(f"Order placed successfully: {place_order_response}")
    #     logger.info(f"Order ID: {order_id_to_manage}, OrderLinkID: {order_link_id_example}")

    #     # --- Example: Amend the order (e.g., change price) ---
    #     # time.sleep(1) # Give time for order to register
    #     # amend_response = connector.amend_order(
    #     #     category="linear",
    #     #     symbol="BTCUSDT",
    #     #     order_id=order_id_to_manage,
    #     #     price="20005" # New price
    #     # )
    #     # if amend_response and amend_response.get("retCode") == 0:
    #     #     logger.info(f"Order amended successfully: {amend_response}")
    #     # else:
    #     #     logger.error(f"Failed to amend order: {amend_response}")

    #     # --- Example: Cancel the order ---
    #     # time.sleep(1)
    #     # cancel_response = connector.cancel_order(
    #     #     category="linear",
    #     #     symbol="BTCUSDT",
    #     #     order_id=order_id_to_manage
    #     # )
    #     # if cancel_response and cancel_response.get("retCode") == 0:
    #     #     logger.info(f"Order cancelled successfully: {cancel_response}")
    #     # else:
    #     #     logger.error(f"Failed to cancel order: {cancel_response}")

    # else:
    #     logger.error(f"Failed to place order: {place_order_response}")


    # --- Example: Get Trade History (executions) ---
    # trade_history = connector.get_trade_history(category="linear", symbol="BTCUSDT", limit=5)
    # if trade_history and trade_history.get("retCode") == 0:
    #     logger.info(f"Trade History (Executions) for BTCUSDT (linear): {len(trade_history.get('result', {}).get('list', []))} trades found.")
    #     # logger.debug(json.dumps(trade_history, indent=2))
    # else:
    #     logger.warning(f"Failed to get trade history or error in response: {trade_history}")

    # --- Example: Get Positions ---
    # positions = connector.get_positions(category="linear", symbol="BTCUSDT")
    # if positions and positions.get("retCode") == 0:
    #     logger.info(f"Positions for BTCUSDT (linear): {positions.get('result', {}).get('list', [])}")
    # else:
    #     logger.warning(f"Failed to get positions or error in response: {positions}")

    # --- Example: Set Leverage (Only for symbols not in Hedge Mode on Testnet) ---
    # Note: Setting leverage might fail if there are open orders or positions for the symbol.
    # set_leverage_response = connector.set_leverage(category="linear", symbol="BTCUSDT", buy_leverage="5", sell_leverage="5")
    # if set_leverage_response and set_leverage_response.get("retCode") == 0:
    #     logger.info(f"Leverage set successfully for BTCUSDT: {set_leverage_response}")
    # else:
    #     logger.warning(f"Failed to set leverage (check for open orders/positions or hedge mode): {set_leverage_response}")

    logger.info("--- BybitConnector Standalone Example Finished ---")
