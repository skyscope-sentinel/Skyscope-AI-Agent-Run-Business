import logging
import pandas as pd
import pandas_ta as ta # For technical indicators

class DataHandler:
    """
    Handles processing of K-line data and calculation of technical indicators.
    """
    def __init__(self, config_logging: bool = True):
        """
        Initializes the DataHandler.
        Args:
            config_logging (bool, optional): Whether to configure basic logging if no handlers are set.
                                            Defaults to True. This is mainly for standalone testing.
        """
        self.logger = logging.getLogger(__name__)
        if config_logging and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
            self.logger.info("Basic logging configured for DataHandler standalone usage.")

        # Conceptual internal data stores for real-time data
        self.current_orderbooks = {} # Keyed by symbol
        self.recent_klines = {}      # Keyed by symbol, then by interval. Could store DataFrames or lists.
        self.last_1m_kline_partials = {} # For aggregating 1m to 3m klines, keyed by symbol

    def process_kline_data(self, kline_raw_data_list: list, symbol: str) -> pd.DataFrame | None:
        """
        Processes raw K-line data into a Pandas DataFrame.

        Args:
            kline_raw_data_list (list): List of lists/tuples from bybit_connector.get_kline().
                                        Expected format: [[timestamp, open, high, low, close, volume, turnover], ...]
            symbol (str): The trading symbol for logging purposes.

        Returns:
            pd.DataFrame | None: A Pandas DataFrame with columns:
                                 ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                                 indexed by datetime 'timestamp'. Returns None if input is empty or malformed.
        """
        if not kline_raw_data_list:
            self.logger.warning(f"Received empty kline_raw_data_list for {symbol}.")
            return None

        try:
            df = pd.DataFrame(kline_raw_data_list, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            # Convert numeric columns to appropriate types
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert timestamp to datetime and set as index
            # Bybit V5 API returns timestamp in milliseconds (string)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Sort by timestamp just in case data isn't sorted (it usually is)
            df.sort_index(ascending=True, inplace=True)

            # Select only the required columns for the output
            processed_df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            self.logger.info(f"Successfully processed {len(processed_df)} k-line data points for {symbol} into DataFrame.")
            self.logger.debug(f"DataFrame for {symbol} head:\n{processed_df.head()}")
            return processed_df

        except ValueError as ve:
            self.logger.error(f"ValueError processing k-line data for {symbol}: {ve}. Input data: {kline_raw_data_list[:2]}...", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error processing k-line data for {symbol}: {e}. Input data: {kline_raw_data_list[:2]}...", exc_info=True)
            return None

    def calculate_technical_indicators(self, kline_df: pd.DataFrame, symbol: str = "Unknown") -> pd.DataFrame:
        """
        Calculates basic technical indicators and appends them to the DataFrame.

        Args:
            kline_df (pd.DataFrame): DataFrame from process_kline_data (must include open, high, low, close, volume).
            symbol (str): Trading symbol for logging.

        Returns:
            pd.DataFrame: The input DataFrame with added indicator columns.
        """
        if kline_df is None or kline_df.empty:
            self.logger.warning(f"Cannot calculate technical indicators for {symbol}: DataFrame is empty or None.")
            return kline_df

        try:
            self.logger.info(f"Calculating technical indicators for {symbol} on {len(kline_df)} data points.")

            # SMA (Simple Moving Average) - example periods
            kline_df.ta.sma(length=10, append=True, col_names=('SMA_10')) # Names column SMA_10
            kline_df.ta.sma(length=20, append=True, col_names=('SMA_20'))

            # EMA (Exponential Moving Average) - example periods
            kline_df.ta.ema(length=12, append=True, col_names=('EMA_12'))
            kline_df.ta.ema(length=26, append=True, col_names=('EMA_26'))

            # RSI (Relative Strength Index)
            kline_df.ta.rsi(length=14, append=True, col_names=('RSI_14'))

            # MACD (Moving Average Convergence Divergence)
            # This typically creates MACD (e.g. MACD_12_26_9), MACDh (histogram), MACDs (signal)
            kline_df.ta.macd(fast=12, slow=26, signal=9, append=True,
                             col_names=('MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'))

            # Bollinger Bands
            kline_df.ta.bbands(length=20, std=2, append=True,
                               col_names=('BBL_20_2', 'BBM_20_2', 'BBU_20_2', 'BBB_20_2', 'BBP_20_2'))


            # Example: ATR (Average True Range)
            kline_df.ta.atr(length=14, append=True, col_names=('ATR_14'))

            self.logger.info(f"Finished calculating indicators for {symbol}. DataFrame columns: {kline_df.columns.tolist()}")
            self.logger.debug(f"DataFrame for {symbol} with indicators (head):\n{kline_df.head()}")

            # pandas-ta might add columns with NaN for initial periods where indicator cannot be calculated.
            # This is standard. For strategies, one might dropna() or ensure enough data exists.
            # kline_df.dropna(inplace=True) # Optional: drop rows with any NaN values (will shorten df)
            # self.logger.info(f"Shape after potential dropna: {kline_df.shape}")


        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}", exc_info=True)
            # Return original DataFrame if errors occur

        return kline_df

    # --- WebSocket Callback Handlers ---

    def handle_ws_kline(self, message: dict):
        """
        Handles incoming Kline messages from WebSocket.
        Processes and potentially aggregates K-line data.
        """
        try:
            # self.logger.debug(f"Received raw WS Kline: {message}")
            # Kline data from Bybit WS is usually under 'data' key, which is a list of kline objects
            kline_list = message.get('data', [])
            topic = message.get('topic', '') # e.g., kline.1.BTCUSDT
            symbol = topic.split('.')[-1] if topic else "UnknownSymbol" # Extract symbol from topic

            for kline_data in kline_list:
                # self.logger.info(f"Processing WS Kline for {symbol}: Start={kline_data.get('start')}, Close={kline_data.get('close')}, Confirm={kline_data.get('confirm')}")
                if kline_data.get('confirm') is True: # Only process confirmed klines
                    self.logger.info(f"Confirmed 1-min WS Kline for {symbol}: StartTime={pd.to_datetime(int(kline_data['start']), unit='ms')}, Close={kline_data['close']}")
                    # Conceptual: Add to a structure for recent klines
                    if symbol not in self.recent_klines:
                        self.recent_klines[symbol] = {}
                    if '1m' not in self.recent_klines[symbol]:
                        self.recent_klines[symbol]['1m'] = []
                    self.recent_klines[symbol]['1m'].append(kline_data)
                    # Keep only last N klines if needed, e.g., last 200
                    self.recent_klines[symbol]['1m'] = self.recent_klines[symbol]['1m'][-200:]

                    # Conceptual: Aggregate 1-minute klines to 3-minute klines
                    self.aggregate_1m_to_3m_klines(symbol, kline_data)
                # else:
                #     self.logger.debug(f"Received unconfirmed (partial) kline for {symbol}: {kline_data}")

        except Exception as e:
            self.logger.error(f"Error in handle_ws_kline: {e}. Message: {message}", exc_info=True)

    def aggregate_1m_to_3m_klines(self, symbol: str, new_1m_kline_data: dict):
        """
        Conceptually aggregates 1-minute K-lines into 3-minute K-lines.
        This is a simplified placeholder. A robust implementation needs proper state management.
        """
        timestamp_ms = int(new_1m_kline_data['start'])
        open_price = float(new_1m_kline_data['open'])
        high_price = float(new_1m_kline_data['high'])
        low_price = float(new_1m_kline_data['low'])
        close_price = float(new_1m_kline_data['close'])
        volume = float(new_1m_kline_data['volume'])

        # Check if this 1m kline starts a new 3m interval (minute % 3 == 0, e.g. 00, 03, 06...)
        # Bybit kline timestamps are start times of the interval.
        # A 1-minute kline starting at 10:00:00 is for 10:00:00-10:00:59.
        # A 3-minute kline starting at 10:00:00 is for 10:00:00-10:02:59.
        # So, if a 1m kline starts at 10:00, 10:01, 10:02, it belongs to the 3m kline starting at 10:00.

        kline_minute = pd.to_datetime(timestamp_ms, unit='ms').minute

        # Get or initialize the current partial 3-minute kline for the symbol
        partial_kline = self.last_1m_kline_partials.get(symbol)

        if partial_kline is None or kline_minute % 3 == 0: # Start of a new 3-min bar
            if partial_kline and partial_kline['is_complete']: # Log the previously completed 3-min bar
                 self.logger.info(f"COMPLETED 3-min KLINE for {symbol} (from aggregation): "
                                  f"T={pd.to_datetime(partial_kline['start'], unit='ms')}, O={partial_kline['open']:.2f}, H={partial_kline['high']:.2f}, "
                                  f"L={partial_kline['low']:.2f}, C={partial_kline['close']:.2f}, V={partial_kline['volume']:.2f}")

            # Initialize new partial 3-min kline
            # The start time of the 3m kline is the start time of the first 1m kline in that 3m window.
            # This means if a 1m kline arrives for 10:00, 10:01, or 10:02, the 3m kline start is 10:00.
            # A 1m kline for 10:00 has start = 10:00. Its minute is 0. 0 % 3 == 0.
            # A 1m kline for 10:01 has start = 10:01. Its minute is 1. 1 % 3 != 0.
            # A 1m kline for 10:02 has start = 10:02. Its minute is 2. 2 % 3 != 0.
            # A 1m kline for 10:03 has start = 10:03. Its minute is 3. 3 % 3 == 0. So this starts a new 3m bar.

            # Correct start time for the 3-min bar:
            three_min_start_ts = timestamp_ms - (timestamp_ms % (3 * 60 * 1000))

            self.last_1m_kline_partials[symbol] = {
                'start': three_min_start_ts,
                'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price, 'volume': volume,
                'count': 1, # How many 1m klines contributed
                'is_complete': False
            }
            # self.logger.debug(f"[{symbol}] New 3-min partial kline started at {pd.to_datetime(three_min_start_ts, unit='ms')} with 1m data from {pd.to_datetime(timestamp_ms, unit='ms')}")
        else:
            # Update existing partial 3-min kline
            partial_kline['high'] = max(partial_kline['high'], high_price)
            partial_kline['low'] = min(partial_kline['low'], low_price)
            partial_kline['close'] = close_price # Close of the latest 1m kline
            partial_kline['volume'] += volume
            partial_kline['count'] += 1
            # self.logger.debug(f"[{symbol}] Updated 3-min partial kline with 1m data from {pd.to_datetime(timestamp_ms, unit='ms')}. Count: {partial_kline['count']}")


        # Check if this 1m kline completes the current 3m kline (i.e., it's the 3rd one, or its minute is 2, 5, 8 ...)
        # A 1m kline starting at 10:02 completes the 10:00-10:02 (3m) bar. (minute + 1) % 3 == 0
        if (kline_minute + 1) % 3 == 0:
            partial_kline['is_complete'] = True
            # The completed kline will be logged at the start of the *next* 3-minute interval OR
            # could be emitted here via a callback / further processing.
            # For this subtask, we log it when the *next* 3-min bar starts (see above).
            self.logger.info(f"Marked 3-min KLINE for {symbol} starting {pd.to_datetime(partial_kline['start'], unit='ms')} as complete. "
                             f"C={partial_kline['close']:.2f}, V={partial_kline['volume']:.2f}, Count={partial_kline['count']}")


    def handle_ws_orderbook(self, message: dict):
        # self.logger.debug(f"Received raw WS Orderbook: {message}")
        topic = message.get('topic', '')
        symbol = topic.split('.')[-1] if topic else "UnknownSymbol"
        data = message.get('data', {})

        # Orderbook data contains 's' (symbol), 'b' (bids), 'a' (asks), 'ts' (timestamp)
        bids = data.get('b', [])
        asks = data.get('a', [])

        if bids or asks:
            self.logger.info(f"Received WS Orderbook for {symbol}: Top Bid {bids[0] if bids else 'N/A'}, Top Ask {asks[0] if asks else 'N/A'}. Update TS: {data.get('ts')}")
            self.current_orderbooks[symbol] = {'bids': bids, 'asks': asks, 'ts': data.get('ts')}
        else:
            self.logger.debug(f"Received WS Orderbook update for {symbol} without bid/ask data (possibly snapshot setup or empty book): {data}")


    def handle_ws_public_trade(self, message: dict):
        # self.logger.debug(f"Received raw WS Public Trade: {message}")
        # Public trade data is a list under 'data' key
        trade_list = message.get('data', [])
        for trade in trade_list:
            # Fields: T (timestamp), s (symbol), S (side: Buy/Sell), v (volume/qty), p (price), L (tickDirection), i (tradeId), BT (blockTrade)
            self.logger.info(f"Received WS Public Trade for {trade.get('s')}: Side={trade.get('S')}, Px={trade.get('p')}, Qty={trade.get('v')}, Time={pd.to_datetime(trade.get('T'), unit='ms')}")

    def handle_ws_order_update(self, message: dict):
        # self.logger.debug(f"Received raw WS Order Update: {message}")
        # Order update data is a list under 'data' key
        order_list = message.get('data', [])
        for order_data in order_list:
            self.logger.info(f"Received WS Order Update: Symbol={order_data.get('symbol')}, OrderID={order_data.get('orderId')}, Status={order_data.get('orderStatus')}, CumExecQty={order_data.get('cumExecQty')}")
            # Conceptual: Update an internal dictionary of open/closed orders

    def handle_ws_position_update(self, message: dict):
        # self.logger.debug(f"Received raw WS Position Update: {message}")
        # Position update data is a list under 'data' key
        position_list = message.get('data', [])
        for pos_data in position_list:
            self.logger.info(f"Received WS Position Update: Symbol={pos_data.get('symbol')}, Side={pos_data.get('side')}, Size={pos_data.get('size')}, EntryPx={pos_data.get('avgPrice')}, LiqPx={pos_data.get('liqPrice')}")
            # Conceptual: Update an internal dictionary of current positions


if __name__ == '__main__':
    print("--- Running DataHandler Standalone Example ---")

    # Sample raw K-line data (simulating Bybit API response)
    # [timestamp, open, high, low, close, volume, turnover]
    sample_raw_kline = [
        ["1678886400000", "25000.0", "25500.5", "24800.0", "25200.0", "1000.5", "25200000.0"],
        ["1678886460000", "25200.0", "25300.0", "25100.0", "25150.0", "800.2", "20120000.0"],
        ["1678886520000", "25150.0", "25250.0", "25050.0", "25200.0", "900.0", "22680000.0"],
        # Add more data points to allow indicators to calculate properly (e.g., > 20 for SMA_20)
        ["1678886580000", "25200.0", "25400.0", "25180.0", "25350.0", "1200.0", "30420000.0"],
        ["1678886640000", "25350.0", "25380.0", "25280.0", "25300.0", "700.0", "17710000.0"],
        ["1678886700000", "25300.0", "25320.0", "25100.0", "25120.0", "1100.0", "27832000.0"],
        ["1678886760000", "25120.0", "25220.0", "25080.0", "25180.0", "950.0", "23921000.0"],
        ["1678886820000", "25180.0", "25400.0", "25150.0", "25380.0", "1300.0", "33000000.0"],
        ["1678886880000", "25380.0", "25500.0", "25350.0", "25480.0", "1000.0", "25480000.0"],
        ["1678886940000", "25480.0", "25600.0", "25450.0", "25550.0", "1400.0", "35770000.0"], # 10th point
        ["1678887000000", "25550.0", "25580.0", "25400.0", "25420.0", "1050.0", "26691000.0"],
        ["1678887060000", "25420.0", "25500.0", "25380.0", "25480.0", "980.0", "24970400.0"],
        ["1678887120000", "25480.0", "25650.0", "25450.0", "25600.0", "1600.0", "40960000.0"],
        ["1678887180000", "25600.0", "25700.0", "25580.0", "25650.0", "1350.0", "34627500.0"],
        ["1678887240000", "25650.0", "25800.0", "25600.0", "25750.0", "1700.0", "43775000.0"],
        ["1678887300000", "25750.0", "25780.0", "25680.0", "25700.0", "1150.0", "29555000.0"],
        ["1678887360000", "25700.0", "25900.0", "25650.0", "25850.0", "1800.0", "46530000.0"],
        ["1678887420000", "25850.0", "25950.0", "25800.0", "25900.0", "1500.0", "38850000.0"],
        ["1678887480000", "25900.0", "26000.0", "25850.0", "25950.0", "1900.0", "49305000.0"],
        ["1678887540000", "25950.0", "26100.0", "25900.0", "26050.0", "2000.0", "52100000.0"], # 20th point
        ["1678887600000", "26050.0", "26150.0", "26000.0", "26100.0", "1750.0", "45675000.0"], # 21st point for MACD signal line
        ["1678887660000", "26100.0", "26200.0", "26050.0", "26180.0", "1850.0", "48433000.0"],
        ["1678887720000", "26180.0", "26300.0", "26150.0", "26250.0", "1950.0", "51187500.0"],
        ["1678887780000", "26250.0", "26280.0", "26100.0", "26120.0", "1650.0", "43098000.0"],
        ["1678887840000", "26120.0", "26180.0", "26000.0", "26020.0", "1450.0", "37729000.0"],
        ["1678887900000", "26020.0", "26050.0", "25850.0", "25880.0", "2100.0", "54348000.0"], # 26th point for MACD slow EMA
        ["1678887960000", "25880.0", "25950.0", "25800.0", "25900.0", "1500.0", "38850000.0"],
        ["1678888020000", "25900.0", "26000.0", "25880.0", "25980.0", "1600.0", "41568000.0"],
        ["1678888080000", "25980.0", "26050.0", "25950.0", "26000.0", "1300.0", "33800000.0"],
        ["1678888140000", "26000.0", "26100.0", "25980.0", "26080.0", "1700.0", "44336000.0"], # 30th point
        ["1678888200000", "26080.0", "26150.0", "26050.0", "26100.0", "1200.0", "31320000.0"],
        ["1678888260000", "26100.0", "26200.0", "26080.0", "26180.0", "1400.0", "36652000.0"],
        ["1678888320000", "26180.0", "26300.0", "26150.0", "26250.0", "1550.0", "40687500.0"],
        ["1678888380000", "26250.0", "26350.0", "26200.0", "26300.0", "1650.0", "43395000.0"], # 34th point for MACD signal line calculation
    ]

    handler = DataHandler()

    print("\n--- Testing process_kline_data ---")
    kline_df = handler.process_kline_data(sample_raw_kline, "BTCUSDT_Test")
    if kline_df is not None:
        print(f"Processed K-line DataFrame (first 5 rows):\n{kline_df.head()}")
        print(f"\nDataFrame Info:")
        kline_df.info()
    else:
        print("Failed to process K-line data.")

    print("\n--- Testing calculate_technical_indicators ---")
    if kline_df is not None:
        df_with_indicators = handler.calculate_technical_indicators(kline_df.copy(), "BTCUSDT_Test") # Use .copy() to avoid modifying original df
        print(f"DataFrame with Indicators (last 5 rows to see values):\n{df_with_indicators.tail()}")
        print(f"\nDataFrame with Indicators Info:")
        df_with_indicators.info()
        # Check for NaN values (expected at the beginning)
        print(f"\nNaN values in indicator columns (head):\n{df_with_indicators.head().isna().sum()}")
        print(f"NaN values in indicator columns (tail):\n{df_with_indicators.tail().isna().sum()}")
    else:
        print("Skipping indicator calculation as K-line DataFrame was not processed.")

    print("\n--- Testing with empty data ---")
    empty_df = handler.process_kline_data([], "EMPTY_Test")
    if empty_df is None:
        print("Correctly handled empty raw kline list.")

    empty_df_with_indicators = handler.calculate_technical_indicators(empty_df, "EMPTY_Test")
    if empty_df_with_indicators is None: # or .empty if it returns an empty df
        print("Correctly handled empty DataFrame for indicator calculation.")

    print("\n--- DataHandler Standalone Example Finished ---")
