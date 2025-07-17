import pandas as pd
import logging
import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP

# Assuming bybit_gemini_bot modules are in PYTHONPATH or same directory level for simplicity here
# For robust execution, ensure PYTHONPATH is set up correctly if running from project root.
try:
    from bybit_gemini_bot.data_handler import DataHandler
    from bybit_gemini_bot.risk_manager import RiskManager
    from bybit_gemini_bot.config import load_config # For loading app config
except ImportError:
    print("Error: Could not import bot modules during initial attempt. Adjusting sys.path for fallback.")
    # Fallback for standalone execution if modules are in a subfolder and script is in root
    import sys
    import os
    # Add project root to sys.path, assuming script is in project_root/
    # and modules are in project_root/bybit_gemini_bot/
    # The structure is:
    # project_root/
    #   backtester.py  (this file)
    #   bybit_gemini_bot/
    #     __init__.py
    #     data_handler.py
    #     risk_manager.py
    #     config.py
    #   configs/
    #     config.dev.json (or config.json.example)

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    # No, bybit_gemini_bot is the package, so PROJECT_ROOT itself should be in sys.path
    # if bybit_gemini_bot is a top-level package.
    # If bybit_gemini_bot is a directory *containing* the package, then MODULE_PATH logic is fine.
    # Let's assume 'bybit_gemini_bot' is the package name and is a directory in PROJECT_ROOT.

    # If backtester.py is in PROJECT_ROOT, and we need to import `from bybit_gemini_bot.module`
    # then PROJECT_ROOT needs to be in sys.path so python can find the `bybit_gemini_bot` package.
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    # Now try importing again
    from bybit_gemini_bot.data_handler import DataHandler
    from bybit_gemini_bot.risk_manager import RiskManager
    from bybit_gemini_bot.config import load_config


class SimpleBacktester:
    """
    A simplified backtesting engine to test core trading logic.
    """
    def __init__(self, config: dict, historical_data_df: pd.DataFrame,
                 instrument_info: dict, initial_capital_usdt: float = 1000.0,
                 symbol: str = "BTCUSDT"):
        """
        Initializes the SimpleBacktester.

        Args:
            config (dict): Bot's configuration object.
            historical_data_df (pd.DataFrame): Processed K-line data with indicators.
            instrument_info (dict): Instrument details (lot sizes, tick sizes).
            initial_capital_usdt (float): Starting capital.
            symbol (str): Trading symbol.
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.historical_data_df = historical_data_df
        self.instrument_info = instrument_info
        self.symbol = symbol

        self.initial_capital_usdt = Decimal(str(initial_capital_usdt))
        self.balance_usdt = Decimal(str(initial_capital_usdt))

        self.current_position_qty = Decimal('0')
        self.current_position_entry_price = Decimal('0')
        self.current_position_side = None  # "LONG" or "SHORT"
        self.current_stop_loss_price = Decimal('0')
        self.current_take_profit_price = Decimal('0')
        self.entry_timestamp = None # To store timestamp of entry

        self.trades_history = []

        if 'risk_per_trade_percentage' not in self.config:
            self.logger.warning("risk_per_trade_percentage not in config, defaulting to 1.0 for backtester.")
            self.config['risk_per_trade_percentage'] = 1.0

        self.risk_manager = RiskManager(config=self.config, config_logging=False)
        self.data_handler = DataHandler(config_logging=False)

        self.taker_fee_rate = Decimal(str(self.config.get('taker_fee_rate', 0.00055)))

        self.logger.info(f"SimpleBacktester initialized for {self.symbol}. Initial Capital: {self.initial_capital_usdt} USDT. Taker Fee: {self.taker_fee_rate*100}%")
        self.logger.info(f"Instrument Info: {self.instrument_info}")
        if self.historical_data_df.empty:
             self.logger.error("Historical data is empty. Backtester cannot run.")
        else:
             self.logger.info(f"Historical data loaded: {len(self.historical_data_df)} bars, from {self.historical_data_df.index.min()} to {self.historical_data_df.index.max()}")


    def _get_entry_signal(self, current_bar_data: pd.Series, prev_bar_data: pd.Series) -> str | None:
        """
        Simplified entry signal logic based on SMA crossover.
        Requires 'SMA_10' and 'SMA_20' to be in current_bar_data and prev_bar_data.
        """
        if not all(k in current_bar_data for k in ['SMA_10', 'SMA_20']) or \
           not all(k in prev_bar_data for k in ['SMA_10', 'SMA_20']):
            # self.logger.debug("SMA_10 or SMA_20 not available in bar data. No signal.")
            return None
        if pd.isna(current_bar_data['SMA_10']) or pd.isna(current_bar_data['SMA_20']) or \
           pd.isna(prev_bar_data['SMA_10']) or pd.isna(prev_bar_data['SMA_20']):
            # self.logger.debug("SMA values are NaN for current or previous bar. No signal.")
            return None

        # Crossover Logic:
        # LONG signal: SMA10 crosses above SMA20
        if prev_bar_data['SMA_10'] <= prev_bar_data['SMA_20'] and current_bar_data['SMA_10'] > current_bar_data['SMA_20']:
            return "LONG"
        # SHORT signal: SMA10 crosses below SMA20
        elif prev_bar_data['SMA_10'] >= prev_bar_data['SMA_20'] and current_bar_data['SMA_10'] < current_bar_data['SMA_20']:
            return "SHORT"
        return None

    def _calculate_fees(self, quantity: Decimal, price: Decimal) -> Decimal:
        return quantity * price * self.taker_fee_rate

    def run(self):
        if self.historical_data_df.empty or len(self.historical_data_df) < 2:
            self.logger.error("Not enough historical data to run backtest (need at least 2 bars for crossover logic).")
            return
        self.logger.info(f"Starting backtest run for {self.symbol} with {len(self.historical_data_df)} bars...")

        for i in range(1, len(self.historical_data_df)): # Start from the second bar
            prev_bar = self.historical_data_df.iloc[i-1]
            current_bar = self.historical_data_df.iloc[i]
            current_timestamp = current_bar.name
            current_price_decimal = Decimal(str(current_bar['close']))

            if self.current_position_qty > Decimal('0'):
                exit_reason = None
                exit_price = current_price_decimal

                # SL/TP checks (simplified: assumes SL/TP prices are fixed upon entry)
                if self.current_position_side == "LONG":
                    if current_bar['low'] <= self.current_stop_loss_price: # Check if low breached SL
                        exit_reason = "STOP_LOSS"
                        exit_price = self.current_stop_loss_price
                    elif current_bar['high'] >= self.current_take_profit_price: # Check if high breached TP
                        exit_reason = "TAKE_PROFIT"
                        exit_price = self.current_take_profit_price
                elif self.current_position_side == "SHORT":
                    if current_bar['high'] >= self.current_stop_loss_price: # Check if high breached SL
                        exit_reason = "STOP_LOSS"
                        exit_price = self.current_stop_loss_price
                    elif current_bar['low'] <= self.current_take_profit_price: # Check if low breached TP
                        exit_reason = "TAKE_PROFIT"
                        exit_price = self.current_take_profit_price

                if exit_reason:
                    entry_fee = self._calculate_fees(self.current_position_qty, self.current_position_entry_price)
                    exit_fee = self._calculate_fees(self.current_position_qty, exit_price)
                    total_fees = entry_fee + exit_fee

                    pnl_per_unit = (exit_price - self.current_position_entry_price) if self.current_position_side == "LONG" \
                                   else (self.current_position_entry_price - exit_price)
                    gross_pnl = pnl_per_unit * self.current_position_qty
                    net_pnl = gross_pnl - total_fees

                    self.balance_usdt += net_pnl

                    self.logger.info(f"{current_timestamp} - EXIT {self.current_position_side} {self.symbol} at {exit_price:.4f} due to {exit_reason}. Qty: {self.current_position_qty}. "
                                     f"Entry: {self.current_position_entry_price:.4f}. Gross PNL: {gross_pnl:.4f}. Net PNL: {net_pnl:.4f}. Fees: {total_fees:.4f}. Balance: {self.balance_usdt:.4f}")

                    self.trades_history.append({
                        "symbol": self.symbol, "entry_time": self.entry_timestamp, "exit_time": current_timestamp,
                        "entry_price": float(self.current_position_entry_price), "exit_price": float(exit_price),
                        "side": self.current_position_side, "qty": float(self.current_position_qty),
                        "gross_pnl": float(gross_pnl), "net_pnl": float(net_pnl), "total_fees": float(total_fees),
                        "exit_reason": exit_reason, "balance_after_trade": float(self.balance_usdt)
                    })
                    self.current_position_qty = Decimal('0')
                    self.current_position_side = None
                    continue # Skip entry logic for this bar as we just exited

            if self.current_position_qty == Decimal('0'): # Check for new entry
                signal = self._get_entry_signal(current_bar, prev_bar)
                if signal:
                    entry_price = current_price_decimal # Assume entry at current close

                    sl_percentage = Decimal(str(self.config.get('stop_loss_percentage', '0.02'))) # e.g. 2% SL from entry
                    if signal == "LONG":
                        stop_loss_price_hypothetical = float(entry_price * (Decimal('1') - sl_percentage))
                    else:
                        stop_loss_price_hypothetical = float(entry_price * (Decimal('1') + sl_percentage))

                    position_qty_str = self.risk_manager.calculate_position_size(
                        available_balance_usdt=float(self.balance_usdt),
                        entry_price=float(entry_price),
                        stop_loss_price=stop_loss_price_hypothetical,
                        instrument_info=self.instrument_info,
                        symbol=self.symbol,
                        side=signal
                    )

                    if position_qty_str:
                        self.current_position_qty = Decimal(position_qty_str)
                        self.current_position_entry_price = entry_price
                        self.current_position_side = signal
                        self.entry_timestamp = current_timestamp

                        self.current_stop_loss_price = Decimal(str(stop_loss_price_hypothetical))
                        tick_size = self.instrument_info['priceFilter']['tickSize']
                        rounding_sl = ROUND_UP if signal == "LONG" else ROUND_DOWN # Make SL price slightly worse for us
                        self.current_stop_loss_price = self.risk_manager._adjust_to_step(self.current_stop_loss_price, tick_size, rounding_sl)

                        tp_price_str = self.risk_manager.calculate_take_profit_price_for_fee_coverage(
                            entry_price=float(self.current_position_entry_price),
                            side=self.current_position_side,
                            order_qty_base=float(self.current_position_qty),
                            fee_rate_taker=float(self.taker_fee_rate),
                            instrument_info=self.instrument_info,
                            symbol=self.symbol,
                            fee_multiplier=float(self.config.get('tp_fee_multiplier', 3.0))
                        )
                        if tp_price_str:
                            self.current_take_profit_price = Decimal(tp_price_str)
                        else: # Fallback TP based on SL distance (e.g. 1.5x R:R)
                            risk_reward_ratio = Decimal(str(self.config.get('risk_reward_ratio', '1.5')))
                            price_diff_to_sl = abs(self.current_position_entry_price - self.current_stop_loss_price)
                            if signal == "LONG":
                                self.current_take_profit_price = self.current_position_entry_price + (price_diff_to_sl * risk_reward_ratio)
                            else:
                                self.current_take_profit_price = self.current_position_entry_price - (price_diff_to_sl * risk_reward_ratio)
                            rounding_tp = ROUND_UP if signal == "LONG" else ROUND_DOWN # TP rounded in our favor
                            self.current_take_profit_price = self.risk_manager._adjust_to_step(self.current_take_profit_price, tick_size, rounding_tp)
                            self.logger.warning(f"Could not calculate TP based on fee coverage for {self.symbol}. Using R:R based TP: {self.current_take_profit_price:.4f}")

                        # Note: Entry fee is deducted when trade is closed in this simplified version for PNL calc.
                        # A more realistic one might pre-deduct or hold margin.
                        self.logger.info(f"{current_timestamp} - ENTRY {signal} {self.symbol} at {self.current_position_entry_price:.4f}. Qty: {self.current_position_qty}. "
                                         f"SL: {self.current_stop_loss_price:.4f}, TP: {self.current_take_profit_price:.4f}. Current Balance: {self.balance_usdt:.4f}")

        self.logger.info(f"Backtest run completed for {self.symbol}.")

    def calculate_performance(self):
        self.logger.info("\n--- Backtest Performance ---")
        if not self.trades_history:
            self.logger.info("No trades were executed.")
            print("No trades were executed.")
            return {}

        total_trades = len(self.trades_history)
        winning_trades = sum(1 for trade in self.trades_history if trade['net_pnl'] > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        total_net_pnl = sum(trade['net_pnl'] for trade in self.trades_history)
        average_net_pnl_per_trade = total_net_pnl / total_trades if total_trades > 0 else 0
        total_fees_paid = sum(trade['total_fees'] for trade in self.trades_history)
        final_balance = float(self.balance_usdt) # Balance already reflects PNL from closed trades

        results = {
            "initial_capital": float(self.initial_capital_usdt),
            "final_balance": final_balance,
            "total_net_pnl": total_net_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_pnl_per_trade": average_net_pnl_per_trade,
            "total_fees": total_fees_paid
        }
        output_str = "\n--- Backtest Performance Results ---\n"
        for key, value in results.items():
            output_str += f"{key.replace('_', ' ').title()}: {value:.2f if isinstance(value, float) and key != 'win_rate' else value}{'%' if key == 'win_rate' else ''}\n"

        self.logger.info(output_str)
        print(output_str) # Also print to console for easy viewing

        trades_df = pd.DataFrame(self.trades_history)
        if not trades_df.empty:
            try:
                trades_df.to_csv("backtest_trades_history.csv", index=False)
                self.logger.info("Trades history saved to backtest_trades_history.csv")
            except Exception as e:
                self.logger.error(f"Failed to save trades history CSV: {e}")
        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s',
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Simplified Backtester Main Execution ---")

    try:
        # This assumes backtester.py is in the project root, and 'configs' is a subdirectory.
        # load_config itself looks for 'configs/config.dev.json' or 'configs/config.json.example'
        # relative to where it's called from, or uses absolute paths if provided.
        # The default load_config in bybit_gemini_bot.config uses paths relative to its own location if no arg.
        # For robustness, let's construct path from this script's location.

        # Determine project root (where backtester.py is)
        import os # Ensure os is imported here
        project_root_path = os.path.dirname(os.path.abspath(__file__))
        dev_config_file = os.path.join(project_root_path, "configs", "config.dev.json")
        example_config_file = os.path.join(project_root_path, "configs", "config.json.example")

        config_to_load = dev_config_file if os.path.exists(dev_config_file) else example_config_file

        if not os.path.exists(config_to_load):
            logger.error(f"No config file found at {dev_config_file} or {example_config_file}. Exiting.")
            exit()

        logger.info(f"Loading configuration from: {config_to_load}")
        app_config = load_config(config_to_load)

    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        app_config = {} # Ensure app_config exists

    if not app_config:
        logger.error("Failed to load configuration. Exiting backtester.")
        exit()

    app_config.setdefault('taker_fee_rate', 0.00055)
    app_config.setdefault('risk_per_trade_percentage', 1.0)
    app_config.setdefault('stop_loss_percentage', 0.02) # 2% SL
    app_config.setdefault('tp_fee_multiplier', 3.0) # TP target is 3x round-trip fees
    app_config.setdefault('risk_reward_ratio', 1.5) # Fallback R:R if fee-based TP fails

    data_file = "sample_btcusdt_1h.csv"
    try:
        raw_df = pd.read_csv(data_file)
        logger.info(f"Successfully loaded historical data from {data_file}")
    except FileNotFoundError:
        logger.error(f"Error: Data file '{data_file}' not found. Assumed in project root: {os.path.join(project_root_path, data_file)}")
        exit()
    except Exception as e:
        logger.error(f"Error loading data from '{data_file}': {e}")
        exit()

    data_handler = DataHandler(config_logging=False)
    kline_list_from_csv = []
    for index, row in raw_df.iterrows():
        kline_list_from_csv.append([
            str(int(row['timestamp'])),
            str(row['open']), str(row['high']), str(row['low']), str(row['close']),
            str(row['volume']), str(Decimal(str(row['open'])) * Decimal(str(row['volume'])))
        ])

    historical_df_processed = data_handler.process_kline_data(kline_list_from_csv, "BTCUSDT_Backtest")
    if historical_df_processed is None or historical_df_processed.empty:
        logger.error("Failed to process historical K-line data. Exiting.")
        exit()

    historical_df_with_ta = data_handler.calculate_technical_indicators(historical_df_processed, "BTCUSDT_Backtest")
    min_periods_for_signal = 20
    # Ensure enough data for indicators like SMA_20
    if len(historical_df_with_ta) < min_periods_for_signal:
        logger.error(f"Not enough data to form indicators requiring {min_periods_for_signal} periods. Have {len(historical_df_with_ta)}. Exiting.")
        exit()
    historical_df_with_ta.dropna(subset=['SMA_10', 'SMA_20'], inplace=True)

    if historical_df_with_ta.empty:
        logger.error(f"DataFrame became empty after dropping NaNs for indicators from {len(raw_df)} initial rows. Ensure sufficient data for indicator warmup.")
        exit()
    logger.info(f"Historical data processed. Shape after TA and NaN drop: {historical_df_with_ta.shape}")

    sample_instrument_info = {
        "symbol": "BTCUSDT",
        "lotSizeFilter": {"maxOrderQty": "100", "minOrderQty": "0.001", "qtyStep": "0.001"},
        "priceFilter": {"minPrice": "0.50", "maxPrice": "999999.00", "tickSize": "0.50"}
    }

    logger.info("Initializing SimpleBacktester...")
    backtester = SimpleBacktester(
        config=app_config,
        historical_data_df=historical_df_with_ta,
        instrument_info=sample_instrument_info,
        initial_capital_usdt=10000.0,
        symbol="BTCUSDT"
    )

    logger.info("Running backtest...")
    backtester.run()
    logger.info("Calculating performance...")
    backtester.calculate_performance()
    logger.info("--- Simplified Backtester Main Execution Finished ---")
