import logging
import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP

class RiskManager:
    """
    Manages risk by calculating position sizes, stop-loss/take-profit levels, and fees.
    """
    def __init__(self, config: dict, config_logging: bool = True):
        """
        Initializes the RiskManager.

        Args:
            config (dict): Configuration dictionary containing risk parameters like
                           'risk_per_trade_percentage'.
            config_logging (bool, optional): Whether to configure basic logging. Defaults to True.
        """
        self.logger = logging.getLogger(__name__)
        if config_logging and not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')
            self.logger.info("Basic logging configured for RiskManager standalone usage.")

        self.config = config
        if 'risk_per_trade_percentage' not in self.config:
            self.logger.warning("'risk_per_trade_percentage' not found in config. Using default 1.0%.")
            self.config['risk_per_trade_percentage'] = 1.0

        self.logger.info(f"RiskManager initialized. Risk per trade: {self.config['risk_per_trade_percentage']}%")

    def _get_qty_precision(self, qty_step_str: str) -> int:
        """Determines the number of decimal places for quantity based on qtyStep."""
        if '.' in qty_step_str:
            return len(qty_step_str.split('.')[1])
        return 0 # No decimal places if qty_step is "1" or similar

    def _get_price_precision(self, tick_size_str: str) -> int:
        """Determines the number of decimal places for price based on tickSize."""
        if '.' in tick_size_str:
            return len(tick_size_str.split('.')[1])
        return 0

    def _adjust_to_step(self, value: Decimal, step_str: str, rounding_mode=ROUND_DOWN) -> Decimal:
        """Adjusts a value to comply with a given step size."""
        step = Decimal(step_str)
        return (value / step).quantize(Decimal('1'), rounding=rounding_mode) * step

    def calculate_position_size(self, available_balance_usdt: float, entry_price: float,
                                stop_loss_price: float, instrument_info: dict,
                                symbol: str, side: str = "Buy") -> str | None:
        """
        Calculates the position size in base currency, adjusted for instrument rules.

        Args:
            available_balance_usdt (float): Available trading capital in USDT.
            entry_price (float): Proposed entry price.
            stop_loss_price (float): Proposed stop-loss price.
            instrument_info (dict): From Bybit's get_instruments_info (V5 API).
                                    Example: {'symbol': 'BTCUSDT', ...,
                                              'lotSizeFilter': {'maxOrderQty': '100', 'minOrderQty': '0.001', 'qtyStep': '0.001'},
                                              'priceFilter': {'minPrice': '0.5', 'maxPrice': '999999', 'tickSize': '0.5'}}
            symbol (str): The trading symbol (e.g., "BTCUSDT").
            side (str): "Buy" for long, "Sell" for short. Default is "Buy".

        Returns:
            str | None: Adjusted quantity as a string, or None if calculation is not possible.
        """
        self.logger.info(f"Calculating position size for {symbol} ({side}): Balance={available_balance_usdt}, Entry={entry_price}, SL={stop_loss_price}")

        if entry_price <= 0 or stop_loss_price <= 0:
            self.logger.error("Entry price and stop-loss price must be positive.")
            return None

        if side == "Buy":
            if entry_price <= stop_loss_price:
                self.logger.error(f"For a BUY order, entry price ({entry_price}) must be greater than stop loss ({stop_loss_price}).")
                return None
            price_diff_per_unit = Decimal(str(entry_price)) - Decimal(str(stop_loss_price))
        elif side == "Sell":
            if entry_price >= stop_loss_price:
                self.logger.error(f"For a SELL order, entry price ({entry_price}) must be less than stop loss ({stop_loss_price}).")
                return None
            price_diff_per_unit = Decimal(str(stop_loss_price)) - Decimal(str(entry_price))
        else:
            self.logger.error(f"Invalid side '{side}'. Must be 'Buy' or 'Sell'.")
            return None

        if price_diff_per_unit <= Decimal('0'): # Should be caught by above checks too
            self.logger.error(f"Price difference for SL calculation is zero or negative for {symbol}, cannot size position.")
            return None

        risk_percentage = Decimal(str(self.config.get('risk_per_trade_percentage', 1.0))) / Decimal('100')
        risk_amount_usdt = Decimal(str(available_balance_usdt)) * risk_percentage
        self.logger.info(f"Risk per trade: {self.config['risk_per_trade_percentage']}%, Risk amount (USDT): {risk_amount_usdt:.4f}")

        # Initial quantity in base currency
        calculated_qty_base = risk_amount_usdt / price_diff_per_unit
        self.logger.info(f"Calculated initial quantity for {symbol}: {calculated_qty_base:.8f}")

        # Extract instrument rules (handle potential missing keys gracefully)
        lot_size_filter = instrument_info.get('lotSizeFilter')
        if not lot_size_filter:
            self.logger.error(f"Missing 'lotSizeFilter' in instrument_info for {symbol}. Cannot adjust quantity.")
            return None

        min_order_qty_str = lot_size_filter.get('minOrderQty')
        max_order_qty_str = lot_size_filter.get('maxOrderQty')
        qty_step_str = lot_size_filter.get('qtyStep')

        if not all([min_order_qty_str, max_order_qty_str, qty_step_str]):
            self.logger.error(f"Missing one or more fields (minOrderQty, maxOrderQty, qtyStep) in lotSizeFilter for {symbol}.")
            return None

        min_order_qty = Decimal(min_order_qty_str)
        max_order_qty = Decimal(max_order_qty_str)
        # qty_step = Decimal(qty_step_str) # Handled by _adjust_to_step

        # Adjust quantity based on instrument rules
        adjusted_qty = self._adjust_to_step(calculated_qty_base, qty_step_str, ROUND_DOWN)
        self.logger.info(f"Quantity for {symbol} after applying qtyStep ('{qty_step_str}'): {adjusted_qty:.8f}")

        if adjusted_qty < min_order_qty:
            self.logger.warning(f"Adjusted quantity {adjusted_qty} for {symbol} is below minOrderQty {min_order_qty}. Cannot place order with this risk/SL setup.")
            # Option: could try to set to min_order_qty if affordable, but that changes risk profile.
            # For now, returning None if calculated risk-based qty is too small.
            return None

        if adjusted_qty > max_order_qty:
            self.logger.warning(f"Adjusted quantity {adjusted_qty} for {symbol} exceeds maxOrderQty {max_order_qty}. Capping at maxOrderQty.")
            adjusted_qty = max_order_qty
            # Recalculate risk if capped by maxOrderQty (optional, for info)
            actual_risk_usdt = adjusted_qty * price_diff_per_unit
            actual_risk_percent = (actual_risk_usdt / Decimal(str(available_balance_usdt))) * Decimal('100')
            self.logger.info(f"Quantity capped at maxOrderQty. Actual risk: {actual_risk_usdt:.2f} USDT ({actual_risk_percent:.2f}% of balance).")


        # Check if the order value is too small based on minOrderValue (if applicable, e.g. for spot, not typically for linear futures qty)
        # Bybit linear futures usually don't have a separate minOrderValue, it's implied by minOrderQty * price.
        # If symbol is spot, one might need to check: e.g. if 'minOrderValue' in lot_size_filter or a similar field exists.
        # For now, assuming linear futures where minOrderQty is the primary constraint.

        qty_precision = self._get_qty_precision(qty_step_str)
        final_qty_str = f"{adjusted_qty:.{qty_precision}f}"

        self.logger.info(f"Final adjusted position size for {symbol}: {final_qty_str}")
        return final_qty_str

    def calculate_fees_usdt(self, order_qty_base: float, execution_price_usdt: float,
                            fee_rate: float, is_futures: bool = True) -> float:
        """
        Calculates estimated trading fee in USDT for one side of a trade.

        Args:
            order_qty_base (float): Quantity of the order in base currency.
            execution_price_usdt (float): Price at which the order is expected to execute.
            fee_rate (float): The fee rate (e.g., 0.00055 for 0.055%).
            is_futures (bool): True for futures (fee on notional value), False for spot
                               (can vary, here assuming fee on quote value for simplicity if spot).

        Returns:
            float: Estimated fee in USDT.
        """
        order_qty_base_d = Decimal(str(order_qty_base))
        execution_price_usdt_d = Decimal(str(execution_price_usdt))
        fee_rate_d = Decimal(str(fee_rate))

        if is_futures:
            # Fee = OrderQuantity x ExecutionPrice x FeeRate (for linear futures)
            fee = order_qty_base_d * execution_price_usdt_d * fee_rate_d
        else: # Spot (example: fee paid in quote currency)
            # This can vary based on exchange and fee settings (paid in base or quote)
            # Assuming fee is on the quote currency value for simplicity.
            fee = order_qty_base_d * execution_price_usdt_d * fee_rate_d

        self.logger.debug(f"Calculated fee: {float(fee):.8f} USDT for Qty={order_qty_base}, Price={execution_price_usdt}, Rate={fee_rate}")
        return float(fee)

    def calculate_take_profit_price_for_fee_coverage(self, entry_price: float, side: str,
                                               order_qty_base: float, fee_rate_taker: float,
                                               instrument_info: dict, symbol: str,
                                               fee_multiplier: float = 3.0) -> str | None:
        """
        Calculates a take-profit price that aims to cover round-trip fees by a multiplier.

        Args:
            entry_price (float): The entry price of the position.
            side (str): "Buy" (for long) or "Sell" (for short).
            order_qty_base (float): Quantity of the base asset for the position.
            fee_rate_taker (float): Taker fee rate (e.g., 0.00075 for Bybit's 0.075% taker).
            instrument_info (dict): Containing `priceFilter.tickSize`.
            symbol (str): The trading symbol.
            fee_multiplier (float): How many times the round-trip fee to target as profit. Default 3x.

        Returns:
            str | None: Adjusted take-profit price as a string, or None if inputs are invalid.
        """
        self.logger.info(f"Calculating TP for {fee_multiplier}x fee coverage for {symbol} ({side}): Entry={entry_price}, Qty={order_qty_base}, FeeRate={fee_rate_taker}")

        if order_qty_base <= 0:
            self.logger.error("Order quantity must be positive to calculate TP for fee coverage.")
            return None

        entry_price_d = Decimal(str(entry_price))
        order_qty_base_d = Decimal(str(order_qty_base))

        # Calculate fee for one side (entry)
        entry_fee_usdt = Decimal(str(self.calculate_fees_usdt(order_qty_base, entry_price, fee_rate_taker, is_futures=True)))
        # Assume exit fee is similar (worst case, exit at same price for fee calc, or slightly adjusted)
        # For simplicity, using entry_price for exit_fee calculation as well. A more precise calc might use expected exit.
        exit_fee_usdt = Decimal(str(self.calculate_fees_usdt(order_qty_base, entry_price, fee_rate_taker, is_futures=True)))

        total_round_trip_fees_usdt = entry_fee_usdt + exit_fee_usdt
        self.logger.info(f"Estimated entry fee: {entry_fee_usdt:.4f}, Estimated exit fee: {exit_fee_usdt:.4f}, Total RT fees: {total_round_trip_fees_usdt:.4f} USDT for {symbol}")

        target_profit_usdt = total_round_trip_fees_usdt * Decimal(str(fee_multiplier))
        self.logger.info(f"Target profit for {fee_multiplier}x fee coverage: {target_profit_usdt:.4f} USDT for {symbol}")

        # Calculate target exit price
        if side.lower() == "buy": # Long position
            price_change_needed = target_profit_usdt / order_qty_base_d
            target_exit_price_d = entry_price_d + price_change_needed
        elif side.lower() == "sell": # Short position
            price_change_needed = target_profit_usdt / order_qty_base_d
            target_exit_price_d = entry_price_d - price_change_needed
        else:
            self.logger.error(f"Invalid side '{side}' for TP calculation.")
            return None

        self.logger.info(f"Calculated raw target exit price for {symbol}: {target_exit_price_d:.8f}")

        # Adjust TP to comply with tickSize
        price_filter = instrument_info.get('priceFilter')
        if not price_filter or 'tickSize' not in price_filter:
            self.logger.error(f"Missing 'priceFilter' or 'tickSize' in instrument_info for {symbol}. Cannot adjust TP price.")
            return None

        tick_size_str = price_filter['tickSize']

        # For TP, round towards more profit: UP for LONG, DOWN for SHORT
        rounding_tp = ROUND_UP if side.lower() == "buy" else ROUND_DOWN
        adjusted_tp_price_d = self._adjust_to_step(target_exit_price_d, tick_size_str, rounding_mode=rounding_tp)

        price_precision = self._get_price_precision(tick_size_str)
        final_tp_price_str = f"{adjusted_tp_price_d:.{price_precision}f}"

        self.logger.info(f"Final adjusted TP price for {symbol} ({side}) aiming for {fee_multiplier}x fee coverage: {final_tp_price_str} (TickSize: {tick_size_str})")
        return final_tp_price_str


if __name__ == '__main__':
    print("--- Running RiskManager Standalone Example ---")

    dummy_config = {
        "risk_per_trade_percentage": 1.5, # 1.5% risk per trade
        # other config params if needed by RM in future
    }

    # Example instrument_info for BTCUSDT (Linear Perpetual) - values are illustrative
    dummy_instrument_info_btc = {
        "symbol": "BTCUSDT",
        "lotSizeFilter": {
            "maxOrderQty": "100",       # Max quantity of BTC
            "minOrderQty": "0.001",     # Min quantity of BTC
            "qtyStep": "0.001"          # Quantity step (precision for BTC)
        },
        "priceFilter": {
            "minPrice": "0.50",         # Min price in USDT
            "maxPrice": "999999.00",    # Max price in USDT
            "tickSize": "0.50"          # Price step (precision for USDT) for BTCUSDT is usually 0.5 or 0.1
        }
    }
    dummy_instrument_info_eth = {
        "symbol": "ETHUSDT",
        "lotSizeFilter": {"maxOrderQty": "1000", "minOrderQty": "0.01", "qtyStep": "0.01"},
        "priceFilter": {"minPrice": "0.05", "maxPrice": "99999.00", "tickSize": "0.05"}
    }

    risk_manager = RiskManager(config=dummy_config)

    print("\n--- Testing calculate_position_size ---")
    # Test case 1: BTCUSDT Long
    pos_size_btc_long = risk_manager.calculate_position_size(
        available_balance_usdt=1000.0,
        entry_price=30000.0,
        stop_loss_price=29700.0, # 300 USDT risk per unit of BTC
        instrument_info=dummy_instrument_info_btc,
        symbol="BTCUSDT",
        side="Buy"
    )
    print(f"BTCUSDT Long Position Size: {pos_size_btc_long}") # Expected risk: 1000 * 1.5% = 15 USDT. Qty = 15 / 300 = 0.05 BTC. Adjusted: 0.050

    # Test case 2: ETHUSDT Short
    pos_size_eth_short = risk_manager.calculate_position_size(
        available_balance_usdt=1000.0,
        entry_price=1800.0,
        stop_loss_price=1830.0, # 30 USDT risk per unit of ETH
        instrument_info=dummy_instrument_info_eth,
        symbol="ETHUSDT",
        side="Sell"
    )
    print(f"ETHUSDT Short Position Size: {pos_size_eth_short}") # Expected risk: 15 USDT. Qty = 15 / 30 = 0.5 ETH. Adjusted: 0.50

    # Test case 3: Quantity too small
    pos_size_too_small = risk_manager.calculate_position_size(
        available_balance_usdt=100.0, # Smaller balance
        entry_price=30000.0,
        stop_loss_price=29990.0, # 10 USDT risk per unit
        instrument_info=dummy_instrument_info_btc,
        symbol="BTCUSDT",
        side="Buy"
    )
    print(f"BTCUSDT Long Position Size (Too Small Risk/Diff): {pos_size_too_small}") # Risk: 1.5 USDT. Qty = 1.5 / 10 = 0.15. Min is 0.001. This should work.
                                                                               # Let's try an even smaller one.
    pos_size_actually_too_small = risk_manager.calculate_position_size(
        available_balance_usdt=10.0, # Very small balance
        entry_price=30000.0,
        stop_loss_price=29999.0, # 1 USDT risk per unit
        instrument_info=dummy_instrument_info_btc,
        symbol="BTCUSDT",
        side="Buy"
    )
    print(f"BTCUSDT Long Position Size (Actually Too Small): {pos_size_actually_too_small}") # Risk: 0.15 USDT. Qty = 0.15 / 1 = 0.15. Min is 0.001. This should be fine.
                                                                                         # The "too small" case is if adjusted_qty < min_order_qty.
                                                                                         # If risk = 0.0001 USDT, qty = 0.0001 / 1 = 0.0001. Adjusted to 0.000. This would be None.

    pos_size_calc_for_min_test = risk_manager.calculate_position_size(
        available_balance_usdt=1.0, # 1 USDT balance
        entry_price=30000.0,
        stop_loss_price=29999.9, # 0.1 USDT risk per unit
        instrument_info=dummy_instrument_info_btc, # minQty 0.001
        symbol="BTCUSDT",
        side="Buy"
    ) # Risk: 1 * 0.015 = 0.015 USDT. Qty = 0.015 / 0.1 = 0.15. Should be fine.
      # Let's make SL very close to trigger minQty issue
    pos_size_calc_for_min_test_tight = risk_manager.calculate_position_size(
        available_balance_usdt=100.0, # 100 USDT balance
        entry_price=30000.0,
        stop_loss_price=29000.0, # Risk per unit: 1000 USDT
        instrument_info=dummy_instrument_info_btc, # minQty 0.001
        symbol="BTCUSDT",
        side="Buy"
    ) # Risk: 100 * 0.015 = 1.5 USDT. Qty = 1.5 / 1000 = 0.0015. Adjusted to 0.001. This should be 0.001.
    print(f"BTCUSDT Long Position Size (Triggering Min Qty): {pos_size_calc_for_min_test_tight}")


    # Test case 4: Invalid SL
    pos_size_invalid_sl = risk_manager.calculate_position_size(
        available_balance_usdt=1000.0,
        entry_price=30000.0,
        stop_loss_price=30000.0, # SL same as entry
        instrument_info=dummy_instrument_info_btc,
        symbol="BTCUSDT",
        side="Buy"
    )
    print(f"BTCUSDT Long Position Size (Invalid SL): {pos_size_invalid_sl}")


    print("\n--- Testing calculate_take_profit_price_for_fee_coverage ---")
    # Bybit taker fee rate for non-VIP futures is often 0.055% (0.00055) or 0.075%
    # Let's use a higher one for more visible impact, e.g. 0.075%
    bybit_taker_fee_rate = 0.00075

    # Test case 1: BTCUSDT Long, 3x fee coverage
    tp_price_btc_long = risk_manager.calculate_take_profit_price_for_fee_coverage(
        entry_price=30000.0,
        side="Buy",
        order_qty_base=0.050, # From previous calculation
        fee_rate_taker=bybit_taker_fee_rate,
        instrument_info=dummy_instrument_info_btc,
        symbol="BTCUSDT",
        fee_multiplier=3.0
    )
    # Entry fee: 0.050 * 30000 * 0.00075 = 1.125 USDT
    # Exit fee: (similar) 1.125 USDT
    # Total RT fees: 2.25 USDT
    # Target profit: 3 * 2.25 = 6.75 USDT
    # Price change needed: 6.75 / 0.050 = 135 USDT
    # Target TP: 30000 + 135 = 30135. Adjusted by tickSize (0.50): 30135.00
    print(f"BTCUSDT Long TP for 3x Fees: {tp_price_btc_long}")

    # Test case 2: ETHUSDT Short, 2x fee coverage
    tp_price_eth_short = risk_manager.calculate_take_profit_price_for_fee_coverage(
        entry_price=1800.0,
        side="Sell",
        order_qty_base=0.50, # From previous calculation
        fee_rate_taker=bybit_taker_fee_rate,
        instrument_info=dummy_instrument_info_eth, # tickSize 0.05
        symbol="ETHUSDT",
        fee_multiplier=2.0
    )
    # Entry fee: 0.50 * 1800 * 0.00075 = 0.675 USDT
    # Exit fee: 0.675 USDT
    # Total RT fees: 1.35 USDT
    # Target profit: 2 * 1.35 = 2.70 USDT
    # Price change needed: 2.70 / 0.50 = 5.4 USDT
    # Target TP: 1800 - 5.4 = 1794.6. Adjusted by tickSize (0.05), ROUND_DOWN for short: 1794.60
    print(f"ETHUSDT Short TP for 2x Fees: {tp_price_eth_short}")

    print("\n--- RiskManager Standalone Example Finished ---")
