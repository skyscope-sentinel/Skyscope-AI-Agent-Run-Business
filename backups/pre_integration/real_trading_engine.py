#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real Trading Engine
==================

This module handles real cryptocurrency trading on actual exchanges.
It connects to real exchanges and executes actual trades with real money.

⚠️ WARNING: This handles real money. Use with extreme caution.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import ccxt
import requests

logger = logging.getLogger('RealTradingEngine')

class RealTradingEngine:
    """Handles real cryptocurrency trading on exchanges"""
    
    def __init__(self, config_file: str = "config/trading_config.json"):
        self.config_file = config_file
        self.exchanges = {}
        self.trading_enabled = False
        self.min_trade_amount = 10.0  # Minimum $10 trades
        self.max_trade_amount = 100.0  # Maximum $100 trades per trade
        self.daily_trade_limit = 1000.0  # Maximum $1000 trades per day
        self.daily_trades_total = 0.0
        
        # Load configuration
        self._load_config()
        
        # Initialize exchanges
        self._initialize_exchanges()
    
    def _load_config(self):
        """Load trading configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Create default config
            self.config = {
                "trading_enabled": False,
                "exchanges": {
                    "binance": {
                        "api_key": "",
                        "api_secret": "",
                        "sandbox": True,
                        "enabled": False
                    },
                    "coinbase": {
                        "api_key": "",
                        "api_secret": "",
                        "passphrase": "",
                        "sandbox": True,
                        "enabled": False
                    },
                    "kraken": {
                        "api_key": "",
                        "api_secret": "",
                        "sandbox": True,
                        "enabled": False
                    }
                },
                "trading_pairs": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
                "risk_management": {
                    "max_position_size": 0.1,  # 10% of portfolio
                    "stop_loss_percentage": 0.02,  # 2% stop loss
                    "take_profit_percentage": 0.05,  # 5% take profit
                    "max_daily_trades": 50
                }
            }
            self._save_config()
    
    def _save_config(self):
        """Save trading configuration"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _initialize_exchanges(self):
        """Initialize exchange connections"""
        for exchange_name, exchange_config in self.config["exchanges"].items():
            if not exchange_config.get("enabled", False):
                continue
            
            try:
                if exchange_name == "binance":
                    exchange = ccxt.binance({
                        'apiKey': exchange_config["api_key"],
                        'secret': exchange_config["api_secret"],
                        'sandbox': exchange_config.get("sandbox", True),
                        'enableRateLimit': True,
                    })
                elif exchange_name == "coinbase":
                    exchange = ccxt.coinbasepro({
                        'apiKey': exchange_config["api_key"],
                        'secret': exchange_config["api_secret"],
                        'passphrase': exchange_config["passphrase"],
                        'sandbox': exchange_config.get("sandbox", True),
                        'enableRateLimit': True,
                    })
                elif exchange_name == "kraken":
                    exchange = ccxt.kraken({
                        'apiKey': exchange_config["api_key"],
                        'secret': exchange_config["api_secret"],
                        'sandbox': exchange_config.get("sandbox", True),
                        'enableRateLimit': True,
                    })
                else:
                    continue
                
                # Test connection
                exchange.load_markets()
                self.exchanges[exchange_name] = exchange
                logger.info(f"Connected to {exchange_name} exchange")
                
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_name}: {e}")
    
    def enable_trading(self, enable: bool = True):
        """Enable or disable real trading"""
        self.trading_enabled = enable
        self.config["trading_enabled"] = enable
        self._save_config()
        
        if enable:
            logger.warning("🚨 REAL TRADING ENABLED - This will use real money!")
        else:
            logger.info("Real trading disabled - simulation mode")
    
    def get_account_balance(self, exchange_name: str) -> Dict:
        """Get account balance from exchange"""
        if exchange_name not in self.exchanges:
            return {}
        
        try:
            exchange = self.exchanges[exchange_name]
            balance = exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance from {exchange_name}: {e}")
            return {}
    
    def get_market_price(self, symbol: str, exchange_name: str = None) -> Optional[float]:
        """Get current market price for a symbol"""
        if not self.exchanges:
            return None
        
        # Use first available exchange if none specified
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            return None
        
        try:
            exchange = self.exchanges[exchange_name]
            ticker = exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Failed to get price for {symbol} from {exchange_name}: {e}")
            return None
    
    def execute_trade(self, symbol: str, side: str, amount: float, 
                     exchange_name: str = None) -> Optional[Dict]:
        """Execute a real trade on the exchange"""
        
        # Safety checks
        if not self.trading_enabled:
            logger.info(f"Trading disabled - would execute {side} {amount} {symbol}")
            return self._simulate_trade(symbol, side, amount)
        
        if amount < self.min_trade_amount:
            logger.warning(f"Trade amount ${amount} below minimum ${self.min_trade_amount}")
            return None
        
        if amount > self.max_trade_amount:
            logger.warning(f"Trade amount ${amount} above maximum ${self.max_trade_amount}")
            return None
        
        if self.daily_trades_total + amount > self.daily_trade_limit:
            logger.warning(f"Daily trade limit ${self.daily_trade_limit} would be exceeded")
            return None
        
        # Use first available exchange if none specified
        if exchange_name is None and self.exchanges:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange {exchange_name} not available")
            return None
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Execute market order
            order = exchange.create_market_order(symbol, side, amount)
            
            # Update daily total
            self.daily_trades_total += amount
            
            logger.info(f"✅ Executed real trade: {side} {amount} {symbol} on {exchange_name}")
            logger.info(f"Order ID: {order['id']}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute trade on {exchange_name}: {e}")
            return None
    
    def _simulate_trade(self, symbol: str, side: str, amount: float) -> Dict:
        """Simulate a trade for testing purposes"""
        price = self.get_market_price(symbol) or 50000.0  # Default BTC price
        
        return {
            'id': f"sim_{int(time.time())}",
            'symbol': symbol,
            'side': side,
            'amount': amount,
            'price': price,
            'cost': amount * price if side == 'buy' else amount,
            'timestamp': datetime.now().isoformat(),
            'status': 'closed',
            'simulated': True
        }
    
    def analyze_market_opportunity(self, symbol: str) -> Dict:
        """Analyze market for trading opportunities"""
        try:
            # Get price data from multiple exchanges
            prices = {}
            for exchange_name in self.exchanges:
                price = self.get_market_price(symbol, exchange_name)
                if price:
                    prices[exchange_name] = price
            
            if len(prices) < 2:
                return {"opportunity": False, "reason": "Insufficient price data"}
            
            # Find arbitrage opportunities
            min_price = min(prices.values())
            max_price = max(prices.values())
            price_diff = max_price - min_price
            price_diff_percent = (price_diff / min_price) * 100
            
            # Consider it an opportunity if price difference > 0.5%
            if price_diff_percent > 0.5:
                return {
                    "opportunity": True,
                    "type": "arbitrage",
                    "symbol": symbol,
                    "price_difference": price_diff,
                    "price_difference_percent": price_diff_percent,
                    "buy_exchange": min(prices, key=prices.get),
                    "sell_exchange": max(prices, key=prices.get),
                    "buy_price": min_price,
                    "sell_price": max_price,
                    "potential_profit": price_diff
                }
            
            return {"opportunity": False, "reason": "No significant arbitrage opportunity"}
            
        except Exception as e:
            logger.error(f"Failed to analyze market opportunity: {e}")
            return {"opportunity": False, "reason": str(e)}
    
    def get_trading_status(self) -> Dict:
        """Get current trading status"""
        return {
            "trading_enabled": self.trading_enabled,
            "exchanges_connected": len(self.exchanges),
            "exchanges": list(self.exchanges.keys()),
            "daily_trades_total": self.daily_trades_total,
            "daily_trade_limit": self.daily_trade_limit,
            "remaining_daily_limit": self.daily_trade_limit - self.daily_trades_total
        }
    
    def setup_api_keys(self, exchange_name: str, api_credentials: Dict):
        """Set up API keys for an exchange"""
        if exchange_name not in self.config["exchanges"]:
            self.config["exchanges"][exchange_name] = {}
        
        self.config["exchanges"][exchange_name].update(api_credentials)
        self.config["exchanges"][exchange_name]["enabled"] = True
        self._save_config()
        
        # Reinitialize exchanges
        self._initialize_exchanges()
        
        logger.info(f"Updated API credentials for {exchange_name}")

def setup_real_trading():
    """Set up real trading engine"""
    print("⚡ Setting up Real Trading Engine")
    print("=" * 40)
    
    trading_engine = RealTradingEngine()
    
    print("📊 Trading Status:")
    status = trading_engine.get_trading_status()
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print("\n⚠️  IMPORTANT TRADING SETUP:")
    print("  1. Add your exchange API keys to config/trading_config.json")
    print("  2. Start with sandbox/testnet mode")
    print("  3. Enable real trading only when ready")
    print("  4. Monitor trades carefully")
    
    print(f"\n📁 Configuration file: {trading_engine.config_file}")
    
    return trading_engine

if __name__ == "__main__":
    setup_real_trading()