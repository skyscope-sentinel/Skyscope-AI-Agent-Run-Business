import os
import sys
import json
import time
import uuid
import hmac
import base64
import hashlib
import logging
import datetime
import threading
import traceback
import requests
import websocket
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import asyncio
import aiohttp
from decimal import Decimal, ROUND_DOWN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Import internal modules
try:
    from database_manager import DatabaseManager
    from agent_manager import AgentManager
    from performance_monitor import PerformanceMonitor
    from live_thinking_rag_system import LiveThinkingRAGSystem
    from enhanced_security_compliance import EncryptionManager
except ImportError:
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/crypto_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_crypto_trading_system")

# Constants
CONFIG_DIR = Path("config")
MODELS_DIR = Path("models")
DATA_DIR = Path("data")
BACKTEST_DIR = Path("backtest_results")
STRATEGIES_DIR = Path("strategies")
KEYS_DIR = Path("keys")
LOGS_DIR = Path("logs")
REPORTS_DIR = Path("reports")

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)
KEYS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "crypto_trading_config.json"

class TradingMode(Enum):
    """Trading modes."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

class TimeFrame(Enum):
    """Candlestick timeframes."""
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MN1 = "1M"

class StrategyType(Enum):
    """Strategy types."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    SENTIMENT = "sentiment"
    ML_PREDICTION = "ml_prediction"
    DEEP_LEARNING = "deep_learning"
    GRID_TRADING = "grid_trading"
    SCALPING = "scalping"
    CUSTOM = "custom"

class RiskLevel(Enum):
    """Risk levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CUSTOM = "custom"

class ExchangeName(Enum):
    """Supported cryptocurrency exchanges."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BITFINEX = "bitfinex"
    BITSTAMP = "bitstamp"
    HUOBI = "huobi"
    KUCOIN = "kucoin"
    BYBIT = "bybit"
    FTX = "ftx"
    DERIBIT = "deribit"
    BITMEX = "bitmex"
    OKEX = "okex"
    CUSTOM = "custom"

@dataclass
class TradingConfig:
    """Configuration for the trading system."""
    mode: TradingMode = TradingMode.PAPER
    base_currency: str = "USDT"
    quote_currencies: List[str] = field(default_factory=lambda: ["BTC", "ETH", "SOL", "BNB", "XRP"])
    exchanges: List[ExchangeName] = field(default_factory=lambda: [ExchangeName.BINANCE])
    default_timeframe: TimeFrame = TimeFrame.H1
    strategy_types: List[StrategyType] = field(default_factory=lambda: [StrategyType.TREND_FOLLOWING, StrategyType.MOMENTUM])
    risk_level: RiskLevel = RiskLevel.MEDIUM
    max_open_positions: int = 5
    position_size_pct: float = 0.05  # 5% of portfolio per position
    stop_loss_pct: float = 0.03  # 3% stop loss
    take_profit_pct: float = 0.06  # 6% take profit
    trailing_stop_pct: Optional[float] = None
    max_daily_drawdown_pct: float = 0.10  # 10% max daily drawdown
    use_ai_predictions: bool = True
    ai_confidence_threshold: float = 0.65
    rebalance_frequency_days: int = 7
    data_update_interval_seconds: int = 60
    trading_hours: Dict[str, List[str]] = field(default_factory=lambda: {
        "monday": ["00:00-23:59"],
        "tuesday": ["00:00-23:59"],
        "wednesday": ["00:00-23:59"],
        "thursday": ["00:00-23:59"],
        "friday": ["00:00-23:59"],
        "saturday": ["00:00-23:59"],
        "sunday": ["00:00-23:59"]
    })
    api_keys: Dict[str, Dict[str, str]] = field(default_factory=dict)
    backtest_start_date: Optional[str] = None
    backtest_end_date: Optional[str] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "base_currency": self.base_currency,
            "quote_currencies": self.quote_currencies,
            "exchanges": [ex.value for ex in self.exchanges],
            "default_timeframe": self.default_timeframe.value,
            "strategy_types": [st.value for st in self.strategy_types],
            "risk_level": self.risk_level.value,
            "max_open_positions": self.max_open_positions,
            "position_size_pct": self.position_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "max_daily_drawdown_pct": self.max_daily_drawdown_pct,
            "use_ai_predictions": self.use_ai_predictions,
            "ai_confidence_threshold": self.ai_confidence_threshold,
            "rebalance_frequency_days": self.rebalance_frequency_days,
            "data_update_interval_seconds": self.data_update_interval_seconds,
            "trading_hours": self.trading_hours,
            "api_keys": self.api_keys,
            "backtest_start_date": self.backtest_start_date,
            "backtest_end_date": self.backtest_end_date,
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingConfig':
        """Create from dictionary."""
        return cls(
            mode=TradingMode(data.get("mode", TradingMode.PAPER.value)),
            base_currency=data.get("base_currency", "USDT"),
            quote_currencies=data.get("quote_currencies", ["BTC", "ETH", "SOL", "BNB", "XRP"]),
            exchanges=[ExchangeName(ex) for ex in data.get("exchanges", [ExchangeName.BINANCE.value])],
            default_timeframe=TimeFrame(data.get("default_timeframe", TimeFrame.H1.value)),
            strategy_types=[StrategyType(st) for st in data.get("strategy_types", [StrategyType.TREND_FOLLOWING.value, StrategyType.MOMENTUM.value])],
            risk_level=RiskLevel(data.get("risk_level", RiskLevel.MEDIUM.value)),
            max_open_positions=data.get("max_open_positions", 5),
            position_size_pct=data.get("position_size_pct", 0.05),
            stop_loss_pct=data.get("stop_loss_pct", 0.03),
            take_profit_pct=data.get("take_profit_pct", 0.06),
            trailing_stop_pct=data.get("trailing_stop_pct"),
            max_daily_drawdown_pct=data.get("max_daily_drawdown_pct", 0.10),
            use_ai_predictions=data.get("use_ai_predictions", True),
            ai_confidence_threshold=data.get("ai_confidence_threshold", 0.65),
            rebalance_frequency_days=data.get("rebalance_frequency_days", 7),
            data_update_interval_seconds=data.get("data_update_interval_seconds", 60),
            trading_hours=data.get("trading_hours", {
                "monday": ["00:00-23:59"],
                "tuesday": ["00:00-23:59"],
                "wednesday": ["00:00-23:59"],
                "thursday": ["00:00-23:59"],
                "friday": ["00:00-23:59"],
                "saturday": ["00:00-23:59"],
                "sunday": ["00:00-23:59"]
            }),
            api_keys=data.get("api_keys", {}),
            backtest_start_date=data.get("backtest_start_date"),
            backtest_end_date=data.get("backtest_end_date"),
            custom_settings=data.get("custom_settings", {})
        )
    
    def save(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to file."""
        try:
            # Ensure sensitive data is not saved to disk
            config_dict = self.to_dict()
            
            # Replace API keys with placeholders if they exist
            if "api_keys" in config_dict:
                for exchange in config_dict["api_keys"]:
                    if "api_key" in config_dict["api_keys"][exchange]:
                        config_dict["api_keys"][exchange]["api_key"] = "***"
                    if "api_secret" in config_dict["api_keys"][exchange]:
                        config_dict["api_keys"][exchange]["api_secret"] = "***"
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Trading configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving trading configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = DEFAULT_CONFIG_PATH) -> 'TradingConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Trading configuration loaded from {filepath}")
                
                # Load API keys from secure storage
                config = cls.from_dict(data)
                config.api_keys = cls._load_api_keys()
                
                return config
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading trading configuration: {e}")
            return cls()
    
    @staticmethod
    def _load_api_keys() -> Dict[str, Dict[str, str]]:
        """Load API keys from secure storage."""
        api_keys = {}
        try:
            # Try to load from encrypted storage
            keys_file = KEYS_DIR / "crypto_api_keys.json"
            if keys_file.exists():
                try:
                    # Check if we have access to the encryption manager
                    encryption_manager = None
                    try:
                        from enhanced_security_compliance import EncryptionManager
                        encryption_manager = EncryptionManager()
                    except ImportError:
                        pass
                    
                    if encryption_manager:
                        # Decrypt the file
                        decrypted_path = encryption_manager.decrypt_file(keys_file)
                        with open(decrypted_path, 'r') as f:
                            api_keys = json.load(f)
                        # Remove the decrypted file
                        os.remove(decrypted_path)
                    else:
                        # Fallback to unencrypted file
                        with open(keys_file, 'r') as f:
                            api_keys = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading API keys: {e}")
            
            # Check environment variables as a backup
            for exchange in ExchangeName:
                env_prefix = exchange.value.upper()
                api_key_env = f"{env_prefix}_API_KEY"
                api_secret_env = f"{env_prefix}_API_SECRET"
                
                if api_key_env in os.environ and api_secret_env in os.environ:
                    if exchange.value not in api_keys:
                        api_keys[exchange.value] = {}
                    
                    api_keys[exchange.value]["api_key"] = os.environ[api_key_env]
                    api_keys[exchange.value]["api_secret"] = os.environ[api_secret_env]
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
        
        return api_keys

@dataclass
class Candle:
    """Candlestick data."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candle':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"]
        )

@dataclass
class Trade:
    """Trade information."""
    id: str
    symbol: str
    exchange: ExchangeName
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    timestamp: int
    status: str
    fee: float = 0.0
    fee_currency: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "timestamp": self.timestamp,
            "status": self.status,
            "fee": self.fee,
            "fee_currency": self.fee_currency,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_id": self.strategy_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            exchange=ExchangeName(data["exchange"]),
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            quantity=data["quantity"],
            price=data["price"],
            timestamp=data["timestamp"],
            status=data["status"],
            fee=data.get("fee", 0.0),
            fee_currency=data.get("fee_currency", ""),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            strategy_id=data.get("strategy_id")
        )

@dataclass
class Position:
    """Trading position."""
    id: str
    symbol: str
    exchange: ExchangeName
    side: OrderSide
    entry_price: float
    current_price: float
    quantity: float
    timestamp: int
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: Optional[str] = None
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def value(self) -> float:
        """Calculate the current value of the position."""
        return self.quantity * self.current_price
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate the profit/loss percentage."""
        if self.side == OrderSide.BUY:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "timestamp": self.timestamp,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_id": self.strategy_id,
            "trades": [t.to_dict() for t in self.trades],
            "value": self.value,
            "pnl_percentage": self.pnl_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            symbol=data["symbol"],
            exchange=ExchangeName(data["exchange"]),
            side=OrderSide(data["side"]),
            entry_price=data["entry_price"],
            current_price=data["current_price"],
            quantity=data["quantity"],
            timestamp=data["timestamp"],
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            strategy_id=data.get("strategy_id"),
            trades=[Trade.from_dict(t) for t in data.get("trades", [])]
        )

@dataclass
class Portfolio:
    """Trading portfolio."""
    id: str
    name: str
    base_currency: str
    total_value: float
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    creation_date: int = field(default_factory=lambda: int(time.time()))
    last_updated: int = field(default_factory=lambda: int(time.time()))
    
    @property
    def total_pnl(self) -> float:
        """Calculate the total profit/loss."""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_pnl_percentage(self) -> float:
        """Calculate the total profit/loss percentage."""
        initial_value = self.total_value - self.total_pnl
        if initial_value == 0:
            return 0.0
        return (self.total_pnl / initial_value) * 100
    
    @property
    def position_value(self) -> float:
        """Calculate the total value of all positions."""
        return sum(position.value for position in self.positions.values())
    
    @property
    def position_allocation(self) -> Dict[str, float]:
        """Calculate the allocation percentage for each position."""
        allocations = {}
        for symbol, position in self.positions.items():
            allocations[symbol] = (position.value / self.total_value) * 100 if self.total_value > 0 else 0.0
        return allocations
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "base_currency": self.base_currency,
            "total_value": self.total_value,
            "cash": self.cash,
            "positions": {symbol: position.to_dict() for symbol, position in self.positions.items()},
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "creation_date": self.creation_date,
            "last_updated": self.last_updated,
            "total_pnl": self.total_pnl,
            "total_pnl_percentage": self.total_pnl_percentage,
            "position_value": self.position_value,
            "position_allocation": self.position_allocation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Create from dictionary."""
        positions = {}
        for symbol, position_data in data.get("positions", {}).items():
            positions[symbol] = Position.from_dict(position_data)
        
        return cls(
            id=data["id"],
            name=data["name"],
            base_currency=data["base_currency"],
            total_value=data["total_value"],
            cash=data["cash"],
            positions=positions,
            realized_pnl=data.get("realized_pnl", 0.0),
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            creation_date=data.get("creation_date", int(time.time())),
            last_updated=data.get("last_updated", int(time.time()))
        )

@dataclass
class TradingSignal:
    """Trading signal generated by a strategy."""
    id: str
    strategy_id: str
    symbol: str
    side: OrderSide
    signal_type: str
    timestamp: int
    price: float
    confidence: float
    timeframe: TimeFrame
    indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    expiration: Optional[int] = None
    
    def is_valid(self) -> bool:
        """Check if the signal is still valid."""
        if self.expiration is None:
            return True
        return int(time.time()) < self.expiration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "price": self.price,
            "confidence": self.confidence,
            "timeframe": self.timeframe.value,
            "indicators": self.indicators,
            "metadata": self.metadata,
            "expiration": self.expiration
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            signal_type=data["signal_type"],
            timestamp=data["timestamp"],
            price=data["price"],
            confidence=data["confidence"],
            timeframe=TimeFrame(data["timeframe"]),
            indicators=data.get("indicators", {}),
            metadata=data.get("metadata", {}),
            expiration=data.get("expiration")
        )

@dataclass
class StrategyPerformance:
    """Performance metrics for a trading strategy."""
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_holding_time: float = 0.0
    
    def update(self, trade: Trade, pnl: float, holding_time: float) -> None:
        """Update performance metrics with a new trade."""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
            self.avg_profit = ((self.avg_profit * (self.winning_trades - 1)) + pnl) / self.winning_trades if self.winning_trades > 0 else 0
        else:
            self.losing_trades += 1
            self.avg_loss = ((self.avg_loss * (self.losing_trades - 1)) + pnl) / self.losing_trades if self.losing_trades > 0 else 0
        
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        self.avg_holding_time = ((self.avg_holding_time * (self.total_trades - 1)) + holding_time) / self.total_trades if self.total_trades > 0 else 0
        
        # Profit factor and expectancy
        if self.losing_trades > 0 and self.avg_loss != 0:
            self.profit_factor = (self.winning_trades * self.avg_profit) / (self.losing_trades * abs(self.avg_loss))
            self.expectancy = (self.win_rate * self.avg_profit) - ((1 - self.win_rate) * abs(self.avg_loss))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy_id": self.strategy_id,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_profit": self.avg_profit,
            "avg_loss": self.avg_loss,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_holding_time": self.avg_holding_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyPerformance':
        """Create from dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            total_pnl=data.get("total_pnl", 0.0),
            win_rate=data.get("win_rate", 0.0),
            avg_profit=data.get("avg_profit", 0.0),
            avg_loss=data.get("avg_loss", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            sortino_ratio=data.get("sortino_ratio", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            expectancy=data.get("expectancy", 0.0),
            avg_holding_time=data.get("avg_holding_time", 0.0)
        )

class TradingStrategy(ABC):
    """Base class for trading strategies."""
    
    def __init__(self, strategy_id: str, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 parameters: Dict[str, Any] = None):
        """Initialize the strategy."""
        self.strategy_id = strategy_id
        self.name = name
        self.symbols = symbols
        self.timeframes = timeframes
        self.parameters = parameters or {}
        self.performance = StrategyPerformance(strategy_id)
        self.signals: List[TradingSignal] = []
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        pass
    
    def save(self, filepath: Optional[Path] = None) -> Path:
        """Save strategy to file."""
        if filepath is None:
            filepath = STRATEGIES_DIR / f"{self.strategy_id}.json"
        
        try:
            strategy_data = {
                "strategy_id": self.strategy_id,
                "name": self.name,
                "type": self.__class__.__name__,
                "symbols": self.symbols,
                "timeframes": [tf.value for tf in self.timeframes],
                "parameters": self.parameters,
                "performance": self.performance.to_dict()
            }
            
            with open(filepath, 'w') as f:
                json.dump(strategy_data, f, indent=2)
            
            logger.info(f"Strategy saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            raise
    
    @classmethod
    def load(cls, filepath: Path) -> 'TradingStrategy':
        """Load strategy from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            strategy_type = data.get("type")
            strategy_class = globals().get(strategy_type)
            
            if not strategy_class or not issubclass(strategy_class, TradingStrategy):
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            strategy = strategy_class(
                strategy_id=data["strategy_id"],
                name=data["name"],
                symbols=data["symbols"],
                timeframes=[TimeFrame(tf) for tf in data["timeframes"]],
                parameters=data.get("parameters", {})
            )
            
            if "performance" in data:
                strategy.performance = StrategyPerformance.from_dict(data["performance"])
            
            return strategy
        except Exception as e:
            logger.error(f"Error loading strategy: {e}")
            raise

class TrendFollowingStrategy(TradingStrategy):
    """Trend following strategy using moving averages."""
    
    def __init__(self, strategy_id: str, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 parameters: Dict[str, Any] = None):
        """Initialize the strategy."""
        super().__init__(strategy_id, name, symbols, timeframes, parameters)
        
        # Set default parameters if not provided
        if not self.parameters:
            self.parameters = {
                "fast_ma_period": 20,
                "slow_ma_period": 50,
                "signal_threshold": 0.0,
                "ma_type": "ema"  # ema, sma, wma
            }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        signals = []
        
        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue
            
            # Calculate moving averages
            fast_period = self.parameters["fast_ma_period"]
            slow_period = self.parameters["slow_ma_period"]
            ma_type = self.parameters["ma_type"]
            
            if ma_type == "ema":
                fast_ma = df['close'].ewm(span=fast_period, adjust=False).mean()
                slow_ma = df['close'].ewm(span=slow_period, adjust=False).mean()
            elif ma_type == "wma":
                # Weighted moving average
                weights_fast = np.arange(1, fast_period + 1)
                weights_slow = np.arange(1, slow_period + 1)
                fast_ma = df['close'].rolling(window=fast_period).apply(lambda x: np.sum(weights_fast * x) / weights_fast.sum(), raw=True)
                slow_ma = df['close'].rolling(window=slow_period).apply(lambda x: np.sum(weights_slow * x) / weights_slow.sum(), raw=True)
            else:  # Default to SMA
                fast_ma = df['close'].rolling(window=fast_period).mean()
                slow_ma = df['close'].rolling(window=slow_period).mean()
            
            # Calculate crossovers
            df['fast_ma'] = fast_ma
            df['slow_ma'] = slow_ma
            df['ma_diff'] = fast_ma - slow_ma
            df['ma_diff_prev'] = df['ma_diff'].shift(1)
            
            # Generate signals for crossovers
            for i in range(1, len(df)):
                # Skip if we don't have enough data
                if pd.isna(df['ma_diff_prev'].iloc[i]):
                    continue
                
                # Check for bullish crossover (fast MA crosses above slow MA)
                if df['ma_diff_prev'].iloc[i] <= self.parameters["signal_threshold"] and df['ma_diff'].iloc[i] > self.parameters["signal_threshold"]:
                    signal = TradingSignal(
                        id=str(uuid.uuid4()),
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        signal_type="ma_crossover_bullish",
                        timestamp=int(df.index[i].timestamp()),
                        price=df['close'].iloc[i],
                        confidence=min(1.0, abs(df['ma_diff'].iloc[i]) / df['close'].iloc[i] * 100),  # Confidence based on MA difference
                        timeframe=self.timeframes[0],  # Use the first timeframe in the list
                        indicators={
                            "fast_ma": df['fast_ma'].iloc[i],
                            "slow_ma": df['slow_ma'].iloc[i],
                            "ma_diff": df['ma_diff'].iloc[i]
                        }
                    )
                    signals.append(signal)
                
                # Check for bearish crossover (fast MA crosses below slow MA)
                elif df['ma_diff_prev'].iloc[i] >= self.parameters["signal_threshold"] and df['ma_diff'].iloc[i] < self.parameters["signal_threshold"]:
                    signal = TradingSignal(
                        id=str(uuid.uuid4()),
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        signal_type="ma_crossover_bearish",
                        timestamp=int(df.index[i].timestamp()),
                        price=df['close'].iloc[i],
                        confidence=min(1.0, abs(df['ma_diff'].iloc[i]) / df['close'].iloc[i] * 100),  # Confidence based on MA difference
                        timeframe=self.timeframes[0],  # Use the first timeframe in the list
                        indicators={
                            "fast_ma": df['fast_ma'].iloc[i],
                            "slow_ma": df['slow_ma'].iloc[i],
                            "ma_diff": df['ma_diff'].iloc[i]
                        }
                    )
                    signals.append(signal)
        
        # Store and return signals
        self.signals = signals
        return signals

class MomentumStrategy(TradingStrategy):
    """Momentum trading strategy using RSI and MACD."""
    
    def __init__(self, strategy_id: str, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 parameters: Dict[str, Any] = None):
        """Initialize the strategy."""
        super().__init__(strategy_id, name, symbols, timeframes, parameters)
        
        # Set default parameters if not provided
        if not self.parameters:
            self.parameters = {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "use_macd": True,
                "use_rsi": True
            }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        signals = []
        
        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue
            
            # Calculate RSI
            if self.parameters["use_rsi"]:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=self.parameters["rsi_period"]).mean()
                avg_loss = loss.rolling(window=self.parameters["rsi_period"]).mean()
                
                rs = avg_gain / avg_loss
                df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            if self.parameters["use_macd"]:
                fast_ema = df['close'].ewm(span=self.parameters["macd_fast"], adjust=False).mean()
                slow_ema = df['close'].ewm(span=self.parameters["macd_slow"], adjust=False).mean()
                df['macd'] = fast_ema - slow_ema
                df['macd_signal'] = df['macd'].ewm(span=self.parameters["macd_signal"], adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                df['macd_hist_prev'] = df['macd_hist'].shift(1)
            
            # Generate signals
            for i in range(1, len(df)):
                confidence = 0.0
                signal_type = ""
                side = None
                
                # RSI signals
                if self.parameters["use_rsi"] and not pd.isna(df['rsi'].iloc[i]):
                    if df['rsi'].iloc[i] < self.parameters["rsi_oversold"]:
                        # Oversold condition - potential buy
                        rsi_confidence = (self.parameters["rsi_oversold"] - df['rsi'].iloc[i]) / self.parameters["rsi_oversold"]
                        confidence = max(confidence, rsi_confidence)
                        signal_type = "rsi_oversold"
                        side = OrderSide.BUY
                    elif df['rsi'].iloc[i] > self.parameters["rsi_overbought"]:
                        # Overbought condition - potential sell
                        rsi_confidence = (df['rsi'].iloc[i] - self.parameters["rsi_overbought"]) / (100 - self.parameters["rsi_overbought"])
                        confidence = max(confidence, rsi_confidence)
                        signal_type = "rsi_overbought"
                        side = OrderSide.SELL
                
                # MACD signals
                if self.parameters["use_macd"] and not pd.isna(df['macd_hist'].iloc[i]) and not pd.isna(df['macd_hist_prev'].iloc[i]):
                    # Bullish MACD crossover
                    if df['macd_hist_prev'].iloc[i] <= 0 and df['macd_hist'].iloc[i] > 0:
                        macd_confidence = min(1.0, abs(df['macd_hist'].iloc[i]) / df['close'].iloc[i] * 100)
                        
                        # If RSI already gave a signal, combine them
                        if side == OrderSide.BUY:
                            confidence = (confidence + macd_confidence) / 2
                            signal_type += "_macd_cross"
                        else:
                            confidence = macd_confidence
                            signal_type = "macd_bullish_cross"
                            side = OrderSide.BUY
                    
                    # Bearish MACD crossover
                    elif df['macd_hist_prev'].iloc[i] >= 0 and df['macd_hist'].iloc[i] < 0:
                        macd_confidence = min(1.0, abs(df['macd_hist'].iloc[i]) / df['close'].iloc[i] * 100)
                        
                        # If RSI already gave a signal, combine them
                        if side == OrderSide.SELL:
                            confidence = (confidence + macd_confidence) / 2
                            signal_type += "_macd_cross"
                        else:
                            confidence = macd_confidence
                            signal_type = "macd_bearish_cross"
                            side = OrderSide.SELL
                
                # Create signal if we have one
                if side is not None:
                    indicators = {}
                    if self.parameters["use_rsi"]:
                        indicators["rsi"] = df['rsi'].iloc[i]
                    if self.parameters["use_macd"]:
                        indicators["macd"] = df['macd'].iloc[i]
                        indicators["macd_signal"] = df['macd_signal'].iloc[i]
                        indicators["macd_hist"] = df['macd_hist'].iloc[i]
                    
                    signal = TradingSignal(
                        id=str(uuid.uuid4()),
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        side=side,
                        signal_type=signal_type,
                        timestamp=int(df.index[i].timestamp()),
                        price=df['close'].iloc[i],
                        confidence=confidence,
                        timeframe=self.timeframes[0],  # Use the first timeframe in the list
                        indicators=indicators
                    )
                    signals.append(signal)
        
        # Store and return signals
        self.signals = signals
        return signals

class MLPredictionStrategy(TradingStrategy):
    """Machine learning based prediction strategy."""
    
    def __init__(self, strategy_id: str, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 parameters: Dict[str, Any] = None):
        """Initialize the strategy."""
        super().__init__(strategy_id, name, symbols, timeframes, parameters)
        
        # Set default parameters if not provided
        if not self.parameters:
            self.parameters = {
                "model_type": "random_forest",  # random_forest, gradient_boosting, lstm
                "prediction_horizon": 3,  # Number of candles to predict ahead
                "training_window": 500,  # Number of candles to use for training
                "feature_window": 20,  # Number of past candles to use as features
                "retrain_frequency": 100,  # Retrain model every N candles
                "confidence_threshold": 0.6,  # Minimum confidence to generate a signal
                "price_change_threshold": 0.01  # Minimum predicted price change to generate a signal
            }
        
        self.models = {}
        self.last_trained = {}
        self.scalers = {}
    
    def _prepare_features(self, df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML model."""
        feature_window = self.parameters["feature_window"]
        prediction_horizon = self.parameters["prediction_horizon"]
        
        # Create features from price and volume data
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=feature_window).std()
        
        # Add technical indicators
        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Create lagged features
        for i in range(1, feature_window + 1):
            df[f'close_lag_{i}'] = df['close'].shift(i)
            df[f'volume_lag_{i}'] = df['volume'].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        if len(df) <= feature_window + prediction_horizon:
            return np.array([]), np.array([])
        
        # Select features
        feature_columns = [
            'returns', 'log_returns', 'volatility',
            'sma_10', 'sma_20', 'ema_10', 'ema_20',
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_width'
        ]
        
        # Add lagged features
        for i in range(1, feature_window + 1):
            feature_columns.extend([f'close_lag_{i}', f'volume_lag_{i}', f'returns_lag_{i}'])
        
        # Create target: future price change
        df['target'] = df['close'].shift(-prediction_horizon) / df['close'] - 1
        
        # Split into features and target
        X = df[feature_columns].values
        y = df['target'].values
        
        # Scale features
        if symbol not in self.scalers:
            self.scalers[symbol] = MinMaxScaler()
            X_scaled = self.scalers[symbol].fit_transform(X)
        else:
            X_scaled = self.scalers[symbol].transform(X)
        
        return X_scaled[:-prediction_horizon], y[:-prediction_horizon]
    
    def _train_model(self, symbol: str, df: pd.DataFrame) -> None:
        """Train a machine learning model for price prediction."""
        model_type = self.parameters["model_type"]
        
        # Prepare features and target
        X, y = self._prepare_features(df, symbol)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"Not enough data to train model for {symbol}")
            return
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model based on type
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train, y_train)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
        elif model_type == "lstm":
            # Reshape data for LSTM [samples, time steps, features]
            feature_dim = X_train.shape[1]
            X_train_lstm = X_train.reshape((X_train.shape[0], 1, feature_dim))
            X_val_lstm = X_val.reshape((X_val.shape[0], 1, feature_dim))
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(1, feature_dim)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(
                X_train_lstm, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val_lstm, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            return
        
        # Store the model
        self.models[symbol] = model
        self.last_trained[symbol] = len(df)
        
        # Save model
        model_dir = MODELS_DIR / "trading" / symbol
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if model_type == "lstm":
            model_path = model_dir / f"{self.strategy_id}_lstm.h5"
            model.save(str(model_path))
        else:
            model_path = model_dir / f"{self.strategy_id}_{model_type}.pkl"
            with open(model_path, 'wb') as f:
                import pickle
                pickle.dump(model, f)
        
        logger.info(f"Trained and saved {model_type} model for {symbol}")
    
    def _predict_price_change(self, symbol: str, df: pd.DataFrame) -> Tuple[float, float]:
        """Predict future price change and confidence."""
        model_type = self.parameters["model_type"]
        
        # Check if we need to train or retrain the model
        if symbol not in self.models or symbol not in self.last_trained:
            self._train_model(symbol, df)
        elif len(df) - self.last_trained[symbol] >= self.parameters["retrain_frequency"]:
            self._train_model(symbol, df)
        
        # If we still don't have a model, return no prediction
        if symbol not in self.models:
            return 0.0, 0.0
        
        # Prepare features for prediction
        X, _ = self._prepare_features(df, symbol)
        
        if len(X) == 0:
            return 0.0, 0.0
        
        # Make prediction based on model type
        if model_type == "lstm":
            feature_dim = X.shape[1]
            X_lstm = X[-1].reshape((1, 1, feature_dim))
            prediction = self.models[symbol].predict(X_lstm, verbose=0)[0][0]
        else:
            prediction = self.models[symbol].predict([X[-1]])[0]
        
        # Calculate confidence
        if model_type == "random_forest":
            # For random forest, use the standard deviation of tree predictions
            predictions = np.array([tree.predict([X[-1]])[0] for tree in self.models[symbol].estimators_])
            confidence = 1.0 - min(1.0, np.std(predictions) / abs(prediction) if prediction != 0 else 1.0)
        elif model_type == "gradient_boosting":
            # For gradient boosting, confidence is harder to estimate, use a placeholder
            confidence = 0.7
        else:
            # For LSTM, use a placeholder confidence
            confidence = 0.7
        
        return prediction, confidence
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        signals = []
        
        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue
            
            # Make prediction
            predicted_change, confidence = self._predict_price_change(symbol, df)
            
            # Generate signal if prediction is significant and confidence is high enough
            if abs(predicted_change) >= self.parameters["price_change_threshold"] and confidence >= self.parameters["confidence_threshold"]:
                side = OrderSide.BUY if predicted_change > 0 else OrderSide.SELL
                
                signal = TradingSignal(
                    id=str(uuid.uuid4()),
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side=side,
                    signal_type="ml_prediction",
                    timestamp=int(df.index[-1].timestamp()),
                    price=df['close'].iloc[-1],
                    confidence=confidence,
                    timeframe=self.timeframes[0],  # Use the first timeframe in the list
                    indicators={
                        "predicted_change": predicted_change,
                        "prediction_horizon": self.parameters["prediction_horizon"]
                    }
                )
                signals.append(signal)
        
        # Store and return signals
        self.signals = signals
        return signals

class DeepLearningStrategy(TradingStrategy):
    """Deep learning strategy using LSTM networks."""
    
    def __init__(self, strategy_id: str, name: str, symbols: List[str], timeframes: List[TimeFrame], 
                 parameters: Dict[str, Any] = None):
        """Initialize the strategy."""
        super().__init__(strategy_id, name, symbols, timeframes, parameters)
        
        # Set default parameters if not provided
        if not self.parameters:
            self.parameters = {
                "sequence_length": 60,  # Number of time steps to use as input
                "prediction_horizon": 5,  # Number of time steps to predict ahead
                "lstm_units": 50,  # Number of LSTM units
                "dropout_rate": 0.2,  # Dropout rate
                "learning_rate": 0.001,  # Learning rate
                "batch_size": 32,  # Batch size for training
                "epochs": 50,  # Number of training epochs
                "train_test_split": 0.8,  # Ratio of training data
                "retrain_frequency": 1000,  # Retrain model every N candles
                "confidence_threshold": 0.65,  # Minimum confidence to generate a signal
                "price_change_threshold": 0.01  # Minimum predicted price change to generate a signal
            }
        
        self.models = {}
        self.scalers = {}
        self.last_trained = {}
    
    def _prepare_data(self, df: pd.DataFrame, symbol: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model."""
        sequence_length = self.parameters["sequence_length"]
        prediction_horizon = self.parameters["prediction_horizon"]
        
        # Extract features
        features = df[['open', 'high', 'low', 'close', 'volume']].values
        
        # Scale features
        if symbol not in self.scalers:
            self.scalers[symbol] = {}
            self.scalers[symbol]['features'] = MinMaxScaler()
            features_scaled = self.scalers[symbol]['features'].fit_transform(features)
            
            # Also create a scaler just for close prices (for inverse transform later)
            self.scalers[symbol]['close'] = MinMaxScaler()
            self.scalers[symbol]['close'].fit_transform(df[['close']].values)
        else:
            features_scaled = self.scalers[symbol]['features'].transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - sequence_length - prediction_horizon):
            X.append(features_scaled[i:i+sequence_length])
            # Target is the close price 'prediction_horizon' steps ahead
            y.append(features_scaled[i+sequence_length+prediction_horizon, 3])  # 3 is the index of close price
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM model."""
        lstm_units = self.parameters["lstm_units"]
        dropout_rate = self.parameters["dropout_rate"]
        learning_rate = self.parameters["learning_rate"]
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        x = LSTM(lstm_units, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(lstm_units)(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1)(x)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        
        return model
    
    def _train_model(self, symbol: str, df: pd.DataFrame) -> None:
        """Train LSTM model."""
        # Prepare data
        X, y = self._prepare_data(df, symbol)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning(f"Not enough data to train model for {symbol}")
            return
        
        # Split into training and validation sets
        split_idx = int(len(X) * self.parameters["train_test_split"])
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        model = self._build_model((X.shape[1], X.shape[2]))
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            filepath=str(MODELS_DIR / "trading" / symbol / f"{self.strategy_id}_best_model.h5"),
            monitor='val_loss',
            save_best_only=True
        )
        
        model.fit(
            X_train, y_train,
            epochs=self.parameters["epochs"],
            batch_size=self.parameters["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )
        
        # Store model
        self.models[symbol] = model
        self.last_trained[symbol] = len(df)
        
        # Save model
        model_dir = MODELS_DIR / "trading" / symbol
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{self.strategy_id}_lstm.h5"
        model.save(str(model_path))
        
        logger.info(f"Trained and saved LSTM model for {symbol}")
    
    def _predict(self, symbol: str, df: pd.DataFrame) -> Tuple[float, float]:
        """Make prediction using LSTM model."""
        # Check if we need to train or retrain the model
        if symbol not in self.models or symbol not in self.last_trained:
            self._train_model(symbol, df)
        elif len(df) - self.last_trained[symbol] >= self.parameters["retrain_frequency"]:
            self._train_model(symbol, df)
        
        # If we still don't have a model, return no prediction
        if symbol not in self.models:
            return 0.0, 0.0
        
        # Prepare input data
        sequence_length = self.parameters["sequence_length"]
        if len(df) < sequence_length:
            return 0.0, 0.0
        
        # Extract and scale features
        features = df[['open', 'high', 'low', 'close', 'volume']].values[-sequence_length:]
        features_scaled = self.scalers[symbol]['features'].transform(features)
        
        # Make prediction
        X = np.array([features_scaled])
        prediction_scaled = self.models[symbol].predict(X, verbose=0)[0][0]
        
        # Convert prediction back to original scale
        prediction = self.scalers[symbol]['close'].inverse_transform([[prediction_scaled]])[0][0]
        current_price = df['close'].iloc[-1]
        
        # Calculate predicted price change
        price_change = (prediction - current_price) / current_price
        
        # Calculate confidence (placeholder)
        confidence = 0.7
        
        return price_change, confidence
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals from market data."""
        signals = []
        
        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue
            
            # Make prediction
            predicted_change, confidence = self._predict(symbol, df)
            
            # Generate signal if prediction is significant and confidence is high enough
            if abs(predicted_change) >= self.parameters["price_change_threshold"] and confidence >= self.parameters["confidence_threshold"]:
                side = OrderSide.BUY if predicted_change > 0 else OrderSide.SELL
                
                signal = TradingSignal(
                    id=str(uuid.uuid4()),
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    side=side,
                    signal_type="deep_learning",
                    timestamp=int(df.index[-1].timestamp()),
                    price=df['close'].iloc[-1],
                    confidence=confidence,
                    timeframe=self.timeframes[0],
                    indicators={
                        "predicted_change": predicted_change,
                        "prediction_horizon": self.parameters["prediction_horizon"]
                    }
                )
                signals.append(signal)
        
        # Store and return signals
        self.signals = signals
        return signals

class StrategyFactory:
    """Factory for creating trading strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: StrategyType, strategy_id: str, name: str, symbols: List[str], 
                       timeframes: List[TimeFrame], parameters: Dict[str, Any] = None) -> TradingStrategy:
        """Create a trading strategy."""
        if strategy_type == StrategyType.TREND_FOLLOWING:
            return TrendFollowingStrategy(strategy_id, name, symbols, timeframes, parameters)
        elif strategy_type == StrategyType.MOMENTUM:
            return MomentumStrategy(strategy_id, name, symbols, timeframes, parameters)
        elif strategy_type == StrategyType.ML_PREDICTION:
            return MLPredictionStrategy(strategy_id, name, symbols, timeframes, parameters)
        elif strategy_type == StrategyType.DEEP_LEARNING:
            return DeepLearningStrategy(strategy_id, name, symbols, timeframes, parameters)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

class ExchangeConnector(ABC):
    """Base class for cryptocurrency exchange connectors."""
    
    def __init__(self, exchange_name: ExchangeName, api_key: str = None, api_secret: str = None):
        """Initialize the exchange connector."""
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.base_url = ""
        self.ws_url = ""
        self.ws_connection = None
        self.ws_thread = None
        self.running = False
        self.data_callbacks = []
    
    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        pass
    
    @abstractmethod
    def get_candles(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[Candle]:
        """Get historical candlestick data."""
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book data."""
        pass
    
    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        pass
    
    @abstractmethod
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                   quantity: float, price: float = None) -> Dict[str, Any]:
        """Place a new order."""
        pass
    
    @abstractmethod
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order."""
        pass
    
    @abstractmethod
    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order details."""
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        pass
    
    def start_websocket(self, symbols: List[str], channels: List[str], callback: Callable) -> None:
        """Start websocket connection."""
        if self.ws_thread and self.ws_thread.is_alive():
            return
        
        self.running = True
        self.data_callbacks.append(callback)
        
        def _websocket_thread():
            while self.running:
                try:
                    self._connect_websocket(symbols, channels)
                except Exception as e:
                    logger.error(f"Websocket error: {e}")
                    time.sleep(5)  # Wait before reconnecting
        
        self.ws_thread = threading.Thread(target=_websocket_thread)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def stop_websocket(self) -> None:
        """Stop websocket connection."""
        self.running = False
        if self.ws_connection:
            self.ws_connection.close()
    
    @abstractmethod
    def _connect_websocket(self, symbols: List[str], channels: List[str]) -> None:
        """Connect to websocket."""
        pass
    
    def _on_websocket_message(self, message: str) -> None:
        """Handle websocket message."""
        try:
            data = json.loads(message)
            for callback in self.data_callbacks:
                callback(data)
        except Exception as e:
            logger.error(f"Error processing websocket message: {e}")

class BinanceConnector(ExchangeConnector):
    """Binance exchange connector."""
    
    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """Initialize the Binance connector."""
        super().__init__(ExchangeName.BINANCE, api_key, api_secret)
        
        if testnet:
            self.base_url = "https://testnet.binance.vision/api"
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            self.base_url = "https://api.binance.com/api"
            self.ws_url = "wss://stream.binance.com:9443/ws"
    
    def _get_timestamp(self) -> int:
        """Get current timestamp in milliseconds."""
        return int(time.time() * 1000)
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate HMAC SHA256 signature."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _send_request(self, method: str, endpoint: str, params: Dict[str, Any] = None, 
                     signed: bool = False) -> Dict[str, Any]:
        """Send request to Binance API."""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        if signed and self.api_secret:
            if params is None:
                params = {}
            
            params['timestamp'] = self._get_timestamp()
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers)
            elif method == 'POST':
                response = self.session.post(url, params=params, headers=headers)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data."""
        return self._send_request('GET', '/v3/ticker/price', {'symbol': symbol})
    
    def get_candles(self, symbol: str, timeframe: TimeFrame, limit: int = 100) -> List[Candle]:
        """Get historical candlestick data."""
        params = {
            'symbol': symbol,
            'interval': timeframe.value,
            'limit': limit
        }
        
        response = self._send_request('GET', '/v3/klines', params)
        
        candles = []
        for kline in response:
            candle = Candle(
                timestamp=kline[0],  # Open time
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5])
            )
            candles.append(candle)
        
        return candles
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book data."""
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        return self._send_request('GET', '/v3/depth', params)
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance."""
        response = self._send_request('GET', '/v3/account', signed=True)
        
        balances = {}
        for asset in response['balances']:
            free = float(asset['free'])
            locked = float(asset['locked'])
            total = free + locked
            
            if total > 0:
                balances[asset['asset']] = {
                    'free': free,
                    'locked': locked,
                    'total': total
                }
        
        return balances
    
    def place_order(self, symbol: str, side: OrderSide, order_type: OrderType, 
                   quantity: float, price: float = None) -> Dict[str, Any]:
        """Place a new order."""
        params = {
            'symbol': symbol,
            'side': side.value.upper(),
            'type': order_type.value.upper(),
            'quantity': quantity,
            'newOrderRespType': 'FULL'
        }
        
        if order_type != OrderType.MARKET and price is not None:
            params['price'] = price
        
        if order_type == OrderType.STOP_LOSS and price is not None:
            params['stopPrice'] = price
        
        return self._send_request('POST', '/v3/order', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an existing order."""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        response = self._send_request('DELETE', '/v3/order', params, signed=True)
        
