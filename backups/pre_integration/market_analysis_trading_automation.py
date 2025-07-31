#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Market Analysis and Trading Automation
=========================================================================

This module provides comprehensive market analysis and trading automation capabilities
for the Skyscope Sentinel Intelligence AI platform, enabling sophisticated crypto
trading strategies, portfolio optimization, and DeFi yield maximization across
multiple blockchains and exchanges.

Features:
- Multi-agent framework with specialized agents (Statistics, Fact, Subjectivity, Trading)
- Real-time market data analysis across multiple exchanges
- Cross-chain trading capabilities (Ethereum, BSC, Polygon, Avalanche, Solana)
- DeFi protocol integration for yield optimization and liquidity provision
- Advanced technical indicators and sentiment analysis
- Risk management with position sizing and stop-loss mechanisms
- Machine learning models for price prediction and pattern recognition
- Backtesting framework for strategy validation
- Portfolio optimization algorithms using modern portfolio theory
- Arbitrage opportunity detection across exchanges and chains
- Market making strategies with dynamic spread adjustment
- Integration with GPT-4o via openai-unofficial for advanced market analysis
- High-frequency trading capabilities with latency optimization
- Order book analysis and liquidity detection

Dependencies:
- pandas, numpy, scipy
- ccxt (for exchange integration)
- web3 (for blockchain integration)
- sklearn, tensorflow, pytorch (for ML models)
- openai-unofficial (for GPT-4o integration)
- ta (for technical indicators)
- nltk, textblob (for sentiment analysis)
- matplotlib, plotly (for visualization)
"""

import os
import time
import json
import uuid
import hmac
import hashlib
import logging
import asyncio
import threading
import traceback
import websockets
import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from abc import ABC, abstractmethod

# Data analysis and ML imports
try:
    import ta
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    ML_AVAILABLE = True
    logging.info("Machine learning libraries loaded successfully")
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Some machine learning libraries not available. ML features will be limited.")

# Exchange API imports
try:
    import ccxt
    import ccxt.async_support as ccxt_async
    CCXT_AVAILABLE = True
    logging.info("CCXT loaded successfully")
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available. Exchange integration will be limited.")

# Blockchain integration imports
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    WEB3_AVAILABLE = True
    logging.info("Web3 loaded successfully")
except ImportError:
    WEB3_AVAILABLE = False
    logging.warning("Web3 not available. Blockchain integration will be limited.")

# NLP and sentiment analysis imports
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    nltk.download('vader_lexicon', quiet=True)
    NLP_AVAILABLE = True
    logging.info("NLP libraries loaded successfully")
except ImportError:
    NLP_AVAILABLE = False
    logging.warning("NLP libraries not available. Sentiment analysis will be limited.")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VISUALIZATION_AVAILABLE = True
    logging.info("Visualization libraries loaded successfully")
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available. Charting will be limited.")

# OpenAI unofficial for GPT-4o integration
try:
    import openai_unofficial
    OPENAI_UNOFFICIAL_AVAILABLE = True
    logging.info("openai-unofficial loaded successfully")
except ImportError:
    try:
        import openai
        OPENAI_AVAILABLE = True
        OPENAI_UNOFFICIAL_AVAILABLE = False
        logging.info("Standard OpenAI loaded as fallback")
    except ImportError:
        OPENAI_AVAILABLE = False
        OPENAI_UNOFFICIAL_AVAILABLE = False
        logging.warning("OpenAI libraries not available. GPT-4o integration will be disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/market_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("market_analysis")

# --- Constants and Enumerations ---

class ExchangeType(Enum):
    """Types of exchanges supported by the system."""
    SPOT = auto()
    FUTURES = auto()
    MARGIN = auto()
    OPTION = auto()
    DEX = auto()

class BlockchainType(Enum):
    """Types of blockchains supported by the system."""
    ETHEREUM = "ethereum"
    BITCOIN = "bitcoin"
    BINANCE_SMART_CHAIN = "bsc"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    SOLANA = "solana"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BASE = "base"

class DeFiProtocolType(Enum):
    """Types of DeFi protocols supported by the system."""
    LENDING = auto()
    DEX = auto()
    YIELD_FARMING = auto()
    LIQUIDITY_MINING = auto()
    STAKING = auto()
    DERIVATIVES = auto()
    INSURANCE = auto()
    ASSET_MANAGEMENT = auto()

class TimeFrame(Enum):
    """Standard timeframes for market data."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class OrderType(Enum):
    """Types of orders supported by the system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSide(Enum):
    """Order sides (buy/sell)."""
    BUY = "buy"
    SELL = "sell"

class PositionType(Enum):
    """Types of positions (long/short)."""
    LONG = "long"
    SHORT = "short"

class TradingStrategy(Enum):
    """Trading strategies supported by the system."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    SENTIMENT_BASED = "sentiment_based"
    ML_PREDICTION = "ml_prediction"
    GRID_TRADING = "grid_trading"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MULTI_FACTOR = "multi_factor"

class RiskLevel(Enum):
    """Risk levels for trading strategies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# --- Data Structures ---

@dataclass
class MarketData:
    """Structure for market data."""
    symbol: str
    exchange: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrderBookData:
    """Structure for order book data."""
    symbol: str
    exchange: str
    timestamp: int
    bids: List[Tuple[float, float]]  # price, amount
    asks: List[Tuple[float, float]]  # price, amount
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeData:
    """Structure for trade data."""
    symbol: str
    exchange: str
    timestamp: int
    price: float
    amount: float
    side: OrderSide
    trade_id: str
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Order:
    """Structure for an order."""
    order_id: str
    symbol: str
    exchange: str
    order_type: OrderType
    side: OrderSide
    price: float = None
    amount: float = None
    filled_amount: float = 0.0
    status: str = "new"
    timestamp: int = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Structure for a position."""
    position_id: str
    symbol: str
    exchange: str
    position_type: PositionType
    entry_price: float
    current_price: float
    amount: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    timestamp: int = None
    stop_loss: float = None
    take_profit: float = None
    liquidation_price: float = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradingSignal:
    """Structure for a trading signal."""
    symbol: str
    exchange: str
    timestamp: int
    signal_type: str
    direction: OrderSide
    strength: float
    timeframe: TimeFrame
    source: str
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Structure for backtesting results."""
    strategy_name: str
    symbol: str
    exchange: str
    start_time: int
    end_time: int
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NewsItem:
    """Structure for news data."""
    title: str
    content: str
    source: str
    url: str
    timestamp: int
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    symbols: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class SentimentAnalysis:
    """Structure for sentiment analysis results."""
    symbol: str
    timestamp: int
    overall_sentiment: float
    news_sentiment: float
    social_sentiment: float
    sources: Dict[str, float] = field(default_factory=dict)
    topics: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

@dataclass
class DeFiProtocol:
    """Structure for DeFi protocol data."""
    name: str
    protocol_type: DeFiProtocolType
    blockchain: BlockchainType
    tvl: float
    apy: float = None
    contract_address: str = None
    url: str = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DeFiPosition:
    """Structure for DeFi position data."""
    position_id: str
    protocol: str
    blockchain: BlockchainType
    wallet_address: str
    asset: str
    amount: float
    apy: float = None
    value_usd: float = None
    timestamp: int = None
    transaction_hash: str = None
    additional_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """Structure for price prediction results."""
    symbol: str
    timestamp: int
    timeframe: TimeFrame
    model_name: str
    prediction_time: int
    predicted_price: float
    confidence: float
    features_used: List[str] = field(default_factory=list)
    additional_data: Dict[str, Any] = field(default_factory=dict)

# --- Agent Base Classes ---

class Agent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: str, name: str):
        """Initialize the agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
        """
        self.agent_id = agent_id
        self.name = name
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.status = "initialized"
        self.logger = logging.getLogger(f"agent.{agent_id}")
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process input data and return results.
        
        Args:
            data: Input data to process
            
        Returns:
            Processing results
        """
        pass
    
    def update_status(self, status: str) -> None:
        """Update the agent's status.
        
        Args:
            status: New status string
        """
        self.status = status
        self.last_active = datetime.now()
        self.logger.debug(f"Status updated to: {status}")
    
    def log_activity(self, activity: str, level: str = "info") -> None:
        """Log an activity performed by the agent.
        
        Args:
            activity: Description of the activity
            level: Log level (debug, info, warning, error, critical)
        """
        self.last_active = datetime.now()
        
        if level == "debug":
            self.logger.debug(activity)
        elif level == "info":
            self.logger.info(activity)
        elif level == "warning":
            self.logger.warning(activity)
        elif level == "error":
            self.logger.error(activity)
        elif level == "critical":
            self.logger.critical(activity)

# --- Specialized Agents ---

class StatisticsAgent(Agent):
    """Agent responsible for statistical analysis of market data."""
    
    def __init__(self, agent_id: str, name: str = "Statistics Agent"):
        """Initialize the statistics agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
        """
        super().__init__(agent_id, name)
        self.data_cache = {}
        self.indicators = {}
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and compute statistical indicators.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Dictionary with computed statistics and indicators
        """
        self.update_status("processing")
        
        try:
            # Extract data
            symbol = data.get("symbol")
            exchange = data.get("exchange")
            timeframe = data.get("timeframe", TimeFrame.H1)
            ohlcv_data = data.get("ohlcv")
            
            if not symbol or not exchange or not ohlcv_data:
                raise ValueError("Missing required data: symbol, exchange, or OHLCV data")
            
            # Convert to pandas DataFrame if needed
            if not isinstance(ohlcv_data, pd.DataFrame):
                df = pd.DataFrame(
                    ohlcv_data,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
            else:
                df = ohlcv_data.copy()
            
            # Cache key
            cache_key = f"{exchange}_{symbol}_{timeframe.value}"
            
            # Store in cache
            self.data_cache[cache_key] = df
            
            # Compute basic statistics
            stats = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe.value,
                "data_points": len(df),
                "last_update": df.index[-1].isoformat(),
                "current_price": df["close"].iloc[-1],
                "daily_change": df["close"].iloc[-1] / df["close"].iloc[-2] - 1 if len(df) > 1 else 0,
                "daily_range": (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["low"].iloc[-1],
                "daily_volume": df["volume"].iloc[-1],
                "avg_volume_10d": df["volume"].tail(10).mean(),
                "price_stats": {
                    "mean": df["close"].mean(),
                    "median": df["close"].median(),
                    "std": df["close"].std(),
                    "min": df["close"].min(),
                    "max": df["close"].max(),
                    "last": df["close"].iloc[-1]
                }
            }
            
            # Compute technical indicators if ta library is available
            if "ta" in globals():
                # Add all indicators
                df_with_indicators = self._add_technical_indicators(df)
                
                # Extract the last values of indicators
                indicators = {}
                for col in df_with_indicators.columns:
                    if col not in ["open", "high", "low", "close", "volume"]:
                        indicators[col] = df_with_indicators[col].iloc[-1]
                
                # Store indicators
                self.indicators[cache_key] = indicators
                
                # Add to stats
                stats["indicators"] = indicators
                
                # Add trend analysis
                stats["trend"] = self._analyze_trend(df_with_indicators)
                
                # Add volatility analysis
                stats["volatility"] = self._analyze_volatility(df_with_indicators)
                
                # Add support/resistance levels
                stats["support_resistance"] = self._find_support_resistance(df)
            
            self.update_status("idle")
            self.log_activity(f"Processed statistics for {symbol} on {exchange}")
            
            return stats
        
        except Exception as e:
            self.update_status("error")
            self.log_activity(f"Error processing statistics: {str(e)}", "error")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Add momentum indicators
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df["macd_signal"] = ta.trend.MACD(df["close"]).macd_signal()
        df["macd_diff"] = ta.trend.MACD(df["close"]).macd_diff()
        df["stoch_k"] = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["stoch_d"] = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch_signal()
        
        # Add trend indicators
        df["sma_20"] = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
        df["sma_50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
        df["sma_200"] = ta.trend.SMAIndicator(df["close"], window=200).sma_indicator()
        df["ema_12"] = ta.trend.EMAIndicator(df["close"], window=12).ema_indicator()
        df["ema_26"] = ta.trend.EMAIndicator(df["close"], window=26).ema_indicator()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
        
        # Add volatility indicators
        df["bb_high"] = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
        df["bb_mid"] = ta.volatility.BollingerBands(df["close"]).bollinger_mavg()
        df["bb_low"] = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
        df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
        df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()
        
        # Add volume indicators
        df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
        df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"]).money_flow_index()
        
        return df
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends using technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with trend analysis
        """
        # Get the last row for analysis
        last = df.iloc[-1]
        
        # Determine short-term trend (based on EMA crossover)
        short_term_trend = "bullish" if last["ema_12"] > last["ema_26"] else "bearish"
        
        # Determine medium-term trend (based on price vs SMA 50)
        medium_term_trend = "bullish" if last["close"] > last["sma_50"] else "bearish"
        
        # Determine long-term trend (based on price vs SMA 200)
        long_term_trend = "bullish" if last["close"] > last["sma_200"] else "bearish"
        
        # Determine trend strength (based on ADX)
        adx = last["adx"]
        if adx < 20:
            trend_strength = "weak"
        elif adx < 40:
            trend_strength = "moderate"
        else:
            trend_strength = "strong"
        
        # Determine if we're in a range or trending market
        bb_width = last["bb_width"]
        market_condition = "ranging" if bb_width < 0.1 else "trending"
        
        # Check for potential reversal signals
        reversal_signals = []
        
        # RSI overbought/oversold
        if last["rsi"] > 70:
            reversal_signals.append("RSI overbought")
        elif last["rsi"] < 30:
            reversal_signals.append("RSI oversold")
        
        # MACD crossover
        if df["macd_diff"].iloc[-2] < 0 and last["macd_diff"] > 0:
            reversal_signals.append("MACD bullish crossover")
        elif df["macd_diff"].iloc[-2] > 0 and last["macd_diff"] < 0:
            reversal_signals.append("MACD bearish crossover")
        
        # Stochastic crossover
        if df["stoch_k"].iloc[-2] < df["stoch_d"].iloc[-2] and last["stoch_k"] > last["stoch_d"]:
            reversal_signals.append("Stochastic bullish crossover")
        elif df["stoch_k"].iloc[-2] > df["stoch_d"].iloc[-2] and last["stoch_k"] < last["stoch_d"]:
            reversal_signals.append("Stochastic bearish crossover")
        
        return {
            "short_term": short_term_trend,
            "medium_term": medium_term_trend,
            "long_term": long_term_trend,
            "strength": trend_strength,
            "market_condition": market_condition,
            "reversal_signals": reversal_signals
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with volatility analysis
        """
        # Calculate daily returns
        df_returns = df["close"].pct_change().dropna()
        
        # Calculate historical volatility (standard deviation of returns)
        volatility_daily = df_returns.std()
        volatility_annualized = volatility_daily * np.sqrt(365)
        
        # Calculate average true range (ATR) as percentage of price
        atr_pct = df["atr"].iloc[-1] / df["close"].iloc[-1]
        
        # Determine volatility regime
        if volatility_daily < 0.01:
            volatility_regime = "low"
        elif volatility_daily < 0.03:
            volatility_regime = "moderate"
        else:
            volatility_regime = "high"
        
        # Calculate Bollinger Band width trend
        bb_width_trend = "expanding" if df["bb_width"].iloc[-1] > df["bb_width"].iloc[-10:].mean() else "contracting"
        
        return {
            "daily": float(volatility_daily),
            "annualized": float(volatility_annualized),
            "atr_pct": float(atr_pct),
            "regime": volatility_regime,
            "bb_width_trend": bb_width_trend
        }
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 20, threshold: float = 0.02) -> Dict[str, List[float]]:
        """Find support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for peak detection
            threshold: Threshold for level significance
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Function to find peaks (for resistance) and troughs (for support)
        def find_peaks(series, window):
            peaks = []
            for i in range(window, len(series) - window):
                if all(series[i] > series[i-j] for j in range(1, window+1)) and \
                   all(series[i] > series[i+j] for j in range(1, window+1)):
                    peaks.append((i, series[i]))
            return peaks
        
        def find_troughs(series, window):
            troughs = []
            for i in range(window, len(series) - window):
                if all(series[i] < series[i-j] for j in range(1, window+1)) and \
                   all(series[i] < series[i+j] for j in range(1, window+1)):
                    troughs.append((i, series[i]))
            return troughs
        
        # Find peaks and troughs
        highs = df["high"].values
        lows = df["low"].values
        
        resistance_points = find_peaks(highs, window)
        support_points = find_troughs(lows, window)
        
        # Cluster levels that are close to each other
        def cluster_levels(levels, threshold):
            if not levels:
                return []
            
            # Sort by price
            sorted_levels = sorted(levels, key=lambda x: x[1])
            
            # Cluster
            clusters = [[sorted_levels[0]]]
            for i in range(1, len(sorted_levels)):
                current_level = sorted_levels[i]
                last_cluster = clusters[-1]
                last_level = last_cluster[-1]
                
                # If close to the last level, add to the same cluster
                if abs(current_level[1] - last_level[1]) / last_level[1] < threshold:
                    last_cluster.append(current_level)
                else:
                    clusters.append([current_level])
            
            # Calculate average price for each cluster
            return [sum(level[1] for level in cluster) / len(cluster) for cluster in clusters]
        
        resistance_levels = cluster_levels(resistance_points, threshold)
        support_levels = cluster_levels(support_points, threshold)
        
        # Sort levels
        resistance_levels.sort(reverse=True)
        support_levels.sort(reverse=True)
        
        return {
            "resistance": resistance_levels[:5],  # Top 5 resistance levels
            "support": support_levels[:5]         # Top 5 support levels
        }

class FactAgent(Agent):
    """Agent responsible for extracting factual information from market data and news."""
    
    def __init__(self, agent_id: str, name: str = "Fact Agent"):
        """Initialize the fact agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
        """
        super().__init__(agent_id, name)
        self.facts_database = {}
        self.news_cache = {}
        self.last_update = {}
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and news to extract factual information.
        
        Args:
            data: Dictionary containing market data and news
            
        Returns:
            Dictionary with extracted factual information
        """
        self.update_status("processing")
        
        try:
            # Extract data
            symbol = data.get("symbol")
            market_data = data.get("market_data", {})
            news_items = data.get("news_items", [])
            on_chain_data = data.get("on_chain_data", {})
            
            if not symbol:
                raise ValueError("Missing required data: symbol")
            
            # Initialize facts for this symbol if not exists
            if symbol not in self.facts_database:
                self.facts_database[symbol] = {
                    "market_facts": {},
                    "news_facts": {},
                    "on_chain_facts": {},
                    "economic_facts": {}
                }
            
            # Extract market facts
            if market_data:
                market_facts = self._extract_market_facts(symbol, market_data)
                self.facts_database[symbol]["market_facts"].update(market_facts)
            
            # Extract news facts
            if news_items:
                news_facts = self._extract_news_facts(symbol, news_items)
                self.facts_database[symbol]["news_facts"].update(news_facts)
            
            # Extract on-chain facts
            if on_chain_data:
                on_chain_facts = self._extract_on_chain_facts(symbol, on_chain_data)
                self.facts_database[symbol]["on_chain_facts"].update(on_chain_facts)
            
            # Update last update timestamp
            self.last_update[symbol] = datetime.now()
            
            # Compile all facts for this symbol
            all_facts = {
                "symbol": symbol,
                "last_update": self.last_update[symbol].isoformat(),
                "market_facts": self.facts_database[symbol]["market_facts"],
                "news_facts": self.facts_database[symbol]["news_facts"],
                "on_chain_facts": self.facts_database[symbol]["on_chain_facts"],
                "economic_facts": self.facts_database[symbol]["economic_facts"]
            }
            
            self.update_status("idle")
            self.log_activity(f"Processed factual information for {symbol}")
            
            return all_facts
        
        except Exception as e:
            self.update_status("error")
            self.log_activity(f"Error processing factual information: {str(e)}", "error")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def _extract_market_facts(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract factual information from market data.
        
        Args:
            symbol: Symbol being analyzed
            market_data: Market data dictionary
            
        Returns:
            Dictionary with extracted market facts
        """
        facts = {}
        
        # Extract price facts
        if "price" in market_data:
            price = market_data["price"]
            facts["current_price"] = price
            
            # Check if we have historical data to compare
            if "historical_prices" in market_data:
                hist_prices = market_data["historical_prices"]
                if len(hist_prices) > 0:
                    # 24h change
                    day_ago = next((p for p in hist_prices if p["timeframe"] == "1d"), None)
                    if day_ago:
                        facts["price_change_24h"] = price - day_ago["price"]
                        facts["price_change_24h_pct"] = (price - day_ago["price"]) / day_ago["price"] * 100
                    
                    # 7d change
                    week_ago = next((p for p in hist_prices if p["timeframe"] == "7d"), None)
                    if week_ago:
                        facts["price_change_7d"] = price - week_ago["price"]
                        facts["price_change_7d_pct"] = (price - week_ago["price"]) / week_ago["price"] * 100
                    
                    # 30d change
                    month_ago = next((p for p in hist_prices if p["timeframe"] == "30d"), None)
                    if month_ago:
                        facts["price_change_30d"] = price - month_ago["price"]
                        facts["price_change_30d_pct"] = (price - month_ago["price"]) / month_ago["price"] * 100
        
        # Extract volume facts
        if "volume" in market_data:
            volume = market_data["volume"]
            facts["current_volume"] = volume
            
            # Check if we have historical data to compare
            if "historical_volumes" in market_data:
                hist_volumes = market_data["historical_volumes"]
                if len(hist_volumes) > 0:
                    # Average volume
                    avg_volume = sum(v["volume"] for v in hist_volumes) / len(hist_volumes)
                    facts["average_volume"] = avg_volume
                    facts["volume_vs_average"] = volume / avg_volume if avg_volume > 0 else 0
        
        # Extract market cap facts
        if "market_cap" in market_data:
            facts["market_cap"] = market_data["market_cap"]
        
        # Extract liquidity facts
        if "liquidity" in market_data:
            liquidity = market_data["liquidity"]
            facts["liquidity"] = liquidity
            
            # Calculate bid-ask spread if available
            if "bid" in market_data and "ask" in market_data:
                bid = market_data["bid"]
                ask = market_data["ask"]
                facts["bid_ask_spread"] = ask - bid
                facts["bid_ask_spread_pct"] = (ask - bid) / bid * 100 if bid > 0 else 0
        
        # Extract order book facts
        if "order_book" in market_data:
            order_book = market_data["order_book"]
            if "bids" in order_book and "asks" in order_book:
                bids = order_book["bids"]
                asks = order_book["asks"]
                
                # Calculate order book imbalance
                bid_volume = sum(b[1] for b in bids)
                ask_volume = sum(a[1] for a in asks)
                total_volume = bid_volume + ask_volume
                
                if total_volume > 0:
                    facts["buy_sell_ratio"] = bid_volume / ask_volume if ask_volume > 0 else float('inf')
                    facts["buy_pressure"] = bid_volume / total_volume
                    facts["sell_pressure"] = ask_volume / total_volume
        
        return facts
    
    def _extract_news_facts(self, symbol: str, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract factual information from news items.
        
        Args:
            symbol: Symbol being analyzed
            news_items: List of news items
            
        Returns:
            Dictionary with extracted news facts
        """
        facts = {}
        
        # Skip if no news items
        if not news_items:
            return facts
        
        # Count news by source
        sources = {}
        for item in news_items:
            source = item.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        facts["news_count"] = len(news_items)
        facts["news_sources"] = sources
        
        # Extract topics from news
        topics = {}
        for item in news_items:
            for tag in item.get("tags", []):
                topics[tag] = topics.get(tag, 0) + 1
        
        facts["news_topics"] = topics
        
        # Extract mentioned entities
        entities = {}
        for item in news_items:
            for entity in item.get("entities", []):
                entity_type = entity.get("type", "unknown")
                entity_name = entity.get("name", "unknown")
                
                if entity_type not in entities:
                    entities[entity_type] = {}
                
                entities[entity_type][entity_name] = entities[entity_type].get(entity_name, 0) + 1
        
        facts["mentioned_entities"] = entities
        
        # Extract factual statements if available
        factual_statements = []
        for item in news_items:
            if "factual_statements" in item:
                factual_statements.extend(item["factual_statements"])
        
        if factual_statements:
            facts["factual_statements"] = factual_statements
        
        return facts
    
    def _extract_on_chain_facts(self, symbol: str, on_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract factual information from on-chain data.
        
        Args:
            symbol: Symbol being analyzed
            on_chain_data: On-chain data dictionary
            
        Returns:
            Dictionary with extracted on-chain facts
        """
        facts = {}
        
        # Extract transaction facts
        if "transactions" in on_chain_data:
            txs = on_chain_data["transactions"]
            facts["transaction_count"] = txs.get("count", 0)
            facts["transaction_volume"] = txs.get("volume", 0)
            facts["average_transaction_size"] = txs.get("average_size", 0)
        
        # Extract address facts
        if "addresses" in on_chain_data:
            addresses = on_chain_data["addresses"]
            facts["active_addresses"] = addresses.get("active", 0)
            facts["new_addresses"] = addresses.get("new", 0)
            
            # Extract whale movements if available
            if "whales" in addresses:
                whales = addresses["whales"]
                facts["whale_inflow"] = whales.get("inflow", 0)
                facts["whale_outflow"] = whales.get("outflow", 0)
                facts["whale_balance_change"] = whales.get("balance_change", 0)
        
        # Extract network facts
        if "network" in on_chain_data:
            network = on_chain_data["network"]
            facts["network_hash_rate"] = network.get("hash_rate", 0)
            facts["network_difficulty"] = network.get("difficulty", 0)
            facts["network_fees"] = network.get("fees", 0)
        
        # Extract smart contract facts
        if "smart_contracts" in on_chain_data:
            contracts = on_chain_data["smart_contracts"]
            facts["contract_interactions"] = contracts.get("interactions", 0)
            facts["new_contracts"] = contracts.get("new", 0)
            
            # Extract DeFi facts if available
            if "defi" in contracts:
                defi = contracts["defi"]
                facts["total_value_locked"] = defi.get("tvl", 0)
                facts["defi_users"] = defi.get("users", 0)
        
        return facts

class SubjectivityAgent(Agent):
    """Agent responsible for analyzing subjective information and sentiment."""
    
    def __init__(self, agent_id: str, name: str = "Subjectivity Agent"):
        """Initialize the subjectivity agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
        """
        super().__init__(agent_id, name)
        self.sentiment_cache = {}
        self.trend_analysis = {}
        
        # Initialize sentiment analyzer if available
        self.sia = None
        if NLP_AVAILABLE:
            try:
                self.sia = SentimentIntensityAnalyzer()
                self.log_activity("Initialized VADER sentiment analyzer")
            except Exception as e:
                self.log_activity(f"Error initializing sentiment analyzer: {str(e)}", "error")
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process news, social media, and market data to extract subjective information.
        
        Args:
            data: Dictionary containing news, social media, and market data
            
        Returns:
            Dictionary with subjective analysis and sentiment
        """
        self.update_status("processing")
        
        try:
            # Extract data
            symbol = data.get("symbol")
            news_items = data.get("news_items", [])
            social_posts = data.get("social_posts", [])
            market_data = data.get("market_data", {})
            
            if not symbol:
                raise ValueError("Missing required data: symbol")
            
            # Analyze news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol, news_items)
            
            # Analyze social media sentiment
            social_sentiment = await self._analyze_social_sentiment(symbol, social_posts)
            
            # Analyze market sentiment
            market_sentiment = self._analyze_market_sentiment(symbol, market_data)
            
            # Combine all sentiment sources
            overall_sentiment = self._combine_sentiment(news_sentiment, social_sentiment, market_sentiment)
            
            # Store in cache
            self.sentiment_cache[symbol] = {
                "timestamp": int(datetime.now().timestamp()),
                "overall": overall_sentiment,
                "news": news_sentiment,
                "social": social_sentiment,
                "market": market_sentiment
            }
            
            # Analyze sentiment trend
            sentiment_trend = self._analyze_sentiment_trend(symbol)
            
            # Create result
            result = {
                "symbol": symbol,
                "timestamp": int(datetime.now().timestamp()),
                "overall_sentiment": overall_sentiment,
                "sentiment_components": {
                    "news": news_sentiment,
                    "social": social_sentiment,
                    "market": market_sentiment
                },
                "sentiment_trend": sentiment_trend,
                "market_mood": self._determine_market_mood(overall_sentiment),
                "confidence": self._calculate_confidence(news_items, social_posts, market_data)
            }
            
            # Add GPT-4o analysis if available
            if OPENAI_UNOFFICIAL_AVAILABLE or OPENAI_AVAILABLE:
                gpt_analysis = await self._get_gpt_sentiment_analysis(symbol, news_items, social_posts, market_data)
                if gpt_analysis:
                    result["gpt_analysis"] = gpt_analysis
            
            self.update_status("idle")
            self.log_activity(f"Processed subjective information for {symbol}")
            
            return result
        
        except Exception as e:
            self.update_status("error")
            self.log_activity(f"Error processing subjective information: {str(e)}", "error")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    async def _analyze_news_sentiment(self, symbol: str, news_items: List[Dict[str, Any]]) -> float:
        """Analyze sentiment from news items.
        
        Args:
            symbol: Symbol being analyzed
            news_items: List of news items
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not news_items:
            return 0.0
        
        # Skip if no sentiment analyzer
        if not self.sia:
            return 0.0
        
        total_sentiment = 0.0
        total_weight = 0.0
        
        for item in news_items:
            # Get text content
            title = item.get("title", "")
            content = item.get("content", "")
            
            # Skip if no content
            if not title and not content:
                continue
            
            # Combine title and content with title having more weight
            text = title + " " + content
            
            # Get sentiment score
            sentiment = self.sia.polarity_scores(text)
            compound_score = sentiment["compound"]
            
            # Weight by relevance or recency
            weight = item.get("relevance", 1.0)
            
            # Add to total
            total_sentiment += compound_score * weight
            total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            return total_sentiment / total_weight
        else:
            return 0.0
    
    async def _analyze_social_sentiment(self, symbol: str, social_posts: List[Dict[str, Any]]) -> float:
        """Analyze sentiment from social media posts.
        
        Args:
            symbol: Symbol being analyzed
            social_posts: List of social media posts
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not social_posts:
            return 0.0
        
        # Skip if no sentiment analyzer
        if not self.sia:
            return 0.0
        
        total_sentiment = 0.0
        total_weight = 0.0
        
        for post in social_posts:
            # Get text content
            content = post.get("content", "")
            
            # Skip if no content
            if not content:
                continue
            
            # Get sentiment score
            sentiment = self.sia.polarity_scores(content)
            compound_score = sentiment["compound"]
            
            # Weight by engagement or followers
            weight = post.get("engagement", 1.0)
            
            # Add to total
            total_sentiment += compound_score * weight
            total_weight += weight
        
        # Calculate weighted average
        if total_weight > 0:
            return total_sentiment / total_weight
        else:
            return 0.0
    
    def _analyze_market_sentiment(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """Analyze sentiment from market data.
        
        Args:
            symbol: Symbol being analyzed
            market_data: Market data dictionary
            
        Returns:
            Sentiment score (-1.0 to 1.0)
        """
        if not market_data:
            return 0.0
        
        # Initialize sentiment components
        components = []
        
        # Price momentum
        if "price_change_24h_pct" in market_data:
            price_change = market_data["price_change_24h_pct"]
            # Normalize to -1 to 1 range
            price_sentiment = min(max(price_change / 10, -1), 1)
            components.append(price_sentiment)
        
        # Volume change
        if "volume_vs_average" in market_data:
            volume_ratio = market_data["volume_vs_average"]
            # Normalize to -1 to 1 range
            volume_sentiment = min(max((volume_ratio - 1) / 2, -1), 1)
            components.append(volume_sentiment)
        
        # Buy/sell pressure
        if "buy_pressure" in market_data and "sell_pressure" in market_data:
            buy = market_data["buy_pressure"]
            sell = market_data["sell_pressure"]
            if buy + sell > 0:
                pressure_sentiment = min(max((buy - sell) / (buy + sell), -1), 1)
                components.append(pressure_sentiment)
        
        # Volatility
        if "volatility" in market_data:
            volatility = market_data["volatility"]
            # High volatility is neutral, low volatility is slightly positive
            volatility_sentiment = min(max((0.05 - volatility) * 5, -0.5), 0.5)
            components.append(volatility_sentiment)
        
        # Calculate average of components
        if components:
            return sum(components) / len(components)
        else:
            return 0.0
    
    def _combine_sentiment(self, news_sentiment: float, social_sentiment: float, market_sentiment: float) -> float:
        """Combine sentiment from different sources.
        
        Args:
            news_sentiment: News sentiment score
            social_sentiment: Social media sentiment score
            market_sentiment: Market data sentiment score
            
        Returns:
            Combined sentiment score
        """
        # Weights for different sources
        news_weight = 0.4
        social_weight = 0.3
        market_weight = 0.3
        
        # Calculate weighted sum
        total_weight = 0.0
        weighted_sum = 0.0
        
        if news_sentiment != 0.0:
            weighted_sum += news_sentiment * news_weight
            total_weight += news_weight
        
        if social_sentiment != 0.0:
            weighted_sum += social_sentiment * social_weight
            total_weight += social_weight
        
        if market_sentiment != 0.0:
            weighted_sum += market_sentiment * market_weight
            total_weight += market_weight
        
        # Return weighted average
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _analyze_sentiment_trend(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment trend over time.
        
        Args:
            symbol: Symbol being analyzed
            
        Returns:
            Dictionary with sentiment trend analysis
        """
        # Check if we have historical sentiment data
        if symbol not in self.trend_analysis:
            self.trend_analysis[symbol] = {
                "history": [],
                "last_update": None
            }
        
        # Get current sentiment
        current = self.sentiment_cache.get(symbol, {})
        current_timestamp = current.get("timestamp")
        current_sentiment = current.get("overall", 0.0)
        
        # Skip if no current sentiment
        if current_timestamp is None:
            return {"trend": "neutral", "change": 0.0}
        
        # Add to history
        history = self.trend_analysis[symbol]["history"]
        history.append((current_timestamp, current_sentiment))
        
        # Keep only last 30 days
        cutoff = int((datetime.now() - timedelta(days=30)).timestamp())
        history = [(ts, sent) for ts, sent in history if ts >= cutoff]
        
        # Update history
        self.trend_analysis[symbol]["history"] = history
        self.trend_analysis[symbol]["last_update"] = datetime.now()
        
        # Calculate trend
        if len(history) < 2:
            return {"trend": "neutral", "change": 0.0}
        
        # Get sentiment change
        oldest = history[0][1]
        newest = history[-1][1]
        change = newest - oldest
        
        # Determine trend
        if change > 0.2:
            trend = "strongly_bullish"
        elif change > 0.05:
            trend = "bullish"
        elif change < -0.2:
            trend = "strongly_bearish"
        elif change < -0.05:
            trend = "bearish"
        else:
            trend = "neutral"
        
        return {"trend": trend, "change": change}
    
    def _determine_market_mood(self, sentiment: float) -> str:
        """Determine market mood based on sentiment score.
        
        Args:
            sentiment: Sentiment score (-1.0 to 1.0)
            
        Returns:
            Market mood description
        """
        if sentiment > 0.6:
            return "euphoric"
        elif sentiment > 0.3:
            return "bullish"
        elif sentiment > 0.1:
            return "mildly_bullish"
        elif sentiment > -0.1:
            return "neutral"
        elif sentiment > -0.3:
            return "mildly_bearish"
        elif sentiment > -0.6:
            return "bearish"
        else:
            return "fearful"
    
    def _calculate_confidence(self, news_items: List, social_posts: List, market_data: Dict) -> float:
        """Calculate confidence in sentiment analysis.
        
        Args:
            news_items: List of news items
            social_posts: List of social media posts
            market_data: Market data dictionary
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = 0.5
        
        # Adjust based on data availability
        if news_items:
            confidence += 0.1 * min(len(news_items) / 10, 1.0)
        
        if social_posts:
            confidence += 0.1 * min(len(social_posts) / 20, 1.0)
        
        if market_data:
            confidence += 0.1
        
        # Adjust based on data freshness
        current_time = datetime.now().timestamp()
        
        if news_items:
            newest_news = max((item.get("timestamp", 0) for item in news_items), default=0)
            if newest_news > current_time - 86400:  # Within last 24 hours
                confidence += 0.1
        
        if social_posts:
            newest_post = max((post.get("timestamp", 0) for post in social_posts), default=0)
            if newest_post > current_time - 43200:  # Within last 12 hours
                confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    async def _get_gpt_sentiment_analysis(self, symbol: str, news_items: List, social_posts: List, market_data: Dict) -> Dict[str, Any]:
        """Get sentiment analysis from GPT-4o.
        
        Args:
            symbol: Symbol being analyzed
            news_items: List of news items
            social_posts: List of social media posts
            market_data: Market data dictionary
            
        Returns:
            Dictionary with GPT sentiment analysis
        """
        # Skip if OpenAI is not available
        if not OPENAI_UNOFFICIAL_AVAILABLE and not OPENAI_AVAILABLE:
            return None
        
        try:
            # Prepare input for GPT
            prompt = f"""
            Please analyze the market sentiment for {symbol} based on the following data:
            
            MARKET DATA:
            {json.dumps(market_data, indent=2)}
            
            NEWS (sample of {min(5, len(news_items))} items):
            {json.dumps(news_items[:5], indent=2)}
            
            SOCIAL MEDIA (sample of {min(5, len(social_posts))} posts):
            {json.dumps(social_posts[:5], indent=2)}
            
            Provide a sentiment analysis with the following structure:
            1. Overall sentiment score (-1.0 to 1.0)
            2. Sentiment breakdown (news, social, market)
            3. Key sentiment drivers
            4. Market mood description
            5. Confidence in analysis
            
            Format your response as JSON.
            """
            
            # Call GPT-4o via openai-unofficial if available, otherwise use standard OpenAI
            if OPENAI_UNOFFICIAL_AVAILABLE:
                response = await openai_unofficial.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                gpt_response = response.choices[0].message.content
            else:
                response = await openai.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=1000
                )
                gpt_response = response.choices[0].message.content
            
            # Parse JSON response
            try:
                return json.loads(gpt_response)
            except json.JSONDecodeError:
                # If not valid JSON, return as text
                return {"text_analysis": gpt_response}
        
        except Exception as e:
            self.log_activity(f"Error getting GPT sentiment analysis: {str(e)}", "error")
            return None

class TradingAgent(Agent):
    """Agent responsible for generating trading signals and executing trades."""
    
    def __init__(self, agent_id: str, name: str = "Trading Agent"):
        """Initialize the trading agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
        """
        super().__init__(agent_id, name)
        self.strategies = {}
        self.active_orders = {}
        self.positions = {}
        self.trade_history = []
        self.risk_manager = RiskManager()
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data, statistics, and sentiment to generate trading signals.
        
        Args:
            data: Dictionary containing market data, statistics, and sentiment
            
        Returns:
            Dictionary with trading signals and executed trades
        """
        self.update_status("processing")
        
        try:
            # Extract data
            symbol = data.get("symbol")
            exchange = data.get("exchange")
            market_data = data.get("market_data", {})
            statistics = data.get("statistics", {})
            facts = data.get("facts", {})
            sentiment = data.get("sentiment", {})
            risk_params = data.get("risk_params", {})
            
            if not symbol or not exchange:
                raise ValueError("Missing required data: symbol or exchange")
            
            # Update risk parameters if provided
            if risk_params:
                self.risk_manager.update_parameters(risk_params)
            
            # Generate signals from all active strategies
            signals = []
            for strategy_id, strategy in self.strategies.items():
                if not strategy.is_active:
                    continue
                
                if strategy.applies_to(symbol, exchange):
                    try:
                        strategy_signals = await strategy.generate_signals(
                            symbol=symbol,
                            exchange=exchange,
                            market_data=market_data,
                            statistics=statistics,
                            facts=facts,
                            sentiment=sentiment
                        )
                        
                        for signal in strategy_signals:
                            signal["strategy_id"] = strategy_id
                            signals.append(signal)
                    except Exception as e:
                        self.log_activity(f"Error generating signals for strategy {strategy_id}: {str(e)}", "error")
            
            # Combine signals if multiple signals for same symbol/timeframe
            combined_signals = self._combine_signals(signals)
            
            # Apply risk management
            filtered_signals = self.risk_manager.filter_signals(combined_signals, self.positions)
            
            # Execute trades based on signals
            executed_trades = []
            for signal in filtered_signals:
                try:
                    trade = await self._execute_signal(signal)
                    if trade:
                        executed_trades.append(trade)
                except Exception as e:
                    self.log_activity(f"Error executing signal: {str(e)}", "error")
            
            # Update positions
            await self._update_positions()
            
            # Check for take profit / stop loss
            closed_positions = await self._check_exit_conditions()
            
            # Prepare result
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": int(datetime.now().timestamp()),
                "signals": filtered_signals,
                "executed_trades": executed_trades,
                "closed_positions": closed_positions,
                "active_positions": list(self.positions.values()),
                "active_orders": list(self.active_orders.values())
            }
            
            self.update_status("idle")
            self.log_activity(f"Processed trading signals for {symbol} on {exchange}")
            
            return result
        
        except Exception as e:
            self.update_status("error")
            self.log_activity(f"Error processing trading signals: {str(e)}", "error")
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def add_strategy(self, strategy: 'TradingStrategy') -> str:
        """Add a trading strategy.
        
        Args:
            strategy: Trading strategy to add
            
        Returns:
            Strategy ID
        """
        strategy_id = str(uuid.uuid4())
        self.strategies[strategy_id] = strategy
        self.log_activity(f"Added strategy: {strategy.name} (ID: {strategy_id})")
        return strategy_id
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a trading strategy.
        
        Args:
            strategy_id: ID of the strategy to remove
            
        Returns:
            True if successful, False otherwise
        """
        if strategy_id in self.strategies:
            del self.strategies[strategy_id]
            self.log_activity(f"Removed strategy with ID: {strategy_id}")
            return True
        else:
            self.log_activity(f"Strategy with ID {strategy_id} not found", "warning")
            return False
    
    def _combine_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine multiple signals for the same symbol and timeframe.
        
        Args:
            signals: List of trading signals
            
        Returns:
            List of combined signals
        """
        if not signals:
            return []
        
        # Group signals by symbol, exchange, and timeframe
        grouped_signals = {}
        for signal in signals:
            key = (signal["symbol"], signal["exchange"], signal["timeframe"])
            if key not in grouped_signals:
                grouped_signals[key] = []
            grouped_signals[key].append(signal)
        
        # Combine signals in each group
        combined = []
        for (symbol, exchange, timeframe), group in grouped_signals.items():
            # If only one signal, use it directly
            if len(group) == 1:
                combined.append(group[0])
                continue
            
            # Count buy and sell signals
            buy_count = sum(1 for s in group if s["direction"] == OrderSide.BUY.value)
            sell_count = sum(1 for s in group if s["direction"] == OrderSide.SELL.value)
            
            # Calculate average strength
            buy_strength = sum(s["strength"] for s in group if s["direction"] == OrderSide.BUY.value) / buy_count if buy_count > 0 else 0
            sell_strength = sum(s["strength"] for s in group if s["direction"] == OrderSide.SELL.value) / sell_count if sell_count > 0 else 0
            
            # Determine overall direction
            if buy_strength > sell_strength:
                direction = OrderSide.BUY.value
                strength = buy_strength - sell_strength
            elif sell_strength > buy_strength:
                direction = OrderSide.SELL.value
                strength = sell_strength - buy_strength
            else:
                # No clear direction, skip
                continue
            
            # Create combined signal
            combined_signal = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": int(datetime.now().timestamp()),
                "signal_type": "combined",
                "direction": direction,
                "strength": strength,
                "timeframe": timeframe,
                "source": "combined",
                "component_signals": len(group),
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "strategies": [s["strategy_id"] for s in group]
            }
            
            combined.append(combined_signal)
        
        return combined
    
    async def _execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            
        Returns:
            Dictionary with executed trade information
        """
        symbol = signal["symbol"]
        exchange = signal["exchange"]
        direction = signal["direction"]
        
        # Calculate position size based on risk management
        position_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            available_balance=self._get_available_balance(exchange)
        )
        
        # Skip if position size is too small
        if position_size <= 0:
            return None
        
        # Create order
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=symbol,
            exchange=exchange,
            order_type=OrderType.MARKET,
            side=OrderSide(direction),
            amount=position_size,
            status="pending",
            timestamp=int(datetime.now().timestamp())
        )
        
        # Store order
        self.active_orders[order_id] = order
        
        # In a real implementation, this would call the exchange API
        # For now, simulate order execution
        executed_price = self._get_current_price(symbol, exchange)
        
        # Update order
        order.price = executed_price
        order.filled_amount = position_size
        order.status = "filled"
        
        # Create position
        position_id = str(uuid.uuid4())
        position = Position(
            position_id=position_id,
            symbol=symbol,
            exchange=exchange,
            position_type=PositionType.LONG if direction == OrderSide.BUY.value else PositionType.SHORT,
            entry_price=executed_price,
            current_price=executed_price,
            amount=position_size,
            timestamp=int(datetime.now().timestamp())
        )
        
        # Set stop loss and take profit
        position.stop_loss = self.risk_manager.calculate_stop_loss(
            symbol=symbol,
            entry_price=executed_price,
            position_type=position.position_type
        )
        
        position.take_profit = self.risk_manager.calculate_take_profit(
            symbol=symbol,
            entry_price=executed_price,
            position_type=position.position_type
        )
        
        # Store position
        self.positions[position_id] = position
        
        # Add to trade history
        trade = {
            "order_id": order_id,
            "position_id": position_id,
            "symbol": symbol,
            "exchange": exchange,
            "direction": direction,
            "price": executed_price,
            "amount": position_size,
            "timestamp": order.timestamp,
            "signal": signal
        }
        
        self.trade_history.append(trade)
        
        # Remove from active orders
        del self.active_orders[order_id]
        
        self.log_activity(f"Executed {direction} order for {position_size} {symbol} at {executed_price} on {exchange}")
        
        return trade
    
    async def _update_positions(self) -> None:
        """Update all active positions with current prices and P&L."""
        for position_id, position in list(self.positions.items()):
            try:
                # Get current price
                current_price = self._get_current_price(position.symbol, position.exchange)
                
                # Update position
                position.current_price = current_price
                
                # Calculate P&L
                if position.position_type == PositionType.LONG:
                    position.pnl = (current_price - position.entry_price) * position.amount
                    position.pnl_percent = (current_price / position.entry_price - 1) * 100
                else:  # SHORT
                    position.pnl = (position.entry_price - current_price) * position.amount
                    position.pnl_percent = (position.entry_price / current_price - 1) * 100
            
            except Exception as e:
                self.log_activity(f"Error updating position {position_id}: {str(e)}", "error")
    
    async def _check_exit_conditions(self) -> List[Dict[str, Any]]:
        """Check if any positions should be closed due to take profit or stop loss.
        
        Returns:
            List of closed positions
        """
        closed_positions = []
        
        for position_id, position in list(self.positions.items()):
            try:
                # Skip if no current price
                if position.current_price is None:
                    continue
                
                # Check stop loss
                if position.stop_loss is not None:
                    if (position.position_type == PositionType.LONG and position.current_price <= position.stop_loss) or \
                       (position.position_type == PositionType.SHORT and position.current_price >= position.stop_loss):
                        # Close position due to stop loss
                        closed = await self._close_position(position_id, "stop_loss")
                        if closed:
                            closed_positions.append(closed)
                        continue
                
                # Check take profit
                if position.take_profit is not None:
                    if (position.position_type == PositionType.LONG and position.current_price >= position.take_profit) or \
                       (position.position_type == PositionType.SHORT and position.current_price <= position.take_profit):
                        # Close position due to take profit
                        closed = await self._close_position(position_id, "take_profit")
                        if closed:
                            closed_positions.append(closed)
                        continue
            
            except Exception as e:
                self.log_activity(f"Error checking exit conditions for position {position_id}: {str(e)}", "error")
        
        return closed_positions
    
    async def _close_position(self, position_id: str, reason: str) -> Dict[str, Any]:
        """Close a position.
        
        Args:
            position_id: ID of the position to close
            reason: Reason for closing the position
            
        Returns:
            Dictionary with closed position information
        """
        if position_id not in self.positions:
            return None
        
        position = self.positions[position_id]
        
        # Create order for closing position
        order_id = str(uuid.uuid4())
        order = Order(
            order_id=order_id,
            symbol=position.symbol,
            exchange=position.exchange,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL if position.position_type == PositionType.LONG else OrderSide.BUY,
            amount=position.amount,
            status="pending",
            timestamp=int(datetime.now().timestamp())
        )
        
        # Store order
        self.active_orders[order_id] = order
        
        # In a real implementation, this would call the exchange API
        # For now, simulate order execution
        executed_price = position.current_price
        
        # Update order
        order.price = executed_price
        order.filled_amount = position.amount
        order.status = "filled"
        
        # Calculate final P&L
        if position.position_type == PositionType.LONG:
            pnl = (executed_price - position.entry_price) * position.amount
            pnl_percent = (executed_price / position.entry_price - 1) * 100
        else:  # SHORT
            pnl = (position.entry_price - executed_price) * position.amount
            pnl_percent = (position.entry_price / executed_price - 1) * 100
        
        # Create result
        closed_position = {
            "position_id": position_id,
            "symbol": position.symbol,
            "exchange": position.exchange,
            "position_type": position.position_type.value,
            "entry_price": position.entry_price,
            "exit_price": executed_price,
            "amount": position.amount,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "entry_time": position.timestamp,
            "exit_time": order.timestamp,
            "duration_seconds": order.timestamp - position.timestamp,
            "reason": reason
        }
        
        # Remove from positions
        del self.positions[position_id]
        
        # Remove from active orders
        del self.active_orders[order_id]
        
        self.log_activity(f"Closed position {position_id} with {reason}, PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
        
        return closed_position
    
    def _get_current_price(self, symbol: str, exchange: str) -> float:
        """Get current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            exchange: Exchange to get price from
            
        Returns:
            Current price
        """
        # In a real implementation, this would call the exchange API
        # For now, return a dummy price
        return 100.0
    
    def _get_available_balance(self, exchange: str) -> float:
        """Get available balance for an exchange.
        
        Args:
            exchange: Exchange to get balance for
            
        Returns:
            Available balance
        """
        # In a real implementation, this would call the exchange API
        # For now, return a dummy balance
        return 10000.0

# --- Risk Management ---

class RiskManager:
    """Risk management system for trading."""
    
    def __init__(self):
        """Initialize the risk manager."""
        self.max_position_size = 0.05  # 5% of available balance
        self.max_risk_per_trade = 0.01  # 1% of available balance
        self.max_open_positions = 10
        self.max_daily_loss = 0.05  # 5% of total balance
        self.max_drawdown = 0.20  # 20% of peak balance
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.06  # 6% take profit (3:1 reward-to-risk ratio)
        self.daily_loss = 0.0
        self.peak_balance = 0.0
        self.current_balance = 0.0
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update risk parameters.
        
        Args:
            params: Dictionary with risk parameters
        """
        if "max_position_size" in params:
            self.max_position_size = params["max_position_size"]
        
        if "max_risk_per_trade" in params:
            self.max_risk_per_trade = params["max_risk_per_trade"]
        
        if "max_open_positions" in params:
            self.max_open_positions = params["max_open_positions"]
        
        if "max_daily_loss" in params:
            self.max_daily_loss = params["max_daily_loss"]
        
        if "max_drawdown" in params:
            self.max_drawdown = params["max_drawdown"]
        
        if "stop_loss_pct" in params:
            self.stop_loss_pct = params["stop_loss_pct"]
        
        if "take_profit_pct