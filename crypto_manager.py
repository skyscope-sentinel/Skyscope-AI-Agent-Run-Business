
import os
import json
import uuid
import time
import logging
import threading
import queue
import random
import asyncio
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures
import requests
import hashlib
import base64
import hmac
import websockets
import re
import io
import tempfile
import shutil
from decimal import Decimal

# Try to import cryptocurrency libraries
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    logging.warning("CCXT not available. Cryptocurrency exchange features will be limited.")

try:
    from web3 import Web3
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logging.warning("Web3 not available. Ethereum blockchain features will be limited.")

try:
    import binance
    from binance.client import Client as BinanceClient
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    logging.warning("Binance API not available. Binance exchange features will be limited.")

try:
    import bybit
    BYBIT_AVAILABLE = True
except ImportError:
    BYBIT_AVAILABLE = False
    logging.warning("ByBit API not available. ByBit exchange features will be limited.")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("Technical Analysis library not available. Trading strategy features will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/crypto_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("crypto_manager")

# Constants
CRYPTO_DIR = Path("crypto")
WALLETS_DIR = Path("wallets")
TRANSACTIONS_DIR = Path("transactions")
STRATEGIES_DIR = Path("strategies")
BACKTEST_DIR = Path("backtest")
EXCHANGE_DIR = Path("exchanges")
MARKET_DATA_DIR = Path("market_data")
REPORTS_DIR = Path("reports")

# Ensure directories exist
for directory in [
    CRYPTO_DIR, 
    WALLETS_DIR, 
    TRANSACTIONS_DIR, 
    STRATEGIES_DIR, 
    BACKTEST_DIR, 
    EXCHANGE_DIR, 
    MARKET_DATA_DIR, 
    REPORTS_DIR
]:
    directory.mkdir(exist_ok=True, parents=True)

class CryptoType(Enum):
    """Enumeration of cryptocurrency types."""
    BITCOIN = "BTC"
    ETHEREUM = "ETH"
    SOLANA = "SOL"
    CARDANO = "ADA"
    RIPPLE = "XRP"
    POLKADOT = "DOT"
    DOGECOIN = "DOGE"
    LITECOIN = "LTC"
    CHAINLINK = "LINK"
    UNISWAP = "UNI"
    POLYGON = "MATIC"
    AVALANCHE = "AVAX"
    KASPA = "KAS"
    BINANCE_COIN = "BNB"
    TETHER = "USDT"
    USD_COIN = "USDC"
    DAI = "DAI"
    OTHER = "OTHER"

class ExchangeType(Enum):
    """Enumeration of cryptocurrency exchanges."""
    BINANCE = "binance"
    BYBIT = "bybit"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    KUCOIN = "kucoin"
    FTX = "ftx"
    HUOBI = "huobi"
    BITFINEX = "bitfinex"
    BITSTAMP = "bitstamp"
    GEMINI = "gemini"
    OKEX = "okex"
    GATE_IO = "gate_io"
    BITTREX = "bittrex"
    POLONIEX = "poloniex"
    OTHER = "other"

class NetworkType(Enum):
    """Enumeration of blockchain networks."""
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"
    ETHEREUM_LAYER2 = "ethereum_l2"
    BINANCE_SMART_CHAIN = "bsc"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    AVALANCHE = "avalanche"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    COSMOS = "cosmos"
    TRON = "tron"
    KASPA = "kaspa"
    OTHER = "other"

class WalletType(Enum):
    """Enumeration of wallet types."""
    HOT = "hot"  # Connected to internet
    COLD = "cold"  # Offline storage
    HARDWARE = "hardware"  # Hardware wallet
    PAPER = "paper"  # Paper wallet
    EXCHANGE = "exchange"  # Exchange wallet
    CUSTODIAL = "custodial"  # Custodial service
    MULTI_SIG = "multi_sig"  # Multi-signature wallet
    SMART_CONTRACT = "smart_contract"  # Smart contract wallet

class TransactionType(Enum):
    """Enumeration of cryptocurrency transaction types."""
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    PURCHASE = "purchase"
    SALE = "sale"
    TRADE = "trade"
    SWAP = "swap"
    STAKE = "stake"
    UNSTAKE = "unstake"
    REWARD = "reward"
    FEE = "fee"
    TRANSFER = "transfer"
    AIRDROP = "airdrop"
    MINING = "mining"
    YIELD = "yield"
    LENDING = "lending"
    BORROWING = "borrowing"
    REPAYMENT = "repayment"
    OTHER = "other"

class TradeDirection(Enum):
    """Enumeration of trade directions."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"

class OrderType(Enum):
    """Enumeration of order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "one_cancels_other"
    ICEBERG = "iceberg"
    TWAP = "time_weighted_average_price"
    VWAP = "volume_weighted_average_price"

class TimeFrame(Enum):
    """Enumeration of time frames for market data."""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

class StrategyType(Enum):
    """Enumeration of trading strategy types."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_MAKING = "market_making"
    GRID_TRADING = "grid_trading"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"
    POSITION_TRADING = "position_trading"
    DOLLAR_COST_AVERAGING = "dollar_cost_averaging"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    MACHINE_LEARNING = "machine_learning"
    CUSTOM = "custom"

class RiskLevel(Enum):
    """Enumeration of risk levels for trading strategies."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class CryptoAddress:
    """Represents a cryptocurrency address."""
    address: str
    crypto_type: CryptoType
    network: NetworkType
    label: str = ""
    is_contract: bool = False
    derivation_path: str = ""
    public_key: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "address": self.address,
            "crypto_type": self.crypto_type.value,
            "network": self.network.value,
            "label": self.label,
            "is_contract": self.is_contract,
            "derivation_path": self.derivation_path,
            "public_key": self.public_key,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoAddress':
        """Create from dictionary."""
        return cls(
            address=data["address"],
            crypto_type=CryptoType(data["crypto_type"]),
            network=NetworkType(data["network"]),
            label=data.get("label", ""),
            is_contract=data.get("is_contract", False),
            derivation_path=data.get("derivation_path", ""),
            public_key=data.get("public_key", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None
        )

@dataclass
class CryptoWallet:
    """Represents a cryptocurrency wallet."""
    wallet_id: str
    name: str
    type: WalletType
    addresses: Dict[str, CryptoAddress] = field(default_factory=dict)  # address -> CryptoAddress
    balances: Dict[str, float] = field(default_factory=dict)  # crypto_type -> balance
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    encrypted_private_keys: Dict[str, str] = field(default_factory=dict)  # address -> encrypted_private_key
    encrypted_seed_phrase: str = ""
    encryption_method: str = "AES-256"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "wallet_id": self.wallet_id,
            "name": self.name,
            "type": self.type.value,
            "addresses": {addr: address.to_dict() for addr, address in self.addresses.items()},
            "balances": self.balances,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "encrypted_private_keys": self.encrypted_private_keys,
            "encrypted_seed_phrase": self.encrypted_seed_phrase,
            "encryption_method": self.encryption_method,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoWallet':
        """Create from dictionary."""
        wallet = cls(
            wallet_id=data["wallet_id"],
            name=data["name"],
            type=WalletType(data["type"]),
            balances=data.get("balances", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            encrypted_private_keys=data.get("encrypted_private_keys", {}),
            encrypted_seed_phrase=data.get("encrypted_seed_phrase", ""),
            encryption_method=data.get("encryption_method", "AES-256"),
            metadata=data.get("metadata", {})
        )
        
        # Load addresses
        addresses = {}
        for addr, addr_data in data.get("addresses", {}).items():
            addresses[addr] = CryptoAddress.from_dict(addr_data)
        wallet.addresses = addresses
        
        return wallet
    
    def save(self, directory: Path = WALLETS_DIR) -> Path:
        """Save wallet to file."""
        directory.mkdir(exist_ok=True, parents=True)
        filepath = directory / f"{self.wallet_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'CryptoWallet':
        """Load wallet from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class CryptoTransaction:
    """Represents a cryptocurrency transaction."""
    transaction_id: str
    wallet_id: str
    type: TransactionType
    crypto_type: CryptoType
    amount: float
    fee: float
    timestamp: datetime
    status: str = "pending"  # pending, completed, failed
    from_address: str = ""
    to_address: str = ""
    transaction_hash: str = ""
    block_number: int = 0
    block_hash: str = ""
    network: NetworkType = NetworkType.OTHER
    exchange: Optional[ExchangeType] = None
    price_usd: float = 0.0
    price_btc: float = 0.0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transaction_id": self.transaction_id,
            "wallet_id": self.wallet_id,
            "type": self.type.value,
            "crypto_type": self.crypto_type.value,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status,
            "from_address": self.from_address,
            "to_address": self.to_address,
            "transaction_hash": self.transaction_hash,
            "block_number": self.block_number,
            "block_hash": self.block_hash,
            "network": self.network.value,
            "exchange": self.exchange.value if self.exchange else None,
            "price_usd": self.price_usd,
            "price_btc": self.price_btc,
            "notes": self.notes,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CryptoTransaction':
        """Create from dictionary."""
        return cls(
            transaction_id=data["transaction_id"],
            wallet_id=data["wallet_id"],
            type=TransactionType(data["type"]),
            crypto_type=CryptoType(data["crypto_type"]),
            amount=data["amount"],
            fee=data["fee"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            status=data.get("status", "pending"),
            from_address=data.get("from_address", ""),
            to_address=data.get("to_address", ""),
            transaction_hash=data.get("transaction_hash", ""),
            block_number=data.get("block_number", 0),
            block_hash=data.get("block_hash", ""),
            network=NetworkType(data.get("network", "other")),
            exchange=ExchangeType(data["exchange"]) if data.get("exchange") else None,
            price_usd=data.get("price_usd", 0.0),
            price_btc=data.get("price_btc", 0.0),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {})
        )
    
    def save(self, directory: Path = TRANSACTIONS_DIR) -> Path:
        """Save transaction to file."""
        directory.mkdir(exist_ok=True, parents=True)
        wallet_dir = directory / self.wallet_id
        wallet_dir.mkdir(exist_ok=True)
        
        # Create year/month subdirectories for better organization
        year_dir = wallet_dir / str(self.timestamp.year)
        year_dir.mkdir(exist_ok=True)
        month_dir = year_dir / f"{self.timestamp.month:02d}"
        month_dir.mkdir(exist_ok=True)
        
        filepath = month_dir / f"{self.transaction_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'CryptoTransaction':
        """Load transaction from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class ExchangeAccount:
    """Represents an account on a cryptocurrency exchange."""
    account_id: str
    exchange: ExchangeType
    name: str
    api_key: str
    api_secret: str
    passphrase: str = ""  # Some exchanges require a passphrase
    subaccount: str = ""  # Some exchanges support subaccounts
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    permissions: List[str] = field(default_factory=list)  # read, trade, withdraw, etc.
    balances: Dict[str, float] = field(default_factory=dict)  # crypto_type -> balance
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "account_id": self.account_id,
            "exchange": self.exchange.value,
            "name": self.name,
            "api_key": self.encrypt(self.api_key),
            "api_secret": self.encrypt(self.api_secret),
            "passphrase": self.encrypt(self.passphrase) if self.passphrase else "",
            "subaccount": self.subaccount,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat(),
            "is_active": self.is_active,
            "permissions": self.permissions,
            "balances": self.balances,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExchangeAccount':
        """Create from dictionary."""
        account = cls(
            account_id=data["account_id"],
            exchange=ExchangeType(data["exchange"]),
            name=data["name"],
            api_key="",  # Will be decrypted below
            api_secret="",  # Will be decrypted below
            passphrase="",  # Will be decrypted below
            subaccount=data.get("subaccount", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_used=datetime.fromisoformat(data["last_used"]),
            is_active=data.get("is_active", True),
            permissions=data.get("permissions", []),
            balances=data.get("balances", {}),
            metadata=data.get("metadata", {})
        )
        
        # Decrypt sensitive fields
        account.api_key = account.decrypt(data["api_key"])
        account.api_secret = account.decrypt(data["api_secret"])
        if data.get("passphrase"):
            account.passphrase = account.decrypt(data["passphrase"])
        
        return account
    
    def encrypt(self, text: str) -> str:
        """Simple encryption for demonstration purposes.
        In production, use proper encryption libraries and secure key management."""
        if not text:
            return ""
        
        # This is a placeholder. In a real system, use proper encryption
        key = hashlib.sha256(f"SKYSCOPE_CRYPTO_{self.account_id}".encode()).digest()
        encoded = base64.b64encode(text.encode())
        return encoded.decode()
    
    def decrypt(self, encrypted_text: str) -> str:
        """Simple decryption for demonstration purposes.
        In production, use proper encryption libraries and secure key management."""
        if not encrypted_text:
            return ""
        
        # This is a placeholder. In a real system, use proper decryption
        try:
            decoded = base64.b64decode(encrypted_text.encode()).decode()
            return decoded
        except:
            return ""
    
    def save(self, directory: Path = EXCHANGE_DIR) -> Path:
        """Save exchange account to file."""
        directory.mkdir(exist_ok=True, parents=True)
        exchange_dir = directory / self.exchange.value
        exchange_dir.mkdir(exist_ok=True)
        
        filepath = exchange_dir / f"{self.account_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExchangeAccount':
        """Load exchange account from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class MarketData:
    """Represents cryptocurrency market data."""
    symbol: str  # Trading pair, e.g., BTC/USDT
    exchange: ExchangeType
    timeframe: TimeFrame
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: Dict[str, float] = field(default_factory=dict)  # Technical indicators
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "indicators": self.indicators
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            exchange=ExchangeType(data["exchange"]),
            timeframe=TimeFrame(data["timeframe"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            volume=data["volume"],
            indicators=data.get("indicators", {})
        )

@dataclass
class TradingStrategy:
    """Represents a cryptocurrency trading strategy."""
    strategy_id: str
    name: str
    type: StrategyType
    description: str
    risk_level: RiskLevel
    symbols: List[str]  # Trading pairs
    timeframes: List[TimeFrame]
    exchanges: List[ExchangeType]
    parameters: Dict[str, Any] = field(default_factory=dict)
    entry_conditions: List[Dict[str, Any]] = field(default_factory=list)
    exit_conditions: List[Dict[str, Any]] = field(default_factory=list)
    position_sizing: Dict[str, Any] = field(default_factory=dict)
    risk_management: Dict[str, Any] = field(default_factory=dict)
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_active: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "symbols": self.symbols,
            "timeframes": [tf.value for tf in self.timeframes],
            "exchanges": [ex.value for ex in self.exchanges],
            "parameters": self.parameters,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "position_sizing": self.position_sizing,
            "risk_management": self.risk_management,
            "backtest_results": self.backtest_results,
            "performance_metrics": self.performance_metrics,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingStrategy':
        """Create from dictionary."""
        return cls(
            strategy_id=data["strategy_id"],
            name=data["name"],
            type=StrategyType(data["type"]),
            description=data["description"],
            risk_level=RiskLevel(data["risk_level"]),
            symbols=data["symbols"],
            timeframes=[TimeFrame(tf) for tf in data["timeframes"]],
            exchanges=[ExchangeType(ex) for ex in data["exchanges"]],
            parameters=data.get("parameters", {}),
            entry_conditions=data.get("entry_conditions", []),
            exit_conditions=data.get("exit_conditions", []),
            position_sizing=data.get("position_sizing", {}),
            risk_management=data.get("risk_management", {}),
            backtest_results=data.get("backtest_results", {}),
            performance_metrics=data.get("performance_metrics", {}),
            is_active=data.get("is_active", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {})
        )
    
    def save(self, directory: Path = STRATEGIES_DIR) -> Path:
        """Save trading strategy to file."""
        directory.mkdir(exist_ok=True, parents=True)
        filepath = directory / f"{self.strategy_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'TradingStrategy':
        """Load trading strategy from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class TradePosition:
    """Represents an open trading position."""
    position_id: str
    strategy_id: str
    symbol: str
    exchange: ExchangeType
    direction: TradeDirection
    entry_price: float
    current_price: float
    quantity: float
    leverage: float = 1.0  # 1.0 means no leverage
    margin: float = 0.0  # Required for margin/futures trading
    liquidation_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_percentage: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "position_id": self.position_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "exchange": self.exchange.value,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "margin": self.margin,
            "liquidation_price": self.liquidation_price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_percentage": self.unrealized_pnl_percentage,
            "entry_time": self.entry_time.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradePosition':
        """Create from dictionary."""
        return cls(
            position_id=data["position_id"],
            strategy_id=data["strategy_id"],
            symbol=data["symbol"],
            exchange=ExchangeType(data["exchange"]),
            direction=TradeDirection(data["direction"]),
            entry_price=data["entry_price"],
            current_price=data["current_price"],
            quantity=data["quantity"],
            leverage=data.get("leverage", 1.0),
            margin=data.get("margin", 0.0),
            liquidation_price=data.get("liquidation_price", 0.0),
            take_profit=data.get("take_profit", 0.0),
            stop_loss=data.get("stop_loss", 0.0),
            unrealized_pnl=data.get("unrealized_pnl", 0.0),
            unrealized_pnl_percentage=data.get("unrealized_pnl_percentage", 0.0),
            entry_time=datetime.fromisoformat(data["entry_time"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {})
        )

@dataclass
class TradeOrder:
    """Represents a cryptocurrency trade order."""
    order_id: str
    exchange_order_id: str = ""
    strategy_id: str = ""
    symbol: str = ""
    exchange: Optional[ExchangeType] = None
    order_type: OrderType = OrderType.MARKET
    direction: TradeDirection = TradeDirection.LONG
    quantity: float = 0.0
    price: float = 0.0
    stop_price: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    status: str = "pending"  # pending, open, filled, cancelled, rejected
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fee: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "order_id": self.order_id,
            "exchange_order_id": self.exchange_order_id,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "exchange": self.exchange.value if self.exchange else None,
            "order_type": self.order_type.value,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "take_profit": self.take_profit,
            "stop_loss": self.stop_loss,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "fee": self.fee,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeOrder':
        """Create from dictionary."""
        return cls(
            order_id=data["order_id"],
            exchange_order_id=data.get("exchange_order_id", ""),
            strategy_id=data.get("strategy_id", ""),
            symbol=data.get("symbol", ""),
            exchange=ExchangeType(data["exchange"]) if data.get("exchange") else None,
            order_type=OrderType(data.get("order_type", "market")),
            direction=TradeDirection(data.get("direction", "long")),
            quantity=data.get("quantity", 0.0),
            price=data.get("price", 0.0),
            stop_price=data.get("stop_price", 0.0),
            take_profit=data.get("take_profit", 0.0),
            stop_loss=data.get("stop_loss", 0.0),
            status=data.get("status", "pending"),
            filled_quantity=data.get("filled_quantity", 0.0),
            average_fill_price=data.get("average_fill_price", 0.0),
            fee=data.get("fee", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            filled_at=datetime.fromisoformat(data["filled_at"]) if data.get("filled_at") else None,
            metadata=data.get("metadata", {})
        )

class CryptoManager:
    """Manager for cryptocurrency operations and trading."""
    
    def __init__(self):
        self.wallets: Dict[str, CryptoWallet] = {}
        self.transactions: Dict[str, List[CryptoTransaction]] = {}  # wallet_id -> transactions
        self.exchange_accounts: Dict[str, ExchangeAccount] = {}
        self.market_data_cache: Dict[str, Dict[str, List[MarketData]]] = {}  # symbol -> timeframe -> data
        self.strategies: Dict[str, TradingStrategy] = {}
        self.active_positions: Dict[str, TradePosition] = {}
        self.orders: Dict[str, TradeOrder] = {}
        
        self.active = False
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # Queues for async operations
        self.market_data_queue: queue.Queue = queue.Queue()
        self.order_queue: queue.Queue = queue.Queue()
        self.wallet_update_queue: queue.Queue = queue.Queue()
        
        # Exchange clients
        self.exchange_clients: Dict[str, Any] = {}
        
        # Web3 providers
        self.web3_providers: Dict[str, Any] = {}
        
        # Load existing data
        self.load_all_data()
        
        logger.info("CryptoManager initialized")
    
    def load_all_data(self) -> None:
        """Load all cryptocurrency data from files."""
        # Load wallets
        if WALLETS_DIR.exists():
            for filepath in WALLETS_DIR.glob("*.json"):
                try:
                    wallet = CryptoWallet.load(filepath)
                    self.wallets[wallet.wallet_id] = wallet
                    logger.info(f"Loaded wallet: {wallet.name} ({wallet.wallet_id})")
                except Exception as e:
                    logger.error(f"Error loading wallet from {filepath}: {str(e)}")
        
        # Load transactions (most recent 100 per wallet)
        if TRANSACTIONS_DIR.exists():
            for wallet_dir in TRANSACTIONS_DIR.iterdir():
                if wallet_dir.is_dir():
                    wallet_id = wallet_dir.name
                    self.transactions[wallet_id] = []
                    
                    # Get all transaction files
                    transaction_files = []
                    for year_dir in wallet_dir.iterdir():
                        if year_dir.is_dir():
                            for month_dir in year_dir.iterdir():
                                if month_dir.is_dir():
                                    transaction_files.extend(month_dir.glob("*.json"))
                    
                    # Sort by modification time (newest first) and take first 100
                    transaction_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    transaction_files = transaction_files[:100]
                    
                    for filepath in transaction_files:
                        try:
                            transaction = CryptoTransaction.load(filepath)
                            self.transactions[wallet_id].append(transaction)
                            logger.debug(f"Loaded transaction: {transaction.transaction_id}")
                        except Exception as e:
                            logger.error(f"Error loading transaction from {filepath}: {str(e)}")
        
        # Load exchange accounts
        if EXCHANGE_DIR.exists():
            for exchange_dir in EXCHANGE_DIR.iterdir():
                if exchange_dir.is_dir():
                    for filepath in exchange_dir.glob("*.json"):
                        try:
                            account = ExchangeAccount.load(filepath)
                            self.exchange_accounts[account.account_id] = account
                            logger.info(f"Loaded exchange account: {account.name} ({account.exchange.value})")
                        except Exception as e:
                            logger.error(f"Error loading exchange account from {filepath}: {str(e)}")
        
        # Load trading strategies
        if STRATEGIES_DIR.exists():
            for filepath in STRATEGIES_DIR.glob("*.json"):
                try:
                    strategy = TradingStrategy.load(filepath)
                    self.strategies[strategy.strategy_id] = strategy
                    logger.info(f"Loaded trading strategy: {strategy.name} ({strategy.strategy_id})")
                except Exception as e:
                    logger.error(f"Error loading trading strategy from {filepath}: {str(e)}")
    
    def start(self) -> bool:
        """Start the crypto manager."""
        if self.active:
            logger.warning("CryptoManager is already running")
            return False
        
        self.active = True
        self.stop_event.clear()
        
        # Initialize exchange clients
        self.initialize_exchange_clients()
        
        # Initialize Web3 providers
        self.initialize_web3_providers()
        
        # Start worker threads
        self.start_workers()
        
        logger.info("CryptoManager started")
        return True
    
    def stop(self) -> bool:
        """Stop the crypto manager."""
        if not self.active:
            logger.warning("CryptoManager is not running")
            return False
        
        self.active = False
        self.stop_event.set()
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.worker_threads = []
        
        logger.info("CryptoManager stopped")
        return True
    
    def initialize_exchange_clients(self) -> None:
        """Initialize clients for cryptocurrency exchanges."""
        if not CCXT_AVAILABLE:
            logger.warning("CCXT not available, exchange functionality will be limited")
            return
        
        for account_id, account in self.exchange_accounts.items():
            if not account.is_active:
                continue
            
            try:
                exchange_id = account.exchange.value
                
                # Initialize CCXT exchange
                exchange_class = getattr(ccxt, exchange_id)
                exchange_client = exchange_class({
                    'apiKey': account.api_key,
                    'secret': account.api_secret,
                    'password': account.passphrase if account.passphrase else None,
                    'enableRateLimit': True
                })
                
                # Set up subaccount if needed
                if account.subaccount:
                    if hasattr(exchange_client, 'options'):
                        exchange_client.options['defaultSubaccount'] = account.subaccount
                
                self.exchange_clients[account_id] = exchange_client
                logger.info(f"Initialized exchange client for {account.name} ({exchange_id})")
            
            except Exception as e:
                logger.error(f"Error initializing exchange client for {account.name}: {str(e)}")
    
    def initialize_web3_providers(self) -> None:
        """Initialize Web3 providers for blockchain networks."""
        if not WEB3_AVAILABLE:
            logger.warning("Web3 not available, blockchain functionality will be limited")
            return
        
        # Initialize Ethereum provider
        try:
            eth_provider = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/your-infura-key"))
            if eth_provider.is_connected():
                self.web3_providers[NetworkType.ETHEREUM.value] = eth_provider
                logger.info("Initialized Ethereum Web3 provider")
        except Exception as e:
            logger.error(f"Error initializing Ethereum Web3 provider: {str(e)}")
        
        # Initialize Binance Smart Chain provider
        try:
            bsc_provider = Web3(Web3.HTTPProvider("https://bsc-dataseed.binance.org/"))
            if bsc_provider.is_connected():
                self.web3_providers[NetworkType.BINANCE_SMART_CHAIN.value] = bsc_provider
                logger.info("Initialized Binance Smart Chain Web3 provider")
        except Exception as e:
            logger.error(f"Error initializing Binance Smart Chain Web3 provider: {str(e)}")
        
        # Other networks could be added similarly
    
    def start_workers(self) -> None:
        """Start worker threads for various cryptocurrency tasks."""
        # Market data worker
        market_data_worker = threading.Thread(
            target=self.market_data_worker,
            name="MarketDataWorker",
            daemon=True
        )
        self.worker_threads.append(market_data_worker)
        market_data_worker.start()
        
        # Order processing worker
        order_worker = threading.Thread(
            target=self.order_worker,
            name="OrderWorker",
            daemon=True
        )
        self.worker_threads.append(order_worker)
        order_worker.start()
        
        # Wallet update worker
        wallet_worker = threading.Thread(
            target=self.wallet_update_worker,
            name="WalletUpdateWorker",
            daemon=True
        )
        self.worker_threads.append(wallet_worker)
        wallet_worker.start()
        
        # Strategy execution worker
        strategy_worker = threading.Thread(
            target=self.strategy_worker,
            name="StrategyWorker",
            daemon=True
        )
        self.worker_threads.append(strategy_worker)
        strategy_worker.start()
        
        # Position monitoring worker
        position_worker = threading.Thread(
            target=self.position_monitoring_worker,
            name="PositionMonitoringWorker",
            daemon=True
        )
        self.worker_threads.append(position_worker)
        position_worker.start()
        
        logger.info("Started worker threads")
    
    def market_data_worker(self) -> None:
        """Worker thread for fetching and processing market data."""
        logger.info("Market data worker started")
        
        while not self.stop_event.is_set():
            try:
                # Process market data requests from queue
                try:
                    market_data_task = self.market_data_queue.get(timeout=1.0)
                    
                    symbol = market_data_task.get("symbol")
                    exchange_id = market_data_task.get("exchange")
                    timeframe = market_data_task.get("timeframe")
                    
                    # Find an exchange client for this exchange
                    exchange_client = None
                    for account_id, client in self.exchange_clients.items():
                        if self.exchange_accounts[account_id].exchange.value == exchange_id:
                            exchange_client = client
                            break
                    
                    if exchange_client:
                        # Fetch OHLCV data
                        ohlcv = exchange_client.fetch_ohlcv(symbol, timeframe)
                        
                        # Process and store data
                        data_list = []
                        for candle in ohlcv:
                            timestamp, open_price, high, low, close, volume = candle
                            
                            market_data = MarketData(
                                symbol=symbol,
                                exchange=ExchangeType(exchange_id),
                                timeframe=TimeFrame(timeframe),
                                timestamp=datetime.fromtimestamp(timestamp / 1000),
                                open=open_price,
                                high=high,
                                low=low,
                                close=close,
                                volume=volume
                            )
                            
                            data_list.append(market_data)
                        
                        # Add technical indicators if available
                        if TA_AVAILABLE and data_list:
                            self.add_technical_indicators(data_list)
                        
                        # Store in cache
                        if symbol not in self.market_data_cache:
                            self.market_data_cache[symbol] = {}
                        
                        self.market_data_cache[symbol][timeframe] = data_list
                        
                        logger.info(f"Fetched {len(data_list)} candles for {symbol} {timeframe} from {exchange_id}")
                    
                    else:
                        logger.warning(f"No exchange client available for {exchange_id}")
                    
                    self.market_data_queue.task_done()
                
                except queue.Empty:
                    pass
                
                # Periodically update market data for active strategies
                for strategy_id, strategy in self.strategies.items():
                    if strategy.is_active:
                        for symbol in strategy.symbols:
                            for exchange in strategy.exchanges:
                                for timeframe in strategy.timeframes:
                                    self.market_data_queue.put({
                                        "symbol": symbol,
                                        "exchange": exchange.value,
                                        "timeframe": timeframe.value
                                    })
            
            except Exception as e:
                logger.error(f"Error in market data worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(10)  # Check every 10 seconds
    
    def add_technical_indicators(self, data_list: List[MarketData]) -> None:
        """Add technical indicators to market data."""
        if not TA_AVAILABLE or not data_list:
            return
        
        # Convert to pandas DataFrame
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'open': d.open,
                'high': d.high,
                'low': d.low,
                'close': d.close,
                'volume': d.volume
            }
            for d in data_list
        ])
        
        # Add RSI
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        except:
            pass
        
        # Add MACD
        try:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        except:
            pass
        
        # Add Bollinger Bands
        try:
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['bollinger_mavg'] = bollinger.bollinger_mavg()
            df['bollinger_hband'] = bollinger.bollinger_hband()
            df['bollinger_lband'] = bollinger.bollinger_lband()
        except:
            pass
        
        # Add moving averages
        try:
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
            df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
        except:
            pass
        
        # Add ATR
        try:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        except:
            pass
        
        # Update market data with indicators
        for i, row in df.iterrows():
            indicators = {}
            for col in df.columns:
                if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] and not pd.isna(row[col]):
                    indicators[col] = row[col]
            
            data_list[i].indicators = indicators
    
    def order_worker(self) -> None:
        """Worker thread for processing cryptocurrency orders."""
        logger.info("Order worker started")
        
        while not self.stop_event.is_set():
            try:
                # Process order requests from queue
                try:
                    order_task = self.order_queue.get(timeout=1.0)
                    
                    task_type = order_task.get("type")
                    
                    if task_type == "create":
                        # Create a new order
                        order = TradeOrder(
                            order_id=order_task.get("order_id", f"order-{str(uuid.uuid4())[:8]}"),
                            strategy_id=order_task.get("strategy_id", ""),
                            symbol=order_task.get("symbol"),
                            exchange=ExchangeType(order_task.get("exchange")),
                            order_type=OrderType(order_task.get("order_type", "market")),
                            direction=TradeDirection(order_task.get("direction", "long")),
                            quantity=order_task.get("quantity"),
                            price=order_task.get("price", 0.0),
                            stop_price=order_task.get("stop_price", 0.0),
                            take_profit=order_task.get("take_profit", 0.0),
                            stop_loss=order_task.get("stop_loss", 0.0),
                            metadata=order_task.get("metadata", {})
                        )
                        
                        # Find exchange account for this exchange
                        account_id = None
                        for acc_id, account in self.exchange_accounts.items():
                            if account.exchange == order.exchange:
                                account_id = acc_id
                                break
                        
                        if account_id and account_id in self.exchange_clients:
                            exchange_client = self.exchange_clients[account_id]
                            
                            # Prepare order parameters
                            params = {}
                            
                            if order.take_profit > 0:
                                params["takeProfit"] = order.take_profit
                            
                            if order.stop_loss > 0:
                                params["stopLoss"] = order.stop_loss
                            
                            try:
                                # Place order on exchange
                                if order.order_type == OrderType.MARKET:
                                    if order.direction == TradeDirection.LONG:
                                        result = exchange_client.create_market_buy_order(
                                            symbol=order.symbol,
                                            amount=order.quantity,
                                            params=params
                                        )
                                    else:
                                        result = exchange_client.create_market_sell_order(
                                            symbol=order.symbol,
                                            amount=order.quantity,
                                            params=params
                                        )
                                
                                elif order.order_type == OrderType.LIMIT:
                                    if order.direction == TradeDirection.LONG:
                                        result = exchange_client.create_limit_buy_order(
                                            symbol=order.symbol,
                                            amount=order.quantity,
                                            price=order.price,
                                            params=params
                                        )
                                    else:
                                        result = exchange_client.create_limit_sell_order(
                                            symbol=order.symbol,
                                            amount=order.quantity,
                                            price=order.price,
                                            params=params
                                        )
                                
                                # Update order with exchange data
                                order.exchange_order_id = result.get("id", "")
                                order.status = "open"
                                
                                # Store order
                                self.orders[order.order_id] = order
                                
                                logger.info(f"Placed {order.order_type.value} {order.direction.value} order for {order.quantity} {order.symbol}")
                            
                            except Exception as e:
                                order.status = "rejected"
                                order.metadata["error"] = str(e)
                                logger.error(f"Error placing order: {str(e)}")
                        
                        else:
                            order.status = "rejected"
                            order.metadata["error"] = "No exchange client available"
                            logger.warning(f"No exchange client available for {order.exchange.value}")
                    
                    elif task_type == "cancel":
                        # Cancel an existing order
                        order_id = order_task.get("order_id")
                        
                        if order_id in self.orders:
                            order = self.orders[order_id]
                            
                            # Find exchange client
                            account_id = None
                            for acc_id, account in self.exchange_accounts.items():
                                if account.exchange == order.exchange:
                                    account_id = acc_id
                                    break
                            
                            if account_id and account_id in self.exchange_clients and order.exchange_order_id:
                                exchange_client = self.exchange_clients[account_id]
                                
                                try:
                                    # Cancel order on exchange
                                    result = exchange_client.cancel_order(order.exchange_order_id, order.symbol)
                                    
                                    # Update order status
                                    order.status = "cancelled"
                                    order.updated_at = datetime.now()
                                    
                                    logger.info(f"Cancelled order {order_id}")
                                
                                except Exception as e:
                                    logger.error(f"Error cancelling order {order_id}: {str(e)}")
                            
                            else:
                                logger.warning(f"Cannot cancel order {order_id}: No exchange client or exchange order ID")
                        
                        else:
                            logger.warning(f"Order {order_id} not found")
                    
                    elif task_type == "update":
                        # Update an existing order
                        order_id = order_task.get("order_id")
                        
                        if order_id in self.orders:
                            order = self.orders[order_id]
                            
                            # Find exchange client
                            account_id = None
                            for acc_id, account in self.exchange_accounts.items():
                                if account.exchange == order.exchange:
                                    account_id = acc_id
                                    break
                            
                            if account_id and account_id in self.exchange_clients and order.exchange_order_id:
                                exchange_client = self.exchange_clients[account_id]
                                
                                try:
                                    # Cancel existing order
                                    exchange_client.cancel_order(order.exchange_order_id, order.symbol)
                                    
                                    # Create new order with updated parameters
                                    new_price = order_task.get("price", order.price)
                                    new_quantity = order_task.get("quantity", order.quantity)
                                    
                                    params = {}
                                    
                                    if order_task.get("take_profit", order.take_profit) > 0:
                                        params["takeProfit"] = order_task.get("take_profit", order.take_profit)
                                    
                                    if order_task.get("stop_loss", order.stop_loss) > 0:
                                        params["stopLoss"] = order_task.get("stop_loss", order.stop_loss)
                                    
                                    # Place updated order
                                    if order.order_type == OrderType.LIMIT:
                                        if order.direction == TradeDirection.LONG:
                                            result = exchange_client.create_limit_buy_order(
                                                symbol=order.symbol,
                                                amount=new_quantity,
                                                price=new_price,
                                                params=params
                                            )
                                        else:
                                            result = exchange_client.create_limit_sell_order(
                                                symbol=order.symbol,
                                                amount=new_quantity,
                                                price=new_price,
                                                params=params
                                            )
                                    
                                    # Update order with new data
                                    order.exchange_order_id = result.get("id", "")
                                    order.price = new_price
                                    order.quantity = new_quantity
                                    order.take_profit = order_task.get("take_profit", order.take_profit)
                                    order.stop_loss = order_task.get("stop_loss", order.stop_loss)
                                    order.updated_at = datetime.now()
                                    
                                    logger.info(f"Updated order {order_id}")
                                
                                except Exception as e:
                                    logger.error(f"Error updating order {order_id}: {str(e)}")
                            
                            else:
                                logger.warning(f"Cannot update order {order_id}: No exchange client or exchange order ID")
                        
                        else:
                            logger.warning(f"Order {order_id} not found")
                    
                    self.order_queue.task_done()
                
                except queue.Empty:
                    pass
                
                # Update order statuses
                self.update_order_statuses()
            
            except Exception as e:
                logger.error(f"Error in order worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(1)
    
    def update_order_statuses(self) -> None:
        """Update the status of open orders."""
        open_orders = [order for order in self.orders.values() if order.status == "open"]
        
        for order in open_orders:
            # Find exchange client
            account_id = None
            for acc_id, account in self.exchange_accounts.items():
                if account.exchange == order.exchange:
                    account_id = acc_id
                    break
            
            if account_id and account_id in self.exchange_clients and order.exchange_order_id:
                exchange_client = self.exchange_clients[account_id]
                
                try:
                    # Fetch order status from exchange
                    order_status = exchange_client.fetch_order(order.exchange_order_id, order.symbol)
                    
                    # Update order
                    order.status = order_status.get("status", order.status)
                    order.filled_quantity = order_status.get("filled", order.filled_quantity)
                    order.average_fill_price = order_status.get("average", order.average_fill_price)
                    order.fee = order_status.get("fee", {}).get("cost", order.fee)
                    order.updated_at = datetime.now()
                    
                    if order.status == "filled" and not order.filled_at:
                        order.filled_at = datetime.now()
                        
                        # If order is filled, create a position
                        if order.strategy_id:
                            position_id = f"pos-{str(uuid.uuid4())[:8]}"
                            
                            position = TradePosition(
                                position_id=position_id,
                                strategy_id=order.strategy_id,
                                symbol=order.symbol,
                                exchange=order.exchange,
                                direction=order.direction,
                                entry_price=order.average_fill_price,
                                current_price=order.average_fill_price,
                                quantity=order.filled_quantity,
                                take_profit=order.take_profit,
                                stop_loss=order.stop_loss,
                                metadata={
                                    "order_id": order.order_id,
                                    "exchange_order_id": order.exchange_order_id
                                }
                            )
                            
                            self.active_positions[position_id] = position
                            logger.info(f"Created position {position_id} from filled order {order.order_id}")
                
                except Exception as e:
                    logger.error(f"Error updating order status for {order.order_id}: {str(e)}")
    
    def wallet_update_worker(self) -> None:
        """Worker thread for updating wallet balances."""
        logger.info("Wallet update worker started")
        
        while not self.stop_event.is_set():
            try:
                # Process wallet update requests from queue
                try:
                    wallet_task = self.wallet_update_queue.get(timeout=1.0)
                    
                    task_type = wallet_task.get("type")
                    
                    if task_type == "update_balance":
                        # Update wallet balance
                        wallet_id = wallet_task.get("wallet_id")
                        
                        if wallet_id in self.wallets:
                            wallet = self.wallets[wallet_id]
                            
                            if wallet.type == WalletType.EXCHANGE:
                                # Update exchange wallet balance
                                exchange_account_id = wallet.metadata.get("exchange_account_id")
                                
                                if exchange_account_id in self.exchange_accounts and exchange_account_id in self.exchange_clients:
                                    exchange_client = self.exchange_clients[exchange_account_id]
                                    
                                    try:
                                        # Fetch balances from exchange
                                        balance_data = exchange_client.fetch_balance()
                                        
                                        # Update wallet balances
                                        for crypto_type in CryptoType:
                                            symbol = crypto_type.value
                                            if symbol in balance_data.get("total", {}):
                                                wallet.balances[symbol] = balance_data["total"][symbol]
                                        
                                        wallet.last_updated = datetime.now()
                                        wallet.save()
                                        
                                        logger.info(f"Updated exchange wallet balance for {wallet.name}")
                                    
                                    except Exception as e:
                                        logger.error(f"Error updating exchange wallet balance: {str(e)}")
                            
                            elif wallet.type in [WalletType.HOT, WalletType.COLD, WalletType.HARDWARE]:
                                # Update blockchain wallet balance
                                for address, addr_obj in wallet.addresses.items():
                                    network = addr_obj.network
                                    crypto_type = addr_obj.crypto_type
                                    
                                    if network.value in self.web3_providers:
                                        web3 = self.web3_providers[network.value]
                                        
                                        try:
                                            # Get balance from blockchain
                                            balance_wei = web3.eth.get_balance(address)
                                            balance_eth = web3.from_wei(balance_wei, 'ether')
                                            
                                            wallet.balances[crypto_type.value] = float(balance_eth)
                                            addr_obj.last_used = datetime.now()
                                            
                                            logger.info(f"Updated blockchain wallet balance for {wallet.name} - {crypto_type.value}")
                                        
                                        except Exception as e:
                                            logger.error(f"Error updating blockchain wallet balance: {str(e)}")
                    
                    elif task_type == "scan_transactions":
                        # Scan for new transactions
                        wallet_id = wallet_task.get("wallet_id")
                        
                        if wallet_id in self.wallets:
                            wallet = self.wallets[wallet_id]
                            
                            # Get the latest transaction timestamp
                            latest_tx_time = datetime.fromtimestamp(0)
                            if wallet_id in self.transactions and self.transactions[wallet_id]:
                                latest_tx_time = max(tx.timestamp for tx in self.transactions[wallet_id])
                            
                            if wallet.type == WalletType.EXCHANGE:
                                # Scan exchange transactions
                                exchange_account_id = wallet.metadata.get("exchange_account_id")
                                
                                if exchange_account_id in self.exchange_accounts and exchange_account_id in self.exchange_clients:
                                    exchange_client = self.exchange_clients[exchange_account_id]
                                    
                                    try:
                                        # Fetch recent transactions
                                        since = int(latest_tx_time.timestamp() * 1000)
                                        deposits = exchange_client.fetch_deposits(None, since)
                                        withdrawals = exchange_client.fetch_withdrawals(None, since)
                                        
                                        # Process deposits
                                        for deposit in deposits:
                                            tx_id = f"tx-{str(uuid.uuid4())[:8]}"
                                            
                                            tx = CryptoTransaction(
                                                transaction_id=tx_id,
                                                wallet_id=wallet_id,
                                                type=TransactionType.DEPOSIT,
                                                crypto_type=CryptoType(deposit.get("currency")),
                                                amount=deposit.get("amount", 0.0),
                                                fee=deposit.get("fee", {}).get("cost", 0.0),
                                                timestamp=datetime.fromtimestamp(deposit.get("timestamp") / 1000),
                                                status=deposit.get("status", "pending"),
                                                from_address=deposit.get("address", ""),
                                                to_address=deposit.get("addressTo", ""),
                                                transaction_hash=deposit.get("txid", ""),
                                                network=NetworkType.OTHER,  # Would need mapping
                                                exchange=wallet.metadata.get("exchange_type"),
                                                metadata={
                                                    "exchange_data": deposit
                                                }
                                            )
                                            
                                            # Save transaction
                                            tx.save()
                                            
                                            # Add to memory
                                            if wallet_id not in self.transactions:
                                                self.transactions[wallet_id] = []
                                            
                                            self.transactions[wallet_id].append(tx)
                                            logger.info(f"Recorded deposit transaction {tx_id}")
                                        
                                        # Process withdrawals
                                        for withdrawal in withdrawals:
                                            tx_id = f"tx-{str(uuid.uuid4())[:8]}"
                                            
                                            tx = CryptoTransaction(
                                                transaction_id=tx_id,
                                                wallet_id=wallet_id,
                                                type=TransactionType.WITHDRAWAL,
                                                crypto_type=CryptoType(withdrawal.get("currency")),
                                                amount=withdrawal.get("amount", 0.0),
                                                fee=withdrawal.get("fee", {}).get("cost", 0.0),
                                                timestamp=datetime.fromtimestamp(withdrawal.get("timestamp") / 1000),
                                                status=withdrawal.get("status", "pending"),
                                                from_address=withdrawal.get("address", ""),
                                                to_address=withdrawal.get("addressTo", ""),
                                                transaction_hash=withdrawal.get("txid", ""),
                                                network=NetworkType.OTHER,  # Would need mapping
                                                exchange=wallet.metadata.get("exchange_type"),
                                                metadata={
                                                    "exchange_data": withdrawal
                                                }
                                            )
                                            
                                            # Save transaction
                                            tx.save()
                                            
                                            # Add to memory
                                            if wallet_id not in self.transactions:
                                                self.transactions[wallet_id] = []
                                            
                                            self.transactions[wallet_id].append(tx)
                                            logger.info(f"Recorded withdrawal transaction {tx_id}")
                                    
                                    except Exception as e:
                                        logger.error(f"Error scanning exchange transactions: {str(e)}")
                            
                            elif wallet.type in [WalletType.HOT, WalletType.COLD, WalletType.HARDWARE]:
                                # Scan blockchain transactions
                                for address, addr_obj in wallet.addresses.items():
                                    network = addr_obj.network
                                    crypto_type = addr_obj.crypto_type
                                    
                                    if network.value in self.web3_providers:
                                        web3 = self.web3_providers[network.value]
                                        
                                        try:
                                            # Get transaction count
                                            tx_count = web3.eth.get_transaction_count(address)
                                            
                                            # Fetch recent blocks
                                            latest_block = web3.eth.block_number
                                            start_block = latest_block - 1000  # Look back 1000 blocks
                                            
                                            # This is simplified - in a real implementation, you'd use an indexer or API
                                            # to efficiently fetch transactions for an address
                                            
                                            logger.info(f"Scanned blockchain transactions for {wallet.name} - {address}")
                                        
                                        except Exception as e:
                                            logger.error(f"Error scanning blockchain transactions: {str(e)}")
                    
                    self.wallet_update_queue.task_done()
                
                except queue.Empty:
                    pass
                
                # Periodically update all wallet balances
                for wallet_id in self.wallets:
                    self.wallet_update_queue.put({
                        "type": "update_balance",
                        "wallet_id": wallet_id
                    })
                    
                    self.wallet_update_queue.put({
                        "type": "scan_transactions",
                        "wallet_id": wallet_id
                    })
            
            except Exception as e:
                logger.error(f"Error in wallet update worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(60)  # Update every minute
    
    def strategy_worker(self) -> None:
        """Worker thread for executing trading strategies."""
        logger.info("Strategy worker started")
        
        while not self.stop_event.is_set():
            try:
                # Execute active strategies
                for strategy_id, strategy in self.strategies.items():
                    if strategy.is_active:
                        try:
                            # Execute strategy
                            self.execute_strategy(strategy)
                        except Exception as e:
                            logger.error(f"Error executing strategy {strategy.name}: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error in strategy worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(10)  # Check every 10 seconds
    
    def execute_strategy(self, strategy: TradingStrategy) -> None:
        """Execute a trading strategy."""
        # Check if we have market data for this strategy
        for symbol in strategy.symbols:
            for timeframe in strategy.timeframes:
                timeframe_str = timeframe.value
                
                if symbol not in self.market_data_cache or timeframe_str not in self.market_data_cache[symbol]:
                    # Request market data
                    for exchange in strategy.exchanges:
                        self.market_data_queue.put({
                            "symbol": symbol,
                            "exchange": exchange.value,
                            "timeframe": timeframe_str
                        })
                    
                    # Skip execution until we have data
                    logger.info(f"Waiting for market data for {symbol} {timeframe_str}")
                    return
        
        # Execute strategy based on type
        if strategy.type == StrategyType.TREND_FOLLOWING:
            self.execute_trend_following_strategy(strategy)
        elif strategy.type == StrategyType.MEAN_REVERSION:
            self.execute_mean_reversion_strategy(strategy)
        elif strategy.type == StrategyType.BREAKOUT:
            self.execute_breakout_strategy(strategy)
        elif strategy.type == StrategyType.MOMENTUM:
            self.execute_momentum_strategy(strategy)
        elif strategy.type == StrategyType.GRID_TRADING:
            self.execute_grid_trading_strategy(strategy)
        else:
            logger.warning(f"Strategy type {strategy.type.value} not implemented")
    
    def execute_trend_following_strategy(self, strategy: TradingStrategy) -> None:
        """Execute a trend following strategy."""
        for symbol in strategy.symbols:
            # Get the primary timeframe data
            primary_timeframe = strategy.timeframes[0].value
            
            if symbol in self.market_data_cache and primary_timeframe in self.market_data_cache[symbol]:
                data = self.market_data_cache[symbol][primary_timeframe]
                
                if not data:
                    continue
                
                # Get the latest candle
                latest = data[-1]
                
                # Check for trend indicators
                if "ema_12" in latest.indicators and "ema_26" in latest.indicators:
                    ema_12 = latest.indicators["ema_12"]
                    ema_26 = latest.indicators["ema_26"]
                    
                    # Check for crossover (short-term EMA crosses above long-term EMA)
                    if len(data) > 1:
                        prev = data[-2]
                        prev_ema_12 = prev.indicators.get("ema_12", 0)
                        prev_ema_26 = prev.indicators.get("ema_26", 0)
                        
                        # Bullish crossover
                        if prev_ema_12 <= prev_ema_26 and ema_12 > ema_26:
                            # Check if we already have a position for this symbol
                            has_position = False
                            for pos in self.active_positions.values():
                                if pos.symbol == symbol and pos.strategy_id == strategy.strategy_id:
                                    has_position = True
                                    break
                            
                            if not has_position:
                                # Calculate position size based on risk management
                                account_balance = self.get_account_balance(strategy.exchanges[0])
                                risk_percentage = strategy.risk_management.get("risk_per_trade", 0.01)  # Default 1%
                                position_size = account_balance * risk_percentage
                                
                                # Calculate quantity
                                price = latest.close
                                quantity = position_size / price
                                
                                # Apply minimum trade size
                                min_trade = strategy.parameters.get("min_trade_size", 0.0)
                                if quantity < min_trade:
                                    logger.info(f"Trade size too small:
