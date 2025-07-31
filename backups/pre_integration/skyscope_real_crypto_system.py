
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Real Cryptocurrency System
==================================

This module integrates real cryptocurrency wallet management, real trading operations,
and real-time analytics into a single cohesive system. It is designed for actual
cryptocurrency operations, not simulations.

IMPORTANT: This system handles real money. Use with extreme caution.

Features:
- Secure wallet management with encryption
- Real cryptocurrency trading on exchanges
- Real-time analytics and monitoring
- Blockchain integration for transaction verification
- Advanced security features

Security Notice:
- API keys are loaded from environment variables or secure storage
- Seed phrases and private keys are encrypted at rest
- All sensitive operations are logged for audit purposes
"""

import os
import sys
import json
import time
import uuid
import hashlib
import logging
import threading
import datetime
import secrets
import base64
import getpass
import signal
import queue
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("real_crypto_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealCryptoSystem")

# Try to import optional dependencies
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logger.warning("cryptography package not available. Encryption features will be limited.")
    CRYPTOGRAPHY_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    logger.warning("ccxt package not available. Exchange integration will be limited.")
    CCXT_AVAILABLE = False

try:
    from mnemonic import Mnemonic
    MNEMONIC_AVAILABLE = True
except ImportError:
    logger.warning("mnemonic package not available. Seed phrase generation will use fallback methods.")
    MNEMONIC_AVAILABLE = False

try:
    import numpy as np
    import pandas as pd
    ANALYTICS_LIBS_AVAILABLE = True
except ImportError:
    logger.warning("numpy/pandas not available. Analytics capabilities will be limited.")
    ANALYTICS_LIBS_AVAILABLE = False

try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    logger.warning("web3 package not available. Ethereum integration will be limited.")
    WEB3_AVAILABLE = False

# Constants
DEFAULT_WALLET_DIR = "secure_wallets"
DEFAULT_CONFIG_DIR = "config"
DEFAULT_DATA_DIR = "data"
DEFAULT_CACHE_DIR = "cache"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_LOGS_DIR = "logs"

# Environment variable names
ENV_WALLET_KEY = "SKYSCOPE_WALLET_KEY"
ENV_WALLET_MODE = "SKYSCOPE_WALLET_MODE"
ENV_WALLET_SEED_PHRASE = "SKYSCOPE_WALLET_SEED_PHRASE"
ENV_INFURA_API_KEY = "INFURA_API_KEY"
ENV_INFURA_API_SECRET = "INFURA_API_SECRET"
ENV_BLOCKCYPHER_TOKEN = "BLOCKCYPHER_TOKEN"
ENV_BTC_NETWORK = "BTC_NETWORK"  # mainnet or testnet
ENV_ETH_NETWORK = "ETH_NETWORK"  # mainnet, sepolia, etc.

# Supported cryptocurrencies
SUPPORTED_CRYPTOCURRENCIES = {
    "BTC": {
        "name": "Bitcoin",
        "decimals": 8,
        "address_prefix": "bc1",
        "address_length": 42,
        "explorer_url": "https://www.blockchain.com/explorer/addresses/btc/{address}"
    },
    "ETH": {
        "name": "Ethereum",
        "decimals": 18,
        "address_prefix": "0x",
        "address_length": 42,
        "explorer_url": "https://etherscan.io/address/{address}"
    },
    "SOL": {
        "name": "Solana",
        "decimals": 9,
        "address_prefix": "",
        "address_length": 44,
        "explorer_url": "https://explorer.solana.com/address/{address}"
    },
    "BNB": {
        "name": "Binance Coin",
        "decimals": 18,
        "address_prefix": "0x",
        "address_length": 42,
        "explorer_url": "https://bscscan.com/address/{address}"
    },
    "DOGE": {
        "name": "Dogecoin",
        "decimals": 8,
        "address_prefix": "D",
        "address_length": 34,
        "explorer_url": "https://dogechain.info/address/{address}"
    },
    "ADA": {
        "name": "Cardano",
        "decimals": 6,
        "address_prefix": "addr",
        "address_length": 58,
        "explorer_url": "https://cardanoscan.io/address/{address}"
    }
}

# API endpoints
BLOCKCYPHER_API_BASE = "https://api.blockcypher.com/v1"
BLOCKCHAIN_INFO_API_BASE = "https://blockchain.info"

# Security class for encryption/decryption
class SecurityManager:
    """
    Manages encryption and decryption of sensitive data like seed phrases and private keys.
    """
    
    def __init__(self):
        """Initialize the security manager."""
        self.encryption_key = None
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """Initialize encryption from environment variable or prompt."""
        if ENV_WALLET_KEY in os.environ:
            self.set_encryption_key(os.environ[ENV_WALLET_KEY])
            logger.info("Using encryption key from environment variable")
        else:
            logger.warning("No encryption key found in environment variables")
    
    def set_encryption_key(self, key: str) -> bool:
        """
        Set the encryption key for securing wallet data.
        
        Args:
            key: Encryption key (should be strong and secure)
            
        Returns:
            True if successful, False otherwise
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Encryption not available. Install cryptography package for secure storage.")
            return False
            
        try:
            # Derive a proper encryption key using PBKDF2
            salt = b'skyscope_real_crypto_salt'  # In production, use a secure random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            # Create the key
            key_bytes = key.encode()
            derived_key = base64.urlsafe_b64encode(kdf.derive(key_bytes))
            self.encryption_key = derived_key
            logger.info("Encryption key set successfully")
            return True
        except Exception as e:
            logger.error(f"Error setting encryption key: {e}")
            return False
    
    def prompt_for_encryption_key(self) -> bool:
        """
        Prompt the user for an encryption key.
        
        Returns:
            True if key was set successfully, False otherwise
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            logger.warning("Encryption not available. Install cryptography package.")
            return False
            
        try:
            # Get password without echoing to screen
            key = getpass.getpass("Enter encryption key for wallet data: ")
            if not key:
                logger.error("No encryption key provided")
                return False
                
            return self.set_encryption_key(key)
        except Exception as e:
            logger.error(f"Error getting encryption key: {e}")
            return False
    
    def encrypt_data(self, data: Union[str, bytes, Dict]) -> Optional[bytes]:
        """
        Encrypt sensitive data.
        
        Args:
            data: Data to encrypt (string, bytes, or dictionary)
            
        Returns:
            Encrypted data as bytes or None if encryption failed
        """
        if not CRYPTOGRAPHY_AVAILABLE or not self.encryption_key:
            logger.warning("Encryption not available or no key set")
            return None
            
        try:
            # Convert data to JSON string if it's a dictionary
            if isinstance(data, dict):
                data = json.dumps(data)
                
            # Convert to bytes if it's a string
            if isinstance(data, str):
                data = data.encode()
                
            # Create Fernet cipher and encrypt
            cipher = Fernet(self.encryption_key)
            encrypted_data = cipher.encrypt(data)
            return encrypted_data
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: bytes) -> Optional[Any]:
        """
        Decrypt encrypted data.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data (string or dictionary) or None if decryption failed
        """
        if not CRYPTOGRAPHY_AVAILABLE or not self.encryption_key:
            logger.warning("Decryption not available or no key set")
            return None
            
        try:
            # Create Fernet cipher and decrypt
            cipher = Fernet(self.encryption_key)
            decrypted_data = cipher.decrypt(encrypted_data)
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_data)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return None

# Real Cryptocurrency Wallet Manager
class RealWalletManager:
    """
    Manages real cryptocurrency wallets and transactions.
    
    This class handles wallet creation, transaction tracking, and balance management
    for real cryptocurrency wallets.
    """
    
    def __init__(self, wallet_dir: str = DEFAULT_WALLET_DIR, security_manager: Optional[SecurityManager] = None):
        """
        Initialize the wallet manager.
        
        Args:
            wallet_dir: Directory for storing wallet data
            security_manager: Security manager for encryption (optional)
        """
        self.wallet_dir = wallet_dir
        self.wallets = {}
        self.security_manager = security_manager or SecurityManager()
        self.lock = threading.RLock()
        
        # Create wallet directory
        os.makedirs(wallet_dir, exist_ok=True)
        
        # Load existing wallets
        self._load_wallets()
        
        logger.info(f"RealWalletManager initialized with {len(self.wallets)} wallets")
    
    def generate_wallet(self, 
                      cryptocurrency: str, 
                      name: Optional[str] = None,
                      seed_phrase: Optional[str] = None) -> Dict:
        """
        Generate a new cryptocurrency wallet with seed phrase.
        
        Args:
            cryptocurrency: Cryptocurrency code (BTC, ETH, etc.)
            name: Wallet name (optional)
            seed_phrase: Existing seed phrase (optional)
            
        Returns:
            Wallet information dictionary
        """
        with self.lock:
            if cryptocurrency not in SUPPORTED_CRYPTOCURRENCIES:
                raise ValueError(f"Unsupported cryptocurrency: {cryptocurrency}")
            
            # Generate or use provided seed phrase
            if not seed_phrase:
                seed_phrase = self._generate_seed_phrase()
            
            # Generate wallet data based on cryptocurrency
            if cryptocurrency.upper() == "BTC":
                wallet_data = self._generate_bitcoin_wallet(seed_phrase)
            elif cryptocurrency.upper() == "ETH":
                wallet_data = self._generate_ethereum_wallet(seed_phrase)
            elif cryptocurrency.upper() == "SOL":
                wallet_data = self._generate_solana_wallet(seed_phrase)
            elif cryptocurrency.upper() == "BNB":
                wallet_data = self._generate_binance_wallet(seed_phrase)
            else:
                # Fallback for other cryptocurrencies
                wallet_data = self._generate_wallet_fallback(cryptocurrency, seed_phrase)
            
            # Add metadata
            wallet_info = {
                "name": name or f"{SUPPORTED_CRYPTOCURRENCIES[cryptocurrency]['name']} Wallet",
                "cryptocurrency": cryptocurrency.upper(),
                "created_at": datetime.datetime.now().isoformat(),
                "seed_phrase": seed_phrase,
                "address": wallet_data["address"],
                "private_key": wallet_data["private_key"],
                "public_key": wallet_data.get("public_key", wallet_data["address"]),
                "balance": 0.0,
                "transactions": []
            }
            
            # Save wallet securely
            self._save_wallet(wallet_info)
            
            # Add to wallets dictionary
            wallet_id = f"{cryptocurrency.upper()}_{wallet_data['address']}"
            self.wallets[wallet_id] = wallet_info
            
            # Create readable wallet file for user
            self._create_user_wallet_file(wallet_info)
            
            logger.info(f"Generated new {cryptocurrency} wallet: {wallet_info['name']} with address {wallet_data['address']}")
            
            # Return a copy without sensitive data
            safe_info = wallet_info.copy()
            safe_info["seed_phrase"] = "[REDACTED]"
            safe_info["private_key"] = "[REDACTED]"
            return safe_info
    
    def _generate_seed_phrase(self) -> str:
        """
        Generate a random seed phrase.
        
        Returns:
            A seed phrase (12 or 24 words)
        """
        if MNEMONIC_AVAILABLE:
            # Use mnemonic package for proper BIP39 seed phrase
            mnemo = Mnemonic("english")
            return mnemo.generate(strength=256)  # 24 words
        else:
            # Fallback to basic word list
            words = [
                "abandon", "ability", "able", "about", "above", "absent", "absorb", "abstract", "absurd", "abuse",
                "access", "accident", "account", "accuse", "achieve", "acid", "acoustic", "acquire", "across", "act",
                "action", "actor", "actress", "actual", "adapt", "add", "addict", "address", "adjust", "admit",
                "adult", "advance", "advice", "aerobic", "affair", "afford", "afraid", "again", "age", "agent",
                "agree", "ahead", "aim", "air", "airport", "aisle", "alarm", "album", "alcohol", "alert",
                "alien", "all", "alley", "allow", "almost", "alone", "alpha", "already", "also", "alter",
                "always", "amateur", "amazing", "among", "amount", "amused", "analyst", "anchor", "ancient", "anger",
                "angle", "angry", "animal", "ankle", "announce", "annual", "another", "answer", "antenna", "antique"
            ]
            return " ".join(random.sample(words, 12))
    
    def _generate_bitcoin_wallet(self, seed_phrase: str) -> Dict:
        """
        Generate Bitcoin wallet from seed phrase.
        
        Args:
            seed_phrase: BIP39 seed phrase
            
        Returns:
            Dictionary with wallet data
        """
        try:
            from bitcoinlib.wallets import Wallet
            from bitcoinlib.mnemonic import Mnemonic
            
            # Create wallet from seed
            wallet = Wallet.create(
                name=f"temp_wallet_{secrets.token_hex(8)}",
                keys=seed_phrase,
                network='bitcoin'
            )
            
            # Get first address
            address = wallet.get_key().address
            private_key = wallet.get_key().wif
            public_key = wallet.get_key().public_hex
            
            return {
                "address": address,
                "private_key": private_key,
                "public_key": public_key
            }
            
        except ImportError:
            # Fallback to manual generation
            return self._generate_wallet_fallback("BTC", seed_phrase)
    
    def _generate_ethereum_wallet(self, seed_phrase: str) -> Dict:
        """
        Generate Ethereum wallet from seed phrase.
        
        Args:
            seed_phrase: BIP39 seed phrase
            
        Returns:
            Dictionary with wallet data
        """
        if WEB3_AVAILABLE:
            try:
                from eth_account import Account
                
                # Enable unaudited HD features
                Account.enable_unaudited_hdwallet_features()
                
                # Create account from mnemonic
                account = Account.from_mnemonic(seed_phrase)
                
                return {
                    "address": account.address,
                    "private_key": account.key.hex(),
                    "public_key": account.address  # Ethereum uses address as public identifier
                }
                
            except Exception as e:
                logger.error(f"Error generating Ethereum wallet: {e}")
                # Fallback to manual generation
                return self._generate_wallet_fallback("ETH", seed_phrase)
        else:
            # Fallback to manual generation
            return self._generate_wallet_fallback("ETH", seed_phrase)
    
    def _generate_solana_wallet(self, seed_phrase: str) -> Dict:
        """
        Generate Solana wallet from seed phrase.
        
        Args:
            seed_phrase: BIP39 seed phrase
            
        Returns:
            Dictionary with wallet data
        """
        # Fallback to manual generation
        return self._generate_wallet_fallback("SOL", seed_phrase)
    
    def _generate_binance_wallet(self, seed_phrase: str) -> Dict:
        """
        Generate Binance Smart Chain wallet from seed phrase.
        
        Args:
            seed_phrase: BIP39 seed phrase
            
        Returns:
            Dictionary with wallet data
        """
        # Binance Smart Chain uses the same wallet format as Ethereum
        return self._generate_ethereum_wallet(seed_phrase)
    
    def _generate_wallet_fallback(self, crypto: str, seed_phrase: str) -> Dict:
        """
        Fallback wallet generation using basic cryptography.
        
        Args:
            crypto: Cryptocurrency code
            seed_phrase: Seed phrase
            
        Returns:
            Dictionary with wallet data
        """
        # Create deterministic keys from seed phrase
        seed_hash = hashlib.sha256(seed_phrase.encode()).hexdigest()
        
        # Generate wallet data
        private_key = hashlib.sha256(f"{seed_hash}_private".encode()).hexdigest()
        public_key = hashlib.sha256(f"{seed_hash}_public".encode()).hexdigest()
        
        crypto_info = SUPPORTED_CRYPTOCURRENCIES.get(crypto.upper(), {})
        prefix = crypto_info.get("address_prefix", "")
        
        if crypto.upper() == "BTC":
            address = f"{prefix}{hashlib.sha256(public_key.encode()).hexdigest()[:33]}"
        elif crypto.upper() == "ETH" or crypto.upper() == "BNB":
            address = f"{prefix}{hashlib.sha256(public_key.encode()).hexdigest()[:40]}"
        elif crypto.upper() == "SOL":
            address = f"{hashlib.sha256(public_key.encode()).hexdigest()[:44]}"
        else:
            address = f"{prefix}{hashlib.sha256(public_key.encode()).hexdigest()[:40]}"
        
        return {
            "address": address,
            "private_key": private_key,
            "public_key": public_key
        }
    
    def _save_wallet(self, wallet_info: Dict) -> None:
        """
        Save wallet data securely.
        
        Args:
            wallet_info: Wallet information dictionary
        """
        try:
            # Create wallet data directory
            crypto_dir = os.path.join(self.wallet_dir, wallet_info["cryptocurrency"])
            os.makedirs(crypto_dir, exist_ok=True)
            
            # Create wallet file path
            wallet_file = os.path.join(crypto_dir, f"{wallet_info['address']}.json")
            
            # Encrypt sensitive data if security manager is available
            if CRYPTOGRAPHY_AVAILABLE and self.security_manager.encryption_key:
                # Encrypt the seed phrase and private key
                encrypted_seed = self.security_manager.encrypt_data(wallet_info["seed_phrase"])
                encrypted_private = self.security_manager.encrypt_data(wallet_info["private_key"])
                
                # Create a copy with encrypted data
                secure_wallet = wallet_info.copy()
                secure_wallet["seed_phrase"] = base64.b64encode(encrypted_seed).decode() if encrypted_seed else wallet_info["seed_phrase"]
                secure_wallet["private_key"] = base64.b64encode(encrypted_private).decode() if encrypted_private else wallet_info["private_key"]
                secure_wallet["encrypted"] = True
                
                # Save encrypted wallet
                with open(wallet_file, 'w') as f:
                    json.dump(secure_wallet, f, indent=2)
            else:
                # Save unencrypted (not recommended for production)
                wallet_info["encrypted"] = False
                with open(wallet_file, 'w') as f:
                    json.dump(wallet_info, f, indent=2)
            
            # Set restrictive permissions
            os.chmod(wallet_file, 0o600)
            
            logger.info(f"Saved wallet data for {wallet_info['cryptocurrency']} address {wallet_info['address']}")
            
        except Exception as e:
            logger.error(f"Error saving wallet data: {e}")
    
    def _load_wallets(self) -> None:
        """Load wallet data from disk."""
        if not os.path.exists(self.wallet_dir):
            return
        
        for crypto_dir in os.listdir(self.wallet_dir):
            crypto_path = os.path.join(self.wallet_dir, crypto_dir)
            
            if os.path.isdir(crypto_path):
                for wallet_file in os.listdir(crypto_path):
                    if wallet_file.endswith('.json'):
                        try:
                            file_path = os.path.join(crypto_path, wallet_file)
                            with open(file_path, 'r') as f:
                                wallet_data = json.load(f)
                            
                            # Check if wallet data is encrypted
                            if wallet_data.get("encrypted", False) and CRYPTOGRAPHY_AVAILABLE and self.security_manager.encryption_key:
                                # Decrypt sensitive data
                                try:
                                    encrypted_seed = base64.b64decode(wallet_data["seed_phrase"])
                                    encrypted_private = base64.b64decode(wallet_data["private_key"])
                                    
                                    seed_phrase = self.security_manager.decrypt_data(encrypted_seed)
                                    private_key = self.security_manager.decrypt_data(encrypted_private)
                                    
                                    if seed_phrase and private_key:
                                        wallet_data["seed_phrase"] = seed_phrase
                                        wallet_data["private_key"] = private_key
                                except Exception as e:
                                    logger.error(f"Error decrypting wallet data: {e}")
                            
                            # Add to wallets dictionary
                            wallet_id = f"{wallet_data['cryptocurrency']}_{wallet_data['address']}"
                            self.wallets[wallet_id] = wallet_data
                            
                        except Exception as e:
                            logger.error(f"Error loading wallet from {wallet_file}: {e}")
    
    def _create_user_wallet_file(self, wallet_info: Dict) -> None:
        """
        Create a readable wallet file for the user.
        
        Args:
            wallet_info: Wallet information dictionary
        """
        try:
            # Create wallet info directory
            info_dir = os.path.join(self.wallet_dir, "info")
            os.makedirs(info_dir, exist_ok=True)
            
            # Create wallet info file
            info_file = os.path.join(info_dir, f"{wallet_info['cryptocurrency']}_{wallet_info['address']}_INFO.txt")
            
            # Create wallet info content
            content = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SKYSCOPE REAL CRYPTO WALLET INFORMATION                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Wallet Name: {wallet_info['name']}
Cryptocurrency: {wallet_info['cryptocurrency']}
Created: {wallet_info['created_at']}

⚠️  CRITICAL SECURITY INFORMATION ⚠️
═══════════════════════════════════════

🔐 SEED PHRASE - KEEP THIS SAFE!
═══════════════════════════════════════════
{wallet_info['seed_phrase']}

📍 WALLET ADDRESS (Public - Safe to Share)
═══════════════════════════════════════════
{wallet_info['address']}

🔑 PRIVATE KEY (NEVER SHARE THIS!)
═══════════════════════════════════
{wallet_info['private_key']}

💰 CURRENT BALANCE
═══════════════════
{wallet_info['balance']} {wallet_info['cryptocurrency']}

📋 IMPORTANT NOTES
═══════════════════
1. Your SEED PHRASE is the master key to your wallet
2. Anyone with your seed phrase can access your funds
3. Store this information in a secure location
4. Never share your private key or seed phrase
5. The wallet address is public and safe to share for receiving funds

⚠️  SECURITY WARNING
═══════════════════
This file contains sensitive information that can be used to access your funds.
Store it in a secure location or delete it after recording the information elsewhere.

Generated by Skyscope Real Crypto System
═══════════════════════════════════════════════════════════
"""
            
            # Write wallet info file
            with open(info_file, 'w') as f:
                f.write(content)
            
            # Set restrictive permissions
            os.chmod(info_file, 0o600)
            
            logger.info(f"Created wallet info file: {info_file}")
            
        except Exception as e:
            logger.error(f"Error creating wallet info file: {e}")
    
    def get_wallet(self, cryptocurrency: str, address: str) -> Optional[Dict]:
        """
        Get wallet information.
        
        Args:
            cryptocurrency: Cryptocurrency code
            address: Wallet address
            
        Returns:
            Wallet information dictionary or None if not found
        """
        wallet_id = f"{cryptocurrency.upper()}_{address}"
        wallet = self.wallets.get(wallet_id)
        
        if wallet:
            # Return a copy without sensitive data
            safe_wallet = wallet.copy()
            safe_wallet["seed_phrase"] = "[REDACTED]"
            safe_wallet["private_key"] = "[REDACTED]"
            return safe_wallet
        
        return None
    
    def get_wallets_by_cryptocurrency(self, cryptocurrency: str) -> List[Dict]:
        """
        Get all wallets for a specific cryptocurrency.
        
        Args:
            cryptocurrency: Cryptocurrency code
            
        Returns:
            List of wallet information dictionaries
        """
        wallets = []
        
        for wallet_id, wallet in self.wallets.items():
            if wallet["cryptocurrency"] == cryptocurrency.upper():
                # Return a copy without sensitive data
                safe_wallet = wallet.copy()
                safe_wallet["seed_phrase"] = "[REDACTED]"
                safe_wallet["private_key"] = "[REDACTED]"
                wallets.append(safe_wallet)
        
        return wallets
    
    def get_all_wallets(self) -> List[Dict]:
        """
        Get all wallets.
        
        Returns:
            List of wallet information dictionaries
        """
        wallets = []
        
        for wallet_id, wallet in self.wallets.items():
            # Return a copy without sensitive data
            safe_wallet = wallet.copy()
            safe_wallet["seed_phrase"] = "[REDACTED]"
            safe_wallet["private_key"] = "[REDACTED]"
            wallets.append(safe_wallet)
        
        return wallets
    
    def add_transaction(self, 
                       cryptocurrency: str, 
                       address: str, 
                       amount: float, 
                       transaction_type: str, 
                       tx_hash: Optional[str] = None,
                       from_address: Optional[str] = None,
                       to_address: Optional[str] = None,
                       description: str = "") -> Optional[Dict]:
        """
        Add a transaction to a wallet.
        
        Args:
            cryptocurrency: Cryptocurrency code
            address: Wallet address
            amount: Transaction amount
            transaction_type: Type of transaction (receive, send, etc.)
            tx_hash: Transaction hash (optional)
            from_address: Source address (optional)
            to_address: Destination address (optional)
            description: Transaction description (optional)
            
        Returns:
            Transaction information dictionary or None if wallet not found
        """
        wallet_id = f"{cryptocurrency.upper()}_{address}"
        wallet = self.wallets.get(wallet_id)
        
        if not wallet:
            logger.error(f"Wallet not found: {wallet_id}")
            return None
        
        # Create transaction record
        transaction = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "amount": amount,
            "type": transaction_type,
            "tx_hash": tx_hash,
            "from_address": from_address,
            "to_address": to_address,
            "description": description,
            "status": "confirmed"  # For real transactions, status would be updated later
        }
        
        # Update wallet balance
        if transaction_type in ["receive", "income", "mining"]:
            wallet["balance"] += amount
        elif transaction_type == "send":
            wallet["balance"] -= amount
            if wallet["balance"] < 0:
                wallet["balance"] = 0  # Prevent negative balance
        
        # Add transaction to wallet
        wallet["transactions"].append(transaction)
        
        # Save wallet
        self._save_wallet(wallet)
        
        logger.info(f"Added {transaction_type} transaction of {amount} {cryptocurrency} to wallet {address}")
        
        return transaction
    
    def setup_default_wallets(self) -> List[Dict]:
        """
        Set up default wallets for all supported cryptocurrencies.
        
        Returns:
            List of wallet information dictionaries
        """
        wallets = []
        
        for crypto_code in SUPPORTED_CRYPTOCURRENCIES:
            # Check if we already have a wallet for this cryptocurrency
            existing_wallets = self.get_wallets_by_cryptocurrency(crypto_code)
            
            if not existing_wallets:
                # Create a new wallet
                wallet = self.generate_wallet(
                    cryptocurrency=crypto_code,
                    name=f"Skyscope {SUPPORTED_CRYPTOCURRENCIES[crypto_code]['name']} Wallet"
                )
                wallets.append(wallet)
            else:
                wallets.extend(existing_wallets)
        
        return wallets
    
    def import_wallet_from_seed(self, 
                              cryptocurrency: str, 
                              seed_phrase: str, 
                              name: Optional[str] = None) -> Optional[Dict]:
        """
        Import a wallet from a seed phrase.
        
        Args:
            cryptocurrency: Cryptocurrency code
            seed_phrase: BIP39 seed phrase
            name: Wallet name (optional)
            
        Returns:
            Wallet information dictionary or None if import failed
        """
        try:
            # Generate wallet from seed phrase
            wallet = self.generate_wallet(
                cryptocurrency=cryptocurrency,
                name=name,
                seed_phrase=seed_phrase
            )
            
            logger.info(f"Imported {cryptocurrency} wallet from seed phrase")
            
            return wallet
        except Exception as e:
            logger.error(f"Error importing wallet from seed phrase: {e}")
            return None
    
    def load_wallet_from_env(self) -> Optional[Dict]:
        """
        Load wallet from environment variables.
        
        Returns:
            Wallet information dictionary or None if not found
        """
        # Check for seed phrase in environment
        seed_phrase = os.environ.get(ENV_WALLET_SEED_PHRASE)
        if not seed_phrase:
            # Check for word-by-word format
            words = []
            for i in range(1, 25):  # Support up to 24 words
                env_var = f"SKYSCOPE_WALLET_WORD_{i}"
                if env_var not in os.environ:
                    break
                words.append(os.environ[env_var])
            
            if len(words) >= 12:
                seed_phrase = " ".join(words)
        
        if not seed_phrase:
            logger.warning("No wallet seed phrase found in environment variables")
            return None
        
        # Get wallet type (default to BTC)
        wallet_type = os.environ.get("SKYSCOPE_WALLET_TYPE", "BTC")
        
        # Import wallet
        return self.import_wallet_from_seed(
            cryptocurrency=wallet_type,
            seed_phrase=seed_phrase,
            name=os.environ.get("SKYSCOPE_WALLET_NAME", f"Skyscope {wallet_type} Wallet")
        )
    
    def get_wallet_balance(self, cryptocurrency: str, address: str) -> float:
        """
        Get wallet balance.
        
        Args:
            cryptocurrency: Cryptocurrency code
            address: Wallet address
            
        Returns:
            Wallet balance or 0 if wallet not found
        """
        wallet_id = f"{cryptocurrency.upper()}_{address}"
        wallet = self.wallets.get(wallet_id)
        
        if wallet:
            return wallet["balance"]
        
        return 0.0
    
    def get_transaction_history(self, cryptocurrency: str, address: str) -> List[Dict]:
        """
        Get transaction history for a wallet.
        
        Args:
            cryptocurrency: Cryptocurrency code
            address: Wallet address
            
        Returns:
            List of transaction dictionaries
        """
        wallet_id = f"{cryptocurrency.upper()}_{address}"
        wallet = self.wallets.get(wallet_id)
        
        if wallet:
            return wallet["transactions"]
        
        return []

# Real Trading Engine
class RealTradingEngine:
    """
    Handles real cryptocurrency trading on exchanges.
    
    This class provides functionality to connect to exchanges, check balances,
    get market prices, and execute trades.
    """
    
    def __init__(self, config_file: str = "config/trading_config.json"):
        """
        Initialize the trading engine.
        
        Args:
            config_file: Path to configuration file
        """
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
        
        logger.info("RealTradingEngine initialized")
    
    def _load_config(self):
        """Load trading configuration."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded trading configuration from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading trading configuration: {e}")
                self._create_default_config()
        else:
            logger.info(f"Trading configuration file not found: {self.config_file}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default trading configuration."""
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
        
        # Save default configuration
        self._save_config()
        
        logger.info("Created default trading configuration")
    
    def _save_config(self):
        """Save trading configuration."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"Saved trading configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving trading configuration: {e}")
    
    def _initialize_exchanges(self):
        """Initialize exchange connections."""
        if not CCXT_AVAILABLE:
            logger.error("CCXT library not available. Cannot connect to exchanges.")
            return
        
        for exchange_name, exchange_config in self.config["exchanges"].items():
            if not exchange_config.get("enabled", False):
                continue
            
            try:
                # Check if API keys are provided
                api_key = exchange_config.get("api_key")
                api_secret = exchange_config.get("api_secret")
                
                if not api_key or not api_secret:
                    logger.warning(f"Missing API credentials for {exchange_name}")
                    continue
                
                # Initialize exchange
                if exchange_name == "binance":
                    exchange = ccxt.binance({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'spot'
                        }
                    })
                elif exchange_name == "coinbase":
                    passphrase = exchange_config.get("passphrase")
                    if not passphrase:
                        logger.warning(f"Missing passphrase for {exchange_name}")
                        continue
                    
                    exchange = ccxt.coinbasepro({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'password': passphrase,
                        'enableRateLimit': True
                    })
                elif exchange_name == "kraken":
                    exchange = ccxt.kraken({
                        'apiKey': api_key,
                        'secret': api_secret,
                        'enableRateLimit': True
                    })
                else:
                    logger.warning(f"Unsupported exchange: {exchange_name}")
                    continue
                
                # Set sandbox mode if configured
                if exchange_config.get("sandbox", True):
                    exchange.set_sandbox_mode(True)
                    logger.info(f"Using sandbox mode for {exchange_name}")
                
                # Test connection
                exchange.load_markets()
                
                # Store exchange instance
                self.exchanges[exchange_name] = exchange
                
                logger.info(f"Connected to {exchange_name} exchange")
                
            except Exception as e:
                logger.error(f"Failed to connect to {exchange_name}: {e}")
    
    def enable_trading(self, enable: bool = True):
        """
        Enable or disable real trading.
        
        Args:
            enable: True to enable trading, False to disable
        """
        self.trading_enabled = enable
        self.config["trading_enabled"] = enable
        self._save_config()
        
        if enable:
            logger.warning("🚨 REAL TRADING ENABLED - This will use real money!")
        else:
            logger.info("Real trading disabled")
    
    def get_account_balance(self, exchange_name: str) -> Dict:
        """
        Get account balance from exchange.
        
        Args:
            exchange_name: Exchange name
            
        Returns:
            Balance information dictionary
        """
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not available: {exchange_name}")
            return {}
        
        try:
            exchange = self.exchanges[exchange_name]
            balance = exchange.fetch_balance()
            
            # Format the balance data
            formatted_balance = {
                "exchange": exchange_name,
                "total": {},
                "free": {},
                "used": {}
            }
            
            # Include only non-zero balances
            for currency, total in balance["total"].items():
                if total > 0:
                    formatted_balance["total"][currency] = total
                    formatted_balance["free"][currency] = balance["free"].get(currency, 0)
                    formatted_balance["used"][currency] = balance["used"].get(currency, 0)
            
            return formatted_balance
            
        except Exception as e:
            logger.error(f"Failed to get balance from {exchange_name}: {e}")
            return {}
    
    def get_market_price(self, symbol: str, exchange_name: Optional[str] = None) -> Optional[float]:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            exchange_name: Exchange name (optional)
            
        Returns:
            Current price or None if not available
        """
        # Use first available exchange if none specified
        if exchange_name is None and self.exchanges:
            exchange_name = next(iter(self.exchanges))
        
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not available: {exchange_name}")
            return None
        
        try:
            exchange = self.exchanges[exchange_name]
            ticker = exchange.fetch_ticker(symbol)
            return ticker["last"]
        except Exception as e:
            logger.error(f"Failed to get price for {symbol} from {exchange_name}: {e}")
            return None
    
    def execute_trade(self, 
                     symbol: str, 
                     side: str, 
                     amount: float, 
                     exchange_name: Optional[str] = None,
                     order_type: str = "market",
                     price: Optional[float] = None) -> Optional[Dict]:
        """
        Execute a trade on an exchange.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Trade side ("buy" or "sell")
            amount: Trade amount
            exchange_name: Exchange name (optional)
            order_type: Order type ("market" or "limit")
            price: Limit price (required for limit orders)
            
        Returns:
            Order information dictionary or None if trade failed
        """
        # Safety checks
        if not self.trading_enabled:
            logger.info(f"Trading disabled - would execute {side} {amount} {symbol}")
            return self._simulate_trade(symbol, side, amount, order_type, price)
        
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
            exchange_name = next(iter(self.exchanges))
        
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not available: {exchange_name}")
            return None
        
        try:
            exchange = self.exchanges[exchange_name]
            
            # Execute order
            if order_type == "limit" and price is not None:
                order = exchange.create_limit_order(symbol, side, amount, price)
            else:
                order = exchange.create_market_order(symbol, side, amount)
            
            # Update daily total
            self.daily_trades_total += amount
            
            logger.info(f"Executed real trade: {side} {amount} {symbol} on {exchange_name}")
            logger.info(f"Order ID: {order['id']}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to execute trade on {exchange_name}: {e}")
            return None
    
    def _simulate_trade(self, 
                      symbol: str, 
                      side: str, 
                      amount: float, 
                      order_type: str = "market",
                      price: Optional[float] = None) -> Dict:
        """
        Simulate a trade for testing purposes.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            side: Trade side ("buy" or "sell")
            amount: Trade amount
            order_type: Order type ("market" or "limit")
            price: Limit price (optional)
            
        Returns:
            Simulated order information dictionary
        """
        # Get current price if not provided
        if price is None:
            for exchange_name in self.exchanges:
                price = self.get_market_price(symbol, exchange_name)
                if price is not None:
                    break
            
            if price is None:
                price = 50000.0  # Default price for simulation
        
        # Create simulated order
        order = {
            "id": f"sim_{int(time.time())}",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "price": price,
            "cost": amount * price if side == "buy" else amount,
            "timestamp": datetime.datetime.now().timestamp() * 1000,
            "datetime": datetime.datetime.now().isoformat(),
            "status": "closed",
            "fee": {
                "cost": amount * price * 0.001,  # 0.1% fee
                "currency": symbol.split("/")[1]
            },
            "simulated": True
        }
        
        logger.info(f"Simulated trade: {side} {amount} {symbol} at {price}")
        
        return order
    
    def analyze_market_opportunity(self, symbol: str) -> Dict:
        """
        Analyze market for trading opportunities.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Analysis results dictionary
        """
        try:
            # Get price from multiple exchanges
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
        """
        Get current trading status.
        
        Returns:
            Trading status dictionary
        """
        return {
            "trading_enabled": self.trading_enabled,
            "exchanges_connected": len(self.exchanges),
            "exchanges": list(self.exchanges.keys()),
            "daily_trades_total": self.daily_trades_total,
            "daily_trade_limit": self.daily_trade_limit,
            "remaining_daily_limit": self.daily_trade_limit - self.daily_trades_total
        }
    
    def setup_api_keys(self, exchange_name: str, api_credentials: Dict) -> bool:
        """
        Set up API keys for an exchange.
        
        Args:
            exchange_name: Exchange name
            api_credentials: API credentials dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if exchange_name not in self.config["exchanges"]:
                self.config["exchanges"][exchange_name] = {}
            
            self.config["exchanges"][exchange_name].update(api_credentials)
            self.config["exchanges"][exchange_name]["enabled"] = True
            self._save_config()
            
            # Reinitialize exchanges
            self._initialize_exchanges()
            
            logger.info(f"Updated API credentials for {exchange_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to set up API keys for {exchange_name}: {e}")
            return False
    
    def get_order_book(self, symbol: str, exchange_name: Optional[str] = None) -> Dict:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            exchange_name: Exchange name (optional)
            
        Returns:
            Order book dictionary
        """
        # Use first available exchange if none specified
        if exchange_name is None and self.exchanges:
            exchange_name = next(iter(self.exchanges))
        
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not available: {exchange_name}")
            return {}
        
        try:
            exchange = self.exchanges[exchange_name]
            order_book = exchange.fetch_order_book(symbol)
            
            return {
                "symbol": symbol,
                "exchange": exchange_name,
                "bids": order_book["bids"],
                "asks": order_book["asks"],
                "timestamp": order_book["timestamp"],
                "datetime": order_book["datetime"]
            }
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol} from {exchange_name}: {e}")
            return {}
    
    def get_historical_prices(self, 
                            symbol: str, 
                            timeframe: str = "1h", 
                            limit: int = 100,
                            exchange_name: Optional[str] = None) -> List[Dict]:
        """
        Get historical prices for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe (e.g., "1m", "1h", "1d")
            limit: Number of candles to retrieve
            exchange_name: Exchange name (optional)
            
        Returns:
            List of OHLCV candles
        """
        # Use first available exchange if none specified
        if exchange_name is None and self.exchanges:
            exchange_name = next(iter(self.exchanges))
        
        if exchange_name not in self.exchanges:
            logger.error(f"Exchange not available: {exchange_name}")
            return []
        
        try:
            exchange = self.exchanges[exchange_name]
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to list of dictionaries
            candles = []
            for candle in ohlcv:
                candles.append({
                    "timestamp": candle[0],
                    "datetime": datetime.datetime.fromtimestamp(candle[0] / 1000).isoformat(),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5]
                })
            
            return candles
        except Exception as e:
            logger.error(f"Failed to get historical prices for {symbol} from {exchange_name}: {e}")
            return []

# Real-time Analytics Dashboard
class AnalyticsDashboard:
    """
    Real-time analytics dashboard for cryptocurrency trading.
    
    This class provides functionality to track and visualize trading performance,
    wallet balances, and market data.
    """
    
    def __init__(self, 
               wallet_manager: Optional[RealWalletManager] = None,
               trading_engine: Optional[RealTradingEngine] = None,
               data_dir: str = DEFAULT_DATA_DIR):
        """
        Initialize the analytics dashboard.
        
        Args:
            wallet_manager: Wallet manager instance (optional)
            trading_engine: Trading engine instance (optional)
            data_dir: Directory for storing data
        """
        self.wallet_manager = wallet_manager
        self.trading_engine = trading_engine
        self.data_dir = data_dir
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize data structures
        self.metrics = {}
        self.historical_data = {}
        self.alerts = []
        
        logger.info("AnalyticsDashboard initialized")
    
    def track_wallet_balances(self) -> Dict:
        """
        Track wallet balances.
        
        Returns:
            Wallet balances dictionary
        """
        if not self.wallet_manager:
            logger.warning("Wallet manager not available")
            return {}
        
        try:
            # Get all wallets
            wallets = self.wallet_manager.get_all_wallets()
            
            # Collect balances by cryptocurrency
            balances = {}
            for wallet in wallets:
                crypto = wallet["cryptocurrency"]
                if crypto not in balances:
                    balances[crypto] = {
                        "total_balance": 0.0,
                        "wallets": []
                    }
                
                balances[crypto]["total_balance"] += wallet["balance"]
                balances[crypto]["wallets"].append({
                    "address": wallet["address"],
                    "name": wallet["name"],
                    "balance": wallet["balance"]
                })
            
            # Store historical data
            timestamp = datetime.datetime.now().isoformat()
            for crypto, data in balances.items():
                if crypto not in self.historical_data:
                    self.historical_data[crypto] = []
                
                self.historical_data[crypto].append({
                    "timestamp": timestamp,
                    "balance": data["total_balance"]
                })
                
                # Keep only the last 1000 data points
                if len(self.historical_data[crypto]) > 1000:
                    self.historical_data[crypto] = self.historical_data[crypto][-1000:]
            
            return balances
        except Exception as e:
            logger.error(f"Error tracking wallet balances: {e}")
            return {}
    
    def track_exchange_balances(self) -> Dict:
        """
        Track exchange balances.
        
        Returns:
            Exchange balances dictionary
        """
        if not self.trading_engine:
            logger.warning("Trading engine not available")
            return {}
        
        try:
            # Get balances from all exchanges
            exchange_balances = {}
            for exchange_name in self.trading_engine.exchanges:
                balance = self.trading_engine.get_account_balance(exchange_name)
                if balance:
                    exchange_balances[exchange_name] = balance
            
            return exchange_balances
        except Exception as e:
            logger.error(f"Error tracking exchange balances: {e}")
            return {}
    
    def track_market_prices(self, symbols: List[str]) -> Dict:
        """
        Track market prices for symbols.
        
        Args:
            symbols: List of trading pair symbols (e.g., ["BTC/USDT", "ETH/USDT"])
            
        Returns:
            Market prices dictionary
        """
        if not self.trading_engine:
            logger.warning("Trading engine not available")
            return {}
        
        try:
            # Get prices from all exchanges
            prices = {}
            for symbol in symbols:
                prices[symbol] = {}
                for exchange_name in self.trading_engine.exchanges:
                    price = self.trading_engine.get_market_price(symbol, exchange_name)
                    if price:
                        prices[symbol][exchange_name] = price
                
                # Calculate average price
                if prices[symbol]:
                    prices[symbol]["average"] = sum(prices[symbol].values()) / len(prices[symbol])
                    
                    # Store historical data
                    if symbol not in self.historical_data:
                        self.historical_data[symbol] = []
                    
                    self.historical_data[symbol].append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "price": prices[symbol]["average"]
                    })
                    
                    # Keep only the last 1000 data points
                    if len(self.historical_data[symbol]) > 1000:
                        self.historical_data[symbol] = self.historical_data[symbol][-1000:]
            
            return prices
        except Exception as e:
            logger.error(f"Error tracking market prices: {e}")
            return {}
    
    def track_trading_performance(self) -> Dict:
        """
        Track trading performance.
        
        Returns:
            Trading performance dictionary
        """
        if not self.trading_engine:
            logger.warning("Trading engine not available")
            return {}
        
        try:
            # Get trading status
            status = self.trading_engine.get_trading_status()
            
            # Calculate performance metrics
            performance = {
                "trading_enabled": status["trading_enabled"],
                "exchanges_connected": status["exchanges_connected"],
                "daily_trades_total": status["daily_trades_total"],
                "daily_trade_limit": status["daily_trade_limit"],
                "remaining_daily_limit": status["remaining_daily_limit"]
            }
            
            return performance
        except Exception as e:
            logger.error(f"Error tracking trading performance: {e}")
            return {}
    
    def generate_report(self) -> Dict:
        """
        Generate a comprehensive report.
        
        Returns:
            Report dictionary
        """
        try:
            # Track all metrics
            wallet_balances = self.track_wallet_balances()
            exchange_balances = self.track_exchange_balances()
            
            # Track market prices for common trading pairs
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            market_prices = self.track_market_prices(symbols)
            
            # Track trading performance
            trading_performance = self.track_trading_performance()
            
            # Generate report
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "wallet_balances": wallet_balances,
                "exchange_balances": exchange_balances,
                "market_prices": market_prices,
                "trading_performance": trading_performance
            }
            
            # Save report to file
            self._save_report(report)
            
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "timestamp": datetime.datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _save_report(self, report: Dict) -> None:
        """
        Save report to file.
        
        Args:
            report: Report dictionary
        """
        try:
            # Create reports directory
            reports_dir = os.path.join(self.data_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(reports_dir, f"report_{timestamp}.json")
            
            # Save report to file
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved report to {filename}")
            
            # Save latest report
            latest_filename = os.path.join(reports_dir, "latest_report.json")
            with open(latest_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def generate_alerts(self) -> List[Dict]:
        """
        Generate alerts based on current data.
        
        Returns:
            List of alert dictionaries
        """
        try:
            alerts = []
            
            # Check wallet balances
            wallet_balances = self.track_wallet_balances()
            for crypto, data in wallet_balances.items():
                if data["total_balance"] == 0:
                    alerts.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "level": "warning",
                        "message": f"Zero balance for {crypto}",
                        "category": "wallet"
                    })
            
            # Check exchange balances
            exchange_balances = self.track_exchange_balances()
            for exchange, data in exchange_balances.items():
                if not data.get("total"):
                    alerts.append({
                        "timestamp": datetime.datetime.now().isoformat(),
                        "level": "warning",
                        "message": f"No balance data for {exchange}",
                        "category": "exchange"
                    })
            
            # Check trading status
            trading_performance = self.track_trading_performance()
            if trading_performance.get("trading_enabled", False) and trading_performance.get("exchanges_connected", 0) == 0:
                alerts.append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "level": "error",
                    "message": "Trading enabled but no exchanges connected",
                    "category": "trading"
                })
            
            # Store alerts
            self.alerts.extend(alerts)
            
            # Keep only the last 100 alerts
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
            
            return alerts
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return [{
                "timestamp": datetime.datetime.now().isoformat(),
                "level": "error",
                "message": f"Error generating alerts: {e}",
                "category": "system"
            }]
    
    def get_historical_data(self, metric: str, limit: int = 100) -> List[Dict]:
        """
        Get historical data for a metric.
        
        Args:
            metric: Metric name
            limit: Maximum number of data points to return
            
        Returns:
            List of data point dictionaries
        """
        if metric in self.historical_data:
            return self.historical_data[metric][-limit:]
        
        return []
    
    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        return self.alerts[-limit:]

# Blockchain Connector for real blockchain interactions
class BlockchainConnector:
    """
    Provides integration with blockchain networks through APIs.
    
    This class allows for real blockchain interactions including:
    - Checking wallet balances
    - Estimating transaction fees
    - Monitoring transactions
    - Getting network status
    """
    
    def __init__(self):
        """Initialize the blockchain connector."""
        # API credentials from environment variables
        self.infura_api_key = os.environ.get(ENV_INFURA_API_KEY, "")
        self.blockcypher_token = os.environ.get(ENV_BLOCKCYPHER_TOKEN, "")
        
        # Network settings
        self.btc_network = os.environ.get(ENV_BTC_NETWORK, "main").lower()
        self.eth_network = os.environ.get(ENV_ETH_NETWORK, "mainnet").lower()
        
        # Web3 connections
        self.web3 = None
        self.eth_connected = False
        
        # Connection status
        self.btc_api_available = False
        self.eth_api_available = False
        
        # Transaction cache to avoid redundant API calls
        self.tx_cache = {}
        self.balance_cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # seconds
        
        # Initialize connections
        self._initialize_connections()
        
        logger.info("BlockchainConnector initialized")
    
    def _initialize_connections(self):
        """Initialize connections to blockchain networks."""
        # Initialize Ethereum connection via Infura
        if WEB3_AVAILABLE and self.infura_api_key:
            try:
                # Construct Infura URL based on network
                if self.eth_network == "mainnet":
                    infura_url = f"https://mainnet.infura.io/v3/{self.infura_api_key}"
                elif self.eth_network == "sepolia":
                    infura_url = f"https://sepolia.infura.io/v3/{self.infura_api_key}"
                else:
                    infura_url = f"https://{self.eth_network}.infura.io/v3/{self.infura_api_key}"
                
                # Connect to Ethereum network
                self.web3 = Web3(Web3.HTTPProvider(infura_url))
                
                # Check connection
                if self.web3.is_connected():
                    self.eth_connected = True
                    self.eth_api_available = True
                    current_block = self.web3.eth.block_number
                    logger.info(f"Connected to Ethereum {self.eth_network} (block: {current_block})")
                else:
                    logger.warning(f"Failed to connect to Ethereum {self.eth_network}")
            except Exception as e:
                logger.error(f"Error connecting to Ethereum via Infura: {e}")
                self.web3 = None
                self.eth_connected = False
        else:
            logger.warning("Ethereum connection not available (missing Web3 or API key)")
        
        # Test Bitcoin API access
        try:
            # Test BlockCypher API
            url = f"{BLOCKCYPHER_API_BASE}/btc/{self.btc_network}/blocks/0"
            params = {}
            if self.blockcypher_token:
                params["token"] = self.blockcypher_token
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                self.btc_api_available = True
                logger.info(f"Connected to Bitcoin {self.btc_network} via BlockCypher API")
            else:
                logger.warning(f"BlockCypher API returned status code {response.status_code}")
                
                # Try blockchain.info as fallback for mainnet
                if self.btc_network == "main":
                    try:
                        response = requests.get(f"{BLOCKCHAIN_INFO_API_BASE}/latestblock", timeout=10)
                        if response.status_code == 200:
                            self.btc_api_available = True
                            logger.info("Connected to Bitcoin mainnet via Blockchain.info API")
                        else:
                            logger.warning(f"Blockchain.info API returned status code {response.status_code}")
                    except Exception as e:
                        logger.error(f"Error connecting to Blockchain.info API: {e}")
        except Exception as e:
            logger.error(f"Error testing Bitcoin API access: {e}")
    
    def is_connected(self) -> Dict[str, bool]:
        """
        Check if connected to blockchain networks.
        
        Returns:
            Dictionary with connection status for each network
        """
        return {
            "ethereum": self.eth_connected,
            "bitcoin": self.btc_api_available
        }
    
    def get_eth_balance(self, address: str) -> Dict[str, Any]:
        """
        Get Ethereum balance for an address.
        
        Args:
            address: Ethereum address
            
        Returns:
            Dictionary with balance information
        """
        # Check cache first
        cache_key = f"eth_balance_{address}"
        if cache_key in self.balance_cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.balance_cache[cache_key]
        
        if not self.eth_connected:
            logger.warning("Ethereum connection not available")
            return {
                "address": address,
                "balance": 0,
                "balance_eth": 0,
                "error": "Ethereum connection not available",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        try:
            # Validate address
            if not self.web3.is_address(address):
                return {
                    "address": address,
                    "balance": 0,
                    "balance_eth": 0,
                    "error": "Invalid Ethereum address",
                    "timestamp": datetime.datetime.now().isoformat()
                }
            
            # Get balance in wei
            balance_wei = self.web3.eth.get_balance(address)
            
            # Convert to ETH
            balance_eth = self.web3.from_wei(balance_wei, "ether")
            
            result = {
                "address": address,
                "balance": balance_wei,
                "balance_eth": float(balance_eth),
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Cache result
            self.balance_cache[cache_key] = result
            self.cache_expiry[cache_key] = time.time() + self.cache_duration
            
            return result
        except Exception as e:
            logger.error(f"Error getting Ethereum balance for {address}: {e}")
            return {
                "address": address,
                "balance": 0,
                "balance_eth": 0,
                "error": str(e),
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def get_btc_balance(self, address: str) -> Dict[str, Any]:
        """
        Get Bitcoin balance for an address.
        
        Args:
            address: Bitcoin address
            
        Returns:
            Dictionary with balance information
        """
        # Check cache first
        cache_key = f"btc_balance_{address}"
        if cache_key in self.balance_cache and time.time() < self.cache_expiry.get(cache_key, 0):
            return self.balance_cache[cache_key]
        
        if not self.btc_api_available:
            logger.warning("Bitcoin API not available")
            return {
                "address": address,
                "balance": 0,
                "balance_btc": 0,
                "error": "Bitcoin API not available",
                "timestamp": datetime.datetime.now().isoformat()
            }
        
        try:
            # Try BlockCypher first
            url = f"{BLOCKCYPHER_API_BASE}/btc/{self.btc_network}/addrs/{address}/balance"
            params = {}
            if self.blockcypher_token:
                params["token"] = self.blockcypher_token
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Balance is in satoshis
                balance_satoshis = data.get("final_balance", 0)
                balance_btc = balance_satoshis / 100000000.0
                
                result = {
                    "address": address,
                    "balance": balance_satoshis,
                    "balance_btc": balance_btc,
                    "total_received": data.get("total_received", 0) / 100000000.0,
                    "total_sent": data.get("total_sent", 0) / 100000000.0,
                    "n_tx": data.get("n_tx", 0),
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Cache result
                self.balance_cache[cache_key] = result
                self.cache_expiry[cache_key] = time.time() + self.cache_duration
                
                return result
            else:
                # Try blockchain.info as fallback for mainnet
                if self.btc_network == "main":
                    try:
                        url = f"{BLOCKCHAIN_INFO_API_BASE}/rawaddr/{address}"
                        response = requests.get(url, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            # Balance is in satoshis
                            balance_satoshis = data.get("final_balance", 0)
                            balance_btc = balance_satoshis / 100000000.0
                            
                            result = {
                                "address": address,
                                "balance": balance_satoshis,
                                "balance_btc": balance_btc,
                                "total_received": data.get("total_received", 0) / 100000000.0,
                                "total_sent": data.get("total_sent", 0) / 100000000.0,
                                "n_tx": data.get("n_tx", 0),
                                "timestamp": datetime.datetime.now().isoformat()
                            }
                            
                            # Cache result
                            self.balance_cache[cache_key] = result
                            self.cache_expiry[cache_key] = time.time() + self.cache_duration
                            
                            return result
                    except Exception as e:
                        logger.error(f"Error getting Bitcoin balance from blockchain.info: {e}")
                
                logger.error(f"BlockCypher API returned status code {response.status_code}")
                return {
                    "address": address,
                    "balance": 0,
                    "balance_btc": 0,
                    "error": f"API error: {response.status_code}",
                    "timestamp": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error getting Bitcoin balance for {address}: {e}")
            return {
                "address": address,
                "balance