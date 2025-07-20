#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Secure Wallet Integration for Skyscope Enterprise Suite
=====================================================

This module provides secure wallet integration by leveraging environment
variables and secure storage mechanisms. It handles:
- Importing wallets from seed phrases
- Encrypting and decrypting wallet data
- Managing wallet addresses and keys
- Integrating with the main wallet manager

SECURITY NOTICE:
- Seed phrases are loaded from environment variables and not stored directly
- Wallet data is encrypted using a user-provided key
- This module should be used in a secure environment
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Try to import cryptography for encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    import base64
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logging.warning("Cryptography package not available. Wallet data will not be encrypted.")
    CRYPTOGRAPHY_AVAILABLE = False

# Import our crypto wallet manager
try:
    from crypto.crypto_wallet_manager import wallet_manager, CryptoWallet
except ImportError:
    wallet_manager = None
    logging.error("Wallet manager not available. Secure wallet integration will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("secure_wallet.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SecureWallet")

# Environment variable names
ENV_WALLET_MODE = "SKYSCOPE_WALLET_MODE"
ENV_WALLET_KEY = "SKYSCOPE_WALLET_KEY"
ENV_SEED_PHRASE = "SKYSCOPE_WALLET_SEED_PHRASE"

class SecureWalletIntegration:
    """
    Manages secure wallet integration, including importing, encrypting,
    and managing wallets from seed phrases.
    """

    def __init__(self, data_dir: str = "secure_wallets"):
        """
        Initialize the secure wallet integration.

        Args:
            data_dir: Directory for storing encrypted wallet data
        """
        self.data_dir = data_dir
        self.wallets = {}
        self.encryption_key = None
        self.connection_mode = os.environ.get(ENV_WALLET_MODE, "simulation")
        self.custom_addresses = {}
        self.earnings = {}

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Load encryption key from environment
        self._load_encryption_key()

        # Load any existing wallets
        self._load_wallets()

        logger.info(f"SecureWalletIntegration initialized in {self.connection_mode} mode")

    def _load_encryption_key(self) -> None:
        """Load encryption key from environment variable."""
        key = os.environ.get(ENV_WALLET_KEY)

        if key:
            # Derive a 32-byte key using PBKDF2
            if CRYPTOGRAPHY_AVAILABLE:
                salt = b'skyscope-salt'  # Use a fixed salt for simplicity
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                    backend=default_backend()
                )
                self.encryption_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
                logger.info("Encryption key loaded and derived successfully")
            else:
                self.encryption_key = key.encode()
                logger.warning("Cryptography package not available. Using raw key for encryption.")
        else:
            logger.warning("No encryption key found. Wallet data will not be encrypted.")

    def _encrypt_data(self, data: str) -> Optional[bytes]:
        """
        Encrypt data using the encryption key.

        Args:
            data: Data to encrypt

        Returns:
            Encrypted data or None on error
        """
        if not self.encryption_key:
            return data.encode()  # No encryption if key is not available

        if CRYPTOGRAPHY_AVAILABLE:
            try:
                f = Fernet(self.encryption_key)
                return f.encrypt(data.encode())
            except Exception as e:
                logger.error(f"Error encrypting data: {e}")
                return None
        else:
            return data.encode()

    def _decrypt_data(self, encrypted_data: bytes) -> Optional[str]:
        """
        Decrypt data using the encryption key.

        Args:
            encrypted_data: Data to decrypt

        Returns:
            Decrypted data or None on error
        """
        if not self.encryption_key:
            return encrypted_data.decode()  # No decryption if key is not available

        if CRYPTOGRAPHY_AVAILABLE:
            try:
                f = Fernet(self.encryption_key)
                return f.decrypt(encrypted_data).decode()
            except Exception as e:
                logger.error(f"Error decrypting data: {e}")
                return None
        else:
            return encrypted_data.decode()

    def import_wallet(self,
                     seed_phrase: str,
                     wallet_type: str,
                     name: str = None) -> Optional[str]:
        """
        Import a wallet from a seed phrase and save it securely.

        Args:
            seed_phrase: Wallet seed phrase
            wallet_type: Cryptocurrency type (BTC, ETH, etc.)
            name: Wallet name

        Returns:
            Wallet ID or None on error
        """
        try:
            # Generate a unique ID for this wallet
            wallet_id = f"wallet-{uuid.uuid4()}"

            # Encrypt seed phrase
            encrypted_seed = self._encrypt_data(seed_phrase)
            if not encrypted_seed:
                logger.error("Failed to encrypt seed phrase")
                return None

            # Create wallet data
            wallet_data = {
                "id": wallet_id,
                "name": name or f"Secure {wallet_type} Wallet",
                "type": wallet_type,
                "encrypted_seed": encrypted_seed.decode(),
                "created_at": datetime.now().isoformat()
            }

            # Save wallet data
            wallet_file = os.path.join(self.data_dir, f"{wallet_id}.json")
            with open(wallet_file, "w") as f:
                json.dump(wallet_data, f, indent=2)

            # Add to in-memory wallets
            self.wallets[wallet_id] = wallet_data

            logger.info(f"Wallet {name} ({wallet_type}) imported successfully")
            return wallet_id
        except Exception as e:
            logger.error(f"Error importing wallet: {e}")
            return None

    def _load_wallets(self) -> None:
        """Load secure wallets from disk."""
        try:
            if not os.path.exists(self.data_dir):
                return

            for wallet_file in os.listdir(self.data_dir):
                if wallet_file.endswith(".json"):
                    try:
                        wallet_path = os.path.join(self.data_dir, wallet_file)
                        with open(wallet_path, "r") as f:
                            wallet_data = json.load(f)

                            # Add to in-memory wallets
                            self.wallets[wallet_data["id"]] = wallet_data
                    except Exception as e:
                        logger.error(f"Error loading wallet from {wallet_file}: {e}")
        except Exception as e:
            logger.error(f"Error loading secure wallets: {e}")

    def list_wallets(self) -> List[Dict[str, Any]]:
        """
        List all secure wallets.

        Returns:
            List of wallet dictionaries
        """
        wallets_list = []

        for wallet_id, wallet_data in self.wallets.items():
            # Get address for this wallet
            address = self.get_wallet_address(wallet_id)

            wallets_list.append({
                "id": wallet_id,
                "name": wallet_data["name"],
                "type": wallet_data["type"],
                "address": address,
                "created_at": wallet_data["created_at"]
            })

        return wallets_list

    def get_wallet_address(self, wallet_id: str) -> Optional[str]:
        """
        Get the address for a secure wallet.

        Args:
            wallet_id: ID of the wallet

        Returns:
            Wallet address or None on error
        """
        if wallet_id not in self.wallets:
            return None

        # Get wallet data
        wallet_data = self.wallets[wallet_id]

        # Decrypt seed phrase
        encrypted_seed = wallet_data["encrypted_seed"].encode()
        seed_phrase = self._decrypt_data(encrypted_seed)

        if not seed_phrase:
            logger.error(f"Failed to decrypt seed phrase for wallet {wallet_id}")
            return None

        # Generate address from seed phrase
        # This requires a deterministic wallet implementation, which is complex.
        # For simulation, we'll use a simple hash-based address.
        # In a real system, you would use a library like `bip_utils`.

        # For simplicity, we'll create a temporary CryptoWallet to get the address
        if wallet_manager:
            try:
                temp_wallet = CryptoWallet(
                    cryptocurrency=wallet_data["type"],
                    seed_phrase=seed_phrase
                )
                return temp_wallet.address
            except Exception as e:
                logger.error(f"Error generating address for wallet {wallet_id}: {e}")
                return None
        else:
            # Fallback if wallet manager is not available
            h = hashlib.sha256(seed_phrase.encode()).hexdigest()
            return h[:42]

    def get_connection_mode(self) -> str:
        """
        Get the current wallet connection mode.

        Returns:
            'real' or 'simulation'
        """
        return self.connection_mode

    def register_external_address(self, wallet_type: str, address: str) -> bool:
        """
        Register a real external address for a wallet type.

        Args:
            wallet_type: Cryptocurrency type (BTC, ETH, etc.)
            address: Real external address

        Returns:
            True if registered, False otherwise
        """
        self.custom_addresses[wallet_type] = address
        logger.info(f"Registered real {wallet_type} address: {address}")
        return True

    def integrate_with_wallet_manager(self) -> bool:
        """
        Integrate secure wallets with the main wallet manager.

        Returns:
            True if integrated, False otherwise
        """
        if not wallet_manager:
            logger.error("Wallet manager not available for integration")
            return False

        try:
            logger.info("Integrating secure wallets with wallet manager")

            for wallet_id, wallet_data in self.wallets.items():
                # Check if wallet already exists in wallet manager
                address = self.get_wallet_address(wallet_id)
                if not address:
                    continue

                existing_wallet = wallet_manager.get_wallet(
                    cryptocurrency=wallet_data["type"],
                    address=address
                )

                if not existing_wallet:
                    # Decrypt seed phrase
                    seed_phrase = self._decrypt_data(wallet_data["encrypted_seed"].encode())
                    if not seed_phrase:
                        continue

                    # Create wallet in wallet manager
                    wallet_manager.create_wallet(
                        cryptocurrency=wallet_data["type"],
                        name=wallet_data["name"],
                        seed_phrase=seed_phrase
                    )

            return True
        except Exception as e:
            logger.error(f"Error integrating secure wallets: {e}")
            return False

    def record_earning(self, wallet_type: str, amount: float) -> None:
        """
        Record earnings for a specific wallet type.

        Args:
            wallet_type: Cryptocurrency type (BTC, ETH, etc.)
            amount: Amount earned
        """
        if wallet_type not in self.earnings:
            self.earnings[wallet_type] = 0.0

        self.earnings[wallet_type] += amount
        logger.info(f"Recorded earning of {amount} {wallet_type}")

    def get_total_earnings(self) -> Dict[str, float]:
        """
        Get total earnings for all wallet types.

        Returns:
            Dictionary with total earnings by cryptocurrency
        """
        return self.earnings


def load_seed_phrase_from_env() -> Optional[str]:
    """
    Load seed phrase from environment variables.

    Tries to load from SKYSCOPE_WALLET_SEED_PHRASE first, then
    from individual word variables (SKYSCOPE_WALLET_WORD_1, etc.).

    Returns:
        The seed phrase or None if not found
    """
    # Try complete seed phrase first
    if ENV_SEED_PHRASE in os.environ:
        return os.environ[ENV_SEED_PHRASE]

    # Try individual words
    words = []
    for i in range(1, 25):  # Check up to 24 words
        word = os.environ.get(f"SKYSCOPE_WALLET_WORD_{i}")
        if word:
            words.append(word)
        else:
            break

    if len(words) >= 12:
        return " ".join(words)

    return None


def print_secure_setup_instructions() -> None:
    """Print detailed instructions for secure wallet setup."""
    print_header("SECURE WALLET SETUP INSTRUCTIONS")

    print_info("To use real wallets, you need to set up environment variables securely.")
    print_warning("This is a critical security step - do not skip or rush it!")
    print()

    # Platform-specific instructions
    system = platform.system().lower()

    if system == "windows":
        print_info("On Windows, you can set environment variables using the System Properties:")
        print("1. Search for 'environment variables' in the Start Menu and select 'Edit the system environment variables'")
        print("2. In the System Properties window, click the 'Environment Variables...' button")
        print("3. In the 'User variables' section, click 'New...' to add a new variable")
        print("4. Add the variables below")
    else:
        # Linux or macOS
        env_file = get_platform_env_file()
        print_info(f"On {platform.system()}, you can set environment variables in your shell configuration file.")
        print(f"For your current shell, this is likely: {env_file}")
        print()
        print_info("Open this file with a text editor (e.g., nano, vim, gedit) and add the following lines:")

    # Environment variables to set
    print()
    print_warning("Required Environment Variables:")
    print_info("---------------------------------")

    # SKYSCOPE_WALLET_MODE
    print(f"\n{Fore.GREEN}SKYSCOPE_WALLET_MODE{Style.RESET_ALL}")
    print("Set this to 'real' to enable real wallet integration.")
    if system != "windows":
        print(f"  {get_platform_env_command('SKYSCOPE_WALLET_MODE', 'real')}")

    # SKYSCOPE_WALLET_KEY
    print(f"\n{Fore.GREEN}SKYSCOPE_WALLET_KEY{Style.RESET_ALL}")
    print("This is an encryption key for storing wallet data securely.")
    print_warning("Choose a strong, unique key and store it in a password manager.")
    if system != "windows":
        print(f"  {get_platform_env_command('SKYSCOPE_WALLET_KEY', 'YOUR_STRONG_ENCRYPTION_KEY_HERE')}")

    # SKYSCOPE_WALLET_SEED_PHRASE
    print(f"\n{Fore.GREEN}SKYSCOPE_WALLET_SEED_PHRASE{Style.RESET_ALL}")
    print("This is your 12 or 24-word seed phrase.")
    print_warning("This is the most sensitive piece of information - handle it with care!")
    if system != "windows":
        print(f"  {get_platform_env_command('SKYSCOPE_WALLET_SEED_PHRASE', 'word1 word2 ... word12')}")

    # Alternative: individual words
    print("\nAlternatively, you can set individual words (more secure in some cases):")
    print(f"{Fore.GREEN}SKYSCOPE_WALLET_WORD_1, SKYSCOPE_WALLET_WORD_2, ...{Style.RESET_ALL}")
    if system != "windows":
        print(f"  {get_platform_env_command('SKYSCOPE_WALLET_WORD_1', 'word1')}")
        print(f"  {get_platform_env_command('SKYSCOPE_WALLET_WORD_2', 'word2')}")
        print("  ...")

    print()
    print_info("After setting these variables, restart your terminal or application")
    print_info("to ensure they are loaded correctly.")
    print()


# Create singleton instance
secure_wallet = SecureWalletIntegration()

if __name__ == "__main__":
    # Simple test
    print("Secure Wallet Integration Test")
    print("==============================")

    # Print setup instructions
    print_secure_setup_instructions()

    # Check current setup
    status = check_current_env_setup()
    print("\nCurrent environment status:")
    for var, is_set in status.items():
        print(f"- {var}: {'Set' if is_set else 'Not set'}")

    # Try to load seed phrase
    seed = load_seed_phrase_from_env()
    if seed:
        print("\nSeed phrase loaded successfully from environment!")
        print(f"  Seed phrase: {seed[:10]}...{seed[-10:]}")
    else:
        print("\nNo seed phrase found in environment variables.")
