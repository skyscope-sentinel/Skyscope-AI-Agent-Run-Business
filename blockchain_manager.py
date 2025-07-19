#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Blockchain Manager for Skyscope Agent Swarm
===========================================

This module handles all interactions with the blockchain, using INFURA endpoints.
"""

import os
import json
import logging
from typing import Dict, Any
from web3 import Web3

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blockchain.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BlockchainManager")

class BlockchainManager:
    """Manages connections to various blockchains via INFURA"""

    def __init__(self):
        self.infura_api_key = config.get("infura.api_key")
        if not self.infura_api_key:
            raise ValueError("INFURA API key not found in configuration.")

        self.web3_instances = {}

    def get_web3_instance(self, network: str = "ethereum_mainnet") -> Web3:
        """
        Get a Web3 instance for a specific network.

        Args:
            network: The name of the network (e.g., "ethereum_mainnet")

        Returns:
            A Web3 instance connected to the specified network.
        """
        if network in self.web3_instances:
            return self.web3_instances[network]

        endpoint_url = config.get(f"infura.endpoints.{network}")
        if not endpoint_url:
            raise ValueError(f"INFURA endpoint for network '{network}' not found in configuration.")

        full_endpoint_url = f"{endpoint_url}{self.infura_api_key}"

        web3 = Web3(Web3.HTTPProvider(full_endpoint_url))

        if web3.is_connected():
            self.web3_instances[network] = web3
            logger.info(f"Successfully connected to {network}")
            return web3
        else:
            raise ConnectionError(f"Failed to connect to {network}")

    def get_balance(self, address: str, network: str = "ethereum_mainnet") -> float:
        """
        Get the balance of an address on a specific network.

        Args:
            address: The address to check the balance of.
            network: The network to check the balance on.

        Returns:
            The balance in the native currency of the network.
        """
        web3 = self.get_web3_instance(network)
        balance_wei = web3.eth.get_balance(address)
        balance_ether = web3.from_wei(balance_wei, 'ether')
        return float(balance_ether)

    def send_transaction(self, from_address: str, to_address: str, amount: float, private_key: str, network: str = "ethereum_mainnet") -> str:
        """
        Send a transaction on a specific network.

        Args:
            from_address: The address to send the transaction from.
            to_address: The address to send the transaction to.
            amount: The amount to send.
            private_key: The private key of the from_address.
            network: The network to send the transaction on.

        Returns:
            The transaction hash.
        """
        web3 = self.get_web3_instance(network)

        nonce = web3.eth.get_transaction_count(from_address)

        tx = {
            'nonce': nonce,
            'to': to_address,
            'value': web3.to_wei(amount, 'ether'),
            'gas': 21000,
            'gasPrice': web3.eth.gas_price
        }

        signed_tx = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

        return web3.to_hex(tx_hash)

# Create a singleton instance for easy import
blockchain_manager = BlockchainManager()
