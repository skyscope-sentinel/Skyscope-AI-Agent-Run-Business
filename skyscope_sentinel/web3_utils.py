
import os
from web3 import Web3
from eth_account import Account
from bip39 import bip39_to_mini_secret, bip39_to_seed
from solders.keypair import Keypair

class Web3Utils:
    def __init__(self, infura_api_key, seed_phrase):
        self.infura_api_key = infura_api_key
        self.seed_phrase = seed_phrase
        self.web3 = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{self.infura_api_key}"))
        self.account = self.get_account_from_seed_phrase()

    def get_account_from_seed_phrase(self):
        private_key = bip39_to_mini_secret(self.seed_phrase, "").hex()
        return Account.from_key(private_key)

    def get_balance(self):
        balance = self.web3.eth.get_balance(self.account.address)
        return self.web3.from_wei(balance, 'ether')

    def send_transaction(self, to_address, amount):
        tx = {
            'to': to_address,
            'value': self.web3.to_wei(amount, 'ether'),
            'gas': 21000,
            'gasPrice': self.web3.eth.gas_price,
            'nonce': self.web3.eth.get_transaction_count(self.account.address),
        }
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.account.key)
        tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        return self.web3.to_hex(tx_hash)

    def get_solana_keypair(self):
        seed = bip39_to_seed(self.seed_phrase)
        return Keypair.from_seed(seed[:32])
