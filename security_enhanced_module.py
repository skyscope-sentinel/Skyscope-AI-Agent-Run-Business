#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - Enhanced Security Module
===========================================================

This module provides comprehensive security features for the Skyscope Sentinel
Intelligence AI system, implementing advanced protection mechanisms, authentication,
encryption, intrusion detection, and security monitoring capabilities.

Features:
- Multi-factor authentication with TOTP, FIDO2/WebAuthn, and biometrics
- Advanced encryption and secure key management
- Real-time intrusion detection and prevention
- ML-based anomaly detection for security events
- Zero-trust architecture implementation
- Secure communication protocols with TLS 1.3 and secure messaging
- Threat intelligence integration with multiple feeds
- Comprehensive security audit logging
- Automated vulnerability scanning and management
- Role-based access control with fine-grained permissions
- Secure API gateway with rate limiting and request validation
- Advanced DDoS protection mechanisms
- Blockchain-based security for immutable audit trails
- Quantum-resistant cryptographic algorithms

Dependencies:
- cryptography, pyotp, fido2, pyjwt, passlib
- scikit-learn, tensorflow (for ML-based detection)
- requests, aiohttp (for API communication)
- sqlalchemy (for database interactions)
- pyyaml (for configuration)
"""

import os
import sys
import json
import time
import uuid
import base64
import socket
import hashlib
import logging
import datetime
import ipaddress
import threading
import traceback
import subprocess
import urllib.parse
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import deque, defaultdict

# Cryptography and authentication
try:
    import cryptography
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, hmac
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec, x25519
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
    from cryptography.x509 import load_pem_x509_certificate
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography library not available. Encryption features will be limited.")

try:
    import pyotp
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False
    logging.warning("PyOTP library not available. TOTP-based MFA will be disabled.")

try:
    from fido2.client import ClientData, ClientError
    from fido2.server import Fido2Server, RelyingParty
    from fido2.webauthn import PublicKeyCredentialRpEntity
    FIDO2_AVAILABLE = True
except ImportError:
    FIDO2_AVAILABLE = False
    logging.warning("FIDO2 library not available. WebAuthn-based MFA will be disabled.")

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT library not available. JWT functionality will be limited.")

try:
    from passlib.hash import argon2
    ARGON2_AVAILABLE = True
except ImportError:
    ARGON2_AVAILABLE = False
    logging.warning("Passlib/Argon2 not available. Will use fallback password hashing.")

# Machine learning for anomaly detection
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ML_BASIC_AVAILABLE = True
except ImportError:
    ML_BASIC_AVAILABLE = False
    logging.warning("Basic ML libraries not available. Anomaly detection will be limited.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
    ML_ADVANCED_AVAILABLE = True
except ImportError:
    ML_ADVANCED_AVAILABLE = False
    logging.warning("TensorFlow not available. Advanced anomaly detection will be disabled.")

# Try to import blockchain integration
try:
    from blockchain_crypto_integration import BlockchainType, WalletManager
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    logging.warning("Blockchain integration not available. Blockchain security features will be limited.")

# Try to import quantum enhanced AI
try:
    from quantum_enhanced_ai import QuantumEnhancedAI, QuantumRandomNumberGenerator
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logging.warning("Quantum Enhanced AI not available. Quantum-resistant features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/security.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("security_enhanced")

# --- Constants and Enums ---

class AuthMethod(Enum):
    """Authentication methods supported by the system."""
    PASSWORD = "password"
    TOTP = "totp"
    FIDO2 = "fido2"
    BIOMETRIC = "biometric"
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"
    CERTIFICATE = "certificate"
    OAUTH = "oauth"
    JWT = "jwt"

class AccessLevel(Enum):
    """Access levels for authorization."""
    NONE = 0
    READ = 10
    WRITE = 20
    UPDATE = 30
    DELETE = 40
    ADMIN = 50
    SYSTEM = 60

class SecurityEventType(Enum):
    """Types of security events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    ENCRYPTION = "encryption"
    INTRUSION = "intrusion"
    ANOMALY = "anomaly"
    VULNERABILITY = "vulnerability"
    CONFIGURATION = "configuration"
    AUDIT = "audit"
    NETWORK = "network"
    SYSTEM = "system"
    APPLICATION = "application"
    DATA = "data"
    API = "api"
    USER = "user"

class SecurityEventSeverity(Enum):
    """Severity levels for security events."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"
    ED25519 = "ed25519"
    X25519 = "x25519"
    
    # Post-quantum algorithms
    DILITHIUM = "dilithium"  # Signature
    KYBER = "kyber"          # Key exchange
    SPHINCS = "sphincs"      # Signature
    NTRU = "ntru"           # Key exchange
    SIKE = "sike"           # Key exchange

class HashAlgorithm(Enum):
    """Supported hash algorithms."""
    SHA256 = "sha256"
    SHA384 = "sha384"
    SHA512 = "sha512"
    SHA3_256 = "sha3-256"
    SHA3_384 = "sha3-384"
    SHA3_512 = "sha3-512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"

class KeyType(Enum):
    """Types of cryptographic keys."""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    HMAC = "hmac"
    PASSWORD = "password"
    SEED = "seed"
    CERTIFICATE = "certificate"

# --- Data Models ---

@dataclass
class SecurityEvent:
    """Data model for security events."""
    id: str
    timestamp: datetime.datetime
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    source: str
    message: str
    details: Dict[str, Any]
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime.datetime] = None
    resolution_details: Optional[str] = None

@dataclass
class UserCredential:
    """Data model for user credentials."""
    user_id: str
    auth_method: AuthMethod
    credential_id: str
    credential_data: Dict[str, Any]
    created_at: datetime.datetime
    last_used: Optional[datetime.datetime] = None
    expires_at: Optional[datetime.datetime] = None
    is_active: bool = True

@dataclass
class UserSession:
    """Data model for user sessions."""
    session_id: str
    user_id: str
    created_at: datetime.datetime
    expires_at: datetime.datetime
    ip_address: str
    user_agent: str
    auth_methods: List[AuthMethod]
    is_active: bool = True
    last_activity: Optional[datetime.datetime] = None
    mfa_completed: bool = False
    access_level: AccessLevel = AccessLevel.READ
    device_id: Optional[str] = None
    location: Optional[Dict[str, Any]] = None

@dataclass
class EncryptionKey:
    """Data model for encryption keys."""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_material: bytes
    created_at: datetime.datetime
    expires_at: Optional[datetime.datetime] = None
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    rotation_policy: Optional[str] = None
    last_rotated: Optional[datetime.datetime] = None

@dataclass
class AccessPolicy:
    """Data model for access control policies."""
    policy_id: str
    name: str
    description: str
    resources: List[str]
    actions: List[str]
    conditions: Optional[Dict[str, Any]] = None
    effect: str = "allow"  # "allow" or "deny"
    priority: int = 0
    created_at: datetime.datetime = datetime.datetime.now()
    updated_at: Optional[datetime.datetime] = None
    version: int = 1

@dataclass
class VulnerabilityReport:
    """Data model for vulnerability reports."""
    id: str
    timestamp: datetime.datetime
    severity: SecurityEventSeverity
    title: str
    description: str
    affected_components: List[str]
    cve_id: Optional[str] = None
    cvss_score: Optional[float] = None
    remediation_steps: Optional[str] = None
    status: str = "open"  # open, in_progress, resolved, false_positive
    assigned_to: Optional[str] = None
    resolved_at: Optional[datetime.datetime] = None
    resolution_details: Optional[str] = None

# --- Multi-Factor Authentication ---

class MFAManager:
    """Manages multi-factor authentication methods."""
    
    def __init__(self, storage_path: str = "security/mfa"):
        """Initialize the MFA manager.
        
        Args:
            storage_path: Path for storing MFA data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        self.user_credentials = {}  # user_id -> List[UserCredential]
        self.totp_secrets = {}  # user_id -> Dict[credential_id, secret]
        
        # Initialize FIDO2 server if available
        self.fido2_server = None
        if FIDO2_AVAILABLE:
            try:
                rp = PublicKeyCredentialRpEntity(
                    id="skyscope.ai",
                    name="Skyscope Sentinel AI"
                )
                self.fido2_server = Fido2Server(rp)
            except Exception as e:
                logger.error(f"Error initializing FIDO2 server: {str(e)}")
    
    def setup_totp(self, user_id: str, issuer: str = "Skyscope Sentinel") -> Dict[str, Any]:
        """Set up TOTP-based MFA for a user.
        
        Args:
            user_id: User identifier
            issuer: Issuer name for TOTP
            
        Returns:
            Dictionary with TOTP setup information
        """
        if not TOTP_AVAILABLE:
            raise ImportError("PyOTP library not available. TOTP-based MFA is disabled.")
        
        # Generate a random secret
        secret = pyotp.random_base32()
        
        # Create TOTP object
        totp = pyotp.TOTP(secret)
        
        # Create a provisioning URI for QR code
        provisioning_uri = totp.provisioning_uri(name=user_id, issuer_name=issuer)
        
        # Generate a credential ID
        credential_id = str(uuid.uuid4())
        
        # Store the TOTP secret
        if user_id not in self.totp_secrets:
            self.totp_secrets[user_id] = {}
        
        self.totp_secrets[user_id][credential_id] = secret
        
        # Create a user credential
        credential = UserCredential(
            user_id=user_id,
            auth_method=AuthMethod.TOTP,
            credential_id=credential_id,
            credential_data={"secret": secret},
            created_at=datetime.datetime.now(),
            is_active=True
        )
        
        # Store the credential
        if user_id not in self.user_credentials:
            self.user_credentials[user_id] = []
        
        self.user_credentials[user_id].append(credential)
        
        # Save to storage
        self._save_totp_secrets()
        
        return {
            "credential_id": credential_id,
            "secret": secret,
            "provisioning_uri": provisioning_uri,
            "qr_code_url": f"https://chart.googleapis.com/chart?chs=200x200&chld=M|0&cht=qr&chl={urllib.parse.quote(provisioning_uri)}"
        }
    
    def verify_totp(self, user_id: str, totp_code: str, credential_id: Optional[str] = None) -> bool:
        """Verify a TOTP code.
        
        Args:
            user_id: User identifier
            totp_code: TOTP code to verify
            credential_id: Optional credential ID (if user has multiple TOTP credentials)
            
        Returns:
            True if verification succeeds, False otherwise
        """
        if not TOTP_AVAILABLE:
            raise ImportError("PyOTP library not available. TOTP-based MFA is disabled.")
        
        if user_id not in self.totp_secrets:
            logger.warning(f"No TOTP secrets found for user {user_id}")
            return False
        
        # If credential_id is provided, verify only that credential
        if credential_id:
            if credential_id not in self.totp_secrets[user_id]:
                logger.warning(f"TOTP credential {credential_id} not found for user {user_id}")
                return False
            
            secret = self.totp_secrets[user_id][credential_id]
            totp = pyotp.TOTP(secret)
            return totp.verify(totp_code)
        
        # Otherwise, try all credentials
        for cred_id, secret in self.totp_secrets[user_id].items():
            totp = pyotp.TOTP(secret)
            if totp.verify(totp_code):
                # Update last used timestamp
                for cred in self.user_credentials.get(user_id, []):
                    if cred.credential_id == cred_id and cred.auth_method == AuthMethod.TOTP:
                        cred.last_used = datetime.datetime.now()
                        break
                
                return True
        
        return False
    
    def begin_fido2_registration(self, user_id: str, display_name: str) -> Dict[str, Any]:
        """Begin FIDO2/WebAuthn registration process.
        
        Args:
            user_id: User identifier
            display_name: User's display name
            
        Returns:
            Dictionary with registration options for the client
        """
        if not FIDO2_AVAILABLE:
            raise ImportError("FIDO2 library not available. WebAuthn-based MFA is disabled.")
        
        if not self.fido2_server:
            raise RuntimeError("FIDO2 server not initialized")
        
        # Generate registration options
        options, state = self.fido2_server.register_begin(
            {
                "id": user_id.encode(),
                "name": user_id,
                "displayName": display_name
            },
            user_verification="preferred",
            authenticator_attachment="cross-platform"
        )
        
        # Store state for later verification
        state_file = self.storage_path / f"fido2_state_{user_id}.json"
        with open(state_file, "w") as f:
            json.dump(state, f)
        
        return options
    
    def complete_fido2_registration(self, user_id: str, registration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete FIDO2/WebAuthn registration process.
        
        Args:
            user_id: User identifier
            registration_data: Registration data from the client
            
        Returns:
            Dictionary with registration result
        """
        if not FIDO2_AVAILABLE:
            raise ImportError("FIDO2 library not available. WebAuthn-based MFA is disabled.")
        
        if not self.fido2_server:
            raise RuntimeError("FIDO2 server not initialized")
        
        # Load state
        state_file = self.storage_path / f"fido2_state_{user_id}.json"
        if not state_file.exists():
            raise ValueError(f"No FIDO2 registration in progress for user {user_id}")
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        # Complete registration
        auth_data = self.fido2_server.register_complete(state, registration_data)
        
        # Generate a credential ID
        credential_id = str(uuid.uuid4())
        
        # Create a user credential
        credential = UserCredential(
            user_id=user_id,
            auth_method=AuthMethod.FIDO2,
            credential_id=credential_id,
            credential_data=auth_data,
            created_at=datetime.datetime.now(),
            is_active=True
        )
        
        # Store the credential
        if user_id not in self.user_credentials:
            self.user_credentials[user_id] = []
        
        self.user_credentials[user_id].append(credential)
        
        # Save to storage
        self._save_fido2_credentials(user_id, auth_data)
        
        # Clean up state file
        state_file.unlink()
        
        return {
            "credential_id": credential_id,
            "status": "registered"
        }
    
    def begin_fido2_authentication(self, user_id: str) -> Dict[str, Any]:
        """Begin FIDO2/WebAuthn authentication process.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with authentication options for the client
        """
        if not FIDO2_AVAILABLE:
            raise ImportError("FIDO2 library not available. WebAuthn-based MFA is disabled.")
        
        if not self.fido2_server:
            raise RuntimeError("FIDO2 server not initialized")
        
        # Load credentials
        credentials = self._load_fido2_credentials(user_id)
        if not credentials:
            raise ValueError(f"No FIDO2 credentials found for user {user_id}")
        
        # Generate authentication options
        options, state = self.fido2_server.authenticate_begin(credentials)
        
        # Store state for later verification
        state_file = self.storage_path / f"fido2_auth_state_{user_id}.json"
        with open(state_file, "w") as f:
            json.dump(state, f)
        
        return options
    
    def complete_fido2_authentication(self, user_id: str, auth_data: Dict[str, Any]) -> bool:
        """Complete FIDO2/WebAuthn authentication process.
        
        Args:
            user_id: User identifier
            auth_data: Authentication data from the client
            
        Returns:
            True if authentication succeeds, False otherwise
        """
        if not FIDO2_AVAILABLE:
            raise ImportError("FIDO2 library not available. WebAuthn-based MFA is disabled.")
        
        if not self.fido2_server:
            raise RuntimeError("FIDO2 server not initialized")
        
        # Load state
        state_file = self.storage_path / f"fido2_auth_state_{user_id}.json"
        if not state_file.exists():
            raise ValueError(f"No FIDO2 authentication in progress for user {user_id}")
        
        with open(state_file, "r") as f:
            state = json.load(f)
        
        # Load credentials
        credentials = self._load_fido2_credentials(user_id)
        if not credentials:
            raise ValueError(f"No FIDO2 credentials found for user {user_id}")
        
        try:
            # Complete authentication
            self.fido2_server.authenticate_complete(
                state,
                credentials,
                auth_data
            )
            
            # Update last used timestamp for the credential
            for cred in self.user_credentials.get(user_id, []):
                if cred.auth_method == AuthMethod.FIDO2:
                    cred.last_used = datetime.datetime.now()
                    break
            
            # Clean up state file
            state_file.unlink()
            
            return True
        except Exception as e:
            logger.error(f"FIDO2 authentication failed: {str(e)}")
            return False
    
    def generate_recovery_codes(self, user_id: str, count: int = 10) -> List[str]:
        """Generate recovery codes for a user.
        
        Args:
            user_id: User identifier
            count: Number of recovery codes to generate
            
        Returns:
            List of recovery codes
        """
        # Generate random recovery codes
        recovery_codes = []
        for _ in range(count):
            code = base64.b32encode(os.urandom(10)).decode('utf-8')
            code = code.replace('=', '').lower()
            # Format as xxxx-xxxx-xxxx
            code = f"{code[:4]}-{code[4:8]}-{code[8:12]}"
            recovery_codes.append(code)
        
        # Hash the recovery codes
        hashed_codes = []
        for code in recovery_codes:
            if ARGON2_AVAILABLE:
                hashed_code = argon2.hash(code)
            else:
                salt = os.urandom(16)
                hashed_code = hashlib.pbkdf2_hmac('sha256', code.encode(), salt, 100000).hex()
                hashed_code = f"pbkdf2:sha256:100000${salt.hex()}${hashed_code}"
            
            hashed_codes.append(hashed_code)
        
        # Create a credential
        credential_id = str(uuid.uuid4())
        credential = UserCredential(
            user_id=user_id,
            auth_method=AuthMethod.PASSWORD,  # Using PASSWORD type for recovery codes
            credential_id=credential_id,
            credential_data={"recovery_codes": hashed_codes, "used_codes": []},
            created_at=datetime.datetime.now(),
            is_active=True
        )
        
        # Store the credential
        if user_id not in self.user_credentials:
            self.user_credentials[user_id] = []
        
        self.user_credentials[user_id].append(credential)
        
        # Save to storage
        self._save_recovery_codes(user_id, credential)
        
        return recovery_codes
    
    def verify_recovery_code(self, user_id: str, recovery_code: str) -> bool:
        """Verify a recovery code.
        
        Args:
            user_id: User identifier
            recovery_code: Recovery code to verify
            
        Returns:
            True if verification succeeds, False otherwise
        """
        if user_id not in self.user_credentials:
            return False
        
        # Find recovery code credential
        recovery_cred = None
        for cred in self.user_credentials[user_id]:
            if cred.auth_method == AuthMethod.PASSWORD and "recovery_codes" in cred.credential_data:
                recovery_cred = cred
                break
        
        if not recovery_cred:
            return False
        
        # Check if the code matches any of the hashed codes
        hashed_codes = recovery_cred.credential_data["recovery_codes"]
        used_codes = recovery_cred.credential_data.get("used_codes", [])
        
        for i, hashed_code in enumerate(hashed_codes):
            if i in used_codes:
                continue
            
            # Verify the code
            if ARGON2_AVAILABLE:
                try:
                    if argon2.verify(recovery_code, hashed_code):
                        # Mark the code as used
                        used_codes.append(i)
                        recovery_cred.credential_data["used_codes"] = used_codes
                        self._save_recovery_codes(user_id, recovery_cred)
                        return True
                except Exception:
                    continue
            else:
                # Parse the hash string
                try:
                    parts = hashed_code.split('$')
                    if len(parts) != 3:
                        continue
                    
                    algorithm_info = parts[0].split(':')
                    if len(algorithm_info) != 3 or algorithm_info[0] != "pbkdf2" or algorithm_info[1] != "sha256":
                        continue
                    
                    iterations = int(algorithm_info[2])
                    salt = bytes.fromhex(parts[1])
                    stored_hash = parts[2]
                    
                    # Compute the hash
                    computed_hash = hashlib.pbkdf2_hmac('sha256', recovery_code.encode(), salt, iterations).hex()
                    
                    if computed_hash == stored_hash:
                        # Mark the code as used
                        used_codes.append(i)
                        recovery_cred.credential_data["used_codes"] = used_codes
                        self._save_recovery_codes(user_id, recovery_cred)
                        return True
                except Exception:
                    continue
        
        return False
    
    def _save_totp_secrets(self):
        """Save TOTP secrets to storage."""
        for user_id, secrets in self.totp_secrets.items():
            user_dir = self.storage_path / user_id
            user_dir.mkdir(exist_ok=True, parents=True)
            
            totp_file = user_dir / "totp_secrets.json"
            with open(totp_file, "w") as f:
                json.dump(secrets, f)
    
    def _save_fido2_credentials(self, user_id: str, auth_data: Dict[str, Any]):
        """Save FIDO2 credentials to storage."""
        user_dir = self.storage_path / user_id
        user_dir.mkdir(exist_ok=True, parents=True)
        
        fido2_file = user_dir / "fido2_credentials.json"
        
        # Load existing credentials
        credentials = []
        if fido2_file.exists():
            with open(fido2_file, "r") as f:
                credentials = json.load(f)
        
        # Add new credential
        credentials.append(auth_data)
        
        # Save credentials
        with open(fido2_file, "w") as f:
            json.dump(credentials, f)
    
    def _load_fido2_credentials(self, user_id: str) -> List[Dict[str, Any]]:
        """Load FIDO2 credentials from storage."""
        fido2_file = self.storage_path / user_id / "fido2_credentials.json"
        if not fido2_file.exists():
            return []
        
        with open(fido2_file, "r") as f:
            return json.load(f)
    
    def _save_recovery_codes(self, user_id: str, credential: UserCredential):
        """Save recovery codes to storage."""
        user_dir = self.storage_path / user_id
        user_dir.mkdir(exist_ok=True, parents=True)
        
        recovery_file = user_dir / f"recovery_codes_{credential.credential_id}.json"
        with open(recovery_file, "w") as f:
            json.dump(credential.credential_data, f)

# --- Encryption and Key Management ---

class KeyManager:
    """Manages encryption keys and cryptographic operations."""
    
    def __init__(self, storage_path: str = "security/keys"):
        """Initialize the key manager.
        
        Args:
            storage_path: Path for storing keys
        """
        if not CRYPTO_AVAILABLE:
            raise ImportError("Cryptography library not available. Encryption features are disabled.")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True, parents=True)
        
        self.keys = {}  # key_id -> EncryptionKey
        self.master_key = None
        self.master_key_id = None
        
        # Load or create master key
        self._initialize_master_key()
        
        # Load existing keys
        self._load_keys()
    
    def _initialize_master_key(self):
        """Initialize the master key for key encryption."""
        master_key_file = self.storage_path / "master_key.json"
        
        if master_key_file.exists():
            # Load existing master key
            try:
                with open(master_key_file, "r") as f:
                    master_key_data = json.load(f)
                
                # Decrypt the master key using the environment variable or prompt for password
                password = os.environ.get("SKYSCOPE_MASTER_PASSWORD")
                if not password:
                    import getpass
                    password = getpass.getpass("Enter master key password: ")
                
                salt = bytes.fromhex(master_key_data["salt"])
                encrypted_key = bytes.fromhex(master_key_data["encrypted_key"])
                
                # Derive key from password
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                
                # Decrypt the master key
                fernet = Fernet(key)
                master_key_bytes = fernet.decrypt(encrypted_key)
                
                self.master_key = master_key_bytes
                self.master_key_id = master_key_data["key_id"]
                
                logger.info("Master key loaded successfully")
            except Exception as e:
                logger.error(f"Error loading master key: {str(e)}")
                raise
        else:
            # Create a new master key
            try:
                # Generate a random key
                self.master_key = os.urandom(32)
                self.master_key_id = str(uuid.uuid4())
                
                # Get password for encrypting the master key
                password = os.environ.get("SKYSCOPE_MASTER_PASSWORD")
                if not password:
                    import getpass
                    password = getpass.getpass("Create master key password: ")
                    password_confirm = getpass.getpass("Confirm master key password: ")
                    
                    if password != password_confirm:
                        raise ValueError("Passwords do not match")
                
                # Generate salt
                salt = os.urandom(16)
                
                # Derive key from password
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
                
                # Encrypt the master key
                fernet = Fernet(key)
                encrypted_key = fernet.encrypt(self.master_key)
                
                # Save the encrypted master key
                master_key_data = {
                    "key_id": self.master_key_id,
                    "salt": salt.hex(),
                    "encrypted_key": encrypted_key.hex(),
                    "created_at": datetime.datetime.now().isoformat()
                }
                
                with open(master_key_file, "w") as f:
                    json.dump(master_key_data, f)
                
                logger.info("New master key created successfully")
            except Exception as e:
                logger.error(f"Error creating master key: {str(e)}")
                raise
    
    def _load_keys(self):
        """Load encryption keys from storage."""
        keys_dir = self.storage_path / "keys"
        keys_dir.mkdir(exist_ok=True, parents=True)
        
        for key_file in keys_dir.glob("*.key"):
            try:
                # Decrypt the key file using the master key
                with open(key_file, "rb") as f:
                    encrypted_data = f.read()
                
                fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
                decrypted_data = fernet.decrypt(encrypted_data)
                
                key_data = json.loads(decrypted_data.decode())
                
                key = EncryptionKey(
                    key_id=key_data["key_id"],
                    key_type=KeyType(key_data["key_type"]),
                    algorithm=EncryptionAlgorithm(key_data["algorithm"]),
                    key_material=bytes.fromhex(key_data["key_material"]),
                    created_at=datetime.datetime.fromisoformat(key_data["created_at"]),
                    expires_at=datetime.datetime.fromisoformat(key_data["expires_at"]) if key_data.get("expires_at") else None,
                    is_active=key_data.get("is_active", True),
                    metadata=key_data.get("metadata"),
                    rotation_policy=key_data.get("rotation_policy"),
                    last_rotated=datetime.datetime.fromisoformat(key_data["last_rotated"]) if key_data.get("last_rotated") else None
                )
                
                self.keys[key.key_id] = key
            except Exception as e:
                logger.error(f"Error loading key from {key_file}: {str(e)}")
    
    def generate_key(self, key_type: KeyType, algorithm: EncryptionAlgorithm, 
                    expires_in_days: Optional[int] = 365, 
                    metadata: Optional[Dict[str, Any]] = None,
                    rotation_policy: Optional[str] = None) -> str:
        """Generate a new encryption key.
        
        Args:
            key_type: Type of key to generate
            algorithm: Encryption algorithm
            expires_in_days: Number of days until the key expires (None for no expiration)
            metadata: Optional metadata for the key
            rotation_policy: Optional rotation policy (e.g., "90days")
            
        Returns:
            Key ID
        """
        key_id = str(uuid.uuid4())
        key_material = None
        
        # Generate key material based on algorithm
        if algorithm == EncryptionAlgorithm.AES_256_GCM or algorithm == EncryptionAlgorithm.AES_256_CBC:
            key_material = os.urandom(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_material = os.urandom(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.RSA_2048:
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                key_material = private_key.private_bytes(
                    encoding=Encoding.DER,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
            elif key_type == KeyType.ASYMMETRIC_PUBLIC:
                raise ValueError("Cannot generate public key directly. Generate private key first.")
        elif algorithm == EncryptionAlgorithm.RSA_4096:
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=4096
                )
                key_material = private_key.private_bytes(
                    encoding=Encoding.DER,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
            elif key_type == KeyType.ASYMMETRIC_PUBLIC:
                raise ValueError("Cannot generate public key directly. Generate private key first.")
        elif algorithm == EncryptionAlgorithm.ECDSA_P256:
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                private_key = ec.generate_private_key(ec.SECP256R1())
                key_material = private_key.private_bytes(
                    encoding=Encoding.DER,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
            elif key_type == KeyType.ASYMMETRIC_PUBLIC:
                raise ValueError("Cannot generate public key directly. Generate private key first.")
        elif algorithm == EncryptionAlgorithm.ECDSA_P384:
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                private_key = ec.generate_private_key(ec.SECP384R1())
                key_material = private_key.private_bytes(
                    encoding=Encoding.DER,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
            elif key_type == KeyType.ASYMMETRIC_PUBLIC:
                raise ValueError("Cannot generate public key directly. Generate private key first.")
        elif algorithm == EncryptionAlgorithm.ED25519:
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                private_key = ec.generate_private_key(ec.Ed25519())
                key_material = private_key.private_bytes(
                    encoding=Encoding.DER,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
            elif key_type == KeyType.ASYMMETRIC_PUBLIC:
                raise ValueError("Cannot generate public key directly. Generate private key first.")
        elif algorithm == EncryptionAlgorithm.X25519:
            if key_type == KeyType.ASYMMETRIC_PRIVATE:
                private_key = x25519.X25519PrivateKey.generate()
                key_material = private_key.private_bytes(
                    encoding=Encoding.DER,
                    format=PrivateFormat.PKCS8,
                    encryption_algorithm=NoEncryption()
                )
            elif key_type == KeyType.ASYMMETRIC_PUBLIC:
                raise ValueError("Cannot generate public key directly. Generate private key first.")
        elif algorithm == EncryptionAlgorithm.HMAC:
            key_material = os.urandom(32)  # 256 bits
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # Calculate expiration date
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.datetime.now() + datetime.timedelta(days=expires_in_days)
        
        # Create key object
        key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_material=key_material,
            created_at=datetime.datetime.now(),
            expires_at=expires_at,
            is_active=True,
            metadata=metadata,
            rotation_policy=rotation_policy
        )
        
        # Store the key
        self.keys[key_id] = key
        
        # Save the key to storage
        self._save_key(key)
        
        return key_id
    
    def derive_public_key(self, private_key_id: str) -> str:
        """Derive a public key from a private key.
        
        Args:
            private_key_id: ID of the private key
            
        Returns:
            ID of the derived public key
        """
        if private_key_id not in self.keys:
            raise ValueError(f"Private key not found: {private_key_id}")
        
        private_key = self.keys[private_key_id]
        if private_key.key_type != KeyType.ASYMMETRIC_PRIVATE:
            raise ValueError(f"Key is not a private key: {private_key_id}")
        
        # Load the private key
        if private_key.algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            priv_key = load_pem_private_key(private_key.key_material, password=None)
            pub_key = priv_key.public_key()
            pub_key_material = pub_key.public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo
            )
        elif private_key.algorithm in [EncryptionAlgorithm.ECDSA_P256, EncryptionAlgorithm.ECDSA_P384]:
            priv_key = load_pem_private_key(private_key.key_material, password=None)
            pub_key = priv_key.public_key()
            pub_key_material = pub_key.public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo
            )
        elif private_key.algorithm == EncryptionAlgorithm.ED25519:
            priv_key = load_pem_private_key(private_key.key_material, password=None)
            pub_key = priv_key.public_key()
            pub_key_material = pub_key.public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo
            )
        elif private_key.algorithm == EncryptionAlgorithm.X25519:
            priv_key = x25519.X25519PrivateKey.from_private_bytes(private_key.key_material)
            pub_key = priv_key.public_key()
            pub_key_material = pub_key.public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo
            )
        else:
            raise ValueError(f"Unsupported algorithm for key derivation: {private_key.algorithm}")
        
        # Create public key
        public_key_id = str(uuid.uuid4())
        public_key = EncryptionKey(
            key_id=public_key_id,
            key_type=KeyType.ASYMMETRIC_PUBLIC,
            algorithm=private_key.algorithm,
            key_material=pub_key_material,
            created_at=datetime.datetime.now(),
            expires_at=private_key.expires_at,
            is_active=True,
            metadata={"derived_from": private_key_id},
            rotation_policy=private_key.rotation_policy
        )
        
        # Store the key
        self.keys[public_key_id] = public_key
        
        # Save the key to storage
        self._save_key(public_key)
        
        return public_key_id
    
    def encrypt(self, data: bytes, key_id: str, aad: Optional[bytes] = None) -> bytes:
        """Encrypt data using the specified key.
        
        Args:
            data: Data to encrypt
            key_id: ID of the key to use
            aad: Additional authenticated data (for AEAD ciphers)
            
        Returns:
            Encrypted data
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        key = self.keys[key_id]
        
        if not key.is_active:
            raise ValueError(f"Key is not active: {key_id}")
        
        if key.expires_at and key.expires_at < datetime.datetime.now():
            raise ValueError(f"Key has expired: {key_id}")
        
        # Encrypt based on algorithm
        if key.algorithm == EncryptionAlgorithm.AES_256_GCM:
            # Generate a random IV
            iv = os.urandom(12)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key.key_material),
                modes.GCM(iv)
            )
            
            encryptor = cipher.encryptor()
            
            # Add AAD if provided
            if aad:
                encryptor.authenticate_additional_data(aad)
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            # Get tag
            tag = encryptor.tag
            
            # Format: IV || Tag || Ciphertext
            return iv + tag + ciphertext
        
        elif key.algorithm == EncryptionAlgorithm.AES_256_CBC:
            # Generate a random IV
            iv = os.urandom(16)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key.key_material),
                modes.CBC(iv)
            )
            
            encryptor = cipher.encryptor()
            
            # Pad data
            padder = padding.PKCS7(algorithms.AES.block_size).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            # Encrypt data
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            # Format: IV || Ciphertext
            return iv + ciphertext
        
        elif key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            # Generate a random nonce
            nonce = os.urandom(12)
            
            # Create cipher
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            cipher = ChaCha20Poly1305(key.key_material)
            
            # Encrypt data
            ciphertext = cipher.encrypt(nonce, data, aad)
            
            # Format: Nonce || Ciphertext
            return nonce + ciphertext
        
        elif key.algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            if key.key_type == KeyType.ASYMMETRIC_PUBLIC:
                # Load public key
                public_key = load_pem_public_key(key.key_material)
                
                # Encrypt data
                ciphertext = public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                return ciphertext
            else:
                raise ValueError("RSA encryption requires a public key")
        
        elif key.algorithm == EncryptionAlgorithm.X25519:
            raise ValueError("X25519 is a key exchange algorithm, not an encryption algorithm")
        
        else:
            raise ValueError(f"Unsupported algorithm for encryption: {key.algorithm}")
    
    def decrypt(self, encrypted_data: bytes, key_id: str, aad: Optional[bytes] = None) -> bytes:
        """Decrypt data using the specified key.
        
        Args:
            encrypted_data: Encrypted data
            key_id: ID of the key to use
            aad: Additional authenticated data (for AEAD ciphers)
            
        Returns:
            Decrypted data
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        key = self.keys[key_id]
        
        if not key.is_active:
            raise ValueError(f"Key is not active: {key_id}")
        
        # Decrypt based on algorithm
        if key.algorithm == EncryptionAlgorithm.AES_256_GCM:
            # Format: IV (12 bytes) || Tag (16 bytes) || Ciphertext
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key.key_material),
                modes.GCM(iv, tag)
            )
            
            decryptor = cipher.decryptor()
            
            # Add AAD if provided
            if aad:
                decryptor.authenticate_additional_data(aad)
            
            # Decrypt data
            return decryptor.update(ciphertext) + decryptor.finalize()
        
        elif key.algorithm == EncryptionAlgorithm.AES_256_CBC:
            # Format: IV (16 bytes) || Ciphertext
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(key.key_material),
                modes.CBC(iv)
            )
            
            decryptor = cipher.decryptor()
            
            # Decrypt data
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Unpad data
            unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()
        
        elif key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            # Format: Nonce (12 bytes) || Ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Create cipher
            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            cipher = ChaCha20Poly1305(key.key_material)
            
            # Decrypt data
            return cipher.decrypt(nonce, ciphertext, aad)
        
        elif key.algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            if key.key_type == KeyType.ASYMMETRIC_PRIVATE:
                # Load private key
                private_key = load_pem_private_key(key.key_material, password=None)
                
                # Decrypt data
                plaintext = private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                
                return plaintext
            else:
                raise ValueError("RSA decryption requires a private key")
        
        else:
            raise ValueError(f"Unsupported algorithm for decryption: {key.algorithm}")
    
    def sign(self, data: bytes, key_id: str) -> bytes:
        """Sign data using the specified key.
        
        Args:
            data: Data to sign
            key_id: ID of the key to use
            
        Returns:
            Signature
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        key = self.keys[key_id]
        
        if not key.is_active:
            raise ValueError(f"Key is not active: {key_id}")
        
        if key.expires_at and key.expires_at < datetime.datetime.now():
            raise ValueError(f"Key has expired: {key_id}")
        
        # Sign based on algorithm
        if key.algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
            if key.key_type == KeyType.ASYMMETRIC_PRIVATE:
                # Load private key
                private_key = load_pem_private_key(key.key_material, password=None)
                
                # Sign data
                signature = private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                return signature
            else:
                raise ValueError("RSA signing requires a private key")
        
        elif key.algorithm in [EncryptionAlgorithm.ECDSA_P256, EncryptionAlgorithm.ECDSA_P384]:
            if key.key_type == KeyType.ASYMMETRIC_PRIVATE:
                # Load private key
                private_key = load_pem_private_key(key.key_material, password=None)
                
                # Sign data
                signature = private_key.sign(
                    data,
                    ec.ECDSA(hashes.SHA256())
                )
                
                return signature
            else:
                raise ValueError("ECDSA signing requires a private key")
        
        elif key.algorithm == EncryptionAlgorithm.ED25519:
            if key.key_type == KeyType.ASYMMETRIC_PRIVATE:
                # Load private key
                private_key = load_pem_private_key(key.key_material, password=None)
                
                # Sign data
                signature = private_key.sign(data)
                
                return signature
            else:
                raise ValueError("Ed25519 signing requires a private key")
        
        elif key.key_type == KeyType.HMAC:
            # Create HMAC
            h = hmac.HMAC(key.key_material, hashes.SHA256())
            h.update(data)
            return h.finalize()
        
        else:
            raise ValueError(f"Unsupported algorithm for signing: {key.algorithm}")
    
    def verify(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify a signature using the specified key.
        
        Args:
            data: Data that was signed
            signature: Signature to verify
            key_id: ID of the key to use
            
        Returns:
            True if signature is valid, False otherwise
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        key = self.keys[key_id]
        
        if not key.is_active:
            raise ValueError(f"Key is not active: {key_id}")
        
        # Verify based on algorithm
        try:
            if key.algorithm in [EncryptionAlgorithm.RSA_2048, EncryptionAlgorithm.RSA_4096]:
                if key.key_type == KeyType.ASYMMETRIC_PUBLIC:
                    # Load public key
                    public_key = load_pem_public_key(key.key_material)
                    
                    # Verify signature
                    public_key.verify(
                        signature,
                        data,
                        padding.PSS(
                            mgf=padding.MGF1(hashes.SHA256()),
                            salt_length=padding.PSS.MAX_LENGTH
                        ),
                        hashes.SHA256()
                    )
                    
                    return True
                else:
                    raise ValueError("RSA verification requires a public key")
            
            elif key.algorithm in [EncryptionAlgorithm.ECDSA_P256, EncryptionAlgorithm.ECDSA_P384]:
                if key.key_type == KeyType.ASYMMETRIC_PUBLIC:
                    # Load public key
                    public_key = load_pem_public_key(key.key_material)
                    
                    # Verify signature
                    public_key.verify(
                        signature,
                        data,
                        ec.ECDSA(hashes.SHA256())
                    )
                    
                    return True
                else:
                    raise ValueError("ECDSA verification requires a public key")
            
            elif key.algorithm == EncryptionAlgorithm.ED25519:
                if key.key_type == KeyType.ASYMMETRIC_PUBLIC:
                    # Load public key
                    public_key = load_pem_public_key(key.key_material)
                    
                    # Verify signature
                    public_key.verify(signature, data)
                    
                    return True
                else:
                    raise ValueError("Ed25519 verification requires a public key")
            
            elif key.key_type == KeyType.HMAC:
                # Create HMAC
                h = hmac.HMAC(key.key_material, hashes.SHA256())
                h.update(data)
                h.verify(signature)
                
                return True
            
            else:
                raise ValueError(f"Unsupported algorithm for verification: {key.algorithm}")
        
        except Exception:
            return False
    
    def rotate_key(self, key_id: str) -> str:
        """Rotate a key by creating a new key with the same properties.
        
        Args:
            key_id: ID of the key to rotate
            
        Returns:
            ID of the new key
        """
        if key_id not in self.keys:
            raise ValueError(f"Key not found: {key_id}")
        
        old_key = self.keys[key_id]
        
        # Generate a new key with the same properties
        new_key_id = self.generate_key(
            key_type=old_key.key_type,
            algorithm=old_key.algorithm,
            expires_in_days=(old_key.expires_at - datetime.datetime.now()).days if old_key.expires_at else None,
            metadata=old_key.metadata,
            rotation_policy=old_key.rotation_policy
        )
        
        # Update the old key
        old_key.is_active = False
        old_key.metadata = old_key.metadata or {}
        old_key.metadata["rotated_to"] = new_key_id
        
        # Update the new key
        new_key = self.keys[new_key_id]
        new_key.metadata = new_key.metadata or {}
        new_key.metadata["rotated_from"] = key_id
        new_key.last_rotated = datetime.datetime.now()
        
        # Save both keys
        self._save_key(old_key)
        self._save_key(new_key)
        
        return new_key_id
    
    def _save_key(self, key: EncryptionKey):
        """Save a key to storage."""
        keys_dir = self.storage_path / "keys"
        keys_dir.mkdir(exist_ok=True, parents=True)
        
        key_file = keys_dir / f"{key.key_id}.key"
        
        # Prepare key data
        key_data = {
            "key_id": key.key_id,
            "key_type": key.key_type.value,
            "algorithm": key.algorithm.value,
            "key_material": key.key_material.hex(),
            "created_at": key.created_at.isoformat(),
            "expires_at": key.expires_at.isoformat() if key.expires_at else None,
            "is_active": key.is_active,
            "metadata": key.metadata,
            "rotation_policy": key.rotation_policy,
            "last_rotated": key.last_rotated.isoformat() if key.last_rotated else None
        }
        
        # Encrypt the key data using the master key
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        encrypted_data = fernet.encrypt(json.dumps(key_data).encode())
        
        # Save the encrypted key
        with open(key_file, "wb") as f:
            f.write(encrypted_data)

class QuantumResistantCrypto:
    """Implements quantum-resistant cryptographic algorithms."""
    
    def __init__(self):
        """Initialize the quantum-resistant crypto module."""
        self.qrng = None
        if QUANTUM_AVAILABLE:
            try:
                self.qrng = QuantumRandomNumberGenerator(n_qubits=8)
                logger.info("Quantum random number generator initialized")
            except Exception as e:
                logger.error(f"Error initializing quantum random number generator: {str(e)}")
        
        # Check for PQC libraries
        try:
            import liboqs
            self.liboqs_available = True
            logger.info("liboqs available for post-quantum cryptography")
        except ImportError:
            self.liboqs_available = False
            logger.warning("liboqs not available. Using classical fallbacks for post-quantum algorithms.")
    
    def generate_random_bytes(self, length: int) -> bytes:
        """Generate random bytes using quantum or classical sources.
        
        Args:
            length: Number of bytes to generate
            
        Returns:
            Random bytes
        """
        if self.qrng:
            try:
                # Use quantum random number generator
                random_numbers = self.qrng.generate_random_numbers(n_samples=length, min_value=0, max_value=256)
                return bytes([int(n) for n in random_numbers])
            except Exception as e:
                logger.error(f"Error generating quantum random numbers: {str(e)}")
                # Fall back to classical RNG
                return os.urandom(length)
        else:
            # Use classical RNG
            return os.urandom(length)
    
    def dilithium_keygen(self) -> Tuple[bytes, bytes]:
        """Generate a Dilithium key pair.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        if self.liboqs_available:
            try:
                import liboqs
                
                # Create a Dilithium signer
                signer = liboqs.Signature("Dilithium2")
                
                # Generate key pair
                public_key = signer.generate_keypair()
                private_key = signer.export_secret_key()
                
                return private_key, public_key
            except Exception as e:
                logger.error(f"Error generating Dilithium key pair: {str(e)}")
        
        # Fallback implementation (not actually quantum-resistant)
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        public_key = private_key.public_key()
        
        priv_bytes = private_key.private_bytes(
            encoding=Encoding.DER,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        )
        
        pub_bytes = public_key.public_bytes(
            encoding=Encoding.DER,
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        return priv_bytes, pub_bytes
    
    def dilithium_sign(self, private_key: bytes, message: bytes) -> bytes:
        """Sign a message using Dilithium.
        
        Args:
            private_key: Dilithium private key
            message: Message to sign
            
        Returns:
            Signature
        """
        if self.liboqs_available:
            try:
                import liboqs
                
                # Create a Dilithium signer
                signer = liboqs.Signature("Dilithium2")
                
                # Import the private key
                signer.import_secret_key(private_key)
                
                # Sign the message
                signature = signer.sign(message)
                
                return signature
            except Exception as e:
                logger.error(f"Error signing with Dilithium: {str(e)}")
        
        # Fallback implementation (not actually quantum-resistant)
        try:
            private_key_obj = load_pem_private_key(private_key, password=None)
            
            signature = private_key_obj.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature
        except Exception as e:
            logger.error(f"Error in fallback signing: {str(e)}")
            raise
    
    def dilithium_verify(self, public_key: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a Dilithium signature.
        
        Args:
            public_key: Dilithium public key
            message: Original message
            signature: Signature to verify
            
        Returns:
            True if signature is valid, False otherwise
        """
        if self.liboqs_available:
            try:
                import liboqs
                
                # Create a Dilithium verifier
                verifier = liboqs.Signature("Dilithium2")
                
                # Verify the signature
                return verifier.verify(message, signature, public_key)
            except Exception as e:
                logger.error(f"Error verifying with Dilithium: {str(e)}")
                return False
        
        # Fallback implementation (not actually quantum-resistant)
        try:
            public_key_obj = load_pem_public_key(public_key)
            
            public_key_obj.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
        except Exception:
            return False
    
    def kyber_keygen(self) -> Tuple[bytes, bytes]:
        """Generate a Kyber key pair.
        
        Returns:
            Tuple of (private_key, public_key)
        """
        if self.liboqs_available:
            try:
                import liboqs
                
                # Create a Kyber KEM
                kem = liboqs.KeyEncapsulation("Kyber512")
                
                # Generate key pair
                public_key = kem.generate_keypair()
                private_key = kem.export_secret_key()
                
                return private_key, public_key
            except Exception as e:
                logger.error(f"Error generating Kyber key pair: {str(e)}")
        
        # Fallback implementation (not actually quantum-resistant)
        private_key = x25519.X25519PrivateKey.generate()
        public_key = private_key.public_key()
        
        priv_bytes = private_key.private_bytes(
            encoding=Encoding.DER,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        )
        
        pub_bytes = public_key.public_bytes(
            encoding=Encoding.DER,
            format=PublicFormat.SubjectPublicKeyInfo
        )
        
        return priv_bytes, pub_bytes
    
    def kyber_encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate a shared secret using Kyber.
        
        Args:
            public_key: Kyber public key
            
        Returns:
            Tuple of (ciphertext, shared_secret)
        """
        if self.liboqs_available:
            try:
                import liboqs
                
                # Create a Kyber KEM
                kem = liboqs.KeyEncapsulation("Kyber512")
                
                # Encapsulate a shared secret
                ciphertext, shared_secret = kem.encap_secret(public_key)
                
                return ciphertext, shared_secret
            except Exception as e:
                logger.error(f"Error encapsulating with Kyber: {str(e)}")
        
        # Fallback implementation (not actually quantum-resistant)
        try:
            # Generate an ephemeral key pair
            ephemeral_private = x25519.X25519PrivateKey.generate()
            ephemeral_public = ephemeral_private.public_key()
            
            # Load the recipient's public key
            recipient_public = load_pem_public_key(public_key)
            
            # Compute shared secret
            shared_secret = ephemeral_private.exchange(recipient_public)
            
            # Return the ephemeral public key as the "ciphertext"
            ciphertext = ephemeral_public.public_bytes(
                encoding=Encoding.DER,
                format=PublicFormat.SubjectPublicKeyInfo
            )
            
            return ciphertext, shared_secret
        except Exception as e:
            logger.error(f"Error in fallback encapsulation: {str(e)}")
            raise
    
    def kyber_decapsulate(self, private_key: bytes, ciphertext: bytes) -> bytes:
        """Decapsulate a shared secret using Kyber.
        
        Args:
            private_key: Kyber private key
            ciphertext: Ciphertext from encapsulation
            
        Returns:
            Shared secret
        """
        if self.liboqs_available:
            try:
                import liboqs
                
                # Create a Kyber KEM
                kem = liboqs.KeyEncapsulation("Kyber512")
                
                # Import the private key
                kem.import_secret_key(private_key)
                
                # Decapsulate the shared secret
                shared_secret = kem.decap_secret(ciphertext)
                
                return shared_secret
            except Exception as e:
                logger.error(f"Error decapsulating with Kyber: {str(e)}")
        
        # Fallback implementation (not actually quantum-resistant)
        try:
            # Load the private key
            recipient_private = load_pem_private_key(private_key, password=None)
            
            # Load the ephemeral public key from the ciphertext
            ephemeral_public = load_pem_public_key(ciphertext)
            
            # Compute shared secret
            shared_secret = recipient_private.exchange(ephemeral_public)
            
            return shared_secret
        except Exception as e:
            logger.error(f"Error in fallback decapsulation: {str(e)}")
            raise

# --- Intrusion Detection ---

class IntrusionDetectionSystem:
    """Detects and responds to intrusion attempts."""
    
    def __init__(self, config_path: str = "security/ids_config.json"):
        """Initialize the intrusion detection system.
        
        Args:
            config_path: Path to the IDS configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.events = deque(maxlen=self.config.get("max_events", 10000))
        self.alerts = deque(maxlen=self.config.get("max_alerts", 1000))
        
        # Initialize event counters
        self.event_counters = defaultdict(int)
        self.ip_counters = defaultdict(int)
        self.user_counters = defaultdict(int)
        
        # Initialize blacklists
        self.ip_blacklist = set(self.config.get("ip_blacklist", []))
        self.user_blacklist = set(self.config.get("user_blacklist", []))
        
        # Initialize rate limiters
        self.rate_limiters = {}
        for resource, limit in self.config.get("rate_limits", {}).items():
            self.rate_limiters[resource] = {
                "limit": limit,
                "counters": defaultdict(lambda: {"count": 0, "last_reset": time.time()})
            }
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_thread)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load the IDS configuration."""
        default_config = {
            "enabled": True,
            "max_events": 10000,
            "max_alerts": 1000,
            "rules": [
                {
                    "name": "Failed login attempts",
                    "type": "threshold",
                    "conditions": {
                        "event_type": "authentication",
                        "status": "failure"
                    },
                    "threshold": 5,
                    "window": 300,  # 5 minutes
                    "groupby": ["ip_address", "user_id"],
                    "severity": "medium",
                    "actions": ["alert", "log"]
                },
                {
                    "name": "Suspicious API access",
                    "type": "pattern",
                    "conditions": {
                        "event_type": "api",
                        "path": {"regex": "/(admin|config|system)/.*"}
                    },
                    "severity": "high",
                    "actions": ["alert", "log"]
                }
            ],
            "ip_blacklist": [],
            "user_blacklist": [],
            "rate_limits": {
                "login": 10,  # 10 attempts per minute
                "api": 100    # 100 requests per minute
            },
            "alert_channels": ["log", "email"],
            "email_recipients": ["security@example.com"]
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                
                # Merge with default config
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                
                return config
            except Exception as e:
                logger.error(f"Error loading IDS config: {str(e)}")
                return default_config
        else:
            # Create default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    
    def process_event(self, event: Dict[str, Any]) -> bool:
        """Process a security event and check for intrusions.
        
        Args:
            event: Security event data
            
        Returns:
            True if the event is allowed, False if it should be blocked
        """
        if not self.config.get("enabled", True):
            return True
        
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.datetime.now().isoformat()
        
        # Check IP blacklist
        ip_address = event.get("ip_address")
        if ip_address and ip_address in self.ip_blacklist:
            self._create