import os
import sys
import json
import time
import uuid
import logging
import hashlib
import datetime
import threading
import traceback
import socket
import ipaddress
import re
import ssl
import hmac
import base64
import secrets
import tempfile
import subprocess
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
import urllib.parse
import urllib.request
import http.client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate
import jwt

# Try to import optional dependencies
try:
    import pyotp
    PYOTP_AVAILABLE = True
except ImportError:
    PYOTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import sqlalchemy
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Import internal modules
try:
    from database_manager import DatabaseManager
    DATABASE_MANAGER_AVAILABLE = True
except ImportError:
    DATABASE_MANAGER_AVAILABLE = False
    print("Warning: DatabaseManager module not available. Some features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/security_compliance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("enhanced_security_compliance")

# Constants
CONFIG_DIR = Path("config")
SECURITY_DIR = Path("security")
KEYS_DIR = SECURITY_DIR / "keys"
CERTS_DIR = SECURITY_DIR / "certificates"
LOGS_DIR = Path("logs")
AUDIT_DIR = LOGS_DIR / "audit"
REPORTS_DIR = Path("reports") / "security"
TEMP_DIR = Path("temp") / "security"

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SECURITY_DIR.mkdir(parents=True, exist_ok=True)
KEYS_DIR.mkdir(parents=True, exist_ok=True)
CERTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Australian compliance regulations
class AustralianRegulation(Enum):
    """Australian regulatory frameworks."""
    PRIVACY_ACT = "privacy_act"  # Privacy Act 1988
    NOTIFIABLE_DATA_BREACHES = "notifiable_data_breaches"  # Notifiable Data Breaches scheme
    SECURITY_LEGISLATION = "security_legislation"  # Security Legislation Amendment Act
    CLOUD_SECURITY = "cloud_security"  # Australian Government Cloud Security Guidelines
    APRA_CPS_234 = "apra_cps_234"  # APRA CPS 234 Information Security
    ASD_ESSENTIAL_EIGHT = "asd_essential_eight"  # ASD Essential Eight
    ISM = "ism"  # Information Security Manual
    PSPF = "pspf"  # Protective Security Policy Framework
    GDPR = "gdpr"  # General Data Protection Regulation (for Australian businesses with EU customers)
    CONSUMER_DATA_RIGHT = "consumer_data_right"  # Consumer Data Right (CDR)
    ANTI_MONEY_LAUNDERING = "anti_money_laundering"  # Anti-Money Laundering and Counter-Terrorism Financing Act

# International compliance frameworks
class ComplianceFramework(Enum):
    """International compliance frameworks."""
    ISO_27001 = "iso_27001"  # ISO/IEC 27001 - Information Security Management
    SOC2 = "soc2"  # SOC 2 - Service Organization Control 2
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    GDPR = "gdpr"  # General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    NIST_CSF = "nist_csf"  # NIST Cybersecurity Framework
    NIST_800_53 = "nist_800_53"  # NIST Special Publication 800-53
    SOX = "sox"  # Sarbanes-Oxley Act
    FISMA = "fisma"  # Federal Information Security Management Act
    FEDRAMP = "fedramp"  # Federal Risk and Authorization Management Program
    CUSTOM = "custom"  # Custom compliance framework

class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"
    TOP_SECRET = "top_secret"

class AuthMethod(Enum):
    """Authentication methods."""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    TOTP = "totp"  # Time-based One-Time Password
    SMS = "sms"
    EMAIL = "email"
    SSO = "sso"  # Single Sign-On
    CUSTOM = "custom"

class MFAType(Enum):
    """Multi-factor authentication types."""
    NONE = "none"
    APP_TOTP = "app_totp"  # Time-based One-Time Password via app
    SMS = "sms"
    EMAIL = "email"
    HARDWARE_TOKEN = "hardware_token"
    BIOMETRIC = "biometric"
    PUSH_NOTIFICATION = "push_notification"

class EncryptionType(Enum):
    """Encryption types."""
    NONE = "none"
    AES_256 = "aes_256"
    RSA = "rsa"
    FERNET = "fernet"
    CUSTOM = "custom"

class ThreatLevel(Enum):
    """Threat levels."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VulnerabilityStatus(Enum):
    """Vulnerability status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"
    FALSE_POSITIVE = "false_positive"

class AuditLogLevel(Enum):
    """Audit log levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"

@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_type: EncryptionType = EncryptionType.AES_256
    auth_methods: List[AuthMethod] = field(default_factory=lambda: [AuthMethod.PASSWORD])
    mfa_required: bool = True
    mfa_type: MFAType = MFAType.APP_TOTP
    password_policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_length": 12,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_numbers": True,
        "require_special_chars": True,
        "max_age_days": 90,
        "history_count": 5
    })
    session_timeout_minutes: int = 30
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    api_rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "default": 100,  # requests per minute
        "auth": 10,
        "sensitive": 20
    })
    ip_whitelist: List[str] = field(default_factory=list)
    ip_blacklist: List[str] = field(default_factory=list)
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block"
    })
    audit_log_retention_days: int = 365
    enable_vulnerability_scanning: bool = True
    vulnerability_scan_schedule: str = "weekly"
    enable_threat_detection: bool = True
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=lambda: [
        ComplianceFramework.ISO_27001
    ])
    australian_regulations: List[AustralianRegulation] = field(default_factory=lambda: [
        AustralianRegulation.PRIVACY_ACT,
        AustralianRegulation.NOTIFIABLE_DATA_BREACHES
    ])
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "encryption_type": self.encryption_type.value,
            "auth_methods": [method.value for method in self.auth_methods],
            "mfa_required": self.mfa_required,
            "mfa_type": self.mfa_type.value,
            "password_policy": self.password_policy,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_login_attempts": self.max_login_attempts,
            "lockout_duration_minutes": self.lockout_duration_minutes,
            "api_rate_limits": self.api_rate_limits,
            "ip_whitelist": self.ip_whitelist,
            "ip_blacklist": self.ip_blacklist,
            "security_headers": self.security_headers,
            "audit_log_retention_days": self.audit_log_retention_days,
            "enable_vulnerability_scanning": self.enable_vulnerability_scanning,
            "vulnerability_scan_schedule": self.vulnerability_scan_schedule,
            "enable_threat_detection": self.enable_threat_detection,
            "compliance_frameworks": [framework.value for framework in self.compliance_frameworks],
            "australian_regulations": [regulation.value for regulation in self.australian_regulations],
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        """Create from dictionary."""
        return cls(
            encryption_type=EncryptionType(data.get("encryption_type", EncryptionType.AES_256.value)),
            auth_methods=[AuthMethod(method) for method in data.get("auth_methods", [AuthMethod.PASSWORD.value])],
            mfa_required=data.get("mfa_required", True),
            mfa_type=MFAType(data.get("mfa_type", MFAType.APP_TOTP.value)),
            password_policy=data.get("password_policy", {
                "min_length": 12,
                "require_uppercase": True,
                "require_lowercase": True,
                "require_numbers": True,
                "require_special_chars": True,
                "max_age_days": 90,
                "history_count": 5
            }),
            session_timeout_minutes=data.get("session_timeout_minutes", 30),
            max_login_attempts=data.get("max_login_attempts", 5),
            lockout_duration_minutes=data.get("lockout_duration_minutes", 30),
            api_rate_limits=data.get("api_rate_limits", {
                "default": 100,
                "auth": 10,
                "sensitive": 20
            }),
            ip_whitelist=data.get("ip_whitelist", []),
            ip_blacklist=data.get("ip_blacklist", []),
            security_headers=data.get("security_headers", {
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block"
            }),
            audit_log_retention_days=data.get("audit_log_retention_days", 365),
            enable_vulnerability_scanning=data.get("enable_vulnerability_scanning", True),
            vulnerability_scan_schedule=data.get("vulnerability_scan_schedule", "weekly"),
            enable_threat_detection=data.get("enable_threat_detection", True),
            compliance_frameworks=[ComplianceFramework(framework) for framework in data.get("compliance_frameworks", [ComplianceFramework.ISO_27001.value])],
            australian_regulations=[AustralianRegulation(regulation) for regulation in data.get("australian_regulations", [AustralianRegulation.PRIVACY_ACT.value, AustralianRegulation.NOTIFIABLE_DATA_BREACHES.value])],
            custom_settings=data.get("custom_settings", {})
        )
    
    def save(self, filepath: Path = CONFIG_DIR / "security_config.json") -> None:
        """Save configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Security configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving security configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = CONFIG_DIR / "security_config.json") -> 'SecurityConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Security configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading security configuration: {e}")
            return cls()

@dataclass
class DataProtectionPolicy:
    """Data protection policy."""
    data_classification: DataClassification
    encryption_required: bool = True
    encryption_type: EncryptionType = EncryptionType.AES_256
    access_control_level: SecurityLevel = SecurityLevel.HIGH
    data_retention_days: Optional[int] = None
    data_masking_required: bool = False
    data_masking_fields: List[str] = field(default_factory=list)
    data_backup_required: bool = True
    backup_frequency: str = "daily"
    backup_retention_days: int = 90
    data_location_restrictions: List[str] = field(default_factory=list)  # e.g., ["Australia", "New Zealand"]
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data_classification": self.data_classification.value,
            "encryption_required": self.encryption_required,
            "encryption_type": self.encryption_type.value,
            "access_control_level": self.access_control_level.value,
            "data_retention_days": self.data_retention_days,
            "data_masking_required": self.data_masking_required,
            "data_masking_fields": self.data_masking_fields,
            "data_backup_required": self.data_backup_required,
            "backup_frequency": self.backup_frequency,
            "backup_retention_days": self.backup_retention_days,
            "data_location_restrictions": self.data_location_restrictions,
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataProtectionPolicy':
        """Create from dictionary."""
        return cls(
            data_classification=DataClassification(data["data_classification"]),
            encryption_required=data.get("encryption_required", True),
            encryption_type=EncryptionType(data.get("encryption_type", EncryptionType.AES_256.value)),
            access_control_level=SecurityLevel(data.get("access_control_level", SecurityLevel.HIGH.value)),
            data_retention_days=data.get("data_retention_days"),
            data_masking_required=data.get("data_masking_required", False),
            data_masking_fields=data.get("data_masking_fields", []),
            data_backup_required=data.get("data_backup_required", True),
            backup_frequency=data.get("backup_frequency", "daily"),
            backup_retention_days=data.get("backup_retention_days", 90),
            data_location_restrictions=data.get("data_location_restrictions", []),
            custom_settings=data.get("custom_settings", {})
        )

@dataclass
class ComplianceRequirement:
    """Compliance requirement."""
    id: str
    framework: Union[ComplianceFramework, AustralianRegulation]
    name: str
    description: str
    controls: List[str]
    status: str = "not_implemented"  # not_implemented, in_progress, implemented, verified, exempt
    evidence: List[str] = field(default_factory=list)
    responsible_party: Optional[str] = None
    due_date: Optional[datetime.date] = None
    last_audit_date: Optional[datetime.date] = None
    next_audit_date: Optional[datetime.date] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "framework": self.framework.value,
            "name": self.name,
            "description": self.description,
            "controls": self.controls,
            "status": self.status,
            "evidence": self.evidence,
            "responsible_party": self.responsible_party,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "last_audit_date": self.last_audit_date.isoformat() if self.last_audit_date else None,
            "next_audit_date": self.next_audit_date.isoformat() if self.next_audit_date else None,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComplianceRequirement':
        """Create from dictionary."""
        # Determine if this is an Australian regulation or international framework
        framework_value = data["framework"]
        try:
            framework = AustralianRegulation(framework_value)
        except ValueError:
            framework = ComplianceFramework(framework_value)
        
        due_date = datetime.date.fromisoformat(data["due_date"]) if data.get("due_date") else None
        last_audit_date = datetime.date.fromisoformat(data["last_audit_date"]) if data.get("last_audit_date") else None
        next_audit_date = datetime.date.fromisoformat(data["next_audit_date"]) if data.get("next_audit_date") else None
        
        return cls(
            id=data["id"],
            framework=framework,
            name=data["name"],
            description=data["description"],
            controls=data["controls"],
            status=data.get("status", "not_implemented"),
            evidence=data.get("evidence", []),
            responsible_party=data.get("responsible_party"),
            due_date=due_date,
            last_audit_date=last_audit_date,
            next_audit_date=next_audit_date,
            notes=data.get("notes", "")
        )

@dataclass
class Vulnerability:
    """Security vulnerability."""
    id: str
    name: str
    description: str
    severity: ThreatLevel
    affected_components: List[str]
    cve_id: Optional[str] = None  # Common Vulnerabilities and Exposures ID
    cvss_score: Optional[float] = None  # Common Vulnerability Scoring System
    status: VulnerabilityStatus = VulnerabilityStatus.OPEN
    discovered_date: datetime.date = field(default_factory=datetime.date.today)
    remediation_plan: Optional[str] = None
    remediation_date: Optional[datetime.date] = None
    assigned_to: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "affected_components": self.affected_components,
            "cve_id": self.cve_id,
            "cvss_score": self.cvss_score,
            "status": self.status.value,
            "discovered_date": self.discovered_date.isoformat(),
            "remediation_plan": self.remediation_plan,
            "remediation_date": self.remediation_date.isoformat() if self.remediation_date else None,
            "assigned_to": self.assigned_to,
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vulnerability':
        """Create from dictionary."""
        discovered_date = datetime.date.fromisoformat(data["discovered_date"]) if isinstance(data["discovered_date"], str) else data["discovered_date"]
        remediation_date = datetime.date.fromisoformat(data["remediation_date"]) if data.get("remediation_date") and isinstance(data["remediation_date"], str) else data.get("remediation_date")
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            severity=ThreatLevel(data["severity"]),
            affected_components=data["affected_components"],
            cve_id=data.get("cve_id"),
            cvss_score=data.get("cvss_score"),
            status=VulnerabilityStatus(data.get("status", VulnerabilityStatus.OPEN.value)),
            discovered_date=discovered_date,
            remediation_plan=data.get("remediation_plan"),
            remediation_date=remediation_date,
            assigned_to=data.get("assigned_to"),
            notes=data.get("notes", "")
        )

@dataclass
class SecurityIncident:
    """Security incident."""
    id: str
    name: str
    description: str
    severity: ThreatLevel
    status: str  # detected, investigating, contained, resolved, closed
    detection_date: datetime.datetime
    affected_systems: List[str]
    affected_data: List[str]
    incident_type: str  # e.g., data_breach, malware, unauthorized_access
    source: Optional[str] = None
    impact_description: str = ""
    resolution_steps: List[str] = field(default_factory=list)
    resolution_date: Optional[datetime.datetime] = None
    reported_to_authorities: bool = False
    authority_report_date: Optional[datetime.date] = None
    lessons_learned: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status,
            "detection_date": self.detection_date.isoformat(),
            "affected_systems": self.affected_systems,
            "affected_data": self.affected_data,
            "incident_type": self.incident_type,
            "source": self.source,
            "impact_description": self.impact_description,
            "resolution_steps": self.resolution_steps,
            "resolution_date": self.resolution_date.isoformat() if self.resolution_date else None,
            "reported_to_authorities": self.reported_to_authorities,
            "authority_report_date": self.authority_report_date.isoformat() if self.authority_report_date else None,
            "lessons_learned": self.lessons_learned
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityIncident':
        """Create from dictionary."""
        detection_date = datetime.datetime.fromisoformat(data["detection_date"]) if isinstance(data["detection_date"], str) else data["detection_date"]
        resolution_date = datetime.datetime.fromisoformat(data["resolution_date"]) if data.get("resolution_date") and isinstance(data["resolution_date"], str) else data.get("resolution_date")
        authority_report_date = datetime.date.fromisoformat(data["authority_report_date"]) if data.get("authority_report_date") and isinstance(data["authority_report_date"], str) else data.get("authority_report_date")
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            severity=ThreatLevel(data["severity"]),
            status=data["status"],
            detection_date=detection_date,
            affected_systems=data["affected_systems"],
            affected_data=data["affected_data"],
            incident_type=data["incident_type"],
            source=data.get("source"),
            impact_description=data.get("impact_description", ""),
            resolution_steps=data.get("resolution_steps", []),
            resolution_date=resolution_date,
            reported_to_authorities=data.get("reported_to_authorities", False),
            authority_report_date=authority_report_date,
            lessons_learned=data.get("lessons_learned", "")
        )

@dataclass
class AuditLogEntry:
    """Audit log entry."""
    id: str
    timestamp: datetime.datetime
    level: AuditLogLevel
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    resource: str
    action: str
    status: str  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "status": self.status,
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditLogEntry':
        """Create from dictionary."""
        timestamp = datetime.datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        
        return cls(
            id=data["id"],
            timestamp=timestamp,
            level=AuditLogLevel(data["level"]),
            event_type=data["event_type"],
            user_id=data.get("user_id"),
            ip_address=data.get("ip_address"),
            resource=data["resource"],
            action=data["action"],
            status=data["status"],
            details=data.get("details", {})
        )

class EncryptionManager:
    """Manager for encryption and decryption operations."""
    
    def __init__(self, config: SecurityConfig = None):
        """Initialize the encryption manager."""
        self.config = config or SecurityConfig()
        self._encryption_keys = {}
        self._load_or_generate_keys()
    
    def _load_or_generate_keys(self) -> None:
        """Load existing keys or generate new ones."""
        try:
            # Fernet key
            fernet_key_path = KEYS_DIR / "fernet.key"
            if fernet_key_path.exists():
                with open(fernet_key_path, 'rb') as f:
                    self._encryption_keys['fernet'] = f.read()
            else:
                self._encryption_keys['fernet'] = Fernet.generate_key()
                with open(fernet_key_path, 'wb') as f:
                    f.write(self._encryption_keys['fernet'])
            
            # RSA keys
            rsa_private_key_path = KEYS_DIR / "rsa_private.pem"
            rsa_public_key_path = KEYS_DIR / "rsa_public.pem"
            
            if rsa_private_key_path.exists() and rsa_public_key_path.exists():
                with open(rsa_private_key_path, 'rb') as f:
                    self._encryption_keys['rsa_private'] = f.read()
                with open(rsa_public_key_path, 'rb') as f:
                    self._encryption_keys['rsa_public'] = f.read()
            else:
                # Generate new RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
                public_key = private_key.public_key()
                
                # Serialize private key
                from cryptography.hazmat.primitives import serialization
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                # Serialize public key
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                # Store keys
                self._encryption_keys['rsa_private'] = private_pem
                self._encryption_keys['rsa_public'] = public_pem
                
                # Save keys to files
                with open(rsa_private_key_path, 'wb') as f:
                    f.write(private_pem)
                with open(rsa_public_key_path, 'wb') as f:
                    f.write(public_pem)
            
            # AES key
            aes_key_path = KEYS_DIR / "aes.key"
            if aes_key_path.exists():
                with open(aes_key_path, 'rb') as f:
                    self._encryption_keys['aes'] = f.read()
            else:
                # Generate a random 32-byte key for AES-256
                self._encryption_keys['aes'] = os.urandom(32)
                with open(aes_key_path, 'wb') as f:
                    f.write(self._encryption_keys['aes'])
            
            logger.info("Encryption keys loaded successfully")
        except Exception as e:
            logger.error(f"Error loading or generating encryption keys: {e}")
            raise
    
    def encrypt_data(self, data: Union[str, bytes], encryption_type: EncryptionType = None) -> bytes:
        """Encrypt data using the specified encryption type."""
        if encryption_type is None:
            encryption_type = self.config.encryption_type
        
        try:
            # Convert string to bytes if needed
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            if encryption_type == EncryptionType.AES_256:
                return self._encrypt_aes(data_bytes)
            elif encryption_type == EncryptionType.RSA:
                return self._encrypt_rsa(data_bytes)
            elif encryption_type == EncryptionType.FERNET:
                return self._encrypt_fernet(data_bytes)
            elif encryption_type == EncryptionType.NONE:
                return data_bytes
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, encryption_type: EncryptionType = None) -> bytes:
        """Decrypt data using the specified encryption type."""
        if encryption_type is None:
            encryption_type = self.config.encryption_type
        
        try:
            if encryption_type == EncryptionType.AES_256:
                return self._decrypt_aes(encrypted_data)
            elif encryption_type == EncryptionType.RSA:
                return self._decrypt_rsa(encrypted_data)
            elif encryption_type == EncryptionType.FERNET:
                return self._decrypt_fernet(encrypted_data)
            elif encryption_type == EncryptionType.NONE:
                return encrypted_data
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_type}")
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def _encrypt_fernet(self, data: bytes) -> bytes:
        """Encrypt data using Fernet (symmetric encryption)."""
        key = self._encryption_keys.get('fernet')
        if not key:
            raise ValueError("Fernet key not available")
        
        f = Fernet(key)
        return f.encrypt(data)
    
    def _decrypt_fernet(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using Fernet (symmetric encryption)."""
        key = self._encryption_keys.get('fernet')
        if not key:
            raise ValueError("Fernet key not available")
        
        f = Fernet(key)
        return f.decrypt(encrypted_data)
    
    def _encrypt_rsa(self, data: bytes) -> bytes:
        """Encrypt data using RSA (asymmetric encryption)."""
        from cryptography.hazmat.primitives import serialization
        
        public_key_pem = self._encryption_keys.get('rsa_public')
        if not public_key_pem:
            raise ValueError("RSA public key not available")
        
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        
        # RSA can only encrypt limited amount of data, so we'll use a hybrid approach
        # Generate a random AES key and encrypt it with RSA
        aes_key = os.urandom(32)
        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Use the AES key to encrypt the actual data
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad the data to be a multiple of AES block size
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt the data
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine everything: [encrypted_aes_key_length (4 bytes)][encrypted_aes_key][iv][encrypted_data]
        key_length = len(encrypted_aes_key).to_bytes(4, byteorder='big')
        return key_length + encrypted_aes_key + iv + encrypted_data
    
    def _decrypt_rsa(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using RSA (asymmetric encryption)."""
        from cryptography.hazmat.primitives import serialization
        
        private_key_pem = self._encryption_keys.get('rsa_private')
        if not private_key_pem:
            raise ValueError("RSA private key not available")
        
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        # Extract the encrypted AES key length
        key_length = int.from_bytes(encrypted_data[:4], byteorder='big')
        
        # Extract the encrypted AES key
        encrypted_aes_key = encrypted_data[4:4+key_length]
        
        # Decrypt the AES key
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Extract the IV and encrypted data
        iv = encrypted_data[4+key_length:4+key_length+16]
        ciphertext = encrypted_data[4+key_length+16:]
        
        # Decrypt the data
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Unpad the data
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    def _encrypt_aes(self, data: bytes) -> bytes:
        """Encrypt data using AES-256 (symmetric encryption)."""
        key = self._encryption_keys.get('aes')
        if not key:
            raise ValueError("AES key not available")
        
        # Generate a random IV
        iv = os.urandom(16)
        
        # Create AES cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Pad the data
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt the data
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Return IV + encrypted data
        return iv + encrypted_data
    
    def _decrypt_aes(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256 (symmetric encryption)."""
        key = self._encryption_keys.get('aes')
        if not key:
            raise ValueError("AES key not available")
        
        # Extract IV and ciphertext
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Create AES cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Decrypt the data
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Unpad the data
        unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    def encrypt_file(self, input_path: Path, output_path: Path = None, encryption_type: EncryptionType = None) -> Path:
        """Encrypt a file."""
        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + '.enc')
        
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            encrypted_data = self.encrypt_data(data, encryption_type)
            
            with open(output_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"File encrypted: {input_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error encrypting file {input_path}: {e}")
            raise
    
    def decrypt_file(self, input_path: Path, output_path: Path = None, encryption_type: EncryptionType = None) -> Path:
        """Decrypt a file."""
        if output_path is None:
            # Remove .enc suffix if present
            if input_path.suffix == '.enc':
                output_path = input_path.with_suffix('')
            else:
                output_path = input_path.with_suffix(input_path.suffix + '.dec')
        
        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.decrypt_data(encrypted_data, encryption_type)
            
            with open(output_path, 'wb') as f:
                f.write(decrypted_data)
            
            logger.info(f"File decrypted: {input_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error decrypting file {input_path}: {e}")
            raise
    
    def generate_hash(self, data: Union[str, bytes], algorithm: str = 'sha256') -> str:
        """Generate a hash of the data."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if algorithm == 'sha256':
                hash_obj = hashlib.sha256(data)
            elif algorithm == 'sha512':
                hash_obj = hashlib.sha512(data)
            elif algorithm == 'md5':
                hash_obj = hashlib.md5(data)
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error generating hash: {e}")
            raise
    
    def generate_hmac(self, data: Union[str, bytes], key: Union[str, bytes] = None, algorithm: str = 'sha256') -> str:
        """Generate an HMAC for the data."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if key is None:
                # Use a portion of the fernet key as the HMAC key
                key = self._encryption_keys.get('fernet')[:16]
            elif isinstance(key, str):
                key = key.encode('utf-8')
            
            if algorithm == 'sha256':
                h = hmac.new(key, data, hashlib.sha256)
            elif algorithm == 'sha512':
                h = hmac.new(key, data, hashlib.sha512)
            elif algorithm == 'md5':
                h = hmac.new(key, data, hashlib.md5)
            else:
                raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
            
            return h.hexdigest()
        except Exception as e:
            logger.error(f"Error generating HMAC: {e}")
            raise

class AuthenticationManager:
    """Manager for authentication and authorization."""
    
    def __init__(self, config: SecurityConfig = None, encryption_manager: EncryptionManager = None):
        """Initialize the authentication manager."""
        self.config = config or SecurityConfig()
        self.encryption_manager = encryption_manager or EncryptionManager(self.config)
        self.db_path = SECURITY_DIR / "auth.db"
        self._init_database()
        self._failed_attempts = {}
        self._sessions = {}
        self._api_keys = {}
        self._load_api_keys()
    
    def _init_database(self) -> None:
        """Initialize the authentication database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                email TEXT,
                role TEXT,
                mfa_secret TEXT,
                mfa_enabled INTEGER DEFAULT 0,
                last_login TIMESTAMP,
                password_changed TIMESTAMP,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create roles table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS roles (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                permissions TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create password history table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS password_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create sessions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                token TEXT NOT NULL,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            # Create API keys table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                key_name TEXT NOT NULL,
                key_prefix TEXT NOT NULL,
                key_hash TEXT NOT NULL,
                permissions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used TIMESTAMP,
                status TEXT DEFAULT 'active',
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Authentication database initialized")
        except Exception as e:
            logger.error(f"Error initializing authentication database: {e}")
            raise
    
    def _load_api_keys(self) -> None:
        """Load API keys from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM api_keys WHERE status = 'active'")
            rows = cursor.fetchall()
            
            for row in rows:
                self._api_keys[row['key_prefix']] = {
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'key_hash': row['key_hash'],
                    'permissions': json.loads(row['permissions']) if row['permissions'] else [],
                    'expires_at': row['expires_at']
                }
            
            conn.close()
            logger.info(f"Loaded {len(self._api_keys)} API keys")
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
    
    def create_user(self, username: str, password: str, email: str = None, role: str = "user") -> str:
        """Create a new user."""
        try:
            # Validate password against policy
            self._validate_password(password)
            
            # Generate a random salt
            salt = secrets.token_hex(16)
            
            # Hash the password with the salt
            password_hash = self._hash_password(password, salt)
            
            # Generate a unique ID
            user_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                conn.close()
                raise ValueError(f"Username '{username}' already exists")
            
            # Insert the new user
            cursor.execute(
                "INSERT INTO users (id, username, password_hash, salt, email, role, password_changed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (user_id, username, password_hash, salt, email, role, datetime.datetime.now())
            )
            
            # Add the password to history
            history_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO password_history (id, user_id, password_hash) VALUES (?, ?, ?)",
                (history_id, user_id, password_hash)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Created user: {username}")
            return user_id
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against policy."""
        policy = self.config.password_policy
        
        if len(password) < policy.get("min_length", 12):
            raise ValueError(f"Password must be at least {policy.get('min_length', 12)} characters long")
        
        if policy.get("require_uppercase", True) and not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        
        if policy.get("require_lowercase", True) and not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        
        if policy.get("require_numbers", True) and not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one number")
        
        if policy.get("require_special_chars", True) and not any(not c.isalnum() for c in password):
            raise ValueError("Password must contain at least one special character")
        
        return True
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash a password with the given salt."""
        # Combine password and salt
        salted_password = (password + salt).encode('utf-8')
        
        # Use SHA-512 for password hashing
        hash_obj = hashlib.sha512(salted_password)
        return hash_obj.hexdigest()
    
    def authenticate(self, username: str, password: str, ip_address: str = None) -> Optional[Dict[str, Any]]:
        """Authenticate a user."""
        try:
            # Check if the IP is blacklisted
            if ip_address and self._is_ip_blacklisted(ip_address):
                logger.warning(f"Authentication attempt from blacklisted IP: {ip_address}")
                return None
            
            # Check for too many failed attempts
            if ip_address and self._check_rate_limit(ip_address, "auth"):
                logger.warning(f"Too many authentication attempts from IP: {ip_address}")
                return None
            
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user by username
            cursor.execute("SELECT * FROM users WHERE username = ? AND status = 'active'", (username,))
            user = cursor.fetchone()
            
            if not user:
                # Record failed attempt
                if ip_address:
                    self._record_failed_attempt(ip_address)
                conn.close()
                return None
            
            # Hash the provided password with the stored salt
            password_hash = self._hash_password(password, user['salt'])
            
            # Check if the password matches
            if password_hash != user['password_hash']:
                # Record failed attempt
                if ip_address:
                    self._record_failed_attempt(ip_address)
                conn.close()
                return None
            
            # Check if password needs to be changed
            password_age = None
            if user['password_changed']:
                password_changed = datetime.datetime.fromisoformat(user['password_changed'])
                password_age = (datetime.datetime.now() - password_changed).days
            
            # Update last login time
            cursor.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.datetime.now(), user['id'])
            )
            
            conn.commit()
            conn.close()
            
            # Clear failed attempts for this IP
            if ip_address and ip_address in self._failed_attempts:
                del self._failed_attempts[ip_address]
            
            # Return user info
            return {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role'],
                'mfa_enabled': bool(user['mfa_enabled']),
                'password_expired': password_age and password_age > self.config.password_policy.get("max_age_days", 90)
            }
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    def verify_mfa(self, user_id: str, code: str) -> bool:
        """Verify a multi-factor authentication code."""
        if not PYOTP_AVAILABLE:
            logger.error("PyOTP library not available for MFA verification")
            return False
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get user's MFA secret
            cursor.execute("SELECT mfa_secret, mfa_enabled FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            conn.close()
            
            if not user or not user['mfa_enabled'] or not user['mfa_secret']:
                return False
            
            # Decrypt the MFA secret
            secret_bytes = self.encryption_manager.decrypt_data(
                base64.b64decode(user['mfa_secret']),
                EncryptionType.FERNET
            )
            secret = secret_bytes.decode('utf-8')
            
            # Verify the code
            import pyotp
            totp = pyotp.TOTP(secret)
            return totp.verify(code)
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
    
    def setup_mfa(self, user_id: str) -> Dict[str, str]:
        """Set up multi-factor authentication for a user."""
        if not PYOTP_AVAILABLE:
            raise ImportError("PyOTP library not available for MFA setup")
        
        try:
            import pyotp
            
            # Generate a new secret
            secret = pyotp.random_base32()
            
            # Encrypt the secret
            encrypted_secret = self.encryption_manager.encrypt_data(
                secret,
                EncryptionType.FERNET
            )
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get user info for the QR code
            cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                raise ValueError(f"User ID {user_id} not found")
            
            # Update the user with the new secret
            cursor.execute(
                "UPDATE users SET mfa_secret = ?, mfa_enabled = 0 WHERE id = ?",
                (base64.b64encode(encrypted_secret).decode('utf-8'), user_id)
            )
            
            conn.commit()
            conn.close()
            
            # Generate provisioning URI for QR code
            totp = pyotp.TOTP(secret)
            uri = totp.provisioning_uri(user[0], issuer_name="Skyscope Sentinel")
            
            return {
                'secret': secret,
                'uri': uri
            }
        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            raise
    
    def enable_mfa(self, user_id: str, code: str) -> bool:
        """Enable MFA after verifying the setup code."""
        if not self.verify_mfa(user_id, code):
            return False
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Enable MFA for the user
            cursor.execute(
                "UPDATE users SET mfa_enabled = 1 WHERE id = ?",
                (user_id,)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"MFA enabled for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error enabling MFA: {e}")
            return False
    
    def create_session(self, user_id: str, ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """Create a new session for a user."""
        try:
            # Generate a unique session ID
            session_id = str(uuid.uuid4())
            
            # Generate a secure token
            token = secrets.token_urlsafe(64)
            
            # Calculate expiration time
            expires_at = datetime.datetime.now() + datetime.timedelta(minutes=self.config.session_timeout_minutes)
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert the new session
            cursor.execute(
                "INSERT INTO sessions (id, user_id, token, ip_address, user_agent, expires_at, last_activity) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, user_id, token, ip_address, user_agent, expires_at, datetime.datetime.now())
            )
            
            conn.commit()
            conn.close()
            
            # Store session in memory
            self._sessions[token] = {
                'id': session_id,
                'user_id': user_id,
                'expires_at': expires_at,
                'ip_address': ip_address
            }
            
            logger.info(f"Created session for user {user_id}")
            return {
                'session_id': session_id,
                'token': token,
                'expires_at': expires_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise
    
    def validate_session(self, token: str, ip_address: str = None) -> Optional[Dict[str, Any]]:
        """Validate a session token."""
        try:
            # Check memory cache first
            session = self._sessions.get(token)
            
            if session:
                # Check if session has expired
                if session['expires_at'] < datetime.datetime.now():
                    # Remove expired session
                    del self._sessions[token]
                    self._remove_session_from_db(session['id'])
                    return None
                
                # Check IP address if provided
                if ip_address and session['ip_address'] and ip_address != session['ip_address']:
                    logger.warning(f"Session IP mismatch: {ip_address} != {session['ip_address']}")
                    return None
                
                # Update last activity
                self._update_session_activity(session['id'])
                
                return {
                    'user_id': session['user_id'],
                    'session_id': session['id']
                }
            
            # Not found in memory, check database
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM sessions WHERE token = ? AND expires_at > ?",
                (token, datetime.datetime.now())
            )
            
            db_session = cursor.fetchone()
            
            if not db_session:
                conn.close()
                return None
            
            # Check IP address if provided
            if ip_address and db_session['ip_address'] and ip_address != db_session['ip_address']:
                logger.warning(f"Session IP mismatch: {ip_address} != {db_session['ip_address']}")
                conn.close()
                return None
            
            # Update last activity
            cursor.execute(
                "UPDATE sessions SET last_activity = ? WHERE id = ?",
                (datetime.datetime.now(), db_session['id'])
            )
            
            conn.commit()
            conn.close()
            
            # Add to memory cache
            self._sessions[token] = {
                'id': db_session['id'],
                'user_id': db_session['user_id'],
                'expires_at': datetime.datetime.fromisoformat(db_session['expires_at']),
                'ip_address': db_session['ip_address']
            }
            
            return {
                'user_id': db_session['user_id'],
                'session_id': db_session['id']
            }
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return None
    
    def _update_session_activity(self, session_id: str) -> None:
        """Update the last activity timestamp for a session."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE sessions SET last_activity = ? WHERE id = ?",
                (datetime.datetime.now(), session_id)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error updating session activity: {e}")
    
    def _remove_session_from_db(self, session_id: str) -> None:
        """Remove a session from the database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error removing session from database: {e}")
    
    def end_session(self, token: str) -> bool:
        """End a user session."""
        try:
            # Remove from memory cache
            if token in self._sessions:
                session_id = self._sessions[token]['id']
                del self._sessions[token]
            else:
                # Look up in database
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("SELECT id FROM sessions WHERE token = ?", (token,))
                result = cursor.fetchone()
                
                if not result:
                    conn.close()
                    return False
                
                session_id = result[0]
                conn.close()
            
            # Remove from database
            self._remove_session_from_db(session_id)
            
            logger.info(f"Ended session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            return False
    
    def create_api_key(self, user_id: str, key_name: str, permissions: List[str] = None, expires_days: int = 365) -> Dict[str, Any]:
        """Create a new API key for a user."""
        try:
            # Generate a unique API key ID
            key_id = str(uuid.uuid4())
            
            # Generate a secure API key
            api_key = f"sk_{secrets.token_urlsafe(32)}"
            
            # Extract the prefix for lookup
            key_prefix = api_key[:7]
            
            # Hash the API key for storage
            key_hash = self.encryption_manager.generate_hash(api_key)
            
            # Calculate expiration date
            expires_at = datetime.datetime.now() + datetime.timedelta(days=expires_days) if expires_days else None
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Insert the new API key
            cursor.execute(
                "INSERT INTO api_keys (id, user_id, key_name, key_prefix, key_hash, permissions, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    key_id, 
                    user_id, 
                    key_name, 
                    key_prefix, 
                    key_hash, 
                    json.dumps(permissions) if permissions else None,
                    expires_at
                )
            )
            
            conn.commit()
            conn.close()
            
            # Add to memory cache
            self._api_keys[key_prefix] = {
                'id': key_id,
                'user_id': user_id,
                'key_hash': key_hash,
                'permissions': permissions or [],
                'expires_at': expires_at.isoformat() if expires_at else None
            }
            
            logger.info(f"Created API key '{key_name}' for user {user_id}")
            return {
                'id': key_id,
                'key': api_key,
                'name': key_name,
                'permissions': permissions or [],
                'expires_at': expires_at.isoformat() if expires_at else None
            }
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key."""
        try:
            # Extract the prefix
            key_prefix = api_key[:7]
            
            # Check if prefix exists
            if key_prefix not in self._api_keys:
                return None
            
            # Get key info
            key_info = self._api_keys[key_prefix]
            
            # Check if key has expired
            if key_info['expires_at']:
                expires_at = datetime.datetime.fromisoformat(key_info['expires_at'])
                if expires_at < datetime.datetime.now():
                    return None
            
            # Hash the API key and compare
            key_hash = self.encryption_manager.generate_hash(api_key)
            if key_hash != key_info['key_hash']:
                return None
            
            # Update last used timestamp
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE api_keys SET last_used = ? WHERE id = ?",
                (datetime.datetime.now(), key_info['id'])
            )
            
            conn.commit()
            conn.close()
            
            return {
                'user_id': key_info['user_id'],
                'permissions': key_info['permissions']
            }
        except Exception as e:
            logger.error(f"API key validation error: {e}")
            return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get the key prefix
            cursor.execute("SELECT key_prefix FROM api_keys WHERE id = ?", (key_id,))
            result = cursor.fetchone()
            
            if not result:
                conn.close()
                return False
            
            key_prefix = result[0]
            
            # Update the key status
            cursor.execute(
                "UPDATE api_keys SET status = 'revoked' WHERE id = ?",
                (key_id,)
            )
            
            conn.commit()
            conn.close()
            
            # Remove from memory cache
            if key_prefix in self._api_keys:
                del self._api_keys[key_prefix]
            
            logger.info(f"Revoked API key {key_id}")
            return True
        except Exception as e:
            logger.error(f"Error revoking API key: {e}")
            return False
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change a user's password."""
        try:
            # Validate the new password
            self._validate_password(new_password)
            
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get the user
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                conn.close()
                raise ValueError(f"User ID {user_id} not found")
            
            # Verify current password
            current_hash = self._hash_password(current_password, user['salt'])
            if current_hash != user['password_hash']:
                conn.close()
                return False
            
            # Check password history
            history_count = self.config.password_policy.get("history_count", 5)
            if history_count > 0:
                cursor.execute(
                    "SELECT password_hash FROM password_history WHERE user_id = ? ORDER BY changed_at DESC LIMIT ?",
                    (user_id, history_count)
                )
                password_history = cursor.fetchall()
                
                # Check if new password matches any recent passwords
                new_hash = self._hash_password(new_password, user['salt'])
                for old_password in password_history:
                    if new_hash == old_password['password_hash']:
                        conn.close()
                        raise ValueError("New password cannot match any of your recent passwords")
            
            # Generate a new salt and hash
            new_salt = secrets.token_hex(16)
            new_hash = self._hash_password(new_password, new_salt)
            
            # Update the user's password
            cursor.execute(
                "UPDATE users SET password_hash = ?, salt = ?, password_changed = ? WHERE id = ?",
                (new_hash, new_salt, datetime.datetime.now(), user_id)
            )
            
            # Add to password history
            history_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO password_history (id, user_id, password_hash) VALUES (?, ?, ?)",
                (history_id, user_id, new_hash)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Password changed for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            raise
    
    def _record_failed_attempt(self, ip_address: str) -> None:
        """Record a failed authentication attempt."""
        if ip_address not in self._failed_attempts:
            self._failed_attempts[ip_address] = {
                'count': 0,
                'first_attempt': datetime.datetime.now(),
                'last_attempt': datetime.datetime.now()
            }
        
        self._failed_attempts[ip_address]['count'] += 1
        self._failed_attempts[ip_address]['last_attempt'] = datetime.datetime.now()
        
        # Check if we should blacklist this IP
        if self._failed_attempts[ip_address]['count'] >= self.config.max_login_attempts:
            self._add_to_blacklist(ip_address, f"Too many failed login attempts ({self._failed_attempts[ip_address]['count']})")
    
    def _check_rate_limit(self, ip_address: str, endpoint_type: str = "default") -> bool:
        """Check if an IP address has exceeded the rate limit."""
        # Get the rate limit for this endpoint type
        rate_limit = self.config.api_rate_limits.get(endpoint_type, self.config.api_rate_limits.get("default", 100))
        
        # Check if this IP is in the failed attempts list
        if ip_address in self._failed_attempts:
            attempts = self._failed_attempts[ip_address]
            
            # Check if we're within the lockout period
            if attempts['count'] >= self.config.max_login_attempts:
                lockout_duration = datetime.timedelta(minutes=self.config.lockout_duration_minutes)
                if datetime.datetime.now() - attempts['last_attempt'] < lockout_duration:
                    return True  # Rate limited
            
            # Check if we've exceeded the rate limit
            time_window = datetime.timedelta(minutes=1)  # 1 minute window
            if attempts['count'] > rate_limit and datetime.datetime.now() - attempts['first_attempt'] < time_window:
                return True  # Rate limited
        
        return False  # Not rate limited
    
    def _is_ip_whitelisted(self, ip_address: str) -> bool:
        """Check if an IP address is whitelisted."""
        if not self.config.ip_whitelist:
            return False
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for whitelist_entry in self.config.ip_whitelist:
                # Check if entry is a CIDR range
                if '/' in whitelist_entry:
                    if ip in ipaddress.ip_network(whitelist_entry):
                        return True
                # Check if entry is a specific IP
                elif ip_address == whitelist_entry:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking IP whitelist: {e}")
            return False
    
    def _is_ip_blacklisted(self, ip_address: str) -> bool:
        """Check if an IP address is blacklisted."""
        if not self.config.ip_blacklist:
            return False
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            for blacklist_entry in self.config.ip_blacklist:
                # Check if entry is a CIDR range
                if '/' in blacklist_entry:
                    if ip in ipaddress.ip_network(blacklist_entry):
                        return True
                # Check if entry is a specific IP
                elif ip_address == blacklist_entry:
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking IP blacklist: {e}")
            return False
    
    def _add_to_blacklist(self, ip_address: str, reason: str = None) -> None:
        """Add an IP address to the blacklist."""
        if ip_address not in self.config.ip_blacklist:
            self.config.ip_blacklist.append(ip_address)
            self.config.save()
            logger.warning(f"Added IP {ip_address} to blacklist. Reason: {reason}")

class AuditLogger:
    """Logger for security audit events."""
    
    def __init__(self, config: SecurityConfig = None):
        """Initialize the audit logger."""
        self.config = config or SecurityConfig()
        self.db_path = AUDIT_DIR / "audit.db"
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize the audit database
