
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
import re
import hashlib
import base64
import csv
import io
import tempfile
import shutil

# PDF generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm

# Try to import web automation libraries
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logging.warning("Playwright not available. Web automation features will be limited.")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("Selenium not available. Web automation features will be limited.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/business_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("business_manager")

# Constants
COMPANY_NAME = "Skyscope Sentinel Intelligence"
COMPANY_ABN = "11287984779"
COMPANY_EMAIL = "admin@skyscope.cloud"
BUSINESS_DIR = Path("business")
REPORTS_DIR = Path("reports")
ACCOUNTS_DIR = Path("accounts")
CREDENTIALS_DIR = Path("credentials")
TEMPLATES_DIR = Path("templates")
TRANSACTIONS_DIR = Path("transactions")
CONTRACTS_DIR = Path("contracts")
INVOICES_DIR = Path("invoices")
RECEIPTS_DIR = Path("receipts")
TAX_DIR = Path("tax")

# Ensure directories exist
for directory in [
    BUSINESS_DIR, 
    REPORTS_DIR, 
    ACCOUNTS_DIR, 
    CREDENTIALS_DIR, 
    TEMPLATES_DIR, 
    TRANSACTIONS_DIR, 
    CONTRACTS_DIR, 
    INVOICES_DIR, 
    RECEIPTS_DIR, 
    TAX_DIR
]:
    directory.mkdir(exist_ok=True, parents=True)

class BusinessType(Enum):
    """Enumeration of business types."""
    FREELANCE = "freelance"
    AFFILIATE_MARKETING = "affiliate_marketing"
    CONTENT_CREATION = "content_creation"
    SOCIAL_MEDIA = "social_media"
    CRYPTO_TRADING = "crypto_trading"
    E_COMMERCE = "e_commerce"
    CONSULTING = "consulting"
    DATA_SERVICES = "data_services"
    SAAS = "saas"
    OTHER = "other"

class BusinessStatus(Enum):
    """Enumeration of business operation statuses."""
    PLANNING = "planning"
    SETTING_UP = "setting_up"
    ACTIVE = "active"
    PAUSED = "paused"
    SCALING = "scaling"
    OPTIMIZING = "optimizing"
    PIVOTING = "pivoting"
    CLOSING = "closing"
    FAILED = "failed"

class TransactionType(Enum):
    """Enumeration of transaction types."""
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"
    REFUND = "refund"
    TAX = "tax"
    FEE = "fee"
    OTHER = "other"

class PaymentMethod(Enum):
    """Enumeration of payment methods."""
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    CRYPTO = "cryptocurrency"
    CASH = "cash"
    CHECK = "check"
    STRIPE = "stripe"
    DIRECT_DEPOSIT = "direct_deposit"
    OTHER = "other"

class ReportType(Enum):
    """Enumeration of report types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    TAX = "tax"
    BAS = "bas"  # Business Activity Statement (Australian)
    FINANCIAL = "financial"
    PERFORMANCE = "performance"
    CUSTOM = "custom"

class TaxCategory(Enum):
    """Enumeration of tax categories for Australian taxation."""
    GST = "gst"  # Goods and Services Tax
    INCOME_TAX = "income_tax"
    PAYG = "payg"  # Pay As You Go
    SUPER = "superannuation"
    FBT = "fbt"  # Fringe Benefits Tax
    CAPITAL_GAINS = "capital_gains"
    DEDUCTION = "deduction"
    OFFSET = "offset"
    REBATE = "rebate"
    OTHER = "other"

@dataclass
class BusinessIdentity:
    """Represents a business identity with registration details."""
    name: str
    business_type: BusinessType
    abn: str = ""  # Australian Business Number
    acn: str = ""  # Australian Company Number
    tfn: str = ""  # Tax File Number
    registration_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    business_email: str = ""
    business_phone: str = ""
    business_address: str = ""
    website: str = ""
    social_media: Dict[str, str] = field(default_factory=dict)
    logo_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "business_type": self.business_type.value,
            "abn": self.abn,
            "acn": self.acn,
            "tfn": self.tfn,
            "registration_date": self.registration_date.isoformat() if self.registration_date else None,
            "expiry_date": self.expiry_date.isoformat() if self.expiry_date else None,
            "business_email": self.business_email,
            "business_phone": self.business_phone,
            "business_address": self.business_address,
            "website": self.website,
            "social_media": self.social_media,
            "logo_path": str(self.logo_path) if self.logo_path else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessIdentity':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            business_type=BusinessType(data["business_type"]),
            abn=data["abn"],
            acn=data.get("acn", ""),
            tfn=data.get("tfn", ""),
            registration_date=datetime.fromisoformat(data["registration_date"]) if data.get("registration_date") else None,
            expiry_date=datetime.fromisoformat(data["expiry_date"]) if data.get("expiry_date") else None,
            business_email=data.get("business_email", ""),
            business_phone=data.get("business_phone", ""),
            business_address=data.get("business_address", ""),
            website=data.get("website", ""),
            social_media=data.get("social_media", {}),
            logo_path=Path(data["logo_path"]) if data.get("logo_path") else None
        )
    
    def save(self, directory: Path = BUSINESS_DIR) -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        safe_name = self.name.lower().replace(" ", "_")
        filepath = directory / f"{safe_name}_identity.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'BusinessIdentity':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class AccountCredential:
    """Represents credentials for a business account."""
    account_id: str
    service_name: str
    username: str
    password: str = ""  # Note: In production, use secure storage
    email: str = ""
    api_key: str = ""
    secret_key: str = ""
    token: str = ""
    mfa_secret: str = ""
    url: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "account_id": self.account_id,
            "service_name": self.service_name,
            "username": self.username,
            "password": self.encrypt(self.password) if self.password else "",
            "email": self.email,
            "api_key": self.encrypt(self.api_key) if self.api_key else "",
            "secret_key": self.encrypt(self.secret_key) if self.secret_key else "",
            "token": self.encrypt(self.token) if self.token else "",
            "mfa_secret": self.encrypt(self.mfa_secret) if self.mfa_secret else "",
            "url": self.url,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccountCredential':
        """Create from dictionary."""
        credential = cls(
            account_id=data["account_id"],
            service_name=data["service_name"],
            username=data["username"],
            email=data.get("email", ""),
            url=data.get("url", ""),
            notes=data.get("notes", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
        
        # Decrypt sensitive fields if they exist
        if data.get("password"):
            credential.password = credential.decrypt(data["password"])
        if data.get("api_key"):
            credential.api_key = credential.decrypt(data["api_key"])
        if data.get("secret_key"):
            credential.secret_key = credential.decrypt(data["secret_key"])
        if data.get("token"):
            credential.token = credential.decrypt(data["token"])
        if data.get("mfa_secret"):
            credential.mfa_secret = credential.decrypt(data["mfa_secret"])
        
        return credential
    
    def encrypt(self, text: str) -> str:
        """Simple encryption for demonstration purposes.
        In production, use proper encryption libraries and secure key management."""
        if not text:
            return ""
        
        # This is a placeholder. In a real system, use proper encryption
        key = hashlib.sha256(f"SKYSCOPE_SECRET_{self.account_id}".encode()).digest()
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
    
    def save(self, directory: Path = CREDENTIALS_DIR) -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        service_dir = directory / self.service_name.lower().replace(" ", "_")
        service_dir.mkdir(exist_ok=True)
        
        filepath = service_dir / f"{self.account_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'AccountCredential':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class BusinessAccount:
    """Represents a business account for a service."""
    account_id: str
    business_id: str
    service_name: str
    account_type: str
    status: str = "active"
    balance: float = 0.0
    currency: str = "AUD"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    credentials_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "account_id": self.account_id,
            "business_id": self.business_id,
            "service_name": self.service_name,
            "account_type": self.account_type,
            "status": self.status,
            "balance": self.balance,
            "currency": self.currency,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "credentials_id": self.credentials_id,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessAccount':
        """Create from dictionary."""
        return cls(
            account_id=data["account_id"],
            business_id=data["business_id"],
            service_name=data["service_name"],
            account_type=data["account_type"],
            status=data.get("status", "active"),
            balance=data.get("balance", 0.0),
            currency=data.get("currency", "AUD"),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            credentials_id=data.get("credentials_id"),
            metadata=data.get("metadata", {})
        )
    
    def save(self, directory: Path = ACCOUNTS_DIR) -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        business_dir = directory / self.business_id.lower().replace(" ", "_")
        business_dir.mkdir(exist_ok=True)
        
        filepath = business_dir / f"{self.account_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'BusinessAccount':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class Transaction:
    """Represents a financial transaction."""
    transaction_id: str
    business_id: str
    account_id: str
    type: TransactionType
    amount: float
    currency: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    payment_method: PaymentMethod = PaymentMethod.OTHER
    status: str = "completed"
    category: str = "uncategorized"
    reference: str = ""
    counterparty: str = ""
    tax_amount: float = 0.0
    tax_category: Optional[TaxCategory] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    receipt_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transaction_id": self.transaction_id,
            "business_id": self.business_id,
            "account_id": self.account_id,
            "type": self.type.value,
            "amount": self.amount,
            "currency": self.currency,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "payment_method": self.payment_method.value,
            "status": self.status,
            "category": self.category,
            "reference": self.reference,
            "counterparty": self.counterparty,
            "tax_amount": self.tax_amount,
            "tax_category": self.tax_category.value if self.tax_category else None,
            "metadata": self.metadata,
            "receipt_path": str(self.receipt_path) if self.receipt_path else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create from dictionary."""
        return cls(
            transaction_id=data["transaction_id"],
            business_id=data["business_id"],
            account_id=data["account_id"],
            type=TransactionType(data["type"]),
            amount=data["amount"],
            currency=data["currency"],
            description=data["description"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payment_method=PaymentMethod(data["payment_method"]),
            status=data.get("status", "completed"),
            category=data.get("category", "uncategorized"),
            reference=data.get("reference", ""),
            counterparty=data.get("counterparty", ""),
            tax_amount=data.get("tax_amount", 0.0),
            tax_category=TaxCategory(data["tax_category"]) if data.get("tax_category") else None,
            metadata=data.get("metadata", {}),
            receipt_path=Path(data["receipt_path"]) if data.get("receipt_path") else None
        )
    
    def save(self, directory: Path = TRANSACTIONS_DIR) -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        business_dir = directory / self.business_id.lower().replace(" ", "_")
        business_dir.mkdir(exist_ok=True)
        
        # Create year/month subdirectories for better organization
        year_dir = business_dir / str(self.timestamp.year)
        year_dir.mkdir(exist_ok=True)
        month_dir = year_dir / f"{self.timestamp.month:02d}"
        month_dir.mkdir(exist_ok=True)
        
        filepath = month_dir / f"{self.transaction_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'Transaction':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class BusinessReport:
    """Represents a business report."""
    report_id: str
    business_id: str
    type: ReportType
    title: str
    start_date: datetime
    end_date: datetime
    created_at: datetime = field(default_factory=datetime.now)
    summary: str = ""
    income: float = 0.0
    expenses: float = 0.0
    profit: float = 0.0
    tax_collected: float = 0.0
    tax_paid: float = 0.0
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    pdf_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "business_id": self.business_id,
            "type": self.type.value,
            "title": self.title,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "created_at": self.created_at.isoformat(),
            "summary": self.summary,
            "income": self.income,
            "expenses": self.expenses,
            "profit": self.profit,
            "tax_collected": self.tax_collected,
            "tax_paid": self.tax_paid,
            "notes": self.notes,
            "metadata": self.metadata,
            "pdf_path": str(self.pdf_path) if self.pdf_path else None,
            "csv_path": str(self.csv_path) if self.csv_path else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessReport':
        """Create from dictionary."""
        return cls(
            report_id=data["report_id"],
            business_id=data["business_id"],
            type=ReportType(data["type"]),
            title=data["title"],
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            summary=data.get("summary", ""),
            income=data.get("income", 0.0),
            expenses=data.get("expenses", 0.0),
            profit=data.get("profit", 0.0),
            tax_collected=data.get("tax_collected", 0.0),
            tax_paid=data.get("tax_paid", 0.0),
            notes=data.get("notes", ""),
            metadata=data.get("metadata", {}),
            pdf_path=Path(data["pdf_path"]) if data.get("pdf_path") else None,
            csv_path=Path(data["csv_path"]) if data.get("csv_path") else None
        )
    
    def save(self, directory: Path = REPORTS_DIR) -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        business_dir = directory / self.business_id.lower().replace(" ", "_")
        business_dir.mkdir(exist_ok=True)
        
        # Create year subdirectory
        year_dir = business_dir / str(self.created_at.year)
        year_dir.mkdir(exist_ok=True)
        
        filepath = year_dir / f"{self.report_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'BusinessReport':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class BusinessOperation:
    """Represents a business operation."""
    operation_id: str
    business_id: str
    name: str
    type: BusinessType
    status: BusinessStatus
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    target_revenue: float = 0.0
    actual_revenue: float = 0.0
    expenses: float = 0.0
    profit: float = 0.0
    assigned_agents: List[str] = field(default_factory=list)
    accounts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_id": self.operation_id,
            "business_id": self.business_id,
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "target_revenue": self.target_revenue,
            "actual_revenue": self.actual_revenue,
            "expenses": self.expenses,
            "profit": self.profit,
            "assigned_agents": self.assigned_agents,
            "accounts": self.accounts,
            "metrics": self.metrics,
            "tasks": self.tasks,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessOperation':
        """Create from dictionary."""
        return cls(
            operation_id=data["operation_id"],
            business_id=data["business_id"],
            name=data["name"],
            type=BusinessType(data["type"]),
            status=BusinessStatus(data["status"]),
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            target_revenue=data.get("target_revenue", 0.0),
            actual_revenue=data.get("actual_revenue", 0.0),
            expenses=data.get("expenses", 0.0),
            profit=data.get("profit", 0.0),
            assigned_agents=data.get("assigned_agents", []),
            accounts=data.get("accounts", []),
            metrics=data.get("metrics", {}),
            tasks=data.get("tasks", []),
            metadata=data.get("metadata", {})
        )
    
    def save(self, directory: Path = BUSINESS_DIR / "operations") -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        business_dir = directory / self.business_id.lower().replace(" ", "_")
        business_dir.mkdir(exist_ok=True)
        
        filepath = business_dir / f"{self.operation_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'BusinessOperation':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class PersonaProfile:
    """Represents a persona profile for business account creation."""
    profile_id: str
    first_name: str
    last_name: str
    email: str
    phone: str = ""
    date_of_birth: Optional[datetime] = None
    address: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country: str = "Australia"
    occupation: str = ""
    bio: str = ""
    interests: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    profile_image_path: Optional[Path] = None
    social_profiles: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "profile_id": self.profile_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "phone": self.phone,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "postal_code": self.postal_code,
            "country": self.country,
            "occupation": self.occupation,
            "bio": self.bio,
            "interests": self.interests,
            "skills": self.skills,
            "profile_image_path": str(self.profile_image_path) if self.profile_image_path else None,
            "social_profiles": self.social_profiles,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaProfile':
        """Create from dictionary."""
        return cls(
            profile_id=data["profile_id"],
            first_name=data["first_name"],
            last_name=data["last_name"],
            email=data["email"],
            phone=data.get("phone", ""),
            date_of_birth=datetime.fromisoformat(data["date_of_birth"]) if data.get("date_of_birth") else None,
            address=data.get("address", ""),
            city=data.get("city", ""),
            state=data.get("state", ""),
            postal_code=data.get("postal_code", ""),
            country=data.get("country", "Australia"),
            occupation=data.get("occupation", ""),
            bio=data.get("bio", ""),
            interests=data.get("interests", []),
            skills=data.get("skills", []),
            profile_image_path=Path(data["profile_image_path"]) if data.get("profile_image_path") else None,
            social_profiles=data.get("social_profiles", {}),
            created_at=datetime.fromisoformat(data["created_at"])
        )
    
    def save(self, directory: Path = BUSINESS_DIR / "personas") -> Path:
        """Save to file."""
        directory.mkdir(exist_ok=True, parents=True)
        
        filepath = directory / f"{self.profile_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'PersonaProfile':
        """Load from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

class BusinessManager:
    """Manager for handling business operations and finances."""
    
    def __init__(self):
        self.businesses: Dict[str, BusinessIdentity] = {}
        self.operations: Dict[str, BusinessOperation] = {}
        self.accounts: Dict[str, Dict[str, BusinessAccount]] = {}  # business_id -> account_id -> account
        self.credentials: Dict[str, AccountCredential] = {}
        self.transactions: Dict[str, List[Transaction]] = {}  # business_id -> transactions
        self.reports: Dict[str, List[BusinessReport]] = {}  # business_id -> reports
        self.personas: Dict[str, PersonaProfile] = {}
        
        self.active = False
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # Task queues
        self.report_queue: queue.Queue = queue.Queue()
        self.transaction_queue: queue.Queue = queue.Queue()
        self.registration_queue: queue.Queue = queue.Queue()
        
        # Load existing data
        self.load_all_data()
        
        logger.info("BusinessManager initialized")
    
    def load_all_data(self) -> None:
        """Load all business data from files."""
        # Load businesses
        if BUSINESS_DIR.exists():
            for filepath in BUSINESS_DIR.glob("*_identity.json"):
                try:
                    business = BusinessIdentity.load(filepath)
                    self.businesses[business.abn] = business
                    logger.info(f"Loaded business: {business.name} (ABN: {business.abn})")
                except Exception as e:
                    logger.error(f"Error loading business from {filepath}: {str(e)}")
        
        # Load personas
        personas_dir = BUSINESS_DIR / "personas"
        if personas_dir.exists():
            for filepath in personas_dir.glob("*.json"):
                try:
                    persona = PersonaProfile.load(filepath)
                    self.personas[persona.profile_id] = persona
                    logger.info(f"Loaded persona: {persona.first_name} {persona.last_name}")
                except Exception as e:
                    logger.error(f"Error loading persona from {filepath}: {str(e)}")
        
        # Load operations
        operations_dir = BUSINESS_DIR / "operations"
        if operations_dir.exists():
            for business_dir in operations_dir.iterdir():
                if business_dir.is_dir():
                    for filepath in business_dir.glob("*.json"):
                        try:
                            operation = BusinessOperation.load(filepath)
                            self.operations[operation.operation_id] = operation
                            logger.info(f"Loaded operation: {operation.name}")
                        except Exception as e:
                            logger.error(f"Error loading operation from {filepath}: {str(e)}")
        
        # Load accounts
        if ACCOUNTS_DIR.exists():
            for business_dir in ACCOUNTS_DIR.iterdir():
                if business_dir.is_dir():
                    business_id = business_dir.name
                    self.accounts[business_id] = {}
                    
                    for filepath in business_dir.glob("*.json"):
                        try:
                            account = BusinessAccount.load(filepath)
                            self.accounts[business_id][account.account_id] = account
                            logger.info(f"Loaded account: {account.service_name} for business {business_id}")
                        except Exception as e:
                            logger.error(f"Error loading account from {filepath}: {str(e)}")
        
        # Load credentials
        if CREDENTIALS_DIR.exists():
            for service_dir in CREDENTIALS_DIR.iterdir():
                if service_dir.is_dir():
                    for filepath in service_dir.glob("*.json"):
                        try:
                            credential = AccountCredential.load(filepath)
                            self.credentials[credential.account_id] = credential
                            logger.info(f"Loaded credential: {credential.service_name} ({credential.account_id})")
                        except Exception as e:
                            logger.error(f"Error loading credential from {filepath}: {str(e)}")
        
        # Load transactions (most recent 100 per business)
        if TRANSACTIONS_DIR.exists():
            for business_dir in TRANSACTIONS_DIR.iterdir():
                if business_dir.is_dir():
                    business_id = business_dir.name
                    self.transactions[business_id] = []
                    
                    # Get all transaction files
                    transaction_files = []
                    for year_dir in business_dir.iterdir():
                        if year_dir.is_dir():
                            for month_dir in year_dir.iterdir():
                                if month_dir.is_dir():
                                    transaction_files.extend(month_dir.glob("*.json"))
                    
                    # Sort by modification time (newest first) and take first 100
                    transaction_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    transaction_files = transaction_files[:100]
                    
                    for filepath in transaction_files:
                        try:
                            transaction = Transaction.load(filepath)
                            self.transactions[business_id].append(transaction)
                            logger.debug(f"Loaded transaction: {transaction.transaction_id}")
                        except Exception as e:
                            logger.error(f"Error loading transaction from {filepath}: {str(e)}")
        
        # Load reports
        if REPORTS_DIR.exists():
            for business_dir in REPORTS_DIR.iterdir():
                if business_dir.is_dir():
                    business_id = business_dir.name
                    self.reports[business_id] = []
                    
                    # Get all report files from all year directories
                    report_files = []
                    for year_dir in business_dir.iterdir():
                        if year_dir.is_dir():
                            report_files.extend(year_dir.glob("*.json"))
                    
                    for filepath in report_files:
                        try:
                            report = BusinessReport.load(filepath)
                            self.reports[business_id].append(report)
                            logger.info(f"Loaded report: {report.title}")
                        except Exception as e:
                            logger.error(f"Error loading report from {filepath}: {str(e)}")
    
    def start(self) -> bool:
        """Start the business manager."""
        if self.active:
            logger.warning("BusinessManager is already running")
            return False
        
        self.active = True
        self.stop_event.clear()
        
        # Start worker threads
        self.start_workers()
        
        logger.info("BusinessManager started")
        return True
    
    def stop(self) -> bool:
        """Stop the business manager."""
        if not self.active:
            logger.warning("BusinessManager is not running")
            return False
        
        self.active = False
        self.stop_event.set()
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.worker_threads = []
        
        logger.info("BusinessManager stopped")
        return True
    
    def start_workers(self) -> None:
        """Start worker threads for various business tasks."""
        # Report generation worker
        report_worker = threading.Thread(
            target=self.report_worker,
            name="ReportWorker",
            daemon=True
        )
        self.worker_threads.append(report_worker)
        report_worker.start()
        
        # Transaction processing worker
        transaction_worker = threading.Thread(
            target=self.transaction_worker,
            name="TransactionWorker",
            daemon=True
        )
        self.worker_threads.append(transaction_worker)
        transaction_worker.start()
        
        # Registration worker
        registration_worker = threading.Thread(
            target=self.registration_worker,
            name="RegistrationWorker",
            daemon=True
        )
        self.worker_threads.append(registration_worker)
        registration_worker.start()
        
        # Periodic tasks worker
        periodic_worker = threading.Thread(
            target=self.periodic_tasks_worker,
            name="PeriodicTasksWorker",
            daemon=True
        )
        self.worker_threads.append(periodic_worker)
        periodic_worker.start()
        
        logger.info("Started worker threads")
    
    def report_worker(self) -> None:
        """Worker thread for generating reports."""
        logger.info("Report worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get report task from queue
                try:
                    report_task = self.report_queue.get(timeout=1.0)
                    
                    # Generate report
                    report_type = report_task.get("type", ReportType.DAILY)
                    business_id = report_task.get("business_id")
                    
                    if report_type == ReportType.DAILY:
                        self.generate_daily_report(business_id)
                    elif report_type == ReportType.WEEKLY:
                        self.generate_weekly_report(business_id)
                    elif report_type == ReportType.MONTHLY:
                        self.generate_monthly_report(business_id)
                    elif report_type == ReportType.QUARTERLY:
                        self.generate_quarterly_report(business_id)
                    elif report_type == ReportType.ANNUAL:
                        self.generate_annual_report(business_id)
                    elif report_type == ReportType.TAX:
                        self.generate_tax_report(business_id)
                    elif report_type == ReportType.BAS:
                        self.generate_bas_report(business_id)
                    else:
                        logger.warning(f"Unknown report type: {report_type}")
                    
                    self.report_queue.task_done()
                
                except queue.Empty:
                    pass
            
            except Exception as e:
                logger.error(f"Error in report worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
    
    def transaction_worker(self) -> None:
        """Worker thread for processing transactions."""
        logger.info("Transaction worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get transaction task from queue
                try:
                    transaction_task = self.transaction_queue.get(timeout=1.0)
                    
                    # Process transaction
                    transaction_type = transaction_task.get("type")
                    
                    if transaction_type == "record":
                        # Record a new transaction
                        transaction = Transaction(
                            transaction_id=transaction_task.get("transaction_id", f"tx-{str(uuid.uuid4())[:8]}"),
                            business_id=transaction_task.get("business_id"),
                            account_id=transaction_task.get("account_id"),
                            type=TransactionType(transaction_task.get("transaction_type", "other")),
                            amount=transaction_task.get("amount", 0.0),
                            currency=transaction_task.get("currency", "AUD"),
                            description=transaction_task.get("description", ""),
                            timestamp=transaction_task.get("timestamp", datetime.now()),
                            payment_method=PaymentMethod(transaction_task.get("payment_method", "other")),
                            status=transaction_task.get("status", "completed"),
                            category=transaction_task.get("category", "uncategorized"),
                            reference=transaction_task.get("reference", ""),
                            counterparty=transaction_task.get("counterparty", ""),
                            tax_amount=transaction_task.get("tax_amount", 0.0),
                            tax_category=TaxCategory(transaction_task.get("tax_category")) if transaction_task.get("tax_category") else None,
                            metadata=transaction_task.get("metadata", {}),
                            receipt_path=Path(transaction_task.get("receipt_path")) if transaction_task.get("receipt_path") else None
                        )
                        
                        self.record_transaction(transaction)
                    
                    elif transaction_type == "update":
                        # Update an existing transaction
                        transaction_id = transaction_task.get("transaction_id")
                        business_id = transaction_task.get("business_id")
                        
                        if business_id in self.transactions:
                            for i, tx in enumerate(self.transactions[business_id]):
                                if tx.transaction_id == transaction_id:
                                    # Update fields
                                    for key, value in transaction_task.items():
                                        if key not in ["type", "transaction_id", "business_id"] and hasattr(tx, key):
                                            setattr(tx, key, value)
                                    
                                    # Save updated transaction
                                    tx.save()
                                    self.transactions[business_id][i] = tx
                                    break
                    
                    self.transaction_queue.task_done()
                
                except queue.Empty:
                    pass
            
            except Exception as e:
                logger.error(f"Error in transaction worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
    
    def registration_worker(self) -> None:
        """Worker thread for handling business and account registrations."""
        logger.info("Registration worker started")
        
        while not self.stop_event.is_set():
            try:
                # Get registration task from queue
                try:
                    registration_task = self.registration_queue.get(timeout=1.0)
                    
                    # Process registration
                    registration_type = registration_task.get("type")
                    
                    if registration_type == "business":
                        # Register a new business
                        business = BusinessIdentity(
                            name=registration_task.get("name"),
                            business_type=BusinessType(registration_task.get("business_type", "other")),
                            abn=registration_task.get("abn", ""),
                            acn=registration_task.get("acn", ""),
                            tfn=registration_task.get("tfn", ""),
                            registration_date=registration_task.get("registration_date", datetime.now()),
                            business_email=registration_task.get("business_email", ""),
                            business_phone=registration_task.get("business_phone", ""),
                            business_address=registration_task.get("business_address", ""),
                            website=registration_task.get("website", ""),
                            social_media=registration_task.get("social_media", {})
                        )
                        
                        self.register_business(business)
                    
                    elif registration_type == "account":
                        # Register a new account
                        business_id = registration_task.get("business_id")
                        service_name = registration_task.get("service_name")
                        
                        if registration_task.get("use_automation", False) and (PLAYWRIGHT_AVAILABLE or SELENIUM_AVAILABLE):
                            # Use web automation to register
                            persona_id = registration_task.get("persona_id")
                            if persona_id in self.personas:
                                persona = self.personas[persona_id]
                                account_id = self.register_account_automated(
                                    business_id=business_id,
                                    service_name=service_name,
                                    service_url=registration_task.get("service_url"),
                                    account_type=registration_task.get("account_type", "standard"),
                                    persona=persona,
                                    automation_steps=registration_task.get("automation_steps", [])
                                )
                                
                                if account_id:
                                    logger.info(f"Successfully registered account {account_id} for {service_name}")
                        else:
                            # Manual registration (just create the account record)
                            account = BusinessAccount(
                                account_id=registration_task.get("account_id", f"acc-{str(uuid.uuid4())[:8]}"),
                                business_id=business_id,
                                service_name=service_name,
                                account_type=registration_task.get("account_type", "standard"),
                                status=registration_task.get("status", "active"),
                                balance=registration_task.get("balance", 0.0),
                                currency=registration_task.get("currency", "AUD"),
                                metadata=registration_task.get("metadata", {})
                            )
                            
                            # Create credentials if provided
                            if registration_task.get("username") and registration_task.get("password"):
                                credential = AccountCredential(
                                    account_id=account.account_id,
                                    service_name=service_name,
                                    username=registration_task.get("username"),
                                    password=registration_task.get("password"),
                                    email=registration_task.get("email", ""),
                                    api_key=registration_task.get("api_key", ""),
                                    secret_key=registration_task.get("secret_key", ""),
                                    token=registration_task.get("token", ""),
                                    url=registration_task.get("url", ""),
                                    notes=registration_task.get("notes", "")
                                )
                                
                                credential.save()
                                self.credentials[credential.account_id] = credential
                                account.credentials_id = credential.account_id
                            
                            self.add_account(account)
                    
                    elif registration_type == "persona":
                        # Create a new persona
                        persona = PersonaProfile(
                            profile_id=registration_task.get("profile_id", f"persona-{str(uuid.uuid4())[:8]}"),
                            first_name=registration_task.get("first_name"),
                            last_name=registration_task.get("last_name"),
                            email=registration_task.get("email"),
                            phone=registration_task.get("phone", ""),
                            date_of_birth=registration_task.get("date_of_birth"),
                            address=registration_task.get("address", ""),
                            city=registration_task.get("city", ""),
                            state=registration_task.get("state", ""),
                            postal_code=registration_task.get("postal_code", ""),
                            country=registration_task.get("country", "Australia"),
                            occupation=registration_task.get("occupation", ""),
                            bio=registration_task.get("bio", ""),
                            interests=registration_task.get("interests", []),
                            skills=registration_task.get("skills", [])
                        )
                        
                        persona.save()
                        self.personas[persona.profile_id] = persona
                    
                    self.registration_queue.task_done()
                
                except queue.Empty:
                    pass
            
            except Exception as e:
                logger.error(f"Error in registration worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
    
    def periodic_tasks_worker(self) -> None:
        """Worker thread for periodic tasks like daily reports and updates."""
        logger.info("Periodic tasks worker started")
        
        last_daily_report = datetime.now().date()
        last_weekly_report = datetime.now().date()
        last_monthly_report = datetime.now().date()
        
        while not self.stop_event.is_set():
            try:
                now = datetime.now()
                today = now.date()
                
                # Daily reports (once per day)
                if today > last_daily_report:
                    logger.info("Generating daily reports for all businesses")
                    for business_id in self.businesses:
                        self.report_queue.put({
                            "type": ReportType.DAILY,
                            "business_id": business_id
                        })
                    last_daily_report = today
                
                # Weekly reports (once per week on Monday)
                if today > last_weekly_report and now.weekday() == 0:  # Monday
                    logger.info("Generating weekly reports for all businesses")
                    for business_id in self.businesses:
                        self.report_queue.put({
                            "type": ReportType.WEEKLY,
                            "business_id": business_id
                        })
                    last_weekly_report = today
                
                # Monthly reports (first day of month)
                if today > last_monthly_report and today.day == 1:
                    logger.info("Generating monthly reports for all businesses")
                    for business_id in self.businesses:
                        self.report_queue.put({
                            "type": ReportType.MONTHLY,
                            "business_id": business_id
                        })
                    last_monthly_report = today
                
                # Update operation metrics
                self.update_operation_metrics()
            
            except Exception as e:
                logger.error(f"Error in periodic tasks worker: {str(e)}")
            
            # Sleep for 1 hour before checking again
            for _ in range(60):  # Check stop_event every minute
                if self.stop_event.is_set():
                    break
                time.sleep(60)
    
    def register_business(self, business: BusinessIdentity) -> bool:
        """Register a new business."""
        if business.abn in self.businesses:
            logger.warning(f"Business with ABN {business.abn} already exists")
            return False
        
        # Save business to file
        business.save()
        
        # Add to in-memory cache
        self.businesses[business.abn] = business
        
        # Initialize containers for this business
        self.accounts[business.abn] = {}
        self.transactions[business.abn] = []
        self.reports[business.abn] = []
        
        logger.info(f"Registered business: {business.name} (ABN: {business.abn})")
        return True
    
    def add_account(self, account: BusinessAccount) -> bool:
        """Add a new business account."""
        if account.business_id not in self.accounts:
            self.accounts[account.business_id] = {}
        
        if account.account_id in self.accounts[account.business_id]:
            logger.warning(f"Account {account.account_id} already exists for business {account.business_id}")
            return False
        
        # Save account to file
        account.save()
        
        # Add to in-memory cache
        self.accounts[account.business_id][account.account_id] = account
        
        logger.info(f"Added account: {account.service_name} ({account.account_id}) for business {account.business_id}")
        return True
    
    def register_account_automated(
        self, 
        business_id: str, 
        service_name: str, 
        service_url: str,
        account_type: str,
        persona: PersonaProfile,
        automation_steps: List[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Register an account using web automation."""
        if not (PLAYWRIGHT_AVAILABLE or SELENIUM_AVAILABLE):
            logger.error("Web automation libraries not available")
            return None
        
        logger.info(f"Attempting automated registration for {service_name}")
        
        account_id = f"acc-{str(uuid.uuid4())[:8]}"
        username = f"{persona.first_name.lower()}{persona.last_name.lower()}{random.randint(100, 999)}"
        password = f"Skyscope{random.randint(1000, 9999)}!"
        
        try:
            if PLAYWRIGHT_AVAILABLE:
                # Use Playwright for automation
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    # Navigate to service URL
                    page.goto(service_url)
                    
                    # Execute automation steps if provided
                    if automation_steps:
                        for step in automation_steps:
                            action = step.get("action")
                            
                            if action == "click":
                                selector = step.get("selector")
                                page.click(selector)
                            elif action == "fill":
                                selector = step.get("selector")
                                value = step.get("value")
                                
                                # Replace placeholders with persona data
                                if value == "{first_name}":
                                    value = persona.first_name
                                elif value == "{last_name}":
                                    value = persona.last_name
                                elif value == "{email}":
                                    value = persona.email
                                elif value == "{phone}":
                                    value = persona.phone
                                elif value == "{username}":
                                    value = username
                                elif value == "{password}":
                                    value = password
                                
                                page.fill(selector, value)
                            elif action == "select":
                                selector = step.get("selector")
                                value = step.get("value")
                                page.select_option(selector, value)
                            elif action == "wait":
                                duration = step.get("duration", 1000)
                                page.wait_for_timeout(duration)
                            elif action == "wait_for_selector":
                                selector = step.get("selector")
                                page.wait_for_selector(selector)
                            elif action == "wait_for_navigation":
                                page.wait_for_navigation()
                    
                    # Take screenshot for verification
                    screenshot_path = CREDENTIALS_DIR / service_name.lower().replace(" ", "_") / f"{account_id}_registration.png"
                    screenshot_path.parent.mkdir(exist_ok=True, parents=True)
                    page.screenshot(path=str(screenshot_path))
                    
                    browser.close()
            
            elif SELENIUM_AVAILABLE:
                # Use Selenium for automation
                driver = webdriver.Chrome()
                
                try:
                    # Navigate to service URL
                    driver.get(service_url)
                    
                    # Execute automation steps if provided
                    if automation_steps:
                        for step in automation_steps:
                            action = step.get("action")
                            
                            if action == "click":
                                selector = step.get("selector")
                                selector_type = step.get("selector_type", "css")
                                
                                element = None
                                if selector_type == "css":
                                    element = driver.find_element(By.CSS_SELECTOR, selector)
                                elif selector_type == "xpath":
                                    element = driver.find_element(By.XPATH, selector)
                                elif selector_type == "id":
                                    element = driver.find_element(By.ID, selector)
                                
                                if element:
                                    element.click()
                            
                            elif action == "fill":
                                selector = step.get("selector")
                                selector_type = step.get("selector_type", "css")
                                value = step.get("value")
                                
                                # Replace placeholders with persona data
                                if value == "{first_name}":
                                    value = persona.first_name
                                elif value == "{last_name}":
                                    value = persona.last_name
                                elif value == "{email}":
                                    value = persona.email
                                elif value == "{phone}":
                                    value = persona.phone
                                elif value == "{username}":
                                    value = username
                                elif value == "{password}":
                                    value = password
                                
                                element = None
                                if selector_type == "css":
                                    element = driver.find_element(By.CSS_SELECTOR, selector)
                                elif selector_type == "xpath":
                                    element = driver.find_element(By.XPATH, selector)
                                elif selector_type == "id":
                                    element = driver.find_element(By.ID, selector)
                                
                                if element:
                                    element.send_keys(value)
                            
                            elif action == "select":
                                # Implementation for select
                                pass
                            
                            elif action == "wait":
                                duration = step.get("duration", 1)
                                time.sleep(duration)
                            
                            elif action == "wait_for_selector":
                                selector = step.get("selector")
                                selector_type = step.get("selector_type", "css")
                                timeout = step.get("timeout", 10)
                                
                                wait = WebDriverWait(driver, timeout)
                                
                                if selector_type == "css":
                                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                                elif selector_type == "xpath":
                                    wait.until(EC.presence_of_element_located((By.XPATH, selector)))
                                elif selector_type == "id":
                                    wait.until(EC.presence_of_element_located((By.ID, selector)))
                    
                    # Take screenshot for verification
                    screenshot_path = CREDENTIALS_DIR / service_name.lower().replace(" ", "_") / f"{account_id}_registration.png"
                    screenshot_path.parent.mkdir(exist_ok=True, parents=True)
                    driver.save_screenshot(str(screenshot_path))
                
                finally:
                    driver.quit()
            
            # Create account record
            account = BusinessAccount(
                account_id=account_id,
                business_id=business_id,
                service_name=service_name,
                account_type=account_type,
                status="active",
                balance=0.0,
                currency="AUD",
                metadata={
                    "registration_method": "automated",
                    "registration_date": datetime.now().isoformat(),
                    "screenshot_path": str(screenshot_path) if 'screenshot_path' in locals() else None
                }
            )
            
            # Create credential record
            credential = AccountCredential(
                account_id=account_id,
                service_name=service_name,
                username=username,
                password=password,
                email=persona.email,
                url=service_url,
                notes=f"Automated registration using persona {persona.profile_id}"
            )
            
            # Save records
            credential.save()
            self.credentials[credential.account_id] = credential
            
            account.credentials_id = credential.account_id
            account.save()
            
            if business_id not in self.accounts:
                self.accounts[business_id] = {}
            
            self.accounts[business_id][account_id] = account
            
            logger.info(f"Successfully registered account {account_id} for {service_name}")
            return account_id
        
        except Exception as e:
            logger.error(f"Error during automated account registration: {str(e)}")
            return None
    
    def create_business_operation(
        self,
        business_id: str,
        name: str,
        type: BusinessType,
        description: str,
        target_revenue: float = 0.0,
        assigned_agents: List[str] = None
    ) -> Optional[str]:
        """Create a new business operation."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        operation_id = f"op-{str(uuid.uuid4())[:8]}"
        
        operation = BusinessOperation(
            operation_id=operation_id,
            business_id=business_id,
            name=name,
            type=type,
            status=BusinessStatus.PLANNING,
            description=description,
            target_revenue=target_revenue,
            assigned_agents=assigned_agents or []
        )
        
        # Save operation to file
        operation.save()
        
        # Add to in-memory cache
        self.operations[operation_id] = operation
        
        logger.info(f"Created business operation: {name} ({operation_id})")
        return operation_id
    
    def update_operation_status(self, operation_id: str, status: BusinessStatus) -> bool:
        """Update the status of a business operation."""
        if operation_id not in self.operations:
            logger.warning(f"Operation {operation_id} not found")
            return False
        
        operation = self.operations[operation_id]
        operation.status = status
        
        if status == BusinessStatus.ACTIVE and not operation.started_at:
            operation.started_at = datetime.now()
        elif status in [BusinessStatus.CLOSING, BusinessStatus.FAILED] and not operation.completed_at:
            operation.completed_at = datetime.now()
        
        # Save updated operation
        operation.save()
        
        logger.info(f"Updated operation {operation_id} status to {status.value}")
        return True
    
    def update_operation_metrics(self) -> None:
        """Update metrics for all active operations."""
        for operation_id, operation in self.operations.items():
            if operation.status != BusinessStatus.ACTIVE:
                continue
            
            # Calculate revenue and expenses
            revenue = 0.0
            expenses = 0.0
            
            if operation.business_id in self.transactions:
                for tx in self.transactions[operation.business_id]:
                    # Check if transaction is related to this operation
                    if tx.metadata.get("operation_id") == operation_id:
                        if tx.type == TransactionType.INCOME:
                            revenue += tx.amount
                        elif tx.type == TransactionType.EXPENSE:
                            expenses += tx.amount
            
            # Update operation metrics
            operation.actual_revenue = revenue
            operation.expenses = expenses
            operation.profit = revenue - expenses
            
            # Update other metrics
            if operation.accounts:
                active_accounts = 0
                for account_id in operation.accounts:
                    if account_id in self.accounts.get(operation.business_id, {}):
                        account = self.accounts[operation.business_id][account_id]
                        if account.status == "active":
                            active_accounts += 1
                
                operation.metrics["active_accounts"] = active_accounts
            
            # Save updated operation
            operation.save()
    
    def record_transaction(self, transaction: Transaction) -> bool:
        """Record a new financial transaction."""
        if transaction.business_id not in self.transactions:
            self.transactions[transaction.business_id] = []
        
        # Save transaction to file
        transaction.save()
        
        # Add to in-memory cache
        self.transactions[transaction.business_id].append(transaction)
        
        # Update account balance if applicable
        if transaction.account_id and transaction.business_id in self.accounts and transaction.account_id in self.accounts[transaction.business_id]:
            account = self.accounts[transaction.business_id][transaction.account_id]
            
            if transaction.type == TransactionType.INCOME:
                account.balance += transaction.amount
            elif transaction.type == TransactionType.EXPENSE:
                account.balance -= transaction.amount
            elif transaction.type == TransactionType.REFUND:
                account.balance += transaction.amount
            
            account.last_updated = datetime.now()
            account.save()
        
        logger.info(f"Recorded transaction {transaction.transaction_id} for business {transaction.business_id}")
        return True
    
    def generate_daily_report(self, business_id: str) -> Optional[str]:
        """Generate a daily business report."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for today
        today = datetime.now().date()
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.DAILY,
            title=f"Daily Business Report - {today.strftime('%Y-%m-%d')}",
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_weekly_report(self, business_id: str) -> Optional[str]:
        """Generate a weekly business report."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for the past week
        today = datetime.now().date()
        start_date = datetime.combine(today - timedelta(days=7), datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.WEEKLY,
            title=f"Weekly Business Report - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_monthly_report(self, business_id: str) -> Optional[str]:
        """Generate a monthly business report."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for the past month
        today = datetime.now().date()
        first_day = today.replace(day=1)
        last_month = first_day - timedelta(days=1)
        start_date = datetime.combine(last_month.replace(day=1), datetime.min.time())
        end_date = datetime.combine(first_day - timedelta(days=1), datetime.max.time())
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.MONTHLY,
            title=f"Monthly Business Report - {start_date.strftime('%B %Y')}",
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_quarterly_report(self, business_id: str) -> Optional[str]:
        """Generate a quarterly business report."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for the past quarter
        today = datetime.now().date()
        current_month = today.month
        current_quarter = (current_month - 1) // 3 + 1
        
        if current_quarter == 1:
            start_month = 10
            start_year = today.year - 1
            end_month = 12
            end_year = today.year - 1
        else:
            start_month = (current_quarter - 2) * 3 + 1
            start_year = today.year
            end_month = (current_quarter - 1) * 3
            end_year = today.year
        
        start_date = datetime(start_year, start_month, 1)
        if end_month == 12:
            end_date = datetime(end_year, end_month, 31, 23, 59, 59)
        else:
            end_date = datetime(end_year, end_month + 1, 1) - timedelta(seconds=1)
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.QUARTERLY,
            title=f"Quarterly Business Report - Q{current_quarter-1} {end_year}",
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_annual_report(self, business_id: str) -> Optional[str]:
        """Generate an annual business report."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for the past year
        today = datetime.now().date()
        start_date = datetime(today.year - 1, 7, 1)  # Australian financial year starts July 1
        end_date = datetime(today.year, 6, 30, 23, 59, 59)  # Australian financial year ends June 30
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.ANNUAL,
            title=f"Annual Business Report - FY{start_date.year}/{end_date.year}",
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_tax_report(self, business_id: str) -> Optional[str]:
        """Generate a tax report for Australian taxation."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for the past financial year
        today = datetime.now().date()
        
        # Australian financial year: July 1 to June 30
        if today.month > 6:
            start_date = datetime(today.year, 7, 1)
            end_date = datetime(today.year + 1, 6, 30, 23, 59, 59)
        else:
            start_date = datetime(today.year - 1, 7, 1)
            end_date = datetime(today.year, 6, 30, 23, 59, 59)
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.TAX,
            title=f"Tax Report - FY{start_date.year}/{end_date.year}",
            start_date=start_date,
            end_date=end_date
        )
    
    def generate_bas_report(self, business_id: str) -> Optional[str]:
        """Generate a Business Activity Statement (BAS) report for Australian taxation."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Set date range for the past quarter (BAS is typically quarterly in Australia)
        today = datetime.now().date()
        current_month = today.month
        current_quarter = (current_month - 1) // 3 + 1
        
        if current_quarter == 1:
            start_month = 10
            start_year = today.year - 1
            end_month = 12
            end_year = today.year - 1
        else:
            start_month = (current_quarter - 2) * 3 + 1
            start_year = today.year
            end_month = (current_quarter - 1) * 3
            end_year = today.year
        
        start_date = datetime(start_year, start_month, 1)
        if end_month == 12:
            end_date = datetime(end_year, end_month, 31, 23, 59, 59)
        else:
            end_date = datetime(end_year, end_month + 1, 1) - timedelta(seconds=1)
        
        return self._generate_report(
            business_id=business_id,
            type=ReportType.BAS,
            title=f"Business Activity Statement - Q{current_quarter-1} {end_year}",
            start_date=start_date,
            end_date=end_date
        )
    
    def _generate_report(
        self,
        business_id: str,
        type: ReportType,
        title: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[str]:
        """Internal method to generate a business report."""
        if business_id not in self.businesses:
            logger.warning(f"Business {business_id} not found")
            return None
        
        business = self.businesses[business_id]
        
        # Calculate financial data for the period
        income = 0.0
        expenses = 0.0
        tax_collected = 0.0
        tax_paid = 0.0
        
        # Get transactions for the period
        period_transactions = []
        if business_id in self.transactions:
            for tx in self.transactions[business_id]:
                if start_date <= tx.timestamp <= end_date:
                    period_transactions.append(tx)
                    
                    if tx.type == TransactionType.INCOME:
                        income += tx.amount
                        if tx.tax_category == TaxCategory.GST:
                            tax_collected += tx.tax_amount
                    elif tx.type == TransactionType.EXPENSE:
                        expenses += tx.amount
                        if tx.tax_category == TaxCategory.GST:
                            tax_paid += tx.tax_amount
        
        profit = income - expenses
        
        # Create report object
        report_id = f"report-{str(uuid.uuid4())[:8]}"
        report = BusinessReport(
            report_id=report_id,
            business_id=business_id,
            type=type,
            title=title,
            start_date=start_date,
            end_date=end_date,
            summary=f"Financial summary for {business.name} from
