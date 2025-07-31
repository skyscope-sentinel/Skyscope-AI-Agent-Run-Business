import os
import sys
import json
import time
import uuid
import logging
import datetime
import threading
import traceback
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Import internal modules
try:
    from database_manager import DatabaseManager
    from agent_manager import AgentManager
    from performance_monitor import PerformanceMonitor
    from crypto_manager import CryptoManager
    from live_thinking_rag_system import LiveThinkingRAGSystem
except ImportError:
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/advanced_business_operations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_business_operations")

# Constants
CONFIG_DIR = Path("config")
REPORTS_DIR = Path("reports")
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
TEMPLATES_DIR = Path("templates")

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Australian tax rates for FY 2023-2024
AUS_TAX_RATES = [
    {"threshold": 0, "rate": 0.0, "base": 0},
    {"threshold": 18200, "rate": 0.19, "base": 0},
    {"threshold": 45000, "rate": 0.325, "base": 5092},
    {"threshold": 120000, "rate": 0.37, "base": 29467},
    {"threshold": 180000, "rate": 0.45, "base": 51667}
]

# GST rate in Australia (10%)
AUS_GST_RATE = 0.10

# Business types and structures in Australia
class BusinessStructure(Enum):
    """Business structures in Australia."""
    SOLE_TRADER = "sole_trader"
    PARTNERSHIP = "partnership"
    COMPANY = "company"
    TRUST = "trust"

class BusinessCategory(Enum):
    """Business categories."""
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    CONSULTING = "consulting"
    RETAIL = "retail"
    WHOLESALE = "wholesale"
    MANUFACTURING = "manufacturing"
    SERVICES = "services"
    CONTENT_CREATION = "content_creation"
    CRYPTO_TRADING = "crypto_trading"
    DROPSHIPPING = "dropshipping"
    AFFILIATE_MARKETING = "affiliate_marketing"
    DIGITAL_PRODUCTS = "digital_products"
    SUBSCRIPTION_SERVICES = "subscription_services"
    CUSTOM = "custom"

class SalesChannel(Enum):
    """Sales channels."""
    WEBSITE = "website"
    MARKETPLACE = "marketplace"
    SOCIAL_MEDIA = "social_media"
    DIRECT = "direct"
    WHOLESALE = "wholesale"
    RETAIL_STORE = "retail_store"
    AFFILIATE = "affiliate"
    EMAIL = "email"
    PHONE = "phone"
    MOBILE_APP = "mobile_app"
    API = "api"
    CUSTOM = "custom"

class PaymentMethod(Enum):
    """Payment methods."""
    CREDIT_CARD = "credit_card"
    BANK_TRANSFER = "bank_transfer"
    PAYPAL = "paypal"
    CRYPTO = "cryptocurrency"
    CASH = "cash"
    CHEQUE = "cheque"
    AFTERPAY = "afterpay"
    STRIPE = "stripe"
    DIRECT_DEBIT = "direct_debit"
    CUSTOM = "custom"

class CustomerSegment(Enum):
    """Customer segments."""
    B2B = "business_to_business"
    B2C = "business_to_consumer"
    B2G = "business_to_government"
    ENTERPRISE = "enterprise"
    SMB = "small_medium_business"
    STARTUP = "startup"
    INDIVIDUAL = "individual"
    CUSTOM = "custom"

class InventoryType(Enum):
    """Inventory types."""
    PHYSICAL = "physical"
    DIGITAL = "digital"
    SERVICE = "service"
    SUBSCRIPTION = "subscription"
    MIXED = "mixed"
    NONE = "none"

class SupplyChainModel(Enum):
    """Supply chain models."""
    DIRECT_MANUFACTURING = "direct_manufacturing"
    DROPSHIPPING = "dropshipping"
    WHOLESALE = "wholesale"
    JUST_IN_TIME = "just_in_time"
    THIRD_PARTY_LOGISTICS = "third_party_logistics"
    CUSTOM = "custom"

class ReportingFrequency(Enum):
    """Reporting frequencies."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    CUSTOM = "custom"

class BusinessMetric(Enum):
    """Business metrics."""
    REVENUE = "revenue"
    PROFIT = "profit"
    MARGIN = "margin"
    ROI = "roi"
    CAC = "customer_acquisition_cost"
    LTV = "lifetime_value"
    CHURN = "churn_rate"
    CONVERSION = "conversion_rate"
    GROWTH = "growth_rate"
    BURN_RATE = "burn_rate"
    RUNWAY = "runway"
    CUSTOM = "custom"

class AnalyticsModel(Enum):
    """Analytics models."""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ARIMA = "arima"
    PROPHET = "prophet"
    CUSTOM = "custom"

@dataclass
class BusinessConfig:
    """Configuration for a business operation."""
    name: str
    abn: str  # Australian Business Number
    structure: BusinessStructure
    category: BusinessCategory
    sales_channels: List[SalesChannel]
    payment_methods: List[PaymentMethod]
    customer_segments: List[CustomerSegment]
    inventory_type: InventoryType
    supply_chain_model: SupplyChainModel
    reporting_frequency: ReportingFrequency
    target_revenue: float
    target_profit_margin: float
    tax_settings: Dict[str, Any] = field(default_factory=dict)
    analytics_models: List[AnalyticsModel] = field(default_factory=list)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "abn": self.abn,
            "structure": self.structure.value,
            "category": self.category.value,
            "sales_channels": [channel.value for channel in self.sales_channels],
            "payment_methods": [method.value for method in self.payment_methods],
            "customer_segments": [segment.value for segment in self.customer_segments],
            "inventory_type": self.inventory_type.value,
            "supply_chain_model": self.supply_chain_model.value,
            "reporting_frequency": self.reporting_frequency.value,
            "target_revenue": self.target_revenue,
            "target_profit_margin": self.target_profit_margin,
            "tax_settings": self.tax_settings,
            "analytics_models": [model.value for model in self.analytics_models],
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BusinessConfig':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            abn=data["abn"],
            structure=BusinessStructure(data["structure"]),
            category=BusinessCategory(data["category"]),
            sales_channels=[SalesChannel(channel) for channel in data["sales_channels"]],
            payment_methods=[PaymentMethod(method) for method in data["payment_methods"]],
            customer_segments=[CustomerSegment(segment) for segment in data["customer_segments"]],
            inventory_type=InventoryType(data["inventory_type"]),
            supply_chain_model=SupplyChainModel(data["supply_chain_model"]),
            reporting_frequency=ReportingFrequency(data["reporting_frequency"]),
            target_revenue=data["target_revenue"],
            target_profit_margin=data["target_profit_margin"],
            tax_settings=data.get("tax_settings", {}),
            analytics_models=[AnalyticsModel(model) for model in data.get("analytics_models", [])],
            custom_settings=data.get("custom_settings", {})
        )
    
    def save(self, filepath: Path = CONFIG_DIR / "business_config.json") -> None:
        """Save configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Business configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving business configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = CONFIG_DIR / "business_config.json") -> 'BusinessConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Business configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                # Return default configuration
                return cls(
                    name="Skyscope Sentinel Business",
                    abn="12345678901",
                    structure=BusinessStructure.COMPANY,
                    category=BusinessCategory.SAAS,
                    sales_channels=[SalesChannel.WEBSITE, SalesChannel.DIRECT],
                    payment_methods=[PaymentMethod.CREDIT_CARD, PaymentMethod.CRYPTO],
                    customer_segments=[CustomerSegment.B2B, CustomerSegment.B2C],
                    inventory_type=InventoryType.DIGITAL,
                    supply_chain_model=SupplyChainModel.DIRECT_MANUFACTURING,
                    reporting_frequency=ReportingFrequency.MONTHLY,
                    target_revenue=100000.0,  # Six-figure target
                    target_profit_margin=0.40,  # 40% profit margin
                    tax_settings={
                        "country": "Australia",
                        "gst_registered": True,
                        "tax_year_end_month": 6  # June (Australian financial year)
                    },
                    analytics_models=[
                        AnalyticsModel.LINEAR_REGRESSION,
                        AnalyticsModel.RANDOM_FOREST,
                        AnalyticsModel.ARIMA
                    ]
                )
        except Exception as e:
            logger.error(f"Error loading business configuration: {e}")
            return cls(
                name="Skyscope Sentinel Business",
                abn="12345678901",
                structure=BusinessStructure.COMPANY,
                category=BusinessCategory.SAAS,
                sales_channels=[SalesChannel.WEBSITE, SalesChannel.DIRECT],
                payment_methods=[PaymentMethod.CREDIT_CARD, PaymentMethod.CRYPTO],
                customer_segments=[CustomerSegment.B2B, CustomerSegment.B2C],
                inventory_type=InventoryType.DIGITAL,
                supply_chain_model=SupplyChainModel.DIRECT_MANUFACTURING,
                reporting_frequency=ReportingFrequency.MONTHLY,
                target_revenue=100000.0,
                target_profit_margin=0.40,
                tax_settings={
                    "country": "Australia",
                    "gst_registered": True,
                    "tax_year_end_month": 6
                }
            )

@dataclass
class Product:
    """Product information."""
    id: str
    name: str
    description: str
    sku: str
    price: float
    cost: float
    inventory_type: InventoryType
    category: str
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    stock_level: int = 0
    reorder_point: int = 0
    supplier_id: Optional[str] = None
    tax_rate: float = AUS_GST_RATE
    digital_asset_url: Optional[str] = None
    is_active: bool = True
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    @property
    def margin(self) -> float:
        """Calculate margin percentage."""
        if self.price == 0:
            return 0.0
        return (self.price - self.cost) / self.price * 100.0
    
    @property
    def profit(self) -> float:
        """Calculate profit per unit."""
        return self.price - self.cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "sku": self.sku,
            "price": self.price,
            "cost": self.cost,
            "inventory_type": self.inventory_type.value,
            "category": self.category,
            "tags": self.tags,
            "attributes": self.attributes,
            "stock_level": self.stock_level,
            "reorder_point": self.reorder_point,
            "supplier_id": self.supplier_id,
            "tax_rate": self.tax_rate,
            "digital_asset_url": self.digital_asset_url,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "margin": self.margin,
            "profit": self.profit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Product':
        """Create from dictionary."""
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            sku=data["sku"],
            price=data["price"],
            cost=data["cost"],
            inventory_type=InventoryType(data["inventory_type"]),
            category=data["category"],
            tags=data.get("tags", []),
            attributes=data.get("attributes", {}),
            stock_level=data.get("stock_level", 0),
            reorder_point=data.get("reorder_point", 0),
            supplier_id=data.get("supplier_id"),
            tax_rate=data.get("tax_rate", AUS_GST_RATE),
            digital_asset_url=data.get("digital_asset_url"),
            is_active=data.get("is_active", True),
            created_at=created_at,
            updated_at=updated_at
        )

@dataclass
class Customer:
    """Customer information."""
    id: str
    name: str
    email: str
    segment: CustomerSegment
    acquisition_channel: SalesChannel
    acquisition_cost: float = 0.0
    lifetime_value: float = 0.0
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    company: Optional[str] = None
    abn: Optional[str] = None  # Australian Business Number for B2B
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    last_purchase_date: Optional[datetime.datetime] = None
    total_purchases: int = 0
    total_spent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "segment": self.segment.value,
            "acquisition_channel": self.acquisition_channel.value,
            "acquisition_cost": self.acquisition_cost,
            "lifetime_value": self.lifetime_value,
            "phone": self.phone,
            "address": self.address,
            "company": self.company,
            "abn": self.abn,
            "attributes": self.attributes,
            "tags": self.tags,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_purchase_date": self.last_purchase_date.isoformat() if self.last_purchase_date else None,
            "total_purchases": self.total_purchases,
            "total_spent": self.total_spent
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create from dictionary."""
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        last_purchase_date = datetime.datetime.fromisoformat(data["last_purchase_date"]) if data.get("last_purchase_date") and isinstance(data["last_purchase_date"], str) else data.get("last_purchase_date")
        
        return cls(
            id=data["id"],
            name=data["name"],
            email=data["email"],
            segment=CustomerSegment(data["segment"]),
            acquisition_channel=SalesChannel(data["acquisition_channel"]),
            acquisition_cost=data.get("acquisition_cost", 0.0),
            lifetime_value=data.get("lifetime_value", 0.0),
            phone=data.get("phone"),
            address=data.get("address"),
            company=data.get("company"),
            abn=data.get("abn"),
            attributes=data.get("attributes", {}),
            tags=data.get("tags", []),
            is_active=data.get("is_active", True),
            created_at=created_at,
            updated_at=updated_at,
            last_purchase_date=last_purchase_date,
            total_purchases=data.get("total_purchases", 0),
            total_spent=data.get("total_spent", 0.0)
        )

@dataclass
class OrderItem:
    """Order item information."""
    product_id: str
    quantity: int
    unit_price: float
    unit_cost: float
    tax_rate: float
    discount: float = 0.0
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal (before tax)."""
        return self.unit_price * self.quantity * (1 - self.discount)
    
    @property
    def tax_amount(self) -> float:
        """Calculate tax amount."""
        return self.subtotal * self.tax_rate
    
    @property
    def total(self) -> float:
        """Calculate total (including tax)."""
        return self.subtotal + self.tax_amount
    
    @property
    def cost(self) -> float:
        """Calculate total cost."""
        return self.unit_cost * self.quantity
    
    @property
    def profit(self) -> float:
        """Calculate profit."""
        return self.subtotal - self.cost
    
    @property
    def margin(self) -> float:
        """Calculate margin percentage."""
        if self.subtotal == 0:
            return 0.0
        return (self.subtotal - self.cost) / self.subtotal * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "product_id": self.product_id,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "unit_cost": self.unit_cost,
            "tax_rate": self.tax_rate,
            "discount": self.discount,
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "total": self.total,
            "cost": self.cost,
            "profit": self.profit,
            "margin": self.margin
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderItem':
        """Create from dictionary."""
        return cls(
            product_id=data["product_id"],
            quantity=data["quantity"],
            unit_price=data["unit_price"],
            unit_cost=data["unit_cost"],
            tax_rate=data["tax_rate"],
            discount=data.get("discount", 0.0)
        )

@dataclass
class Order:
    """Order information."""
    id: str
    customer_id: str
    items: List[OrderItem]
    status: str
    payment_method: PaymentMethod
    sales_channel: SalesChannel
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    paid_at: Optional[datetime.datetime] = None
    shipped_at: Optional[datetime.datetime] = None
    completed_at: Optional[datetime.datetime] = None
    cancelled_at: Optional[datetime.datetime] = None
    notes: str = ""
    shipping_address: Optional[Dict[str, str]] = None
    billing_address: Optional[Dict[str, str]] = None
    shipping_cost: float = 0.0
    shipping_tax_rate: float = AUS_GST_RATE
    discount_code: Optional[str] = None
    discount_amount: float = 0.0
    external_reference: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal (before tax)."""
        return sum(item.subtotal for item in self.items)
    
    @property
    def tax_amount(self) -> float:
        """Calculate tax amount."""
        return sum(item.tax_amount for item in self.items) + (self.shipping_cost * self.shipping_tax_rate)
    
    @property
    def shipping_tax_amount(self) -> float:
        """Calculate shipping tax amount."""
        return self.shipping_cost * self.shipping_tax_rate
    
    @property
    def total(self) -> float:
        """Calculate total (including tax and shipping)."""
        return self.subtotal + self.tax_amount + self.shipping_cost - self.discount_amount
    
    @property
    def cost(self) -> float:
        """Calculate total cost."""
        return sum(item.cost for item in self.items)
    
    @property
    def profit(self) -> float:
        """Calculate profit."""
        return self.subtotal - self.cost - self.shipping_cost - self.discount_amount
    
    @property
    def margin(self) -> float:
        """Calculate margin percentage."""
        if self.subtotal == 0:
            return 0.0
        return (self.subtotal - self.cost) / self.subtotal * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "items": [item.to_dict() for item in self.items],
            "status": self.status,
            "payment_method": self.payment_method.value,
            "sales_channel": self.sales_channel.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "shipped_at": self.shipped_at.isoformat() if self.shipped_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "notes": self.notes,
            "shipping_address": self.shipping_address,
            "billing_address": self.billing_address,
            "shipping_cost": self.shipping_cost,
            "shipping_tax_rate": self.shipping_tax_rate,
            "discount_code": self.discount_code,
            "discount_amount": self.discount_amount,
            "external_reference": self.external_reference,
            "attributes": self.attributes,
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "shipping_tax_amount": self.shipping_tax_amount,
            "total": self.total,
            "cost": self.cost,
            "profit": self.profit,
            "margin": self.margin
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create from dictionary."""
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        paid_at = datetime.datetime.fromisoformat(data["paid_at"]) if data.get("paid_at") and isinstance(data["paid_at"], str) else data.get("paid_at")
        shipped_at = datetime.datetime.fromisoformat(data["shipped_at"]) if data.get("shipped_at") and isinstance(data["shipped_at"], str) else data.get("shipped_at")
        completed_at = datetime.datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") and isinstance(data["completed_at"], str) else data.get("completed_at")
        cancelled_at = datetime.datetime.fromisoformat(data["cancelled_at"]) if data.get("cancelled_at") and isinstance(data["cancelled_at"], str) else data.get("cancelled_at")
        
        return cls(
            id=data["id"],
            customer_id=data["customer_id"],
            items=[OrderItem.from_dict(item) for item in data["items"]],
            status=data["status"],
            payment_method=PaymentMethod(data["payment_method"]),
            sales_channel=SalesChannel(data["sales_channel"]),
            created_at=created_at,
            updated_at=updated_at,
            paid_at=paid_at,
            shipped_at=shipped_at,
            completed_at=completed_at,
            cancelled_at=cancelled_at,
            notes=data.get("notes", ""),
            shipping_address=data.get("shipping_address"),
            billing_address=data.get("billing_address"),
            shipping_cost=data.get("shipping_cost", 0.0),
            shipping_tax_rate=data.get("shipping_tax_rate", AUS_GST_RATE),
            discount_code=data.get("discount_code"),
            discount_amount=data.get("discount_amount", 0.0),
            external_reference=data.get("external_reference"),
            attributes=data.get("attributes", {})
        )

@dataclass
class Supplier:
    """Supplier information."""
    id: str
    name: str
    contact_name: str
    email: str
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    abn: Optional[str] = None  # Australian Business Number
    website: Optional[str] = None
    payment_terms: str = "30 days"
    lead_time_days: int = 7
    minimum_order_value: float = 0.0
    currency: str = "AUD"
    notes: str = ""
    is_active: bool = True
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "contact_name": self.contact_name,
            "email": self.email,
            "phone": self.phone,
            "address": self.address,
            "abn": self.abn,
            "website": self.website,
            "payment_terms": self.payment_terms,
            "lead_time_days": self.lead_time_days,
            "minimum_order_value": self.minimum_order_value,
            "currency": self.currency,
            "notes": self.notes,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Supplier':
        """Create from dictionary."""
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        
        return cls(
            id=data["id"],
            name=data["name"],
            contact_name=data["contact_name"],
            email=data["email"],
            phone=data.get("phone"),
            address=data.get("address"),
            abn=data.get("abn"),
            website=data.get("website"),
            payment_terms=data.get("payment_terms", "30 days"),
            lead_time_days=data.get("lead_time_days", 7),
            minimum_order_value=data.get("minimum_order_value", 0.0),
            currency=data.get("currency", "AUD"),
            notes=data.get("notes", ""),
            is_active=data.get("is_active", True),
            created_at=created_at,
            updated_at=updated_at
        )

@dataclass
class InventoryTransaction:
    """Inventory transaction information."""
    id: str
    product_id: str
    transaction_type: str  # purchase, sale, adjustment, transfer
    quantity: int
    reference_id: Optional[str] = None  # Order ID, Purchase Order ID, etc.
    notes: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    location_id: Optional[str] = None
    cost_per_unit: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "product_id": self.product_id,
            "transaction_type": self.transaction_type,
            "quantity": self.quantity,
            "reference_id": self.reference_id,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "location_id": self.location_id,
            "cost_per_unit": self.cost_per_unit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InventoryTransaction':
        """Create from dictionary."""
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        
        return cls(
            id=data["id"],
            product_id=data["product_id"],
            transaction_type=data["transaction_type"],
            quantity=data["quantity"],
            reference_id=data.get("reference_id"),
            notes=data.get("notes", ""),
            created_at=created_at,
            location_id=data.get("location_id"),
            cost_per_unit=data.get("cost_per_unit")
        )

@dataclass
class FinancialTransaction:
    """Financial transaction information."""
    id: str
    transaction_type: str  # income, expense, transfer, adjustment
    amount: float
    description: str
    category: str
    date: datetime.date = field(default_factory=datetime.date.today)
    reference_id: Optional[str] = None  # Order ID, Invoice ID, etc.
    payment_method: Optional[PaymentMethod] = None
    tax_amount: float = 0.0
    tax_code: str = "GST"  # Australian tax code
    is_tax_deductible: bool = False
    notes: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "transaction_type": self.transaction_type,
            "amount": self.amount,
            "description": self.description,
            "category": self.category,
            "date": self.date.isoformat(),
            "reference_id": self.reference_id,
            "payment_method": self.payment_method.value if self.payment_method else None,
            "tax_amount": self.tax_amount,
            "tax_code": self.tax_code,
            "is_tax_deductible": self.is_tax_deductible,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FinancialTransaction':
        """Create from dictionary."""
        date = datetime.date.fromisoformat(data["date"]) if isinstance(data["date"], str) else data["date"]
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        payment_method = PaymentMethod(data["payment_method"]) if data.get("payment_method") else None
        
        return cls(
            id=data["id"],
            transaction_type=data["transaction_type"],
            amount=data["amount"],
            description=data["description"],
            category=data["category"],
            date=date,
            reference_id=data.get("reference_id"),
            payment_method=payment_method,
            tax_amount=data.get("tax_amount", 0.0),
            tax_code=data.get("tax_code", "GST"),
            is_tax_deductible=data.get("is_tax_deductible", False),
            notes=data.get("notes", ""),
            created_at=created_at,
            updated_at=updated_at
        )

@dataclass
class MarketingCampaign:
    """Marketing campaign information."""
    id: str
    name: str
    channel: SalesChannel
    start_date: datetime.date
    end_date: Optional[datetime.date] = None
    budget: float = 0.0
    spend: float = 0.0
    target_audience: List[CustomerSegment] = field(default_factory=list)
    goals: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "draft"  # draft, active, paused, completed
    notes: str = ""
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    
    @property
    def roi(self) -> float:
        """Calculate ROI (Return on Investment)."""
        if self.spend == 0:
            return 0.0
        revenue = self.metrics.get("revenue", 0.0)
        return (revenue - self.spend) / self.spend * 100.0 if self.spend > 0 else 0.0
    
    @property
    def conversion_rate(self) -> float:
        """Calculate conversion rate."""
        impressions = self.metrics.get("impressions", 0)
        conversions = self.metrics.get("conversions", 0)
        return (conversions / impressions * 100.0) if impressions > 0 else 0.0
    
    @property
    def cost_per_acquisition(self) -> float:
        """Calculate cost per acquisition (CPA)."""
        conversions = self.metrics.get("conversions", 0)
        return self.spend / conversions if conversions > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "channel": self.channel.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "budget": self.budget,
            "spend": self.spend,
            "target_audience": [segment.value for segment in self.target_audience],
            "goals": self.goals,
            "metrics": self.metrics,
            "status": self.status,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "roi": self.roi,
            "conversion_rate": self.conversion_rate,
            "cost_per_acquisition": self.cost_per_acquisition
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketingCampaign':
        """Create from dictionary."""
        start_date = datetime.date.fromisoformat(data["start_date"]) if isinstance(data["start_date"], str) else data["start_date"]
        end_date = datetime.date.fromisoformat(data["end_date"]) if data.get("end_date") and isinstance(data["end_date"], str) else data.get("end_date")
        created_at = datetime.datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        updated_at = datetime.datetime.fromisoformat(data["updated_at"]) if isinstance(data["updated_at"], str) else data["updated_at"]
        
        return cls(
            id=data["id"],
            name=data["name"],
            channel=SalesChannel(data["channel"]),
            start_date=start_date,
            end_date=end_date,
            budget=data.get("budget", 0.0),
            spend=data.get("spend", 0.0),
            target_audience=[CustomerSegment(segment) for segment in data.get("target_audience", [])],
            goals=data.get("goals", {}),
            metrics=data.get("metrics", {}),
            status=data.get("status", "draft"),
            notes=data.get("notes", ""),
            created_at=created_at,
            updated_at=updated_at
        )

class AnalyticsEngine:
    """Engine for business analytics and forecasting."""
    
    def __init__(self, db_manager=None):
        """Initialize the analytics engine."""
        self.db_manager = db_manager
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.reports = {}
    
    def load_data(self, data_type: str, start_date: Optional[datetime.date] = None, 
                 end_date: Optional[datetime.date] = None) -> pd.DataFrame:
        """Load data for analysis."""
        try:
            if self.db_manager:
                # Load from database
                query = f"SELECT * FROM {data_type}"
                params = []
                
                if start_date or end_date:
                    query += " WHERE "
                    if start_date:
                        query += "date >= ?"
                        params.append(start_date.isoformat())
                    if end_date:
                        if start_date:
                            query += " AND "
                        query += "date <= ?"
                        params.append(end_date.isoformat())
                
                df = self.db_manager.execute_query(query, params)
                return df
            else:
                # Load from CSV file
                file_path = DATA_DIR / f"{data_type}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    
                    # Convert date columns
                    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                    for col in date_columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            pass
                    
                    # Filter by date if needed
                    if start_date or end_date:
                        date_col = next((col for col in date_columns if 'date' in col.lower()), None)
                        if date_col:
                            if start_date:
                                df = df[df[date_col].dt.date >= start_date]
                            if end_date:
                                df = df[df[date_col].dt.date <= end_date]
                    
                    return df
                else:
                    logger.warning(f"Data file {file_path} not found")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, data_type: str) -> bool:
        """Save data to storage."""
        try:
            if self.db_manager:
                # Save to database
                return self.db_manager.insert_dataframe(data, data_type)
            else:
                # Save to CSV file
                file_path = DATA_DIR / f"{data_type}.csv"
                data.to_csv(file_path, index=False)
                return True
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False
    
    def train_model(self, model_type: AnalyticsModel, data: pd.DataFrame, 
                   target_column: str, feature_columns: List[str], 
                   model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train a predictive model."""
        try:
            # Prepare data
            X = data[feature_columns]
            y = data[target_column]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize model
            if model_type == AnalyticsModel.LINEAR_REGRESSION:
                model = LinearRegression(**(model_params or {}))
            elif model_type == AnalyticsModel.RANDOM_FOREST:
                model = RandomForestRegressor(**(model_params or {}))
            elif model_type == AnalyticsModel.GRADIENT_BOOSTING:
                model = GradientBoostingRegressor(**(model_params or {}))
            elif model_type == AnalyticsModel.ARIMA:
                # ARIMA requires time series data
                return self._train_arima_model(data, target_column, model_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_scaled, y)
            
            # Evaluate model
            y_pred = model.predict(X_scaled)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Store model
            model_id = str(uuid.uuid4())
            self.models[model_id] = {
                "model": model,
                "scaler": scaler,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "model_type": model_type,
                "metrics": {
                    "mse": mse,
                    "r2": r2
                },
                "created_at": datetime.datetime.now()
            }
            
            # Save model metadata
            model_metadata = {
                "id": model_id,
                "model_type": model_type.value,
                "feature_columns": feature_columns,
                "target_column": target_column,
                "metrics": {
                    "mse": mse,
                    "r2": r2
                },
                "created_at": datetime.datetime.now().isoformat()
            }
            
            # Return model info
            return model_metadata
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {"error": str(e)}
    
    def _train_arima_model(self, data: pd.DataFrame, target_column: str, 
                          model_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train an ARIMA time series model."""
        try:
            # Ensure data is sorted by date
            date_col = next((col for col in data.columns if 'date' in col.lower()), None)
            if date_col:
                data = data.sort_values(by=date_col)
            
            # Extract time series
            y = data[target_column]
            
            # Set default parameters if not provided
            if not model_params:
                model_params = {"order": (1, 1, 1)}
            
            # Train ARIMA model
            model = ARIMA(y, **model_params)
            model_fit = model.fit()
            
            # Evaluate model
            predictions = model_fit.predict()
            mse = mean_squared_error(y[1:], predictions[:-1])
            
            # Store model
            model_id = str(uuid.uuid4())
            self.models[model_id] = {
                "model": model_fit,
                "target_column": target_column,
                "model_type": AnalyticsModel.ARIMA,
                "metrics": {
                    "mse": mse
                },
                "created_at": datetime.datetime.now()
            }
            
            # Return model info
            return {
                "id": model_id,
                "model_type": AnalyticsModel.ARIMA.value,
                "target_column": target_column,
                "metrics": {
                    "mse": mse
                },
                "created_at": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {"error": str(e)}
    
    def predict(self, model_id: str, data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using a trained model."""
        try:
            model_info = self.models.get(model_id)
            if not model_info:
                raise ValueError(f"Model {model_id} not found")
            
            model = model_info["model"]
            model_type = model_info["model_type"]
            
            if model_type == AnalyticsModel.ARIMA:
                # ARIMA prediction
                forecast_steps = len(data) if not data.empty else 12  # Default to 12 periods
                forecast = model.forecast(steps=forecast_steps)
                return pd.DataFrame({
                    "forecast": forecast
                })
            else:
                # Regression model prediction
                feature_columns = model_info["feature_columns"]
                scaler = model_info["scaler"]
                
                # Prepare input data
                X = data[feature_columns]
                X = X.fillna(X.mean())
                X_scaled = scaler.transform(X)
                
                # Make predictions
                predictions = model.predict(X_scaled)
                
                # Add predictions to dataframe
                result = data.copy()
                result["prediction"] = predictions
                
                return result
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.DataFrame()
    
    def forecast_revenue(self, historical_data: pd.DataFrame, periods: int = 12, 
                        model_type: AnalyticsModel = AnalyticsModel.ARIMA) -> Dict[str, Any]:
        """Forecast revenue for future periods."""
        try:
            # Prepare time series data
            date_col = next((col for col in historical_data.columns if 'date' in col.lower()), None)
            revenue_col = next((col for col in historical_data.columns if 'revenue' in col.lower() or 'amount' in col.lower()), None)
            
            if not date_col or not revenue_col:
                raise ValueError("Data must contain date and revenue columns")
            
            # Ensure data is sorted by date
            historical_data = historical_data.sort_values(by=date_col)
            
            # Group by month if data is daily
            if historical_data[date_col].dtype == 'datetime64[ns]':
                historical_data['month'] = historical_data[date_col].dt.to_period('M')
                monthly_data = historical_data.groupby('month')[revenue_col].sum().reset_index()
                monthly_data[date_col] = monthly_data['month'].dt.to_timestamp()
                data = monthly_data[[date_col, revenue_col]]
            else:
                data = historical_data[[date_col, revenue_col]]
            
            # Train model
            if model_type == AnalyticsModel.ARIMA:
                # ARIMA model
                model = ARIMA(data[revenue_col], order=(1, 1, 1))
                model_fit = model.fit()
                
                # Generate forecast
                forecast = model_fit.forecast(steps=periods)
                forecast_dates = pd.date_range(
                    start=data[date_col].iloc[-1] + pd.Timedelta(days=30),
                    periods=periods,
                    freq='M'
                )
                
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'forecast': forecast
                })
                
                # Calculate confidence intervals
                forecast_df['lower_95'] = forecast - 1.96 * model_fit.params_obj['sigma2'] ** 0.5
                forecast_df['upper_95'] = forecast + 1.96 * model_fit.params_obj['sigma2'] ** 0.5
                
                # Store forecast
                forecast_id = str(uuid.uuid4())
                self.forecasts[forecast_id] = {
                    "data": forecast_df,
                    "model_type": model_type,
                    "periods": periods,
                    "created_at": datetime.datetime.now()
                }
                
                return {
                    "id": forecast_id,
                    "forecast": forecast_df.to_dict(orient="records"),
                    "model_type": model_type.value,
                    "periods": periods,
                    "total_forecast": float(forecast.sum()),
                    "average_forecast": float(forecast.mean()),
                    "created_at": datetime.datetime.now().isoformat()
                }
            else:
                # For other model types, we need to create features
                data = data.copy()
                data['month'] = pd.to_datetime(data[date_col]).dt.month
                data['year'] = pd.to_datetime(data[date_col]).dt.year
                data['trend'] = range(1, len(data) + 1)
                
                # Train model
                model_metadata = self.train_model(
                    model_type=model_type,
                    data=data,
                    target_column=revenue_col,
                    feature_columns=['month', 'year', 'trend'],
                    model_params=None
                )
                
                if "error" in model_metadata:
                    raise ValueError(f"Error training model: {model_metadata['error']}")
                
                # Generate future dates
                last_date = pd.to_datetime(data[date_col].iloc[-1])
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
                
                # Create future dataframe
                future_data = pd.DataFrame({
                    'date': future_dates,
                    'month': [d.month for d in future_dates],
                    'year': [d.year for d in future_dates],
                    'trend': range(len(data) + 1, len(data) + periods + 1)
                })
                
                # Make predictions
                predictions = self.predict(model_metadata["id"], future_data)
                
                # Store forecast
                forecast_id = str(uuid.uuid4())
                self.forecasts[forecast_id] = {
                    "data": predictions,
                    "model_type": model_type,
                    "periods": periods,
                    "created_at": datetime.datetime.now()
                }
                
                return {
                    "id": forecast_id,
                    "forecast": predictions.to_dict(orient="records"),
                    "model_type": model_type.value,
                    "periods": periods,
                    "total_forecast": float(predictions["prediction"].sum()),
                    "average_forecast": float(predictions["prediction"].mean()),
                    "created_at": datetime.datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error forecasting revenue: {e}")
            return {"error": str(e)}
    
    def segment_customers(self, customer_data: pd.DataFrame, n_clusters: int = 3) -> Dict[str, Any]:
        """Segment customers based on their behavior."""
        try:
            # Select relevant features
            features = []
            if 'lifetime_value' in customer_data.columns:
                features.append('lifetime_value')
            if 'total_spent' in customer_data.columns:
                features.append('total_spent')
            if 'total_purchases' in customer_data.columns:
                features.append('total_purchases')
            if 'acquisition_cost' in customer_data.columns:
                features.append('acquisition_cost')
            
            if not features:
                raise ValueError("Customer data must contain at least one of: lifetime_value, total_spent, total_purchases, acquisition_cost")
            
            # Prepare data
            X = customer_data[features]
            X = X.fillna(X.mean())
            
            # Normalize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to data
            result = customer_data.copy()
            result['cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = []
            for i in range(n_clusters):
                cluster_data = result[result['cluster'] == i]
                stats = {
                    "cluster_id": i,
                    "size": len(cluster_data),
                    "percentage": len(cluster_data) / len(result) * 100
                }
                
                # Add feature averages
                for feature in features:
                    stats[f"avg_{feature}"] = float(cluster_data[feature].mean())
                
                cluster_stats.append(stats)
            
            # Store segmentation
            segmentation_id = str(uuid.uuid4())
            self.metrics[segmentation_id] = {
                "type": "customer_segmentation",
                "data": result,
                "cluster_stats": cluster_stats,
                "n_clusters": n_clusters,
                "features": features,
                "created_at": datetime.datetime.now()
            }
            
            return {
                "id": segmentation_id,
                "clusters": cluster_stats,
                "features": features,
                "n_clusters": n_clusters,
                "created_at": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error segmenting customers: {e}")
            return {"error": str(e)}
    
    def calculate_business_metrics(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate key business metrics."""
        try:
            metrics = {}
            
            # Revenue metrics
            if "orders" in data:
                orders_df = data["orders"]
                
                # Total revenue
                if "total" in orders_df.columns:
                    metrics["total_revenue"] = float(orders_df["total"].sum())
                
                # Revenue by channel
                if "sales_channel" in orders_df.columns and "total" in orders_df.columns:
                    channel_revenue = orders_df.groupby("sales_channel")["total"].sum().to_dict()
                    metrics["revenue_by_channel"] = channel_revenue
                
                # Revenue by payment method
                if "payment_method" in orders_df.columns and "total" in orders_df.columns:
                    payment_revenue = orders_df.groupby("payment_method")["total"].sum().to_dict()
                    metrics["revenue_by_payment"] = payment_revenue
                
                # Average order value
                if "total" in orders_df.columns:
                    metrics["average_order_value"] = float(orders_df["total"].mean())
                
                # Total profit
                if "profit" in orders_df.columns:
                    metrics["total_profit"] = float(orders_df["profit"].sum())
                
                # Average profit margin
                if "margin" in orders_df.columns:
                    metrics["average_margin"] = float(orders_df["margin"].mean())
            
            # Customer metrics
            if "customers" in data:
                customers_df = data["customers"]
                
                # Total customers
                metrics["total_customers"] = len(customers_df)
                
                # Customers by segment
                if "segment" in customers_df.columns:
                    segment_counts = customers_df["segment"].value_counts().to_dict()
                    metrics["customers_by_segment"] = segment_counts
                
                # Average lifetime value
                if "lifetime_value" in customers_df.columns:
                    metrics["average_ltv"] = float(customers_df["lifetime_value"].mean())
                
                # Average acquisition cost
                if "acquisition_cost" in customers_df.columns:
                    metrics["average_cac"] = float(customers_df["acquisition_cost"].mean())
                
                # LTV:CAC ratio
                if "lifetime_value" in customers_df.columns and "acquisition_cost" in customers_df.columns:
                    valid_customers = customers_df[customers_df["acquisition_cost"] > 0]
                    if len(valid_customers) > 0:
                        ltv_cac_ratio = valid_customers["lifetime_value"] / valid_customers["acquisition_cost"]
                        metrics["ltv_cac_ratio"] = float(ltv_cac_ratio.mean())
            
            # Product metrics
            if "products" in data:
                products_df = data["products"]
                
                # Total products
                metrics["total_products"] = len(products_df)
                
                # Average product margin
                if "margin" in products_df.columns:
                    metrics["average_product_margin"] = float(products_df["margin"].mean())
                
                # Products by category
                if "category" in products_df.columns:
                    category_counts = products_df["category"].value_counts().to_dict()
                    metrics["products_by_category"] = category_counts
            
            # Inventory metrics
            if "inventory_transactions" in data and "products" in data:
                inventory_df = data["inventory_transactions"]
                products_df = data["products"]
                
                # Total inventory value
                if "stock_level" in products_df.columns and "cost" in products_df.columns:
                    inventory_value = (products_df["stock_level"] * products_df["cost"]).sum()
                    metrics["total_inventory_value"] = float(inventory_value)
                
                # Products below reorder point
                if "stock_level" in products_df.columns and "reorder_point" in products_df.columns:
                    below_reorder = products_df[products_df["stock_level"] < products_df["reorder_point"]]
                    metrics["products_below_reorder"] = len(below_reorder)
            
            # Marketing metrics
            if "marketing_campaigns" in data:
                campaigns_df = data["marketing_campaigns"]
                
                # Total marketing spend
                if "spend" in campaigns_df.columns:
                    metrics["total_marketing_spend"] = float(campaigns_df["spend"].sum())
                
                # Average ROI
                if "roi" in campaigns_df.columns:
                    metrics["average_marketing_roi"] = float(campaigns_df["roi"].mean())
                
                # Average conversion rate
                if "conversion_rate" in campaigns_df.columns:
                    metrics["average_conversion_rate"] = float(campaigns_df["conversion_rate"].mean())
            
            # Store metrics
            metrics_id = str(uuid.uuid4())
            self.metrics[metrics_id] = {
                "type": "business_metrics",
                "data": metrics,
                "created_at": datetime.datetime.now()
            }
            
            return {
                "id": metrics_id,
                "metrics": metrics,
                "created_at": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating business metrics: {e}")
            return {"error": str(e)}
    
    def generate_report(self, report_type: str, data: Dict[str, Any], 
                       format: str = "json", output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Generate a business report."""
        try:
            report = {
                "type": report_type,
                "generated_at": datetime.datetime.now().isoformat(),
                "data": data
            }
            
            # Generate report ID
            report_id = str(uuid.uuid4())
            
            # Store report
            self.reports[report_id] = {
                "type": report_type,
                "data": data,
                "format": format,
                "created_at": datetime.datetime.now()
            }
            
            # Export report if path provided
            if output_path:
                if format == "json":
                    with open(output_path, 'w') as f:
                        json.dump(report, f, indent=2)
                elif format == "csv":
                    if isinstance(data, pd.DataFrame):
                        data.to_csv(output_path, index=False)
                    else:
                        pd.DataFrame(data).to_csv(output_path, index=False)
                elif format == "html":
                    # Generate HTML report
                    html_content = self._generate_html_report(report_type, data)
                    with open(output_path, 'w') as f:
                        f.write(html_content)
            
            return {
                "id": report_id,
                "type": report_type,
                "format": format,
                "output_path": str(output_path) if output_path else None,
                "created_at": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def _generate_html_report(self, report_type: str, data: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        try:
            # Basic HTML template
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report_type.title()} Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f8f9fa; 
                              border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 200px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
                    .metric-label {{ font-size: 14px; color: #7f8c8d; }}
                    .chart-container {{ margin: 20px 0; height: 400px; }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>{report_type.title()} Report</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            # Add report content based on type
            if report_type == "business_metrics":
                html += self._generate_metrics_html(data)
            elif report_type == "revenue_forecast":
                html += self._generate_forecast_html(data)
            elif report_type == "customer_segmentation":
                html += self._generate_segmentation_html(data)
            else:
                # Generic data display
                html += "<h2>Report Data</h2>"
                html += "<pre>" + json.dumps(data, indent=2) + "</pre>"
            
            # Close HTML
            html += """
            </body>
            </html>
            """
            
            return html
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return f"<html><body><h1>Error Generating Report</h1><p>{str(e)}</p></body></html>"
    
    def _generate_metrics_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for metrics report."""
        html = "<h2>Key Business Metrics</h2><div>"
        
        # Display key metrics as cards
        metrics = data.get("metrics", {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                formatted_value = f"${value:,.2f}" if "revenue" in key or "profit" in key or "value" in key or "cost" in key or "spend" in key else f"{value:,.2f}"
                html += f"""
                <div class="metric">
                    <div class="metric-value">{formatted_value}</div>
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                </div>
                """
        
        html += "</div>"
        
        # Display breakdown tables
        for key, value in metrics.items():
            if isinstance(value, dict) and key.startswith("revenue_by") or key.endswith("_by_segment") or key.endswith("_by_category"):
                html += f"<h3>{key.replace('_', ' ').title()}</h3>"
                html += "<table><tr><th>Category</th><th>Value</th></tr>"
                
                for category, amount in value.items():
                    formatted_amount = f"${amount:,.2f}" if "revenue" in key else f"{amount:,.2f}"
                    html += f"<tr><td>{category}</td><td>{formatted_amount}</td></tr>"
                
                html += "</table>"
        
        return html
    
    def _generate_forecast_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for forecast report."""
        html = "<h2>Revenue Forecast</h2>"
        
        # Summary metrics
        if "total_forecast" in data:
            html += f"""
            <div class="metric">
                <div class="metric-value">${data['total_forecast']:,.2f}</div>
                <div class="metric-label">Total Forecast Revenue</div>
            </div>
            <div class="metric">
                <div class="metric-value">${data['average_forecast']:,.2f}</div>
                <div class="metric-label">Average Monthly Forecast</div>
            </div>
            """
        
        # Forecast chart
        if "forecast" in data and isinstance(data["forecast"], list) and len(data["forecast"]) > 0:
            forecast_data = data["forecast"]
            dates = [item.get("date") for item in forecast_data]
            values = [item.get("forecast") for item in forecast_data]
            lower_bounds = [item.get("lower_95") for item in forecast_data if "lower_95" in item]
            upper_bounds = [item.get("upper_95") for item in forecast_data if "upper_95" in item]
            
            html += """
            <div class="chart-container" id="forecast-chart"></div>
            <script>
                var dates = """ + json.dumps(dates) + """;
                var values = """ + json.dumps(values) + """;
            """
            
            if lower_bounds and upper_bounds:
                html += """
                var lower_bounds = """ + json.dumps(lower_bounds) + """;
                var upper_bounds = """ + json.dumps(upper_bounds) + """;
                
                var trace1 = {
                    x: dates,
                    y: values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Forecast',
                    line: {color: '#3498db'}
                };
                
                var trace2 = {
                    x: dates.concat(dates.slice().reverse()),
                    y: upper_bounds.concat(lower_bounds.slice().reverse()),
                    fill: 'toself',
                    fillcolor: 'rgba(52, 152, 219, 0.2)',
                    line: {color: 'transparent'},
                    name: '95% Confidence',
                    showlegend: false
                };
                
                var data = [trace2, trace1];
                """
            else:
                html += """
                var trace1 = {
                    x: dates,
                    y: values,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Forecast',
                    line: {color: '#3498db'}
                };
                
                var data = [trace1];
                """
            
            html += """
                var layout = {
                    title: 'Revenue Forecast',
                    xaxis: {title: 'Date'},
                    yaxis: {title: 'Revenue ($)'},
                    margin: {l: 60, r: 40, t: 40, b: 60}
                };
                
                Plotly.newPlot('forecast-chart', data, layout);
            </script>
            """
        
        return html
    
    def _generate_segmentation_html(self, data: Dict[str, Any]) -> str:
        """Generate HTML for customer segmentation report."""
        html = "<h2>Customer Segmentation Analysis</h2>"
        
        # Cluster information
        if "clusters" in data and isinstance(data["clusters"], list):
            html += "<h3>Customer Segments</h3>"
            html += "<table><tr><th>Segment</th><th>Size</th><th>Percentage</th>"
            
            # Add feature columns
            features = data.get("features", [])
            for feature in features:
                html += f"<th>Avg. {feature.replace('_', ' ').title()}</th>"
            
            html += "</tr>"
            
            # Add cluster data
            for cluster in data["clusters"]:
                html += f"<tr><td>Segment {cluster['cluster_id'] + 1}</td><td>{cluster['size']}</td><td>{cluster['percentage']:.1f}%</td>"
                
                for feature in features:
                    feature_key = f"avg_{feature}"
                    if feature_key in cluster:
                        value = cluster[feature_key]
                        formatted_value = f"${value:,.2f}" if "value" in feature or "spent" in feature or "cost" in feature else f"{value:,.2f}"
                        html += f"<td>{formatted_value}</td>"
                    else:
                        html += "<td>N/A</td>"
                
                html += "</tr>"
            
            html += "</table>"
            
            # Add visualization
            if len(data["clusters"]) > 0 and len(features) >= 2:
                html += """
                <div class="chart-container" id="cluster-chart"></div>
                <script>
                    var clusters = """ + json.dumps(data["clusters"]) + """;
                    var features = """ + json.dumps(features) + """;
                    
                    // Prepare data for each cluster
                    var traces = [];
                    for (var i = 0; i < clusters.length; i++) {
                        var cluster = clusters[i];
                        traces.push({
                            x: [cluster['avg_' + features[0]]],
                            y: [cluster['avg_' + features[1]]],
                            mode: 'markers',
                            marker: {
                                size: Math.sqrt(cluster['size']) * 5,
                                sizemode: 'area',
                                sizeref: 0.1
                            },
                            name: 'Segment ' + (cluster['cluster_id'] + 1) + ' (' + cluster['percentage'].toFixed(1) + '%)',
                            text: ['Size: ' + cluster['size']]
                        });
                    }
                    
                    var layout = {
                        title: 'Customer Segments',
                        xaxis: {title: features[0].replace('_', ' ').charAt(0).toUpperCase() + features[0].replace('_', ' ').slice(1)},
                        yaxis: {title: features[1].replace('_', ' ').charAt(0).toUpperCase() + features[1].replace('_', ' 
