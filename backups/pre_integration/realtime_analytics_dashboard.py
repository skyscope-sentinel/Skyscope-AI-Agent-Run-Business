import os
import sys
import json
import time
import uuid
import logging
import datetime
import threading
import traceback
import asyncio
import queue
import socket
import signal
import math
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Generator
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod
from collections import deque, defaultdict
import random

# Data processing and analysis
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import networkx as nx

# Dashboard and UI
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from streamlit_plotly_events import plotly_events
import altair as alt

# Web and networking
import requests
import websocket
import socketio

# Import internal modules
try:
    from agent_manager import AgentManager
    from business_manager import BusinessManager
    from crypto_manager import CryptoManager
    from database_manager import DatabaseManager
    from performance_monitor import PerformanceMonitor
    from live_thinking_rag_system import LiveThinkingRAGSystem
    from advanced_business_operations import AnalyticsEngine
    from enhanced_security_compliance import AuditLogger
except ImportError:
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/analytics_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("realtime_analytics_dashboard")

# Constants
CONFIG_DIR = Path("config")
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
REPORTS_DIR = Path("reports")
DASHBOARD_DIR = Path("dashboard")
TEMP_DIR = Path("temp")
MODELS_DIR = Path("models")

# Ensure directories exist
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dashboard configuration
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dashboard_config.json"
DEFAULT_REFRESH_RATE = 5  # seconds
DEFAULT_HISTORY_LENGTH = 1000  # data points
DEFAULT_PREDICTION_HORIZON = 24  # hours
DEFAULT_ANOMALY_THRESHOLD = 3.0  # standard deviations
DEFAULT_VISUALIZATION_LIMIT = 100  # max points to visualize

class DashboardMode(Enum):
    """Dashboard viewing modes."""
    OVERVIEW = "overview"
    AGENT_METRICS = "agent_metrics"
    BUSINESS_METRICS = "business_metrics"
    SYSTEM_PERFORMANCE = "system_performance"
    SECURITY_MONITORING = "security_monitoring"
    PREDICTIVE_ANALYTICS = "predictive_analytics"
    NETWORK_ANALYSIS = "network_analysis"
    CUSTOM = "custom"

class ChartType(Enum):
    """Chart types for visualizations."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"
    AREA = "area"
    CANDLESTICK = "candlestick"
    NETWORK = "network"
    MAP = "map"
    TABLE = "table"
    GAUGE = "gauge"
    CUSTOM = "custom"

class DataSource(Enum):
    """Data sources for the dashboard."""
    AGENT_TELEMETRY = "agent_telemetry"
    BUSINESS_METRICS = "business_metrics"
    SYSTEM_METRICS = "system_metrics"
    DATABASE_METRICS = "database_metrics"
    SECURITY_LOGS = "security_logs"
    API_METRICS = "api_metrics"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert levels for notifications."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class DashboardConfig:
    """Configuration for the analytics dashboard."""
    refresh_rate: int = DEFAULT_REFRESH_RATE
    history_length: int = DEFAULT_HISTORY_LENGTH
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON
    anomaly_threshold: float = DEFAULT_ANOMALY_THRESHOLD
    visualization_limit: int = DEFAULT_VISUALIZATION_LIMIT
    default_mode: DashboardMode = DashboardMode.OVERVIEW
    enabled_data_sources: List[DataSource] = field(default_factory=lambda: list(DataSource))
    chart_preferences: Dict[str, ChartType] = field(default_factory=dict)
    custom_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    layout_config: Dict[str, Any] = field(default_factory=dict)
    theme: str = "dark"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "refresh_rate": self.refresh_rate,
            "history_length": self.history_length,
            "prediction_horizon": self.prediction_horizon,
            "anomaly_threshold": self.anomaly_threshold,
            "visualization_limit": self.visualization_limit,
            "default_mode": self.default_mode.value,
            "enabled_data_sources": [source.value for source in self.enabled_data_sources],
            "chart_preferences": {k: v.value for k, v in self.chart_preferences.items()},
            "custom_metrics": self.custom_metrics,
            "alert_thresholds": self.alert_thresholds,
            "layout_config": self.layout_config,
            "theme": self.theme
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DashboardConfig':
        """Create from dictionary."""
        return cls(
            refresh_rate=data.get("refresh_rate", DEFAULT_REFRESH_RATE),
            history_length=data.get("history_length", DEFAULT_HISTORY_LENGTH),
            prediction_horizon=data.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON),
            anomaly_threshold=data.get("anomaly_threshold", DEFAULT_ANOMALY_THRESHOLD),
            visualization_limit=data.get("visualization_limit", DEFAULT_VISUALIZATION_LIMIT),
            default_mode=DashboardMode(data.get("default_mode", DashboardMode.OVERVIEW.value)),
            enabled_data_sources=[DataSource(source) for source in data.get("enabled_data_sources", [ds.value for ds in DataSource])],
            chart_preferences={k: ChartType(v) for k, v in data.get("chart_preferences", {}).items()},
            custom_metrics=data.get("custom_metrics", {}),
            alert_thresholds=data.get("alert_thresholds", {}),
            layout_config=data.get("layout_config", {}),
            theme=data.get("theme", "dark")
        )
    
    def save(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Dashboard configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving dashboard configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = DEFAULT_CONFIG_PATH) -> 'DashboardConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Dashboard configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading dashboard configuration: {e}")
            return cls()

@dataclass
class MetricDefinition:
    """Definition of a metric to be tracked and visualized."""
    id: str
    name: str
    description: str
    data_source: DataSource
    unit: str = ""
    aggregation: str = "mean"  # mean, sum, min, max, count
    chart_type: ChartType = ChartType.LINE
    color: str = "#3366CC"
    alert_thresholds: Dict[AlertLevel, float] = field(default_factory=dict)
    historical_data: List[Tuple[datetime.datetime, float]] = field(default_factory=list)
    forecast_data: List[Tuple[datetime.datetime, float]] = field(default_factory=list)
    anomaly_data: List[Tuple[datetime.datetime, float, float]] = field(default_factory=list)
    related_metrics: List[str] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "data_source": self.data_source.value,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "chart_type": self.chart_type.value,
            "color": self.color,
            "alert_thresholds": {level.value: value for level, value in self.alert_thresholds.items()},
            "related_metrics": self.related_metrics,
            "custom_properties": self.custom_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricDefinition':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            data_source=DataSource(data["data_source"]),
            unit=data.get("unit", ""),
            aggregation=data.get("aggregation", "mean"),
            chart_type=ChartType(data.get("chart_type", ChartType.LINE.value)),
            color=data.get("color", "#3366CC"),
            alert_thresholds={AlertLevel(level): value for level, value in data.get("alert_thresholds", {}).items()},
            related_metrics=data.get("related_metrics", []),
            custom_properties=data.get("custom_properties", {})
        )

@dataclass
class Alert:
    """Alert notification."""
    id: str
    timestamp: datetime.datetime
    level: AlertLevel
    metric_id: str
    metric_name: str
    value: float
    threshold: float
    message: str
    is_acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime.datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "metric_id": self.metric_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "is_acknowledged": self.is_acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary."""
        timestamp = datetime.datetime.fromisoformat(data["timestamp"])
        acknowledged_at = datetime.datetime.fromisoformat(data["acknowledged_at"]) if data.get("acknowledged_at") else None
        
        return cls(
            id=data["id"],
            timestamp=timestamp,
            level=AlertLevel(data["level"]),
            metric_id=data["metric_id"],
            metric_name=data["metric_name"],
            value=data["value"],
            threshold=data["threshold"],
            message=data["message"],
            is_acknowledged=data.get("is_acknowledged", False),
            acknowledged_by=data.get("acknowledged_by"),
            acknowledged_at=acknowledged_at
        )

class DataProcessor:
    """Processor for streaming and historical data."""
    
    def __init__(self, config: DashboardConfig):
        """Initialize the data processor."""
        self.config = config
        self.metrics: Dict[str, MetricDefinition] = {}
        self.data_streams: Dict[str, deque] = {}
        self.alerts: List[Alert] = []
        self.anomaly_detectors: Dict[str, Any] = {}
        self.forecasting_models: Dict[str, Any] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_update = datetime.datetime.now()
        self._load_metrics()
        self._init_data_streams()
    
    def _load_metrics(self) -> None:
        """Load metric definitions."""
        metrics_path = CONFIG_DIR / "dashboard_metrics.json"
        try:
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                
                for metric_data in metrics_data:
                    metric = MetricDefinition.from_dict(metric_data)
                    self.metrics[metric.id] = metric
                
                logger.info(f"Loaded {len(self.metrics)} metric definitions")
            else:
                logger.info("No metric definitions found, using defaults")
                self._create_default_metrics()
        except Exception as e:
            logger.error(f"Error loading metric definitions: {e}")
            self._create_default_metrics()
    
    def _create_default_metrics(self) -> None:
        """Create default metric definitions."""
        # Agent metrics
        self.metrics["agent_count"] = MetricDefinition(
            id="agent_count",
            name="Active Agents",
            description="Number of active agents in the system",
            data_source=DataSource.AGENT_TELEMETRY,
            unit="agents",
            aggregation="count",
            chart_type=ChartType.LINE
        )
        
        self.metrics["agent_tasks_completed"] = MetricDefinition(
            id="agent_tasks_completed",
            name="Tasks Completed",
            description="Number of tasks completed by agents",
            data_source=DataSource.AGENT_TELEMETRY,
            unit="tasks",
            aggregation="sum",
            chart_type=ChartType.LINE
        )
        
        self.metrics["agent_success_rate"] = MetricDefinition(
            id="agent_success_rate",
            name="Agent Success Rate",
            description="Percentage of tasks successfully completed",
            data_source=DataSource.AGENT_TELEMETRY,
            unit="%",
            aggregation="mean",
            chart_type=ChartType.LINE
        )
        
        # Business metrics
        self.metrics["revenue"] = MetricDefinition(
            id="revenue",
            name="Revenue",
            description="Total revenue generated",
            data_source=DataSource.BUSINESS_METRICS,
            unit="$",
            aggregation="sum",
            chart_type=ChartType.LINE
        )
        
        self.metrics["profit"] = MetricDefinition(
            id="profit",
            name="Profit",
            description="Total profit",
            data_source=DataSource.BUSINESS_METRICS,
            unit="$",
            aggregation="sum",
            chart_type=ChartType.LINE
        )
        
        self.metrics["roi"] = MetricDefinition(
            id="roi",
            name="ROI",
            description="Return on Investment",
            data_source=DataSource.BUSINESS_METRICS,
            unit="%",
            aggregation="mean",
            chart_type=ChartType.LINE
        )
        
        # System metrics
        self.metrics["cpu_usage"] = MetricDefinition(
            id="cpu_usage",
            name="CPU Usage",
            description="System CPU usage",
            data_source=DataSource.SYSTEM_METRICS,
            unit="%",
            aggregation="mean",
            chart_type=ChartType.LINE
        )
        
        self.metrics["memory_usage"] = MetricDefinition(
            id="memory_usage",
            name="Memory Usage",
            description="System memory usage",
            data_source=DataSource.SYSTEM_METRICS,
            unit="%",
            aggregation="mean",
            chart_type=ChartType.LINE
        )
        
        self.metrics["api_latency"] = MetricDefinition(
            id="api_latency",
            name="API Latency",
            description="Average API response time",
            data_source=DataSource.API_METRICS,
            unit="ms",
            aggregation="mean",
            chart_type=ChartType.LINE
        )
        
        # Database metrics
        self.metrics["db_queries"] = MetricDefinition(
            id="db_queries",
            name="Database Queries",
            description="Number of database queries",
            data_source=DataSource.DATABASE_METRICS,
            unit="queries",
            aggregation="sum",
            chart_type=ChartType.LINE
        )
        
        self.metrics["db_latency"] = MetricDefinition(
            id="db_latency",
            name="Database Latency",
            description="Average database query latency",
            data_source=DataSource.DATABASE_METRICS,
            unit="ms",
            aggregation="mean",
            chart_type=ChartType.LINE
        )
        
        # Security metrics
        self.metrics["security_events"] = MetricDefinition(
            id="security_events",
            name="Security Events",
            description="Number of security events detected",
            data_source=DataSource.SECURITY_LOGS,
            unit="events",
            aggregation="sum",
            chart_type=ChartType.LINE
        )
        
        logger.info(f"Created {len(self.metrics)} default metric definitions")
    
    def _init_data_streams(self) -> None:
        """Initialize data streams for each metric."""
        for metric_id in self.metrics:
            self.data_streams[metric_id] = deque(maxlen=self.config.history_length)
    
    def add_data_point(self, metric_id: str, value: float, timestamp: Optional[datetime.datetime] = None) -> None:
        """Add a data point to a metric's stream."""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        if metric_id in self.data_streams:
            self.data_streams[metric_id].append((timestamp, value))
            
            # Check for alerts
            self._check_alerts(metric_id, value)
            
            # Update last update time
            self.last_update = timestamp
        else:
            logger.warning(f"Unknown metric ID: {metric_id}")
    
    def _check_alerts(self, metric_id: str, value: float) -> None:
        """Check if a metric value triggers any alerts."""
        if metric_id not in self.metrics:
            return
        
        metric = self.metrics[metric_id]
        
        for level, threshold in metric.alert_thresholds.items():
            if (level == AlertLevel.WARNING and value >= threshold) or \
               (level == AlertLevel.ERROR and value >= threshold) or \
               (level == AlertLevel.CRITICAL and value >= threshold):
                
                # Create alert
                alert = Alert(
                    id=str(uuid.uuid4()),
                    timestamp=datetime.datetime.now(),
                    level=level,
                    metric_id=metric_id,
                    metric_name=metric.name,
                    value=value,
                    threshold=threshold,
                    message=f"{metric.name} has reached {value} {metric.unit}, threshold: {threshold} {metric.unit}"
                )
                
                self.alerts.append(alert)
                logger.warning(f"Alert triggered: {alert.message}")
    
    def get_metric_data(self, metric_id: str, limit: Optional[int] = None) -> List[Tuple[datetime.datetime, float]]:
        """Get historical data for a metric."""
        if metric_id not in self.data_streams:
            return []
        
        data = list(self.data_streams[metric_id])
        
        if limit is not None and limit > 0:
            return data[-limit:]
        
        return data
    
    def get_metrics_dataframe(self, metric_ids: List[str], limit: Optional[int] = None) -> pd.DataFrame:
        """Get a DataFrame containing multiple metrics."""
        # Get data for each metric
        data_dict = {}
        timestamps = set()
        
        for metric_id in metric_ids:
            if metric_id in self.data_streams:
                metric_data = self.get_metric_data(metric_id, limit)
                data_dict[metric_id] = {ts: val for ts, val in metric_data}
                timestamps.update(ts for ts, _ in metric_data)
        
        # Create DataFrame with timestamps as index
        df = pd.DataFrame(index=sorted(timestamps))
        
        # Add each metric as a column
        for metric_id, data in data_dict.items():
            df[metric_id] = df.index.map(lambda ts: data.get(ts, None))
        
        return df
    
    def detect_anomalies(self, metric_id: str, window_size: int = 20) -> List[Tuple[datetime.datetime, float, float]]:
        """Detect anomalies in a metric's data stream."""
        if metric_id not in self.data_streams or len(self.data_streams[metric_id]) < window_size:
            return []
        
        # Get the data
        data = self.get_metric_data(metric_id)
        timestamps, values = zip(*data)
        
        # Convert to numpy array
        values_array = np.array(values)
        
        # Use rolling window statistics for anomaly detection
        anomalies = []
        
        for i in range(window_size, len(values_array)):
            window = values_array[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            # Z-score
            if std > 0:
                z_score = (values_array[i] - mean) / std
                
                if abs(z_score) > self.config.anomaly_threshold:
                    anomalies.append((timestamps[i], values_array[i], z_score))
        
        return anomalies
    
    def train_anomaly_detector(self, metric_id: str) -> bool:
        """Train an anomaly detection model for a metric."""
        if metric_id not in self.data_streams or len(self.data_streams[metric_id]) < 50:
            return False
        
        try:
            # Get the data
            data = self.get_metric_data(metric_id)
            _, values = zip(*data)
            
            # Reshape for sklearn
            X = np.array(values).reshape(-1, 1)
            
            # Train isolation forest
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X)
            
            # Store the model
            self.anomaly_detectors[metric_id] = model
            
            return True
        except Exception as e:
            logger.error(f"Error training anomaly detector for {metric_id}: {e}")
            return False
    
    def predict_anomalies(self, metric_id: str) -> List[Tuple[datetime.datetime, float, bool]]:
        """Predict anomalies using a trained model."""
        if metric_id not in self.anomaly_detectors or metric_id not in self.data_streams:
            return []
        
        try:
            # Get the data
            data = self.get_metric_data(metric_id)
            timestamps, values = zip(*data)
            
            # Reshape for sklearn
            X = np.array(values).reshape(-1, 1)
            
            # Predict anomalies
            model = self.anomaly_detectors[metric_id]
            predictions = model.predict(X)
            
            # Convert predictions to list of tuples
            # -1 indicates an anomaly, 1 indicates normal
            anomalies = [(timestamps[i], values[i], predictions[i] == -1) for i in range(len(predictions))]
            
            return anomalies
        except Exception as e:
            logger.error(f"Error predicting anomalies for {metric_id}: {e}")
            return []
    
    def train_forecasting_model(self, metric_id: str, model_type: str = "arima") -> bool:
        """Train a forecasting model for a metric."""
        if metric_id not in self.data_streams or len(self.data_streams[metric_id]) < 30:
            return False
        
        try:
            # Get the data
            data = self.get_metric_data(metric_id)
            _, values = zip(*data)
            
            if model_type == "arima":
                # Train ARIMA model
                model = ARIMA(values, order=(5, 1, 0))
                model_fit = model.fit()
                
                # Store the model
                self.forecasting_models[metric_id] = {
                    "type": "arima",
                    "model": model_fit,
                    "last_value": values[-1]
                }
                
            elif model_type == "exponential_smoothing":
                # Train exponential smoothing model
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=24
                )
                model_fit = model.fit()
                
                # Store the model
                self.forecasting_models[metric_id] = {
                    "type": "exponential_smoothing",
                    "model": model_fit,
                    "last_value": values[-1]
                }
                
            elif model_type == "linear_regression":
                # Create features (just using index for now)
                X = np.arange(len(values)).reshape(-1, 1)
                y = np.array(values)
                
                # Train linear regression model
                model = LinearRegression()
                model.fit(X, y)
                
                # Store the model
                self.forecasting_models[metric_id] = {
                    "type": "linear_regression",
                    "model": model,
                    "last_index": len(values) - 1,
                    "last_value": values[-1]
                }
            
            return True
        except Exception as e:
            logger.error(f"Error training forecasting model for {metric_id}: {e}")
            return False
    
    def forecast_metric(self, metric_id: str, steps: int = 24) -> List[float]:
        """Forecast future values for a metric."""
        if metric_id not in self.forecasting_models:
            return []
        
        try:
            model_info = self.forecasting_models[metric_id]
            model_type = model_info["type"]
            model = model_info["model"]
            
            if model_type == "arima":
                # Forecast using ARIMA
                forecast = model.forecast(steps=steps)
                return forecast.tolist()
                
            elif model_type == "exponential_smoothing":
                # Forecast using exponential smoothing
                forecast = model.forecast(steps)
                return forecast.tolist()
                
            elif model_type == "linear_regression":
                # Forecast using linear regression
                last_index = model_info["last_index"]
                X_future = np.arange(last_index + 1, last_index + steps + 1).reshape(-1, 1)
                forecast = model.predict(X_future)
                return forecast.tolist()
            
            return []
        except Exception as e:
            logger.error(f"Error forecasting for {metric_id}: {e}")
            return []
    
    def calculate_correlations(self, metric_ids: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix between metrics."""
        if not metric_ids or len(metric_ids) < 2:
            return pd.DataFrame()
        
        try:
            # Get data for all metrics
            df = self.get_metrics_dataframe(metric_ids)
            
            # Drop rows with missing values
            df = df.dropna()
            
            if df.empty:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = df.corr()
            
            # Store the matrix
            self.correlation_matrix = corr_matrix
            
            return corr_matrix
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return pd.DataFrame()
    
    def simulate_data(self, duration_seconds: int = 60, interval_seconds: float = 1.0) -> None:
        """Simulate data for testing purposes."""
        start_time = datetime.datetime.now()
        current_time = start_time
        end_time = start_time + datetime.timedelta(seconds=duration_seconds)
        
        # Base values for each metric
        base_values = {
            "agent_count": 8500,
            "agent_tasks_completed": 0,
            "agent_success_rate": 95.0,
            "revenue": 0.0,
            "profit": 0.0,
            "roi": 15.0,
            "cpu_usage": 45.0,
            "memory_usage": 60.0,
            "api_latency": 120.0,
            "db_queries": 0,
            "db_latency": 25.0,
            "security_events": 0
        }
        
        # Trends and patterns
        trends = {
            "agent_count": lambda t: 5 * math.sin(t / 10) + random.normalvariate(0, 2),
            "agent_tasks_completed": lambda t: 50 + 10 * math.sin(t / 5) + random.normalvariate(0, 5),
            "agent_success_rate": lambda t: -0.5 * math.sin(t / 15) + random.normalvariate(0, 0.2),
            "revenue": lambda t: 100 + 20 * math.sin(t / 20) + random.normalvariate(0, 5),
            "profit": lambda t: 40 + 10 * math.sin(t / 20) + random.normalvariate(0, 2),
            "roi": lambda t: 0.2 * math.sin(t / 30) + random.normalvariate(0, 0.1),
            "cpu_usage": lambda t: 5 * math.sin(t / 8) + random.normalvariate(0, 1),
            "memory_usage": lambda t: 2 * math.sin(t / 12) + random.normalvariate(0, 0.5),
            "api_latency": lambda t: 10 * math.sin(t / 7) + random.normalvariate(0, 2),
            "db_queries": lambda t: 20 + 5 * math.sin(t / 6) + random.normalvariate(0, 2),
            "db_latency": lambda t: 2 * math.sin(t / 9) + random.normalvariate(0, 0.5),
            "security_events": lambda t: 0.5 + 0.5 * math.sin(t / 25) + random.normalvariate(0, 0.1)
        }
        
        # Simulate data points
        while current_time < end_time:
            elapsed = (current_time - start_time).total_seconds()
            
            for metric_id, base_value in base_values.items():
                trend_func = trends.get(metric_id, lambda t: random.normalvariate(0, 1))
                
                # Calculate new value
                if metric_id == "agent_tasks_completed" or metric_id == "db_queries" or metric_id == "security_events":
                    # These are cumulative metrics
                    increment = max(0, trend_func(elapsed))
                    new_value = base_values[metric_id] + increment
                    base_values[metric_id] = new_value
                else:
                    # These are gauge metrics
                    new_value = base_value + trend_func(elapsed)
                    
                    # Ensure values stay in reasonable ranges
                    if metric_id == "agent_success_rate":
                        new_value = max(80, min(100, new_value))
                    elif metric_id == "cpu_usage" or metric_id == "memory_usage":
                        new_value = max(0, min(100, new_value))
                    elif metric_id == "roi":
                        new_value = max(0, new_value)
                    elif metric_id == "api_latency" or metric_id == "db_latency":
                        new_value = max(1, new_value)
                
                # Add the data point
                self.add_data_point(metric_id, new_value, current_time)
            
            # Increment time
            current_time += datetime.timedelta(seconds=interval_seconds)
            time.sleep(interval_seconds)
        
        logger.info(f"Simulated {duration_seconds} seconds of data at {interval_seconds}s intervals")

class VisualizationEngine:
    """Engine for creating visualizations."""
    
    def __init__(self, data_processor: DataProcessor, config: DashboardConfig):
        """Initialize the visualization engine."""
        self.data_processor = data_processor
        self.config = config
        self.color_palette = px.colors.qualitative.Plotly
    
    def create_line_chart(self, metric_id: str, title: Optional[str] = None, height: int = 400) -> go.Figure:
        """Create a line chart for a metric."""
        # Get the metric definition
        metric = self.data_processor.metrics.get(metric_id)
        if not metric:
            return go.Figure()
        
        # Get the data
        data = self.data_processor.get_metric_data(metric_id, self.config.visualization_limit)
        if not data:
            return go.Figure()
        
        # Split into timestamps and values
        timestamps, values = zip(*data)
        
        # Create the figure
        fig = go.Figure()
        
        # Add the line trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name=metric.name,
            line=dict(color=metric.color, width=2)
        ))
        
        # Add forecasted data if available
        if metric_id in self.data_processor.forecasting_models:
            forecast_values = self.data_processor.forecast_metric(metric_id, 24)
            if forecast_values:
                last_timestamp = timestamps[-1]
                forecast_timestamps = [last_timestamp + datetime.timedelta(hours=i+1) for i in range(len(forecast_values))]
                
                fig.add_trace(go.Scatter(
                    x=forecast_timestamps,
                    y=forecast_values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color=metric.color, width=2, dash='dash')
                ))
        
        # Add anomalies if available
        anomalies = self.data_processor.detect_anomalies(metric_id)
        if anomalies:
            anomaly_timestamps = [a[0] for a in anomalies]
            anomaly_values = [a[1] for a in anomalies]
            
            fig.add_trace(go.Scatter(
                x=anomaly_timestamps,
                y=anomaly_values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='circle')
            ))
        
        # Set chart title
        chart_title = title if title else f"{metric.name} ({metric.unit})"
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title='Time',
            yaxis_title=metric.unit if metric.unit else 'Value',
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_bar_chart(self, metric_id: str, title: Optional[str] = None, height: int = 400) -> go.Figure:
        """Create a bar chart for a metric."""
        # Get the metric definition
        metric = self.data_processor.metrics.get(metric_id)
        if not metric:
            return go.Figure()
        
        # Get the data
        data = self.data_processor.get_metric_data(metric_id, self.config.visualization_limit)
        if not data:
            return go.Figure()
        
        # Split into timestamps and values
        timestamps, values = zip(*data)
        
        # Create the figure
        fig = go.Figure()
        
        # Add the bar trace
        fig.add_trace(go.Bar(
            x=timestamps,
            y=values,
            name=metric.name,
            marker_color=metric.color
        ))
        
        # Set chart title
        chart_title = title if title else f"{metric.name} ({metric.unit})"
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title='Time',
            yaxis_title=metric.unit if metric.unit else 'Value',
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_area_chart(self, metric_id: str, title: Optional[str] = None, height: int = 400) -> go.Figure:
        """Create an area chart for a metric."""
        # Get the metric definition
        metric = self.data_processor.metrics.get(metric_id)
        if not metric:
            return go.Figure()
        
        # Get the data
        data = self.data_processor.get_metric_data(metric_id, self.config.visualization_limit)
        if not data:
            return go.Figure()
        
        # Split into timestamps and values
        timestamps, values = zip(*data)
        
        # Create the figure
        fig = go.Figure()
        
        # Add the area trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines',
            name=metric.name,
            fill='tozeroy',
            line=dict(color=metric.color, width=1)
        ))
        
        # Set chart title
        chart_title = title if title else f"{metric.name} ({metric.unit})"
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title='Time',
            yaxis_title=metric.unit if metric.unit else 'Value',
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_gauge_chart(self, metric_id: str, title: Optional[str] = None, height: int = 400) -> go.Figure:
        """Create a gauge chart for a metric."""
        # Get the metric definition
        metric = self.data_processor.metrics.get(metric_id)
        if not metric:
            return go.Figure()
        
        # Get the latest data point
        data = self.data_processor.get_metric_data(metric_id, 1)
        if not data:
            return go.Figure()
        
        # Get the latest value
        _, value = data[-1]
        
        # Determine gauge range and colors
        if metric.unit == '%':
            min_val, max_val = 0, 100
            steps = [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ]
        else:
            # Get min and max from historical data
            all_data = self.data_processor.get_metric_data(metric_id)
            if all_data:
                values = [v for _, v in all_data]
                min_val, max_val = min(values), max(values)
                # Add some padding
                range_size = max_val - min_val
                min_val = max(0, min_val - 0.1 * range_size)
                max_val = max_val + 0.1 * range_size
                
                # Create steps
                third = (max_val - min_val) / 3
                steps = [
                    {'range': [min_val, min_val + third], 'color': 'green'},
                    {'range': [min_val + third, min_val + 2*third], 'color': 'yellow'},
                    {'range': [min_val + 2*third, max_val], 'color': 'red'}
                ]
            else:
                min_val, max_val = 0, 100
                steps = [
                    {'range': [0, 33], 'color': 'green'},
                    {'range': [33, 66], 'color': 'yellow'},
                    {'range': [66, 100], 'color': 'red'}
                ]
        
        # Set chart title
        chart_title = title if title else f"{metric.name} ({metric.unit})"
        
        # Create the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': chart_title},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': metric.color},
                'steps': steps,
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        # Update layout
        fig.update_layout(
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_histogram(self, metric_id: str, title: Optional[str] = None, height: int = 400) -> go.Figure:
        """Create a histogram for a metric."""
        # Get the metric definition
        metric = self.data_processor.metrics.get(metric_id)
        if not metric:
            return go.Figure()
        
        # Get the data
        data = self.data_processor.get_metric_data(metric_id)
        if not data:
            return go.Figure()
        
        # Extract values
        _, values = zip(*data)
        
        # Create the figure
        fig = go.Figure()
        
        # Add the histogram trace
        fig.add_trace(go.Histogram(
            x=values,
            name=metric.name,
            marker_color=metric.color,
            nbinsx=30
        ))
        
        # Set chart title
        chart_title = title if title else f"{metric.name} Distribution ({metric.unit})"
        
        # Update layout
        fig.update_layout(
            title=chart_title,
            xaxis_title=metric.unit if metric.unit else 'Value',
            yaxis_title='Frequency',
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_heatmap(self, metric_ids: List[str], title: str = "Correlation Matrix", height: int = 500) -> go.Figure:
        """Create a correlation heatmap for multiple metrics."""
        if not metric_ids or len(metric_ids) < 2:
            return go.Figure()
        
        # Calculate correlations
        corr_matrix = self.data_processor.calculate_correlations(metric_ids)
        if corr_matrix.empty:
            return go.Figure()
        
        # Replace metric IDs with names for better readability
        metric_names = {}
        for metric_id in metric_ids:
            if metric_id in self.data_processor.metrics:
                metric_names[metric_id] = self.data_processor.metrics[metric_id].name
        
        if metric_names:
            corr_matrix = corr_matrix.rename(index=metric_names, columns=metric_names)
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            colorbar=dict(title='Correlation')
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_network_graph(self, metric_ids: List[str], title: str = "Metric Relationships", height: int = 600) -> go.Figure:
        """Create a network graph showing relationships between metrics."""
        if not metric_ids or len(metric_ids) < 2:
            return go.Figure()
        
        # Calculate correlations
        corr_matrix = self.data_processor.calculate_correlations(metric_ids)
        if corr_matrix.empty:
            return go.Figure()
        
        # Create a network graph
        G = nx.Graph()
        
        # Add nodes
        for metric_id in metric_ids:
            if metric_id in self.data_processor.metrics:
                metric = self.data_processor.metrics[metric_id]
                G.add_node(metric_id, name=metric.name, group=metric.data_source.value)
        
        # Add edges based on correlation strength
        for i, metric_id1 in enumerate(metric_ids):
            for j, metric_id2 in enumerate(metric_ids[i+1:], i+1):
                corr = abs(corr_matrix.iloc[i, j])
                if corr > 0.5:  # Only add edges for strong correlations
                    G.add_edge(metric_id1, metric_id2, weight=corr)
        
        # Use networkx spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Extract node positions
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=15,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            ),
            text=[G.nodes[node]['name'] for node in G.nodes()],
            textposition="top center"
        )
        
        # Color nodes by group
        node_groups = [G.nodes[node]['group'] for node in G.nodes()]
        unique_groups = list(set(node_groups))
        node_colors = [unique_groups.index(group) for group in node_groups]
        node_trace.marker.color = node_colors
        
        # Create edge traces
        edge_traces = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G.edges[edge]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight * 5, color='rgba(150,150,150,0.7)'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
        
        # Create the figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        # Update layout
        fig.update_layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
        )
        
        return fig
    
    def create_multi_line_chart(self, metric_ids: List[str], title: str = "Metrics Comparison", height: int = 500) -> go.Figure:
        """Create a multi-line chart comparing multiple metrics."""
        if not metric_ids:
            return go.Figure()
        
        # Create the figure
        fig = go.Figure()
        
        # Add a trace for each metric
        for i, metric_id in enumerate(metric_ids):
            if metric_id in self.data_processor.metrics:
                metric = self.data_processor.metrics[metric_id]
                data = self.data_processor.get_metric_data(metric_id, self.config.visualization_limit)
                
                if data:
                    timestamps, values = zip(*data)
                    
                    # Normalize values for better comparison
                    if len(values) > 1:
                        min_val, max_val = min(values), max(values)
                        if max_val > min_val:
                            normalized = [(v - min_val) / (max_val - min_val) for v in values]
                        else:
                            normalized = [0.5 for _ in values]
                    else:
                        normalized = [0.5]
                    
                    # Use color from metric or from palette
                    color = metric.color if metric.color != "#3366CC" else self.color_palette[i % len(self.color_palette)]
                    
                    fig.add_trace(go.Scatter(
                        x=timestamps,
                        y=normalized,
                        mode='lines',
                        name=f"{metric.name} ({metric.unit})",
                        line=dict(color=color, width=2)
                    ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Normalized Value (0-1)',
            height=height,
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_summary_cards(self, metric_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create summary cards for metrics."""
        cards = {}
        
        for metric_id in metric_ids:
            if metric_id in self.data_processor.metrics:
                metric = self.data_processor.metrics[metric_id]
                data = self.data_processor.get_metric_data(metric_id)
                
                if data:
                    # Get latest value
                    latest_timestamp, latest_value = data[-1]
                    
                    # Calculate change if we have enough data
                    change = None
                    change_percent = None
                    if len(data) > 1:
                        prev_timestamp, prev_value = data[-2]
                        change = latest_value - prev_value
                        if prev_value != 0:
                            change_percent = (change / prev_value) * 100
                    
                    # Calculate statistics
                    values = [v for _, v in data]
                    stats = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "median": sorted(values)[len(values) // 2]
                    }
                    
                    cards[metric_id] = {
                        "name": metric.name,
                        "value": latest_value,
                        "unit": metric.unit,
                        "timestamp": latest_timestamp,
                        "change": change,
                        "change_percent": change_percent,
                        "stats": stats
                    }
        
        return cards
    
    def create_alerts_table(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Create a table of recent alerts."""
        # Sort alerts by timestamp (newest first)
        sorted_alerts = sorted(
            self.data_processor.alerts,
            key=lambda a: a.timestamp,
            reverse=True
        )
        
        # Limit the number of alerts
        alerts_to_show = sorted_alerts[:limit]
        
        # Convert to dictionaries for display
        return [alert.to_dict() for alert in alerts_to_show]
    
    def create_chart(self, metric_id: str, chart_type: ChartType, title: Optional[str] = None, height: int = 400) -> go.Figure:
        """Create a chart based on the specified type."""
        if chart_type == ChartType.LINE:
            return self.create_line_chart(metric_id, title, height)
        elif chart_type == ChartType.BAR:
            return self.create_bar_chart(metric_id, title, height)
        elif chart_type == ChartType.AREA:
            return self.create_area_chart(metric_id, title, height)
        elif chart_type == ChartType.GAUGE:
            return self.create_gauge_chart(metric_id, title, height)
        elif chart_type == ChartType.HISTOGRAM:
            return self.create_histogram(metric_id, title, height)
        else:
            logger.warning(f"Unsupported chart type: {chart_type}")
            return go.Figure()

class DashboardUI:
    """User interface for the analytics dashboard."""
    
    def __init__(self, data_processor: DataProcessor, viz_engine: VisualizationEngine, config: DashboardConfig):
        """Initialize the dashboard UI."""
        self.data_processor = data_processor
        self.viz_engine = viz_engine
        self.config = config
        self.current_mode = config.default_mode
    
    def run(self) -> None:
        """Run the Streamlit dashboard."""
        # Set page config
        st.set_page_config(
            page_title="Skyscope Sentinel Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Set theme
        if self.config.theme == "dark":
            st.markdown("""
            <style>
                .reportview-container {
                    background-color: #0e1117;
                    color: #fafafa;
                }
                .sidebar .sidebar-content {
                    background-color: #262730;
                    color: #fafafa;
                }
                h1, h2, h3, h4, h5, h6 {
                    color: #fafafa;
                }
                .stButton>button {
                    background-color: #4e8cff;
                    color: white;
                }
                .stTextInput>div>div>input {
                    background-color: #262730;
                    color: #fafafa;
                }
                .stSelectbox>div>div>select {
                    background-color: #262730;
                    color: #fafafa;
                }
            </style>
            """, unsafe_allow_html=True)
        
        # Auto-refresh the dashboard
        if self.config.refresh_rate > 0:
            st_autorefresh(interval=self.config.refresh_rate * 1000, key="data_refresh")
        
        # Create sidebar
        self._create_sidebar()
        
        # Create header
        self._create_header()
        
        # Display the selected mode
        if self.current_mode == DashboardMode.OVERVIEW:
            self._show_overview()
        elif self.current_mode == DashboardMode.AGENT_METRICS:
            self._show_agent_metrics()
        elif self.current_mode == DashboardMode.BUSINESS_METRICS:
            self._show_business_metrics()
        elif self.current_mode == DashboardMode.SYSTEM_PERFORMANCE:
            self._show_system_performance()
        elif self.current_mode == DashboardMode.SECURITY_MONITORING:
            self._show_security_monitoring()
        elif self.current_mode == DashboardMode.PREDICTIVE_ANALYTICS:
            self._show_predictive_analytics()
        elif self.current_mode == DashboardMode.NETWORK_ANALYSIS:
            self._show_network_analysis()
        elif self.current_mode == DashboardMode.CUSTOM:
            self._show_custom_dashboard()
    
    def _create_sidebar(self) -> None:
        """Create the sidebar navigation."""
        st.sidebar.title("Skyscope Sentinel")
        st.sidebar.subheader("Analytics Dashboard")
        
        # Add navigation
        selected_mode = st.sidebar.radio(
            "View Mode",
            [mode.value.title() for mode in DashboardMode]
        )
        
        # Update current mode
        for mode in DashboardMode:
            if mode.value.title() == selected_mode:
                self.current_mode = mode
                break
        
        # Add refresh rate control
        st.sidebar.subheader("Settings")
        refresh_rate = st.sidebar.slider(
            "Refresh Rate (seconds)",
            min_value=1,
            max_value=60,
            value=self.config.refresh_rate,
            step=1
        )
        
        if refresh_rate != self.config.refresh_rate:
            self.config.refresh_rate = refresh_rate
            self.config.save()
        
        # Add data simulation control
        st.sidebar.subheader("Data Simulation")
        if st.sidebar.button("Generate Test Data (1 minute)"):
            self.data_processor.simulate_data(60, 1.0)
        
        # Add about section
        st.sidebar.markdown("---")
        st.sidebar.info(
            "Skyscope Sentinel Intelligence AI System\n\n"
            "Real-time Analytics Dashboard\n\n"
            "Version 1.0.0"
        )
    
    def _create_header(self) -> None:
        """Create the dashboard header."""
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.title(f"📊 {self.current_mode.value.title()} Dashboard")
        
        with col2:
            st.text(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
            if self.data_processor.alerts:
                alert_count = len(self.data_processor.alerts)
                st.warning(f"⚠️ {alert_count} active alerts")
    
    def _show_overview(self) -> None:
        """Show the overview dashboard."""
        # Key metrics summary
        st.subheader("Key Metrics")
        key_metrics = [
            "agent_count", "agent_success_rate", "revenue", 
            "profit", "roi", "cpu_usage"
        ]
        
        # Create summary cards
        cards = self.viz_engine.create_summary_cards(key_metrics)
        
        # Display cards in a grid
        cols = st.columns(3)
        for i, metric_id in enumerate(key_metrics):
            if metric_id in cards:
                card = cards[metric_id]
                with cols[i % 3]:
                    st.metric(
                        label=f"{card['name']} ({card['unit']})",
                        value=f"{card['value']:.2f}",
                        delta=f"{card['change']:.2f}" if card['change'] is not None else None
                    )
        
        # Recent alerts
        st.subheader("Recent Alerts")
        alerts = self.viz_engine.create_alerts_table(5)
        if alerts:
            # Convert to DataFrame for better display
            alerts_df = pd.DataFrame(alerts)
            alerts_df = alerts_df[['timestamp', 'level', 'metric_name', 'value', 'threshold', 'message']]
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            alerts_df['timestamp'] = alerts_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("No alerts to display")
        
        # Main metrics charts
        st.subheader("Key Performance Indicators")
        
        # Business metrics
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_line_chart("revenue", "Revenue Over Time"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.viz_engine.create_line_chart("profit", "Profit Over Time"),
                use_container_width=True
            )
        
        # Agent metrics
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_line_chart("agent_count", "Active Agents"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.viz_engine.create_line_chart("agent_success_rate", "Agent Success Rate"),
                use_container_width=True
            )
        
        # System metrics
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_gauge_chart("cpu_usage", "CPU Usage"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.viz_engine.create_gauge_chart("memory_usage", "Memory Usage"),
                use_container_width=True
            )
        
        # Metrics comparison
        st.subheader("Metrics Comparison")
        comparison_metrics = ["revenue", "profit", "agent_tasks_completed", "cpu_usage"]
        st.plotly_chart(
            self.viz_engine.create_multi_line_chart(comparison_metrics, "Key Metrics Trends (Normalized)"),
            use_container_width=True
        )
    
    def _show_agent_metrics(self) -> None:
        """Show the agent metrics dashboard."""
        st.subheader("Agent Performance Metrics")
        
        # Summary statistics
        agent_metrics = ["agent_count", "agent_tasks_completed", "agent_success_rate"]
        cards = self.viz_engine.create_summary_cards(agent_metrics)
        
        cols = st.columns(3)
        for i, metric_id in enumerate(agent_metrics):
            if metric_id in cards:
                card = cards[metric_id]
                with cols[i]:
                    st.metric(
                        label=f"{card['name']} ({card['unit']})",
                        value=f"{card['value']:.2f}",
                        delta=f"{card['change']:.2f}" if card['change'] is not None else None
                    )
        
        # Agent count over time
        st.subheader("Agent Count Over Time")
        st.plotly_chart(
            self.viz_engine.create_line_chart("agent_count", "Active Agents"),
            use_container_width=True
        )
        
        # Tasks completed and success rate
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_area_chart("agent_tasks_completed", "Tasks Completed"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.viz_engine.create_line_chart("agent_success_rate", "Success Rate"),
                use_container_width=True
            )
        
        # Agent distribution (placeholder)
        st.subheader("Agent Distribution")
        
        # Create a placeholder pie chart for agent types
        agent_types = {
            "Research": 2500,
            "Analysis": 2000,
            "Content Creation": 1500,
            "Business Operations": 1000,
            "Crypto Trading": 800,
            "Security": 700,
            "Other": 1500
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(agent_types.keys()),
            values=list(agent_types.values()),
            hole=.3
        )])
        
        fig.update_layout(
            title="Agent Types Distribution",
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Agent performance metrics
        st.subheader("Agent Performance Analysis")
        
        # Create a placeholder bar chart for agent performance by type
        performance_by_type = {
            "Research": 92,
            "Analysis": 88,
            "Content Creation": 95,
            "Business Operations": 91,
            "Crypto Trading": 87,
            "Security": 96,
            "Other": 90
        }
        
        fig = go.Figure(data=[go.Bar(
            x=list(performance_by_type.keys()),
            y=list(performance_by_type.values()),
            marker_color='royalblue'
        )])
        
        fig.update_layout(
            title="Success Rate by Agent Type (%)",
            xaxis_title="Agent Type",
            yaxis_title="Success Rate (%)",
            yaxis=dict(range=[80, 100]),
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_business_metrics(self) -> None:
        """Show the business metrics dashboard."""
        st.subheader("Business Performance Metrics")
        
        # Summary statistics
        business_metrics = ["revenue", "profit", "roi"]
        cards = self.viz_engine.create_summary_cards(business_metrics)
        
        cols = st.columns(3)
        for i, metric_id in enumerate(business_metrics):
            if metric_id in cards:
                card = cards[metric_id]
                with cols[i]:
                    st.metric(
                        label=f"{card['name']} ({card['unit']})",
                        value=f"{card['value']:.2f}",
                        delta=f"{card['change']:.2f}" if card['change'] is not None else None
                    )
        
        # Revenue and profit over time
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_line_chart("revenue", "Revenue Over Time"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.viz_engine.create_line_chart("profit", "Profit Over Time"),
                use_container_width=True
            )
        
        # ROI and profit margin
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_line_chart("roi", "Return on Investment"),
                use_container_width=True
            )
        
        with col2:
            # Create a placeholder for profit margin
            # In a real implementation, this would use actual data
            dates = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(24, 0, -1)]
            values = [35 + random.normalvariate(0, 2) for _ in range(24)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='Profit Margin',
                line=dict(color='green', width=2)
            ))
            
            fig.update_layout(
                title="Profit Margin (%)",
                xaxis_title='Time',
                yaxis_title='%',
                template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Revenue forecast
        st.subheader("Revenue Forecast")
        
        # Train forecasting model if not already trained
        if "revenue" not in self.data_processor.forecasting_models:
            self.data_processor.train_forecasting_model("revenue", "arima")
        
        # Get forecast
        forecast_values = self.data_processor.forecast_metric("revenue", 24)
        
        if forecast_values:
            # Get historical data
            historical_data = self.data_processor.get_metric_data("revenue", 48)
            
            if historical_data:
                # Create forecast chart
                fig = go.Figure()
                
                # Add historical data
                hist_timestamps, hist_values = zip(*historical_data)
                fig.add_trace(go.Scatter(
                    x=hist_timestamps,
                    y=hist_values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Add forecast data
                last_timestamp = hist_timestamps[-1]
                forecast_timestamps = [last_timestamp + datetime.timedelta(hours=i+1) for i in range(len(forecast_values))]
                
                fig.add_trace(go.Scatter(
                    x=forecast_timestamps,
                    y=forecast_values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                # Add confidence intervals (placeholder)
                lower_bound = [value * 0.9 for value in forecast_values]
                upper_bound = [value * 1.1 for value in forecast_values]
                
                fig.add_trace(go.Scatter(
                    x=forecast_timestamps + forecast_timestamps[::-1],
                    y=upper_bound + lower_bound[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    hoverinfo='skip',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title="Revenue Forecast (24 hours)",
                    xaxis_title='Time',
                    yaxis_title='Revenue ($)',
                    template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add forecast summary
                total_forecast = sum(forecast_values)
                avg_forecast = total_forecast / len(forecast_values)
                
                st.info(f"Forecasted total revenue for next 24 hours: ${total_forecast:.2f}")
                st.info(f"Average hourly revenue forecast: ${avg_forecast:.2f}")
            else:
                st.warning("Not enough historical data for forecasting")
        else:
            st.warning("Forecasting model not available")
        
        # Revenue breakdown (placeholder)
        st.subheader("Revenue Breakdown")
        
        # Create a placeholder for revenue by source
        revenue_sources = {
            "AI Services": 45000,
            "Data Analysis": 30000,
            "Consulting": 15000,
            "Crypto Trading": 12000,
            "Content Creation": 8000,
            "Other": 5000
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(revenue_sources.keys()),
            values=list(revenue_sources.values()),
            hole=.3
        )])
        
        fig.update_layout(
            title="Revenue by Source",
            template='plotly_dark' if self.config.theme == 'dark' else 'plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_system_performance(self) -> None:
        """Show the system performance dashboard."""
        st.subheader("System Performance Metrics")
        
        # Summary statistics
        system_metrics = ["cpu_usage", "memory_usage", "api_latency", "db_latency"]
        cards = self.viz_engine.create_summary_cards(system_metrics)
        
        cols = st.columns(4)
        for i, metric_id in enumerate(system_metrics):
            if metric_id in cards:
                card = cards[metric_id]
                with cols[i]:
                    st.metric(
                        label=f"{card['name']} ({card['unit']})",
                        value=f"{card['value']:.2f}",
                        delta=f"{card['change']:.2f}" if card['change'] is not None else None
                    )
        
        # CPU and memory gauges
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_gauge_chart("cpu_usage", "CPU Usage"),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                self.viz_engine.create_gauge_chart("memory_usage", "Memory Usage"),
                use_container_width=True
            )
        
        # CPU and memory over time
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                self.viz_engine.create_line_
