import os
import sys
import time
import json
import logging
import threading
import queue
import uuid
import datetime
import statistics
import math
import re
import socket
import platform
import subprocess
import warnings
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Generator
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque, defaultdict, Counter
import concurrent.futures
import multiprocessing as mp

# System monitoring
import psutil
import GPUtil
try:
    import pynvml
    NVIDIA_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception):
    NVIDIA_AVAILABLE = False

# Data processing and visualization
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import MaxNLocator
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# For web dashboard
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/performance_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("performance_monitor")

# Constants
METRICS_DIR = Path("metrics")
REPORTS_DIR = Path("reports")
ALERTS_DIR = Path("alerts")
OPTIMIZATIONS_DIR = Path("optimizations")
CONFIG_DIR = Path("config")
PERFORMANCE_CONFIG_PATH = CONFIG_DIR / "performance_config.json"
DEFAULT_SAMPLING_INTERVAL = 5  # seconds
DEFAULT_AGGREGATION_INTERVAL = 60  # seconds
DEFAULT_RETENTION_PERIOD = 7  # days
DEFAULT_ALERT_CHECK_INTERVAL = 30  # seconds
MAX_METRICS_HISTORY = 10000  # Maximum number of data points to keep in memory
MAX_EVENTS_HISTORY = 1000  # Maximum number of events to keep in memory

# Ensure directories exist
METRICS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
OPTIMIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

class MetricType(Enum):
    """Types of performance metrics."""
    SYSTEM = "system"
    AGENT = "agent"
    PIPELINE = "pipeline"
    TASK = "task"
    BUSINESS = "business"
    NETWORK = "network"
    DATABASE = "database"
    CUSTOM = "custom"

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class OptimizationType(Enum):
    """Types of optimization suggestions."""
    RESOURCE_ALLOCATION = "resource_allocation"
    AGENT_DISTRIBUTION = "agent_distribution"
    PIPELINE_CONFIGURATION = "pipeline_configuration"
    TASK_PRIORITIZATION = "task_prioritization"
    SCALING = "scaling"
    DATABASE = "database"
    NETWORK = "network"
    CUSTOM = "custom"

class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"

@dataclass
class MetricValue:
    """Represents a single metric value."""
    timestamp: float
    value: Union[float, int, bool, str]
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricValue':
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            value=data["value"],
            tags=data.get("tags", {})
        )

@dataclass
class Metric:
    """Represents a performance metric."""
    name: str
    type: MetricType
    unit: str
    description: str
    values: List[MetricValue] = field(default_factory=list)
    
    def add_value(self, value: Union[float, int, bool, str], tags: Dict[str, str] = None) -> None:
        """Add a value to the metric."""
        metric_value = MetricValue(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        self.values.append(metric_value)
        
        # Limit the number of values kept in memory
        if len(self.values) > MAX_METRICS_HISTORY:
            self.values = self.values[-MAX_METRICS_HISTORY:]
    
    def get_latest_value(self) -> Optional[MetricValue]:
        """Get the latest value of the metric."""
        if not self.values:
            return None
        return self.values[-1]
    
    def get_values_in_range(self, start_time: float, end_time: float) -> List[MetricValue]:
        """Get values in a time range."""
        return [v for v in self.values if start_time <= v.timestamp <= end_time]
    
    def get_average(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[float]:
        """Get average value in a time range."""
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = float('inf')
        
        values = [float(v.value) for v in self.get_values_in_range(start_time, end_time) 
                 if isinstance(v.value, (int, float))]
        
        if not values:
            return None
        
        return sum(values) / len(values)
    
    def get_percentile(self, percentile: float, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Optional[float]:
        """Get percentile value in a time range."""
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = float('inf')
        
        values = [float(v.value) for v in self.get_values_in_range(start_time, end_time) 
                 if isinstance(v.value, (int, float))]
        
        if not values:
            return None
        
        return np.percentile(values, percentile)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type.value,
            "unit": self.unit,
            "description": self.description,
            "values": [v.to_dict() for v in self.values]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            type=MetricType(data["type"]),
            unit=data["unit"],
            description=data["description"],
            values=[MetricValue.from_dict(v) for v in data.get("values", [])]
        )

@dataclass
class Alert:
    """Represents a performance alert."""
    id: str
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: Optional[str] = None
    metric_value: Optional[Any] = None
    threshold: Optional[Any] = None
    source: Optional[str] = None
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    resolution_message: Optional[str] = None
    
    @classmethod
    def create(cls, level: AlertLevel, message: str, metric_name: Optional[str] = None, 
              metric_value: Optional[Any] = None, threshold: Optional[Any] = None, 
              source: Optional[str] = None) -> 'Alert':
        """Create a new alert."""
        return cls(
            id=str(uuid.uuid4()),
            level=level,
            message=message,
            timestamp=time.time(),
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            source=source
        )
    
    def resolve(self, message: Optional[str] = None) -> None:
        """Resolve the alert."""
        self.resolved = True
        self.resolved_timestamp = time.time()
        self.resolution_message = message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "source": self.source,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp,
            "resolution_message": self.resolution_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            level=AlertLevel(data["level"]),
            message=data["message"],
            timestamp=data["timestamp"],
            metric_name=data.get("metric_name"),
            metric_value=data.get("metric_value"),
            threshold=data.get("threshold"),
            source=data.get("source"),
            resolved=data.get("resolved", False),
            resolved_timestamp=data.get("resolved_timestamp"),
            resolution_message=data.get("resolution_message")
        )

@dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion."""
    id: str
    type: OptimizationType
    description: str
    timestamp: float
    priority: int  # 1-10, with 10 being highest
    estimated_impact: float  # 0-1, with 1 being highest
    metrics_affected: List[str]
    implementation_steps: List[str]
    implemented: bool = False
    implemented_timestamp: Optional[float] = None
    result: Optional[str] = None
    
    @classmethod
    def create(cls, type: OptimizationType, description: str, priority: int, 
              estimated_impact: float, metrics_affected: List[str], 
              implementation_steps: List[str]) -> 'OptimizationSuggestion':
        """Create a new optimization suggestion."""
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            description=description,
            timestamp=time.time(),
            priority=priority,
            estimated_impact=estimated_impact,
            metrics_affected=metrics_affected,
            implementation_steps=implementation_steps
        )
    
    def mark_implemented(self, result: Optional[str] = None) -> None:
        """Mark the optimization as implemented."""
        self.implemented = True
        self.implemented_timestamp = time.time()
        self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "description": self.description,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "estimated_impact": self.estimated_impact,
            "metrics_affected": self.metrics_affected,
            "implementation_steps": self.implementation_steps,
            "implemented": self.implemented,
            "implemented_timestamp": self.implemented_timestamp,
            "result": self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationSuggestion':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=OptimizationType(data["type"]),
            description=data["description"],
            timestamp=data["timestamp"],
            priority=data["priority"],
            estimated_impact=data["estimated_impact"],
            metrics_affected=data["metrics_affected"],
            implementation_steps=data["implementation_steps"],
            implemented=data.get("implemented", False),
            implemented_timestamp=data.get("implemented_timestamp"),
            result=data.get("result")
        )

@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    id: str
    action: ScalingAction
    timestamp: float
    reason: str
    metrics_triggered: List[str]
    previous_state: Dict[str, Any]
    new_state: Dict[str, Any]
    success: bool = False
    completion_timestamp: Optional[float] = None
    result: Optional[str] = None
    
    @classmethod
    def create(cls, action: ScalingAction, reason: str, metrics_triggered: List[str],
              previous_state: Dict[str, Any], new_state: Dict[str, Any]) -> 'ScalingEvent':
        """Create a new scaling event."""
        return cls(
            id=str(uuid.uuid4()),
            action=action,
            timestamp=time.time(),
            reason=reason,
            metrics_triggered=metrics_triggered,
            previous_state=previous_state,
            new_state=new_state
        )
    
    def complete(self, success: bool, result: Optional[str] = None) -> None:
        """Mark the scaling event as complete."""
        self.success = success
        self.completion_timestamp = time.time()
        self.result = result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "action": self.action.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metrics_triggered": self.metrics_triggered,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "success": self.success,
            "completion_timestamp": self.completion_timestamp,
            "result": self.result
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScalingEvent':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            action=ScalingAction(data["action"]),
            timestamp=data["timestamp"],
            reason=data["reason"],
            metrics_triggered=data["metrics_triggered"],
            previous_state=data["previous_state"],
            new_state=data["new_state"],
            success=data.get("success", False),
            completion_timestamp=data.get("completion_timestamp"),
            result=data.get("result")
        )

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    sampling_interval: float = DEFAULT_SAMPLING_INTERVAL
    aggregation_interval: float = DEFAULT_AGGREGATION_INTERVAL
    retention_period: int = DEFAULT_RETENTION_PERIOD
    alert_check_interval: float = DEFAULT_ALERT_CHECK_INTERVAL
    enabled_metric_types: List[str] = field(default_factory=lambda: [m.value for m in MetricType])
    alert_thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    scaling_thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimization_settings: Dict[str, Any] = field(default_factory=dict)
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sampling_interval": self.sampling_interval,
            "aggregation_interval": self.aggregation_interval,
            "retention_period": self.retention_period,
            "alert_check_interval": self.alert_check_interval,
            "enabled_metric_types": self.enabled_metric_types,
            "alert_thresholds": self.alert_thresholds,
            "scaling_thresholds": self.scaling_thresholds,
            "optimization_settings": self.optimization_settings,
            "log_level": self.log_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceConfig':
        """Create from dictionary."""
        return cls(
            sampling_interval=data.get("sampling_interval", DEFAULT_SAMPLING_INTERVAL),
            aggregation_interval=data.get("aggregation_interval", DEFAULT_AGGREGATION_INTERVAL),
            retention_period=data.get("retention_period", DEFAULT_RETENTION_PERIOD),
            alert_check_interval=data.get("alert_check_interval", DEFAULT_ALERT_CHECK_INTERVAL),
            enabled_metric_types=data.get("enabled_metric_types", [m.value for m in MetricType]),
            alert_thresholds=data.get("alert_thresholds", {}),
            scaling_thresholds=data.get("scaling_thresholds", {}),
            optimization_settings=data.get("optimization_settings", {}),
            log_level=data.get("log_level", "INFO")
        )
    
    @classmethod
    def load(cls) -> 'PerformanceConfig':
        """Load configuration from file."""
        if not PERFORMANCE_CONFIG_PATH.exists():
            # Create default config
            config = cls()
            config.save()
            return config
        
        try:
            with open(PERFORMANCE_CONFIG_PATH, 'r') as f:
                data = json.load(f)
            
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading performance config: {e}")
            return cls()
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(PERFORMANCE_CONFIG_PATH, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance config: {e}")

class MetricsRegistry:
    """Registry for performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Metric] = {}
        self.lock = threading.RLock()
    
    def register_metric(self, name: str, type: MetricType, unit: str, description: str) -> Metric:
        """Register a new metric."""
        with self.lock:
            if name in self.metrics:
                return self.metrics[name]
            
            metric = Metric(name=name, type=type, unit=unit, description=description)
            self.metrics[name] = metric
            return metric
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        with self.lock:
            return self.metrics.get(name)
    
    def add_metric_value(self, name: str, value: Union[float, int, bool, str], tags: Dict[str, str] = None) -> None:
        """Add a value to a metric."""
        with self.lock:
            metric = self.metrics.get(name)
            if metric:
                metric.add_value(value, tags)
    
    def get_metrics_by_type(self, type: MetricType) -> List[Metric]:
        """Get metrics by type."""
        with self.lock:
            return [m for m in self.metrics.values() if m.type == type]
    
    def get_all_metrics(self) -> List[Metric]:
        """Get all metrics."""
        with self.lock:
            return list(self.metrics.values())
    
    def save_metrics(self, directory: Path = METRICS_DIR) -> None:
        """Save metrics to disk."""
        with self.lock:
            try:
                # Create directory if it doesn't exist
                directory.mkdir(parents=True, exist_ok=True)
                
                # Save each metric type to a separate file
                for type in MetricType:
                    metrics = self.get_metrics_by_type(type)
                    if not metrics:
                        continue
                    
                    filepath = directory / f"{type.value}_metrics.json"
                    with open(filepath, 'w') as f:
                        json.dump({m.name: m.to_dict() for m in metrics}, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving metrics: {e}")
    
    def load_metrics(self, directory: Path = METRICS_DIR) -> None:
        """Load metrics from disk."""
        with self.lock:
            try:
                # Check if directory exists
                if not directory.exists():
                    return
                
                # Load metrics from each file
                for type in MetricType:
                    filepath = directory / f"{type.value}_metrics.json"
                    if not filepath.exists():
                        continue
                    
                    with open(filepath, 'r') as f:
                        metrics_data = json.load(f)
                    
                    for name, data in metrics_data.items():
                        metric = Metric.from_dict(data)
                        self.metrics[name] = metric
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")

class AlertManager:
    """Manager for performance alerts."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()
    
    def add_alert(self, level: AlertLevel, message: str, metric_name: Optional[str] = None,
                 metric_value: Optional[Any] = None, threshold: Optional[Any] = None,
                 source: Optional[str] = None) -> Alert:
        """Add a new alert."""
        with self.lock:
            alert = Alert.create(
                level=level,
                message=message,
                metric_name=metric_name,
                metric_value=metric_value,
                threshold=threshold,
                source=source
            )
            
            self.alerts.append(alert)
            
            # Limit the number of alerts kept in memory
            if len(self.alerts) > MAX_EVENTS_HISTORY:
                self.alerts = self.alerts[-MAX_EVENTS_HISTORY:]
            
            # Notify handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
            
            # Save alert to disk
            self._save_alert(alert)
            
            return alert
    
    def resolve_alert(self, alert_id: str, message: Optional[str] = None) -> bool:
        """Resolve an alert."""
        with self.lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolve(message)
                    
                    # Save alert to disk
                    self._save_alert(alert)
                    
                    return True
            
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        with self.lock:
            return [a for a in self.alerts if not a.resolved]
    
    def get_alerts_by_level(self, level: AlertLevel) -> List[Alert]:
        """Get alerts by level."""
        with self.lock:
            return [a for a in self.alerts if a.level == level]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        with self.lock:
            return list(self.alerts)
    
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        with self.lock:
            self.alert_handlers.append(handler)
    
    def _save_alert(self, alert: Alert) -> None:
        """Save an alert to disk."""
        try:
            # Create alerts directory if it doesn't exist
            ALERTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save alert to file
            filepath = ALERTS_DIR / f"{alert.id}.json"
            with open(filepath, 'w') as f:
                json.dump(alert.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    def load_alerts(self) -> None:
        """Load alerts from disk."""
        with self.lock:
            try:
                # Check if directory exists
                if not ALERTS_DIR.exists():
                    return
                
                # Load alerts from files
                for filepath in ALERTS_DIR.glob("*.json"):
                    try:
                        with open(filepath, 'r') as f:
                            alert_data = json.load(f)
                        
                        alert = Alert.from_dict(alert_data)
                        
                        # Only load recent alerts
                        cutoff_time = time.time() - (self.config.retention_period * 24 * 60 * 60)
                        if alert.timestamp >= cutoff_time:
                            self.alerts.append(alert)
                    except Exception as e:
                        logger.error(f"Error loading alert from {filepath}: {e}")
                
                # Sort alerts by timestamp
                self.alerts.sort(key=lambda a: a.timestamp)
                
                # Limit the number of alerts kept in memory
                if len(self.alerts) > MAX_EVENTS_HISTORY:
                    self.alerts = self.alerts[-MAX_EVENTS_HISTORY:]
            except Exception as e:
                logger.error(f"Error loading alerts: {e}")

class OptimizationEngine:
    """Engine for generating optimization suggestions."""
    
    def __init__(self, config: PerformanceConfig, metrics_registry: MetricsRegistry):
        self.config = config
        self.metrics_registry = metrics_registry
        self.suggestions: List[OptimizationSuggestion] = []
        self.suggestion_handlers: List[Callable[[OptimizationSuggestion], None]] = []
        self.lock = threading.RLock()
    
    def add_suggestion(self, type: OptimizationType, description: str, priority: int,
                      estimated_impact: float, metrics_affected: List[str],
                      implementation_steps: List[str]) -> OptimizationSuggestion:
        """Add a new optimization suggestion."""
        with self.lock:
            suggestion = OptimizationSuggestion.create(
                type=type,
                description=description,
                priority=priority,
                estimated_impact=estimated_impact,
                metrics_affected=metrics_affected,
                implementation_steps=implementation_steps
            )
            
            self.suggestions.append(suggestion)
            
            # Limit the number of suggestions kept in memory
            if len(self.suggestions) > MAX_EVENTS_HISTORY:
                self.suggestions = self.suggestions[-MAX_EVENTS_HISTORY:]
            
            # Notify handlers
            for handler in self.suggestion_handlers:
                try:
                    handler(suggestion)
                except Exception as e:
                    logger.error(f"Error in suggestion handler: {e}")
            
            # Save suggestion to disk
            self._save_suggestion(suggestion)
            
            return suggestion
    
    def mark_implemented(self, suggestion_id: str, result: Optional[str] = None) -> bool:
        """Mark a suggestion as implemented."""
        with self.lock:
            for suggestion in self.suggestions:
                if suggestion.id == suggestion_id and not suggestion.implemented:
                    suggestion.mark_implemented(result)
                    
                    # Save suggestion to disk
                    self._save_suggestion(suggestion)
                    
                    return True
            
            return False
    
    def get_pending_suggestions(self) -> List[OptimizationSuggestion]:
        """Get all pending (not implemented) suggestions."""
        with self.lock:
            return [s for s in self.suggestions if not s.implemented]
    
    def get_suggestions_by_type(self, type: OptimizationType) -> List[OptimizationSuggestion]:
        """Get suggestions by type."""
        with self.lock:
            return [s for s in self.suggestions if s.type == type]
    
    def get_all_suggestions(self) -> List[OptimizationSuggestion]:
        """Get all suggestions."""
        with self.lock:
            return list(self.suggestions)
    
    def add_suggestion_handler(self, handler: Callable[[OptimizationSuggestion], None]) -> None:
        """Add a suggestion handler."""
        with self.lock:
            self.suggestion_handlers.append(handler)
    
    def _save_suggestion(self, suggestion: OptimizationSuggestion) -> None:
        """Save a suggestion to disk."""
        try:
            # Create optimizations directory if it doesn't exist
            OPTIMIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save suggestion to file
            filepath = OPTIMIZATIONS_DIR / f"{suggestion.id}.json"
            with open(filepath, 'w') as f:
                json.dump(suggestion.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving optimization suggestion: {e}")
    
    def load_suggestions(self) -> None:
        """Load suggestions from disk."""
        with self.lock:
            try:
                # Check if directory exists
                if not OPTIMIZATIONS_DIR.exists():
                    return
                
                # Load suggestions from files
                for filepath in OPTIMIZATIONS_DIR.glob("*.json"):
                    try:
                        with open(filepath, 'r') as f:
                            suggestion_data = json.load(f)
                        
                        suggestion = OptimizationSuggestion.from_dict(suggestion_data)
                        
                        # Only load recent suggestions
                        cutoff_time = time.time() - (self.config.retention_period * 24 * 60 * 60)
                        if suggestion.timestamp >= cutoff_time:
                            self.suggestions.append(suggestion)
                    except Exception as e:
                        logger.error(f"Error loading suggestion from {filepath}: {e}")
                
                # Sort suggestions by priority and timestamp
                self.suggestions.sort(key=lambda s: (-s.priority, s.timestamp))
                
                # Limit the number of suggestions kept in memory
                if len(self.suggestions) > MAX_EVENTS_HISTORY:
                    self.suggestions = self.suggestions[-MAX_EVENTS_HISTORY:]
            except Exception as e:
                logger.error(f"Error loading optimization suggestions: {e}")
    
    def analyze_metrics(self) -> List[OptimizationSuggestion]:
        """Analyze metrics and generate optimization suggestions."""
        new_suggestions = []
        
        try:
            # Check CPU usage
            cpu_metric = self.metrics_registry.get_metric("system.cpu.usage")
            if cpu_metric:
                cpu_values = [float(v.value) for v in cpu_metric.values[-10:] if isinstance(v.value, (int, float))]
                if cpu_values and sum(cpu_values) / len(cpu_values) > 80:
                    suggestion = self.add_suggestion(
                        type=OptimizationType.RESOURCE_ALLOCATION,
                        description="High CPU usage detected. Consider optimizing agent workload distribution.",
                        priority=8,
                        estimated_impact=0.7,
                        metrics_affected=["system.cpu.usage", "agent.efficiency"],
                        implementation_steps=[
                            "Identify CPU-intensive agents",
                            "Redistribute workload across more agents",
                            "Consider scaling up hardware resources"
                        ]
                    )
                    new_suggestions.append(suggestion)
            
            # Check memory usage
            memory_metric = self.metrics_registry.get_metric("system.memory.usage_percent")
            if memory_metric:
                memory_values = [float(v.value) for v in memory_metric.values[-10:] if isinstance(v.value, (int, float))]
                if memory_values and sum(memory_values) / len(memory_values) > 85:
                    suggestion = self.add_suggestion(
                        type=OptimizationType.RESOURCE_ALLOCATION,
                        description="High memory usage detected. Consider optimizing memory allocation.",
                        priority=7,
                        estimated_impact=0.6,
                        metrics_affected=["system.memory.usage_percent", "system.memory.available"],
                        implementation_steps=[
                            "Identify memory-intensive agents",
                            "Implement memory usage limits",
                            "Consider increasing system memory"
                        ]
                    )
                    new_suggestions.append(suggestion)
            
            # Check agent efficiency
            efficiency_metric = self.metrics_registry.get_metric("agent.efficiency")
            if efficiency_metric:
                efficiency_values = [float(v.value) for v in efficiency_metric.values[-20:] if isinstance(v.value, (int, float))]
                if efficiency_values and sum(efficiency_values) / len(efficiency_values) < 0.7:
                    suggestion = self.add_suggestion(
                        type=OptimizationType.AGENT_DISTRIBUTION,
                        description="Low agent efficiency detected. Consider rebalancing agent workload.",
                        priority=6,
                        estimated_impact=0.8,
                        metrics_affected=["agent.efficiency", "task.completion_rate"],
                        implementation_steps=[
                            "Analyze agent performance metrics",
                            "Redistribute tasks based on agent specializations",
                            "Consider agent training or replacement"
                        ]
                    )
                    new_suggestions.append(suggestion)
            
            # Check pipeline throughput
            throughput_metric = self.metrics_registry.get_metric("pipeline.throughput")
            if throughput_metric:
                throughput_values = [float(v.value) for v in throughput_metric.values[-20:] if isinstance(v.value, (int, float))]
                if throughput_values:
                    avg_throughput = sum(throughput_values) / len(throughput_values)
                    if avg_throughput < self.config.optimization_settings.get("min_pipeline_throughput", 10):
                        suggestion = self.add_suggestion(
                            type=OptimizationType.PIPELINE_CONFIGURATION,
                            description="Low pipeline throughput detected. Consider optimizing pipeline configuration.",
                            priority=7,
                            estimated_impact=0.7,
                            metrics_affected=["pipeline.throughput", "task.completion_rate"],
                            implementation_steps=[
                                "Identify bottleneck stages in pipeline",
                                "Increase agent allocation to bottleneck stages",
                                "Consider pipeline architecture changes"
                            ]
                        )
                        new_suggestions.append(suggestion)
            
            # Check task completion rate
            completion_metric = self.metrics_registry.get_metric("task.completion_rate")
            if completion_metric:
                completion_values = [float(v.value) for v in completion_metric.values[-20:] if isinstance(v.value, (int, float))]
                if completion_values and sum(completion_values) / len(completion_values) < 0.9:
                    suggestion = self.add_suggestion(
                        type=OptimizationType.TASK_PRIORITIZATION,
                        description="Low task completion rate detected. Consider optimizing task prioritization.",
                        priority=8,
                        estimated_impact=0.9,
                        metrics_affected=["task.completion_rate", "business.revenue"],
                        implementation_steps=[
                            "Analyze task failure patterns",
                            "Implement smarter task prioritization",
                            "Consider task simplification or subdivision"
                        ]
                    )
                    new_suggestions.append(suggestion)
            
            # Check database performance
            db_query_metric = self.metrics_registry.get_metric("database.query_time")
            if db_query_metric:
                query_values = [float(v.value) for v in db_query_metric.values[-20:] if isinstance(v.value, (int, float))]
                if query_values and sum(query_values) / len(query_values) > 0.5:  # 500ms average query time
                    suggestion = self.add_suggestion(
                        type=OptimizationType.DATABASE,
                        description="Slow database queries detected. Consider optimizing database performance.",
                        priority=6,
                        estimated_impact=0.6,
                        metrics_affected=["database.query_time", "system.responsiveness"],
                        implementation_steps=[
                            "Identify slow queries",
                            "Add appropriate indexes",
                            "Consider query optimization or caching"
                        ]
                    )
                    new_suggestions.append(suggestion)
        except Exception as e:
            logger.error(f"Error analyzing metrics for optimization suggestions: {e}")
        
        return new_suggestions

class ScalingManager:
    """Manager for automatic scaling."""
    
    def __init__(self, config: PerformanceConfig, metrics_registry: MetricsRegistry):
        self.config = config
        self.metrics_registry = metrics_registry
        self.scaling_events: List[ScalingEvent] = []
        self.scaling_handlers: List[Callable[[ScalingEvent], None]] = []
        self.lock = threading.RLock()
    
    def add_scaling_event(self, action: ScalingAction, reason: str, metrics_triggered: List[str],
                         previous_state: Dict[str, Any], new_state: Dict[str, Any]) -> ScalingEvent:
        """Add a new scaling event."""
        with self.lock:
            event = ScalingEvent.create(
                action=action,
                reason=reason,
                metrics_triggered=metrics_triggered,
                previous_state=previous_state,
                new_state=new_state
            )
            
            self.scaling_events.append(event)
            
            # Limit the number of events kept in memory
            if len(self.scaling_events) > MAX_EVENTS_HISTORY:
                self.scaling_events = self.scaling_events[-MAX_EVENTS_HISTORY:]
            
            # Notify handlers
            for handler in self.scaling_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in scaling event handler: {e}")
            
            # Save event to disk
            self._save_event(event)
            
            return event
    
    def complete_scaling_event(self, event_id: str, success: bool, result: Optional[str] = None) -> bool:
        """Mark a scaling event as complete."""
        with self.lock:
            for event in self.scaling_events:
                if event.id == event_id and event.completion_timestamp is None:
                    event.complete(success, result)
                    
                    # Save event to disk
                    self._save_event(event)
                    
                    return True
            
            return False
    
    def get_recent_events(self, limit: int = 10) -> List[ScalingEvent]:
        """Get recent scaling events."""
        with self.lock:
            return sorted(self.scaling_events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_events_by_action(self, action: ScalingAction) -> List[ScalingEvent]:
        """Get events by action type."""
        with self.lock:
            return [e for e in self.scaling_events if e.action == action]
    
    def get_all_events(self) -> List[ScalingEvent]:
        """Get all scaling events."""
        with self.lock:
            return list(self.scaling_events)
    
    def add_scaling_handler(self, handler: Callable[[ScalingEvent], None]) -> None:
        """Add a scaling event handler."""
        with self.lock:
            self.scaling_handlers.append(handler)
    
    def _save_event(self, event: ScalingEvent) -> None:
        """Save a scaling event to disk."""
        try:
            # Create directory if it doesn't exist
            scaling_dir = METRICS_DIR / "scaling"
            scaling_dir.mkdir(parents=True, exist_ok=True)
            
            # Save event to file
            filepath = scaling_dir / f"{event.id}.json"
            with open(filepath, 'w') as f:
                json.dump(event.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving scaling event: {e}")
    
    def load_events(self) -> None:
        """Load scaling events from disk."""
        with self.lock:
            try:
                # Check if directory exists
                scaling_dir = METRICS_DIR / "scaling"
                if not scaling_dir.exists():
                    return
                
                # Load events from files
                for filepath in scaling_dir.glob("*.json"):
                    try:
                        with open(filepath, 'r') as f:
                            event_data = json.load(f)
                        
                        event = ScalingEvent.from_dict(event_data)
                        
                        # Only load recent events
                        cutoff_time = time.time() - (self.config.retention_period * 24 * 60 * 60)
                        if event.timestamp >= cutoff_time:
                            self.scaling_events.append(event)
                    except Exception as e:
                        logger.error(f"Error loading scaling event from {filepath}: {e}")
                
                # Sort events by timestamp
                self.scaling_events.sort(key=lambda e: e.timestamp)
                
                # Limit the number of events kept in memory
                if len(self.scaling_events) > MAX_EVENTS_HISTORY:
                    self.scaling_events = self.scaling_events[-MAX_EVENTS_HISTORY:]
            except Exception as e:
                logger.error(f"Error loading scaling events: {e}")
    
    def evaluate_scaling_needs(self) -> Optional[ScalingAction]:
        """Evaluate current metrics and determine if scaling is needed."""
        try:
            # Check CPU usage for scaling up/down
            cpu_metric = self.metrics_registry.get_metric("system.cpu.usage")
            if cpu_metric:
                cpu_values = [float(v.value) for v in cpu_metric.values[-10:] if isinstance(v.value, (int, float))]
                if cpu_values:
                    avg_cpu = sum(cpu_values) / len(cpu_values)
                    cpu_threshold_high = self.config.scaling_thresholds.get("cpu_high", 85)
                    cpu_threshold_low = self.config.scaling_thresholds.get("cpu_low", 20)
                    
                    if avg_cpu > cpu_threshold_high:
                        return ScalingAction.SCALE_UP
                    elif avg_cpu < cpu_threshold_low:
                        return ScalingAction.SCALE_DOWN
            
            # Check memory usage for scaling up/down
            memory_metric = self.metrics_registry.get_metric("system.memory.usage_percent")
            if memory_metric:
                memory_values = [float(v.value) for v in memory_metric.values[-10:] if isinstance(v.value, (int, float))]
                if memory_values:
                    avg_memory = sum(memory_values) / len(memory_values)
                    memory_threshold_high = self.config.scaling_thresholds.get("memory_high", 80)
                    memory_threshold_low = self.config.scaling_thresholds.get("memory_low", 30)
                    
                    if avg_memory > memory_threshold_high:
                        return ScalingAction.SCALE_UP
                    elif avg_memory < memory_threshold_low:
                        return ScalingAction.SCALE_DOWN
            
            # Check agent count for scaling out/in
            agent_count_metric = self.metrics_registry.get_metric("agent.count")
            agent_utilization_metric = self.metrics_registry.get_metric("agent.utilization")
            
            if agent_count_metric and agent_utilization_metric:
                count_values = [int(v.value) for v in agent_count_metric.values[-5:] if isinstance(v.value, (int, float))]
                utilization_values = [float(v.value) for v in agent_utilization_metric.values[-10:] if isinstance(v.value, (int, float))]
                
                if count_values and utilization_values:
                    avg_count = sum(count_values) / len(count_values)
                    avg_utilization = sum(utilization_values) / len(utilization_values)
                    
                    utilization_threshold_high = self.config.scaling_thresholds.get("utilization_high", 0.8)
                    utilization_threshold_low = self.config.scaling_thresholds.get("utilization_low", 0.3)
                    
                    if avg_utilization > utilization_threshold_high:
                        return ScalingAction.SCALE_OUT
                    elif avg_utilization < utilization_threshold_low:
                        return ScalingAction.SCALE_IN
            
            # Check task queue for scaling out/in
            queue_metric = self.metrics_registry.get_metric("task.queue_length")
            if queue_metric:
                queue_values = [int(v.value) for v in queue_metric.values[-10:] if isinstance(v.value, (int, float))]
                if queue_values:
                    avg_queue = sum(queue_values) / len(queue_values)
                    queue_threshold_high = self.config.scaling_thresholds.get("queue_high", 100)
                    queue_threshold_low = self.config.scaling_thresholds.get("queue_low", 10)
                    
                    if avg_queue > queue_threshold_high:
                        return ScalingAction.SCALE_OUT
                    elif avg_queue < queue_threshold_low:
                        return ScalingAction.SCALE_IN
        except Exception as e:
            logger.error(f"Error evaluating scaling needs: {e}")
        
        return ScalingAction.NO_ACTION

class SystemMonitor:
    """Monitor for system-level metrics."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics_registry = metrics_registry
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize system metrics."""
        try:
            # Register CPU metrics
            self.metrics_registry.register_metric(
                name="system.cpu.usage",
                type=MetricType.SYSTEM,
                unit="percent",
                description="CPU usage percentage"
            )
            
            self.metrics_registry.register_metric(
                name="system.cpu.count",
                type=MetricType.SYSTEM,
                unit="count",
                description="Number of CPU cores"
            )
            
            # Register memory metrics
            self.metrics_registry.register_metric(
                name="system.memory.total",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Total system memory"
            )
            
            self.metrics_registry.register_metric(
                name="system.memory.available",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Available system memory"
            )
            
            self.metrics_registry.register_metric(
                name="system.memory.usage_percent",
                type=MetricType.SYSTEM,
                unit="percent",
                description="Memory usage percentage"
            )
            
            # Register disk metrics
            self.metrics_registry.register_metric(
                name="system.disk.total",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Total disk space"
            )
            
            self.metrics_registry.register_metric(
                name="system.disk.free",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Free disk space"
            )
            
            self.metrics_registry.register_metric(
                name="system.disk.usage_percent",
                type=MetricType.SYSTEM,
                unit="percent",
                description="Disk usage percentage"
            )
            
            # Register network metrics
            self.metrics_registry.register_metric(
                name="system.network.sent",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Network bytes sent"
            )
            
            self.metrics_registry.register_metric(
                name="system.network.received",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Network bytes received"
            )
            
            # Register process metrics
            self.metrics_registry.register_metric(
                name="system.process.count",
                type=MetricType.SYSTEM,
                unit="count",
                description="Number of processes"
            )
            
            self.metrics_registry.register_metric(
                name="system.process.python_memory",
                type=MetricType.SYSTEM,
                unit="bytes",
                description="Memory used by Python processes"
            )
            
            # Register GPU metrics if available
            if NVIDIA_AVAILABLE:
                self.metrics_registry.register_metric(
                    name="system.gpu.count",
                    type=MetricType.SYSTEM,
                    unit="count",
                    description="Number of GPUs"
                )
                
                self.metrics_registry.register_metric(
                    name="system.gpu.memory_total",
                    type=MetricType.SYSTEM,
                    unit="bytes",
                    description="Total GPU memory"
                )
                
                self.metrics_registry.register_metric(
                    name="system.gpu.memory_used",
                    type=MetricType.SYSTEM,
                    unit="bytes",
                    description="Used GPU memory"
                )
                
                self.metrics_registry.register_metric(
                    name="system.gpu.utilization",
                    type=MetricType.SYSTEM,
                    unit="percent",
                    description="GPU utilization percentage"
                )
            
            # Register system uptime metric
            self.metrics_registry.register_metric(
                name="system.uptime",
                type=MetricType.SYSTEM,
                unit="seconds",
                description="System uptime"
            )
            
            # Set initial values
            self._update_static_metrics()
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing system metrics: {e}")
            return False
    
    def _update_static_metrics(self) -> None:
        """Update metrics that don't change often."""
        try:
            # CPU count
            cpu_count = psutil.cpu_count(logical=True)
            self.metrics_registry.add_metric_value("system.cpu.count", cpu_count)
            
            # Total memory
            memory = psutil.virtual_memory()
            self.metrics_registry.add_metric_value("system.memory.total", memory.total)
            
            # Total disk space
            disk = psutil.disk_usage('/')
            self.metrics_registry.add_metric_value("system.disk.total", disk.total)
            
            # GPU count if available
            if NVIDIA_AVAILABLE:
                gpu_count = pynvml.nvmlDeviceGetCount()
                self.metrics_registry.add_metric_value("system.gpu.count", gpu_count)
                
                # Total GPU memory
                total_gpu_memory = 0
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_gpu_memory += info.total
                
                self.metrics_registry.add_metric_value("system.gpu.memory_total", total_gpu_memory)
        except Exception as e:
            logger.error(f"Error updating static system metrics: {e}")
    
    def collect_metrics(self) -> None:
        """Collect current system metrics."""
        if not self.initialized:
            if not self.initialize():
                return
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.metrics_registry.add_metric_value("system.cpu.usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics_registry.add_metric_value("system.memory.available", memory.available)
            self.metrics_registry.add_metric_value("system.memory.usage_percent", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.metrics_registry.add_metric_value("system.disk.free", disk.free)
            self.metrics_registry.add_metric_value("system.disk.usage_percent", disk.percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.metrics_registry.add_metric_value("system.network.sent", net_io.bytes_sent)
            self.metrics_registry.add_metric_value("system.network.received", net_io.bytes_recv)
            
            # Process information
            process_count = len(psutil.pids())
            self.metrics_registry.add_metric_value("system.process.count", process_count)
            
            # Python process memory
            python_memory = 0
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_memory += proc.info['memory_info'].rss
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            self.metrics_registry.add_metric_value("system.process.python_memory", python_memory)
            
            # GPU metrics if available
            if NVIDIA_AVAILABLE:
                try:
                    gpu_count = pynvml.nvmlDeviceGetCount()
                    
                    total_gpu_memory_used = 0
                    total_gpu_utilization = 0
                    
                    for i in range(gpu_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # Memory info
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        total_gpu_memory_used += info.used
                        
                        # Utilization info
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        total_gpu_utilization += util.gpu
                    
                    avg_gpu_utilization = total_gpu_utilization / gpu_count if gpu_count > 0 else 0
                    
                    self.metrics_registry.add_metric_value("system.gpu.memory_used", total_gpu_memory_used)
                    self.metrics_registry.add_metric_value("system.gpu.utilization", avg_gpu_utilization)
                except Exception as e:
                    logger.error(f"Error collecting GPU metrics: {e}")
            
            # System uptime
            uptime = time.time() - psutil.boot_time()
            self.metrics_registry.add_metric_value("system.uptime", uptime)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

class AgentMonitor:
    """Monitor for agent-specific metrics."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics_registry = metrics_registry
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize agent metrics."""
        try:
            # Register agent count metrics
            self.metrics_registry.register_metric(
                name="agent.count",
                type=MetricType.AGENT,
                unit="count",
                description="Total number of agents"
            )
            
            self.metrics_registry.register_metric(
                name="agent.active_count",
                type=MetricType.AGENT,
                unit="count",
                description="Number of active agents"
            )
            
            # Register agent performance metrics
            self.metrics_registry.register_metric(
                name="agent.efficiency",
                type=MetricType.AGENT,
                unit="ratio",
                description="Agent efficiency ratio (tasks completed / tasks assigned)"
            )
            
            self.metrics_registry.register_metric(
                name="agent.utilization",
                type=MetricType.AGENT,
                unit="ratio",
                description="Agent utilization ratio (time busy / total time)"
            )
            
            self.metrics_registry.register_metric(
                name="agent.error_rate",
                type=MetricType.AGENT,
                unit="ratio",
                description="Agent error rate (errors / total tasks)"
            )
            
            self.metrics_registry.register_metric(
                name="agent.response_time",
                type=MetricType.AGENT,
                unit="seconds",
                description="Average agent response time"
            )
            
            # Register agent resource metrics
            self.metrics_registry.register_metric(
                name="agent.memory_usage",
                type=MetricType.AGENT,
                unit="bytes",
                description="Total memory used by agents"
            )
            
            self.metrics_registry.register_metric(
                name="agent.cpu_usage",
                type=MetricType.AGENT,
                unit="percent",
                description="Total CPU used by agents"
            )
            
            # Register agent type metrics
            self.metrics_registry.register_metric(
                name="agent.type_distribution",
                type=MetricType.AGENT,
                unit="count",
                description="Distribution of agent types"
            )
            
            # Register agent lifecycle metrics
            self.metrics_registry.register_metric(
                name="agent.creation_rate",
                type=MetricType.AGENT,
                unit="count/minute",
                description="Rate of agent creation"
            )
            
            self.metrics_registry.register_metric(
                name="agent.termination_rate",
                type=MetricType.AGENT,
                unit="count/minute",
                description="Rate of agent termination"
            )
            
            self.metrics_registry.register_metric(
                name="agent.lifespan",
                type=MetricType.AGENT,
                unit="seconds",
                description="Average agent lifespan"
            )
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing agent metrics: {e}")
            return False
    
    def collect_metrics(self, agent_manager=None) -> None:
        """Collect current agent metrics."""
        if not self.initialized:
            if not self.initialize():
                return
        
        try:
            # If agent_manager is provided, use it to get real metrics
            if agent_manager:
                # Get agent counts
                total_agents = agent_manager.get_agent_count()
                active_agents = agent_manager.get_active_agent_count()
                
                self.metrics_registry.add_metric_value("agent.count", total_agents)
                self.metrics_registry.add_metric_value("agent.active_count", active_agents)
                
                # Get agent performance metrics
                efficiency = agent_manager.get_agent_efficiency()
                utilization = agent_manager.get_agent_utilization()
                error_rate = agent_manager.get_agent_error_rate()
                response_time = agent_manager.get_agent_response_time()
                
                self.metrics_registry.add_metric_value("agent.efficiency", efficiency)
                self.metrics_registry.add_metric_value("agent.utilization", utilization)
                self.metrics_registry.add_metric_value("agent.error_rate", error_rate)
                self.metrics_registry.add_metric_value("agent.response_time", response_time)
                
                # Get agent resource usage
                memory_usage = agent_manager.get_agent_memory_usage()
                cpu_usage = agent_manager.get_agent_cpu_usage()
                
                self.metrics_registry.add_metric_value("agent.memory_usage", memory_usage)
                self.metrics_registry.add_metric_value("agent.cpu_usage", cpu_usage)
                
                # Get agent type distribution
                type_distribution = agent_manager.get_agent_type_distribution()
                
                self.metrics_registry.add_metric_value("agent.type_distribution", type_distribution)
                
                # Get agent lifecycle metrics
                creation_rate = agent_manager.get_agent_creation_rate()
                termination_rate = agent_manager.get_agent_termination_rate()
                lifespan = agent_manager.get_agent_lifespan()
                
                self.metrics_registry.add_metric_value("agent.creation_rate", creation_rate)
                self.metrics_registry.add_metric_value("agent.termination_rate", termination_rate)
                self.metrics_registry.add_metric_value("agent.lifespan", lifespan)
            else:
                # Generate simulated metrics for testing
                self._generate_simulated_metrics()
        except Exception as e:
            logger.error(f"Error collecting agent metrics: {e}")
    
    def _generate_simulated_metrics(self) -> None:
        """Generate simulated agent metrics for testing."""
        try:
            # Simulated agent counts
            total_agents = 10000  # 10,000 agents
            active_ratio = 0.8 + 0.1 * math.sin(time.time() / 3600)  # Varies between 0.7 and 0.9
            active_agents = int(total_agents * active_ratio)
            
            self.metrics_registry.add_metric_value("agent.count", total_agents)
            self.metrics_registry.add_metric_value("agent.active_count", active_agents)
            
            # Simulated performance metrics
            efficiency = 0.85 + 0.1 * math.sin(time.time() / 7200)  # Varies between 0.75 and 0.95
            utilization = active_ratio * 0.9  # Related to active ratio
            error_rate = 0.05 + 0.03 * math.sin(time.time() / 3600)  # Varies between 0.02 and 0.08
            response_time = 0.5 + 0.3 * math.sin(time.time() / 1800)  # Varies between 0.2 and 0.8 seconds
            
            self.metrics_registry.add_metric_value("agent.efficiency", efficiency)
            self.metrics_registry.add_metric_value("agent.utilization", utilization)
            self.metrics_registry.add_metric_value("agent.error_rate", error_rate)
            self.metrics_registry.add_metric_value("agent.response_time", response_time)
            
            # Simulated resource usage
            memory_per_agent = 10 * 1024 * 1024  # 10 MB per agent
            memory_usage = active_agents * memory_per_agent
            cpu_usage = active_agents / total_agents * 100  # CPU usage proportional to active agents
            
            self.metrics_registry.add_metric_value("agent.memory_usage", memory_usage)
            self.metrics_registry.add_metric_value("agent.cpu_usage", cpu_usage)
            
            # Simulated type distribution
            type_distribution = {
                "researcher": int(total_agents * 0.2),
                "analyst": int(total_agents * 0.3),
                "executor": int(total_agents * 0.4),
                "manager": int(total_agents * 0.1)
            }
            
            self.metrics_registry.add_metric_value("agent.type_distribution", type_distribution)
            
            # Simulated lifecycle metrics
            creation_rate = 10 + 5 * math.sin(time.time() / 900)  # Varies between 5 and 15 per minute
            termination_rate = creation_rate * (0.9 + 0.1 * math.sin(time.time() / 1200))  # Slightly less than creation rate
            lifespan = 3600 + 1800 * math.sin(time.time() / 7200)  # Varies between 0.5 and 1.5 hours
            
            self.metrics_registry.add_metric_value("agent.creation_rate", creation_rate)
            self.metrics_registry.add_metric_value("agent.termination_rate", termination_rate)
            self.metrics_registry.add_metric_value("agent.lifespan", lifespan)
        except Exception as e:
            logger.error(f"Error generating simulated agent metrics: {e}")

class PipelineMonitor:
    """Monitor for pipeline-specific metrics."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics_registry = metrics_registry
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize pipeline metrics."""
        try:
            # Register pipeline count metrics
            self.metrics_registry.register_metric(
                name="pipeline.count",
                type=MetricType.PIPELINE,
                unit="count",
                description="Total number of pipelines"
            )
            
            self.metrics_registry.register_metric(
                name="pipeline.active_count",
                type=MetricType.PIPELINE,
                unit="count",
                description="Number of active pipelines"
            )
            
            # Register pipeline performance metrics
            self.metrics_registry.register_metric(
                name="pipeline.throughput",
                type=MetricType.PIPELINE,
                unit="tasks/minute",
                description="Pipeline throughput (tasks completed per minute)"
            )
            
            self.metrics_registry.register_metric(
                name="pipeline.latency",
                type=MetricType.PIPELINE,
                unit="seconds",
                description="Average pipeline latency (time to complete a task)"
            )
            
            self.metrics_registry.register_metric(
                name="pipeline.error_rate",
                type=MetricType.PIPELINE,
                unit="ratio",
                description="Pipeline error rate (errors / total tasks)"
            )
            
            self.metrics_registry.register_metric(
                name="pipeline.utilization",
                type=MetricType.PIPELINE,
                unit="ratio",
                description="Pipeline utilization ratio (capacity used / total capacity)"
            )
            
            # Register pipeline stage metrics
            self.metrics_registry.register_metric(
                name="pipeline.stage_times",
                type=MetricType.PIPELINE,
                unit="seconds",
                description="Time spent in each pipeline stage"
            )
            
            self.metrics_registry.register_metric(
                name="pipeline.bottleneck_stage",
                type=MetricType.PIPELINE,
                unit="name",
                description="Current pipeline bottleneck stage"
            )
            
            # Register pipeline resource metrics
            self.metrics_registry.register_metric(
                name="pipeline.memory_usage",
                type=MetricType.PIPELINE,
                unit="bytes",
                description="Total memory used by pipelines"
            )
            
            self.metrics_registry.register_metric(
                name="pipeline.cpu_usage",
                type=MetricType.PIPELINE,
                unit="percent",
                description="Total CPU used by pipelines"
            )
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing pipeline metrics: {e}")
            return False
    
    def collect_metrics(self, pipeline_manager=None) -> None:
        """Collect current pipeline metrics."""
        if not self.initialized:
            if not self.initialize():
                return
        
        try:
            # If pipeline_manager is provided, use it to get real metrics
            if pipeline_manager:
                # Get pipeline counts
                total_pipelines = pipeline_manager.get_pipeline_count()
                active_pipelines = pipeline_manager.get_active_pipeline_count()
                
                self.metrics_registry.add_metric_value("pipeline.count", total_pipelines)
                self.metrics_registry.add_metric_value("pipeline.active_count", active_pipelines)
                
                # Get pipeline performance metrics
                throughput = pipeline_manager.get_pipeline_throughput()
                latency = pipeline_manager.get_pipeline_latency()
                error_rate = pipeline_manager.get_pipeline_error_rate()
                utilization = pipeline_manager.get_pipeline_utilization()
                
                self.metrics_registry.add_metric_value("pipeline.throughput", throughput)
                self.metrics_registry.add_metric_value("pipeline.latency", latency)
                self.metrics_registry.add_metric_value("pipeline.error_rate", error_rate)
                self.metrics_registry.add_metric_value("pipeline.utilization", utilization)
                
                # Get pipeline stage metrics
                stage_times = pipeline_manager.get_pipeline_stage_times()
                bottleneck_stage = pipeline_manager.get_pipeline_bottleneck_stage()
                
                self.metrics_registry.add_metric_value("pipeline.stage_times", stage_times)
                self.metrics_registry.add_metric_value("pipeline.bottleneck_stage", bottleneck_stage)
                
                # Get pipeline resource usage
                memory_usage = pipeline_manager.get_pipeline_memory_usage()
                cpu_usage = pipeline_manager.get_pipeline_cpu_usage()
                
                self.metrics_registry.add_metric_value("pipeline.memory_usage", memory_usage)
                self.metrics_registry.add_metric_value("pipeline.cpu_usage", cpu_usage)
            else:
                # Generate simulated metrics for testing
                self._generate_simulated_metrics()
        except Exception as e:
            logger.error(f"Error collecting pipeline metrics: {e}")
    
    def _generate_simulated_metrics(self) -> None:
        """Generate simulated pipeline metrics for testing."""
        try:
            # Simulated pipeline counts
            total_pipelines = 100  # 100 pipelines
            active_ratio = 0.9 + 0.1 * math.sin(time.time() / 3600)  # Varies between 0.8 and 1.0
            active_pipelines = int(total_pipelines * active_ratio)
            
            self.metrics_registry.add_metric_value("pipeline.count", total_pipelines)
            self.metrics_registry.add_metric_value("pipeline.active_count", active_pipelines)
            
            # Simulated performance metrics
            throughput = 50 + 20 * math.sin(time.time() / 1800)  # Varies between 30 and 70 tasks/minute
            latency = 30 + 15 * math.sin(time.time() / 2400)  # Varies between 15 and 45 seconds
            error_rate = 0.03 + 0.02 * math.sin(time.time() / 3600)  # Varies between 0.01 and 0.05
            utilization = active_ratio * (0.7 + 0.2 * math.sin(time.time() / 1200))  # Varies with active ratio
            
            self.metrics_registry.add_metric_value("pipeline.throughput", throughput)
            self.metrics_registry.add_metric_value("pipeline.latency", latency)
            self.metrics_registry.add_metric_value("pipeline.error_rate", error_rate)
            self.metrics_registry.add_metric_value("pipeline.utilization", utilization)
            
            # Simulated stage metrics
            stage_times = {
                "input": 5 + 2 * math.sin(time.time() / 600),
                "processing": 15 + 5 * math.sin(time.time() / 900),
                "analysis": 8 + 4 * math.sin(time.time() / 1200),
                "output": 2 + 1 * math.sin(time.time() / 300)
            }
            
            # Determine bottleneck stage
            bottleneck_stage = max(stage_times.items(), key=lambda x: x[1])[0]
            
            self.metrics_registry.add_metric_value("pipeline.stage_times", stage_times)
            self.metrics_registry.add_metric_value("pipeline.bottleneck_stage", bottleneck_stage)
            
            # Simulated resource usage
            memory_per_pipeline = 50 * 1024 * 1024  # 50 MB per pipeline
            memory_usage = active_pipelines * memory_per_pipeline
            cpu_usage = active_pipelines / total_pipelines * 80  # CPU usage proportional to active pipelines
            
            self.metrics_registry.add_metric_value("pipeline.memory_usage", memory_usage)
            self.metrics_registry.add_metric_value("pipeline.cpu_usage", cpu_usage)
        except Exception as e:
            logger.error(f"Error generating simulated pipeline metrics: {e}")

class TaskMonitor:
    """Monitor for task-specific metrics."""
    
    def __init__(self, metrics_registry: MetricsRegistry):
        self.metrics_registry = metrics_registry
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize task metrics."""
        try:
            # Register task count metrics
            self.metrics_registry.register_metric(
                name="task.count",
                type=MetricType.TASK,
                unit="count",
                description="Total number of tasks"
            )
            
            self.metrics_registry.register_metric(
                name="task.active_count",
                type=MetricType.TASK,
                unit="count",
                description="Number of active tasks"
            )
            
            self.metrics_registry.register_metric(
                name="task.queue_length",
                type=MetricType.TASK,
                unit="count",
                description="Number of tasks in queue"
            )
            
            # Register task performance metrics
            self.metrics_registry.register_metric(
                name="task.completion_rate",
                type=MetricType.TASK,
                unit="ratio",
                description="Task completion rate (completed / total)"
            )
            
            self.metrics_registry.register_metric(
                name="task.error_rate",
                type=MetricType.TASK,
                unit="ratio",
                description="Task error rate (errors / total)"
            )
            
            self.metrics_registry.register_metric(
                name="task.processing_time",
                type=MetricType.TASK,
                unit="seconds",
                description="Average task processing time"
            )
            
            self.metrics_registry.register_metric(
                name="task.waiting_time",
                type=MetricType.TASK,
                unit="seconds",
                description="Average task waiting time"
            )
            
            # Register task type metrics
            self.metrics_registry.register_metric(
                name="task.type_distribution",
                type=MetricType.TASK,
                unit="count",
                description="Distribution of task types"
            )
            
            self.metrics_registry.register_metric(
                name="task.priority_distribution",
                type=MetricType.TASK,
                unit="count",
                description="Distribution of task priorities"
            )
            
            # Register task resource metrics
            self.metrics_registry.register_metric(
                name="task.memory_usage",
                type=MetricType.TASK,
                unit="bytes",
                description="Average memory usage per task"
            )
            
            self.metrics_registry.register_metric(
                name="task.cpu_usage",
                type=MetricType.TASK,
                unit="percent",
                description="Average CPU usage per task"
            )
            
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error initializing task metrics: {e}")
            return False
    
    def collect_metrics(self, task_manager=None) -> None:
        """Collect current task metrics."""
        if not self.initialized:
            if not self.initialize():
                return
        
        try:
            # If task_manager is provided, use it to get real metrics
            if task_manager:
                # Get task counts
                total_tasks = task_manager.get_task_count()
                active_tasks = task_manager.get_active_task_count()
                queue_length = task_manager.get
