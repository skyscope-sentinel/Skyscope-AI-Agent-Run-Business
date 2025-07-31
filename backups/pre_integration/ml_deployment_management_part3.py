
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_deployment_management_part3.py - Final part of ML Deployment and Management System

This module completes the ML deployment and management system with:
1. Completion of the ModelMonitoringManager class
2. ComplianceManager for security, privacy, and regulatory compliance
3. FastAPI application with REST and WebSocket endpoints
4. Main entry point and initialization
5. Integration scripts for the complete system

Part of Skyscope Sentinel Intelligence AI - ITERATION 11
"""

import asyncio
import base64
import datetime
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from functools import lru_cache, partial, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, BinaryIO, Generator

try:
    import aiohttp
    import numpy as np
    import pandas as pd
    import psutil
    import pydantic
    import requests
    import torch
    import websockets
    import yaml
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, WebSocket, WebSocketDisconnect, File, UploadFile, Request, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.openapi.docs import get_swagger_ui_html
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    from pydantic import BaseModel, Field, validator
    from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, DateTime, create_engine, func
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response
    
    # Import local modules with error handling
    try:
        from ml_deployment_management import (
            Employee, Team, Project, MLModel, ModelDeployment, DeploymentMetric, AuditLog,
            EmployeeHierarchyManager, MLModelManager, OpenAIUnofficialManager,
            KubernetesDeploymentManager, Base, Session
        )
        LOCAL_IMPORTS_AVAILABLE = True
    except ImportError:
        warnings.warn("Local modules not available. Running in standalone mode.")
        LOCAL_IMPORTS_AVAILABLE = False
        # Create simplified Base for standalone mode
        Base = declarative_base()
        engine = create_engine('sqlite:///ml_deployment.db')
        Session = sessionmaker(bind=engine)

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "aiohttp", "numpy", "pandas", "psutil", "pydantic", 
                          "requests", "torch", "websockets", "pyyaml", "fastapi", 
                          "uvicorn", "prometheus-client", "sqlalchemy", "python-multipart",
                          "python-jose[cryptography]", "passlib[bcrypt]"])
    print("Please restart the application.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_deployment_part3.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_EMPLOYEES = 10000
DEFAULT_MODEL = "gpt-4o-2024-05-13"
AUDIO_PREVIEW_MODEL = "gpt-4o-audio-preview-2025-06-03"
REALTIME_PREVIEW_MODEL = "gpt-4o-realtime-preview-2024-10-01"
DEFAULT_OLLAMA_MODEL = "llama3"
KUBERNETES_NAMESPACE = "skyscope-ml"
MODEL_REGISTRY_PATH = Path("./model_registry")
MODEL_REGISTRY_PATH.mkdir(exist_ok=True)
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "skyscope_sentinel_super_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

#######################################################
# ModelMonitoringManager Class Completion
#######################################################

class ModelMonitoringManager:
    """
    Manages model monitoring, including performance tracking,
    drift detection, anomaly detection, and alert management.
    """
    
    def __init__(self, db_session=None):
        self.session = db_session or Session()
        self.monitored_models = {}
        self.drift_detectors = {}
        self.anomaly_detectors = {}
        self.alert_thresholds = {}
        self.alert_subscribers = defaultdict(list)
        self.alert_history = defaultdict(list)
        self.performance_history = defaultdict(lambda: defaultdict(list))
        
        # Initialize metrics
        self.model_performance_gauge = Gauge('model_performance', 'Model performance metrics', ['model_id', 'metric'])
        self.model_drift_gauge = Gauge('model_drift', 'Model drift metrics', ['model_id', 'feature'])
        self.model_latency_histogram = Histogram('model_inference_latency', 'Model inference latency', ['model_id'])
        self.model_prediction_counter = Counter('model_predictions_total', 'Total number of model predictions', ['model_id', 'result'])
        self.model_anomaly_counter = Counter('model_anomalies_total', 'Total number of anomalies detected', ['model_id', 'type'])
    
    def register_model_for_monitoring(self, model_id: int, 
                                     reference_data_path: Optional[str] = None,
                                     performance_metrics: List[str] = None,
                                     drift_detection_features: List[str] = None,
                                     alert_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """Register a model for monitoring."""
        try:
            # Get model
            model = self.session.query(MLModel).filter_by(id=model_id).first()
            if not model:
                return {"status": "error", "message": f"Model with ID {model_id} not found"}
            
            # Default metrics if not provided
            if not performance_metrics:
                performance_metrics = ["accuracy", "latency", "throughput"]
            
            # Default alert thresholds if not provided
            if not alert_thresholds:
                alert_thresholds = {
                    "accuracy_drop": 0.05,  # Alert if accuracy drops by 5%
                    "latency_increase": 0.2,  # Alert if latency increases by 20%
                    "drift_threshold": 0.1,  # Alert if drift score exceeds 0.1
                    "error_rate": 0.02,  # Alert if error rate exceeds 2%
                    "anomaly_threshold": 3.0  # Alert if anomaly score exceeds 3 standard deviations
                }
            
            # Store monitoring configuration
            self.monitored_models[model_id] = {
                "model_name": model.name,
                "model_version": model.version,
                "reference_data_path": reference_data_path,
                "performance_metrics": performance_metrics,
                "drift_detection_features": drift_detection_features,
                "registered_at": datetime.datetime.now().isoformat(),
                "last_updated": datetime.datetime.now().isoformat(),
                "status": "active"
            }
            
            # Store alert thresholds
            self.alert_thresholds[model_id] = alert_thresholds
            
            # Initialize drift detector if reference data provided
            if reference_data_path and drift_detection_features:
                self._initialize_drift_detector(model_id, reference_data_path, drift_detection_features)
            
            # Initialize anomaly detector
            self._initialize_anomaly_detector(model_id)
            
            return {
                "status": "success",
                "message": f"Model {model.name} v{model.version} registered for monitoring",
                "model_id": model_id,
                "metrics": performance_metrics,
                "alert_thresholds": alert_thresholds
            }
        except Exception as e:
            logger.error(f"Error registering model for monitoring: {e}")
            return {"status": "error", "message": str(e)}
    
    def _initialize_drift_detector(self, model_id: int, reference_data_path: str,
                                  features: List[str]) -> None:
        """Initialize a drift detector for a model."""
        try:
            # Load reference data
            if reference_data_path.endswith('.csv'):
                reference_data = pd.read_csv(reference_data_path)
            elif reference_data_path.endswith('.parquet'):
                reference_data = pd.read_parquet(reference_data_path)
            else:
                raise ValueError(f"Unsupported reference data format: {reference_data_path}")
            
            # Validate features
            for feature in features:
                if feature not in reference_data.columns:
                    raise ValueError(f"Feature '{feature}' not found in reference data")
            
            # Store reference statistics
            feature_stats = {}
            for feature in features:
                feature_data = reference_data[feature]
                
                if pd.api.types.is_numeric_dtype(feature_data):
                    # Numeric feature
                    feature_stats[feature] = {
                        "mean": feature_data.mean(),
                        "std": feature_data.std(),
                        "min": feature_data.min(),
                        "max": feature_data.max(),
                        "median": feature_data.median(),
                        "type": "numeric"
                    }
                else:
                    # Categorical feature
                    value_counts = feature_data.value_counts(normalize=True).to_dict()
                    feature_stats[feature] = {
                        "distribution": value_counts,
                        "unique_values": feature_data.nunique(),
                        "type": "categorical"
                    }
            
            # Store drift detector
            self.drift_detectors[model_id] = {
                "reference_stats": feature_stats,
                "features": features,
                "last_updated": datetime.datetime.now().isoformat(),
                "drift_scores": {}
            }
            
            logger.info(f"Drift detector initialized for model {model_id} with {len(features)} features")
        except Exception as e:
            logger.error(f"Error initializing drift detector: {e}")
            raise
    
    def _initialize_anomaly_detector(self, model_id: int) -> None:
        """Initialize an anomaly detector for a model."""
        try:
            # Create anomaly detector configuration
            self.anomaly_detectors[model_id] = {
                "window_size": 100,  # Number of observations to maintain
                "metrics": {},  # Metrics history for anomaly detection
                "last_updated": datetime.datetime.now().isoformat()
            }
            
            logger.info(f"Anomaly detector initialized for model {model_id}")
        except Exception as e:
            logger.error(f"Error initializing anomaly detector: {e}")
            raise
    
    def update_model_performance(self, model_id: int, 
                               metrics: Dict[str, float]) -> Dict[str, Any]:
        """Update performance metrics for a model."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Record timestamp
            timestamp = datetime.datetime.now().isoformat()
            
            # Update metrics
            for metric_name, metric_value in metrics.items():
                self.model_performance_gauge.labels(model_id=str(model_id), metric=metric_name).set(metric_value)
                
                # Store in performance history
                self.performance_history[model_id][metric_name].append({
                    "value": metric_value,
                    "timestamp": timestamp
                })
                
                # Keep only the last 1000 entries
                if len(self.performance_history[model_id][metric_name]) > 1000:
                    self.performance_history[model_id][metric_name].pop(0)
            
            # Store metrics in database
            for metric_name, metric_value in metrics.items():
                deployment_metric = DeploymentMetric(
                    deployment_id=model_id,  # Assuming model_id is the deployment_id
                    metric_name=metric_name,
                    metric_value=metric_value
                )
                self.session.add(deployment_metric)
            
            self.session.commit()
            
            # Check for anomalies
            anomalies = self._check_for_anomalies(model_id, metrics)
            
            # Check for performance degradation alerts
            alerts = []
            if model_id in self.alert_thresholds:
                thresholds = self.alert_thresholds[model_id]
                
                # Check accuracy drop
                if "accuracy" in metrics and "accuracy_drop" in thresholds:
                    # Get baseline accuracy (average of last 10 measurements)
                    baseline_accuracy = self._get_baseline_metric(model_id, "accuracy", 10)
                    if baseline_accuracy and metrics["accuracy"] < baseline_accuracy - thresholds["accuracy_drop"]:
                        alerts.append({
                            "type": "accuracy_drop",
                            "current": metrics["accuracy"],
                            "baseline": baseline_accuracy,
                            "threshold": thresholds["accuracy_drop"],
                            "timestamp": timestamp
                        })
                
                # Check latency increase
                if "latency" in metrics and "latency_increase" in thresholds:
                    baseline_latency = self._get_baseline_metric(model_id, "latency", 10)
                    if baseline_latency and metrics["latency"] > baseline_latency * (1 + thresholds["latency_increase"]):
                        alerts.append({
                            "type": "latency_increase",
                            "current": metrics["latency"],
                            "baseline": baseline_latency,
                            "threshold": thresholds["latency_increase"],
                            "timestamp": timestamp
                        })
                
                # Check error rate
                if "error_rate" in metrics and "error_rate" in thresholds:
                    if metrics["error_rate"] > thresholds["error_rate"]:
                        alerts.append({
                            "type": "high_error_rate",
                            "current": metrics["error_rate"],
                            "threshold": thresholds["error_rate"],
                            "timestamp": timestamp
                        })
            
            # Combine with anomaly alerts
            alerts.extend(anomalies)
            
            # Send alerts if any
            if alerts:
                self._send_alerts(model_id, alerts)
            
            return {
                "status": "success",
                "model_id": model_id,
                "timestamp": timestamp,
                "metrics": metrics,
                "anomalies": anomalies,
                "alerts": alerts
            }
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
            return {"status": "error", "message": str(e)}
    
    def _get_baseline_metric(self, model_id: int, metric_name: str, n: int = 10) -> Optional[float]:
        """Get baseline value for a metric based on historical data."""
        try:
            if model_id not in self.performance_history or metric_name not in self.performance_history[model_id]:
                return None
            
            history = self.performance_history[model_id][metric_name]
            if len(history) < n:
                return None
            
            # Calculate average of last n measurements
            values = [entry["value"] for entry in history[-n:]]
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error getting baseline metric: {e}")
            return None
    
    def _check_for_anomalies(self, model_id: int, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for anomalies in model metrics."""
        try:
            if model_id not in self.anomaly_detectors:
                return []
            
            anomaly_detector = self.anomaly_detectors[model_id]
            anomalies = []
            timestamp = datetime.datetime.now().isoformat()
            
            for metric_name, metric_value in metrics.items():
                # Initialize metric history if not exists
                if metric_name not in anomaly_detector["metrics"]:
                    anomaly_detector["metrics"][metric_name] = deque(maxlen=anomaly_detector["window_size"])
                
                metric_history = anomaly_detector["metrics"][metric_name]
                
                # Add current value to history
                metric_history.append(metric_value)
                
                # Need enough data points for anomaly detection
                if len(metric_history) < 10:
                    continue
                
                # Calculate mean and standard deviation
                values = np.array(metric_history)
                mean = np.mean(values)
                std = np.std(values)
                
                # Calculate z-score
                z_score = abs((metric_value - mean) / max(std, 1e-10))
                
                # Check if value is an anomaly (z-score > threshold)
                anomaly_threshold = self.alert_thresholds.get(model_id, {}).get("anomaly_threshold", 3.0)
                if z_score > anomaly_threshold:
                    anomaly = {
                        "type": "anomaly",
                        "metric": metric_name,
                        "value": metric_value,
                        "z_score": z_score,
                        "mean": mean,
                        "std": std,
                        "threshold": anomaly_threshold,
                        "timestamp": timestamp
                    }
                    anomalies.append(anomaly)
                    
                    # Increment anomaly counter
                    self.model_anomaly_counter.labels(model_id=str(model_id), type=metric_name).inc()
            
            return anomalies
        except Exception as e:
            logger.error(f"Error checking for anomalies: {e}")
            return []
    
    def track_inference(self, model_id: int, input_data: Dict[str, Any],
                       prediction: Any, ground_truth: Optional[Any] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       latency_ms: Optional[float] = None) -> Dict[str, Any]:
        """Track a model inference for monitoring."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Record prediction
            timestamp = datetime.datetime.now().isoformat()
            
            # Check for feature drift
            drift_detected = False
            drift_features = []
            
            if model_id in self.drift_detectors:
                drift_results = self._check_feature_drift(model_id, input_data)
                drift_detected = drift_results["drift_detected"]
                drift_features = drift_results["drift_features"]
                
                # Update drift scores in detector
                self.drift_detectors[model_id]["drift_scores"] = drift_results["drift_scores"]
            
            # Record metrics
            if latency_ms is not None:
                self.model_latency_histogram.labels(model_id=str(model_id)).observe(latency_ms)
            
            # Increment prediction counter
            result = "unknown"
            if ground_truth is not None:
                if isinstance(prediction, (list, np.ndarray)) and isinstance(ground_truth, (list, np.ndarray)):
                    # For array predictions, check if they're equal
                    result = "correct" if np.array_equal(np.array(prediction), np.array(ground_truth)) else "incorrect"
                else:
                    # For scalar predictions
                    result = "correct" if prediction == ground_truth else "incorrect"
            
            self.model_prediction_counter.labels(model_id=str(model_id), result=result).inc()
            
            # Check for alerts
            alerts = []
            if drift_detected and model_id in self.alert_thresholds:
                drift_threshold = self.alert_thresholds[model_id].get("drift_threshold", 0.1)
                for feature, score in self.drift_detectors[model_id]["drift_scores"].items():
                    if score > drift_threshold:
                        alerts.append({
                            "type": "drift_alert",
                            "feature": feature,
                            "drift_score": score,
                            "threshold": drift_threshold,
                            "timestamp": timestamp
                        })
            
            # Send alerts if any
            if alerts:
                self._send_alerts(model_id, alerts)
            
            return {
                "status": "success",
                "model_id": model_id,
                "timestamp": timestamp,
                "drift_detected": drift_detected,
                "drift_features": drift_features,
                "alerts": alerts
            }
        except Exception as e:
            logger.error(f"Error tracking inference: {e}")
            return {"status": "error", "message": str(e)}
    
    def _check_feature_drift(self, model_id: int, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for feature drift in input data."""
        drift_detector = self.drift_detectors.get(model_id)
        if not drift_detector:
            return {"drift_detected": False, "drift_features": [], "drift_scores": {}}
        
        reference_stats = drift_detector["reference_stats"]
        features = drift_detector["features"]
        
        drift_scores = {}
        drift_features = []
        drift_detected = False
        
        for feature in features:
            if feature not in input_data:
                continue
            
            value = input_data[feature]
            ref_stats = reference_stats[feature]
            
            if ref_stats["type"] == "numeric":
                # Calculate z-score for numeric features
                z_score = abs((value - ref_stats["mean"]) / max(ref_stats["std"], 1e-10))
                drift_score = min(1.0, z_score / 10.0)  # Normalize to [0, 1]
            else:
                # For categorical features, check if value is in reference distribution
                if value in ref_stats["distribution"]:
                    # Use inverse of probability as drift score
                    drift_score = 1.0 - ref_stats["distribution"].get(value, 0)
                else:
                    # Value not seen in reference data
                    drift_score = 1.0
            
            drift_scores[feature] = drift_score
            
            # Check if drift exceeds threshold
            if drift_score > 0.5:  # Arbitrary threshold for demonstration
                drift_features.append(feature)
                drift_detected = True
            
            # Update drift gauge
            self.model_drift_gauge.labels(model_id=str(model_id), feature=feature).set(drift_score)
        
        return {
            "drift_detected": drift_detected,
            "drift_features": drift_features,
            "drift_scores": drift_scores
        }
    
    def _send_alerts(self, model_id: int, alerts: List[Dict[str, Any]]) -> None:
        """Send alerts to subscribers."""
        try:
            # Add alerts to history
            self.alert_history[model_id].extend(alerts)
            
            # Limit history size
            max_history = 1000
            if len(self.alert_history[model_id]) > max_history:
                self.alert_history[model_id] = self.alert_history[model_id][-max_history:]
            
            # Get subscribers for this model
            subscribers = self.alert_subscribers.get(model_id, [])
            
            if not subscribers:
                logger.info(f"No subscribers for model {model_id} alerts")
                return
            
            # Format alert message
            model_info = self.monitored_models.get(model_id, {})
            model_name = model_info.get("model_name", f"Model {model_id}")
            model_version = model_info.get("model_version", "unknown")
            
            alert_message = {
                "model_id": model_id,
                "model_name": model_name,
                "model_version": model_version,
                "timestamp": datetime.datetime.now().isoformat(),
                "alerts": alerts
            }
            
            # Send to each subscriber
            for subscriber in subscribers:
                try:
                    # If subscriber is a callback function
                    if callable(subscriber):
                        subscriber(alert_message)
                    # If subscriber is a webhook URL
                    elif isinstance(subscriber, str) and subscriber.startswith("http"):
                        threading.Thread(
                            target=self._send_webhook_alert,
                            args=(subscriber, alert_message),
                            daemon=True
                        ).start()
                except Exception as e:
                    logger.error(f"Error sending alert to subscriber: {e}")
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    def _send_webhook_alert(self, webhook_url: str, alert_message: Dict[str, Any]) -> None:
        """Send alert to webhook URL."""
        try:
            response = requests.post(
                webhook_url,
                json=alert_message,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                logger.info(f"Alert sent to webhook {webhook_url}")
            else:
                logger.warning(f"Failed to send alert to webhook {webhook_url}: {response.status_code}")
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def subscribe_to_alerts(self, model_id: int, subscriber: Union[Callable, str]) -> Dict[str, Any]:
        """Subscribe to alerts for a model."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Add subscriber
            self.alert_subscribers[model_id].append(subscriber)
            
            return {
                "status": "success",
                "message": f"Subscribed to alerts for model {model_id}",
                "subscriber_count": len(self.alert_subscribers[model_id])
            }
        except Exception as e:
            logger.error(f"Error subscribing to alerts: {e}")
            return {"status": "error", "message": str(e)}
    
    def unsubscribe_from_alerts(self, model_id: int, subscriber: Union[Callable, str]) -> Dict[str, Any]:
        """Unsubscribe from alerts for a model."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Remove subscriber
            if subscriber in self.alert_subscribers[model_id]:
                self.alert_subscribers[model_id].remove(subscriber)
            
            return {
                "status": "success",
                "message": f"Unsubscribed from alerts for model {model_id}",
                "subscriber_count": len(self.alert_subscribers[model_id])
            }
        except Exception as e:
            logger.error(f"Error unsubscribing from alerts: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_alert_history(self, model_id: int, limit: int = 100) -> Dict[str, Any]:
        """Get alert history for a model."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Get alert history
            alerts = self.alert_history.get(model_id, [])
            
            return {
                "status": "success",
                "model_id": model_id,
                "alert_count": len(alerts),
                "alerts": alerts[-limit:]
            }
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_performance_report(self, model_id: int, metrics: List[str] = None,
                              start_time: Optional[str] = None,
                              end_time: Optional[str] = None) -> Dict[str, Any]:
        """Generate a performance report for a model."""
        try:
            if model_id not in self.monitored_models:
                return {"status": "error", "message": f"Model {model_id} not registered for monitoring"}
            
            # Default to all metrics if not specified
            if not metrics:
                metrics = list(self.performance_history[model_id].keys())
            
            # Convert time strings to datetime objects
            start_dt = None
            end_dt = None
            if start_time:
                start_dt = datetime.datetime.fromisoformat(start_time)
            if end_time:
                end_dt = datetime.datetime.fromisoformat(end_time)
            
            # Collect metric data
            report_data = {}
            for metric in metrics:
                if metric not in self.performance_history[model_id]:
                    continue
                
                # Filter by time range if specified
                metric_data = self.performance_history[model_id][metric]
                if start_dt or end_dt:
                    filtered_data = []
                    for entry in metric_data:
                        entry_dt = datetime.datetime.fromisoformat(entry["timestamp"])
                        if start_dt and entry_dt < start_dt:
                            continue
                        if end_dt and entry_dt > end_dt:
                            continue
                        filtered_data.append(entry)
                    metric_data = filtered_data
                
                # Calculate statistics
                if metric_data:
                    values = [entry["value"] for entry in metric_data]
                    report_data[metric] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "median": np.median(values),
                        "p95": np.percentile(values, 95),
                        "p99": np.percentile(values, 99),
                        "latest": values[-1],
                        "trend": self._calculate_trend(values)
                    }
            
            # Get model info
            model_info = self.monitored_models[model_id]
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_name": model_info["model_name"],
                "model_version": model_info["model_version"],
                "report_time": datetime.datetime.now().isoformat(),
                "start_time": start_time,
                "end_time": end_time,
                "metrics": report_data
            }
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < window:
            return "insufficient_data"
        
        # Compare average of first half to average of second half
        half = len(values) // 2
        first_half_avg = np.mean(values[:half])
        second_half_avg = np.mean(values[half:])
        
        # Calculate percent change
        percent_change = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg != 0 else 0
        
        # Determine trend
        if abs(percent_change) < 0.01:  # Less than 1% change
            return "stable"
        elif percent_change > 0:
            return "improving" if first_half_avg < second_half_avg else "degrading"
        else:
            return "degrading" if first_half_avg > second_half_avg else "improving"
    
    def close(self):
        """Close the database session."""
        self.session.close()


#######################################################
# ComplianceManager Class
#######################################################

class ComplianceManager:
    """
    Manages compliance aspects of ML models, including security scanning,
    privacy-preserving techniques, regulatory compliance, and audit trails.
    """
    
    def __init__(self, db_session=None):
        self.session = db_session or Session()
        self.compliance_checks = {}
        self.privacy_settings = {}
        self.security_scans = {}
        self.audit_trails = defaultdict(list)
        
        # Initialize default privacy settings
        self.default_privacy_settings = {
            "differential_privacy": {
                "enabled": False,
                "epsilon": 1.0,
                "delta": 1e-5
            },
            "homomorphic_encryption": {
                "enabled": False,
                "scheme": "BFV",  # Brakerski/Fan-Vercauteren scheme
                "key_size": 2048
            },
            "federated_learning": {
                "enabled": False,
                "secure_aggregation": True
            },
            "data_minimization": {
                "enabled": True,
                "retention_days": 90
            },
            "pii_detection": {
                "enabled": True,
                "redaction": True
            }
        }
        
        # Initialize default compliance requirements
        self.default_compliance_requirements = {
            "gdpr": {
                "enabled": True,
                "data_subject_rights": True,
                "data_protection_impact_assessment": True,
                "breach_notification": True,
                "consent_management": True,
                "cross_border_transfers": False
            },
            "ccpa": {
                "enabled": True,
                "consumer_rights": True,
                "opt_out_rights": True,
                "disclosure_requirements": True
            },
            "hipaa": {
                "enabled": False,
                "phi_protection": True,
                "security_rule": True,
                "privacy_rule": True
            },
            "model_transparency": {
                "enabled": True,
                "explainability_reports": True,
                "bias_assessments": True,
                "version_control": True
            },
            "security_standards": {
                "enabled": True,
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "access_controls": True,
                "vulnerability_scanning": True
            }
        }
    
    def register_model_for_compliance(self, model_id: int, 
                                     privacy_settings: Optional[Dict[str, Any]] = None,
                                     compliance_requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a model for compliance monitoring."""
        try:
            # Get model
            model = self.session.query(MLModel).filter_by(id=model_id).first()
            if not model:
                return {"status": "error", "message": f"Model with ID {model_id} not found"}
            
            # Use default settings if not provided
            if not privacy_settings:
                privacy_settings = self.default_privacy_settings
            
            if not compliance_requirements:
                compliance_requirements = self.default_compliance_requirements
            
            # Store compliance configuration
            self.compliance_checks[model_id] = {
                "model_name": model.name,
                "model_version": model.version,
                "compliance_requirements": compliance_requirements,
                "last_compliance_check": None,
                "compliance_status": "pending",
                "registered_at": datetime.datetime.now().isoformat()
            }
            
            # Store privacy settings
            self.privacy_settings[model_id] = privacy_settings
            
            # Create initial audit trail entry
            self._add_audit_trail(
                model_id=model_id,
                action="register_compliance",
                details={
                    "privacy_settings": privacy_settings,
                    "compliance_requirements": compliance_requirements
                }
            )
            
            return {
                "status": "success",
                "message": f"Model {model.name} v{model.version} registered for compliance",
                "model_id": model_id,
                "privacy_settings": privacy_settings,
                "compliance_requirements": compliance_requirements
            }
        except Exception as e:
            logger.error(f"Error registering model for compliance: {e}")
            return {"status": "error", "message": str(e)}
    
    def scan_model_security(self, model_id: int) -> Dict[str, Any]:
        """Perform a security scan on a model."""
        try:
            if model_id not in self.compliance_checks:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Get model
            model = self.session.query(MLModel).filter_by(id=model_id).first()
            if not model:
                return {"status": "error", "message": f"Model with ID {model_id} not found"}
            
            # Simulate security scan
            scan_results = {
                "scan_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_id": model_id,
                "model_name": model.name,
                "model_version": model.version,
                "vulnerabilities": self._simulate_vulnerability_scan(),
                "dependency_check": self._simulate_dependency_check(),
                "code_quality": self._simulate_code_quality_check(),
                "overall_risk_level": "low"  # Could be low, medium, high, critical
            }
            
            # Calculate overall risk level
            vuln_count = sum(1 for v in scan_results["vulnerabilities"] if v["severity"] in ["high", "critical"])
            dep_vuln_count = sum(1 for d in scan_results["dependency_check"] if d["vulnerable"])
            
            if vuln_count > 3 or dep_vuln_count > 5:
                scan_results["overall_risk_level"] = "critical"
            elif vuln_count > 1 or dep_vuln_count > 2:
                scan_results["overall_risk_level"] = "high"
            elif vuln_count > 0 or dep_vuln_count > 0:
                scan_results["overall_risk_level"] = "medium"
            
            # Store scan results
            self.security_scans[model_id] = scan_results
            
            # Update compliance status
            self.compliance_checks[model_id]["last_security_scan"] = datetime.datetime.now().isoformat()
            if scan_results["overall_risk_level"] in ["high", "critical"]:
                self.compliance_checks[model_id]["compliance_status"] = "failed"
            
            # Add audit trail entry
            self._add_audit_trail(
                model_id=model_id,
                action="security_scan",
                details={
                    "scan_id": scan_results["scan_id"],
                    "overall_risk_level": scan_results["overall_risk_level"],
                    "vulnerability_count": len(scan_results["vulnerabilities"]),
                    "dependency_vulnerability_count": sum(1 for d in scan_results["dependency_check"] if d["vulnerable"])
                }
            )
            
            return {
                "status": "success",
                "scan_id": scan_results["scan_id"],
                "model_id": model_id,
                "timestamp": scan_results["timestamp"],
                "overall_risk_level": scan_results["overall_risk_level"],
                "vulnerability_count": len(scan_results["vulnerabilities"]),
                "dependency_vulnerability_count": sum(1 for d in scan_results["dependency_check"] if d["vulnerable"]),
                "full_report": scan_results
            }
        except Exception as e:
            logger.error(f"Error scanning model security: {e}")
            return {"status": "error", "message": str(e)}
    
    def _simulate_vulnerability_scan(self) -> List[Dict[str, Any]]:
        """Simulate a vulnerability scan."""
        vulnerabilities = []
        
        # Simulate finding 0-5 vulnerabilities
        import random
        vuln_count = random.randint(0, 5)
        
        severity_levels = ["low", "medium", "high", "critical"]
        vulnerability_types = [
            "Insecure Deserialization",
            "Cross-Site Scripting",
            "SQL Injection",
            "Path Traversal",
            "Remote Code Execution",
            "Authentication Bypass",
            "Sensitive Data Exposure",
            "Insufficient Logging & Monitoring",
            "Broken Access Control",
            "Security Misconfiguration"
        ]
        
        for _ in range(vuln_count):
            vulnerabilities.append({
                "id": f"VULN-{uuid.uuid4().hex[:8]}",
                "type": random.choice(vulnerability_types),
                "severity": random.choice(severity_levels),
                "description": f"Simulated vulnerability for testing purposes",
                "remediation": "Update to latest version and apply security patches"
            })
        
        return vulnerabilities
    
    def _simulate_dependency_check(self) -> List[Dict[str, Any]]:
        """Simulate a dependency check."""
        dependencies = []
        
        # Simulate 5-15 dependencies
        import random
        dep_count = random.randint(5, 15)
        
        for i in range(dep_count):
            # 20% chance of being vulnerable
            is_vulnerable = random.random() < 0.2
            
            dependencies.append({
                "name": f"dependency-{i}",
                "version": f"1.{random.randint(0, 9)}.{random.randint(0, 9)}",
                "vulnerable": is_vulnerable,
                "vulnerabilities": [
                    {
                        "id": f"CVE-2023-{random.randint(1000, 9999)}",
                        "severity": random.choice(["low", "medium", "high", "critical"]),
                        "fixed_version": f"1.{random.randint(0, 9)}.{random.randint(10, 20)}"
                    }
                ] if is_vulnerable else []
            })
        
        return dependencies
    
    def _simulate_code_quality_check(self) -> Dict[str, Any]:
        """Simulate a code quality check."""
        import random
        
        return {
            "maintainability_index": random.uniform(50, 100),
            "cyclomatic_complexity": random.uniform(5, 30),
            "code_duplication": random.uniform(0, 20),
            "test_coverage": random.uniform(60, 100),
            "issues": [
                {
                    "type": "code_smell",
                    "severity": random.choice(["low", "medium", "high"]),
                    "description": "Simulated code quality issue"
                }
                for _ in range(random.randint(0, 5))
            ]
        }
    
    def apply_differential_privacy(self, model_id: int, epsilon: float = 1.0,
                                  delta: float = 1e-5) -> Dict[str, Any]:
        """Apply differential privacy to a model."""
        try:
            if model_id not in self.privacy_settings:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Update privacy settings
            self.privacy_settings[model_id]["differential_privacy"] = {
                "enabled": True,
                "epsilon": epsilon,
                "delta": delta,
                "applied_at": datetime.datetime.now().isoformat()
            }
            
            # Add audit trail entry
            self._add_audit_trail(
                model_id=model_id,
                action="apply_differential_privacy",
                details={
                    "epsilon": epsilon,
                    "delta": delta
                }
            )
            
            return {
                "status": "success",
                "message": f"Differential privacy applied to model {model_id}",
                "epsilon": epsilon,
                "delta": delta
            }
        except Exception as e:
            logger.error(f"Error applying differential privacy: {e}")
            return {"status": "error", "message": str(e)}
    
    def apply_homomorphic_encryption(self, model_id: int, scheme: str = "BFV",
                                    key_size: int = 2048) -> Dict[str, Any]:
        """Apply homomorphic encryption to a model."""
        try:
            if model_id not in self.privacy_settings:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Validate scheme
            valid_schemes = ["BFV", "CKKS", "BGV", "TFHE"]
            if scheme not in valid_schemes:
                return {"status": "error", "message": f"Invalid scheme: {scheme}. Must be one of {valid_schemes}"}
            
            # Update privacy settings
            self.privacy_settings[model_id]["homomorphic_encryption"] = {
                "enabled": True,
                "scheme": scheme,
                "key_size": key_size,
                "applied_at": datetime.datetime.now().isoformat()
            }
            
            # Add audit trail entry
            self._add_audit_trail(
                model_id=model_id,
                action="apply_homomorphic_encryption",
                details={
                    "scheme": scheme,
                    "key_size": key_size
                }
            )
            
            return {
                "status": "success",
                "message": f"Homomorphic encryption applied to model {model_id}",
                "scheme": scheme,
                "key_size": key_size
            }
        except Exception as e:
            logger.error(f"Error applying homomorphic encryption: {e}")
            return {"status": "error", "message": str(e)}
    
    def perform_gdpr_compliance_check(self, model_id: int) -> Dict[str, Any]:
        """Perform a GDPR compliance check on a model."""
        try:
            if model_id not in self.compliance_checks:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Get model
            model = self.session.query(MLModel).filter_by(id=model_id).first()
            if not model:
                return {"status": "error", "message": f"Model with ID {model_id} not found"}
            
            # Check if GDPR compliance is enabled
            if not self.compliance_checks[model_id]["compliance_requirements"].get("gdpr", {}).get("enabled", False):
                return {"status": "error", "message": f"GDPR compliance not enabled for model {model_id}"}
            
            # Simulate GDPR compliance check
            check_results = {
                "check_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_id": model_id,
                "model_name": model.name,
                "model_version": model.version,
                "regulation": "GDPR",
                "checks": self._simulate_gdpr_checks(),
                "overall_status": "compliant"  # Could be compliant, partially_compliant, non_compliant
            }
            
            # Calculate overall status
            failed_checks = sum(1 for c in check_results["checks"] if not c["compliant"])
            total_checks = len(check_results["checks"])
            
            if failed_checks == 0:
                check_results["overall_status"] = "compliant"
            elif failed_checks < total_checks / 2:
                check_results["overall_status"] = "partially_compliant"
            else:
                check_results["overall_status"] = "non_compliant"
            
            # Update compliance status
            self.compliance_checks[model_id]["last_gdpr_check"] = datetime.datetime.now().isoformat()
            self.compliance_checks[model_id]["gdpr_status"] = check_results["overall_status"]
            
            if check_results["overall_status"] == "non_compliant":
                self.compliance_checks[model_id]["compliance_status"] = "failed"
            
            # Add audit trail entry
            self._add_audit_trail(
                model_id=model_id,
                action="gdpr_compliance_check",
                details={
                    "check_id": check_results["check_id"],
                    "overall_status": check_results["overall_status"],
                    "failed_checks": failed_checks,
                    "total_checks": total_checks
                }
            )
            
            return {
                "status": "success",
                "check_id": check_results["check_id"],
                "model_id": model_id,
                "timestamp": check_results["timestamp"],
                "regulation": "GDPR",
                "overall_status": check_results["overall_status"],
                "failed_checks": failed_checks,
                "total_checks": total_checks,
                "full_report": check_results
            }
        except Exception as e:
            logger.error(f"Error performing GDPR compliance check: {e}")
            return {"status": "error", "message": str(e)}
    
    def _simulate_gdpr_checks(self) -> List[Dict[str, Any]]:
        """Simulate GDPR compliance checks."""
        import random
        
        checks = [
            {
                "id": "gdpr-1",
                "name": "Data Minimization",
                "description": "Check if only necessary data is collected and processed",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement data minimization practices" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-2",
                "name": "Lawful Basis for Processing",
                "description": "Check if there is a lawful basis for processing personal data",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Establish clear lawful basis for processing" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-3",
                "name": "Data Subject Rights",
                "description": "Check if data subject rights (access, rectification, erasure, etc.) are supported",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement data subject rights mechanisms" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-4",
                "name": "Data Protection by Design",
                "description": "Check if data protection is considered from the design phase",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement privacy by design principles" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-5",
                "name": "Data Protection Impact Assessment",
                "description": "Check if a DPIA has been conducted",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Conduct a comprehensive DPIA" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-6",
                "name": "Cross-Border Data Transfers",
                "description": "Check if cross-border data transfers comply with GDPR requirements",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement appropriate safeguards for cross-border transfers" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-7",
                "name": "Data Breach Notification",
                "description": "Check if data breach notification procedures are in place",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Establish data breach notification procedures" if random.random() > 0.8 else None
            },
            {
                "id": "gdpr-8",
                "name": "Records of Processing Activities",
                "description": "Check if records of processing activities are maintained",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Maintain comprehensive records of processing activities" if random.random() > 0.8 else None
            }
        ]
        
        return checks
    
    def perform_ccpa_compliance_check(self, model_id: int) -> Dict[str, Any]:
        """Perform a CCPA compliance check on a model."""
        try:
            if model_id not in self.compliance_checks:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Get model
            model = self.session.query(MLModel).filter_by(id=model_id).first()
            if not model:
                return {"status": "error", "message": f"Model with ID {model_id} not found"}
            
            # Check if CCPA compliance is enabled
            if not self.compliance_checks[model_id]["compliance_requirements"].get("ccpa", {}).get("enabled", False):
                return {"status": "error", "message": f"CCPA compliance not enabled for model {model_id}"}
            
            # Simulate CCPA compliance check
            check_results = {
                "check_id": str(uuid.uuid4()),
                "timestamp": datetime.datetime.now().isoformat(),
                "model_id": model_id,
                "model_name": model.name,
                "model_version": model.version,
                "regulation": "CCPA",
                "checks": self._simulate_ccpa_checks(),
                "overall_status": "compliant"  # Could be compliant, partially_compliant, non_compliant
            }
            
            # Calculate overall status
            failed_checks = sum(1 for c in check_results["checks"] if not c["compliant"])
            total_checks = len(check_results["checks"])
            
            if failed_checks == 0:
                check_results["overall_status"] = "compliant"
            elif failed_checks < total_checks / 2:
                check_results["overall_status"] = "partially_compliant"
            else:
                check_results["overall_status"] = "non_compliant"
            
            # Update compliance status
            self.compliance_checks[model_id]["last_ccpa_check"] = datetime.datetime.now().isoformat()
            self.compliance_checks[model_id]["ccpa_status"] = check_results["overall_status"]
            
            if check_results["overall_status"] == "non_compliant":
                self.compliance_checks[model_id]["compliance_status"] = "failed"
            
            # Add audit trail entry
            self._add_audit_trail(
                model_id=model_id,
                action="ccpa_compliance_check",
                details={
                    "check_id": check_results["check_id"],
                    "overall_status": check_results["overall_status"],
                    "failed_checks": failed_checks,
                    "total_checks": total_checks
                }
            )
            
            return {
                "status": "success",
                "check_id": check_results["check_id"],
                "model_id": model_id,
                "timestamp": check_results["timestamp"],
                "regulation": "CCPA",
                "overall_status": check_results["overall_status"],
                "failed_checks": failed_checks,
                "total_checks": total_checks,
                "full_report": check_results
            }
        except Exception as e:
            logger.error(f"Error performing CCPA compliance check: {e}")
            return {"status": "error", "message": str(e)}
    
    def _simulate_ccpa_checks(self) -> List[Dict[str, Any]]:
        """Simulate CCPA compliance checks."""
        import random
        
        checks = [
            {
                "id": "ccpa-1",
                "name": "Right to Know",
                "description": "Check if consumers can request disclosure of personal information collected",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement right to know request mechanism" if random.random() > 0.8 else None
            },
            {
                "id": "ccpa-2",
                "name": "Right to Delete",
                "description": "Check if consumers can request deletion of personal information",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement right to delete request mechanism" if random.random() > 0.8 else None
            },
            {
                "id": "ccpa-3",
                "name": "Right to Opt-Out",
                "description": "Check if consumers can opt-out of the sale of personal information",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement opt-out mechanism" if random.random() > 0.8 else None
            },
            {
                "id": "ccpa-4",
                "name": "Non-Discrimination",
                "description": "Check if consumers are not discriminated against for exercising their rights",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Review and update non-discrimination policies" if random.random() > 0.8 else None
            },
            {
                "id": "ccpa-5",
                "name": "Privacy Notice",
                "description": "Check if privacy notice includes required CCPA disclosures",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Update privacy notice with CCPA disclosures" if random.random() > 0.8 else None
            },
            {
                "id": "ccpa-6",
                "name": "Verification Process",
                "description": "Check if there is a process to verify consumer requests",
                "compliant": random.random() > 0.2,
                "evidence": "Simulated evidence for testing purposes",
                "remediation": "Implement robust verification process" if random.random() > 0.8 else None
            }
        ]
        
        return checks
    
    def _add_audit_trail(self, model_id: int, action: str, 
                        details: Optional[Dict[str, Any]] = None) -> None:
        """Add an entry to the audit trail."""
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            audit_entry = {
                "timestamp": timestamp,
                "action": action,
                "model_id": model_id,
                "details": details or {}
            }
            
            # Add to in-memory audit trail
            self.audit_trails[model_id].append(audit_entry)
            
            # Limit audit trail size
            max_trail_size = 1000
            if len(self.audit_trails[model_id]) > max_trail_size:
                self.audit_trails[model_id] = self.audit_trails[model_id][-max_trail_size:]
            
            # Add to database audit log
            try:
                audit_log = AuditLog(
                    action=action,
                    entity_type="model",
                    entity_id=model_id,
                    details=json.dumps(details) if details else None
                )
                
                self.session.add(audit_log)
                self.session.commit()
            except Exception as e:
                logger.error(f"Error adding to database audit log: {e}")
                self.session.rollback()
        except Exception as e:
            logger.error(f"Error adding audit trail: {e}")
    
    def get_audit_trail(self, model_id: int, limit: int = 100) -> Dict[str, Any]:
        """Get the audit trail for a model."""
        try:
            if model_id not in self.compliance_checks:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Get audit trail
            audit_trail = self.audit_trails.get(model_id, [])
            
            return {
                "status": "success",
                "model_id": model_id,
                "audit_count": len(audit_trail),
                "audit_trail": audit_trail[-limit:]
            }
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_compliance_status(self, model_id: int) -> Dict[str, Any]:
        """Get the compliance status for a model."""
        try:
            if model_id not in self.compliance_checks:
                return {"status": "error", "message": f"Model {model_id} not registered for compliance"}
            
            # Get compliance status
            compliance_status = self.compliance_checks[model_id]
            
            # Get privacy settings
            privacy_settings = self.privacy_settings.get(model_id, {})
            
            # Get security scan results
            security_scan = self.security_scans.get(model_id, {})
            
            return {
                "status": "success",
                "model_id": model_id,
                "model_name": compliance_status["model_name"],
                "model_version": compliance_status["model_version"],
                "compliance_status": compliance_status["compliance_status"],
                "last_compliance_check": compliance_status["last_compliance_check"],
                "gdpr_status": compliance_status.get("gdpr_status", "unknown"),
                "ccpa_status": compliance_status.get("ccpa_status", "unknown"),
                "security_risk_level": security_scan.get("overall_risk_level", "unknown"),
                "privacy_settings": {
                    "differential_privacy": privacy_settings.get("differential_privacy", {}).get("enabled", False),
                    "homomorphic_encryption": privacy_settings.get("homomorphic_encryption", {}).get("enabled", False),
                    "federated_learning": privacy_settings.get("federated_learning", {}).get("enabled", False),
                    "data_minimization": privacy_settings.get("data_minimization", {}).get("enabled", False),
                    "pii_detection": privacy_settings.get("pii_detection", {}).get("enabled", False)
                }
            }
        except Exception as e:
            logger.error(f"Error getting compliance status: {e}")
            return {"status": "error", "message": str(e)}
    
    def close(self):
        """Close the database session."""
        self.session.close()


#######################################################
# Authentication and Authorization
#######################################################

# Pydantic models for authentication
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None
    employee_id: Optional[int] = None


class UserInDB(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    hashed_password: str
    role: str = "user"
    employee_id: Optional[int] = None


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Security bearer scheme for API keys
security = HTTPBearer()

# In-memory user database (replace with database in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@skyscope.ai",
        "hashed_password": pwd_context.hash("admin"),
        "disabled": False,
        "role": "admin",
        "employee_id": 1
    },
    "user": {
        "username": "user",
        "full_name": "Regular User",
        "email": "user@skyscope.ai",
        "hashed_password": pwd_context.hash("user"),
        "disabled": False,
        "role": "user",
        "employee_id": 2
    }
}


def verify_password(plain_password, hashed_password):
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Get password hash."""
    return pwd_context.hash(password)


def get_user(db, username: str):
    """Get user from database."""
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(fake_db, username: str, password: str):
    """Authenticate user."""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"), employee_id=payload.get("employee_id"))
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


#######################################################
# Rate Limiting Middleware
#######################################################

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, limit: int = 100, window: int = 60):
        self.limit = limit  # Number of requests
        self.window = window  # Time window in seconds
        self.requests = defaultdict(list)
        self.lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed based on rate limit."""
        with self.lock:
            now = time.time()
            
            # Remove old requests
            self.requests[key] = [req_time for req_time in self.requests[key] if now - req_time < self.window]
            
            # Check if limit is reached
            if len(self.requests[key]) >= self.limit:
                return False
            
            # Add current request
            self.requests[key].append(now)
            return True


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""
    
    def __init__(self, app, limiter: RateLimiter):
        super().__init__(app)
        self.limiter = limiter
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host
        
        # Check rate limit
        if not self.limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"}
            )
        
        # Process request
        response = await call_next(request)
        return response


#######################################################
# FastAPI Application
#######################################################

# Create FastAPI app
app = FastAPI(
    title="Skyscope ML Deployment and Management API",
    description="API for managing ML models, employees, and deployments",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
rate_limiter = RateLimiter(limit=100, window=60)
app.add_middleware(RateLimitMiddleware, limiter=rate_limiter)

# Create managers
employee_manager = None
model_manager = None
openai_manager = None
kubernetes_manager = None
monitoring_manager = None
compliance_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize managers on startup."""
    global employee_manager, model_manager, openai_manager, kubernetes_manager, monitoring_manager, compliance_manager
    
    try:
        # Create database session
        session = Session()
        
        # Initialize managers
        employee_manager = EmployeeHierarchyManager(session) if LOCAL_IMPORTS_AVAILABLE else None
        model_manager = MLModelManager(session) if LOCAL_IMPORTS_AVAILABLE else None
        openai_manager = OpenAIUnofficialManager() if LOCAL_IMPORTS_AVAILABLE else None
        kubernetes_manager = KubernetesDeploymentManager() if LOCAL_IMPORTS_AVAILABLE else None
        monitoring_manager = ModelMonitoringManager(session)
        compliance_manager = ComplianceManager(session)
        
        # Start Prometheus metrics server
        start_http_server(8000)
        
        logger.info("Managers initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing managers: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global employee_manager, model_manager, openai_manager, kubernetes_manager, monitoring_manager, compliance_manager
    
    try:
        # Close database sessions
        if employee_manager:
            employee_manager.close()
        
        if model_manager:
            model_manager.close()
        
        if monitoring_manager:
            monitoring_manager.close()
        
        if compliance_manager:
            compliance_manager.close()
        
        logger.info("Managers closed successfully")
    except Exception as e:
        logger.error(f"Error closing managers: {e}")


#######################################################
# Authentication Endpoints
#######################################################

@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get access token."""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role, "employee_id": user.employee_id},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/users/me", response_model=dict)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    """Get current user info."""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role,
        "employee_id": current_user.employee_id
    }


#######################################################
# Employee Endpoints
#######################################################

@app.post("/api/employees", status_code=status.HTTP_201_CREATED)
async def create_employee(
    employee: dict,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Create a new employee."""
    if not employee_manager:
        raise HTTPException(status_code=501, detail="Employee management not available")
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create employees")
    
    result = employee_manager.create_employee(employee)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@app.post("/api/teams", status_code=status.HTTP_201_CREATED)
async def create_team(
    team: dict,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Create a new team."""
    if not employee_manager:
        raise HTTPException(status_code=501, detail="Employee management not available")
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create teams")
    
    result = employee_manager.create_team(team)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@app.put("/api/employees/{employee_id}/team/{team_id}")
async def assign_employee_to_team(
    employee_id: int,
    team_id: int,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """Assign an employee to a team."""
    if not employee_manager:
        raise HTTPException(status_code=501, detail="Employee management not available")
    
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to assign employees")
    
    result = employee_manager.assign_employee_to_team(employee_id, team_id)
    if not result:
        raise HTTPException(status_code=400, detail="Failed to assign employee to team")
    
    return {"status": "success", "message": f"Employee {employee_id} assigned to team {team_id}"}


@app.put("/api/teams/{team_id}/manager/{employee_id}")
async def assign_manager_to_team(
    team_id: int,
    employee_id: int,
    current_user: UserInDB = Depends(get_current_active_user)
):
    """