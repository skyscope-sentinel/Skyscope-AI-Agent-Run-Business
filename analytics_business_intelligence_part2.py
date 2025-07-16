#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analytics_business_intelligence_part2.py - Advanced Analytics and Business Intelligence System (Part 2)

This module continues the comprehensive analytics and business intelligence system for the
Skyscope Sentinel Intelligence AI platform, implementing advanced predictive analytics,
anomaly detection, pattern recognition, OLAP analysis, and executive dashboards.

Features:
1. Ensemble forecasting methods
2. Anomaly detection algorithms
3. Pattern recognition and data mining
4. Multi-dimensional OLAP analysis
5. GPT-4o integration for automated insights
6. Executive dashboard management
7. KPI tracking and alerting
8. Reporting automation
9. Real-time streaming analytics
10. Data visualization for business metrics

Part of Skyscope Sentinel Intelligence AI - ITERATION 12
"""

import asyncio
import base64
import datetime
import io
import json
import logging
import os
import re
import signal
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator

# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask import Flask, request, jsonify, send_file

# Data processing and ML
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from prophet import Prophet
from sklearn.ensemble import IsolationForest, RandomForestRegressor as SkRandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import lightgbm as lgb
import xgboost as xgb
import shap
import umap
import hdbscan

# Database and storage
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, func, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import pymongo
import redis
from kafka import KafkaProducer, KafkaConsumer

# Web and API
import aiohttp
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

# Visualization and reporting
import networkx as nx
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
import pydeck as pdk
import openpyxl
import xlsxwriter
from jinja2 import Environment, FileSystemLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analytics_bi_part2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import openai-unofficial for GPT-4o integration
try:
    import openai_unofficial
    from openai_unofficial import OpenAI
    OPENAI_UNOFFICIAL_AVAILABLE = True
except ImportError:
    OPENAI_UNOFFICIAL_AVAILABLE = False
    logger.warning("openai-unofficial package not available. Using standard OpenAI package as fallback.")
    try:
        import openai
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        logger.warning("OpenAI package not found. GPT-4o insights generation will be limited.")

# Constants
DEFAULT_MODEL = "gpt-4o-2024-05-13"
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///analytics_bi.db")
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
MAX_EMPLOYEES = 10000
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8050"))
DATA_WAREHOUSE_PATH = Path("./data_warehouse")
REPORTS_PATH = Path("./reports")

# Create necessary directories
DATA_WAREHOUSE_PATH.mkdir(exist_ok=True)
REPORTS_PATH.mkdir(exist_ok=True)

# SQLAlchemy Base
Base = declarative_base()

#######################################################
# Predictive Analytics and Forecasting (Continued)
#######################################################

class PredictiveAnalytics:
    """
    Predictive analytics and forecasting for business metrics.
    Implements time series forecasting, anomaly detection, and pattern recognition.
    """
    
    def __init__(self):
        """Initialize the predictive analytics module."""
        self.models = {}
        self.forecasts = {}
        self.anomaly_detectors = {}
    
    def train_ensemble_forecast(self, df: pd.DataFrame, date_col: str, value_col: str,
                              model_name: str, forecast_periods: int = 30,
                              models: List[str] = None) -> Dict[str, Any]:
        """
        Train an ensemble forecasting model combining multiple forecasting methods.
        
        Args:
            df: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            model_name: Name for the model
            forecast_periods: Number of periods to forecast
            models: List of model types to include in ensemble ('prophet', 'arima', 'lstm')
            
        Returns:
            Dictionary with model and forecast results
        """
        try:
            if models is None:
                models = ['prophet', 'arima', 'lstm']
            
            forecasts = {}
            
            # Train individual models
            if 'prophet' in models:
                prophet_result = self.train_prophet_model(
                    df, date_col, value_col, f"{model_name}_prophet", 
                    forecast_periods=forecast_periods
                )
                if 'error' not in prophet_result:
                    forecasts['prophet'] = prophet_result['forecast']
            
            if 'arima' in models:
                arima_result = self.train_arima_model(
                    df, date_col, value_col, f"{model_name}_arima", 
                    forecast_periods=forecast_periods
                )
                if 'error' not in arima_result:
                    forecasts['arima'] = arima_result['forecast']
            
            if 'lstm' in models:
                lstm_result = self.train_lstm_model(
                    df, date_col, value_col, f"{model_name}_lstm", 
                    forecast_periods=forecast_periods
                )
                if 'error' not in lstm_result:
                    forecasts['lstm'] = lstm_result['forecast']
            
            if not forecasts:
                return {'error': 'No models were successfully trained'}
            
            # Combine forecasts
            ensemble_forecast = pd.DataFrame()
            
            # Get common date range for forecasts
            all_dates = set()
            for model_type, forecast in forecasts.items():
                all_dates.update(forecast['ds'].tolist())
            
            all_dates = sorted(all_dates)
            ensemble_forecast['ds'] = all_dates
            
            # Add individual model forecasts
            for model_type, forecast in forecasts.items():
                # Merge with ensemble forecast
                forecast_subset = forecast[['ds', 'yhat']].copy()
                forecast_subset.columns = ['ds', f'yhat_{model_type}']
                ensemble_forecast = pd.merge(ensemble_forecast, forecast_subset, on='ds', how='left')
            
            # Calculate ensemble forecast (simple average)
            yhat_columns = [col for col in ensemble_forecast.columns if col.startswith('yhat_')]
            ensemble_forecast['yhat'] = ensemble_forecast[yhat_columns].mean(axis=1)
            
            # Calculate confidence interval (simplified)
            ensemble_forecast['yhat_lower'] = ensemble_forecast[yhat_columns].min(axis=1)
            ensemble_forecast['yhat_upper'] = ensemble_forecast[yhat_columns].max(axis=1)
            
            # Add actual values if available
            ensemble_forecast = pd.merge(
                ensemble_forecast, 
                df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'}),
                on='ds', how='left'
            )
            
            # Store model and forecast
            self.models[model_name] = {
                'model': 'ensemble',
                'type': 'ensemble',
                'component_models': list(forecasts.keys()),
                'date_col': date_col,
                'value_col': value_col,
                'trained_at': datetime.datetime.now()
            }
            
            self.forecasts[model_name] = ensemble_forecast
            
            # Calculate accuracy metrics on training data
            mask = ~ensemble_forecast['y'].isna()
            if mask.sum() > 0:
                y_true = ensemble_forecast.loc[mask, 'y'].values
                y_pred = ensemble_forecast.loc[mask, 'yhat'].values
                
                metrics = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred)
                }
            else:
                metrics = {}
            
            # Prepare result
            result = {
                'model_name': model_name,
                'forecast': ensemble_forecast,
                'metrics': metrics,
                'component_models': list(forecasts.keys())
            }
            
            # Store model in database
            self._store_model_in_db(model_name, 'ensemble', metrics)
            
            # Store forecasts in database
            self._store_forecasts_in_db(model_name, ensemble_forecast)
            
            return result
        
        except Exception as e:
            logger.error(f"Error training ensemble forecast model: {e}")
            return {'error': str(e)}
    
    def _store_model_in_db(self, model_name: str, model_type: str, metrics: Dict[str, float]) -> bool:
        """
        Store model metadata in the database.
        
        Args:
            model_name: Name of the model
            model_type: Type of the model
            metrics: Dictionary with accuracy metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with Session() as session:
                # Check if model already exists
                model = session.query(ForecastModel).filter_by(name=model_name).first()
                
                if model:
                    # Update existing model
                    model.model_type = model_type
                    model.parameters = json.dumps(metrics)
                    model.accuracy = metrics.get('r2', 0.0)
                    model.last_trained = datetime.datetime.now()
                else:
                    # Create new model
                    model = ForecastModel(
                        name=model_name,
                        metric_name=model_name.split('_')[0],  # Simplified approach
                        model_type=model_type,
                        parameters=json.dumps(metrics),
                        accuracy=metrics.get('r2', 0.0),
                        last_trained=datetime.datetime.now()
                    )
                    session.add(model)
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing model in database: {e}")
            return False
    
    def _store_forecasts_in_db(self, model_name: str, forecast_df: pd.DataFrame) -> bool:
        """
        Store forecasts in the database.
        
        Args:
            model_name: Name of the model
            forecast_df: DataFrame with forecasts
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with Session() as session:
                # Get model ID
                model = session.query(ForecastModel).filter_by(name=model_name).first()
                
                if not model:
                    logger.error(f"Model {model_name} not found in database")
                    return False
                
                # Delete existing forecasts for this model
                session.query(Forecast).filter_by(model_id=model.id).delete()
                
                # Add new forecasts
                for _, row in forecast_df.iterrows():
                    if pd.isna(row.get('y', None)):  # Only store forecasts, not historical data
                        forecast = Forecast(
                            model_id=model.id,
                            timestamp=row['ds'],
                            value=row['yhat'],
                            lower_bound=row.get('yhat_lower'),
                            upper_bound=row.get('yhat_upper'),
                            created_at=datetime.datetime.now()
                        )
                        session.add(forecast)
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing forecasts in database: {e}")
            return False
    
    def get_forecast(self, model_name: str) -> pd.DataFrame:
        """
        Get forecast for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with forecast
        """
        if model_name in self.forecasts:
            return self.forecasts[model_name]
        else:
            logger.error(f"Forecast for model {model_name} not found")
            return pd.DataFrame()
    
    def get_model_metrics(self, model_name: str) -> Dict[str, float]:
        """
        Get accuracy metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with accuracy metrics
        """
        try:
            with Session() as session:
                model = session.query(ForecastModel).filter_by(name=model_name).first()
                
                if model and model.parameters:
                    return json.loads(model.parameters)
                else:
                    logger.error(f"Model {model_name} not found in database")
                    return {}
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {}
    
    def detect_anomalies_isolation_forest(self, df: pd.DataFrame, value_col: str,
                                        contamination: float = 0.05,
                                        detector_name: str = None) -> Dict[str, Any]:
        """
        Detect anomalies using Isolation Forest algorithm.
        
        Args:
            df: DataFrame with data
            value_col: Column name for values
            contamination: Expected proportion of anomalies
            detector_name: Name for the detector
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Generate detector name if not provided
            if detector_name is None:
                detector_name = f"isolation_forest_{value_col}"
            
            # Prepare data
            X = df[value_col].values.reshape(-1, 1)
            
            # Create and train detector
            detector = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Fit and predict
            y_pred = detector.fit_predict(X)
            scores = detector.decision_function(X)
            
            # Convert predictions (-1 for anomalies, 1 for normal)
            anomalies = y_pred == -1
            
            # Create result DataFrame
            result_df = df.copy()
            result_df['anomaly'] = anomalies
            result_df['anomaly_score'] = scores
            
            # Store detector
            self.anomaly_detectors[detector_name] = {
                'detector': detector,
                'type': 'isolation_forest',
                'value_col': value_col,
                'contamination': contamination,
                'trained_at': datetime.datetime.now()
            }
            
            # Store anomalies in database
            self._store_anomalies_in_db(result_df, value_col, 'isolation_forest')
            
            return {
                'detector_name': detector_name,
                'anomalies': result_df[anomalies],
                'anomaly_count': anomalies.sum(),
                'total_count': len(df),
                'anomaly_percentage': anomalies.mean() * 100
            }
        
        except Exception as e:
            logger.error(f"Error detecting anomalies with Isolation Forest: {e}")
            return {'error': str(e)}
    
    def detect_anomalies_zscore(self, df: pd.DataFrame, value_col: str,
                              date_col: str = None, threshold: float = 3.0,
                              detector_name: str = None) -> Dict[str, Any]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            df: DataFrame with data
            value_col: Column name for values
            date_col: Column name for dates (optional)
            threshold: Z-score threshold for anomalies
            detector_name: Name for the detector
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Generate detector name if not provided
            if detector_name is None:
                detector_name = f"zscore_{value_col}"
            
            # Prepare data
            values = df[value_col].values
            
            # Calculate Z-scores
            mean = np.mean(values)
            std = np.std(values)
            z_scores = np.abs((values - mean) / std)
            
            # Detect anomalies
            anomalies = z_scores > threshold
            
            # Create result DataFrame
            result_df = df.copy()
            result_df['anomaly'] = anomalies
            result_df['anomaly_score'] = z_scores
            
            # Store detector
            self.anomaly_detectors[detector_name] = {
                'detector': 'zscore',
                'type': 'zscore',
                'value_col': value_col,
                'threshold': threshold,
                'mean': mean,
                'std': std,
                'trained_at': datetime.datetime.now()
            }
            
            # Store anomalies in database
            self._store_anomalies_in_db(result_df, value_col, 'zscore')
            
            return {
                'detector_name': detector_name,
                'anomalies': result_df[anomalies],
                'anomaly_count': anomalies.sum(),
                'total_count': len(df),
                'anomaly_percentage': anomalies.mean() * 100
            }
        
        except Exception as e:
            logger.error(f"Error detecting anomalies with Z-score: {e}")
            return {'error': str(e)}
    
    def detect_anomalies_prophet(self, df: pd.DataFrame, date_col: str, value_col: str,
                               interval_width: float = 0.99,
                               detector_name: str = None) -> Dict[str, Any]:
        """
        Detect anomalies using Prophet forecasting model.
        
        Args:
            df: DataFrame with data
            date_col: Column name for dates
            value_col: Column name for values
            interval_width: Width of prediction intervals
            detector_name: Name for the detector
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Generate detector name if not provided
            if detector_name is None:
                detector_name = f"prophet_{value_col}"
            
            # Prepare data for Prophet
            prophet_df = df[[date_col, value_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Create and train Prophet model
            model = Prophet(interval_width=interval_width)
            model.fit(prophet_df)
            
            # Make predictions for historical data
            forecast = model.predict(prophet_df[['ds']])
            
            # Merge with actual data
            result_df = pd.merge(prophet_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')
            
            # Detect anomalies
            anomalies = (result_df['y'] < result_df['yhat_lower']) | (result_df['y'] > result_df['yhat_upper'])
            
            # Calculate anomaly scores (distance from prediction normalized by interval width)
            result_df['anomaly'] = anomalies
            result_df['anomaly_score'] = np.maximum(
                (result_df['y'] - result_df['yhat_upper']) / (result_df['yhat_upper'] - result_df['yhat']),
                (result_df['yhat_lower'] - result_df['y']) / (result_df['yhat'] - result_df['yhat_lower'])
            )
            result_df.loc[~anomalies, 'anomaly_score'] = 0
            
            # Store detector
            self.anomaly_detectors[detector_name] = {
                'detector': model,
                'type': 'prophet',
                'date_col': date_col,
                'value_col': value_col,
                'interval_width': interval_width,
                'trained_at': datetime.datetime.now()
            }
            
            # Store anomalies in database
            self._store_anomalies_in_db(result_df, value_col, 'prophet')
            
            return {
                'detector_name': detector_name,
                'anomalies': result_df[anomalies],
                'anomaly_count': anomalies.sum(),
                'total_count': len(df),
                'anomaly_percentage': anomalies.mean() * 100,
                'forecast': forecast
            }
        
        except Exception as e:
            logger.error(f"Error detecting anomalies with Prophet: {e}")
            return {'error': str(e)}
    
    def _store_anomalies_in_db(self, result_df: pd.DataFrame, metric_name: str, 
                             detection_method: str) -> bool:
        """
        Store detected anomalies in the database.
        
        Args:
            result_df: DataFrame with anomaly detection results
            metric_name: Name of the metric
            detection_method: Method used for anomaly detection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with Session() as session:
                # Filter anomalies
                anomalies_df = result_df[result_df['anomaly']]
                
                for _, row in anomalies_df.iterrows():
                    # Calculate expected value if available
                    expected_value = row.get('yhat', None)
                    
                    anomaly = AnomalyDetection(
                        metric_name=metric_name,
                        metric_value=row[metric_name],
                        expected_value=expected_value,
                        deviation_score=row['anomaly_score'],
                        is_anomaly=1,
                        detection_method=detection_method,
                        timestamp=row.get('ds', datetime.datetime.now())
                    )
                    session.add(anomaly)
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing anomalies in database: {e}")
            return False
    
    def find_patterns_clustering(self, df: pd.DataFrame, feature_cols: List[str],
                               n_clusters: int = 5, method: str = 'kmeans') -> Dict[str, Any]:
        """
        Find patterns in data using clustering algorithms.
        
        Args:
            df: DataFrame with data
            feature_cols: List of column names for features
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'dbscan', 'hdbscan')
            
        Returns:
            Dictionary with clustering results
        """
        try:
            # Prepare data
            X = df[feature_cols].values
            
            # Scale features
            scaler = SkStandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply clustering
            if method == 'kmeans':
                model = SkKMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X_scaled)
                
                # Calculate cluster centers
                cluster_centers = scaler.inverse_transform(model.cluster_centers_)
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(X_scaled, clusters)
                
                # Calculate inertia
                inertia = model.inertia_
                
                # Create result DataFrame
                result_df = df.copy()
                result_df['cluster'] = clusters
                
                # Calculate cluster statistics
                cluster_stats = result_df.groupby('cluster').agg({
                    feature_cols[0]: ['mean', 'std', 'min', 'max', 'count']
                })
                
                return {
                    'method': 'kmeans',
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_avg,
                    'inertia': inertia,
                    'cluster_centers': cluster_centers,
                    'clusters': result_df,
                    'cluster_stats': cluster_stats
                }
            
            elif method == 'dbscan':
                from sklearn.cluster import DBSCAN
                
                # Find optimal eps using nearest neighbors
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=2)
                nn.fit(X_scaled)
                distances, _ = nn.kneighbors(X_scaled)
                distances = np.sort(distances[:, 1])
                
                # Use knee point as eps
                from kneed import KneeLocator
                knee_locator = KneeLocator(
                    range(len(distances)), distances, 
                    curve='convex', direction='increasing'
                )
                eps = distances[knee_locator.knee] if knee_locator.knee else 0.5
                
                # Apply DBSCAN
                model = DBSCAN(eps=eps, min_samples=5)
                clusters = model.fit_predict(X_scaled)
                
                # Create result DataFrame
                result_df = df.copy()
                result_df['cluster'] = clusters
                
                # Calculate cluster statistics
                n_clusters_actual = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                return {
                    'method': 'dbscan',
                    'eps': eps,
                    'n_clusters': n_clusters_actual,
                    'n_noise': n_noise,
                    'clusters': result_df
                }
            
            elif method == 'hdbscan':
                # Apply HDBSCAN
                model = hdbscan.HDBSCAN(
                    min_cluster_size=5,
                    min_samples=None,
                    cluster_selection_epsilon=0.0,
                    alpha=1.0,
                    cluster_selection_method='eom'
                )
                clusters = model.fit_predict(X_scaled)
                
                # Create result DataFrame
                result_df = df.copy()
                result_df['cluster'] = clusters
                result_df['cluster_probability'] = model.probabilities_
                
                # Calculate cluster statistics
                n_clusters_actual = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                
                return {
                    'method': 'hdbscan',
                    'n_clusters': n_clusters_actual,
                    'n_noise': n_noise,
                    'clusters': result_df,
                    'cluster_persistence': model.cluster_persistence_
                }
            
            else:
                return {'error': f"Unsupported clustering method: {method}"}
        
        except Exception as e:
            logger.error(f"Error finding patterns with clustering: {e}")
            return {'error': str(e)}
    
    def find_patterns_association_rules(self, df: pd.DataFrame, 
                                      min_support: float = 0.1,
                                      min_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Find patterns using association rule mining.
        
        Args:
            df: DataFrame with transaction data (binary or categorical)
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with association rules
        """
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            
            # Apply one-hot encoding for categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                df_encoded = pd.get_dummies(df, columns=categorical_cols)
            else:
                df_encoded = df
            
            # Find frequent itemsets
            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            
            # Generate association rules
            rules = association_rules(
                frequent_itemsets, metric="confidence", min_threshold=min_confidence
            )
            
            return {
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'rule_count': len(rules),
                'itemset_count': len(frequent_itemsets)
            }
        
        except Exception as e:
            logger.error(f"Error finding patterns with association rules: {e}")
            return {'error': str(e)}
    
    def find_patterns_sequential(self, df: pd.DataFrame, id_col: str, 
                               sequence_col: str, event_col: str,
                               min_support: float = 0.1) -> Dict[str, Any]:
        """
        Find sequential patterns in event data.
        
        Args:
            df: DataFrame with event data
            id_col: Column name for sequence ID
            sequence_col: Column name for sequence position or timestamp
            event_col: Column name for event type
            min_support: Minimum support threshold
            
        Returns:
            Dictionary with sequential patterns
        """
        try:
            # Sort data by ID and sequence
            df_sorted = df.sort_values([id_col, sequence_col])
            
            # Group events by ID
            sequences = df_sorted.groupby(id_col)[event_col].apply(list).tolist()
            
            # Find sequential patterns
            from prefixspan import PrefixSpan
            ps = PrefixSpan(sequences)
            
            # Mine frequent sequential patterns
            min_support_count = int(min_support * len(sequences))
            patterns = ps.frequent(min_support_count)
            
            # Format results
            pattern_results = []
            for support, pattern in patterns:
                pattern_results.append({
                    'pattern': pattern,
                    'support': support / len(sequences),
                    'count': support
                })
            
            # Sort by support
            pattern_results.sort(key=lambda x: x['support'], reverse=True)
            
            return {
                'patterns': pattern_results,
                'pattern_count': len(pattern_results),
                'sequence_count': len(sequences)
            }
        
        except Exception as e:
            logger.error(f"Error finding sequential patterns: {e}")
            return {'error': str(e)}


#######################################################
# OLAP Analysis and Data Mining
#######################################################

class OLAPAnalysis:
    """
    Multi-dimensional OLAP analysis for business intelligence.
    Implements data cube operations, drill-down, roll-up, and slice-and-dice.
    """
    
    def __init__(self):
        """Initialize the OLAP analysis module."""
        self.data_cubes = {}
        self.spark = None
        self.initialize_spark()
    
    def initialize_spark(self):
        """Initialize Spark session for OLAP processing."""
        try:
            self.spark = SparkSession.builder \
                .appName("Skyscope OLAP Analysis") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .getOrCreate()
            
            logger.info("Spark session initialized successfully for OLAP analysis")
        except Exception as e:
            logger.error(f"Error initializing Spark session for OLAP: {e}")
    
    def create_data_cube(self, df: pd.DataFrame, cube_name: str, 
                       dimensions: List[str], measures: List[str],
                       agg_functions: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Create a data cube for OLAP analysis.
        
        Args:
            df: DataFrame with data
            cube_name: Name for the data cube
            dimensions: List of dimension columns
            measures: List of measure columns
            agg_functions: Dictionary mapping measures to aggregation functions
            
        Returns:
            Dictionary with data cube information
        """
        try:
            # Default aggregation functions if not provided
            if agg_functions is None:
                agg_functions = {measure: 'sum' for measure in measures}
            
            # Create Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Register as temp view
            spark_df.createOrReplaceTempView(cube_name)
            
            # Store cube metadata
            self.data_cubes[cube_name] = {
                'dimensions': dimensions,
                'measures': measures,
                'agg_functions': agg_functions,
                'created_at': datetime.datetime.now()
            }
            
            return {
                'cube_name': cube_name,
                'dimensions': dimensions,
                'measures': measures,
                'row_count': df.shape[0]
            }
        
        except Exception as e:
            logger.error(f"Error creating data cube: {e}")
            return {'error': str(e)}
    
    def cube_query(self, cube_name: str, dimensions: List[str] = None,
                 measures: List[str] = None, filters: Dict[str, Any] = None,
                 group_by: List[str] = None) -> pd.DataFrame:
        """
        Query a data cube with OLAP operations.
        
        Args:
            cube_name: Name of the data cube
            dimensions: List of dimensions to include (None for all)
            measures: List of measures to include (None for all)
            filters: Dictionary with filter conditions
            group_by: List of dimensions to group by
            
        Returns:
            DataFrame with query results
        """
        try:
            if cube_name not in self.data_cubes:
                logger.error(f"Data cube {cube_name} not found")
                return pd.DataFrame()
            
            cube_info = self.data_cubes[cube_name]
            
            # Use all dimensions and measures if not specified
            if dimensions is None:
                dimensions = cube_info['dimensions']
            
            if measures is None:
                measures = cube_info['measures']
            
            # Build SQL query
            select_clause = []
            
            # Add dimensions to select clause
            for dim in dimensions:
                select_clause.append(f"`{dim}`")
            
            # Add measures with aggregation functions to select clause
            for measure in measures:
                agg_func = cube_info['agg_functions'].get(measure, 'sum')
                select_clause.append(f"{agg_func}(`{measure}`) as `{measure}`")
            
            # Build query
            query = f"SELECT {', '.join(select_clause)} FROM {cube_name}"
            
            # Add filters
            if filters:
                where_clauses = []
                for col, value in filters.items():
                    if isinstance(value, list):
                        where_clauses.append(f"`{col}` IN ({', '.join(map(repr, value))})")
                    elif isinstance(value, tuple) and len(value) == 2:
                        where_clauses.append(f"`{col}` BETWEEN {value[0]} AND {value[1]}")
                    else:
                        where_clauses.append(f"`{col}` = '{value}'")
                
                query += f" WHERE {' AND '.join(where_clauses)}"
            
            # Add group by
            if group_by:
                query += f" GROUP BY {', '.join([f'`{dim}`' for dim in group_by])}"
            elif dimensions:
                query += f" GROUP BY {', '.join([f'`{dim}`' for dim in dimensions])}"
            
            # Execute query
            result = self.spark.sql(query)
            
            # Convert to pandas DataFrame
            return result.toPandas()
        
        except Exception as e:
            logger.error(f"Error querying data cube: {e}")
            return pd.DataFrame()
    
    def drill_down(self, cube_name: str, from_dim: str, to_dim: str,
                 measures: List[str] = None, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Perform drill-down operation on a data cube.
        
        Args:
            cube_name: Name of the data cube
            from_dim: Higher-level dimension
            to_dim: Lower-level dimension
            measures: List of measures to include
            filters: Dictionary with filter conditions
            
        Returns:
            DataFrame with drill-down results
        """
        try:
            if cube_name not in self.data_cubes:
                logger.error(f"Data cube {cube_name} not found")
                return pd.DataFrame()
            
            # Get dimensions for drill-down
            dimensions = [from_dim, to_dim]
            
            # Query cube with drill-down dimensions
            return self.cube_query(
                cube_name=cube_name,
                dimensions=dimensions,
                measures=measures,
                filters=filters,
                group_by=dimensions
            )
        
        except Exception as e:
            logger.error(f"Error performing drill-down: {e}")
            return pd.DataFrame()
    
    def roll_up(self, cube_name: str, from_dim: str, to_dim: str = None,
              measures: List[str] = None, filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Perform roll-up operation on a data cube.
        
        Args:
            cube_name: Name of the data cube
            from_dim: Lower-level dimension
            to_dim: Higher-level dimension (None for grand total)
            measures: List of measures to include
            filters: Dictionary with filter conditions
            
        Returns:
            DataFrame with roll-up results
        """
        try:
            if cube_name not in self.data_cubes:
                logger.error(f"Data cube {cube_name} not found")
                return pd.DataFrame()
            
            # Get dimensions for roll-up
            dimensions = [to_dim] if to_dim else []
            
            # Query cube with roll-up dimensions
            return self.cube_query(
                cube_name=cube_name,
                dimensions=dimensions,
                measures=measures,
                filters=filters,
                group_by=dimensions
            )
        
        except Exception as e:
            logger.error(f"Error performing roll-up: {e}")
            return pd.DataFrame()
    
    def slice_and_dice(self, cube_name: str, dimensions: List[str],
                     measures: List[str], filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Perform slice-and-dice operation on a data cube.
        
        Args:
            cube_name: Name of the data cube
            dimensions: List of dimensions to include
            measures: List of measures to include
            filters: Dictionary with filter conditions
            
        Returns:
            DataFrame with slice-and-dice results
        """
        try:
            # Query cube with specified dimensions, measures, and filters
            return self.cube_query(
                cube_name=cube_name,
                dimensions=dimensions,
                measures=measures,
                filters=filters,
                group_by=dimensions
            )
        
        except Exception as e:
            logger.error(f"Error performing slice-and-dice: {e}")
            return pd.DataFrame()
    
    def pivot_analysis(self, df: pd.DataFrame, index: List[str],
                     columns: List[str], values: List[str],
                     aggfunc: str = 'sum') -> pd.DataFrame:
        """
        Perform pivot table analysis.
        
        Args:
            df: DataFrame with data
            index: List of columns for index
            columns: List of columns for pivot columns
            values: List of columns for values
            aggfunc: Aggregation function
            
        Returns:
            Pivot table DataFrame
        """
        try:
            # Create pivot table
            pivot_table = pd.pivot_table(
                df,
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )
            
            return pivot_table
        
        except Exception as e:
            logger.error(f"Error creating pivot table: {e}")
            return pd.DataFrame()


#######################################################
# GPT-4o Integration for Automated Insights
#######################################################

class InsightGenerator:
    """
    Automated insight generation using GPT-4o integration.
    Analyzes data and generates natural language insights.
    """
    
    def __init__(self):
        """Initialize the insight generator."""
        self.openai_client = None
        self.initialize_openai()
    
    def initialize_openai(self):
        """Initialize OpenAI client for GPT-4o integration."""
        try:
            if OPENAI_UNOFFICIAL_AVAILABLE:
                from openai_unofficial import OpenAI
                self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("OpenAI Unofficial client initialized for insight generation")
            elif OPENAI_AVAILABLE:
                import openai
                self.openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                logger.info("Standard OpenAI client initialized for insight generation")
            else:
                logger.warning("OpenAI client not available. Insight generation will be limited.")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
    
    def generate_insights_from_data(self, df: pd.DataFrame, context: str = None,
                                  max_insights: int = 5) -> List[Dict[str, str]]:
        """
        Generate insights from data using GPT-4o.
        
        Args:
            df: DataFrame with data
            context: Additional context about the data
            max_insights: Maximum number of insights to generate
            
        Returns:
            List of insights
        """
        try:
            if not self.openai_client:
                return [{"title": "OpenAI API Not Available", 
                        "content": "OpenAI API is not available for insight generation."}]
            
            # Prepare data summary
            data_summary = self._prepare_data_summary(df)
            
            # Build prompt
            prompt = f"""
            You are an expert data analyst. Analyze the following data and provide {max_insights} key insights.
            
            Data Summary:
            {data_summary}
            
            {f'Context: {context}' if context else ''}
            
            For each insight:
            1. Provide a concise title
            2. Provide a detailed explanation with specific numbers and trends
            3. If relevant, suggest business implications or actions
            
            Format your response as a JSON array with 'title' and 'content' fields for each insight.
            """
            
            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst providing insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                insights = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                insights = [{"title": "Data Analysis", "content": content}]
            
            # Store insights in database
            self._store_insights_in_db(insights, "data_analysis", df.shape[0])
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights from data: {e}")
            return [{"title": "Error Generating Insights", "content": str(e)}]
    
    def generate_insights_from_forecast(self, forecast_df: pd.DataFrame, 
                                      model_name: str, metrics: Dict[str, float] = None) -> List[Dict[str, str]]:
        """
        Generate insights from forecast results using GPT-4o.
        
        Args:
            forecast_df: DataFrame with forecast results
            model_name: Name of the forecast model
            metrics: Dictionary with model metrics
            
        Returns:
            List of insights
        """
        try:
            if not self.openai_client:
                return [{"title": "OpenAI API Not Available", 
                        "content": "OpenAI API is not available for insight generation."}]
            
            # Prepare forecast summary
            forecast_summary = self._prepare_forecast_summary(forecast_df, metrics)
            
            # Build prompt
            prompt = f"""
            You are an expert forecasting analyst. Analyze the following forecast results and provide key insights.
            
            Forecast Model: {model_name}
            
            Forecast Summary:
            {forecast_summary}
            
            For each insight:
            1. Provide a concise title
            2. Provide a detailed explanation with specific numbers and trends
            3. If relevant, suggest business implications or actions
            
            Format your response as a JSON array with 'title' and 'content' fields for each insight.
            """
            
            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert forecasting analyst providing insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                insights = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                insights = [{"title": "Forecast Analysis", "content": content}]
            
            # Store insights in database
            self._store_insights_in_db(insights, "forecast_analysis", forecast_df.shape[0])
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights from forecast: {e}")
            return [{"title": "Error Generating Insights", "content": str(e)}]
    
    def generate_insights_from_kpis(self, kpis: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate insights from KPI data using GPT-4o.
        
        Args:
            kpis: List of KPI dictionaries
            
        Returns:
            List of insights
        """
        try:
            if not self.openai_client:
                return [{"title": "OpenAI API Not Available", 
                        "content": "OpenAI API is not available for insight generation."}]
            
            # Prepare KPI summary
            kpi_summary = json.dumps(kpis, indent=2)
            
            # Build prompt
            prompt = f"""
            You are an expert business analyst. Analyze the following KPIs and provide key insights.
            
            KPI Data:
            {kpi_summary}
            
            For each insight:
            1. Provide a concise title
            2. Provide a detailed explanation focusing on KPIs that are off-target or showing significant changes
            3. Suggest business implications or actions
            
            Format your response as a JSON array with 'title' and 'content' fields for each insight.
            """
            
            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert business analyst providing insights in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            # Extract JSON
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                insights = json.loads(json_str)
            else:
                # Fallback if JSON parsing fails
                insights = [{"title": "KPI Analysis", "content": content}]
            
            # Store insights in database
            self._store_insights_in_db(insights, "kpi_analysis", len(kpis))
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights from KPIs: {e}")
            return [{"title": "Error Generating Insights", "content": str(e)}]
    
    def _prepare_data_summary(self, df: pd.DataFrame) -> str:
        """
        Prepare a summary of the data for insight generation.
        
        Args:
            df: DataFrame with data
            
        Returns:
            String with data summary
        """
        # Basic statistics
        stats = df.describe().to_string()
        
        # Column types
        dtypes = df.dtypes.to_string()
        
        # Sample data (first 5 rows)
        sample = df.head(5).to_string()
        
        # Missing values
        missing = df.isna().sum().to_string()
        
        # Combine into summary
        summary = f"""
        DataFrame Shape: {df.shape}
        
        Column Types:
        {dtypes}
        
        Missing Values:
        {missing}
        
        Basic Statistics:
        {stats}
        
        Sample Data:
        {sample}
        """
        
        return summary
    
    def _prepare_forecast_summary(self, forecast_df: pd.DataFrame, 
                                metrics: Dict[str, float] = None) -> str:
        """
        Prepare a summary of forecast results for insight generation.
        
        Args:
            forecast_df: DataFrame with forecast results
            metrics: Dictionary with model metrics
            
        Returns:
            String with forecast summary
        """
        # Basic statistics
        stats = forecast_df.describe().to_string()
        
        # Sample forecast data
        sample = forecast_df.tail(10).to_string()
        
        # Metrics summary
        metrics_str = ""
        if metrics:
            metrics_str = "\nModel Metrics:\n"
            for metric, value in metrics.items():
                metrics_str += f"{metric}: {value}\n"
        
        # Combine into summary
        summary = f"""
        Forecast DataFrame Shape: {forecast_df.shape}
        
        Forecast Statistics:
        {stats}
        
        {metrics_str}
        
        Recent Forecast Data:
        {sample}
        """
        
        return summary
    
    def _store_insights_in_db(self, insights: List[Dict[str, str]], 
                            source: str, source_id: int) -> bool:
        """
        Store generated insights in the database.
        
        Args:
            insights: List of insight dictionaries
            source: Source of the insights
            source_id: ID of the source
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with Session() as session:
                for insight in insights:
                    db_insight = Insight(
                        title=insight['title'],
                        content=insight['content'],
                        source=source,
                        source_id=source_id,
                        confidence=0.8,  # Default confidence
                        generated_at=datetime.datetime.now(),
                        model=DEFAULT_MODEL
                    )
                    session.add(db_insight)
                
                session.commit()
                return True
        except Exception as e:
            logger.error(f"Error storing insights in database: {e}")
            return False


#######################################################
# Executive Dashboard and KPI Tracking
#######################################################

class ExecutiveDashboard:
    """
    Executive dashboard management and KPI tracking.
    Provides real-time monitoring of key business metrics.
    """
    
    def __init__(self):
        """Initialize the executive dashboard."""
        self.dashboards = {}
        self.kpis = {}
        self.alerts = []
    
    def create_dashboard(self, name: str, description: str = None,
                       layout: Dict[str, Any] = None, owner_id: int = None,
                       is_public: bool = False) -> int:
        """
        Create a new executive dashboard.
        
        Args:
            name: Dashboard name
            description: Dashboard description
            layout: Dashboard layout configuration
            owner_id: ID of the dashboard owner
            is_public: Whether the dashboard is public
            
        Returns:
            ID of the created dashboard
        """
        try:
            with Session() as session:
                # Create dashboard
                dashboard = Dashboard(
                    name=name,
                    description=description,
                    layout=json.dumps(layout) if layout else None,
                    owner_id=owner_id,
                    is_public=1 if is_public else 0,
                    created_at=datetime.datetime.now()
                )
                session.add(dashboard)
                session.commit()
                
                # Store in memory
                self.dashboards[dashboard.id] = {
                    'name': name,
                    'description': description,
                    'layout': layout,
                    'widgets': []
                }
                
                return dashboard.id
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            return -1
    
    def add_widget(self, dashboard_id: int, widget_type: str, title: str,
                 config: Dict[str, Any], position_x: int = 0,
                 position_y: int = 0, width: int = 4, height: int = 4) -> int:
        """
        Add a widget to a dashboard.
        
        Args:
            dashboard_id: ID of the dashboard
            widget_type: Type of widget ('chart', 'table', 'kpi', etc.)
            title: Widget title
            config: Widget configuration
            position_x: X position on dashboard grid
            position_y: Y position on dashboard grid
            width: Widget width in grid units
            height: Widget height in grid units
            
        Returns:
            ID of the created widget
        """
        try:
            with Session() as session:
                # Create widget
                widget = DashboardWidget(
                    dashboard_id=dashboard_id,
                    widget_type=widget_type,
                    title=title,
                    config=json.dumps(config),
                    position_x=position_x,
                    position_y=position_y,
                    width=width,
                    height=height
                )
                session.add(widget)
                session.commit()
                
                # Store in memory
                if dashboard_id in self.dashboards:
                    self.dashboards[dashboard_id]['widgets'].append({
                        'id': widget.id,
                        'type': widget_type,
                        'title': title,
                        'config': config,
                        'position': {'x': position_x, 'y': position_y},
                        'size': {'width': width, 'height': height}
                    })
                
                return widget.id
        except Exception as e:
            logger.error(f"Error adding widget to dashboard: {e}")
            return -1
    
    def get_dashboard(self, dashboard_id: int) -> Dict[str, Any]:
        """
        Get a dashboard with all its widgets.
        
        Args:
            dashboard_id: ID of the dashboard
            
        Returns:
            Dictionary with dashboard data
        """
        try:
            with Session() as session:
                # Get dashboard
                dashboard = session.query(Dashboard).filter_by(id=dashboard_id).first()
                
                if not dashboard:
                    logger.error(f"Dashboard {dashboard_id} not found")
                    return {}
                
                # Get widgets
                widgets = session.query(DashboardWidget).filter_by(dashboard_id=dashboard_id).all()
                
                # Prepare result
                result = {
                    'id': dashboard.id,
                    'name': dashboard.name,
                    'description': dashboard.description,
                    'layout': json.loads(dashboard.layout) if dashboard.layout else None,
                    'owner_id': dashboard.owner_id,
                    'is_public': dashboard.is_public == 1,
                    'created_at': dashboard.created_at.isoformat(),
                    'updated_at': dashboard.updated_at.isoformat(),
                    'widgets': []
                }
                
                for widget in widgets:
                    result['widgets'].append({
                        'id': widget.id,
                        'type': widget.widget_type,
                        'title': widget.title,
                        'config': json.loads(widget.config) if widget.config else None,
                        'position': {'x': widget.position_x, 'y': widget.position_y},
                        'size': {'width': widget.width, 'height': widget.height}
                    })
                
                # Store in memory
                self.dashboards[dashboard.id] = result
                
                return result
        except Exception as e:
            logger.error(f"Error getting dashboard: {e}")
            return {}
    
    def create_kpi(self, name: str, description: str = None, category: str = None,
                 current_value: float = 0.0, target_value: float = None,
                 unit: str = None) -> int:
        """
        Create a new KPI.
        
        Args:
            name: KPI name
            description: KPI description
            category: KPI category
            current_value: Current value of the KPI
            target_value: Target value of the KPI
            unit: Unit of measurement
            
        Returns:
            ID of the created KPI
        """
        try:
            with Session() as session:
                # Calculate status
                status = 'on_track'
                if target_value is not None and current_value is not None:
                    if current_value < target_value * 0.8:
                        status = 'off_track'
                    elif current_value < target_value * 0.95:
                        status = 'at_risk'
                
                # Create KPI
                kpi = KPI(
                    name=name,
                    description=description,
                    category=category,
                    current_value=current_value,
                    target_value=target_value,
                    unit=unit,
                    status=status,
                    updated_at=datetime.datetime.now()
                )
                session.add(kpi)
                session.commit()
                
                # Create initial history entry
                history = KPIHistory(
                    kpi_id=kpi.id,
                    value=current_value,
                    timestamp=datetime.datetime.now()
                )
                session.add(history)
                session.commit()
                
                # Store in memory
                self.kpis[kpi.id] = {
                    'name': name,
                    'description': description,
                    'category': category,
                    'current_value': current_value,
                    'target_value': target_value,
                    'unit': unit,
                    'status': status
                }
                
                return kpi.id
        except Exception as e:
            logger.error(f"Error creating KPI: {e}")
            return -1
    
    def update_kpi(self, kpi_id: int, new_value: float) -> bool:
        """
        Update a KPI with a new value.
        
        Args:
            kpi_id: ID of the KPI
            new_value: New value for the KPI
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with Session() as session:
                # Get KPI
                kpi = session.query(KPI).filter_by(id=kpi_id).first()
                
                if not kpi:
                    logger.error(f"KPI {kpi_id} not found")
                    return False
                
                # Update KPI
                old_value = kpi.current_value
                kpi.current_value = new_value
                kpi.updated_at = datetime.datetime.now()
                
                # Calculate status
                if kpi.target_value is not None:
                    if new_value < kpi.target_value * 0.8:
                        kpi.status = 'off_track'
                    elif new_value < kpi.target_value * 0.95:
                        kpi.status = 'at_risk'
                    else:
                        kpi.status = 'on_track'
                
                # Create history entry
                history = KPIHistory(
                    kpi_id=kpi.id,
                    value=new_value,
                    timestamp=datetime.datetime.now()
                )
                session.add(history)
                
                session.commit()
                
                # Update in memory
                if kpi_id in self.kpis:
                    self.kpis[kpi_id]['current_value'] = new_value
                    self.kpis[kpi_id]['status'] = kpi.status
                
                # Check for alerts
                if kpi.status == 'off_track' and (old_value >= kpi.target_value * 0.8 or old_value == 0):
                    self._create_alert(
                        kpi_id=kpi_id,
                        alert_type='kpi_off_track',
                        message=f"KPI {kpi.name} is now off track. Current value: {new_value} {kpi.unit or ''}, Target: {kpi.target_value} {kpi.unit or ''}"
                    )
                elif kpi.status == 'at_risk' and old_value >= kpi.target_value * 0.95:
                    self._create_alert(
                        kpi_id=kpi_id,
                        alert_type='kpi_at_risk',
                        message=f"KPI {kpi.name} is now at risk. Current value: {new_value} {kpi.unit or ''}, Target: {kpi.target_value} {kpi.unit or ''}"
                    )
                
                return True
        except Exception as e:
            logger.error(f"Error updating KPI: {e}")
            return False
    
    def get_kpi(self, kpi_id: int, with_history: bool = False) -> Dict[str, Any]:
        """
        Get a KPI with its history.
        
        Args:
            kpi_id: ID of the KPI
            with_history: Whether to include history
            
        Returns:
            Dictionary with KPI data
        """
        try:
            with Session() as session:
                # Get KPI
                kpi = session.query(KPI).filter_by(id=kpi_id).first()
                
                if not kpi:
                    logger.error(f"KPI {kpi_id} not found")
                    return {}
                
                # Prepare result
                result = {
                    'id': kpi.id,
                    'name': kpi.name,
                    'description': kpi.description,
                    'category': kpi.category,
                    'current_value': kpi.current_value,
                    'target_value': kpi.target_value,
                    'unit': kpi.unit,
                    'status': kpi.status,
                    'updated_at': kpi.updated_at.isoformat()
                }
                
                # Add history if requested
                if with_history:
                    history = session.query(KPIHistory).filter_by(kpi_id=kpi_id).order_by(KPIHistory.timestamp).all()
                    result['history'] = [
                        {
                            'value': h.value,
                            'timestamp': h.timestamp.isoformat()
                        }
                        for h in history
                    ]
                
                # Store in memory
                self.kpis[kpi.id] = {
                    'name': kpi.name,
                    'description': kpi.description,
                    'category': kpi.category,
                    'current_value': kpi.current_value,
                    'target_value': kpi.target_value,
                    'unit': kpi.unit,
                    'status': kpi.status
                }
                
                return result
        except Exception as e:
            logger.error(f"Error getting KPI: {e}")
            return {}
    
    def get_kpis_by_category(self, category: str = None) -> List[Dict[str, Any]]:
        """
        Get KPIs by category.
        
        Args:
            category: KPI category (None for all)
            
        Returns:
            List of KPI dictionaries
        """
        try:
            with Session() as session:
                # Query KPIs
                if category:
                    kpis = session.query(KPI).filter_by(category=category).all()
                else:
                    kpis = session.query(KPI).all()
                
                # Prepare result
                result = []
                for kpi in kpis:
                    result.append({
                        'id': kpi.id,
                        'name': kpi.name,
                        'description': kpi.description,
                        'category': kpi.category,
                        'current_value': kpi.current_value,
                        'target_value': kpi.target_value,
                        'unit': kpi.unit,
                        'status': kpi.status,
                        'updated_at': kpi.updated_at.isoformat()
                    })
                
                return result
        except Exception as e:
            logger.error(f"Error getting KPIs by category: {e}")
            return []
    
    def _create_alert(self, kpi_id: int, alert_type: str, message: str) -> bool:
        """
        Create an alert for a KPI.
        
        Args:
            kpi_id: ID of the KPI
            alert_type: Type of alert
            message: Alert message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Store alert in memory
            alert = {
                'kpi_id': kpi_id,
                'type': alert_type,
                'message': message,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self.alerts.append(alert)
            
            # Log alert
            logger.warning(f"KPI Alert: {message}")
            
            # In a real implementation, this would send notifications
            # via email, Slack, or other channels
            
            return True
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return False
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        # Return alerts from memory
        return self.alerts[-limit:]


#######################################################
# Reporting Automation
#######################################################

class ReportingAutomation:
    """
    Automated report generation and distribution.
    Creates PDF, Excel, and web-based reports.
    """
    
    def __init__(self):
        """Initialize the reporting automation module."""
        self.reports = {}
        self.templates = {}
        self.scheduled_reports = {}
        self.load_reports()
    
    def load_reports(self):
        """Load reports from the database."""
        try:
            with Session() as session:
                reports = session.query(Report).all()
                
                for report in reports:
                    self.reports[report.id] = {
                        'name': report.name,
                        'description': report.description,
                        'report_type': report.report_type,
                        'schedule': report.schedule,
                        'recipients': json.loads(report.recipients) if report.recipients else [],
                        'query': report.query,
                        'template': report.template,
                        'last_generated': report.last_generated
                    }
            
            logger.info(f"Loaded {len(self.reports)} reports")
        except Exception as e:
            logger.error(f"Error loading reports: {e}")
    
    def create_report(self, name: str, description: str = None,
                    report_type: str = 'pdf', schedule: str = None,
                    recipients: List[str] = None, query: str = None,
                    template: str = None) -> int:
        """
        Create a new report.
        
        Args:
            name: Report name
            description: Report description
            report_type: Report type ('pdf', 'excel', 'web')
            schedule: Cron expression for scheduling
            recipients: List of recipient email addresses
            query: SQL or query definition
            template: Template name or path
            
        Returns:
            ID of the created report
        """
        try:
            with Session() as session:
                # Create report
                report = Report(
                    name=name,
                    description=description,
                    report_type=report_type,
                    schedule=schedule,
                    recipients=json.dumps(recipients) if recipients else None,
                    query=query,
                    template=template,
                    created_at=datetime.datetime.now()
                )
                session.add(report)
                session.commit()
                
                # Store in memory
                self.reports[report.id] = {
                    'name': name,
                    'description': description,
                    'report_type': report_type,
                    'schedule': schedule,
                    'recipients': recipients or [],
                    'query': query,
                    'template': template,
                    'last_generated': None
                }
                
                # Schedule report if needed
                if schedule:
                    self._schedule_report(report.id, schedule)
                
                return report.id
        except Exception as e:
            logger.error(f"Error creating report: {e}")
            return -1
    
    def _schedule_report(self, report_id: int, schedule: str):
        """
        Schedule a report using the provided cron expression.
        
        Args:
            report_id: ID of the report
            schedule: Cron expression
            
        Returns:
            None
        """
        # This is a placeholder for scheduling logic
        # In a real implementation, this would use a scheduler like APScheduler
        self.scheduled_reports[report_id] = {
            'schedule': schedule,
            'next_run': None  # Would be calculated based on cron expression
        }
    
    def generate_report(self, report_id: int) -> Dict[str, Any]:
        """
        Generate a report.
        
        Args:
            report_id: ID of the report
            
        Returns:
            Dictionary with report generation result
        """
        try:
            if report_id not in self.reports:
                logger.error(f"Report {report_id} not found")
                return {'error': f"Report {report_id} not found"}
            
            report_info = self.reports[report_id]
            
            # Execute query
            data = self._execute_query(report_info['query'])
            
            if isinstance(data, dict) and 'error' in data:
                return data
            
            # Generate report based on type
            if report_info['report_type'] == 'pdf':
                report_path = self._generate_pdf_report(
                    data=data,
                    report_name=report_info['name'],
                    template=report_info['template']
                )
            elif report_info['report_type'] == 'excel':
                report_path = self._generate_excel_report(
                    data=data,
                    report_name=report_info['name'],
                    template=report_info['template']
                )
            elif report_info['report_type'] == 'web':
                report_path = self._generate_web_report(
                    data=data,
                    report_name=report_info['name'],
                    template=report_info['template']
                )
            else:
                return {'error': f"Unsupported report type: {report_info['report_type']}"}
            
            # Update last generated timestamp
            with Session() as session:
                report = session.query(Report).filter_by(id=report_id).first()
                if report:
                    report.last_generated = datetime.datetime.now()
                    session.commit()
            
            # Update in memory
            report_info['last_generated'] = datetime.datetime.now()
            
            # Send report if recipients are specified
            if report_info['recipients']:
                self._send_report(
                    report_path=report_path,
                    report_name=report_info['name'],
                    recipients=report_info['recipients']
                )
            
            return {
                'report_id': report_id,
                'report_name': report_info['name'],
                'report_type': report_info['report_type'],
                'report_path': report_path,
                'generated_at': datetime.datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a query and return the results.
        
        Args:
            query: SQL or query definition
            
        Returns:
            DataFrame with query results
        """
        try:
            # Check if query is SQL or JSON definition
            if query.strip().upper().startswith('SELECT'):
                # Execute SQL query
                with Session() as session:
                    result = session.execute(text(query))
                    columns = result.keys()
                    data = result.fetchall()
                    return pd.DataFrame(data, columns=columns)
            else:
                # Parse JSON query definition
                try:
                    query_def = json.loads(query)
                    
                    # Handle different query types
                    if query_def.get('type') == 'kpi':
                        # Query KPIs
                        category = query_def.get('category')
                        with Session() as session:
                            if category:
                                kpis = session.query(KPI).filter_by