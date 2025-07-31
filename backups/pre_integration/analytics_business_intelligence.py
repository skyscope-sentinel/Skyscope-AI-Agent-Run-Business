#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analytics_business_intelligence.py - Advanced Analytics and Business Intelligence System

This module provides a comprehensive analytics and business intelligence system for the
Skyscope Sentinel Intelligence AI platform, supporting real-time dashboards, data warehousing,
predictive analytics, and executive reporting for a 10,000 employee organization.

Features:
1. Real-time analytics dashboards with interactive visualizations
2. Data warehouse implementation with ETL pipelines (Apache Spark patterns)
3. Advanced visualization components (Plotly, D3.js integration)
4. Predictive analytics with Prophet and ML models
5. Executive dashboards with KPI tracking
6. Data mining and pattern recognition algorithms
7. GPT-4o integration for automated insights generation
8. Time series forecasting for business metrics
9. Anomaly detection for critical metrics
10. Multi-dimensional OLAP analysis
11. Real-time data streaming (Kafka-like patterns)
12. Comprehensive reporting automation

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

# Check for required packages
REQUIRED_PACKAGES = [
    "numpy", "pandas", "matplotlib", "seaborn", "plotly", "dash", "flask",
    "pyspark", "prophet", "scikit-learn", "tensorflow", "statsmodels",
    "sqlalchemy", "pymongo", "redis", "kafka-python", "aiohttp", "websockets",
    "dash_bootstrap_components", "dash_core_components", "dash_html_components",
    "reportlab", "openpyxl", "xlsxwriter", "pydeck", "networkx", "scipy",
    "lightgbm", "xgboost", "shap", "lime", "umap", "hdbscan", "fastapi",
    "uvicorn", "python-jose", "passlib", "jinja2", "python-multipart"
]

# Try to import required packages, install if missing
missing_packages = []
for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    import subprocess
    print(f"Installing missing packages: {', '.join(missing_packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    print("Please restart the application to use the newly installed packages.")
    sys.exit(1)

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
        logging.FileHandler("analytics_bi.log"),
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
# Database Models
#######################################################

class AnalyticsMetric(Base):
    """Database model for analytics metrics."""
    __tablename__ = 'analytics_metrics'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    source = Column(String)
    dimensions = Column(String)  # JSON serialized dimensions
    
    def __repr__(self):
        return f"<AnalyticsMetric(id={self.id}, name='{self.name}', value={self.value})>"


class KPI(Base):
    """Database model for Key Performance Indicators."""
    __tablename__ = 'kpis'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    category = Column(String, nullable=False)
    current_value = Column(Float, nullable=False)
    target_value = Column(Float)
    unit = Column(String)
    status = Column(String)  # on_track, at_risk, off_track
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    history = relationship("KPIHistory", back_populates="kpi")
    
    def __repr__(self):
        return f"<KPI(id={self.id}, name='{self.name}', current_value={self.current_value})>"


class KPIHistory(Base):
    """Database model for KPI historical values."""
    __tablename__ = 'kpi_history'
    
    id = Column(Integer, primary_key=True)
    kpi_id = Column(Integer, ForeignKey('kpis.id'))
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    kpi = relationship("KPI", back_populates="history")
    
    def __repr__(self):
        return f"<KPIHistory(id={self.id}, kpi_id={self.kpi_id}, value={self.value})>"


class Dashboard(Base):
    """Database model for dashboards."""
    __tablename__ = 'dashboards'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    layout = Column(String)  # JSON serialized layout
    owner_id = Column(Integer)
    is_public = Column(Integer, default=0)  # 0=private, 1=public
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    widgets = relationship("DashboardWidget", back_populates="dashboard")
    
    def __repr__(self):
        return f"<Dashboard(id={self.id}, name='{self.name}')>"


class DashboardWidget(Base):
    """Database model for dashboard widgets."""
    __tablename__ = 'dashboard_widgets'
    
    id = Column(Integer, primary_key=True)
    dashboard_id = Column(Integer, ForeignKey('dashboards.id'))
    widget_type = Column(String, nullable=False)  # chart, table, kpi, etc.
    title = Column(String, nullable=False)
    config = Column(String)  # JSON serialized configuration
    position_x = Column(Integer, default=0)
    position_y = Column(Integer, default=0)
    width = Column(Integer, default=4)
    height = Column(Integer, default=4)
    
    # Relationships
    dashboard = relationship("Dashboard", back_populates="widgets")
    
    def __repr__(self):
        return f"<DashboardWidget(id={self.id}, title='{self.title}', type='{self.widget_type}')>"


class Report(Base):
    """Database model for reports."""
    __tablename__ = 'reports'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    report_type = Column(String, nullable=False)  # pdf, excel, etc.
    schedule = Column(String)  # cron expression
    recipients = Column(String)  # JSON serialized list of recipients
    query = Column(String)  # SQL or query definition
    template = Column(String)  # Template name or path
    last_generated = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Report(id={self.id}, name='{self.name}', type='{self.report_type}')>"


class DataSource(Base):
    """Database model for data sources."""
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    source_type = Column(String, nullable=False)  # database, api, file, etc.
    connection_string = Column(String)
    credentials = Column(String)  # Encrypted credentials
    refresh_interval = Column(Integer)  # In seconds
    last_refresh = Column(DateTime)
    schema = Column(String)  # JSON serialized schema
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<DataSource(id={self.id}, name='{self.name}', type='{self.source_type}')>"


class DataWarehouseTable(Base):
    """Database model for data warehouse tables."""
    __tablename__ = 'data_warehouse_tables'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    schema = Column(String)  # JSON serialized schema
    source_id = Column(Integer, ForeignKey('data_sources.id'))
    etl_job_id = Column(Integer)
    row_count = Column(Integer, default=0)
    last_updated = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<DataWarehouseTable(id={self.id}, name='{self.name}')>"


class Insight(Base):
    """Database model for AI-generated insights."""
    __tablename__ = 'insights'
    
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(String, nullable=False)
    source = Column(String)  # dashboard, report, kpi, etc.
    source_id = Column(Integer)
    confidence = Column(Float)
    generated_at = Column(DateTime, default=func.now())
    model = Column(String)
    
    def __repr__(self):
        return f"<Insight(id={self.id}, title='{self.title}')>"


class AnomalyDetection(Base):
    """Database model for anomaly detection records."""
    __tablename__ = 'anomaly_detections'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    expected_value = Column(Float)
    deviation_score = Column(Float, nullable=False)
    is_anomaly = Column(Integer, default=0)  # 0=normal, 1=anomaly
    detection_method = Column(String)
    timestamp = Column(DateTime, default=func.now())
    
    def __repr__(self):
        return f"<AnomalyDetection(id={self.id}, metric='{self.metric_name}', score={self.deviation_score})>"


class ForecastModel(Base):
    """Database model for forecast models."""
    __tablename__ = 'forecast_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    metric_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # prophet, arima, lstm, etc.
    parameters = Column(String)  # JSON serialized parameters
    accuracy = Column(Float)
    last_trained = Column(DateTime)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    forecasts = relationship("Forecast", back_populates="model")
    
    def __repr__(self):
        return f"<ForecastModel(id={self.id}, name='{self.name}', type='{self.model_type}')>"


class Forecast(Base):
    """Database model for forecasts."""
    __tablename__ = 'forecasts'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('forecast_models.id'))
    timestamp = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    lower_bound = Column(Float)
    upper_bound = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    model = relationship("ForecastModel", back_populates="forecasts")
    
    def __repr__(self):
        return f"<Forecast(id={self.id}, model_id={self.model_id}, value={self.value})>"


# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


#######################################################
# Data Warehouse and ETL Pipeline
#######################################################

class DataWarehouse:
    """
    Data Warehouse implementation with ETL pipelines using Apache Spark patterns.
    Provides data storage, transformation, and retrieval capabilities.
    """
    
    def __init__(self):
        """Initialize the Data Warehouse."""
        self.spark = None
        self.mongo_client = None
        self.redis_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
        self.etl_jobs = {}
        self.data_sources = {}
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize connections to various data systems."""
        try:
            # Initialize Spark session
            self.spark = SparkSession.builder \
                .appName("Skyscope Analytics BI") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.sql.warehouse.dir", str(DATA_WAREHOUSE_PATH / "spark-warehouse")) \
                .getOrCreate()
            
            logger.info("Spark session initialized successfully")
            
            # Initialize MongoDB connection
            try:
                self.mongo_client = pymongo.MongoClient(MONGODB_URI)
                self.mongo_client.admin.command('ping')
                logger.info("MongoDB connection established successfully")
            except Exception as e:
                logger.warning(f"MongoDB connection failed: {e}")
                self.mongo_client = None
            
            # Initialize Redis connection
            try:
                self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
                self.redis_client.ping()
                logger.info("Redis connection established successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
            
            # Initialize Kafka producer
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info("Kafka producer initialized successfully")
            except Exception as e:
                logger.warning(f"Kafka producer initialization failed: {e}")
                self.kafka_producer = None
            
            # Load data sources from database
            self.load_data_sources()
            
        except Exception as e:
            logger.error(f"Error initializing data warehouse connections: {e}")
    
    def load_data_sources(self):
        """Load data sources from the database."""
        try:
            with Session() as session:
                sources = session.query(DataSource).all()
                for source in sources:
                    self.data_sources[source.id] = {
                        "name": source.name,
                        "type": source.source_type,
                        "connection_string": source.connection_string,
                        "refresh_interval": source.refresh_interval
                    }
            logger.info(f"Loaded {len(self.data_sources)} data sources")
        except Exception as e:
            logger.error(f"Error loading data sources: {e}")
    
    def create_etl_job(self, name: str, source_id: int, target_table: str, 
                      transformation_script: str = None, schedule: str = None) -> int:
        """
        Create a new ETL job.
        
        Args:
            name: Name of the ETL job
            source_id: ID of the data source
            target_table: Target table name
            transformation_script: PySpark transformation script
            schedule: Cron expression for scheduling
            
        Returns:
            ID of the created ETL job
        """
        try:
            job_id = len(self.etl_jobs) + 1
            self.etl_jobs[job_id] = {
                "name": name,
                "source_id": source_id,
                "target_table": target_table,
                "transformation_script": transformation_script,
                "schedule": schedule,
                "status": "created",
                "last_run": None,
                "next_run": None
            }
            
            # Schedule the job if a schedule is provided
            if schedule:
                self._schedule_etl_job(job_id, schedule)
            
            logger.info(f"Created ETL job {job_id}: {name}")
            return job_id
        except Exception as e:
            logger.error(f"Error creating ETL job: {e}")
            return -1
    
    def _schedule_etl_job(self, job_id: int, schedule: str):
        """Schedule an ETL job using the provided cron expression."""
        # This is a placeholder for scheduling logic
        # In a real implementation, this would use a scheduler like APScheduler
        pass
    
    def run_etl_job(self, job_id: int) -> bool:
        """
        Run an ETL job.
        
        Args:
            job_id: ID of the ETL job to run
            
        Returns:
            True if successful, False otherwise
        """
        if job_id not in self.etl_jobs:
            logger.error(f"ETL job {job_id} not found")
            return False
        
        job = self.etl_jobs[job_id]
        source_id = job["source_id"]
        
        if source_id not in self.data_sources:
            logger.error(f"Data source {source_id} not found")
            return False
        
        source = self.data_sources[source_id]
        
        try:
            # Update job status
            job["status"] = "running"
            job["last_run"] = datetime.datetime.now()
            
            # Extract data from source
            df = self._extract_data(source)
            
            # Transform data
            if job["transformation_script"]:
                df = self._transform_data(df, job["transformation_script"])
            
            # Load data to target
            success = self._load_data(df, job["target_table"])
            
            # Update job status
            job["status"] = "completed" if success else "failed"
            
            logger.info(f"ETL job {job_id} completed with status: {job['status']}")
            return success
        except Exception as e:
            logger.error(f"Error running ETL job {job_id}: {e}")
            job["status"] = "failed"
            return False
    
    def _extract_data(self, source: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract data from a source.
        
        Args:
            source: Source configuration
            
        Returns:
            DataFrame with extracted data
        """
        source_type = source["type"]
        
        if source_type == "database":
            # Extract from database
            return self._extract_from_database(source["connection_string"])
        elif source_type == "api":
            # Extract from API
            return self._extract_from_api(source["connection_string"])
        elif source_type == "file":
            # Extract from file
            return self._extract_from_file(source["connection_string"])
        elif source_type == "kafka":
            # Extract from Kafka
            return self._extract_from_kafka(source["connection_string"])
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _extract_from_database(self, connection_string: str) -> pd.DataFrame:
        """Extract data from a database."""
        # This is a placeholder implementation
        # In a real implementation, this would use SQLAlchemy to connect to the database
        engine = create_engine(connection_string)
        query = "SELECT * FROM sample_table"
        return pd.read_sql(query, engine)
    
    def _extract_from_api(self, api_url: str) -> pd.DataFrame:
        """Extract data from an API."""
        # This is a placeholder implementation
        # In a real implementation, this would use requests or aiohttp to call the API
        import requests
        response = requests.get(api_url)
        data = response.json()
        return pd.DataFrame(data)
    
    def _extract_from_file(self, file_path: str) -> pd.DataFrame:
        """Extract data from a file."""
        # Determine file type from extension
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")
    
    def _extract_from_kafka(self, topic: str) -> pd.DataFrame:
        """Extract data from Kafka."""
        if not self.kafka_consumer:
            self.kafka_consumer = KafkaConsumer(
                topic,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id='analytics_bi',
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
        
        # Collect messages for a short period
        messages = []
        start_time = time.time()
        while time.time() - start_time < 5:  # Collect for 5 seconds
            msg_pack = self.kafka_consumer.poll(timeout_ms=100)
            for tp, msgs in msg_pack.items():
                for msg in msgs:
                    messages.append(msg.value)
        
        return pd.DataFrame(messages)
    
    def _transform_data(self, df: pd.DataFrame, transformation_script: str) -> pd.DataFrame:
        """
        Transform data using a PySpark transformation script.
        
        Args:
            df: Input DataFrame
            transformation_script: PySpark transformation script
            
        Returns:
            Transformed DataFrame
        """
        try:
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Execute transformation script
            # This is a simplified implementation
            # In a real implementation, this would use a safer method to execute code
            local_dict = {"spark": self.spark, "df": spark_df}
            exec(transformation_script, {}, local_dict)
            transformed_df = local_dict.get("result", spark_df)
            
            # Convert back to pandas DataFrame
            return transformed_df.toPandas()
        except Exception as e:
            logger.error(f"Error transforming data: {e}")
            return df
    
    def _load_data(self, df: pd.DataFrame, target_table: str) -> bool:
        """
        Load data to a target table.
        
        Args:
            df: DataFrame to load
            target_table: Target table name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(df)
            
            # Write to Parquet file
            parquet_path = str(DATA_WAREHOUSE_PATH / f"{target_table}.parquet")
            spark_df.write.mode("overwrite").parquet(parquet_path)
            
            # Update metadata in database
            with Session() as session:
                table = session.query(DataWarehouseTable).filter_by(name=target_table).first()
                if table:
                    table.row_count = df.shape[0]
                    table.last_updated = datetime.datetime.now()
                else:
                    table = DataWarehouseTable(
                        name=target_table,
                        description=f"Data warehouse table for {target_table}",
                        schema=json.dumps(df.dtypes.apply(lambda x: str(x)).to_dict()),
                        row_count=df.shape[0],
                        last_updated=datetime.datetime.now()
                    )
                    session.add(table)
                session.commit()
            
            logger.info(f"Loaded {df.shape[0]} rows to {target_table}")
            return True
        except Exception as e:
            logger.error(f"Error loading data to {target_table}: {e}")
            return False
    
    def query_data_warehouse(self, table_name: str, query: str = None) -> pd.DataFrame:
        """
        Query data from the data warehouse.
        
        Args:
            table_name: Name of the table to query
            query: SQL query to execute (if None, returns all data)
            
        Returns:
            DataFrame with query results
        """
        try:
            # Check if table exists
            parquet_path = DATA_WAREHOUSE_PATH / f"{table_name}.parquet"
            if not parquet_path.exists():
                logger.error(f"Table {table_name} not found in data warehouse")
                return pd.DataFrame()
            
            # Read Parquet file
            df = self.spark.read.parquet(str(parquet_path))
            
            # Execute query if provided
            if query:
                df.createOrReplaceTempView(table_name)
                result = self.spark.sql(query)
                return result.toPandas()
            
            return df.toPandas()
        except Exception as e:
            logger.error(f"Error querying data warehouse: {e}")
            return pd.DataFrame()
    
    def get_data_warehouse_tables(self) -> List[Dict[str, Any]]:
        """
        Get a list of tables in the data warehouse.
        
        Returns:
            List of table metadata
        """
        try:
            with Session() as session:
                tables = session.query(DataWarehouseTable).all()
                return [
                    {
                        "id": table.id,
                        "name": table.name,
                        "description": table.description,
                        "row_count": table.row_count,
                        "last_updated": table.last_updated.isoformat() if table.last_updated else None
                    }
                    for table in tables
                ]
        except Exception as e:
            logger.error(f"Error getting data warehouse tables: {e}")
            return []
    
    def stream_data_to_kafka(self, data: Dict[str, Any], topic: str) -> bool:
        """
        Stream data to a Kafka topic.
        
        Args:
            data: Data to stream
            topic: Kafka topic
            
        Returns:
            True if successful, False otherwise
        """
        if not self.kafka_producer:
            logger.error("Kafka producer not initialized")
            return False
        
        try:
            self.kafka_producer.send(topic, data)
            return True
        except Exception as e:
            logger.error(f"Error streaming data to Kafka: {e}")
            return False
    
    def close(self):
        """Close all connections."""
        if self.spark:
            self.spark.stop()
        
        if self.mongo_client:
            self.mongo_client.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        if self.kafka_consumer:
            self.kafka_consumer.close()


#######################################################
# Analytics and Visualization
#######################################################

class AdvancedVisualization:
    """
    Advanced visualization components using Plotly and D3.js integration.
    Provides interactive charts, dashboards, and visual analytics.
    """
    
    def __init__(self):
        """Initialize the visualization components."""
        self.color_schemes = {
            "default": px.colors.qualitative.Plotly,
            "dark": px.colors.qualitative.Dark24,
            "light": px.colors.qualitative.Pastel,
            "diverging": px.colors.diverging.Spectral,
            "sequential": px.colors.sequential.Viridis
        }
    
    def create_time_series_chart(self, df: pd.DataFrame, x_col: str, y_col: str, 
                                title: str = None, color_col: str = None, 
                                color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive time series chart.
        
        Args:
            df: DataFrame with time series data
            x_col: Column name for x-axis (time)
            y_col: Column name for y-axis (value)
            title: Chart title
            color_col: Column name for color grouping
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        if color_col:
            fig = px.line(
                df, x=x_col, y=y_col, color=color_col, 
                title=title, color_discrete_sequence=colors
            )
        else:
            fig = px.line(
                df, x=x_col, y=y_col, 
                title=title, color_discrete_sequence=colors
            )
        
        # Add range slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str,
                        title: str = None, color_col: str = None,
                        color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive bar chart.
        
        Args:
            df: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Column name for color grouping
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        if color_col:
            fig = px.bar(
                df, x=x_col, y=y_col, color=color_col,
                title=title, color_discrete_sequence=colors
            )
        else:
            fig = px.bar(
                df, x=x_col, y=y_col,
                title=title, color_discrete_sequence=colors
            )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                           title: str = None, color_col: str = None, size_col: str = None,
                           color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive scatter plot.
        
        Args:
            df: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Column name for color grouping
            size_col: Column name for point size
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_col, size=size_col,
            title=title, color_discrete_sequence=colors,
            hover_data=df.columns
        )
        
        # Add trendline
        fig.update_traces(marker=dict(line=dict(width=0.5, color='DarkSlateGrey')))
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_heatmap(self, df: pd.DataFrame, x_cols: List[str] = None, 
                      y_cols: List[str] = None, title: str = None,
                      color_scheme: str = "default") -> go.Figure:
        """
        Create a heatmap for correlation analysis.
        
        Args:
            df: DataFrame with data
            x_cols: List of column names for x-axis (if None, use all numeric columns)
            y_cols: List of column names for y-axis (if None, use all numeric columns)
            title: Chart title
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        # Select numeric columns if not specified
        if x_cols is None:
            x_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if y_cols is None:
            y_cols = x_cols
        
        # Calculate correlation matrix
        corr_df = df[x_cols].corr()
        
        # Create heatmap
        if color_scheme == "diverging":
            colorscale = px.colors.diverging.RdBu_r
        else:
            colorscale = px.colors.sequential.Viridis
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale=colorscale,
            zmin=-1, zmax=1
        ))
        
        # Add correlation values as text
        annotations = []
        for i, row in enumerate(corr_df.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=corr_df.columns[j],
                        y=corr_df.columns[i],
                        text=f"{value:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(value) > 0.5 else "black")
                    )
                )
        
        # Improve layout
        fig.update_layout(
            title=title if title else "Correlation Matrix",
            template="plotly_white",
            annotations=annotations,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame, names_col: str, values_col: str,
                        title: str = None, color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive pie chart.
        
        Args:
            df: DataFrame with data
            names_col: Column name for pie slice names
            values_col: Column name for pie slice values
            title: Chart title
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.pie(
            df, names=names_col, values=values_col,
            title=title, color_discrete_sequence=colors
        )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, x_col: str, y_col: str,
                       title: str = None, color_col: str = None,
                       color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive box plot.
        
        Args:
            df: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title
            color_col: Column name for color grouping
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.box(
            df, x=x_col, y=y_col, color=color_col,
            title=title, color_discrete_sequence=colors
        )
        
        # Add individual points
        fig.update_traces(boxpoints='all', jitter=0.3, pointpos=-1.8)
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        return fig
    
    def create_histogram(self, df: pd.DataFrame, x_col: str, nbins: int = 20,
                        title: str = None, color_col: str = None,
                        color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive histogram.
        
        Args:
            df: DataFrame with data
            x_col: Column name for x-axis
            nbins: Number of bins
            title: Chart title
            color_col: Column name for color grouping
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.histogram(
            df, x=x_col, color=color_col, nbins=nbins,
            title=title, color_discrete_sequence=colors
        )
        
        # Add KDE curve
        kde = df[x_col].plot.kde(ind=np.linspace(df[x_col].min(), df[x_col].max(), 100))
        kde_x = kde.lines[0].get_xdata()
        kde_y = kde.lines[0].get_ydata()
        plt.close()
        
        # Scale KDE curve to match histogram
        hist_max = fig.data[0].y.max()
        kde_max = kde_y.max()
        kde_y = kde_y * (hist_max / kde_max)
        
        fig.add_trace(
            go.Scatter(
                x=kde_x, y=kde_y,
                mode='lines', name='KDE',
                line=dict(color='rgba(0,0,0,0.8)', width=2)
            )
        )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40),
            bargap=0.1
        )
        
        return fig
    
    def create_3d_scatter(self, df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                         title: str = None, color_col: str = None, size_col: str = None,
                         color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive 3D scatter plot.
        
        Args:
            df: DataFrame with data
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            z_col: Column name for z-axis
            title: Chart title
            color_col: Column name for color grouping
            size_col: Column name for point size
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.scatter_3d(
            df, x=x_col, y=y_col, z=z_col, color=color_col, size=size_col,
            title=title, color_discrete_sequence=colors,
            hover_data=df.columns
        )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col
            )
        )
        
        return fig
    
    def create_network_graph(self, nodes: pd.DataFrame, edges: pd.DataFrame,
                            title: str = None, color_col: str = None,
                            color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive network graph.
        
        Args:
            nodes: DataFrame with node data (must have 'id' column)
            edges: DataFrame with edge data (must have 'source' and 'target' columns)
            title: Chart title
            color_col: Column name for node color
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes
        for _, node in nodes.iterrows():
            node_attrs = node.to_dict()
            G.add_node(node['id'], **node_attrs)
        
        # Add edges
        for _, edge in edges.iterrows():
            G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        if color_col and color_col in nodes.columns:
            node_colors = nodes[color_col].tolist()
            colorscale = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        else:
            node_colors = [len(list(G.neighbors(node))) for node in G.nodes()]
            colorscale = 'YlGnBu'
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale=colorscale,
                color=node_colors,
                size=10,
                line_width=2
            )
        )
        
        # Add node text for hover
        node_text = [f"ID: {node}<br>Connections: {len(list(G.neighbors(node)))}" for node in G.nodes()]
        node_trace.text = node_text
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          title=title if title else 'Network Graph',
                          titlefont_size=16,
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                      ))
        
        return fig
    
    def create_sunburst_chart(self, df: pd.DataFrame, path_cols: List[str], values_col: str,
                             title: str = None, color_col: str = None,
                             color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive sunburst chart for hierarchical data.
        
        Args:
            df: DataFrame with data
            path_cols: List of column names for hierarchical path
            values_col: Column name for values
            title: Chart title
            color_col: Column name for color
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.sunburst(
            df, path=path_cols, values=values_col, color=color_col,
            title=title, color_discrete_sequence=colors
        )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def create_treemap(self, df: pd.DataFrame, path_cols: List[str], values_col: str,
                      title: str = None, color_col: str = None,
                      color_scheme: str = "default") -> go.Figure:
        """
        Create an interactive treemap for hierarchical data.
        
        Args:
            df: DataFrame with data
            path_cols: List of column names for hierarchical path
            values_col: Column name for values
            title: Chart title
            color_col: Column name for color
            color_scheme: Color scheme name
            
        Returns:
            Plotly Figure object
        """
        colors = self.color_schemes.get(color_scheme, self.color_schemes["default"])
        
        fig = px.treemap(
            df, path=path_cols, values=values_col, color=color_col,
            title=title, color_discrete_sequence=colors
        )
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
    
    def create_gauge_chart(self, value: float, title: str = None, min_val: float = 0,
                          max_val: float = 100, threshold_values: List[float] = None,
                          threshold_colors: List[str] = None) -> go.Figure:
        """
        Create a gauge chart for KPI visualization.
        
        Args:
            value: Value to display
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            threshold_values: List of threshold values
            threshold_colors: List of colors for thresholds
            
        Returns:
            Plotly Figure object
        """
        if threshold_values is None:
            threshold_values = [max_val * 0.33, max_val * 0.66, max_val]
        
        if threshold_colors is None:
            threshold_colors = ["red", "yellow", "green"]
        
        # Calculate steps for gauge
        steps = []
        for i in range(len(threshold_values)):
            start = threshold_values[i - 1] if i > 0 else min_val
            end = threshold_values[i]
            steps.append(
                dict(
                    range=[start, end],
                    color=threshold_colors[i],
                    thickness=0.75
                )
            )
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text=title if title else "Gauge Chart"),
            gauge=dict(
                axis=dict(range=[min_val, max_val]),
                bar=dict(color="darkblue"),
                steps=steps,
                threshold=dict(
                    line=dict(color="black", width=4),
                    thickness=0.75,
                    value=value
                )
            )
        ))
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def create_bullet_chart(self, value: float, target: float, title: str = None,
                           min_val: float = 0, max_val: float = 100,
                           ranges: List[float] = None, colors: List[str] = None) -> go.Figure:
        """
        Create a bullet chart for KPI visualization.
        
        Args:
            value: Actual value
            target: Target value
            title: Chart title
            min_val: Minimum value
            max_val: Maximum value
            ranges: List of range values
            colors: List of colors for ranges
            
        Returns:
            Plotly Figure object
        """
        if ranges is None:
            ranges = [max_val * 0.33, max_val * 0.66, max_val]
        
        if colors is None:
            colors = ["#F25757", "#FFD15C", "#AAD922"]
        
        # Create bullet chart
        fig = go.Figure()
        
        # Add ranges
        for i in range(len(ranges)):
            start = ranges[i - 1] if i > 0 else min_val
            end = ranges[i]
            fig.add_trace(go.Bar(
                x=[end - start],
                y=["Performance"],
                orientation="h",
                marker=dict(color=colors[i]),
                base=start,
                name=f"Range {i+1}",
                hoverinfo="none"
            ))
        
        # Add actual value bar
        fig.add_trace(go.Bar(
            x=[value],
            y=["Performance"],
            orientation="h",
            marker=dict(color="black"),
            width=0.2,
            name="Actual",
            hovertemplate="Actual: %{x}<extra></extra>"
        ))
        
        # Add target marker
        fig.add_trace(go.Scatter(
            x=[target],
            y=["Performance"],
            mode="markers",
            marker=dict(symbol="line-ns", color="red", size=20, line=dict(width=2)),
            name="Target",
            hovertemplate="Target: %{x}<extra></extra>"
        ))
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            title=title if title else "Bullet Chart",
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis=dict(range=[min_val, max_val]),
            barmode="overlay",
            height=150
        )
        
        return fig
    
    def create_sparkline(self, data: List[float], title: str = None, 
                        show_markers: bool = False) -> go.Figure:
        """
        Create a sparkline chart for compact trend visualization.
        
        Args:
            data: List of values
            title: Chart title
            show_markers: Whether to show markers
            
        Returns:
            Plotly Figure object
        """
        # Create x values
        x = list(range(len(data)))
        
        # Create sparkline
        fig = go.Figure()
        
        # Add line
        fig.add_trace(go.Scatter(
            x=x,
            y=data,
            mode="lines" if not show_markers else "lines+markers",
            line=dict(width=2, color="blue"),
            marker=dict(size=4),
            hoverinfo="y"
        ))
        
        # Add endpoints
        fig.add_trace(go.Scatter(
            x=[0, len(data) - 1],
            y=[data[0], data[-1]],
            mode="markers",
            marker=dict(size=6, color="red"),
            hoverinfo="y"
        ))
        
        # Improve layout
        fig.update_layout(
            template="plotly_white",
            title=title,
            showlegend=False,
            margin=dict(l=5, r=5, t=30, b=5),
            height=100,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
        )
        
        return fig
    
    def create_d3_visualization(self, viz_type: str, data: Dict[str, Any]) -> str:
        """
        Create a D3.js visualization.
        
        Args:
            viz_type: Type of visualization
            data: Data for visualization
            
        Returns:
            HTML string with D3.js visualization
        """
        # This is a placeholder implementation
        # In a real implementation, this would generate D3.js code
        
        if viz_type == "force_directed":
            # Force-directed graph
            html = """
            <div id="d3-force-directed"></div>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>
                // D3.js force-directed graph code would go here
                // This is a placeholder
                const width = 800;
                const height = 600;
                
                const svg = d3.select("#d3-force-directed")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // Force simulation setup would go here
                // Nodes and links would be created from data
            </script>
            """
        elif viz_type == "choropleth":
            # Choropleth map
            html = """
            <div id="d3-choropleth"></div>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script src="https://d3js.org/topojson.v3.min.js"></script>
            <script>
                // D3.js choropleth map code would go here
                // This is a placeholder
                const width = 800;
                const height = 500;
                
                const svg = d3.select("#d3-choropleth")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // Map projection and path generator would go here
                // GeoJSON data would be loaded and rendered
            </script>
            """
        else:
            # Generic D3.js visualization
            html = """
            <div id="d3-visualization"></div>
            <script src="https://d3js.org/d3.v7.min.js"></script>
            <script>
                // Generic D3.js visualization code would go here
                // This is a placeholder
                const width = 800;
                const height = 400;
                
                const svg = d3.select("#d3-visualization")
                    .append("svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // Visualization-specific code would go here
            </script>
            """
        
        return html


#######################################################
# Predictive Analytics and Forecasting
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
    
    def train_prophet_model(self, df: pd.DataFrame, date_col: str, value_col: str,
                           model_name: str, forecast_periods: int = 30,
                           seasonality_mode: str = "additive",
                           include_history: bool = True) -> Dict[str, Any]:
        """
        Train a Prophet forecasting model.
        
        Args:
            df: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            model_name: Name for the model
            forecast_periods: Number of periods to forecast
            seasonality_mode: Seasonality mode ('additive' or 'multiplicative')
            include_history: Whether to include history in forecast
            
        Returns:
            Dictionary with model and forecast results
        """
        try:
            # Prepare data for Prophet
            prophet_df = df[[date_col, value_col]].copy()
            prophet_df.columns = ['ds', 'y']
            
            # Create and train Prophet model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality='auto'
            )
            
            # Add holidays if available
            # model.add_country_holidays(country_name='US')
            
            # Fit the model
            model.fit(prophet_df)
            
            # Create future dataframe for forecasting
            future = model.make_future_dataframe(periods=forecast_periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Store model and forecast
            self.models[model_name] = {
                'model': model,
                'type': 'prophet',
                'date_col': date_col,
                'value_col': value_col,
                'trained_at': datetime.datetime.now()
            }
            
            self.forecasts[model_name] = forecast
            
            # Calculate accuracy metrics on training data
            if include_history:
                y_true = prophet_df['y'].values
                y_pred = forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'yhat'].values
                
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
                'forecast': forecast,
                'metrics': metrics,
                'components': {
                    'trend': forecast['trend'].values,
                    'seasonal': forecast['yearly'] + forecast['weekly'] + forecast['daily'] \
                        if 'daily' in forecast.columns else forecast['yearly'] + forecast['weekly']
                }
            }
            
            # Store model in database
            self._store_model_in_db(model_name, 'prophet', metrics)
            
            # Store forecasts in database
            self._store_forecasts_in_db(model_name, forecast)
            
            return result
        
        except Exception as e:
            logger.error(f"Error training Prophet model: {e}")
            return {'error': str(e)}
    
    def train_arima_model(self, df: pd.DataFrame, date_col: str, value_col: str,
                         model_name: str, forecast_periods: int = 30,
                         order: Tuple[int, int, int] = (1, 1, 1),
                         seasonal_order: Tuple[int, int, int, int] = None) -> Dict[str, Any]:
        """
        Train an ARIMA forecasting model.
        
        Args:
            df: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            model_name: Name for the model
            forecast_periods: Number of periods to forecast
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
            
        Returns:
            Dictionary with model and forecast results
        """
        try:
            # Prepare data for ARIMA
            df = df.sort_values(date_col)
            ts = df[value_col].values
            
            # Create and train ARIMA model
            if seasonal_order:
                model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(ts, order=order)
            
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            forecast_dates = pd.date_range(
                start=df[date_col].max() + pd.Timedelta(days=1),
                periods=forecast_periods
            )
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast,
                'yhat_lower': forecast - 1.96 * model_fit.params['sigma2'] ** 0.5,
                'yhat_upper': forecast + 1.96 * model_fit.params['sigma2'] ** 0.5
            })
            
            # Add historical data
            historical_df = pd.DataFrame({
                'ds': df[date_col],
                'y': df[value_col],
                'yhat': model_fit.fittedvalues,
                'yhat_lower': model_fit.fittedvalues - 1.96 * model_fit.params['sigma2'] ** 0.5,
                'yhat_upper': model_fit.fittedvalues + 1.96 * model_fit.params['sigma2'] ** 0.5
            })
            
            combined_df = pd.concat([historical_df, forecast_df])
            
            # Store model and forecast
            self.models[model_name] = {
                'model': model_fit,
                'type': 'arima',
                'date_col': date_col,
                'value_col': value_col,
                'trained_at': datetime.datetime.now()
            }
            
            self.forecasts[model_name] = combined_df
            
            # Calculate accuracy metrics on training data
            y_true = df[value_col].values
            y_pred = model_fit.fittedvalues
            
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'aic': model_fit.aic,
                'bic': model_fit.bic
            }
            
            # Prepare result
            result = {
                'model_name': model_name,
                'forecast': combined_df,
                'metrics': metrics
            }
            
            # Store model in database
            self._store_model_in_db(model_name, 'arima', metrics)
            
            # Store forecasts in database
            self._store_forecasts_in_db(model_name, combined_df)
            
            return result
        
        except Exception as e:
            logger.error(f"Error training ARIMA model: {e}")
            return {'error': str(e)}
    
    def train_lstm_model(self, df: pd.DataFrame, date_col: str, value_col: str,
                        model_name: str, forecast_periods: int = 30,
                        sequence_length: int = 10, epochs: int = 50,
                        batch_size: int = 32) -> Dict[str, Any]:
        """
        Train an LSTM forecasting model.
        
        Args:
            df: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            model_name: Name for the model
            forecast_periods: Number of periods to forecast
            sequence_length: Length of input sequences
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary with model and forecast results
        """
        try:
            # Prepare data for LSTM
            df = df.sort_values(date_col)
            ts = df[value_col].values
            
            # Scale data
            scaler = SkStandardScaler()
            ts_scaled = scaler.fit_transform(ts.reshape(-1, 1)).flatten()
            
            # Create sequences
            X = []
            y = []
            for i in range(len(ts_scaled) - sequence_length):
                X.append(ts_scaled[i:i+sequence_length])
                y.append(ts_scaled[i+sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Reshape for LSTM [samples, time steps, features]
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                verbose=0
            )
            
            # Generate forecast
            forecast = []
            current_sequence = ts_scaled[-sequence_length:].tolist()
            
            for _ in range(forecast_periods):
                # Predict next value
                current_sequence_array = np.array(current_sequence).reshape((1, sequence_length, 1))
                next_val = model.predict(current_sequence_array, verbose=0)[0][0]
                
                # Add to forecast
                forecast.append(next_val)
                
                # Update sequence
                current_sequence = current_sequence[1:] + [next_val]
            
            # Inverse transform
            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
            
            # Create forecast dates
            forecast_dates = pd.date_range(
                start=df[date_col].max() + pd.Timedelta(days=1),
                periods=forecast_periods
            )
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast
            })
            
            # Calculate confidence intervals (simplified)
            forecast_std = np.std(ts) * 0.5  # Simplified approach
            forecast_df['yhat_lower'] = forecast_df['yhat'] - 1.96 * forecast_std
            forecast_df['yhat_upper'] = forecast_df['yhat'] + 1.96 * forecast_std
            
            # Add historical data
            y_pred_train = model.predict(X_train, verbose=0).flatten()
            y_pred_test = model.predict(X_test, verbose=0).flatten()
            y_pred = np.concatenate([y_pred_train, y_pred_test])
            y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            
            # Pad with NaN for the first sequence_length values
            y_pred_full = np.concatenate([np.full(sequence_length, np.nan), y_pred])
            
            historical_df = pd.DataFrame({
                'ds': df[date_col],
                'y': df[value_col],
                'yhat': y_pred_full[:len(df)]
            })
            
            historical_df['yhat_lower'] = historical_df['yhat'] - 1.96 * forecast_std
            historical_df['yhat_upper'] = historical_df['yhat'] + 1.96 * forecast_std
            
            combined_df = pd.concat([historical_df, forecast_df])
            
            # Store model and forecast
            self.models[model_name] = {
                'model': model,
                'type': 'lstm',
                'date_col': date_col,
                'value_col': value_col,
                'scaler': scaler,
                'sequence_length': sequence_length,
                'trained_at': datetime.datetime.now()
            }
            
            self.forecasts[model_name] = combined_df
            
            # Calculate accuracy metrics on training data
            y_true = df[value_col].values[sequence_length:]
            y_pred = y_pred_full[sequence_length:len(df)]
            
            # Filter out NaN values
            mask = ~np.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Prepare result
            result = {
                'model_name': model_name,
                'forecast': combined_df,
                'metrics': metrics,
                'training_history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss']
                }
            }
            
            # Store model in database
            self._store_model_in_db(model_name, 'lstm', metrics)
            
            # Store forecasts in database
            self._store_forecasts_in_db(model_name, combined_df)
            
            return result
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {'error': str(e)}
    
    def train_ensemble_forecast(self, df: pd.DataFrame, date_col: str, value_col: str,
                              model_name: str, forecast_periods: int = 30,
                              models: List[str] = None) -> Dict[str, Any]:
        """
        Train an ensemble forecasting model.
        
        Args:
            df: DataFrame with time series data
            date_col: