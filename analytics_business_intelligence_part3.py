#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analytics_business_intelligence_part3.py - Advanced Analytics and Business Intelligence System (Part 3)

This module completes the comprehensive analytics and business intelligence system for the
Skyscope Sentinel Intelligence AI platform, implementing reporting automation, real-time
streaming analytics, FastAPI application, and integration with the main Skyscope system.

Features:
1. Reporting automation with PDF, Excel, and web reports
2. Real-time streaming analytics with WebSocket broadcasting
3. FastAPI application with comprehensive endpoints
4. Dashboard and KPI management endpoints
5. WebSocket endpoints for real-time updates
6. Integration with the main Skyscope system
7. Performance optimization for 10,000 employees
8. Example usage and initialization scripts
9. Comprehensive documentation and summary

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
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, File, UploadFile, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
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
        logging.FileHandler("analytics_bi_part3.log"),
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
TEMPLATES_PATH = Path("./templates")

# Create necessary directories
DATA_WAREHOUSE_PATH.mkdir(exist_ok=True)
REPORTS_PATH.mkdir(exist_ok=True)
TEMPLATES_PATH.mkdir(exist_ok=True)

# SQLAlchemy Base
Base = declarative_base()

#######################################################
# Reporting Automation (Continued)
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
        self.jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_PATH))
        self.load_reports()
        self.load_templates()
    
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
    
    def load_templates(self):
        """Load report templates."""
        try:
            # Create default templates if they don't exist
            self._create_default_templates()
            
            # Load templates from the templates directory
            for template_file in TEMPLATES_PATH.glob("*.html"):
                template_name = template_file.stem
                self.templates[template_name] = str(template_file)
            
            logger.info(f"Loaded {len(self.templates)} report templates")
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def _create_default_templates(self):
        """Create default report templates if they don't exist."""
        # Default PDF template
        pdf_template_path = TEMPLATES_PATH / "default_pdf_template.html"
        if not pdf_template_path.exists():
            pdf_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{{ report_title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; text-align: center; }
                    h2 { color: #34495e; margin-top: 20px; }
                    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    th { background-color: #3498db; color: white; padding: 8px; text-align: left; }
                    td { padding: 8px; border-bottom: 1px solid #ddd; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    .footer { text-align: center; margin-top: 30px; font-size: 0.8em; color: #7f8c8d; }
                    .chart { width: 100%; margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>{{ report_title }}</h1>
                <p>Generated on: {{ generation_date }}</p>
                
                {% for section in sections %}
                    <h2>{{ section.title }}</h2>
                    <p>{{ section.description }}</p>
                    
                    {% if section.table %}
                        <table>
                            <thead>
                                <tr>
                                {% for column in section.table.columns %}
                                    <th>{{ column }}</th>
                                {% endfor %}
                                </tr>
                            </thead>
                            <tbody>
                                {% for row in section.table.rows %}
                                <tr>
                                    {% for cell in row %}
                                    <td>{{ cell }}</td>
                                    {% endfor %}
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    {% endif %}
                    
                    {% if section.chart %}
                        <div class="chart">
                            <img src="data:image/png;base64,{{ section.chart }}" alt="Chart" style="width: 100%;">
                        </div>
                    {% endif %}
                {% endfor %}
                
                <div class="footer">
                    <p>Skyscope Sentinel Intelligence AI - Analytics & Business Intelligence</p>
                </div>
            </body>
            </html>
            """
            with open(pdf_template_path, 'w') as f:
                f.write(pdf_template)
        
        # Default Excel template (metadata)
        excel_template_path = TEMPLATES_PATH / "default_excel_template.json"
        if not excel_template_path.exists():
            excel_template = {
                "sheets": [
                    {
                        "name": "Summary",
                        "columns": ["Metric", "Value", "Change", "Status"],
                        "formats": {
                            "header": {"bold": True, "bg_color": "#3498db", "font_color": "white"},
                            "data": {"num_format": "0.00"},
                            "status_good": {"bg_color": "#2ecc71"},
                            "status_warning": {"bg_color": "#f39c12"},
                            "status_bad": {"bg_color": "#e74c3c"}
                        }
                    },
                    {
                        "name": "Details",
                        "dynamic": True
                    },
                    {
                        "name": "Charts",
                        "charts": [
                            {
                                "type": "line",
                                "data_range": "Details!$A$1:$D$10",
                                "title": "Trend Analysis"
                            }
                        ]
                    }
                ]
            }
            with open(excel_template_path, 'w') as f:
                json.dump(excel_template, f, indent=2)
        
        # Default web template
        web_template_path = TEMPLATES_PATH / "default_web_template.html"
        if not web_template_path.exists():
            web_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{{ report_title }}</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body { padding-top: 20px; }
                    .report-header { margin-bottom: 30px; }
                    .section { margin-bottom: 40px; }
                    .chart-container { height: 300px; margin-bottom: 20px; }
                    .table-responsive { margin-bottom: 20px; }
                    .footer { text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #eee; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="report-header">
                        <h1 class="text-center">{{ report_title }}</h1>
                        <p class="text-center text-muted">Generated on: {{ generation_date }}</p>
                    </div>
                    
                    {% for section in sections %}
                    <div class="section">
                        <h2>{{ section.title }}</h2>
                        <p>{{ section.description }}</p>
                        
                        {% if section.table %}
                        <div class="table-responsive">
                            <table class="table table-striped table-hover">
                                <thead class="table-primary">
                                    <tr>
                                    {% for column in section.table.columns %}
                                        <th>{{ column }}</th>
                                    {% endfor %}
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for row in section.table.rows %}
                                    <tr>
                                        {% for cell in row %}
                                        <td>{{ cell }}</td>
                                        {% endfor %}
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% endif %}
                        
                        {% if section.chart %}
                        <div class="chart-container">
                            <canvas id="chart-{{ loop.index }}"></canvas>
                            <script>
                                var ctx = document.getElementById('chart-{{ loop.index }}').getContext('2d');
                                var chartData = {{ section.chart_data|safe }};
                                var chart = new Chart(ctx, chartData);
                            </script>
                        </div>
                        {% endif %}
                    </div>
                    {% endfor %}
                    
                    <div class="footer">
                        <p class="text-muted">Skyscope Sentinel Intelligence AI - Analytics & Business Intelligence</p>
                    </div>
                </div>
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            </body>
            </html>
            """
            with open(web_template_path, 'w') as f:
                f.write(web_template)
    
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
                                kpis = session.query(KPI).filter_by(category=category).all()
                            else:
                                kpis = session.query(KPI).all()
                            
                            # Convert to DataFrame
                            kpi_data = []
                            for kpi in kpis:
                                kpi_data.append({
                                    'id': kpi.id,
                                    'name': kpi.name,
                                    'description': kpi.description,
                                    'category': kpi.category,
                                    'current_value': kpi.current_value,
                                    'target_value': kpi.target_value,
                                    'unit': kpi.unit,
                                    'status': kpi.status,
                                    'updated_at': kpi.updated_at
                                })
                            
                            return pd.DataFrame(kpi_data)
                    
                    elif query_def.get('type') == 'forecast':
                        # Query forecast data
                        model_name = query_def.get('model_name')
                        with Session() as session:
                            model = session.query(ForecastModel).filter_by(name=model_name).first()
                            if not model:
                                return pd.DataFrame()
                            
                            forecasts = session.query(Forecast).filter_by(model_id=model.id).all()
                            
                            # Convert to DataFrame
                            forecast_data = []
                            for forecast in forecasts:
                                forecast_data.append({
                                    'timestamp': forecast.timestamp,
                                    'value': forecast.value,
                                    'lower_bound': forecast.lower_bound,
                                    'upper_bound': forecast.upper_bound
                                })
                            
                            return pd.DataFrame(forecast_data)
                    
                    elif query_def.get('type') == 'anomaly':
                        # Query anomaly detection data
                        metric_name = query_def.get('metric_name')
                        with Session() as session:
                            if metric_name:
                                anomalies = session.query(AnomalyDetection).filter_by(metric_name=metric_name).all()
                            else:
                                anomalies = session.query(AnomalyDetection).all()
                            
                            # Convert to DataFrame
                            anomaly_data = []
                            for anomaly in anomalies:
                                anomaly_data.append({
                                    'id': anomaly.id,
                                    'metric_name': anomaly.metric_name,
                                    'metric_value': anomaly.metric_value,
                                    'expected_value': anomaly.expected_value,
                                    'deviation_score': anomaly.deviation_score,
                                    'is_anomaly': anomaly.is_anomaly,
                                    'detection_method': anomaly.detection_method,
                                    'timestamp': anomaly.timestamp
                                })
                            
                            return pd.DataFrame(anomaly_data)
                    
                    elif query_def.get('type') == 'olap':
                        # Execute OLAP query
                        cube_name = query_def.get('cube_name')
                        dimensions = query_def.get('dimensions')
                        measures = query_def.get('measures')
                        filters = query_def.get('filters')
                        group_by = query_def.get('group_by')
                        
                        # Create OLAP analysis instance
                        olap = OLAPAnalysis()
                        
                        # Execute query
                        return olap.cube_query(
                            cube_name=cube_name,
                            dimensions=dimensions,
                            measures=measures,
                            filters=filters,
                            group_by=group_by
                        )
                    
                    elif query_def.get('type') == 'data_warehouse':
                        # Query data warehouse
                        table_name = query_def.get('table_name')
                        custom_query = query_def.get('query')
                        
                        # Create data warehouse instance
                        dw = DataWarehouse()
                        
                        # Execute query
                        return dw.query_data_warehouse(
                            table_name=table_name,
                            query=custom_query
                        )
                    
                    else:
                        return pd.DataFrame()
                
                except json.JSONDecodeError:
                    return {'error': f"Invalid query definition: {query}"}
        
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {'error': str(e)}
    
    def _generate_pdf_report(self, data: pd.DataFrame, report_name: str, 
                           template: str = None) -> str:
        """
        Generate a PDF report.
        
        Args:
            data: DataFrame with report data
            report_name: Name of the report
            template: Template name or path
            
        Returns:
            Path to the generated report
        """
        try:
            # Use default template if not specified
            if not template:
                template = "default_pdf_template"
            
            # Create report directory if it doesn't exist
            report_dir = REPORTS_PATH / "pdf"
            report_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name.replace(' ', '_')}_{timestamp}.pdf"
            report_path = report_dir / filename
            
            # Create document
            doc = SimpleDocTemplate(
                str(report_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Create content
            content = []
            
            # Add title
            content.append(Paragraph(report_name, title_style))
            content.append(Spacer(1, 12))
            
            # Add generation date
            generation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content.append(Paragraph(f"Generated on: {generation_date}", normal_style))
            content.append(Spacer(1, 24))
            
            # Add data summary
            content.append(Paragraph("Data Summary", heading_style))
            content.append(Spacer(1, 12))
            
            # Add data shape
            content.append(Paragraph(f"Number of records: {len(data)}", normal_style))
            content.append(Paragraph(f"Number of columns: {len(data.columns)}", normal_style))
            content.append(Spacer(1, 12))
            
            # Add data table
            if not data.empty:
                # Prepare table data
                table_data = [data.columns.tolist()]
                for _, row in data.head(100).iterrows():  # Limit to 100 rows
                    table_data.append([str(cell) for cell in row])
                
                # Create table
                data_table = Table(table_data)
                
                # Add table style
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ])
                data_table.setStyle(table_style)
                
                content.append(data_table)
                content.append(Spacer(1, 24))
            
            # Add data visualization if applicable
            if len(data) > 0 and data.select_dtypes(include=[np.number]).columns.any():
                content.append(Paragraph("Data Visualization", heading_style))
                content.append(Spacer(1, 12))
                
                # Create a simple bar chart
                numeric_cols = data.select_dtypes(include=[np.number]).columns[:5]  # Limit to 5 columns
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(8, 6))
                    data[numeric_cols].mean().plot(kind='bar')
                    plt.title("Average Values")
                    plt.tight_layout()
                    
                    # Save chart to temp file
                    chart_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
                    plt.savefig(chart_path)
                    plt.close()
                    
                    # Add chart to content
                    content.append(Image(chart_path, width=450, height=300))
                    
                    # Clean up temp file
                    os.unlink(chart_path)
            
            # Add footer
            content.append(Spacer(1, 36))
            content.append(Paragraph("Skyscope Sentinel Intelligence AI - Analytics & Business Intelligence", normal_style))
            
            # Build PDF
            doc.build(content)
            
            logger.info(f"Generated PDF report: {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return str(e)
    
    def _generate_excel_report(self, data: pd.DataFrame, report_name: str, 
                             template: str = None) -> str:
        """
        Generate an Excel report.
        
        Args:
            data: DataFrame with report data
            report_name: Name of the report
            template: Template name or path
            
        Returns:
            Path to the generated report
        """
        try:
            # Create report directory if it doesn't exist
            report_dir = REPORTS_PATH / "excel"
            report_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name.replace(' ', '_')}_{timestamp}.xlsx"
            report_path = report_dir / filename
            
            # Load template configuration if provided
            template_config = None
            if template:
                template_path = TEMPLATES_PATH / f"{template}.json"
                if template_path.exists():
                    with open(template_path, 'r') as f:
                        template_config = json.load(f)
            
            # Use default template if not specified or not found
            if not template_config:
                template_path = TEMPLATES_PATH / "default_excel_template.json"
                if template_path.exists():
                    with open(template_path, 'r') as f:
                        template_config = json.load(f)
                else:
                    # Simple default configuration
                    template_config = {
                        "sheets": [
                            {
                                "name": "Data",
                                "dynamic": True
                            },
                            {
                                "name": "Summary",
                                "columns": ["Metric", "Value"]
                            }
                        ]
                    }
            
            # Create Excel workbook
            workbook = xlsxwriter.Workbook(str(report_path))
            
            # Add data sheet
            data_sheet = workbook.add_worksheet("Data")
            
            # Write headers
            for col_idx, col_name in enumerate(data.columns):
                data_sheet.write(0, col_idx, col_name, workbook.add_format({'bold': True, 'bg_color': '#3498db', 'font_color': 'white'}))
            
            # Write data
            for row_idx, row in enumerate(data.values):
                for col_idx, cell_value in enumerate(row):
                    data_sheet.write(row_idx + 1, col_idx, cell_value)
            
            # Add summary sheet
            summary_sheet = workbook.add_worksheet("Summary")
            
            # Write summary headers
            summary_headers = ["Metric", "Value"]
            for col_idx, header in enumerate(summary_headers):
                summary_sheet.write(0, col_idx, header, workbook.add_format({'bold': True, 'bg_color': '#3498db', 'font_color': 'white'}))
            
            # Write summary data
            summary_data = [
                ["Report Name", report_name],
                ["Generated On", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ["Number of Records", len(data)],
                ["Number of Columns", len(data.columns)]
            ]
            
            for row_idx, (metric, value) in enumerate(summary_data):
                summary_sheet.write(row_idx + 1, 0, metric)
                summary_sheet.write(row_idx + 1, 1, value)
            
            # Add numeric summaries if applicable
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                row_idx = len(summary_data) + 2
                summary_sheet.write(row_idx, 0, "Numeric Columns Summary", workbook.add_format({'bold': True}))
                row_idx += 1
                
                for col in numeric_cols:
                    summary_sheet.write(row_idx, 0, f"{col} (Mean)")
                    summary_sheet.write(row_idx, 1, data[col].mean())
                    row_idx += 1
                    
                    summary_sheet.write(row_idx, 0, f"{col} (Min)")
                    summary_sheet.write(row_idx, 1, data[col].min())
                    row_idx += 1
                    
                    summary_sheet.write(row_idx, 0, f"{col} (Max)")
                    summary_sheet.write(row_idx, 1, data[col].max())
                    row_idx += 1
                    
                    summary_sheet.write(row_idx, 0, f"{col} (Std)")
                    summary_sheet.write(row_idx, 1, data[col].std())
                    row_idx += 1
            
            # Add chart if applicable
            if len(numeric_cols) > 0:
                chart_sheet = workbook.add_worksheet("Charts")
                
                # Create a column chart
                chart = workbook.add_chart({'type': 'column'})
                
                # Configure the chart
                for i, col in enumerate(numeric_cols[:5]):  # Limit to 5 columns
                    col_letter = chr(66 + i)  # B, C, D, ...
                    chart.add_series({
                        'name': f'Data!${col_letter}$1',
                        'categories': 'Data!$A$2:$A$11',
                        'values': f'Data!${col_letter}$2:${col_letter}$11',
                    })
                
                # Set chart title and labels
                chart.set_title({'name': 'Data Visualization'})
                chart.set_x_axis({'name': 'Index'})
                chart.set_y_axis({'name': 'Value'})
                
                # Insert the chart into the chart sheet
                chart_sheet.insert_chart('B2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
            
            # Close workbook
            workbook.close()
            
            logger.info(f"Generated Excel report: {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"Error generating Excel report: {e}")
            return str(e)
    
    def _generate_web_report(self, data: pd.DataFrame, report_name: str, 
                           template: str = None) -> str:
        """
        Generate a web-based report.
        
        Args:
            data: DataFrame with report data
            report_name: Name of the report
            template: Template name or path
            
        Returns:
            Path to the generated report
        """
        try:
            # Use default template if not specified
            if not template:
                template = "default_web_template"
            
            # Create report directory if it doesn't exist
            report_dir = REPORTS_PATH / "web"
            report_dir.mkdir(exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report_name.replace(' ', '_')}_{timestamp}.html"
            report_path = report_dir / filename
            
            # Prepare template data
            template_data = {
                'report_title': report_name,
                'generation_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'sections': []
            }
            
            # Add data summary section
            data_summary_section = {
                'title': 'Data Summary',
                'description': f'This report contains {len(data)} records with {len(data.columns)} columns.',
                'table': {
                    'columns': ['Metric', 'Value'],
                    'rows': [
                        ['Number of Records', len(data)],
                        ['Number of Columns', len(data.columns)]
                    ]
                }
            }
            template_data['sections'].append(data_summary_section)
            
            # Add data preview section
            data_preview_section = {
                'title': 'Data Preview',
                'description': 'Preview of the first 10 records:',
                'table': {
                    'columns': data.columns.tolist(),
                    'rows': data.head(10).values.tolist()
                }
            }
            template_data['sections'].append(data_preview_section)
            
            # Add data visualization if applicable
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Create chart data for JavaScript Chart.js
                chart_data = {
                    'type': 'bar',
                    'data': {
                        'labels': data.index[:20].astype(str).tolist(),  # Limit to 20 points
                        'datasets': []
                    },
                    'options': {
                        'responsive': True,
                        'plugins': {
                            'legend': {
                                'position': 'top',
                            },
                            'title': {
                                'display': True,
                                'text': 'Data Visualization'
                            }
                        }
                    }
                }
                
                # Add datasets for each numeric column (limit to 5)
                colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
                for i, col in enumerate(numeric_cols[:5]):
                    dataset = {
                        'label': col,
                        'data': data[col][:20].tolist(),  # Limit to 20 points
                        'backgroundColor': colors[i % len(colors)]
                    }
                    chart_data['data']['datasets'].append(dataset)
                
                # Add visualization section
                visualization_section = {
                    'title': 'Data Visualization',
                    'description': 'Visual representation of the data:',
                    'chart': True,
                    'chart_data': json.dumps(chart_data)
                }
                template_data['sections'].append(visualization_section)
            
            # Add statistics section if applicable
            if len(numeric_cols) > 0:
                stats_section = {
                    'title': 'Statistical Summary',
                    'description': 'Statistical summary of numeric columns:',
                    'table': {
                        'columns': ['Column', 'Mean', 'Min', 'Max', 'Std'],
                        'rows': []
                    }
                }
                
                for col in numeric_cols:
                    stats_section['table']['rows'].append([
                        col,
                        f"{data[col].mean():.2f}",
                        f"{data[col].min():.2f}",
                        f"{data[col].max():.2f}",
                        f"{data[col].std():.2f}"
                    ])
                
                template_data['sections'].append(stats_section)
            
            # Render template
            template_obj = self.jinja_env.get_template(f"{template}.html")
            html_content = template_obj.render(**template_data)
            
            # Write to file
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated web report: {report_path}")
            return str(report_path)
        
        except Exception as e:
            logger.error(f"Error generating web report: {e}")
            return str(e)
    
    def _send_report(self, report_path: str, report_name: str, recipients: List[str]) -> bool:
        """
        Send a report to recipients.
        
        Args:
            report_path: Path to the report
            report_name: Name of the report
            recipients: List of recipient email addresses
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This is a placeholder implementation
            # In a real implementation, this would use an email library or API
            
            logger.info(f"Sending report {report_name} to {len(recipients)} recipients")
            
            # Log the action
            for recipient in recipients:
                logger.info(f"Would send {report_path} to {recipient}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error sending report: {e}")
            return False


#######################################################
# Real-time Streaming Analytics
#######################################################

class StreamingAnalytics:
    """
    Real-time streaming analytics with WebSocket broadcasting.
    Processes data streams and calculates metrics in real-time.
    """
    
    def __init__(self):
        """Initialize the streaming analytics module."""
        self.streams = {}
        self.metrics = {}
        self.subscribers = {}
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.initialize_connections()
    
    def initialize_connections(self):
        """Initialize connections to streaming systems."""
        try:
            # Initialize Redis connection
            try:
                self.redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
                self.redis_client.ping()
                logger.info("Redis connection established successfully for streaming analytics")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
            
            # Initialize Kafka producer
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info("Kafka producer initialized successfully for streaming analytics")
            except Exception as e:
                logger.warning(f"Kafka producer initialization failed: {e}")
                self.kafka_producer = None
            
            # Initialize Kafka consumer (will be created when needed)
            self.kafka_consumer = None
            
        except Exception as e:
            logger.error(f"Error initializing streaming analytics connections: {e}")
    
    def create_stream(self, stream_id: str, source_type: str, 
                    source_config: Dict[str, Any], metrics_config: List[Dict[str, Any]]) -> bool:
        """
        Create a new data stream for real-time analytics.
        
        Args:
            stream_id: ID for the stream
            source_type: Type of data source ('kafka', 'websocket', 'api', etc.)
            source_config: Configuration for the data source
            metrics_config: Configuration for metrics to calculate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if stream already exists
            if stream_id in self.streams:
                logger.warning(f"Stream {stream_id} already exists")
                return False
            
            # Create stream configuration
            self.streams[stream_id] = {
                'source_type': source_type,
                'source_config': source_config,
                'metrics_config': metrics_config,
                'status': 'created',
                'created_at': datetime.datetime.now(),
                'last_active': None,
                'message_count': 0,
                'error_count': 0
            }
            
            # Initialize metrics for this stream
            self.metrics[stream_id] = {}
            for metric_config in metrics_config:
                metric_id = metric_config.get('id')
                if metric_id:
                    self.metrics[stream_id][metric_id] = {
                        'config': metric_config,
                        'current_value': None,
                        'history': deque(maxlen=100)  # Keep last 100 values
                    }
            
            # Initialize subscribers list
            self.subscribers[stream_id] = set()
            
            logger.info(f"Created stream {stream_id} of type {source_type}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating stream: {e}")
            return False
    
    def start_stream(self, stream_id: str) -> bool:
        """
        Start processing a data stream.
        
        Args:
            stream_id: ID of the stream to start
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if stream exists
            if stream_id not in self.streams:
                logger.error(f"Stream {stream_id} not found")
                return False
            
            # Get stream configuration
            stream_config = self.streams[stream_id]
            source_type = stream_config['source_type']
            source_config = stream_config['source_config']
            
            # Start stream based on source type
            if source_type == 'kafka':
                # Start Kafka consumer in a separate thread
                thread = threading.Thread(
                    target=self._process_kafka_stream,
                    args=(stream_id, source_config),
                    daemon=True
                )
                thread.start()
                
                # Update stream status
                stream_config['status'] = 'running'
                stream_config['thread'] = thread
                
                logger.info(f"Started Kafka stream {stream_id}")
                return True
            
            elif source_type == 'redis':
                # Start Redis subscriber in a separate thread
                thread = threading.Thread(
                    target=self._process_redis_stream,
                    args=(stream_id, source_config),
                    daemon=True
                )
                thread.start()
                
                # Update stream status
                stream_config['status'] = 'running'
                stream_config['thread'] = thread
                
                logger.info(f"Started Redis stream {stream_id}")
                return True
            
            elif source_type == 'simulation':
                # Start simulated stream in a separate thread
                thread = threading.Thread(
                    target=self._process_simulated_stream,
                    args=(stream_id, source_config),
                    daemon=True
                )
                thread.start()
                
                # Update stream status
                stream_config['status'] = 'running'
                stream_config['thread'] = thread
                
                logger.info(f"Started simulated stream {stream_id}")
                return True
            
            else:
                logger.error(f"Unsupported stream source type: {source_type}")
                return False
        
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            return False
    
    def _process_kafka_stream(self, stream_id: str, source_config: Dict[str, Any]):
        """
        Process a Kafka stream.
        
        Args:
            stream_id: ID of the stream
            source_config: Configuration for the Kafka source
            
        Returns:
            None
        """
        try:
            # Get Kafka configuration
            topic = source_config.get('topic')
            group_id = source_config.get('group_id', f'analytics_bi_{stream_id}')
            
            # Create Kafka consumer
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            # Process messages
            for message in consumer:
                try:
                    # Get message value
                    data = message.value
                    
                    # Process message
                    self._process_message(stream_id, data)
                    
                    # Update stream status
                    self.streams[stream_id]['message_count'] += 1
                    self.streams[stream_id]['last_active'] = datetime.datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error processing Kafka message: {e}")
                    self.streams[stream_id]['error_count'] += 1
                
                # Check if stream should stop
                if self.streams[stream_id]['status'] != 'running':
                    break
            
            # Close consumer
            consumer.close()
            
        except Exception as e:
            logger.error(f"Error processing Kafka stream: {e}")
            self.streams[stream_id]['status'] = 'error'
            self.streams[stream_id]['error'] = str(e)
    
    def _process_redis_stream(self, stream_id: str, source_config: Dict[str, Any]):
        """
        Process a Redis stream.
        
        Args:
            stream_id: ID of the stream
            source_config: Configuration for the Redis source
            
        Returns:
            None
        """
        try:
            # Check if Redis client is available
            if not self.redis_client:
                logger.error("Redis client not available")
                self.streams[stream_id]['status'] = 'error'
                self.streams[stream_id]['error'] = "Redis client not available"
                return
            
            # Get Redis configuration
            channel = source_config.get('channel')
            
            # Create Redis pubsub
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel)
            
            # Process messages
            for message in pubsub.listen():
                try:
                    # Skip subscription confirmation messages
                    if message['type'] != 'message':
                        continue
                    
                    # Get message data
                    data = json.loads(message['data'].decode('utf-8'))
                    
                    # Process message
                    self._process_message(stream_id, data)
                    
                    # Update stream status
                    self.streams[stream_id]['message_count'] += 1
                    self.streams[stream_id]['last_active'] = datetime.datetime.now()
                    
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")
                    self.streams[stream_id]['error_count'] += 1
                
                # Check if stream should stop
                if self.streams[stream_id]['status'] != 'running':
                    break
            
            # Close pubsub
            pubsub.unsubscribe()
            
        except Exception as e:
            logger.error(f"Error processing Redis stream: {e}")
            self.streams[stream_id]['status'] = 'error'
            self.streams[stream_id]['error'] = str(e)
    
    def _process_simulated_stream(self, stream_id: str, source_config: Dict[str, Any]):
        """
        Process a simulated stream for testing.
        
        Args:
            stream_id: ID of the stream
            source_config: Configuration for the simulated source
            
        Returns:
            None
        """
        try:
            # Get simulation configuration
            interval = source_config.get('interval', 1.0)  # seconds
            duration = source_config.get('duration')  # seconds, None for indefinite
            metrics = source_config.get('metrics', {})
            
            # Initialize simulation time
            start_time = time.time()
            
            # Generate simulated data
            while True:
                try:
                    # Generate timestamp
                    timestamp = datetime.datetime.now().isoformat()
                    
                    # Generate data for each metric
                    data = {
                        'timestamp': timestamp,
                        'metrics': {}
                    }
                    
                    for metric_id, metric_config in metrics.items():
                        base_value = metric_config.get('base_value', 100)
                        amplitude = metric_config.get('amplitude', 10)
                        
                        # Generate value with some randomness
                        value = base_value + amplitude * (np.sin(time.time() / 10) + 0.5 * np.random.randn())
                        
                        # Add to data
                        data['metrics'][metric_id] = value
                    
                    # Process message
                    self._process_message(stream_id, data)
                    
                    # Update stream status
                    self.streams[stream_id]['message_count'] += 1
                    self.streams[stream_id]['last_active'] = datetime.datetime.now()
                    
                    # Sleep for interval
                    time.sleep(interval)
                    
                    # Check if duration has elapsed
                    if duration and time.time() - start_time > duration:
                        break
                    
                except Exception as e:
                    logger.error(f"Error generating simulated data: {e}")
                    self.streams[stream_id]['error_count'] += 1
                
                # Check if stream should stop
                if self.streams[stream_id]['status'] != 'running':
                    break
            
            # Update stream status
            self.streams[stream_id]['status'] = 'completed'
            
        except Exception as e:
            logger.error(f"Error processing simulated stream: {e}")
            self.streams[stream_id]['status'] = 'error'
            self.streams[stream_id]['error'] = str(e)
    
    def _process_message(self, stream_id: str, data: Dict[str, Any]):
        """
        Process a message from a stream.
        
        Args:
            stream_id: ID of the stream
            data: Message data
            
        Returns:
            None
        """
        try:
            # Check if stream exists
            if stream_id not in self.streams:
                logger.error(f"Stream {stream_id} not found")
                return
            
            # Get metrics configuration
            metrics_config = self.streams[stream_id]['metrics_config']
            
            # Calculate metrics
            calculated_metrics = {}
            
            for metric_config in metrics_config:
                metric_id = metric_config.get('id')
                if not metric_id:
                    continue
                
                # Get metric type
                metric_type = metric_config.get('type', 'simple')
                
                # Calculate metric based on type
                if metric_type == 'simple':
                    # Simple metric directly from data
                    path = metric_config.get('path', f'metrics.{metric_id}')
                    value = self._get_value_from_path(data, path)
                    
                    if value is not None:
                        calculated_metrics[metric_id] = value
                
                elif metric_type == 'derived':
                    # Derived metric calculated from other metrics
                    formula = metric_config.get('formula')
                    if formula:
                        # Extract required values
                        values = {}
                        for var_name, var_path in metric_config.get('variables', {}).items():
                            values[var_name] = self._get_value_from_path(data, var_path)
                        
                        # Evaluate formula
                        try:
                            # Simple formula evaluation (not safe for production)
                            # In a real implementation, use a safer method
                            formula_str = formula
                            for var_name, var_value in values.items():
                                formula_str = formula_str.replace(var_name, str(var_value))
                            
                            value = eval(formula_str)
                            calculated_metrics[metric_id] = value
                        except Exception as e:
                            logger.error(f"Error evaluating formula for metric {metric_id}: {e}")
                
                elif metric_type == 'window':
                    # Window-based metric (e.g., moving average)
                    window_size = metric_config.get('window_size', 10)
                    path = metric_config.get('path', f'metrics.{metric_id}')
                    value = self._get_value_from_path(data, path)
                    
                    if value is not None:
                        # Get metric history
                        metric_data = self.metrics[stream_id].get(metric_id, {})
                        history = metric_data.get('history', deque(maxlen=window_size))
                        
                        # Add current value to history
                        history.append(value)
                        
                        # Calculate window metric
                        window_function = metric_config.get('window_function', 'mean')
                        if window_function == 'mean':
                            calculated_metrics[metric_id] = sum(history) / len(history)
                        elif window_function == 'sum':
                            calculated_metrics[metric_id] = sum(history)
                        elif window_function == 'min':
                            calculated_metrics[metric_id] = min(history)
                        elif window_function == 'max':
                            calculated_metrics[metric_id] = max(history)
                        
                        # Update history
                        if metric_id not in self.metrics[stream_id]:
                            self.metrics[stream_id][metric_id] = {'history': history}
                        else:
                            self.metrics[stream_id][metric_id]['history'] = history
            
            # Update metrics
            for metric_id, value in calculated_metrics.items():
                if metric_id in self.metrics[stream_id]:
                    self.metrics[stream_id][metric_id]['current_value'] = value
                    self.metrics[stream_id][metric_id]['last_updated'] = datetime.datetime.now()
                    
                    # Add to history if not already added
                    if 'history' not in self.metrics[stream_id][metric_id]:
                        self.metrics[stream_id][metric_id]['history'] = deque(maxlen=100)
                    
                    self.metrics[stream_id][metric_id]['history'].append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'value': value
                    })
            
            # Broadcast to subscribers
            self._broadcast_metrics(stream_id, calculated_metrics)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _get_value_from_path(self, data: Dict[str, Any], path: str) -> Any:
        """
        Get a value from a nested dictionary using a dot-separated path.
        
        Args:
            data: Dictionary to extract value from
            path: Dot-separated path to the value
            
        Returns:
            Value at the specified path, or None if not found
        """
        try:
            # Split path into parts
            parts = path.split('.')
            
            # Traverse the dictionary
            current = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            
            return current
        
        except Exception:
            return None
    
    def _broadcast_metrics(self, stream_id: str, metrics: Dict[str, Any]):
        """
        Broadcast metrics to subscribers.
        
        Args:
            stream_id: ID of the stream
            metrics: Dictionary of calculated metrics
            
        Returns:
            None
        """
        try:
            # Check if stream has subscribers
            if stream_id not in self.subscribers or not self.subscribers[stream_id]:
                return
            
            # Prepare message
            message = {
                'stream_id': stream_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'metrics': metrics
            }
            
            # Broadcast to all subscribers
            for websocket in list(self.subscribers[stream_id]):
                try:
                    # Send message asynchronously
                    asyncio.run_coroutine_threadsafe(
                        websocket.send_json(message),
                        asyncio.get_event_loop()
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting to subscriber: {e}")
                    # Remove failed subscriber
                    self.subscribers[stream_id].discard(websocket)
            
        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")
    
    def subscribe(self, stream_id: str, websocket: WebSocket) -> bool:
        """
        Subscribe a WebSocket to a stream.
        
        Args:
            stream_id: ID of the stream
            websocket: WebSocket connection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if stream exists
            if stream_id not in self.streams:
                logger.error(f"Stream {stream_id} not found")
                return False
            
            # Add to subscribers
            if stream_id not in self.subscribers:
                self.subscribers[stream_id] = set()
            
            self.subscribers[stream_id].add(websocket)
            
            logger.info(f"Added subscriber to stream {stream_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error subscribing to stream: {e}")
            return False
    
    def unsubscribe(self, stream_id: str, websocket: WebSocket) -> bool:
        """
        Unsubscribe a WebSocket from a stream.
        
        Args:
            stream_id: ID of the stream
            websocket: WebSocket connection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if stream exists
            if stream_id not in self.streams or stream_id not in self.subscribers:
                logger.error(f"Stream {stream_id} not found")
                return False
            
            # Remove from subscribers
            self.subscribers[stream_id].discard(websocket)
            
            logger.info(f"Removed subscriber from stream {stream_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error unsubscribing from stream: {e}")
            return False
    
    def stop_stream(self, stream_id: str) -> bool:
        """
        Stop processing a data stream.
        
        Args:
            stream_id: ID of the stream to stop
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if stream exists
            if stream_id not in self.streams:
                logger.error(f"Stream {stream_id} not found")
                return False
            
            # Update stream status
            self.streams[stream_id]['status'] = 'stopped'
            
            logger.info(f"Stopped stream {stream_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            return False
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """
        Get the status of a stream.
        
        Args:
            stream_id: ID of the stream
            
        Returns:
            Dictionary with stream status
        """
        try:
            # Check if stream exists
            if stream_id not in self.streams:
                logger.error(f"Stream {stream_id} not found")
                return {}
            
            # Get stream configuration
            stream_config = self.streams[stream_id]
            
            # Prepare status
            status = {
                'stream_id': stream_id,
                'source_type': stream_config['source_type'],
                'status': stream_config['status'],
                'created_at': stream_config['created_at'].isoformat(),
                'last_active': stream_config['last_active'].isoformat() if stream_config['last_active'] else None,
                'message_count': stream_config['message_count'],
                'error_count': stream_config['error_count'],
                'subscriber_count': len(self.subscribers.get(stream_id, set())),
                'metrics': {}
            }
            
            # Add metrics
            for metric_id, metric_data in self.metrics.get(stream_id, {}).items():
                status['metrics'][metric_id] = {
                    'current_value': metric_data.get('current_value'),
                    'last_updated': metric_data.get('last_updated', '').isoformat() if metric_data.get('last_updated') else None
                }
            
            return status
        
        except Exception as e:
            logger.error(f"Error getting stream status: {e}")
            return {}


#######################################################
# FastAPI Application
#######################################################

class AnalyticsAPI:
    """
    FastAPI application for the analytics and business intelligence system.
    Provides REST API endpoints and WebSocket connections.
    """
    
    def __init__(self):
        """Initialize the FastAPI application."""
        self.app = FastAPI(
            title="Skyscope Analytics & Business Intelligence API",
            description="API for advanced analytics, business intelligence, and real-time monitoring",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize components
        self.data_warehouse = DataWarehouse()
        self.olap_analysis = OLAPAnalysis()
        self.predictive_analytics = PredictiveAnalytics()
        self.executive_dashboard = ExecutiveDashboard()
        self.reporting_automation = ReportingAutomation()
        self.streaming_analytics = StreamingAnalytics()
        self.insight_generator = InsightGenerator()
        self.visualization = AdvancedVisualization()
        
        # Set up routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes."""
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.datetime.now().isoformat(),
                "version": "1.0.0",
                "components": {
                    "data_warehouse": self._check_component(self.data_warehouse),
                    "olap_analysis": self._check_component(self.olap_analysis),
                    "predictive_analytics": self._check_component(self.predictive_analytics),
                    "executive_dashboard": self._check_component(self.executive_dashboard),
                    "reporting_automation": self._check_component(self.reporting_automation),
                    "streaming_analytics": self._check_component(self.streaming_analytics),
                    "insight_generator": self._check_component(self.insight_generator)
                }
            }
        
        # Dashboard endpoints
        @self.app.get("/api/dashboards")
        async def get_dashboards():
            """Get all dashboards."""
            try:
                with Session() as session:
                    dashboards = session.query(Dashboard).all()
                    return [
                        {
                            "id": dashboard.id,
                            "name": dashboard.name,
                            "description": dashboard.description,
                            "owner_id": dashboard.owner_id,
                            "is_public": dashboard.is_public == 1,
                            "created_at": dashboard.created_at.isoformat(),
                            "updated_at": dashboard.updated_at.isoformat()
                        }
                        for dashboard in dashboards
                    ]
            except Exception as e:
                logger.error(f"Error getting dashboards: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/dashboards/{dashboard_id}")
        async def get_dashboard(dashboard_id: int):
            """Get a dashboard by ID."""
            try:
                dashboard = self.executive_dashboard.get_dashboard(dashboard_id)
                if not dashboard:
                    raise HTTPException(status_code=404, detail=f"Dashboard {dashboard_id} not found")
                return dashboard
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/dashboards")
        async def create_dashboard(dashboard: dict):
            """Create a new dashboard."""
            try:
                dashboard_id = self.executive_dashboard.create_dashboard(
                    name=dashboard.get("name"),
                    description=dashboard.get("description"),
                    layout=dashboard.get("layout"),
                    owner_id=dashboard.get("owner_id"),
                    is_public=dashboard.get("is_public", False)
                )
                
                if dashboard_id < 0:
                    raise HTTPException(status_code=500, detail="Failed to create dashboard")
                
                return {"id": dashboard_id, "message": "Dashboard created successfully"}
            except Exception as e:
                logger.error(f"Error creating dashboard: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/dashboards/{dashboard_id}/widgets")
        async def add_widget(dashboard_id: int, widget: dict):
            """Add a widget to a dashboard."""
            try:
                widget_id = self.executive_dashboard.add_widget(
                    dashboard_id=dashboard_id,
                    widget_type=widget.get("type"),
                    title=widget.get("title"),
                    config=widget.get("config"),
                    position_x=widget.get("position_x", 0),
                    position_y=widget.get("position_y", 0),
                    width=widget.get("width", 4),
                    height=widget.get("height", 4)
                )
                
                if widget_id < 0:
                    raise HTTPException(status_code=500, detail="Failed to add widget")
                
                return {"id": widget_id, "message": "Widget added successfully"}
            except Exception as e:
                logger.error(f"Error adding widget: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # KPI endpoints
        @self.app.get("/api/kpis")
        async def get_kpis(category: str = None):
            """Get all KPIs, optionally filtered by category."""
            try:
                kpis = self.executive_dashboard.get_kpis_by_category(category)
                return kpis
            except Exception as e:
                logger.error(f"Error getting KPIs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/kpis/{kpi_id}")
        async def get_kpi(kpi_id: int, with_history: bool = False):
            """Get a KPI by ID."""
            try:
                kpi = self.executive_dashboard.get_kpi(kpi_id, with_history)
                if not kpi:
                    raise HTTPException(status_code=404, detail=f"KPI {kpi_id} not found")
                return kpi
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting KPI: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/kpis")
        async def create_kpi(kpi: dict):
            """Create a new KPI."""
            try:
                kpi_id = self.executive_dashboard.create_kpi(
                    name=kpi.get("name"),
                    description=kpi.get("description"),
                    category=kpi.get("category"),
                    current_value=kpi.get("current_value", 0.0),
                    target_value=kpi.get("target_value"),
                    unit=kpi.get("unit")
                )
                
                if kpi_id < 0:
                    raise HTTPException(status_code=500, detail="Failed to create KPI")
                
                return {"id": kpi_id, "message": "KPI created successfully"}
            except Exception as e:
                logger.error(f"Error creating KPI: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/api/kpis/{kpi_id}")
        async def update_kpi(kpi_id: int, value: float):
            """Update a KPI value."""
            try:
                success = self.executive_dashboard.update_kpi(kpi_id, value)
                if not success:
                    raise HTTPException(status_code=404, detail=f"KPI {kpi_id} not found")
                return {"message": "KPI updated successfully"}
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating KPI: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/kpis/alerts")
        async def get_kpi_alerts(limit: int = 100):
            """Get KPI alerts."""
            try:
                alerts = self.executive_dashboard.get_alerts(limit)
                return alerts
            except Exception as e:
                logger.error(f"Error getting KPI alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Report endpoints
        @self.app.get("/api/reports")
        async def get_reports():
            """Get all reports."""
            try:
                with Session() as session:
                    reports = session.query(Report).all()
                    return [
                        {
                            "id": report.id,
                            "name": report.name,
                            "description": report.description,
                            "report_type": report.report_type,
                            "schedule": report.schedule,
                            "last_generated": report.last_generated.isoformat() if report.last_generated else None,
                            "created_at": report.created_at.isoformat()
                        }
                        for report in reports
                    ]
            except Exception as e:
                logger.error(f"Error getting reports: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/reports")
        async def create_report(report: dict):
            """Create a new report."""
            try:
                report_id = self.reporting_automation.create_report(
                    name=report.get("name"),
                    description=report.get("description"),
                    report_type=report.get("report_type", "pdf"),
                    schedule=report.get("schedule"),
                    recipients=report.get("recipients"),
                    query=report.get("query"),
                    template=report.get("template")
                )
                
                if report_id < 0:
                    raise HTTPException(status_code=500, detail="Failed to create report")
                
                return {"id": report_id, "message": "Report created successfully"}
            except Exception as e:
                logger.error(f"Error creating report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/reports/{report_id}/generate")
        async def generate_report(report_id: int):
            """Generate a report."""
            try:
                result = self.reporting_automation.generate_report(report_id)
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])
                return result
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error generating report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/reports/{report_id}/download")
        async def download_report(report_id: int):
            """Download a generated report."""
            try:
                with Session() as session:
                    report = session.query(Report).filter_by(id=report_id).first()
                    if not report:
                        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
                    
                    # Check if report was generated
                    if not report.last_generated:
                        raise HTTPException(status_code=400, detail="Report has not been generated yet")
                    
                    # Determine report path
                    report_type = report.report_type
                    report_name = report.name.replace(" ", "_")
                    timestamp = report.last_generated.strftime("%Y%m%d_%H%M%S")
                    
                    if report_type == "pdf":
                        report_path = REPORTS_PATH / "pdf" / f"{report_name}_{timestamp}.pdf"
                        media_type = "application/pdf"
                    elif report_type == "excel":
                        report_path = REPORTS_PATH / "excel" / f"{report_name}_{timestamp}.xlsx"
                        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    elif report_type == "web":
                        report_path = REPORTS_PATH / "web" / f"{report_name}_{timestamp}.html"
                        media_type = "text/html"
                    else:
                        raise HTTPException(status_code=400, detail=f"Unsupported report type: {report_type}")
                    
                    # Check if file exists
                    if not report_path.exists():
                        raise HTTPException(status_code=404, detail="Report file not found")
                    
                    # Return file
                    return FileResponse(
                        path=str(report_path),
                        filename=report_path.name,
                        media_type=media_type
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error downloading report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Forecast endpoints
        @self.app.post("/api/forecasts/prophet")
        async def create_prophet_forecast(request: dict):
            """Create a Prophet forecast."""
            try:
                # Get parameters
                data = pd.DataFrame(request.get("data", []))
                date_col = request.get("date_col")
                value_col = request.get("value_col")
                model_name = request.get("model_name")
                forecast_periods = request.get("forecast_periods", 30)
                seasonality_mode = request.get("seasonality_mode", "additive")
                
                # Create forecast
                result = self.predictive_analytics.train_prophet_model(
                    df=data,
                    date_col=date_col,
                    value_col=value_col,
                    model_name=model_name,
                    forecast_periods=forecast_periods,
                    seasonality_mode=seasonality_mode
                )
                
                if "error" in result:
                    raise HTTPException(status_code=500, detail=result["error"])
                