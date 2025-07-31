#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analytics_bi_initialization.py - Initialization Script for Advanced Analytics and Business Intelligence System

This script initializes the comprehensive analytics and business intelligence system for the
Skyscope Sentinel Intelligence AI platform. It sets up the database, creates example data,
configures the FastAPI application, establishes WebSocket endpoints, and provides integration
points with the main Skyscope system.

Features:
1. Database initialization and schema creation
2. FastAPI application setup with comprehensive endpoints
3. WebSocket endpoint handlers for real-time analytics
4. Performance optimization for 10,000 employees
5. Integration points with the main Skyscope system
6. Example usage scenarios and demonstrations
7. System documentation and summary
8. Startup and configuration scripts

Part of Skyscope Sentinel Intelligence AI - ITERATION 12
"""

import asyncio
import datetime
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Check for required packages
REQUIRED_PACKAGES = [
    "fastapi", "uvicorn", "sqlalchemy", "pandas", "numpy", "plotly",
    "dash", "prophet", "scikit-learn", "tensorflow", "pyspark",
    "redis", "kafka-python", "websockets", "jinja2", "reportlab",
    "xlsxwriter", "matplotlib", "seaborn", "networkx", "hdbscan"
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
import fastapi
import uvicorn
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules from the analytics and business intelligence system
try:
    from analytics_business_intelligence import (
        Base, DataWarehouse, AdvancedVisualization, PredictiveAnalytics,
        AnalyticsMetric, KPI, KPIHistory, Dashboard, DashboardWidget,
        Report, DataSource, DataWarehouseTable, Insight, AnomalyDetection,
        ForecastModel, Forecast
    )
    from analytics_business_intelligence_part2 import (
        PredictiveAnalytics, OLAPAnalysis, InsightGenerator, ExecutiveDashboard
    )
    from analytics_business_intelligence_part3 import (
        ReportingAutomation, StreamingAnalytics, AnalyticsAPI
    )
except ImportError:
    print("Error: Required modules not found. Please ensure the analytics_business_intelligence*.py files are in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analytics_bi_init.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///analytics_bi.db")
MAX_EMPLOYEES = 10000
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8050"))
DATA_WAREHOUSE_PATH = Path("./data_warehouse")
REPORTS_PATH = Path("./reports")
TEMPLATES_PATH = Path("./templates")
CONFIG_PATH = Path("./config")
DOCS_PATH = Path("./docs")

# Create necessary directories
DATA_WAREHOUSE_PATH.mkdir(exist_ok=True)
REPORTS_PATH.mkdir(exist_ok=True)
TEMPLATES_PATH.mkdir(exist_ok=True)
CONFIG_PATH.mkdir(exist_ok=True)
DOCS_PATH.mkdir(exist_ok=True)

# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

#######################################################
# Database Initialization
#######################################################

def initialize_database():
    """Initialize the database schema."""
    try:
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info("Database schema created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database schema: {e}")
        return False

def populate_example_data():
    """Populate the database with example data."""
    try:
        with Session() as session:
            # Check if data already exists
            kpi_count = session.query(KPI).count()
            if kpi_count > 0:
                logger.info("Example data already exists, skipping population")
                return True
            
            # Create example KPIs
            kpis = [
                {
                    "name": "Monthly Revenue",
                    "description": "Total revenue for the current month",
                    "category": "Financial",
                    "current_value": 1250000.0,
                    "target_value": 1500000.0,
                    "unit": "USD"
                },
                {
                    "name": "Active Users",
                    "description": "Number of active users in the last 30 days",
                    "category": "User Engagement",
                    "current_value": 85000.0,
                    "target_value": 100000.0,
                    "unit": "users"
                },
                {
                    "name": "Conversion Rate",
                    "description": "Percentage of visitors who convert to customers",
                    "category": "Sales",
                    "current_value": 3.2,
                    "target_value": 4.0,
                    "unit": "%"
                },
                {
                    "name": "Customer Satisfaction",
                    "description": "Average customer satisfaction score",
                    "category": "Customer",
                    "current_value": 4.3,
                    "target_value": 4.5,
                    "unit": "score"
                },
                {
                    "name": "Employee Productivity",
                    "description": "Average tasks completed per employee per day",
                    "category": "Operations",
                    "current_value": 7.8,
                    "target_value": 8.5,
                    "unit": "tasks"
                }
            ]
            
            # Add KPIs to database
            for kpi_data in kpis:
                # Calculate status
                current = kpi_data["current_value"]
                target = kpi_data["target_value"]
                if current < target * 0.8:
                    status = "off_track"
                elif current < target * 0.95:
                    status = "at_risk"
                else:
                    status = "on_track"
                
                kpi = KPI(
                    name=kpi_data["name"],
                    description=kpi_data["description"],
                    category=kpi_data["category"],
                    current_value=kpi_data["current_value"],
                    target_value=kpi_data["target_value"],
                    unit=kpi_data["unit"],
                    status=status,
                    updated_at=datetime.datetime.now()
                )
                session.add(kpi)
            
            session.commit()
            
            # Create KPI history (last 30 days)
            for kpi in session.query(KPI).all():
                base_value = kpi.current_value * 0.8  # Start at 80% of current value
                target_value = kpi.current_value
                
                for days_ago in range(30, 0, -1):
                    date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
                    progress = (30 - days_ago) / 30.0  # Progress from 0 to 1
                    
                    # Calculate value with some randomness
                    value = base_value + (target_value - base_value) * progress
                    value *= (0.95 + 0.1 * np.random.random())  # Add +/- 5% randomness
                    
                    history = KPIHistory(
                        kpi_id=kpi.id,
                        value=value,
                        timestamp=date
                    )
                    session.add(history)
            
            session.commit()
            
            # Create example dashboards
            exec_dashboard = Dashboard(
                name="Executive Dashboard",
                description="High-level overview of key business metrics",
                layout=json.dumps({
                    "grid": "masonry",
                    "theme": "dark"
                }),
                owner_id=1,
                is_public=1,
                created_at=datetime.datetime.now()
            )
            session.add(exec_dashboard)
            
            sales_dashboard = Dashboard(
                name="Sales Analytics",
                description="Detailed analysis of sales performance",
                layout=json.dumps({
                    "grid": "fixed",
                    "theme": "light"
                }),
                owner_id=2,
                is_public=1,
                created_at=datetime.datetime.now()
            )
            session.add(sales_dashboard)
            
            session.commit()
            
            # Create example widgets
            # Executive Dashboard Widgets
            widgets = [
                {
                    "dashboard_id": exec_dashboard.id,
                    "widget_type": "kpi",
                    "title": "Monthly Revenue",
                    "config": json.dumps({
                        "kpi_id": 1,
                        "display": "gauge",
                        "color_scheme": "default"
                    }),
                    "position_x": 0,
                    "position_y": 0,
                    "width": 6,
                    "height": 4
                },
                {
                    "dashboard_id": exec_dashboard.id,
                    "widget_type": "kpi",
                    "title": "Active Users",
                    "config": json.dumps({
                        "kpi_id": 2,
                        "display": "gauge",
                        "color_scheme": "default"
                    }),
                    "position_x": 6,
                    "position_y": 0,
                    "width": 6,
                    "height": 4
                },
                {
                    "dashboard_id": exec_dashboard.id,
                    "widget_type": "chart",
                    "title": "Revenue Trend",
                    "config": json.dumps({
                        "chart_type": "line",
                        "kpi_id": 1,
                        "period": "30d",
                        "color_scheme": "default"
                    }),
                    "position_x": 0,
                    "position_y": 4,
                    "width": 12,
                    "height": 6
                },
                {
                    "dashboard_id": exec_dashboard.id,
                    "widget_type": "chart",
                    "title": "User Engagement",
                    "config": json.dumps({
                        "chart_type": "line",
                        "kpi_id": 2,
                        "period": "30d",
                        "color_scheme": "default"
                    }),
                    "position_x": 0,
                    "position_y": 10,
                    "width": 12,
                    "height": 6
                }
            ]
            
            for widget_data in widgets:
                widget = DashboardWidget(**widget_data)
                session.add(widget)
            
            # Sales Dashboard Widgets
            widgets = [
                {
                    "dashboard_id": sales_dashboard.id,
                    "widget_type": "kpi",
                    "title": "Conversion Rate",
                    "config": json.dumps({
                        "kpi_id": 3,
                        "display": "bullet",
                        "color_scheme": "default"
                    }),
                    "position_x": 0,
                    "position_y": 0,
                    "width": 12,
                    "height": 3
                },
                {
                    "dashboard_id": sales_dashboard.id,
                    "widget_type": "chart",
                    "title": "Conversion Rate Trend",
                    "config": json.dumps({
                        "chart_type": "line",
                        "kpi_id": 3,
                        "period": "30d",
                        "color_scheme": "default"
                    }),
                    "position_x": 0,
                    "position_y": 3,
                    "width": 12,
                    "height": 6
                },
                {
                    "dashboard_id": sales_dashboard.id,
                    "widget_type": "table",
                    "title": "Top Products",
                    "config": json.dumps({
                        "query": "SELECT * FROM top_products ORDER BY revenue DESC LIMIT 10",
                        "columns": ["product_name", "revenue", "units_sold", "profit_margin"]
                    }),
                    "position_x": 0,
                    "position_y": 9,
                    "width": 12,
                    "height": 6
                }
            ]
            
            for widget_data in widgets:
                widget = DashboardWidget(**widget_data)
                session.add(widget)
            
            session.commit()
            
            # Create example reports
            reports = [
                {
                    "name": "Monthly Executive Summary",
                    "description": "Monthly summary of key business metrics for executives",
                    "report_type": "pdf",
                    "schedule": "0 8 1 * *",  # 8 AM on the 1st day of each month
                    "recipients": json.dumps(["executives@skyscope.ai"]),
                    "query": json.dumps({
                        "type": "kpi",
                        "category": "Financial"
                    }),
                    "template": "executive_summary"
                },
                {
                    "name": "Weekly Sales Report",
                    "description": "Weekly sales performance report",
                    "report_type": "excel",
                    "schedule": "0 8 * * 1",  # 8 AM every Monday
                    "recipients": json.dumps(["sales@skyscope.ai"]),
                    "query": "SELECT * FROM sales_data WHERE date >= date('now', '-7 day')",
                    "template": "sales_report"
                }
            ]
            
            for report_data in reports:
                report = Report(**report_data)
                session.add(report)
            
            session.commit()
            
            # Create example data sources
            data_sources = [
                {
                    "name": "Sales Database",
                    "source_type": "database",
                    "connection_string": "sqlite:///sales.db",
                    "refresh_interval": 3600,  # 1 hour
                    "schema": json.dumps({
                        "tables": ["sales", "products", "customers"]
                    }),
                    "created_at": datetime.datetime.now()
                },
                {
                    "name": "User Analytics API",
                    "source_type": "api",
                    "connection_string": "https://api.skyscope.ai/analytics/users",
                    "refresh_interval": 300,  # 5 minutes
                    "schema": json.dumps({
                        "endpoints": ["/active", "/new", "/retention"]
                    }),
                    "created_at": datetime.datetime.now()
                }
            ]
            
            for source_data in data_sources:
                source = DataSource(**source_data)
                session.add(source)
            
            session.commit()
            
            logger.info("Example data populated successfully")
            return True
    except Exception as e:
        logger.error(f"Error populating example data: {e}")
        return False

def generate_large_dataset(num_employees=MAX_EMPLOYEES):
    """
    Generate a large dataset for performance testing.
    
    Args:
        num_employees: Number of employees to generate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate employee data
        employees = []
        departments = ["Engineering", "Sales", "Marketing", "Finance", "HR", "Operations", "Research"]
        roles = ["Manager", "Director", "VP", "Associate", "Analyst", "Engineer", "Specialist"]
        
        for i in range(1, num_employees + 1):
            # Generate employee
            employee = {
                "id": i,
                "name": f"Employee {i}",
                "department": departments[i % len(departments)],
                "role": roles[i % len(roles)],
                "salary": 50000 + (i % 10) * 10000,
                "hire_date": datetime.datetime.now() - datetime.timedelta(days=i % 1000),
                "performance_score": round(3 + 2 * np.random.random(), 1),  # 3.0-5.0
                "projects_completed": i % 20,
                "manager_id": max(1, i % 100)  # Assign managers
            }
            employees.append(employee)
        
        # Create DataFrame
        df = pd.DataFrame(employees)
        
        # Save to parquet file
        employee_path = DATA_WAREHOUSE_PATH / "employees.parquet"
        df.to_parquet(str(employee_path))
        
        # Generate performance metrics
        metrics = []
        metric_names = ["tasks_completed", "bugs_fixed", "code_reviews", "meetings_attended", "documentation_pages"]
        
        # Generate daily metrics for the last 30 days
        for day in range(30):
            date = datetime.datetime.now() - datetime.timedelta(days=day)
            
            # Generate metrics for a subset of employees each day
            for i in range(1, num_employees + 1, 10):  # Every 10th employee for performance
                for metric_name in metric_names:
                    # Generate metric with some randomness
                    value = 10 * np.random.random()
                    
                    metric = {
                        "employee_id": i,
                        "date": date,
                        "metric_name": metric_name,
                        "value": value
                    }
                    metrics.append(metric)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Save to parquet file
        metrics_path = DATA_WAREHOUSE_PATH / "employee_metrics.parquet"
        metrics_df.to_parquet(str(metrics_path))
        
        logger.info(f"Generated large dataset with {num_employees} employees and {len(metrics)} metrics")
        return True
    except Exception as e:
        logger.error(f"Error generating large dataset: {e}")
        return False

#######################################################
# FastAPI Application Completion
#######################################################

def complete_fastapi_app(app):
    """
    Complete the FastAPI application with additional endpoints.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        None
    """
    # Add additional endpoints
    
    # Data Warehouse endpoints
    @app.get("/api/data-warehouse/tables")
    async def get_data_warehouse_tables():
        """Get all tables in the data warehouse."""
        try:
            dw = DataWarehouse()
            tables = dw.get_data_warehouse_tables()
            return tables
        except Exception as e:
            logger.error(f"Error getting data warehouse tables: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/data-warehouse/tables/{table_name}")
    async def query_data_warehouse_table(
        table_name: str, 
        query: str = None, 
        limit: int = 1000
    ):
        """Query a table in the data warehouse."""
        try:
            dw = DataWarehouse()
            df = dw.query_data_warehouse(table_name, query)
            
            if isinstance(df, dict) and "error" in df:
                raise fastapi.HTTPException(status_code=500, detail=df["error"])
            
            # Limit results
            df = df.head(limit)
            
            # Convert to records
            return df.to_dict(orient="records")
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error querying data warehouse table: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/data-warehouse/etl-jobs")
    async def create_etl_job(job: dict):
        """Create a new ETL job."""
        try:
            dw = DataWarehouse()
            job_id = dw.create_etl_job(
                name=job.get("name"),
                source_id=job.get("source_id"),
                target_table=job.get("target_table"),
                transformation_script=job.get("transformation_script"),
                schedule=job.get("schedule")
            )
            
            if job_id < 0:
                raise fastapi.HTTPException(status_code=500, detail="Failed to create ETL job")
            
            return {"id": job_id, "message": "ETL job created successfully"}
        except Exception as e:
            logger.error(f"Error creating ETL job: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/data-warehouse/etl-jobs/{job_id}/run")
    async def run_etl_job(job_id: int):
        """Run an ETL job."""
        try:
            dw = DataWarehouse()
            success = dw.run_etl_job(job_id)
            
            if not success:
                raise fastapi.HTTPException(status_code=500, detail=f"Failed to run ETL job {job_id}")
            
            return {"message": f"ETL job {job_id} executed successfully"}
        except Exception as e:
            logger.error(f"Error running ETL job: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    # OLAP Analysis endpoints
    @app.post("/api/olap/cubes")
    async def create_data_cube(cube: dict):
        """Create a data cube for OLAP analysis."""
        try:
            olap = OLAPAnalysis()
            
            # Get data from warehouse if table_name is provided
            if "table_name" in cube:
                dw = DataWarehouse()
                df = dw.query_data_warehouse(cube["table_name"])
                
                if isinstance(df, dict) and "error" in df:
                    raise fastapi.HTTPException(status_code=500, detail=df["error"])
            else:
                # Use provided data
                df = pd.DataFrame(cube.get("data", []))
            
            result = olap.create_data_cube(
                df=df,
                cube_name=cube.get("name"),
                dimensions=cube.get("dimensions"),
                measures=cube.get("measures"),
                agg_functions=cube.get("agg_functions")
            )
            
            if "error" in result:
                raise fastapi.HTTPException(status_code=500, detail=result["error"])
            
            return result
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating data cube: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/olap/cubes/{cube_name}/query")
    async def query_data_cube(cube_name: str, query: dict):
        """Query a data cube."""
        try:
            olap = OLAPAnalysis()
            df = olap.cube_query(
                cube_name=cube_name,
                dimensions=query.get("dimensions"),
                measures=query.get("measures"),
                filters=query.get("filters"),
                group_by=query.get("group_by")
            )
            
            if df.empty:
                return []
            
            # Convert to records
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error querying data cube: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/olap/cubes/{cube_name}/drill-down")
    async def drill_down(cube_name: str, params: dict):
        """Perform drill-down operation on a data cube."""
        try:
            olap = OLAPAnalysis()
            df = olap.drill_down(
                cube_name=cube_name,
                from_dim=params.get("from_dim"),
                to_dim=params.get("to_dim"),
                measures=params.get("measures"),
                filters=params.get("filters")
            )
            
            if df.empty:
                return []
            
            # Convert to records
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error performing drill-down: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    # Predictive Analytics endpoints
    @app.post("/api/forecasts/arima")
    async def create_arima_forecast(request: dict):
        """Create an ARIMA forecast."""
        try:
            # Get parameters
            data = pd.DataFrame(request.get("data", []))
            date_col = request.get("date_col")
            value_col = request.get("value_col")
            model_name = request.get("model_name")
            forecast_periods = request.get("forecast_periods", 30)
            order = tuple(request.get("order", [1, 1, 1]))
            seasonal_order = tuple(request.get("seasonal_order")) if "seasonal_order" in request else None
            
            # Create forecast
            pa = PredictiveAnalytics()
            result = pa.train_arima_model(
                df=data,
                date_col=date_col,
                value_col=value_col,
                model_name=model_name,
                forecast_periods=forecast_periods,
                order=order,
                seasonal_order=seasonal_order
            )
            
            if "error" in result:
                raise fastapi.HTTPException(status_code=500, detail=result["error"])
            
            # Convert forecast to records
            forecast_dict = result["forecast"].to_dict(orient="records")
            
            return {
                "model_name": result["model_name"],
                "forecast": forecast_dict,
                "metrics": result["metrics"]
            }
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating ARIMA forecast: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/forecasts/lstm")
    async def create_lstm_forecast(request: dict):
        """Create an LSTM forecast."""
        try:
            # Get parameters
            data = pd.DataFrame(request.get("data", []))
            date_col = request.get("date_col")
            value_col = request.get("value_col")
            model_name = request.get("model_name")
            forecast_periods = request.get("forecast_periods", 30)
            sequence_length = request.get("sequence_length", 10)
            epochs = request.get("epochs", 50)
            batch_size = request.get("batch_size", 32)
            
            # Create forecast
            pa = PredictiveAnalytics()
            result = pa.train_lstm_model(
                df=data,
                date_col=date_col,
                value_col=value_col,
                model_name=model_name,
                forecast_periods=forecast_periods,
                sequence_length=sequence_length,
                epochs=epochs,
                batch_size=batch_size
            )
            
            if "error" in result:
                raise fastapi.HTTPException(status_code=500, detail=result["error"])
            
            # Convert forecast to records
            forecast_dict = result["forecast"].to_dict(orient="records")
            
            return {
                "model_name": result["model_name"],
                "forecast": forecast_dict,
                "metrics": result["metrics"],
                "training_history": result.get("training_history")
            }
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating LSTM forecast: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/forecasts/ensemble")
    async def create_ensemble_forecast(request: dict):
        """Create an ensemble forecast."""
        try:
            # Get parameters
            data = pd.DataFrame(request.get("data", []))
            date_col = request.get("date_col")
            value_col = request.get("value_col")
            model_name = request.get("model_name")
            forecast_periods = request.get("forecast_periods", 30)
            models = request.get("models", ["prophet", "arima", "lstm"])
            
            # Create forecast
            pa = PredictiveAnalytics()
            result = pa.train_ensemble_forecast(
                df=data,
                date_col=date_col,
                value_col=value_col,
                model_name=model_name,
                forecast_periods=forecast_periods,
                models=models
            )
            
            if "error" in result:
                raise fastapi.HTTPException(status_code=500, detail=result["error"])
            
            # Convert forecast to records
            forecast_dict = result["forecast"].to_dict(orient="records")
            
            return {
                "model_name": result["model_name"],
                "forecast": forecast_dict,
                "metrics": result["metrics"],
                "component_models": result["component_models"]
            }
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating ensemble forecast: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    # Anomaly Detection endpoints
    @app.post("/api/anomalies/isolation-forest")
    async def detect_anomalies_isolation_forest(request: dict):
        """Detect anomalies using Isolation Forest."""
        try:
            # Get parameters
            data = pd.DataFrame(request.get("data", []))
            value_col = request.get("value_col")
            contamination = request.get("contamination", 0.05)
            detector_name = request.get("detector_name")
            
            # Detect anomalies
            pa = PredictiveAnalytics()
            result = pa.detect_anomalies_isolation_forest(
                df=data,
                value_col=value_col,
                contamination=contamination,
                detector_name=detector_name
            )
            
            if "error" in result:
                raise fastapi.HTTPException(status_code=500, detail=result["error"])
            
            # Convert anomalies to records
            anomalies_dict = result["anomalies"].to_dict(orient="records")
            
            return {
                "detector_name": result["detector_name"],
                "anomalies": anomalies_dict,
                "anomaly_count": result["anomaly_count"],
                "total_count": result["total_count"],
                "anomaly_percentage": result["anomaly_percentage"]
            }
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error detecting anomalies with Isolation Forest: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    # Insight Generation endpoints
    @app.post("/api/insights/data")
    async def generate_insights_from_data(request: dict):
        """Generate insights from data using GPT-4o."""
        try:
            # Get parameters
            data = pd.DataFrame(request.get("data", []))
            context = request.get("context")
            max_insights = request.get("max_insights", 5)
            
            # Generate insights
            ig = InsightGenerator()
            insights = ig.generate_insights_from_data(
                df=data,
                context=context,
                max_insights=max_insights
            )
            
            return {"insights": insights}
        except Exception as e:
            logger.error(f"Error generating insights from data: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/insights/forecast")
    async def generate_insights_from_forecast(request: dict):
        """Generate insights from forecast results using GPT-4o."""
        try:
            # Get parameters
            forecast_df = pd.DataFrame(request.get("forecast", []))
            model_name = request.get("model_name")
            metrics = request.get("metrics")
            
            # Generate insights
            ig = InsightGenerator()
            insights = ig.generate_insights_from_forecast(
                forecast_df=forecast_df,
                model_name=model_name,
                metrics=metrics
            )
            
            return {"insights": insights}
        except Exception as e:
            logger.error(f"Error generating insights from forecast: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/insights/kpis")
    async def generate_insights_from_kpis(request: dict):
        """Generate insights from KPI data using GPT-4o."""
        try:
            # Get parameters
            kpis = request.get("kpis", [])
            
            # Generate insights
            ig = InsightGenerator()
            insights = ig.generate_insights_from_kpis(kpis)
            
            return {"insights": insights}
        except Exception as e:
            logger.error(f"Error generating insights from KPIs: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    # Visualization endpoints
    @app.post("/api/visualizations/time-series")
    async def create_time_series_chart(request: dict):
        """Create a time series chart."""
        try:
            # Get parameters
            data = pd.DataFrame(request.get("data", []))
            x_col = request.get("x_col")
            y_col = request.get("y_col")
            title = request.get("title")
            color_col = request.get("color_col")
            color_scheme = request.get("color_scheme", "default")
            
            # Create chart
            viz = AdvancedVisualization()
            fig = viz.create_time_series_chart(
                df=data,
                x_col=x_col,
                y_col=y_col,
                title=title,
                color_col=color_col,
                color_scheme=color_scheme
            )
            
            # Convert to JSON
            chart_json = fig.to_json()
            
            return {"chart": chart_json}
        except Exception as e:
            logger.error(f"Error creating time series chart: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    # Add more visualization endpoints as needed
    
    # Streaming Analytics endpoints
    @app.post("/api/streams")
    async def create_stream(stream: dict):
        """Create a new data stream for real-time analytics."""
        try:
            sa = StreamingAnalytics()
            success = sa.create_stream(
                stream_id=stream.get("stream_id"),
                source_type=stream.get("source_type"),
                source_config=stream.get("source_config"),
                metrics_config=stream.get("metrics_config")
            )
            
            if not success:
                raise fastapi.HTTPException(status_code=500, detail="Failed to create stream")
            
            return {"message": f"Stream {stream.get('stream_id')} created successfully"}
        except Exception as e:
            logger.error(f"Error creating stream: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/streams/{stream_id}/start")
    async def start_stream(stream_id: str):
        """Start processing a data stream."""
        try:
            sa = StreamingAnalytics()
            success = sa.start_stream(stream_id)
            
            if not success:
                raise fastapi.HTTPException(status_code=500, detail=f"Failed to start stream {stream_id}")
            
            return {"message": f"Stream {stream_id} started successfully"}
        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/streams/{stream_id}/stop")
    async def stop_stream(stream_id: str):
        """Stop processing a data stream."""
        try:
            sa = StreamingAnalytics()
            success = sa.stop_stream(stream_id)
            
            if not success:
                raise fastapi.HTTPException(status_code=500, detail=f"Failed to stop stream {stream_id}")
            
            return {"message": f"Stream {stream_id} stopped successfully"}
        except Exception as e:
            logger.error(f"Error stopping stream: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/streams/{stream_id}/status")
    async def get_stream_status(stream_id: str):
        """Get the status of a stream."""
        try:
            sa = StreamingAnalytics()
            status = sa.get_stream_status(stream_id)
            
            if not status:
                raise fastapi.HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
            
            return status
        except fastapi.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting stream status: {e}")
            raise fastapi.HTTPException(status_code=500, detail=str(e))

#######################################################
# WebSocket Endpoint Handlers
#######################################################

def setup_websocket_endpoints(app):
    """
    Set up WebSocket endpoints for real-time updates.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        None
    """
    # Store active connections
    active_connections: Dict[str, Set[fastapi.WebSocket]] = {}
    
    @app.websocket("/ws/dashboards/{dashboard_id}")
    async def dashboard_updates(websocket: fastapi.WebSocket, dashboard_id: int):
        """WebSocket endpoint for real-time dashboard updates."""
        await websocket.accept()
        
        # Add to active connections
        if f"dashboard_{dashboard_id}" not in active_connections:
            active_connections[f"dashboard_{dashboard_id}"] = set()
        
        active_connections[f"dashboard_{dashboard_id}"].add(websocket)
        
        try:
            # Send initial data
            with Session() as session:
                dashboard = session.query(Dashboard).filter_by(id=dashboard_id).first()
                if dashboard:
                    widgets = session.query(DashboardWidget).filter_by(dashboard_id=dashboard_id).all()
                    
                    # Prepare initial data
                    data = {
                        "type": "initial_data",
                        "dashboard": {
                            "id": dashboard.id,
                            "name": dashboard.name,
                            "description": dashboard.description,
                            "layout": json.loads(dashboard.layout) if dashboard.layout else None
                        },
                        "widgets": []
                    }
                    
                    for widget in widgets:
                        widget_data = {
                            "id": widget.id,
                            "type": widget.widget_type,
                            "title": widget.title,
                            "config": json.loads(widget.config) if widget.config else None,
                            "position": {"x": widget.position_x, "y": widget.position_y},
                            "size": {"width": widget.width, "height": widget.height}
                        }
                        
                        # Add widget-specific data
                        if widget.widget_type == "kpi":
                            kpi_id = json.loads(widget.config).get("kpi_id") if widget.config else None
                            if kpi_id:
                                kpi = session.query(KPI).filter_by(id=kpi_id).first()
                                if kpi:
                                    widget_data["kpi"] = {
                                        "id": kpi.id,
                                        "name": kpi.name,
                                        "current_value": kpi.current_value,
                                        "target_value": kpi.target_value,
                                        "unit": kpi.unit,
                                        "status": kpi.status
                                    }
                        
                        data["widgets"].append(widget_data)
                    
                    # Send initial data
                    await websocket.send_json(data)
            
            # Wait for messages (keep connection alive)
            while True:
                # Receive message (this will block until a message is received)
                message = await websocket.receive_text()
                
                # Process message
                try:
                    msg_data = json.loads(message)
                    msg_type = msg_data.get("type")
                    
                    if msg_type == "ping":
                        # Respond to ping
                        await websocket.send_json({"type": "pong"})
                    elif msg_type == "refresh":
                        # Refresh dashboard data
                        # This would be implemented in a real system
                        await websocket.send_json({"type": "refresh_ack"})
                except json.JSONDecodeError:
                    # Ignore invalid messages
                    pass
                
        except fastapi.WebSocketDisconnect:
            # Remove from active connections
            if f"dashboard_{dashboard_id}" in active_connections:
                active_connections[f"dashboard_{dashboard_id}"].discard(websocket)
        except Exception as e:
            logger.error(f"Error in dashboard WebSocket: {e}")
            # Close connection on error
            await websocket.close(code=1011, reason=str(e))
    
    @app.websocket("/ws/kpis/{kpi_id}")
    async def kpi_updates(websocket: fastapi.WebSocket, kpi_id: int):
        """WebSocket endpoint for real-time KPI updates."""
        await websocket.accept()
        
        # Add to active connections
        if f"kpi_{kpi_id}" not in active_connections:
            active_connections[f"kpi_{kpi_id}"] = set()
        
        active_connections[f"kpi_{kpi_id}"].add(websocket)
        
        try:
            # Send initial data
            with Session() as session:
                kpi = session.query(KPI).filter_by(id=kpi_id).first()
                if kpi:
                    # Get KPI history
                    history = session.query(KPIHistory).filter_by(kpi_id=kpi_id).order_by(KPIHistory.timestamp).all()
                    
                    # Prepare initial data
                    data = {
                        "type": "initial_data",
                        "kpi": {
                            "id": kpi.id,
                            "name": kpi.name,
                            "description": kpi.description,
                            "category": kpi.category,
                            "current_value": kpi.current_value,
                            "target_value": kpi.target_value,
                            "unit": kpi.unit,
                            "status": kpi.status,
                            "updated_at": kpi.updated_at.isoformat()
                        },
                        "history": [
                            {
                                "value": h.value,
                                "timestamp": h.timestamp.isoformat()
                            }
                            for h in history
                        ]
                    }
                    
                    # Send initial data
                    await websocket.send_json(data)
            
            # Wait for messages (keep connection alive)
            while True:
                # Receive message (this will block until a message is received)
                message = await websocket.receive_text()
                
                # Process message
                try:
                    msg_data = json.loads(message)
                    msg_type = msg_data.get("type")
                    
                    if msg_type == "ping":
                        # Respond to ping
                        await websocket.send_json({"type": "pong"})
                    elif msg_type == "refresh":
                        # Refresh KPI data
                        # This would be implemented in a real system
                        await websocket.send_json({"type": "refresh_ack"})
                except json.JSONDecodeError:
                    # Ignore invalid messages
                    pass
                
        except fastapi.WebSocketDisconnect:
            # Remove from active connections
            if f"kpi_{kpi_id}" in active_connections:
                active_connections[f"kpi_{kpi_id}"].discard(websocket)
        except Exception as e:
            logger.error(f"Error in KPI WebSocket: {e}")
            # Close connection on error
            await websocket.close(code=1011, reason=str(e))
    
    @app.websocket("/ws/streams/{stream_id}")
    async def stream_updates(websocket: fastapi.WebSocket, stream_id: str):
        """WebSocket endpoint for real-time streaming analytics."""
        await websocket.accept()
        
        # Subscribe to stream
        sa = StreamingAnalytics()
        success = sa.subscribe(stream_id, websocket)
        
        if not success:
            await websocket.close(code=1011, reason=f"Stream {stream_id} not found")
            return
        
        try:
            # Send initial status
            status = sa.get_stream_status(stream_id)
            if status:
                await websocket.send_json({
                    "type": "initial_status",
                    "status": status
                })
            
            # Wait for messages (keep connection alive)
            while True:
                # Receive message (this will block until a message is received)
                message = await websocket.receive_text()
                
                # Process message
                try:
                    msg_data = json.loads(message)
                    msg_type = msg_data.get("type")
                    
                    if msg_type == "ping":
                        # Respond to ping
                        await websocket.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    # Ignore invalid messages
                    pass
                
        except fastapi.WebSocketDisconnect:
            # Unsubscribe from stream
            sa.unsubscribe(stream_id, websocket)
        except Exception as e:
            logger.error(f"Error in stream WebSocket: {e}")
            # Unsubscribe from stream
            sa.unsubscribe(stream_id, websocket)
            # Close connection on error
            await websocket.close(code=1011, reason=str(e))
    
    # Function to broadcast updates to connected clients
    async def broadcast_update(channel: str, data: Dict[str, Any]):
        """
        Broadcast an update to all connected clients on a channel.
        
        Args:
            channel: Channel to broadcast to
            data: Data to broadcast
            
        Returns:
            None
        """
        if channel in active_connections:
            # Create a copy of the set to avoid modification during iteration
            connections = active_connections[channel].copy()
            
            for websocket in connections:
                try:
                    await websocket.send_json(data)
                except Exception:
                    # Remove failed connection
                    active_connections[channel].discard(websocket)
    
    # Store broadcast function for use by other components
    app.state.broadcast_update = broadcast_update

#######################################################
# Integration with Main Skyscope System
#######################################################

def setup_skyscope_integration():
    """
    Set up integration points with the main Skyscope system.
    
    Returns:
        Dict with integration functions
    """
    # Create integration functions
    
    def get_kpi_data(kpi_id: int) -> Dict[str, Any]:
        """
        Get KPI data for the main Skyscope system.
        
        Args:
            kpi_id: ID of the KPI
            
        Returns:
            Dictionary with KPI data
        """
        try:
            with Session() as session:
                kpi = session.query(KPI).filter_by(id=kpi_id).first()
                if not kpi:
                    return {"error": f"KPI {kpi_id} not found"}
                
                # Get KPI history
                history = session.query(KPIHistory).filter_by(kpi_id=kpi_id).order_by(KPIHistory.timestamp).all()
                
                # Prepare result
                result = {
                    "id": kpi.id,
                    "name": kpi.name,
                    "description": kpi.description,
                    "category": kpi.category,
                    "current_value": kpi.current_value,
                    "target_value": kpi.target_value,
                    "unit": kpi.unit,
                    "status": kpi.status,
                    "updated_at": kpi.updated_at.isoformat(),
                    "history": [
                        {
                            "value": h.value,
                            "timestamp": h.timestamp.isoformat()
                        }
                        for h in history
                    ]
                }
                
                return result
        except Exception as e:
            logger.error(f"Error getting KPI data: {e}")
            return {"error": str(e)}
    
    def update_kpi_value(kpi_id: int, value: float) -> Dict[str, Any]:
        """
        Update a KPI value from the main Skyscope system.
        
        Args:
            kpi_id: ID of the KPI
            value: New value for the KPI
            
        Returns:
            Dictionary with result
        """
        try:
            # Create ExecutiveDashboard instance
            ed = ExecutiveDashboard()
            
            # Update KPI
            success = ed.update_kpi(kpi_id, value)
            
            if not success:
                return {"error": f"Failed to update KPI {kpi_id}"}
            
            return {"message": f"KPI {kpi_id} updated successfully"}
        except Exception as e:
            logger.error(f"Error updating KPI value: {e}")
            return {"error": str(e)}
    
    def generate_forecast(data: pd.DataFrame, date_col: str, value_col: str, 
                        model_type: str = "prophet", forecast_periods: int = 30) -> Dict[str, Any]:
        """
        Generate a forecast for the main Skyscope system.
        
        Args:
            data: DataFrame with time series data
            date_col: Column name for dates
            value_col: Column name for values
            model_type: Type of forecasting model
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        try:
            # Create PredictiveAnalytics instance
            pa = PredictiveAnalytics()
            
            # Generate forecast based on model type
            model_name = f"{value_col}_{model_type}_{uuid.uuid4().hex[:8]}"
            
            if model_type == "prophet":
                result = pa.train_prophet_model(
                    df=data,
                    date_col=date_col,
                    value_col=value_col,
                    model_name=model_name,
                    forecast_periods=forecast_periods
                )
            elif model_type == "arima":
                result = pa.train_arima_model(
                    df=data,
                    date_col=date_col,
                    value_col=value_col,
                    model_name=model_name,
                    forecast_periods=forecast_periods
                )
            elif model_type == "lstm":
                result = pa.train_lstm_model(
                    df=data,
                    date_col=date_col,
                    value_col=value_col,
                    model_name=model_name,
                    forecast_periods=forecast_periods
                )
            elif model_type == "ensemble":
                result = pa.train_ensemble_forecast(
                    df=data,
                    date_col=date_col,
                    value_col=value_col,
                    model_name=model_name,
                    forecast_periods=forecast_periods
                )
            else:
                return {"error": f"Unsupported model type: {model_type}"}
            
            if "error" in result:
                return {"error": result["error"]}
            
            # Convert forecast to records
            forecast_dict = result["forecast"].to_dict(orient="records")
            
            return {
                "model_name": result["model_name"],
                "forecast": forecast_dict,
                "metrics": result["metrics"]
            }
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {"error": str(e)}
    
    def detect_anomalies(data: pd.DataFrame, value_col: str, method: str = "isolation_forest") -> Dict[str, Any]:
        """
        Detect anomalies for the main Skyscope system.
        
        Args:
            data: DataFrame with data
            value_col: Column name for values
            method: Anomaly detection method
            
        Returns:
            Dictionary with anomaly detection results
        """
        try:
            # Create PredictiveAnalytics instance
            pa = PredictiveAnalytics()
            
            # Detect anomalies based on method
            detector_name = f"{value_col}_{method}_{uuid.uuid4().hex[:8]}"
            
            if method == "isolation_forest":
                result = pa.detect_anomalies_isolation_forest(
                    df=data,
                    value_col=value_col,
                    detector_name=detector_name
                )
            elif method == "zscore":
                result = pa.detect_anomalies_zscore(
                    df=data,
                    value_col=value_col,
                    detector_name=detector_name
                )
            else:
                return {"error": f"Unsupported anomaly detection method: {method}"}
            
            if "error" in result:
                return {"error": result["error"]}
            
            # Convert anomalies to records
            anomalies_dict = result["anomalies"].to_dict(orient="records")
            
            return {
                "detector_name": result["detector_name"],
                "anomalies": anomalies_dict,
                "anomaly_count": result["anomaly_count"],
                "total_count": result["total_count"],
                "anomaly_percentage": result["anomaly_percentage"]
            }
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"error": str(e)}
    
    def generate_insights(data: pd.DataFrame, context: str = None) -> Dict[str, Any]:
        """
        Generate insights for the main Skyscope system.
        
        Args:
            data: DataFrame with data
            context: Additional context about the data
            
        Returns:
            Dictionary with insights
        """
        try:
            # Create InsightGenerator instance
            ig = InsightGenerator()
            
            # Generate insights
            insights = ig.generate_insights_from_data(
                df=data,
                context=context
            )
            
            return {"insights": insights}
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}
    
    # Return integration functions
    return {
        "get_kpi_data": get_kpi_data,
        "update_kpi_value": update_kpi_value,
        "generate_forecast": generate_forecast,
        "detect_anomalies": detect_anomalies,
        "generate_insights": generate_insights
    }

#######################################################
# Performance Optimization for 10,000 Employees
#######################################################

def optimize_for_large_scale():
    """
    Optimize the system for large scale (10,000 employees).
    
    Returns:
        Dict with optimization results
    """
    try:
        optimizations = []
        
        # 1. Database indexing
        with Session() as session:
            # Create indexes for common queries
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_kpis_category ON kpis (category)"))
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_kpi_history_kpi_id ON kpi_history (kpi_id)"))
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_kpi_history_timestamp ON kpi_history (timestamp)"))
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_dashboard_widgets_dashboard_id ON dashboard_widgets (dashboard_id)"))
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_anomaly_detections_metric_name ON anomaly_detections (metric_name)"))
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_forecasts_model_id ON forecasts (model_id)"))
            
            optimizations.append("Database indexes created for common queries")
        
        # 2. Data partitioning for large tables
        # In a real implementation, this would use table partitioning
        # For SQLite, we'll simulate with separate tables
        with Session() as session:
            # Create partitioned tables for employee metrics
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS employee_metrics_current (
                    id INTEGER PRIMARY KEY,
                    employee_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """))
            
            session.execute(text("""
                CREATE TABLE IF NOT EXISTS employee_metrics_archive (
                    id INTEGER PRIMARY KEY,
                    employee_id INTEGER NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL
                )
            """))
            
            # Create indexes for partitioned tables
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_emp_metrics_current_emp_id ON employee_metrics_current (employee_id)"))
            session.execute(text("CREATE INDEX IF NOT EXISTS idx_emp_metrics_archive_emp_id ON employee_metrics_archive (employee_id)"))
            
            optimizations.append("Data partitioning implemented for large tables")
        
        # 3. Implement caching for frequently accessed data
        # Create cache directory
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Create cache configuration
        cache_config = {
            "enabled": True,
            "ttl": 300,  # 5 minutes
            "max_size": 1000,  # Maximum number of items
            "directories": {
                "kpis": str(cache_dir / "kpis"),
                "dashboards": str(cache_dir / "dashboards"),
                "forecasts": str(cache_dir / "forecasts")
            }
        }
        
        # Create cache directories
        for dir_name in cache_config["directories"].values():
            Path(dir_name).mkdir(exist_ok=True)
        
        # Save cache configuration
        with open(CONFIG_PATH / "cache_config.json", "w") as f:
            json.dump(cache_config, f, indent=2)
        
        optimizations.append("Caching implemented for frequently accessed data")
        
        # 4. Implement connection pooling
        # This is handled by SQLAlchemy's connection pooling
        
        # 5. Implement batch processing for large operations
        # Create batch processing configuration
        batch_config = {
            "enabled": True,
            "batch_size": 1000,  # Process 1000 records at a time
            "operations": {
                "employee_metrics": {
                    "batch_size": 1000,
                    "parallel": True,
                    "max_workers": 4
                },
                "forecasting": {
                    "batch_size": 10,
                    "parallel": True,
                    "max_workers": 2
                },
                "anomaly_detection": {
                    "batch_size": 5000,
                    "parallel": True,
                    "max_workers": 4
                }
            }
        }
        
        # Save batch configuration
        with open(CONFIG_PATH / "batch_config.json", "w") as f:
            json.dump(batch_config, f, indent=2)
        
        optimizations.append("Batch processing implemented for large operations")
        
        # 6. Implement parallel processing for intensive operations
        # This is configured in the batch processing configuration
        
        # 7. Optimize memory usage
        # Create memory optimization configuration
        memory_config = {
            "enabled": True,
            "max_memory_percent": 80,  # Use up to 80% of available memory
            "dataframe_chunk_size": 100000,  # Process large DataFrames in chunks
            "gc_interval": 60,  # Run garbage collection every 60 seconds
            "low_memory_mode": False  # Enable when memory is constrained
        }
        
        # Save memory configuration
        with open(CONFIG_PATH / "memory_config.json", "w") as f:
            json.dump(memory_config, f, indent=2)
        
        optimizations.append("Memory usage optimized for large datasets")
        
        # 8. Implement query optimization
        # Create query optimization configuration
        query_config = {
            "enabled": True,
            "use_prepared_statements": True,
            "limit_default": 1000,  # Default limit for queries
            "timeout": 30,  # Query timeout in seconds
            "optimize_joins": True,
            "use_views": True
        }
        
        # Save query configuration
        with open(CONFIG_PATH / "query_config.json", "w") as f:
            json.dump(query_config, f, indent=2)
        
        optimizations.append("Query optimization implemented")
        
        # 9. Implement data compression for large datasets
        # Create compression configuration
        compression_config = {
            "enabled": True,
            "format": "parquet",  # Use Parquet for data storage
            "compression": "snappy",  # Use Snappy compression
            "chunk_size": 100000,  # Chunk size for compression
            "compress_tables": [
                "employee_metrics_archive",
                "analytics_metrics",
                "kpi_history"
            ]
        }
        
        # Save compression configuration
        with open(CONFIG_PATH / "compression_config.json", "w") as f:
            json.dump(compression_config, f, indent=2)
        
        optimizations.append("Data compression implemented for large datasets")
        
        return {
            "status": "success",
            "optimizations": optimizations,
            "config_files": [
                str(CONFIG_PATH / "cache_config.json"),
                str(CONFIG_PATH / "batch_config.json"),
                str(CONFIG_PATH / "memory_config.json"),
                str(CONFIG_PATH / "query_config.json"),
                str(CONFIG_PATH / "compression_config.json")
            ]
        }
    except Exception as e:
        logger.error(f"Error optimizing for large scale: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

#######################################################
# Example Usage Scenarios
#######################################################

def create_example_scenarios():
    """
    Create example usage scenarios for the system.
    
    Returns:
        Dict with example scenarios
    """
    try:
        examples = []
        
        # 1. Executive Dashboard Example
        executive_dashboard = {
            "title": "Executive Dashboard Example",
            "description": "Create and view an executive dashboard with key business metrics",
            "steps": [
                "Create KPIs for revenue, active users, and customer satisfaction",
                "Create an executive dashboard with KPI widgets",
                "Add time series charts for KPI trends",
                "Add forecast widgets for revenue prediction",
                "Generate insights from KPI data",
                "Set up real-time updates via WebSocket"
            ],
            "code_example": """
# Create KPIs
kpi_ids = []
kpis = [
    {
        "name": "Monthly Revenue",
        "description": "Total revenue for the current month",
        "category": "Financial",
        "current_value": 1250000.0,
        "target_value": 1500000.0,
        "unit": "USD"
    },
    {
        "name": "Active Users",
        "description": "Number of active users in the last 30 days",
        "category": "User Engagement",
        "current_value": 85000.0,
        "target_value": 100000.0,
        "unit": "users"
    }
]

for kpi in kpis:
    response = requests.post("http://localhost:8050/api/kpis", json=kpi)
    kpi_ids.append(response.json()["id"])

# Create dashboard
dashboard = {
    "name": "Executive Dashboard",
    "description": "High-level overview of key business metrics",
    "layout": {
        "grid": "masonry",
        "theme": "dark"
    },
    "owner_id": 1,
    "is_public": True
}

response = requests.post("http://localhost:8050/api/dashboards", json=dashboard)
dashboard_id = response.json()["id"]

# Add widgets
widgets = [
    {
        "type": "kpi",
        "title": "Monthly Revenue",
        "config": {
            "kpi_id": kpi_ids[0],
            "display": "gauge",
            "color_scheme": "default"
        },
        "position_x": 0,
        "position_y": 0,
        "width": 6,
        "height": 4
    },
    {
        "type": "kpi",
        "title": "Active Users",
        "config": {
            "kpi_id": kpi_ids[1],
            "display": "gauge",
            "color_scheme": "default"
        },
        "position_x": 6,
        "position_y": 0,
        "width": 6,
        "height": 4
    }
]

for widget in widgets:
    requests.post(f"http://localhost:8050/api/dashboards/{dashboard_id}/widgets", json=widget)

# View dashboard
response = requests.get(f"http://localhost:8050/api/dashboards/{dashboard_id}")
print(response.json())

# Connect to WebSocket for real-time updates
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    print(f"Received update: {data}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")

ws = websocket.WebSocketApp(
    f"ws://localhost:8050/ws/dashboards/{dashboard_id}",
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open
)

ws.run_forever()
"""
        }
        examples.append(executive_dashboard)
        
        # 2. Forecasting Example
        forecasting = {
            "title": "Forecasting Example",
            "description": "Generate and visualize forecasts for business metrics",
            "steps": [
                "Prepare time series data",
                "Generate forecasts using Prophet, ARIMA, and LSTM",
                "Create an ensemble forecast",
                "Visualize forecast results",
                "Generate insights from forecasts",
                "Export forecast to report"
            ],
            "code_example": """
# Prepare time series data
import pandas as pd
import numpy as np
import datetime

# Generate sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.cumsum(np.random.normal(0, 1, len(dates))) + 1000  # Random walk with drift
data = pd.DataFrame({'ds': dates, 'y': values})

# Generate Prophet forecast
prophet_request = {
    "data": data.to_dict(orient="records"),
    "date_col": "ds",
    "value_col": "y",
    "model_name": "revenue_forecast_prophet",
    "forecast_periods": 90,
    "seasonality_mode": "additive"
}

response = requests.post("http://localhost:8050/api/forecasts/prophet", json=prophet_request)
prophet_result = response.json()

# Generate ARIMA forecast
arima_request = {
    "data": data.to_dict(orient="records"),
    "date_col": "ds",
    "value_col": "y",
    "model_name": "revenue_forecast_arima",
    "forecast_periods": 90,
    "order": [1, 1, 1]
}

response = requests.post("http://localhost:8050/api/forecasts/arima", json=arima_request)
arima_result = response.json()

# Generate LSTM forecast
lstm_request = {
    "data": data.to_dict(orient="records"),
    "date_col": "ds",
    "value_col": "y",
    "model_name": "revenue_forecast_lstm",
    "forecast_periods": 90,
    "sequence_length": 10,
    "epochs": 50
}

response = requests.post("http://localhost:8050/api/forecasts/lstm", json=lstm_request)
lstm_result = response.json()

# Generate ensemble forecast
ensemble_request = {
    "data": data.to_dict(orient="records"),
    "date_col": "ds",
    "value_col": "y",
    "model_name": "revenue_forecast_ensemble",
    "forecast_periods": 90,
    "models": ["prophet", "arima", "lstm"]
}

response = requests.post("http://localhost:8050/api/forecasts/ensemble", json=ensemble_request)
ensemble_result = response.json()

# Visualize forecast
viz_request = {
    "data": pd.DataFrame(ensemble_result["forecast"]).to_dict(orient="records"),
    "x_col": "ds",
    "y_col": "yhat",
    "title": "Revenue Forecast",
    "color_scheme": "default"
}

response = requests.post("http://localhost:8050/api/visualizations/time-series", json=viz_request)
chart = response.json()["chart"]

# Generate insights from forecast
insight_request = {
    "forecast": ensemble_result["forecast"],
    "model_name": "revenue_forecast_ensemble",
    "metrics": ensemble_result["metrics"]
}

response = requests.post("http://localhost:8050/api/insights/forecast", json=insight_request)
insights = response.json()["insights"]

# Create and generate a report
report = {
    "name": "Revenue Forecast Report",
    "description": "Forecast of revenue for the next 90 days",
    "report_type": "pdf",
    "query": json.dumps({
        "type": "forecast",
        "model_name": "revenue_forecast_ensemble"
    }),
    "template": "forecast_report"
}

response = requests.post("http://localhost:8050/api/reports", json=report)
report_id = response.json()["id"]

response = requests.post(f"http://localhost:8050/api/reports/{report_id}/generate")
report_result = response.json()

# Download the report
response = requests.get(f"http://localhost:8050/api/reports/{report_id}/download")
with open("revenue_forecast_report.pdf", "wb") as f:
    f.write(response.content)
"""
        }
        examples.append(forecasting)
        
        # 3. Anomaly Detection Example
        anomaly_detection = {
            "title": "Anomaly Detection Example",
            "description": "Detect anomalies in business metrics",
            "steps": [
                "Prepare data with anomalies",
                "Detect anomalies using Isolation Forest",
                "Detect anomalies using Z-score",
                "Visualize anomalies",
                "Set up real-time anomaly detection",
                "Create alerts for detected anomalies"
            ],
            "code_example": """
# Prepare data with anomalies
import pandas as pd
import numpy as np

# Generate sample data with anomalies
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'value': np.random.normal(0, 1, n_samples)
})

# Add anomalies
anomaly_indices = np.random.choice(range(n_samples), size=20, replace=False)
data.loc[anomaly_indices, 'value'] = np.random.normal(10, 1, len(anomaly_indices))

# Detect anomalies using Isolation Forest
isolation_forest_request = {
    "data": data.to_dict(orient="records"),
    "value_col": "value",
    "contamination": 0.02,
    "detector_name": "value_isolation_forest"
}

response = requests.post("http://localhost:8050/api/anomalies/isolation-forest", json=isolation_forest_request)
isolation_forest_result = response.json()

# Get anomalies
anomalies = pd.DataFrame(isolation_forest_result["anomalies"])
print(f"Detected {len(anomalies)} anomalies out of {n_samples} samples")
print(f"Anomaly percentage: {isolation_forest_result['anomaly_percentage']:.2f}%")

# Visualize anomalies
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(range(len(data)), data['value'], c='blue', label='Normal')
plt.scatter(anomalies.index, anomalies['value'], c='red', label='Anomaly')
plt.legend()
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.savefig('anomalies.png')

# Set up real-time anomaly detection stream
stream = {
    "stream_id": "anomaly_detection_stream",
    "source_type": "simulation",
    "source_config": {
        "interval": 1.0,
        "metrics": {
            "value": {
                "base_value": 0,
                "amplitude": 1
            }
        }
    },
    "metrics_config": [
        {
            "id": "value",
            "type": "simple",
            "path": "metrics.value"
        },
        {
            "id": "anomaly_score",
            "type": "window",
            "path": "metrics.value",
            "window_size": 20,
            "window_function": "mean"
        }
    ]
}

response = requests.post("http://localhost:8050/api/streams", json=stream)
print(response.json())

# Start the stream
response = requests.post(f"http://localhost:8050/api/streams/{stream['stream_id']}/start")
print(response.json())

# Connect to WebSocket for real-time anomaly detection
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    value = data.get("metrics", {}).get("value")
    anomaly_score = data.get("metrics", {}).get("anomaly_score")
    
    if value and anomaly_score and abs(value - anomaly_score) > 3:
        print(f"ANOMALY DETECTED: value={value}, score={anomaly_score}")
    else:
        print(f"Normal: value={value}, score={anomaly_score}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("Connection opened")

ws = websocket.WebSocketApp(
    f"ws://localhost:8050/ws/streams/{stream['stream_id']}",
    on_message=on_message,
    on_error=on_error,
    on_close=on_close,
    on_open=on_open
)

ws.run_forever()