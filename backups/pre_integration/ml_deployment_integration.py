
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_deployment_integration.py - Integration Script for ML Deployment and Management System

This script integrates all components of the ML Deployment and Management System:
1. Employee Hierarchy Management (10,000 employees)
2. ML Model Lifecycle Management
3. OpenAI GPT-4o Integration (including audio-preview & realtime-preview)
4. Kubernetes Deployment
5. Model Monitoring
6. Compliance Management
7. FastAPI Application

Part of Skyscope Sentinel Intelligence AI - ITERATION 11
"""

import asyncio
import base64
import datetime
import hashlib
import json
import logging
import os
import random
import re
import shutil
import signal
import string
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
    import uvicorn
    import websockets
    import yaml
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, WebSocket, WebSocketDisconnect, File, UploadFile, Request, status, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    from pydantic import BaseModel, Field, validator
    from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, DateTime, create_engine, func, text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker, Session as SQLAlchemySession
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    # Try to import Kubernetes client
    try:
        import kubernetes
        from kubernetes import client, config, watch
        KUBERNETES_AVAILABLE = True
    except ImportError:
        KUBERNETES_AVAILABLE = False
        warnings.warn("Kubernetes client not available. Kubernetes deployment features disabled.")
    
    # Import openai-unofficial if available, otherwise prepare for fallback
    try:
        import openai_unofficial
        from openai_unofficial import OpenAI
        from openai_unofficial.types.audio import Speech
        from openai_unofficial.types.chat import ChatCompletion
        OPENAI_UNOFFICIAL_AVAILABLE = True
    except ImportError:
        OPENAI_UNOFFICIAL_AVAILABLE = False
        warnings.warn("openai-unofficial package not found. Using standard OpenAI package with fallback to Ollama.")
        try:
            import openai
        except ImportError:
            warnings.warn("OpenAI package not found. Only Ollama will be available.")
    
    # Try to import Ollama for fallback
    try:
        import ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False
        if not OPENAI_UNOFFICIAL_AVAILABLE:
            warnings.warn("Neither openai-unofficial nor Ollama are available. Limited functionality.")

except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "aiohttp", "numpy", "pandas", "psutil", "pydantic", 
                          "requests", "torch", "websockets", "pyyaml", "fastapi", 
                          "uvicorn", "prometheus-client", "sqlalchemy", "python-multipart",
                          "python-jose[cryptography]", "passlib[bcrypt]", "jinja2"])
    print("Please restart the application.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_deployment_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import local modules with error handling
try:
    from ml_deployment_management import (
        Employee, Team, Project, MLModel, ModelDeployment, DeploymentMetric, AuditLog,
        EmployeeHierarchyManager, MLModelManager, OpenAIUnofficialManager,
        KubernetesDeploymentManager, Base, Session
    )
    from ml_deployment_management_part2 import ModelMonitoringManager
    from ml_deployment_management_part3 import ComplianceManager
    LOCAL_IMPORTS_AVAILABLE = True
except ImportError:
    logger.warning("Local modules not available. Creating standalone version.")
    LOCAL_IMPORTS_AVAILABLE = False
    
    # Create database models for standalone version
    Base = declarative_base()
    
    class Employee(Base):
        __tablename__ = 'employees'
        
        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)
        employee_id = Column(String, unique=True, nullable=False)
        role = Column(String, nullable=False)
        department = Column(String, nullable=False)
        division = Column(String, nullable=False)
        team_id = Column(Integer, ForeignKey('teams.id'))
        skills = Column(String)  # JSON serialized list of skills
        performance_score = Column(Float, default=0.0)
        created_at = Column(DateTime, default=func.now())
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        # Relationships
        team = relationship("Team", back_populates="employees")
        deployments = relationship("ModelDeployment", back_populates="owner")
        
        def __repr__(self):
            return f"<Employee(id={self.id}, name='{self.name}', role='{self.role}')>"
    
    
    class Team(Base):
        __tablename__ = 'teams'
        
        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)
        department = Column(String, nullable=False)
        division = Column(String, nullable=False)
        manager_id = Column(Integer, ForeignKey('employees.id'))
        created_at = Column(DateTime, default=func.now())
        
        # Relationships
        employees = relationship("Employee", back_populates="team", foreign_keys=[Employee.team_id])
        manager = relationship("Employee", foreign_keys=[manager_id])
        projects = relationship("Project", back_populates="team")
        
        def __repr__(self):
            return f"<Team(id={self.id}, name='{self.name}', department='{self.department}')>"
    
    
    class Project(Base):
        __tablename__ = 'projects'
        
        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)
        description = Column(String)
        team_id = Column(Integer, ForeignKey('teams.id'))
        status = Column(String, default="planning")  # planning, active, completed, cancelled
        created_at = Column(DateTime, default=func.now())
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        # Relationships
        team = relationship("Team", back_populates="projects")
        models = relationship("MLModel", back_populates="project")
        
        def __repr__(self):
            return f"<Project(id={self.id}, name='{self.name}', status='{self.status}')>"
    
    
    class MLModel(Base):
        __tablename__ = 'ml_models'
        
        id = Column(Integer, primary_key=True)
        name = Column(String, nullable=False)
        version = Column(String, nullable=False)
        project_id = Column(Integer, ForeignKey('projects.id'))
        model_type = Column(String, nullable=False)  # classification, regression, nlp, vision, etc.
        framework = Column(String)  # pytorch, tensorflow, sklearn, etc.
        metrics = Column(String)  # JSON serialized metrics
        path = Column(String)  # Path to model artifacts
        created_at = Column(DateTime, default=func.now())
        
        # Relationships
        project = relationship("Project", back_populates="models")
        deployments = relationship("ModelDeployment", back_populates="model")
        
        def __repr__(self):
            return f"<MLModel(id={self.id}, name='{self.name}', version='{self.version}')>"
    
    
    class ModelDeployment(Base):
        __tablename__ = 'model_deployments'
        
        id = Column(Integer, primary_key=True)
        model_id = Column(Integer, ForeignKey('ml_models.id'))
        owner_id = Column(Integer, ForeignKey('employees.id'))
        environment = Column(String, nullable=False)  # dev, staging, production
        status = Column(String, default="pending")  # pending, active, failed, terminated
        deployment_type = Column(String)  # kubernetes, edge, cloud
        endpoint = Column(String)
        created_at = Column(DateTime, default=func.now())
        updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
        
        # Relationships
        model = relationship("MLModel", back_populates="deployments")
        owner = relationship("Employee", back_populates="deployments")
        metrics = relationship("DeploymentMetric", back_populates="deployment")
        
        def __repr__(self):
            return f"<ModelDeployment(id={self.id}, environment='{self.environment}', status='{self.status}')>"
    
    
    class DeploymentMetric(Base):
        __tablename__ = 'deployment_metrics'
        
        id = Column(Integer, primary_key=True)
        deployment_id = Column(Integer, ForeignKey('model_deployments.id'))
        metric_name = Column(String, nullable=False)
        metric_value = Column(Float, nullable=False)
        timestamp = Column(DateTime, default=func.now())
        
        # Relationships
        deployment = relationship("ModelDeployment", back_populates="metrics")
        
        def __repr__(self):
            return f"<DeploymentMetric(id={self.id}, name='{self.metric_name}', value={self.metric_value})>"
    
    
    class AuditLog(Base):
        __tablename__ = 'audit_logs'
        
        id = Column(Integer, primary_key=True)
        action = Column(String, nullable=False)
        entity_type = Column(String, nullable=False)
        entity_id = Column(Integer, nullable=False)
        user_id = Column(Integer, ForeignKey('employees.id'))
        details = Column(String)
        timestamp = Column(DateTime, default=func.now())
        
        def __repr__(self):
            return f"<AuditLog(id={self.id}, action='{self.action}', entity_type='{self.entity_type}')>"
    
    # Create engine and session
    engine = create_engine('sqlite:///ml_deployment.db')
    Session = sessionmaker(bind=engine)
    
    # Import manager classes from the modules
    from ml_deployment_management import EmployeeHierarchyManager, MLModelManager, OpenAIUnofficialManager, KubernetesDeploymentManager
    from ml_deployment_management_part2 import ModelMonitoringManager
    from ml_deployment_management_part3 import ComplianceManager

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

# Create FastAPI app
app = FastAPI(
    title="Skyscope ML Deployment and Management API",
    description="API for managing ML models, employees, and deployments with 10,000 employee structure",
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

# Create static files directory
static_dir = Path("./static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create templates
templates = Jinja2Templates(directory="templates")

# Global managers
employee_manager = None
model_manager = None
openai_manager = None
kubernetes_manager = None
monitoring_manager = None
compliance_manager = None

# Authentication setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/token")
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


#######################################################
# Authentication Functions
#######################################################

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
        return user_dict


def authenticate_user(fake_db, username: str, password: str):
    """Authenticate user."""
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
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
        token_data = {"username": username, "role": payload.get("role"), "employee_id": payload.get("employee_id")}
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data["username"])
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Get current active user."""
    if current_user.get("disabled", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


#######################################################
# Helper Functions
#######################################################

def generate_employee_data(count: int = MAX_EMPLOYEES) -> List[Dict[str, Any]]:
    """Generate data for employees."""
    roles = [
        "Data Scientist", "ML Engineer", "Data Engineer", "DevOps Engineer", 
        "Research Scientist", "Software Engineer", "Product Manager", "Project Manager",
        "UI/UX Designer", "Business Analyst", "Technical Writer", "QA Engineer"
    ]
    
    departments = [
        "Research", "Engineering", "Product", "Operations", "Marketing", 
        "Sales", "Finance", "HR", "Legal", "Customer Support"
    ]
    
    divisions = [
        "AI Research", "Platform", "Applications", "Infrastructure", "Enterprise",
        "Consumer", "Healthcare", "Finance", "Retail", "Manufacturing"
    ]
    
    skills_pool = [
        "python", "pytorch", "tensorflow", "sklearn", "nlp", "computer_vision",
        "reinforcement_learning", "data_engineering", "mlops", "devops",
        "kubernetes", "docker", "aws", "azure", "gcp", "database_management",
        "data_visualization", "statistics", "deep_learning", "quantum_ml",
        "federated_learning", "edge_deployment", "model_optimization",
        "feature_engineering", "time_series", "recommendation_systems",
        "anomaly_detection", "natural_language_generation", "speech_recognition",
        "image_generation", "model_compression", "automl", "neural_architecture_search",
        "explainable_ai", "ethical_ai", "privacy_preserving_ml", "graph_neural_networks",
        "knowledge_graphs", "transformers", "gans", "vae", "diffusion_models"
    ]
    
    employees = []
    for i in range(1, count + 1):
        # Determine if this is a manager (approximately 10% of employees)
        is_manager = random.random() < 0.1
        role = "Manager, " + random.choice(roles) if is_manager else random.choice(roles)
        
        # Generate random skills (3-8 skills per employee)
        num_skills = random.randint(3, 8)
        skills = random.sample(skills_pool, num_skills)
        
        # Generate employee data
        employee = {
            "name": f"Employee {i}",
            "employee_id": f"EMP-{i:06d}",
            "role": role,
            "department": random.choice(departments),
            "division": random.choice(divisions),
            "skills": skills,
            "performance_score": round(random.uniform(0.5, 1.0), 2)
        }
        
        employees.append(employee)
    
    return employees


def generate_team_data(count: int = 100) -> List[Dict[str, Any]]:
    """Generate data for teams."""
    departments = [
        "Research", "Engineering", "Product", "Operations", "Marketing", 
        "Sales", "Finance", "HR", "Legal", "Customer Support"
    ]
    
    divisions = [
        "AI Research", "Platform", "Applications", "Infrastructure", "Enterprise",
        "Consumer", "Healthcare", "Finance", "Retail", "Manufacturing"
    ]
    
    team_names = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
        "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi", "Rho",
        "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"
    ]
    
    teams = []
    for i in range(1, count + 1):
        department = random.choice(departments)
        division = random.choice(divisions)
        team_name = f"Team {random.choice(team_names)} {i}"
        
        team = {
            "name": team_name,
            "department": department,
            "division": division
        }
        
        teams.append(team)
    
    return teams


def generate_project_data(count: int = 50, team_ids: List[int] = None) -> List[Dict[str, Any]]:
    """Generate data for projects."""
    if not team_ids:
        team_ids = list(range(1, 101))  # Default to teams 1-100
    
    project_prefixes = [
        "Project", "Initiative", "Program", "Venture", "Endeavor",
        "Mission", "Operation", "Task", "Assignment", "Undertaking"
    ]
    
    project_names = [
        "Phoenix", "Horizon", "Nexus", "Quantum", "Zenith", "Apex", "Vortex",
        "Catalyst", "Fusion", "Pinnacle", "Summit", "Odyssey", "Voyager",
        "Discovery", "Explorer", "Pioneer", "Trailblazer", "Pathfinder",
        "Navigator", "Compass", "Beacon", "Lighthouse", "Sentinel", "Guardian"
    ]
    
    statuses = ["planning", "active", "completed", "cancelled"]
    weights = [0.3, 0.5, 0.15, 0.05]  # Probability weights for statuses
    
    projects = []
    for i in range(1, count + 1):
        prefix = random.choice(project_prefixes)
        name = random.choice(project_names)
        team_id = random.choice(team_ids)
        status = random.choices(statuses, weights=weights)[0]
        
        project = {
            "name": f"{prefix} {name} {i}",
            "description": f"This is a description for {prefix} {name} {i}.",
            "team_id": team_id,
            "status": status
        }
        
        projects.append(project)
    
    return projects


def generate_model_data(count: int = 20, project_ids: List[int] = None) -> List[Dict[str, Any]]:
    """Generate data for ML models."""
    if not project_ids:
        project_ids = list(range(1, 51))  # Default to projects 1-50
    
    model_types = [
        "classification", "regression", "nlp", "vision", "time_series",
        "recommendation", "anomaly_detection", "reinforcement_learning",
        "generative", "clustering"
    ]
    
    frameworks = [
        "pytorch", "tensorflow", "sklearn", "xgboost", "lightgbm",
        "keras", "huggingface", "jax", "mxnet", "fastai"
    ]
    
    models = []
    for i in range(1, count + 1):
        model_type = random.choice(model_types)
        framework = random.choice(frameworks)
        project_id = random.choice(project_ids)
        version = f"0.{random.randint(1, 9)}.{random.randint(0, 9)}"
        
        # Generate random metrics
        metrics = {
            "accuracy": round(random.uniform(0.7, 0.99), 4),
            "f1_score": round(random.uniform(0.7, 0.99), 4),
            "precision": round(random.uniform(0.7, 0.99), 4),
            "recall": round(random.uniform(0.7, 0.99), 4),
            "training_time": round(random.uniform(10, 1000), 2)
        }
        
        model = {
            "name": f"{model_type}_{framework}_model_{i}",
            "version": version,
            "project_id": project_id,
            "model_type": model_type,
            "framework": framework,
            "metrics": metrics,
            "path": f"./model_registry/models/{model_type}_{framework}_model_{i}_{version}"
        }
        
        models.append(model)
    
    return models


def generate_deployment_data(count: int = 10, model_ids: List[int] = None, owner_ids: List[int] = None) -> List[Dict[str, Any]]:
    """Generate data for model deployments."""
    if not model_ids:
        model_ids = list(range(1, 21))  # Default to models 1-20
    
    if not owner_ids:
        owner_ids = list(range(1, 101))  # Default to employees 1-100
    
    environments = ["dev", "staging", "production"]
    weights = [0.3, 0.3, 0.4]  # Probability weights for environments
    
    deployment_types = ["kubernetes", "edge", "cloud"]
    
    statuses = ["pending", "active", "failed", "terminated"]
    status_weights = [0.1, 0.7, 0.1, 0.1]  # Probability weights for statuses
    
    deployments = []
    for i in range(1, count + 1):
        model_id = random.choice(model_ids)
        owner_id = random.choice(owner_ids)
        environment = random.choices(environments, weights=weights)[0]
        deployment_type = random.choice(deployment_types)
        status = random.choices(statuses, weights=status_weights)[0]
        
        # Generate random endpoint
        if deployment_type == "kubernetes":
            endpoint = f"http://model-{model_id}.{KUBERNETES_NAMESPACE}.svc.cluster.local"
        elif deployment_type == "edge":
            endpoint = f"http://edge-device-{random.randint(1, 100)}:8000"
        else:  # cloud
            endpoint = f"https://api.skyscope.ai/models/{model_id}"
        
        deployment = {
            "model_id": model_id,
            "owner_id": owner_id,
            "environment": environment,
            "status": status,
            "deployment_type": deployment_type,
            "endpoint": endpoint
        }
        
        deployments.append(deployment)
    
    return deployments


def generate_metric_data(count: int = 100, deployment_ids: List[int] = None) -> List[Dict[str, Any]]:
    """Generate data for deployment metrics."""
    if not deployment_ids:
        deployment_ids = list(range(1, 11))  # Default to deployments 1-10
    
    metric_names = [
        "accuracy", "f1_score", "precision", "recall", "latency",
        "throughput", "error_rate", "cpu_usage", "memory_usage", "gpu_usage"
    ]
    
    metrics = []
    for i in range(1, count + 1):
        deployment_id = random.choice(deployment_ids)
        metric_name = random.choice(metric_names)
        
        # Generate appropriate metric value based on name
        if metric_name in ["accuracy", "f1_score", "precision", "recall"]:
            metric_value = round(random.uniform(0.7, 0.99), 4)
        elif metric_name == "latency":
            metric_value = round(random.uniform(10, 500), 2)  # ms
        elif metric_name == "throughput":
            metric_value = round(random.uniform(10, 1000), 2)  # requests/sec
        elif metric_name == "error_rate":
            metric_value = round(random.uniform(0.001, 0.05), 4)
        elif metric_name in ["cpu_usage", "memory_usage", "gpu_usage"]:
            metric_value = round(random.uniform(10, 90), 2)  # percentage
        else:
            metric_value = round(random.uniform(0, 100), 2)
        
        metric = {
            "deployment_id": deployment_id,
            "metric_name": metric_name,
            "metric_value": metric_value
        }
        
        metrics.append(metric)
    
    return metrics


#######################################################
# Initialization Functions
#######################################################

def initialize_database():
    """Initialize the database and create tables."""
    try:
        # Create tables
        Base.metadata.create_all(engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False


def initialize_employee_hierarchy(session: SQLAlchemySession, count: int = MAX_EMPLOYEES):
    """Initialize the employee hierarchy with employees, teams, etc."""
    try:
        # Check if employees already exist
        employee_count = session.query(func.count(Employee.id)).scalar()
        if employee_count > 0:
            logger.info(f"Employee hierarchy already initialized with {employee_count} employees")
            return True
        
        logger.info(f"Initializing employee hierarchy with {count} employees...")
        
        # Generate and create teams
        teams_data = generate_team_data(count=100)
        teams = []
        for team_data in teams_data:
            team = Team(
                name=team_data["name"],
                department=team_data["department"],
                division=team_data["division"]
            )
            session.add(team)
            teams.append(team)
        
        session.commit()
        logger.info(f"Created {len(teams)} teams")
        
        # Generate and create employees
        employees_data = generate_employee_data(count=count)
        employees = []
        for employee_data in employees_data:
            # Assign to a random team
            team_id = random.choice(teams).id if teams else None
            
            employee = Employee(
                name=employee_data["name"],
                employee_id=employee_data["employee_id"],
                role=employee_data["role"],
                department=employee_data["department"],
                division=employee_data["division"],
                team_id=team_id,
                skills=json.dumps(employee_data["skills"]),
                performance_score=employee_data["performance_score"]
            )
            session.add(employee)
            employees.append(employee)
            
            # Commit in batches to avoid memory issues
            if len(employees) % 1000 == 0:
                session.commit()
                logger.info(f"Created {len(employees)} employees so far...")
        
        session.commit()
        logger.info(f"Created {len(employees)} employees")
        
        # Assign managers to teams (select from employees with "Manager" in role)
        managers = [e for e in employees if "Manager" in e.role]
        for team in teams:
            if managers:
                manager = random.choice(managers)
                team.manager_id = manager.id
        
        session.commit()
        logger.info(f"Assigned managers to teams")
        
        # Generate and create projects
        team_ids = [team.id for team in teams]
        projects_data = generate_project_data(count=50, team_ids=team_ids)
        projects = []
        for project_data in projects_data:
            project = Project(
                name=project_data["name"],
                description=project_data["description"],
                team_id=project_data["team_id"],
                status=project_data["status"]
            )
            session.add(project)
            projects.append(project)
        
        session.commit()
        logger.info(f"Created {len(projects)} projects")
        
        # Generate and create ML models
        project_ids = [project.id for project in projects]
        models_data = generate_model_data(count=20, project_ids=project_ids)
        models = []
        for model_data in models_data:
            model = MLModel(
                name=model_data["name"],
                version=model_data["version"],
                project_id=model_data["project_id"],
                model_type=model_data["model_type"],
                framework=model_data["framework"],
                metrics=json.dumps(model_data["metrics"]),
                path=model_data["path"]
            )
            session.add(model)
            models.append(model)
        
        session.commit()
        logger.info(f"Created {len(models)} ML models")
        
        # Generate and create model deployments
        model_ids = [model.id for model in models]
        employee_ids = [employee.id for employee in employees]
        deployments_data = generate_deployment_data(count=10, model_ids=model_ids, owner_ids=employee_ids)
        deployments = []
        for deployment_data in deployments_data:
            deployment = ModelDeployment(
                model_id=deployment_data["model_id"],
                owner_id=deployment_data["owner_id"],
                environment=deployment_data["environment"],
                status=deployment_data["status"],
                deployment_type=deployment_data["deployment_type"],
                endpoint=deployment_data["endpoint"]
            )
            session.add(deployment)
            deployments.append(deployment)
        
        session.commit()
        logger.info(f"Created {len(deployments)} model deployments")
        
        # Generate and create deployment metrics
        deployment_ids = [deployment.id for deployment in deployments]
        metrics_data = generate_metric_data(count=100, deployment_ids=deployment_ids)
        metrics = []
        for metric_data in metrics_data:
            metric = DeploymentMetric(
                deployment_id=metric_data["deployment_id"],
                metric_name=metric_data["metric_name"],
                metric_value=metric_data["metric_value"]
            )
            session.add(metric)
            metrics.append(metric)
        
        session.commit()
        logger.info(f"Created {len(metrics)} deployment metrics")
        
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error initializing employee hierarchy: {e}")
        return False


def initialize_managers(session: SQLAlchemySession):
    """Initialize all manager instances."""
    global employee_manager, model_manager, openai_manager, kubernetes_manager, monitoring_manager, compliance_manager
    
    try:
        # Initialize managers
        employee_manager = EmployeeHierarchyManager(session)
        model_manager = MLModelManager(session)
        openai_manager = OpenAIUnofficialManager()
        
        if KUBERNETES_AVAILABLE:
            kubernetes_manager = KubernetesDeploymentManager(KUBERNETES_NAMESPACE)
        else:
            kubernetes_manager = None
            logger.warning("Kubernetes manager not initialized due to missing dependencies")
        
        monitoring_manager = ModelMonitoringManager(session)
        compliance_manager = ComplianceManager(session)
        
        logger.info("All managers initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing managers: {e}")
        return False


def setup_monitoring_for_models(session: SQLAlchemySession):
    """Set up monitoring for all models."""
    try:
        if not monitoring_manager:
            logger.warning("Monitoring manager not initialized")
            return False
        
        # Get all models
        models = session.query(MLModel).all()
        
        for model in models:
            # Register model for monitoring
            monitoring_manager.register_model_for_monitoring(
                model_id=model.id,
                performance_metrics=["accuracy", "latency", "throughput", "error_rate"],
                drift_detection_features=None  # No reference data available in this example
            )
        
        logger.info(f"Set up monitoring for {len(models)} models")
        return True
    except Exception as e:
        logger.error(f"Error setting up monitoring for models: {e}")
        return False


def setup_compliance_for_models(session: SQLAlchemySession):
    """Set up compliance for all models."""
    try:
        if not compliance_manager:
            logger.warning("Compliance manager not initialized")
            return False
        
        # Get all models
        models = session.query(MLModel).all()
        
        for model in models:
            # Register model for compliance
            compliance_manager.register_model_for_compliance(model_id=model.id)
        
        logger.info(f"Set up compliance for {len(models)} models")
        return True
    except Exception as e:
        logger.error(f"Error setting up compliance for models: {e}")
        return False


def create_model_registry_structure():
    """Create the model registry directory structure."""
    try:
        for subdir in ["models", "metadata", "experiments", "deployments", "federated", "ab_tests"]:
            path = MODEL_REGISTRY_PATH / subdir
            path.mkdir(exist_ok=True)
        
        logger.info("Model registry structure created")
        return True
    except Exception as e:
        logger.error(f"Error creating model registry structure: {e}")
        return False


def initialize_system():
    """Initialize the entire system."""
    try:
        # Create a session
        session = Session()
        
        # Initialize database
        if not initialize_database():
            logger.error("Failed to initialize database")
            return False
        
        # Create model registry structure
        if not create_model_registry_structure():
            logger.error("Failed to create model registry structure")
            return False
        
        # Initialize employee hierarchy
        if not initialize_employee_hierarchy(session):
            logger.error("Failed to initialize employee hierarchy")
            return False
        
        # Initialize managers
        if not initialize_managers(session):
            logger.error("Failed to initialize managers")
            return False
        
        # Set up monitoring for models
        if not setup_monitoring_for_models(session):
            logger.warning("Failed to set up monitoring for models")
        
        # Set up compliance for models
        if not setup_compliance_for_models(session):
            logger.warning("Failed to set up compliance for models")
        
        # Start Prometheus metrics server
        try:
            start_http_server(8000)
            logger.info("Prometheus metrics server started on port 8000")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus metrics server: {e}")
        
        logger.info("System initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return False
    finally:
        session.close()


#######################################################
# FastAPI Endpoints
#######################################################

# Authentication endpoints
@app.post("/api/token", response_model=dict)
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
        data={"sub": user["username"], "role": user["role"], "employee_id": user["employee_id"]},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/users/me", response_model=dict)
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Get current user info."""
    return {
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
        "employee_id": current_user["employee_id"]
    }


# Employee endpoints
@app.get("/api/employees", response_model=List[dict])
async def get_employees(
    skip: int = 0, 
    limit: int = 100,
    department: Optional[str] = None,
    division: Optional[str] = None,
    role: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """Get employees with optional filtering."""
    try:
        with Session() as session:
            query = session.query(Employee)
            
            if department:
                query = query.filter(Employee.department == department)
            
            if division:
                query = query.filter(Employee.division == division)
            
            if role:
                query = query.filter(Employee.role.like(f"%{role}%"))
            
            total = query.count()
            employees = query.offset(skip).limit(limit).all()
            
            result = []
            for employee in employees:
                try:
                    skills = json.loads(employee.skills) if employee.skills else []
                except:
                    skills = []
                
                result.append({
                    "id": employee.id,
                    "name": employee.name,
                    "employee_id": employee.employee_id,
                    "role": employee.role,
                    "department": employee.department,
                    "division": employee.division,
                    "team_id": employee.team_id,
                    "skills": skills,
                    "performance_score": employee.performance_score
                })
            
            return result
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/employees/{employee_id}", response_model=dict)
async def get_employee(
    employee_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Get employee by ID."""
    try:
        with Session() as session:
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                raise HTTPException(status_code=404, detail="Employee not found")
            
            try:
                skills = json.loads(employee.skills) if employee.skills else []
            except:
                skills = []
            
            return {
                "id": employee.id,
                "name": employee.name,
                "employee_id": employee.employee_id,
                "role": employee.role,
                "department": employee.department,
                "division": employee.division,
                "team_id": employee.team_id,
                "skills": skills,
                "performance_score": employee.performance_score
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/employees", status_code=status.HTTP_201_CREATED)
async def create_employee(
    employee: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new employee."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create employees")
    
    try:
        with Session() as session:
            # Check if employee_id already exists
            existing = session.query(Employee).filter(Employee.employee_id == employee["employee_id"]).first()
            if existing:
                raise HTTPException(status_code=400, detail="Employee ID already exists")
            
            # Create employee
            new_employee = Employee(
                name=employee["name"],
                employee_id=employee["employee_id"],
                role=employee["role"],
                department=employee["department"],
                division=employee["division"],
                team_id=employee.get("team_id"),
                skills=json.dumps(employee.get("skills", [])),
                performance_score=employee.get("performance_score", 0.0)
            )
            
            session.add(new_employee)
            session.commit()
            
            return {
                "id": new_employee.id,
                "name": new_employee.name,
                "employee_id": new_employee.employee_id,
                "message": "Employee created successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/employees/{employee_id}")
async def update_employee(
    employee_id: int,
    employee: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Update an employee."""
    if current_user["role"] != "admin" and current_user["employee_id"] != employee_id:
        raise HTTPException(status_code=403, detail="Not authorized to update this employee")
    
    try:
        with Session() as session:
            db_employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not db_employee:
                raise HTTPException(status_code=404, detail="Employee not found")
            
            # Update fields
            if "name" in employee:
                db_employee.name = employee["name"]
            if "role" in employee:
                db_employee.role = employee["role"]
            if "department" in employee:
                db_employee.department = employee["department"]
            if "division" in employee:
                db_employee.division = employee["division"]
            if "team_id" in employee:
                db_employee.team_id = employee["team_id"]
            if "skills" in employee:
                db_employee.skills = json.dumps(employee["skills"])
            if "performance_score" in employee:
                db_employee.performance_score = employee["performance_score"]
            
            session.commit()
            
            return {
                "id": db_employee.id,
                "message": "Employee updated successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/employees/{employee_id}")
async def delete_employee(
    employee_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Delete an employee."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete employees")
    
    try:
        with Session() as session:
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                raise HTTPException(status_code=404, detail="Employee not found")
            
            session.delete(employee)
            session.commit()
            
            return {"message": f"Employee {employee_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting employee: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Team endpoints
@app.get("/api/teams", response_model=List[dict])
async def get_teams(
    skip: int = 0, 
    limit: int = 100,
    department: Optional[str] = None,
    division: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """Get teams with optional filtering."""
    try:
        with Session() as session:
            query = session.query(Team)
            
            if department:
                query = query.filter(Team.department == department)
            
            if division:
                query = query.filter(Team.division == division)
            
            total = query.count()
            teams = query.offset(skip).limit(limit).all()
            
            result = []
            for team in teams:
                result.append({
                    "id": team.id,
                    "name": team.name,
                    "department": team.department,
                    "division": team.division,
                    "manager_id": team.manager_id
                })
            
            return result
    except Exception as e:
        logger.error(f"Error getting teams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/teams/{team_id}", response_model=dict)
async def get_team(
    team_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Get team by ID."""
    try:
        with Session() as session:
            team = session.query(Team).filter(Team.id == team_id).first()
            if not team:
                raise HTTPException(status_code=404, detail="Team not found")
            
            # Get team members
            employees = session.query(Employee).filter(Employee.team_id == team_id).all()
            employee_list = []
            for employee in employees:
                try:
                    skills = json.loads(employee.skills) if employee.skills else []
                except:
                    skills = []
                
                employee_list.append({
                    "id": employee.id,
                    "name": employee.name,
                    "employee_id": employee.employee_id,
                    "role": employee.role,
                    "skills": skills,
                    "performance_score": employee.performance_score
                })
            
            # Get manager
            manager = None
            if team.manager_id:
                manager_obj = session.query(Employee).filter(Employee.id == team.manager_id).first()
                if manager_obj:
                    manager = {
                        "id": manager_obj.id,
                        "name": manager_obj.name,
                        "employee_id": manager_obj.employee_id,
                        "role": manager_obj.role
                    }
            
            return {
                "id": team.id,
                "name": team.name,
                "department": team.department,
                "division": team.division,
                "manager": manager,
                "employees": employee_list,
                "employee_count": len(employee_list)
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/teams", status_code=status.HTTP_201_CREATED)
async def create_team(
    team: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new team."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to create teams")
    
    try:
        with Session() as session:
            # Create team
            new_team = Team(
                name=team["name"],
                department=team["department"],
                division=team["division"],
                manager_id=team.get("manager_id")
            )
            
            session.add(new_team)
            session.commit()
            
            return {
                "id": new_team.id,
                "name": new_team.name,
                "message": "Team created successfully"
            }
    except Exception as e:
        logger.error(f"Error creating team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/teams/{team_id}")
async def update_team(
    team_id: int,
    team: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Update a team."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to update teams")
    
    try:
        with Session() as session:
            db_team = session.query(Team).filter(Team.id == team_id).first()
            if not db_team:
                raise HTTPException(status_code=404, detail="Team not found")
            
            # Update fields
            if "name" in team:
                db_team.name = team["name"]
            if "department" in team:
                db_team.department = team["department"]
            if "division" in team:
                db_team.division = team["division"]
            if "manager_id" in team:
                db_team.manager_id = team["manager_id"]
            
            session.commit()
            
            return {
                "id": db_team.id,
                "message": "Team updated successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/teams/{team_id}/manager/{employee_id}")
async def assign_manager_to_team(
    team_id: int,
    employee_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Assign a manager to a team."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to assign managers")
    
    try:
        with Session() as session:
            # Check if team exists
            team = session.query(Team).filter(Team.id == team_id).first()
            if not team:
                raise HTTPException(status_code=404, detail="Team not found")
            
            # Check if employee exists
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                raise HTTPException(status_code=404, detail="Employee not found")
            
            # Update employee role if not already a manager
            if "manager" not in employee.role.lower():
                employee.role = f"Manager, {employee.role}"
            
            # Assign manager to team
            team.manager_id = employee_id
            session.commit()
            
            return {
                "message": f"Employee {employee_id} assigned as manager of team {team_id}"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning manager to team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/employees/{employee_id}/team/{team_id}")
async def assign_employee_to_team(
    employee_id: int,
    team_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Assign an employee to a team."""
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to assign employees")
    
    try:
        with Session() as session:
            # Check if employee exists
            employee = session.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                raise HTTPException(status_code=404, detail="Employee not found")
            
            # Check if team exists
            team = session.query(Team).filter(Team.id == team_id).first()
            if not team:
                raise HTTPException(status_code=404, detail="Team not found")
            
            # Assign employee to team
            employee.team_id = team_id
            session.commit()
            
            return {
                "message": f"Employee {employee_id} assigned to team {team_id}"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning employee to team: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Project endpoints
@app.get("/api/projects", response_model=List[dict])
async def get_projects(
    skip: int = 0, 
    limit: int = 100,
    status: Optional[str] = None,
    team_id: Optional[int] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """Get projects with optional filtering."""
    try:
        with Session() as session:
            query = session.query(Project)
            
            if status:
                query = query.filter(Project.status == status)
            
            if team_id:
                query = query.filter(Project.team_id == team_id)
            
            total = query.count()
            projects = query.offset(skip).limit(limit).all()
            
            result = []
            for project in projects:
                result.append({
                    "id": project.id,
                    "name": project.name,
                    "description": project.description,
                    "team_id": project.team_id,
                    "status": project.status
                })
            
            return result
    except Exception as e:
        logger.error(f"Error getting projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/projects/{project_id}", response_model=dict)
async def get_project(
    project_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Get project by ID."""
    try:
        with Session() as session:
            project = session.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # Get team
            team = None
            if project.team_id:
                team_obj = session.query(Team).filter(Team.id == project.team_id).first()
                if team_obj:
                    team = {
                        "id": team_obj.id,
                        "name": team_obj.name,
                        "department": team_obj.department,
                        "division": team_obj.division
                    }
            
            # Get models
            models = session.query(MLModel).filter(MLModel.project_id == project_id).all()
            model_list = []
            for model in models:
                try:
                    metrics = json.loads(model.metrics) if model.metrics else {}
                except:
                    metrics = {}
                
                model_list.append({
                    "id": model.id,
                    "name": model.name,
                    "version": model.version,
                    "model_type": model.model_type,
                    "framework": model.framework,
                    "metrics": metrics
                })
            
            return {
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "team": team,
                "status": project.status,
                "models": model_list,
                "model_count": len(model_list),
                "created_at": project.created_at.isoformat() if project.created_at else None,
                "updated_at": project.updated_at.isoformat() if project.updated_at else None
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/projects", status_code=status.HTTP_201_CREATED)
async def create_project(
    project: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new project."""
    try:
        with Session() as session:
            # Check if team exists
            if "team_id" in project:
                team = session.query(Team).filter(Team.id == project["team_id"]).first()
                if not team:
                    raise HTTPException(status_code=404, detail="Team not found")
            
            # Create project
            new_project = Project(
                name=project["name"],
                description=project.get("description"),
                team_id=project.get("team_id"),
                status=project.get("status", "planning")
            )
            
            session.add(new_project)
            session.commit()
            
            return {
                "id": new_project.id,
                "name": new_project.name,
                "message": "Project created successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/projects/{project_id}")
async def update_project(
    project_id: int,
    project: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Update a project."""
    try:
        with Session() as session:
            db_project = session.query(Project).filter(Project.id == project_id).first()
            if not db_project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # Update fields
            if "name" in project:
                db_project.name = project["name"]
            if "description" in project:
                db_project.description = project["description"]
            if "team_id" in project:
                # Check if team exists
                team = session.query(Team).filter(Team.id == project["team_id"]).first()
                if not team:
                    raise HTTPException(status_code=404, detail="Team not found")
                db_project.team_id = project["team_id"]
            if "status" in project:
                db_project.status = project["status"]
            
            session.commit()
            
            return {
                "id": db_project.id,
                "message": "Project updated successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ML Model endpoints
@app.get("/api/models", response_model=List[dict])
async def get_models(
    skip: int = 0, 
    limit: int = 100,
    model_type: Optional[str] = None,
    framework: Optional[str] = None,
    project_id: Optional[int] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """Get ML models with optional filtering."""
    try:
        with Session() as session:
            query = session.query(MLModel)
            
            if model_type:
                query = query.filter(MLModel.model_type == model_type)
            
            if framework:
                query = query.filter(MLModel.framework == framework)
            
            if project_id:
                query = query.filter(MLModel.project_id == project_id)
            
            total = query.count()
            models = query.offset(skip).limit(limit).all()
            
            result = []
            for model in models:
                try:
                    metrics = json.loads(model.metrics) if model.metrics else {}
                except:
                    metrics = {}
                
                result.append({
                    "id": model.id,
                    "name": model.name,
                    "version": model.version,
                    "project_id": model.project_id,
                    "model_type": model.model_type,
                    "framework": model.framework,
                    "metrics": metrics,
                    "path": model.path
                })
            
            return result
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/{model_id}", response_model=dict)
async def get_model(
    model_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Get ML model by ID."""
    try:
        with Session() as session:
            model = session.query(MLModel).filter(MLModel.id == model_id).first()
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            try:
                metrics = json.loads(model.metrics) if model.metrics else {}
            except:
                metrics = {}
            
            # Get project
            project = None
            if model.project_id:
                project_obj = session.query(Project).filter(Project.id == model.project_id).first()
                if project_obj:
                    project = {
                        "id": project_obj.id,
                        "name": project_obj.name,
                        "status": project_obj.status
                    }
            
            # Get deployments
            deployments = session.query(ModelDeployment).filter(ModelDeployment.model_id == model_id).all()
            deployment_list = []
            for deployment in deployments:
                deployment_list.append({
                    "id": deployment.id,
                    "environment": deployment.environment,
                    "status": deployment.status,
                    "deployment_type": deployment.deployment_type,
                    "endpoint": deployment.endpoint,
                    "created_at": deployment.created_at.isoformat() if deployment.created_at else None
                })
            
            return {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "project": project,
                "model_type": model.model_type,
                "framework": model.framework,
                "metrics": metrics,
                "path": model.path,
                "deployments": deployment_list,
                "deployment_count": len(deployment_list),
                "created_at": model.created_at.isoformat() if model.created_at else None
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models", status_code=status.HTTP_201_CREATED)
async def create_model(
    model: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new ML model."""
    try:
        with Session() as session:
            # Check if project exists
            if "project_id" in model:
                project = session.query(Project).filter(Project.id == model["project_id"]).first()
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")
            
            # Create model
            new_model = MLModel(
                name=model["name"],
                version=model["version"],
                project_id=model.get("project_id"),
                model_type=model["model_type"],
                framework=model.get("framework"),
                metrics=json.dumps(model.get("metrics", {})),
                path=model.get("path")
            )
            
            session.add(new_model)
            session.commit()
            
            # Create model directory in registry if path not provided
            if not model.get("path"):
                model_dir = MODEL_REGISTRY_PATH / "models" / f"{new_model.id}_{new_model.name}_{new_model.version}"
                model_dir.mkdir(exist_ok=True)
                
                # Update model path
                new_model.path = str(model_dir)
                session.commit()
            
            return {
                "id": new_model.id,
                "name": new_model.name,
                "version": new_model.version,
                "message": "Model created successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/models/{model_id}")
async def update_model(
    model_id: int,
    model: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Update an ML model."""
    try:
        with Session() as session:
            db_model = session.query(MLModel).filter(MLModel.id == model_id).first()
            if not db_model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Update fields
            if "name" in model:
                db_model.name = model["name"]
            if "version" in model:
                db_model.version = model["version"]
            if "project_id" in model:
                # Check if project exists
                project = session.query(Project).filter(Project.id == model["project_id"]).first()
                if not project:
                    raise HTTPException(status_code=404, detail="Project not found")
                db_model.project_id = model["project_id"]
            if "model_type" in model:
                db_model.model_type = model["model_type"]
            if "framework" in model:
                db_model.framework = model["framework"]
            if "metrics" in model:
                db_model.metrics = json.dumps(model["metrics"])
            if "path" in model:
                db_model.path = model["path"]
            
            session.commit()
            
            return {
                "id": db_model.id,
                "message": "Model updated successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Deployment endpoints
@app.get("/api/deployments", response_model=List[dict])
async def get_deployments(
    skip: int = 0, 
    limit: int = 100,
    environment: Optional[str] = None,
    status: Optional[str] = None,
    deployment_type: Optional[str] = None,
    model_id: Optional[int] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """Get model deployments with optional filtering."""
    try:
        with Session() as session:
            query = session.query(ModelDeployment)
            
            if environment:
                query = query.filter(ModelDeployment.environment == environment)
            
            if status:
                query = query.filter(ModelDeployment.status == status)
            
            if deployment_type:
                query = query.filter(ModelDeployment.deployment_type == deployment_type)
            
            if model_id:
                query = query.filter(ModelDeployment.model_id == model_id)
            
            total = query.count()
            deployments = query.offset(skip).limit(limit).all()
            
            result = []
            for deployment in deployments:
                result.append({
                    "id": deployment.id,
                    "model_id": deployment.model_id,
                    "owner_id": deployment.owner_id,
                    "environment": deployment.environment,
                    "status": deployment.status,
                    "deployment_type": deployment.deployment_type,
                    "endpoint": deployment.endpoint,
                    "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
                    "updated_at": deployment.updated_at.isoformat() if deployment.updated_at else None
                })
            
            return result
    except Exception as e:
        logger.error(f"Error getting deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/deployments/{deployment_id}", response_model=dict)
async def get_deployment(
    deployment_id: int,
    current_user: dict = Depends(get_current_active_user)
):
    """Get model deployment by ID."""
    try:
        with Session() as session:
            deployment = session.query(ModelDeployment).filter(ModelDeployment.id == deployment_id).first()
            if not deployment:
                raise HTTPException(status_code=404, detail="Deployment not found")
            
            # Get model
            model = None
            if deployment.model_id:
                model_obj = session.query(MLModel).filter(MLModel.id == deployment.model_id).first()
                if model_obj:
                    try:
                        metrics = json.loads(model_obj.metrics) if model_obj.metrics else {}
                    except:
                        metrics = {}
                    
                    model = {
                        "id": model_obj.id,
                        "name": model_obj.name,
                        "version": model_obj.version,
                        "model_type": model_obj.model_type,
                        "framework": model_obj.framework,
                        "metrics": metrics
                    }
            
            # Get owner
            owner = None
            if deployment.owner_id:
                owner_obj = session.query(Employee).filter(Employee.id == deployment.owner_id).first()
                if owner_obj:
                    owner = {
                        "id": owner_obj.id,
                        "name": owner_obj.name,
                        "employee_id": owner_obj.employee_id,
                        "role": owner_obj.role
                    }
            
            # Get metrics
            metrics = session.query(DeploymentMetric).filter(DeploymentMetric.deployment_id == deployment_id).all()
            metric_list = []
            for metric in metrics:
                metric_list.append({
                    "id": metric.id,
                    "metric_name": metric.metric_name,
                    "metric_value": metric.metric_value,
                    "timestamp": metric.timestamp.isoformat() if metric.timestamp else None
                })
            
            return {
                "id": deployment.id,
                "model": model,
                "owner": owner,
                "environment": deployment.environment,
                "status": deployment.status,
                "deployment_type": deployment.deployment_type,
                "endpoint": deployment.endpoint,
                "metrics": metric_list,
                "created_at": deployment.created_at.isoformat() if deployment.created_at else None,
                "updated_at": deployment.updated_at.isoformat() if deployment.updated_at else None
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/deployments", status_code=status.HTTP_201_CREATED)
async def create_deployment(
    deployment: dict,
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new model deployment."""
    try:
        with Session() as session:
            # Check if model exists
            model = session.query(MLModel).filter(MLModel.id == deployment["model_id"]).first()
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Check if owner exists
            owner = session.query(Employee).filter(Employee.id == deployment["owner_id"]).first()
            if not owner:
                raise HTTPException(status_code=404, detail="Owner not found")
            
            # Create deployment
            new_deployment = ModelDeployment(
                model_id=deployment["model_id"],
                owner_id=deployment["owner_id"],
                environment=deployment["environment"],
                status=deployment.get("status", "pending"),
                deployment_type=deployment["deployment_type"],
                endpoint=deployment.get("endpoint")
            )
            
            session.add(new_deployment)
            session.commit()
            
            # If Kubernetes deployment, initiate deployment process
            if deployment["deployment_type"] == "kubernetes" and kubernetes_manager:
                # Run deployment in background
                threading.Thread(
                    target=deploy_model_to_kubernetes,
                    args=(new_deployment.id, model.id, model.path, f"model-{model.id}", 2),
                    daemon=True
                ).start()
            
            return {
                "id": new_deployment.id,
                "model_id": new_deployment.model_id,
                "environment": new_deployment.environment,
                "message": "Deployment created successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def deploy_model_to_kubernetes