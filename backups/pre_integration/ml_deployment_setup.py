#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_deployment_setup.py - Setup and Example Usage for ML Deployment Management System

This script provides a comprehensive setup and demonstration of the ML Deployment
and Management System with 10,000 employee structure, GPT-4o integration, and
advanced monitoring capabilities.

Features:
1. System setup and initialization
2. 10,000 employee hierarchy creation
3. Example model deployments
4. API usage examples
5. Configuration templates
6. Health check and monitoring endpoints
7. OpenAI unofficial with GPT-4o models (audio & realtime)
8. WebSocket examples
9. Complete system summary

Part of Skyscope Sentinel Intelligence AI - ITERATION 11
"""

import asyncio
import base64
import datetime
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Check for required packages
REQUIRED_PACKAGES = [
    "aiohttp", "fastapi", "uvicorn", "pydantic", "sqlalchemy", 
    "numpy", "pandas", "torch", "websockets", "kubernetes", 
    "prometheus_client", "python-jose", "passlib", "jinja2",
    "python-multipart"
]

MISSING_PACKAGES = []
for package in REQUIRED_PACKAGES:
    try:
        __import__(package)
    except ImportError:
        MISSING_PACKAGES.append(package)

if MISSING_PACKAGES:
    print(f"Missing required packages: {', '.join(MISSING_PACKAGES)}")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + MISSING_PACKAGES)
    print("Please restart the script after installation.")
    sys.exit(1)

# Now import required packages
import aiohttp
import fastapi
import numpy as np
import pandas as pd
import pydantic
import requests
import torch
import uvicorn
import websockets
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Boolean, DateTime, create_engine, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# Try to import Kubernetes client
try:
    import kubernetes
    from kubernetes import client, config, watch
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    warnings.warn("Kubernetes client not available. Kubernetes deployment features disabled.")

# Try to import OpenAI unofficial
try:
    import openai_unofficial
    from openai_unofficial import OpenAI
    OPENAI_UNOFFICIAL_AVAILABLE = True
except ImportError:
    OPENAI_UNOFFICIAL_AVAILABLE = False
    warnings.warn("openai-unofficial package not available. Using standard OpenAI package or Ollama as fallback.")
    try:
        import openai
    except ImportError:
        warnings.warn("OpenAI package not found. Only Ollama will be available if installed.")

# Try to import Ollama for fallback
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    if not OPENAI_UNOFFICIAL_AVAILABLE:
        warnings.warn("Neither openai-unofficial nor Ollama are available. LLM features will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_deployment_setup.log"),
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
DB_PATH = "sqlite:///ml_deployment.db"

# Create database models
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
    team = relationship("Team", back_populates="employees", foreign_keys=[team_id])
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
engine = create_engine(DB_PATH)
Session = sessionmaker(bind=engine)

#######################################################
# Helper Functions for Initialization
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


def initialize_employee_hierarchy(session, count: int = MAX_EMPLOYEES):
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


def create_model_registry_structure():
    """Create the model registry directory structure."""
    try:
        MODEL_REGISTRY_PATH.mkdir(exist_ok=True)
        for subdir in ["models", "metadata", "experiments", "deployments", "federated", "ab_tests"]:
            path = MODEL_REGISTRY_PATH / subdir
            path.mkdir(exist_ok=True)
        
        logger.info("Model registry structure created")
        return True
    except Exception as e:
        logger.error(f"Error creating model registry structure: {e}")
        return False


#######################################################
# OpenAI Unofficial Integration
#######################################################

class OpenAIManager:
    """
    Manages integration with openai-unofficial package for GPT-4o models
    including audio-preview & realtime-preview variants.
    """
    
    def __init__(self):
        self.client = None
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.token_usage = defaultdict(int)
        self.fallback_to_ollama = True
        self.ollama_client = None
        
        # Initialize clients
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize API clients."""
        try:
            if OPENAI_UNOFFICIAL_AVAILABLE:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI Unofficial client initialized")
            elif "openai" in sys.modules:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
                logger.info("Standard OpenAI client initialized as fallback")
            
            if OLLAMA_AVAILABLE and self.fallback_to_ollama:
                self.ollama_client = ollama
                logger.info("Ollama client initialized for fallback")
        except Exception as e:
            logger.error(f"Error initializing API clients: {e}")
    
    def set_api_key(self, api_key: str) -> bool:
        """Set the OpenAI API key."""
        try:
            self.api_key = api_key
            
            if OPENAI_UNOFFICIAL_AVAILABLE:
                self.client = OpenAI(api_key=api_key)
            elif "openai" in sys.modules:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            
            return True
        except Exception as e:
            logger.error(f"Error setting API key: {e}")
            return False
    
    def chat_completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a chat completion using GPT-4o models."""
        try:
            if not self.client:
                return {"error": "No API client available"}
            
            # Check if audio output is requested
            audio_output = request.get("audio_output", False)
            
            # Select appropriate model based on audio output
            model = request.get("model", DEFAULT_MODEL)
            if audio_output and not model.endswith("audio-preview"):
                model = AUDIO_PREVIEW_MODEL
                logger.info(f"Switching to audio-preview model: {model}")
            
            # Prepare messages
            messages = request.get("messages", [])
            
            # Set up response format for audio if needed
            response_format = None
            if audio_output:
                response_format = {
                    "type": "text_and_audio",
                    "audio": {
                        "format": "mp3"
                    }
                }
            
            # Try with openai-unofficial first
            try:
                if OPENAI_UNOFFICIAL_AVAILABLE:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=request.get("stream", False),
                        response_format=response_format
                    )
                    
                    # Extract audio if present
                    audio_data = None
                    if audio_output and hasattr(response.choices[0].message, "audio"):
                        audio_data = response.choices[0].message.audio
                    
                    return {
                        "id": response.id,
                        "model": response.model,
                        "content": response.choices[0].message.content,
                        "audio": audio_data,
                        "finish_reason": response.choices[0].finish_reason
                    }
                elif "openai" in sys.modules:
                    # Fallback to standard OpenAI package
                    import openai
                    response = self.client.chat.completions.create(
                        model=model if "gpt-4o" in model else "gpt-4o",  # Fallback to standard model
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=request.get("stream", False)
                    )
                    
                    return {
                        "id": response.id,
                        "model": response.model,
                        "content": response.choices[0].message.content,
                        "finish_reason": response.choices[0].finish_reason
                    }
            except Exception as e:
                logger.warning(f"OpenAI API error: {e}. Trying Ollama fallback.")
                
                # Fallback to Ollama if available
                if OLLAMA_AVAILABLE and self.fallback_to_ollama:
                    # Convert messages to Ollama format
                    ollama_messages = []
                    for msg in messages:
                        ollama_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    
                    response = self.ollama_client.chat(
                        model=DEFAULT_OLLAMA_MODEL,
                        messages=ollama_messages,
                        stream=False
                    )
                    
                    return {
                        "id": str(uuid.uuid4()),
                        "model": DEFAULT_OLLAMA_MODEL,
                        "content": response["message"]["content"],
                        "finish_reason": "stop",
                        "using_fallback": True
                    }
                else:
                    raise
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return {"error": str(e)}
    
    async def streaming_chat_completion(self, request: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate a streaming chat completion."""
        try:
            if not self.client:
                yield {"error": "No API client available"}
                return
            
            # Prepare messages
            messages = request.get("messages", [])
            model = request.get("model", DEFAULT_MODEL)
            
            # Try with openai-unofficial first
            try:
                if OPENAI_UNOFFICIAL_AVAILABLE:
                    stream = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield {
                                "id": chunk.id,
                                "model": chunk.model,
                                "content": chunk.choices[0].delta.content,
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                elif "openai" in sys.modules:
                    # Fallback to standard OpenAI package
                    import openai
                    stream = self.client.chat.completions.create(
                        model=model if "gpt-4o" in model else "gpt-4o",
                        messages=messages,
                        temperature=request.get("temperature", 0.7),
                        max_tokens=request.get("max_tokens"),
                        stream=True
                    )
                    
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield {
                                "id": chunk.id,
                                "model": chunk.model,
                                "content": chunk.choices[0].delta.content,
                                "finish_reason": chunk.choices[0].finish_reason
                            }
            except Exception as e:
                logger.warning(f"OpenAI API streaming error: {e}. Trying Ollama fallback.")
                
                # Fallback to Ollama if available
                if OLLAMA_AVAILABLE and self.fallback_to_ollama:
                    # Convert messages to Ollama format
                    ollama_messages = []
                    for msg in messages:
                        ollama_messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
                    
                    stream = self.ollama_client.chat(
                        model=DEFAULT_OLLAMA_MODEL,
                        messages=ollama_messages,
                        stream=True
                    )
                    
                    for chunk in stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            yield {
                                "id": str(uuid.uuid4()),
                                "model": DEFAULT_OLLAMA_MODEL,
                                "content": chunk["message"]["content"],
                                "finish_reason": None,
                                "using_fallback": True
                            }
                else:
                    yield {"error": str(e)}
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            yield {"error": str(e)}
    
    def text_to_speech(self, text: str, voice: str = "alloy", output_format: str = "mp3") -> Optional[bytes]:
        """Convert text to speech using OpenAI TTS."""
        try:
            if not self.client:
                return None
            
            if OPENAI_UNOFFICIAL_AVAILABLE:
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format=output_format
                )
                
                # Get audio data
                audio_data = response.content
                
                return audio_data
            elif "openai" in sys.modules:
                # Fallback to standard OpenAI package
                import openai
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=text,
                    response_format=output_format
                )
                
                # Get audio data
                audio_data = response.content
                
                return audio_data
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            return None


#######################################################
# FastAPI Application Setup
#######################################################

def create_fastapi_app():
    """Create and configure the FastAPI application."""
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
    
    # Create OpenAI manager
    openai_manager = OpenAIManager()
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
            "database": check_database_connection(),
            "openai_available": OPENAI_UNOFFICIAL_AVAILABLE or "openai" in sys.modules,
            "ollama_available": OLLAMA_AVAILABLE,
            "kubernetes_available": KUBERNETES_AVAILABLE
        }
    
    def check_database_connection():
        """Check database connection."""
        try:
            with Session() as session:
                session.execute("SELECT 1")
            return True
        except:
            return False
    
    # Metrics endpoint
    @app.get("/metrics")
    async def get_metrics():
        """Get system metrics."""
        try:
            with Session() as session:
                employee_count = session.query(func.count(Employee.id)).scalar()
                team_count = session.query(func.count(Team.id)).scalar()
                project_count = session.query(func.count(Project.id)).scalar()
                model_count = session.query(func.count(MLModel.id)).scalar()
                deployment_count = session.query(func.count(ModelDeployment.id)).scalar()
                
                # Get active deployments
                active_deployments = session.query(func.count(ModelDeployment.id)).filter(
                    ModelDeployment.status == "active"
                ).scalar()
                
                # Get deployment types
                deployment_types = {}
                for deployment_type in ["kubernetes", "edge", "cloud"]:
                    count = session.query(func.count(ModelDeployment.id)).filter(
                        ModelDeployment.deployment_type == deployment_type
                    ).scalar()
                    deployment_types[deployment_type] = count
                
                # Get system metrics
                system_metrics = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                }
                
                return {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "employee_count": employee_count,
                    "team_count": team_count,
                    "project_count": project_count,
                    "model_count": model_count,
                    "deployment_count": deployment_count,
                    "active_deployments": active_deployments,
                    "deployment_types": deployment_types,
                    "system_metrics": system_metrics
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # OpenAI chat completion endpoint
    @app.post("/api/chat")
    async def chat_completion(request: dict):
        """Generate a chat completion using GPT-4o models."""
        try:
            result = openai_manager.chat_completion(request)
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # OpenAI streaming chat completion endpoint
    @app.websocket("/api/chat/stream")
    async def streaming_chat_completion(websocket: WebSocket):
        """Generate a streaming chat completion over WebSocket."""
        try:
            await websocket.accept()
            
            # Receive request
            request_data = await websocket.receive_json()
            
            # Generate streaming completion
            async for chunk in openai_manager.streaming_chat_completion(request_data):
                await websocket.send_json(chunk)
            
            # Send completion message
            await websocket.send_json({"finish_reason": "stop", "done": True})
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error in streaming chat completion: {e}")
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
    
    # Text-to-speech endpoint
    @app.post("/api/tts")
    async def text_to_speech(text: str = Form(...), voice: str = Form("alloy")):
        """Convert text to speech using OpenAI TTS."""
        try:
            audio_data = openai_manager.text_to_speech(text, voice)
            if not audio_data:
                raise HTTPException(status_code=500, detail="Failed to generate speech")
            
            return Response(
                content=audio_data,
                media_type="audio/mpeg",
                headers={"Content-Disposition": f'attachment; filename="speech_{int(time.time())}.mp3"'}
            )
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket for real-time audio (simulated for demonstration)
    @app.websocket("/api/realtime-audio")
    async def realtime_audio_websocket(websocket: WebSocket):
        """Simulate a real-time audio WebSocket connection."""
        try:
            await websocket.accept()
            
            # Send welcome message
            await websocket.send_json({
                "type": "info",
                "message": "Connected to simulated GPT-4o realtime-preview WebSocket"
            })
            
            # Wait for configuration
            config = await websocket.receive_json()
            
            # Acknowledge configuration
            await websocket.send_json({
                "type": "config_ack",
                "message": "Configuration received",
                "config": config
            })
            
            # Process messages
            while True:
                try:
                    # Wait for audio data or text
                    message = await websocket.receive()
                    
                    if "bytes" in message:
                        # Simulate processing audio
                        await asyncio.sleep(0.5)  # Simulate processing time
                        
                        # Send simulated transcription
                        await websocket.send_json({
                            "type": "transcription",
                            "text": "This is a simulated transcription of the audio you sent."
                        })
                        
                        # Send simulated response
                        await websocket.send_json({
                            "type": "response",
                            "text": "This is a simulated response from GPT-4o realtime-preview model."
                        })
                        
                        # Send simulated audio response (just sending a message instead of actual audio)
                        await websocket.send_json({
                            "type": "audio_response",
                            "message": "This would be audio data in a real implementation."
                        })
                    elif "text" in message:
                        text_data = message.get("text", "")
                        
                        # Simulate processing text
                        await asyncio.sleep(0.2)  # Simulate processing time
                        
                        # Send simulated response
                        await websocket.send_json({
                            "type": "response",
                            "text": f"Simulated response to: {text_data}"
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Unsupported message format"
                        })
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected")
                    break
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected during setup")
        except Exception as e:
            logger.error(f"Error in realtime audio WebSocket: {e}")
    
    return app


#######################################################
# Configuration Templates
#######################################################

def generate_configuration_templates():
    """Generate configuration templates for the system."""
    try:
        # Create config directory
        config_dir = Path("./config")
        config_dir.mkdir(exist_ok=True)
        
        # Create main configuration file
        main_config = {
            "system": {
                "name": "Skyscope ML Deployment and Management System",
                "version": "1.0.0",
                "log_level": "INFO",
                "max_employees": MAX_EMPLOYEES,
                "database_url": DB_PATH,
                "model_registry_path": str(MODEL_REGISTRY_PATH)
            },
            "openai": {
                "api_key": "${OPENAI_API_KEY}",  # Use environment variable
                "default_model": DEFAULT_MODEL,
                "audio_preview_model": AUDIO_PREVIEW_MODEL,
                "realtime_preview_model": REALTIME_PREVIEW_MODEL,
                "fallback_to_ollama": True
            },
            "ollama": {
                "enabled": True,
                "default_model": DEFAULT_OLLAMA_MODEL,
                "host": "http://localhost:11434"
            },
            "kubernetes": {
                "enabled": KUBERNETES_AVAILABLE,
                "namespace": KUBERNETES_NAMESPACE,
                "use_in_cluster_config": False,
                "kubeconfig_path": "${HOME}/.kube/config"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "cors_origins": ["*"],
                "docs_url": "/api/docs",
                "redoc_url": "/api/redoc"
            },
            "monitoring": {
                "prometheus_port": 9090,
                "metrics_endpoint": "/metrics",
                "health_endpoint": "/health",
                "log_metrics_interval": 60  # seconds
            }
        }
        
        with open(config_dir / "config.json", "w") as f:
            json.dump(main_config, f, indent=2)
        
        # Create monitoring configuration
        monitoring_config = {
            "metrics": {
                "model_performance": True,
                "model_drift": True,
                "system_resources": True,
                "api_requests": True,
                "deployment_status": True
            },
            "alerts": {
                "enabled": True,
                "channels": {
                    "email": {
                        "enabled": False,
                        "recipients": ["alerts@skyscope.ai"]
                    },
                    "slack": {
                        "enabled": False,
                        "webhook_url": "${SLACK_WEBHOOK_URL}"
                    },
                    "webhook": {
                        "enabled": False,
                        "url": "https://alerts.skyscope.ai/webhook"
                    }
                },
                "thresholds": {
                    "accuracy_drop": 0.05,
                    "latency_increase": 0.2,
                    "drift_threshold": 0.1,
                    "error_rate": 0.02,
                    "cpu_usage": 90,
                    "memory_usage": 90,
                    "disk_usage": 90
                }
            },
            "logging": {
                "file": "logs/monitoring.log",
                "level": "INFO",
                "rotation": "1 day",
                "retention": "30 days"
            }
        }
        
        with open(config_dir / "monitoring.json", "w") as f:
            json.dump(monitoring_config, f, indent=2)
        
        # Create deployment configuration
        deployment_config = {
            "environments": {
                "dev": {
                    "replicas": 1,
                    "resources": {
                        "requests": {
                            "cpu": "100m",
                            "memory": "256Mi"
                        },
                        "limits": {
                            "cpu": "500m",
                            "memory": "512Mi"
                        }
                    },
                    "auto_scaling": False
                },
                "staging": {
                    "replicas": 2,
                    "resources": {
                        "requests": {
                            "cpu": "500m",
                            "memory": "1Gi"
                        },
                        "limits": {
                            "cpu": "1",
                            "memory": "2Gi"
                        }
                    },
                    "auto_scaling": True,
                    "min_replicas": 2,
                    "max_replicas": 5
                },
                "production": {
                    "replicas": 3,
                    "resources": {
                        "requests": {
                            "cpu": "1",
                            "memory": "2Gi"
                        },
                        "limits": {
                            "cpu": "2",
                            "memory": "4Gi"
                        }
                    },
                    "auto_scaling": True,
                    "min_replicas": 3,
                    "max_replicas": 10
                }
            },
            "strategies": {
                "rolling": {
                    "max_surge": "25%",
                    "max_unavailable": "25%"
                },
                "blue_green": {
                    "pre_switch_validation_period": 60  # seconds
                },
                "canary": {
                    "initial_weight": 20,  # percent
                    "increment": 20,  # percent
                    "interval": 300,  # seconds
                    "max_failures": 2
                }
            },
            "registry": {
                "url": "docker.io/skyscope",
                "credentials_secret": "registry-credentials"
            }
        }
        
        with open(config_dir / "deployment.json", "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        # Create .env.template file
        env_template = """# Skyscope ML Deployment and Management System
# Environment Variables Template

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=sqlite:///ml_deployment.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Kubernetes Configuration
KUBECONFIG=/path/to/kubeconfig
KUBERNETES_NAMESPACE=skyscope-ml

# Monitoring Configuration
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO

# Alert Channels
SLACK_WEBHOOK_URL=your_slack_webhook_url
ALERT_EMAIL=alerts@skyscope.ai

# JWT Authentication
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
"""
        
        with open(config_dir / ".env.template", "w") as f:
            f.write(env_template)
        
        logger.info("Configuration templates generated successfully")
        return True
    except Exception as e:
        logger.error(f"Error generating configuration templates: {e}")
        return False


#######################################################
# Example API Usage
#######################################################

def example_api_usage():
    """Demonstrate example API usage."""
    print("\n" + "="*80)
    print("EXAMPLE API USAGE")
    print("="*80)
    
    # Create a session
    session = Session()
    
    try:
        # Get employee count
        employee_count = session.query(func.count(Employee.id)).scalar()
        print(f"\n1. System has {employee_count} employees initialized")
        
        # Get a random employee
        employee = session.query(Employee).order_by(func.random()).first()
        print(f"\n2. Random employee: {employee.name} (ID: {employee.id}, Role: {employee.role})")
        
        # Get employee's team
        if employee.team_id:
            team = session.query(Team).filter(Team.id == employee.team_id).first()
            print(f"   Team: {team.name} (Department: {team.department}, Division: {team.division})")
            
            # Get team manager
            if team.manager_id:
                manager = session.query(Employee).filter(Employee.id == team.manager_id).first()
                print(f"   Manager: {manager.name} (ID: {manager.id})")
        
        # Get a random model
        model = session.query(MLModel).order_by(func.random()).first()
        if model:
            print(f"\n3. Random ML model: {model.name} v{model.version} (ID: {model.id}, Type: {model.model_type})")
            
            # Get model's project
            if model.project_id:
                project = session.query(Project).filter(Project.id == model.project_id).first()
                print(f"   Project: {project.name} (Status: {project.status})")
            
            # Get model's deployments
            deployments = session.query(ModelDeployment).filter(ModelDeployment.model_id == model.id).all()
            if deployments:
                print(f"   Deployments: {len(deployments)}")
                for deployment in deployments:
                    print(f"     - Environment: {deployment.environment}, Status: {deployment.status}, Type: {deployment.deployment_type}")
                    print(f"       Endpoint: {deployment.endpoint}")
        
        # Example REST API calls (using requests)
        base_url = "http://localhost:8000"
        print("\n4. Example REST API calls (these would be executed when the API is running):")
        
        print(f"\n   GET {base_url}/health")
        print("   Response: {'status': 'healthy', 'timestamp': '2025-07-16T12:34:56.789012', ...}")
        
        print(f"\n   GET {base_url}/api/employees?department=Research&limit=5")
        print("   Response: [{'id': 1, 'name': 'Employee 1', 'role': 'Data Scientist', ...}, ...]")
        
        print(f"\n   POST {base_url}/api/models")
        print("   Request: {'name': 'new_classification_model', 'version': '1.0.0', 'model_type': 'classification', ...}")
        print("   Response: {'id': 21, 'name': 'new_classification_model', 'message': 'Model created successfully'}")
        
        print(f"\n   POST {base_url}/api/chat")
        print("   Request: {'model': 'gpt-4o-2024-05-13', 'messages': [{'role': 'user', 'content': 'Hello'}]}")
        print("   Response: {'id': 'chatcmpl-123', 'model': 'gpt-4o-2024-05-13', 'content': 'Hello! How can I assist you today?', ...}")
        
        # Example WebSocket usage
        print("\n5. Example WebSocket usage (these would be executed when the API is running):")
        
        print("\n   WebSocket connection to ws://localhost:8000/api/chat/stream")
        print("   Send: {'model': 'gpt-4o-2024-05-13', 'messages': [{'role': 'user', 'content': 'Write a poem about AI'}]}")
        print("   Receive stream: {'id': 'chatcmpl-123', 'content': 'In silicon dreams', ...}")
        print("   Receive stream: {'id': 'chatcmpl-123', 'content': ' and neural light,', ...}")
        print("   ... (streaming continues)")
        
        print("\n   WebSocket connection to ws://localhost:8000/api/realtime-audio")
        print("   Send config: {'speech_recognition': {'model': 'whisper-1'}, 'text_to_speech': {'voice': 'alloy'}}")
        print("   Send audio bytes: <binary audio data>")
        print("   Receive: {'type': 'transcription', 'text': 'What's the weather like today?'}")
        print("   Receive: {'type': 'response', 'text': 'The weather today is sunny with a high of 75°F.'}")
        print("   Receive: {'type': 'audio_response', 'data': <binary audio data>}")
        
        # OpenAI unofficial example (simulated)
        print("\n6. OpenAI Unofficial Integration Example:")
        
        openai_manager = OpenAIManager()
        if openai_manager.client:
            print("\n   Chat completion with GPT-4o:")
            print("   Request: {'model': 'gpt-4o-2024-05-13', 'messages': [{'role': 'user', 'content': 'Hello'}]}")
            print("   (This would make an actual API call if OpenAI API key is configured)")
            
            print("\n   Audio output with GPT-4o-audio-preview:")
            print("   Request: {'model': 'gpt-4o-audio-preview-2025-06-03', 'messages': [{'role': 'user', 'content': 'Tell me a story'}], 'audio_output': True}")
            print("   (This would make an actual API call if OpenAI API key is configured)")
            
            print("\n   Text-to-speech with OpenAI TTS:")
            print("   Request: text='Hello, this is a test of text-to-speech', voice='alloy'")
            print("   (This would make an actual API call if OpenAI API key is configured)")
        else:
            print("\n   OpenAI client not available. Set OPENAI_API_KEY environment variable to use this feature.")
        
        # Kubernetes deployment example (simulated)
        print("\n7. Kubernetes Deployment Example (simulated):")
        
        if KUBERNETES_AVAILABLE:
            print("\n   Deploying model to Kubernetes:")
            print("   Request: {'model_id': 1, 'deployment_name': 'model-1', 'replicas': 2, 'environment': 'production'}")
            print("   (This would create actual Kubernetes resources if configured)")
            
            print("\n   Kubernetes deployment with blue-green strategy:")
            print("   Request: {'model_id': 2, 'deployment_name': 'model-2', 'replicas': 3, 'strategy': 'blue_green'}")
            print("   (This would create blue and green deployments in Kubernetes if configured)")
        else:
            print("\n   Kubernetes client not available. Install kubernetes package to use this feature.")
    except Exception as e:
        print(f"\nError in example API usage: {e}")
    finally:
        session.close()


#######################################################
# System Summary
#######################################################

def print_system_summary():
    """Print a summary of the complete system."""
    print("\n" + "="*80)
    print("SKYSCOPE ML DEPLOYMENT AND MANAGEMENT SYSTEM SUMMARY")
    print("="*80)
    
    print("\nSystem Components:")
    print("------------------")
    print("1. Employee Hierarchy Management")
    print("   - 10,000 employee structure with teams, departments, and divisions")
    print("   - Manager assignment and performance tracking")
    print("   - Skill registry and employee-team relationships")
    
    print("\n2. ML Model Lifecycle Management")
    print("   - Model registration, versioning, and metadata tracking")
    print("   - Model training with AutoML and Neural Architecture Search")
    print("   - Federated learning across teams")
    print("   - A/B testing and model comparison")
    
    print("\n3. OpenAI GPT-4o Integration")
    print("   - Support for GPT-4o models via openai-unofficial package")
    print("   - Audio-preview model for text-to-speech capabilities")
    print("   - Realtime-preview model for real-time audio conversations")
    print("   - Fallback to standard OpenAI package or Ollama")
    
    print("\n4. Kubernetes Deployment")
    print("   - Automated deployment to Kubernetes clusters")
    print("   - Support for rolling updates, blue-green, and canary deployments")
    print("   - Resource management and auto-scaling")
    print("   - Multi-region deployment")
    
    print("\n5. Model Monitoring")
    print("   - Performance tracking and metrics collection")
    print("   - Drift detection and anomaly detection")
    print("   - Alerting and notification system")
    print("   - Compliance and audit trails")
    
    print("\n6. API and Interface")
    print("   - RESTful API with FastAPI")
    print("   - WebSocket support for streaming and real-time audio")
    print("   - Health check and monitoring endpoints")
    print("   - Authentication and authorization")
    
    print("\nSystem Requirements:")
    print("-------------------")
    print("- Python 3.9+")
    print("- SQLite (or other SQL database)")
    print("- OpenAI API key (for GPT-4o integration)")
    print("- Kubernetes cluster (optional, for deployment)")
    print("- Prometheus (optional, for metrics)")
    
    print("\nSetup Instructions:")
    print("------------------")
    print("1. Install required packages:")
    print("   pip install -r requirements.txt")
    print("\n2. Set environment variables:")
    print("   export OPENAI_API_KEY=your_openai_api_key")
    print("\n3. Initialize the system:")
    print("   python ml_deployment_setup.py --init")
    print("\n4. Start the API server:")
    print("   uvicorn ml_deployment_api:app --host 0.0.0.0 --port 8000 --workers 4")
    
    print("\nAPI Documentation:")
    print("----------------")
    print("- Swagger UI: http://localhost:8000/api/docs")
    print("- ReDoc: http://localhost:8000/api/redoc")
    
    print("\nMonitoring:")
    print("----------")
    print("- Health check: http://localhost:8000/health")
    print("- Metrics: http://localhost:8000/metrics")
    print("- Prometheus: http://localhost:9090")
    
    print("\nConfiguration:")
    print("-------------")
    print("- Main config: ./config/config.json")
    print("- Monitoring config: ./config/monitoring.json")
    print("- Deployment config: ./config/deployment.json")
    print("- Environment variables: ./config/.env")
    
    print("\n" + "="*80)


#######################################################
# Main Function
#######################################################

def main():
    """Main function to run the setup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ML Deployment and Management System Setup")
    parser.add_argument("--init", action="store_true", help="Initialize the system")
    parser.add_argument("--config", action="store_true", help="Generate configuration templates")
    parser.add_argument("--examples", action="store_true", help="Show example API usage")
    parser.add_argument("--summary", action="store_true", help="Print system summary")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    parser.add_argument("--api", action="store_true", help="Start the API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Run all steps if --all is specified
    if args.all:
        args.init = True
        args.config = True
        args.examples = True
        args.summary = True
    
    # Initialize the system
    if args.init:
        print("\nInitializing the system...")
        
        # Initialize database
        if initialize_database():
            print("Database initialized successfully")
        else:
            print("Failed to initialize database")
            return
        
        # Create model registry structure
        if create_model_registry_structure():
            print("Model registry structure created successfully")
        else:
            print("Failed to create model registry structure")
        
        # Initialize employee hierarchy
        with Session() as session:
            if initialize_employee_hierarchy(session):
                print(f"Employee hierarchy initialized successfully with {MAX_EMPLOYEES} employees")
            else:
                print("Failed to initialize employee hierarchy")
    
    # Generate configuration templates
    if args.config:
        if generate_configuration_templates():
            print("Configuration templates generated successfully")
        else:
            print("Failed to generate configuration templates")
    
    # Show example API usage
    if args.examples:
        example_api_usage()
    
    # Print system summary
    if args.summary:
        print_system_summary()
    
    # Start the API server
    if args.api:
        print(f"\nStarting API server on {args.host}:{args.port}...")
        app = create_fastapi_app()
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
