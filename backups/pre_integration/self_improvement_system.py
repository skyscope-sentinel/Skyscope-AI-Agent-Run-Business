#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skyscope Sentinel Intelligence AI Platform - Self-Improvement System

This module provides comprehensive self-improvement capabilities for the Skyscope platform,
including continuous learning from agent experiences, performance monitoring, hyperparameter
optimization, strategy evolution, knowledge distillation, adaptive learning rate scheduling,
multi-objective optimization, automated architecture search, experience replay, meta-learning,
curriculum learning, and online learning updates.

Created on: July 16, 2025
Author: Skyscope Sentinel Intelligence
"""

import os
import sys
import time
import json
import uuid
import logging
import threading
import multiprocessing
import pickle
import hashlib
import datetime
import random
import copy
import warnings
import tempfile
import shutil
import re
import math
from typing import Dict, List, Tuple, Set, Any, Optional, Union, Callable, TypeVar, Generic
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque, Counter, OrderedDict
from abc import ABC, abstractmethod
import heapq
import itertools

# Import ML libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("self_improvement_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Self_Improvement_System')

# Constants
SYSTEM_VERSION = "1.0.0"
DEFAULT_CONFIG_PATH = "config/self_improvement_system.json"
DEFAULT_HISTORY_DIR = "data/improvement_history"
DEFAULT_POPULATION_SIZE = 50
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_CROSSOVER_RATE = 0.7
DEFAULT_TOURNAMENT_SIZE = 5
DEFAULT_GENERATIONS = 20
DEFAULT_EXPERIENCE_BUFFER_SIZE = 100000
DEFAULT_PRIORITY_ALPHA = 0.6
DEFAULT_PRIORITY_BETA_START = 0.4
DEFAULT_PRIORITY_BETA_END = 1.0
DEFAULT_CURRICULUM_LEVELS = 10
DEFAULT_META_LEARNING_STEPS = 5

# Improvement types
class ImprovementType(Enum):
    """Enumeration of improvement types"""
    HYPERPARAMETER_OPTIMIZATION = "hyperparameter_optimization"
    ARCHITECTURE_SEARCH = "architecture_search"
    STRATEGY_EVOLUTION = "strategy_evolution"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    EXPERIENCE_REPLAY = "experience_replay"
    META_LEARNING = "meta_learning"
    CURRICULUM_LEARNING = "curriculum_learning"
    ONLINE_LEARNING = "online_learning"
    ENSEMBLE_CREATION = "ensemble_creation"
    PRUNING = "pruning"

# Improvement status
class ImprovementStatus(Enum):
    """Enumeration of improvement statuses"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SCHEDULED = "scheduled"

@dataclass
class ImprovementResult:
    """Data class for storing improvement results"""
    improvement_id: str
    original_model_id: str
    improved_model_id: str
    improvement_type: ImprovementType
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    improvement_percentage: Dict[str, float]
    parameters_before: Dict[str, Any]
    parameters_after: Dict[str, Any]
    duration: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    notes: str = ""

@dataclass
class ExperienceRecord:
    """Data class for storing agent experience records"""
    agent_id: str
    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    priority: float = 1.0
    
@dataclass
class PerformanceRecord:
    """Data class for storing agent performance records"""
    agent_id: str
    metrics: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyGene:
    """Data class for representing a strategy gene in genetic algorithms"""
    gene_id: str
    parameters: Dict[str, Any]
    fitness: float = 0.0
    age: int = 0
    mutation_history: List[str] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)

@dataclass
class ArchitectureCandidate:
    """Data class for representing an architecture candidate in architecture search"""
    architecture_id: str
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    performance: float = 0.0
    parameters_count: int = 0
    flops: int = 0
    training_time: float = 0.0
    inference_time: float = 0.0

@dataclass
class CurriculumTask:
    """Data class for representing a task in curriculum learning"""
    task_id: str
    difficulty: float
    description: str
    parameters: Dict[str, Any]
    prerequisites: List[str] = field(default_factory=list)
    completion_criteria: Dict[str, float] = field(default_factory=dict)
    
@dataclass
class SelfImprovementConfig:
    """Configuration for self-improvement system"""
    history_dir: str = DEFAULT_HISTORY_DIR
    
    # Continuous learning settings
    cl_update_frequency: int = 100
    cl_min_experiences: int = 1000
    cl_batch_size: int = 64
    
    # Performance monitoring settings
    pm_metrics_history_length: int = 100
    pm_alert_threshold: float = 0.2
    pm_trend_window: int = 10
    
    # Hyperparameter optimization settings
    hpo_method: str = "optuna"  # "optuna", "grid", "random", "evolution"
    hpo_trials: int = 50
    hpo_timeout: int = 3600  # seconds
    hpo_cross_validation: int = 3
    
    # Strategy evolution settings
    se_population_size: int = DEFAULT_POPULATION_SIZE
    se_mutation_rate: float = DEFAULT_MUTATION_RATE
    se_crossover_rate: float = DEFAULT_CROSSOVER_RATE
    se_tournament_size: int = DEFAULT_TOURNAMENT_SIZE
    se_generations: int = DEFAULT_GENERATIONS
    se_elitism_count: int = 2
    
    # Knowledge distillation settings
    kd_temperature: float = 3.0
    kd_alpha: float = 0.5
    kd_teacher_count: int = 3
    
    # Adaptive learning rate settings
    alr_method: str = "cosine"  # "cosine", "step", "plateau", "linear", "exponential"
    alr_patience: int = 10
    alr_factor: float = 0.5
    alr_min_lr: float = 1e-6
    
    # Multi-objective optimization settings
    moo_method: str = "nsga2"  # "nsga2", "moea/d", "spea2"
    moo_population_size: int = 100
    moo_generations: int = 30
    moo_objectives: List[str] = field(default_factory=lambda: ["performance", "complexity", "inference_time"])
    
    # Automated architecture search settings
    aas_method: str = "evolution"  # "evolution", "random", "bayesian"
    aas_max_layers: int = 10
    aas_max_evaluations: int = 100
    aas_time_budget: int = 7200  # seconds
    
    # Experience replay settings
    er_buffer_size: int = DEFAULT_EXPERIENCE_BUFFER_SIZE
    er_priority_alpha: float = DEFAULT_PRIORITY_ALPHA
    er_priority_beta_start: float = DEFAULT_PRIORITY_BETA_START
    er_priority_beta_end: float = DEFAULT_PRIORITY_BETA_END
    er_priority_epsilon: float = 0.01
    
    # Meta-learning settings
    ml_adaptation_steps: int = DEFAULT_META_LEARNING_STEPS
    ml_meta_batch_size: int = 32
    ml_meta_lr: float = 0.001
    ml_task_lr: float = 0.1
    
    # Curriculum learning settings
    cl_levels: int = DEFAULT_CURRICULUM_LEVELS
    cl_promotion_threshold: float = 0.8
    cl_difficulty_step: float = 0.1
    cl_min_tasks_per_level: int = 5
    
    # Online learning settings
    ol_update_frequency: int = 50
    ol_batch_size: int = 16
    ol_learning_rate: float = 0.001
    ol_max_drift: float = 0.3
    
    # General settings
    random_seed: int = 42
    verbose: bool = False
    debug: bool = False
    max_parallel_improvements: int = multiprocessing.cpu_count()
    improvement_history_length: int = 100
    auto_improvement_enabled: bool = True

class SelfImprovementSystem:
    """
    Main self-improvement system class for Skyscope Sentinel Intelligence AI Platform.
    
    This class orchestrates all self-improvement capabilities including continuous learning,
    performance monitoring, hyperparameter optimization, strategy evolution, knowledge distillation,
    adaptive learning rate scheduling, multi-objective optimization, automated architecture search,
    experience replay, meta-learning, curriculum learning, and online learning updates.
    """
    
    def __init__(self, config: Optional[SelfImprovementConfig] = None):
        """
        Initialize the self-improvement system.
        
        Args:
            config: Self-improvement configuration (optional)
        """
        self.config = config or SelfImprovementConfig()
        
        # Initialize components
        self.continuous_learning = ContinuousLearningManager(self.config)
        self.performance_monitoring = PerformanceMonitoringManager(self.config)
        self.hyperparameter_optimization = HyperparameterOptimizationManager(self.config)
        self.strategy_evolution = StrategyEvolutionManager(self.config)
        self.knowledge_distillation = KnowledgeDistillationManager(self.config)
        self.adaptive_learning_rate = AdaptiveLearningRateManager(self.config)
        self.multi_objective_optimization = MultiObjectiveOptimizationManager(self.config)
        self.automated_architecture_search = AutomatedArchitectureSearchManager(self.config)
        self.experience_replay = ExperienceReplayManager(self.config)
        self.meta_learning = MetaLearningManager(self.config)
        self.curriculum_learning = CurriculumLearningManager(self.config)
        self.online_learning = OnlineLearningManager(self.config)
        
        # Improvement history
        self.improvement_history: List[ImprovementResult] = []
        self.pending_improvements: Dict[str, Tuple[ImprovementType, Dict[str, Any]]] = {}
        self.active_improvements: Dict[str, threading.Thread] = {}
        
        # Create directories
        os.makedirs(self.config.history_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Load improvement history
        self._load_improvement_history()
        
        # Start auto-improvement thread if enabled
        if self.config.auto_improvement_enabled:
            self.auto_improvement_thread = threading.Thread(
                target=self._auto_improvement_loop,
                daemon=True
            )
            self.auto_improvement_thread.start()
        
        logger.info(f"Self-improvement system initialized (version {SYSTEM_VERSION})")
    
    @classmethod
    def from_config_file(cls, config_path: str = DEFAULT_CONFIG_PATH) -> 'SelfImprovementSystem':
        """
        Create a self-improvement system instance from a configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            SelfImprovementSystem instance
        """
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            config = SelfImprovementConfig(**config_data)
            return cls(config)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        seed = self.config.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _load_improvement_history(self) -> None:
        """Load improvement history from files."""
        history_file = os.path.join(self.config.history_dir, "improvement_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                
                for item in history_data:
                    # Convert string enums back to enum values
                    item['improvement_type'] = ImprovementType(item['improvement_type'])
                    
                    self.improvement_history.append(ImprovementResult(**item))
                
                # Trim history if needed
                if len(self.improvement_history) > self.config.improvement_history_length:
                    self.improvement_history = self.improvement_history[-self.config.improvement_history_length:]
                
                logger.info(f"Loaded {len(self.improvement_history)} improvement history records")
            except Exception as e:
                logger.error(f"Error loading improvement history: {e}")
                self.improvement_history = []
    
    def _save_improvement_history(self) -> None:
        """Save improvement history to files."""
        try:
            # Convert to serializable format
            history_data = []
            for result in self.improvement_history:
                # Convert enum values to strings for JSON serialization
                history_data.append({
                    **result.__dict__,
                    'improvement_type': result.improvement_type.value
                })
            
            history_file = os.path.join(self.config.history_dir, "improvement_history.json")
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving improvement history: {e}")
    
    def _auto_improvement_loop(self) -> None:
        """Background thread for automatic improvement scheduling."""
        logger.info("Starting auto-improvement thread")
        
        while True:
            try:
                # Check if we can schedule more improvements
                if len(self.active_improvements) < self.config.max_parallel_improvements:
                    # Check for models that need improvement
                    models_to_improve = self.performance_monitoring.get_models_needing_improvement()
                    
                    for model_id, metrics in models_to_improve.items():
                        # Determine best improvement method based on metrics
                        improvement_type = self._determine_best_improvement_method(model_id, metrics)
                        
                        # Schedule improvement
                        if improvement_type is not None:
                            self.schedule_improvement(model_id, improvement_type)
                
                # Process pending improvements
                self._process_pending_improvements()
                
                # Sleep for a while
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in auto-improvement loop: {e}")
                time.sleep(300)  # Sleep longer after an error
    
    def _determine_best_improvement_method(self, model_id: str, 
                                         metrics: Dict[str, float]) -> Optional[ImprovementType]:
        """
        Determine the best improvement method for a model based on its metrics.
        
        Args:
            model_id: Model ID
            metrics: Model performance metrics
            
        Returns:
            ImprovementType or None
        """
        # Get performance history
        history = self.performance_monitoring.get_performance_history(model_id)
        
        if not history:
            # Not enough history, default to hyperparameter optimization
            return ImprovementType.HYPERPARAMETER_OPTIMIZATION
        
        # Check for performance degradation
        if self.performance_monitoring.detect_performance_degradation(model_id):
            # If performance is degrading, try online learning
            return ImprovementType.ONLINE_LEARNING
        
        # Check for plateauing performance
        if self.performance_monitoring.detect_performance_plateau(model_id):
            # If performance is plateauing, alternate between different methods
            last_improvements = [
                result for result in self.improvement_history 
                if result.original_model_id == model_id
            ]
            
            if not last_improvements:
                return ImprovementType.HYPERPARAMETER_OPTIMIZATION
            
            last_type = last_improvements[-1].improvement_type
            
            # Cycle through improvement types
            if last_type == ImprovementType.HYPERPARAMETER_OPTIMIZATION:
                return ImprovementType.ARCHITECTURE_SEARCH
            elif last_type == ImprovementType.ARCHITECTURE_SEARCH:
                return ImprovementType.STRATEGY_EVOLUTION
            elif last_type == ImprovementType.STRATEGY_EVOLUTION:
                return ImprovementType.KNOWLEDGE_DISTILLATION
            elif last_type == ImprovementType.KNOWLEDGE_DISTILLATION:
                return ImprovementType.EXPERIENCE_REPLAY
            else:
                return ImprovementType.HYPERPARAMETER_OPTIMIZATION
        
        # No specific issues detected
        return None
    
    def _process_pending_improvements(self) -> None:
        """Process pending improvements if slots are available."""
        # Check if we have capacity for more active improvements
        while (len(self.active_improvements) < self.config.max_parallel_improvements and 
               self.pending_improvements):
            # Get next improvement
            improvement_id = next(iter(self.pending_improvements.keys()))
            improvement_type, params = self.pending_improvements.pop(improvement_id)
            
            # Start improvement in a separate thread
            thread = threading.Thread(
                target=self._run_improvement,
                args=(improvement_id, improvement_type, params),
                daemon=True
            )
            thread.start()
            
            # Add to active improvements
            self.active_improvements[improvement_id] = thread
            
            logger.info(f"Started improvement {improvement_id} of type {improvement_type.value}")
    
    def _run_improvement(self, improvement_id: str, improvement_type: ImprovementType, 
                       params: Dict[str, Any]) -> None:
        """
        Run an improvement task in a separate thread.
        
        Args:
            improvement_id: Improvement ID
            improvement_type: Type of improvement
            params: Improvement parameters
        """
        try:
            # Get model ID
            model_id = params.get("model_id")
            if model_id is None:
                logger.error(f"No model ID provided for improvement {improvement_id}")
                return
            
            # Get current metrics and parameters
            metrics_before = self.performance_monitoring.get_latest_metrics(model_id)
            parameters_before = params.get("parameters", {})
            
            # Start timing
            start_time = time.time()
            
            # Run improvement based on type
            if improvement_type == ImprovementType.HYPERPARAMETER_OPTIMIZATION:
                improved_model_id = self.hyperparameter_optimization.optimize(model_id, **params)
            elif improvement_type == ImprovementType.ARCHITECTURE_SEARCH:
                improved_model_id = self.automated_architecture_search.search(model_id, **params)
            elif improvement_type == ImprovementType.STRATEGY_EVOLUTION:
                improved_model_id = self.strategy_evolution.evolve(model_id, **params)
            elif improvement_type == ImprovementType.KNOWLEDGE_DISTILLATION:
                improved_model_id = self.knowledge_distillation.distill(model_id, **params)
            elif improvement_type == ImprovementType.EXPERIENCE_REPLAY:
                improved_model_id = self.experience_replay.replay(model_id, **params)
            elif improvement_type == ImprovementType.META_LEARNING:
                improved_model_id = self.meta_learning.meta_train(model_id, **params)
            elif improvement_type == ImprovementType.CURRICULUM_LEARNING:
                improved_model_id = self.curriculum_learning.train_curriculum(model_id, **params)
            elif improvement_type == ImprovementType.ONLINE_LEARNING:
                improved_model_id = self.online_learning.update(model_id, **params)
            else:
                logger.error(f"Unsupported improvement type: {improvement_type}")
                return
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Get new metrics and parameters
            metrics_after = self.performance_monitoring.get_latest_metrics(improved_model_id)
            parameters_after = params.get("new_parameters", {})
            
            # Calculate improvement percentages
            improvement_percentage = {}
            for metric, value_after in metrics_after.items():
                if metric in metrics_before:
                    value_before = metrics_before[metric]
                    if value_before != 0:
                        improvement_percentage[metric] = (value_after - value_before) / abs(value_before) * 100
            
            # Create improvement result
            result = ImprovementResult(
                improvement_id=improvement_id,
                original_model_id=model_id,
                improved_model_id=improved_model_id,
                improvement_type=improvement_type,
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                improvement_percentage=improvement_percentage,
                parameters_before=parameters_before,
                parameters_after=parameters_after,
                duration=duration,
                notes=f"Improvement completed successfully in {duration:.2f} seconds"
            )
            
            # Add to history
            self.improvement_history.append(result)
            if len(self.improvement_history) > self.config.improvement_history_length:
                self.improvement_history = self.improvement_history[-self.config.improvement_history_length:]
            
            # Save history
            self._save_improvement_history()
            
            logger.info(f"Completed improvement {improvement_id} of type {improvement_type.value}")
            
            # Log improvement metrics
            for metric, percentage in improvement_percentage.items():
                logger.info(f"  {metric}: {percentage:.2f}% improvement")
        
        except Exception as e:
            logger.error(f"Error in improvement {improvement_id}: {e}")
        
        finally:
            # Remove from active improvements
            if improvement_id in self.active_improvements:
                del self.active_improvements[improvement_id]
    
    def schedule_improvement(self, model_id: str, improvement_type: ImprovementType, 
                           **params) -> str:
        """
        Schedule an improvement task.
        
        Args:
            model_id: Model ID
            improvement_type: Type of improvement
            **params: Additional parameters for the improvement
            
        Returns:
            Improvement ID
        """
        # Generate improvement ID
        improvement_id = str(uuid.uuid4())
        
        # Add to parameters
        all_params = {
            "model_id": model_id,
            **params
        }
        
        # Add to pending improvements
        self.pending_improvements[improvement_id] = (improvement_type, all_params)
        
        # Process pending improvements
        self._process_pending_improvements()
        
        logger.info(f"Scheduled {improvement_type.value} improvement for model {model_id}")
        return improvement_id
    
    def cancel_improvement(self, improvement_id: str) -> bool:
        """
        Cancel a scheduled or active improvement task.
        
        Args:
            improvement_id: Improvement ID
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        # Check if in pending improvements
        if improvement_id in self.pending_improvements:
            del self.pending_improvements[improvement_id]
            logger.info(f"Cancelled pending improvement {improvement_id}")
            return True
        
        # Check if in active improvements
        if improvement_id in self.active_improvements:
            # We can't actually stop the thread, but we can mark it as cancelled
            logger.info(f"Marked active improvement {improvement_id} as cancelled, but it will continue running")
            return True
        
        logger.warning(f"Improvement {improvement_id} not found")
        return False
    
    def get_improvement_status(self, improvement_id: str) -> Optional[ImprovementStatus]:
        """
        Get the status of an improvement task.
        
        Args:
            improvement_id: Improvement ID
            
        Returns:
            ImprovementStatus or None if not found
        """
        # Check if in pending improvements
        if improvement_id in self.pending_improvements:
            return ImprovementStatus.PENDING
        
        # Check if in active improvements
        if improvement_id in self.active_improvements:
            return ImprovementStatus.IN_PROGRESS
        
        # Check if in history
        for result in self.improvement_history:
            if result.improvement_id == improvement_id:
                return ImprovementStatus.COMPLETED
        
        return None
    
    def get_improvement_result(self, improvement_id: str) -> Optional[ImprovementResult]:
        """
        Get the result of a completed improvement task.
        
        Args:
            improvement_id: Improvement ID
            
        Returns:
            ImprovementResult or None if not found or not completed
        """
        for result in self.improvement_history:
            if result.improvement_id == improvement_id:
                return result
        
        return None
    
    def list_improvements(self, model_id: Optional[str] = None, 
                        improvement_type: Optional[ImprovementType] = None) -> List[ImprovementResult]:
        """
        List improvement results with optional filtering.
        
        Args:
            model_id: Filter by model ID (optional)
            improvement_type: Filter by improvement type (optional)
            
        Returns:
            List of ImprovementResult objects
        """
        results = self.improvement_history.copy()
        
        if model_id is not None:
            results = [r for r in results if r.original_model_id == model_id]
        
        if improvement_type is not None:
            results = [r for r in results if r.improvement_type == improvement_type]
        
        return results
    
    def add_experience(self, agent_id: str, state: Any, action: Any, reward: float, 
                     next_state: Any, done: bool, info: Dict[str, Any] = None) -> None:
        """
        Add an experience record for continuous learning.
        
        Args:
            agent_id: Agent ID
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information (optional)
        """
        self.experience_replay.add_experience(
            agent_id, state, action, reward, next_state, done, info or {}
        )
    
    def record_performance(self, agent_id: str, metrics: Dict[str, float], 
                         context: Dict[str, Any] = None) -> None:
        """
        Record agent performance metrics.
        
        Args:
            agent_id: Agent ID
            metrics: Performance metrics
            context: Additional context information (optional)
        """
        self.performance_monitoring.record_performance(
            agent_id, metrics, context or {}
        )
    
    def optimize_hyperparameters(self, model_id: str, param_space: Dict[str, Any], 
                               **kwargs) -> str:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_id: Model ID
            param_space: Hyperparameter search space
            **kwargs: Additional optimization parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            model_id,
            ImprovementType.HYPERPARAMETER_OPTIMIZATION,
            param_space=param_space,
            **kwargs
        )
    
    def evolve_strategy(self, model_id: str, strategy_params: Dict[str, Any], 
                      fitness_function: Callable, **kwargs) -> str:
        """
        Evolve a strategy using genetic algorithms.
        
        Args:
            model_id: Model ID
            strategy_params: Strategy parameters to evolve
            fitness_function: Function to evaluate fitness
            **kwargs: Additional evolution parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            model_id,
            ImprovementType.STRATEGY_EVOLUTION,
            strategy_params=strategy_params,
            fitness_function=fitness_function,
            **kwargs
        )
    
    def distill_knowledge(self, student_model_id: str, teacher_model_ids: List[str], 
                        **kwargs) -> str:
        """
        Distill knowledge from teacher models to a student model.
        
        Args:
            student_model_id: Student model ID
            teacher_model_ids: List of teacher model IDs
            **kwargs: Additional distillation parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            student_model_id,
            ImprovementType.KNOWLEDGE_DISTILLATION,
            teacher_model_ids=teacher_model_ids,
            **kwargs
        )
    
    def search_architecture(self, model_id: str, search_space: Dict[str, Any], 
                          **kwargs) -> str:
        """
        Search for an optimal neural network architecture.
        
        Args:
            model_id: Model ID
            search_space: Architecture search space
            **kwargs: Additional search parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            model_id,
            ImprovementType.ARCHITECTURE_SEARCH,
            search_space=search_space,
            **kwargs
        )
    
    def prioritized_replay(self, model_id: str, batch_size: int = None, 
                         **kwargs) -> str:
        """
        Perform prioritized experience replay for a model.
        
        Args:
            model_id: Model ID
            batch_size: Batch size (optional)
            **kwargs: Additional replay parameters
            
        Returns:
            Improvement ID
        """
        if batch_size is None:
            batch_size = self.config.cl_batch_size
        
        return self.schedule_improvement(
            model_id,
            ImprovementType.EXPERIENCE_REPLAY,
            batch_size=batch_size,
            **kwargs
        )
    
    def meta_train(self, model_id: str, task_datasets: List[Any], 
                 **kwargs) -> str:
        """
        Perform meta-learning for a model.
        
        Args:
            model_id: Model ID
            task_datasets: List of datasets for different tasks
            **kwargs: Additional meta-learning parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            model_id,
            ImprovementType.META_LEARNING,
            task_datasets=task_datasets,
            **kwargs
        )
    
    def train_with_curriculum(self, model_id: str, curriculum: List[CurriculumTask], 
                            **kwargs) -> str:
        """
        Train a model using curriculum learning.
        
        Args:
            model_id: Model ID
            curriculum: List of curriculum tasks
            **kwargs: Additional curriculum learning parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            model_id,
            ImprovementType.CURRICULUM_LEARNING,
            curriculum=curriculum,
            **kwargs
        )
    
    def update_online(self, model_id: str, new_data: Any, 
                    **kwargs) -> str:
        """
        Update a model online with new data.
        
        Args:
            model_id: Model ID
            new_data: New data for online learning
            **kwargs: Additional online learning parameters
            
        Returns:
            Improvement ID
        """
        return self.schedule_improvement(
            model_id,
            ImprovementType.ONLINE_LEARNING,
            new_data=new_data,
            **kwargs
        )
    
    def get_improvement_recommendations(self, model_id: str) -> List[Tuple[ImprovementType, float]]:
        """
        Get recommendations for improvements for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            List of (ImprovementType, confidence) tuples, sorted by confidence
        """
        # Get performance history
        history = self.performance_monitoring.get_performance_history(model_id)
        
        if not history:
            # Not enough history, recommend basic improvements
            return [
                (ImprovementType.HYPERPARAMETER_OPTIMIZATION, 0.9),
                (ImprovementType.EXPERIENCE_REPLAY, 0.7),
                (ImprovementType.ONLINE_LEARNING, 0.5)
            ]
        
        recommendations = []
        
        # Check for performance degradation
        if self.performance_monitoring.detect_performance_degradation(model_id):
            recommendations.append((ImprovementType.ONLINE_LEARNING, 0.9))
            recommendations.append((ImprovementType.EXPERIENCE_REPLAY, 0.8))
        
        # Check for performance plateau
        if self.performance_monitoring.detect_performance_plateau(model_id):
            recommendations.append((ImprovementType.HYPERPARAMETER_OPTIMIZATION, 0.8))
            recommendations.append((ImprovementType.ARCHITECTURE_SEARCH, 0.7))
            recommendations.append((ImprovementType.STRATEGY_EVOLUTION, 0.6))
        
        # Check improvement history
        past_improvements = [
            result for result in self.improvement_history 
            if result.original_model_id == model_id
        ]
        
        # If certain improvements haven't been tried, recommend them
        improvement_types_tried = {result.improvement_type for result in past_improvements}
        
        if ImprovementType.HYPERPARAMETER_OPTIMIZATION not in improvement_types_tried:
            recommendations.append((ImprovementType.HYPERPARAMETER_OPTIMIZATION, 0.85))
        
        if ImprovementType.KNOWLEDGE_DISTILLATION not in improvement_types_tried:
            recommendations.append((ImprovementType.KNOWLEDGE_DISTILLATION, 0.75))
        
        if ImprovementType.CURRICULUM_LEARNING not in improvement_types_tried:
            recommendations.append((ImprovementType.CURRICULUM_LEARNING, 0.7))
        
        # Add some general recommendations with lower confidence
        if ImprovementType.META_LEARNING not in improvement_types_tried:
            recommendations.append((ImprovementType.META_LEARNING, 0.5))
        
        if ImprovementType.ARCHITECTURE_SEARCH not in improvement_types_tried:
            recommendations.append((ImprovementType.ARCHITECTURE_SEARCH, 0.6))
        
        # Sort by confidence and remove duplicates
        seen_types = set()
        unique_recommendations = []
        
        for imp_type, confidence in sorted(recommendations, key=lambda x: x[1], reverse=True):
            if imp_type not in seen_types:
                unique_recommendations.append((imp_type, confidence))
                seen_types.add(imp_type)
        
        return unique_recommendations
    
    def analyze_improvement_history(self, model_id: str) -> Dict[str, Any]:
        """
        Analyze the improvement history for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary with analysis results
        """
        # Get improvements for this model
        improvements = [
            result for result in self.improvement_history 
            if result.original_model_id == model_id
        ]
        
        if not improvements:
            return {
                "model_id": model_id,
                "total_improvements": 0,
                "message": "No improvement history found for this model"
            }
        
        # Count by type
        type_counts = Counter([imp.improvement_type for imp in improvements])
        
        # Calculate average improvement by metric
        metric_improvements = defaultdict(list)
        for imp in improvements:
            for metric, percentage in imp.improvement_percentage.items():
                metric_improvements[metric].append(percentage)
        
        avg_improvements = {
            metric: sum(values) / len(values) 
            for metric, values in metric_improvements.items()
        }
        
        # Find most effective improvement type for each metric
        best_type_by_metric = {}
        for metric in metric_improvements:
            best_improvement = None
            best_percentage = float('-inf')
            
            for imp in improvements:
                if metric in imp.improvement_percentage:
                    percentage = imp.improvement_percentage[metric]
                    if percentage > best_percentage:
                        best_percentage = percentage
                        best_improvement = imp
            
            if best_improvement:
                best_type_by_metric[metric] = (best_improvement.improvement_type, best_percentage)
        
        # Calculate total improvement over time
        cumulative_improvement = {}
        for metric in metric_improvements:
            cumulative = 0
            cumulative_over_time = []
            
            for imp in sorted(improvements, key=lambda x: x.timestamp):
                if metric in imp.improvement_percentage:
                    cumulative += imp.improvement_percentage[metric]
                    cumulative_over_time.append((imp.timestamp, cumulative))
            
            if cumulative_over_time:
                cumulative_improvement[metric] = cumulative_over_time
        
        # Return analysis
        return {
            "model_id": model_id,
            "total_improvements": len(improvements),
            "improvement_types": {t.value: c for t, c in type_counts.items()},
            "average_improvement_by_metric": avg_improvements,
            "best_improvement_type_by_metric": {
                metric: (imp_type.value, percentage) 
                for metric, (imp_type, percentage) in best_type_by_metric.items()
            },
            "cumulative_improvement": cumulative_improvement,
            "first_improvement": improvements[0].timestamp,
            "last_improvement": improvements[-1].timestamp
        }
    
    def generate_improvement_report(self, model_id: str) -> str:
        """
        Generate a human-readable report of improvements for a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Report string
        """
        analysis = self.analyze_improvement_history(model_id)
        
        if analysis["total_improvements"] == 0:
            return f"No improvement history found for model {model_id}"
        
        # Build report
        report = [
            f"Improvement Report for Model {model_id}",
            f"=" * 50,
            f"Total improvements: {analysis['total_improvements']}",
            f"First improvement: {analysis['first_improvement']}",
            f"Last improvement: {analysis['last_improvement']}",
            "",
            "Improvement Types:",
            "-" * 20
        ]
        
        for imp_type, count in analysis["improvement_types"].items():
            report.append(f"- {imp_type}: {count}")
        
        report.extend([
            "",
            "Average Improvement by Metric:",
            "-" * 30
        ])
        
        for metric, avg in analysis["average_improvement_by_metric"].items():
            report.append(f"- {metric}: {avg:.2f}%")
        
        report.extend([
            "",
            "Best Improvement Type by Metric:",
            "-" * 30
        ])
        
        for metric, (imp_type, percentage) in analysis["best_improvement_type_by_metric"].items():
            report.append(f"- {metric}: {imp_type} ({percentage:.2f}%)")
        
        report.extend([
            "",
            "Recommendations:",
            "-" * 20
        ])
        
        recommendations = self.get_improvement_recommendations(model_id)
        for imp_type, confidence in recommendations:
            report.append(f"- {imp_type.value}: {confidence:.2f} confidence")
        
        return "\n".join(report)

class ContinuousLearningManager:
    """
    Manages continuous learning from agent experiences.
    """
    
    def __init__(self, config: SelfImprovementConfig):
        """
        Initialize the continuous learning manager.
        
        Args:
            config: Self-improvement configuration
        """
        self.config = config
        self.experience_counts = defaultdict(int)
        self.last_update_time = {}
        
        logger.info("Continuous learning manager initialized")
    
    def should_update(self, agent_id: str) -> bool:
        """
        Check if an agent should be updated based on new experiences.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if agent should be updated, False otherwise
        """
        # Check if we have enough experiences
        if self.experience_counts[agent_id] < self.config.cl_min_experiences:
            return False
        
        # Check if it's time for an update
        current_time = time.time()
        last_time = self.last_update_time.get(agent_id, 0)
        
        if current_time - last_time < self.config.cl_update_frequency:
            return False
        
        return True
    
    def update_agent(self, agent_id: str, model, experience_buffer: List[ExperienceRecord]) -> bool:
        """
        Update an agent's model with new experiences.
        
        Args:
            agent_id: Agent ID
            model: Agent's model
            experience_buffer: Buffer of experience records
            
        Returns:
            True if update was successful, False otherwise
        """
        if not experience_buffer:
            logger.warning(f"No experiences available for agent {agent_id}")
            return False
        
        try:
            # Prepare batch data
            batch_size = min(self.config.cl_batch_size, len(experience_buffer))
            batch = random.sample(experience_buffer, batch_size)
            
            # Extract states, actions, rewards, next_states, dones
            states = [exp.state for exp in batch]
            actions = [exp.action for exp in batch]
            rewards = [exp.reward for exp in batch]
            next_states = [exp.next_state for exp in batch]
            dones = [exp.done for exp in batch]
            
            # Update model (implementation depends on model type)
            if hasattr(model, 'update_from_experiences'):
                model.update_from_experiences(states, actions, rewards, next_states, dones)
            elif hasattr(model, 'fit'):
                # For supervised learning models, we need to create X, y pairs
                X = np.array(states)
                y = np.array(rewards)  # Simplified - actual target depends on the task
                model.fit(X, y)
            else:
                logger.warning(f"Model for agent {agent_id} doesn't support continuous learning")
                return False
            
            # Update last update time
            self.last_update_time[agent_id] = time.time()
            
            # Reset experience count
            self.experience_counts[agent_id] = 0
            
            logger.info(f"Updated agent {agent_id} with {batch_size} experiences")
            return True
        
        except Exception as e:
            logger.error(f"Error updating agent {agent_id}: {e}")
            return False
    
    def record_experience(self, agent_id: str) -> None:
        """
        Record that an agent has had a new experience.
        
        Args:
            agent_id: Agent ID
        """
        self.experience_counts[agent_id] += 1

class PerformanceMonitoringManager:
    """
    Manages performance monitoring and trend analysis.
    """
    
    def __init__(self, config: SelfImprovementConfig):
        """
        Initialize the performance monitoring manager.
        
        Args:
            config: Self-improvement configuration
        """
        self.config = config
        self.performance_history: Dict[str, List[PerformanceRecord]] = defaultdict(list)
        
        logger.info("Performance monitoring manager initialized")
    
    def record_performance(self, agent_id: str, metrics: Dict[str, float], 
                         context: Dict[str, Any] = None) -> None:
        """
        Record performance metrics for an agent.
        
        Args:
            agent_id: Agent ID
            metrics: Performance metrics
            context: Additional context information (optional)
        """
        record = PerformanceRecord(
            agent_id=agent_id,
            metrics=metrics,
            context=context or {}
        )
        
        # Add to history
        self.performance_history[agent_id].append(record)
        
        # Trim history if needed
        if len(self.performance_history[agent_id]) > self.config.pm_metrics_history_length:
            self.performance_history[agent_id] = self.performance_history[agent_id][-self.config.pm_metrics_history_length:]
        
        # Check for alerts
        self._check_alerts(agent_id, record)
    
    def _check_alerts(self, agent_id: str, record: PerformanceRecord) -> None:
        """
        Check for performance alerts based on new metrics.
        
        Args:
            agent_id: Agent ID
            record: New performance record
        """
        history = self.performance_history[agent_id]
        
        if len(history) < 2:
            return  # Not enough history
        
        # Get previous record
        prev_record = history[-2]
        
        # Check for significant drops in performance
        for metric, value in record.metrics.items():
            if metric in prev_record.metrics:
                prev_value = prev_record.metrics[metric]
                if prev_value != 0 and abs(value - prev_value) / abs(prev_value) > self.config.pm_alert_threshold:
                    logger.warning(f"Performance alert for agent {agent_id}: {metric} changed by {(value - prev_value) / prev_value * 100:.2f}%")
    
    def get_performance_history(self, agent_id: str) -> List[PerformanceRecord]:
        """
        Get performance history for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            List of performance records
        """
        return self.performance_history.get(agent_id, [])
    
    def get_latest_metrics(self, agent_id: str) -> Dict[str, float]:
        """
        Get the latest performance metrics for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary of metrics or empty dict if no history
        """
        history = self.performance_history.get(agent_id, [])
        if not history:
            return {}
        
        return history[-1].metrics
    
    def detect_performance_degradation(self, agent_id: str) -> bool:
        """
        Detect if an agent's performance is degrading.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if performance is degrading, False otherwise
        """
        history = self.performance_history.get(agent_id, [])
        
        if len(history) < self.config.pm_trend_window:
            return False  # Not enough history
        
        # Get recent history
        recent = history[-self.config.pm_trend_window:]
        
        # Check for consistent degradation in any metric
        for metric in recent[0].metrics:
            # Check if metric exists in all records
            if not all(metric in record.metrics for record in recent):
                continue
            
            # Extract values
            values = [record.metrics[metric] for record in recent]
            
            # Check if consistently decreasing (for metrics where higher is better)
            # or consistently increasing (for metrics where lower is better)
            # This is a simplified approach - in practice, you'd need to know which metrics are better when higher/lower
            
            # Check if values are consistently decreasing
            decreasing = all(values[i] > values[i+1] for i in range(len(values)-1))
            
            # Check if values are consistently increasing
            increasing = all(values[i] < values[i+1] for i in range(len(values)-1))
            
            if decreasing or increasing:
                return True
        
        return False
    
    def detect_performance_plateau(self, agent_id: str) -> bool:
        """
        Detect if an agent's performance has plateaued.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            True if performance has plateaued, False otherwise
        """
        history = self.performance_history.get(agent_id, [])
        
        if len(history) < self.config.pm_trend_window:
            return False  # Not enough history
        
        # Get recent history
        recent = history[-self.config.pm_trend_window:]
        
        # Check for plateau in any metric
        for metric in recent[0].metrics:
            # Check if metric exists in all records
            if not all(metric in record.metrics for record in recent):
                continue
            
            # Extract values
            values = [record.metrics[metric] for record in recent]
            
            # Calculate mean and standard deviation
            mean = sum(values) / len(values)
            std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            
            # Check if standard deviation is very small relative to mean
            if abs(mean) > 0 and std_dev / abs(mean) < 0.01:
                return True
        
        return False
    
    def analyze_trends(self, agent_id: str) -> Dict[str, Any]:
        """
        Analyze performance trends for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Dictionary with trend analysis
        """
        history = self.performance_history.get(agent_id, [])
        
        if len(history) < 2:
            return {
                "agent_id": agent_id,
                "message": "Not enough history for trend analysis"
            }
        
        # Find all metrics that appear in the history
        all_metrics = set()
        for record in history:
            all_metrics.update(record.metrics.keys())
        
        # Analyze each metric
        trends = {}
        for metric in all_metrics:
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for record in history:
                if metric in record.metrics:
                    values.append(record.metrics[metric])
                    timestamps.append(record.timestamp)
            
            if len(values) < 2:
                continue  # Not enough data for this metric
            
            # Calculate trend statistics
            mean = sum(values) / len(values)
            std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            min_val = min(values)
            max_val = max(values)
            
            # Calculate slope (simple linear regression)
            n = len(values)
            indices = list(range(n))
            
            mean_x = sum(indices) / n
            mean_y = sum(values) / n
            
            numerator = sum((indices[i] - mean_x) * (values[i] - mean_y) for i in range(n))
            denominator = sum((indices[i] - mean_x) ** 2 for i in range(n))
            
            slope = numerator / denominator if denominator != 0 else 0
            
            # Determine trend direction
            if abs(slope) < 0.001:
                direction = "stable"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"
            
            # Calculate recent change (last 5 points or fewer)
            recent_window = min(5, len(values))
            recent_values = values[-recent_window:]
            recent_change = (recent_values[-1] - recent_values[0]) / recent_values[0] if recent_values[0] != 0 else 0
            
            # Store trend data
            trends[metric] = {
                "mean": mean,
                "std_dev": std_dev,
                "min": min_val,
                "max": max_val,
                "slope": slope,
                "direction": direction,
                "recent_change_pct": recent_change * 100,
                "values": values,
                "timestamps": timestamps
            }
        
        return {
            "agent_id": agent_id,
            "num_records": len(history),
            "first_timestamp": history[0].timestamp,
            "last_timestamp": history[-1].timestamp,
            "trends": trends
        }
    
    def get_models_needing_improvement(self) -> Dict[str, Dict[str, float]]:
        """
        Get models that need improvement based on performance metrics.
        
        Returns:
            Dictionary mapping model IDs to their latest metrics
        """
        models_to_improve = {}
        
        for agent_id, history in self.performance_history.items():
            if not history:
                continue
            
            # Check if performance is degrading or has plateaued
            if (self.detect_performance_degradation(agent_id) or 
                self.detect_performance_plateau(agent_id)):
                models_to_improve[agent_id] = history[-1].metrics
        
        return models_to_improve
    
    def visualize_performance(self, agent_id: str, metric: str, 
                            output_file: str = None) -> Optional[plt.Figure]:
        """
        Visualize performance trends for a specific metric.
        
        Args:
            agent_id: Agent ID
            metric: Metric name
            output_file: Output file path (optional)
            
        Returns:
            Matplotlib figure or None if visualization failed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            history = self.performance_history.get(agent_id, [])
            
            if len(history) < 2:
                logger.warning(f"Not enough history for agent {agent_id} to visualize")
                return None
            
            # Extract values and timestamps
            values = []
            timestamps = []
            
            for record in history:
                if metric in record.metrics:
                    values.append(record.metrics[metric])
                    timestamps.append(record.timestamp)
            
            if len(values) < 2:
                logger.warning(f"Not enough data for metric {metric}")
                return None
            
            # Create figure
            plt.figure(figsize=(10, 6))
            sns.set_style("darkgrid")
            
            # Plot metric
            plt.plot(range(len(values)), values, marker='o', linestyle='-', linewidth=2)
            
            # Add trend line
            z = np.polyfit(range(len(values)), values, 1)
            p = np.poly1d(z)
            plt.plot(range(len(values)), p(range(len(values))), "r--", linewidth=1)
            
            # Add labels and title
            plt.xlabel("Time")
            plt.ylabel(metric)
            plt.title(f"{metric} Performance for Agent {agent_id}")
            
            # Add grid
            plt.grid(True)
            
            # Save if output file is provided
            if output_file:
                plt.savefig(output_file)
                logger.info(f"Saved performance visualization to {output_file}")
            
            return plt.gcf()
        
        except Exception as e:
            logger.error(f"Error visualizing performance: {e}")
            return None

class HyperparameterOptimizationManager:
    """
    Manages automatic hyperparameter optimization.
    """
    
    def __init__(self, config: SelfImprovementConfig):
        """
        Initialize the hyperparameter optimization manager.
        
        Args:
            config: Self-improvement configuration
        """
        self.config = config
        
        logger.info("Hyperparameter optimization manager initialized")
    
    def optimize(self, model_id: str, param_space: Dict[str, Any], 
               eval_function: Callable = None, **kwargs) -> str:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_id: Model ID
            param_space: Hyperparameter search space
            eval_function: Function to evaluate parameter sets (optional)
            **kwargs: Additional optimization parameters
            
        Returns:
            Improved model ID
        """
        # Get optimization method
        method = kwargs.get("method", self.config.hpo_method)
        
        # Get number of trials
        trials = kwargs.get("trials", self.config.hpo_trials)
        
        # Get timeout
        timeout = kwargs.get("timeout", self.config.hpo_timeout)
        
        logger.info(f"Starting hyperparameter optimization for model {model_id} using {method}")
        
        # Run optimization based on method
        if method == "optuna":
            best_params = self._optimize_with_optuna(model_id, param_space, eval_function, trials, timeout)
        elif method == "grid":
            best_params = self._optimize_with_grid_search(model_id, param_space, eval_function)
        elif method == "random":
            best_params = self._optimize_with_random_search(model_id, param_space, eval_function, trials)
        elif method == "evolution":
            best_params = self._optimize_with_evolution(model_id, param_space, eval_function, trials)
        else:
            logger.error(f"Unsupported optimization method: {method}")
            raise ValueError(f"Unsupported optimization method: {method}")
        
        # Create improved model with best parameters
        improved_model_id = self._create_improved_model(model_id, best_params)
        
        logger.info(f"Hyperparameter optimization completed for model {model_id}")
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Created improved model: {improved_model_id}")
        
        return improved_model_id
    
    def _optimize_with_optuna(self, model_id: str, param_space: Dict[str, Any], 
                            eval_function: Callable, trials: int, timeout: int) -> Dict[str, Any]:
        """Optimize using Optuna."""
        import optuna
        
        # Define objective function for Optuna
        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(param_name, param_config["values"])
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config["low"], 
                        param_config["high"],
                        step=param_config.get("step", 1)
                    )
                elif param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                else:
                    logger.warning(f"Unsupported parameter type: {param_config['type']}")
            
            # Evaluate parameters
            return eval_function(model_id, params)
        
        # Create study
        study = optuna.create_study(direction="maximize")
        
        # Optimize
        study.optimize(objective, n_trials=trials, timeout=timeout)
        
        # Get best parameters
        return study.best_params
    
    def _optimize_with_grid_search(self, model_id: str, param_space: Dict[str, Any], 
                                 eval_function: Callable) -> Dict[str, Any]:
        """Optimize using grid search."""
        # Convert param_space to grid search format
        param_grid = {}
        for param_name, param_config in param_space.items():
            if param_config["type"] == "categorical":
                param_grid[param_name] = param_config["values"]
            elif param_config["type"] == "int":
                low = param_config["low"]
                high = param_config["high"]
                step = param_config.get("step", 1)
                param_grid[param_name] = list(range(low, high + 1, step))
            elif param_config["type"] == "float":
                low = param_config["low"]
                high = param_config["high"]
                num_points = param_config.get("num_points", 10)
                if param_config.get("log", False):
                    param_grid[param_name] = np.logspace(np.log10(low), np.log10(high), num_points).tolist()
                else:
                    param_grid[param_name] = np.linspace(low, high, num_points).tolist()
        
        # Generate all combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        # Evaluate all combinations
        best_score = float('-inf')
        best_params = None
        
        for params in combinations:
            score = eval_function(model_id, params)
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params
    
    def _optimize_with_random_search(self, model_id: str, param_space: Dict[str, Any], 
                                   eval_function: Callable, trials: int) -> Dict[str, Any]:
        """Optimize using random search."""
        best_score = float('-inf')
        best_params = None
        
        for _ in range(trials):
            # Sample random parameters
            params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "categorical":
                    params[param_name] = random.choice(param_config["values"])
                elif param_config["type"] == "int":
                    low = param_config["low"]
                    high = param_config["high"]
                    step = param_config.get("step", 1)
                    params[param_name] = random.randrange(low, high + 1, step)
                elif param_config["type"] == "float":
                    low = param_config["low"]
                    high = param_config["high"]
                    if param_config.get("log", False):
                        params[param_name] = 10 ** (random.uniform(np.log10(low), np.log10(high)))
                    else:
                        params[param_name] = random.uniform(low, high)
            
            # Evaluate parameters
            score = eval_function(model_id, params)
            if score > best_score:
                best_score = score
                best_params = params
        
        return best_params
    
    def _optimize_with_evolution(self, model_id: str, param_space: Dict[str, Any], 
                               eval_function: Callable, generations: int) -> Dict[str, Any]:
        """Optimize using evolutionary algorithms."""
        # Define bounds and parameter names
        bounds = []
        param_names = []
        param_types = {}
        
        for param_name, param_config in param_space.items():
            param_names.append(param_name)
            param_types[param_name] = param_config["type"]
            
            if param_config["type"] == "categorical":
                # For categorical, we'll use an index into the values list
                bounds.append((0, len(param_config["values"]) - 1))
            elif param_config["type"] in ["int", "float"]:
                bounds.append((param_config["low"], param_config["high"]))
        
        # Define objective function for differential evolution
        def objective(x):
            # Convert x to parameters
            params = {}
            for i, param_name in enumerate(param_names):
                param_config = param_space[param_name]
                
                if param_types[param_name] == "categorical":
                    # Convert index to actual value
                    index = int(x[i])
                    params[param_name] = param_config["values"][index]
                elif param_types[param_name] == "int":
                    params[param_name] = int(x[i])
                elif param_types[param_name] == "float":
                    params[param_name] = float(x[i])
            
            # Evaluate parameters
            return -eval_function(model_id, params)  # Negative because we're minimizing
        
        # Run differential evolution
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=generations,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        # Convert result back to parameters
        best_params = {}
        for i, param_name in enumerate(param_names):
            param_config = param_space[param_name]
            
            if param_types[param_name] == "categorical":
                index = int(result.x[i])
                best_params[param_name] = param_config["values"][index]
            elif param_types[param_name] == "int":
                best_params[param_name] = int(result.x[i])
            elif param_types[param_name] == "float":
                best_params[param_name] = float(result.x[i])
        
        return best_params
    
    def _create_improved_model(self, model_id: str, best_params: Dict[str, Any]) -> str:
        """
        Create an improved model with the best parameters.
        
        Args:
            model_id: Original model ID
            best_params: Best hyperparameters
            
        Returns:
            Improved model ID
        """
        # This is a placeholder - in a real implementation, you would:
        # 1. Load the original model
        # 2. Create a new model with the best parameters
        # 3. Train the new model
        # 4. Save and register the new model
        
        # For now, we'll just return a dummy ID
        improved_model_id = f"{model_id}_improved_{uuid.uuid4().hex[:8]}"
        
        return improved_model_id

class StrategyEvolutionManager:
    """
    Manages strategy evolution using genetic algorithms.
    """
    
    def __init__(self, config: SelfImprovementConfig):
        """
        Initialize the strategy evolution manager.
        
        Args:
            config: Self-improvement configuration
        """
        self.config = config
        self.populations: Dict[str, List[StrategyGene]] = {}
        
        logger.info("Strategy evolution manager initialized")
    
    def evolve(self, model_id: str, strategy_params: Dict[str, Any], 
             fitness_function: Callable, **kwargs) -> str:
        """
        Evolve a strategy using genetic algorithms.
        
        Args:
            model_id: Model ID
            strategy_params: Strategy parameters to evolve
            fitness_function: Function to evaluate fitness
            **kwargs: Additional evolution parameters
            
        Returns:
            Improved model ID
        """
        # Get evolution parameters
        population_size = kwargs.get("population_size", self.config.se_population_size)
        mutation_rate = kwargs.get("mutation_rate", self.config.se_mutation_rate)
        crossover_rate = kwargs.get("crossover_rate", self.config.se_crossover_rate)
        tournament_size = kwargs.get("tournament_size", self.config.se_tournament_size)
        generations = kwargs.get("generations", self.config.se_generations)
        elitism_count = kwargs.get("elitism_count", self.config.se_elitism_count)
        
        logger.info(f"Starting strategy evolution for model {model_id}")
        
        # Initialize population if not exists
        if model_id not in self.populations:
            self.populations[model_id] = self._initialize_population(
                strategy_params, population_size
            )
        
        population = self.populations[model_id]
        
        # Evaluate initial population
        for gene in population:
            if gene.fitness == 0.0:  # Only evaluate if not already evaluated
                gene.fitness = fitness_function(model_id, gene.parameters)
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Evolution loop
        for generation in range(generations):
            logger.info(f"Generation {generation+1}/{generations}, best fitness: {population[0].fitness:.4f}")
            
            # Create new generation
            new_population = []
            
            # Elitism - keep best individuals
            new_population.extend(population[:elitism_count])
            
            # Create rest of population through selection, crossover, and mutation
            while len(new_population) < population_size:
                # Selection
                parent1 = self._tournament_selection(population, tournament_size)
                parent2 = self._tournament_selection(population, tournament_size)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # Mutation
                if random.random() < mutation_rate:
                    self._mutate(child1, strategy_params)
                
                if random.random() < mutation_rate:
                    self._mutate(child2, strategy_params)
                
                # Reset fitness
                child1.fitness = 0.0
                child2.fitness = 0.0
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)
            
            # Evaluate new population
            for gene in new_population:
                if gene.fitness == 0.0:  # Only evaluate if not already evaluated
                    gene.fitness = fitness_function(model_id, gene.parameters)
            
            # Replace old population
            population = new_population
            
            # Sort population by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update stored population
        self.populations[model_id] = population
        
        # Create improved model with best strategy
        best_strategy = population[0]
        improved_model_id = self._create_improved_model(model_id, best_strategy)
        
        logger.info(f"Strategy evolution completed for model {model_id}")
        logger.info(f"Best fitness: {best_strategy.fitness:.4f}")
        logger.info(f"Created improved model: {improved_model_id}")
        
        return improved_model_id
    
    def _initialize_population(self, strategy_params: Dict[str, Any], 
                             population_size: int) -> List[StrategyGene]:
        """Initialize a random population."""
        population = []
        
        for _ in range(population_size):
            # Generate random parameters
            params = {}
            for param_name, param_config in strategy_params.items():
                if isinstance(param_config, dict) and "type" in param_config:
                    # Structured parameter definition
                    if param_config["type"] == "categorical":
                        params[param_name] = random.choice(param_config["values"])
                    elif param_config["type"] == "int":
                        low = param_config["low"]
                        high = param_config["high"]
                        step = param_config.get("step", 1)
                        params[param_name] = random.randrange(low, high + 1, step)
                    elif param_config["type"] == "float":
                        low = param_config["low"]
                        high = param_config["high"]
                        if param_config.get("log", False):
                            params[param_name] = 10 ** (random.uniform(np.log10(low), np.log10(high)))
                        else:
                            params[param_name] = random.uniform(low, high)
                    elif param_config["type"] == "bool":
                        params[param_name] = random.choice([True, False])
                else:
                    # Simple parameter range [min, max]
                    if isinstance(param_config, list) and len(param_config) == 2:
                        min_val, max_val = param_config
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = random.randint(min_val, max_val)
                        else:
                            params[param_name] = random.uniform(min_val, max_val)
            
            # Create gene
            gene = StrategyGene(
                gene_id=str(uuid.uuid4()),
                parameters=params,
                fitness=0.0
            )
            
            population.append(gene)
        
        return population
    
    def _tournament_selection(self, population: List[StrategyGene], 
                            tournament_size: int) -> StrategyGene:
        """Select an individual using tournament selection."""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: StrategyGene, parent2: StrategyGene) -> Tuple[StrategyGene, StrategyGene]:
        """Perform crossover between two parents."""
        # Create children with new IDs
        child1 = StrategyGene(
            gene_id=str(uuid.uuid4()),
            parameters={},
            fitness=0.0,
            parent_ids=[parent1.gene_id, parent2.gene_id]
        )
        
        child2 = StrategyGene(
            gene_id=str(uuid.uuid4()),
            parameters={},
            fitness=0.0,
            parent_ids=[parent1.gene_id, parent2.gene_id]
        )
        
        # Get all parameter names
        all_params = set(parent1.parameters.keys()) | set(parent2.parameters.keys())
        
        # Perform crossover for each parameter
        for param_name in all_params:
            if param_name in parent1.parameters and param_name in parent2.parameters:
                # Both parents have this parameter
                if random.random() < 0.5:
                    child1.parameters[param_name] = parent1.parameters[param_name]
                    child2.parameters[param_name] = parent2.parameters[param_name]
                else:
                    child1.parameters[param_name] = parent2.parameters[param_name]
                    child2.parameters[param_name] = parent1.parameters[param_name]
            elif param_name in parent1.parameters:
                # Only parent1 has this parameter
                child1.parameters[param_name] = parent1.parameters[param_name]
                child2.parameters[param_name] = parent1.parameters[param_name]
            else: