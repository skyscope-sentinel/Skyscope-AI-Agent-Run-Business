import os
import sys
import json
import time
import uuid
import math
import random
import logging
import asyncio
import threading
import multiprocessing
import numpy as np
import networkx as nx
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, TypeVar, Generic, Iterable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import heapq
import itertools
import traceback

# Import internal modules
try:
    from agent_manager import AgentManager
    from performance_monitor import PerformanceMonitor
    from database_manager import DatabaseManager
    from live_thinking_rag_system import LiveThinkingRAGSystem
    from enhanced_security_compliance import SecurityManager
except ImportError:
    print("Warning: Some internal modules could not be imported. Running in standalone mode.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/agent_orchestration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_agent_orchestration")

# Constants
MAX_AGENTS = 10000
DEFAULT_PIPELINE_COUNT = 100
DEFAULT_AGENTS_PER_PIPELINE = 100
CONFIG_DIR = Path("config")
MODELS_DIR = Path("models")
PIPELINES_DIR = Path("pipelines")
TASKS_DIR = Path("tasks")
LOGS_DIR = Path("logs")
METRICS_DIR = Path("metrics")

# Ensure directories exist
for directory in [CONFIG_DIR, MODELS_DIR, PIPELINES_DIR, TASKS_DIR, LOGS_DIR, METRICS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Default configuration path
DEFAULT_CONFIG_PATH = CONFIG_DIR / "agent_orchestration_config.json"

# Type variables for generics
T = TypeVar('T')
U = TypeVar('U')

class AgentRole(Enum):
    """Agent roles in the orchestration system."""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    CRITIC = "critic"
    OPTIMIZER = "optimizer"
    INTEGRATOR = "integrator"
    MONITOR = "monitor"
    COMMUNICATOR = "communicator"
    CUSTOM = "custom"

class AgentSpecialization(Enum):
    """Agent specializations within roles."""
    GENERAL = "general"
    BUSINESS = "business"
    FINANCE = "finance"
    MARKETING = "marketing"
    TECHNOLOGY = "technology"
    CREATIVE = "creative"
    LEGAL = "legal"
    OPERATIONS = "operations"
    DATA_SCIENCE = "data_science"
    RESEARCH = "research"
    STRATEGY = "strategy"
    CUSTOM = "custom"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Task status states."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

class PipelineStatus(Enum):
    """Pipeline status states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

class ResourceType(Enum):
    """Resource types for agents."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    TOKENS = "tokens"
    CUSTOM = "custom"

class OrchestrationStrategy(Enum):
    """Orchestration strategies."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_BASED = "priority_based"
    CAPABILITY_MATCHED = "capability_matched"
    SWARM_OPTIMIZED = "swarm_optimized"
    HIERARCHICAL = "hierarchical"
    MARKET_BASED = "market_based"
    CUSTOM = "custom"

class CollaborationPattern(Enum):
    """Collaboration patterns between agents."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    STAR = "star"
    BLACKBOARD = "blackboard"
    MARKET = "market"
    SWARM = "swarm"
    CUSTOM = "custom"

@dataclass
class AgentCapability:
    """Capability of an agent."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "performance_score": self.performance_score,
            "confidence_score": self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            performance_score=data.get("performance_score", 0.0),
            confidence_score=data.get("confidence_score", 0.0)
        )

@dataclass
class AgentProfile:
    """Profile of an agent."""
    id: str
    name: str
    role: AgentRole
    specialization: AgentSpecialization
    capabilities: List[AgentCapability]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)
    collaboration_score: float = 0.0
    reliability_score: float = 0.0
    adaptation_score: float = 0.0
    learning_rate: float = 0.01
    creation_time: int = field(default_factory=lambda: int(time.time()))
    last_active: int = field(default_factory=lambda: int(time.time()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role.value,
            "specialization": self.specialization.value,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "performance_metrics": self.performance_metrics,
            "resource_usage": {rt.value: usage for rt, usage in self.resource_usage.items()},
            "collaboration_score": self.collaboration_score,
            "reliability_score": self.reliability_score,
            "adaptation_score": self.adaptation_score,
            "learning_rate": self.learning_rate,
            "creation_time": self.creation_time,
            "last_active": self.last_active,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentProfile':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            role=AgentRole(data["role"]),
            specialization=AgentSpecialization(data["specialization"]),
            capabilities=[AgentCapability.from_dict(cap) for cap in data["capabilities"]],
            performance_metrics=data.get("performance_metrics", {}),
            resource_usage={ResourceType(rt): usage for rt, usage in data.get("resource_usage", {}).items()},
            collaboration_score=data.get("collaboration_score", 0.0),
            reliability_score=data.get("reliability_score", 0.0),
            adaptation_score=data.get("adaptation_score", 0.0),
            learning_rate=data.get("learning_rate", 0.01),
            creation_time=data.get("creation_time", int(time.time())),
            last_active=data.get("last_active", int(time.time())),
            metadata=data.get("metadata", {})
        )
    
    def calculate_capability_score(self, capability_name: str) -> float:
        """Calculate score for a specific capability."""
        for cap in self.capabilities:
            if cap.name == capability_name:
                return (cap.performance_score * 0.7) + (cap.confidence_score * 0.3)
        return 0.0
    
    def calculate_overall_score(self) -> float:
        """Calculate overall agent score."""
        capability_scores = [self.calculate_capability_score(cap.name) for cap in self.capabilities]
        avg_capability_score = sum(capability_scores) / len(capability_scores) if capability_scores else 0
        
        return (
            avg_capability_score * 0.4 +
            self.collaboration_score * 0.2 +
            self.reliability_score * 0.3 +
            self.adaptation_score * 0.1
        )
    
    def update_metrics(self, task_result: Dict[str, Any]) -> None:
        """Update agent metrics based on task result."""
        # Update last active time
        self.last_active = int(time.time())
        
        # Update performance metrics
        if "execution_time" in task_result:
            if "avg_execution_time" not in self.performance_metrics:
                self.performance_metrics["avg_execution_time"] = task_result["execution_time"]
            else:
                self.performance_metrics["avg_execution_time"] = (
                    self.performance_metrics["avg_execution_time"] * 0.9 +
                    task_result["execution_time"] * 0.1
                )
        
        if "success" in task_result:
            if "success_rate" not in self.performance_metrics:
                self.performance_metrics["success_rate"] = 1.0 if task_result["success"] else 0.0
            else:
                self.performance_metrics["success_rate"] = (
                    self.performance_metrics["success_rate"] * 0.95 +
                    (1.0 if task_result["success"] else 0.0) * 0.05
                )
        
        if "quality_score" in task_result:
            if "avg_quality_score" not in self.performance_metrics:
                self.performance_metrics["avg_quality_score"] = task_result["quality_score"]
            else:
                self.performance_metrics["avg_quality_score"] = (
                    self.performance_metrics["avg_quality_score"] * 0.9 +
                    task_result["quality_score"] * 0.1
                )
        
        # Update reliability score
        if "success" in task_result:
            self.reliability_score = self.reliability_score * 0.95 + (1.0 if task_result["success"] else 0.0) * 0.05
        
        # Update collaboration score if applicable
        if "collaboration_quality" in task_result:
            self.collaboration_score = self.collaboration_score * 0.9 + task_result["collaboration_quality"] * 0.1
        
        # Update adaptation score
        if "adaptation_score" in task_result:
            self.adaptation_score = self.adaptation_score * 0.9 + task_result["adaptation_score"] * 0.1
        
        # Update capability scores
        if "capability_metrics" in task_result:
            for cap_name, metrics in task_result["capability_metrics"].items():
                for cap in self.capabilities:
                    if cap.name == cap_name:
                        if "performance" in metrics:
                            cap.performance_score = cap.performance_score * 0.9 + metrics["performance"] * 0.1
                        if "confidence" in metrics:
                            cap.confidence_score = cap.confidence_score * 0.9 + metrics["confidence"] * 0.1
        
        # Update resource usage
        if "resource_usage" in task_result:
            for resource_type_str, usage in task_result["resource_usage"].items():
                try:
                    resource_type = ResourceType(resource_type_str)
                    if resource_type not in self.resource_usage:
                        self.resource_usage[resource_type] = usage
                    else:
                        self.resource_usage[resource_type] = (
                            self.resource_usage[resource_type] * 0.9 + usage * 0.1
                        )
                except ValueError:
                    logger.warning(f"Unknown resource type: {resource_type_str}")

@dataclass
class Task:
    """Task to be executed by agents."""
    id: str
    name: str
    description: str
    priority: TaskPriority
    required_capabilities: List[str]
    input_data: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: Optional[str] = None
    assigned_pipeline_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # IDs of tasks that must complete before this one
    dependents: List[str] = field(default_factory=list)    # IDs of tasks that depend on this one
    creation_time: int = field(default_factory=lambda: int(time.time()))
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    estimated_duration: Optional[float] = None
    actual_duration: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority.value,
            "required_capabilities": self.required_capabilities,
            "input_data": self.input_data,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "assigned_pipeline_id": self.assigned_pipeline_id,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "creation_time": self.creation_time,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            priority=TaskPriority(data["priority"]),
            required_capabilities=data["required_capabilities"],
            input_data=data["input_data"],
            status=TaskStatus(data["status"]),
            assigned_agent_id=data.get("assigned_agent_id"),
            assigned_pipeline_id=data.get("assigned_pipeline_id"),
            dependencies=data.get("dependencies", []),
            dependents=data.get("dependents", []),
            creation_time=data.get("creation_time", int(time.time())),
            start_time=data.get("start_time"),
            completion_time=data.get("completion_time"),
            estimated_duration=data.get("estimated_duration"),
            actual_duration=data.get("actual_duration"),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            metadata=data.get("metadata", {})
        )
    
    def is_ready(self, completed_task_ids: Set[str]) -> bool:
        """Check if task is ready to be executed."""
        return all(dep_id in completed_task_ids for dep_id in self.dependencies)
    
    def can_be_retried(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

@dataclass
class Pipeline:
    """Pipeline of agents and tasks."""
    id: str
    name: str
    description: str
    agent_ids: List[str]
    status: PipelineStatus = PipelineStatus.INITIALIZING
    task_ids: List[str] = field(default_factory=list)
    completed_task_ids: Set[str] = field(default_factory=set)
    failed_task_ids: Set[str] = field(default_factory=set)
    creation_time: int = field(default_factory=lambda: int(time.time()))
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    coordinator_agent_id: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_allocation: Dict[ResourceType, float] = field(default_factory=dict)
    collaboration_pattern: CollaborationPattern = CollaborationPattern.MESH
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agent_ids": self.agent_ids,
            "status": self.status.value,
            "task_ids": self.task_ids,
            "completed_task_ids": list(self.completed_task_ids),
            "failed_task_ids": list(self.failed_task_ids),
            "creation_time": self.creation_time,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "coordinator_agent_id": self.coordinator_agent_id,
            "performance_metrics": self.performance_metrics,
            "resource_allocation": {rt.value: alloc for rt, alloc in self.resource_allocation.items()},
            "collaboration_pattern": self.collaboration_pattern.value,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            agent_ids=data["agent_ids"],
            status=PipelineStatus(data["status"]),
            task_ids=data.get("task_ids", []),
            completed_task_ids=set(data.get("completed_task_ids", [])),
            failed_task_ids=set(data.get("failed_task_ids", [])),
            creation_time=data.get("creation_time", int(time.time())),
            start_time=data.get("start_time"),
            completion_time=data.get("completion_time"),
            coordinator_agent_id=data.get("coordinator_agent_id"),
            performance_metrics=data.get("performance_metrics", {}),
            resource_allocation={ResourceType(rt): alloc for rt, alloc in data.get("resource_allocation", {}).items()},
            collaboration_pattern=CollaborationPattern(data.get("collaboration_pattern", CollaborationPattern.MESH.value)),
            metadata=data.get("metadata", {})
        )
    
    def calculate_completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if not self.task_ids:
            return 0.0
        return len(self.completed_task_ids) / len(self.task_ids) * 100
    
    def update_metrics(self, task_result: Dict[str, Any]) -> None:
        """Update pipeline metrics based on task result."""
        # Update performance metrics
        if "execution_time" in task_result:
            if "avg_task_execution_time" not in self.performance_metrics:
                self.performance_metrics["avg_task_execution_time"] = task_result["execution_time"]
            else:
                self.performance_metrics["avg_task_execution_time"] = (
                    self.performance_metrics["avg_task_execution_time"] * 0.95 +
                    task_result["execution_time"] * 0.05
                )
        
        # Update success rate
        if "success" in task_result:
            success_count = self.performance_metrics.get("success_count", 0)
            total_count = self.performance_metrics.get("total_count", 0)
            
            success_count += 1 if task_result["success"] else 0
            total_count += 1
            
            self.performance_metrics["success_count"] = success_count
            self.performance_metrics["total_count"] = total_count
            self.performance_metrics["success_rate"] = success_count / total_count if total_count > 0 else 0.0
        
        # Update resource usage
        if "resource_usage" in task_result:
            for resource_type_str, usage in task_result["resource_usage"].items():
                try:
                    resource_type = ResourceType(resource_type_str)
                    if resource_type not in self.resource_allocation:
                        self.resource_allocation[resource_type] = usage
                    else:
                        self.resource_allocation[resource_type] += usage
                except ValueError:
                    logger.warning(f"Unknown resource type: {resource_type_str}")
        
        # Check if pipeline is complete
        if len(self.completed_task_ids) + len(self.failed_task_ids) >= len(self.task_ids) and self.task_ids:
            if not self.failed_task_ids:
                self.status = PipelineStatus.COMPLETED
                self.completion_time = int(time.time())
            else:
                self.status = PipelineStatus.FAILED

@dataclass
class AgentOrchestrationConfig:
    """Configuration for agent orchestration."""
    max_agents: int = MAX_AGENTS
    pipeline_count: int = DEFAULT_PIPELINE_COUNT
    agents_per_pipeline: int = DEFAULT_AGENTS_PER_PIPELINE
    orchestration_strategy: OrchestrationStrategy = OrchestrationStrategy.SWARM_OPTIMIZED
    default_collaboration_pattern: CollaborationPattern = CollaborationPattern.MESH
    task_assignment_batch_size: int = 10
    resource_allocation_strategy: str = "dynamic"
    load_balancing_threshold: float = 0.8
    agent_specialization_ratio: Dict[AgentSpecialization, float] = field(default_factory=lambda: {
        AgentSpecialization.GENERAL: 0.3,
        AgentSpecialization.BUSINESS: 0.1,
        AgentSpecialization.FINANCE: 0.1,
        AgentSpecialization.MARKETING: 0.1,
        AgentSpecialization.TECHNOLOGY: 0.1,
        AgentSpecialization.CREATIVE: 0.05,
        AgentSpecialization.LEGAL: 0.05,
        AgentSpecialization.OPERATIONS: 0.05,
        AgentSpecialization.DATA_SCIENCE: 0.1,
        AgentSpecialization.RESEARCH: 0.05
    })
    agent_role_ratio: Dict[AgentRole, float] = field(default_factory=lambda: {
        AgentRole.COORDINATOR: 0.05,
        AgentRole.EXECUTOR: 0.5,
        AgentRole.ANALYZER: 0.1,
        AgentRole.RESEARCHER: 0.1,
        AgentRole.PLANNER: 0.05,
        AgentRole.CRITIC: 0.05,
        AgentRole.OPTIMIZER: 0.05,
        AgentRole.INTEGRATOR: 0.05,
        AgentRole.MONITOR: 0.03,
        AgentRole.COMMUNICATOR: 0.02
    })
    swarm_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "cohesion_factor": 0.7,
        "separation_factor": 0.5,
        "alignment_factor": 0.8,
        "exploration_rate": 0.2,
        "exploitation_rate": 0.8,
        "adaptation_rate": 0.05
    })
    resource_limits: Dict[ResourceType, float] = field(default_factory=lambda: {
        ResourceType.CPU: 0.8,
        ResourceType.MEMORY: 0.8,
        ResourceType.GPU: 0.9,
        ResourceType.NETWORK: 0.7,
        ResourceType.API_CALLS: 5000,
        ResourceType.TOKENS: 1000000
    })
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_success_rate": 0.7,
        "max_response_time": 5.0,
        "min_quality_score": 0.6
    })
    scaling_parameters: Dict[str, Any] = field(default_factory=lambda: {
        "auto_scaling": True,
        "min_pipeline_count": 10,
        "max_pipeline_count": 200,
        "scaling_cooldown": 300,  # seconds
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.3
    })
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_agents": self.max_agents,
            "pipeline_count": self.pipeline_count,
            "agents_per_pipeline": self.agents_per_pipeline,
            "orchestration_strategy": self.orchestration_strategy.value,
            "default_collaboration_pattern": self.default_collaboration_pattern.value,
            "task_assignment_batch_size": self.task_assignment_batch_size,
            "resource_allocation_strategy": self.resource_allocation_strategy,
            "load_balancing_threshold": self.load_balancing_threshold,
            "agent_specialization_ratio": {spec.value: ratio for spec, ratio in self.agent_specialization_ratio.items()},
            "agent_role_ratio": {role.value: ratio for role, ratio in self.agent_role_ratio.items()},
            "swarm_parameters": self.swarm_parameters,
            "resource_limits": {rt.value: limit for rt, limit in self.resource_limits.items()},
            "performance_thresholds": self.performance_thresholds,
            "scaling_parameters": self.scaling_parameters,
            "custom_settings": self.custom_settings
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentOrchestrationConfig':
        """Create from dictionary."""
        config = cls(
            max_agents=data.get("max_agents", MAX_AGENTS),
            pipeline_count=data.get("pipeline_count", DEFAULT_PIPELINE_COUNT),
            agents_per_pipeline=data.get("agents_per_pipeline", DEFAULT_AGENTS_PER_PIPELINE),
            orchestration_strategy=OrchestrationStrategy(data.get("orchestration_strategy", OrchestrationStrategy.SWARM_OPTIMIZED.value)),
            default_collaboration_pattern=CollaborationPattern(data.get("default_collaboration_pattern", CollaborationPattern.MESH.value)),
            task_assignment_batch_size=data.get("task_assignment_batch_size", 10),
            resource_allocation_strategy=data.get("resource_allocation_strategy", "dynamic"),
            load_balancing_threshold=data.get("load_balancing_threshold", 0.8)
        )
        
        # Handle agent specialization ratio
        if "agent_specialization_ratio" in data:
            config.agent_specialization_ratio = {
                AgentSpecialization(spec): ratio 
                for spec, ratio in data["agent_specialization_ratio"].items()
            }
        
        # Handle agent role ratio
        if "agent_role_ratio" in data:
            config.agent_role_ratio = {
                AgentRole(role): ratio 
                for role, ratio in data["agent_role_ratio"].items()
            }
        
        # Handle swarm parameters
        if "swarm_parameters" in data:
            config.swarm_parameters = data["swarm_parameters"]
        
        # Handle resource limits
        if "resource_limits" in data:
            config.resource_limits = {
                ResourceType(rt): limit 
                for rt, limit in data["resource_limits"].items()
            }
        
        # Handle performance thresholds
        if "performance_thresholds" in data:
            config.performance_thresholds = data["performance_thresholds"]
        
        # Handle scaling parameters
        if "scaling_parameters" in data:
            config.scaling_parameters = data["scaling_parameters"]
        
        # Handle custom settings
        if "custom_settings" in data:
            config.custom_settings = data["custom_settings"]
        
        return config
    
    def save(self, filepath: Path = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Agent orchestration configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving agent orchestration configuration: {e}")
    
    @classmethod
    def load(cls, filepath: Path = DEFAULT_CONFIG_PATH) -> 'AgentOrchestrationConfig':
        """Load configuration from file."""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Agent orchestration configuration loaded from {filepath}")
                return cls.from_dict(data)
            else:
                logger.info(f"Configuration file {filepath} not found, using defaults")
                return cls()
        except Exception as e:
            logger.error(f"Error loading agent orchestration configuration: {e}")
            return cls()

class PriorityQueue(Generic[T]):
    """Priority queue implementation."""
    
    def __init__(self):
        self._queue = []
        self._index = 0
    
    def push(self, item: T, priority: int):
        """Push an item onto the queue."""
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    
    def pop(self) -> T:
        """Pop the highest priority item."""
        return heapq.heappop(self._queue)[2]
    
    def peek(self) -> Optional[T]:
        """Peek at the highest priority item without removing it."""
        if not self._queue:
            return None
        return self._queue[0][2]
    
    def __len__(self) -> int:
        """Get the number of items in the queue."""
        return len(self._queue)
    
    def __bool__(self) -> bool:
        """Check if the queue is not empty."""
        return bool(self._queue)

class TaskQueue:
    """Queue for managing tasks."""
    
    def __init__(self):
        """Initialize the task queue."""
        self.queues = {priority: deque() for priority in TaskPriority}
        self.task_map = {}  # Map of task ID to (priority, index in queue)
    
    def add_task(self, task: Task) -> None:
        """Add a task to the queue."""
        self.queues[task.priority].append(task)
        self.task_map[task.id] = (task.priority, len(self.queues[task.priority]) - 1)
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task based on priority."""
        for priority in TaskPriority:
            if self.queues[priority]:
                task = self.queues[priority].popleft()
                if task.id in self.task_map:
                    del self.task_map[task.id]
                return task
        return None
    
    def get_tasks_batch(self, batch_size: int) -> List[Task]:
        """Get a batch of tasks based on priority."""
        tasks = []
        for _ in range(batch_size):
            task = self.get_next_task()
            if task:
                tasks.append(task)
            else:
                break
        return tasks
    
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the queue."""
        if task_id not in self.task_map:
            return False
        
        priority, _ = self.task_map[task_id]
        for i, task in enumerate(self.queues[priority]):
            if task.id == task_id:
                del self.queues[priority][i]
                del self.task_map[task_id]
                return True
        
        return False
    
    def update_task_priority(self, task_id: str, new_priority: TaskPriority) -> bool:
        """Update the priority of a task."""
        if task_id not in self.task_map:
            return False
        
        old_priority, _ = self.task_map[task_id]
        for i, task in enumerate(self.queues[old_priority]):
            if task.id == task_id:
                task.priority = new_priority
                del self.queues[old_priority][i]
                self.queues[new_priority].append(task)
                self.task_map[task_id] = (new_priority, len(self.queues[new_priority]) - 1)
                return True
        
        return False
    
    def __len__(self) -> int:
        """Get the total number of tasks in the queue."""
        return sum(len(queue) for queue in self.queues.values())
    
    def __bool__(self) -> bool:
        """Check if the queue is not empty."""
        return any(queue for queue in self.queues.values())
    
    def get_tasks_by_priority(self) -> Dict[TaskPriority, List[Task]]:
        """Get all tasks grouped by priority."""
        return {priority: list(queue) for priority, queue in self.queues.items()}

class AgentPool:
    """Pool of agents for task execution."""
    
    def __init__(self, config: AgentOrchestrationConfig):
        """Initialize the agent pool."""
        self.config = config
        self.agents = {}  # Map of agent ID to AgentProfile
        self.available_agents = set()  # Set of available agent IDs
        self.busy_agents = set()  # Set of busy agent IDs
        self.agent_manager = None
        self.lock = threading.RLock()
        
        # Try to initialize agent manager
        try:
            from agent_manager import AgentManager
            self.agent_manager = AgentManager()
        except ImportError:
            logger.warning("AgentManager not available. Some features will be limited.")
    
    def add_agent(self, agent: AgentProfile) -> None:
        """Add an agent to the pool."""
        with self.lock:
            self.agents[agent.id] = agent
            self.available_agents.add(agent.id)
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the pool."""
        with self.lock:
            if agent_id not in self.agents:
                return False
            
            if agent_id in self.available_agents:
                self.available_agents.remove(agent_id)
            
            if agent_id in self.busy_agents:
                self.busy_agents.remove(agent_id)
            
            del self.agents[agent_id]
            return True
    
    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
    
    def mark_agent_busy(self, agent_id: str) -> bool:
        """Mark an agent as busy."""
        with self.lock:
            if agent_id not in self.agents or agent_id not in self.available_agents:
                return False
            
            self.available_agents.remove(agent_id)
            self.busy_agents.add(agent_id)
            return True
    
    def mark_agent_available(self, agent_id: str) -> bool:
        """Mark an agent as available."""
        with self.lock:
            if agent_id not in self.agents or agent_id not in self.busy_agents:
                return False
            
            self.busy_agents.remove(agent_id)
            self.available_agents.add(agent_id)
            return True
    
    def get_available_agents(self) -> List[AgentProfile]:
        """Get all available agents."""
        with self.lock:
            return [self.agents[agent_id] for agent_id in self.available_agents]
    
    def get_busy_agents(self) -> List[AgentProfile]:
        """Get all busy agents."""
        with self.lock:
            return [self.agents[agent_id] for agent_id in self.busy_agents]
    
    def get_agents_by_role(self, role: AgentRole) -> List[AgentProfile]:
        """Get agents by role."""
        return [agent for agent in self.agents.values() if agent.role == role]
    
    def get_agents_by_specialization(self, specialization: AgentSpecialization) -> List[AgentProfile]:
        """Get agents by specialization."""
        return [agent for agent in self.agents.values() if agent.specialization == specialization]
    
    def get_agents_by_capability(self, capability_name: str) -> List[AgentProfile]:
        """Get agents by capability."""
        return [
            agent for agent in self.agents.values() 
            if any(cap.name == capability_name for cap in agent.capabilities)
        ]
    
    def find_best_agent_for_task(self, task: Task) -> Optional[str]:
        """Find the best available agent for a task."""
        with self.lock:
            available_agents = self.get_available_agents()
            if not available_agents:
                return None
            
            # Filter agents by required capabilities
            capable_agents = []
            for agent in available_agents:
                agent_capabilities = {cap.name for cap in agent.capabilities}
                if all(req_cap in agent_capabilities for req_cap in task.required_capabilities):
                    capable_agents.append(agent)
            
            if not capable_agents:
                return None
            
            # Score agents based on capabilities and other factors
            agent_scores = []
            for agent in capable_agents:
                # Calculate capability score
                capability_scores = []
                for req_cap in task.required_capabilities:
                    capability_scores.append(agent.calculate_capability_score(req_cap))
                
                avg_capability_score = sum(capability_scores) / len(capability_scores) if capability_scores else 0
                
                # Calculate overall score
                score = (
                    avg_capability_score * 0.6 +
                    agent.reliability_score * 0.3 +
                    agent.adaptation_score * 0.1
                )
                
                agent_scores.append((agent.id, score))
            
            # Return the agent with the highest score
            if agent_scores:
                return max(agent_scores, key=lambda x: x[1])[0]
            
            return None
    
    def create_agents(self, count: int) -> List[str]:
        """Create new agents."""
        agent_ids = []
        
        # Calculate counts for each role and specialization
        role_counts = {role: int(count * ratio) for role, ratio in self.config.agent_role_ratio.items()}
        spec_counts = {spec: int(count * ratio) for spec, ratio in self.config.agent_specialization_ratio.items()}
        
        # Ensure we create exactly 'count' agents
        total_role_count = sum(role_counts.values())
        if total_role_count < count:
            # Add remaining to EXECUTOR role
            role_counts[AgentRole.EXECUTOR] += count - total_role_count
        
        total_spec_count = sum(spec_counts.values())
        if total_spec_count < count:
            # Add remaining to GENERAL specialization
            spec_counts[AgentSpecialization.GENERAL] += count - total_spec_count
        
        # Create agents with different roles and specializations
        roles = list(role_counts.keys())
        specializations = list(spec_counts.keys())
        
        for i in range(count):
            # Select role and specialization
            role = None
            for r in roles:
                if role_counts[r] > 0:
                    role = r
                    role_counts[r] -= 1
                    break
            
            if role is None:
                role = AgentRole.EXECUTOR
            
            specialization = None
            for s in specializations:
                if spec_counts[s] > 0:
                    specialization = s
                    spec_counts[s] -= 1
                    break
            
            if specialization is None:
                specialization = AgentSpecialization.GENERAL
            
            # Create agent profile
            agent_id = str(uuid.uuid4())
            agent_name = f"Agent-{role.value.capitalize()}-{specialization.value.capitalize()}-{agent_id[:8]}"
            
            # Create capabilities based on role and specialization
            capabilities = self._generate_capabilities_for_agent(role, specialization)
            
            agent = AgentProfile(
                id=agent_id,
                name=agent_name,
                role=role,
                specialization=specialization,
                capabilities=capabilities,
                reliability_score=0.7 + random.random() * 0.3,  # Initial reliability between 0.7 and 1.0
                adaptation_score=0.5 + random.random() * 0.5    # Initial adaptation between 0.5 and 1.0
            )
            
            self.add_agent(agent)
            agent_ids.append(agent_id)
        
        return agent_ids
    
    def _generate_capabilities_for_agent(self, role: AgentRole, specialization: AgentSpecialization) -> List[AgentCapability]:
        """Generate capabilities for an agent based on role and specialization."""
        capabilities = []
        
        # Add role-specific capabilities
        if role == AgentRole.COORDINATOR:
            capabilities.extend([
                AgentCapability(
                    name="task_coordination",
                    description="Coordinate tasks among multiple agents",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="resource_allocation",
                    description="Allocate resources efficiently",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.EXECUTOR:
            capabilities.extend([
                AgentCapability(
                    name="task_execution",
                    description="Execute tasks efficiently",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="problem_solving",
                    description="Solve problems efficiently",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.ANALYZER:
            capabilities.extend([
                AgentCapability(
                    name="data_analysis",
                    description="Analyze data and extract insights",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="pattern_recognition",
                    description="Recognize patterns in data",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.RESEARCHER:
            capabilities.extend([
                AgentCapability(
                    name="information_gathering",
                    description="Gather information from various sources",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="knowledge_synthesis",
                    description="Synthesize knowledge from multiple sources",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.PLANNER:
            capabilities.extend([
                AgentCapability(
                    name="strategic_planning",
                    description="Create strategic plans",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="resource_planning",
                    description="Plan resource allocation efficiently",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.CRITIC:
            capabilities.extend([
                AgentCapability(
                    name="quality_assessment",
                    description="Assess quality of work",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="error_detection",
                    description="Detect errors in work",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.OPTIMIZER:
            capabilities.extend([
                AgentCapability(
                    name="performance_optimization",
                    description="Optimize performance of systems",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="efficiency_improvement",
                    description="Improve efficiency of processes",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.INTEGRATOR:
            capabilities.extend([
                AgentCapability(
                    name="system_integration",
                    description="Integrate different systems",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="compatibility_management",
                    description="Manage compatibility between systems",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.MONITOR:
            capabilities.extend([
                AgentCapability(
                    name="system_monitoring",
                    description="Monitor system performance",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="anomaly_detection",
                    description="Detect anomalies in system behavior",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif role == AgentRole.COMMUNICATOR:
            capabilities.extend([
                AgentCapability(
                    name="information_sharing",
                    description="Share information efficiently",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="clear_communication",
                    description="Communicate clearly and effectively",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        
        # Add specialization-specific capabilities
        if specialization == AgentSpecialization.BUSINESS:
            capabilities.extend([
                AgentCapability(
                    name="business_analysis",
                    description="Analyze business processes and metrics",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="market_research",
                    description="Research market trends and opportunities",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.FINANCE:
            capabilities.extend([
                AgentCapability(
                    name="financial_analysis",
                    description="Analyze financial data and metrics",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="investment_strategy",
                    description="Develop investment strategies",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.MARKETING:
            capabilities.extend([
                AgentCapability(
                    name="marketing_strategy",
                    description="Develop marketing strategies",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="content_creation",
                    description="Create marketing content",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.TECHNOLOGY:
            capabilities.extend([
                AgentCapability(
                    name="software_development",
                    description="Develop software solutions",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="system_architecture",
                    description="Design system architectures",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.CREATIVE:
            capabilities.extend([
                AgentCapability(
                    name="creative_design",
                    description="Create creative designs",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="content_writing",
                    description="Write creative content",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.LEGAL:
            capabilities.extend([
                AgentCapability(
                    name="legal_analysis",
                    description="Analyze legal documents and issues",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="compliance_management",
                    description="Manage compliance with regulations",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.OPERATIONS:
            capabilities.extend([
                AgentCapability(
                    name="process_optimization",
                    description="Optimize operational processes",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="logistics_management",
                    description="Manage logistics operations",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.DATA_SCIENCE:
            capabilities.extend([
                AgentCapability(
                    name="data_mining",
                    description="Extract insights from large datasets",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="predictive_modeling",
                    description="Create predictive models",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        elif specialization == AgentSpecialization.RESEARCH:
            capabilities.extend([
                AgentCapability(
                    name="academic_research",
                    description="Conduct academic research",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                ),
                AgentCapability(
                    name="literature_review",
                    description="Review academic literature",
                    performance_score=0.7 + random.random() * 0.3,
                    confidence_score=0.7 + random.random() * 0.3
                )
            ])
        
        # Add general capabilities
        capabilities.extend([
            AgentCapability(
                name="critical_thinking",
                description="Think critically about problems",
                performance_score=0.6 + random.random() * 0.4,
                confidence_score=0.6 + random.random() * 0.4
            ),
            AgentCapability(
                name="decision_making",
                description="Make decisions based on available information",
                performance_score=0.6 + random.random() * 0.4,
                confidence_score=0.6 + random.random() * 0.4
            )
        ])
        
        return capabilities

class CollaborationNetwork:
    """Network for agent collaboration."""
    
    def __init__(self, pattern: CollaborationPattern = CollaborationPattern.MESH):
        """Initialize the collaboration network."""
        self.pattern = pattern
        self.graph = nx.DiGraph()
        self.agent_positions = {}  # For swarm pattern
        self.blackboard = {}  # For blackboard pattern
        self.market = {}  # For market pattern
    
    def add_agent(self, agent_id: str) -> None:
        """Add an agent to the network."""
        self.graph.add_node(agent_id)
        
        # Initialize agent position for swarm pattern
        if self.pattern == CollaborationPattern.SWARM:
            self.agent_positions[agent_id] = {
                'position': np.random.rand(2),  # 2D position
                'velocity': np.random.rand(2) * 0.1 - 0.05,  # Initial velocity
                'best_position': None,
                'best_score': 0.0
            }
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the network."""
        if agent_id in self.graph:
            self.graph.remove_node(agent_id)
        
        # Remove agent position for swarm pattern
        if agent_id in self.agent_positions:
            del self.agent_positions[agent_id]
    
    def add_connection(self, from_agent_id: str, to_agent_id: str, weight: float = 1.0) -> None:
        """Add a connection between agents."""
        self.graph.add_edge(from_agent_id, to_agent_id, weight=weight)
    
    def remove_connection(self, from_agent_id: str, to_agent_id: str) -> None:
        """Remove a connection between agents."""
        if self.graph.has_edge(from_agent_id, to_agent_id):
            self.graph.remove_edge(from_agent_id, to_agent_id)
    
    def get_connections(self, agent_id: str) -> List[Tuple[str, float]]:
        """Get all connections for an agent."""
        if agent_id not in self.graph:
            return []
        
        return [(neighbor, self.graph[agent_id][neighbor]['weight']) for neighbor in self.graph.neighbors(agent_id)]
    
    def get_strongest_connections(self, agent_id: str, limit: int = 5) -> List[Tuple[str, float]]:
        """Get the strongest connections for an agent."""
        connections = self.get_connections(agent_id)
        return sorted(connections, key=lambda x: x[1], reverse=True)[:limit]
    
    def update_connection_weight(self, from_agent_id: str, to_agent_id: str, weight: float) -> None:
        """Update the weight of a connection."""
        if self.graph.has_edge(from_agent_id, to_agent_id):
            self.graph[from_agent_id][to_agent_id]['weight'] = weight
    
    def get_central_agents(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get the most central agents in the network."""
        if not self.graph:
            return []
        
        centrality = nx.eigenvector_centrality_numpy(self.graph, weight='weight')
        return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:limit]
    
    def get_communities(self) -> List[Set[str]]:
        """Get communities in the network."""
        if not self.graph:
            return []
        
        # Convert to undirected graph for community detection
        undirected = self.graph.to_undirected()
        communities = list(nx.community.greedy_modularity_communities(undirected, weight='weight'))
        return [set(community) for community in communities]
    
    def configure_pattern(self, pattern: CollaborationPattern, agents: List[str]) -> None:
        """Configure the network according to the specified pattern."""
        self.pattern = pattern
        
        # Clear existing graph
        self.graph.clear()
        
        # Add all agents
        for agent_id in agents:
            self.add_agent(agent_id)
        
        # Configure based on pattern
        if pattern == CollaborationPattern.SEQUENTIAL:
            # Connect agents in a chain
            for i in range(len(agents) - 1):
                self.add_connection(agents[i], agents[i + 1], 1.0)
        
        elif pattern == CollaborationPattern.PARALLEL:
            # No connections, agents work independently
            pass
        
        elif pattern == CollaborationPattern.HIERARCHICAL:
            # Create a tree structure
            if len(agents) > 1:
                # Use the first agent as root
                root = agents[0]
                
                # Connect root to all other agents
                for agent_id in agents[1:]:
                    self.add_connection(root, agent_id, 1.0)
                
                # Create additional layers if there are enough agents
                if len(agents) > 10:
                    # Second layer
                    second_layer = agents[1:min(10, len(agents))]
                    remaining = agents[10:]
                    
                    # Distribute remaining agents among second layer nodes
                    for i, agent_id in enumerate(remaining):
                        parent = second_layer[i % len(second_layer)]
                        self.add_connection(parent, agent_id, 1.0)
        
        elif pattern == CollaborationPattern.MESH:
            # Connect all agents to all other agents
            for i in range(len(agents)):
                for j in range(len(agents)):
                    if i != j:
                        self.add_connection(agents[i], agents[j], 1.0)
        
        elif pattern == CollaborationPattern.STAR:
            # Use the first agent as center
            if len(agents) > 1:
                center = agents[0]
                
                # Connect center to all other agents
                for agent_id in agents[1:]:
                    self.add_connection(center, agent_id, 1.0)
                    self.add_connection(agent_id, center, 1.0)
        
        elif pattern == CollaborationPattern.BLACKBOARD:
            # Initialize blackboard
            self.blackboard = {}
            
            # No direct connections in blackboard pattern
            pass
        
        elif pattern == CollaborationPattern.MARKET:
            # Initialize market
            self.market = {
                'tasks': {},
                'bids': {},
                'assignments': {}
            }
            
            # No direct connections in market pattern
            pass
        
        elif pattern == CollaborationPattern.SWARM:
            # Initialize agent positions
            self.agent_positions = {}
            for agent_id in agents:
                self.agent_positions[agent_id] = {
                    'position': np.random.rand(2),  # 2D position
                    'velocity': np.random.rand(2) * 0.1 - 0.05,  # Initial velocity
                    'best_position': None,
                    'best_score': 0.0
                }
    
    def update_swarm(self, agent_scores: Dict[str, float], swarm_params: Dict[str, float]) -> None:
        """Update swarm positions based on agent scores."""
        if self.pattern != CollaborationPattern.SWARM:
            return
        
        # Extract parameters
        cohesion_factor = swarm_params.get('cohesion_factor', 0.7)
        separation_factor = swarm_params.get('separation_factor', 0.5)
        alignment_factor = swarm_params.get('alignment_factor', 0.8)
        
        # Calculate swarm center
        positions = np.array([data['position'] for data in self.agent_positions.values()])
        swarm_center = np.mean(positions, axis=0) if len(positions) > 0 else np.zeros(2)
        
        # Update each agent
        for agent_id, score in agent_scores.items():
            if agent_id not in self.agent_positions:
                continue
            
            agent_data = self.agent_positions[agent_id]
            
            # Update best position if current score is better
            if agent_data['best_position'] is None or score > agent_data['best_score']:
                agent_data['best_position'] = agent_data['position'].copy()
                agent_data['best_score'] = score
            
            # Calculate cohesion vector (towards swarm center)
            cohesion = (swarm_center - agent_data['position']) * cohesion_factor
            
            # Calculate separation vector (away from close neighbors)
            separation = np.zeros(2)
            for other_id, other_data in self.agent_positions.items():
                if other_id != agent_id:
                    diff = agent_data['position'] - other_data['position']
                    dist = np.linalg.norm(diff)
                    if dist < 0.1 and dist > 0:  # Avoid division by zero
                        separation += diff / (dist ** 2)
            separation *= separation_factor
            
            # Calculate alignment vector (average velocity of neighbors)
            alignment = np.zeros(2)
            neighbor_count = 0
            for other_id, other_data in self.agent_positions.items():
                if other_id != agent_id:
                    dist = np.linalg.norm(agent_data['position'] - other_data['position'])
                    if dist < 0.2:  # Only consider close neighbors
                        alignment += other_data['velocity']
                        neighbor_count += 1
            if neighbor_count > 0:
                alignment /= neighbor_count
            alignment *= alignment_factor
            
            # Update velocity
            agent_data['velocity'] = (
                agent_data['velocity'] * 0.9 +  # Inertia
                cohesion +
                separation +
                alignment
            )
            
            # Limit velocity
            max_velocity = 0.1
            velocity_norm = np.linalg.norm(agent_data['velocity'])
            if velocity_norm > max_velocity:
                agent_data['velocity'] = agent_data['velocity'] / velocity_norm * max_velocity
            
            # Update position
            agent_data['position'] += agent_data['velocity']
            
            # Ensure position stays within bounds [0, 1]
            agent_data['position'] = np.clip(agent_data['position'], 0, 1)
    
    def post_to_blackboard(self, agent_id: str, key: str, value: Any) -> None:
        """Post information to the blackboard."""
        if self.pattern != CollaborationPattern.BLACKBOARD:
            return
        
        if key not in self.blackboard:
            self.blackboard[key] = []
        
        self.blackboard[key].append({
            'agent_id': agent_id,
            'value': value,
            'timestamp': time.time()
        })
    
    def get_from_blackboard(self, key: str, limit: int = None) -> List[Dict[str, Any]]:
        """Get information from the blackboard."""
        if self.pattern != CollaborationPattern.BLACKBOARD:
            return []
        
        if key not in self.blackboard:
            return []
        
        entries = sorted(self.blackboard[key], key=lambda x: x['timestamp'], reverse=True)
        return entries[:limit] if limit else entries
    
    def place_bid(self, agent_id: str, task_id: str, bid_value: float) -> None:
        """Place a bid in the market pattern."""
        if self.pattern != CollaborationPattern.MARKET:
            return
        
        if task_id not in self.market['bids']:
            self.market['bids'][task_id] = []
        
        self.market['bids'][task_id].append({
            'agent_id': agent_id,
            'value': bid_value,
            'timestamp': time.time()
        })
    
    def get_best_bid(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the best bid for a task."""
        if self.pattern != CollaborationPattern.MARKET:
            return None
        
        if task_id not in self.market['bids'] or not self.market['bids'][task_id]:
            return None
        
        return min(self.market['bids'][task_id], key=lambda x: x['value'])
    
    def assign_task(self, task_id: str, agent_id: str) -> None:
        """Assign a task to an agent in the market pattern."""
        if self.pattern != CollaborationPattern.MARKET:
            return
        
        self.market['assignments'][task_id] = agent_id

class ResourceManager:
    """Manager for system resources."""
    
    def __init__(self, config: AgentOrchestrationConfig):
        """Initialize the resource manager."""
        self.config = config
        self.resource_usage = {rt: 0.0 for rt in ResourceType}
        self.resource_limits = config.resource_limits
        self.agent_resource_usage = {}  # Map of agent ID to resource usage
        self.pipeline_resource_usage = {}  # Map of pipeline ID to resource usage
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.performance_monitor = None
        try:
            from performance_monitor import PerformanceMonitor
            self.performance_monitor = PerformanceMonitor()
        except ImportError:
            logger.warning("PerformanceMonitor not available. Resource monitoring will be limited.")
    
    def allocate_resources(self, agent_id: str, resources: Dict[ResourceType, float]) -> bool:
        """Allocate resources to an agent."""
        with self.lock:
            # Check if allocation would exceed limits
            for resource_type, amount in resources.items():
                if self.resource_usage.get(resource_type, 0) + amount > self.resource_limits.get(resource_type, float('inf')):
                    logger.warning(f"Resource allocation for {agent_id} would exceed limits for {resource_type}")
                    return False
            
            # Allocate resources
            for resource_type, amount in resources.items():
                self.resource_usage[resource_type] = self.resource_usage.get(resource_type, 0) + amount
                
                if agent_id not in self.agent_resource_usage:
                    self.agent_resource_usage[agent_id] = {}
                
                self.agent_resource_usage[agent_id][resource_type] = self.agent_resource_usage[agent_id].get(resource_type, 0) + amount
            
            return True
    
    def release_resources(self, agent_id: str, resources: Dict[ResourceType, float] = None) -> None:
        """Release resources allocated to an agent."""
        with self.lock:
            if agent_id not in self.agent_resource_usage:
                return
            
            # If specific resources are provided, release only those
            if resources:
                for resource_type, amount in resources.items():
                    if resource_type in self.agent_resource_usage[agent_id]:
                        used_amount = min(amount, self.agent_resource_usage[agent_id][resource_type])
                        self.agent_resource_usage[agent_id][resource_type] -= used_amount
                        self.resource_usage[resource_type] -= used_amount
                        
                        # Remove resource type if usage is zero
                        if self.agent_resource_usage[agent_id][resource_type] <= 0:
                            del self.agent_resource_usage[agent_id][resource_type]
            else:
                # Release all resources for the agent
                for resource_type, amount in self.agent_resource_usage[agent_id].items():
                    self.resource_usage[resource_type] -= amount
                
                del self.agent_resource_usage[agent_id]
    
    def allocate_pipeline_resources(self, pipeline_id: str, resources: Dict[ResourceType, float]) -> bool:
        """Allocate resources to a pipeline."""
        with self.lock:
            # Check if allocation would exceed limits
            for resource_type, amount in resources.items():
                if self.resource_usage.get(resource_type, 0) + amount > self.resource_limits.get(resource_type, float('inf')):
                    logger.warning(f"Resource allocation for pipeline {pipeline_id} would exceed limits for {resource_type}")
                    return False
            
            # Allocate resources
            for resource_type, amount in resources.items():
                self.resource_usage[resource_type] = self.resource_usage.get(resource_type, 0) + amount
                
                if pipeline_id not in self.pipeline_resource_usage:
                    self.pipeline_resource_usage[pipeline_id] = {}
                
                self.pipeline_resource_usage[pipeline_id][resource_type] = self.pipeline_resource_usage[pipeline_id].get(resource_type, 0) + amount
            
            return True
    
    def release_pipeline_resources(self, pipeline_id: str, resources: Dict[ResourceType, float] = None) -> None:
        """Release resources allocated to a pipeline."""
        with self.lock:
            if pipeline_id not in self.pipeline_resource_usage:
                return
            
            # If specific resources are provided, release only those
            if resources:
                for resource_type, amount in resources.items():
                    if resource_type in self.pipeline_resource_usage[pipeline_id]:
                        used_amount = min(amount, self.pipeline_resource_usage[pipeline_id][resource_type])
                        self.pipeline_resource_usage[pipeline_id][resource_type] -= used_amount
                        self.resource_usage[resource_type] -= used_amount
                        
                        # Remove resource type if usage is zero
                        if self.pipeline_resource_usage[pipeline_id][resource_type] <= 0:
                            del self.pipeline_resource_usage[pipeline_id][resource_type]
            else:
                # Release all resources for the pipeline
                for resource_type, amount in self.pipeline_resource_usage[pipeline_id].items():
                    self.resource_usage[resource_type] -= amount
                
                del self.pipeline_resource_usage[pipeline_id]
    
    def get_resource_usage(self) -> Dict[ResourceType, float]:
        """Get current resource usage."""
        with self.lock:
            return {rt: usage for rt, usage in self.resource_usage.items()}
    
    def get_resource_usage_percentage(self) -> Dict[ResourceType, float]:
        """Get resource usage as percentage of limits."""
        with self.lock:
            return {
                rt: (usage / self.resource_limits.get(rt, 1.0) * 100) if self.resource_limits.get(rt, 0) > 0 else 0.0
                for rt, usage in self.resource_usage.items()
            }
    
    def get_agent_resource_usage(self, agent_id: str) -> Dict[ResourceType, float]:
        """Get resource usage for a specific agent."""
        with self.lock:
            return self.agent_resource_usage.get(agent_id, {})
    
    def get_pipeline_resource_usage(self, pipeline_id: str) -> Dict[ResourceType, float]:
        """Get resource usage for a specific pipeline."""
        with self.lock:
            return self.pipeline_resource_usage.get(pipeline_id, {})
    
    def get_available_resources(self) -> Dict[ResourceType, float]:
        """Get available resources."""
        with self.lock:
            return {
                rt: max(0, self.resource_limits.get(rt, float('inf')) - self.resource_usage.get(rt, 0))
                for rt in ResourceType
                if rt in self.resource_limits
            }
    
    def is_resource_available(self, resources: Dict[ResourceType, float]) -> bool:
        """Check if requested resources are available."""
        with self.lock:
            for resource_type, amount in resources.items():
                if self.resource_usage.get(resource_type, 0) + amount > self.resource_limits.get(resource_type, float('inf')):
                    return False
            return True
    
    def optimize_resource_allocation(self) -> Dict[str, Dict[ResourceType, float]]:
        """Optimize resource allocation across agents."""
        with self.lock:
            # This is a simplified optimization strategy
            # In
