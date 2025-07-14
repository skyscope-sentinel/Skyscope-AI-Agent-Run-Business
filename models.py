from __future__ import annotations
from typing import TypedDict, Literal, List, Optional, Any, Dict, Union
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# ==============================================================================
# --- Enumerations for Categorical Data ---
# ==============================================================================

class AgentRole(Enum):
    """Enumeration of specialized roles an AI agent can assume."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CODER = "coder"
    REVIEWER = "reviewer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    BROWSER = "browser"
    QUANTUM = "quantum"
    CREATIVE = "creative"
    DATA_SCIENTIST = "data_scientist"
    SECURITY = "security"
    DEBUGGER = "debugger"
    TESTER = "tester"
    UI_DESIGNER = "ui_designer"
    DOMAIN_EXPERT = "domain_expert"
    SUMMARIZER = "summarizer"
    CRITIC = "critic"
    INTEGRATOR = "integrator"

class SwarmPattern(Enum):
    """Enumeration of different multi-agent collaboration architectures."""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    MIXTURE_OF_AGENTS = "mixture_of_agents"
    GROUP_CHAT = "group_chat"
    FOREST = "forest"
    SPREADSHEET = "spreadsheet"
    AGENT_REARRANGE = "agent_rearrange"
    ROUTER = "router"

class TaskStatus(Enum):
    """Enumeration of the possible lifecycle statuses for a task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Enumeration of task priority levels for the execution queue."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

# ==============================================================================
# --- TypedDicts for Structured Data ---
# ==============================================================================

class Message(TypedDict):
    """Represents a single message in the chat history."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str

class KnowledgeItem(TypedDict):
    """Represents an item in the knowledge stack."""
    id: str
    title: str
    content: str
    source: Optional[str]
    timestamp: str

class UploadedFile(TypedDict):
    """Represents a file uploaded by the user."""
    id: str
    name: str
    type: str
    content: bytes
    timestamp: str

class ApiKeys(TypedDict):
    """A structured dictionary for storing API keys for various services."""
    openai: str
    anthropic: str
    google: str
    huggingface: str

class ToolSettings(TypedDict):
    """A structured dictionary for managing the enabled/disabled state of various tools."""
    web_search: bool
    deep_research: bool
    deep_thinking: bool
    browser_automation: bool
    quantum_computing: bool
    filesystem_access: bool

class BusinessPlan(TypedDict):
    """A structured dictionary representing a complete business plan."""
    executive_summary: str
    company_description: str
    market_analysis: str
    organization_and_management: str
    service_or_product_line: str
    marketing_and_sales_strategy: str
    financial_projections: str

class ExecutionSteps(TypedDict):
    """A structured dictionary for a phased execution plan."""
    phase_1: List[str]
    phase_2: List[str]
    phase_3: List[str]
    phase_4: List[str]

class GateSpecification(TypedDict):
    """A dictionary specifying a quantum gate and its properties for circuit creation."""
    gate: str
    targets: List[int]
    controls: Optional[List[int]]
    parameters: Optional[List[float]]

# ==============================================================================
# --- Dataclasses for Complex Objects with Behavior ---
# ==============================================================================

@dataclass
class AgentTask:
    """
    Represents a task to be performed by an agent or swarm.
    This dataclass holds all information about a task's state and definition.
    """
    description: str
    input_data: Any = None
    assigned_to: Optional[Union[str, List[str]]] = None
    depends_on: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[datetime] = None
    max_retries: int = 3
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0

    def update_status(self, status: TaskStatus):
        """Updates the task's status and relevant timestamps."""
        self.status = status
        self.updated_at = datetime.now()
        if status == TaskStatus.IN_PROGRESS and not self.started_at:
            self.started_at = datetime.now()
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.completed_at = datetime.now()

    def increment_retry(self):
        """Increments the retry count for the task."""
        self.retry_count += 1
        self.updated_at = datetime.now()

    def can_retry(self) -> bool:
        """Checks if the task can be retried based on its max_retries limit."""
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        """Converts the task object to a JSON-serializable dictionary."""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "assigned_to": self.assigned_to,
            "depends_on": self.depends_on,
            "priority": self.priority.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AgentTask:
        """Creates an AgentTask instance from a dictionary."""
        return cls(
            task_id=data["task_id"],
            description=data["description"],
            assigned_to=data["assigned_to"],
            depends_on=data["depends_on"],
            priority=TaskPriority(data["priority"]),
            max_retries=data["max_retries"],
            status=TaskStatus(data["status"]),
            error=data["error"],
            metadata=data["metadata"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            retry_count=data["retry_count"],
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
        )
