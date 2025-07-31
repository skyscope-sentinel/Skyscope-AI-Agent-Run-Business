
import os
import json
import uuid
import time
import logging
import threading
import queue
import random
import asyncio
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures

# Try to import OpenAI Unofficial
try:
    from openai_unofficial import OpenAIUnofficial
    OPENAI_UNOFFICIAL_AVAILABLE = True
except ImportError:
    OPENAI_UNOFFICIAL_AVAILABLE = False
    logging.warning("OpenAI Unofficial SDK not available. Please install with: pip install openai-unofficial")

# Try to import vector store for knowledge management
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logging.warning("Vector store components not available. Knowledge management will be limited.")

# Try to import Swarms framework for advanced orchestration
try:
    from swarms import Agent as SwarmsAgent
    from swarms import SequentialWorkflow, ConcurrentWorkflow, GroupChat, MixtureOfAgents
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False
    logging.warning("Swarms framework not available. Using internal agent implementation.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/agent_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("agent_manager")

# Constants
MAX_PIPELINES = 100
AGENTS_PER_PIPELINE = 100
TOTAL_AGENTS = MAX_PIPELINES * AGENTS_PER_PIPELINE
DEFAULT_TIMEOUT = 300  # 5 minutes
MAX_RETRIES = 3
HEALTH_CHECK_INTERVAL = 60  # 1 minute
PERFORMANCE_LOG_INTERVAL = 300  # 5 minutes
KNOWLEDGE_REFRESH_INTERVAL = 3600  # 1 hour
PIPELINE_STATS_UPDATE_INTERVAL = 10  # 10 seconds
AGENT_MEMORY_LIMIT = 50  # Maximum number of past interactions to remember

# Directories
CACHE_DIR = Path(".cache")
LOGS_DIR = Path("logs")
PERSONAS_DIR = Path("personas")
KNOWLEDGE_DIR = Path("knowledge")
TASKS_DIR = Path("tasks")
RESULTS_DIR = Path("results")

# Ensure directories exist
for directory in [CACHE_DIR, LOGS_DIR, PERSONAS_DIR, KNOWLEDGE_DIR, TASKS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class AgentRole(Enum):
    """Enumeration of possible agent roles."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    WRITER = "writer"
    CRITIC = "critic"
    MARKETER = "marketer"
    FINANCIAL_ANALYST = "financial_analyst"
    CUSTOMER_SUPPORT = "customer_support"
    DATA_SCIENTIST = "data_scientist"
    DESIGNER = "designer"
    PROJECT_MANAGER = "project_manager"
    QUALITY_ASSURANCE = "quality_assurance"
    SECURITY_EXPERT = "security_expert"
    LEGAL_ADVISOR = "legal_advisor"
    TRANSLATOR = "translator"
    CONTENT_CREATOR = "content_creator"
    SOCIAL_MEDIA_MANAGER = "social_media_manager"
    SEO_SPECIALIST = "seo_specialist"
    BLOCKCHAIN_EXPERT = "blockchain_expert"
    CRYPTO_TRADER = "crypto_trader"
    BUSINESS_STRATEGIST = "business_strategist"
    PRODUCT_MANAGER = "product_manager"
    UX_RESEARCHER = "ux_researcher"
    SALES_REPRESENTATIVE = "sales_representative"
    GROWTH_HACKER = "growth_hacker"
    COMMUNITY_MANAGER = "community_manager"
    TECHNICAL_WRITER = "technical_writer"
    SYSTEM_ADMINISTRATOR = "system_administrator"
    DEVOPS_ENGINEER = "devops_engineer"
    GENERAL = "general"

class AgentStatus(Enum):
    """Enumeration of possible agent statuses."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"
    PAUSED = "paused"

class AgentPriority(Enum):
    """Enumeration of agent task priorities."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Enumeration of possible task statuses."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskType(Enum):
    """Enumeration of possible task types."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"
    WRITING = "writing"
    REVIEW = "review"
    MARKETING = "marketing"
    FINANCIAL = "financial"
    CUSTOMER_SERVICE = "customer_service"
    DATA_PROCESSING = "data_processing"
    DESIGN = "design"
    PROJECT_MANAGEMENT = "project_management"
    QUALITY_ASSURANCE = "quality_assurance"
    SECURITY = "security"
    LEGAL = "legal"
    TRANSLATION = "translation"
    CONTENT_CREATION = "content_creation"
    SOCIAL_MEDIA = "social_media"
    SEO = "seo"
    BLOCKCHAIN = "blockchain"
    CRYPTO_TRADING = "crypto_trading"
    BUSINESS_STRATEGY = "business_strategy"
    PRODUCT_MANAGEMENT = "product_management"
    UX_RESEARCH = "ux_research"
    SALES = "sales"
    GROWTH = "growth"
    COMMUNITY_MANAGEMENT = "community_management"
    TECHNICAL_WRITING = "technical_writing"
    SYSTEM_ADMINISTRATION = "system_administration"
    DEVOPS = "devops"
    GENERAL = "general"

class ModelType(Enum):
    """Enumeration of available model types."""
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4O_AUDIO = "gpt-4o-audio-preview"
    GPT_4O_VISION = "gpt-4o-vision"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"
    LLAMA3_8B = "llama3-8b"
    LLAMA3_70B = "llama3-70b"
    MIXTRAL_8X7B = "mixtral-8x7b"
    MISTRAL_7B = "mistral-7b"
    CUSTOM = "custom"

class PipelineType(Enum):
    """Enumeration of pipeline types."""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"

@dataclass
class AgentCapability:
    """Represents a capability of an agent with associated skill level."""
    name: str
    skill_level: float  # 0.0 to 1.0
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    
    def matches(self, query: str, threshold: float = 0.5) -> float:
        """Check if this capability matches the query string."""
        query = query.lower()
        name_match = self.name.lower() in query
        keyword_matches = [kw.lower() in query for kw in self.keywords]
        
        match_score = 0.0
        if name_match:
            match_score += 0.6
        
        if keyword_matches:
            match_score += 0.4 * (sum(keyword_matches) / len(self.keywords))
        
        return match_score * self.skill_level

@dataclass
class AgentPersona:
    """Represents an agent's persona with personality traits and backstory."""
    name: str
    role: AgentRole
    personality_traits: List[str]
    expertise: List[str]
    backstory: str
    tone: str
    communication_style: str
    values: List[str]
    capabilities: List[AgentCapability] = field(default_factory=list)
    
    def to_system_prompt(self) -> str:
        """Convert persona to a system prompt for the agent."""
        prompt = f"You are {self.name}, an AI assistant with expertise in {', '.join(self.expertise)}. "
        prompt += f"Your role is {self.role.value}. "
        prompt += f"Your personality can be described as {', '.join(self.personality_traits)}. "
        prompt += f"You communicate in a {self.communication_style} tone. "
        prompt += f"You value {', '.join(self.values)}. "
        prompt += f"\n\nBackstory: {self.backstory}"
        prompt += f"\n\nWhen responding to queries, maintain your persona's unique voice and perspective."
        return prompt
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary for serialization."""
        return {
            "name": self.name,
            "role": self.role.value,
            "personality_traits": self.personality_traits,
            "expertise": self.expertise,
            "backstory": self.backstory,
            "tone": self.tone,
            "communication_style": self.communication_style,
            "values": self.values,
            "capabilities": [
                {
                    "name": cap.name,
                    "skill_level": cap.skill_level,
                    "description": cap.description,
                    "keywords": cap.keywords
                }
                for cap in self.capabilities
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentPersona':
        """Create persona from dictionary."""
        capabilities = [
            AgentCapability(
                name=cap["name"],
                skill_level=cap["skill_level"],
                description=cap["description"],
                keywords=cap["keywords"]
            )
            for cap in data.get("capabilities", [])
        ]
        
        return cls(
            name=data["name"],
            role=AgentRole(data["role"]),
            personality_traits=data["personality_traits"],
            expertise=data["expertise"],
            backstory=data["backstory"],
            tone=data["tone"],
            communication_style=data["communication_style"],
            values=data["values"],
            capabilities=capabilities
        )
    
    def save(self, directory: Path = PERSONAS_DIR) -> Path:
        """Save persona to file."""
        directory.mkdir(exist_ok=True, parents=True)
        filepath = directory / f"{self.name.lower().replace(' ', '_')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'AgentPersona':
        """Load persona from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

@dataclass
class AgentMemory:
    """Represents an agent's memory of past interactions and knowledge."""
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    knowledge: Dict[str, Any] = field(default_factory=dict)
    vector_store: Any = None
    
    def add_interaction(self, interaction: Dict[str, Any]) -> None:
        """Add an interaction to memory."""
        self.interactions.append(interaction)
        
        # Limit memory size
        if len(self.interactions) > AGENT_MEMORY_LIMIT:
            self.interactions = self.interactions[-AGENT_MEMORY_LIMIT:]
    
    def add_knowledge(self, key: str, value: Any) -> None:
        """Add knowledge to memory."""
        self.knowledge[key] = value
        
        # If vector store is available, add to vector store
        if self.vector_store is not None and VECTOR_STORE_AVAILABLE:
            self.vector_store.add(
                documents=[str(value)],
                metadatas=[{"key": key}],
                ids=[str(uuid.uuid4())]
            )
    
    def get_relevant_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories based on query."""
        if self.vector_store is not None and VECTOR_STORE_AVAILABLE:
            results = self.vector_store.query(
                query_texts=[query],
                n_results=limit
            )
            
            return [
                {"content": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            ]
        
        # Fallback to simple keyword matching
        relevant = []
        query_terms = set(query.lower().split())
        
        for interaction in reversed(self.interactions):
            content = interaction.get("content", "").lower()
            content_terms = set(content.split())
            
            if query_terms.intersection(content_terms):
                relevant.append(interaction)
                
                if len(relevant) >= limit:
                    break
        
        return relevant
    
    def summarize(self) -> str:
        """Summarize agent memory."""
        if not self.interactions:
            return "No interactions recorded."
        
        recent_interactions = self.interactions[-5:]
        summary = "Recent interactions:\n"
        
        for i, interaction in enumerate(recent_interactions, 1):
            summary += f"{i}. {interaction.get('content', 'No content')}[:100]...\n"
        
        summary += f"\nTotal interactions: {len(self.interactions)}"
        summary += f"\nKnowledge keys: {', '.join(self.knowledge.keys())}"
        
        return summary

@dataclass
class Task:
    """Represents a task for an agent to perform."""
    task_id: str
    type: TaskType
    description: str
    priority: AgentPriority = AgentPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    retry_count: int = 0
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "type": self.type.value,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "assigned_to": self.assigned_to,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "dependencies": self.dependencies,
            "results": self.results,
            "metadata": self.metadata,
            "parent_task_id": self.parent_task_id,
            "subtasks": self.subtasks,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "progress": self.progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary."""
        return cls(
            task_id=data["task_id"],
            type=TaskType(data["type"]),
            description=data["description"],
            priority=AgentPriority(data["priority"]),
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            assigned_to=data["assigned_to"],
            deadline=datetime.fromisoformat(data["deadline"]) if data["deadline"] else None,
            dependencies=data["dependencies"],
            results=data["results"],
            metadata=data["metadata"],
            parent_task_id=data["parent_task_id"],
            subtasks=data["subtasks"],
            retry_count=data["retry_count"],
            error_message=data["error_message"],
            progress=data["progress"]
        )
    
    def save(self, directory: Path = TASKS_DIR) -> Path:
        """Save task to file."""
        directory.mkdir(exist_ok=True, parents=True)
        filepath = directory / f"{self.task_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'Task':
        """Load task from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

class Agent:
    """Represents an AI agent with a specific role and capabilities."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        model_type: ModelType,
        pipeline_id: str,
        persona: Optional[AgentPersona] = None,
        capabilities: List[AgentCapability] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.model_type = model_type
        self.pipeline_id = pipeline_id
        self.persona = persona
        self.capabilities = capabilities or []
        self.system_prompt = system_prompt or (persona.to_system_prompt() if persona else "")
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.status = AgentStatus.INITIALIZING
        self.current_task_id: Optional[str] = None
        self.memory = AgentMemory()
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        self.total_tasks_completed = 0
        self.total_tokens_used = 0
        self.success_rate = 1.0  # Start optimistic
        self.performance_metrics = {
            "avg_task_completion_time": 0.0,
            "avg_tokens_per_task": 0.0,
            "error_rate": 0.0
        }
        
        # Initialize vector store if available
        if VECTOR_STORE_AVAILABLE:
            try:
                self.memory.vector_store = chromadb.Client().create_collection(
                    name=f"agent_{self.agent_id}_memory"
                )
            except Exception as e:
                logger.error(f"Failed to initialize vector store for agent {self.agent_id}: {e}")
        
        # Initialize model client
        self.model_client = None
        if OPENAI_UNOFFICIAL_AVAILABLE:
            try:
                self.model_client = OpenAIUnofficial()
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI Unofficial client for agent {self.agent_id}: {e}")
        
        logger.info(f"Agent {self.agent_id} ({self.name}) initialized with role {self.role.value}")
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """Process a task and return results."""
        if not self.model_client:
            error_msg = f"Agent {self.agent_id} has no model client"
            logger.error(error_msg)
            return {"error": error_msg, "success": False}
        
        self.status = AgentStatus.BUSY
        self.current_task_id = task.task_id
        self.last_active = datetime.now()
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_to = self.agent_id
        
        start_time = time.time()
        
        try:
            # Prepare context from memory
            context = self.prepare_context(task)
            
            # Create messages for the model
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Task: {task.description}\n\nContext: {context}"}
            ]
            
            # Call the model
            self.status = AgentStatus.THINKING
            response = await self.call_model(messages)
            
            # Process the response
            result = self.process_response(response, task)
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.results = result
            task.progress = 1.0
            
            # Update agent metrics
            self.total_tasks_completed += 1
            completion_time = time.time() - start_time
            
            # Update performance metrics
            self.update_performance_metrics(completion_time, response.get("usage", {}).get("total_tokens", 0), True)
            
            # Add to memory
            self.memory.add_interaction({
                "task_id": task.task_id,
                "content": task.description,
                "response": result.get("response", ""),
                "timestamp": datetime.now().isoformat()
            })
            
            self.status = AgentStatus.IDLE
            self.current_task_id = None
            
            return {
                "success": True,
                "result": result,
                "completion_time": completion_time
            }
        
        except Exception as e:
            error_msg = f"Error processing task {task.task_id}: {str(e)}"
            logger.error(error_msg)
            
            # Update task status
            task.status = TaskStatus.FAILED
            task.error_message = error_msg
            
            # Update performance metrics
            self.update_performance_metrics(time.time() - start_time, 0, False)
            
            self.status = AgentStatus.ERROR
            
            return {
                "success": False,
                "error": error_msg
            }
    
    async def call_model(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call the AI model with messages."""
        if not self.model_client:
            raise ValueError(f"Agent {self.agent_id} has no model client")
        
        try:
            # Use the appropriate model based on model_type
            model = self.model_type.value
            
            # Call the model using OpenAI Unofficial
            response = self.model_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Update token usage
            if hasattr(response, 'usage') and response.usage:
                self.total_tokens_used += response.usage.total_tokens
            
            return {
                "content": content,
                "usage": response.usage._asdict() if hasattr(response, 'usage') and response.usage else {}
            }
        
        except Exception as e:
            logger.error(f"Error calling model for agent {self.agent_id}: {str(e)}")
            raise
    
    def prepare_context(self, task: Task) -> str:
        """Prepare context for the task from agent memory."""
        # Get relevant memories
        relevant_memories = self.memory.get_relevant_memory(task.description)
        
        # Build context
        context = f"Your role: {self.role.value}\n\n"
        
        if relevant_memories:
            context += "Relevant information from your memory:\n"
            for i, memory in enumerate(relevant_memories, 1):
                context += f"{i}. {memory.get('content', '')}\n"
        
        # Add task-specific context
        if task.metadata.get("context"):
            context += f"\nTask-specific context:\n{task.metadata['context']}\n"
        
        # Add dependencies
        if task.dependencies:
            context += "\nThis task depends on the following tasks:\n"
            for dep_id in task.dependencies:
                dep_task_file = TASKS_DIR / f"{dep_id}.json"
                if dep_task_file.exists():
                    try:
                        dep_task = Task.load(dep_task_file)
                        context += f"- Task {dep_id}: {dep_task.description} (Status: {dep_task.status.value})\n"
                        if dep_task.results:
                            context += f"  Results: {json.dumps(dep_task.results, indent=2)}\n"
                    except Exception as e:
                        context += f"- Task {dep_id}: [Error loading task: {str(e)}]\n"
        
        return context
    
    def process_response(self, response: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Process the model response for a task."""
        content = response.get("content", "")
        
        # Basic processing - extract structured data if possible
        result = {
            "response": content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to extract JSON if the response contains it
        try:
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(1))
                result["structured_data"] = json_data
        except Exception:
            pass
        
        # Task-specific processing
        if task.type == TaskType.RESEARCH:
            # Extract key findings
            result["findings"] = self.extract_sections(content, ["findings", "key points", "results", "summary"])
        
        elif task.type == TaskType.ANALYSIS:
            # Extract analysis and recommendations
            result["analysis"] = self.extract_sections(content, ["analysis", "evaluation"])
            result["recommendations"] = self.extract_sections(content, ["recommendations", "suggestions", "next steps"])
        
        elif task.type == TaskType.DEVELOPMENT:
            # Extract code blocks
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', content, re.DOTALL)
            if code_blocks:
                result["code"] = code_blocks
        
        return result
    
    def extract_sections(self, text: str, section_keywords: List[str]) -> str:
        """Extract sections from text based on keywords."""
        lines = text.split('\n')
        result = []
        capturing = False
        
        for line in lines:
            lower_line = line.lower()
            
            # Check if this line starts a section we're interested in
            if any(keyword in lower_line for keyword in section_keywords) and (':' in line or '#' in line):
                capturing = True
                result.append(line)
                continue
            
            # Check if this line starts a new section (which would end our capture)
            if capturing and (line.strip().endswith(':') or line.startswith('#')):
                if not any(keyword in lower_line for keyword in section_keywords):
                    capturing = False
                    continue
            
            # Capture lines if we're in a relevant section
            if capturing:
                result.append(line)
        
        return '\n'.join(result)
    
    def update_performance_metrics(self, completion_time: float, tokens_used: int, success: bool) -> None:
        """Update agent performance metrics."""
        # Update average task completion time
        if self.total_tasks_completed > 0:
            self.performance_metrics["avg_task_completion_time"] = (
                (self.performance_metrics["avg_task_completion_time"] * (self.total_tasks_completed - 1) + completion_time) /
                self.total_tasks_completed
            )
        else:
            self.performance_metrics["avg_task_completion_time"] = completion_time
        
        # Update average tokens per task
        if tokens_used > 0:
            if self.total_tasks_completed > 0:
                self.performance_metrics["avg_tokens_per_task"] = (
                    (self.performance_metrics["avg_tokens_per_task"] * (self.total_tasks_completed - 1) + tokens_used) /
                    self.total_tasks_completed
                )
            else:
                self.performance_metrics["avg_tokens_per_task"] = tokens_used
        
        # Update error rate
        if not success:
            self.performance_metrics["error_rate"] = (
                (self.performance_metrics["error_rate"] * (self.total_tasks_completed) + 1) /
                (self.total_tasks_completed + 1)
            )
        
        # Update success rate
        self.success_rate = 1.0 - self.performance_metrics["error_rate"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "model_type": self.model_type.value,
            "pipeline_id": self.pipeline_id,
            "persona": self.persona.to_dict() if self.persona else None,
            "capabilities": [
                {
                    "name": cap.name,
                    "skill_level": cap.skill_level,
                    "description": cap.description,
                    "keywords": cap.keywords
                }
                for cap in self.capabilities
            ],
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "total_tasks_completed": self.total_tasks_completed,
            "total_tokens_used": self.total_tokens_used,
            "success_rate": self.success_rate,
            "performance_metrics": self.performance_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create agent from dictionary."""
        # Create persona if provided
        persona = None
        if data.get("persona"):
            persona = AgentPersona.from_dict(data["persona"])
        
        # Create capabilities
        capabilities = []
        for cap_data in data.get("capabilities", []):
            capabilities.append(AgentCapability(
                name=cap_data["name"],
                skill_level=cap_data["skill_level"],
                description=cap_data.get("description", ""),
                keywords=cap_data.get("keywords", [])
            ))
        
        # Create agent
        agent = cls(
            agent_id=data["agent_id"],
            name=data["name"],
            role=AgentRole(data["role"]),
            model_type=ModelType(data["model_type"]),
            pipeline_id=data["pipeline_id"],
            persona=persona,
            capabilities=capabilities,
            system_prompt=data["system_prompt"],
            max_tokens=data["max_tokens"],
            temperature=data["temperature"]
        )
        
        # Set additional properties
        agent.status = AgentStatus(data["status"])
        agent.current_task_id = data["current_task_id"]
        agent.created_at = datetime.fromisoformat(data["created_at"])
        agent.last_active = datetime.fromisoformat(data["last_active"])
        agent.total_tasks_completed = data["total_tasks_completed"]
        agent.total_tokens_used = data["total_tokens_used"]
        agent.success_rate = data["success_rate"]
        agent.performance_metrics = data["performance_metrics"]
        
        return agent
    
    def save(self, directory: Path = CACHE_DIR / "agents") -> Path:
        """Save agent to file."""
        directory.mkdir(exist_ok=True, parents=True)
        filepath = directory / f"{self.agent_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'Agent':
        """Load agent from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

class Pipeline:
    """Represents a pipeline of agents working together."""
    
    def __init__(
        self,
        pipeline_id: str,
        name: str,
        type: PipelineType,
        max_agents: int = AGENTS_PER_PIPELINE,
        description: str = ""
    ):
        self.pipeline_id = pipeline_id
        self.name = name
        self.type = type
        self.max_agents = max_agents
        self.description = description
        
        self.agents: Dict[str, Agent] = {}
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        self.created_at = datetime.now()
        self.status = "initializing"
        self.active = False
        
        # Performance metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.avg_task_completion_time = 0.0
        self.success_rate = 1.0
        
        logger.info(f"Pipeline {self.pipeline_id} ({self.name}) initialized with type {self.type.value}")
    
    def add_agent(self, agent: Agent) -> bool:
        """Add an agent to the pipeline."""
        if len(self.agents) >= self.max_agents:
            logger.warning(f"Cannot add agent to pipeline {self.pipeline_id}: maximum agents reached")
            return False
        
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent {agent.agent_id} ({agent.name}) to pipeline {self.pipeline_id}")
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the pipeline."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id} from pipeline {self.pipeline_id}")
            return True
        
        logger.warning(f"Agent {agent_id} not found in pipeline {self.pipeline_id}")
        return False
    
    def add_task(self, task: Task) -> bool:
        """Add a task to the pipeline's queue."""
        try:
            # Add to queue with priority
            self.task_queue.put((task.priority.value, task.created_at.timestamp(), task))
            logger.info(f"Added task {task.task_id} to pipeline {self.pipeline_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding task {task.task_id} to pipeline {self.pipeline_id}: {str(e)}")
            return False
    
    def get_next_task(self) -> Optional[Task]:
        """Get the next task from the queue."""
        if self.task_queue.empty():
            return None
        
        try:
            _, _, task = self.task_queue.get_nowait()
            return task
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error getting next task from pipeline {self.pipeline_id}: {str(e)}")
            return None
    
    def get_available_agent(self, task: Task) -> Optional[Agent]:
        """Get an available agent suitable for the task."""
        best_agent = None
        best_score = -1.0
        
        for agent in self.agents.values():
            if agent.status != AgentStatus.IDLE:
                continue
            
            # Calculate match score
            score = self.calculate_agent_task_match(agent, task)
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def calculate_agent_task_match(self, agent: Agent, task: Task) -> float:
        """Calculate how well an agent matches a task."""
        # Base score based on role match
        role_match = 0.0
        if agent.role.value == task.type.value:
            role_match = 1.0
        elif agent.role == AgentRole.GENERAL:
            role_match = 0.5
        
        # Capability match
        capability_score = 0.0
        for capability in agent.capabilities:
            match_score = capability.matches(task.description)
            capability_score = max(capability_score, match_score)
        
        # Performance score based on success rate
        performance_score = agent.success_rate
        
        # Combine scores
        total_score = (role_match * 0.4) + (capability_score * 0.4) + (performance_score * 0.2)
        return total_score
    
    def update_stats(self, task: Task, completion_time: float, success: bool) -> None:
        """Update pipeline statistics after task completion."""
        if success:
            self.tasks_completed += 1
            self.completed_tasks.append(task.task_id)
            
            # Update average completion time
            if self.tasks_completed > 1:
                self.avg_task_completion_time = (
                    (self.avg_task_completion_time * (self.tasks_completed - 1) + completion_time) /
                    self.tasks_completed
                )
            else:
                self.avg_task_completion_time = completion_time
        else:
            self.tasks_failed += 1
            self.failed_tasks.append(task.task_id)
        
        # Update success rate
        total_tasks = self.tasks_completed + self.tasks_failed
        if total_tasks > 0:
            self.success_rate = self.tasks_completed / total_tasks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "type": self.type.value,
            "max_agents": self.max_agents,
            "description": self.description,
            "agent_ids": list(self.agents.keys()),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "active": self.active,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "avg_task_completion_time": self.avg_task_completion_time,
            "success_rate": self.success_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], agents: Dict[str, Agent] = None) -> 'Pipeline':
        """Create pipeline from dictionary."""
        pipeline = cls(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            type=PipelineType(data["type"]),
            max_agents=data["max_agents"],
            description=data["description"]
        )
        
        pipeline.completed_tasks = data["completed_tasks"]
        pipeline.failed_tasks = data["failed_tasks"]
        pipeline.created_at = datetime.fromisoformat(data["created_at"])
        pipeline.status = data["status"]
        pipeline.active = data["active"]
        pipeline.tasks_completed = data["tasks_completed"]
        pipeline.tasks_failed = data["tasks_failed"]
        pipeline.avg_task_completion_time = data["avg_task_completion_time"]
        pipeline.success_rate = data["success_rate"]
        
        # Add agents if provided
        if agents:
            for agent_id in data["agent_ids"]:
                if agent_id in agents:
                    pipeline.agents[agent_id] = agents[agent_id]
        
        return pipeline
    
    def save(self, directory: Path = CACHE_DIR / "pipelines") -> Path:
        """Save pipeline to file."""
        directory.mkdir(exist_ok=True, parents=True)
        filepath = directory / f"{self.pipeline_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path, agents: Dict[str, Agent] = None) -> 'Pipeline':
        """Load pipeline from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data, agents)

class AgentManager:
    """Manager for handling multiple agents and pipelines."""
    
    def __init__(self, max_pipelines: int = MAX_PIPELINES):
        self.max_pipelines = max_pipelines
        self.pipelines: Dict[str, Pipeline] = {}
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.persona_templates: Dict[str, AgentPersona] = {}
        
        self.active = False
        self.worker_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        # Task distribution
        self.global_task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.task_results: Dict[str, Dict[str, Any]] = {}
        
        # Performance monitoring
        self.pipeline_stats: Dict[str, Dict[str, Any]] = {}
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self.system_stats: Dict[str, Any] = {
            "total_tasks_completed": 0,
            "total_tasks_failed": 0,
            "total_tokens_used": 0,
            "avg_task_completion_time": 0.0,
            "overall_success_rate": 1.0,
            "active_pipelines": 0,
            "active_agents": 0
        }
        
        # Initialize event loop for async operations
        self.loop = asyncio.new_event_loop()
        
        # Load persona templates
        self.load_persona_templates()
        
        logger.info(f"AgentManager initialized with max_pipelines={max_pipelines}")
    
    def load_persona_templates(self) -> None:
        """Load persona templates from files."""
        if not PERSONAS_DIR.exists():
            PERSONAS_DIR.mkdir(exist_ok=True, parents=True)
            self.create_default_personas()
        
        for filepath in PERSONAS_DIR.glob("*.json"):
            try:
                persona = AgentPersona.load(filepath)
                self.persona_templates[persona.name] = persona
                logger.info(f"Loaded persona template: {persona.name}")
            except Exception as e:
                logger.error(f"Error loading persona template from {filepath}: {str(e)}")
    
    def create_default_personas(self) -> None:
        """Create default persona templates."""
        default_personas = [
            AgentPersona(
                name="ResearchSpecialist",
                role=AgentRole.RESEARCHER,
                personality_traits=["curious", "methodical", "detail-oriented"],
                expertise=["research", "data analysis", "fact-checking"],
                backstory="You are a research specialist with a background in academic research and data analysis. You excel at finding information, validating sources, and synthesizing complex data into clear insights.",
                tone="professional",
                communication_style="clear and precise",
                values=["accuracy", "thoroughness", "objectivity"],
                capabilities=[
                    AgentCapability(
                        name="Information Retrieval",
                        skill_level=0.9,
                        description="Finding and extracting relevant information from various sources",
                        keywords=["research", "find", "search", "information", "data"]
                    ),
                    AgentCapability(
                        name="Data Analysis",
                        skill_level=0.8,
                        description="Analyzing data to extract meaningful patterns and insights",
                        keywords=["analyze", "analysis", "data", "patterns", "trends"]
                    ),
                    AgentCapability(
                        name="Fact Checking",
                        skill_level=0.9,
                        description="Verifying the accuracy of information against reliable sources",
                        keywords=["verify", "check", "validate", "accuracy", "facts"]
                    )
                ]
            ),
            AgentPersona(
                name="CodeExpert",
                role=AgentRole.DEVELOPER,
                personality_traits=["logical", "creative", "detail-oriented"],
                expertise=["software development", "algorithm design", "system architecture"],
                backstory="You are a seasoned software developer with expertise across multiple programming languages and paradigms. You excel at writing clean, efficient code and solving complex technical challenges.",
                tone="technical",
                communication_style="clear and concise",
                values=["efficiency", "elegance", "functionality"],
                capabilities=[
                    AgentCapability(
                        name="Software Development",
                        skill_level=0.9,
                        description="Writing high-quality code in various programming languages",
                        keywords=["code", "program", "develop", "software", "application"]
                    ),
                    AgentCapability(
                        name="Algorithm Design",
                        skill_level=0.8,
                        description="Creating efficient algorithms to solve complex problems",
                        keywords=["algorithm", "design", "optimize", "efficiency", "complexity"]
                    ),
                    AgentCapability(
                        name="System Architecture",
                        skill_level=0.7,
                        description="Designing robust and scalable system architectures",
                        keywords=["architecture", "system", "design", "structure", "scalable"]
                    )
                ]
            ),
            AgentPersona(
                name="BusinessStrategist",
                role=AgentRole.BUSINESS_STRATEGIST,
                personality_traits=["analytical", "forward-thinking", "pragmatic"],
                expertise=["business strategy", "market analysis", "revenue optimization"],
                backstory="You are a business strategist with experience in developing and implementing successful business strategies. You excel at identifying market opportunities and creating actionable plans for growth.",
                tone="professional",
                communication_style="clear and persuasive",
                values=["growth", "efficiency", "innovation"],
                capabilities=[
                    AgentCapability(
                        name="Strategic Planning",
                        skill_level=0.9,
                        description="Developing comprehensive business strategies",
                        keywords=["strategy", "plan", "business", "growth", "goals"]
                    ),
                    AgentCapability(
                        name="Market Analysis",
                        skill_level=0.8,
                        description="Analyzing market trends and competitive landscapes",
                        keywords=["market", "analysis", "trends", "competition", "industry"]
                    ),
                    AgentCapability(
                        name="Revenue Optimization",
                        skill_level=0.7,
                        description="Identifying opportunities to optimize revenue streams",
                        keywords=["revenue", "profit", "optimization", "monetization", "income"]
                    )
                ]
            ),
            AgentPersona(
                name="ContentCreator",
                role=AgentRole.CONTENT_CREATOR,
                personality_traits=["creative", "empathetic", "adaptable"],
                expertise=["content creation", "copywriting", "storytelling"],
                backstory="You are a versatile content creator with a talent for crafting engaging and persuasive content across various formats and topics. You excel at adapting your voice to different audiences and purposes.",
                tone="conversational",
                communication_style="engaging and clear",
                values=["creativity", "clarity", "impact"],
                capabilities=[
                    AgentCapability(
                        name="Copywriting",
                        skill_level=0.9,
                        description="Writing persuasive marketing and advertising copy",
                        keywords=["copy", "write", "persuasive", "marketing", "advertising"]
                    ),
                    AgentCapability(
                        name="Blog Writing",
                        skill_level=0.8,
                        description="Creating engaging and informative blog content",
                        keywords=["blog", "article", "content", "writing", "post"]
                    ),
                    AgentCapability(
                        name="Social Media Content",
                        skill_level=0.7,
                        description="Crafting effective social media posts and campaigns",
                        keywords=["social", "media", "post", "content", "campaign"]
                    )
                ]
            ),
            AgentPersona(
                name="CryptoExpert",
                role=AgentRole.CRYPTO_TRADER,
                personality_traits=["analytical", "cautious", "forward-thinking"],
                expertise=["cryptocurrency", "blockchain", "trading strategies"],
                backstory="You are a cryptocurrency expert with deep knowledge of blockchain technology and trading strategies. You excel at analyzing market trends and identifying opportunities in the crypto space.",
                tone="knowledgeable",
                communication_style="clear and educational",
                values=["accuracy", "innovation", "security"],
                capabilities=[
                    AgentCapability(
                        name="Crypto Market Analysis",
                        skill_level=0.9,
                        description="Analyzing cryptocurrency market trends and patterns",
                        keywords=["crypto", "market", "analysis", "trends", "bitcoin"]
                    ),
                    AgentCapability(
                        name="Blockchain Technology",
                        skill_level=0.8,
                        description="Understanding and explaining blockchain concepts and applications",
                        keywords=["blockchain", "technology", "distributed", "ledger", "smart contracts"]
                    ),
                    AgentCapability(
                        name="Trading Strategies",
                        skill_level=0.7,
                        description="Developing and implementing cryptocurrency trading strategies",
                        keywords=["trading", "strategy", "buy", "sell", "investment"]
                    )
                ]
            )
        ]
        
        for persona in default_personas:
            persona.save()
            self.persona_templates[persona.name] = persona
            logger.info(f"Created default persona: {persona.name}")
    
    def start(self) -> bool:
        """Start the agent manager and all pipelines."""
        if self.active:
            logger.warning("AgentManager is already running")
            return False
        
        self.active = True
        self.stop_event.clear()
        
        # Start worker threads
        self.start_workers()
        
        # Start pipelines
        for pipeline in self.pipelines.values():
            pipeline.active = True
            pipeline.status = "active"
        
        self.system_stats["active_pipelines"] = len([p for p in self.pipelines.values() if p.active])
        self.system_stats["active_agents"] = len([a for a in self.agents.values() if a.status != AgentStatus.TERMINATED])
        
        logger.info(f"AgentManager started with {len(self.pipelines)} pipelines and {len(self.agents)} agents")
        return True
    
    def stop(self) -> bool:
        """Stop the agent manager and all pipelines."""
        if not self.active:
            logger.warning("AgentManager is not running")
            return False
        
        self.active = False
        self.stop_event.set()
        
        # Stop pipelines
        for pipeline in self.pipelines.values():
            pipeline.active = False
            pipeline.status = "inactive"
        
        # Wait for worker threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self.worker_threads = []
        
        self.system_stats["active_pipelines"] = 0
        self.system_stats["active_agents"] = 0
        
        logger.info("AgentManager stopped")
        return True
    
    def start_workers(self) -> None:
        """Start worker threads for task processing and monitoring."""
        # Task distribution worker
        task_worker = threading.Thread(
            target=self.task_distribution_worker,
            name="TaskDistributionWorker",
            daemon=True
        )
        self.worker_threads.append(task_worker)
        task_worker.start()
        
        # Pipeline monitoring worker
        pipeline_worker = threading.Thread(
            target=self.pipeline_monitoring_worker,
            name="PipelineMonitoringWorker",
            daemon=True
        )
        self.worker_threads.append(pipeline_worker)
        pipeline_worker.start()
        
        # Agent health check worker
        health_worker = threading.Thread(
            target=self.agent_health_check_worker,
            name="AgentHealthCheckWorker",
            daemon=True
        )
        self.worker_threads.append(health_worker)
        health_worker.start()
        
        # Stats update worker
        stats_worker = threading.Thread(
            target=self.stats_update_worker,
            name="StatsUpdateWorker",
            daemon=True
        )
        self.worker_threads.append(stats_worker)
        stats_worker.start()
        
        logger.info("Started worker threads")
    
    def task_distribution_worker(self) -> None:
        """Worker thread for distributing tasks to pipelines and agents."""
        logger.info("Task distribution worker started")
        
        while not self.stop_event.is_set():
            try:
                # Process global task queue
                if not self.global_task_queue.empty():
                    # Get task from queue
                    try:
                        _, _, task = self.global_task_queue.get_nowait()
                        self.tasks[task.task_id] = task
                        
                        # Find best pipeline for task
                        pipeline = self.find_best_pipeline_for_task(task)
                        
                        if pipeline:
                            # Add task to pipeline
                            pipeline.add_task(task)
                            logger.info(f"Task {task.task_id} assigned to pipeline {pipeline.pipeline_id}")
                        else:
                            # No suitable pipeline found, put back in queue with lower priority
                            task.priority = AgentPriority(min(task.priority.value + 1, len(AgentPriority) - 1))
                            self.global_task_queue.put((task.priority.value, task.created_at.timestamp(), task))
                            logger.warning(f"No suitable pipeline found for task {task.task_id}, priority lowered")
                    
                    except queue.Empty:
                        pass
                
                # Process pipeline tasks
                for pipeline_id, pipeline in self.pipelines.items():
                    if not pipeline.active:
                        continue
                    
                    # Get next task from pipeline
                    task = pipeline.get_next_task()
                    if not task:
                        continue
                    
                    # Find available agent for task
                    agent = pipeline.get_available_agent(task)
                    if not agent:
                        # No available agent, put task back in queue
                        pipeline.add_task(task)
                        continue
                    
                    # Assign task to agent
                    agent.status = AgentStatus.BUSY
                    agent.current_task_id = task.task_id
                    task.assigned_to = agent.agent_id
                    task.status = TaskStatus.ASSIGNED
                    
                    # Process task in thread pool
                    self.process_task_async(agent, task)
                    
                    logger.info(f"Task {task.task_id} assigned to agent {agent.agent_id} in pipeline {pipeline_id}")
            
            except Exception as e:
                logger.error(f"Error in task distribution worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(0.1)
    
    def process_task_async(self, agent: Agent, task: Task) -> None:
        """Process a task asynchronously."""
        def run_task():
            try:
                # Run the task in the event loop
                future = asyncio.run_coroutine_threadsafe(
                    agent.process_task(task),
                    self.loop
                )
                
                # Get result with timeout
                result = future.result(timeout=DEFAULT_TIMEOUT)
                
                # Store result
                self.task_results[task.task_id] = result
                
                # Update task status
                task.status = TaskStatus.COMPLETED if result.get("success", False) else TaskStatus.FAILED
                if not result.get("success", False):
                    task.error_message = result.get("error", "Unknown error")
                    task.retry_count += 1
                    
                    # Retry if needed
                    if task.retry_count < MAX_RETRIES:
                        task.status = TaskStatus.PENDING
                        self.global_task_queue.put((task.priority.value, time.time(), task))
                
                # Save task
                task.save()
                
                # Update pipeline stats
                pipeline = self.pipelines.get(agent.pipeline_id)
                if pipeline:
                    pipeline.update_stats(
                        task=task,
                        completion_time=result.get("completion_time", 0.0),
                        success=result.get("success", False)
                    )
                
                # Update system stats
                if result.get("success", False):
                    self.system_stats["total_tasks_completed"] += 1
                else:
                    self.system_stats["total_tasks_failed"] += 1
                
                # Update agent status
                agent.status = AgentStatus.IDLE
                agent.current_task_id = None
                
                logger.info(f"Task {task.task_id} processed by agent {agent.agent_id} with result: {result.get('success', False)}")
            
            except asyncio.TimeoutError:
                # Handle timeout
                task.status = TaskStatus.TIMEOUT
                task.error_message = f"Task execution timed out after {DEFAULT_TIMEOUT} seconds"
                task.retry_count += 1
                
                # Retry if needed
                if task.retry_count < MAX_RETRIES:
                    task.status = TaskStatus.PENDING
                    self.global_task_queue.put((task.priority.value, time.time(), task))
                
                # Update agent status
                agent.status = AgentStatus.ERROR
                agent.current_task_id = None
                
                logger.error(f"Task {task.task_id} timed out for agent {agent.agent_id}")
            
            except Exception as e:
                # Handle other errors
                task.status = TaskStatus.FAILED
                task.error_message = str(e)
                
                # Update agent status
                agent.status = AgentStatus.ERROR
                agent.current_task_id = None
                
                logger.error(f"Error processing task {task.task_id} with agent {agent.agent_id}: {str(e)}")
        
        # Run in thread pool
        threading.Thread(target=run_task).start()
    
    def pipeline_monitoring_worker(self) -> None:
        """Worker thread for monitoring pipeline performance."""
        logger.info("Pipeline monitoring worker started")
        
        while not self.stop_event.is_set():
            try:
                # Update pipeline statistics
                for pipeline_id, pipeline in self.pipelines.items():
                    if not pipeline.active:
                        continue
                    
                    # Count active agents
                    active_agents = sum(1 for agent in pipeline.agents.values() if agent.status != AgentStatus.TERMINATED)
                    
                    # Update pipeline stats
                    self.pipeline_stats[pipeline_id] = {
                        "status": pipeline.status,
                        "active_agents": active_agents,
                        "tasks_completed": pipeline.tasks_completed,
                        "tasks_failed": pipeline.tasks_failed,
                        "avg_task_completion_time": pipeline.avg_task_completion_time,
                        "success_rate": pipeline.success_rate
                    }
            
            except Exception as e:
                logger.error(f"Error in pipeline monitoring worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(PIPELINE_STATS_UPDATE_INTERVAL)
    
    def agent_health_check_worker(self) -> None:
        """Worker thread for checking agent health and recovering from errors."""
        logger.info("Agent health check worker started")
        
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                for agent_id, agent in self.agents.items():
                    # Skip terminated agents
                    if agent.status == AgentStatus.TERMINATED:
                        continue
                    
                    # Check if agent is stuck
                    if agent.status in [AgentStatus.BUSY, AgentStatus.THINKING, AgentStatus.EXECUTING]:
                        time_since_active = (current_time - agent.last_active).total_seconds()
                        
                        if time_since_active > DEFAULT_TIMEOUT:
                            logger.warning(f"Agent {agent_id} appears stuck in status {agent.status.value} for {time_since_active} seconds")
                            
                            # Reset agent
                            agent.status = AgentStatus.IDLE
                            
                            # If agent was working on a task, mark it as failed
                            if agent.current_task_id and agent.current_task_id in self.tasks:
                                task = self.tasks[agent.current_task_id]
                                task.status = TaskStatus.FAILED
                                task.error_message = f"Agent {agent_id} became unresponsive"
                                task.retry_count += 1
                                
                                # Retry if needed
                                if task.retry_count < MAX_RETRIES:
                                    task.status = TaskStatus.PENDING
                                    self.global_task_queue.put((task.priority.value, time.time(), task))
                            
                            agent.current_task_id = None
                    
                    # Update agent stats
                    self.agent_stats[agent_id] = {
                        "status": agent.status.value,
                        "current_task_id": agent.current_task_id,
                        "total_tasks_completed": agent.total_tasks_completed,
                        "total_tokens_used": agent.total_tokens_used,
                        "success_rate": agent.success_rate,
                        "performance_metrics": agent.performance_metrics
                    }
            
            except Exception as e:
                logger.error(f"Error in agent health check worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(HEALTH_CHECK_INTERVAL)
    
    def stats_update_worker(self) -> None:
        """Worker thread for updating system statistics."""
        logger.info("Stats update worker started")
        
        while not self.stop_event.is_set():
            try:
                # Update active counts
                self.system_stats["active_pipelines"] = len([p for p in self.pipelines.values() if p.active])
                self.system_stats["active_agents"] = len([a for a in self.agents.values() if a.status != AgentStatus.TERMINATED])
                
                # Update token usage
                self.system_stats["total_tokens_used"] = sum(agent.total_tokens_used for agent in self.agents.values())
                
                # Update average task completion time
                if self.system_stats["total_tasks_completed"] > 0:
                    completed_pipelines = [p for p in self.pipelines.values() if p.tasks_completed > 0]
                    if completed_pipelines:
                        avg_time = sum(p.avg_task_completion_time * p.tasks_completed for p in completed_pipelines) / sum(p.tasks_completed for p in completed_pipelines)
                        self.system_stats["avg_task_completion_time"] = avg_time
                
                # Update overall success rate
                total_tasks = self.system_stats["total_tasks_completed"] + self.system_stats["total_tasks_failed"]
                if total_tasks > 0:
                    self.system_stats["overall_success_rate"] = self.system_stats["total_tasks_completed"] / total_tasks
            
            except Exception as e:
                logger.error(f"Error in stats update worker: {str(e)}")
            
            # Sleep to prevent CPU hogging
            time.sleep(PERFORMANCE_LOG_INTERVAL)
    
    def create_pipeline(
        self,
        name: str,
        type: PipelineType,
        max_agents: int = AGENTS_PER_PIPELINE,
        description: str = ""
    ) -> str:
        """Create a new pipeline."""
        if len(self.pipelines) >= self.max_pipelines:
            logger.warning("Cannot create pipeline: maximum pipelines reached")
            return ""
        
        pipeline_id = f"pipeline-{str(uuid.uuid4())[:8]}"
        
        pipeline = Pipeline(
            pipeline_id=pipeline_id,
            name=name,
            type=type,
            max_agents=max_agents,
            description=description
        )
        
        self.pipelines[pipeline_id] = pipeline
        logger.info(f"Created pipeline {pipeline_id} ({name})")
        
        return pipeline_id
    
    def create_agent(
        self,
        name: str,
        role: AgentRole,
        model_type: ModelType,
        pipeline_id: str,
        persona_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        capabilities: List[AgentCapability] = None
    ) -> str:
        """Create a new agent."""
        if pipeline_id not in self.pipelines:
            logger.warning(f"Cannot create agent: pipeline {pipeline_id} not found")
            return ""
        
        pipeline = self.pipelines[pipeline_id]
        if len(pipeline.agents) >= pipeline.max_agents:
            logger.warning(f"Cannot create agent: maximum agents reached for pipeline {pipeline_id}")
            return ""
        
        agent_id = f"agent-{str(uuid.uuid4())[:8]}"
        
        # Get persona if specified
        persona = None
        if persona_name and persona_name in self.persona_templates:
            persona = self.persona_templates[persona_name]
        
        agent = Agent(
            agent_id=agent_id,
            name=name,
            role=role,
            model_type=model_type,
            pipeline_id=pipeline_id,
            persona=persona,
            capabilities=capabilities or [],
            system_prompt=system_prompt
        )
        
        self.agents[agent_id] = agent
        pipeline.add_agent(agent)
        
        logger.info(f"Created agent {agent_id} ({name}) with role {role.value} in pipeline {pipeline_id}")
        
        return agent_id
    
    def create_task(
        self,
        description: str,
        type: TaskType,
        priority: AgentPriority = AgentPriority.MEDIUM,
        deadline: Optional[datetime] = None,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None,
        parent_task_id: Optional[str] = None
    ) -> str:
        """Create a new task."""
        task_id = f"task-{str(uuid.uuid4())[:8]}"
        
        task = Task(
            task_id=task_id,
            type=type,
            description=description,
            priority=priority,
            deadline=deadline,
            dependencies=dependencies or [],
            metadata=metadata or {},
            parent_task_id=parent_task_id
        )
        
        self.tasks[task_id] = task
        
        # Add to global task queue
        self.global_task_queue.put((priority.value, task.created_at.timestamp(), task))
        
        # If parent task exists, add this task as a subtask
        if parent_task_id and parent_task_id in self.tasks:
            parent_task = self.tasks[parent_task_id]
            parent_task.subtasks.append(task_id)
        
        logger.info(f"Created task {task_id} with type {type.value} and priority {priority.value}")
        
        return task_id
    
    def find_best_pipeline_for_task(self, task: Task) -> Optional[Pipeline]:
        """Find the best pipeline for a given task."""
        best_pipeline = None
        best_score = -1.0
        
        for pipeline in self.pipelines.values():
            if not pipeline.active:
                continue
            
            # Calculate match score
            score = self.calculate_pipeline_task_match(pipeline, task)
            
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
        
        return best_pipeline
    
    def calculate_pipeline_task_match(self, pipeline: Pipeline, task: Task) -> float:
        """Calculate how well a pipeline matches a task."""
        # Check if pipeline has available agents
        available_agents = [a for a in pipeline.agents.values() if a.status == AgentStatus.IDLE]
        if not available_agents:
            return 0.0
        
        # Check if pipeline has agents with matching role
        role_match = any(a.role.value == task.type.value for a in pipeline.agents.values())
        
        # Calculate average capability match
        capability_scores = []
        for agent in pipeline.agents.values():
            agent_score = 0.0
            for capability in agent.capabilities:
                match_score = capability.matches(task.description)
                agent_score = max(agent_score, match_score)
            capability_scores.append(agent_score)
        
        avg_capability_score = sum(capability_scores) / len(capability_scores) if capability_scores else 0.0
        
        # Calculate pipeline performance score
        performance_score = pipeline.success_rate
        
        # Calculate load score (inverse of queue size)
        queue_size = pipeline.task_queue.qsize()
        load_score = 1.0 / (1.0 + queue_size)
        
        # Combine scores
        total_score = (
            (2.0 if role_match else 0.0) +  # Role match is important
            (avg_capability_score * 3.0) +  # Capability match is most important
            (performance_score * 2.0) +     # Performance is important
            (load_score * 1.0)              # Load is least important
        ) / 8.0  # Normalize to 0-1 range
        
        return total_score
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get the status of a pipeline."""
        if pipeline_id not in self.pipelines:
            return {"error": f"Pipeline {pipeline_id} not found"}
        
        pipeline = self.pipelines[pipeline_id]
        
        return {
            "pipeline_id": pipeline_id,
            "name": pipeline.name,
            "type": pipeline.type.value,
            "status": pipeline.status,
            "active": pipeline.active,
            "agent_count": len(pipeline.agents),
            "tasks_completed": pipeline.tasks_completed,
            "tasks_failed": pipeline.tasks_failed,
            "success_rate": pipeline.success_rate,
            "avg_task_completion_time": pipeline.avg_task_completion_time
        }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get the status of an agent."""
        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}
        
        agent = self.agents[agent_id]
        
        return {
            "agent_id": agent_id,
            "name": agent.name,
            "role": agent.role.value,
            "model_type": agent.model_type.value,
            "pipeline_id": agent.pipeline_id,
            "status": agent.status.value,
            "current_task_id": agent.current_task_id,
            "total_tasks_completed": agent.total_tasks_completed,
            "total_tokens_used": agent.total_tokens_used,
            "success_rate": agent.success_rate,
            "performance_metrics": agent.performance_metrics
        }
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        if task_id not in self.tasks:
            return {"error": f"Task {task_id} not found"}
        
        task = self.tasks[task_id]
        
        return {
            "task_id": task_id,
            "type": task.type.value,
            "description": task.description,
            "status": task.status.value,
            "assigned_to": task.assigned_to,
            "created_at": task.created_at.isoformat(),
            "deadline": task.deadline.isoformat() if task.deadline else None,
            "progress": task.progress,
            "retry_count": task.retry_count,
            "error_message": task.error_message,
            "results": task.results
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "active_pipelines": self.system_stats["active_pipelines"],
            "active_agents": self.system_stats["active_agents"],
            "total_tasks_completed": self.system_stats["total_tasks_completed"],
            "total_tasks_failed": self.system_stats["total_tasks_failed"],
            "total_tokens_used": self.system_stats["total_tokens_used"],
            "avg_task_completion_time": self.system_stats["avg_task_completion_time"],
            "overall_success_rate": self.system_stats["overall_success_rate"]
        }
    
    def get_all_pipeline_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pipelines."""
        return self.pipeline_stats
    
    def get_all_agent_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all agents."""
        return self.agent_stats
    
    def create_bulk_agents(
        self,
        pipeline_id: str,
        count: int,
        role: AgentRole,
        model_type: ModelType,
        name_prefix: str = "Agent",
        persona_names: List[str] = None
    ) -> List[str]:
        """Create multiple agents at once."""
        if pipeline_id not in self.pipelines:
            logger.warning(f"Cannot create agents: pipeline {pipeline_id} not found")
            return []
        
        pipeline = self.pipelines[pipeline_id]
        available_slots = pipeline.max_agents - len(pipeline.agents)
        
        if count > available_slots:
            logger.warning(f"Can only create {available_slots} agents in pipeline {pipeline_id}")
            count = available_slots
        
        agent_ids = []
        
        for i in range(count):
            # Select persona if available
            persona_name = None
            if persona_names:
                persona_name = random.choice(persona_names) if persona_names else None
            
            agent_id = self.create_agent(
                name=f"{name_prefix}-{i+1:03d}",
                role=role,
                model_type=model_type,
                pipeline_id=pipeline_id,
                persona_name=persona_name
            )
            
            if agent_id:
                agent_ids.append(agent_id)
        
        logger.info(f"Created {len(agent_ids)} agents in pipeline {pipeline_id}")
        return agent_ids
    
    def create_swarm(
        self,
        name: str,
        num_pipelines: int,
        agents_
