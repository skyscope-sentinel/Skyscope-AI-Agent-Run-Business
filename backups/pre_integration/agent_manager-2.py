import os
import json
import uuid
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path
import numpy as np

# Try to import Swarms framework
try:
    from swarms import Agent, SequentialWorkflow, ConcurrentWorkflow, MixtureOfAgents
    from swarms import GroupChat, ForestSwarm, SpreadSheetSwarm, SwarmRouter, SwarmType
    from swarms.structs.agent_rearrange import AgentRearrange
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False
    logging.warning("Swarms framework not available. Some features will be disabled.")

# Try to import vector store for knowledge management
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logging.warning("Vector store components not available. Knowledge management will be limited.")

# Import local modules
try:
    from config import get_config, get, set, get_api_key
except ImportError:
    # Fallback for testing
    def get_config(): return {}
    def get(key_path, default=None): return default
    def set(key_path, value): return True
    def get_api_key(provider): return ""

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agent_manager")

class AgentRole(Enum):
    """Enumeration of agent roles"""
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
    """Enumeration of swarm patterns"""
    SEQUENTIAL = "sequential"
    CONCURRENT = "concurrent"
    MIXTURE_OF_AGENTS = "mixture_of_agents"
    GROUP_CHAT = "group_chat"
    FOREST = "forest"
    SPREADSHEET = "spreadsheet"
    AGENT_REARRANGE = "agent_rearrange"
    ROUTER = "router"
    # New advanced collaboration modes
    HIERARCHICAL = "hierarchical"      # Tree-structured team / sub-team workflow
    GRAPH = "graph"                    # Arbitrary DAG-based workflow

class TaskStatus(Enum):
    """Enumeration of task statuses"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    """Enumeration of task priorities"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class AgentTask:
    """Represents a task to be performed by an agent or swarm"""
    
    def __init__(
        self,
        task_id: Optional[str] = None,
        description: str = "",
        input_data: Any = None,
        assigned_to: Optional[Union[str, List[str]]] = None,
        depends_on: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        deadline: Optional[datetime] = None,
        max_retries: int = 3,
        status: TaskStatus = TaskStatus.PENDING,
        result: Any = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a task
        
        Args:
            task_id: Unique identifier for the task
            description: Human-readable description of the task
            input_data: Input data for the task
            assigned_to: Agent or swarm assigned to the task
            depends_on: List of task IDs that must be completed before this task
            priority: Priority of the task
            deadline: Deadline for task completion
            max_retries: Maximum number of retries if the task fails
            status: Current status of the task
            result: Result of the task (if completed)
            error: Error message (if failed)
            metadata: Additional metadata for the task
        """
        self.task_id = task_id or str(uuid.uuid4())
        self.description = description
        self.input_data = input_data
        self.assigned_to = assigned_to
        self.depends_on = depends_on or []
        self.priority = priority
        self.deadline = deadline
        self.max_retries = max_retries
        self.status = status
        self.result = result
        self.error = error
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.started_at = None
        self.completed_at = None
        self.retry_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
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
            "retry_count": self.retry_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentTask':
        """Create task from dictionary"""
        task = cls(
            task_id=data["task_id"],
            description=data["description"],
            assigned_to=data["assigned_to"],
            depends_on=data["depends_on"],
            priority=TaskPriority(data["priority"]),
            max_retries=data["max_retries"],
            status=TaskStatus(data["status"]),
            error=data["error"],
            metadata=data["metadata"]
        )
        
        task.created_at = datetime.fromisoformat(data["created_at"])
        task.updated_at = datetime.fromisoformat(data["updated_at"])
        
        if data["started_at"]:
            task.started_at = datetime.fromisoformat(data["started_at"])
        
        if data["completed_at"]:
            task.completed_at = datetime.fromisoformat(data["completed_at"])
        
        task.retry_count = data["retry_count"]
        
        if data["deadline"]:
            task.deadline = datetime.fromisoformat(data["deadline"])
        
        return task
    
    def update_status(self, status: TaskStatus):
        """Update task status"""
        self.status = status
        self.updated_at = datetime.now()
        
        if status == TaskStatus.IN_PROGRESS and not self.started_at:
            self.started_at = datetime.now()
        
        if status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            self.completed_at = datetime.now()
    
    def increment_retry(self):
        """Increment retry count"""
        self.retry_count += 1
        self.updated_at = datetime.now()
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries

class AgentMemory:
    """Memory system for agents to store and retrieve information"""
    
    def __init__(self, memory_path: Optional[str] = None):
        """
        Initialize agent memory
        
        Args:
            memory_path: Path to store memory data
        """
        self.memory_path = memory_path or get("swarms.memory_path", "data/memory")
        self.short_term_memory = []
        self.long_term_memory = {}
        self.vector_store = None
        self.embedding_model = None
        
        # Initialize vector store if available
        if VECTOR_STORE_AVAILABLE:
            self._init_vector_store()
        
        # Create memory directory if it doesn't exist
        os.makedirs(self.memory_path, exist_ok=True)
        
        # Load existing memory if available
        self._load_memory()
    
    def _init_vector_store(self):
        """Initialize vector store for semantic search"""
        try:
            # Initialize ChromaDB
            self.vector_store = chromadb.PersistentClient(path=os.path.join(self.memory_path, "vector_db"))
            
            # Create or get collection
            self.collection = self.vector_store.get_or_create_collection("agent_memory")
            
            # Initialize embedding model
            model_name = get("knowledge_stack.embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            
            logger.info(f"Vector store initialized with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self.vector_store = None
            self.embedding_model = None
    
    def _load_memory(self):
        """Load memory from disk"""
        try:
            memory_file = os.path.join(self.memory_path, "long_term_memory.json")
            if os.path.exists(memory_file):
                with open(memory_file, "r") as f:
                    self.long_term_memory = json.load(f)
                logger.info(f"Loaded {len(self.long_term_memory)} memory items from disk")
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def _save_memory(self):
        """Save memory to disk"""
        try:
            memory_file = os.path.join(self.memory_path, "long_term_memory.json")
            with open(memory_file, "w") as f:
                json.dump(self.long_term_memory, f, indent=2)
            logger.info(f"Saved {len(self.long_term_memory)} memory items to disk")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def add(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an item to memory
        
        Args:
            key: Key to identify the memory item
            value: Value to store
            metadata: Additional metadata for the memory item
        """
        metadata = metadata or {}
        timestamp = datetime.now().isoformat()
        
        # Add to short-term memory
        self.short_term_memory.append({
            "key": key,
            "value": value,
            "metadata": metadata,
            "timestamp": timestamp
        })
        
        # Limit short-term memory size
        max_items = get("swarms.memory.short_term_limit", 100)
        if len(self.short_term_memory) > max_items:
            self.short_term_memory = self.short_term_memory[-max_items:]
        
        # Add to long-term memory
        self.long_term_memory[key] = {
            "value": value,
            "metadata": metadata,
            "timestamp": timestamp,
            "access_count": 0,
            "last_accessed": None
        }
        
        # Add to vector store if available
        if self.vector_store is not None and self.embedding_model is not None:
            try:
                # Convert value to string if it's not already
                if not isinstance(value, str):
                    if isinstance(value, dict) or isinstance(value, list):
                        text_value = json.dumps(value)
                    else:
                        text_value = str(value)
                else:
                    text_value = value
                
                # Add to vector store
                self.collection.add(
                    documents=[text_value],
                    metadatas=[{"key": key, **metadata}],
                    ids=[key]
                )
            except Exception as e:
                logger.error(f"Error adding to vector store: {e}")
        
        # Save memory periodically
        if len(self.long_term_memory) % 10 == 0:
            self._save_memory()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from memory
        
        Args:
            key: Key to identify the memory item
            
        Returns:
            The memory item or None if not found
        """
        if key in self.long_term_memory:
            # Update access statistics
            self.long_term_memory[key]["access_count"] += 1
            self.long_term_memory[key]["last_accessed"] = datetime.now().isoformat()
            
            return self.long_term_memory[key]["value"]
        
        return None
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search memory using semantic similarity
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of memory items matching the query
        """
        if self.vector_store is None or self.embedding_model is None:
            # Fallback to simple keyword search
            results = []
            for key, item in self.long_term_memory.items():
                value = item["value"]
                if isinstance(value, str) and query.lower() in value.lower():
                    results.append({
                        "key": key,
                        "value": value,
                        "metadata": item["metadata"],
                        "timestamp": item["timestamp"],
                        "score": 0.5  # Arbitrary score for keyword match
                    })
            
            # Sort by recency
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results[:limit]
        
        try:
            # Query vector store
            query_results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            results = []
            for i, document_id in enumerate(query_results["ids"][0]):
                if document_id in self.long_term_memory:
                    results.append({
                        "key": document_id,
                        "value": self.long_term_memory[document_id]["value"],
                        "metadata": self.long_term_memory[document_id]["metadata"],
                        "timestamp": self.long_term_memory[document_id]["timestamp"],
                        "score": float(query_results["distances"][0][i]) if "distances" in query_results else 1.0
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def get_recent(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent memory items
        
        Args:
            limit: Maximum number of items
            
        Returns:
            List of recent memory items
        """
        return self.short_term_memory[-limit:]
    
    def clear_short_term(self):
        """Clear short-term memory"""
        self.short_term_memory = []
    
    def delete(self, key: str) -> bool:
        """
        Delete an item from memory
        
        Args:
            key: Key to identify the memory item
            
        Returns:
            True if the item was deleted, False otherwise
        """
        if key in self.long_term_memory:
            del self.long_term_memory[key]
            
            # Delete from vector store if available
            if self.vector_store is not None:
                try:
                    self.collection.delete(ids=[key])
                except Exception as e:
                    logger.error(f"Error deleting from vector store: {e}")
            
            return True
        
        return False
    
    def summarize(self) -> Dict[str, Any]:
        """
        Summarize memory statistics
        
        Returns:
            Dictionary of memory statistics
        """
        return {
            "short_term_count": len(self.short_term_memory),
            "long_term_count": len(self.long_term_memory),
            "vector_store_available": self.vector_store is not None,
            "memory_path": self.memory_path
        }

class KnowledgeManager:
    """Manager for shared knowledge between agents"""
    
    def __init__(self, vector_db_path: Optional[str] = None):
        """
        Initialize knowledge manager
        
        Args:
            vector_db_path: Path to vector database
        """
        self.vector_db_path = vector_db_path or get("knowledge_stack.vector_db_path", "data/vector_db")
        self.knowledge_items = {}
        self.vector_store = None
        self.embedding_model = None
        
        # Initialize vector store if available
        if VECTOR_STORE_AVAILABLE:
            self._init_vector_store()
        
        # Create knowledge directory if it doesn't exist
        os.makedirs(self.vector_db_path, exist_ok=True)
    
    def _init_vector_store(self):
        """Initialize vector store for semantic search"""
        try:
            # Initialize ChromaDB
            self.vector_store = chromadb.PersistentClient(path=self.vector_db_path)
            
            # Create or get collection
            self.collection = self.vector_store.get_or_create_collection("knowledge_stack")
            
            # Initialize embedding model
            model_name = get("knowledge_stack.embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            
            logger.info(f"Knowledge vector store initialized with model {model_name}")
        except Exception as e:
            logger.error(f"Error initializing knowledge vector store: {e}")
            self.vector_store = None
            self.embedding_model = None
    
    def add_item(self, title: str, content: str, source: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an item to the knowledge stack
        
        Args:
            title: Title of the knowledge item
            content: Content of the knowledge item
            source: Source of the knowledge item
            metadata: Additional metadata for the knowledge item
        """
        item_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        metadata = metadata or {}
        if source:
            metadata["source"] = source
        
        # Add to knowledge items
        self.knowledge_items[item_id] = {
            "id": item_id,
            "title": title,
            "content": content,
            "metadata": metadata,
            "timestamp": timestamp
        }
        
        # Add to vector store if available
        if self.vector_store is not None and self.embedding_model is not None:
            try:
                self.collection.add(
                    documents=[content],
                    metadatas=[{"id": item_id, "title": title, **metadata}],
                    ids=[item_id]
                )
            except Exception as e:
                logger.error(f"Error adding to knowledge vector store: {e}")
        
        return item_id
    
    def get_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an item from the knowledge stack
        
        Args:
            item_id: ID of the knowledge item
            
        Returns:
            The knowledge item or None if not found
        """
        return self.knowledge_items.get(item_id)
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge stack
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of knowledge items matching the query
        """
        if self.vector_store is None or self.embedding_model is None:
            # Fallback to simple keyword search
            results = []
            for item_id, item in self.knowledge_items.items():
                if query.lower() in item["title"].lower() or query.lower() in item["content"].lower():
                    results.append(item)
            
            # Sort by recency
            results.sort(key=lambda x: x["timestamp"], reverse=True)
            return results[:limit]
        
        try:
            # Query vector store
            query_results = self.collection.query(
                query_texts=[query],
                n_results=limit
            )
            
            results = []
            for i, document_id in enumerate(query_results["ids"][0]):
                if document_id in self.knowledge_items:
                    item = self.knowledge_items[document_id].copy()
                    item["score"] = float(query_results["distances"][0][i]) if "distances" in query_results else 1.0
                    results.append(item)
            
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge vector store: {e}")
            return []
    
    def get_all_items(self) -> List[Dict[str, Any]]:
        """
        Get all knowledge items
        
        Returns:
            List of all knowledge items
        """
        return list(self.knowledge_items.values())
    
    def delete_item(self, item_id: str) -> bool:
        """
        Delete an item from the knowledge stack
        
        Args:
            item_id: ID of the knowledge item
            
        Returns:
            True if the item was deleted, False otherwise
        """
        if item_id in self.knowledge_items:
            del self.knowledge_items[item_id]
            
            # Delete from vector store if available
            if self.vector_store is not None:
                try:
                    self.collection.delete(ids=[item_id])
                except Exception as e:
                    logger.error(f"Error deleting from knowledge vector store: {e}")
            
            return True
        
        return False
    
    def clear(self):
        """Clear the knowledge stack"""
        self.knowledge_items = {}
        
        # Clear vector store if available
        if self.vector_store is not None:
            try:
                self.collection.delete(ids=list(self.knowledge_items.keys()))
            except Exception as e:
                logger.error(f"Error clearing knowledge vector store: {e}")

class ToolRegistry:
    """Registry for tools that can be used by agents"""
    
    def __init__(self):
        """Initialize tool registry"""
        self.tools = {}
    
    def register_tool(self, name: str, func: Callable, description: str, parameters: Optional[Dict[str, Any]] = None):
        """
        Register a tool
        
        Args:
            name: Name of the tool
            func: Function to call when the tool is used
            description: Description of the tool
            parameters: Parameters for the tool
        """
        self.tools[name] = {
            "name": name,
            "func": func,
            "description": description,
            "parameters": parameters or {}
        }
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool or None if not found
        """
        return self.tools.get(name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools
        
        Returns:
            List of tools
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """
        Execute a tool
        
        Args:
            name: Name of the tool
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Result of the tool execution
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool {name} not found")
        
        return tool["func"](**kwargs)

class AgentFactory:
    """Factory for creating agents"""
    
    def __init__(self, memory: Optional[AgentMemory] = None, knowledge_manager: Optional[KnowledgeManager] = None, tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize agent factory
        
        Args:
            memory: Shared memory for agents
            knowledge_manager: Knowledge manager for agents
            tool_registry: Tool registry for agents
        """
        self.memory = memory or AgentMemory()
        self.knowledge_manager = knowledge_manager or KnowledgeManager()
        self.tool_registry = tool_registry or ToolRegistry()
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools"""
        # Memory tools
        self.tool_registry.register_tool(
            name="memory_add",
            func=lambda key, value, metadata=None: self.memory.add(key, value, metadata),
            description="Add an item to memory",
            parameters={
                "key": "Key to identify the memory item",
                "value": "Value to store",
                "metadata": "Additional metadata for the memory item (optional)"
            }
        )
        
        self.tool_registry.register_tool(
            name="memory_get",
            func=lambda key: self.memory.get(key),
            description="Get an item from memory",
            parameters={
                "key": "Key to identify the memory item"
            }
        )
        
        self.tool_registry.register_tool(
            name="memory_search",
            func=lambda query, limit=5: self.memory.search(query, limit),
            description="Search memory using semantic similarity",
            parameters={
                "query": "Search query",
                "limit": "Maximum number of results (default: 5)"
            }
        )
        
        # Knowledge tools
        self.tool_registry.register_tool(
            name="knowledge_add",
            func=lambda title, content, source=None, metadata=None: self.knowledge_manager.add_item(title, content, source, metadata),
            description="Add an item to the knowledge stack",
            parameters={
                "title": "Title of the knowledge item",
                "content": "Content of the knowledge item",
                "source": "Source of the knowledge item (optional)",
                "metadata": "Additional metadata for the knowledge item (optional)"
            }
        )
        
        self.tool_registry.register_tool(
            name="knowledge_search",
            func=lambda query, limit=5: self.knowledge_manager.search(query, limit),
            description="Search the knowledge stack",
            parameters={
                "query": "Search query",
                "limit": "Maximum number of results (default: 5)"
            }
        )
        
        # File tools
        self.tool_registry.register_tool(
            name="read_file",
            func=lambda file_path: self._read_file(file_path),
            description="Read a file from the filesystem",
            parameters={
                "file_path": "Path to the file"
            }
        )
        
        self.tool_registry.register_tool(
            name="write_file",
            func=lambda file_path, content: self._write_file(file_path, content),
            description="Write to a file on the filesystem",
            parameters={
                "file_path": "Path to the file",
                "content": "Content to write"
            }
        )
    
    def _read_file(self, file_path: str) -> str:
        """
        Read a file from the filesystem
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content of the file
        """
        if not get("tools.filesystem_access.enabled", False):
            raise PermissionError("Filesystem access is disabled")
        
        # Check if the file path is allowed
        allowed_dirs = get("tools.filesystem_access.allowed_directories", [])
        allowed = False
        
        file_path = os.path.abspath(file_path)
        for allowed_dir in allowed_dirs:
            allowed_path = os.path.abspath(allowed_dir)
            if file_path.startswith(allowed_path):
                allowed = True
                break
        
        if not allowed:
            raise PermissionError(f"Access to {file_path} is not allowed")
        
        with open(file_path, "r") as f:
            return f.read()
    
    def _write_file(self, file_path: str, content: str) -> bool:
        """
        Write to a file on the filesystem
        
        Args:
            file_path: Path to the file
            content: Content to write
            
        Returns:
            True if the file was written successfully
        """
        if not get("tools.filesystem_access.enabled", False):
            raise PermissionError("Filesystem access is disabled")
        
        # Check if the file path is allowed
        allowed_dirs = get("tools.filesystem_access.allowed_directories", [])
        allowed = False
        
        file_path = os.path.abspath(file_path)
        for allowed_dir in allowed_dirs:
            allowed_path = os.path.abspath(allowed_dir)
            if file_path.startswith(allowed_path):
                allowed = True
                break
        
        if not allowed:
            raise PermissionError(f"Access to {file_path} is not allowed")
        
        # Check if the file extension is allowed
        allowed_exts = get("tools.filesystem_access.allowed_extensions", [])
        if allowed_exts:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in allowed_exts:
                raise PermissionError(f"File extension {ext} is not allowed")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
        
        return True
    
    def create_agent(
        self,
        agent_name: str,
        role: AgentRole,
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Create an agent
        
        Args:
            agent_name: Name of the agent
            role: Role of the agent
            system_prompt: System prompt for the agent
            model_name: Model to use for the agent
            tools: List of tool names to enable for the agent
            **kwargs: Additional arguments for the agent
            
        Returns:
            The created agent
        """
        if not SWARMS_AVAILABLE:
            raise ImportError("Swarms framework is not available")
        
        # Get default model if not specified
        model_name = model_name or get("models.default_model")
        
        # Get default system prompt if not specified
        if system_prompt is None:
            system_prompt = self._get_role_prompt(role)
        
        # Create the agent
        agent = Agent(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            **kwargs
        )
        
        # Add tools
        if tools:
            for tool_name in tools:
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    agent.add_tool(tool_name, tool["description"])
        
        return agent
    
    def _get_role_prompt(self, role: AgentRole) -> str:
        """
        Get system prompt for a role
        
        Args:
            role: Role of the agent
            
        Returns:
            System prompt for the role
        """
        base_prompt = get("prompts.system")
        
        role_prompts = {
            AgentRole.COORDINATOR: """You are a Coordinator agent responsible for managing and orchestrating other agents.
Your primary tasks include:
1. Breaking down complex problems into smaller tasks
2. Assigning tasks to appropriate specialized agents
3. Monitoring progress and ensuring task completion
4. Integrating results from different agents
5. Making high-level decisions about the workflow

Always maintain a clear overview of the entire process and ensure that all agents are working effectively towards the common goal.""",
            
            AgentRole.RESEARCHER: """You are a Researcher agent specialized in gathering and analyzing information.
Your primary tasks include:
1. Searching for relevant information on a given topic
2. Evaluating the credibility and relevance of sources
3. Extracting key insights and data points
4. Organizing information in a structured format
5. Identifying gaps in knowledge that require further investigation

Be thorough, objective, and critical in your research approach.""",
            
            AgentRole.ANALYST: """You are an Analyst agent specialized in data analysis and interpretation.
Your primary tasks include:
1. Processing and cleaning data for analysis
2. Applying statistical methods to identify patterns and trends
3. Creating visualizations to represent data insights
4. Drawing conclusions based on data evidence
5. Providing actionable recommendations based on analysis

Be precise, methodical, and data-driven in your approach.""",
            
            AgentRole.WRITER: """You are a Writer agent specialized in creating high-quality written content.
Your primary tasks include:
1. Drafting clear and engaging text on various topics
2. Adapting writing style to different audiences and purposes
3. Structuring content for maximum readability and impact
4. Editing and refining text for grammar, clarity, and flow
5. Incorporating feedback to improve written materials

Be creative, clear, and audience-focused in your writing.""",
            
            AgentRole.CODER: """You are a Coder agent specialized in software development.
Your primary tasks include:
1. Writing clean, efficient, and well-documented code
2. Implementing algorithms and data structures
3. Debugging and fixing issues in existing code
4. Optimizing code for performance and readability
5. Following best practices and coding standards

Be precise, logical, and solution-oriented in your approach.""",
            
            AgentRole.REVIEWER: """You are a Reviewer agent specialized in evaluating and improving work.
Your primary tasks include:
1. Critically assessing the quality and accuracy of content
2. Identifying errors, inconsistencies, and areas for improvement
3. Providing constructive feedback with specific suggestions
4. Ensuring adherence to standards and requirements
5. Verifying that all objectives have been met

Be thorough, fair, and constructive in your reviews.""",
            
            AgentRole.PLANNER: """You are a Planner agent specialized in strategic planning and organization.
Your primary tasks include:
1. Developing comprehensive project plans with clear milestones
2. Identifying required resources and potential constraints
3. Creating timelines and schedules for task completion
4. Anticipating risks and developing contingency plans
5. Aligning plans with overall objectives and priorities

Be strategic, detail-oriented, and forward-thinking in your planning.""",
            
            AgentRole.EXECUTOR: """You are an Executor agent specialized in implementing plans and completing tasks.
Your primary tasks include:
1. Following instructions precisely to complete assigned work
2. Managing time and resources efficiently
3. Adapting to changing circumstances while maintaining focus
4. Reporting progress and identifying obstacles
5. Ensuring high-quality execution of all tasks

Be reliable, efficient, and results-oriented in your approach.""",
            
            AgentRole.BROWSER: """You are a Browser agent specialized in web navigation and information retrieval.
Your primary tasks include:
1. Navigating websites and web applications
2. Extracting specific information from web pages
3. Filling forms and interacting with web elements
4. Following links and exploring web content
5. Capturing screenshots and saving web content

Be precise, thorough, and security-conscious in your web interactions.""",
            
            AgentRole.QUANTUM: """You are a Quantum Computing agent specialized in quantum algorithms and simulations.
Your primary tasks include:
1. Translating classical problems into quantum frameworks
2. Designing and implementing quantum algorithms
3. Simulating quantum systems and interpreting results
4. Optimizing quantum circuits for efficiency
5. Explaining quantum concepts and results in accessible terms

Be innovative, precise, and scientifically rigorous in your approach.""",
            
            AgentRole.CREATIVE: """You are a Creative agent specialized in generating innovative ideas and content.
Your primary tasks include:
1. Brainstorming original concepts and approaches
2. Developing creative solutions to complex problems
3. Generating engaging and imaginative content
4. Thinking outside conventional boundaries
5. Adapting creative output to specific requirements

Be imaginative, original, and bold in your creative thinking.""",
            
            AgentRole.DATA_SCIENTIST: """You are a Data Scientist agent specialized in advanced data analysis and modeling.
Your primary tasks include:
1. Applying machine learning algorithms to extract insights
2. Building predictive models and evaluating their performance
3. Processing and transforming complex datasets
4. Interpreting results and communicating findings
5. Recommending data-driven strategies and solutions

Be analytical, methodical, and innovative in your data science approach.""",
            
            AgentRole.SECURITY: """You are a Security agent specialized in identifying and addressing security concerns.
Your primary tasks include:
1. Evaluating code and systems for potential vulnerabilities
2. Recommending security best practices and improvements
3. Ensuring compliance with security standards and regulations
4. Detecting and responding to security threats
5. Implementing security measures and controls

Be vigilant, thorough, and proactive in your security approach.""",
            
            AgentRole.DEBUGGER: """You are a Debugger agent specialized in identifying and fixing issues.
Your primary tasks include:
1. Analyzing error messages and unexpected behaviors
2. Tracing code execution to locate problems
3. Developing and testing potential solutions
4. Implementing fixes and verifying their effectiveness
5. Documenting issues and solutions for future reference

Be systematic, persistent, and detail-oriented in your debugging approach.""",
            
            AgentRole.TESTER: """You are a Tester agent specialized in quality assurance and validation.
Your primary tasks include:
1. Designing and executing test cases to verify functionality
2. Identifying edge cases and potential failure scenarios
3. Reporting bugs and issues with clear reproduction steps
4. Validating that requirements have been met
5. Ensuring overall quality and reliability

Be methodical, thorough, and quality-focused in your testing approach.""",
            
            AgentRole.UI_DESIGNER: """You are a UI Designer agent specialized in creating user interfaces.
Your primary tasks include:
1. Designing intuitive and visually appealing interfaces
2. Creating wireframes and mockups for user interfaces
3. Applying design principles and best practices
4. Ensuring consistency and accessibility in design
5. Translating user needs into effective interface elements

Be user-centered, creative, and detail-oriented in your design approach.""",
            
            AgentRole.DOMAIN_EXPERT: """You are a Domain Expert agent with specialized knowledge in a particular field.
Your primary tasks include:
1. Providing accurate and detailed information about your domain
2. Explaining complex concepts in accessible terms
3. Applying domain-specific methodologies and best practices
4. Evaluating work from a domain expertise perspective
5. Staying current with developments in your field

Be knowledgeable, precise, and authoritative in your domain expertise.""",
            
            AgentRole.SUMMARIZER: """You are a Summarizer agent specialized in condensing and synthesizing information.
Your primary tasks include:
1. Extracting key points from lengthy content
2. Identifying the most important information and main themes
3. Creating concise summaries that retain essential meaning
4. Adapting summary length and detail to specific needs
5. Organizing information in a clear and logical structure

Be concise, accurate, and focused on the most relevant information.""",
            
            AgentRole.CRITIC: """You are a Critic agent specialized in providing constructive criticism.
Your primary tasks include:
1. Evaluating work against objective quality criteria
2. Identifying strengths and weaknesses in a balanced manner
3. Providing specific and actionable feedback for improvement
4. Challenging assumptions and identifying logical flaws
5. Suggesting alternative approaches and solutions

Be honest, fair, and constructive in your critical analysis.""",
            
            AgentRole.INTEGRATOR: """You are an Integrator agent specialized in combining and synthesizing diverse inputs.
Your primary tasks include:
1. Merging contributions from multiple sources or agents
2. Resolving conflicts and inconsistencies between different inputs
3. Creating cohesive and unified outputs from disparate elements
4. Ensuring consistency in style, format, and content
5. Preserving the value of all contributions while creating a coherent whole

Be collaborative, balanced, and synthesis-oriented in your approach."""
        }
        
        role_prompt = role_prompts.get(role, "")
        
        if role_prompt:
            return f"{base_prompt}\n\n{role_prompt}"
        
        return base_prompt
    
    def create_swarm(
        self,
        swarm_name: str,
        swarm_type: SwarmPattern,
        agents: List[Any],
        **kwargs
    ) -> Any:
        """
        Create a swarm
        
        Args:
            swarm_name: Name of the swarm
            swarm_type: Type of swarm
            agents: List of agents in the swarm
            **kwargs: Additional arguments for the swarm
            
        Returns:
            The created swarm
        """
        if not SWARMS_AVAILABLE:
            raise ImportError("Swarms framework is not available")
        
        # Create the swarm based on the type
        if swarm_type == SwarmPattern.SEQUENTIAL:
            return SequentialWorkflow(
                name=swarm_name,
                agents=agents,
                **kwargs
            )
        
        elif swarm_type == SwarmPattern.CONCURRENT:
            return ConcurrentWorkflow(
                name=swarm_name,
                agents=agents,
                **kwargs
            )
        
        elif swarm_type == SwarmPattern.MIXTURE_OF_AGENTS:
            # Extract aggregator agent if provided
            aggregator_agent = kwargs.pop("aggregator_agent", None)
            
            if aggregator_agent is None:
                # Create a default aggregator agent
                aggregator_agent = self.create_agent(
                    agent_name=f"{swarm_name}_aggregator",
                    role=AgentRole.INTEGRATOR,
                    system_prompt=f"You are the aggregator for the {swarm_name} swarm. Your job is to synthesize the outputs from multiple expert agents into a coherent and comprehensive response.",
                    model_name=get("models.default_model")
                )
            
            return MixtureOfAgents(
                name=swarm_name,
                agents=agents,
                aggregator_agent=aggregator_agent,
                **kwargs
            )
        
        elif swarm_type == SwarmPattern.GROUP_CHAT:
            return GroupChat(
                name=swarm_name,
                agents=agents,
                **kwargs
            )
        
        elif swarm_type == SwarmPattern.FOREST:
            return ForestSwarm(
                name=swarm_name,
                agents=agents,
                **kwargs
            )
        
        elif swarm_type == SwarmPattern.SPREADSHEET:
            return SpreadSheetSwarm(
                name=swarm_name,
                agents=agents,
                **kwargs
            )
        
        # --- Hierarchical Swarm (tree structure) -------------------------
        # We leverage ForestSwarm (already imported) which organises agents
        # in a hierarchical tree – ideal for team / sub-team delegation.
        elif swarm_type == SwarmPattern.HIERARCHICAL:
            return ForestSwarm(
                name=swarm_name,
                agents=agents,
                **kwargs
            )

        # --- Graph-Based Swarm -------------------------------------------
        # A flexible non-linear DAG using AgentRearrange under the hood.
        elif swarm_type == SwarmPattern.GRAPH:
            graph_flow = kwargs.pop("graph_flow", None)
            if graph_flow is None:
                # Fallback: fully connected chain
                node_names = [ag.agent_name for ag in agents]
                graph_flow = " , ".join(node_names)
            return AgentRearrange(
                name=swarm_name,
                agents=agents,
                flow=graph_flow,
                **kwargs
            )

        elif swarm_type == SwarmPattern.AGENT_REARRANGE:
            # Extract flow if provided
            flow = kwargs.pop("flow", None)
            
            if flow is None:
                # Create a default sequential flow
                agent_names = [agent.agent_name for agent in agents]
                flow = " -> ".join(agent_names)
            
            return AgentRearrange(
                name=swarm_name,
                agents=agents,
                flow=flow,
                **kwargs
            )
        
        elif swarm_type == SwarmPattern.ROUTER:
            # Map SwarmPattern to SwarmType
            pattern_to_type = {
                SwarmPattern.SEQUENTIAL: SwarmType.SequentialWorkflow,
                SwarmPattern.CONCURRENT: SwarmType.ConcurrentWorkflow,
                SwarmPattern.MIXTURE_OF_AGENTS: SwarmType.MixtureOfAgents,
                SwarmPattern.GROUP_CHAT: SwarmType.GroupChat,
                SwarmPattern.FOREST: SwarmType.ForestSwarm,
                SwarmPattern.SPREADSHEET: SwarmType.SpreadsheetSwarm,
                SwarmPattern.AGENT_REARRANGE: SwarmType.AgentRearrange
            }
            
            # Extract default_swarm_type if provided
            default_swarm_type = kwargs.pop("default_swarm_type", SwarmPattern.SEQUENTIAL)
            swarm_type_enum = pattern_to_type.get(default_swarm_type, SwarmType.SequentialWorkflow)
            
            return SwarmRouter(
                name=swarm_name,
                swarm_type=swarm_type_enum,
                agents=agents,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unknown swarm type: {swarm_type}")

class Pipeline:
    """Represents a workflow pipeline of tasks"""
    
    def __init__(
        self,
        pipeline_id: Optional[str] = None,
        name: str = "",
        description: str = "",
        tasks: Optional[List[AgentTask]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a pipeline
        
        Args:
            pipeline_id: Unique identifier for the pipeline
            name: Name of the pipeline
            description: Description of the pipeline
            tasks: List of tasks in the pipeline
            metadata: Additional metadata for the pipeline
        """
        self.pipeline_id = pipeline_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.tasks = tasks or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.status = "pending"
        self.result = None
        self.error = None
    
    def add_task(self, task: AgentTask):
        """
        Add a task to the pipeline
        
        Args:
            task: Task to add
        """
        self.tasks.append(task)
        self.updated_at = datetime.now()
    
    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """
        Get a task by ID
        
        Args:
            task_id: ID of the task
            
        Returns:
            The task or None if not found
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        
        return None
    
    def get_next_tasks(self) -> List[AgentTask]:
        """
        Get the next tasks that are ready to be executed
        
        Returns:
            List of tasks ready to be executed
        """
        ready_tasks = []
        
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                dependencies_completed = True
                
                for dep_id in task.depends_on:
                    dep_task = self.get_task(dep_id)
                    if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                        dependencies_completed = False
                        break
                
                if dependencies_completed:
                    ready_tasks.append(task)
        
        # Sort by priority (higher priority first)
        ready_tasks.sort(key=lambda t: t.priority.value, reverse=True)
        
        return ready_tasks
    
    def is_completed(self) -> bool:
        """
        Check if the pipeline is completed
        
        Returns:
            True if all tasks are completed, failed, or cancelled
        """
        for task in self.tasks:
            if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False
        
        return True
    
    def update_status(self):
        """Update the status of the pipeline based on task statuses"""
        if not self.tasks:
            self.status = "pending"
            return
        
        # Check if any tasks are failed
        failed_tasks = [task for task in self.tasks if task.status == TaskStatus.FAILED]
        if failed_tasks:
            self.status = "failed"
            self.error = failed_tasks[0].error
            return
        
        # Check if all tasks are completed
        if all(task.status == TaskStatus.COMPLETED for task in self.tasks):
            self.status = "completed"
            return
        
        # Check if all tasks are cancelled
        if all(task.status == TaskStatus.CANCELLED for task in self.tasks):
            self.status = "cancelled"
            return
        
        # Check if any tasks are in progress
        if any(task.status == TaskStatus.IN_PROGRESS for task in self.tasks):
            self.status = "in_progress"
            return
        
        # Check if any tasks are assigned
        if any(task.status == TaskStatus.ASSIGNED for task in self.tasks):
            self.status = "in_progress"
            return
        
        # Default to pending
        self.status = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary"""
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status,
            "result": self.result,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pipeline':
        """Create pipeline from dictionary"""
        pipeline = cls(
            pipeline_id=data["pipeline_id"],
            name=data["name"],
            description=data["description"],
            tasks=[AgentTask.from_dict(task_data) for task_data in data["tasks"]],
            metadata=data["metadata"]
        )
        
        pipeline.created_at = datetime.fromisoformat(data["created_at"])
        pipeline.updated_at = datetime.fromisoformat(data["updated_at"])
        pipeline.status = data["status"]
        pipeline.result = data["result"]
        pipeline.error = data["error"]
        
        return pipeline

class AgentManager:
    """Manager for orchestrating multiple agents and swarms"""
    
    def __init__(self):
        """Initialize agent manager"""
        self.memory = AgentMemory()
        self.knowledge_manager = KnowledgeManager()
        self.tool_registry = ToolRegistry()
        self.agent_factory = AgentFactory(self.memory, self.knowledge_manager, self.tool_registry)
        
        self.agents = {}
        self.swarms = {}
        self.pipelines = {}
        
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.task_results = {}
        
        self.worker_thread = None
        self.stop_event = threading.Event()
    
    def create_agent(
        self,
        agent_name: str,
        role: Union[AgentRole, str],
        system_prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Create an agent
        
        Args:
            agent_name: Name of the agent
            role: Role of the agent (can be AgentRole enum or string)
            system_prompt: System prompt for the agent
            model_name: Model to use for the agent
            tools: List of tool names to enable for the agent
            **kwargs: Additional arguments for the agent
            
        Returns:
            ID of the created agent
        """
        # Convert string role to enum if needed
        if isinstance(role, str):
            try:
                role = AgentRole(role)
            except ValueError:
                raise ValueError(f"Unknown agent role: {role}")
        
        # Create the agent
        agent = self.agent_factory.create_agent(
            agent_name=agent_name,
            role=role,
            system_prompt=system_prompt,
            model_name=model_name,
            tools=tools,
            **kwargs
        )
        
        # Generate agent ID
        agent_id = str(uuid.uuid4())
        
        # Store the agent
        self.agents[agent_id] = {
            "id": agent_id,
            "name": agent_name,
            "role": role,
            "agent": agent,
            "created_at": datetime.now().isoformat()
        }
        
        return agent_id
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an agent by ID
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            The agent or None if not found
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all agents
        
        Returns:
            List of agents
        """
        return [
            {
                "id": agent_id,
                "name": agent_info["name"],
                "role": agent_info["role"].value,
                "created_at": agent_info["created_at"]
            }
            for agent_id, agent_info in self.agents.items()
        ]
    
    def create_swarm(
        self,
        swarm_name: str,
        swarm_type: Union[SwarmPattern, str],
        agent_ids: List[str],
        **kwargs
    ) -> str:
        """
        Create a swarm
        
        Args:
            swarm_name: Name of the swarm
            swarm_type: Type of swarm (can be SwarmPattern enum or string)
            agent_ids: List of agent IDs in the swarm
            **kwargs: Additional arguments for the swarm
            
        Returns:
            ID of the created swarm
        """
        # Convert string swarm type to enum if needed
        if isinstance(swarm_type, str):
            try:
                swarm_type = SwarmPattern(swarm_type)
            except ValueError:
                raise ValueError(f"Unknown swarm type: {swarm_type}")
        
        # Get agents
        agents = []
        for agent_id in agent_ids:
            agent_info = self.get_agent(agent_id)
            if agent_info is None:
                raise ValueError(f"Agent {agent_id} not found")
            
            agents.append(agent_info["agent"])
        
        # Handle special case for MixtureOfAgents
        if swarm_type == SwarmPattern.MIXTURE_OF_AGENTS and "aggregator_agent_id" in kwargs:
            aggregator_agent_id = kwargs.pop("aggregator_agent_id")
            aggregator_agent_info = self.get_agent(aggregator_agent_id)
            
            if aggregator_agent_info is None:
                raise ValueError(f"Aggregator agent {aggregator_agent_id} not found")
            
            kwargs["aggregator_agent"] = aggregator_agent_info["agent"]
        
        # Create the swarm
        swarm = self.agent_factory.create_swarm(
            swarm_name=swarm_name,
            swarm_type=swarm_type,
            agents=agents,
            **kwargs
        )
        
        # Generate swarm ID
        swarm_id = str(uuid.uuid4())
        
        # Store the swarm
        self.swarms[swarm_id] = {
            "id": swarm_id,
            "name": swarm_name,
            "type": swarm_type,
            "agent_ids": agent_ids,
            "swarm": swarm,
            "created_at": datetime.now().isoformat()
        }
        
        return swarm_id
    
    def get_swarm(self, swarm_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a swarm by ID
        
        Args:
            swarm_id: ID of the swarm
            
        Returns:
            The swarm or None if not found
        """
        return self.swarms.get(swarm_id)
    
    def list_swarms(self) -> List[Dict[str, Any]]:
        """
        List all swarms
        
        Returns:
            List of swarms
        """
        return [
            {
                "id": swarm_id,
                "name": swarm_info["name"],
                "type": swarm_info["type"].value,
                "agent_ids": swarm_info["agent_ids"],
                "created_at": swarm_info["created_at"]
            }
            for swarm_id, swarm_info in self.swarms.items()
        ]
    
    def create_pipeline(
        self,
        name: str,
        description: str = "",
        tasks: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a pipeline
        
        Args:
            name: Name of the pipeline
            description: Description of the pipeline
            tasks: List of task definitions
            metadata: Additional metadata for the pipeline
            
        Returns:
            ID of the created pipeline
        """
        # Create the pipeline
        pipeline = Pipeline(
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        # Add tasks if provided
        if tasks:
            for task_def in tasks:
                task = AgentTask(
                    description=task_def.get("description", ""),
                    input_data=task_def.get("input_data"),
                    assigned_to=task_def.get("assigned_to"),
                    depends_on=task_def.get("depends_on", []),
                    priority=TaskPriority(task_def.get("priority", TaskPriority.MEDIUM.value)),
                    deadline=datetime.fromisoformat(task_def["deadline"]) if "deadline" in task_def else None,
                    max_retries=task_def.get("max_retries", 3),
                    metadata=task_def.get("metadata", {})
                )
                
                pipeline.add_task(task)
        
        # Generate pipeline ID
        pipeline_id = pipeline.pipeline_id
        
        # Store the pipeline
        self.pipelines[pipeline_id] = pipeline
        
        return pipeline_id
    
    def get_pipeline(self, pipeline_id: str) -> Optional[Pipeline]:
        """
        Get a pipeline by ID
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            The pipeline or None if not found
        """
        return self.pipelines.get(pipeline_id)
    
    def list_pipelines(self) -> List[Dict[str, Any]]:
        """
        List all pipelines
        
        Returns:
            List of pipelines
        """
        return [
            {
                "id": pipeline.pipeline_id,
                "name": pipeline.name,
                "description": pipeline.description,
                "status": pipeline.status,
                "task_count": len(pipeline.tasks),
                "created_at": pipeline.created_at.isoformat()
            }
            for pipeline in self.pipelines.values()
        ]
    
    def add_task_to_pipeline(self, pipeline_id: str, task: AgentTask) -> bool:
        """
        Add a task to a pipeline
        
        Args:
            pipeline_id: ID of the pipeline
            task: Task to add
            
        Returns:
            True if the task was added successfully
        """
        pipeline = self.get_pipeline(pipeline_id)
        if pipeline is None:
            return False
        
        pipeline.add_task(task)
        return True
    
    def run_pipeline(self, pipeline_id: str) -> bool:
        """
        Run a pipeline
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            True if the pipeline was started successfully
        """
        pipeline = self.get_pipeline(pipeline_id)
        if pipeline is None:
            return False
        
        # Add all pending tasks to the queue
        for task in pipeline.tasks:
            if task.status == TaskStatus.PENDING:
                # Set priority based on task priority (higher priority = lower value in queue)
                priority = 4 - task.priority.value  # Convert CRITICAL(3) to 1, LOW(0) to 4
                self.task_queue.put((priority, task.task_id, pipeline_id))
        
        # Start the worker thread if not already running
        self._ensure_worker_thread()
        
        return True
    
    def run_agent(self, agent_id: str, input_data: Any) -> str:
        """
        Run an agent directly
        
        Args:
            agent_id: ID of the agent
            input_data: Input data for the agent
            
        Returns:
            ID of the created task
        """
        agent_info = self.get_agent(agent_id)
        if agent_info is None:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create a task
        task = AgentTask(
            description=f"Direct execution of agent {agent_info['name']}",
            input_data=input_data,
            assigned_to=agent_id,
            priority=TaskPriority.HIGH
        )
        
        # Create a pipeline for this task
        pipeline = Pipeline(
            name=f"Direct execution of {agent_info['name']}",
            description=f"Pipeline for direct execution of agent {agent_info['name']}",
            tasks=[task]
        )
        
        # Store the pipeline
        pipeline_id = pipeline.pipeline_id
        self.pipelines[pipeline_id] = pipeline
        
        # Add the task to the queue
        self.task_queue.put((1, task.task_id, pipeline_id))  # Priority 1 (HIGH)
        
        # Start the worker thread if not already running
        self._ensure_worker_thread()
        
        return task.task_id
    
    def run_swarm(self, swarm_id: str, input_data: Any) -> str:
        """
        Run a swarm directly
        
        Args:
            swarm_id: ID of the swarm
            input_data: Input data for the swarm
            
        Returns:
            ID of the created task
        """
        swarm_info = self.get_swarm(swarm_id)
        if swarm_info is None:
            raise ValueError(f"Swarm {swarm_id} not found")
        
        # Create a task
        task = AgentTask(
            description=f"Direct execution of swarm {swarm_info['name']}",
            input_data=input_data,
            assigned_to=swarm_id,
            priority=TaskPriority.HIGH
        )
        
        # Create a pipeline for this task
        pipeline = Pipeline(
            name=f"Direct execution of {swarm_info['name']}",
            description=f"Pipeline for direct execution of swarm {swarm_info['name']}",
            tasks=[task]
        )
        
        # Store the pipeline
        pipeline_id = pipeline.pipeline_id
        self.pipelines[pipeline_id] = pipeline
        
        # Add the task to the queue
        self.task_queue.put((1, task.task_id, pipeline_id))  # Priority 1 (HIGH)
        
        # Start the worker thread if not already running
        self._ensure_worker_thread()
        
        return task.task_id
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the result of a task
        
        Args:
            task_id: ID of the task
            
        Returns:
            Task result or None if not available
        """
        # Check if the task is in the results
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        # Check if the task is in a pipeline
        for pipeline in self.pipelines.values():
            task = pipeline.get_task(task_id)
            if task is not None:
                if task.status == TaskStatus.COMPLETED:
                    return {
                        "task_id": task_id,
                        "status": task.status.value,
                        "result": task.result,
                        "completed_at": task.completed_at.isoformat() if task.completed_at else None
                    }
                else:
                    return {
                        "task_id": task_id,
                        "status": task.status.value,
                        "error": task.error,
                        "updated_at": task.updated_at.isoformat()
                    }
        
        return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task
        
        Args:
            task_id: ID of the task
            
        Returns:
            True if the task was cancelled successfully
        """
        # Check if the task is running
        if task_id in self.running_tasks:
            # Mark the task as cancelled
            self.running_tasks[task_id]["cancel"] = True
            return True
        
        # Check if the task is in a pipeline
        for pipeline in self.pipelines.values():
            task = pipeline.get_task(task_id)
            if task is not None and task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                task.update_status(TaskStatus.CANCELLED)
                pipeline.update_status()
                return True
        
        return False
    
    def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel a pipeline
        
        Args:
            pipeline_id: ID of the pipeline
            
        Returns:
            True if the pipeline was cancelled successfully
        """
        pipeline = self.get_pipeline(pipeline_id)
        if pipeline is None:
            return False
        
        # Cancel all pending and assigned tasks
        for task in pipeline.tasks:
            if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                task.update_status(TaskStatus.CANCELLED)
        
        # Cancel running tasks
        for task_id in list(self.running_tasks.keys()):
            if self.running_tasks[task_id]["pipeline_id"] == pipeline_id:
                self.running_tasks[task_id]["cancel"] = True
        
        pipeline.update_status()
        return True
    
    def _ensure_worker_thread(self):
        """Ensure that the worker thread is running"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.stop_event.clear()
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
    
    def _worker_loop(self):
        """Worker loop for processing tasks"""
        while not self.stop_event.is_set():
            try:
                # Get a task from the queue with a timeout
                try:
                    priority, task_id, pipeline_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Get the pipeline and task
                pipeline = self.get_pipeline(pipeline_id)
                if pipeline is None:
                    logger.error(f"Pipeline {pipeline_id} not found")
                    self.task_queue.task_done()
                    continue
                
                task = pipeline.get_task(task_id)
                if task is None:
                    logger.error(f"Task {task_id} not found in pipeline {pipeline_id}")
                    self.task_queue.task_done()
                    continue
                
                # Check if the task is still pending
                if task.status != TaskStatus.PENDING:
                    logger.info(f"Task {task_id} is not pending (status: {task.status.value})")
                    self.task_queue.task_done()
                    continue
                
                # Check if all dependencies are completed
                dependencies_completed = True
                for dep_id in task.depends_on:
                    dep_task = pipeline.get_task(dep_id)
                    if dep_task is None or dep_task.status != TaskStatus.COMPLETED:
                        dependencies_completed = False
                        break
                
                if not dependencies_completed:
                    # Put the task back in the queue with a delay
                    logger.info(f"Task {task_id} has unmet dependencies, re-queueing")
                    self.task_queue.put((priority, task_id, pipeline_id))
                    self.task_queue.task_done()
                    time.sleep(1.0)
                    continue
                
                # Update task status
                task.update_status(TaskStatus.ASSIGNED)
                pipeline.update_status()
                
                # Execute the task
                self._execute_task(task, pipeline)
                
                # Mark the task as done in the queue
                self.task_queue.task_done()
            
            except Exception as e:
                logger.error(f"Error in worker loop: {e}")
                time.sleep(1.0)
    
    def _execute_task(self, task: AgentTask, pipeline: Pipeline):
        """
        Execute a task
        
        Args:
            task: Task to execute
            pipeline: Pipeline containing the task
        """
        # Mark the task as in progress
        task.update_status(TaskStatus.IN_PROGRESS)
        pipeline.update_status()
        
        # Add to running tasks
        self.running_tasks[task.task_id] = {
            "task": task,
            "pipeline": pipeline,
            "cancel": False
        }
        
        try:
            # Get the assigned agent or swarm
            assigned_to = task.assigned_to
            
            if isinstance(assigned_to, list):
                # Multiple agents assigned (use a swarm)
                agents = []
                for agent_id in assigned_to:
                    agent_info = self.get_agent(agent_id)
                    if agent_info is None:
                        raise ValueError(f"Agent {agent_id} not found")
                    
                    agents.append(agent_info["agent"])
                
                # Create a temporary swarm
                swarm = self.agent_factory.create_swarm(
                    swarm_name=f"temp_swarm_{task.task_id}",
                    swarm_type=SwarmPattern.SEQUENTIAL,
                    agents=agents
                )
                
                # Execute the swarm
                result = swarm.run(task.input_data)
            
            elif assigned_to in self.agents:
                # Single agent assigned
                agent_info = self.agents[assigned_to]
                agent = agent_info["agent"]
                
                # Execute the agent
                result = agent.run(task.input_data)
            
            elif assigned_to in self.swarms:
                # Swarm assigned
                swarm_info = self.swarms[assigned_to]
                swarm = swarm_info["swarm"]
                
                # Execute the swarm
                result = swarm.run(task.input_data)
            
            else:
                raise ValueError(f"Invalid assignment: {assigned_to}")
            
            # Check if the task was cancelled
            if self.running_tasks[task.task_id]["cancel"]:
                task.update_status(TaskStatus.CANCELLED)
                pipeline.update_status()
                return
            
            # Update task with result
            task.result = result
            task.update_status(TaskStatus.COMPLETED)
            
            # Store the result
            self.task_results[task.task_id] = {
                "task_id": task.task_id,
                "status": task.status.value,
                "result": task.result,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None
            }
            
            # Check if all tasks in the pipeline are completed
            pipeline.update_status()
            if pipeline.is_completed():
                # Collect all results
                results = {}
                for t in pipeline.tasks:
                    if t.status == TaskStatus.COMPLETED:
                        results[t.task_id] = t.result
                
                # Set pipeline result
                pipeline.result = results
            
            # Check if there are more tasks to execute
            next_tasks = pipeline.get_next_tasks()
            for next_task in next_tasks:
                # Add to queue with priority
                priority = 4 - next_task.priority.value
                self.task_queue.put((priority, next_task.task_id, pipeline.pipeline_id))
        
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            
            # Update task with error
            task.error = str(e)
            
            # Check if the task can be retried
            if task.can_retry():
                task.increment_retry()
                task.update_status(TaskStatus.PENDING)
                
                # Re-queue the task with a delay
                priority = 4 - task.priority.value
                self.task_queue.put((priority, task.task_id, pipeline.pipeline_id))
            else:
                task.update_status(TaskStatus.FAILED)
            
            pipeline.update_status()
        
        finally:
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
    
    def stop(self):
        """Stop the agent manager"""
        # Signal the worker thread to stop
        self.stop_event.set()
        
        # Wait for the worker thread to finish
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Cancel all running tasks
        for task_info in self.running_tasks.values():
            task_info["cancel"] = True
        
        # Clear the task queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break
    
    def create_expert_pipeline(self, task_description: str, num_experts: int = 5) -> str:
        """
        Create a pipeline with expert agents
        
        Args:
            task_description: Description of the task
            num_experts: Number of expert agents to create
            
        Returns:
            ID of the created pipeline
        """
        # Create a coordinator agent
        coordinator_id = self.create_agent(
            agent_name="Coordinator",
            role=AgentRole.COORDINATOR,
            model_name=get("models.default_model")
        )
        
        # Create expert agents
        expert_ids = []
        expert_roles = [
            AgentRole.RESEARCHER,
            AgentRole.ANALYST,
            AgentRole.WRITER,
            AgentRole.CODER,
            AgentRole.REVIEWER,
            AgentRole.PLANNER,
            AgentRole.EXECUTOR,
            AgentRole.CREATIVE,
            AgentRole.DOMAIN_EXPERT,
            AgentRole.SUMMARIZER
        ]
        
        # Select random roles if num_experts < len(expert_roles)
        if num_experts < len(expert_roles):
            import random
            selected_roles = random.sample(expert_roles, num_experts)
        else:
            selected_roles = expert_roles[:num_experts]
        
        for i, role in enumerate(selected_roles):
            expert_id = self.create_agent(
                agent_name=f"Expert_{role.value}",
                role=role,
                model_name=get("models.default_model")
            )
            expert_ids.append(expert_id)
        
        # Create an integrator agent
        integrator_id = self.create_agent(
            agent_name="Integrator",
            role=AgentRole.INTEGRATOR,
            model_name=get("models.default_model")
        )
        
        # Create tasks
        tasks = []
        
        # Task 1: Coordinator analyzes the task
        coordinator_task = AgentTask(
            description="Analyze the task and create a plan",
            input_
