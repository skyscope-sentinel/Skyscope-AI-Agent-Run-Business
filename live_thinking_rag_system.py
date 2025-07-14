import os
import sys
import json
import time
import logging
import threading
import queue
import uuid
import re
import inspect
import traceback
import datetime
import hashlib
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Set, Generator
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque, defaultdict

# For vector storage and embeddings
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

# For visualization
import streamlit as st
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.table import Table

# Local imports
from ui_themes import create_thinking_animation, create_progress_bar, create_ocr_text, create_glitch_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/live_thinking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_thinking_rag")

# Constants
CACHE_DIR = Path("cache")
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
KNOWLEDGE_BASE_DIR = Path("knowledge_base")
THINKING_LOGS_DIR = Path("logs/thinking")
MAX_THINKING_HISTORY = 1000
MAX_RAG_RESULTS = 10
MAX_CONCURRENT_RETRIEVALS = 5
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
THINKING_UPDATE_INTERVAL = 0.1  # seconds
QUALITY_THRESHOLD = 0.75  # 0-1 scale

# Ensure directories exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
THINKING_LOGS_DIR.mkdir(parents=True, exist_ok=True)

class ThoughtType(Enum):
    """Types of thoughts in the thinking process."""
    PLANNING = "planning"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    ANALYSIS = "analysis"
    DECISION = "decision"
    CRITIQUE = "critique"
    REFLECTION = "reflection"
    ACTION = "action"
    ERROR = "error"
    SUMMARY = "summary"

class CritiqueLevel(Enum):
    """Levels of self-critique."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4

class RetrievalStrategy(Enum):
    """Strategies for knowledge retrieval."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    CONTEXTUAL = "contextual"
    CHRONOLOGICAL = "chronological"

@dataclass
class ThoughtNode:
    """Represents a single thought in the thinking process."""
    id: str
    type: ThoughtType
    content: str
    timestamp: float
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, type: ThoughtType, content: str, parent_id: Optional[str] = None) -> 'ThoughtNode':
        """Create a new thought node."""
        return cls(
            id=str(uuid.uuid4()),
            type=type,
            content=content,
            timestamp=time.time(),
            parent_id=parent_id,
            children_ids=[],
            metadata={}
        )
    
    def add_child(self, child_id: str) -> None:
        """Add a child thought."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThoughtNode':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=ThoughtType(data["type"]),
            content=data["content"],
            timestamp=data["timestamp"],
            parent_id=data["parent_id"],
            children_ids=data["children_ids"],
            metadata=data["metadata"]
        )

@dataclass
class RetrievedDocument:
    """Represents a document retrieved from the knowledge base."""
    id: str
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            source=data["source"],
            score=data["score"],
            metadata=data["metadata"]
        )

@dataclass
class ThinkingSession:
    """Represents a thinking session."""
    id: str
    task: str
    agent_id: str
    start_time: float
    end_time: Optional[float] = None
    root_thought_id: Optional[str] = None
    current_thought_id: Optional[str] = None
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, task: str, agent_id: str) -> 'ThinkingSession':
        """Create a new thinking session."""
        return cls(
            id=str(uuid.uuid4()),
            task=task,
            agent_id=agent_id,
            start_time=time.time(),
            metadata={}
        )
    
    def complete(self) -> None:
        """Mark the session as complete."""
        self.end_time = time.time()
        self.status = "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "task": self.task,
            "agent_id": self.agent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "root_thought_id": self.root_thought_id,
            "current_thought_id": self.current_thought_id,
            "status": self.status,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThinkingSession':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            task=data["task"],
            agent_id=data["agent_id"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            root_thought_id=data.get("root_thought_id"),
            current_thought_id=data.get("current_thought_id"),
            status=data["status"],
            metadata=data["metadata"]
        )

@dataclass
class QualityMetrics:
    """Metrics for quality assessment."""
    relevance: float = 0.0  # 0-1 scale
    coherence: float = 0.0  # 0-1 scale
    accuracy: float = 0.0  # 0-1 scale
    completeness: float = 0.0  # 0-1 scale
    efficiency: float = 0.0  # 0-1 scale
    critique_level: CritiqueLevel = CritiqueLevel.MEDIUM
    
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (self.relevance + self.coherence + self.accuracy + self.completeness + self.efficiency) / 5.0
    
    def meets_threshold(self, threshold: float = QUALITY_THRESHOLD) -> bool:
        """Check if quality meets threshold."""
        return self.overall_score() >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "relevance": self.relevance,
            "coherence": self.coherence,
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "efficiency": self.efficiency,
            "critique_level": self.critique_level.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityMetrics':
        """Create from dictionary."""
        return cls(
            relevance=data["relevance"],
            coherence=data["coherence"],
            accuracy=data["accuracy"],
            completeness=data["completeness"],
            efficiency=data["efficiency"],
            critique_level=CritiqueLevel(data["critique_level"])
        )

class ThinkingGraph:
    """Graph of thoughts in a thinking session."""
    
    def __init__(self):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.sessions: Dict[str, ThinkingSession] = {}
        self.lock = threading.RLock()
    
    def create_session(self, task: str, agent_id: str) -> ThinkingSession:
        """Create a new thinking session."""
        with self.lock:
            session = ThinkingSession.create(task, agent_id)
            self.sessions[session.id] = session
            return session
    
    def add_thought(self, session_id: str, type: ThoughtType, content: str, parent_id: Optional[str] = None) -> ThoughtNode:
        """Add a thought to a session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Create thought
            thought = ThoughtNode.create(type, content, parent_id)
            self.nodes[thought.id] = thought
            
            # Update parent if provided
            if parent_id:
                parent = self.nodes.get(parent_id)
                if parent:
                    parent.add_child(thought.id)
            else:
                # If no parent, this is the root thought
                if not session.root_thought_id:
                    session.root_thought_id = thought.id
            
            # Update session's current thought
            session.current_thought_id = thought.id
            
            return thought
    
    def get_thought(self, thought_id: str) -> Optional[ThoughtNode]:
        """Get a thought by ID."""
        with self.lock:
            return self.nodes.get(thought_id)
    
    def get_session(self, session_id: str) -> Optional[ThinkingSession]:
        """Get a session by ID."""
        with self.lock:
            return self.sessions.get(session_id)
    
    def get_session_thoughts(self, session_id: str) -> List[ThoughtNode]:
        """Get all thoughts in a session."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session or not session.root_thought_id:
                return []
            
            # Traverse the thought tree
            thoughts = []
            queue = [session.root_thought_id]
            visited = set()
            
            while queue:
                thought_id = queue.pop(0)
                if thought_id in visited:
                    continue
                
                visited.add(thought_id)
                thought = self.nodes.get(thought_id)
                if thought:
                    thoughts.append(thought)
                    queue.extend(thought.children_ids)
            
            return thoughts
    
    def get_thought_path(self, thought_id: str) -> List[ThoughtNode]:
        """Get the path from root to a thought."""
        with self.lock:
            path = []
            current_id = thought_id
            
            while current_id:
                thought = self.nodes.get(current_id)
                if not thought:
                    break
                
                path.insert(0, thought)
                current_id = thought.parent_id
            
            return path
    
    def complete_session(self, session_id: str) -> bool:
        """Mark a session as complete."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            session.complete()
            return True
    
    def save_session(self, session_id: str) -> bool:
        """Save a session to disk."""
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return False
            
            # Create session directory
            session_dir = THINKING_LOGS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Save session metadata
            with open(session_dir / "session.json", "w") as f:
                json.dump(session.to_dict(), f, indent=2)
            
            # Save thoughts
            thoughts_dir = session_dir / "thoughts"
            thoughts_dir.mkdir(parents=True, exist_ok=True)
            
            for thought in self.get_session_thoughts(session_id):
                with open(thoughts_dir / f"{thought.id}.json", "w") as f:
                    json.dump(thought.to_dict(), f, indent=2)
            
            return True
    
    def load_session(self, session_id: str) -> Optional[ThinkingSession]:
        """Load a session from disk."""
        session_dir = THINKING_LOGS_DIR / session_id
        if not session_dir.exists():
            return None
        
        try:
            # Load session metadata
            with open(session_dir / "session.json", "r") as f:
                session_data = json.load(f)
            
            session = ThinkingSession.from_dict(session_data)
            
            # Load thoughts
            thoughts_dir = session_dir / "thoughts"
            if thoughts_dir.exists():
                for thought_file in thoughts_dir.glob("*.json"):
                    with open(thought_file, "r") as f:
                        thought_data = json.load(f)
                    
                    thought = ThoughtNode.from_dict(thought_data)
                    self.nodes[thought.id] = thought
            
            self.sessions[session.id] = session
            return session
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    def list_sessions(self) -> List[ThinkingSession]:
        """List all available sessions."""
        sessions = []
        
        # Load from disk
        for session_dir in THINKING_LOGS_DIR.glob("*"):
            if session_dir.is_dir() and (session_dir / "session.json").exists():
                session_id = session_dir.name
                if session_id not in self.sessions:
                    self.load_session(session_id)
        
        # Return all sessions
        return list(self.sessions.values())

class VectorStore:
    """Vector store for document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.document_lookup: Dict[int, RetrievedDocument] = {}
        self.lock = threading.RLock()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the vector store."""
        with self.lock:
            if self.initialized:
                return True
            
            try:
                # Load model
                self.model = SentenceTransformer(self.model_name)
                
                # Create empty index
                embedding_dim = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(embedding_dim)
                
                self.initialized = True
                logger.info(f"Vector store initialized with model {self.model_name}")
                return True
            except Exception as e:
                logger.error(f"Error initializing vector store: {e}")
                return False
    
    def add_documents(self, documents: List[RetrievedDocument]) -> bool:
        """Add documents to the vector store."""
        if not self.initialize():
            return False
        
        with self.lock:
            try:
                # Get current index size
                current_size = self.index.ntotal
                
                # Create embeddings
                texts = [doc.content for doc in documents]
                embeddings = self.model.encode(texts)
                
                # Add to index
                self.index.add(np.array(embeddings).astype('float32'))
                
                # Update document lookup
                for i, doc in enumerate(documents):
                    self.document_lookup[current_size + i] = doc
                
                return True
            except Exception as e:
                logger.error(f"Error adding documents to vector store: {e}")
                return False
    
    def search(self, query: str, k: int = MAX_RAG_RESULTS) -> List[RetrievedDocument]:
        """Search for similar documents."""
        if not self.initialize():
            return []
        
        with self.lock:
            try:
                # Create query embedding
                query_embedding = self.model.encode([query])
                
                # Search index
                distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
                
                # Get documents
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1 and idx in self.document_lookup:
                        doc = self.document_lookup[idx]
                        doc.score = 1.0 / (1.0 + distances[0][i])  # Convert distance to score
                        results.append(doc)
                
                return results
            except Exception as e:
                logger.error(f"Error searching vector store: {e}")
                return []
    
    def save(self, directory: Path = EMBEDDINGS_DIR) -> bool:
        """Save the vector store to disk."""
        if not self.initialized:
            return False
        
        with self.lock:
            try:
                # Create directory
                directory.mkdir(parents=True, exist_ok=True)
                
                # Save index
                faiss.write_index(self.index, str(directory / "index.faiss"))
                
                # Save document lookup
                documents = {str(k): v.to_dict() for k, v in self.document_lookup.items()}
                with open(directory / "documents.json", "w") as f:
                    json.dump(documents, f)
                
                return True
            except Exception as e:
                logger.error(f"Error saving vector store: {e}")
                return False
    
    def load(self, directory: Path = EMBEDDINGS_DIR) -> bool:
        """Load the vector store from disk."""
        if not self.initialize():
            return False
        
        with self.lock:
            try:
                # Check if files exist
                if not (directory / "index.faiss").exists() or not (directory / "documents.json").exists():
                    return False
                
                # Load index
                self.index = faiss.read_index(str(directory / "index.faiss"))
                
                # Load document lookup
                with open(directory / "documents.json", "r") as f:
                    documents = json.load(f)
                
                self.document_lookup = {int(k): RetrievedDocument.from_dict(v) for k, v in documents.items()}
                
                return True
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                return False

class KnowledgeBase:
    """Knowledge base for document storage and retrieval."""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.documents: Dict[str, RetrievedDocument] = {}
        self.lock = threading.RLock()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the knowledge base."""
        with self.lock:
            if self.initialized:
                return True
            
            try:
                # Initialize vector store
                if not self.vector_store.initialize():
                    return False
                
                # Load existing documents
                self.load()
                
                self.initialized = True
                logger.info("Knowledge base initialized")
                return True
            except Exception as e:
                logger.error(f"Error initializing knowledge base: {e}")
                return False
    
    def add_document(self, content: str, source: str, metadata: Dict[str, Any] = None) -> Optional[str]:
        """Add a document to the knowledge base."""
        if not self.initialize():
            return None
        
        with self.lock:
            try:
                # Create document
                doc_id = str(uuid.uuid4())
                doc = RetrievedDocument(
                    id=doc_id,
                    content=content,
                    source=source,
                    score=1.0,
                    metadata=metadata or {}
                )
                
                # Add to documents
                self.documents[doc_id] = doc
                
                # Add to vector store
                self.vector_store.add_documents([doc])
                
                return doc_id
            except Exception as e:
                logger.error(f"Error adding document to knowledge base: {e}")
                return None
    
    def add_documents_from_directory(self, directory: Path, recursive: bool = True) -> int:
        """Add documents from a directory."""
        if not directory.exists() or not directory.is_dir():
            return 0
        
        count = 0
        pattern = "**/*.*" if recursive else "*.*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    
                    # Add document
                    if self.add_document(
                        content=content,
                        source=str(file_path),
                        metadata={"filename": file_path.name, "extension": file_path.suffix}
                    ):
                        count += 1
                except Exception as e:
                    logger.error(f"Error adding document {file_path}: {e}")
        
        return count
    
    def search(self, query: str, k: int = MAX_RAG_RESULTS, strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> List[RetrievedDocument]:
        """Search for documents."""
        if not self.initialize():
            return []
        
        with self.lock:
            if strategy == RetrievalStrategy.SEMANTIC:
                # Semantic search using vector store
                return self.vector_store.search(query, k)
            elif strategy == RetrievalStrategy.KEYWORD:
                # Keyword search
                results = []
                query_terms = set(query.lower().split())
                
                for doc in self.documents.values():
                    content_terms = set(doc.content.lower().split())
                    overlap = len(query_terms.intersection(content_terms))
                    if overlap > 0:
                        score = overlap / len(query_terms)
                        doc_copy = RetrievedDocument(
                            id=doc.id,
                            content=doc.content,
                            source=doc.source,
                            score=score,
                            metadata=doc.metadata.copy()
                        )
                        results.append(doc_copy)
                
                # Sort by score
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:k]
            elif strategy == RetrievalStrategy.HYBRID:
                # Combine semantic and keyword search
                semantic_results = self.vector_store.search(query, k)
                keyword_results = self.search(query, k, RetrievalStrategy.KEYWORD)
                
                # Merge results
                results_dict = {}
                for doc in semantic_results:
                    results_dict[doc.id] = doc
                
                for doc in keyword_results:
                    if doc.id in results_dict:
                        # Combine scores
                        results_dict[doc.id].score = max(results_dict[doc.id].score, doc.score)
                    else:
                        results_dict[doc.id] = doc
                
                # Sort by score
                results = list(results_dict.values())
                results.sort(key=lambda x: x.score, reverse=True)
                return results[:k]
            else:
                logger.warning(f"Unsupported retrieval strategy: {strategy}")
                return []
    
    def get_document(self, doc_id: str) -> Optional[RetrievedDocument]:
        """Get a document by ID."""
        with self.lock:
            return self.documents.get(doc_id)
    
    def save(self) -> bool:
        """Save the knowledge base to disk."""
        with self.lock:
            try:
                # Save vector store
                if not self.vector_store.save():
                    return False
                
                # Save documents
                documents_dir = KNOWLEDGE_BASE_DIR / "documents"
                documents_dir.mkdir(parents=True, exist_ok=True)
                
                with open(documents_dir / "documents.json", "w") as f:
                    documents = {k: v.to_dict() for k, v in self.documents.items()}
                    json.dump(documents, f)
                
                return True
            except Exception as e:
                logger.error(f"Error saving knowledge base: {e}")
                return False
    
    def load(self) -> bool:
        """Load the knowledge base from disk."""
        with self.lock:
            try:
                # Load vector store
                if not self.vector_store.load():
                    # If vector store doesn't exist, that's okay for a new knowledge base
                    pass
                
                # Load documents
                documents_path = KNOWLEDGE_BASE_DIR / "documents" / "documents.json"
                if documents_path.exists():
                    with open(documents_path, "r") as f:
                        documents = json.load(f)
                    
                    self.documents = {k: RetrievedDocument.from_dict(v) for k, v in documents.items()}
                
                return True
            except Exception as e:
                logger.error(f"Error loading knowledge base: {e}")
                return False

class ThinkingEngine:
    """Engine for live thinking and self-criticism."""
    
    def __init__(self):
        self.graph = ThinkingGraph()
        self.knowledge_base = KnowledgeBase()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.thinking_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RETRIEVALS)
        self.lock = threading.RLock()
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the thinking engine."""
        with self.lock:
            if self.initialized:
                return True
            
            try:
                # Initialize knowledge base
                if not self.knowledge_base.initialize():
                    return False
                
                self.initialized = True
                logger.info("Thinking engine initialized")
                return True
            except Exception as e:
                logger.error(f"Error initializing thinking engine: {e}")
                return False
    
    def start_thinking(self, task: str, agent_id: str) -> str:
        """Start a new thinking session."""
        if not self.initialize():
            raise RuntimeError("Thinking engine not initialized")
        
        with self.lock:
            # Create session
            session = self.graph.create_session(task, agent_id)
            
            # Add initial thought
            thought = self.graph.add_thought(
                session_id=session.id,
                type=ThoughtType.PLANNING,
                content=f"Starting to think about: {task}"
            )
            
            # Set up active session
            self.active_sessions[session.id] = {
                "session": session,
                "current_thought": thought,
                "quality_metrics": QualityMetrics(),
                "start_time": time.time(),
                "last_update": time.time(),
                "retrieved_documents": [],
                "thinking_queue": queue.Queue(),
                "thinking_thread": None
            }
            
            # Start thinking thread
            self._start_thinking_thread(session.id)
            
            return session.id
    
    def _start_thinking_thread(self, session_id: str) -> None:
        """Start the thinking thread for a session."""
        def thinking_loop():
            while session_id in self.active_sessions:
                try:
                    # Get next thought from queue
                    try:
                        thought_data = self.active_sessions[session_id]["thinking_queue"].get(timeout=0.1)
                    except queue.Empty:
                        continue
                    
                    # Process thought
                    thought_type = thought_data.get("type", ThoughtType.REASONING)
                    thought_content = thought_data.get("content", "")
                    parent_id = thought_data.get("parent_id")
                    
                    # Add thought to graph
                    thought = self.graph.add_thought(
                        session_id=session_id,
                        type=thought_type,
                        content=thought_content,
                        parent_id=parent_id
                    )
                    
                    # Update active session
                    with self.lock:
                        if session_id in self.active_sessions:
                            self.active_sessions[session_id]["current_thought"] = thought
                            self.active_sessions[session_id]["last_update"] = time.time()
                    
                    # Notify callbacks
                    self._notify_thinking_callbacks(session_id, thought)
                    
                    # Mark task as done
                    self.active_sessions[session_id]["thinking_queue"].task_done()
                    
                    # Small delay to simulate thinking
                    time.sleep(THINKING_UPDATE_INTERVAL)
                except Exception as e:
                    logger.error(f"Error in thinking thread for session {session_id}: {e}")
        
        # Start thread
        thread = threading.Thread(target=thinking_loop, daemon=True)
        thread.start()
        
        with self.lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["thinking_thread"] = thread
    
    def _notify_thinking_callbacks(self, session_id: str, thought: ThoughtNode) -> None:
        """Notify thinking callbacks."""
        callbacks = self.thinking_callbacks.get(session_id, [])
        for callback in callbacks:
            try:
                callback(thought)
            except Exception as e:
                logger.error(f"Error in thinking callback: {e}")
    
    def add_thought(self, session_id: str, type: ThoughtType, content: str, parent_id: Optional[str] = None) -> Optional[str]:
        """Add a thought to a thinking session."""
        if session_id not in self.active_sessions:
            return None
        
        # Add to thinking queue
        self.active_sessions[session_id]["thinking_queue"].put({
            "type": type,
            "content": content,
            "parent_id": parent_id
        })
        
        # Return parent ID for chaining
        return parent_id or self.active_sessions[session_id]["current_thought"].id
    
    def retrieve_knowledge(self, session_id: str, query: str, strategy: RetrievalStrategy = RetrievalStrategy.HYBRID) -> List[RetrievedDocument]:
        """Retrieve knowledge for a thinking session."""
        if session_id not in self.active_sessions:
            return []
        
        # Add retrieval thought
        self.add_thought(
            session_id=session_id,
            type=ThoughtType.RETRIEVAL,
            content=f"Retrieving knowledge for: {query}"
        )
        
        # Perform retrieval
        results = self.knowledge_base.search(query, strategy=strategy)
        
        # Store results
        with self.lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["retrieved_documents"] = results
        
        # Add analysis thought
        if results:
            self.add_thought(
                session_id=session_id,
                type=ThoughtType.ANALYSIS,
                content=f"Retrieved {len(results)} relevant documents. Most relevant: {results[0].content[:100]}..."
            )
        else:
            self.add_thought(
                session_id=session_id,
                type=ThoughtType.ANALYSIS,
                content="No relevant documents found."
            )
        
        return results
    
    def evaluate_quality(self, session_id: str, metrics: QualityMetrics) -> bool:
        """Evaluate quality of thinking."""
        if session_id not in self.active_sessions:
            return False
        
        with self.lock:
            # Update metrics
            self.active_sessions[session_id]["quality_metrics"] = metrics
            
            # Add critique thought based on metrics
            overall_score = metrics.overall_score()
            critique_level = metrics.critique_level
            
            if overall_score < QUALITY_THRESHOLD:
                critique_content = f"Self-critique: Quality assessment indicates room for improvement (score: {overall_score:.2f}). "
                
                if metrics.relevance < QUALITY_THRESHOLD:
                    critique_content += "Relevance is low. Need to focus more on the task. "
                
                if metrics.coherence < QUALITY_THRESHOLD:
                    critique_content += "Coherence is low. Need to improve logical flow. "
                
                if metrics.accuracy < QUALITY_THRESHOLD:
                    critique_content += "Accuracy concerns detected. Need to verify information. "
                
                if metrics.completeness < QUALITY_THRESHOLD:
                    critique_content += "Solution may be incomplete. Need to address all aspects. "
                
                if metrics.efficiency < QUALITY_THRESHOLD:
                    critique_content += "Efficiency could be improved. Consider more direct approaches. "
                
                self.add_thought(
                    session_id=session_id,
                    type=ThoughtType.CRITIQUE,
                    content=critique_content
                )
                
                return False
            else:
                self.add_thought(
                    session_id=session_id,
                    type=ThoughtType.REFLECTION,
                    content=f"Self-assessment: Quality metrics are satisfactory (score: {overall_score:.2f})."
                )
                
                return True
    
    def complete_thinking(self, session_id: str, summary: str) -> bool:
        """Complete a thinking session."""
        if session_id not in self.active_sessions:
            return False
        
        # Add summary thought
        self.add_thought(
            session_id=session_id,
            type=ThoughtType.SUMMARY,
            content=summary
        )
        
        # Wait for thinking queue to empty
        self.active_sessions[session_id]["thinking_queue"].join()
        
        # Complete session
        self.graph.complete_session(session_id)
        
        # Save session
        self.graph.save_session(session_id)
        
        # Clean up
        with self.lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
        
        return True
    
    def get_thinking_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a thinking session."""
        with self.lock:
            if session_id not in self.active_sessions:
                # Try to load from saved sessions
                session = self.graph.get_session(session_id)
                if not session:
                    return {"error": f"Session not found: {session_id}"}
                
                # Get thoughts
                thoughts = self.graph.get_session_thoughts(session_id)
                
                return {
                    "session_id": session_id,
                    "task": session.task,
                    "agent_id": session.agent_id,
                    "status": session.status,
                    "start_time": session.start_time,
                    "end_time": session.end_time,
                    "duration": session.end_time - session.start_time if session.end_time else None,
                    "thought_count": len(thoughts),
                    "current_thought": None,
                    "quality_metrics": None,
                    "retrieved_documents": []
                }
            
            # Get active session
            session_data = self.active_sessions[session_id]
            session = session_data["session"]
            current_thought = session_data["current_thought"]
            quality_metrics = session_data["quality_metrics"]
            
            return {
                "session_id": session_id,
                "task": session.task,
                "agent_id": session.agent_id,
                "status": session.status,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": time.time() - session.start_time,
                "thought_count": len(self.graph.get_session_thoughts(session_id)),
                "current_thought": current_thought.to_dict() if current_thought else None,
                "quality_metrics": quality_metrics.to_dict() if quality_metrics else None,
                "retrieved_documents": [doc.to_dict() for doc in session_data["retrieved_documents"]]
            }
    
    def register_thinking_callback(self, session_id: str, callback: Callable) -> None:
        """Register a callback for thinking updates."""
        with self.lock:
            self.thinking_callbacks[session_id].append(callback)
    
    def unregister_thinking_callback(self, session_id: str, callback: Callable) -> None:
        """Unregister a thinking callback."""
        with self.lock:
            if session_id in self.thinking_callbacks:
                try:
                    self.thinking_callbacks[session_id].remove(callback)
                except ValueError:
                    pass
    
    def get_thinking_visualization(self, session_id: str, format: str = "html") -> str:
        """Get visualization of thinking process."""
        thoughts = self.graph.get_session_thoughts(session_id)
        
        if not thoughts:
            return "No thoughts found."
        
        if format == "html":
            return self._generate_html_visualization(session_id, thoughts)
        elif format == "text":
            return self._generate_text_visualization(session_id, thoughts)
        elif format == "markdown":
            return self._generate_markdown_visualization(session_id, thoughts)
        else:
            return f"Unsupported visualization format: {format}"
    
    def _generate_html_visualization(self, session_id: str, thoughts: List[ThoughtNode]) -> str:
        """Generate HTML visualization."""
        session = self.graph.get_session(session_id)
        if not session:
            return "<p>Session not found.</p>"
        
        html = f"""
        <div class="thinking-visualization">
            <h2>Thinking Session: {session.task}</h2>
            <p>Agent: {session.agent_id}</p>
            <p>Status: {session.status}</p>
            <p>Started: {datetime.datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="thinking-timeline">
        """
        
        for thought in thoughts:
            thought_time = datetime.datetime.fromtimestamp(thought.timestamp).strftime('%H:%M:%S')
            thought_class = f"thought-{thought.type.value}"
            
            html += f"""
            <div class="thought {thought_class}">
                <div class="thought-header">
                    <span class="thought-type">{thought.type.value}</span>
                    <span class="thought-time">{thought_time}</span>
                </div>
                <div class="thought-content">{thought.content}</div>
            </div>
            """
        
        html += """
            </div>
        </div>
        <style>
            .thinking-visualization {
                font-family: 'Arial', sans-serif;
                max-width: 800px;
                margin: 0 auto;
            }
            .thinking-timeline {
                border-left: 2px solid #ccc;
                padding-left: 20px;
                margin-left: 10px;
            }
            .thought {
                margin-bottom: 15px;
                padding: 10px;
                border-radius: 5px;
                position: relative;
            }
            .thought:before {
                content: '';
                position: absolute;
                left: -26px;
                top: 10px;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                background: #fff;
                border: 2px solid #ccc;
            }
            .thought-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 5px;
                font-size: 0.8em;
            }
            .thought-type {
                font-weight: bold;
                text-transform: uppercase;
            }
            .thought-planning { background-color: #e3f2fd; }
            .thought-reasoning { background-color: #f1f8e9; }
            .thought-retrieval { background-color: #fff8e1; }
            .thought-analysis { background-color: #e8eaf6; }
            .thought-decision { background-color: #e0f2f1; }
            .thought-critique { background-color: #ffebee; }
            .thought-reflection { background-color: #f3e5f5; }
            .thought-action { background-color: #e8f5e9; }
            .thought-error { background-color: #ffebee; }
            .thought-summary { background-color: #fce4ec; }
        </style>
        """
        
        return html
    
    def _generate_text_visualization(self, session_id: str, thoughts: List[ThoughtNode]) -> str:
        """Generate text visualization."""
        session = self.graph.get_session(session_id)
        if not session:
            return "Session not found."
        
        text = f"Thinking Session: {session.task}\n"
        text += f"Agent: {session.agent_id}\n"
        text += f"Status: {session.status}\n"
        text += f"Started: {datetime.datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for thought in thoughts:
            thought_time = datetime.datetime.fromtimestamp(thought.timestamp).strftime('%H:%M:%S')
            text += f"[{thought_time}] {thought.type.value.upper()}: {thought.content}\n\n"
        
        return text
    
    def _generate_markdown_visualization(self, session_id: str, thoughts: List[ThoughtNode]) -> str:
        """Generate markdown visualization."""
        session = self.graph.get_session(session_id)
        if not session:
            return "Session not found."
        
        md = f"# Thinking Session: {session.task}\n\n"
        md += f"**Agent:** {session.agent_id}  \n"
        md += f"**Status:** {session.status}  \n"
        md += f"**Started:** {datetime.datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d %H:%M:%S')}  \n\n"
        
        md += "## Thinking Process\n\n"
        
        for thought in thoughts:
            thought_time = datetime.datetime.fromtimestamp(thought.timestamp).strftime('%H:%M:%S')
            md += f"### {thought.type.value.title()} ({thought_time})\n\n"
            md += f"{thought.content}\n\n"
            
            # Add horizontal rule between thoughts
            md += "---\n\n"
        
        return md

class LiveThinkingWidget:
    """Streamlit widget for displaying live thinking."""
    
    def __init__(self, thinking_engine: ThinkingEngine):
        self.thinking_engine = thinking_engine
        self.active_sessions = {}
        self.update_interval = 0.5  # seconds
    
    def display_session(self, session_id: str, container=None, max_thoughts: int = 10) -> None:
        """Display a thinking session."""
        if not container:
            container = st
        
        # Get session status
        status = self.thinking_engine.get_thinking_status(session_id)
        
        if "error" in status:
            container.error(status["error"])
            return
        
        # Display session info
        container.subheader(f"Thinking about: {status['task']}")
        
        col1, col2, col3 = container.columns(3)
        col1.metric("Agent", status["agent_id"])
        col2.metric("Status", status["status"])
        col3.metric("Thoughts", status["thought_count"])
        
        # Display current thought if active
        if status["status"] == "active" and status["current_thought"]:
            current_thought = status["current_thought"]
            
            container.markdown("### Current Thought Process")
            
            # Display thinking animation
            container.markdown(create_thinking_animation(), unsafe_allow_html=True)
            
            # Display current thought
            thought_type = current_thought["type"]
            thought_content = current_thought["content"]
            
            container.markdown(f"**{thought_type.title()}**: {thought_content}")
        
        # Display quality metrics if available
        if status["quality_metrics"]:
            metrics = status["quality_metrics"]
            container.markdown("### Quality Assessment")
            
            metrics_cols = container.columns(5)
            metrics_cols[0].metric("Relevance", f"{metrics['relevance']:.2f}")
            metrics_cols[1].metric("Coherence", f"{metrics['coherence']:.2f}")
            metrics_cols[2].metric("Accuracy", f"{metrics['accuracy']:.2f}")
            metrics_cols[3].metric("Completeness", f"{metrics['completeness']:.2f}")
            metrics_cols[4].metric("Efficiency", f"{metrics['efficiency']:.2f}")
            
            # Display progress bar
            overall_score = (metrics['relevance'] + metrics['coherence'] + metrics['accuracy'] + 
                            metrics['completeness'] + metrics['efficiency']) / 5.0
            
            container.markdown(create_progress_bar(overall_score, "Overall Quality"), unsafe_allow_html=True)
        
        # Display retrieved documents if available
        if status["retrieved_documents"]:
            container.markdown("### Retrieved Knowledge")
            
            for i, doc in enumerate(status["retrieved_documents"][:3]):
                container.markdown(f"**Document {i+1}** (Score: {doc['score']:.2f})")
                container.text_area(f"Content {i+1}", doc["content"], height=100, key=f"doc_{session_id}_{i}")
        
        # Register for updates if active
        if status["status"] == "active" and session_id not in self.active_sessions:
            self.active_sessions[session_id] = True
            
            # Schedule updates
            self._schedule_updates(session_id, container)
    
    def _schedule_updates(self, session_id: str, container) -> None:
        """Schedule updates for a session."""
        def update_callback():
            while session_id in self.active_sessions:
                # Check if session is still active
                status = self.thinking_engine.get_thinking_status(session_id)
                if status.get("status") != "active":
                    self.active_sessions.pop(session_id, None)
                    break
                
                # Wait for next update
                time.sleep(self.update_interval)
        
        # Start update thread
        threading.Thread(target=update_callback, daemon=True).start()
    
    def display_thinking_history(self, session_id: str, container=None) -> None:
        """Display thinking history."""
        if not container:
            container = st
        
        # Get visualization
        visualization = self.thinking_engine.get_thinking_visualization(session_id, format="html")
        
        # Display visualization
        container.markdown(visualization, unsafe_allow_html=True)

class RagShellWidget:
    """Streamlit widget for RAG shell interaction."""
    
    def __init__(self, thinking_engine: ThinkingEngine):
        self.thinking_engine = thinking_engine
        self.history = []
    
    def display(self, container=None) -> None:
        """Display the RAG shell."""
        if not container:
            container = st
        
        container.markdown("## Knowledge Retrieval Shell")
        
        # Display shell prompt
        container.markdown(create_ocr_text("Enter a query to search the knowledge base:"), unsafe_allow_html=True)
        
        # Input box
        query = container.text_input("Query", key="rag_query")
        
        col1, col2, col3 = container.columns([1, 1, 1])
        semantic = col1.button("Semantic Search")
        keyword = col2.button("Keyword Search")
        hybrid = col3.button("Hybrid Search")
        
        # Handle search
        if semantic or keyword or hybrid:
            if not query:
                container.warning("Please enter a query.")
                return
            
            # Determine strategy
            if semantic:
                strategy = RetrievalStrategy.SEMANTIC
            elif keyword:
                strategy = RetrievalStrategy.KEYWORD
            else:
                strategy = RetrievalStrategy.HYBRID
            
            # Create thinking session
            session_id = self.thinking_engine.start_thinking(f"RAG Query: {query}", "rag_shell")
            
            # Retrieve knowledge
            results = self.thinking_engine.retrieve_knowledge(session_id, query, strategy)
            
            # Add to history
            self.history.append({
                "query": query,
                "strategy": strategy,
                "session_id": session_id,
                "results": results,
                "timestamp": time.time()
            })
            
            # Complete thinking
            self.thinking_engine.complete_thinking(
                session_id,
                f"Completed retrieval for query: {query}. Found {len(results)} results."
            )
        
        # Display history
        if self.history:
            container.markdown("### Recent Queries")
            
            for i, entry in enumerate(reversed(self.history[-5:])):
                with container.expander(f"{entry['query']} ({entry['strategy'].value})"):
                    if entry["results"]:
                        for j, doc in enumerate(entry["results"][:3]):
                            st.markdown(f"**Result {j+1}** (Score: {doc.score:.2f})")
                            st.text_area(f"Content", doc.content, height=100, key=f"hist_{i}_{j}")
                    else:
                        st.info("No results found.")

class SelfCriticismEngine:
    """Engine for self-criticism and quality assessment."""
    
    def __init__(self, thinking_engine: ThinkingEngine):
        self.thinking_engine = thinking_engine
        self.quality_thresholds = {
            "relevance": QUALITY_THRESHOLD,
            "coherence": QUALITY_THRESHOLD,
            "accuracy": QUALITY_THRESHOLD,
            "completeness": QUALITY_THRESHOLD,
            "efficiency": QUALITY_THRESHOLD
        }
    
    def assess_quality(self, session_id: str, task: str, solution: str, context: Dict[str, Any] = None) -> QualityMetrics:
        """Assess quality of a solution."""
        context = context or {}
        
        # Create metrics
        metrics = QualityMetrics(
            critique_level=CritiqueLevel(context.get("critique_level", CritiqueLevel.MEDIUM.value))
        )
        
        # Assess relevance
        metrics.relevance = self._assess_relevance(task, solution)
        
        # Assess coherence
        metrics.coherence = self._assess_coherence(solution)
        
        # Assess accuracy
        metrics.accuracy = self._assess_accuracy(task, solution, context)
        
        # Assess completeness
        metrics.completeness = self._assess_completeness(task, solution)
        
        # Assess efficiency
        metrics.efficiency = self._assess_efficiency(solution, context)
        
        # Add critique thought
        self._add_critique(session_id, metrics)
        
        return metrics
    
    def _assess_relevance(self, task: str, solution: str) -> float:
        """Assess relevance of solution to task."""
        # Simple keyword matching for demonstration
        task_words = set(task.lower().split())
        solution_words = set(solution.lower().split())
        
        overlap = len(task_words.intersection(solution_words))
        
        if not task_words:
            return 1.0
        
        return min(1.0, overlap / len(task_words) * 2)
    
    def _assess_coherence(self, solution: str) -> float:
        """Assess coherence of solution."""
        # Simple heuristic based on sentence length and structure
        sentences = re.split(r'[.!?]+', solution)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check for very short or very long sentences
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        
        # Penalize extreme variance in sentence length
        variance = sum((length - avg_length) ** 2 for length in sentence_lengths) / len(sentence_lengths)
        normalized_variance = min(1.0, variance / 100)
        
        return 1.0 - normalized_variance
    
    def _assess_accuracy(self, task: str, solution: str, context: Dict[str, Any]) -> float:
        """Assess accuracy of solution."""
        # In a real implementation, this would verify facts against knowledge base
        # For demonstration, use a simple heuristic
        
        # Check for presence of uncertainty markers
        uncertainty_markers = ["maybe", "perhaps", "possibly", "might", "could be", "not sure", "uncertain"]
        uncertainty_count = sum(solution.lower().count(marker) for marker in uncertainty_markers)
        
        # Penalize uncertainty
        uncertainty_penalty = min(0.5, uncertainty_count * 0.1)
        
        return 1.0 - uncertainty_penalty
    
    def _assess_completeness(self, task: str, solution: str) -> float:
        """Assess completeness of solution."""
        # Extract key requirements from task
        requirements = self._extract_requirements(task)
        
        if not requirements:
            return 1.0
        
        # Check if solution addresses requirements
        addressed = 0
        for req in requirements:
            if req.lower() in solution.lower():
                addressed += 1
        
        return addressed / len(requirements)
    
    def _extract_requirements(self, task: str) -> List[str]:
        """Extract requirements from task."""
        # Simple extraction based on common patterns
        requirements = []
        
        # Look for bullet points
        bullet_pattern = r'[•\-*]\s*([^\n]+)'
        requirements.extend(re.findall(bullet_pattern, task))
        
        # Look for numbered items
        numbered_pattern = r'\d+\.\s*([^\n]+)'
        requirements.extend(re.findall(numbered_pattern, task))
        
        # Look for "need to", "should", "must"
        modal_pattern = r'(need to|should|must|has to|have to)\s+([^,.;]+)'
        for match in re.finditer(modal_pattern, task, re.IGNORECASE):
            requirements.append(match.group(2).strip())
        
        return requirements
    
    def _assess_efficiency(self, solution: str, context: Dict[str, Any]) -> float:
        """Assess efficiency of solution."""
        # In a real implementation, this would analyze computational complexity
        # For demonstration, use solution length as a proxy
        
        # Get expected length from context or use default
        expected_length = context.get("expected_length", 500)
        actual_length = len(solution)
        
        # Penalize solutions that are too long
        if actual_length > expected_length * 2:
            return 0.5
        elif actual_length > expected_length * 1.5:
            return 0.75
        else:
            return 1.0
    
    def _add_critique(self, session_id: str, metrics: QualityMetrics) -> None:
        """Add critique thought based on metrics."""
        overall_score = metrics.overall_score()
        
        if overall_score < QUALITY_THRESHOLD:
            critique = f"Self-critique: Overall quality ({overall_score:.2f}) is below threshold ({QUALITY_THRESHOLD}).\n\n"
            
            for metric, value in [
                ("Relevance", metrics.relevance),
                ("Coherence", metrics.coherence),
                ("Accuracy", metrics.accuracy),
                ("Completeness", metrics.completeness),
                ("Efficiency", metrics.efficiency)
            ]:
                if value < self.quality_thresholds.get(metric.lower(), QUALITY_THRESHOLD):
                    critique += f"- {metric}: {value:.2f} (below threshold)\n"
                    
                    if metric == "Relevance":
                        critique += "  Solution may not be addressing the task directly.\n"
                    elif metric == "Coherence":
                        critique += "  Solution may lack logical flow or structure.\n"
                    elif metric == "Accuracy":
                        critique += "  Solution may contain factual errors or uncertainties.\n"
                    elif metric == "Completeness":
                        critique += "  Solution may not address all requirements.\n"
                    elif metric == "Efficiency":
                        critique += "  Solution may be unnecessarily complex or verbose.\n"
            
            self.thinking_engine.add_thought(
                session_id=session_id,
                type=ThoughtType.CRITIQUE,
                content=critique
            )
        else:
            self.thinking_engine.add_thought(
                session_id=session_id,
                type=ThoughtType.REFLECTION,
                content=f"Self-assessment: Quality metrics are satisfactory (score: {overall_score:.2f})."
            )
    
    def suggest_improvements(self, session_id: str, metrics: QualityMetrics, solution: str) -> str:
        """Suggest improvements based on quality assessment."""
        if metrics.overall_score() >= QUALITY_THRESHOLD:
            return "No significant improvements needed."
        
        improvements = "Suggested improvements:\n\n"
        
        if metrics.relevance < self.quality_thresholds["relevance"]:
            improvements += "1. Improve relevance:\n"
            improvements += "   - Focus more directly on the task requirements\n"
            improvements += "   - Ensure key terms from the task are addressed\n"
            improvements += "   - Remove tangential or unrelated content\n\n"
        
        if metrics.coherence < self.quality_thresholds["coherence"]:
            improvements += "2. Improve coherence:\n"
            improvements += "   - Ensure logical flow between sentences and paragraphs\n"
            improvements += "   - Add transition phrases between sections\n"
            improvements += "   - Maintain consistent terminology throughout\n\n"
        
        if metrics.accuracy < self.quality_thresholds["accuracy"]:
            improvements += "3. Improve accuracy:\n"
            improvements += "   - Verify factual claims against reliable sources\n"
            improvements += "   - Remove or qualify uncertain statements\n"
            improvements += "   - Cite sources for important claims\n\n"
        
        if metrics.completeness < self.quality_thresholds["completeness"]:
            improvements += "4. Improve completeness:\n"
            improvements += "   - Address all requirements from the task\n"
            improvements += "   - Consider edge cases and potential issues\n"
            improvements += "   - Provide concrete examples where appropriate\n\n"
        
        if metrics.efficiency < self.quality_thresholds["efficiency"]:
            improvements += "5. Improve efficiency:\n"
            improvements += "   - Remove redundant or repetitive content\n"
            improvements += "   - Use more concise phrasing\n"
            improvements += "   - Focus on the most important aspects\n\n"
        
        # Add improvement thought
        self.thinking_engine.add_thought(
            session_id=session_id,
            type=ThoughtType.REFLECTION,
            content=improvements
        )
        
        return improvements

# Initialize the thinking engine as a singleton
_thinking_engine = None

def get_thinking_engine() -> ThinkingEngine:
    """Get the thinking engine singleton."""
    global _thinking_engine
    if _thinking_engine is None:
        _thinking_engine = ThinkingEngine()
        _thinking_engine.initialize()
    return _thinking_engine

# Example usage
if __name__ == "__main__":
    # Initialize the thinking engine
    engine = get_thinking_engine()
    
    # Initialize knowledge base with sample data
    kb = engine.knowledge_base
    kb.add_document(
        content="The Skyscope Sentinel Intelligence system is an AI agentic swarm system with 10,000 agents organized into 100 pipelines.",
        source="system_description.txt"
    )
    kb.add_document(
        content="Each pipeline in the Skyscope system contains 100 specialized agents with different roles and capabilities.",
        source="pipeline_structure.txt"
    )
    kb.add_document(
        content="The system can generate income through freelance work, affiliate marketing, crypto trading, and content creation.",
        source="business_operations.txt"
    )
    
    # Save knowledge base
    kb.save()
    
    # Create a thinking session
    session_id = engine.start_thinking(
        task="Design a new affiliate marketing strategy for the Skyscope system",
        agent_id="marketing_specialist_1"
    )
    
    # Add some thoughts
    engine.add_thought(
        session_id=session_id,
        type=ThoughtType.PLANNING,
        content="I'll start by researching current affiliate marketing trends and then develop a strategy tailored to Skyscope's capabilities."
    )
    
    # Retrieve knowledge
    engine.retrieve_knowledge(
        session_id=session_id,
        query="Skyscope system capabilities"
    )
    
    # Add more thoughts
    engine.add_thought(
        session_id=session_id,
        type=ThoughtType.REASONING,
        content="Based on the retrieved information, Skyscope has 10,000 agents that can be leveraged for affiliate marketing. I'll focus on using these agents to identify high-value affiliate opportunities and automate content creation."
    )
    
    engine.add_thought(
        session_id=session_id,
        type=ThoughtType.DECISION,
        content="I've decided to recommend a multi-channel affiliate strategy focusing on SaaS products with high commissions. The strategy will use Skyscope's agents to create targeted content, track performance, and optimize campaigns automatically."
    )
    
    # Evaluate quality
    metrics = QualityMetrics(
        relevance=0.9,
        coherence=0.85,
        accuracy=0.95,
        completeness=0.7,  # Below threshold
        efficiency=0.8
    )
    
    engine.evaluate_quality(session_id, metrics)
    
    # Add final thoughts after self-criticism
    engine.add_thought(
        session_id=session_id,
        type=ThoughtType.ACTION,
        content="Based on self-criticism, I need to improve the completeness of the strategy. I'll add specific details about implementation steps, required resources, and expected ROI."
    )
    
    # Complete the session
    engine.complete_thinking(
        session_id=session_id,
        summary="Developed an affiliate marketing strategy for Skyscope focusing on SaaS products with high commissions. The strategy leverages Skyscope's 10,000 agents for content creation, campaign optimization, and performance tracking. Self-criticism identified a need for more implementation details, which were subsequently added."
    )
    
    # Display thinking visualization
    visualization = engine.get_thinking_visualization(session_id, format="text")
    print(visualization)
