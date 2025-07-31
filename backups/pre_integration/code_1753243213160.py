# Core RAG System - Parquet Indexer
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

class ParquetCodeIndexer:
    """Fast indexing system for Parquet code files"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "indexes" / "code_index.db"
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with optimized schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Main code files table
                CREATE TABLE IF NOT EXISTS code_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_hash TEXT UNIQUE NOT NULL,
                    repo_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    language TEXT,
                    license TEXT,
                    size INTEGER,
                    content TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding_id TEXT
                );
                
                -- Indexes for fast searching
                CREATE INDEX IF NOT EXISTS idx_repo_name ON code_files(repo_name);
                CREATE INDEX IF NOT EXISTS idx_language ON code_files(language);
                CREATE INDEX IF NOT EXISTS idx_license ON code_files(license);
                CREATE INDEX IF NOT EXISTS idx_size ON code_files(size);
                CREATE INDEX IF NOT EXISTS idx_file_path ON code_files(file_path);
                CREATE INDEX IF NOT EXISTS idx_hash ON code_files(file_hash);
                
                -- Full-text search table
                CREATE VIRTUAL TABLE IF NOT EXISTS code_fts USING fts5(
                    content, repo_name, file_path,
                    content_id UNINDEXED
                );
                
                -- Chunks table for semantic search
                CREATE TABLE IF NOT EXISTS code_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER REFERENCES code_files(id),
                    chunk_index INTEGER,
                    content TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    embedding_vector BLOB
                );
                
                CREATE INDEX IF NOT EXISTS idx_file_id ON code_chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_chunk_index ON code_chunks(chunk_index);
                
                -- Metadata and statistics
                CREATE TABLE IF NOT EXISTS index_stats (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
        self.logger.info(f"Database initialized: {self.db_path}")
    
    def get_language_from_path(self, file_path: str) -> str:
        """Determine programming language from file extension"""
        ext_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp',
            '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php',
            '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
            '.r': 'r', '.sql': 'sql', '.html': 'html', '.css': 'css',
            '.sh': 'shell', '.bash': 'bash', '.ps1': 'powershell',
            '.yml': 'yaml', '.yaml': 'yaml', '.json': 'json',
            '.xml': 'xml', '.md': 'markdown', '.dockerfile': 'dockerfile',
            '.makefile': 'makefile', '.cmake': 'cmake', '.lua': 'lua',
            '.pl': 'perl', '.hs': 'haskell', '.clj': 'clojure',
            '.ex': 'elixir', '.erl': 'erlang'
        }
        
        path_lower = file_path.lower()
        for ext, lang in ext_map.items():
            if path_lower.endswith(ext):
                return lang
        
        # Special cases
        if 'dockerfile' in path_lower:
            return 'dockerfile'
        elif 'makefile' in path_lower:
            return 'makefile'
        
        return 'unknown'
    
    def hash_content(self, content: str) -> str:
        """Generate hash for deduplication"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def chunk_content(self, content: str, chunk_size: int = 8192, overlap: int = 512) -> List[Tuple[str, int, int]]:
        """Split content into overlapping chunks"""
        lines = content.split('\n')
        chunks = []
        
        start_line = 0
        while start_line < len(lines):
            end_line = min(start_line + chunk_size, len(lines))
            chunk_lines = lines[start_line:end_line]
            chunk_content = '\n'.join(chunk_lines)
            
            if chunk_content.strip():  # Only add non-empty chunks
                chunks.append((chunk_content, start_line, end_line))
            
            if end_line >= len(lines):
                break
            
            start_line = end_line - overlap
        
        return chunks

# Initialize the indexer
indexer = ParquetCodeIndexer("/home/user/skyscope_rag")
print("✅ ParquetCodeIndexer created")
print(f"📊 Database: {indexer.db_path}")
print("🔧 Schema initialized with FTS5 and chunk support")