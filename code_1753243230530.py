# Core RAG System Configuration
import json
import sqlite3
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Generator
from datetime import datetime
import threading
import time

# Create the main configuration file
config = {
    "system": {
        "name": "Skyscope RAG System",
        "version": "1.0.0",
        "base_dir": "/home/user/skyscope_rag",
        "parquet_source": "/Users/skyscope.cloud/Documents/github-code",
        "max_workers": 20,
        "batch_size": 10000
    },
    "indexing": {
        "chunk_size": 8192,
        "overlap": 512,
        "min_file_size": 50,
        "max_file_size": 1048576,
        "supported_languages": [
            "python", "javascript", "typescript", "java", "cpp", "c", "csharp",
            "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
            "sql", "html", "css", "shell", "bash", "powershell", "yaml",
            "json", "xml", "markdown", "dockerfile", "makefile", "cmake",
            "lua", "perl", "haskell", "clojure", "elixir", "erlang"
        ]
    },
    "embeddings": {
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 32,
        "cache_embeddings": True
    },
    "search": {
        "max_results": 100,
        "similarity_threshold": 0.7,
        "keyword_boost": 1.5,
        "semantic_boost": 1.0
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "default_model": "codellama",
        "context_window": 8192,
        "temperature": 0.1,
        "timeout": 300
    },
    "agents": {
        "max_concurrent": 5,
        "memory_limit": "8GB",
        "terminal_timeout": 30,
        "browser_timeout": 60
    }
}

# Save configuration
config_path = Path("/home/user/skyscope_rag/config/config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Created system configuration:")
print(f"📄 Config saved to: {config_path}")
print(f"🔧 System name: {config['system']['name']}")
print(f"📊 Max workers: {config['system']['max_workers']}")
print(f"🤖 Default Ollama model: {config['ollama']['default_model']}")
print(f"🔍 Supported languages: {len(config['indexing']['supported_languages'])}")