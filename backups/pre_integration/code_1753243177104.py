# Main RAG System Interface
import argparse
import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Optional

class SkyscopeRAGSystem:
    """Main interface for the Skyscope RAG system"""
    
    def __init__(self, config_path: str = "/home/user/skyscope_rag/config/config.json"):
        self.config_path = config_path
        self.load_config()
        
        # Initialize all components
        self.indexer = indexer
        self.embedding_engine = embedding_engine
        self.search_engine = search_engine
        self.rag_engine = rag_engine
        self.execution_agent = execution_agent
        self.parquet_processor = parquet_processor
        self.memory_manager = memory_manager
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Skyscope RAG System initialized")
    
    def load_config(self):
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            self.config = {"system": {"name": "Skyscope RAG"}}
    
    def index_parquet_files(self, directory_path: str, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Index Parquet files from the GitHub codebase"""
        self.logger.info(f"Starting indexing of parquet files from: {directory_path}")
        
        if not Path(directory_path).exists():
            return {
                'success': False,
                'error': f'Directory not found: {directory_path}'
            }
        
        start_time = time.time()
        result = self.parquet_processor.process_directory(directory_path, max_files)
        elapsed = time.time() - start_time
        
        result['elapsed_total'] = elapsed
        result['files_per_second'] = result['total_processed'] / elapsed if elapsed > 0 else 0
        
        self.logger.info(f"Indexing completed in {elapsed:.1f}s")
        return result
    
    def search(self, query: str, search_type: str = "hybrid", 
              language: Optional[str] = None, max_results: int = 10) -> List[Dict]:
        """Search the codebase"""
        self.logger.info(f"Searching for: {query} (type: {search_type})")
        
        if search_type == "keyword":
            return self.search_engine.keyword_search(query, language, max_results=max_results)
        elif search_type == "semantic":
            return self.search_engine.semantic_search(query, language, max_results=max_results)
        else:  # hybrid
            return self.search_engine.hybrid_search(query, language, max_results=max_results)
    
    def ask(self, question: str, language: Optional[str] = None, 
           execute_code: bool = False) -> Dict[str, Any]:
        """Ask a question and get RAG-enhanced response"""
        self.logger.info(f"Processing question: {question}")
        
        return self.execution_agent.search_and_execute(
            query=question,
            language=language,
            execute_code=execute_code
        )
    
    def switch_ollama_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model"""
        return self.rag_engine.switch_model(model_name)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        with sqlite3.connect(self.indexer.db_path) as conn:
            stats = {}
            
            # File counts
            cursor = conn.execute("SELECT COUNT(*) FROM code_files")
            stats['total_files'] = cursor.fetchone()[0]
            
            # Language distribution
            cursor = conn.execute("""
                SELECT language, COUNT(*) as count 
                FROM code_files 
                GROUP BY language 
                ORDER BY count DESC 
                LIMIT 10
            """)
            stats['top_languages'] = dict(cursor.fetchall())
            
            # Total chunks
            cursor = conn.execute("SELECT COUNT(*) FROM code_chunks")
            stats['total_chunks'] = cursor.fetchone()[0]
            
            # Chunks with embeddings
            cursor = conn.execute("SELECT COUNT(*) FROM code_chunks WHERE embedding_vector IS NOT NULL")
            stats['chunks_with_embeddings'] = cursor.fetchone()[0]
            
            # Memory usage
            stats['memory_keys'] = len(self.memory_manager.list_keys())
            stats['execution_history'] = len(self.execution_agent.execution_history)
            
            # Available models
            stats['available_ollama_models'] = self.rag_engine.available_models
            stats['current_model'] = self.rag_engine.default_model
        
        return stats
    
    def export_results(self, results: List[Dict], output_file: str):
        """Export search results to file"""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results exported to {output_file}")
        except Exception as e:
            self.logger.error(f"Export failed: {e}")

def create_cli():
    """Create command-line interface"""
    parser = argparse.ArgumentParser(description="Skyscope RAG System - Code Search and Generation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index Parquet files')
    index_parser.add_argument('directory', help='Directory containing Parquet files')
    index_parser.add_argument('--max-files', type=int, help='Maximum files to process')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search codebase')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--type', choices=['keyword', 'semantic', 'hybrid'], 
                              default='hybrid', help='Search type')
    search_parser.add_argument('--language', help='Filter by programming language')
    search_parser.add_argument('--max-results', type=int, default=10, help='Maximum results')
    search_parser.add_argument('--output', help='Export results to file')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask question with RAG')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--language', help='Filter by programming language')
    ask_parser.add_argument('--execute', action='store_true', help='Execute found code')
    
    # Stats command
    subparsers.add_parser('stats', help='Show system statistics')
    
    # Model command
    model_parser = subparsers.add_parser('model', help='Switch Ollama model')
    model_parser.add_argument('name', help='Model name')
    
    return parser

# Initialize main system
rag_system = SkyscopeRAGSystem()

print("🎉 Skyscope RAG System Ready!")
print("="*60)
print("🚀 System Components Initialized:")
print("  ✅ Parquet Code Indexer")
print("  ✅ Local Semantic Search Engine")
print("  ✅ Keyword Search (FTS5)")
print("  ✅ Ollama RAG Integration")
print("  ✅ Agent Execution Framework")
print("  ✅ Memory Management System")
print("  ✅ CLI Interface")
print()
print("🔧 Available Commands:")
print("  • index <directory>     - Index Parquet files")
print("  • search <query>        - Search codebase")
print("  • ask <question>        - RAG-enhanced Q&A")
print("  • stats                 - System statistics")
print("  • model <name>          - Switch Ollama model")
print()
print("💡 Example Usage:")
print("  rag_system.ask('How to implement a REST API in Python?')")
print("  rag_system.search('machine learning algorithms', language='python')")
print("  rag_system.index_parquet_files('/path/to/github-code')")
print()
print(f"📊 Current Stats: {rag_system.get_stats()}")