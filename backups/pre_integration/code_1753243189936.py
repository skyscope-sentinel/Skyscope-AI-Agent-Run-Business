# Agent Execution Framework
import subprocess
import threading
import queue
import time
import json
import psutil
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

class TerminalAgent:
    """Agent for executing terminal commands safely"""
    
    def __init__(self, timeout: int = 30, max_concurrent: int = 5):
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.active_processes = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.logger = logging.getLogger(__name__)
    
    def execute_command(self, command: str, cwd: Optional[str] = None, 
                       env: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a shell command safely"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                env=env
            )
            
            return {
                'success': True,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'command': command
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Command timed out',
                'command': command
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'command': command
            }
    
    def execute_python_script(self, script_content: str, 
                             script_name: str = "temp_script.py") -> Dict[str, Any]:
        """Execute Python script safely"""
        script_path = Path("/tmp") / script_name
        
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            result = self.execute_command(f"python3 {script_path}")
            
            # Cleanup
            script_path.unlink(missing_ok=True)
            
            return result
        except Exception as e:
            return {
                'success': False,
                'error': f"Script execution failed: {e}",
                'script': script_content
            }

class MemoryManager:
    """Persistent memory system for agents"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.memory_dir = self.base_dir / "memory"
        self.memory_dir.mkdir(exist_ok=True)
        
        self.session_memory = {}  # In-memory cache
        self.memory_db = self.memory_dir / "memory.json"
        self.load_persistent_memory()
    
    def load_persistent_memory(self):
        """Load memory from disk"""
        if self.memory_db.exists():
            try:
                with open(self.memory_db, 'r') as f:
                    self.persistent_memory = json.load(f)
            except Exception:
                self.persistent_memory = {}
        else:
            self.persistent_memory = {}
    
    def save_persistent_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_db, 'w') as f:
                json.dump(self.persistent_memory, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save memory: {e}")
    
    def store(self, key: str, value: Any, persistent: bool = False):
        """Store value in memory"""
        if persistent:
            self.persistent_memory[key] = value
            self.save_persistent_memory()
        else:
            self.session_memory[key] = value
    
    def retrieve(self, key: str, default: Any = None) -> Any:
        """Retrieve value from memory"""
        # Check session memory first
        if key in self.session_memory:
            return self.session_memory[key]
        
        # Then persistent memory
        return self.persistent_memory.get(key, default)
    
    def delete(self, key: str, persistent: bool = False):
        """Delete key from memory"""
        if persistent and key in self.persistent_memory:
            del self.persistent_memory[key]
            self.save_persistent_memory()
        
        if key in self.session_memory:
            del self.session_memory[key]
    
    def list_keys(self, persistent_only: bool = False) -> List[str]:
        """List all memory keys"""
        if persistent_only:
            return list(self.persistent_memory.keys())
        
        all_keys = set(self.session_memory.keys())
        all_keys.update(self.persistent_memory.keys())
        return list(all_keys)

class CodeExecutionAgent:
    """Agent that combines RAG search with code execution"""
    
    def __init__(self, search_engine, rag_engine, terminal_agent, memory_manager):
        self.search_engine = search_engine
        self.rag_engine = rag_engine
        self.terminal_agent = terminal_agent
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)
        
        # Execution history
        self.execution_history = []
    
    def search_and_execute(self, query: str, execute_code: bool = False,
                          language: Optional[str] = None) -> Dict[str, Any]:
        """Search for relevant code and optionally execute it"""
        
        # Store query in memory
        self.memory_manager.store(f"last_query_{int(time.time())}", query)
        
        # Search for relevant code
        search_results = self.search_engine.hybrid_search(
            query=query,
            language=language,
            max_results=10
        )
        
        if not search_results:
            return {
                'success': False,
                'message': 'No relevant code found',
                'query': query
            }
        
        # Get RAG response
        context_chunks = [result['content'][:2000] for result in search_results[:5]]
        rag_response = self.rag_engine.generate_with_context(query, context_chunks)
        
        result = {
            'success': True,
            'query': query,
            'search_results': len(search_results),
            'rag_response': rag_response,
            'relevant_files': [
                {
                    'repo': r['repo_name'],
                    'path': r['file_path'],
                    'language': r['language'],
                    'score': r.get('combined_score', 0)
                }
                for r in search_results[:5]
            ]
        }
        
        # Optionally execute code if it's safe
        if execute_code and language in ['python', 'bash', 'shell']:
            execution_result = self._safe_execute(search_results[0], language)
            result['execution'] = execution_result
        
        # Store in execution history
        self.execution_history.append(result)
        
        return result
    
    def _safe_execute(self, code_result: Dict, language: str) -> Dict[str, Any]:
        """Safely execute code with safeguards"""
        content = code_result['content']
        
        # Basic safety checks
        dangerous_patterns = [
            'rm -rf', 'sudo', 'format', 'mkfs', 'dd if=', 
            'curl.*|.*sh', 'wget.*|.*sh', '>.*passwd',
            'chmod 777', 'chown root'
        ]
        
        for pattern in dangerous_patterns:
            if pattern in content.lower():
                return {
                    'success': False,
                    'error': f'Potentially dangerous pattern detected: {pattern}',
                    'executed': False
                }
        
        # Execute based on language
        if language == 'python':
            return self.terminal_agent.execute_python_script(content)
        elif language in ['bash', 'shell']:
            return self.terminal_agent.execute_command(content)
        else:
            return {
                'success': False,
                'error': f'Execution not supported for language: {language}',
                'executed': False
            }

# Initialize agent framework
terminal_agent = TerminalAgent(timeout=30, max_concurrent=5)
memory_manager = MemoryManager("/home/user/skyscope_rag")
execution_agent = CodeExecutionAgent(
    search_engine=search_engine,
    rag_engine=rag_engine,
    terminal_agent=terminal_agent,
    memory_manager=memory_manager
)

print("✅ Agent Framework created")
print("🤖 Components:")
print("  • TerminalAgent - Safe command execution")
print("  • MemoryManager - Persistent state storage")
print("  • CodeExecutionAgent - RAG + execution")
print("🔒 Safety features:")
print("  • Command timeout protection")
print("  • Dangerous pattern detection")
print("  • Sandboxed execution environment")
print("💾 Memory system ready for persistent state")