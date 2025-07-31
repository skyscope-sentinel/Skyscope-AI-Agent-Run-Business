# 2. MacOS-Use Integration (Fixed)
macos_use_integration = '''"""
Skyscope RAG - MacOS-Use Integration
Enables macOS system automation using code found through RAG system
"""

import json
import subprocess
import asyncio
from typing import Dict, List, Optional, Any
import requests
import os
from pathlib import Path

class MacOSUseIntegration:
    """Integration with macos-use Pinokio app for system automation"""
    
    def __init__(self, rag_system, macos_use_endpoint: str = "http://localhost:8081"):
        self.rag_system = rag_system
        self.macos_use_endpoint = macos_use_endpoint
        self.session_id = None
        
    async def initialize_macos_session(self) -> Dict[str, Any]:
        """Initialize macOS automation session"""
        try:
            # Connect to macos-use Pinokio app
            response = requests.post(f"{self.macos_use_endpoint}/session/create", 
                                   json={"platform": "darwin"})
            
            if response.status_code == 200:
                session_data = response.json()
                self.session_id = session_data.get("session_id")
                
                return {
                    "status": "success",
                    "session_id": self.session_id,
                    "message": "macOS session initialized"
                }
            else:
                # Fallback to local system commands
                return await self._initialize_local_system()
                
        except Exception as e:
            return await self._initialize_local_system()
    
    async def find_system_automation_code(self, task: str, language: str = "bash") -> Dict[str, Any]:
        """Find relevant system automation code from RAG system"""
        
        # Search for macOS/system automation code
        search_queries = [
            f"macOS {task} automation",
            f"{task} shell script",
            f"AppleScript {task}",
            f"{task} system administration",
            f"osascript {task}",
            f"{task} terminal command"
        ]
        
        results = []
        for query in search_queries:
            search_results = self.rag_system.search(
                query=query,
                language=language,
                max_results=5
            )
            results.extend(search_results)
        
        return {
            "task": task,
            "language": language,
            "code_examples": results[:10],
            "total_found": len(results)
        }

class MacOSUseAPI:
    """API interface for Pinokio orchestration"""
    
    def __init__(self, rag_system):
        self.macos_integration = MacOSUseIntegration(rag_system)
    
    async def handle_system_request(self, request_data: Dict) -> Dict:
        """Handle incoming system automation requests from Pinokio"""
        
        task_type = request_data.get("type", "")
        
        if task_type == "initialize":
            return await self.macos_integration.initialize_macos_session()
        
        elif task_type == "find_code":
            task = request_data.get("task", "")
            language = request_data.get("language", "bash")
            return await self.macos_integration.find_system_automation_code(task, language)
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
'''

# Save macOS-use integration
with open(integration_dir / "macos_use" / "integration.py", "w") as f:
    f.write(macos_use_integration)

print("✅ Created MacOS-Use integration module")