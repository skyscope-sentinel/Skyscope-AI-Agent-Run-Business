# 2. MacOS-Use Integration
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
    
    async def _initialize_local_system(self) -> Dict[str, Any]:
        """Fallback local system initialization"""
        try:
            # Check if running on macOS
            result = subprocess.run(['uname', '-s'], capture_output=True, text=True)
            if 'Darwin' in result.stdout:
                return {
                    "status": "success",
                    "session_id": "local_macos_session",
                    "message": "Local macOS session initialized",
                    "platform": "Darwin"
                }
            else:
                return {
                    "status": "warning",
                    "session_id": "local_unix_session", 
                    "message": "Non-macOS system detected",
                    "platform": result.stdout.strip()
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to initialize system: {str(e)}"
            }
    
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
        
        # Also search for Python system automation
        if language != "python":
            python_results = self.rag_system.search(
                f"Python {task} system automation",
                language="python",
                max_results=3
            )
            results.extend(python_results)
        
        # Deduplicate and rank results
        unique_results = []
        seen_files = set()
        
        for result in results:
            file_id = result.get('file_path', '') + result.get('content_hash', '')
            if file_id not in seen_files:
                seen_files.add(file_id)
                unique_results.append(result)
        
        return {
            "task": task,
            "language": language,
            "code_examples": unique_results[:10],
            "total_found": len(unique_results)
        }
    
    async def execute_system_task(self, task: str, parameters: Dict = None) -> Dict[str, Any]:
        """Execute system automation task using RAG-found code"""
        
        # Find relevant system automation code
        code_results = await self.find_system_automation_code(task, "bash")
        
        if not code_results["code_examples"]:
            # Try other languages
            for lang in ["python", "applescript", "shell"]:
                code_results = await self.find_system_automation_code(task, lang)
                if code_results["code_examples"]:
                    break
        
        if not code_results["code_examples"]:
            return {
                "status": "error",
                "message": f"No system automation code found for task: {task}"
            }
        
        # Generate execution plan using RAG
        rag_response = self.rag_system.ask(
            f"How to automate this macOS system task: {task}. Provide safe shell commands or AppleScript.",
            max_results=5
        )
        
        execution_plan = {
            "task": task,
            "parameters": parameters or {},
            "code_examples": code_results["code_examples"],
            "rag_guidance": rag_response.get("rag_response", ""),
            "execution_steps": []
        }
        
        # If macos-use endpoint is available, send the plan
        if self.session_id and self.macos_use_endpoint:
            try:
                response = requests.post(
                    f"{self.macos_use_endpoint}/execute",
                    json={
                        "session_id": self.session_id,
                        "task": task,
                        "parameters": parameters,
                        "code_context": code_results["code_examples"][:3]
                    }
                )
                
                if response.status_code == 200:
                    execution_result = response.json()
                    execution_plan["execution_result"] = execution_result
                    execution_plan["status"] = "completed"
                else:
                    execution_plan["status"] = "failed"
                    execution_plan["error"] = response.text
                    
            except Exception as e:
                execution_plan["status"] = "error"
                execution_plan["error"] = str(e)
        
        return execution_plan
    
    async def generate_system_script(self, tasks: List[str], output_format: str = "bash") -> Dict[str, Any]:
        """Generate comprehensive system automation script"""
        
        script_components = []
        
        for task in tasks:
            # Find relevant automation code
            code_results = await self.find_system_automation_code(task, output_format)
            
            # Generate script code using RAG
            rag_response = self.rag_system.ask(
                f"Generate {output_format} script for macOS task: {task}",
                max_results=3
            )
            
            script_components.append({
                "task": task,
                "code_examples": code_results["code_examples"][:3],
                "generated_code": rag_response.get("rag_response", ""),
                "files_referenced": len(code_results["code_examples"])
            })
        
        # Combine into full script
        full_script = self._generate_complete_system_script(
            tasks, script_components, output_format
        )
        
        return {
            "tasks": tasks,
            "output_format": output_format,
            "components": script_components,
            "full_script": full_script,
            "total_code_references": sum(comp["files_referenced"] for comp in script_components)
        }
    
    def _generate_complete_system_script(self, tasks: List[str], components: List[Dict], format_type: str) -> str:
        """Generate a complete executable system script"""
        
        if format_type == "bash":
            script_template = f'''#!/bin/bash
# Generated macOS System Automation Script
# Created using Skyscope RAG System with {sum(comp["files_referenced"] for comp in components)} code references
# Tasks: {", ".join(tasks)}

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
LOG_FILE="$HOME/skyscope_automation.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Logging function
log() {{
    echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}}

# Error handling
error_exit() {{
    log "ERROR: $1"
    exit 1
}}

log "Starting macOS automation tasks..."

'''
            
            # Add functions for each task
            for i, component in enumerate(components):
                func_name = component["task"].replace(" ", "_").replace("-", "_").lower()
                script_template += f'''
# Task {i+1}: {component["task"]}
# Code references: {component["files_referenced"]} files
{func_name}() {{
    log "Executing task: {component["task"]}"
    
    # Generated automation code
{self._indent_code(component["generated_code"], 4)}
    
    log "Completed task: {component["task"]}"
}}

'''
        
            # Add main execution
            script_template += '''
# Main execution
main() {
    log "macOS automation script started"
    
'''
            for component in components:
                func_name = component["task"].replace(" ", "_").replace("-", "_").lower()
                script_template += f'    {func_name}\n'
            
            script_template += '''    
    log "All tasks completed successfully"
}

# Run main function
main "$@"
'''
        
        elif format_type == "python":
            script_template = f'''#!/usr/bin/env python3
"""
Generated macOS System Automation Script
Created using Skyscope RAG System with {sum(comp["files_referenced"] for comp in components)} code references
Tasks: {", ".join(tasks)}
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
log_file = Path.home() / "skyscope_automation.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

class MacOSAutomation:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing macOS automation")
    
'''
            
            # Add methods for each task
            for i, component in enumerate(components):
                method_name = component["task"].replace(" ", "_").replace("-", "_").lower()
                script_template += f'''
    def {method_name}(self):
        """Task: {component["task"]}"""
        # Code references: {component["files_referenced"]} files
        
        self.logger.info("Executing task: {component["task"]}")
        
        try:
{self._indent_code(component["generated_code"], 12)}
            
            self.logger.info("Completed task: {component["task"]}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed task {component["task"]}: {{str(e)}}")
            return False
'''
        
            # Add main execution
            script_template += '''
    
    def run_all_tasks(self):
        """Execute all automation tasks"""
        self.logger.info("Starting all automation tasks")
        
        tasks = [
'''
            for component in components:
                method_name = component["task"].replace(" ", "_").replace("-", "_").lower()
                script_template += f'            self.{method_name},\n'
            
            script_template += '''        ]
        
        failed_tasks = []
        for task in tasks:
            if not task():
                failed_tasks.append(task.__name__)
        
        if failed_tasks:
            self.logger.error(f"Failed tasks: {failed_tasks}")
            return False
        else:
            self.logger.info("All tasks completed successfully")
            return True

if __name__ == "__main__":
    automation = MacOSAutomation()
    success = automation.run_all_tasks()
    sys.exit(0 if success else 1)
'''
        
        return script_template
    
    def _indent_code(self, code: str, spaces: int) -> str:
        """Indent code block"""
        lines = code.split('\n')
        indented_lines = [' ' * spaces + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)

class MacOSUseAPI:
    """API interface for Pinokio orchestration"""
    
    def __init__(self, rag_system):
        self.macos_integration = MacOSUseIntegration(rag_system)
    
    async def handle_system_request(self, request_data: Dict) -> Dict:
        """Handle incoming system automation requests from Pinokio"""
        
        task_type = request_data.get("type", "")
        
        if task_type == "initialize":
            return await self.macos_integration.initialize_macos_session()
        
        elif task_type == "automate":
            task = request_data.get("task", "")
            parameters = request_data.get("parameters", {})
            return await self.macos_integration.execute_system_task(task, parameters)
        
        elif task_type == "generate_script":
            tasks = request_data.get("tasks", [])
            format_type = request_data.get("format", "bash")
            return await self.macos_integration.generate_system_script(tasks, format_type)
        
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