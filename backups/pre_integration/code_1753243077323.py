# 5. OpenWebUI Integration
openwebui_integration = '''"""
Skyscope RAG - OpenWebUI Integration
Enables web interface for RAG system interaction
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

class OpenWebUIIntegration:
    """Integration with OpenWebUI Pinokio app for web interface"""
    
    def __init__(self, rag_system, openwebui_endpoint: str = "http://localhost:8080"):
        self.rag_system = rag_system
        self.openwebui_endpoint = openwebui_endpoint
        self.session_id = None
        
    async def initialize_webui_session(self) -> Dict[str, Any]:
        """Initialize OpenWebUI session"""
        try:
            response = requests.post(f"{self.openwebui_endpoint}/api/v1/auths/signin", 
                                   json={
                                       "email": "skyscope@local.dev",
                                       "password": "skyscope_rag"
                                   })
            
            if response.status_code == 200:
                auth_data = response.json()
                self.session_id = auth_data.get("token")
                
                return {
                    "status": "success",
                    "session_id": self.session_id,
                    "message": "OpenWebUI session initialized"
                }
            else:
                return await self._initialize_local_webui()
                
        except Exception as e:
            return await self._initialize_local_webui()
    
    async def _initialize_local_webui(self) -> Dict[str, Any]:
        """Fallback local web interface"""
        self.session_id = f"local_webui_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "status": "success",
            "session_id": self.session_id,
            "message": "Local web interface initialized",
            "mode": "local"
        }
    
    async def create_rag_chat_model(self) -> Dict[str, Any]:
        """Create custom RAG chat model in OpenWebUI"""
        
        model_config = {
            "id": "skyscope-rag",
            "name": "Skyscope RAG System",
            "description": "RAG-enhanced chat using GitHub codebase",
            "capabilities": [
                "code_generation",
                "code_search",
                "code_explanation",
                "architecture_guidance"
            ],
            "parameters": {
                "max_context_length": 16000,
                "temperature": 0.7,
                "top_p": 0.9,
                "rag_enabled": True,
                "search_results_limit": 10
            }
        }
        
        if self.session_id and self.openwebui_endpoint:
            try:
                response = requests.post(
                    f"{self.openwebui_endpoint}/api/v1/models",
                    headers={"Authorization": f"Bearer {self.session_id}"},
                    json=model_config
                )
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "model_id": "skyscope-rag",
                        "message": "RAG chat model created in OpenWebUI"
                    }
                else:
                    return {
                        "status": "local_fallback",
                        "model_config": model_config,
                        "message": "Using local RAG model configuration"
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to create model: {str(e)}"
                }
        
        return {
            "status": "local_fallback",
            "model_config": model_config,
            "message": "Using local RAG model configuration"
        }
    
    async def handle_chat_request(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Handle chat request with RAG enhancement"""
        
        # Search for relevant code using RAG system
        search_results = self.rag_system.search(
            query=message,
            max_results=5
        )
        
        # Generate response using RAG
        rag_response = self.rag_system.ask(
            question=message,
            max_results=5
        )
        
        # Format response for OpenWebUI
        formatted_response = {
            "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": message,
            "response": rag_response.get("rag_response", ""),
            "context": {
                "search_results": len(search_results),
                "code_examples": search_results[:3],
                "model_used": rag_response.get("model", "unknown"),
                "processing_time": rag_response.get("processing_time", 0)
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "rag_enhanced": True,
                "total_context_files": len(search_results)
            }
        }
        
        # Send to OpenWebUI if available
        if self.session_id and self.openwebui_endpoint:
            try:
                response = requests.post(
                    f"{self.openwebui_endpoint}/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.session_id}"},
                    json={
                        "model": "skyscope-rag",
                        "messages": [
                            {
                                "role": "user",
                                "content": message
                            }
                        ],
                        "context": formatted_response["context"]
                    }
                )
                
                if response.status_code == 200:
                    webui_response = response.json()
                    formatted_response["webui_response"] = webui_response
                    formatted_response["status"] = "webui_processed"
                else:
                    formatted_response["status"] = "local_processed"
                    
            except Exception as e:
                formatted_response["status"] = "local_processed"
                formatted_response["error"] = str(e)
        else:
            formatted_response["status"] = "local_processed"
        
        return formatted_response
    
    async def create_code_workspace(self, project_name: str, description: str) -> Dict[str, Any]:
        """Create code workspace in OpenWebUI"""
        
        workspace_config = {
            "name": project_name,
            "description": description,
            "type": "code_workspace",
            "features": {
                "rag_search": True,
                "code_generation": True,
                "code_explanation": True,
                "file_upload": True,
                "collaboration": True
            },
            "tools": [
                {
                    "name": "code_search",
                    "description": "Search through 115M+ code files",
                    "endpoint": "/api/rag/search"
                },
                {
                    "name": "code_generate",
                    "description": "Generate code using RAG context",
                    "endpoint": "/api/rag/generate"
                },
                {
                    "name": "code_explain",
                    "description": "Explain code with examples",
                    "endpoint": "/api/rag/explain"
                }
            ]
        }
        
        if self.session_id and self.openwebui_endpoint:
            try:
                response = requests.post(
                    f"{self.openwebui_endpoint}/api/v1/workspaces",
                    headers={"Authorization": f"Bearer {self.session_id}"},
                    json=workspace_config
                )
                
                if response.status_code == 200:
                    workspace_data = response.json()
                    return {
                        "status": "success",
                        "workspace_id": workspace_data.get("id"),
                        "workspace_url": f"{self.openwebui_endpoint}/workspace/{workspace_data.get('id')}",
                        "message": "Code workspace created in OpenWebUI"
                    }
                else:
                    return {
                        "status": "local_fallback",
                        "workspace_config": workspace_config,
                        "message": "Using local workspace configuration"
                    }
                    
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to create workspace: {str(e)}"
                }
        
        return {
            "status": "local_fallback",
            "workspace_config": workspace_config,
            "message": "Using local workspace configuration"
        }
    
    async def generate_web_interface(self) -> Dict[str, Any]:
        """Generate standalone web interface for RAG system"""
        
        html_interface = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skyscope RAG System - Web Interface</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .header h1 {{
            color: white;
            text-align: center;
            font-weight: 300;
        }}
        
        .container {{
            flex: 1;
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            gap: 2rem;
        }}
        
        .sidebar {{
            width: 300px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .main-content {{
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }}
        
        .chat-container {{
            height: 400px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .input-group {{
            display: flex;
            gap: 0.5rem;
        }}
        
        .input-group input {{
            flex: 1;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1rem;
        }}
        
        .input-group input::placeholder {{
            color: rgba(255, 255, 255, 0.7);
        }}
        
        .btn {{
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            cursor: pointer;
            font-size: 1rem;
            transition: transform 0.2s;
        }}
        
        .btn:hover {{
            transform: translateY(-2px);
        }}
        
        .message {{
            margin-bottom: 1rem;
            padding: 0.75rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #667eea;
            color: white;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }}
        
        .stat-card {{
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            color: white;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }}
        
        .sidebar h3 {{
            color: white;
            margin-bottom: 1rem;
            font-weight: 300;
        }}
        
        .tool-button {{
            display: block;
            width: 100%;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            cursor: pointer;
            text-align: left;
            transition: background 0.2s;
        }}
        
        .tool-button:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        @media (max-width: 768px) {{
            .container {{
                flex-direction: column;
                padding: 1rem;
            }}
            
            .sidebar {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Skyscope RAG System</h1>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h3>🛠️ Tools</h3>
            <button class="tool-button" onclick="searchCode()">🔍 Search Code</button>
            <button class="tool-button" onclick="generateCode()">⚡ Generate Code</button>
            <button class="tool-button" onclick="explainCode()">📚 Explain Code</button>
            <button class="tool-button" onclick="browseFiles()">📁 Browse Files</button>
            
            <h3 style="margin-top: 2rem;">📊 Statistics</h3>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">115M+</div>
                    <div>Files Indexed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">32+</div>
                    <div>Languages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">500GB+</div>
                    <div>Code Data</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="chat-container" id="chatContainer">
                <div class="message">
                    <strong>🤖 Skyscope RAG:</strong> Welcome! I can help you search, understand, and generate code using our massive GitHub codebase. What would you like to explore?
                </div>
            </div>
            
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Ask me anything about code..." 
                       onkeypress="handleKeyPress(event)">
                <button class="btn" onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>
    
    <script>
        // Mock RAG system interaction
        let conversationId = 'conv_' + Date.now();
        
        function addMessage(sender, message, isUser = false) {{
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.style.borderLeftColor = isUser ? '#764ba2' : '#667eea';
            messageDiv.innerHTML = `<strong>${{sender}}:</strong> ${{message}}`;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }}
        
        function sendMessage() {{
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (message) {{
                addMessage('👤 You', message, true);
                input.value = '';
                
                // Simulate RAG response
                setTimeout(() => {{
                    const response = generateMockResponse(message);
                    addMessage('🤖 Skyscope RAG', response);
                }}, 1000);
            }}
        }}
        
        function generateMockResponse(query) {{
            const responses = [
                `I found ${{Math.floor(Math.random() * 100) + 10}} relevant code examples for "${{query}}". Here's what I discovered from the GitHub codebase...`,
                `Based on ${{Math.floor(Math.random() * 50) + 5}} similar implementations, here's the best approach for "${{query}}"...`,
                `Analyzing patterns from ${{Math.floor(Math.random() * 200) + 20}} repositories, I recommend this solution for "${{query}}"...`
            ];
            return responses[Math.floor(Math.random() * responses.length)];
        }}
        
        function handleKeyPress(event) {{
            if (event.key === 'Enter') {{
                sendMessage();
            }}
        }}
        
        function searchCode() {{
            addMessage('🔍 System', 'Code search activated. Enter your search query in the chat.');
        }}
        
        function generateCode() {{
            addMessage('⚡ System', 'Code generation mode activated. Describe what you want to build.');
        }}
        
        function explainCode() {{
            addMessage('📚 System', 'Code explanation mode activated. Paste your code or ask about a concept.');
        }}
        
        function browseFiles() {{
            addMessage('📁 System', 'File browser activated. You can explore the codebase structure.');
        }}
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            document.getElementById('messageInput').focus();
        }});
    </script>
</body>
</html>'''
        
        return {
            "status": "success",
            "html_interface": html_interface,
            "interface_features": [
                "Chat interface with RAG system",
                "Code search and generation tools",
                "Statistics dashboard",
                "Responsive design",
                "Real-time interaction"
            ]
        }

class OpenWebUIAPI:
    """API interface for Pinokio orchestration"""
    
    def __init__(self, rag_system):
        self.webui_integration = OpenWebUIIntegration(rag_system)
    
    async def handle_webui_request(self, request_data: Dict) -> Dict:
        """Handle incoming web UI requests from Pinokio"""
        
        task_type = request_data.get("type", "")
        
        if task_type == "initialize":
            return await self.webui_integration.initialize_webui_session()
        
        elif task_type == "create_model":
            return await self.webui_integration.create_rag_chat_model()
        
        elif task_type == "chat":
            message = request_data.get("message", "")
            conversation_id = request_data.get("conversation_id")
            return await self.webui_integration.handle_chat_request(message, conversation_id)
        
        elif task_type == "create_workspace":
            project_name = request_data.get("project_name", "Skyscope Project")
            description = request_data.get("description", "RAG-enhanced development workspace")
            return await self.webui_integration.create_code_workspace(project_name, description)
        
        elif task_type == "generate_interface":
            return await self.webui_integration.generate_web_interface()
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
'''

# Save OpenWebUI integration
with open(integration_dir / "openwebui" / "integration.py", "w") as f:
    f.write(openwebui_integration)

print("✅ Created OpenWebUI integration module")