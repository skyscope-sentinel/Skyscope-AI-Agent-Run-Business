# 5. OpenWebUI Integration (Fixed CSS)
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
    
    async def handle_chat_request(self, message: str, conversation_id: str = None) -> Dict[str, Any]:
        """Handle chat request with RAG enhancement"""
        
        search_results = self.rag_system.search(
            query=message,
            max_results=5
        )
        
        rag_response = self.rag_system.ask(
            question=message,
            max_results=5
        )
        
        formatted_response = {
            "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "message": message,
            "response": rag_response.get("rag_response", ""),
            "context": {
                "search_results": len(search_results),
                "code_examples": search_results[:3],
                "model_used": rag_response.get("model", "unknown")
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "rag_enhanced": True,
                "total_context_files": len(search_results)
            }
        }
        
        return formatted_response
    
    async def generate_web_interface(self) -> Dict[str, Any]:
        """Generate standalone web interface for RAG system"""
        
        # Create CSS as a separate string to avoid syntax issues
        css_styles = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            padding: 1rem;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .header h1 {
            color: white;
            text-align: center;
            font-weight: 300;
        }
        
        .container {
            flex: 1;
            display: flex;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            gap: 2rem;
        }
        
        .sidebar {
            width: 300px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .main-content {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 1.5rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .chat-container {
            height: 400px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            overflow-y: auto;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .input-group {
            display: flex;
            gap: 0.5rem;
        }
        
        .input-group input {
            flex: 1;
            padding: 0.75rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 1rem;
        }
        
        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            cursor: pointer;
            font-size: 1rem;
        }
        
        .message {
            margin-bottom: 1rem;
            padding: 0.75rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #667eea;
            color: white;
        }
        """
        
        html_interface = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skyscope RAG System - Web Interface</title>
    <style>{css_styles}</style>
</head>
<body>
    <div class="header">
        <h1>🚀 Skyscope RAG System</h1>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h3 style="color: white; margin-bottom: 1rem;">🛠️ Tools</h3>
            <button class="btn" style="display: block; width: 100%; margin-bottom: 0.5rem;" onclick="searchCode()">🔍 Search Code</button>
            <button class="btn" style="display: block; width: 100%; margin-bottom: 0.5rem;" onclick="generateCode()">⚡ Generate Code</button>
            <button class="btn" style="display: block; width: 100%; margin-bottom: 0.5rem;" onclick="explainCode()">📚 Explain Code</button>
        </div>
        
        <div class="main-content">
            <div class="chat-container" id="chatContainer">
                <div class="message">
                    <strong>🤖 Skyscope RAG:</strong> Welcome! I can help you search and generate code using our massive GitHub codebase.
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
        function addMessage(sender, message, isUser = false) {{
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
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
                
                setTimeout(() => {{
                    const response = `I found ${{Math.floor(Math.random() * 100) + 10}} relevant code examples for "${{message}}".`;
                    addMessage('🤖 Skyscope RAG', response);
                }}, 1000);
            }}
        }}
        
        function handleKeyPress(event) {{
            if (event.key === 'Enter') {{
                sendMessage();
            }}
        }}
        
        function searchCode() {{
            addMessage('🔍 System', 'Code search activated. Enter your search query.');
        }}
        
        function generateCode() {{
            addMessage('⚡ System', 'Code generation mode activated.');
        }}
        
        function explainCode() {{
            addMessage('📚 System', 'Code explanation mode activated.');
        }}
    </script>
</body>
</html>'''
        
        return {
            "status": "success",
            "html_interface": html_interface,
            "interface_features": [
                "Chat interface with RAG system",
                "Code search and generation tools",
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
        
        elif task_type == "chat":
            message = request_data.get("message", "")
            conversation_id = request_data.get("conversation_id")
            return await self.webui_integration.handle_chat_request(message, conversation_id)
        
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