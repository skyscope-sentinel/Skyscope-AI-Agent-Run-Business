# 4. Bolt.diy Integration (Fixed CSS syntax)
bolt_diy_integration = '''"""
Skyscope RAG - Bolt.diy Integration
Enables rapid prototyping using code found through RAG system
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

class BoltDiyIntegration:
    """Integration with Bolt.diy Pinokio app for rapid prototyping"""
    
    def __init__(self, rag_system, bolt_endpoint: str = "http://localhost:8083"):
        self.rag_system = rag_system
        self.bolt_endpoint = bolt_endpoint
        self.session_id = None
        
    async def initialize_bolt_session(self) -> Dict[str, Any]:
        """Initialize Bolt.diy rapid prototyping session"""
        try:
            response = requests.post(f"{self.bolt_endpoint}/api/session/create", 
                                   json={"environment": "fullstack", "framework": "auto"})
            
            if response.status_code == 200:
                session_data = response.json()
                self.session_id = session_data.get("session_id")
                
                return {
                    "status": "success",
                    "session_id": self.session_id,
                    "message": "Bolt.diy prototyping session initialized"
                }
            else:
                return await self._initialize_local_prototyping()
                
        except Exception as e:
            return await self._initialize_local_prototyping()
    
    async def _initialize_local_prototyping(self) -> Dict[str, Any]:
        """Fallback local prototyping environment"""
        self.session_id = f"local_bolt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "status": "success",
            "session_id": self.session_id,
            "message": "Local prototyping environment initialized",
            "mode": "local"
        }
    
    async def find_prototype_components(self, app_description: str, framework: str = "") -> Dict[str, Any]:
        """Find relevant prototype components from RAG system"""
        
        search_queries = [
            f"{app_description} prototype",
            f"{app_description} starter template",
            f"{framework} {app_description}",
            f"{app_description} boilerplate"
        ]
        
        results = []
        for query in search_queries:
            search_results = self.rag_system.search(
                query=query,
                max_results=8
            )
            results.extend(search_results)
        
        return {
            "app_description": app_description,
            "framework": framework,
            "components_found": len(results),
            "examples": results[:10]
        }
    
    async def generate_rapid_prototype(self, app_description: str, requirements: List[str], framework: str = "auto") -> Dict[str, Any]:
        """Generate rapid prototype using RAG-found components"""
        
        components = await self.find_prototype_components(app_description, framework)
        
        rag_response = self.rag_system.ask(
            f"Create a rapid prototype for: {app_description}. Requirements: {', '.join(requirements)}. Framework: {framework}",
            max_results=10
        )
        
        prototype_plan = {
            "app_description": app_description,
            "requirements": requirements,
            "framework": framework,
            "components": components,
            "architecture_guidance": rag_response.get("rag_response", ""),
            "generated_files": self._generate_basic_files(app_description, requirements)
        }
        
        return prototype_plan
    
    def _generate_basic_files(self, app_description: str, requirements: List[str]) -> List[Dict]:
        """Generate basic prototype files"""
        
        files = []
        
        # Package.json
        package_json = {
            "name": app_description.lower().replace(" ", "-"),
            "version": "1.0.0",
            "description": f"Rapid prototype for {app_description}",
            "main": "server.js",
            "scripts": {
                "start": "node server.js",
                "dev": "nodemon server.js"
            },
            "dependencies": {
                "express": "^4.18.0",
                "cors": "^2.8.5"
            }
        }
        
        files.append({
            "filename": "package.json",
            "content": json.dumps(package_json, indent=2),
            "type": "config"
        })
        
        # HTML template
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{app_description}</title>
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
        }}
        .header {{ 
            text-align: center; 
            margin-bottom: 30px; 
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{app_description}</h1>
            <p>Rapid prototype generated using Skyscope RAG System</p>
        </div>
        <div id="app">
            <p>Start building your application!</p>
        </div>
    </div>
</body>
</html>"""
        
        files.append({
            "filename": "index.html",
            "content": html_content,
            "type": "frontend"
        })
        
        # Basic server
        server_content = f"""const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static('.'));

app.get('/', (req, res) => {{
    res.sendFile(path.join(__dirname, 'index.html'));
}});

app.get('/api/health', (req, res) => {{
    res.json({{ 
        status: 'OK', 
        message: '{app_description} server is running'
    }});
}});

app.listen(PORT, () => {{
    console.log(`{app_description} server running on port ${{PORT}}`);
}});
"""
        
        files.append({
            "filename": "server.js",
            "content": server_content,
            "type": "backend"
        })
        
        return files

class BoltDiyAPI:
    """API interface for Pinokio orchestration"""
    
    def __init__(self, rag_system):
        self.bolt_integration = BoltDiyIntegration(rag_system)
    
    async def handle_prototyping_request(self, request_data: Dict) -> Dict:
        """Handle incoming prototyping requests from Pinokio"""
        
        task_type = request_data.get("type", "")
        
        if task_type == "initialize":
            return await self.bolt_integration.initialize_bolt_session()
        
        elif task_type == "generate_prototype":
            app_description = request_data.get("app_description", "")
            requirements = request_data.get("requirements", [])
            framework = request_data.get("framework", "auto")
            return await self.bolt_integration.generate_rapid_prototype(app_description, requirements, framework)
        
        elif task_type == "find_components":
            app_description = request_data.get("app_description", "")
            framework = request_data.get("framework", "")
            return await self.bolt_integration.find_prototype_components(app_description, framework)
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
'''

# Save Bolt.diy integration
with open(integration_dir / "bolt_diy" / "integration.py", "w") as f:
    f.write(bolt_diy_integration)

print("✅ Created Bolt.diy integration module")