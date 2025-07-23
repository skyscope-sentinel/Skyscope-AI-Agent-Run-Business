# 3. Devika Integration
devika_integration = '''"""
Skyscope RAG - Devika Integration
Enables AI development workflows using code found through RAG system
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime

class DevikaIntegration:
    """Integration with Devika Pinokio app for AI development workflows"""
    
    def __init__(self, rag_system, devika_endpoint: str = "http://localhost:8082"):
        self.rag_system = rag_system
        self.devika_endpoint = devika_endpoint
        self.session_id = None
        self.project_id = None
        
    async def initialize_devika_session(self, project_name: str = "skyscope_project") -> Dict[str, Any]:
        """Initialize Devika AI development session"""
        try:
            # Connect to Devika Pinokio app
            response = requests.post(f"{self.devika_endpoint}/api/create-project", 
                                   json={"name": project_name, "description": "Skyscope RAG enhanced development"})
            
            if response.status_code == 200:
                project_data = response.json()
                self.project_id = project_data.get("project_id")
                
                return {
                    "status": "success",
                    "project_id": self.project_id,
                    "project_name": project_name,
                    "message": "Devika development session initialized"
                }
            else:
                # Fallback to local development session
                return await self._initialize_local_development()
                
        except Exception as e:
            return await self._initialize_local_development()
    
    async def _initialize_local_development(self) -> Dict[str, Any]:
        """Fallback local development session"""
        self.project_id = f"local_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "status": "success",
            "project_id": self.project_id,
            "message": "Local development session initialized",
            "mode": "local"
        }
    
    async def find_development_patterns(self, requirement: str, technology: str = "") -> Dict[str, Any]:
        """Find relevant development patterns from RAG system"""
        
        # Search for development patterns
        search_queries = [
            f"{requirement} implementation {technology}",
            f"{requirement} design pattern",
            f"{technology} {requirement} example",
            f"{requirement} best practices",
            f"{requirement} architecture pattern"
        ]
        
        results = []
        for query in search_queries:
            search_results = self.rag_system.search(
                query=query,
                max_results=5
            )
            results.extend(search_results)
        
        # Categorize results by language/technology
        categorized_results = {}
        for result in results:
            language = result.get('language', 'unknown')
            if language not in categorized_results:
                categorized_results[language] = []
            categorized_results[language].append(result)
        
        return {
            "requirement": requirement,
            "technology": technology,
            "patterns_found": categorized_results,
            "total_examples": len(results)
        }
    
    async def generate_development_plan(self, requirements: List[str], technology_stack: List[str]) -> Dict[str, Any]:
        """Generate comprehensive development plan using RAG insights"""
        
        development_components = []
        
        for requirement in requirements:
            for tech in technology_stack:
                # Find relevant patterns
                patterns = await self.find_development_patterns(requirement, tech)
                
                # Generate implementation guidance using RAG
                rag_response = self.rag_system.ask(
                    f"How to implement {requirement} using {tech}? Provide architecture and code examples.",
                    max_results=5
                )
                
                development_components.append({
                    "requirement": requirement,
                    "technology": tech,
                    "patterns": patterns["patterns_found"],
                    "implementation_guidance": rag_response.get("rag_response", ""),
                    "examples_found": patterns["total_examples"]
                })
        
        # Create overall development plan
        development_plan = {
            "project_id": self.project_id,
            "requirements": requirements,
            "technology_stack": technology_stack,
            "components": development_components,
            "total_code_references": sum(comp["examples_found"] for comp in development_components),
            "generated_at": datetime.now().isoformat()
        }
        
        # If Devika endpoint is available, send the plan
        if self.project_id and self.devika_endpoint:
            try:
                response = requests.post(
                    f"{self.devika_endpoint}/api/add-context",
                    json={
                        "project_id": self.project_id,
                        "context": development_plan,
                        "type": "rag_enhanced_plan"
                    }
                )
                
                if response.status_code == 200:
                    development_plan["devika_status"] = "plan_uploaded"
                else:
                    development_plan["devika_status"] = "upload_failed"
                    
            except Exception as e:
                development_plan["devika_status"] = f"error: {str(e)}"
        
        return development_plan
    
    async def enhance_devika_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Enhance Devika prompts with RAG context"""
        
        # Extract key terms from the prompt for RAG search
        key_terms = self._extract_key_terms(user_prompt)
        
        # Search for relevant code examples
        context_results = []
        for term in key_terms:
            search_results = self.rag_system.search(
                query=term,
                max_results=3
            )
            context_results.extend(search_results)
        
        # Generate enhanced prompt with context
        enhanced_prompt = self._create_enhanced_prompt(user_prompt, context_results)
        
        # Send enhanced prompt to Devika if available
        devika_response = None
        if self.project_id and self.devika_endpoint:
            try:
                response = requests.post(
                    f"{self.devika_endpoint}/api/execute-agent",
                    json={
                        "project_id": self.project_id,
                        "prompt": enhanced_prompt,
                        "context_enhanced": True
                    }
                )
                
                if response.status_code == 200:
                    devika_response = response.json()
                    
            except Exception as e:
                devika_response = {"error": str(e)}
        
        return {
            "original_prompt": user_prompt,
            "enhanced_prompt": enhanced_prompt,
            "context_examples": len(context_results),
            "key_terms": key_terms,
            "devika_response": devika_response
        }
    
    def _extract_key_terms(self, prompt: str) -> List[str]:
        """Extract key technical terms from prompt"""
        # Simple keyword extraction (can be enhanced with NLP)
        tech_keywords = [
            "API", "REST", "GraphQL", "database", "authentication", "authorization",
            "React", "Vue", "Angular", "Node.js", "Python", "Django", "Flask",
            "microservices", "docker", "kubernetes", "AWS", "Azure", "GCP",
            "machine learning", "AI", "neural network", "blockchain", "web3",
            "testing", "CI/CD", "deployment", "monitoring", "logging"
        ]
        
        found_terms = []
        prompt_lower = prompt.lower()
        
        for keyword in tech_keywords:
            if keyword.lower() in prompt_lower:
                found_terms.append(keyword)
        
        # Add generic terms from prompt
        words = prompt.split()
        for word in words:
            if len(word) > 4 and word.isalpha():
                found_terms.append(word)
        
        return list(set(found_terms))[:10]  # Limit to 10 terms
    
    def _create_enhanced_prompt(self, original_prompt: str, context_results: List[Dict]) -> str:
        """Create enhanced prompt with RAG context"""
        
        if not context_results:
            return original_prompt
        
        context_section = "## Relevant Code Examples Found:\n\n"
        
        for i, result in enumerate(context_results[:5]):  # Limit to 5 examples
            context_section += f"### Example {i+1}: {result.get('file_path', 'Unknown')}\n"
            context_section += f"Language: {result.get('language', 'Unknown')}\n"
            
            content = result.get('content', '')
            if len(content) > 500:
                content = content[:500] + "..."
            
            context_section += f"```{result.get('language', '')}\n{content}\n```\n\n"
        
        enhanced_prompt = f"""
{original_prompt}

{context_section}

## Instructions:
Please consider the above code examples when implementing the solution. Use similar patterns, best practices, and architectural approaches found in the examples. Ensure the solution follows established conventions from the codebase.
"""
        
        return enhanced_prompt

class DevikaAPI:
    """API interface for Pinokio orchestration"""
    
    def __init__(self, rag_system):
        self.devika_integration = DevikaIntegration(rag_system)
    
    async def handle_development_request(self, request_data: Dict) -> Dict:
        """Handle incoming development requests from Pinokio"""
        
        task_type = request_data.get("type", "")
        
        if task_type == "initialize":
            project_name = request_data.get("project_name", "skyscope_project")
            return await self.devika_integration.initialize_devika_session(project_name)
        
        elif task_type == "generate_plan":
            requirements = request_data.get("requirements", [])
            tech_stack = request_data.get("technology_stack", [])
            return await self.devika_integration.generate_development_plan(requirements, tech_stack)
        
        elif task_type == "enhance_prompt":
            prompt = request_data.get("prompt", "")
            return await self.devika_integration.enhance_devika_prompt(prompt)
        
        elif task_type == "find_patterns":
            requirement = request_data.get("requirement", "")
            technology = request_data.get("technology", "")
            return await self.devika_integration.find_development_patterns(requirement, technology)
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
'''

# Save Devika integration
with open(integration_dir / "devika" / "integration.py", "w") as f:
    f.write(devika_integration)

print("✅ Created Devika integration module")