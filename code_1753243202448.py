# Semantic Search Engine with Local Embeddings
import requests
import numpy as np
import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
import logging

class LocalEmbeddingEngine:
    """Local embedding engine using sentence-transformers or Ollama embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", ollama_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.logger = logging.getLogger(__name__)
        self.embedding_cache = {}
        
        # Try to use sentence-transformers first, fallback to Ollama
        self.use_local_model = self._init_local_model()
        if not self.use_local_model:
            self.logger.info("Using Ollama for embeddings")
    
    def _init_local_model(self) -> bool:
        """Initialize local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.logger.info(f"Loaded local model: {self.model_name}")
            return True
        except ImportError:
            self.logger.warning("sentence-transformers not available, will use Ollama")
            return False
        except Exception as e:
            self.logger.error(f"Failed to load local model: {e}")
            return False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text using local model or Ollama"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            if self.use_local_model:
                embedding = self.model.encode(text)
            else:
                embedding = self._get_ollama_embedding(text)
            
            if embedding is not None:
                self.embedding_cache[text] = embedding
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to get embedding: {e}")
            return None
    
    def _get_ollama_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": "nomic-embed-text",  # Good for code
                    "prompt": text
                },
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                return np.array(data.get('embedding', []))
            else:
                self.logger.error(f"Ollama embedding API error: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Ollama embedding request failed: {e}")
            return None
    
    def batch_embeddings(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Get embeddings for multiple texts"""
        if self.use_local_model:
            try:
                embeddings = self.model.encode(texts)
                return [emb for emb in embeddings]
            except Exception as e:
                self.logger.error(f"Batch embedding failed: {e}")
                return [None] * len(texts)
        else:
            # Process one by one for Ollama
            return [self.get_embedding(text) for text in texts]

class OllamaRAGEngine:
    """RAG engine that integrates with Ollama for code generation and reasoning"""
    
    def __init__(self, base_url: str = "http://localhost:11434", default_model: str = "codellama"):
        self.base_url = base_url
        self.default_model = default_model
        self.logger = logging.getLogger(__name__)
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            self.logger.error(f"Failed to get Ollama models: {e}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model"""
        if model_name in self.available_models:
            self.default_model = model_name
            self.logger.info(f"Switched to model: {model_name}")
            return True
        else:
            self.logger.error(f"Model {model_name} not available")
            return False
    
    def generate_with_context(self, query: str, context_chunks: List[str], 
                            model: Optional[str] = None) -> str:
        """Generate response using RAG context"""
        model = model or self.default_model
        
        # Build context-aware prompt
        context = "\n\n".join([f"Code Context {i+1}:\n{chunk}" 
                              for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""You are an expert code assistant with access to a large codebase. Use the provided code context to answer the query accurately and comprehensively.

Code Context:
{context}

Query: {query}

Instructions:
- Analyze the provided code context carefully
- Provide specific, actionable answers
- Include relevant code examples from the context
- Explain complex concepts clearly
- If the context doesn't contain relevant information, say so

Response:"""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_ctx": 8192
                    }
                },
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json().get('response', 'No response generated')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Generation failed: {str(e)}"

# Initialize components
embedding_engine = LocalEmbeddingEngine()
rag_engine = OllamaRAGEngine()

print("✅ Semantic search components created")
print(f"🧠 Embedding engine: {'Local model' if embedding_engine.use_local_model else 'Ollama'}")
print(f"🤖 Available Ollama models: {len(rag_engine.available_models)}")
if rag_engine.available_models:
    print(f"📋 Models: {', '.join(rag_engine.available_models)}")
print(f"🎯 Default model: {rag_engine.default_model}")