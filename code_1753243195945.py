# Advanced Search Engine with Keyword and Semantic Search
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import json
from pathlib import Path

class CodeSearchEngine:
    """Advanced search engine combining keyword and semantic search"""
    
    def __init__(self, db_path: str, embedding_engine):
        self.db_path = db_path
        self.embedding_engine = embedding_engine
        self.logger = logging.getLogger(__name__)
    
    def keyword_search(self, query: str, language: Optional[str] = None, 
                      license: Optional[str] = None, max_results: int = 100) -> List[Dict]:
        """Fast keyword search using FTS5"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build FTS query
            fts_query = self._build_fts_query(query)
            
            # Build filters
            filters = []
            params = [fts_query]
            
            if language:
                filters.append("cf.language = ?")
                params.append(language)
            
            if license:
                filters.append("cf.license = ?")
                params.append(license)
            
            filter_clause = " AND " + " AND ".join(filters) if filters else ""
            
            sql = f"""
                SELECT cf.id, cf.repo_name, cf.file_path, cf.language, 
                       cf.license, cf.size, cf.content,
                       fts.rank
                FROM code_fts fts
                JOIN code_files cf ON cf.id = fts.content_id
                WHERE fts MATCH ?{filter_clause}
                ORDER BY fts.rank
                LIMIT ?
            """
            params.append(max_results)
            
            try:
                cursor = conn.execute(sql, params)
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row['id'],
                        'repo_name': row['repo_name'],
                        'file_path': row['file_path'],
                        'language': row['language'],
                        'license': row['license'],
                        'size': row['size'],
                        'content': row['content'],
                        'score': row['rank'],
                        'search_type': 'keyword'
                    })
                return results
            except Exception as e:
                self.logger.error(f"Keyword search failed: {e}")
                return []
    
    def _build_fts_query(self, query: str) -> str:
        """Build FTS5 query from user input"""
        # Clean and prepare query for FTS5
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        words = query.split()
        
        # Build query with AND logic for multiple terms
        if len(words) > 1:
            return ' AND '.join(f'"{word}"' for word in words if len(word) > 2)
        elif words:
            return f'"{words[0]}"'
        return query
    
    def semantic_search(self, query: str, language: Optional[str] = None,
                       threshold: float = 0.7, max_results: int = 50) -> List[Dict]:
        """Semantic search using embeddings"""
        query_embedding = self.embedding_engine.get_embedding(query)
        if query_embedding is None:
            return []
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all chunks with embeddings
            filters = []
            params = []
            
            if language:
                filters.append("cf.language = ?")
                params.append(language)
            
            filter_clause = " AND " + " AND ".join(filters) if filters else ""
            
            sql = f"""
                SELECT cc.id, cc.file_id, cc.content, cc.start_line, cc.end_line,
                       cc.embedding_vector, cf.repo_name, cf.file_path, 
                       cf.language, cf.license
                FROM code_chunks cc
                JOIN code_files cf ON cf.id = cc.file_id
                WHERE cc.embedding_vector IS NOT NULL{filter_clause}
            """
            
            try:
                cursor = conn.execute(sql, params)
                results = []
                
                for row in cursor.fetchall():
                    if row['embedding_vector']:
                        # Deserialize embedding
                        chunk_embedding = np.frombuffer(row['embedding_vector'], dtype=np.float32)
                        
                        # Calculate similarity
                        similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                        
                        if similarity >= threshold:
                            results.append({
                                'id': row['id'],
                                'file_id': row['file_id'],
                                'repo_name': row['repo_name'],
                                'file_path': row['file_path'],
                                'language': row['language'],
                                'license': row['license'],
                                'content': row['content'],
                                'start_line': row['start_line'],
                                'end_line': row['end_line'],
                                'similarity': similarity,
                                'search_type': 'semantic'
                            })
                
                # Sort by similarity
                results.sort(key=lambda x: x['similarity'], reverse=True)
                return results[:max_results]
                
            except Exception as e:
                self.logger.error(f"Semantic search failed: {e}")
                return []
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except:
            return 0.0
    
    def hybrid_search(self, query: str, language: Optional[str] = None,
                     keyword_weight: float = 0.6, semantic_weight: float = 0.4,
                     max_results: int = 50) -> List[Dict]:
        """Combine keyword and semantic search results"""
        # Get results from both search methods
        keyword_results = self.keyword_search(query, language, max_results=max_results//2)
        semantic_results = self.semantic_search(query, language, max_results=max_results//2)
        
        # Combine and deduplicate
        combined = {}
        
        # Add keyword results
        for result in keyword_results:
            key = f"{result['repo_name']}:{result['file_path']}"
            result['combined_score'] = result.get('score', 1.0) * keyword_weight
            combined[key] = result
        
        # Add semantic results
        for result in semantic_results:
            key = f"{result['repo_name']}:{result['file_path']}"
            if key in combined:
                # Boost score for items found in both
                combined[key]['combined_score'] += result['similarity'] * semantic_weight
                combined[key]['search_type'] = 'hybrid'
            else:
                result['combined_score'] = result['similarity'] * semantic_weight
                combined[key] = result
        
        # Sort by combined score
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return final_results[:max_results]

# Initialize search engine
search_engine = CodeSearchEngine(
    db_path="/home/user/skyscope_rag/indexes/code_index.db",
    embedding_engine=embedding_engine
)

print("✅ Advanced Search Engine created")
print("🔍 Features:")
print("  • FTS5 keyword search with filters")
print("  • Semantic search with embeddings")
print("  • Hybrid search combining both")
print("  • Language and license filtering")
print("  • Similarity thresholding")