# Parquet Processing System
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sqlite3
import numpy as np
from typing import Iterator, Dict, List, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

class ParquetProcessor:
    """Process Parquet files and build searchable indexes"""
    
    def __init__(self, indexer, embedding_engine, batch_size: int = 1000):
        self.indexer = indexer
        self.embedding_engine = embedding_engine
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.processed_files = set()
        self.load_processed_files()
    
    def load_processed_files(self):
        """Load list of already processed files"""
        try:
            with sqlite3.connect(self.indexer.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT file_hash FROM code_files")
                self.processed_files = {row[0] for row in cursor.fetchall()}
            self.logger.info(f"Found {len(self.processed_files)} already processed files")
        except Exception:
            self.processed_files = set()
    
    def process_parquet_file(self, parquet_path: str) -> Dict[str, Any]:
        """Process a single Parquet file"""
        start_time = time.time()
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        try:
            # Read parquet file in batches
            parquet_file = pq.ParquetFile(parquet_path)
            
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                
                for _, row in df.iterrows():
                    try:
                        # Extract data
                        content = row.get('content', '')
                        repo_name = row.get('repo_name', '')
                        file_path = row.get('path', '')
                        license_type = row.get('license', '')
                        file_size = row.get('size', 0)
                        
                        # Skip if invalid
                        if not content or not repo_name or not file_path:
                            skipped_count += 1
                            continue
                        
                        # Generate hash for deduplication
                        file_hash = self.indexer.hash_content(content)
                        
                        # Skip if already processed
                        if file_hash in self.processed_files:
                            skipped_count += 1
                            continue
                        
                        # Determine language
                        language = self.indexer.get_language_from_path(file_path)
                        
                        # Skip if file is too large or too small
                        if file_size < 50 or file_size > 1048576:
                            skipped_count += 1
                            continue
                        
                        # Process the file
                        self.process_code_file(
                            content=content,
                            repo_name=repo_name,
                            file_path=file_path,
                            language=language,
                            license_type=license_type,
                            file_size=file_size,
                            file_hash=file_hash
                        )
                        
                        processed_count += 1
                        self.processed_files.add(file_hash)
                        
                        # Progress update
                        if processed_count % 100 == 0:
                            elapsed = time.time() - start_time
                            self.logger.info(f"Processed {processed_count} files in {elapsed:.1f}s")
                        
                    except Exception as e:
                        error_count += 1
                        self.logger.error(f"Error processing row: {e}")
        
        except Exception as e:
            self.logger.error(f"Error reading parquet file {parquet_path}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': parquet_path
            }
        
        elapsed = time.time() - start_time
        return {
            'success': True,
            'file': parquet_path,
            'processed': processed_count,
            'skipped': skipped_count,
            'errors': error_count,
            'elapsed': elapsed
        }
    
    def process_code_file(self, content: str, repo_name: str, file_path: str,
                         language: str, license_type: str, file_size: int, file_hash: str):
        """Process individual code file"""
        
        with sqlite3.connect(self.indexer.db_path) as conn:
            # Insert main file record
            cursor = conn.execute("""
                INSERT INTO code_files 
                (file_hash, repo_name, file_path, language, license, size, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (file_hash, repo_name, file_path, language, license_type, file_size, content))
            
            file_id = cursor.lastrowid
            
            # Insert into FTS table
            conn.execute("""
                INSERT INTO code_fts (content, repo_name, file_path, content_id)
                VALUES (?, ?, ?, ?)
            """, (content, repo_name, file_path, file_id))
            
            # Create chunks for semantic search
            chunks = self.indexer.chunk_content(content, chunk_size=1000, overlap=100)
            
            for i, (chunk_content, start_line, end_line) in enumerate(chunks):
                # Get embedding for chunk
                embedding = self.embedding_engine.get_embedding(chunk_content)
                embedding_blob = None
                
                if embedding is not None:
                    embedding_blob = embedding.astype(np.float32).tobytes()
                
                # Insert chunk
                conn.execute("""
                    INSERT INTO code_chunks 
                    (file_id, chunk_index, content, start_line, end_line, embedding_vector)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (file_id, i, chunk_content, start_line, end_line, embedding_blob))
    
    def process_directory(self, directory_path: str, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Process all Parquet files in a directory"""
        directory = Path(directory_path)
        parquet_files = list(directory.glob("*.parquet"))
        
        if max_files:
            parquet_files = parquet_files[:max_files]
        
        self.logger.info(f"Found {len(parquet_files)} parquet files to process")
        
        results = []
        total_processed = 0
        total_errors = 0
        
        for i, parquet_file in enumerate(parquet_files):
            self.logger.info(f"Processing file {i+1}/{len(parquet_files)}: {parquet_file.name}")
            
            result = self.process_parquet_file(str(parquet_file))
            results.append(result)
            
            if result['success']:
                total_processed += result['processed']
            else:
                total_errors += 1
        
        return {
            'total_files': len(parquet_files),
            'total_processed': total_processed,
            'total_errors': total_errors,
            'results': results
        }

# Initialize Parquet processor
parquet_processor = ParquetProcessor(
    indexer=indexer,
    embedding_engine=embedding_engine,
    batch_size=1000
)

print("✅ Parquet Processing System created")
print("📊 Features:")
print("  • Batch processing with progress tracking")
print("  • Deduplication using content hashes")
print("  • Automatic language detection")
print("  • Chunking for semantic search")
print("  • Embedding generation and storage")
print("  • FTS5 indexing for keyword search")
print("💡 Ready to process Parquet files from GitHub codebase")