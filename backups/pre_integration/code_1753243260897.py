#!/usr/bin/env python3
"""
Script to analyze the Parquet files in the github-code dataset
and extract schema information for indexing strategy.
"""

import pandas as pd
import pyarrow.parquet as pq
import json
import os
from pathlib import Path

def analyze_parquet_sample(parquet_path):
    """Analyze a sample Parquet file to understand schema and content."""
    print(f"Analyzing: {parquet_path}")
    
    try:
        # Read schema without loading all data
        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema
        
        print("\n=== SCHEMA INFORMATION ===")
        print(f"Number of columns: {len(schema)}")
        for i, field in enumerate(schema):
            print(f"{i}: {field.name} - {field.type}")
        
        print(f"\nTotal rows in file: {parquet_file.metadata.num_rows}")
        print(f"Number of row groups: {parquet_file.metadata.num_row_groups}")
        
        # Read first few rows to understand data structure
        print("\n=== SAMPLE DATA (first 3 rows) ===")
        df_sample = pd.read_parquet(parquet_path, engine='pyarrow').head(3)
        
        for idx, row in df_sample.iterrows():
            print(f"\n--- Row {idx+1} ---")
            for col in df_sample.columns:
                value = str(row[col])
                if len(value) > 200:
                    value = value[:200] + "... [TRUNCATED]"
                print(f"{col}: {value}")
        
        print("\n=== COLUMN STATISTICS ===")
        for col in df_sample.columns:
            if col in df_sample.columns:
                print(f"\n{col}:")
                if df_sample[col].dtype == 'object':
                    # String column
                    avg_length = df_sample[col].astype(str).str.len().mean()
                    max_length = df_sample[col].astype(str).str.len().max()
                    print(f"  Average length: {avg_length:.1f}")
                    print(f"  Max length: {max_length}")
                    
                    # Sample unique values for categorical columns
                    if col in ['language', 'license']:
                        unique_vals = df_sample[col].unique()
                        print(f"  Unique values in sample: {list(unique_vals)}")
                elif df_sample[col].dtype in ['int64', 'int32', 'float64']:
                    # Numeric column
                    print(f"  Min: {df_sample[col].min()}")
                    print(f"  Max: {df_sample[col].max()}")
                    print(f"  Mean: {df_sample[col].mean():.2f}")
        
        return {
            'schema': [{'name': field.name, 'type': str(field.type)} for field in schema],
            'num_rows': parquet_file.metadata.num_rows,
            'num_row_groups': parquet_file.metadata.num_row_groups,
            'file_size_mb': os.path.getsize(parquet_path) / (1024 * 1024)
        }
        
    except Exception as e:
        print(f"Error analyzing {parquet_path}: {e}")
        return None

def main():
    # Path to the first Parquet file
    parquet_path = "/Users/skyscope.cloud/Documents/github-code/data/train-00000-of-01126.parquet"
    
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        return
    
    result = analyze_parquet_sample(parquet_path)
    
    if result:
        # Save analysis results
        output_file = "/Users/skyscope.cloud/Library/Application Support/Fellou/FellouUserTempFileData/parquet_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nAnalysis saved to: {output_file}")

if __name__ == "__main__":
    main()