#!/usr/bin/env python3
"""
Simple script to analyze Parquet file structure using only pyarrow
"""

import pyarrow.parquet as pq
import pyarrow as pa
import json
import os

def analyze_parquet_structure():
    """Analyze the Parquet file structure"""
    parquet_path = "/Users/skyscope.cloud/Documents/github-code/data/train-00000-of-01126.parquet"
    
    print(f"Analyzing: {parquet_path}")
    
    try:
        # Open the Parquet file
        parquet_file = pq.ParquetFile(parquet_path)
        
        print("\n=== SCHEMA INFORMATION ===")
        schema = parquet_file.schema
        print(f"Number of columns: {len(schema)}")
        
        schema_info = []
        for i, field in enumerate(schema):
            field_info = {
                'index': i,
                'name': field.name,
                'type': str(field.type),
                'nullable': field.nullable
            }
            schema_info.append(field_info)
            print(f"{i}: {field.name} - {field.type} (nullable: {field.nullable})")
        
        print(f"\nFile metadata:")
        metadata = parquet_file.metadata
        print(f"Total rows: {metadata.num_rows}")
        print(f"Number of row groups: {metadata.num_row_groups}")
        print(f"File size: {os.path.getsize(parquet_path) / (1024*1024):.1f} MB")
        
        # Read first batch to see actual data structure
        print("\n=== SAMPLE DATA ===")
        first_batch = parquet_file.read_row_group(0, columns=None)
        
        # Convert to Python objects for easier inspection
        sample_data = first_batch.slice(0, 3).to_pydict()
        
        print(f"Sample contains {len(next(iter(sample_data.values())))} rows")
        print("Column names and sample values:")
        
        for column_name, values in sample_data.items():
            print(f"\n{column_name}:")
            for i, value in enumerate(values):
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "... [TRUNCATED]"
                else:
                    display_value = value
                print(f"  Row {i}: {display_value}")
        
        # Save analysis
        analysis_result = {
            'file_path': parquet_path,
            'schema': schema_info,
            'total_rows': metadata.num_rows,
            'row_groups': metadata.num_row_groups,
            'file_size_mb': os.path.getsize(parquet_path) / (1024*1024),
            'sample_data_types': {col: type(values[0]).__name__ if values else 'unknown' 
                                for col, values in sample_data.items()}
        }
        
        output_file = "/Users/skyscope.cloud/Library/Application Support/Fellou/FellouUserTempFileData/parquet_schema.json"
        with open(output_file, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Results saved to: {output_file}")
        
        return analysis_result
        
    except Exception as e:
        print(f"Error analyzing file: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_parquet_structure()