# Create basic configuration
import json
from pathlib import Path

base_dir = Path("/home/user/skyscope_rag")
config_path = base_dir / "config" / "config.json"

# Basic config
config = {
    "system": {
        "name": "Skyscope RAG System",
        "version": "1.0.0",
        "base_dir": str(base_dir),
        "parquet_source": "/Users/skyscope.cloud/Documents/github-code"
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "default_model": "codellama"
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✅ Config created: {config_path}")
print("📋 Basic system ready")