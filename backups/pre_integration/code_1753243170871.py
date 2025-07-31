# Create standalone Python scripts for easy deployment
scripts_dir = Path("/home/user/skyscope_rag/scripts")

# Main executable script
main_script = '''#!/usr/bin/env python3
"""
Skyscope RAG System - Main Executable
Comprehensive RAG system for GitHub codebase search and generation
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path

# Add the system to Python path
sys.path.insert(0, '/home/user/skyscope_rag')

def setup_logging(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/home/user/skyscope_rag/logs/system.log')
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Skyscope RAG System")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index Parquet files')
    index_parser.add_argument('directory', help='Directory with Parquet files')
    index_parser.add_argument('--max-files', type=int, help='Max files to process')
    index_parser.add_argument('--batch-size', type=int, default=1000, help='Batch size')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search codebase')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--type', choices=['keyword', 'semantic', 'hybrid'], 
                              default='hybrid')
    search_parser.add_argument('--language', help='Programming language filter')
    search_parser.add_argument('--max-results', type=int, default=10)
    search_parser.add_argument('--output', help='Output file for results')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask question with RAG')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--language', help='Language filter')
    ask_parser.add_argument('--execute', action='store_true', help='Execute code')
    ask_parser.add_argument('--model', help='Ollama model to use')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    args = parser.parse_args()
    setup_logging(args.debug)
    
    if not args.command:
        parser.print_help()
        return
    
    # Import here to avoid issues if dependencies are missing
    try:
        from rag_system import SkyscopeRAGSystem
        system = SkyscopeRAGSystem()
    except ImportError as e:
        print(f"Error: Missing dependencies. Run setup.py first. {e}")
        return 1
    
    if args.command == 'index':
        print(f"🔄 Indexing files from: {args.directory}")
        result = system.index_parquet_files(args.directory, args.max_files)
        print(f"✅ Indexed {result.get('total_processed', 0)} files")
        
    elif args.command == 'search':
        print(f"🔍 Searching for: {args.query}")
        results = system.search(args.query, args.type, args.language, args.max_results)
        
        print(f"📊 Found {len(results)} results:")
        for i, result in enumerate(results[:5]):
            print(f"  {i+1}. {result['repo_name']}/{result['file_path']} ({result['language']})")
        
        if args.output:
            system.export_results(results, args.output)
            print(f"💾 Results saved to {args.output}")
            
    elif args.command == 'ask':
        if args.model:
            system.switch_ollama_model(args.model)
        
        print(f"❓ Question: {args.question}")
        response = system.ask(args.question, args.language, args.execute)
        
        print("🤖 Response:")
        print(response.get('rag_response', 'No response'))
        
        if 'relevant_files' in response:
            print("📁 Relevant files:")
            for file_info in response['relevant_files']:
                print(f"  • {file_info['repo']}/{file_info['path']}")
                
    elif args.command == 'interactive':
        interactive_mode(system)
        
    elif args.command == 'stats':
        stats = system.get_stats()
        print("📊 System Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

def interactive_mode(system):
    """Interactive REPL mode"""
    print("🎮 Skyscope RAG Interactive Mode")
    print("Commands: search, ask, stats, model, quit")
    print("=" * 50)
    
    while True:
        try:
            command = input("skyscope> ").strip()
            if not command:
                continue
                
            if command == 'quit' or command == 'exit':
                break
            elif command == 'stats':
                stats = system.get_stats()
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            elif command.startswith('search '):
                query = command[7:]
                results = system.search(query)
                print(f"Found {len(results)} results:")
                for i, r in enumerate(results[:3]):
                    print(f"  {i+1}. {r['repo_name']}/{r['file_path']}")
            elif command.startswith('ask '):
                question = command[4:]
                response = system.ask(question)
                print("Response:", response.get('rag_response', 'No response'))
            elif command.startswith('model '):
                model_name = command[6:]
                if system.switch_ollama_model(model_name):
                    print(f"Switched to model: {model_name}")
                else:
                    print(f"Model not available: {model_name}")
            else:
                print("Unknown command. Try: search, ask, stats, model, quit")
                
        except KeyboardInterrupt:
            print("\\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    sys.exit(main())
'''

# Setup script
setup_script = '''#!/usr/bin/env python3
"""
Skyscope RAG System Setup Script
Installs dependencies and configures the system
"""

import subprocess
import sys
import os
from pathlib import Path
import json

def install_packages():
    """Install required Python packages"""
    packages = [
        "sentence-transformers",
        "faiss-cpu",
        "pyarrow",
        "fastparquet", 
        "pandas",
        "numpy",
        "requests",
        "aiohttp",
        "psutil",
        "nltk",
        "tqdm"
    ]
    
    print("📦 Installing Python packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def setup_directories():
    """Create necessary directories"""
    base_dir = Path("/home/user/skyscope_rag")
    dirs = ["indexes", "embeddings", "cache", "logs", "config", "agents", "models", "scripts"]
    
    print("📁 Creating directories...")
    for dir_name in dirs:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✅ {base_dir / dir_name}")

def create_config():
    """Create default configuration"""
    config = {
        "system": {
            "name": "Skyscope RAG System",
            "version": "1.0.0",
            "base_dir": "/home/user/skyscope_rag",
            "parquet_source": "/Users/skyscope.cloud/Documents/github-code"
        },
        "indexing": {
            "batch_size": 1000,
            "chunk_size": 8192,
            "overlap": 512,
            "min_file_size": 50,
            "max_file_size": 1048576
        },
        "embeddings": {
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 32
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "default_model": "codellama",
            "timeout": 300
        },
        "search": {
            "max_results": 100,
            "similarity_threshold": 0.7
        }
    }
    
    config_path = Path("/home/user/skyscope_rag/config/config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"⚙️  Configuration created: {config_path}")

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"🤖 Ollama running with {len(models)} models")
            for model in models:
                print(f"  • {model['name']}")
        else:
            print("⚠️  Ollama not responding properly")
    except Exception:
        print("❌ Ollama not running. Please start Ollama first.")

def main():
    print("🚀 Skyscope RAG System Setup")
    print("=" * 40)
    
    setup_directories()
    install_packages()
    create_config()
    check_ollama()
    
    print("\\n✅ Setup complete!")
    print("🎯 Next steps:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Pull models: ollama pull codellama")
    print("  3. Index files: python main.py index /path/to/parquet/files")
    print("  4. Start searching: python main.py search 'your query'")

if __name__ == "__main__":
    main()
'''

# Save scripts
with open(scripts_dir / "main.py", "w") as f:
    f.write(main_script)

with open(scripts_dir / "setup.py", "w") as f:
    f.write(setup_script)

# Make scripts executable
os.chmod(scripts_dir / "main.py", 0o755)
os.chmod(scripts_dir / "setup.py", 0o755)

print("✅ Created executable scripts:")
print(f"  📜 {scripts_dir / 'main.py'} - Main system interface")
print(f"  🔧 {scripts_dir / 'setup.py'} - Setup and installation")