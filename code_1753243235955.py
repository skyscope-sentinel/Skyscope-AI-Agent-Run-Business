# Install required dependencies
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
packages = [
    "sentence-transformers",
    "faiss-cpu", 
    "chromadb",
    "pyarrow",
    "fastparquet",
    "whoosh",
    "requests",
    "aiohttp",
    "psutil",
    "nltk"
]

print("📦 Installing required packages...")
for package in packages:
    try:
        install_package(package)
        print(f"✅ Installed {package}")
    except Exception as e:
        print(f"⚠️  Failed to install {package}: {e}")

print("\n" + "="*60 + "\n")