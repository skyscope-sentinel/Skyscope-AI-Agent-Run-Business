import tarfile
import os
from pathlib import Path

# Create a comprehensive deployment package
def create_deployment_package():
    base_dir = Path("/home/user/skyscope_rag")
    package_path = "/home/user/skyscope_rag_deployment.tar.gz"
    
    with tarfile.open(package_path, "w:gz") as tar:
        tar.add(base_dir, arcname="skyscope_rag")
    
    return package_path

# Create the package
package_path = create_deployment_package()
print(f"✅ Deployment package created: {package_path}")

# Get package size
package_size = os.path.getsize(package_path)
print(f"📦 Package size: {package_size / (1024*1024):.1f} MB")

# List contents of the system
print("\n📁 System Contents:")
for root, dirs, files in os.walk("/home/user/skyscope_rag"):
    level = root.replace("/home/user/skyscope_rag", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")