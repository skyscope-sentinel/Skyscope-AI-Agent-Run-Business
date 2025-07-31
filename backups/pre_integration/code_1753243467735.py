import os
import pathlib

# Create directory structure for documentation
dirs = [
    'docs',
    'docs/agents',
    'docs/installation',
    'docs/configuration',
    'docs/api',
    'docs/troubleshooting',
    'docs/tutorials',
    'docs/examples',
    'config',
    'config/agents',
    'src',
    'src/agents',
    'src/orchestration',
    'src/utils',
    'tests',
    'tests/agents',
    'tests/integration',
    'scripts',
    'examples',
    'templates'
]

base_path = '/tmp/skyscope-repo'
for dir_path in dirs:
    full_path = os.path.join(base_path, dir_path)
    pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
    print(f"Created: {full_path}")

print("Directory structure created successfully!")