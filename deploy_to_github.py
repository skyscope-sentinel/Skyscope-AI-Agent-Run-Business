#!/usr/bin/env python3
"""
Skyscope Sentinel Intelligence AI - GitHub Deployment Script
===========================================================

This script automates the deployment of the Skyscope Sentinel Intelligence AI system
to a GitHub repository. It organizes files according to the target repository structure,
removes redundancies, and sets up the proper branch structure for integration.

Usage:
    python deploy_to_github.py [--repo REPO_PATH] [--branch BRANCH_NAME] [--dry-run]

Options:
    --repo REPO_PATH       Path to the local GitHub repository (default: current directory)
    --branch BRANCH_NAME   Name of the branch to create for integration (default: integration/v1.0)
    --dry-run             Run without making any changes (simulation mode)
    --force               Overwrite existing files without prompting
    --no-cleanup          Skip redundancy cleanup step
    --help                Show this help message and exit

Example:
    python deploy_to_github.py --repo ~/github/skyscope-sentinel --branch feature/enhanced-ui
"""

import os
import sys
import shutil
import argparse
import subprocess
import json
import re
import logging
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deploy_to_github.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deploy_to_github")

# Constants
VERSION = "1.0.0"
COMPANY_NAME = "Skyscope Sentinel Intelligence"
DEFAULT_BRANCH = "integration/v1.0"

# Target repository structure as defined in INTEGRATION_PLAN.md
TARGET_STRUCTURE = {
    ".github/workflows": "CI definitions",
    "docs": "Documentation",
    "scripts": "Helper & maintenance scripts",
    "skyscope": "Python package root",
    "skyscope/core": "Core modules (agent_manager, crypto_manager, etc.)",
    "skyscope/ui": "UI components (Streamlit apps, themes, assets)",
    "skyscope/ops": "Operations modules (business_manager, etc.)",
    "skyscope/data": "Embedded small reference data",
    "skyscope/tests": "Unit tests",
    "assets": "Static assets (fonts, css, images, animations)",
    "config": "Configuration files",
    "data": "Runtime-generated data (git-ignored)",
}

# File mapping: source -> target
FILE_MAPPING = {
    "app.py": "skyscope/ui/app.py",
    "enhanced_chat_interface.py": "skyscope/ui/enhanced_chat_interface.py",
    "agent_manager.py": "skyscope/core/agent_manager.py",
    "business_manager.py": "skyscope/ops/business_manager.py",
    "crypto_manager.py": "skyscope/core/crypto_manager.py",
    "autonomous_business_operations.py": "skyscope/ops/autonomous_business_operations.py",
    "main_launcher.py": "main_launcher.py",  # Keep at root for easy access
    "install.py": "install.py",  # Keep at root for easy access
    "requirements.txt": "requirements.txt",  # Keep at root for easy access
    "README.md": "README.md",  # Keep at root for easy access
    "SYSTEM_OVERVIEW.md": "docs/SYSTEM_OVERVIEW.md",
    "INTEGRATION_PLAN.md": "docs/INTEGRATION_PLAN.md",
    "DEPLOYMENT_INSTRUCTIONS.md": "docs/DEPLOYMENT_INSTRUCTIONS.md",
}

# Files to ignore (not copied to target)
IGNORE_FILES = [
    ".git",
    ".gitignore",
    ".DS_Store",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dylib",
    "*.dll",
    "*.exe",
    "*.log",
    "venv",
    ".venv",
    "env",
    ".env",
    "node_modules",
    "*.egg-info",
    "dist",
    "build",
]

# Redundancy patterns to detect and clean up
REDUNDANCY_PATTERNS = [
    r"^(.+)-\d+\.py$",  # Files like app-2.py
    r"^(.+)_backup\.py$",  # Files like app_backup.py
    r"^(.+)\.py\.bak$",  # Files like app.py.bak
    r"^copy_of_(.+)\.py$",  # Files like copy_of_app.py
]

class GitHubDeployer:
    """Handles deployment of Skyscope Sentinel Intelligence AI to GitHub."""
    
    def __init__(self, repo_path: str, branch_name: str, dry_run: bool = False, 
                 force: bool = False, no_cleanup: bool = False):
        """Initialize the GitHub deployer."""
        self.repo_path = Path(repo_path).absolute()
        self.branch_name = branch_name
        self.dry_run = dry_run
        self.force = force
        self.no_cleanup = no_cleanup
        self.current_dir = Path.cwd().absolute()
        
        # Validate repo path
        if not self.repo_path.exists():
            raise ValueError(f"Repository path '{self.repo_path}' does not exist")
        
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"'{self.repo_path}' is not a Git repository")
        
        logger.info(f"Initializing GitHub deployer for repository: {self.repo_path}")
        logger.info(f"Target branch: {self.branch_name}")
        logger.info(f"Dry run: {self.dry_run}")
    
    def _run_git_command(self, command: List[str]) -> Tuple[int, str, str]:
        """Run a Git command and return exit code, stdout, and stderr."""
        if self.dry_run and command[0] not in ["status", "branch", "log", "diff"]:
            logger.info(f"[DRY RUN] Would run: git {' '.join(command)}")
            return 0, "", ""
        
        try:
            process = subprocess.Popen(
                ["git"] + command,
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()
            return process.returncode, stdout, stderr
        except Exception as e:
            logger.error(f"Error running Git command: {e}")
            return 1, "", str(e)
    
    def setup_branch(self) -> bool:
        """Set up the target branch for integration."""
        logger.info(f"Setting up branch: {self.branch_name}")
        
        # Check if branch exists
        exit_code, stdout, stderr = self._run_git_command(["branch", "--list", self.branch_name])
        branch_exists = self.branch_name in stdout
        
        if branch_exists:
            logger.info(f"Branch '{self.branch_name}' already exists")
            if not self.force:
                response = input(f"Branch '{self.branch_name}' already exists. Switch to it? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Aborting branch setup")
                    return False
        
        # Create or switch to branch
        if branch_exists:
            exit_code, stdout, stderr = self._run_git_command(["checkout", self.branch_name])
        else:
            exit_code, stdout, stderr = self._run_git_command(["checkout", "-b", self.branch_name])
        
        if exit_code != 0:
            logger.error(f"Failed to set up branch: {stderr}")
            return False
        
        logger.info(f"Successfully set up branch: {self.branch_name}")
        return True
    
    def create_directory_structure(self) -> bool:
        """Create the target directory structure."""
        logger.info("Creating directory structure")
        
        for directory in TARGET_STRUCTURE:
            dir_path = self.repo_path / directory
            if not dir_path.exists():
                logger.info(f"Creating directory: {directory}")
                if not self.dry_run:
                    dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files for Python packages
        python_packages = [
            "skyscope",
            "skyscope/core",
            "skyscope/ui",
            "skyscope/ops",
            "skyscope/data",
            "skyscope/tests",
        ]
        
        for package in python_packages:
            init_file = self.repo_path / package / "__init__.py"
            if not init_file.exists():
                logger.info(f"Creating __init__.py in {package}")
                if not self.dry_run:
                    with open(init_file, 'w') as f:
                        f.write(f'"""\n{COMPANY_NAME} - {package.split("/")[-1].title()} Package\n"""\n\n')
        
        logger.info("Directory structure created successfully")
        return True
    
    def map_files(self) -> Dict[str, str]:
        """Map source files to target locations based on FILE_MAPPING."""
        logger.info("Mapping files to target locations")
        
        file_map = {}
        for source, target in FILE_MAPPING.items():
            source_path = self.current_dir / source
            if source_path.exists():
                file_map[str(source_path)] = str(self.repo_path / target)
            else:
                logger.warning(f"Source file not found: {source}")
        
        logger.info(f"Mapped {len(file_map)} files")
        return file_map
    
    def detect_redundancies(self) -> List[Path]:
        """Detect redundant files in the repository."""
        if self.no_cleanup:
            logger.info("Skipping redundancy detection as requested")
            return []
        
        logger.info("Detecting redundant files")
        
        redundant_files = []
        python_files = list(self.repo_path.glob("**/*.py"))
        
        # Group files by base name (without redundancy patterns)
        file_groups = {}
        for file_path in python_files:
            file_name = file_path.name
            base_name = file_name
            
            # Check if file matches any redundancy pattern
            for pattern in REDUNDANCY_PATTERNS:
                match = re.match(pattern, file_name)
                if match:
                    base_name = match.group(1) + ".py"
                    break
            
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)
        
        # Identify redundant files (more than one file with same base name)
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Keep the newest file, mark others as redundant
                newest_file = max(files, key=lambda f: f.stat().st_mtime)
                for file_path in files:
                    if file_path != newest_file:
                        redundant_files.append(file_path)
                logger.info(f"Found redundant files for {base_name}: keeping {newest_file.name}, "
                           f"redundant: {[f.name for f in files if f != newest_file]}")
        
        logger.info(f"Detected {len(redundant_files)} redundant files")
        return redundant_files
    
    def copy_files(self, file_map: Dict[str, str]) -> bool:
        """Copy files to their target locations."""
        logger.info("Copying files to target locations")
        
        for source, target in file_map.items():
            source_path = Path(source)
            target_path = Path(target)
            
            # Create target directory if it doesn't exist
            target_dir = target_path.parent
            if not target_dir.exists() and not self.dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if target file exists
            if target_path.exists() and not self.force:
                if not self.dry_run:
                    # Compare file contents
                    with open(source_path, 'rb') as f1, open(target_path, 'rb') as f2:
                        if hashlib.md5(f1.read()).hexdigest() == hashlib.md5(f2.read()).hexdigest():
                            logger.info(f"Skipping identical file: {target_path}")
                            continue
                    
                    response = input(f"File '{target_path}' already exists. Overwrite? (y/n): ")
                    if response.lower() != 'y':
                        logger.info(f"Skipping file: {target_path}")
                        continue
            
            # Copy file
            logger.info(f"Copying {source_path} -> {target_path}")
            if not self.dry_run:
                shutil.copy2(source_path, target_path)
        
        logger.info("Files copied successfully")
        return True
    
    def cleanup_redundancies(self, redundant_files: List[Path]) -> bool:
        """Clean up redundant files."""
        if self.no_cleanup:
            logger.info("Skipping redundancy cleanup as requested")
            return True
        
        logger.info("Cleaning up redundant files")
        
        for file_path in redundant_files:
            if not self.force:
                if not self.dry_run:
                    response = input(f"Delete redundant file '{file_path}'? (y/n): ")
                    if response.lower() != 'y':
                        logger.info(f"Skipping deletion of: {file_path}")
                        continue
            
            logger.info(f"Deleting redundant file: {file_path}")
            if not self.dry_run:
                file_path.unlink()
        
        logger.info("Redundancy cleanup completed")
        return True
    
    def update_imports(self) -> bool:
        """Update import statements in Python files to match the new structure."""
        logger.info("Updating import statements")
        
        # Map of old import to new import
        import_map = {
            "from agent_manager import": "from skyscope.core.agent_manager import",
            "import agent_manager": "import skyscope.core.agent_manager as agent_manager",
            "from business_manager import": "from skyscope.ops.business_manager import",
            "import business_manager": "import skyscope.ops.business_manager as business_manager",
            "from crypto_manager import": "from skyscope.core.crypto_manager import",
            "import crypto_manager": "import skyscope.core.crypto_manager as crypto_manager",
            "from autonomous_business_operations import": "from skyscope.ops.autonomous_business_operations import",
            "import autonomous_business_operations": "import skyscope.ops.autonomous_business_operations as autonomous_business_operations",
            "from enhanced_chat_interface import": "from skyscope.ui.enhanced_chat_interface import",
            "import enhanced_chat_interface": "import skyscope.ui.enhanced_chat_interface as enhanced_chat_interface",
            "from app import": "from skyscope.ui.app import",
            "import app": "import skyscope.ui.app as app",
        }
        
        # Find all Python files in the repository
        python_files = list(self.repo_path.glob("**/*.py"))
        
        for file_path in python_files:
            # Skip files in the skyscope package itself
            if "skyscope" in str(file_path) and any(pkg in str(file_path) for pkg in ["core", "ui", "ops", "data", "tests"]):
                continue
            
            logger.info(f"Updating imports in: {file_path}")
            
            if not self.dry_run:
                # Read file content
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Apply import replacements
                new_content = content
                for old_import, new_import in import_map.items():
                    new_content = new_content.replace(old_import, new_import)
                
                # Write updated content if changed
                if new_content != content:
                    with open(file_path, 'w') as f:
                        f.write(new_content)
        
        logger.info("Import statements updated successfully")
        return True
    
    def create_gitignore(self) -> bool:
        """Create or update .gitignore file."""
        logger.info("Creating/updating .gitignore file")
        
        gitignore_path = self.repo_path / ".gitignore"
        gitignore_entries = [
            "# Python",
            "__pycache__/",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "env/",
            "venv/",
            ".venv/",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "*.egg-info/",
            ".installed.cfg",
            "*.egg",
            "",
            "# Skyscope specific",
            "data/*",
            "!data/.gitkeep",
            "logs/*",
            "!logs/.gitkeep",
            "config/*",
            "!config/.gitkeep",
            "credentials/*",
            "!credentials/.gitkeep",
            ".env",
            "*.log",
            "",
            "# OS specific",
            ".DS_Store",
            "Thumbs.db",
            ".directory",
            "",
            "# IDE specific",
            ".idea/",
            ".vscode/",
            "*.swp",
            "*.swo",
        ]
        
        if not self.dry_run:
            # Read existing .gitignore if it exists
            existing_entries = []
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    existing_entries = f.read().splitlines()
            
            # Merge entries, keeping existing ones
            merged_entries = existing_entries.copy()
            for entry in gitignore_entries:
                if entry not in existing_entries:
                    merged_entries.append(entry)
            
            # Write updated .gitignore
            with open(gitignore_path, 'w') as f:
                f.write('\n'.join(merged_entries))
        
        logger.info(".gitignore file created/updated successfully")
        return True
    
    def commit_changes(self) -> bool:
        """Commit changes to the repository."""
        logger.info("Committing changes")
        
        # Add all changes
        exit_code, stdout, stderr = self._run_git_command(["add", "."])
        if exit_code != 0:
            logger.error(f"Failed to add changes: {stderr}")
            return False
        
        # Commit changes
        commit_message = f"chore: integrate Skyscope Sentinel Intelligence AI v{VERSION}\n\n" \
                         f"- Restructured repository according to INTEGRATION_PLAN.md\n" \
                         f"- Updated import statements\n" \
                         f"- Cleaned up redundant files\n" \
                         f"- Added/updated documentation"
        
        exit_code, stdout, stderr = self._run_git_command(["commit", "-m", commit_message])
        if exit_code != 0:
            logger.error(f"Failed to commit changes: {stderr}")
            return False
        
        logger.info("Changes committed successfully")
        return True
    
    def run(self) -> bool:
        """Run the deployment process."""
        logger.info(f"Starting deployment of {COMPANY_NAME} to GitHub")
        
        # Set up branch
        if not self.setup_branch():
            return False
        
        # Create directory structure
        if not self.create_directory_structure():
            return False
        
        # Map files
        file_map = self.map_files()
        
        # Detect redundancies
        redundant_files = self.detect_redundancies()
        
        # Copy files
        if not self.copy_files(file_map):
            return False
        
        # Clean up redundancies
        if not self.cleanup_redundancies(redundant_files):
            return False
        
        # Update imports
        if not self.update_imports():
            return False
        
        # Create/update .gitignore
        if not self.create_gitignore():
            return False
        
        # Commit changes
        if not self.commit_changes():
            return False
        
        logger.info(f"Deployment of {COMPANY_NAME} to GitHub completed successfully")
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description=f"Deploy {COMPANY_NAME} to GitHub")
    parser.add_argument("--repo", default=".", help="Path to the local GitHub repository")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Name of the branch to create for integration")
    parser.add_argument("--dry-run", action="store_true", help="Run without making any changes")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files without prompting")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip redundancy cleanup step")
    
    args = parser.parse_args()
    
    try:
        deployer = GitHubDeployer(
            repo_path=args.repo,
            branch_name=args.branch,
            dry_run=args.dry_run,
            force=args.force,
            no_cleanup=args.no_cleanup
        )
        
        success = deployer.run()
        if not success:
            logger.error("Deployment failed")
            sys.exit(1)
        
        # Print next steps
        print("\n" + "="*80)
        print(f"\n✅ {COMPANY_NAME} deployed to GitHub successfully!")
        print("\nNext steps:")
        print(f"1. Review the changes in branch '{args.branch}'")
        print("2. Create a pull request to merge the changes into the main branch")
        print("3. After the pull request is approved and merged, tag the release:")
        print(f"   git tag -a v{VERSION} -m '{COMPANY_NAME} v{VERSION}'")
        print(f"   git push origin v{VERSION}")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
