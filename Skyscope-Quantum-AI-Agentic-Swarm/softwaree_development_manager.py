import os
import json
import logging
import shutil
import subprocess
from enum import Enum
from typing import Dict, List, Optional, Any, Literal

# Configure logging for clear and informative output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Enums for Software Types and Frameworks ---

class SoftwareType(Enum):
    DESKTOP_APP = "Desktop Application"
    BROWSER_EXTENSION = "Browser Extension"

class Framework(Enum):
    ELECTRON = "Electron"
    TAURI = "Tauri"
    REACT = "React"
    VUE = "Vue"

class BrowserTarget(Enum):
    CHROME = "Chrome"
    FIREFOX = "Firefox"
    SAFARI = "Safari"

# --- Mock/Placeholder AI Agent for Code Generation ---

class MockAgent:
    """A mock AI agent to simulate the generation of code and configuration files."""
    def run(self, task: str) -> str:
        logger.info(f"MockAgent generating code for task: {task[:100]}...")
        
        # Desktop App Generation
        if "package.json for an Electron app" in task:
            return json.dumps({
                "name": "my-electron-app",
                "version": "1.0.0",
                "description": "An autonomously generated desktop application.",
                "main": "main.js",
                "scripts": {
                    "start": "electron .",
                    "test": "echo \"Error: no test specified\" && exit 1",
                    "package": "electron-packager . --platform=win32,darwin,linux --arch=x64"
                },
                "devDependencies": {
                    "electron": "^28.0.0",
                    "electron-packager": "^17.1.2"
                }
            }, indent=2)
        if "main.js for an Electron app" in task:
            return """
const { app, BrowserWindow } = require('electron');

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });
  win.loadFile('index.html');
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
"""
        if "index.html for a desktop app" in task:
            return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>My Autonomous App</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <h1>Welcome to Your AI-Generated App!</h1>
    <p>This application was created autonomously.</p>
    <script src="renderer.js"></script>
</body>
</html>
"""
        if "style.css for a desktop app" in task:
            return "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #282c34; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; }"
        if "renderer.js for a desktop app" in task:
            return "console.log('Renderer process started.');"

        # Browser Extension Generation
        if "manifest.json for a Chrome extension" in task:
            return json.dumps({
                "manifest_version": 3,
                "name": "My Autonomous Extension",
                "version": "1.0",
                "description": "An autonomously generated browser extension.",
                "permissions": ["storage", "activeTab", "scripting"],
                "action": {
                    "default_popup": "popup.html"
                },
                "background": {
                    "service_worker": "background.js"
                }
            }, indent=2)
        if "popup.html for a browser extension" in task:
            return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { width: 200px; text-align: center; }
    </style>
</head>
<body>
    <h3>Autonomous Extension</h3>
    <button id="actionBtn">Click Me</button>
    <script src="popup.js"></script>
</body>
</html>
"""
        if "background.js for a browser extension" in task:
            return "chrome.runtime.onInstalled.addListener(() => { console.log('Extension installed.'); });"
        if "popup.js for a browser extension" in task:
            return "document.getElementById('actionBtn').addEventListener('click', () => { console.log('Button clicked!'); });"

        # CI/CD Workflow Generation
        if "GitHub Actions workflow" in task:
            return """
name: Build and Test
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    - name: Install dependencies
      run: npm install
    - name: Run tests
      run: npm test
    - name: Package application
      run: npm run package
"""
        return ""

# --- Main Software Development Manager Class ---

class SoftwareDevelopmentManager:
    """
    Orchestrates the autonomous creation of desktop software and browser extensions.
    """

    def __init__(self, agent: Any, base_output_dir: str = "generated_software"):
        """
        Initializes the SoftwareDevelopmentManager.

        Args:
            agent (Any): An AI agent instance for code generation.
            base_output_dir (str): The root directory to save generated projects.
        """
        self.agent = agent
        self.base_output_dir = base_output_dir
        os.makedirs(self.base_output_dir, exist_ok=True)

    def _generate_and_write_file(self, project_dir: str, file_path: str, prompt: str):
        """Helper to generate code and write it to a file."""
        code = self.agent.run(prompt)
        full_path = os.path.join(project_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
        logger.info(f"Generated file: {file_path}")

    def _run_command(self, command: str, cwd: str) -> bool:
        """Helper to run a shell command in a specified directory."""
        logger.info(f"Running command: `{command}` in `{cwd}`")
        try:
            subprocess.run(command, cwd=cwd, check=True, shell=True, capture_output=True, text=True)
            logger.info("Command executed successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}.")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False

    def setup_version_control(self, project_dir: str):
        """Initializes a Git repository and makes the first commit."""
        logger.info("Setting up Git version control...")
        if not self._run_command("git init", cwd=project_dir): return
        if not self._run_command("git add .", cwd=project_dir): return
        if not self._run_command("git commit -m \"Initial autonomous commit\"", cwd=project_dir): return

    def setup_ci_cd_pipeline(self, project_dir: str):
        """Generates a basic GitHub Actions workflow file."""
        logger.info("Setting up CI/CD pipeline...")
        prompt = "Generate a GitHub Actions workflow for a Node.js project to build and test."
        self._generate_and_write_file(project_dir, ".github/workflows/build.yml", prompt)

    def create_desktop_app(self, project_name: str, framework: Framework = Framework.ELECTRON) -> Optional[str]:
        """
        Generates a complete, cross-platform desktop application.

        Args:
            project_name (str): The name of the project (e.g., "my-cool-app").
            framework (Framework): The development framework to use.

        Returns:
            The path to the generated project directory, or None on failure.
        """
        if framework != Framework.ELECTRON:
            logger.error(f"Framework '{framework.value}' is not yet supported for autonomous desktop app creation.")
            return None

        logger.info(f"--- Starting Autonomous Desktop App Creation: {project_name} ---")
        project_dir = os.path.join(self.base_output_dir, project_name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        os.makedirs(project_dir)

        # Generate core files
        self._generate_and_write_file(project_dir, "package.json", f"Generate a package.json for an Electron app named '{project_name}'.")
        self._generate_and_write_file(project_dir, "main.js", "Generate a main.js for an Electron app.")
        self._generate_and_write_file(project_dir, "index.html", "Generate a basic index.html for a desktop app.")
        self._generate_and_write_file(project_dir, "style.css", "Generate a simple style.css for a desktop app.")
        self._generate_and_write_file(project_dir, "renderer.js", "Generate a basic renderer.js for a desktop app.")

        # Setup project infrastructure
        self.setup_version_control(project_dir)
        self.setup_ci_cd_pipeline(project_dir)
        
        logger.info(f"--- Desktop App '{project_name}' created successfully at '{project_dir}' ---")
        return project_dir

    def create_browser_extension(self, project_name: str, target: BrowserTarget = BrowserTarget.CHROME) -> Optional[str]:
        """
        Generates a complete browser extension.

        Args:
            project_name (str): The name of the project (e.g., "my-ad-blocker").
            target (BrowserTarget): The target browser for the extension.

        Returns:
            The path to the generated project directory, or None on failure.
        """
        if target != BrowserTarget.CHROME:
            logger.error(f"Browser target '{target.value}' is not yet supported for autonomous extension creation.")
            return None

        logger.info(f"--- Starting Autonomous Browser Extension Creation: {project_name} ---")
        project_dir = os.path.join(self.base_output_dir, project_name)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)
        os.makedirs(project_dir)

        # Generate core files
        self._generate_and_write_file(project_dir, "manifest.json", f"Generate a manifest.json for a Chrome extension named '{project_name}'.")
        self._generate_and_write_file(project_dir, "popup.html", "Generate a popup.html for a browser extension.")
        self._generate_and_write_file(project_dir, "popup.js", "Generate a popup.js for a browser extension.")
        self._generate_and_write_file(project_dir, "background.js", "Generate a background.js for a browser extension.")

        # Setup project infrastructure
        self.setup_version_control(project_dir)

        logger.info(f"--- Browser Extension '{project_name}' created successfully at '{project_dir}' ---")
        return project_dir

    def run_automated_tests(self, project_dir: str) -> bool:
        """Installs dependencies and runs tests for a generated project."""
        logger.info(f"Running automated tests for project at '{project_dir}'...")
        if not os.path.exists(os.path.join(project_dir, "package.json")):
            logger.warning("No package.json found, skipping tests.")
            return True # No tests to run is not a failure
        
        if not self._run_command("npm install", cwd=project_dir):
            logger.error("Failed to install dependencies.")
            return False
        
        return self._run_command("npm test", cwd=project_dir)

    def package_application(self, project_dir: str) -> bool:
        """Packages the application for distribution."""
        logger.info(f"Packaging application at '{project_dir}'...")
        if not os.path.exists(os.path.join(project_dir, "package.json")):
            logger.error("No package.json found, cannot package application.")
            return False
        
        # Ensure dependencies are installed before packaging
        if not os.path.exists(os.path.join(project_dir, "node_modules")):
            if not self._run_command("npm install", cwd=project_dir):
                logger.error("Failed to install dependencies before packaging.")
                return False

        return self._run_command("npm run package", cwd=project_dir)


if __name__ == '__main__':
    logger.info("--- SoftwareDevelopmentManager Demonstration ---")
    
    # 1. Setup
    mock_agent = MockAgent()
    dev_manager = SoftwareDevelopmentManager(agent=mock_agent)
    
    # 2. Create a Desktop Application
    desktop_app_path = dev_manager.create_desktop_app("MyFirstDesktopApp")
    if desktop_app_path:
        logger.info("\n--- Testing and Packaging Desktop App ---")
        # dev_manager.run_automated_tests(desktop_app_path) # This would fail as mock test script exits with 1
        dev_manager.package_application(desktop_app_path)
    
    # 3. Create a Browser Extension
    extension_path = dev_manager.create_browser_extension("MyFirstExtension")
    if extension_path:
        logger.info(f"\nBrowser extension created at '{extension_path}'. No automated test/package step for this demo.")

    logger.info("\n--- Demonstration Finished ---")
    logger.info(f"Check the '{dev_manager.base_output_dir}' directory for generated projects.")
