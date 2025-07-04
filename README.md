# Skyscope Sentinel Intelligence

**Skyscope Sentinel Intelligence** is a fully automated, AI-driven business platform designed to empower entrepreneurs with cutting-edge tools for website generation, cryptocurrency mining, trading, decentralized node operations, social media management, and real-time analytics. 

Founded by Miss Casey Jay Topojani, this platform operates locally on your Ubuntu Linux system, leveraging the lightweight **Gemma 3** (1B parameter) AI model via Ollama. 

It eliminates reliance on external APIs, subscriptions, or manual inputs, ensuring autonomy, security, and high revenue potential through continuous evolution driven by intelligent agents.


## Key Features

**AI-Driven Automation* 
Multi-agent system for research, website creation, crypto trading, and social media content generation.
    *   **Opportunity Scouting Swarm**: An autonomous multi-agent swarm (built with `kyegomez/swarms`) that researches and identifies potential income-generating opportunities, saving reports to the workspace. Triggerable via the GUI.
  
**Kaspa Mining*
Optimized CPU-based mining for Kaspa blockchain using `kaspad`.
  
**Cryptocurrency Trading*
Simulated trading strategies with plans for real exchange integration.
  
**Web3 Nodes*
Lightweight Ethereum nodes for DeFi and staking opportunities.
  
**Website Generation*
Beautiful, AI-generated websites with Gen AI apps, hosted locally.
  
**Social Media Influence*
Viral content creation for platforms like X, maximizing engagement.
  
**Live Chat & Support*
Real-time customer interaction powered by AI agents.
  
**Compliance & Security*
GDPR/CCPA compliance checks and secure credential storage with Vault.
  
**Monitoring & Analytics*
Prometheus and Grafana for real-time performance insights.
  
**Local Operation*
Runs on your Intel i7 12700 PC with 32 GB RAM, no GPU required.

**Please make modifications in script/s for your personal workstation/PC/Mac or other system hardware variables for compatibility.**

## Repository

**The platform's source code is hosted at**[https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business]

## Prerequisites

**Operating System*
Ubuntu Linux (20.04 or later recommended)
  
**Hardware*
Intel i7 12700 CPU, 32 GB DDR4 RAM, 50 GB free disk space **(default)*

**Please make modifications in script/s for your personal workstation/PC/Mac or other system hardware variables for compatibility.**

**Internet* 
Required for initial setup and dependency downloads.
  
**User Permissions*
sudo access for installing system packages.

## Installation Steps

**To set up Skyscope Sentinel Intelligence, follow these steps to install prerequisites and deploy the platform.**

**Two scripts are provided*

**`01-run-first.sh` for prerequisites*
**and** **`01-main.sh` for platform setup.*

## Step 1 Clone the Repository

**Clone the repository to your local machine*

``` bash
git clone https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business.git
cd Skyscope-AI-Agent-Run-Business
```

## Step 2

**Run the Prerequisite Script (01-run-first.sh)The 01-run-first.sh script installs all necessary dependencies, including Anaconda3, Python 3, Node.js, Docker, Ollama, kaspad, and required Python packages.**

**Make the script executable*
`chmod +x 01-run-first.sh`

**Execute the script:*
`./01-run-first.sh`

**Monitor the installation log*
`tail -f ~/skyscope_prereq_install.log`

**This script**

**Updates the system and installs tools curl, git, wget, etc.*

**Installs Anaconda3-2024.06-1 from*
`https://repo.anaconda.com/archive/`

**Installs Python 3.10, python3-pip, and creates a virtual environment venv.*

**Installs Node.js v18, npm, Docker, Docker Compose, Ollama with gemma3:1b, and kaspad.*

**Installs Python packages fastapi, uvicorn, requests, pandas, tensorflow, etc.*

**Verifies installations and provides instructions for the next step.*

## Step 3

**Run the Main Script (01-main.sh)The 01-main.sh script configures and deploys the Skyscope Sentinel Intelligence platform, setting up the frontend, backend, agents, and monitoring.**

**Activate the virtual environment created by running*

`01-run-first.sh`

**followed by*

`source venv/bin/activate`

**Make the script executable*

`chmod +x 01-main.sh`

**Execute the script*

`./01-main.sh`

**Monitor the setup log*

`tail -f ~/skyscope-sentinel/logs/setup.log`

**This script:Verifies hardware (Intel i7 12700, 32 GB RAM).*

**Configures Docker services frontend, backend, MariaDB, Redis, Vault, Prometheus, Grafana.*

**Sets up the Next.js frontend, FastAPI backend, and AI agents.*

**Generates cryptocurrency wallets (Ethereum, Kaspa) and stores credentials in*

`~/Skyscope_manager.txt`

**Deploys the platform, accessible at*

`http://localhost:3000`

**API*

`http://localhost:6000`

**Grafana*

`http://localhost:3001`

## Step 4 

**Access the Platform**

**Frontend*

**Open* `http://localhost:3000` **in a browser to interact with the platform.*

**API**

**Test the API health endpoint*

`curl http://localhost:6000/health`

**Grafana**

**View metrics at*

`http://localhost:3001`

**Credentials**
**Check wallet addresses and credentials*

`cat ~/Skyscope_manager.txt`


## Step 5

**Verify Operation**

 `ps`

**Check platform logs for errors*

`cat ~/skyscope-sentinel/logs/platform.log`

**Monitor mining and trading activities in* **Grafana dashboards**

**Troubleshooting**

**Review logs*

**Prerequisite log*

`~/skyscope_prereq_install.log`

**Setup log*

`~/skyscope-sentinel/logs/setup.log`

**Docker Issues**

**Enter Directory & Restart services*

`cd ~/skyscope-sentinel`

**followed by*

``` bash
sudo docker-compose restart
```

### Permission Errors

**Run scripts with sudo if needed**

`sudo ./01-run-first.sh`

**then*

`sudo ./01-main.sh`

**Security Notes**

**Skyscope_manager.txt: An agent generated .txt for users, this contains sensitive wallet mnemonics and credentials.*

**Keep it secure and never upload it to GitHub.**

### Vault

**Credentials are stored securely using HashiCorp Vault.**

### Contributing

**To contribute to Skyscope Sentinel Intelligence**

**Add new agents in agents/ directory.*

**Extend the frontend in frontend/pages/.*

**Update API endpoints in backend/main.py.*

**Submit pull requests to https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business.*

### License

**This project is licensed under the MIT License.**

**See the LICENSE file in the repository for details.*

### Contact

**For support or inquiries, contact Miss Casey Jay Topojani via the business email for customer and developer
related concerns*

**Business Email**

**admin@skyscopeglobal.net* **or alternatively** **skyscopesentinel@gmail.com.*

**Powered by Skyscope Sentinel Intelligence**

**Founded and developed by Miss Casey Jay Topojani.*


## Skyscope Sentinel Windows GUI

Alongside the web platform, Skyscope Sentinel also offers a dedicated Windows GUI application for enhanced local management of specific platform components, particularly Ollama model management and application configuration. This GUI is built using Python with the PySide6 framework, featuring a modern, themeable interface with acrylic/glassy effects and rounded window corners (dark theme by default).

*(Placeholder for Screenshot: A general view of the Skyscope Sentinel Windows GUI main window, showcasing the dark theme, acrylic sidebar, rounded frameless window, and the Dashboard page.)*

### Purpose

The Windows GUI provides a user-friendly interface for:
*   Managing local Ollama models (listing, downloading, viewing details).
*   Configuring application-specific settings.
*   Monitoring the status of the Ollama service.
*   A streamlined experience for Windows users interacting with the AI backend components.

### Launching the GUI

Currently, the GUI can be launched from the source code:

1.  Ensure Python (3.8+) and pip are installed.
2.  Clone the repository (if you haven't already).
3.  Navigate to the repository root.
4.  It's recommended to create a virtual environment:
    ```bash
    python -m venv venv_gui
    source venv_gui/bin/activate  # On Linux/macOS
    .\venv_gui\Scripts\activate   # On Windows
    ```
5.  Install required GUI dependencies:
    ```bash
    pip install -r requirements_gui.txt
    ```
    (`requirements_gui.txt` includes PySide6, moviepy, and other necessary GUI packages. This file ensures all specific dependencies for the GUI are installed.)
6.  Run the GUI using:
    ```bash
    python -m skyscope_sentinel.main
    ```

*(Future: Instructions for launching a packaged executable created with PyInstaller will be added here.)*

### Key Features & Sections

The GUI provides the following key sections, accessible via the sidebar navigation:

*(Placeholder for Screenshot: A composite image showing snippets of the Model Hub, the enhanced Settings page (Appearance tab with Accent Color and UI Scaling), the Agent Control page, and the Log Stream page.)*

*   **Dashboard (Placeholder):** Provides an overview of system status, active agents, and key metrics. Displays information in a card layout (e.g., "Active Agents," "System Status," "Recent Activity," "Model Performance").
*   **Agent Control (Placeholder):** Designed for managing and configuring AI agents. Includes a list view for agents (with placeholder statuses like "Offline", "Running"), controls to (placeholder) Start, Stop, Configure selected agents, View Logs, and an "Add New Agent..." button.
*   **Video Tools:** Provides utilities for video processing. Currently includes:
    *   **Video Colorization (Simulated):** A placeholder for future AI-driven video colorization. Allows selecting an input B&W video and specifying an output path.
    *   **Images to Video:** Allows users to select multiple images, set FPS and duration per image, and create a slideshow video.
*   **Model Hub:**
    *   **Installed Models:** View a list of all Ollama models available on your local system (Name, Size, Family, Quantization). You can refresh this list and view detailed information (including Modelfile and License) for each model.
    *   **Download Models:** Search for and download new models from the Ollama Hub directly within the application. Download progress is displayed.
    *   **Ollama Service Status:** Check the current status and version of your local Ollama service.
*   **Log Stream (Placeholder):** Provides a centralized view for application logs. Includes a main display area for logs (populated with sample logs), a dropdown to filter by source (e.g., "Application Logs," "Ollama Service," specific agents), a disabled search bar, and a disabled "Clear Logs" button.
*   **Settings:**
    *   **General:** Configure application startup behavior (preference saved, actual OS-level autostart is a future enhancement), system tray icon preferences (enable/disable, minimize on close, show notification on minimize to tray).
    *   **Appearance:** Switch between Dark (default) and Light themes, toggle acrylic/transparency effects for the sidebar.
        *   **Accent Color:** Choose a custom accent color for UI highlights (preview swatch updates, saved for future theme use).
        *   **UI Scaling:** Select UI scaling preference (Small, Medium, Large). Requires application restart to take effect (preference saved, actual scaling implementation is a future enhancement).
    *   **Ollama:** Configure the Ollama service URL, test connectivity, and set preference for attempting to auto-start the local Ollama service when the application launches (actual auto-start action happens at application startup if enabled).
    *   **Agents:** Set default agent log levels (preference saved). Set preference for automatically restarting crashed agents (actual monitoring and restart logic is part of a future agent management system).
    *   **Advanced:** Manage application data folder location (browse and set) and reset all settings to default. Placeholder for clearing application cache.
    *   Settings are persisted locally using `QSettings` (typically in the system registry on Windows or a `.ini` file).
*   **System Tray Icon:**
    *   Provides quick access to show/hide the application window via left-click or context menu.
    *   Context menu includes "Quit" option to close the application.
    *   The application can be configured to minimize to the tray instead of closing (default is True).
    *   Uses a placeholder `app_icon.png` (custom icon recommended for packaging).

### Core Feature Usage

*   **Managing Ollama Models:**
    1.  Navigate to the "Model Hub".
    2.  Click "Refresh List" to see your locally installed models. Select a model and click "View Details" (or double-click) for more information.
    3.  To download a new model, enter its name (e.g., `llama3:latest`) in the "Download New Models" section and click "Download".
*   **Changing Themes:**
    1.  Click the "Toggle Theme" button in the sidebar for a quick switch between Dark and Light themes. This change is saved.
    2.  Alternatively, go to "Settings" -> "Appearance" and select your desired theme from the dropdown.
*   **Configuring Settings:**
    1.  Navigate to "Settings".
    2.  Select the appropriate tab (General, Appearance, etc.).
    3.  Modify the settings as needed. Changes are saved automatically when a UI element's state changes (e.g., checkbox toggled, input field loses focus).
    4.  Status bar messages provide feedback on saved settings or actions.

## Dependencies

### Python Packages

The main application dependencies are listed in `requirements.txt`. For the GUI specifically, install using:

```bash
pip install -r requirements_gui.txt
```

The `requirements_gui.txt` file includes:
*   `PySide6>=6.0.0`: For the application's graphical user interface.
*   `moviepy>=1.0.3`: For creating videos from image sequences in the Video Tools page. Also uses `numpy`, `imageio`, `Pillow`, `tqdm`, `decorator`.
*   `Pillow>=9.0.0`: For image manipulation tasks, also a dependency for moviepy.

### System-Level Dependencies

*   **ffmpeg**: `moviepy` (used for the Images to Video feature) requires `ffmpeg` to be installed on your system and accessible in the system's PATH.
    *   **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the `bin` directory to your PATH.
    *   **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install ffmpeg`
    *   **macOS (using Homebrew)**: `brew install ffmpeg`

### Conceptual Dependencies (for future full DeOldify integration)

The current video colorization feature is a simulation. A full integration of the DeOldify library would require additional heavy dependencies, including:
*   `torch` (PyTorch)
*   `fastai==1.0.61` (specific older version often required by DeOldify)
*   `torchvision`
*   `opencv-python`
*   Pre-trained DeOldify model weights.
