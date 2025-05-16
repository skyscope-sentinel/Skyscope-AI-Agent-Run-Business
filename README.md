# Skyscope Sentinel Intelligence

**Skyscope Sentinel Intelligence** is a fully automated, AI-driven business platform designed to empower entrepreneurs with cutting-edge tools for website generation, cryptocurrency mining, trading, decentralized node operations, social media management, and real-time analytics. 

Founded by Miss Casey Jay Topojani, this platform operates locally on your Ubuntu Linux system, leveraging the lightweight **Gemma 3** (1B parameter) AI model via Ollama. 

It eliminates reliance on external APIs, subscriptions, or manual inputs, ensuring autonomy, security, and high revenue potential through continuous evolution driven by intelligent agents.

## Key Features

- **AI-Driven Automation**: Multi-agent system for research, website creation, crypto trading, and social media content generation.
  
- **Kaspa Mining*
Optimized CPU-based mining for Kaspa blockchain using `kaspad`.
  
- **Cryptocurrency Trading*
Simulated trading strategies with plans for real exchange integration.
  
- **Web3 Nodes*
Lightweight Ethereum nodes for DeFi and staking opportunities.
  
- **Website Generation*
Beautiful, AI-generated websites with Gen AI apps, hosted locally.
  
- **Social Media Influence*
Viral content creation for platforms like X, maximizing engagement.
  
- **Live Chat & Support*
Real-time customer interaction powered by AI agents.
  
- **Compliance & Security*
GDPR/CCPA compliance checks and secure credential storage with Vault.
  
- **Monitoring & Analytics*
Prometheus and Grafana for real-time performance insights.
  
- **Local Operation*
Runs on your Intel i7 12700 PC with 32 GB RAM, no GPU required.

**Please make modifications in script/s for your personal workstation/PC/Mac or other system hardware variables for compatibility.*

## Repository

**The platform's source code is hosted at [https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business]**

## Prerequisites

- **Operating System** Ubuntu Linux (20.04 or later recommended)
  
- **Hardware** Intel i7 12700 CPU, 32 GB DDR4 RAM, 50 GB free disk space **(default)**

**Please make modifications in script/s for your personal workstation/PC/Mac or other system hardware variables for compatibility.*

- **Internet** Required for initial setup and dependency downloads.
  
- **User Permissions** sudo access for installing system packages.

## Installation Steps

**To set up Skyscope Sentinel Intelligence, follow these steps to install prerequisites and deploy the platform.*

**Two scripts are provided**
**`01-run-first.sh` for prerequisites*
**and** **`01-main.sh` for platform setup.*

## Step 1 Clone the Repository

**Clone the repository to your local machine*

``` bash
git clone https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business.git
cd Skyscope-AI-Agent-Run-Business
```

## Step 2

**Run the Prerequisite Script (01-run-first.sh)The 01-run-first.sh script installs all necessary dependencies, including Anaconda3, Python 3, Node.js, Docker, Ollama, kaspad, and required Python packages.*

**Make the script executable**
`chmod +x 01-run-first.sh`

**Execute the script:**
`./01-run-first.sh`

**Monitor the installation log**
`tail -f ~/skyscope_prereq_install.log`

**This script**

**Updates the system and installs tools curl, git, wget, etc.*

**Installs Anaconda3-2024.06-1 from https://repo.anaconda.com/archive/.*

**Installs Python 3.10, python3-pip, and creates a virtual environment venv.*

**Installs Node.js v18, npm, Docker, Docker Compose, Ollama with gemma3:1b, and kaspad.*

**Installs Python packages fastapi, uvicorn, requests, pandas, tensorflow, etc.*

**Verifies installations and provides instructions for the next step.*

## Step 3

**Run the Main Script (01-main.sh)The 01-main.sh script configures and deploys the Skyscope Sentinel Intelligence platform, setting up the frontend, backend, agents, and monitoring.**

**Activate the virtual environment created by `01-run-first.sh`*

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

Generates cryptocurrency wallets (Ethereum, Kaspa) and stores credentials in `~/Skyscope_manager.txt`

**Deploys the platform, accessible at*

`http://localhost:3000`

**API*
`http://localhost:6000`

**Grafana*
`http://localhost:3001`

## Step 4 

**Access the Platform**

**Frontend*

**Open `http://localhost:3000` in a browser to interact with the platform.*

**API**

**Test the API health endpoint*

`curl http://localhost:6000/health`

**Grafana**

**View metrics at*

`http://localhost:3001`

**Credentials*

**Check wallet addresses and credentials*

`cat ~/Skyscope_manager.txt`

## Step 5

**Verify Operation**

 `ps`

**Check platform logs for errors*

`cat ~/skyscope-sentinel/logs/platform.log`

**Monitor mining and trading activities in Grafana dashboards.*

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

**then**

`sudo ./01-main.sh`


**If any files are missing, re-run 01-main.sh**


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
related concerns** 

**Business Email**

**admin@skyscopeglobal.net* **or alternatively** **skyscopesentinel@gmail.com.*

**Powered by Skyscope Sentinel Intelligence**

**Founded and developed by Miss Casey Jay Topojani.*
