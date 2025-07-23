---
description: Repository Information Overview
alwaysApply: true
---

# Skyscope Sentinel Intelligence Information

## Summary
Skyscope Sentinel Intelligence is a multi-agent AI operating system built on the Swarms framework. It orchestrates specialized expert pipelines to research, code, analyze, automate browser interactions, manipulate files, and run quantum-inspired simulations through a Streamlit-based UI.

## Structure
- **skyscope_sentinel/**: Core application module with agent implementations, tools, and UI components
- **swarm-agent/**: TypeScript-based agent framework with client/server architecture
- **tests/**: Unit tests for file I/O and vector store utilities
- **scripts/**: Utility scripts for running demos and deployment
- **docs/**: Documentation for cooperative agents framework and system design

## Language & Runtime
**Language**: Python 3.10
**Framework**: Streamlit
**Build System**: pip
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- swarms (≥0.3.0): Multi-agent orchestration framework
- langchain (≥0.0.267): LLM application framework
- crewai (≥0.28.0): Agent collaboration framework
- streamlit (≥1.28.0): Web UI framework
- playwright (≥1.39.0): Browser automation
- chromadb (≥0.4.15): Vector database

**Development Dependencies**:
- pytest (≥7.4.2): Testing framework
- black (≥23.9.1): Code formatter
- flake8 (≥6.1.0): Linter
- pre-commit (≥3.5.0): Git hooks manager

## Build & Installation
```bash
# Create environment
python -m venv venv && source venv/bin/activate
# or
conda create -n sentinel python=3.10 && conda activate sentinel

# Install dependencies
pip install -r requirements.txt

# Optional: Install Playwright browsers for automation
playwright install --with-deps
```

## Docker
**Dockerfile**: Multi-Agent-Template-App-main/Dockerfile
**Base Image**: python:3.10-slim
**Configuration**: Sets up Python environment, installs dependencies, and copies application files

## Testing
**Framework**: unittest
**Test Location**: tests/
**Naming Convention**: test_*.py
**Run Command**:
```bash
python -m unittest discover tests
# or
python -m tests.test_file_io_tools
```

## Main Components
- **app.py**: Streamlit frontend, routing, UI state management
- **agent_manager.py**: Factory for agents/swarms, task orchestration, vector DB integration
- **config.py**: Configuration management and environment variable handling
- **quantum_manager.py**: Quantum computing simulation and visualization
- **browser_automation.py**: Playwright-based web automation

## API Integration
**Supported Providers**:
- OpenAI (GPT-4o)
- Anthropic (Claude-3)
- Google (Gemini)
- HuggingFace (Llama3)
- Local Ollama models (Llama3, Mistral, Gemma, etc.)

## File System Access
**Capabilities**:
- Read/write files with directory and extension safelists
- Upload/download files through the UI
- Process PDFs, text files, images, and videos

## Browser Automation
**Features**:
- Chromium browser launch within the application
- Natural language commands for navigation and interaction
- Screenshot capture and content extraction