# AGENTS.md - Guidance for AI Agents

This document provides guidance and context for AI agents (like Jules, Aider, etc.) working on the Skyscope Sentinel Intelligence codebase. Its purpose is to help agents understand the project's architecture, key components, and conventions.

## Core Principles for Agents

*   **Understand the Goal:** The primary objective of Skyscope Sentinel Intelligence is to become a multi-agent powered enterprise platform capable of autonomously generating high income through various online means, starting with no initial funds.
*   **Local First (Ollama):** Prioritize using local Ollama models for LLM tasks to ensure privacy, reduce costs, and enable offline capabilities where possible. User-provided API keys for other services (OpenAI, Serper, E2B) are supported for enhanced capabilities.
*   **Modularity:** Strive for modular design in agent and tool development.
*   **Swarm Intelligence:** The project is increasingly leveraging the `kyegomez/swarms` framework for orchestrating multiple agents. Familiarize yourself with its concepts (Agents, Swarms, Workflows, Tools).
*   **User Experience:** The application has a PySide6 GUI. Changes should be reflected appropriately in the GUI, and tasks should run asynchronously to keep the UI responsive.
*   **Workspace:** Agents should use the `./workspace/` directory for file-based outputs (reports, generated code, etc.). Configuration for this is in `config.py`.

## Core Agent Frameworks & Integrations

The project utilizes several AI agent frameworks and libraries:

1.  **`kyegomez/swarms`**: This is becoming the primary framework for multi-agent orchestration.
    *   **`SkyscopeSwarmAgent`**: Located in `skyscope_sentinel/swarms_integration/skyscope_swarm_agent.py`, this is the base class for all agents intended to work within the `swarms` framework in this project. It integrates Skyscope's identity system and default Ollama configuration.
    *   Specialized swarms (like the Opportunity Scouting Swarm) are built using these agents and `swarms` workflow structures (e.g., `SequentialWorkflow`).

2.  **CrewAI**: Used for role-based task decomposition, particularly for research tasks.
    *   See `skyscope_sentinel/crews/research_crew.py`.
    *   Currently integrated into the "Opportunity Research" GUI page.

3.  **AutoGen**: Used for creating conversational agents and managing interactions, including human-in-the-loop scenarios.
    *   See `skyscope_sentinel/autogen_interface.py`.
    *   Currently used to orchestrate the CrewAI research crew.

4.  **LangChain**: While not a primary orchestration framework, its components or patterns might be used by other frameworks (e.g., some tools or LLM wrappers).

5.  **Ollama**: The primary local LLM provider. Integration is managed via `OllamaIntegration` class and configured through the GUI settings, affecting `litellm` configurations used by agent frameworks.

6.  **ChromaDB (Vector Store)**: Used for local RAG capabilities.
    *   Integrated via `skyscope_sentinel/tools/vector_store_utils.py`.
    *   Opportunity reports are automatically chunked, embedded (using Ollama via `litellm`), and stored for future contextual retrieval.

## Implemented Swarms

### 1. Opportunity Scouting Swarm

*   **Purpose**: To autonomously research and identify potential income-generating opportunities based on an initial topic (user-provided or self-generated).
*   **Location**: `skyscope_sentinel/swarms_integration/opportunity_scouting_swarm.py`
*   **Workflow**: Utilizes `swarms.SequentialWorkflow`.
    1.  **`TopicGeneratorAgent`**: Generates or refines research topics. Enhanced to potentially use a search tool for inspiration and produce more unique, actionable ideas for low-capital AI ventures.
    2.  **`ResearchAgent`**: Gathers information on the topic using search tools (DuckDuckGo, Serper if API key is available) and web browsing tools (Playwright via `browse_web_page_and_extract_text`). Tools now include basic error handling.
    3.  **`AnalysisAgent`**: Analyzes gathered information. Enhanced to focus critically on zero-cost startup strategies and provide a detailed 8-point analysis for each opportunity, including actionable first steps for AI agents.
    4.  **`ReportingAgent`**: Consolidates analysis into a structured Markdown report. Enhanced to follow a specific, detailed Markdown template for clarity and professionalism.
*   **Output**: Reports are saved in the `workspace/opportunity_reports/` directory. These reports are also automatically added to a ChromaDB vector store for RAG.
*   **GUI Integration**: Triggered from the "Opportunity Research" tab. Users can select "Swarm Opportunity Scouting" mode. Generated Markdown reports are now rendered directly in the GUI.
    1.  **`TopicGeneratorAgent`**: Generates or refines a research topic.
    2.  **`ResearchAgent`**: Gathers information on the topic using search tools (DuckDuckGo, Serper if API key is available) and web browsing tools (Playwright via `browse_web_page_and_extract_text`).
    3.  **`AnalysisAgent`**: Analyzes the gathered information to identify viable opportunities, assess potential revenue, risks, and resource requirements (aiming for low/no-cost).
    4.  **`ReportingAgent`**: Consolidates the analysis into a structured Markdown report.
*   **Output**: Reports are saved in the `workspace/opportunity_reports/` directory with filenames like `opportunity_report_[topic_slug]_[timestamp].md`.
*   **GUI Integration**: Triggered from the "Opportunity Research" tab. Users can select "Swarm Opportunity Scouting" mode and optionally provide an initial topic.

## Key Code Locations

*   **Main Application**: `skyscope_sentinel/main.py` (GUI structure, page switching)
*   **Agent Base Classes**:
    *   `skyscope_sentinel/agents/base_agent.py` (Older base class)
    *   `skyscope_sentinel/owl_integration/owl_base_agent.py` (OWL/CAMEL conceptual base)
    *   `skyscope_sentinel/swarms_integration/skyscope_swarm_agent.py` (Current base for swarms)
*   **Configuration**: `skyscope_sentinel/config.py` (handles API keys, workspace paths, default models) and `skyscope_sentinel/settings_manager.py` (GUI settings persistence).
*   **Tools**: Utility functions for search, browsing, file I/O, code execution, and vector store operations are in `skyscope_sentinel/tools/`.
*   **Identity Management**: `skyscope_sentinel/agent_identity.py` (generates identities for agents).
*   **Unit Tests**: Located in `tests/`. Contains tests for some utility functions.
*   **Design Documents**: Located in `docs/swarms/` for planned future swarms.

## Working with an Agentic Environment

*   **Tool Usage**: When implementing new agent capabilities, prefer creating reusable tool functions (like those in `skyscope_sentinel/tools/`) and providing them to agents rather than having agents generate complex code for common tasks like web searches or file I/O directly.
*   **Error Handling**: Ensure robust error handling, especially for operations involving external APIs, file system access, or web interactions. Search and browser tools now have basic error catching.
*   **Asynchronous Operations**: GUI interactions that trigger long-running agent tasks MUST be run in separate threads (e.g., using `AsyncRunnerThread` in `main.py`) to keep the UI responsive.
*   **RAG System**: Be aware of the ChromaDB vector store for opportunity reports. Future agents might query this store for contextual information.
*   **Tools**: Utility functions for search, browsing, file I/O, code execution are in `skyscope_sentinel/utils/`.
*   **Identity Management**: `skyscope_sentinel/agent_identity.py` (generates identities for agents).

## Working with an Agentic Environment

*   **Tool Usage**: When implementing new agent capabilities, prefer creating reusable tool functions (like those in `utils/`) and providing them to agents rather than having agents generate complex code for common tasks like web searches or file I/O directly.
*   **Error Handling**: Ensure robust error handling, especially for operations involving external APIs, file system access, or web interactions.
*   **Asynchronous Operations**: GUI interactions that trigger long-running agent tasks MUST be run in separate threads (e.g., using `AsyncRunnerThread` in `main.py`) to keep the UI responsive.

This document will be updated as the project evolves. If you make significant architectural changes or add new core agent systems, please update this file accordingly.
