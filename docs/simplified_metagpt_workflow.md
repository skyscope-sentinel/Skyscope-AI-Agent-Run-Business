# Simplified MetaGPT-Inspired Workflow for Ollama-based Agents

This document outlines a minimal, simplified workflow inspired by MetaGPT, designed for an initial implementation using Ollama-based AI agents. The goal is to capture the essence of role-based task decomposition and document handoff.

## Core Principle

Simulate a basic product development flow where a requirement is transformed into a specification, which is then used to generate code. This involves two primary agent roles.

## Agent Roles and Workflow

1.  **User (Initiator):**
    *   **Action:** Provides an initial idea or requirement for a simple software component.
    *   **Output:** A natural language string (e.g., "Create a command-line calculator that can add and subtract two numbers and display the result.").

2.  **ProductManagerAgent (`PM_Agent`):**
    *   **Input:** User Idea/Requirement (string).
    *   **Process:**
        *   Leverages an Ollama-based LLM (via an `OllamaWorkerAgent` or similar mechanism).
        *   The LLM is prompted to analyze the User Idea/Requirement and structure it into a basic Product Requirement Document (PRD).
        *   The PRD should be simple, for example, a dictionary or structured string containing:
            *   `project_name`: (e.g., "CLI Calculator")
            *   `description`: A brief summary of the project.
            *   `features`: A list of key functionalities (e.g., ["Accept two numeric inputs", "Allow user to choose 'add' or 'subtract' operation", "Display the calculated result"]).
            *   `target_platform`: (e.g., "Command Line Interface")
    *   **Output:** PRD (structured text or dictionary).

3.  **EngineerAgent (`Eng_Agent`):**
    *   **Input:** PRD (structured text or dictionary) from the `ProductManagerAgent`.
    *   **Process:**
        *   Leverages an Ollama-based LLM (via an `OllamaWorkerAgent` or similar mechanism).
        *   The LLM is prompted with the PRD (specifically the features and description) to generate source code for the application (e.g., in Python).
        *   The focus is on generating a single file or a few simple functions that attempt to meet the PRD.
    *   **Output:** Source Code (string).

## Data Flow (Message Content)

*   **User to `PM_Agent`:**
    *   `message_content = {"requirement_text": "Create a CLI calculator..."}`
*   **`PM_Agent` to `Eng_Agent`:**
    *   `message_content = {"prd": {"project_name": "CLI Calculator", "description": "...", "features": [...]}}`
*   **`Eng_Agent` to User/Log (for MVP):**
    *   `message_content = {"source_code": "def add(a, b): return a + b..."}`

## Simplifications for MVP

*   **Two Core Roles:** Only Product Manager and Engineer are implemented initially.
*   **Text-Based "Documents":** PRD and Code are passed as strings or simple structured data (dictionaries) within messages. No complex file handling or rich document formats.
*   **Direct Handoff:** PM sends PRD directly to Engineer. No intermediate roles like Architect.
*   **No Code Review/QA:** Engineer's output is considered final for the MVP.
*   **Single Iteration:** The flow is linear (User -> PM -> Engineer). No feedback loops or iterative refinement in this first version.
*   **Ollama Dependency:** Both agents rely on an underlying Ollama-powered service for their core "thinking" (generating PRD content, generating code). The quality of output will depend on the model and prompts used.

This simplified workflow provides a clear path for a first implementation, allowing us to test the basic agent interaction, message passing, and Ollama integration in a MetaGPT-inspired context.
