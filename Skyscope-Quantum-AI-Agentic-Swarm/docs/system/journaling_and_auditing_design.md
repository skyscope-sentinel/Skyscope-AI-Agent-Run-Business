# Enhanced Journaling and Auditing System - Design Document

**Phase RW1 - Design Only**

## 1. Introduction & Goals

With Skyscope Sentinel Intelligence transitioning to real-world operations, a robust journaling, logging, and auditing system is critical. This system must provide comprehensive, structured, and auditable records of all significant AI agent and swarm activities.

**Goals:**

*   **Traceability & Debugging:** Facilitate understanding of agent behavior, decision-making processes, and error diagnosis.
*   **Performance Analysis:** Allow for monitoring and analysis of swarm/agent efficiency and effectiveness.
*   **Security Auditing:** Track access to sensitive resources (like API keys) and identify anomalous behavior.
*   **Compliance & Record Keeping:** Provide necessary records for business operations and potential future regulatory requirements.
*   **Financial Tracking (Future):** Log events related to (simulated and real) financial transactions.
*   **User Oversight:** Eventually feed into a GUI component for user monitoring.

## 2. Data Points to Log

Each log entry should ideally be a structured record (e.g., JSON) containing the following fields where applicable:

*   **`timestamp`**: ISO 8601 format (e.g., `YYYY-MM-DDTHH:MM:SS.ffffffZ`).
*   **`run_id`**: A unique identifier for a complete swarm execution run (e.g., a UUID generated at the start of `run_opportunity_scouting_swarm`).
*   **`swarm_name`**: Name of the parent swarm (e.g., "OpportunityScoutingSwarm", "ContentGenerationSwarm").
*   **`agent_name`**: Name of the agent performing the action (e.g., "TopicGenerator", "ResearchAgent-Alice_123").
*   **`agent_id`**: Unique Skyscope ID of the agent.
*   **`event_type`**: Categorical type of the event (e.g., "SWARM_START", "SWARM_END", "AGENT_START", "AGENT_END", "TOOL_CALL_START", "TOOL_CALL_SUCCESS", "TOOL_CALL_FAILURE", "LLM_CALL_START", "LLM_CALL_SUCCESS", "LLM_CALL_FAILURE", "DECISION_MADE", "ARTIFACT_GENERATED", "ERROR_ENCOUNTERED", "FINANCIAL_EVENT_SIMULATED", "API_KEY_ACCESS_ATTEMPT").
*   **`message`**: A human-readable summary of the event.
*   **`details_json`**: A JSON object containing event-specific details:
    *   For `TOOL_CALL_*`: `tool_name`, `tool_input` (potentially summarized/truncated if large), `tool_output` (summarized/truncated or error message).
    *   For `LLM_CALL_*`: `model_name`, `prompt_snippet` (first/last N chars to avoid logging full sensitive prompts), `response_snippet` or error.
    *   For `ARTIFACT_GENERATED`: `artifact_type` (e.g., "report_file", "proposal_file"), `artifact_path`.
    *   For `DECISION_MADE`: `decision_summary`, `reasoning_snippet` (if extractable from LLM thought process).
    *   For `FINANCIAL_EVENT_SIMULATED`: `event_description`, `amount`, `currency`, `status`.
    *   For `API_KEY_ACCESS_ATTEMPT`: `key_alias_or_type` (e.g., "SERPER_API_KEY", "OPENAI_API_KEY" - never the key itself), `status` ("SUCCESS", "NOT_FOUND", "ACCESS_DENIED").
*   **`duration_ms`**: For events with a duration (e.g., tool calls, LLM calls).
*   **`status`**: "SUCCESS", "FAILURE", "INFO", "WARNING", "DEBUG".

## 3. Log Structure & Format

*   **Primary Format:** JSON per log entry. This allows for easy parsing, querying, and integration with log management systems.
*   **Fallback/Human-Readable:** For console output during development or direct file viewing, a human-readable format can be derived from the JSON structure, but the underlying stored log should be structured.

**Example JSON Log Entry:**
```json
{
  "timestamp": "2024-07-30T10:30:05.123456Z",
  "run_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "swarm_name": "OpportunityScoutingSwarm",
  "agent_name": "ResearchAgent-Bob_456",
  "agent_id": "SSA_Res_002",
  "event_type": "TOOL_CALL_SUCCESS",
  "message": "DuckDuckGo search for 'AI in agriculture' completed.",
  "details_json": {
    "tool_name": "duckduckgo_search_function",
    "tool_input": {"query": "AI in agriculture", "max_results": 5},
    "tool_output_snippet": "Result 1: AI Farming...\nResult 2: Precision Agriculture..."
  },
  "duration_ms": 1560,
  "status": "SUCCESS"
}
```

## 4. Log Storage

*   **Level 1 (Short-Term - File-based per run):**
    *   Each swarm execution run (identified by `run_id`) will generate its own detailed log file.
    *   Location: `workspace/logs/swarm_runs/{swarm_name}/{run_id_timestamp}.jsonl` (JSON Lines format, one JSON object per line).
    *   This provides easy isolation of logs for a specific operation.
*   **Level 2 (Mid-Term - Centralized SQLite Database):**
    *   A local SQLite database (`workspace/logs/skyscope_audit.db`) will store key structured log entries.
    *   **Schema Idea:**
        *   `logs` table: `id (PK)`, `timestamp`, `run_id`, `swarm_name`, `agent_name`, `agent_id`, `event_type`, `status`, `message`, `details_json_ref (TEXT/BLOB or link to details if too large)`.
    *   The file-based logs (Level 1) can be periodically parsed and key information ingested into this SQLite DB for more complex querying and analysis. Or, critical events can be logged to both simultaneously.
*   **Level 3 (Long-Term - Centralized Logging Platform - Conceptual):**
    *   For scaled real-world operations, integration with platforms like Elasticsearch/Logstash/Kibana (ELK Stack), Grafana Loki, or cloud-native solutions (AWS CloudWatch Logs, Google Cloud Logging) would be considered. This is beyond the initial scope.

## 5. Accessibility & Review

*   **Initial:** Manual review of JSONL files or querying the SQLite database using DB tools.
*   **Future GUI Integration:** A dedicated "Audit Log Explorer" or "Activity Stream" page in the Skyscope Sentinel GUI. This would allow:
    *   Filtering by `run_id`, `swarm_name`, `agent_name`, `event_type`, `status`, `timestamp` range.
    *   Searching log messages.
    *   Viewing detailed JSON for selected entries.

## 6. Agent & Swarm Responsibility for Logging

*   **Utility Logging Function:** A centralized logging utility function (e.g., `log_swarm_event(...)` in `skyscope_sentinel.utils.logging_utils` - to be created) should be developed.
    *   This function would take standardized parameters (event_type, message, details, etc.) and handle formatting to JSON and writing to the appropriate store (file and/or SQLite).
*   **Swarm Orchestrators (`run_..._swarm` functions):**
    *   Generate `run_id` at the start.
    *   Log `SWARM_START` and `SWARM_END` events, including overall status and duration.
    *   Pass `run_id` and `swarm_name` to agents they instantiate.
*   **`SkyscopeSwarmAgent` Base Class:**
    *   Could be enhanced to automatically log `AGENT_START`, `AGENT_END`.
    *   Provide a method for agents to easily call the utility logging function (e.g., `self.log_event(...)`).
*   **Tool Functions:** Should log their execution start, parameters, and success/failure/output through the agent that calls them, or the agent logs this information.
*   **LLM Interactions:** The `SkyscopeSwarmAgent` or the `swarms` library itself (if it offers hooks) should log LLM call attempts, successes, failures, model used, and snippets of prompts/responses (careful with sensitivity).

## 7. Security & Privacy in Logs

*   **Sensitive Data Masking:** Implement mechanisms to avoid logging raw sensitive data (API keys, private financial info, full PII from prompts/responses).
    *   Log API key *usage* by alias/type, not the key itself.
    *   Summarize or truncate potentially large/sensitive inputs/outputs in `details_json`.
    *   If full prompts/responses are needed for debugging, they should be logged to a separate, highly restricted debug store, not the main audit log, and only when debug mode is explicitly enabled.
*   **Access Control:** Secure access to log storage (file permissions, database credentials).

## 8. Initial Implementation Steps (Next Phase after RW1)

1.  Create `skyscope_sentinel/utils/logging_utils.py`.
2.  Implement the basic `log_swarm_event` function to write structured JSONL entries to `workspace/logs/swarm_runs/`.
3.  Integrate `log_swarm_event` calls into `run_opportunity_scouting_swarm` for `SWARM_START`, `SWARM_END`.
4.  Modify `SkyscopeSwarmAgent` to accept `run_id` and `swarm_name`, and provide a `self.log_event()` method.
5.  Add basic `AGENT_START`/`AGENT_END` logging in `SkyscopeSwarmAgent`.
6.  Instruct agents (via prompts or by modifying their `run` methods if necessary) to log key decisions or tool calls using `self.log_event()`.

This design document provides a comprehensive foundation for building an essential journaling and auditing system for Skyscope Sentinel Intelligence.
