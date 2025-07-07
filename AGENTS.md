# AGENTS.md - Guidance for AI Agents

This document provides guidance and context for AI agents (like Jules, Aider, etc.) working on the Skyscope Sentinel Intelligence codebase. Its purpose is to help agents understand the project's architecture, key components, and conventions.

## Core Principles for Agents

*   **Understand the Goal:** The primary objective of Skyscope Sentinel Intelligence is to become a multi-agent powered enterprise platform capable of autonomously generating high income through various online means, starting with no initial funds. **Update:** As of Phase RW1, the platform is authorized for real-world operations with legal and liability coverage, shifting focus from simulation to actual income generation.
*   **Understand the Goal:** The primary objective of Skyscope Sentinel Intelligence is to become a multi-agent powered enterprise platform capable of autonomously generating high income through various online means, starting with no initial funds.
*   **Local First (Ollama):** Prioritize using local Ollama models for LLM tasks to ensure privacy, reduce costs, and enable offline capabilities where possible. User-provided API keys for other services (OpenAI, Serper, E2B) are supported for enhanced capabilities.
*   **Modularity:** Strive for modular design in agent and tool development.
*   **Swarm Intelligence:** The project is increasingly leveraging the `kyegomez/swarms` framework for orchestrating multiple agents. Familiarize yourself with its concepts (Agents, Swarms, Workflows, Tools).
*   **User Experience:** The application has a PySide6 GUI. Changes should be reflected appropriately in the GUI, and tasks should run asynchronously to keep the UI responsive.
*   **Workspace:** Agents should use the `./workspace/` directory for file-based outputs (reports, generated code, etc.). Configuration for this is in `config.py`.
*   **Real-World Operations**: All development from Phase RW1 onwards should assume real-world application. This means heightened attention to security, error handling, data integrity, financial transaction safety (even if simulated initially), and auditable logging.

## Real-World Operations Directive & Pilot Strategy (Phase RW1 Onwards)

Skyscope Sentinel Intelligence has received full legal authorization to operate as a real-world autonomous enterprise, including appropriate liability and indemnity coverage. This marks a significant shift from simulated activities to actual business operations aimed at generating real income.

### Chosen Pilot Income Stream

*   **Pilot Activity:** "Content Creation for Skyscope Sentinel's Own Platform(s) & Initial Service Offering Preparation."
*   **Rationale:**
    *   **Low Barrier to Entry:** Allows immediate real-world application without complex external client acquisition or payment gateway integrations for the very first step.
    *   **Leverages Existing Strengths:** Utilizes the `ContentGenerationSwarm` which is already capable of producing articles, blog posts, and other content formats with SEO considerations.
    *   **Builds Tangible Assets:** Creates high-quality content that can be used for Skyscope Sentinel's own website/blog, establishing an online presence, demonstrating capabilities, and building a portfolio.
    *   **Foundation for Future Services:** The generated content and refined swarm serve as a direct foundation for offering "AI-Powered Content Creation as a Service" to real clients.
    *   **Path to Other Monetization:** High-quality content can later be used for affiliate marketing or to drive traffic to other Skyscope ventures.
*   *   **Initial Pilot Tasks:**
    1.  Define 2-3 initial content pieces (e.g., blog posts) relevant to Skyscope Sentinel's brand and expertise (e.g., "The Future of Autonomous Businesses," "Leveraging Local LLMs for Enterprise").
    2.  Utilize the `ContentGenerationSwarm` (with prompts refined for this pilot) to produce these articles.
    3.  User (Casey Topojani) reviews and approves/edits content for quality and brand alignment.
    4.  (Future) Publish content on a Skyscope platform.
    5.  (Concurrent/Future) Use generated content as samples to draft service descriptions and proposals for offering content creation services.

### Secure Credential & Financial Information Management Plan

Given the transition to real-world operations, managing sensitive data securely is paramount.

**I. Types of Sensitive Information:**

1.  **API Keys:** For freelance platforms, social media, payment gateways, crypto exchanges, third-party services.
2.  **Financial Account Details:**
    *   **Skyscope's Receiving Addresses:** BTC (`1Fr2rKPrZGokKQoWfBGSMLBmDxCe7jdzQa`), other user-provided crypto addresses, bank details for fiat.
    *   **Skyscope's Platform Credentials:** Logins for payment platforms (PayPal, Stripe), exchange API keys (if Skyscope trades directly).
3.  **User Configuration Data:** User's preferred payout addresses, other sensitive settings from the GUI.

**II. Current Method & Limitations:**

*   Currently, API keys are managed via `.env` files (dev) or GUI settings (`SettingsManager` -> `QSettings`, potentially plain text).
*   This is insufficient for real-world financial credentials or critical API keys due to security risks.

**III. Proposed Credential Management Strategy (Tiered Approach):**

*   **Tier 1: Non-Critical/Less Sensitive API Keys** (e.g., search APIs, free-tier utility APIs):
    *   **Storage:** Continue with `.env` (dev) and GUI Settings (`SettingsManager` with user education on securing their local settings).
    *   **Access:** Via `global_config` or `os.getenv()`.

*   **Tier 2: Important Platform API Keys** (e.g., social media posting, non-financial platform interactions requiring login):
    *   **Storage (Short-Term):** User provides via GUI. `SettingsManager` MUST implement encryption for these values before saving (e.g., using a user-provided master password at runtime for decryption/encryption, or leveraging OS keychain/secret service if feasible libraries are integrated).
    *   **Storage (Mid-Term):** Prioritize OS Keychain/Secret Service integration (e.g., via `keyring` library) for storing these keys.
    *   **Access:** Backend processes need a secure way to access decrypted keys, ideally without exposing a master password directly to all agent code.

*   **Tier 3: Highly Sensitive Financial Credentials** (Private Keys, Exchange API Keys with withdrawal/trade rights, Payment Gateway Keys for direct processing):
    *   **Principle: Zero Direct Agent Access.** AI agents should NEVER directly handle or store private keys or credentials that can unilaterally move significant funds or alter financial account security.
    *   **Storage for Skyscope's Operational Keys (Long-Term):** Dedicated Secrets Manager (e.g., HashiCorp Vault, Doppler, cloud provider's secrets manager). The core Skyscope backend (not individual agents) would authenticate to this service to fetch credentials *just-in-time* for specific, authorized operations, ideally performed by a dedicated Transaction Orchestration Service (TOS).
    *   **Storage for User's Public Payout Addresses:** These are public keys for *receiving* funds (e.g., BTC `1Fr2rKPrZGokKQoWfBGSMLBmDxCe7jdzQa`). Store via GUI "Financials" tab, managed by `SettingsManager`. Encryption is good for privacy.
    *   **Access by Agents (for using public addresses or *triggering* actions via TOS):**
    *   **Receiving Payments:** Agents are instructed to use/display the configured public receiving addresses.
    *   **Initiating Payments/Trades (Highly Controlled):** Agents generate a *request* (structured data for a payment/trade). This request goes to the secure TOS. The TOS validates, applies rules/limits, potentially requires Human-in-the-Loop (HITL) approval via GUI/notification for significant actions, and then uses credentials from the Secrets Manager to execute the transaction. Agents only get status updates, not direct access to keys.

**IV. Iterative Implementation Plan for Credential Management:**

1.  **RW1-RW2 (Short-Term):**
    *   Implement basic encryption in `SettingsManager` for API keys entered via GUI.
    *   Continue storing public payout addresses (like the BTC address) via GUI settings (with this new encryption).
    *   Strictly enforce "no direct agent handling of private keys" in all new code.
2.  **Mid-Term (Post-Pilot Success & Revenue):**
    *   Investigate and implement OS Keychain/Secret Service integration (`keyring` library) for Tier 2 API keys.
    *   Design and stub out the API for the Transaction Orchestration Service (TOS). Implement a version that logs requests and requires manual execution by the user.
3.  **Long-Term (Scaling Real Operations):**
    *   Integrate a full Secrets Manager (e.g., HashiCorp Vault) for Tier 3 and critical Tier 2 credentials.
    *   Develop a full Transaction Orchestration Service (TOS) with robust validation, rule engine, and HITL approval mechanisms.

### Payment Reception Plan

This plan outlines how Skyscope Sentinel Intelligence will handle incoming payments, initially focusing on the user-provided BTC address and conceptualizing other methods.

**I. Core Principles:**

1.  **User Control:** The user (Casey Topojani) maintains ultimate control over all financial accounts and wallets.
2.  **Transparency:** The system will aim to provide clear visibility into payment statuses (initially manual, later automated where possible).
3.  **Security:** Secure handling of any payment-related information is critical. Public addresses are less sensitive than private keys or platform logins.
4.  **Automation:** Agents will automate the generation of payment instructions and potentially some aspects of status tracking in the future.
5.  **Simplicity First:** Start with straightforward methods, adding complexity as required.

**II. Bitcoin (BTC) Payment Reception:**

*   **Primary Receiving Address:** `1Fr2rKPrZGokKQoWfBGSMLBmDxCe7jdzQa` (as provided by user).
*   **Storage:** This address will be configured via the GUI's "Financials" settings tab and managed by `SettingsManager` (to be encrypted as per the Credential Management Plan).
*   **Agent Usage:**
    *   **Service Invoicing:** If agents facilitate a service for which BTC payment is agreed:
    *   A designated agent/swarm (e.g., future "BillingAgent") will generate an invoice or payment instruction.
    *   This instruction will include the Skyscope BTC receiving address.
    *   It may also include a USD/AUD equivalent amount, calculated using a crypto price feed tool at the time of invoicing (tool to be developed).
    *   The prepared invoice/instruction will be saved to the workspace or presented to the user (Casey) for manual dispatch to the client. **Agents will not directly send financial requests to external parties without explicit HITL approval in early stages.**
    *   **Affiliate/Ad Revenue:** For platforms paying out in BTC, the user will configure them directly with the Skyscope BTC address. Agents might later be tasked with monitoring these platforms for earnings reports.
*   **Payment Confirmation:**
    *   **Initial:** Manual confirmation by the user (Casey) via block explorer or personal wallet.
    *   **Future:** Potential integration of a block explorer API tool for agents to query transaction status for the specific address.

**III. Other Payment Methods (Conceptual for Pilot - Content Creation Service):**

*   **PayPal:**
    *   **Setup:** User provides their PayPal.Me link or PayPal email address via encrypted GUI "Financials" settings.
    *   **Agent Usage:** Agents include PayPal details in payment instructions prepared for the user to send.
    *   **Confirmation:** Manual via PayPal notifications to the user.
*   **Stripe (for more formal service offerings):**
    *   **Setup:** Requires a user-managed Stripe account.
    *   **Agent Usage (Advanced):** Agents generate a payment request. This request goes to the TOS, which uses Stripe API keys (from Secrets Manager) to create a Payment Link. Agent includes this link in communications prepared for the user.
    *   **Confirmation:** Stripe webhooks to a secure backend, updating an internal ledger.
*   **Direct Bank Transfer (OSKO/PayID - Australia):**
    *   **Setup:** User provides PayID or BSB/Account details via encrypted GUI "Financials" settings.
    *   **Agent Usage:** Agents include these details in payment instructions.
    *   **Confirmation:** Manual by user.

**IV. Key Considerations for Payment Systems:**

1.  **Currency Conversion Tool:** Essential for services priced in fiat but payable in crypto. (To be developed).
2.  **Invoice/Order Tracking System:** Needed for managing services and payments. (To be developed).
3.  **User Notification Workflow:** To inform user of pending actions/payments.
4.  **Record Keeping for Taxation:** Log all real received payments for business accounting.

This plan will evolve as real-world operations commence.

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
    3.  **`AnalysisAgent`**: Analyzes gathered information. Enhanced to focus critically on zero-cost startup strategies and provide a detailed 8-point analysis for each opportunity, including actionable first steps for AI agents. Is now equipped with a RAG tool (`get_contextual_information_for_topic`) to query past reports from ChromaDB.
    4.  **`ReportingAgent`**: Consolidates analysis into a structured Markdown report. Enhanced to follow a specific, detailed Markdown template for clarity and professionalism.
*   **Output**: Reports are saved in the `workspace/opportunity_reports/` directory. These reports are also automatically added to a ChromaDB vector store for RAG.
*   **GUI Integration**: Triggered from the "Opportunity Research" tab. Users can select "Swarm Opportunity Scouting" mode and configure max search results for DuckDuckGo. Generated Markdown reports are now rendered directly in the GUI.
*   **RAG Capability**: The `AnalysisAgent`'s prompt guides it to use its RAG tool. Logging is in place to help verify activation.
    3.  **`AnalysisAgent`**: Analyzes gathered information. Enhanced to focus critically on zero-cost startup strategies and provide a detailed 8-point analysis for each opportunity, including actionable first steps for AI agents.
    4.  **`ReportingAgent`**: Consolidates analysis into a structured Markdown report. Enhanced to follow a specific, detailed Markdown template for clarity and professionalism.
*   **Output**: Reports are saved in the `workspace/opportunity_reports/` directory. These reports are also automatically added to a ChromaDB vector store for RAG.
*   **GUI Integration**: Triggered from the "Opportunity Research" tab. Users can select "Swarm Opportunity Scouting" mode and configure max search results for DuckDuckGo. Generated Markdown reports are now rendered directly in the GUI.
*   **RAG Capability**: The `AnalysisAgent` is now equipped with a tool (`get_contextual_information_for_topic`) to query the ChromaDB vector store of past reports, and its prompt guides it to use this context.

### 2. Content Generation Swarm

*   **Purpose**: To autonomously create various types of textual content (blog posts, tweet threads, short articles) based on a user-provided topic, target audience, and tone. Includes SEO optimization.
*   **Location**: `skyscope_sentinel/swarms_integration/content_generation_swarm.py`
*   **Workflow**: Utilizes `swarms.SequentialWorkflow`.
    1.  **`ContentStrategistAgent`**: Defines content strategy, outline, keywords (using a search tool), and tailors the brief for the specific content type.
    2.  **`DraftWriterAgent`**: Generates the first draft, adapting format and style to the content type (e.g., numbered tweets for a thread).
    3.  **`ReviewEditorAgent`**: Reviews and edits for quality, coherence, grammar, and adherence to the brief and content type.
    4.  **`SeoOptimizerAgent`**: Optimizes the edited content for SEO, generates meta descriptions, SEO titles, and relevant hashtags (for tweet threads).
*   **Output**: SEO-optimized content, along with metadata (SEO title, meta description, hashtags), saved to a Markdown file in `workspace/generated_content/`.
*   **GUI Integration**: Triggered from the "Content Studio" tab, allowing users to specify topic, audience, content type, and tone. Generated content (with metadata) is rendered in the GUI.

### 3. Freelance Task Simulation Swarm

*   **Purpose**: To simulate identifying suitable freelance tasks from an opportunity report and drafting initial bid proposals. (Transitioning to identify real tasks for manual submission by user).
*   **Location**: `skyscope_sentinel/swarms_integration/freelance_simulation_swarm.py`
*   **Workflow**:
    1.  `TaskProspectorAgent`: Parses an opportunity report, uses a search tool for conceptual task examples, and formulates 1-2 mock (soon to be real) freelance task descriptions.
*   **Purpose**: To simulate identifying suitable freelance tasks from an opportunity report and drafting initial bid proposals. (Currently a simulation, does not interact with live platforms).
*   **Location**: `skyscope_sentinel/swarms_integration/freelance_simulation_swarm.py`
*   **Workflow**:
    1.  `TaskProspectorAgent`: Parses an opportunity report, uses a search tool for conceptual task examples, and formulates 1-2 mock freelance task descriptions.
    2.  For each mock task:
       *   `BidStrategistAgent`: Devises a bid strategy (fulfillment plan, selling points, conceptual pricing).
       *   `ProposalDraftingAgent`: Drafts a proposal based on the task and strategy.
*   **Output**: Drafted proposals are saved as Markdown files in `workspace/simulated_proposals/`.
*   **GUI Integration**: Triggered from the "Freelance Hub" tab. User provides a path to an opportunity report. Paths to generated proposals are displayed.
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
*   **RAG System**: Be aware of the ChromaDB vector store for opportunity reports. Agents like `AnalysisAgent` are designed to query this store for contextual information.
*   **Real-World Implications**: With the shift to real-world operations, all new features must be designed with robust security, data integrity, financial safety, and comprehensive audit logging in mind.

This document will be updated as the project evolves. If you make significant architectural changes or add new core agent systems, please update this file accordingly.
*   **RAG System**: Be aware of the ChromaDB vector store for opportunity reports. Future agents might query this store for contextual information.

This document will be updated as the project evolves. If you make significant architectural changes or add new core agent systems, please update this file accordingly.
*   **Tools**: Utility functions for search, browsing, file I/O, code execution are in `skyscope_sentinel/utils/`.
*   **Identity Management**: `skyscope_sentinel/agent_identity.py` (generates identities for agents).

## Working with an Agentic Environment

*   **Tool Usage**: When implementing new agent capabilities, prefer creating reusable tool functions (like those in `utils/`) and providing them to agents rather than having agents generate complex code for common tasks like web searches or file I/O directly.
*   **Error Handling**: Ensure robust error handling, especially for operations involving external APIs, file system access, or web interactions.
*   **Asynchronous Operations**: GUI interactions that trigger long-running agent tasks MUST be run in separate threads (e.g., using `AsyncRunnerThread` in `main.py`) to keep the UI responsive.

This document will be updated as the project evolves. 
