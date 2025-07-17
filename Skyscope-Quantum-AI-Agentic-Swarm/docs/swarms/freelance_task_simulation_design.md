# Freelance Task Simulation Swarm - Design Document

**Phase E1 - Design Only**

## 1. Objective

The Freelance Task Simulation Swarm aims to simulate the process of identifying suitable freelance tasks based on opportunities highlighted by the Opportunity Scouting Swarm, and then drafting initial proposals or expressions of interest for these tasks. This swarm focuses on tasks that can be predominantly executed by AI agents (e.g., content writing, data analysis, simple coding, research).

**Note:** This is a *simulation*. It will not actually interact with live freelance platforms or submit bids initially. Its output will be drafted proposals/applications saved to the workspace.

## 2. Key Agent Roles & Responsibilities

A `SequentialWorkflow` is likely appropriate for the initial simulation.

*   **`TaskProspectorAgent (SkyscopeSwarmAgent)`**:
    *   **Input**: An opportunity report (Markdown) from the Opportunity Scouting Swarm, particularly the "Actionable First Steps" and "Key AI-Leveraged Activities" sections.
    *   **Responsibilities**:
        *   Parse the input report to understand the nature of the opportunity.
        *   Identify specific, simple freelance-style tasks that align with the opportunity and Skyscope's AI capabilities (content creation, data entry/analysis, research, basic coding).
        *   (Simulated) Search for examples of such tasks on conceptual freelance platforms or job boards (using a search tool or its internal knowledge).
        *   Formulate 1-2 concrete, mock "freelance task descriptions" based on its findings. These descriptions should be what an AI agent could bid on.
    *   **Output**: A list of 1-2 mock freelance task descriptions (e.g., "Need 5 blog posts about Topic X, 500 words each", "Looking for data entry from 10 PDFs to CSV").
    *   **Tools**: LLM for reasoning and parsing, potentially a search tool for conceptual task examples.

*   **`BidStrategistAgent (SkyscopeSwarmAgent)`**:
    *   **Input**: A mock freelance task description from `TaskProspectorAgent`.
    *   **Responsibilities**:
        *   Analyze the task description to determine the core requirements.
        *   Define a strategy for how Skyscope's AI agents could fulfill this task efficiently and with high quality.
        *   Outline key selling points emphasizing AI capabilities (speed, consistency, data-driven insights, etc.).
        *   (Simulated) Determine a competitive "bid" or pricing strategy (e.g., per hour, per project – initially, could be qualitative like "low cost, high value"). This is highly conceptual for simulation.
    *   **Output**: A bid strategy document or structured data including requirements, fulfillment plan, selling points, and conceptual pricing.
    *   **Tools**: LLM for strategic reasoning.

*   **`ProposalDraftingAgent (SkyscopeSwarmAgent)`**:
    *   **Input**: Mock freelance task description and the bid strategy from `BidStrategistAgent`.
    *   **Responsibilities**:
        *   Draft a compelling proposal or expression of interest tailored to the mock task description.
        *   Highlight Skyscope's AI capabilities and the proposed fulfillment plan.
        *   Maintain a professional and persuasive tone.
        *   Incorporate the conceptual bid/pricing if provided.
    *   **Output**: A drafted proposal text (e.g., Markdown or plain text).
    *   **Tools**: LLM for content generation.

*   **`ProposalReviewAgent (SkyscopeSwarmAgent)` (Optional):**
    *   **Input**: Drafted proposal from `ProposalDraftingAgent`.
    *   **Responsibilities**: Review the proposal for clarity, grammar, persuasiveness, and alignment with the bid strategy and task description.
    *   **Output**: Finalized or annotated proposal.
    *   **Tools**: LLM for review.

## 3. Workflow

*   **Simplified Sequential Workflow:**
    1.  Input: An opportunity report (or relevant sections).
    2.  `TaskProspectorAgent` identifies/creates mock tasks.
    3.  *For each mock task:*
        a.  `BidStrategistAgent` devises a strategy.
        b.  `ProposalDraftingAgent` writes a proposal.
        c.  (`ProposalReviewAgent` reviews - optional).
    4.  Output: Collection of drafted proposals saved to workspace.

## 4. Required Tools (Beyond LLM)

*   **Search Tool**: For `TaskProspectorAgent` to find conceptual examples of freelance tasks.
*   **File I/O Tool**: To save the generated proposals (likely handled by the main swarm runner function).

## 5. LLM Model Considerations

*   **`TaskProspectorAgent`**: Needs good parsing, understanding of AI capabilities, and some creativity.
*   **`BidStrategistAgent`**: Requires strategic thinking and understanding of value propositions.
*   **`ProposalDraftingAgent`**: Needs strong persuasive writing skills.

Local Ollama models will be prioritized.

## 6. Data Flow & Output

*   **Input**: Opportunity report from the scouting swarm.
*   **Intermediate Data**: Mock task descriptions, bid strategies.
*   **Final Output**: A set of drafted proposals/applications for the simulated freelance tasks, saved to a designated subdirectory in the workspace (e.g., `workspace/simulated_proposals/`).

## 7. Next Steps (Beyond Simulation)

*   **Real Platform Interaction (Future - Complex):** Adapting agents to interact with actual freelance platform APIs or UIs (via browser automation) would be a significant next step but is out of scope for initial simulation.
*   **Skill Matching:** More advanced skill matching between task requirements and available Skyscope AI agent capabilities.
*   **Automated Portfolio Generation:** AI agents could generate sample work or case studies to include with proposals.

This design focuses on simulating the initial stages of freelance task acquisition, providing a basis for understanding how AI agents might approach this income stream.
