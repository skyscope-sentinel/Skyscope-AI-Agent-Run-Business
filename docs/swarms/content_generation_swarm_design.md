# Content Generation Swarm - Design Document

**Phase E1 - Design Only**

## 1. Objective

The Content Generation Swarm will be responsible for autonomously creating various types of textual content based on a given topic, research summary, or specific instructions. Initial focus will be on generating blog posts or articles, with future extensions for social media updates, scripts, etc. The content should be coherent, informative, engaging, and optimized for SEO where applicable.

## 2. Key Agent Roles & Responsibilities

The swarm will be structured, likely as a `SequentialWorkflow` or a more complex `GraphWorkflow` / `SwarmRouter` for advanced versions.

*   **`ContentStrategistAgent (SkyscopeSwarmAgent)`**:
    *   **Input**: Main topic/keyword, target audience, desired content type (e.g., blog post, tweet thread), desired tone/style.
    *   **Responsibilities**:
        *   Refine the topic and identify key sub-topics or angles to cover.
        *   Define the overall structure/outline of the content.
        *   Identify relevant keywords for SEO (may use a search tool).
        *   Specify the target length and any specific formatting requirements.
    *   **Output**: A detailed content brief (outline, keywords, target audience, tone, length, structure) for the `DraftWriterAgent`.
    *   **Tools**: Search tool (for keyword research), LLM for reasoning.

*   **`DraftWriterAgent (SkyscopeSwarmAgent)`**:
    *   **Input**: Content brief from `ContentStrategistAgent`.
    *   **Responsibilities**:
        *   Generate a first draft of the content according to the brief.
        *   Ensure factual accuracy (may require limited use of search tool if explicitly allowed for fact-checking specific points from the brief, or rely on provided research summaries).
        *   Maintain the specified tone and style.
    *   **Output**: Raw first draft of the content.
    *   **Tools**: LLM for generation, potentially a search tool (limited use for fact-checking if allowed by strategist).

*   **`ReviewEditorAgent (SkyscopeSwarmAgent)`**:
    *   **Input**: Raw draft from `DraftWriterAgent`, original content brief.
    *   **Responsibilities**:
        *   Review the draft for clarity, coherence, grammar, spelling, and style.
        *   Ensure the draft adheres to the original brief (outline, keywords, tone).
        *   Fact-check critical claims if necessary (again, limited tool use or based on provided context).
        *   Suggest improvements or directly edit the content.
    *   **Output**: Edited and improved version of the content.
    *   **Tools**: LLM for review and editing. Potentially a grammar/style checking API/tool (or use advanced LLM capabilities).

*   **`SeoOptimizerAgent (SkyscopeSwarmAgent)` (Optional/Advanced):**
    *   **Input**: Edited content from `ReviewEditorAgent`, target keywords from brief.
    *   **Responsibilities**:
        *   Analyze the content for SEO best practices (keyword density, headings, readability, internal/external linking opportunities if context is available).
        *   Suggest or apply SEO improvements.
        *   Generate meta descriptions, SEO titles.
    *   **Output**: SEO-optimized content.
    *   **Tools**: LLM for analysis, potentially SEO analysis tools/APIs if integrated.

*   **`FinalFormatterAgent (SkyscopeSwarmAgent)` (Optional):**
    *   **Input**: SEO-optimized (or edited) content.
    *   **Responsibilities**:
        *   Format the content into the final desired output (e.g., clean Markdown, HTML snippet).
        *   Ensure consistent formatting.
    *   **Output**: Final, publish-ready content.
    *   **Tools**: LLM for formatting.

## 3. Workflow

*   **Initial Simple Workflow (Sequential):**
    `ContentStrategistAgent` -> `DraftWriterAgent` -> `ReviewEditorAgent` -> `FinalFormatterAgent` (if used)
    *   The output of one agent directly feeds into the next.

*   **Advanced Workflow (Potential for `GraphWorkflow` or custom loop with `SwarmRouter`):**
    *   `ContentStrategistAgent` creates brief.
    *   `DraftWriterAgent` produces draft.
    *   `ReviewEditorAgent` reviews. If major revisions needed, it could send feedback *back* to `DraftWriterAgent` for a re-draft (iterative refinement loop).
    *   Once `ReviewEditorAgent` approves, it goes to `SeoOptimizerAgent`.
    *   Then to `FinalFormatterAgent`.

## 4. Required Tools (Beyond LLM)

*   **Search Tool**: (e.g., `duckduckgo_search_function`, `serper_search_function`) - Primarily for `ContentStrategistAgent` (keyword research) and potentially limited use by `DraftWriterAgent` / `ReviewEditorAgent` for fact-checking.
*   **(Future) Grammar/Style Checker API/Tool**: Could enhance `ReviewEditorAgent`.
*   **(Future) SEO Analysis Tool/API**: For `SeoOptimizerAgent`.

## 5. LLM Model Considerations

*   **`ContentStrategistAgent`**: Needs good reasoning and planning abilities (e.g., Mistral, Llama 3, GPT-3.5/4 if API used).
*   **`DraftWriterAgent`**: Needs strong creative writing and instruction-following capabilities (e.g., Mistral, Llama 3, GPT-3.5/4). Model choice might vary based on desired content type (e.g., a model fine-tuned for creative writing vs. technical writing).
*   **`ReviewEditorAgent`**: Needs good analytical, grammar, and editing skills.
*   **`SeoOptimizerAgent`**: Needs understanding of SEO concepts.

Using local Ollama models will be prioritized. The specific model (e.g., `ollama/mistral`, `ollama/llama3`) for each agent can be configured via `SkyscopeSwarmAgent`.

## 6. Data Flow & Output

*   **Input**: A topic, keywords, or a detailed request.
*   **Intermediate Data**: Content briefs, raw drafts, edited drafts. These could be passed as strings or structured data (e.g., JSON) between agents.
*   **Final Output**: Publish-ready content in the desired format (e.g., Markdown text). This output would likely be saved to the workspace or passed to another system/agent for publishing.

## 7. Integration Points

*   Could be triggered by the Opportunity Scouting Swarm (e.g., to write content about an identified opportunity).
*   Could be triggered manually via GUI for specific content requests.
*   Output could feed into a future "Publishing Swarm" or social media management tools.

This design provides a starting point for implementing the Content Generation Swarm. Initial implementation will focus on the simple sequential workflow and core agent roles.
