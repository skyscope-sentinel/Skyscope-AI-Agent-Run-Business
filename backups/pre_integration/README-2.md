# Skyscope Sentinel Intelligence – AI Agentic Swarm

![Skyscope Sentinel Banner](https://raw.githubusercontent.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/main/images/swarmslogobanner.png)

---

## 1 • Introduction  
**Skyscope Sentinel Intelligence** is a modern Streamlit desktop/web application that places a full-featured *multi-agent AI operating system* on your machine.  
Built on top of the Swarms framework, it orchestrates up to **20 specialised expert pipelines** to research, code, analyse, automate the browser, manipulate files, and even run quantum-inspired simulations – all through one sleek dark-themed UI.

---

## 2 • Key Features
| Category | Highlights |
|----------|------------|
| **AI-Driven Automation** | • Hierarchical & concurrent multi-agent swarms<br>• 20 configurable expert roles (Researcher, Coder, Critic, …) |
| **Model Flexibility** | • Local Ollama models (Llama 3, Mistral, Gemma, etc.)<br>• API models (OpenAI GPT-4o, Anthropic Claude-3, Google Gemini, HuggingFace) |
| **Advanced UI** | • Two-pane Chat + Pop-in Code Window<br>• Editable *System Prompt* panel<br>• Persistent **Knowledge Stack** with semantic search |
| **Browser Automation** | • One-click Chromium launch inside the app<br>• Natural-language commands: “go to…”, “click…”, “type…”, “screenshot” |
| **File Management** | • Upload / preview PDF, TXT, images, video<br>• Download artefacts directly to your *Downloads* folder |
| **Quantum AI** | • Simulated & hybrid classical-quantum computation toolkit<br>• Visualise qubit probability histograms |
| **Local Filesystem Access** | • Toggle-controlled read/write utilities with directory & extension safelists |

---

## 3 • Installation

```bash
# 1. Clone
git clone https://github.com/skyscope-sentinel/Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI.git
cd Skyscope-Quantum-AI-Agentic-Swarm-Autonomous-System-WebUI

# 2. Create env (choose one)
python -m venv venv && source venv/bin/activate        # or
conda create -n sentinel python=3.10 && conda activate sentinel

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install Playwright browsers for automation
playwright install --with-deps
```

### 3.1 Env Setup
Create a `.env` file in the project root:

``` nano .env
```

### 3.2 API Key Setup

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
HUGGINGFACE_API_KEY=...
SERPER_API_KEY=...         # for web search (optional)
BROWSERLESS_API_KEY=...    # for remote chromium (optional)
```

> Local Ollama users: ensure `ollama serve` is running on `http://localhost:11434`.


## 4 • Running the App

```
streamlit run app.py
```

The UI opens at `http://localhost:8501`.

### Quick Tour
1. **Chat** – converse with the swarm.  
2. **Code Window** – view / edit / run snippets returned by the agents.  
3. **System Prompt** – fine-tune the global persona in real time.  
4. **Knowledge Stack** – upload docs or inject chat context with one click.  
5. **Files** – manage uploads & downloads.  
6. **Quantum AI** – run simulated circuits, inspect probabilities.  
7. **Browser Automation** – type natural commands to drive Chromium.


## 5 • Architecture Overview
| Module | Role |
|--------|------|
| **`app.py`** | Streamlit frontend, routing, UI state |
| **`config.py`** | Encrypted configuration & `.env` ingestion |
| **`agent_manager.py`** | Factory for agents/swarms, task & pipeline orchestrator, in-memory vector DB |

Agents talk to models via Swarms; tasks are queued, executed in worker threads, and results stream back to the UI.


## 6 • Contributing

1. Fork ➜ create feature branch (`git checkout -b feat/awesome`)  
2. Follow existing code style (PEP-8 + type hints).  
3. Add/adjust tests where relevant.  
4. Run `pre-commit run --all-files` (format/lint).  
5. Open a PR describing **what** & **why**.

All PRs run CI checks, security scan, and require at least one approving review.

---

## 7 • License
Released under the **MIT License** – see `LICENSE` for full text.

---

## 8 • Contact

| Purpose | Contact |
|---------|---------|
| General inquiries | [admin@skyscopeglobal.net](mailto:admin@skyscopeglobal.net) |
| Issues / Bugs | GitHub Issues |

---

*Powered by Swarms ⚡ &nbsp;Built with ♥ by Miss Casey Jay Topojani and contributors.*
