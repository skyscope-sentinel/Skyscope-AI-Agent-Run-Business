# Skyscope Sentinel Intelligence – AI Agentic Swarm

![Skyscope Sentinel Banner](https://raw.githubusercontent.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/main/images/swarmslogobanner.png)

---

## 1 • Introduction  
# 🚀 Skyscope AI Agent Business Automation System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![macOS](https://img.shields.io/badge/macOS-compatible-green.svg)](https://www.apple.com/macos/)
[![Ollama](https://img.shields.io/badge/Ollama-compatible-orange.svg)](https://ollama.ai/)

> **The Future of Autonomous Business Operations is Here**

A cutting-edge, production-ready multi-agent swarm framework that orchestrates autonomous business operations through intelligent AI agents. Powered by advanced Ollama integration, sophisticated orchestration patterns, and built specifically for macOS with Apple Silicon optimizations.

![Skyscope Banner](https://raw.githubusercontent.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/main/images/swarmslogobanner.png)

---

## 🎯 **Mission Statement**

Transform business operations through autonomous multi-agent systems that research, create, deploy, and manage entire business workflows without human intervention. From market research to content creation, from freelance operations to web deployment—all powered by local LLMs and advanced swarm intelligence.

---

## ⚡ **Quick Start**

```bash
# Clone the repository
git clone https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business.git
cd Skyscope-AI-Agent-Run-Business

# Run macOS setup (automatically installs Ollama, dependencies, and optimizes for Apple Silicon)
python setup_macos.py

# Start the swarm orchestrator
python swarm_orchestrator.py
```

---

## 🌟 **Key Features**

### 🔮 **Advanced Multi-Agent Swarm Orchestration**
- **6 Orchestration Modes**: Hierarchical, Collaborative, Sequential, Parallel, Swarm Intelligence, Consensus
- **Intelligent Supervisor Agent** with continuous learning and crisis management
- **Dynamic Agent Selection** with performance-based routing and load balancing
- **Real-time Performance Monitoring** with predictive analytics and trend analysis

### 🧠 **Ollama-Powered Local LLM Integration**
- **15+ Supported Models**: Llama 3, Mistral, Gemma, CodeLlama, Neural Chat, and more
- **macOS Metal GPU Acceleration** for Apple Silicon (M1/M2/M3) optimization
- **Intelligent Model Selection** based on task complexity and requirements
- **Automatic Fallback Systems** with multi-model redundancy

### 🤖 **Specialized Business Automation Agents**

#### 📊 **Research & Development Agent**
- Multi-source research capabilities (web, academic, patents, news, social media)
- Automated market analysis and competitive intelligence
- Technology trend monitoring and innovation tracking
- Research report generation with citations and analysis

#### ✍️ **Creative Content Agent**
- AI-driven content generation for 12+ formats and platforms
- Multi-language support with cultural adaptation
- SEO optimization and performance tracking
- Brand voice consistency and style adaptation

#### 💼 **Freelance Operations Agent**
- Complete client relationship management (CRM)
- Automated project tracking and milestone management
- Invoice generation and payment processing
- Performance analytics and business intelligence

#### 🌐 **Web Deployment Agent**
- Multi-platform deployment (Vercel, Netlify, AWS, Heroku, DigitalOcean)
- Infrastructure as Code with automated provisioning
- SSL certificate management and domain configuration
- Performance monitoring and scaling optimization

### 🔧 **Technical Excellence**
- **Asynchronous Processing** with concurrent task execution
- **Configuration-Driven Architecture** with YAML-based setup
- **Docker Containerization** ready for production deployment
- **Comprehensive API** with REST endpoints and WebSocket support
- **Security-First Design** with encrypted communications and secure storage

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                        │
│              (Learning & Crisis Management)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                SWARM ORCHESTRATOR                          │
│         (6 Orchestration Modes Available)                 │
└─┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┘
  │         │         │         │         │         │
┌─▼─┐   ┌─▼─┐   ┌───▼───┐   ┌─▼─┐   ┌───▼───┐   ┌─▼─┐
│R&D│   │CTN│   │FREELANCE│ │WEB│   │MONITOR│   │...│
│AGT│   │AGT│   │  AGENT  │ │AGT│   │ AGENT │   │AGT│
└───┘   └───┘   └───────┘   └───┘   └───────┘   └───┘
  │       │         │         │         │         │
  └───────┴─────────┴─────────┴─────────┴─────────┘
                      │
            ┌─────────▼─────────┐
            │  OLLAMA ENHANCED  │
            │   LLM INTEGRATION │
            └───────────────────┘
```

---

## 🚀 **Orchestration Modes**

### 1. 🏢 **Hierarchical Mode**
- **Use Case**: Complex business operations requiring oversight
- **Pattern**: Supervisor → Team Leads → Specialized Agents
- **Best For**: Large-scale projects, compliance-heavy workflows

### 2. 🤝 **Collaborative Mode**
- **Use Case**: Creative projects requiring peer coordination
- **Pattern**: Agents work as equals with shared decision-making
- **Best For**: Content creation, brainstorming, innovation projects

### 3. 🔄 **Sequential Mode**
- **Use Case**: Linear workflows with clear dependencies
- **Pattern**: Agent A → Agent B → Agent C → Output
- **Best For**: Research pipelines, content production workflows

### 4. ⚡ **Parallel Mode**
- **Use Case**: Independent tasks requiring concurrent execution
- **Pattern**: Multiple agents working simultaneously
- **Best For**: Data processing, multi-platform deployments

### 5. 🧠 **Swarm Intelligence Mode**
- **Use Case**: Complex problem-solving requiring emergent behavior
- **Pattern**: Collective intelligence with adaptive learning
- **Best For**: Market analysis, optimization problems

### 6. 🗳️ **Consensus Mode**
- **Use Case**: Critical decisions requiring validation
- **Pattern**: Democratic voting with weighted opinions
- **Best For**: Strategic decisions, quality assurance

---

## 💼 **Business Use Cases**

### 🔬 **Automated Research & Development**
```python
# Example: Comprehensive Market Research
research_team = SwarmOrchestrator(
    mode="SWARM_INTELLIGENCE",
    agents=[
        ResearchAgent(sources=["web", "academic", "patents"]),
        AnalysisAgent(focus="competitive_intelligence"),
        ReportAgent(format="executive_summary")
    ]
)

result = await research_team.execute(
    task="Analyze emerging AI trends in healthcare for Q1 2024"
)
```

### 📝 **AI-Driven Content Campaigns**
```python
# Example: Multi-Platform Content Generation
content_campaign = SwarmOrchestrator(
    mode="COLLABORATIVE",
    agents=[
        ContentAgent(platforms=["blog", "social", "email"]),
        SEOAgent(keywords=["AI automation", "business intelligence"]),
        DesignAgent(brand_guidelines="modern_tech")
    ]
)

campaign = await content_campaign.execute(
    task="Create comprehensive product launch campaign"
)
```

### 💰 **Freelance Business Automation**
```python
# Example: Complete Client Project Management
freelance_ops = SwarmOrchestrator(
    mode="HIERARCHICAL",
    agents=[
        ClientAgent(crm_integration=True),
        ProjectAgent(tracking="agile"),
        InvoiceAgent(payment_gateway="stripe"),
        AnalyticsAgent(metrics=["revenue", "client_satisfaction"])
    ]
)

business_report = await freelance_ops.execute(
    task="Manage Q4 client portfolio and generate business insights"
)
```

---

## 🛠️ **Installation & Setup**

### 📋 **Prerequisites**
- **macOS 11+** (Intel or Apple Silicon)
- **Python 3.8+** 
- **4GB+ RAM** (8GB+ recommended for Apple Silicon)
- **10GB+ Storage** for models and dependencies

### ⚙️ **Automated macOS Setup**
```bash
# One-command setup for macOS
python setup_macos.py

# What this installs:
# ✅ Ollama with macOS optimizations
# ✅ Python dependencies with conflict resolution
# ✅ Apple Silicon optimizations (Metal GPU support)
# ✅ Configuration templates
# ✅ Model downloads (Llama 3, Mistral, etc.)
# ✅ Security configurations
```

### 🔧 **Manual Installation**
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download recommended models
ollama pull llama3
ollama pull mistral
ollama pull codellama

# 4. Configure the system
cp config/config_template.yaml config/config.yaml
# Edit config.yaml with your preferences

# 5. Initialize the database
python -c "from supervisor_agent import SupervisorAgent; SupervisorAgent.init_database()"
```

### 🧪 **Verification**
```bash
# Test Ollama integration
python -c "from ollama_integration_enhanced import OllamaEnhanced; print(OllamaEnhanced.test_connection())"

# Test swarm orchestrator
python -c "from swarm_orchestrator import SwarmOrchestrator; print('✅ System Ready')"

# Run health check
python health_check.py
```

---

## ⚡ **Quick Start Examples**

### 🔬 **Market Research Automation**
```python
from swarm_orchestrator import SwarmOrchestrator
from research_development_agent import ResearchAgent

# Initialize research swarm
research_swarm = SwarmOrchestrator(
    mode="SWARM_INTELLIGENCE",
    config_path="config/research_config.yaml"
)

# Add specialized research agents
research_swarm.add_agent(ResearchAgent(
    sources=["web", "academic", "news", "social"],
    expertise="market_analysis"
))

# Execute comprehensive research
results = await research_swarm.execute(
    task="Analyze the autonomous vehicle market trends for 2024",
    output_format="executive_report"
)

print(f"Research completed: {results.summary}")
print(f"Sources analyzed: {results.source_count}")
print(f"Key insights: {len(results.insights)}")
```

### 📝 **Content Generation Pipeline**
```python
from creative_content_agent import CreativeContentAgent

# Initialize content generation agent
content_agent = CreativeContentAgent(
    model="llama3",
    brand_voice="professional_tech",
    output_formats=["blog_post", "social_media", "email_campaign"]
)

# Generate multi-format content
campaign = await content_agent.generate_campaign(
    topic="AI-driven business automation",
    target_audience="small_business_owners",
    campaign_goals=["awareness", "lead_generation"]
)

print(f"Generated {len(campaign.assets)} content assets")
print(f"Platforms covered: {campaign.platforms}")
print(f"SEO score: {campaign.seo_metrics.overall_score}")
```

### 💼 **Freelance Business Management**
```python
from freelance_operations_agent import FreelanceAgent

# Initialize freelance operations
freelance_agent = FreelanceAgent(
    crm_enabled=True,
    payment_gateway="stripe",
    project_management="notion"
)

# Automate client onboarding
new_client = await freelance_agent.onboard_client(
    name="TechStartup Inc",
    project_type="web_development",
    budget_range="$10k-25k",
    timeline="3_months"
)

# Generate project proposal
proposal = await freelance_agent.generate_proposal(
    client_id=new_client.id,
    requirements=new_client.requirements,
    include_timeline=True,
    include_pricing=True
)

print(f"Client onboarded: {new_client.name}")
print(f"Proposal generated: {proposal.filename}")
print(f"Estimated value: ${proposal.total_value:,}")
```

---

## 🎛️ **Configuration**

### 📄 **Main Configuration (config/config.yaml)**
```yaml
# System Configuration
system:
  name: "Skyscope AI Business Automation"
  version: "1.0.0"
  environment: "production"  # development, staging, production
  
# Ollama Configuration
ollama:
  host: "localhost"
  port: 11434
  models:
    primary: "llama3"
    fallback: ["mistral", "codellama"]
    specialized:
      research: "llama3"
      content: "mistral"
      code: "codellama"
  
# Orchestration Settings
orchestration:
  default_mode: "HIERARCHICAL"
  max_concurrent_agents: 5
  timeout_seconds: 300
  retry_attempts: 3
  
# Performance Settings
performance:
  enable_gpu: true  # Metal GPU for Apple Silicon
  memory_limit_gb: 8
  cache_enabled: true
  monitoring_interval: 30
  
# Security Settings
security:
  encryption_enabled: true
  api_key_required: true
  rate_limiting: true
  audit_logging: true
```

### 🤖 **Agent-Specific Configurations**
```yaml
# Research Agent Configuration
research_agent:
  sources:
    web_search:
      enabled: true
      max_results: 20
      timeout: 30
    academic:
      enabled: true
      databases: ["pubmed", "arxiv", "ieee"]
    patents:
      enabled: true
      regions: ["US", "EU", "WO"]
  
  analysis:
    sentiment_analysis: true
    trend_detection: true
    competitive_intelligence: true
    
  output:
    formats: ["report", "summary", "json", "csv"]
    include_citations: true
    confidence_scores: true

# Content Agent Configuration  
content_agent:
  generation:
    creativity_level: 0.7
    max_length: 2000
    tone_consistency: true
    
  platforms:
    blog:
      seo_optimization: true
      readability_score: "flesch_kincaid"
    social_media:
      platforms: ["twitter", "linkedin", "facebook"]
      hashtag_generation: true
    email:
      personalization: true
      a_b_testing: true
      
  brand_guidelines:
    voice: "professional_friendly"
    avoid_words: ["cheap", "basic", "simple"]
    required_elements: ["call_to_action", "brand_mention"]
```

---

## 📊 **Monitoring & Analytics**

### 📈 **Real-Time Dashboards**
- **System Performance**: CPU, Memory, GPU utilization
- **Agent Metrics**: Task completion rates, error rates, response times
- **Business KPIs**: Revenue generated, clients served, content published
- **Model Performance**: Accuracy scores, inference times, model usage

### 🔍 **Advanced Analytics**
```python
# Performance Analytics Example
from supervisor_agent import SupervisorAgent

supervisor = SupervisorAgent()

# Get comprehensive system metrics
metrics = await supervisor.get_analytics()

print(f"System Uptime: {metrics.uptime}")
print(f"Tasks Completed: {metrics.tasks_completed}")
print(f"Success Rate: {metrics.success_rate:.2%}")
print(f"Average Response Time: {metrics.avg_response_time}ms")

# Agent-specific performance
for agent_id, stats in metrics.agent_performance.items():
    print(f"{agent_id}: {stats.efficiency_score:.2f}")
```

### 📊 **Business Intelligence**
```python
# Business Metrics Dashboard
business_metrics = await supervisor.get_business_analytics()

print(f"Revenue Generated: ${business_metrics.revenue_generated:,}")
print(f"Clients Served: {business_metrics.clients_served}")
print(f"Content Published: {business_metrics.content_published}")
print(f"Projects Completed: {business_metrics.projects_completed}")
```

---

## 🔒 **Security & Privacy**

### 🛡️ **Security Features**
- **End-to-End Encryption** for all agent communications
- **Local LLM Processing** - no data sent to external services
- **API Key Management** with secure storage and rotation
- **Rate Limiting** and DDoS protection
- **Audit Logging** for compliance and monitoring

### 🔐 **Privacy Protection**
- **Data Localization** - all processing happens on your machine
- **Configurable Data Retention** policies
- **Anonymization** options for sensitive data
- **GDPR Compliance** features for European operations

### 🚨 **Crisis Management**
```python
# Automatic crisis detection and response
crisis_scenarios = {
    "model_failure": "Switch to fallback model and alert admin",
    "high_error_rate": "Reduce agent load and investigate",
    "memory_overload": "Scale down operations and optimize",
    "security_breach": "Lock down system and alert security team"
}
```

---

## 🛠️ **Development & Customization**

### 🧩 **Creating Custom Agents**
```python
from base_agent import BaseAgent
from ollama_integration_enhanced import OllamaEnhanced

class CustomBusinessAgent(BaseAgent):
    def __init__(self, specialized_function):
        super().__init__()
        self.ollama = OllamaEnhanced()
        self.function = specialized_function
    
    async def execute_task(self, task):
        # Implement your custom business logic
        response = await self.ollama.generate(
            model="llama3",
            prompt=f"Execute {self.function}: {task}",
            context=self.get_context()
        )
        return self.process_response(response)
    
    def get_capabilities(self):
        return {
            "name": "Custom Business Agent",
            "functions": [self.function],
            "supported_modes": ["HIERARCHICAL", "COLLABORATIVE"]
        }
```

### 🔧 **Custom Orchestration Patterns**
```python
class CustomOrchestrationMode:
    def __init__(self, name, pattern):
        self.name = name
        self.pattern = pattern
    
    async def execute(self, agents, task):
        # Implement your custom orchestration logic
        results = []
        for agent in agents:
            if self.should_execute_agent(agent, task):
                result = await agent.execute_task(task)
                results.append(result)
        
        return self.combine_results(results)
```

### 🧪 **Testing Framework**
```bash
# Run comprehensive tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_ollama_integration.py
python -m pytest tests/test_swarm_orchestrator.py
python -m pytest tests/test_agents.py

# Performance benchmarks
python benchmarks/run_performance_tests.py
```

---

## 🚀 **Deployment Options**

### 💻 **Local Development**
```bash
# Development mode with hot reloading
python swarm_orchestrator.py --mode development --debug true
```

### 🐳 **Docker Deployment**
```bash
# Build container
docker build -t skyscope-ai-business .

# Run with GPU support (macOS)
docker run --gpus all -p 8000:8000 skyscope-ai-business
```

### ☁️ **Cloud Deployment**
```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: skyscope-ai-business
spec:
  replicas: 3
  selector:
    matchLabels:
      app: skyscope-ai-business
  template:
    metadata:
      labels:
        app: skyscope-ai-business
    spec:
      containers:
      - name: skyscope-ai-business
        image: skyscope-ai-business:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

---

## 📚 **API Documentation**

### 🔌 **REST API Endpoints**

#### **System Management**
```http
GET /api/v1/system/health
GET /api/v1/system/metrics
POST /api/v1/system/restart
```

#### **Orchestration API**
```http
POST /api/v1/orchestration/create
GET /api/v1/orchestration/{id}/status
POST /api/v1/orchestration/{id}/execute
DELETE /api/v1/orchestration/{id}
```

#### **Agent Management**
```http
GET /api/v1/agents
POST /api/v1/agents
GET /api/v1/agents/{id}
PUT /api/v1/agents/{id}
DELETE /api/v1/agents/{id}
POST /api/v1/agents/{id}/execute
```

### 🔌 **WebSocket API**
```javascript
// Real-time agent communication
const ws = new WebSocket('ws://localhost:8000/ws/agents');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Agent update:', data);
};

// Send task to agent
ws.send(JSON.stringify({
    'agent_id': 'research_agent_001',
    'task': 'Analyze market trends',
    'priority': 'high'
}));
```

### 📱 **Python SDK**
```python
from skyscope_ai import SkyscopeClient

# Initialize client
client = SkyscopeClient(api_key="your_api_key")

# Create orchestration
orchestration = client.create_orchestration(
    mode="SWARM_INTELLIGENCE",
    agents=["research", "content", "deployment"]
)

# Execute task
result = await client.execute_task(
    orchestration_id=orchestration.id,
    task="Launch new product campaign"
)
```

---

## 🤝 **Contributing**

### 🛠️ **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/your-username/Skyscope-AI-Agent-Run-Business.git
cd Skyscope-AI-Agent-Run-Business

# Create development environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

### 📝 **Contribution Guidelines**
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 🐛 **Bug Reports**
Please use the GitHub issue tracker to report bugs. Include:
- **System Information** (macOS version, Python version, hardware)
- **Steps to Reproduce** the issue
- **Expected vs Actual** behavior
- **Relevant Logs** and error messages

### 💡 **Feature Requests**
We welcome feature requests! Please provide:
- **Clear Description** of the proposed feature
- **Use Case** and business justification
- **Implementation Ideas** (if you have any)

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Ollama Team** for creating an exceptional local LLM platform
- **Swarms Framework** contributors for swarm intelligence inspiration
- **CrewAI Community** for multi-agent collaboration patterns
- **LangChain** for foundational AI application patterns
- **Apple** for Metal GPU acceleration and Apple Silicon optimization

---

## 🔗 **Links & Resources**

- **Documentation**: [Full Documentation](docs/)
- **Installation Guide**: [macOS Installation](INSTALLATION_MACOS.md)
- **Usage Examples**: [Examples & Tutorials](USAGE_EXAMPLES.md)
- **API Reference**: [API Documentation](API_DOCUMENTATION.md)
- **Troubleshooting**: [FAQ & Solutions](TROUBLESHOOTING_FAQ.md)
- **Community**: [Discussions](https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/discussions)

---

## 📞 **Support**

- **GitHub Issues**: [Report Bugs & Feature Requests](https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/issues)
- **Discussions**: [Community Support](https://github.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/discussions)
- **Email**: support@skyscope-sentinel.com
- **Documentation**: [Comprehensive Guides](docs/)

---

## 🌟 **Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=skyscope-sentinel/Skyscope-AI-Agent-Run-Business&type=Date)](https://star-history.com/#skyscope-sentinel/Skyscope-AI-Agent-Run-Business&Date)

---

**Built with ❤️ by the Skyscope Sentinel Team**

*Empowering businesses through autonomous AI agent orchestration*
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
