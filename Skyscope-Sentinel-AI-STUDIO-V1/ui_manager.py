import streamlit as st
from pathlib import Path
import os
import json
import subprocess
import sys
from datetime import datetime

# Import managers and models from the project structure
from state_manager import StateManager, get_state_manager
from agent_manager import AgentManager
from quantum_manager import QuantumManager
from browser_automation import BrowserAutomator
from filesystem_manager import FileSystemManager
from opencore_manager import OpenCoreManager
from business_generator import BusinessGenerator
from utils import log_message, sanitize_filename

# Constants from the original app structure
MODEL_OPTIONS = {
    "Local (Ollama)": [
        "llama3:latest", 
        "mistral:latest", 
        "gemma:latest", 
        "codellama:latest",
        "phi3:latest",
        "qwen:latest"
    ],
    "API": [
        "gpt-4o",
        "claude-3-opus",
        "claude-3-sonnet",
        "gemini-pro",
        "gemini-1.5-pro",
        "anthropic.claude-3-haiku-20240307",
        "meta.llama3-70b-instruct",
        "meta.llama3-8b-instruct"
    ]
}

# ------------------------------------------------------------------ #
#  Custom CSS for a polished dark theme with subtle animations       #
# ------------------------------------------------------------------ #

CUSTOM_CSS = """
/* ---- Global Palette ---- */
:root {
  --accent: #4b5eff;
  --bg-main: #0f111a;
  --bg-panel: #1a1d29;
  --bg-card: #222638;
  --text-primary: #f0f2fa;
  --text-secondary: #9fa4c4;
}

/* ---- Core Elements ---- */
.main, .block-container  {
    background-color: var(--bg-main) !important;
    color: var(--text-primary);
}

.stTabs [data-baseweb="tab"]{
    background: var(--bg-panel);
    color: var(--text-secondary);
    border-radius: 8px 8px 0 0;
    transition: background 0.25s ease;
}
.stTabs [data-baseweb="tab"][aria-selected="true"]{
    background: var(--accent);
    color: var(--text-primary);
}

div.stButton>button{
    background: var(--accent);
    color:#fff;
    border:none;
    transition:transform .15s ease, box-shadow .15s ease;
}
div.stButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 4px 14px rgba(75,94,255,.35);
}

.stTextInput>div>input,
.stTextArea>div>textarea,
.stSelectbox>div>div>div{
    background: var(--bg-card);
    color: var(--text-primary);
    border-radius:8px;
    border:1px solid var(--bg-panel);
}

.stToast{
    backdrop-filter:blur(6px);
}
"""

class UIManager:
    """
    Manages the rendering and logic of all UI components for the Skyscope Sentinel application.
    This class acts as a controller, connecting the Streamlit view to the backend managers.
    """

    def __init__(
        self,
        state_manager: StateManager,
        agent_manager: AgentManager,
        quantum_manager: QuantumManager,
        browser_automator: BrowserAutomator,
        fs_manager: FileSystemManager,
        oc_manager: OpenCoreManager,
        business_generator: BusinessGenerator,
    ):
        """
        Initializes the UIManager with all necessary backend components.

        Args:
            state_manager: Manages the Streamlit session state.
            agent_manager: Orchestrates AI agents and swarms.
            quantum_manager: Handles quantum computations.
            browser_automator: Controls the web browser.
            fs_manager: Manages safe filesystem operations.
            oc_manager: Manages OpenCore config.plist files.
            business_generator: Generates autonomous business plans.
        """
        self.state = state_manager
        self.agent_manager = agent_manager
        self.quantum_manager = quantum_manager
        self.browser_automator = browser_automator
        self.fs_manager = fs_manager
        self.oc_manager = oc_manager
        self.business_generator = business_generator
        self._css_injected = False

    def render_sidebar(self):
        """Renders the sidebar with configuration options."""
        with st.sidebar:
            st.image("https://raw.githubusercontent.com/skyscope-sentinel/Skyscope-AI-Agent-Run-Business/main/images/swarmslogobanner.png", use_column_width=True)
            st.title("Configuration")

            # --- Model Settings ---
            st.subheader("Model Settings")
            model_provider = st.selectbox(
                "Model Provider",
                options=list(MODEL_OPTIONS.keys()),
                index=list(MODEL_OPTIONS.keys()).index(self.state.get("model_provider")),
            )
            self.state.set("model_provider", model_provider)

            model_options = MODEL_OPTIONS[model_provider]
            model_name = st.selectbox(
                "Model",
                options=model_options,
                index=0 if self.state.get("model_name") not in model_options else model_options.index(self.state.get("model_name")),
            )
            self.state.set("model_name", model_name)

            if model_provider == "API":
                with st.expander("Configure API Keys"):
                    for provider in self.state.get("api_keys", {}).keys():
                        key = st.text_input(f"{provider.capitalize()} API Key", value=self.state.get_api_key(provider), type="password")
                        self.state.set_api_key(provider, key)

            # --- AI Capabilities ---
            st.subheader("AI Capabilities")
            tools = self.state.get("tools_enabled", {})
            for tool_name, status in tools.items():
                label = tool_name.replace('_', ' ').title()
                new_status = st.toggle(label, value=status)
                if new_status != status:
                    self.state.set_tool_enabled(tool_name, new_status)
            
            # --- Browser Automation Controls ---
            if self.state.get_tool_enabled("browser_automation"):
                st.subheader("Browser Automation")
                if not self.browser_automator.is_running():
                    if st.button("Start Browser"):
                        st.switch_page("browser_automation.py") # Placeholder for actual start logic
                else:
                    if st.button("Stop Browser"):
                        st.switch_page("browser_automation.py") # Placeholder for actual stop logic

            # --- Global Actions ---
            st.subheader("Global Actions")
            if st.button("Clear Chat History"):
                self.state.clear_messages()
                self.state.set("code_window_content", "")
                st.rerun()

    def render_chat_interface(self):
        """Renders the main chat interface."""
        st.subheader("Conversation")

        for message in self.state.get_messages():
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What can I help you with today?"):
            self.state.add_message("user", prompt)
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # This is where the agent manager would be called
                    # For now, we'll just echo the prompt as a mock response
                    # response = self.agent_manager.run_main_swarm(prompt)
                    response = f"Received your message: '{prompt}'. The agent swarm would process this."
                    st.markdown(response)
                    self.state.add_message("assistant", response)

    def render_code_window(self):
        """Renders the code editor and execution controls."""
        st.subheader("Code Window")
        
        code_content = st.text_area(
            "Code Editor",
            value=self.state.get("code_window_content", ""),
            height=400,
            key="code_editor"
        )
        self.state.set("code_window_content", code_content)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Code", use_container_width=True):
                if code_content:
                    with st.spinner("Running code..."):
                        # In a real app, this should be sandboxed
                        try:
                            result = subprocess.run(
                                [sys.executable, "-c", code_content],
                                capture_output=True, text=True, timeout=30
                            )
                            st.subheader("Output")
                            if result.stdout:
                                st.code(result.stdout, language='bash')
                            if result.stderr:
                                st.code(result.stderr, language='bash')
                        except Exception as e:
                            st.error(f"Failed to execute code: {e}")
                else:
                    st.warning("Code editor is empty.")
        with col2:
            if st.button("Clear Code", use_container_width=True):
                self.state.set("code_window_content", "")
                st.rerun()
        with col3:
            if code_content:
                file_name = sanitize_filename(f"code_snippet_{datetime.now().strftime('%Y%m%d%H%M')}.py")
                st.download_button(
                    label="Save Code",
                    data=code_content,
                    file_name=file_name,
                    mime="text/python",
                    use_container_width=True,
                )

    def render_system_prompt_editor(self):
        """Renders the editor for the main system prompt."""
        st.subheader("Advanced System Prompt Editor")
        
        system_prompt = st.text_area(
            "System Prompt",
            value=self.state.get("system_prompt"),
            height=300
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update System Prompt", use_container_width=True):
                self.state.set("system_prompt", system_prompt)
                st.toast("System prompt updated successfully!", icon="✅")
        with col2:
            if st.button("Reset to Default", use_container_width=True):
                from state_manager import DEFAULT_SYSTEM_PROMPT
                self.state.set("system_prompt", DEFAULT_SYSTEM_PROMPT)
                st.rerun()

    def render_knowledge_stack(self):
        """Renders the interface for managing the knowledge stack."""
        st.subheader("Knowledge Stack")
        
        uploaded_file = st.file_uploader(
            "Upload a document to the knowledge stack",
            type=["txt", "md", "pdf", "json"]
        )
        if uploaded_file:
            content = uploaded_file.getvalue().decode("utf-8")
            self.state.add_to_knowledge_stack(
                title=uploaded_file.name,
                content=content,
                source="upload"
            )
            st.toast(f"Added '{uploaded_file.name}' to knowledge stack.", icon="📚")

        st.divider()

        knowledge_items = self.state.get_knowledge_stack()
        if not knowledge_items:
            st.info("The knowledge stack is empty. Upload a document to get started.")
        else:
            for item in knowledge_items:
                with st.expander(f"**{item['title']}** (Source: {item['source']})"):
                    st.text(item['content'][:500] + "...")
                    if st.button("Remove", key=f"remove_knowledge_{item['id']}"):
                        self.state.remove_from_knowledge_stack(item['id'])
                        st.rerun()

    def render_file_manager(self):
        """Renders the interface for file uploads and downloads."""
        st.subheader("File Manager")

        uploaded_files = st.file_uploader(
            "Upload files for agent use",
            accept_multiple_files=True
        )
        if uploaded_files:
            for file in uploaded_files:
                self.state.add_uploaded_file(
                    name=file.name,
                    file_type=file.type,
                    content=file.getvalue()
                )
            st.toast(f"Uploaded {len(uploaded_files)} file(s).", icon="📄")
            st.rerun()

        st.divider()

        files = self.state.get_uploaded_files()
        if not files:
            st.info("No files have been uploaded.")
        else:
            for file in files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file['name']}")
                with col2:
                    st.download_button(
                        "Download",
                        data=file['content'],
                        file_name=file['name'],
                        mime=file['type'],
                        key=f"download_file_{file['id']}",
                        use_container_width=True,
                    )
                with col3:
                    if st.button("Delete", key=f"delete_file_{file['id']}", use_container_width=True):
                        self.state.remove_uploaded_file(file['id'])
                        st.rerun()

    def render_quantum_interface(self):
        """Renders the UI for quantum computing tasks."""
        st.subheader("Quantum AI Interface")
        
        problem_desc = st.text_area("Describe the problem to solve with a quantum algorithm:")
        
        col1, col2 = st.columns(2)
        with col1:
            num_qubits = st.slider("Number of Qubits", 2, 10, 3)
        with col2:
            shots = st.select_slider("Number of Shots", options=[1024, 2048, 4096, 8192], value=1024)
            
        if st.button("Run Quantum Simulation"):
            if problem_desc:
                with st.spinner("Running quantum simulation..."):
                    # This is a mock implementation
                    spec = [{'gate': 'h', 'targets': list(range(num_qubits))}, {'gate': 'measure', 'targets': list(range(num_qubits))}]
                    circuit = self.quantum_manager.create_circuit(num_qubits, spec)
                    counts = self.quantum_manager.execute_circuit(circuit, shots)
                    probs = self.quantum_manager.process_results(counts)
                    
                    st.success("Simulation Complete!")
                    st.bar_chart(probs)
                    st.json(probs)
            else:
                st.warning("Please describe the problem first.")

    def render_browser_automation(self):
        """Renders the UI for browser automation tasks."""
        st.subheader("Browser Automation")
        st.info("This feature allows the AI to control a web browser to perform tasks.", icon="🌐")

        command = st.text_input("Enter a natural language command for the browser:", placeholder="e.g., 'Go to google.com and search for Swarms AI'")
        
        if st.button("Execute Browser Command"):
            if command:
                with st.spinner("Executing browser command..."):
                    # Mock execution
                    st.success(f"Command '{command}' executed successfully (simulated).")
                    st.image("https://raw.githubusercontent.com/kyegomez/swarms/master/images/swarmslogobanner.png", caption="Simulated browser view after command.")
            else:
                st.warning("Please enter a command.")

    def render_opencore_configurator(self):
        """Renders the UI for configuring OpenCore config.plist files."""
        st.subheader("OpenCore `config.plist` Configurator")

        uploaded_plist = st.file_uploader("Upload your config.plist", type=["plist"])
        if uploaded_plist:
            self.oc_manager.load_config(uploaded_plist)
            st.toast("`config.plist` loaded successfully.", icon="✅")

        if self.oc_manager.config:
            with st.expander("SMBIOS Settings", expanded=True):
                model = self.oc_manager.get_value("PlatformInfo.Generic.SystemProductName")
                serial = self.oc_manager.get_value("PlatformInfo.Generic.SystemSerialNumber")
                mlb = self.oc_manager.get_value("PlatformInfo.Generic.MLB")

                st.text_input("System Product Name", value=model, key="smbios_model")
                st.text_input("System Serial Number", value=serial, key="smbios_serial")
                st.text_input("Board Serial Number (MLB)", value=mlb, key="smbios_mlb")

                if st.button("Generate New SMBIOS"):
                    new_model = st.session_state.smbios_model
                    self.oc_manager.generate_smbios(new_model)
                    st.success("New SMBIOS data generated. Save the file to apply.")
                    st.rerun()
            
            with st.expander("Kernel Quirks"):
                quirks_path = 'Kernel.Quirks'
                quirks = self.oc_manager.get_value(quirks_path) or {}
                for key, value in quirks.items():
                    if isinstance(value, bool):
                        new_val = st.toggle(key, value=value, key=f"quirk_{key}")
                        if new_val != value:
                            self.oc_manager.set_value(f"{quirks_path}.{key}", new_val)
            
            st.divider()
            # Serialize the current config to bytes for download
            from io import BytesIO
            buf = BytesIO()
            import plistlib
            plistlib.dump(self.oc_manager.config, buf)
            buf.seek(0)
            
            st.download_button(
                label="Download Modified config.plist",
                data=buf,
                file_name="config_modified.plist",
                mime="application/x-plist"
            )

    def render_business_generator(self):
        """Renders the UI for the autonomous business generator."""
        st.subheader("Autonomous Business Generator")
        st.info("Generate a complete business proposal from a simple theme.", icon="💡")

        theme = st.text_input("Enter a business theme or industry:", placeholder="e.g., 'Sustainable urban farming'")
        company_name = st.text_input("Enter a name for the new company:", placeholder="e.g., 'UrbanHarvest'")

        if st.button("Generate Business Proposal"):
            if theme and company_name:
                with st.spinner(f"Generating proposal for {company_name}..."):
                    # This is a mock call
                    # In a real app, this would be a long-running task
                    st.session_state.proposal_path = self.business_generator.generate_full_business_proposal(theme, company_name)
                    if st.session_state.proposal_path:
                        st.success("Business proposal generated successfully!")
                    else:
                        st.error("Failed to generate business proposal.")
            else:
                st.warning("Please provide both a theme and a company name.")

        if "proposal_path" in st.session_state and st.session_state.proposal_path:
            with open(st.session_state.proposal_path, "r") as f:
                proposal_content = f.read()
            st.markdown("---")
            st.markdown(proposal_content)
            st.download_button(
                "Download Proposal",
                data=proposal_content,
                file_name=os.path.basename(st.session_state.proposal_path),
                mime="text/markdown"
            )

    def render_main_layout(self):
        """Renders the main layout and tabs of the application."""
        st.set_page_config(
            page_title="Skyscope Sentinel Intelligence",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        # Inject custom CSS once
        if not self._css_injected:
            st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
            self._css_injected = True

        st.title("Skyscope Sentinel Intelligence - AI Agentic Swarm")

        self.render_sidebar()

        main_view, code_view = st.columns([3, 2])

        with main_view:
            tab_titles = [
                "Chat", "System Prompt", "Knowledge Stack", "File Manager", 
                "Browser Automation", "Quantum AI", "OpenCore Config", "Business Generator"
            ]
            tabs = st.tabs(tab_titles)
            
            with tabs[0]:
                self.render_chat_interface()
            with tabs[1]:
                self.render_system_prompt_editor()
            with tabs[2]:
                self.render_knowledge_stack()
            with tabs[3]:
                self.render_file_manager()
            with tabs[4]:
                self.render_browser_automation()
            with tabs[5]:
                self.render_quantum_interface()
            with tabs[6]:
                self.render_opencore_configurator()
            with tabs[7]:
                self.render_business_generator()

        with code_view:
            self.render_code_window()
