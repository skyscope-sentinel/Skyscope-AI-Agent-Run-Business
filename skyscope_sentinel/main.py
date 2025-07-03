import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QStatusBar, QSystemTrayIcon, QMenu, QGridLayout, QSizePolicy,
    QListWidget, QGroupBox, QComboBox, QTextEdit, QLineEdit, QMessageBox # Added QMessageBox
)
from PySide6.QtCore import Qt, QSize, QFile, QTextStream, Slot
from PySide6.QtGui import QColor, QPalette, QIcon, QAction, QFont # QFont might be useful for list items

from .model_hub_page import ModelHubPage
from .settings_page import SettingsPage, SETTING_THEME, SETTING_ACRYLIC_EFFECT, SETTING_AUTOSTART, SETTING_MINIMIZE_TO_TRAY # Added keys
from .settings_manager import SettingsManager
from .video_agent_page import VideoAgentPage # Import VideoAgentPage


def load_stylesheet(filename):
    """Loads a QSS file and returns its content as a string."""
    file = QFile(filename)
    if file.open(QFile.ReadOnly | QFile.Text):
        stream = QTextStream(file)
        stylesheet = stream.readAll()
        file.close()
        return stylesheet
    print(f"Error loading stylesheet: {filename}")
    return ""

DARK_STYLE_PATH = "skyscope_sentinel/dark_theme.qss"
LIGHT_STYLE_PATH = "skyscope_sentinel/light_theme.qss"

# Agent imports - remove the old AIAgent class definition
from .agents.metagpt_pm_agent import ProductManagerAgent
from .agents.metagpt_engineer_agent import EngineerAgent
from .agents.metagpt_reviewer_agent import ReviewerAgent
from .owl_integration.owl_base_agent import OwlBaseAgent
# We'll also need OllamaIntegration if we want to initialize these agents fully here for display
from .ollama_integration import OllamaIntegration
# For research task page
from .autogen_interface import initiate_research_via_autogen
from .swarms_integration.opportunity_scouting_swarm import run_opportunity_scouting_swarm # Import swarm runner
import asyncio # For running autogen interface
from PySide6.QtCore import QThread, Signal # For running async tasks in background
from .config import Config # Import the Config class

# Global config instance to be used by other modules
# This should be one of the first things initialized so other modules can import it.
# However, it needs to be updated by SettingsManager from the GUI later.
global_config = Config()


# --- QThread for running asyncio tasks ---
class AsyncRunnerThread(QThread):
    task_completed = Signal(object) # Signal to emit result
    task_failed = Signal(str)       # Signal to emit error message

    def __init__(self, coro, *args, parent=None):
        super().__init__(parent)
        self.coro = coro
        self.args = args

    def run(self):
        try:
            # Create a new event loop for this thread if one doesn't exist,
            # or get the existing one if already set by a higher-level async manager for Qt.
            # For simple cases, asyncio.run() creates and closes its own loop.
            # However, for QThread, it's often better to manage the loop explicitly
            # if there are multiple async tasks or complex interactions.
            # For this specific case where we run one main async function,
            # asyncio.run() might be sufficient if it handles loop creation/closing cleanly.
            # A more robust approach for Qt+asyncio is often libraries like qasync or asyncqt.
            # Sticking to asyncio.run() for now for simplicity.
            result = asyncio.run(self.coro(*self.args))
            self.task_completed.emit(result)
        except Exception as e:
            self.task_failed.emit(str(e))


class ResearchTaskPage(QWidget):
    status_message_requested = Signal(str, str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("researchTaskPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_label = QLabel("Market Opportunity Research Task")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        form_layout = QGridLayout()
        form_layout.setSpacing(10)

        # Research Mode Selection
        mode_label = QLabel("Research Mode:")
        self.mode_selection_combo = QComboBox()
        self.mode_selection_combo.addItems(["CrewAI Research", "Swarm Opportunity Scouting"])
        self.mode_selection_combo.setToolTip("Select the AI agent system to perform the research.")
        form_layout.addWidget(mode_label, 0, 0)
        form_layout.addWidget(self.mode_selection_combo, 0, 1)

        topic_label = QLabel("Research Topic:")
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g., AI tools for content creation (optional for Swarm)")
        form_layout.addWidget(topic_label, 1, 0)
        form_layout.addWidget(self.topic_input, 1, 1)

        layout.addLayout(form_layout)

        self.run_button = QPushButton(QIcon.fromTheme("system-run"), "Start Task") # Renamed button
        self.run_button.setToolTip("Initiate the selected AI agent system to investigate the topic.")
        self.run_button.clicked.connect(self.handle_run_task) # Renamed handler
        layout.addWidget(self.run_button, 0, Qt.AlignCenter)

        results_group = QGroupBox("Task Output") # Renamed group
        results_layout = QVBoxLayout(results_group)
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Task output will appear here...") # Updated placeholder
        results_layout.addWidget(self.results_display)
        layout.addWidget(results_group)

        self.async_thread = None # To hold the thread

    def handle_run_task(self): # Renamed from handle_run_research
        topic = self.topic_input.text().strip()
        selected_mode = self.mode_selection_combo.currentText()

        if selected_mode == "CrewAI Research" and not topic:
            self.status_message_requested.emit("Research topic cannot be empty for CrewAI Research.", "warning", 3000)
            QMessageBox.warning(self, "Input Error", "Please enter a research topic for CrewAI mode.")
            return

        # For Swarm mode, topic is optional. If empty, run_opportunity_scouting_swarm will handle it.

        self.run_button.setEnabled(False)
        task_description = f"Task started using {selected_mode}."
        if topic:
            task_description += f" Topic: '{topic}'."
        self.results_display.setText(f"{task_description}\nPlease wait, this may take some time.")
        QApplication.processEvents() # Update UI

        if self.async_thread and self.async_thread.isRunning():
            self.status_message_requested.emit("A task is already in progress.", "warning", 3000)
            return

        if selected_mode == "CrewAI Research":
            self.async_thread = AsyncRunnerThread(initiate_research_via_autogen, topic)
            target_function_name = "CrewAI Research"
        elif selected_mode == "Swarm Opportunity Scouting":
            # run_opportunity_scouting_swarm takes initial_topic and verbose
            # We pass the topic (which can be None/empty for the swarm)
            self.async_thread = AsyncRunnerThread(run_opportunity_scouting_swarm, topic, True) # verbose=True
            target_function_name = "Swarm Scouting"
        else:
            self.status_message_requested.emit(f"Unknown research mode: {selected_mode}", "error", 5000)
            self.run_button.setEnabled(True)
            return

        self.async_thread.task_completed.connect(self.on_task_completed) # Renamed slot
        self.async_thread.task_failed.connect(self.on_task_failed)       # Renamed slot
        self.async_thread.start()
        self.status_message_requested.emit(f"{target_function_name} task started for topic '{topic if topic else 'auto-generated'}'.", "info", 0)


    @Slot(object)
    def on_task_completed(self, result): # Renamed from on_research_completed
        self.results_display.setText(str(result)) # Result could be markdown report or file path
        self.run_button.setEnabled(True)
        self.status_message_requested.emit("Task completed successfully.", "success", 5000)
        QMessageBox.information(self, "Task Complete", "The AI task has finished.")
        self.async_thread = None

    @Slot(str)
    def on_task_failed(self, error_message): # Renamed from on_research_failed
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.run_button.setEnabled(True)
        self.status_message_requested.emit(f"Task failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Task Failed", f"An error occurred during the AI task:\n{error_message}")
        self.async_thread = None


class PlaceholderPage(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setObjectName("placeholderPage")
        layout = QVBoxLayout(self)
        self.label = QLabel(f"Welcome to the {name} Page", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888888;") # More subtle placeholder text
        layout.addWidget(self.label)

        if name == "Dashboard":
            # Simple card-like layout for Dashboard placeholder
            grid_layout = QGridLayout()
            
            card_titles = ["Active Agents", "System Status", "Recent Activity", "Model Performance"]
            for i, title in enumerate(card_titles):
                card = QFrame()
                card.setObjectName(f"dashboardCard{i}")
                card.setFrameShape(QFrame.StyledPanel)
                card.setFrameShadow(QFrame.Raised)
                card.setMinimumSize(200, 100)
                card.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 6px; background-color: rgba(255,255,255,0.05); }") # Basic card style
                
                card_layout = QVBoxLayout(card)
                title_label = QLabel(title)
                title_label.setStyleSheet("font-size: 16px; font-weight: bold; border: none; background: transparent;")
                content_label = QLabel("Details and metrics will appear here.")
                content_label.setStyleSheet("font-size: 12px; border: none; background: transparent;")
                card_layout.addWidget(title_label)
                card_layout.addWidget(content_label)
                card_layout.addStretch()
                grid_layout.addWidget(card, i // 2, i % 2)
            layout.addLayout(grid_layout)

        elif name == "Agent Control":
            self.label.setText("Manage and Configure Your AI Agents") # Main title
            self.label.setAlignment(Qt.AlignTop | Qt.AlignHCenter) # Adjust alignment
            self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888888; margin-bottom: 10px;")


            # Agents List Section
            agents_group = QGroupBox("Available Agents")
            agents_group_layout = QVBoxLayout()

            self.agent_list_widget = QListWidget()
            self.agent_list_widget.setToolTip("List of configured AI agents and their status.")
            agents_group_layout.addWidget(self.agent_list_widget)
            agents_group.setLayout(agents_group_layout)
            layout.addWidget(agents_group)

            # Instantiate OllamaIntegration (assuming it's light enough to do here)
            # In a real app, this might be a shared instance.
            self.ollama_integration_instance = OllamaIntegration() # Ensure this doesn't block

            # Sample Agents Data - Using new agent classes
            self.agents = [] # Initialize empty list
            try:
                pm_agent = ProductManagerAgent(agent_id="PM001", ollama_integration_instance=self.ollama_integration_instance)
                self.agents.append(pm_agent)

                eng_agent = EngineerAgent(agent_id="ENG001", ollama_integration_instance=self.ollama_integration_instance)
                self.agents.append(eng_agent)

                rev_agent = ReviewerAgent(agent_id="REV001", ollama_integration_instance=self.ollama_integration_instance)
                self.agents.append(rev_agent)

                research_agent = OwlBaseAgent(agent_id="RES001", department="Researchers", role_description="Conducts web research.")
                self.agents.append(research_agent)

                hr_agent = OwlBaseAgent(agent_id="HR001", department="HR", role_description="Manages personnel records (simulated).")
                self.agents.append(hr_agent)

            except Exception as e:
                print(f"Error instantiating sample agents for GUI: {e}")
                # Add a placeholder if agent instantiation fails
                error_agent = OwlBaseAgent(agent_id="ERR999", department="System")
                error_agent.identity['first_name'] = "Error"
                error_agent.identity['last_name'] = "State"
                error_agent.identity['employee_title'] = "Agent Init Failed"
                error_agent.status = "Error"
                self.agents.append(error_agent)


            for agent_instance in self.agents:
                display_text = f"{agent_instance.identity.get('first_name', 'N/A')} {agent_instance.identity.get('last_name', '')} " \
                               f"({agent_instance.identity.get('employee_title', 'N/A')}) - Status: {agent_instance.status}"
                self.agent_list_widget.addItem(display_text)
                list_item = self.agent_list_widget.item(self.agent_list_widget.count() - 1)
                list_item.setData(Qt.UserRole, agent_instance) # Store the actual agent object

            # Agent Actions Section
            actions_layout = QHBoxLayout()
            self.btn_start_agent = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Selected")
            self.btn_start_agent.setToolTip("Start the selected agent.")
            self.btn_start_agent.setEnabled(False)
            self.btn_start_agent.clicked.connect(self.start_selected_agent)
            actions_layout.addWidget(self.btn_start_agent)

            self.btn_stop_agent = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Selected")
            self.btn_stop_agent.setToolTip("Stop the selected agent.")
            self.btn_stop_agent.setEnabled(False)
            self.btn_stop_agent.clicked.connect(self.stop_selected_agent)
            actions_layout.addWidget(self.btn_stop_agent)

            self.btn_config_agent = QPushButton(QIcon.fromTheme("preferences-system"), "Configure Selected")
            self.btn_config_agent.setToolTip("Configure the selected agent.")
            self.btn_config_agent.setEnabled(False)
            self.btn_config_agent.clicked.connect(self.configure_selected_agent)
            actions_layout.addWidget(self.btn_config_agent)

            self.btn_view_logs = QPushButton(QIcon.fromTheme("document-preview"), "View Logs")
            self.btn_view_logs.setToolTip("View logs for the selected agent.")
            self.btn_view_logs.setEnabled(False)
            self.btn_view_logs.clicked.connect(self.view_agent_logs)
            actions_layout.addWidget(self.btn_view_logs)

            layout.addLayout(actions_layout)

            # Add New Agent Button
            self.add_agent_btn = QPushButton(QIcon.fromTheme("list-add"), "Add New Agent...")
            self.add_agent_btn.setToolTip("Define and configure a new AI agent.")
            self.add_agent_btn.clicked.connect(self.add_new_agent)
            self.add_agent_btn.setStyleSheet("QPushButton { text-align: center; padding-left: 0px; margin-top: 10px; }")
            layout.addWidget(self.add_agent_btn, 0, Qt.AlignCenter)

            layout.addStretch() # Ensure content pushes to the top

            # Connect agent selection change to method
            self.agent_list_widget.currentItemChanged.connect(self.on_agent_selection_changed)


        elif name == "Log Stream":
            # Keep original label for this page, but adjust its style for consistency if needed.
            self.label.setText("Centralized Log Monitoring")
            self.label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
            self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888888; margin-bottom: 10px;")

            info_label = QLabel("View real-time logs from the application, Ollama service, and all active agents. Filter by source, log level, and search for specific messages.")
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("font-size: 14px; color: #777777; margin: 5px 20px 15px 20px;") # Adjusted margins
            layout.addWidget(info_label)

            # Controls Area
            controls_layout = QHBoxLayout()

            filter_label = QLabel("Filter by source:")
            controls_layout.addWidget(filter_label)

            self.log_filter_combo = QComboBox()
            self.log_filter_combo.addItems(["All Logs", "Application Logs", "Ollama Service", "Agent Alpha", "Agent Beta"])
            self.log_filter_combo.setToolTip("Filter logs by their source.")
            controls_layout.addWidget(self.log_filter_combo)

            self.search_log_input = QLineEdit()
            self.search_log_input.setPlaceholderText("Search logs...")
            self.search_log_input.setToolTip("Enter keywords to search in logs.")
            # self.search_log_input.setEnabled(False) # Keep disabled for now as per original, or enable for typing
            self.search_log_input.textChanged.connect(self.on_log_search_changed) # Connect for future use
            controls_layout.addWidget(self.search_log_input)

            self.clear_logs_btn = QPushButton(QIcon.fromTheme("edit-clear"), "Clear Logs")
            self.clear_logs_btn.setToolTip("Clear the displayed logs.")
            # self.clear_logs_btn.setEnabled(False) # Keep disabled for now, or enable
            self.clear_logs_btn.clicked.connect(self.clear_logs_display)
            controls_layout.addWidget(self.clear_logs_btn)

            layout.addLayout(controls_layout)

            # Connect filter combo
            self.log_filter_combo.currentIndexChanged.connect(self.on_log_filter_changed)


            # Log Display Area
            self.log_display_area = QTextEdit()
            self.log_display_area.setReadOnly(True)
            sample_logs = (
                "[INFO]  2023-10-27 10:00:00 - Application successfully initialized.\n"
                "[DEBUG] 2023-10-27 10:00:05 - Checking Ollama service status...\n"
                "[OLLAMA]2023-10-27 10:00:06 - Ollama service detected, version: 0.1.15\n"
                "[AGENT_ALPHA] 2023-10-27 10:01:00 - Starting Website Content Generation task.\n"
                "[ERROR] 2023-10-27 10:01:30 - Agent Beta failed to connect to external API: Timeout.\n"
                "[WARN]  2023-10-27 10:02:00 - Low disk space detected on /var/log.\n"
                "[INFO]  2023-10-27 10:03:00 - User 'admin' logged in from 192.168.1.10.\n"
                "[DEBUG] 2023-10-27 10:03:05 - Processing request ID: ax7812gf.\n"
                "[OLLAMA]2023-10-27 10:04:10 - Model 'llama2:7b' loaded successfully.\n"
                "[AGENT_BETA] 2023-10-27 10:05:00 - Task 'ImageAnalysis' completed. Results saved to /output/img_xyz.json.\n"
                "[ERROR] 2023-10-27 10:05:30 - Application database connection lost. Attempting reconnect...\n"
                "[WARN]  2023-10-27 10:06:00 - Configuration file 'settings.json' has new unrecognized keys.\n"
            )
            self.log_display_area.setPlaceholderText("Logs will appear here...")
            self.log_display_area.setText(sample_logs)

            log_font = QFont("Monospace") # Or "Courier", "Consolas"
            log_font.setStyleHint(QFont.Monospace)
            log_font.setPointSize(10) # Slightly smaller for more content
            self.log_display_area.setFont(log_font)
            layout.addWidget(self.log_display_area)


        self.setLayout(layout)

    # --- Agent Control Page Methods ---
    def on_agent_selection_changed(self, current_item, previous_item):
        is_selected = current_item is not None
        if hasattr(self, 'btn_start_agent'): # Check if buttons exist (only for Agent Control page)
            self.btn_start_agent.setEnabled(is_selected)
            self.btn_stop_agent.setEnabled(is_selected)
            self.btn_config_agent.setEnabled(is_selected)
            self.btn_view_logs.setEnabled(is_selected)

        if current_item and hasattr(self, 'agents'): # Check if agent data is relevant
            agent = current_item.data(Qt.UserRole)
            if agent: # Ensure agent data is present
                 print(f"Selected agent: {agent}") # For debugging

    def start_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            agent_title = agent_instance.identity.get('employee_title', 'N/A')
            print(f"Attempting to start agent: {agent_name} (Title: {agent_title}, Current Status: {agent_instance.status})")
            agent_instance.status = "Running" # Simulate status change
            current_item.setText(f"{agent_name} ({agent_title}) - Status: {agent_instance.status}")
            self.show_status_message(f"Agent '{agent_name}' started (simulated).", "success")
        else:
            print("No agent selected to start.")
            self.show_status_message("No agent selected.", "warning")

    def stop_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            agent_title = agent_instance.identity.get('employee_title', 'N/A')
            print(f"Attempting to stop agent: {agent_name} (Title: {agent_title})")
            agent_instance.status = "Offline" # Simulate status change
            current_item.setText(f"{agent_name} ({agent_title}) - Status: {agent_instance.status}")
            self.show_status_message(f"Agent '{agent_name}' stopped (simulated).", "info")
        else:
            print("No agent selected to stop.")
            self.show_status_message("No agent selected.", "warning")

    def configure_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            print(f"Attempting to configure agent: {agent_name}")
            # Placeholder for actual configuration. For OwlBaseAgent, we might show role_description or toolkits.
            config_details = {
                "agent_id": agent_instance.agent_id,
                "name": agent_name,
                "title": agent_instance.identity.get('employee_title'),
                "department": agent_instance.identity.get('department'),
                "status": agent_instance.status,
            }
            if isinstance(agent_instance, OwlBaseAgent):
                config_details["role_description"] = agent_instance.role_description
                config_details["toolkits"] = [type(tk).__name__ for tk in agent_instance.available_toolkits] if agent_instance.available_toolkits else "None"

            print(f"Current config/details: {config_details}")
            self.show_status_message(f"Configuration for '{agent_name}' would open here (details in console).", "info")
        else:
            print("No agent selected to configure.")
            self.show_status_message("No agent selected.", "warning")

    def view_agent_logs(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            print(f"Attempting to view logs for agent: {agent_name}")
            # In a real app, this would switch to a log view filtered for this agent.
            # For now, just a message. We can use agent_instance.message_log (from BaseAgent)
            print(f"Simulated log view for {agent_name}. Message log: {agent_instance.message_log[-5:] if agent_instance.message_log else 'Empty'}")
            self.show_status_message(f"Log view for '{agent_name}' would open here (sample in console).", "info")
        else:
            print("No agent selected to view logs.")
            self.show_status_message("No agent selected.", "warning")

    def add_new_agent(self):
        # This function will now add a generic OwlBaseAgent as a placeholder.
        # A more complex UI would be needed to choose agent type, department, etc.
        self.show_status_message("Adding a new generic agent (simulated)...", "info")
        new_agent_id_num = len(self.agents) + 1
        # For simplicity, assign to a random department or a default like "Staff"
        # For this placeholder, we'll use OwlBaseAgent directly.
        try:
            new_agent_instance = OwlBaseAgent(
                agent_id=f"AGENT{new_agent_id_num:03d}",
                department="Staff" # Default department for new generic agents
            )
            self.agents.append(new_agent_instance)

            display_text = f"{new_agent_instance.identity.get('first_name', 'N/A')} {new_agent_instance.identity.get('last_name', '')} " \
                           f"({new_agent_instance.identity.get('employee_title', 'N/A')}) - Status: {new_agent_instance.status}"
            self.agent_list_widget.addItem(display_text)
            new_list_item = self.agent_list_widget.item(self.agent_list_widget.count() - 1)
            new_list_item.setData(Qt.UserRole, new_agent_instance)

            agent_name = f"{new_agent_instance.identity.get('first_name', 'Agent')} {new_agent_instance.identity.get('last_name', new_agent_instance.agent_id)}"
            self.show_status_message(f"New generic agent '{agent_name}' added.", "success")
            print(f"Added new agent: {agent_name}, ID: {new_agent_instance.agent_id}, Dept: {new_agent_instance.identity.get('department')}")
        except Exception as e:
            print(f"Error adding new agent: {e}")
            self.show_status_message(f"Failed to add new agent: {e}", "error")


    # --- Log Stream Page Methods ---
    def on_log_filter_changed(self, index):
        if hasattr(self, 'log_filter_combo'): # Ensure this method is called for the correct page
            selected_filter = self.log_filter_combo.itemText(index)
            print(f"Log filter changed to: {selected_filter}")
            # Actual log filtering logic would go here.
            # For now, we just show a message.
            self.show_status_message(f"Log filter set to: {selected_filter}", "info")
            # Example: self.update_log_display(filter_text=selected_filter)

    def on_log_search_changed(self, text):
        if hasattr(self, 'search_log_input'):
            print(f"Log search text: {text}")
            # Actual search/filter logic would go here.
            # Example: self.update_log_display(search_term=text)
            # For now, this is a placeholder.
            if text: # Only show message if there's text, to avoid spamming on clear
                self.show_status_message(f"Searching logs for: {text}", "info", 1500)


    def clear_logs_display(self):
        if hasattr(self, 'log_display_area'): # Ensure this method is called for the correct page
            self.log_display_area.clear()
            print("Log display cleared.")
            self.show_status_message("Log display cleared.", "info")

    def show_status_message(self, message, msg_type="info", duration=3000):
        main_window = self.parent().parent().parent()
        if hasattr(main_window, 'show_status_message') and callable(getattr(main_window, 'show_status_message')):
            main_window.show_status_message(message, msg_type, duration)
        else:
            print(f"PlaceholderPage Status ({msg_type}): {message} (MainWindow not found or method missing)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyscope Sentinel Intelligence")
        self.setGeometry(100, 100, 1200, 700)

        # --- Enable Translucent Background and Frameless Window ---
        # This makes the main window background transparent and removes the default OS window frame.
        # The QSS styling for QWidget#centralWidget (with border-radius and background-color)
        # will now create the visual effect of a rounded window.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        # NOTE: Enabling Qt.FramelessWindowHint removes the standard title bar.
        # Custom controls for minimize, maximize, close, and window dragging would be
        # needed for full desktop application behavior. For now, closing and interaction
        # will be primarily via the system tray icon or task manager.

        # --- Main Widget and Layout ---
        self.central_widget = QWidget()
        self.central_widget.setObjectName("centralWidget") # For QSS styling
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0,0,0,0) # No margin for the main layout
        self.main_layout.setSpacing(0) # No spacing between sidebar and content

        # --- Sidebar ---
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar") # For QSS styling
        self.sidebar.setFixedWidth(220)
        self.sidebar_layout = QVBoxLayout(self.sidebar)
        self.sidebar_layout.setAlignment(Qt.AlignTop)
        self.sidebar_layout.setContentsMargins(10, 20, 10, 20) # Adjusted margins
        self.sidebar_layout.setSpacing(12)

        self.main_layout.addWidget(self.sidebar)

        # --- Content Area ---
        self.content_area = QStackedWidget()
        self.content_area.setObjectName("contentArea")
        self.main_layout.addWidget(self.content_area)

        # --- Navigation Buttons and Placeholder Pages ---
        self.settings_manager = SettingsManager() # Initialize settings manager for main window use

        self.nav_buttons = {}
        # Added "Opportunity Research" to sections
        self.sections = ["Dashboard", "Opportunity Research", "Agent Control", "Video Tools", "Model Hub", "Log Stream", "Settings"]
        
        app_title_label = QLabel("Skyscope Sentinel")
        app_title_label.setAlignment(Qt.AlignCenter)
        app_title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding-bottom: 10px; margin-top: 5px;")
        self.sidebar_layout.addWidget(app_title_label)

        # Define icons for sidebar items (using QIcon.fromTheme for now)
        # These might not show up on Windows if a proper icon theme isn't installed or if the names are Linux-specific.
        # For production, embedding actual icon files (SVGs) would be more reliable.
        icon_map = {
            "Dashboard": "view-dashboard",
            "Opportunity Research": "system-search", # Using a search icon
            "Agent Control": "applications-system",
            "Video Tools": "applications-multimedia",
            "Model Hub": "drive-harddisk",
            "Log Stream": "document-view",
            "Settings": "preferences-configure"
        }
        tooltips = {
            "Dashboard": "View system overview and key metrics",
            "Opportunity Research": "Run AI agents to research market opportunities",
            "Agent Control": "Manage and configure AI agents",
            "Video Tools": "Access video processing utilities",
            "Model Hub": "Explore and manage Ollama models",
            "Log Stream": "Monitor real-time application and agent logs",
            "Settings": "Configure application settings"
        }

        for section_name in self.sections:
            icon_name = icon_map.get(section_name, "application-default-icon") # Fallback icon name
            button = QPushButton(section_name)
            button.setIcon(QIcon.fromTheme(icon_name))
            button.setToolTip(tooltips.get(section_name, f"Navigate to {section_name}"))
            button.setCheckable(True)
            button.clicked.connect(lambda checked, name=section_name: self.switch_page(name))
            self.sidebar_layout.addWidget(button)
            self.nav_buttons[section_name] = button

            if section_name == "Model Hub":
                self.model_hub_page = ModelHubPage()
                self.model_hub_page.status_message_requested.connect(self.show_status_message)
                self.content_area.addWidget(self.model_hub_page)
            elif section_name == "Settings":
                self.settings_page = SettingsPage()
                self.settings_page.status_message_requested.connect(self.show_status_message)
                self.settings_page.theme_change_requested.connect(self.apply_theme_by_name)
                self.settings_page.acrylic_effect_requested.connect(self.apply_acrylic_effect)
                self.settings_page.tray_icon_visibility_requested.connect(self.set_tray_icon_visibility) # Connect new signal
                self.content_area.addWidget(self.settings_page)
            elif section_name == "Video Tools": # Condition for Video Tools
                self.video_agent_page = VideoAgentPage()
                self.video_agent_page.status_message_requested.connect(self.show_status_message)
                self.content_area.addWidget(self.video_agent_page)
            elif section_name == "Opportunity Research":
                self.research_task_page = ResearchTaskPage()
                self.research_task_page.status_message_requested.connect(self.show_status_message_slot) # Connect its status signal
                self.content_area.addWidget(self.research_task_page)
            else:
                page = PlaceholderPage(section_name)
                self.content_area.addWidget(page)
        
        self.sidebar_layout.addStretch()

        # --- Founder and Contact Info ---
        founder_label = QLabel("Founded by: Miss Casey Jay Topojani")
        founder_label.setAlignment(Qt.AlignCenter)
        founder_label.setStyleSheet("font-size: 10px; color: #999999; padding-top: 10px;")
        self.sidebar_layout.addWidget(founder_label)

        contact_label = QLabel("Contact: admin@skyscope.cloud")
        contact_label.setAlignment(Qt.AlignCenter)
        contact_label.setStyleSheet("font-size: 10px; color: #999999; padding-bottom: 5px;")
        self.sidebar_layout.addWidget(contact_label)

        # --- Theme Switch Button ---
        self.theme_button = QPushButton("Toggle Theme")
        self.theme_button.setCheckable(False)
        self.theme_button.setText("Toggle Theme")
        self.theme_button.setIcon(QIcon.fromTheme("preferences-desktop-theme"))
        self.theme_button.setToolTip("Quickly switch between dark and light themes.")
        self.theme_button.clicked.connect(self.toggle_theme_directly)
        self.sidebar_layout.addWidget(self.theme_button)

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.show_status_message("Welcome to Skyscope Sentinel!", "info", 5000)

        # --- Initialize Theme and Settings ---
        self.load_initial_settings() # Loads theme, acrylic based on SettingsManager

        # Update the global_config with values from SettingsManager
        # This ensures that any GUI-set API keys or Ollama settings are propagated
        # to the config object that other modules (like crews, autogen_interface) will use.
        if hasattr(self, 'settings_manager') and self.settings_manager:
            global_config.update_from_settings_manager(self.settings_manager)
            # Also, ensure environment variables are set if possible, so tools relying on os.getenv() directly get them.
            # This is a bit of a workaround for tools that don't take config objects directly.
            # Be cautious with this if keys are sensitive and many modules use os.getenv().
            if global_config.get_serper_api_key():
                os.environ["SERPER_API_KEY"] = global_config.get_serper_api_key()
            if global_config.get_openai_api_key():
                os.environ["OPENAI_API_KEY"] = global_config.get_openai_api_key()
            # Ollama settings are usually passed directly to LLM configs, but if some tool expected env vars:
            # os.environ["OLLAMA_MODEL"] = global_config.get_ollama_model_name()
            # os.environ["OLLAMA_BASE_URL"] = global_config.get_ollama_base_url()

        
        # --- System Tray Icon ---
        self.create_system_tray_icon()


        # Show the first page and set its button to active
        if self.sections:
            self.switch_page(self.sections[0])
            if self.sections[0] in self.nav_buttons: 
                 self.nav_buttons[self.sections[0]].setChecked(True)


    def load_initial_settings(self):
        """Load initial settings like theme and acrylic effect."""
        initial_theme_name = self.settings_manager.load_setting(SETTING_THEME, "dark")
        self.apply_theme_by_name(initial_theme_name)
        
        initial_acrylic = self.settings_manager.load_setting(SETTING_ACRYLIC_EFFECT, True)
        self.apply_acrylic_effect(initial_acrylic)


    def switch_page(self, section_name):
        target_widget_label = f"This is the {section_name} Page"
        page_found = False
        for i in range(self.content_area.count()):
            widget = self.content_area.widget(i)
            
            current_page_matches = False
            if section_name == "Model Hub" and isinstance(widget, ModelHubPage):
                current_page_matches = True
            elif section_name == "Settings" and isinstance(widget, SettingsPage):
                current_page_matches = True
            elif section_name == "Video Tools" and isinstance(widget, VideoAgentPage):
                current_page_matches = True
            elif section_name == "Opportunity Research" and isinstance(widget, ResearchTaskPage): # Match ResearchTaskPage
                current_page_matches = True
            # elif isinstance(widget, PlaceholderPage) and widget.label.text() == target_widget_label: # Old problematic check
            # Need a more robust way to identify PlaceholderPages if their title label changes or is removed.
            # For now, we rely on the order and type for ModelHubPage and SettingsPage.
            # For PlaceholderPage, we can check its internal 'name' attribute if we store it.
            # Or, more simply, match by section_name directly as ModelHub/Settings are specific classes.

            # Revised check for PlaceholderPage
            elif isinstance(widget, PlaceholderPage):
                # Access the 'name' attribute we set during PlaceholderPage creation (if we do)
                # For now, we'll assume that if it's a PlaceholderPage and not ModelHub/Settings,
                # it's matched by its position/order in self.sections.
                # This part of the logic might need refinement if pages are dynamically added/removed
                # or if PlaceholderPage's title is the only identifier.
                # The current PlaceholderPage __init__ takes 'name', but it's used for the label.
                # Let's assume the order in QStackedWidget matches self.sections for placeholders.
                if self.sections[i] == section_name: # This assumes order is preserved.
                    current_page_matches = True

            if current_page_matches:
                self.content_area.setCurrentIndex(i)
                page_found = True
                # Update the title label of the PlaceholderPage if it's the one being switched to
                # This is a bit of a workaround because the label text was previously used for matching.
                if isinstance(widget, PlaceholderPage) and hasattr(widget, 'label'):
                    # We might want to avoid changing the "Agent Control" specific title
                    if section_name != "Agent Control" and section_name != "Log Stream" and section_name != "Dashboard": # Avoid overwriting custom titles
                         widget.label.setText(f"Welcome to the {section_name} Page")
                break
        
        if page_found:
            for name, btn in self.nav_buttons.items():
                btn.setChecked(name == section_name)
            print(f"Switched to {section_name}")
            self.show_status_message(f"{section_name} page loaded.", "info", 3000)
        else:
            # This case should ideally not happen if all sections have a corresponding page
            print(f"Warning: Page for section '{section_name}' not found or match logic failed for index {i}.")
            # Fallback: try to find by objectName if set, or create if truly missing.
            # For now, we print a warning. A robust solution would be to store pages in a dictionary.
            # Example: self.pages[section_name] = page_object
            # Then: self.content_area.setCurrentWidget(self.pages[section_name])

            # Check if a PlaceholderPage exists with the given name (if we stored it)
            # This is a simplified check, assuming `widget.page_name_attribute == section_name`
            found_by_probing = False
            for idx in range(self.content_area.count()):
                page_widget = self.content_area.widget(idx)
                if isinstance(page_widget, PlaceholderPage):
                    # PlaceholderPage needs an attribute like `self.page_name = name` in its __init__
                    # For this example, let's assume the label text is still a semi-reliable way to find it if not Agent Control
                    if hasattr(page_widget, 'label') and page_widget.label.text().startswith(f"Welcome to the {section_name}"):
                        self.content_area.setCurrentIndex(idx)
                        page_found = True
                        break
            if not page_found:
                 print(f"Critical Warning: Page for section '{section_name}' truly not found.")



    def apply_theme_from_file(self, qss_file_path):
        """Applies a theme from a QSS file."""
        style_sheet = load_stylesheet(qss_file_path)
        if style_sheet:
            self.setStyleSheet(style_sheet)
            # Update current_theme_path after successfully applying
            self.current_theme_path = qss_file_path 
            # Update status bar color to match new theme
            self.show_status_message(self.status_bar.currentMessage().split(" (")[0], "info", 0) # Resets color
        else:
            print(f"Could not load theme from {qss_file_path}")
            self.show_status_message(f"Error loading theme: {qss_file_path}", "error", 5000)


    @Slot(str)
    def apply_theme_by_name(self, theme_name):
        """Applies theme based on name ('dark' or 'light')."""
        print(f"Main window applying theme: {theme_name}")
        if theme_name == "light":
            self.apply_theme_from_file(LIGHT_STYLE_PATH)
        else: # Default to dark
            self.apply_theme_from_file(DARK_STYLE_PATH)
        # No need to save here, settings_page saves it.
        self.show_status_message(f"{theme_name.capitalize()} theme applied.", "info", 3000)


    def toggle_theme_directly(self):
        """Allows user to toggle theme using the button, also updates setting."""
        if self.current_theme_path == DARK_STYLE_PATH:
            new_theme_name = "light"
        else:
            new_theme_name = "dark"
        
        self.settings_manager.save_setting(SETTING_THEME, new_theme_name)
        self.apply_theme_by_name(new_theme_name)
        
        # Update the settings page UI if it's currently visible or loaded
        if hasattr(self, 'settings_page') and self.settings_page:
            self.settings_page.combo_theme.setCurrentText(new_theme_name.capitalize())


    @Slot(bool)
    def apply_acrylic_effect(self, enabled):
        """Placeholder for applying acrylic effect. Currently, QSS handles transparency."""
        # In a real scenario, this might involve reloading QSS or specific platform calls.
        # For now, our QSS uses rgba for sidebar transparency.
        # We could modify the alpha value in QSS and reload, or have separate QSS versions.
        print(f"Acrylic effect requested: {enabled}. (Handled by QSS alpha for now)")
        self.show_status_message(f"Acrylic effect {'enabled' if enabled else 'disabled'}.", "info", 3000)
        # This setting is saved by settings_page. We might need to reload QSS if it changes alpha.
        # For simplicity, current QSS files have fixed alpha. A more dynamic approach:
        # 1. Read current QSS.
        # 2. Modify relevant rgba alpha values.
        # 3. Re-apply modified QSS.
        # Or, have theme_dark_acrylic.qss, theme_dark_no_acrylic.qss etc.
        # For now, we just acknowledge the setting. The QSS files already have some transparency.

    def create_system_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self)
        
        # Attempt to load a custom icon.
        # Ensure 'skyscope_sentinel/assets/app_icon.png' exists.
        # For testing, I'll assume it does. If not, QIcon will be null.
        custom_icon_path = "skyscope_sentinel/assets/app_icon.png" 
        app_icon = QIcon(custom_icon_path)

        if app_icon.isNull():
            print(f"Warning: Custom icon at '{custom_icon_path}' not found or invalid. Using theme fallback.")
            app_icon = QIcon.fromTheme("application-x-executable", self.style().standardIcon(getattr(QStyle, "SP_ComputerIcon", QStyle.SP_DesktopIcon)))
        
        self.tray_icon.setIcon(app_icon)
        self.setWindowIcon(app_icon) # Also set window icon

        tray_menu = QMenu(self) # Pass parent to menu
        show_action = QAction(QIcon.fromTheme("view-reveal"), "Show/Hide Window", self)
        show_action.setToolTip("Toggle the main application window visibility.")
        show_action.triggered.connect(self.toggle_window_visibility)
        tray_menu.addAction(show_action)

        tray_menu.addSeparator()

        quit_action = QAction(QIcon.fromTheme("application-exit"), "Quit Skyscope Sentinel", self)
        quit_action.setToolTip("Close the application.")
        quit_action.triggered.connect(self.quit_application) # Use a custom quit handler
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_icon_activated)
        
        # Only show tray icon if setting allows, or by default
        if self.settings_manager.load_setting("general/enable_tray_icon", True): # Default to true
            self.tray_icon.show()
        self.tray_icon.setToolTip("Skyscope Sentinel Intelligence Platform")

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger: # Typically left click
            self.toggle_window_visibility()
        elif reason == QSystemTrayIcon.Context: # Right click, menu already shown
            pass

    def toggle_window_visibility(self):
        if self.isVisible() and not self.isMinimized(): # if visible and not minimized
            self.hide()
            if self.settings_manager.load_setting("general/notify_on_minimize_to_tray", True): # New setting
                 self.tray_icon.showMessage(
                    "Skyscope Sentinel",
                    "Application hidden to system tray.",
                    QSystemTrayIcon.Information,
                    2000
                )
        else:
            self.showNormal() # Restores from minimized or hidden state
            self.activateWindow() # Brings to front

    def closeEvent(self, event):
        """Override close event to allow minimizing to tray if configured."""
        minimize_on_close = self.settings_manager.load_setting(SETTING_MINIMIZE_TO_TRAY, True) # Default to true
        
        if minimize_on_close and self.tray_icon.isVisible():
            self.hide()
            if self.settings_manager.load_setting("general/notify_on_minimize_to_tray", True):
                self.tray_icon.showMessage(
                    "Skyscope Sentinel",
                    "Application minimized to system tray. Right-click to quit.",
                    QSystemTrayIcon.Information,
                    2000
                )
            event.ignore()  # Ignore the original close event
        else:
            self.quit_application() # Proceed with normal quit sequence
            event.accept() 

    def quit_application(self):
        """Handles application cleanup and quitting."""
        self.show_status_message("Exiting Skyscope Sentinel...", "info", 2000)
        if self.tray_icon:
            self.tray_icon.hide() # Hide tray icon before quitting
        QApplication.instance().quit()

    @Slot(bool)
    def set_tray_icon_visibility(self, visible):
        if hasattr(self, 'tray_icon'):
            if visible:
                if not self.tray_icon.isVisible():
                    self.tray_icon.show()
                    self.show_status_message("System tray icon enabled.", "info", 3000)
            else:
                if self.tray_icon.isVisible():
                    self.tray_icon.hide()
                    # The message in SettingsPage.on_enable_tray_icon_toggled is already good.
                    # This one can be more direct about the immediate action.
                    self.show_status_message("System tray icon hidden for this session.", "info", 3000)
        else:
            print("DEBUG: Tray icon not available when trying to set visibility.")


    @Slot(str, str, int)
    def show_status_message(self, message, msg_type="info", duration=7000):
        """Displays a message in the status bar."""
        print(f"Status Update ({msg_type}): {message}")
        
        status_color = ""
        # Ensure current_theme_path is initialized
        if not hasattr(self, 'current_theme_path') or not self.current_theme_path:
             # Ensure settings_manager is available or load default directly
            if hasattr(self, 'settings_manager') and self.settings_manager:
                initial_theme_name = self.settings_manager.load_setting(SETTING_THEME, "dark")
                self.current_theme_path = DARK_STYLE_PATH if initial_theme_name == "dark" else LIGHT_STYLE_PATH
            else: # Fallback if called before settings_manager fully init (should not happen in normal flow)
                self.current_theme_path = DARK_STYLE_PATH # Default to dark path

        theme_path_to_check = self.current_theme_path

        if theme_path_to_check == DARK_STYLE_PATH:
            if msg_type == "error": status_color = "color: #E74C3C;" # Red
            elif msg_type == "success": status_color = "color: #2ECC71;" # Green
            else: status_color = "color: #ECF0F1;" # Default light text
        else: # Light theme (theme_path_to_check == LIGHT_STYLE_PATH)
            if msg_type == "error": status_color = "color: #C0392B;" # Darker Red
            elif msg_type == "success": status_color = "color: #27AE60;" # Darker Green
            else: status_color = "color: #2C3E50;" # Default dark text
        
        self.status_bar.setStyleSheet(status_color)
        self.status_bar.showMessage(message, duration if duration > 0 else 0) # Show indefinitely if duration is 0 or less

    @Slot(str, str, int) # Slot for direct connection from other pages
    def show_status_message_slot(self, message, msg_type="info", duration=7000):
        self.show_status_message(message, msg_type, duration)

if __name__ == "__main__":
    import os # Ensure os is imported
    # Fix for Wayland:
    # Check if running under Wayland and XDG_SESSION_TYPE is not set
    # This is a common workaround for Qt apps on some Wayland setups.
    if os.environ.get("XDG_SESSION_TYPE") == "wayland":
        if not os.environ.get("QT_QPA_PLATFORM"):
            # os.environ["QT_QPA_PLATFORM"] = "wayland" # Enable this if 'wayland' is preferred and works
            pass # Or let Qt auto-detect; often works. Forcing xcb might also be an option if wayland native is problematic.

    app = QApplication(sys.argv)
    # Ensure the application doesn't quit when the last window is closed, if using tray icon extensively
    app.setQuitOnLastWindowClosed(False) # Modify based on desired tray behavior
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
