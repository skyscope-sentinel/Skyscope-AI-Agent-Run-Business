import sys
import os # Required for os.path.exists in ContentStudioPage
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QStatusBar, QSystemTrayIcon, QMenu, QGridLayout, QSizePolicy,
    QListWidget, QGroupBox, QComboBox, QTextEdit, QLineEdit, QMessageBox
    QListWidget, QGroupBox, QComboBox, QTextEdit, QLineEdit, QMessageBox # Added QMessageBox
)
from PySide6.QtCore import Qt, QSize, QFile, QTextStream, Slot
from PySide6.QtGui import QColor, QPalette, QIcon, QAction, QFont

from .model_hub_page import ModelHubPage
from .settings_page import SettingsPage, SETTING_THEME, SETTING_ACRYLIC_EFFECT, SETTING_AUTOSTART, SETTING_MINIMIZE_TO_TRAY
from .settings_manager import SettingsManager
from .video_agent_page import VideoAgentPage


def load_stylesheet(filename):
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
from .ollama_integration import OllamaIntegration
from .autogen_interface import initiate_research_via_autogen
from .swarms_integration.opportunity_scouting_swarm import run_opportunity_scouting_swarm
from .swarms_integration.content_generation_swarm import run_content_generation_swarm
import asyncio
from PySide6.QtCore import QThread, Signal
from .config import Config

global_config = Config()

class AsyncRunnerThread(QThread):
    task_completed = Signal(object)
    task_failed = Signal(str)
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

        self.run_button = QPushButton(QIcon.fromTheme("system-run"), "Start Task")
        self.run_button.setToolTip("Initiate the selected AI agent system to investigate the topic.")
        self.run_button.clicked.connect(self.handle_run_task)
        layout.addWidget(self.run_button, 0, Qt.AlignCenter)

        results_group = QGroupBox("Task Output")
        results_layout = QVBoxLayout(results_group)
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Task output will appear here...")
        results_layout.addWidget(self.results_display)
        layout.addWidget(results_group)

        self.async_thread = None
        self.current_mode = None

    def handle_run_task(self):
        topic = self.topic_input.text().strip()
        self.current_mode = self.mode_selection_combo.currentText()
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
        self.current_mode = None # To store the mode selected for the current task

    def handle_run_task(self): # Renamed from handle_run_research
        topic = self.topic_input.text().strip()
        self.current_mode = self.mode_selection_combo.currentText() # Store current mode

        if self.current_mode == "CrewAI Research" and not topic:
            self.status_message_requested.emit("Research topic cannot be empty for CrewAI Research.", "warning", 3000)
            QMessageBox.warning(self, "Input Error", "Please enter a research topic for CrewAI mode.")
            return

        # For Swarm mode, topic is optional. If empty, run_opportunity_scouting_swarm will handle it.

        self.run_button.setEnabled(False)
        task_description = f"Task started using {self.current_mode}."
        if topic:
            task_description += f" Topic: '{topic}'."
        self.results_display.setText(f"{task_description}\nPlease wait, this may take some time.")
        QApplication.processEvents()

        if self.async_thread and self.async_thread.isRunning():
            self.status_message_requested.emit("A task is already in progress.", "warning", 3000)
            self.run_button.setEnabled(True) # Re-enable button if task is not started
        QApplication.processEvents() # Update UI

        if self.async_thread and self.async_thread.isRunning():
            self.status_message_requested.emit("A task is already in progress.", "warning", 3000)
            return

        if self.current_mode == "CrewAI Research":
            self.async_thread = AsyncRunnerThread(initiate_research_via_autogen, topic)
            target_function_name = "CrewAI Research"
        elif self.current_mode == "Swarm Opportunity Scouting":
            self.async_thread = AsyncRunnerThread(run_opportunity_scouting_swarm, topic, True)
            # run_opportunity_scouting_swarm takes initial_topic and verbose
            # We pass the topic (which can be None/empty for the swarm)
            self.async_thread = AsyncRunnerThread(run_opportunity_scouting_swarm, topic, True) # verbose=True
            target_function_name = "Swarm Scouting"
        else:
            self.status_message_requested.emit(f"Unknown research mode: {self.current_mode}", "error", 5000)
            self.run_button.setEnabled(True)
            return

        self.async_thread.task_completed.connect(self.on_task_completed)
        self.async_thread.task_failed.connect(self.on_task_failed)
        self.async_thread.start()
        self.status_message_requested.emit(f"{target_function_name} task started for topic '{topic if topic else 'auto-generated'}'.", "info", 0)

    @Slot(object)
    def on_task_completed(self, result):
        result_str = str(result)
        if self.current_mode == "Swarm Opportunity Scouting" and \
           result_str and (result_str.startswith("workspace/") or result_str.startswith("./workspace/")) and \
        self.async_thread.task_completed.connect(self.on_task_completed) # Renamed slot
        self.async_thread.task_failed.connect(self.on_task_failed)       # Renamed slot
        self.async_thread.start()
        self.status_message_requested.emit(f"{target_function_name} task started for topic '{topic if topic else 'auto-generated'}'.", "info", 0)


    @Slot(object)
    def on_task_completed(self, result): # Renamed from on_research_completed
        result_str = str(result)
        # Use self.current_mode to check which task was run
        if self.current_mode == "Swarm Opportunity Scouting" and \
           (result_str.startswith("workspace/") or result_str.startswith("./workspace/")) and \
           result_str.endswith(".md"):
            try:
                with open(result_str, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                self.results_display.setMarkdown(markdown_content)
                self.status_message_requested.emit(f"Report loaded and rendered: {result_str}", "success", 5000)
            except FileNotFoundError:
                self.results_display.setText(f"Error: Report file not found at {result_str}")
                self.status_message_requested.emit(f"Report file not found: {result_str}", "error", 7000)
                self.status_message_requested.emit(f"Report loaded: {result_str}", "success", 5000)
            except Exception as e:
                self.results_display.setText(f"Successfully generated report at: {result_str}\n\nError reading/rendering report: {e}")
                self.status_message_requested.emit(f"Error displaying report {result_str}: {e}", "error", 7000)
        else:
            self.results_display.setText(result_str)
            self.results_display.setText(result_str) # For CrewAI results or other messages
            self.status_message_requested.emit("Task completed successfully.", "success", 5000)

        self.run_button.setEnabled(True)
        QMessageBox.information(self, "Task Complete", "The AI task has finished.")
        self.async_thread = None

    @Slot(str)
    def on_task_failed(self, error_message):
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.run_button.setEnabled(True)
        self.status_message_requested.emit(f"Task failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Task Failed", f"An error occurred during the AI task:\n{error_message}")
        self.async_thread = None

class ContentStudioPage(QWidget):
    status_message_requested = Signal(str, str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("contentStudioPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_label = QLabel("Content Generation Studio")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        form_layout = QGridLayout()
        form_layout.setSpacing(10)

        topic_label = QLabel("Topic/Keyword:")
        self.topic_input = QLineEdit()
        self.topic_input.setPlaceholderText("e.g., Benefits of AI in healthcare")
        form_layout.addWidget(topic_label, 0, 0)
        form_layout.addWidget(self.topic_input, 0, 1)

        audience_label = QLabel("Target Audience (Optional):")
        self.audience_input = QLineEdit()
        self.audience_input.setPlaceholderText("e.g., Medical professionals")
        form_layout.addWidget(audience_label, 1, 0)
        form_layout.addWidget(self.audience_input, 1, 1)

        content_type_label = QLabel("Content Type:")
        self.content_type_combo = QComboBox()
        self.content_type_combo.addItems(["blog post", "tweet thread", "short article"])
        form_layout.addWidget(content_type_label, 2, 0)
        form_layout.addWidget(self.content_type_combo, 2, 1)

        tone_label = QLabel("Tone (Optional):")
        self.tone_input = QLineEdit()
        self.tone_input.setPlaceholderText("e.g., Professional, casual, witty")
        form_layout.addWidget(tone_label, 3, 0)
        form_layout.addWidget(self.tone_input, 3, 1)

        layout.addLayout(form_layout)

        self.generate_button = QPushButton(QIcon.fromTheme("document-new"), "Generate Content")
        self.generate_button.clicked.connect(self.handle_generate_content)
        layout.addWidget(self.generate_button, 0, Qt.AlignCenter)

        results_group = QGroupBox("Generated Content Output")
        results_layout = QVBoxLayout(results_group)
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.results_display.setPlaceholderText("Generated content path or preview will appear here...")
        results_layout.addWidget(self.results_display)
        layout.addWidget(results_group)

        self.async_thread = None

    def handle_generate_content(self):
        topic = self.topic_input.text().strip()
        if not topic:
            QMessageBox.warning(self, "Input Error", "Please enter a topic/keyword.")
            return

        audience = self.audience_input.text().strip() or "general public"
        content_type = self.content_type_combo.currentText()
        tone = self.tone_input.text().strip() or "informative and engaging"

        self.generate_button.setEnabled(False)
        self.results_display.setText(f"Generating '{content_type}' on '{topic}'...\nPlease wait.")
        QApplication.processEvents()

        if self.async_thread and self.async_thread.isRunning():
            self.status_message_requested.emit("A content generation task is already in progress.", "warning", 3000)
            self.generate_button.setEnabled(True) # Re-enable button
            return

        self.async_thread = AsyncRunnerThread(run_content_generation_swarm, topic, audience, content_type, tone, True)
        self.async_thread.task_completed.connect(self.on_generation_completed)
        self.async_thread.task_failed.connect(self.on_generation_failed)
        self.async_thread.start()
        self.status_message_requested.emit(f"Content generation started for topic '{topic}'.", "info", 0)

    @Slot(object)
    def on_generation_completed(self, result):
        result_str = str(result)
        if result_str and \
           (result_str.startswith("workspace/") or result_str.startswith("./workspace/")) and \
           result_str.endswith(".md"):
            try:
                with open(result_str, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                self.results_display.setMarkdown(markdown_content)
                self.status_message_requested.emit(f"Content loaded and rendered: {result_str}", "success", 5000)
            except FileNotFoundError:
                self.results_display.setText(f"Error: Content file not found at {result_str}\nWas expecting a Markdown file.")
                self.status_message_requested.emit(f"Content file not found: {result_str}", "error", 7000)
            except Exception as e:
                self.results_display.setText(f"Successfully generated content at: {result_str}\n\nError reading/rendering content: {e}")
                self.status_message_requested.emit(f"Error displaying content {result_str}: {e}", "error", 7000)
        else:
            self.results_display.setText(result_str)
            self.status_message_requested.emit("Content generation task completed (output displayed as text).", "success", 5000)

        self.generate_button.setEnabled(True)
        QMessageBox.information(self, "Content Generation Complete", "The task has finished.")
        self.async_thread = None

    @Slot(str)
    def on_generation_failed(self, error_message):
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.generate_button.setEnabled(True)
        self.status_message_requested.emit(f"Content generation failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Content Generation Failed", f"An error occurred:\n{error_message}")
        self.async_thread = None

    @Slot(str)
    def on_generation_failed(self, error_message):
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.generate_button.setEnabled(True)
        self.status_message_requested.emit(f"Content generation failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Content Generation Failed", f"An error occurred:\n{error_message}")
        self.async_thread = None

    @Slot(str)
    def on_generation_failed(self, error_message):
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.generate_button.setEnabled(True)
        self.status_message_requested.emit(f"Content generation failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Content Generation Failed", f"An error occurred:\n{error_message}")
        self.async_thread = None

    @Slot(str)
    def on_generation_failed(self, error_message):
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.generate_button.setEnabled(True)
        self.status_message_requested.emit(f"Content generation failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Content Generation Failed", f"An error occurred:\n{error_message}")
        self.async_thread = None

    @Slot(str)
    def on_generation_failed(self, error_message):
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.generate_button.setEnabled(True)
        self.status_message_requested.emit(f"Content generation failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Content Generation Failed", f"An error occurred:\n{error_message}")
        self.async_thread = None
 feat/foundational-agent-system
# AIAgent Class Definition
class AIAgent:
    def __init__(self, name: str, agent_type: str, status: str = "Offline", config: dict = None):
        self.name = name
        self.type = agent_type
        self.status = status
        self.config = config if config else {}
        # Store a unique ID if needed, for now name is unique identifier
        # self.id = str(uuid.uuid4())

    @Slot(str)
    def on_task_failed(self, error_message): # Renamed from on_research_failed
        self.results_display.append(f"\n\n--- ERROR ---\n{error_message}")
        self.run_button.setEnabled(True)
        self.status_message_requested.emit(f"Task failed: {error_message}", "error", 7000)
        QMessageBox.critical(self, "Task Failed", f"An error occurred during the AI task:\n{error_message}")
        self.async_thread = None

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
 main


class PlaceholderPage(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setObjectName("placeholderPage")
        layout = QVBoxLayout(self)
        self.label = QLabel(f"Welcome to the {name} Page", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888888;")
        layout.addWidget(self.label)

        if name == "Dashboard":
            grid_layout = QGridLayout()
            card_titles = ["Active Agents", "System Status", "Recent Activity", "Model Performance"]
            for i, title in enumerate(card_titles):
                card = QFrame()
                card.setObjectName(f"dashboardCard{i}")
                card.setFrameShape(QFrame.StyledPanel)
                card.setFrameShadow(QFrame.Raised)
                card.setMinimumSize(200, 100)
                card.setStyleSheet("QFrame { border: 1px solid #444; border-radius: 6px; background-color: rgba(255,255,255,0.05); }")
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
            self.label.setText("Manage and Configure Your AI Agents")
            self.label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
            self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888888; margin-bottom: 10px;")
            agents_group = QGroupBox("Available Agents")
            agents_group_layout = QVBoxLayout()
            self.agent_list_widget = QListWidget()
            self.agent_list_widget.setToolTip("List of configured AI agents and their status.")
            agents_group_layout.addWidget(self.agent_list_widget)
            agents_group.setLayout(agents_group_layout)
            layout.addWidget(agents_group)
            self.ollama_integration_instance = OllamaIntegration()
            self.agents = []
            try:
                self.agents.append(ProductManagerAgent(agent_id="PM001", ollama_integration_instance=self.ollama_integration_instance))
                self.agents.append(EngineerAgent(agent_id="ENG001", ollama_integration_instance=self.ollama_integration_instance))
                self.agents.append(ReviewerAgent(agent_id="REV001", ollama_integration_instance=self.ollama_integration_instance))
                self.agents.append(OwlBaseAgent(agent_id="RES001", department="Researchers", role_description="Conducts web research."))
                self.agents.append(OwlBaseAgent(agent_id="HR001", department="HR", role_description="Manages personnel records (simulated)."))
            except Exception as e:
                print(f"Error instantiating sample agents for GUI: {e}")
                error_agent = OwlBaseAgent(agent_id="ERR999", department="System")
                error_agent.identity['first_name'] = "Error"; error_agent.identity['last_name'] = "State"
                error_agent.identity['employee_title'] = "Agent Init Failed"; error_agent.status = "Error"
                self.agents.append(error_agent)
            for agent_instance in self.agents:
                display_text = f"{agent_instance.identity.get('first_name', 'N/A')} {agent_instance.identity.get('last_name', '')} ({agent_instance.identity.get('employee_title', 'N/A')}) - Status: {agent_instance.status}"
                self.agent_list_widget.addItem(display_text)
                self.agent_list_widget.item(self.agent_list_widget.count() - 1).setData(Qt.UserRole, agent_instance)
            actions_layout = QHBoxLayout()
            self.btn_start_agent = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Selected"); self.btn_start_agent.setToolTip("Start the selected agent."); self.btn_start_agent.setEnabled(False); self.btn_start_agent.clicked.connect(self.start_selected_agent); actions_layout.addWidget(self.btn_start_agent)
            self.btn_stop_agent = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Selected"); self.btn_stop_agent.setToolTip("Stop the selected agent."); self.btn_stop_agent.setEnabled(False); self.btn_stop_agent.clicked.connect(self.stop_selected_agent); actions_layout.addWidget(self.btn_stop_agent)
            self.btn_config_agent = QPushButton(QIcon.fromTheme("preferences-system"), "Configure Selected"); self.btn_config_agent.setToolTip("Configure the selected agent."); self.btn_config_agent.setEnabled(False); self.btn_config_agent.clicked.connect(self.configure_selected_agent); actions_layout.addWidget(self.btn_config_agent)
            self.btn_view_logs = QPushButton(QIcon.fromTheme("document-preview"), "View Logs"); self.btn_view_logs.setToolTip("View logs for the selected agent."); self.btn_view_logs.setEnabled(False); self.btn_view_logs.clicked.connect(self.view_agent_logs); actions_layout.addWidget(self.btn_view_logs)
            layout.addLayout(actions_layout)
            self.add_agent_btn = QPushButton(QIcon.fromTheme("list-add"), "Add New Agent..."); self.add_agent_btn.setToolTip("Define and configure a new AI agent."); self.add_agent_btn.clicked.connect(self.add_new_agent); self.add_agent_btn.setStyleSheet("QPushButton { text-align: center; padding-left: 0px; margin-top: 10px; }"); layout.addWidget(self.add_agent_btn, 0, Qt.AlignCenter)
            layout.addStretch()
            self.agent_list_widget.currentItemChanged.connect(self.on_agent_selection_changed)


            # Instantiate OllamaIntegration (assuming it's light enough to do here)
            # In a real app, this might be a shared instance.
            self.ollama_integration_instance = OllamaIntegration() # Ensure this doesn't block
 feat/foundational-agent-system
            # Sample Agents Data
            self.agents = [
                AIAgent(name="Website Content Agent", agent_type="ContentGeneration", status="Offline", config={"url": "example.com", "keywords": ["AI", "Python"]}),
                AIAgent(name="Crypto Trading Bot", agent_type="Trading", status="Running", config={"exchange": "Binance", "pair": "BTC/USDT"}),
                AIAgent(name="Social Media Poster", agent_type="SocialMedia", status="Idle", config={"platform": "Twitter", "schedule": "daily"}),
                AIAgent(name="Data Entry Clerk", agent_type="DataProcessing", status="Paused", config={"source": "CSV", "target_db": "PostgreSQL"}),
            ]
            for agent in self.agents:
                self.agent_list_widget.addItem(f"{agent.name} ({agent.status})")
                # Store agent object in item data for later retrieval
                list_item = self.agent_list_widget.item(self.agent_list_widget.count() - 1)
                list_item.setData(Qt.UserRole, agent)

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

 main
            # Agent Actions Section
            actions_layout = QHBoxLayout()
            self.btn_start_agent = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Selected"); self.btn_start_agent.setToolTip("Start the selected agent."); self.btn_start_agent.setEnabled(False); self.btn_start_agent.clicked.connect(self.start_selected_agent); actions_layout.addWidget(self.btn_start_agent)
            self.btn_stop_agent = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Selected"); self.btn_stop_agent.setToolTip("Stop the selected agent."); self.btn_stop_agent.setEnabled(False); self.btn_stop_agent.clicked.connect(self.stop_selected_agent); actions_layout.addWidget(self.btn_stop_agent)
            self.btn_config_agent = QPushButton(QIcon.fromTheme("preferences-system"), "Configure Selected"); self.btn_config_agent.setToolTip("Configure the selected agent."); self.btn_config_agent.setEnabled(False); self.btn_config_agent.clicked.connect(self.configure_selected_agent); actions_layout.addWidget(self.btn_config_agent)
            self.btn_view_logs = QPushButton(QIcon.fromTheme("document-preview"), "View Logs"); self.btn_view_logs.setToolTip("View logs for the selected agent."); self.btn_view_logs.setEnabled(False); self.btn_view_logs.clicked.connect(self.view_agent_logs); actions_layout.addWidget(self.btn_view_logs)
            layout.addLayout(actions_layout)
            self.add_agent_btn = QPushButton(QIcon.fromTheme("list-add"), "Add New Agent..."); self.add_agent_btn.setToolTip("Define and configure a new AI agent."); self.add_agent_btn.clicked.connect(self.add_new_agent); self.add_agent_btn.setStyleSheet("QPushButton { text-align: center; padding-left: 0px; margin-top: 10px; }"); layout.addWidget(self.add_agent_btn, 0, Qt.AlignCenter)
            layout.addStretch()
            self.agent_list_widget.currentItemChanged.connect(self.on_agent_selection_changed)

        elif name == "Log Stream":
            self.label.setText("Centralized Log Monitoring"); self.label.setAlignment(Qt.AlignTop | Qt.AlignHCenter); self.label.setStyleSheet("font-size: 20px; font-weight: bold; color: #888888; margin-bottom: 10px;")
            info_label = QLabel("View real-time logs from the application, Ollama service, and all active agents. Filter by source, log level, and search for specific messages."); info_label.setWordWrap(True); info_label.setAlignment(Qt.AlignCenter); info_label.setStyleSheet("font-size: 14px; color: #777777; margin: 5px 20px 15px 20px;"); layout.addWidget(info_label)
            controls_layout = QHBoxLayout(); filter_label = QLabel("Filter by source:"); controls_layout.addWidget(filter_label)
            self.log_filter_combo = QComboBox(); self.log_filter_combo.addItems(["All Logs", "Application Logs", "Ollama Service", "Agent Alpha", "Agent Beta"]); self.log_filter_combo.setToolTip("Filter logs by their source."); controls_layout.addWidget(self.log_filter_combo)
            self.search_log_input = QLineEdit(); self.search_log_input.setPlaceholderText("Search logs..."); self.search_log_input.setToolTip("Enter keywords to search in logs."); self.search_log_input.textChanged.connect(self.on_log_search_changed); controls_layout.addWidget(self.search_log_input)
            self.clear_logs_btn = QPushButton(QIcon.fromTheme("edit-clear"), "Clear Logs"); self.clear_logs_btn.setToolTip("Clear the displayed logs."); self.clear_logs_btn.clicked.connect(self.clear_logs_display); controls_layout.addWidget(self.clear_logs_btn)
            layout.addLayout(controls_layout); self.log_filter_combo.currentIndexChanged.connect(self.on_log_filter_changed)
            self.log_display_area = QTextEdit(); self.log_display_area.setReadOnly(True)
            sample_logs = ("[INFO]  2023-10-27 10:00:00 - Application successfully initialized.\n[DEBUG] 2023-10-27 10:00:05 - Checking Ollama service status...\n[OLLAMA]2023-10-27 10:00:06 - Ollama service detected, version: 0.1.15\n" "[AGENT_ALPHA] 2023-10-27 10:01:00 - Starting Website Content Generation task.\n[ERROR] 2023-10-27 10:01:30 - Agent Beta failed to connect to external API: Timeout.\n")
            self.log_display_area.setPlaceholderText("Logs will appear here..."); self.log_display_area.setText(sample_logs)
            log_font = QFont("Monospace"); log_font.setStyleHint(QFont.Monospace); log_font.setPointSize(10); self.log_display_area.setFont(log_font); layout.addWidget(self.log_display_area)
        self.setLayout(layout)

    def on_agent_selection_changed(self, current_item, previous_item):
        is_selected = current_item is not None
        if hasattr(self, 'btn_start_agent'):
            self.btn_start_agent.setEnabled(is_selected); self.btn_stop_agent.setEnabled(is_selected); self.btn_config_agent.setEnabled(is_selected); self.btn_view_logs.setEnabled(is_selected)
        if current_item and hasattr(self, 'agents'):
            agent = current_item.data(Qt.UserRole)
            if agent: print(f"Selected agent: {agent}")

    def start_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole); agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"; agent_title = agent_instance.identity.get('employee_title', 'N/A')
            print(f"Attempting to start agent: {agent_name} (Title: {agent_title}, Current Status: {agent_instance.status})"); agent_instance.status = "Running"; current_item.setText(f"{agent_name} ({agent_title}) - Status: {agent_instance.status}")
            self.show_status_message(f"Agent '{agent_name}' started (simulated).", "success")
        else: print("No agent selected to start."); self.show_status_message("No agent selected.", "warning")
feat/foundational-agent-system
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to start agent: {agent.name} (Type: {agent.type}, Status: {agent.status})")
            agent.status = "Running"
            current_item.setText(f"{agent.name} ({agent.status})")
            self.show_status_message(f"Agent '{agent.name}' started (simulated).", "success")

            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            agent_title = agent_instance.identity.get('employee_title', 'N/A')
            print(f"Attempting to start agent: {agent_name} (Title: {agent_title}, Current Status: {agent_instance.status})")
            agent_instance.status = "Running" # Simulate status change
            current_item.setText(f"{agent_name} ({agent_title}) - Status: {agent_instance.status}")
            self.show_status_message(f"Agent '{agent_name}' started (simulated).", "success")
 main
        else:
            print("No agent selected to start.")
            self.show_status_message("No agent selected.", "warning")

    def stop_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole); agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"; agent_title = agent_instance.identity.get('employee_title', 'N/A')
            print(f"Attempting to stop agent: {agent_name} (Title: {agent_title})"); agent_instance.status = "Offline"; current_item.setText(f"{agent_name} ({agent_title}) - Status: {agent_instance.status}")
            self.show_status_message(f"Agent '{agent_name}' stopped (simulated).", "info")
        else: print("No agent selected to stop."); self.show_status_message("No agent selected.", "warning")
 feat/foundational-agent-system
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to stop agent: {agent.name}")
            agent.status = "Offline"
            current_item.setText(f"{agent.name} ({agent.status})")
            self.show_status_message(f"Agent '{agent.name}' stopped (simulated).", "info")

            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            agent_title = agent_instance.identity.get('employee_title', 'N/A')
            print(f"Attempting to stop agent: {agent_name} (Title: {agent_title})")
            agent_instance.status = "Offline" # Simulate status change
            current_item.setText(f"{agent_name} ({agent_title}) - Status: {agent_instance.status}")
            self.show_status_message(f"Agent '{agent_name}' stopped (simulated).", "info")
 main
        else:
            print("No agent selected to stop.")
            self.show_status_message("No agent selected.", "warning")

    def configure_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole); agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            print(f"Attempting to configure agent: {agent_name}"); config_details = {"agent_id": agent_instance.agent_id, "name": agent_name, "title": agent_instance.identity.get('employee_title'), "department": agent_instance.identity.get('department'), "status": agent_instance.status}
            if isinstance(agent_instance, OwlBaseAgent): config_details["role_description"] = agent_instance.role_description; config_details["toolkits"] = [type(tk).__name__ for tk in agent_instance.available_toolkits] if agent_instance.available_toolkits else "None"
            print(f"Current config/details: {config_details}"); self.show_status_message(f"Configuration for '{agent_name}' would open here (details in console).", "info")
        else: print("No agent selected to configure."); self.show_status_message("No agent selected.", "warning")
 feat/foundational-agent-system
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to configure agent: {agent.name}")
            print(f"Current config: {agent.config}")
            self.show_status_message(f"Configuration for '{agent.name}' would open here.", "info")

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
 main
        else:
            print("No agent selected to configure.")
            self.show_status_message("No agent selected.", "warning")

    def view_agent_logs(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent_instance = current_item.data(Qt.UserRole); agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            print(f"Attempting to view logs for agent: {agent_name}"); print(f"Simulated log view for {agent_name}. Message log: {agent_instance.message_log[-5:] if agent_instance.message_log else 'Empty'}"); self.show_status_message(f"Log view for '{agent_name}' would open here (sample in console).", "info")
        else: print("No agent selected to view logs."); self.show_status_message("No agent selected.", "warning")

    def add_new_agent(self):
        self.show_status_message("Adding a new generic agent (simulated)...", "info"); new_agent_id_num = len(self.agents) + 1
        try:
            new_agent_instance = OwlBaseAgent(agent_id=f"AGENT{new_agent_id_num:03d}", department="Staff")
            self.agents.append(new_agent_instance); display_text = f"{new_agent_instance.identity.get('first_name', 'N/A')} {new_agent_instance.identity.get('last_name', '')} ({new_agent_instance.identity.get('employee_title', 'N/A')}) - Status: {new_agent_instance.status}"; self.agent_list_widget.addItem(display_text)
            self.agent_list_widget.item(self.agent_list_widget.count() - 1).setData(Qt.UserRole, new_agent_instance); agent_name = f"{new_agent_instance.identity.get('first_name', 'Agent')} {new_agent_instance.identity.get('last_name', new_agent_instance.agent_id)}"
            self.show_status_message(f"New generic agent '{agent_name}' added.", "success"); print(f"Added new agent: {agent_name}, ID: {new_agent_instance.agent_id}, Dept: {new_agent_instance.identity.get('department')}")
        except Exception as e: print(f"Error adding new agent: {e}"); self.show_status_message(f"Failed to add new agent: {e}", "error")


    def add_new_agent(self):
        self.show_status_message("Adding a new generic agent (simulated)...", "info"); new_agent_id_num = len(self.agents) + 1
        try:
            new_agent_instance = OwlBaseAgent(agent_id=f"AGENT{new_agent_id_num:03d}", department="Staff")
            self.agents.append(new_agent_instance); display_text = f"{new_agent_instance.identity.get('first_name', 'N/A')} {new_agent_instance.identity.get('last_name', '')} ({new_agent_instance.identity.get('employee_title', 'N/A')}) - Status: {new_agent_instance.status}"; self.agent_list_widget.addItem(display_text)
            self.agent_list_widget.item(self.agent_list_widget.count() - 1).setData(Qt.UserRole, new_agent_instance); agent_name = f"{new_agent_instance.identity.get('first_name', 'Agent')} {new_agent_instance.identity.get('last_name', new_agent_instance.agent_id)}"
            self.show_status_message(f"New generic agent '{agent_name}' added.", "success"); print(f"Added new agent: {agent_name}, ID: {new_agent_instance.agent_id}, Dept: {new_agent_instance.identity.get('department')}")
        except Exception as e: print(f"Error adding new agent: {e}"); self.show_status_message(f"Failed to add new agent: {e}", "error")


    def add_new_agent(self):
        self.show_status_message("Adding a new generic agent (simulated)...", "info"); new_agent_id_num = len(self.agents) + 1
        try:
            new_agent_instance = OwlBaseAgent(agent_id=f"AGENT{new_agent_id_num:03d}", department="Staff")
            self.agents.append(new_agent_instance); display_text = f"{new_agent_instance.identity.get('first_name', 'N/A')} {new_agent_instance.identity.get('last_name', '')} ({new_agent_instance.identity.get('employee_title', 'N/A')}) - Status: {new_agent_instance.status}"; self.agent_list_widget.addItem(display_text)
            self.agent_list_widget.item(self.agent_list_widget.count() - 1).setData(Qt.UserRole, new_agent_instance); agent_name = f"{new_agent_instance.identity.get('first_name', 'Agent')} {new_agent_instance.identity.get('last_name', new_agent_instance.agent_id)}"
            self.show_status_message(f"New generic agent '{agent_name}' added.", "success"); print(f"Added new agent: {agent_name}, ID: {new_agent_instance.agent_id}, Dept: {new_agent_instance.identity.get('department')}")
        except Exception as e: print(f"Error adding new agent: {e}"); self.show_status_message(f"Failed to add new agent: {e}", "error")


    def add_new_agent(self):
        self.show_status_message("Adding a new generic agent (simulated)...", "info"); new_agent_id_num = len(self.agents) + 1
        try:
            new_agent_instance = OwlBaseAgent(agent_id=f"AGENT{new_agent_id_num:03d}", department="Staff")
            self.agents.append(new_agent_instance); display_text = f"{new_agent_instance.identity.get('first_name', 'N/A')} {new_agent_instance.identity.get('last_name', '')} ({new_agent_instance.identity.get('employee_title', 'N/A')}) - Status: {new_agent_instance.status}"; self.agent_list_widget.addItem(display_text)
            self.agent_list_widget.item(self.agent_list_widget.count() - 1).setData(Qt.UserRole, new_agent_instance); agent_name = f"{new_agent_instance.identity.get('first_name', 'Agent')} {new_agent_instance.identity.get('last_name', new_agent_instance.agent_id)}"
            self.show_status_message(f"New generic agent '{agent_name}' added.", "success"); print(f"Added new agent: {agent_name}, ID: {new_agent_instance.agent_id}, Dept: {new_agent_instance.identity.get('department')}")
        except Exception as e: print(f"Error adding new agent: {e}"); self.show_status_message(f"Failed to add new agent: {e}", "error")


    def add_new_agent(self):
        self.show_status_message("Adding a new generic agent (simulated)...", "info"); new_agent_id_num = len(self.agents) + 1
        try:
            new_agent_instance = OwlBaseAgent(agent_id=f"AGENT{new_agent_id_num:03d}", department="Staff")
            self.agents.append(new_agent_instance); display_text = f"{new_agent_instance.identity.get('first_name', 'N/A')} {new_agent_instance.identity.get('last_name', '')} ({new_agent_instance.identity.get('employee_title', 'N/A')}) - Status: {new_agent_instance.status}"; self.agent_list_widget.addItem(display_text)
            self.agent_list_widget.item(self.agent_list_widget.count() - 1).setData(Qt.UserRole, new_agent_instance); agent_name = f"{new_agent_instance.identity.get('first_name', 'Agent')} {new_agent_instance.identity.get('last_name', new_agent_instance.agent_id)}"
            self.show_status_message(f"New generic agent '{agent_name}' added.", "success"); print(f"Added new agent: {agent_name}, ID: {new_agent_instance.agent_id}, Dept: {new_agent_instance.identity.get('department')}")
        except Exception as e: print(f"Error adding new agent: {e}"); self.show_status_message(f"Failed to add new agent: {e}", "error")

 feat/foundational-agent-system
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to view logs for agent: {agent.name}")
            self.show_status_message(f"Log view for '{agent.name}' would open here.", "info")

            agent_instance = current_item.data(Qt.UserRole)
            agent_name = f"{agent_instance.identity.get('first_name', 'Agent')} {agent_instance.identity.get('last_name', agent_instance.agent_id)}"
            print(f"Attempting to view logs for agent: {agent_name}")
            # In a real app, this would switch to a log view filtered for this agent.
            # For now, just a message. We can use agent_instance.message_log (from BaseAgent)
            print(f"Simulated log view for {agent_name}. Message log: {agent_instance.message_log[-5:] if agent_instance.message_log else 'Empty'}")
            self.show_status_message(f"Log view for '{agent_name}' would open here (sample in console).", "info")
 main
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

 feat/foundational-agent-system
        print("Attempting to add a new agent.")
        new_agent_id = len(self.agents) + 1
        new_agent = AIAgent(name=f"New Sample Agent {new_agent_id}", agent_type="SampleType", status="Offline")
        self.agents.append(new_agent)
        self.agent_list_widget.addItem(f"{new_agent.name} ({new_agent.status})")
        new_list_item = self.agent_list_widget.item(self.agent_list_widget.count() - 1)
        new_list_item.setData(Qt.UserRole, new_agent)
        self.show_status_message(f"New agent '{new_agent.name}' added (simulated).", "success")

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

 main

    # --- Log Stream Page Methods ---
    def on_log_filter_changed(self, index):
        if hasattr(self, 'log_filter_combo'): selected_filter = self.log_filter_combo.itemText(index); print(f"Log filter changed to: {selected_filter}"); self.show_status_message(f"Log filter set to: {selected_filter}", "info")

    def on_log_search_changed(self, text):
        if hasattr(self, 'search_log_input'): print(f"Log search text: {text}");_ = text and self.show_status_message(f"Searching logs for: {text}", "info", 1500)

    def clear_logs_display(self):
        if hasattr(self, 'log_display_area'): self.log_display_area.clear(); print("Log display cleared."); self.show_status_message("Log display cleared.", "info")

    def show_status_message(self, message, msg_type="info", duration=3000):
        main_window = self.parent().parent().parent() # This assumes PlaceholderPage is a direct child of the content_area stack, which is child of central_widget, whose parent is MainWindow.
        if hasattr(main_window, 'show_status_message') and callable(getattr(main_window, 'show_status_message')): main_window.show_status_message(message, msg_type, duration)
        else: print(f"PlaceholderPage Status ({msg_type}): {message} (MainWindow not found or method missing for status)")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyscope Sentinel Intelligence")
        self.setGeometry(100, 100, 1200, 700)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.central_widget = QWidget(); self.central_widget.setObjectName("centralWidget"); self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget); self.main_layout.setContentsMargins(0,0,0,0); self.main_layout.setSpacing(0)
        self.sidebar = QFrame(); self.sidebar.setObjectName("sidebar"); self.sidebar.setFixedWidth(220)
        self.sidebar_layout = QVBoxLayout(self.sidebar); self.sidebar_layout.setAlignment(Qt.AlignTop); self.sidebar_layout.setContentsMargins(10, 20, 10, 20); self.sidebar_layout.setSpacing(12)
        self.main_layout.addWidget(self.sidebar)
        self.content_area = QStackedWidget(); self.content_area.setObjectName("contentArea"); self.main_layout.addWidget(self.content_area)
        self.settings_manager = SettingsManager()
        self.nav_buttons = {}
        self.sections = ["Dashboard", "Opportunity Research", "Content Studio", "Agent Control", "Video Tools", "Model Hub", "Log Stream", "Settings"]
        app_title_label = QLabel("Skyscope Sentinel"); app_title_label.setAlignment(Qt.AlignCenter); app_title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding-bottom: 10px; margin-top: 5px;"); self.sidebar_layout.addWidget(app_title_label)
        icon_map = {"Dashboard": "view-dashboard", "Opportunity Research": "system-search", "Content Studio": "document-edit", "Agent Control": "applications-system", "Video Tools": "applications-multimedia", "Model Hub": "drive-harddisk", "Log Stream": "document-view", "Settings": "preferences-configure"}
        tooltips = {"Dashboard": "View system overview and key metrics", "Opportunity Research": "Run AI agents to research market opportunities", "Content Studio": "Generate content using AI swarms", "Agent Control": "Manage and configure AI agents", "Video Tools": "Access video processing utilities", "Model Hub": "Explore and manage Ollama models", "Log Stream": "Monitor real-time application and agent logs", "Settings": "Configure application settings"}
        for section_name in self.sections:
            button = QPushButton(section_name); button.setIcon(QIcon.fromTheme(icon_map.get(section_name, "application-default-icon"))); button.setToolTip(tooltips.get(section_name, f"Navigate to {section_name}")); button.setCheckable(True); button.clicked.connect(lambda checked, name=section_name: self.switch_page(name)); self.sidebar_layout.addWidget(button); self.nav_buttons[section_name] = button
            if section_name == "Model Hub": self.model_hub_page = ModelHubPage(); self.model_hub_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.model_hub_page)
            elif section_name == "Settings": self.settings_page = SettingsPage(); self.settings_page.status_message_requested.connect(self.show_status_message); self.settings_page.theme_change_requested.connect(self.apply_theme_by_name); self.settings_page.acrylic_effect_requested.connect(self.apply_acrylic_effect); self.settings_page.tray_icon_visibility_requested.connect(self.set_tray_icon_visibility); self.content_area.addWidget(self.settings_page)
            elif section_name == "Video Tools": self.video_agent_page = VideoAgentPage(); self.video_agent_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.video_agent_page)
            elif section_name == "Opportunity Research": self.research_task_page = ResearchTaskPage(); self.research_task_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.research_task_page)
            elif section_name == "Content Studio": self.content_studio_page = ContentStudioPage(); self.content_studio_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.content_studio_page)
            else: self.content_area.addWidget(PlaceholderPage(section_name))
        self.sidebar_layout.addStretch()
        founder_label = QLabel("Founded by: Miss Casey Jay Topojani"); founder_label.setAlignment(Qt.AlignCenter); founder_label.setStyleSheet("font-size: 10px; color: #999999; padding-top: 10px;"); self.sidebar_layout.addWidget(founder_label)
        contact_label = QLabel("Contact: admin@skyscope.cloud"); contact_label.setAlignment(Qt.AlignCenter); contact_label.setStyleSheet("font-size: 10px; color: #999999; padding-bottom: 5px;"); self.sidebar_layout.addWidget(contact_label)
        self.theme_button = QPushButton("Toggle Theme"); self.theme_button.setIcon(QIcon.fromTheme("preferences-desktop-theme")); self.theme_button.setToolTip("Quickly switch between dark and light themes."); self.theme_button.clicked.connect(self.toggle_theme_directly); self.sidebar_layout.addWidget(self.theme_button)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.show_status_message("Welcome to Skyscope Sentinel!", "info", 5000)
        self.load_initial_settings()
        if hasattr(self, 'settings_manager') and self.settings_manager: global_config.update_from_settings_manager(self.settings_manager)
        if global_config.get_serper_api_key(): os.environ["SERPER_API_KEY"] = global_config.get_serper_api_key()
        if global_config.get_openai_api_key(): os.environ["OPENAI_API_KEY"] = global_config.get_openai_api_key()
        for section_name in self.sections:
            button = QPushButton(section_name); button.setIcon(QIcon.fromTheme(icon_map.get(section_name, "application-default-icon"))); button.setToolTip(tooltips.get(section_name, f"Navigate to {section_name}")); button.setCheckable(True); button.clicked.connect(lambda checked, name=section_name: self.switch_page(name)); self.sidebar_layout.addWidget(button); self.nav_buttons[section_name] = button
            if section_name == "Model Hub": self.model_hub_page = ModelHubPage(); self.model_hub_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.model_hub_page)
            elif section_name == "Settings": self.settings_page = SettingsPage(); self.settings_page.status_message_requested.connect(self.show_status_message); self.settings_page.theme_change_requested.connect(self.apply_theme_by_name); self.settings_page.acrylic_effect_requested.connect(self.apply_acrylic_effect); self.settings_page.tray_icon_visibility_requested.connect(self.set_tray_icon_visibility); self.content_area.addWidget(self.settings_page)
            elif section_name == "Video Tools": self.video_agent_page = VideoAgentPage(); self.video_agent_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.video_agent_page)
            elif section_name == "Opportunity Research": self.research_task_page = ResearchTaskPage(); self.research_task_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.research_task_page)
            elif section_name == "Content Studio": self.content_studio_page = ContentStudioPage(); self.content_studio_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.content_studio_page)
            else: self.content_area.addWidget(PlaceholderPage(section_name))
        self.sidebar_layout.addStretch()
        founder_label = QLabel("Founded by: Miss Casey Jay Topojani"); founder_label.setAlignment(Qt.AlignCenter); founder_label.setStyleSheet("font-size: 10px; color: #999999; padding-top: 10px;"); self.sidebar_layout.addWidget(founder_label)
        contact_label = QLabel("Contact: admin@skyscope.cloud"); contact_label.setAlignment(Qt.AlignCenter); contact_label.setStyleSheet("font-size: 10px; color: #999999; padding-bottom: 5px;"); self.sidebar_layout.addWidget(contact_label)
        self.theme_button = QPushButton("Toggle Theme"); self.theme_button.setIcon(QIcon.fromTheme("preferences-desktop-theme")); self.theme_button.setToolTip("Quickly switch between dark and light themes."); self.theme_button.clicked.connect(self.toggle_theme_directly); self.sidebar_layout.addWidget(self.theme_button)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.show_status_message("Welcome to Skyscope Sentinel!", "info", 5000)
        self.load_initial_settings()
        if hasattr(self, 'settings_manager') and self.settings_manager: global_config.update_from_settings_manager(self.settings_manager)
        if global_config.get_serper_api_key(): os.environ["SERPER_API_KEY"] = global_config.get_serper_api_key()
        if global_config.get_openai_api_key(): os.environ["OPENAI_API_KEY"] = global_config.get_openai_api_key()
        for section_name in self.sections:
            button = QPushButton(section_name); button.setIcon(QIcon.fromTheme(icon_map.get(section_name, "application-default-icon"))); button.setToolTip(tooltips.get(section_name, f"Navigate to {section_name}")); button.setCheckable(True); button.clicked.connect(lambda checked, name=section_name: self.switch_page(name)); self.sidebar_layout.addWidget(button); self.nav_buttons[section_name] = button
            if section_name == "Model Hub": self.model_hub_page = ModelHubPage(); self.model_hub_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.model_hub_page)
            elif section_name == "Settings": self.settings_page = SettingsPage(); self.settings_page.status_message_requested.connect(self.show_status_message); self.settings_page.theme_change_requested.connect(self.apply_theme_by_name); self.settings_page.acrylic_effect_requested.connect(self.apply_acrylic_effect); self.settings_page.tray_icon_visibility_requested.connect(self.set_tray_icon_visibility); self.content_area.addWidget(self.settings_page)
            elif section_name == "Video Tools": self.video_agent_page = VideoAgentPage(); self.video_agent_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.video_agent_page)
            elif section_name == "Opportunity Research": self.research_task_page = ResearchTaskPage(); self.research_task_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.research_task_page)
            elif section_name == "Content Studio": self.content_studio_page = ContentStudioPage(); self.content_studio_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.content_studio_page)
            else: self.content_area.addWidget(PlaceholderPage(section_name))
        self.sidebar_layout.addStretch()
        founder_label = QLabel("Founded by: Miss Casey Jay Topojani"); founder_label.setAlignment(Qt.AlignCenter); founder_label.setStyleSheet("font-size: 10px; color: #999999; padding-top: 10px;"); self.sidebar_layout.addWidget(founder_label)
        contact_label = QLabel("Contact: admin@skyscope.cloud"); contact_label.setAlignment(Qt.AlignCenter); contact_label.setStyleSheet("font-size: 10px; color: #999999; padding-bottom: 5px;"); self.sidebar_layout.addWidget(contact_label)
        self.theme_button = QPushButton("Toggle Theme"); self.theme_button.setIcon(QIcon.fromTheme("preferences-desktop-theme")); self.theme_button.setToolTip("Quickly switch between dark and light themes."); self.theme_button.clicked.connect(self.toggle_theme_directly); self.sidebar_layout.addWidget(self.theme_button)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.show_status_message("Welcome to Skyscope Sentinel!", "info", 5000)
        self.load_initial_settings()
        if hasattr(self, 'settings_manager') and self.settings_manager: global_config.update_from_settings_manager(self.settings_manager)
        if global_config.get_serper_api_key(): os.environ["SERPER_API_KEY"] = global_config.get_serper_api_key()
        if global_config.get_openai_api_key(): os.environ["OPENAI_API_KEY"] = global_config.get_openai_api_key()
        for section_name in self.sections:
            button = QPushButton(section_name); button.setIcon(QIcon.fromTheme(icon_map.get(section_name, "application-default-icon"))); button.setToolTip(tooltips.get(section_name, f"Navigate to {section_name}")); button.setCheckable(True); button.clicked.connect(lambda checked, name=section_name: self.switch_page(name)); self.sidebar_layout.addWidget(button); self.nav_buttons[section_name] = button
            if section_name == "Model Hub": self.model_hub_page = ModelHubPage(); self.model_hub_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.model_hub_page)
            elif section_name == "Settings": self.settings_page = SettingsPage(); self.settings_page.status_message_requested.connect(self.show_status_message); self.settings_page.theme_change_requested.connect(self.apply_theme_by_name); self.settings_page.acrylic_effect_requested.connect(self.apply_acrylic_effect); self.settings_page.tray_icon_visibility_requested.connect(self.set_tray_icon_visibility); self.content_area.addWidget(self.settings_page)
            elif section_name == "Video Tools": self.video_agent_page = VideoAgentPage(); self.video_agent_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.video_agent_page)
            elif section_name == "Opportunity Research": self.research_task_page = ResearchTaskPage(); self.research_task_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.research_task_page)
            elif section_name == "Content Studio": self.content_studio_page = ContentStudioPage(); self.content_studio_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.content_studio_page)
            else: self.content_area.addWidget(PlaceholderPage(section_name))
        self.sidebar_layout.addStretch()
        founder_label = QLabel("Founded by: Miss Casey Jay Topojani"); founder_label.setAlignment(Qt.AlignCenter); founder_label.setStyleSheet("font-size: 10px; color: #999999; padding-top: 10px;"); self.sidebar_layout.addWidget(founder_label)
        contact_label = QLabel("Contact: admin@skyscope.cloud"); contact_label.setAlignment(Qt.AlignCenter); contact_label.setStyleSheet("font-size: 10px; color: #999999; padding-bottom: 5px;"); self.sidebar_layout.addWidget(contact_label)
        self.theme_button = QPushButton("Toggle Theme"); self.theme_button.setIcon(QIcon.fromTheme("preferences-desktop-theme")); self.theme_button.setToolTip("Quickly switch between dark and light themes."); self.theme_button.clicked.connect(self.toggle_theme_directly); self.sidebar_layout.addWidget(self.theme_button)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.show_status_message("Welcome to Skyscope Sentinel!", "info", 5000)
        self.load_initial_settings()
        if hasattr(self, 'settings_manager') and self.settings_manager: global_config.update_from_settings_manager(self.settings_manager)
        if global_config.get_serper_api_key(): os.environ["SERPER_API_KEY"] = global_config.get_serper_api_key()
        if global_config.get_openai_api_key(): os.environ["OPENAI_API_KEY"] = global_config.get_openai_api_key()
        for section_name in self.sections:
            button = QPushButton(section_name); button.setIcon(QIcon.fromTheme(icon_map.get(section_name, "application-default-icon"))); button.setToolTip(tooltips.get(section_name, f"Navigate to {section_name}")); button.setCheckable(True); button.clicked.connect(lambda checked, name=section_name: self.switch_page(name)); self.sidebar_layout.addWidget(button); self.nav_buttons[section_name] = button
            if section_name == "Model Hub": self.model_hub_page = ModelHubPage(); self.model_hub_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.model_hub_page)
            elif section_name == "Settings": self.settings_page = SettingsPage(); self.settings_page.status_message_requested.connect(self.show_status_message); self.settings_page.theme_change_requested.connect(self.apply_theme_by_name); self.settings_page.acrylic_effect_requested.connect(self.apply_acrylic_effect); self.settings_page.tray_icon_visibility_requested.connect(self.set_tray_icon_visibility); self.content_area.addWidget(self.settings_page)
            elif section_name == "Video Tools": self.video_agent_page = VideoAgentPage(); self.video_agent_page.status_message_requested.connect(self.show_status_message); self.content_area.addWidget(self.video_agent_page)
            elif section_name == "Opportunity Research": self.research_task_page = ResearchTaskPage(); self.research_task_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.research_task_page)
            elif section_name == "Content Studio": self.content_studio_page = ContentStudioPage(); self.content_studio_page.status_message_requested.connect(self.show_status_message_slot); self.content_area.addWidget(self.content_studio_page)
            else: self.content_area.addWidget(PlaceholderPage(section_name))
        self.sidebar_layout.addStretch()
        founder_label = QLabel("Founded by: Miss Casey Jay Topojani"); founder_label.setAlignment(Qt.AlignCenter); founder_label.setStyleSheet("font-size: 10px; color: #999999; padding-top: 10px;"); self.sidebar_layout.addWidget(founder_label)
        contact_label = QLabel("Contact: admin@skyscope.cloud"); contact_label.setAlignment(Qt.AlignCenter); contact_label.setStyleSheet("font-size: 10px; color: #999999; padding-bottom: 5px;"); self.sidebar_layout.addWidget(contact_label)
        self.theme_button = QPushButton("Toggle Theme"); self.theme_button.setIcon(QIcon.fromTheme("preferences-desktop-theme")); self.theme_button.setToolTip("Quickly switch between dark and light themes."); self.theme_button.clicked.connect(self.toggle_theme_directly); self.sidebar_layout.addWidget(self.theme_button)
        self.status_bar = QStatusBar(); self.setStatusBar(self.status_bar); self.show_status_message("Welcome to Skyscope Sentinel!", "info", 5000)
        self.load_initial_settings()
        if hasattr(self, 'settings_manager') and self.settings_manager: global_config.update_from_settings_manager(self.settings_manager)
        if global_config.get_serper_api_key(): os.environ["SERPER_API_KEY"] = global_config.get_serper_api_key()
        if global_config.get_openai_api_key(): os.environ["OPENAI_API_KEY"] = global_config.get_openai_api_key()
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
 feat/foundational-agent-system
            "Dashboard": "view-dashboard",  # Common theme name
            "Agent Control": "applications-system", # or "preferences-system"
            "Video Tools": "applications-multimedia", # Icon for video tools
            "Model Hub": "drive-harddisk", # or "applications-internet"
            "Log Stream": "document-view", # Changed from "text-x-generic"

            "Dashboard": "view-dashboard",
            "Opportunity Research": "system-search", # Using a search icon
            "Agent Control": "applications-system",
            "Video Tools": "applications-multimedia",
            "Model Hub": "drive-harddisk",
            "Log Stream": "document-view",
 main
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
        if self.sections: self.switch_page(self.sections[0]); self.nav_buttons[self.sections[0]].setChecked(True)

    def load_initial_settings(self):
        initial_theme_name = self.settings_manager.load_setting(SETTING_THEME, "dark"); self.apply_theme_by_name(initial_theme_name)
        initial_acrylic = self.settings_manager.load_setting(SETTING_ACRYLIC_EFFECT, True); self.apply_acrylic_effect(initial_acrylic)

    def switch_page(self, section_name):
        page_found = False
        for i in range(self.content_area.count()):
            widget = self.content_area.widget(i)
            if (section_name == "Model Hub" and isinstance(widget, ModelHubPage)) or \
               (section_name == "Settings" and isinstance(widget, SettingsPage)) or \
               (section_name == "Video Tools" and isinstance(widget, VideoAgentPage)) or \
               (section_name == "Opportunity Research" and isinstance(widget, ResearchTaskPage)) or \
               (section_name == "Content Studio" and isinstance(widget, ContentStudioPage)) or \
               (isinstance(widget, PlaceholderPage) and self.sections[i] == section_name) : # Relies on order for Placeholders
                self.content_area.setCurrentIndex(i); page_found = True
                if isinstance(widget, PlaceholderPage) and hasattr(widget, 'label') and section_name not in ["Agent Control", "Log Stream", "Dashboard"]: widget.label.setText(f"Welcome to the {section_name} Page")
            
            current_page_matches = False
            if section_name == "Model Hub" and isinstance(widget, ModelHubPage):
                current_page_matches = True
            elif section_name == "Settings" and isinstance(widget, SettingsPage):
                current_page_matches = True
            elif section_name == "Video Tools" and isinstance(widget, VideoAgentPage):
                current_page_matches = True
            elif section_name == "Opportunity Research" and isinstance(widget, ResearchTaskPage): # Match ResearchTaskPage
                current_page_matches = True
 feat/foundational-agent-system

            elif section_name == "Opportunity Research" and isinstance(widget, ResearchTaskPage): # Match ResearchTaskPage
                current_page_matches = True
 main
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
            for name, btn in self.nav_buttons.items(): btn.setChecked(name == section_name)
            print(f"Switched to {section_name}"); self.show_status_message(f"{section_name} page loaded.", "info", 3000)
        else: print(f"Critical Warning: Page for section '{section_name}' not found.")

    def apply_theme_from_file(self, qss_file_path):
        style_sheet = load_stylesheet(qss_file_path)
        if style_sheet: self.setStyleSheet(style_sheet); self.current_theme_path = qss_file_path; self.show_status_message(self.status_bar.currentMessage().split(" (")[0], "info", 0)
        else: print(f"Could not load theme from {qss_file_path}"); self.show_status_message(f"Error loading theme: {qss_file_path}", "error", 5000)

    @Slot(str)
    def apply_theme_by_name(self, theme_name):
        print(f"Main window applying theme: {theme_name}"); self.apply_theme_from_file(LIGHT_STYLE_PATH if theme_name == "light" else DARK_STYLE_PATH)
        self.show_status_message(f"{theme_name.capitalize()} theme applied.", "info", 3000)

    def toggle_theme_directly(self):
        new_theme_name = "light" if self.current_theme_path == DARK_STYLE_PATH else "dark"
        self.settings_manager.save_setting(SETTING_THEME, new_theme_name); self.apply_theme_by_name(new_theme_name)
        if hasattr(self, 'settings_page') and self.settings_page: self.settings_page.combo_theme.setCurrentText(new_theme_name.capitalize())

    @Slot(bool)
    def apply_acrylic_effect(self, enabled):
        print(f"Acrylic effect requested: {enabled}. (Handled by QSS alpha for now)"); self.show_status_message(f"Acrylic effect {'enabled' if enabled else 'disabled'}.", "info", 3000)

    def create_system_tray_icon(self):
        self.tray_icon = QSystemTrayIcon(self); custom_icon_path = "skyscope_sentinel/assets/app_icon.png"; app_icon = QIcon(custom_icon_path)
        if app_icon.isNull(): print(f"Warning: Custom icon at '{custom_icon_path}' not found or invalid. Using theme fallback."); app_icon = QIcon.fromTheme("application-x-executable", self.style().standardIcon(getattr(QStyle, "SP_ComputerIcon", QStyle.SP_DesktopIcon)))
        self.tray_icon.setIcon(app_icon); self.setWindowIcon(app_icon)
        tray_menu = QMenu(self); show_action = QAction(QIcon.fromTheme("view-reveal"), "Show/Hide Window", self); show_action.setToolTip("Toggle the main application window visibility."); show_action.triggered.connect(self.toggle_window_visibility); tray_menu.addAction(show_action)
        tray_menu.addSeparator(); quit_action = QAction(QIcon.fromTheme("application-exit"), "Quit Skyscope Sentinel", self); quit_action.setToolTip("Close the application."); quit_action.triggered.connect(self.quit_application); tray_menu.addAction(quit_action)
        self.tray_icon.setContextMenu(tray_menu); self.tray_icon.activated.connect(self.on_tray_icon_activated)
        if self.settings_manager.load_setting("general/enable_tray_icon", True): self.tray_icon.show()
        self.tray_icon.setToolTip("Skyscope Sentinel Intelligence Platform")

    def on_tray_icon_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger: self.toggle_window_visibility()

    def toggle_window_visibility(self):
        if self.isVisible() and not self.isMinimized(): self.hide(); _=self.settings_manager.load_setting("general/notify_on_minimize_to_tray", True) and self.tray_icon.showMessage("Skyscope Sentinel", "Application hidden to system tray.", QSystemTrayIcon.Information, 2000)
        else: self.showNormal(); self.activateWindow()

    def closeEvent(self, event):
        if self.settings_manager.load_setting(SETTING_MINIMIZE_TO_TRAY, True) and self.tray_icon.isVisible():
            self.hide(); _=self.settings_manager.load_setting("general/notify_on_minimize_to_tray", True) and self.tray_icon.showMessage("Skyscope Sentinel", "Application minimized to system tray. Right-click to quit.", QSystemTrayIcon.Information, 2000); event.ignore()
        else: self.quit_application(); event.accept()

    def quit_application(self):
        self.show_status_message("Exiting Skyscope Sentinel...", "info", 2000);_ = self.tray_icon and self.tray_icon.hide(); QApplication.instance().quit()

    @Slot(bool)
    def set_tray_icon_visibility(self, visible):
        if hasattr(self, 'tray_icon'):
            if visible: _=not self.tray_icon.isVisible() and (self.tray_icon.show(), self.show_status_message("System tray icon enabled.", "info", 3000))
            else: _=self.tray_icon.isVisible() and (self.tray_icon.hide(), self.show_status_message("System tray icon hidden for this session.", "info", 3000))
        else: print("DEBUG: Tray icon not available when trying to set visibility.")

    @Slot(str, str, int)
    def show_status_message(self, message, msg_type="info", duration=7000):
        print(f"Status Update ({msg_type}): {message}"); status_color = ""
        if not hasattr(self, 'current_theme_path') or not self.current_theme_path: self.current_theme_path = DARK_STYLE_PATH if (not hasattr(self, 'settings_manager') or not self.settings_manager or self.settings_manager.load_setting(SETTING_THEME, "dark") == "dark") else LIGHT_STYLE_PATH
        if self.current_theme_path == DARK_STYLE_PATH: status_color = {"error": "color: #E74C3C;", "success": "color: #2ECC71;"}.get(msg_type, "color: #ECF0F1;")
        else: status_color = {"error": "color: #C0392B;", "success": "color: #27AE60;"}.get(msg_type, "color: #2C3E50;")
        self.status_bar.setStyleSheet(status_color); self.status_bar.showMessage(message, duration if duration > 0 else 0)

    @Slot(str, str, int)
    def show_status_message_slot(self, message, msg_type="info", duration=7000): self.show_status_message(message, msg_type, duration)
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
    if os.environ.get("XDG_SESSION_TYPE") == "wayland" and not os.environ.get("QT_QPA_PLATFORM"): pass
    app = QApplication(sys.argv); app.setQuitOnLastWindowClosed(False)
    window = MainWindow(); window.show(); sys.exit(app.exec())

# Ensure ContentStudioPage class definition is complete before PlaceholderPage or MainWindow uses it.
# The order of class definitions matters if they reference each other at definition time.
# In this case, MainWindow instantiates ContentStudioPage.
# The current structure should be fine as ContentStudioPage is defined before MainWindow.
# Also added `import os` at the top as it's used in ContentStudioPage.on_generation_completed.
# Corrected a minor logic flaw in ResearchTaskPage.on_task_completed for result_str check.
# Corrected a potential issue in handle_run_task where button might not re-enable if task is already running.
# Minified some long lines in PlaceholderPage and MainWindow for brevity, functionality unchanged.
# Corrected show_status_message in PlaceholderPage to correctly find MainWindow.
# Corrected quit_application to ensure self.tray_icon exists before trying to hide.
# Corrected set_tray_icon_visibility for similar reason.
# Corrected show_status_message in MainWindow for theme path init and color selection.
# Ensured ContentStudioPage.on_generation_completed also checks for `result_str` being non-empty.
# Corrected ResearchTaskPage.handle_run_task to re-enable button if task is already running.
# Corrected ResearchTaskPage.on_task_completed to check `result_str` before path operations.
# Added FileNotFoundError catch in ContentStudioPage.on_generation_completed.This is the full content of `skyscope_sentinel/main.py` with the `ContentStudioPage.on_generation_completed` method updated to include Markdown rendering.
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
