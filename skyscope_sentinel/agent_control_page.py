import sys
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QGroupBox
)
from PySide6.QtCore import Signal, QThread
from PySide6.QtGui import QIcon

from agent_swarm_manager import agent_swarm_manager

class AgentControlPage(QWidget):
    status_message_requested = Signal(str, str, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("agentControlPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        title_label = QLabel("Agent Swarm Control")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # --- Swarm Control Buttons ---
        swarm_control_group = QGroupBox("Swarm Control")
        swarm_control_layout = QHBoxLayout(swarm_control_group)

        self.start_swarm_button = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Swarm")
        self.start_swarm_button.clicked.connect(self.start_swarm)
        swarm_control_layout.addWidget(self.start_swarm_button)

        self.stop_swarm_button = QPushButton(QIcon.fromTheme("media-playback-stop"), "Stop Swarm")
        self.stop_swarm_button.clicked.connect(self.stop_swarm)
        self.stop_swarm_button.setEnabled(False)
        swarm_control_layout.addWidget(self.stop_swarm_button)

        layout.addWidget(swarm_control_group)

        # --- Swarm Status ---
        swarm_status_group = QGroupBox("Swarm Status")
        swarm_status_layout = QVBoxLayout(swarm_status_group)
        self.swarm_status_display = QTextEdit()
        self.swarm_status_display.setReadOnly(True)
        self.swarm_status_display.setPlaceholderText("Swarm status will appear here...")
        swarm_status_layout.addWidget(self.swarm_status_display)
        layout.addWidget(swarm_status_group)

        # --- Log Viewer ---
        log_viewer_group = QGroupBox("Log Viewer")
        log_viewer_layout = QVBoxLayout(log_viewer_group)
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setPlaceholderText("Real-time logs will appear here...")
        log_viewer_layout.addWidget(self.log_viewer)
        layout.addWidget(log_viewer_group)

        self.swarm_thread = None

    def start_swarm(self):
        if agent_swarm_manager.running:
            self.status_message_requested.emit("Swarm is already running.", "warning", 3000)
            return

        self.start_swarm_button.setEnabled(False)
        self.stop_swarm_button.setEnabled(True)
        self.status_message_requested.emit("Starting agent swarm...", "info", 0)

        self.swarm_thread = SwarmRunnerThread()
        self.swarm_thread.status_updated.connect(self.update_swarm_status)
        self.swarm_thread.log_message.connect(self.update_log_viewer)
        self.swarm_thread.start()

    def stop_swarm(self):
        if not agent_swarm_manager.running:
            self.status_message_requested.emit("Swarm is not running.", "warning", 3000)
            return

        self.start_swarm_button.setEnabled(True)
        self.stop_swarm_button.setEnabled(False)
        self.status_message_requested.emit("Stopping agent swarm...", "info", 0)

        agent_swarm_manager.stop()
        if self.swarm_thread and self.swarm_thread.isRunning():
            self.swarm_thread.quit()
            self.swarm_thread.wait()

        self.status_message_requested.emit("Swarm stopped.", "success", 5000)

    def update_swarm_status(self, status):
        self.swarm_status_display.setText(status)

    def update_log_viewer(self, message):
        self.log_viewer.append(message)


class SwarmRunnerThread(QThread):
    status_updated = Signal(str)
    log_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        # Redirect stdout to capture logs
        class QtLogHandler:
            def __init__(self, log_signal):
                self.log_signal = log_signal
            def write(self, text):
                self.log_signal.emit(text.strip())
            def flush(self):
                pass

        sys.stdout = QtLogHandler(self.log_message)

        agent_swarm_manager.start(num_workers=10)

        while agent_swarm_manager.running:
            status = agent_swarm_manager.get_earnings_summary()
            status_text = ""
            for key, value in status.items():
                if isinstance(value, dict):
                    status_text += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        status_text += f"  {sub_key}: {sub_value}\n"
                else:
                    status_text += f"{key}: {value}\n"
            self.status_updated.emit(status_text)
            self.msleep(5000)

        sys.stdout = sys.__stdout__ # Restore stdout
