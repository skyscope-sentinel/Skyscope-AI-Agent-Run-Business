import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QStatusBar, QSystemTrayIcon, QMenu, QGridLayout, QSizePolicy,
    QListWidget, QGroupBox, QComboBox, QTextEdit, QLineEdit # Added QTextEdit, QLineEdit for Log Stream
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

# AIAgent Class Definition
class AIAgent:
    def __init__(self, name: str, agent_type: str, status: str = "Offline", config: dict = None):
        self.name = name
        self.type = agent_type
        self.status = status
        self.config = config if config else {}
        # Store a unique ID if needed, for now name is unique identifier
        # self.id = str(uuid.uuid4())

    def __repr__(self):
        return f"AIAgent(name='{self.name}', type='{self.type}', status='{self.status}')"


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
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to start agent: {agent.name} (Type: {agent.type}, Status: {agent.status})")
            agent.status = "Running"
            current_item.setText(f"{agent.name} ({agent.status})")
            self.show_status_message(f"Agent '{agent.name}' started (simulated).", "success")
        else:
            print("No agent selected to start.")
            self.show_status_message("No agent selected.", "warning")

    def stop_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to stop agent: {agent.name}")
            agent.status = "Offline"
            current_item.setText(f"{agent.name} ({agent.status})")
            self.show_status_message(f"Agent '{agent.name}' stopped (simulated).", "info")
        else:
            print("No agent selected to stop.")
            self.show_status_message("No agent selected.", "warning")

    def configure_selected_agent(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to configure agent: {agent.name}")
            print(f"Current config: {agent.config}")
            self.show_status_message(f"Configuration for '{agent.name}' would open here.", "info")
        else:
            print("No agent selected to configure.")
            self.show_status_message("No agent selected.", "warning")

    def view_agent_logs(self):
        current_item = self.agent_list_widget.currentItem()
        if current_item:
            agent = current_item.data(Qt.UserRole)
            print(f"Attempting to view logs for agent: {agent.name}")
            self.show_status_message(f"Log view for '{agent.name}' would open here.", "info")
        else:
            print("No agent selected to view logs.")
            self.show_status_message("No agent selected.", "warning")

    def add_new_agent(self):
        print("Attempting to add a new agent.")
        new_agent_id = len(self.agents) + 1
        new_agent = AIAgent(name=f"New Sample Agent {new_agent_id}", agent_type="SampleType", status="Offline")
        self.agents.append(new_agent)
        self.agent_list_widget.addItem(f"{new_agent.name} ({new_agent.status})")
        new_list_item = self.agent_list_widget.item(self.agent_list_widget.count() - 1)
        new_list_item.setData(Qt.UserRole, new_agent)
        self.show_status_message(f"New agent '{new_agent.name}' added (simulated).", "success")

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
        self.sections = ["Dashboard", "Agent Control", "Video Tools", "Model Hub", "Log Stream", "Settings"] # Added "Video Tools"
        
        app_title_label = QLabel("Skyscope Sentinel")
        app_title_label.setAlignment(Qt.AlignCenter)
        app_title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding-bottom: 10px; margin-top: 5px;")
        self.sidebar_layout.addWidget(app_title_label)

        # Define icons for sidebar items (using QIcon.fromTheme for now)
        # These might not show up on Windows if a proper icon theme isn't installed or if the names are Linux-specific.
        # For production, embedding actual icon files (SVGs) would be more reliable.
        icon_map = {
            "Dashboard": "view-dashboard",  # Common theme name
            "Agent Control": "applications-system", # or "preferences-system"
            "Video Tools": "applications-multimedia", # Icon for video tools
            "Model Hub": "drive-harddisk", # or "applications-internet"
            "Log Stream": "document-view", # Changed from "text-x-generic"
            "Settings": "preferences-configure"
        }
        tooltips = {
            "Dashboard": "View system overview and key metrics",
            "Agent Control": "Manage and configure AI agents",
            "Video Tools": "Access video processing utilities", # Tooltip for Video Tools
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
                self.video_agent_page.status_message_requested.connect(self.show_status_message) # Connect the signal
                self.content_area.addWidget(self.video_agent_page)
            else:
                page = PlaceholderPage(section_name)
                self.content_area.addWidget(page)
        
        self.sidebar_layout.addStretch()

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
        self.load_initial_settings()
        
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
            elif section_name == "Video Tools" and isinstance(widget, VideoAgentPage): # Condition for VideoAgentPage
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
        theme_path = getattr(self, 'current_theme_path', DARK_STYLE_PATH)

        if theme_path == DARK_STYLE_PATH:
            if msg_type == "error": status_color = "color: #E74C3C;" # Red
            elif msg_type == "success": status_color = "color: #2ECC71;" # Green
            else: status_color = "color: #ECF0F1;" # Default light text
        else: # Light theme
            if msg_type == "error": status_color = "color: #C0392B;" # Darker Red
            elif msg_type == "success": status_color = "color: #27AE60;" # Darker Green
            else: status_color = "color: #2C3E50;" # Default dark text
        
        self.status_bar.setStyleSheet(status_color)
        self.status_bar.showMessage(message, duration if duration > 0 else 0) # Show indefinitely if duration is 0 or less

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Ensure the application doesn't quit when the last window is closed, if using tray icon extensively
    app.setQuitOnLastWindowClosed(False) # Modify based on desired tray behavior
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
