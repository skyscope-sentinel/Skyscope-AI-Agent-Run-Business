import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFrame, QStatusBar, QSystemTrayIcon, QMenu, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, QFile, QTextStream, Slot
from PySide6.QtGui import QColor, QPalette, QIcon, QAction

from .model_hub_page import ModelHubPage
from .settings_page import SettingsPage, SETTING_THEME, SETTING_ACRYLIC_EFFECT, SETTING_AUTOSTART, SETTING_MINIMIZE_TO_TRAY # Added keys
from .settings_manager import SettingsManager


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
            self.label.setText("Manage and Configure Your AI Agents")
            info_label = QLabel("This section will allow you to start, stop, configure, and monitor various AI agents. You'll be able to assign models, set tasks, and view their operational logs.")
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("font-size: 14px; color: #777777; margin: 20px;")
            layout.addWidget(info_label)
            add_agent_btn = QPushButton(QIcon.fromTheme("list-add"), "Add New Agent (Placeholder)")
            add_agent_btn.setToolTip("Define and configure a new AI agent.")
            add_agent_btn.setStyleSheet("QPushButton { text-align: center; padding-left: 0px; }") # Center text when icon is present
            layout.addWidget(add_agent_btn, 0, Qt.AlignCenter)


        elif name == "Log Stream":
            self.label.setText("Centralized Log Monitoring")
            info_label = QLabel("View real-time logs from the application, Ollama service, and all active agents. Filter by source, log level, and search for specific messages.")
            info_label.setWordWrap(True)
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("font-size: 14px; color: #777777; margin: 20px;")
            layout.addWidget(info_label)
            filter_combo = QComboBox()
            filter_combo.addItems(["All Logs", "Application Logs", "Ollama Logs", "Agent Logs"])
            filter_combo.setToolTip("Filter logs by source.")
            layout.addWidget(filter_combo, 0, Qt.AlignCenter)


        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skyscope Sentinel Intelligence")
        self.setGeometry(100, 100, 1200, 700)
        
        # For better rounded corners, especially if planning for custom title bar later:
        # self.setAttribute(Qt.WA_TranslucentBackground, True)
        # self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        # However, this requires custom handling for window dragging and min/max/close buttons.
        # For now, we rely on QSS border-radius for the central widget.

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
        self.sections = ["Dashboard", "Agent Control", "Model Hub", "Log Stream", "Settings"]
        
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
            "Model Hub": "drive-harddisk", # or "applications-internet"
            "Log Stream": "document-view", 
            "Settings": "preferences-configure"
        }
        tooltips = {
            "Dashboard": "View system overview and key metrics",
            "Agent Control": "Manage and configure AI agents",
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
                self.content_area.addWidget(self.settings_page)
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
            elif isinstance(widget, PlaceholderPage) and widget.label.text() == target_widget_label:
                current_page_matches = True

            if current_page_matches:
                self.content_area.setCurrentIndex(i)
                page_found = True
                break # Found the page
        
        if page_found:
            for name, btn in self.nav_buttons.items():
                btn.setChecked(name == section_name)
            print(f"Switched to {section_name}")
            self.show_status_message(f"{section_name} page loaded.", "info", 3000)
        else:
            print(f"Warning: Page for section '{section_name}' not found.")


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
