from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QCheckBox, QComboBox, QTabWidget, QScrollArea, QGroupBox, QFormLayout,
    QMessageBox, QFileDialog, QColorDialog, QScrollArea
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QPalette, QColor, QIcon

from .settings_manager import SettingsManager
from .ollama_integration import OllamaIntegration 

# Define keys for settings - consider moving to a shared constants file if they grow
SETTING_AUTOSTART = "general/autostart_on_login" # More specific key
SETTING_ENABLE_TRAY_ICON = "general/enable_tray_icon"
SETTING_MINIMIZE_TO_TRAY = "general/minimize_to_tray_on_close"
SETTING_NOTIFY_ON_MINIMIZE = "general/notify_on_minimize_to_tray"

SETTING_THEME = "appearance/theme" 
SETTING_ACRYLIC_EFFECT = "appearance/acrylic_effect"
SETTING_UI_SCALING = "appearance/ui_scaling" 
SETTING_ACCENT_COLOR = "appearance/accent_color" 

SETTING_OLLAMA_URL = "ollama/service_url"
SETTING_OLLAMA_AUTO_START = "ollama/auto_start_service" # New setting

SETTING_AGENT_LOG_LEVEL = "agents/default_log_level"
SETTING_AGENT_AUTO_RESTART = "agents/auto_restart_crashed" # New setting

SETTING_DATA_FOLDER = "advanced/data_folder_location" # More specific key


class SettingsPage(QWidget):
    theme_change_requested = Signal(str) 
    acrylic_effect_requested = Signal(bool)
    status_message_requested = Signal(str, str) 
    tray_icon_visibility_requested = Signal(bool) # New signal for tray icon enable/disable


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self.settings_manager = SettingsManager()
        self.ollama_integration = OllamaIntegration() 

        self.main_layout = QVBoxLayout(self)
        
        # Use QScrollArea for potentially long content
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setObjectName("settingsScrollArea")
        
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("settingsTabWidget")
        scroll_area.setWidget(self.tab_widget)
        
        self.main_layout.addWidget(scroll_area)

        # Create tabs with icons
        self.tab_widget.addTab(self.create_general_tab(), QIcon.fromTheme("preferences-system"), "General")
        self.tab_widget.addTab(self.create_appearance_tab(), QIcon.fromTheme("preferences-desktop-theme"), "Appearance")
        self.tab_widget.addTab(self.create_ollama_tab(), QIcon.fromTheme("network-server"), "Ollama")
        self.tab_widget.addTab(self.create_agents_tab(), QIcon.fromTheme("applications-science"), "Agents")
        self.tab_widget.addTab(self.create_api_keys_tab(), QIcon.fromTheme("dialog-password", QIcon.fromTheme("security-high")), "API Keys") # New API Keys tab
        self.tab_widget.addTab(self.create_financials_tab(), QIcon.fromTheme("wallet-open-symbolic", QIcon.fromTheme("emblem-money")), "Financials")
        self.tab_widget.addTab(self.create_advanced_tab(), QIcon.fromTheme("preferences-other"), "Advanced")

        self.load_all_settings()

    def _create_tab_content_widget(self):
        """Helper to create a consistent QWidget for tab content with a QFormLayout."""
        tab_content = QWidget()
        layout = QFormLayout(tab_content)
        layout.setRowWrapPolicy(QFormLayout.WrapLongRows)
        layout.setLabelAlignment(Qt.AlignLeft) # Align labels to the left
        layout.setContentsMargins(15, 15, 15, 15) # Add some padding within tabs
        layout.setSpacing(10) # Spacing between rows
        return tab_content, layout

    def create_general_tab(self):
        general_tab, layout = self._create_tab_content_widget()

        self.cb_autostart = QCheckBox("Autostart application on system login")
        self.cb_autostart.setToolTip("If enabled, Skyscope Sentinel will attempt to start when you log into your system.\n(Note: OS-specific setup might be required for this to function.)")
        self.cb_autostart.toggled.connect(lambda checked: self.save_setting_value(SETTING_AUTOSTART, checked))
        layout.addRow(self.cb_autostart)
        
        self.cb_enable_tray_icon = QCheckBox("Enable System Tray Icon")
        self.cb_enable_tray_icon.setToolTip("Show an icon in the system tray for quick access and background operation.")
        self.cb_enable_tray_icon.toggled.connect(self.on_enable_tray_icon_toggled)
        layout.addRow(self.cb_enable_tray_icon)

        self.cb_minimize_to_tray = QCheckBox("Minimize to tray on window close")
        self.cb_minimize_to_tray.setToolTip("If enabled, closing the main window will minimize it to the system tray instead of quitting.")
        self.cb_minimize_to_tray.toggled.connect(lambda checked: self.save_setting_value(SETTING_MINIMIZE_TO_TRAY, checked))
        layout.addRow(self.cb_minimize_to_tray)
        
        self.cb_notify_on_minimize = QCheckBox("Show notification on minimize to tray")
        self.cb_notify_on_minimize.setToolTip("Display a notification when the application is minimized to the tray.")
        self.cb_notify_on_minimize.toggled.connect(lambda checked: self.save_setting_value(SETTING_NOTIFY_ON_MINIMIZE, checked))
        layout.addRow(self.cb_notify_on_minimize)

        self.btn_check_updates = QPushButton(QIcon.fromTheme("system-software-update"), "Check for Updates")
        self.btn_check_updates.setToolTip("Check for new versions of Skyscope Sentinel (internet connection required).")
        self.btn_check_updates.clicked.connect(self.check_for_updates_placeholder)
        layout.addRow(self.btn_check_updates)
        
        return general_tab

    def create_appearance_tab(self):
        appearance_tab, layout = self._create_tab_content_widget()
        
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["Dark", "Light"]) 
        self.combo_theme.setToolTip("Select the application's color theme.")
        self.combo_theme.currentTextChanged.connect(self.on_theme_selected)
        layout.addRow("Application Theme:", self.combo_theme)

        self.cb_acrylic_effect = QCheckBox("Enable Acrylic/Transparency Effects (Sidebar)")
        self.cb_acrylic_effect.setToolTip("Enable subtle transparency for the sidebar. May impact performance on some systems.\n(A restart might be needed for full effect on theme changes.)")
        self.cb_acrylic_effect.toggled.connect(self.on_acrylic_effect_toggled)
        layout.addRow(self.cb_acrylic_effect)
        
        # Accent Color
        self.btn_accent_color = QPushButton(QIcon.fromTheme("preferences-desktop-color"), "Choose Accent Color")
        self.btn_accent_color.setToolTip("Select a custom accent color for UI highlights.")
        self.btn_accent_color.clicked.connect(self.choose_accent_color)
        self.btn_accent_color.setEnabled(True) # Enabled for functionality
        
        self.lbl_accent_color_preview = QLabel() 
        self.lbl_accent_color_preview.setFixedSize(50,25) # Slightly larger preview
        self.lbl_accent_color_preview.setAutoFillBackground(True)
        self.lbl_accent_color_preview.setToolTip("Preview of the selected accent color.")
        
        accent_layout = QHBoxLayout()
        accent_layout.addWidget(self.btn_accent_color)
        accent_layout.addWidget(self.lbl_accent_color_preview)
        accent_layout.addStretch()
        layout.addRow("Accent Color:", accent_layout)

        self.combo_ui_scaling = QComboBox()
        self.combo_ui_scaling.addItems(["Small (80%)", "Medium (100%)", "Large (120%)"])
        self.combo_ui_scaling.setToolTip("Adjust the overall size of UI elements. Requires application restart to take effect.")
        self.combo_ui_scaling.setEnabled(True)
        self.combo_ui_scaling.currentTextChanged.connect(self.on_ui_scaling_changed)
        layout.addRow("UI Scaling:", self.combo_ui_scaling)
        # NOTE: Applying UI scaling (e.g., via QT_SCALE_FACTOR or similar) typically needs to be done
        # before QApplication is initialized. This setting only saves the preference.
        # The main application entry point would need to read this setting and apply it.
        
        return appearance_tab

    def create_ollama_tab(self):
        ollama_tab, layout = self._create_tab_content_widget()

        self.le_ollama_url = QLineEdit()
        self.le_ollama_url.setPlaceholderText("e.g., http://localhost:11434")
        self.le_ollama_url.setToolTip("The URL where the Ollama service is running.")
        self.le_ollama_url.editingFinished.connect(lambda: self.save_setting_value(SETTING_OLLAMA_URL, self.le_ollama_url.text().strip()))
        layout.addRow("Ollama Service URL:", self.le_ollama_url)

        self.btn_test_ollama = QPushButton(QIcon.fromTheme("network-test"), "Test Connection")
        self.btn_test_ollama.setToolTip("Verify connectivity to the specified Ollama service URL.")
        self.btn_test_ollama.clicked.connect(self.test_ollama_connection)
        layout.addRow(self.btn_test_ollama)
        
        self.cb_ollama_auto_start = QCheckBox("Attempt to start local Ollama service with application")
        self.cb_ollama_auto_start.setToolTip("If checked, Skyscope Sentinel will attempt to start a local Ollama instance at application startup if one isn't detected.\n(Depends on Ollama's CLI availability and system configuration.)")
        self.cb_ollama_auto_start.toggled.connect(lambda checked: self.save_setting_value(SETTING_OLLAMA_AUTO_START, checked))
        self.cb_ollama_auto_start.setEnabled(True)
        # Note: This checkbox enables the *setting* for attempting to autostart Ollama.
        # The actual logic to check and start Ollama based on this setting should be
        # implemented in the application's main startup sequence (e.g., in main.py
        # before or during MainWindow initialization, or when Ollama features are first accessed).
        layout.addRow(self.cb_ollama_auto_start)

        return ollama_tab

    def create_agents_tab(self):
        agents_tab, layout = self._create_tab_content_widget()

        self.combo_agent_log_level = QComboBox()
        self.combo_agent_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.combo_agent_log_level.setToolTip("Set the default logging verbosity for AI agents.")
        self.combo_agent_log_level.currentTextChanged.connect(
            lambda text: self.save_setting_value(SETTING_AGENT_LOG_LEVEL, text)
        )
        layout.addRow("Default Agent Log Level:", self.combo_agent_log_level)

        self.cb_agent_auto_restart = QCheckBox("Automatically restart crashed agents")
        self.cb_agent_auto_restart.setToolTip("If enabled, the system will attempt to automatically restart agents that stop unexpectedly.\n(Requires agent monitoring capabilities.)")
        self.cb_agent_auto_restart.toggled.connect(lambda checked: self.save_setting_value(SETTING_AGENT_AUTO_RESTART, checked))
        self.cb_agent_auto_restart.setEnabled(True)
        # Note: This checkbox enables the *setting* for attempting to auto-restart crashed agents.
        # The actual logic for monitoring agent health and performing restarts based on this
        # setting would be part of a dedicated agent management system within the application.
        layout.addRow(self.cb_agent_auto_restart)
        
        return agents_tab

    def create_advanced_tab(self):
        advanced_tab, layout = self._create_tab_content_widget()
        
        data_folder_layout = QHBoxLayout()
        self.le_data_folder = QLineEdit()
        self.le_data_folder.setToolTip("Location where Skyscope Sentinel stores its data, configurations, and logs.")
        self.le_data_folder.setReadOnly(True) # Typically set via browse
        # self.le_data_folder.editingFinished.connect(lambda: self.save_setting_value(SETTING_DATA_FOLDER, self.le_data_folder.text()))
        
        self.btn_browse_data_folder = QPushButton(QIcon.fromTheme("document-open-folder"), "Browse...")
        self.btn_browse_data_folder.setToolTip("Select the directory for application data.")
        self.btn_browse_data_folder.clicked.connect(self.browse_data_folder)
        data_folder_layout.addWidget(self.le_data_folder)
        data_folder_layout.addWidget(self.btn_browse_data_folder)
        layout.addRow("Application Data Folder:", data_folder_layout)
        
        self.btn_clear_cache = QPushButton(QIcon.fromTheme("edit-clear"), "Clear Application Cache")
        self.btn_clear_cache.setToolTip("Clear temporary cache files (e.g., downloaded model indexes). Does not delete models.")
        self.btn_clear_cache.clicked.connect(self.clear_application_cache_placeholder)
        layout.addRow(self.btn_clear_cache)

        self.btn_reset_settings = QPushButton(QIcon.fromTheme("edit-undo"), "Reset All Settings")
        self.btn_reset_settings.setToolTip("Restore all application settings to their original default values.")
        self.btn_reset_settings.clicked.connect(self.reset_all_settings)
        layout.addRow(self.btn_reset_settings)

        return advanced_tab

    def save_setting_value(self, key, value):
        self.settings_manager.save_setting(key, value)
        # Provide more specific feedback for certain settings
        if key == SETTING_THEME:
             self.status_message_requested.emit(f"Theme set to '{value.capitalize()}'.", "info")
        elif key == SETTING_ACRYLIC_EFFECT:
             self.status_message_requested.emit(f"Acrylic effect {'enabled' if value else 'disabled'}.", "info")
        else:
             self.status_message_requested.emit(f"Setting '{key.split('/')[-1].replace('_', ' ').capitalize()}' saved.", "info")


    def load_all_settings(self):
        # General
        self.cb_autostart.setChecked(self.settings_manager.load_setting(SETTING_AUTOSTART, False))
        self.cb_enable_tray_icon.setChecked(self.settings_manager.load_setting(SETTING_ENABLE_TRAY_ICON, True))
        self.cb_minimize_to_tray.setChecked(self.settings_manager.load_setting(SETTING_MINIMIZE_TO_TRAY, True))
        self.cb_notify_on_minimize.setChecked(self.settings_manager.load_setting(SETTING_NOTIFY_ON_MINIMIZE, True))
        # Ensure tray icon state matches loaded setting initially
        self.tray_icon_visibility_requested.emit(self.cb_enable_tray_icon.isChecked())


        # Appearance
        current_theme_name = self.settings_manager.load_setting(SETTING_THEME, "dark")
        self.combo_theme.setCurrentText(current_theme_name.capitalize())
        
        self.cb_acrylic_effect.setChecked(self.settings_manager.load_setting(SETTING_ACRYLIC_EFFECT, True))
        
        accent_color_hex = self.settings_manager.load_setting(SETTING_ACCENT_COLOR, "#1abc9c") 
        self.update_accent_color_preview(accent_color_hex)

        ui_scaling_text = self.settings_manager.load_setting(SETTING_UI_SCALING, "Medium (100%)")
        self.combo_ui_scaling.setCurrentText(ui_scaling_text)


        # Ollama
        self.le_ollama_url.setText(self.settings_manager.load_setting(SETTING_OLLAMA_URL, "http://localhost:11434"))
        self.cb_ollama_auto_start.setChecked(self.settings_manager.load_setting(SETTING_OLLAMA_AUTO_START, False))


        # Agents
        self.combo_agent_log_level.setCurrentText(self.settings_manager.load_setting(SETTING_AGENT_LOG_LEVEL, "INFO"))
        self.cb_agent_auto_restart.setChecked(self.settings_manager.load_setting(SETTING_AGENT_AUTO_RESTART, False))

        # Advanced
        # For data folder, ensure it's a valid path or use a default from QStandardPaths
        import os
        from PySide6.QtCore import QStandardPaths
        default_data_path = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        data_folder_value = self.settings_manager.load_setting(SETTING_DATA_FOLDER, os.path.join(default_data_path, "data"))
        if hasattr(self, 'le_data_folder'): # Check if advanced tab and its elements are initialized
            self.le_data_folder.setText(data_folder_value)

        # Financials - Load saved values or defaults
        if hasattr(self, 'le_paypal_email'): # Check if financial tab elements are initialized
            self.le_paypal_email.setText(self.settings_manager.load_setting("financials/paypal_email", ""))
            self.le_btc_address.setText(self.settings_manager.load_setting("financials/btc_address", ""))
            self.le_eth_address.setText(self.settings_manager.load_setting("financials/eth_address", ""))
            self.le_bank_bsb.setText(self.settings_manager.load_setting("financials/bank_bsb", ""))
            self.le_bank_account.setText(self.settings_manager.load_setting("financials/bank_account_number", ""))
            self.le_payid.setText(self.settings_manager.load_setting("financials/payid", ""))

        # API Keys - Load saved values
        # API Keys - Load saved values
        if hasattr(self, 'le_openai_api_key'):
            self.le_openai_api_key.setText(self.settings_manager.load_setting("api_keys/openai_api_key", ""))
        if hasattr(self, 'le_serper_api_key'):
            self.le_serper_api_key.setText(self.settings_manager.load_setting("api_keys/serper_api_key", ""))
        if hasattr(self, 'le_e2b_api_key'): # Load E2B API Key
            self.le_e2b_api_key.setText(self.settings_manager.load_setting("api_keys/e2b_api_key", ""))

        self.status_message_requested.emit("Settings loaded.", "info")

    # --- API Keys Tab Creator ---
    def create_api_keys_tab(self):
        api_keys_tab, layout = self._create_tab_content_widget()

        api_instructions = QLabel(
            "Enter your API keys for various services here. These keys are stored locally and securely.\n"
            "Some features or agents may require these keys to function correctly."
        )
        api_instructions.setWordWrap(True)
        api_instructions.setStyleSheet("font-style: italic; color: #888888; margin-bottom: 10px;")
        layout.addRow(api_instructions)

        # OpenAI API Key
        self.le_openai_api_key = QLineEdit()
        self.le_openai_api_key.setPlaceholderText("sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.le_openai_api_key.setToolTip("Your OpenAI API Key (used by some agents or if cloud LLMs are preferred).")
        self.le_openai_api_key.setEchoMode(QLineEdit.Password)
        self.le_openai_api_key.editingFinished.connect(
            lambda: self.save_setting_value("api_keys/openai_api_key", self.le_openai_api_key.text().strip())
        )
        layout.addRow("OpenAI API Key:", self.le_openai_api_key)

        # Serper API Key
        self.le_serper_api_key = QLineEdit()
        self.le_serper_api_key.setPlaceholderText("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.le_serper_api_key.setToolTip("Your Serper.dev API Key for the SerperDevTool (enhanced web search).")
        self.le_serper_api_key.setEchoMode(QLineEdit.Password)
        self.le_serper_api_key.editingFinished.connect(
            lambda: self.save_setting_value("api_keys/serper_api_key", self.le_serper_api_key.text().strip())
        )
        layout.addRow("Serper API Key:", self.le_serper_api_key)

        # E2B API Key
        self.le_e2b_api_key = QLineEdit()
        self.le_e2b_api_key.setPlaceholderText("e2b_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.le_e2b_api_key.setToolTip("Your E2B API Key for secure sandboxed code execution.")
        self.le_e2b_api_key.setEchoMode(QLineEdit.Password)
        self.le_e2b_api_key.editingFinished.connect(
            lambda: self.save_setting_value("api_keys/e2b_api_key", self.le_e2b_api_key.text().strip())
        )
        layout.addRow("E2B API Key:", self.le_e2b_api_key)

        # Add more API key fields here as needed in the future

        return api_keys_tab

    # --- Financials Tab Creator and Methods ---
    def create_financials_tab(self):
        financials_tab, layout = self._create_tab_content_widget()

        instructions = QLabel(
            "Enter your payment details below. This information will be used by agents to receive earnings.\n"
            "Skyscope Sentinel Intelligence and its founder are not responsible for the security of this information if your system is compromised.\n"
            "Ensure your system is secure. All details are stored locally."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("font-style: italic; color: #888888; margin-bottom: 10px;")
        layout.addRow(instructions)

        # PayPal
        self.le_paypal_email = QLineEdit()
        self.le_paypal_email.setPlaceholderText("your.email@example.com")
        self.le_paypal_email.setToolTip("Your PayPal email address for receiving payments.")
        self.le_paypal_email.editingFinished.connect(
            lambda: self.save_setting_value("financials/paypal_email", self.le_paypal_email.text().strip())
        )
        layout.addRow("PayPal Email:", self.le_paypal_email)

        # Crypto Wallets
        crypto_group = QGroupBox("Cryptocurrency Wallets")
        crypto_layout = QFormLayout(crypto_group)

        self.le_btc_address = QLineEdit()
        self.le_btc_address.setPlaceholderText("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")
        self.le_btc_address.setToolTip("Your Bitcoin (BTC) wallet address.")
        self.le_btc_address.editingFinished.connect(
            lambda: self.save_setting_value("financials/btc_address", self.le_btc_address.text().strip())
        )
        crypto_layout.addRow("Bitcoin (BTC) Address:", self.le_btc_address)

        self.le_eth_address = QLineEdit()
        self.le_eth_address.setPlaceholderText("0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B")
        self.le_eth_address.setToolTip("Your Ethereum (ETH) or ERC-20 compatible wallet address.")
        self.le_eth_address.editingFinished.connect(
            lambda: self.save_setting_value("financials/eth_address", self.le_eth_address.text().strip())
        )
        crypto_layout.addRow("Ethereum (ETH) Address:", self.le_eth_address)
        layout.addRow(crypto_group)

        # Australian Bank Details
        bank_group = QGroupBox("Australian Bank Details (OSKO/PayID)")
        bank_layout = QFormLayout(bank_group)

        self.le_bank_bsb = QLineEdit()
        self.le_bank_bsb.setPlaceholderText("XXX-XXX")
        self.le_bank_bsb.setToolTip("Your BSB (Bank-State-Branch) number.")
        self.le_bank_bsb.editingFinished.connect(
            lambda: self.save_setting_value("financials/bank_bsb", self.le_bank_bsb.text().strip())
        )
        bank_layout.addRow("BSB:", self.le_bank_bsb)

        self.le_bank_account = QLineEdit()
        self.le_bank_account.setPlaceholderText("XXXXXXXXX")
        self.le_bank_account.setToolTip("Your bank account number.")
        self.le_bank_account.editingFinished.connect(
            lambda: self.save_setting_value("financials/bank_account_number", self.le_bank_account.text().strip())
        )
        bank_layout.addRow("Account Number:", self.le_bank_account)

        self.le_payid = QLineEdit()
        self.le_payid.setPlaceholderText("your.email@example.com or +614XX XXX XXX")
        self.le_payid.setToolTip("Your PayID (often an email or phone number).")
        self.le_payid.editingFinished.connect(
            lambda: self.save_setting_value("financials/payid", self.le_payid.text().strip())
        )
        bank_layout.addRow("PayID (Optional):", self.le_payid)
        layout.addRow(bank_group)

        return financials_tab

    @Slot()
    def check_for_updates_placeholder(self):
        self.status_message_requested.emit("Checking for updates (placeholder)...", "info")
        QMessageBox.information(self, "Check for Updates", "This feature is a placeholder. No update check performed.")

    @Slot(str)
    def on_theme_selected(self, theme_name_capitalized):
        theme_name_lower = theme_name_capitalized.lower()
        self.save_setting_value(SETTING_THEME, theme_name_lower)
        self.theme_change_requested.emit(theme_name_lower) 

    @Slot(bool)
    def on_acrylic_effect_toggled(self, checked):
        self.save_setting_value(SETTING_ACRYLIC_EFFECT, checked)
        self.acrylic_effect_requested.emit(checked)
        
    @Slot(bool)
    def on_enable_tray_icon_toggled(self, checked):
        self.save_setting_value(SETTING_ENABLE_TRAY_ICON, checked)
        self.tray_icon_visibility_requested.emit(checked)
        self.cb_minimize_to_tray.setEnabled(checked) # Enable/disable minimize_to_tray based on this
        self.cb_notify_on_minimize.setEnabled(checked)
        self.status_message_requested.emit(f"System tray icon {'enabled' if checked else 'disabled'}.", "info")

    @Slot()
    def choose_accent_color(self):
        current_color_hex = self.settings_manager.load_setting(SETTING_ACCENT_COLOR, "#1abc9c")
        try:
            initial_color = QColor(current_color_hex)
            if not initial_color.isValid(): # Fallback if stored color is invalid
                print(f"Warning: Invalid current accent color '{current_color_hex}' from settings. Defaulting.")
                initial_color = QColor("#1abc9c")
        except Exception as e: # Catch potential errors from QColor constructor with bad hex
            print(f"Error creating QColor from '{current_color_hex}': {e}. Defaulting.")
            initial_color = QColor("#1abc9c")

        color = QColorDialog.getColor(initial_color, self, "Choose Accent Color")

        if color.isValid():
            new_color_hex = color.name()
            self.save_setting_value(SETTING_ACCENT_COLOR, new_color_hex) # save_setting_value already emits a generic status
            self.update_accent_color_preview(new_color_hex)
            # Emit a more specific status message for this action
            self.status_message_requested.emit(f"Accent color changed to {new_color_hex}.", "info")
            # Note: Dynamic application of this accent color across the app is not part of this subtask.
        else:
            self.status_message_requested.emit("Accent color selection cancelled.", "info")


    @Slot(str) # Explicitly mark as a slot that accepts a string argument
    def update_accent_color_preview(self, color_hex):
        try:
            # It's good practice to ensure the label exists
            if hasattr(self, 'lbl_accent_color_preview'):
                palette = self.lbl_accent_color_preview.palette()
                palette.setColor(QPalette.Window, QColor(color_hex)) # Use QPalette.Window for background of QLabel
                self.lbl_accent_color_preview.setPalette(palette)
            else:
                print("Error: lbl_accent_color_preview not found.")
        except Exception as e:
            print(f"Error updating accent color preview with '{color_hex}': {e}") # Log error
            self.status_message_requested.emit(f"Error applying accent color preview: {e}", "error")

    @Slot(str)
    def on_ui_scaling_changed(self, scaling_text):
        self.save_setting_value(SETTING_UI_SCALING, scaling_text)
        self.status_message_requested.emit(
            f"UI Scaling set to '{scaling_text}'. Please restart the application for changes to take effect.",
            "info"
        )
        # Add a more prominent QMessageBox to inform the user about restart
        QMessageBox.information(self, "UI Scaling Changed",
                                f"UI scaling preference has been set to '{scaling_text}'.\n\n"
                                "A restart of the application is required for this change to apply.",
                                QMessageBox.Ok)

    @Slot()
    def test_ollama_connection(self):
        url = self.le_ollama_url.text().strip()
        if not url:
            self.status_message_requested.emit("Ollama Service URL cannot be empty for connection test.", "warning")
            QMessageBox.warning(self, "Test Connection", "Ollama Service URL cannot be empty.")
            return

        self.status_message_requested.emit(f"Testing Ollama connection to {url}...", "info")
        self.btn_test_ollama.setEnabled(False)
        self.btn_test_ollama.setIcon(QIcon.fromTheme("network-test-symbolic"))
        
        # Note: OllamaIntegration might not support custom URLs per call yet.
        # This test will use its internally configured URL or default.
        version, error = self.ollama_integration.get_ollama_version_sync()
        
        self.btn_test_ollama.setEnabled(True)
        self.btn_test_ollama.setIcon(QIcon.fromTheme("network-test"))
        if error:
            err_msg = f"Failed to connect to Ollama service (using configured default).\nError: {error}"
            self.status_message_requested.emit(err_msg, "error")
            QMessageBox.critical(self, "Connection Failed", err_msg)
        else:
            msg = f"Successfully connected to Ollama (using configured default).\nVersion: {version}"
            self.status_message_requested.emit(msg, "success")
            QMessageBox.information(self, "Connection Successful", msg)

    @Slot()
    def browse_data_folder(self):
        current_path = self.le_data_folder.text()
        if not current_path: # If path is empty, default to user's app data location
            from PySide6.QtCore import QStandardPaths
            current_path = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)

        directory = QFileDialog.getExistingDirectory(self, "Select Application Data Folder", current_path)
        if directory:
            self.le_data_folder.setText(directory)
            self.save_setting_value(SETTING_DATA_FOLDER, directory)
            self.status_message_requested.emit(f"Data folder set to: {directory}", "info")

    @Slot()
    def clear_application_cache_placeholder(self):
        self.status_message_requested.emit("Attempting to clear application cache (placeholder)...", "info")
        reply = QMessageBox.question(self, "Clear Application Cache",
                                    "This is a placeholder action. No actual data will be cleared.\n\nProceed to simulate cache clearing?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.status_message_requested.emit("Application cache cleared (simulated).", "success")
            QMessageBox.information(self, "Clear Cache", "Application cache cleared (Placeholder Action).")


    @Slot()
    def reset_all_settings(self):
        reply = QMessageBox.warning(self, "Reset All Settings",
                                    "Are you sure you want to reset ALL application settings to their defaults?\nThis action cannot be undone and may require an application restart to take full effect.",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.settings_manager.clear_all_settings()
            self.load_all_settings() # Reload UI with default values
            
            # Emit signals for settings that affect main window immediately
            current_theme_name = self.settings_manager.load_setting(SETTING_THEME, "dark") 
            self.theme_change_requested.emit(current_theme_name)
            self.acrylic_effect_requested.emit(self.settings_manager.load_setting(SETTING_ACRYLIC_EFFECT, True))
            self.tray_icon_visibility_requested.emit(self.settings_manager.load_setting(SETTING_ENABLE_TRAY_ICON, True))

            self.status_message_requested.emit("All application settings have been reset to defaults.", "success")
            QMessageBox.information(self, "Settings Reset", "All application settings have been reset to their defaults.\nPlease restart the application if some changes are not immediately visible.")

if __name__ == '__main__':
    from PySide6.QtWidgets import QApplication
    import sys

    class DummyMainWindow(QWidget): # To catch signals for testing
        def __init__(self):
            super().__init__()
            self.settings_page = SettingsPage()
            self.settings_page.theme_change_requested.connect(self.handle_theme_change)
            self.settings_page.acrylic_effect_requested.connect(self.handle_acrylic_toggle)
            self.settings_page.status_message_requested.connect(self.handle_status_message)
            
            layout = QVBoxLayout(self)
            layout.addWidget(self.settings_page)
            self.setWindowTitle("Settings Page Test Container")
            self.setGeometry(200, 200, 700, 500)

        @Slot(str)
        def handle_theme_change(self, theme_name):
            print(f"MAIN WINDOW: Theme change requested to: {theme_name}")
            # Apply theme to this dummy window for visual feedback
            if theme_name == "dark":
                self.setStyleSheet("background-color: #2c3e50; color: white;")
            else:
                self.setStyleSheet("background-color: #ecf0f1; color: black;")


        @Slot(bool)
        def handle_acrylic_toggle(self, enabled):
            print(f"MAIN WINDOW: Acrylic effect requested: {enabled}")

        @Slot(str, str)
        def handle_status_message(self, message, msg_type):
            print(f"STATUS [{msg_type.upper()}]: {message}")


    app = QApplication(sys.argv)
    # Test with a settings manager instance
    # sm = SettingsManager("SkyscopeTest", "SentinelTest")
    # sm.clear_all_settings() # Clear before test if needed

    test_window = DummyMainWindow()
    test_window.show()
    sys.exit(app.exec())
