from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QTextEdit, QProgressBar, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QDialog,
    QDialogButtonBox, QSplitter, QAbstractItemView
)
from PySide6.QtCore import Qt, Slot, QThread, Signal
from PySide6.QtGui import QColor, QIcon

from .ollama_integration import OllamaIntegration

# A simple dialog for showing model details
class ModelDetailsDialog(QDialog):
    def __init__(self, model_info_json, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Details")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        
        try:
            # Assuming model_info_json is already a dict here, if not parse it
            import json
            if isinstance(model_info_json, str):
                 parsed_json = json.loads(model_info_json)
            else: # assume it's already a dict
                 parsed_json = model_info_json

            # Basic pretty printing for now
            details_text = f"Family: {parsed_json.get('details', {}).get('family', 'N/A')}\n"
            details_text += f"Format: {parsed_json.get('details', {}).get('format', 'N/A')}\n"
            details_text += f"Parameter Size: {parsed_json.get('details', {}).get('parameter_size', 'N/A')}\n"
            details_text += f"Quantization Level: {parsed_json.get('details', {}).get('quantization_level', 'N/A')}\n\n"
            
            details_text += "--- Modelfile ---\n"
            details_text += parsed_json.get('modelfile', 'N/A') + "\n\n"
            
            details_text += "--- License ---\n"
            details_text += parsed_json.get('license', 'N/A') + "\n"

            self.text_edit.setText(details_text)

        except Exception as e:
            self.text_edit.setText(f"Error parsing model details:\n{str(e)}\n\nRaw data:\n{model_info_json}")

        layout.addWidget(self.text_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class ModelHubPage(QWidget):
    # Signal to emit status messages to a potential main status bar or log
    status_message_requested = Signal(str, str) # message, type (info, error, success)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("modelHubPage")
        self.ollama_integration = OllamaIntegration()

        # Main layout for Model Hub
        main_splitter = QSplitter(Qt.Horizontal)
        
        # --- Installed Models Section ---
        installed_models_widget = QWidget()
        installed_layout = QVBoxLayout(installed_models_widget)
        installed_layout.setContentsMargins(0,0,0,0) # Remove margins if splitter handles spacing

        installed_title = QLabel("Installed Models")
        installed_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px;")
        installed_layout.addWidget(installed_title)

        self.models_table = QTableWidget()
        self.models_table.setColumnCount(4) 
        self.models_table.setHorizontalHeaderLabels(["Name", "Size (GB)", "Family", "Quantization"])
        self.models_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.models_table.horizontalHeader().setStretchLastSection(True)
        self.models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.models_table.itemDoubleClicked.connect(self.show_selected_model_details)
        self.models_table.setToolTip("List of locally available Ollama models. Double-click for details.")
        installed_layout.addWidget(self.models_table)

        installed_actions_layout = QHBoxLayout()
        self.refresh_button = QPushButton(QIcon.fromTheme("view-refresh"), "Refresh List")
        self.refresh_button.setToolTip("Reload the list of installed Ollama models.")
        self.refresh_button.clicked.connect(self.load_installed_models)
        installed_actions_layout.addWidget(self.refresh_button)

        self.details_button = QPushButton(QIcon.fromTheme("dialog-information"), "View Details")
        self.details_button.setToolTip("Show detailed information for the selected model.")
        self.details_button.clicked.connect(self.show_selected_model_details)
        installed_actions_layout.addWidget(self.details_button)
        installed_layout.addLayout(installed_actions_layout)
        
        main_splitter.addWidget(installed_models_widget)

        # --- Online Models Section ---
        online_models_widget = QWidget()
        online_layout = QVBoxLayout(online_models_widget)
        online_layout.setContentsMargins(0,0,0,0)

        online_title = QLabel("Download New Models")
        online_title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px;")
        online_layout.addWidget(online_title)
        
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter model name (e.g., llama3, mistral:7b)")
        self.search_input.setToolTip("Specify the full model name and tag from Ollama Hub (e.g., 'mistral:latest').")
        search_layout.addWidget(self.search_input)
        
        self.download_button = QPushButton(QIcon.fromTheme("arrow-down"), "Download")
        self.download_button.setToolTip("Download the specified model from Ollama Hub.")
        self.download_button.clicked.connect(self.download_model)
        search_layout.addWidget(self.download_button)
        online_layout.addLayout(search_layout)

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setVisible(False)
        self.download_progress_bar.setTextVisible(True)
        self.download_progress_bar.setRange(0,0) 
        self.download_progress_bar.setToolTip("Download progress for the current model.")
        online_layout.addWidget(self.download_progress_bar)
        
        self.download_status_label = QLabel("") 
        self.download_status_label.setWordWrap(True)
        self.download_status_label.setToolTip("Detailed progress messages during model download.")
        online_layout.addWidget(self.download_status_label)
        online_layout.addStretch() 

        main_splitter.addWidget(online_models_widget)
        main_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])


        # --- Ollama Service Status ---
        service_status_group = QGroupBox("Ollama Service")
        service_status_layout = QHBoxLayout(service_status_group)
        
        self.ollama_status_label = QLabel("Status: Unknown")
        self.ollama_status_label.setToolTip("Current status of the local Ollama service.")
        service_status_layout.addWidget(self.ollama_status_label)
        
        self.check_ollama_button = QPushButton(QIcon.fromTheme("network-wired"), "Check Status")
        self.check_ollama_button.setToolTip("Verify connection to the Ollama service and get its version.")
        self.check_ollama_button.clicked.connect(self.check_ollama_status)
        service_status_layout.addWidget(self.check_ollama_button)
        service_status_layout.addStretch()

        # Add service status group and splitter to the page's main layout
        page_layout = QVBoxLayout(self)
        page_layout.addWidget(service_status_group)
        page_layout.addWidget(main_splitter)


        self.load_installed_models()
        self.check_ollama_status()


    @Slot()
    def load_installed_models(self):
        self.status_message_requested.emit("Loading installed models...", "info")
        self.models_table.setRowCount(0) 
        self.refresh_button.setEnabled(False)
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh-symbolic")) # Indicate activity
        self.details_button.setEnabled(False)
        self.ollama_integration.list_models(self.handle_list_models_complete)

    @Slot(str, str, bool)
    def handle_list_models_complete(self, command_name, output, success):
        self.refresh_button.setEnabled(True)
        self.refresh_button.setIcon(QIcon.fromTheme("view-refresh"))
        if success:
            try:
                models_data = [json.loads(line) for line in output.strip().split('\n') if line.strip()]
                self.models_table.setRowCount(len(models_data))
                for row, model_data in enumerate(models_data):
                    name = QTableWidgetItem(model_data.get("name", "N/A"))
                    size_gb = QTableWidgetItem(f"{model_data.get('size', 0) / (1024**3):.2f}")
                    name.setData(Qt.UserRole, model_data) 
                    family = QTableWidgetItem(model_data.get("details", {}).get("family", "N/A"))
                    quant = QTableWidgetItem(model_data.get("details", {}).get("quantization_level", "N/A"))

                    self.models_table.setItem(row, 0, name)
                    self.models_table.setItem(row, 1, size_gb)
                    self.models_table.setItem(row, 2, family)
                    self.models_table.setItem(row, 3, quant)
                self.status_message_requested.emit("Installed models loaded successfully.", "success")
                if self.models_table.rowCount() > 0:
                    self.details_button.setEnabled(True)
                else:
                    self.status_message_requested.emit("No local models found.", "info")
            except json.JSONDecodeError as e:
                err_msg = f"Error parsing model list from Ollama: {e}. Output: {output[:200]}..."
                self.status_message_requested.emit(err_msg, "error")
                QMessageBox.critical(self, "List Models Error", err_msg)
            except Exception as e:
                 err_msg = f"Unexpected error processing model list: {str(e)}"
                 self.status_message_requested.emit(err_msg, "error")
                 QMessageBox.critical(self, "List Models Error", err_msg)
        else:
            err_msg = f"Failed to list models from Ollama: {output}"
            self.status_message_requested.emit(err_msg, "error")
            QMessageBox.warning(self, "List Models Error", err_msg)

    @Slot()
    def show_selected_model_details(self):
        selected_items = self.models_table.selectedItems()
        if not selected_items:
            self.status_message_requested.emit("No model selected to view details.", "info")
            QMessageBox.information(self, "View Details", "Please select a model from the list to view its details.")
            return
        
        selected_row = selected_items[0].row()
        name_item = self.models_table.item(selected_row, 0)
        model_name = name_item.text()
        
        model_data = name_item.data(Qt.UserRole)
        if model_data and 'modelfile' in model_data and 'details' in model_data :
             dialog = ModelDetailsDialog(model_data, self)
             dialog.exec()
        else: 
            self.status_message_requested.emit(f"Fetching details for {model_name}...", "info")
            self.details_button.setEnabled(False)
            self.details_button.setIcon(QIcon.fromTheme("dialog-information-symbolic")) # Indicate activity
            self.refresh_button.setEnabled(False) # Prevent refresh during fetch
            self.ollama_integration.show_model_info(model_name, self.handle_show_model_details_complete)
            
    @Slot(str, str, bool)
    def handle_show_model_details_complete(self, command_name, output, success):
        self.details_button.setEnabled(True)
        self.details_button.setIcon(QIcon.fromTheme("dialog-information"))
        self.refresh_button.setEnabled(True)
        if success:
            try:
                model_info = json.loads(output)
                dialog = ModelDetailsDialog(model_info, self)
                dialog.exec()
                self.status_message_requested.emit(f"Details for '{command_name.split()[-1]}' loaded.", "success")
            except json.JSONDecodeError as e:
                err_msg = f"Error parsing model details from Ollama: {e}. Output: {output[:200]}..."
                self.status_message_requested.emit(err_msg, "error")
                QMessageBox.critical(self, "Model Details Error", err_msg)
        else:
            err_msg = f"Failed to get model details from Ollama: {output}"
            self.status_message_requested.emit(err_msg, "error")
            QMessageBox.warning(self, "Model Details Error", err_msg)


    @Slot()
    def download_model(self):
        model_name_to_download = self.search_input.text().strip()
        if not model_name_to_download:
            QMessageBox.warning(self, "Download Model", "Please enter a model name to download.")
            return

        self.download_button.setEnabled(False)
        self.download_button.setIcon(QIcon.fromTheme("arrow-down-symbolic")) # Indicate activity
        self.search_input.setEnabled(False)
        self.download_progress_bar.setVisible(True)
        self.download_progress_bar.setRange(0,0) 
        self.download_progress_bar.setFormat(f"Downloading {model_name_to_download}...")
        self.download_status_label.setText(f"Starting download of {model_name_to_download}...")
        self.status_message_requested.emit(f"Attempting to download model: '{model_name_to_download}'...", "info")

        self.ollama_integration.pull_model(
            model_name_to_download,
            self.handle_download_complete,
            self.handle_download_progress
        )

    @Slot(str, str)
    def handle_download_progress(self, model_name, progress_line):
        self.download_status_label.setText(f"Progress: {progress_line}")
        # Very basic progress parsing for Ollama pull
        if "pulling manifest" in progress_line:
            self.download_progress_bar.setRange(0, 100) 
            self.download_progress_bar.setValue(5)
        elif "verifying sha256" in progress_line:
             self.download_progress_bar.setValue(20) 
        elif "writing manifest" in progress_line:
             self.download_progress_bar.setValue(30)
        elif "removing any unused layers" in progress_line:
             self.download_progress_bar.setValue(40)
        elif "success" in progress_line.lower():
             self.download_progress_bar.setValue(100)
        elif "%" in progress_line: # Try to parse percentage
            try:
                parts = progress_line.split()
                for part in parts:
                    if part.endswith('%'):
                        percentage = int(float(part[:-1]))
                        self.download_progress_bar.setRange(0, 100)
                        self.download_progress_bar.setValue(percentage)
                        break
            except ValueError: # If parsing fails, keep indeterminate or last value
                self.download_progress_bar.setRange(0,0) # Revert to indeterminate if parsing is unreliable
        self.download_progress_bar.setFormat(f"Downloading {model_name}: {progress_line[:30]}...")


    @Slot(str, str, bool)
    def handle_download_complete(self, command_name, output, success):
        model_name_pulled = command_name.split()[-1] # Get model name from command
        self.download_button.setEnabled(True)
        self.download_button.setIcon(QIcon.fromTheme("arrow-down"))
        self.search_input.setEnabled(True)
        self.download_progress_bar.setVisible(False)
        self.download_status_label.setText("") 

        if success:
            msg = f"Model '{model_name_pulled}' downloaded successfully."
            self.status_message_requested.emit(msg, "success")
            QMessageBox.information(self, "Download Complete", f"{msg}\nOutput:\n{output[:200]}...")
            self.load_installed_models() 
            self.search_input.clear()
        else:
            err_msg = f"Failed to download model '{model_name_pulled}'. Error: {output}"
            self.status_message_requested.emit(err_msg, "error")
            QMessageBox.critical(self, "Download Failed", err_msg)
            self.download_progress_bar.setRange(0,100) 
            self.download_progress_bar.setValue(0)

    @Slot()
    def check_ollama_status(self):
        self.ollama_status_label.setText("Status: Checking...")
        self.check_ollama_button.setEnabled(False)
        self.check_ollama_button.setIcon(QIcon.fromTheme("network-wired-symbolic")) # Indicate activity
        
        version, error = self.ollama_integration.get_ollama_version_sync() # Using sync for quick check
        
        self.check_ollama_button.setEnabled(True)
        self.check_ollama_button.setIcon(QIcon.fromTheme("network-wired"))
        if error:
            self.ollama_status_label.setText(f"<font color='red'>Status: Error ({error})</font>")
            self.status_message_requested.emit(f"Ollama service error: {error}", "error")
        else:
            self.ollama_status_label.setText(f"<font color='green'>Status: Running (Version: {version})</font>")
            self.status_message_requested.emit(f"Ollama service running (Version: {version})", "success")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'main_splitter'): # Check if splitter exists
            self.main_splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])


if __name__ == '__main__':
    # This is for testing the ModelHubPage independently
    from PySide6.QtWidgets import QApplication
    import sys

    # Dummy main.py functions/classes if needed for signals, or connect locally
    class DummyMainWindow(QWidget):
        @Slot(str, str)
        def handle_status_message(self, message, type):
            print(f"STATUS [{type.upper()}]: {message}")

    app = QApplication(sys.argv)
    
    # Load a stylesheet for testing
    def load_stylesheet_test(filename):
        file = QFile(filename)
        if file.open(QFile.ReadOnly | QFile.Text):
            stream = QTextStream(file)
            stylesheet = stream.readAll()
            file.close()
            return stylesheet
        return ""

    # Assuming your qss files are in ../skyscope_sentinel relative to this test execution if run from within the dir
    # Adjust path as necessary if running from project root
    # For testing, let's try to load dark_theme.qss from the expected location
    
    # Correct path when running model_hub_page.py directly for testing
    # assuming it's in skyscope_sentinel/ and qss files are also there
    # To run this test: python -m skyscope_sentinel.model_hub_page
    
    # If skyscope_sentinel is in PYTHONPATH, then it can be:
    # from skyscope_sentinel.main import DARK_STYLE_PATH # This would be better
    # For direct run:
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dark_style_path = os.path.join(current_dir, "dark_theme.qss")
    
    stylesheet = load_stylesheet_test(dark_style_path)
    if stylesheet:
        app.setStyleSheet(stylesheet)
    else:
        print(f"Warning: Stylesheet {dark_style_path} not loaded for test.")


    main_window_dummy = DummyMainWindow() # To catch signals
    page = ModelHubPage()
    page.status_message_requested.connect(main_window_dummy.handle_status_message)
    
    # To make it look more like it's in the main app for testing
    test_container = QWidget()
    layout = QVBoxLayout(test_container)
    layout.addWidget(page)
    test_container.setWindowTitle("Model Hub Page Test")
    test_container.setGeometry(100, 100, 1000, 600) # Similar size to content area
    if stylesheet: # Apply to the container as well
        test_container.setStyleSheet("QWidget { background-color: #2c3e50; }") # Match dark theme background
    test_container.show()
    
    sys.exit(app.exec())
```

A quick note on the `ollama_integration.py` regarding `ollama list --format=json`:
The output of `ollama list --format=json` is a stream of JSON objects, one per line. The `list_models_sync` was already updated to handle this. The asynchronous `list_models` in `OllamaRunner`'s `run` method captures stdout line by line for `pull`, but for other commands it uses `communicate()`. This should be fine as `communicate()` will capture the full multi-line JSON stream, and the parsing logic in `handle_list_models_complete` splits by newline and parses each line.

This `model_hub_page.py` includes:
-   Two main sections using a `QSplitter`: "Installed Models" and "Download Models".
-   **Installed Models:**
    -   `QTableWidget` to display model name, size, family, and quantization.
    -   "Refresh List" button to call `ollama_integration.list_models`.
    -   "View Details" button to call `ollama_integration.show_model_info` for the selected model, displayed in a `ModelDetailsDialog`.
-   **Download Models:**
    -   `QLineEdit` for model name input.
    -   "Download Model" button to call `ollama_integration.pull_model`.
    -   `QProgressBar` and `QLabel` to show download progress and status.
-   **Ollama Service Status:**
    -   A label to show if Ollama is running (checked via `ollama_integration.get_ollama_version_sync`).
    -   A button to re-check the status.
-   Error handling using `QMessageBox` for CLI command failures or parsing issues.
-   Asynchronous operations for listing, showing details, and pulling models, with UI updates handled in `@Slot` decorated methods.
-   A `status_message_requested` signal to communicate messages back to the main window (e.g., for a global status bar or log).
-   Basic `ModelDetailsDialog` to show parsed model information.

Next, I will update `main.py` to integrate this `ModelHubPage`.I have created `ollama_integration.py` and `model_hub_page.py`.
Now I will update `main.py` to:
1.  Import `ModelHubPage`.
2.  Replace the `PlaceholderPage` for "Model Hub" with an instance of `ModelHubPage`.
3.  Connect the `status_message_requested` signal from `ModelHubPage` to a handler in `MainWindow` (e.g., print to console for now, or update a status bar if one existed).
4.  Ensure the Model Hub page is correctly displayed and functions within the main application window and theme structure.
