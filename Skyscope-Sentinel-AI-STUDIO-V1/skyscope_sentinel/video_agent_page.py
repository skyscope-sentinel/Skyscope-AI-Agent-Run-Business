import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QFrame, QGroupBox, QProgressBar, QFileDialog, QListWidget, QAbstractItemView,
    QSpinBox, QDoubleSpinBox, QMessageBox, QSizePolicy, QScrollArea, QTabWidget
)
from PySide6.QtCore import Qt, Slot, Signal # Added Signal
from PySide6.QtGui import QIcon

from .video_agent import VideoAgent # Assuming video_agent.py is in the same directory

class VideoAgentPage(QWidget):
    status_message_requested = Signal(str, str)  # msg, type ('info', 'error', 'success', 'warning')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("videoAgentPage")

        self.video_agent = VideoAgent(self) # Instantiate the agent

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)

        title_label = QLabel("Video Processing Tools")
        title_label.setObjectName("pageTitle")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title_label)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.colorize_widget = self._create_colorize_ui()
        self.images_to_video_widget = self._create_images_to_video_ui()

        self.tab_widget.addTab(self.colorize_widget, "Video Colorization (Simulated)")
        self.tab_widget.addTab(self.images_to_video_widget, "Images to Video")

    def _create_common_output_controls(self, prefix):
        output_group = QGroupBox("Output Settings")
        output_layout = QGridLayout(output_group)
        lbl_output_path = QLabel("Output File Path:")
        le_output_path = QLineEdit()
        le_output_path.setObjectName(f"{prefix}OutputPath")
        le_output_path.setPlaceholderText("Specify output video file path...")
        btn_browse_output = QPushButton(QIcon.fromTheme("document-save"), "Browse...")
        btn_browse_output.setObjectName(f"{prefix}BrowseOutput")
        output_layout.addWidget(lbl_output_path, 0, 0)
        output_layout.addWidget(le_output_path, 0, 1)
        output_layout.addWidget(btn_browse_output, 0, 2)
        return output_group, le_output_path, btn_browse_output

    def _create_colorize_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        input_group = QGroupBox("Input Black & White Video")
        input_layout = QGridLayout(input_group)
        self.lbl_colorize_input_video = QLabel("Input Video:")
        self.le_colorize_input_video = QLineEdit()
        self.le_colorize_input_video.setPlaceholderText("Select a B&W video file...")
        self.le_colorize_input_video.setReadOnly(True)
        self.btn_browse_colorize_input = QPushButton(QIcon.fromTheme("document-open"), "Browse...")
        input_layout.addWidget(self.lbl_colorize_input_video, 0, 0)
        input_layout.addWidget(self.le_colorize_input_video, 0, 1)
        input_layout.addWidget(self.btn_browse_colorize_input, 0, 2)
        layout.addWidget(input_group)

        output_group, self.le_colorize_output_path, self.btn_browse_colorize_output = \
            self._create_common_output_controls("colorize")
        layout.addWidget(output_group)

        action_layout = QHBoxLayout()
        self.btn_start_colorize = QPushButton(QIcon.fromTheme("media-playback-start"), "Start Colorization")
        self.btn_start_colorize.setStyleSheet("padding: 8px;")
        action_layout.addWidget(self.btn_start_colorize)

        self.btn_cancel_colorize = QPushButton(QIcon.fromTheme("process-stop"), "Cancel")
        self.btn_cancel_colorize.setStyleSheet("padding: 8px;")
        self.btn_cancel_colorize.setEnabled(False)
        action_layout.addWidget(self.btn_cancel_colorize)
        layout.addLayout(action_layout)

        self.progress_colorize = QProgressBar()
        self.progress_colorize.setVisible(False)
        layout.addWidget(self.progress_colorize)

        self.lbl_colorize_status = QLabel("Ready.")
        self.lbl_colorize_status.setObjectName("statusLabel")
        layout.addWidget(self.lbl_colorize_status)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _create_images_to_video_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        image_group = QGroupBox("Input Images")
        image_layout = QVBoxLayout(image_group)
        self.list_images = QListWidget()
        self.list_images.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_images.setToolTip("List of images to include in the video.")
        image_layout.addWidget(self.list_images)

        image_buttons_layout = QHBoxLayout()
        self.btn_add_images = QPushButton(QIcon.fromTheme("list-add"), "Add Images...")
        self.btn_remove_images = QPushButton(QIcon.fromTheme("list-remove"), "Remove Selected")
        self.btn_clear_images = QPushButton(QIcon.fromTheme("edit-clear"), "Clear All")
        image_buttons_layout.addWidget(self.btn_add_images)
        image_buttons_layout.addWidget(self.btn_remove_images)
        image_buttons_layout.addWidget(self.btn_clear_images)
        image_layout.addLayout(image_buttons_layout)
        layout.addWidget(image_group)

        settings_group = QGroupBox("Video Settings")
        settings_layout = QGridLayout(settings_group)
        lbl_fps = QLabel("Output FPS:")
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(24)
        self.spin_fps.setToolTip("Frames per second for the output video.")

        lbl_duration = QLabel("Duration per Image (seconds):")
        self.dspin_duration = QDoubleSpinBox()
        self.dspin_duration.setRange(0.0, 600.0) # Allow 0 to signify using FPS for duration
        self.dspin_duration.setDecimals(1)
        self.dspin_duration.setValue(1.0) # Default to 1s, user can set to 0
        self.dspin_duration.setSuffix(" s")
        self.dspin_duration.setToolTip("How long each image should be displayed. Set to 0 to base duration on FPS (1/FPS per image).")

        duration_note = QLabel("Note: If 'Duration per Image' > 0, it dictates display time. If 0, duration is 1/FPS. 'Output FPS' is for video encoding.")
        duration_note.setStyleSheet("font-size: 9pt; color: #666;")
        duration_note.setWordWrap(True)

        settings_layout.addWidget(lbl_fps, 0, 0)
        settings_layout.addWidget(self.spin_fps, 0, 1)
        settings_layout.addWidget(lbl_duration, 1, 0)
        settings_layout.addWidget(self.dspin_duration, 1, 1)
        settings_layout.addWidget(duration_note, 2, 0, 1, 2)
        layout.addWidget(settings_group)

        output_group, self.le_i2v_output_path, self.btn_browse_i2v_output = \
            self._create_common_output_controls("i2v")
        layout.addWidget(output_group)

        action_layout_i2v = QHBoxLayout()
        self.btn_create_video = QPushButton(QIcon.fromTheme("media-record"), "Create Video")
        self.btn_create_video.setStyleSheet("padding: 8px;")
        action_layout_i2v.addWidget(self.btn_create_video)

        self.btn_cancel_i2v = QPushButton(QIcon.fromTheme("process-stop"), "Cancel")
        self.btn_cancel_i2v.setStyleSheet("padding: 8px;")
        self.btn_cancel_i2v.setEnabled(False)
        action_layout_i2v.addWidget(self.btn_cancel_i2v)
        layout.addLayout(action_layout_i2v)

        self.progress_i2v = QProgressBar()
        self.progress_i2v.setVisible(False)
        layout.addWidget(self.progress_i2v)

        self.lbl_i2v_status = QLabel("Ready.")
        self.lbl_i2v_status.setObjectName("statusLabel")
        layout.addWidget(self.lbl_i2v_status)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def _connect_signals(self):
        self.btn_browse_colorize_input.clicked.connect(self._browse_colorize_input_video)
        self.btn_browse_colorize_output.clicked.connect(lambda: self._browse_output_video_path(self.le_colorize_output_path, "Colorized Video"))
        self.btn_start_colorize.clicked.connect(self._start_colorization_task)
        self.btn_cancel_colorize.clicked.connect(self._cancel_current_task)

        self.btn_add_images.clicked.connect(self._add_images_to_list)
        self.btn_remove_images.clicked.connect(self._remove_selected_images)
        self.btn_clear_images.clicked.connect(lambda: self.list_images.clear())
        self.btn_browse_i2v_output.clicked.connect(lambda: self._browse_output_video_path(self.le_i2v_output_path, "Image Slideshow Video"))
        self.btn_create_video.clicked.connect(self._start_images_to_video_task)
        self.btn_cancel_i2v.clicked.connect(self._cancel_current_task)

        self.video_agent.overall_task_progress.connect(self._update_progress)
        self.video_agent.overall_task_finished.connect(self._on_task_finished)
        self.video_agent.overall_task_error.connect(self._on_task_error)
        self.video_agent.overall_status_update.connect(self._update_status_label)

    def _get_current_task_type_from_tab(self):
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == 0: return "colorize_video"
        elif current_tab_index == 1: return "images_to_video"
        return None

    def _set_task_buttons_enabled_state(self, is_task_running, task_type_just_started=None):
        # Disable/Enable start buttons based on whether any task is running
        self.btn_start_colorize.setEnabled(not is_task_running)
        self.btn_create_video.setEnabled(not is_task_running)

        # Enable relevant cancel button only if its specific task started
        self.btn_cancel_colorize.setEnabled(is_task_running and task_type_just_started == "colorize_video")
        self.btn_cancel_i2v.setEnabled(is_task_running and task_type_just_started == "images_to_video")

        # Disable tab switching while any task is running
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, not is_task_running)

    def _browse_colorize_input_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input B&W Video", os.path.expanduser("~"), "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        if file_path:
            self.le_colorize_input_video.setText(file_path)
            base_name, ext = os.path.splitext(os.path.basename(file_path))
            output_dir = os.path.dirname(file_path)
            self.le_colorize_output_path.setText(os.path.join(output_dir, f"{base_name}_colorized{ext}"))
            self.status_message_requested.emit(f"Input video selected: {os.path.basename(file_path)}", "info")

    def _browse_output_video_path(self, line_edit_widget, title_suffix="Video"):
        current_path = line_edit_widget.text()
        default_dir = os.path.dirname(current_path) if current_path and os.path.isdir(os.path.dirname(current_path)) else os.path.expanduser("~")

        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Specify Output Path for {title_suffix}", default_dir,
            "MP4 Video (*.mp4);;AVI Video (*.avi);;MOV Video (*.mov);;All Files (*)"
        )
        if file_path:
            if not os.path.splitext(file_path)[1] and title_suffix != "Any": # Add default extension if missing
                 file_path += ".mp4"
            line_edit_widget.setText(file_path)
            self.status_message_requested.emit(f"Output path set: {os.path.basename(file_path)}", "info")

    def _start_colorization_task(self):
        input_path = self.le_colorize_input_video.text()
        output_path = self.le_colorize_output_path.text()

        if not input_path or not os.path.exists(input_path):
            QMessageBox.warning(self, "Input Error", "Please select a valid input video file.")
            return
        if not output_path:
            QMessageBox.warning(self, "Output Error", "Please specify an output file path.")
            return

        self._set_task_buttons_enabled_state(True, "colorize_video")
        self.progress_colorize.setVisible(True)
        self.progress_colorize.setValue(0)
        self.lbl_colorize_status.setText("Starting colorization...")
        self.status_message_requested.emit("Colorization task started.", "info")
        self.video_agent.colorize_video(input_path, output_path)

    def _add_images_to_list(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", os.path.expanduser("~"), "Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        if files:
            for f_path in files:
                if not self.list_images.findItems(f_path, Qt.MatchExactly):
                    self.list_images.addItem(f_path)
            if self.list_images.count() > 0 and not self.le_i2v_output_path.text():
                first_image_dir = os.path.dirname(self.list_images.item(0).text())
                self.le_i2v_output_path.setText(os.path.join(first_image_dir, "slideshow_output.mp4"))
            self.status_message_requested.emit(f"{len(files)} images added to list.", "info")

    def _remove_selected_images(self):
        selected_items = self.list_images.selectedItems()
        if not selected_items: return
        count_removed = len(selected_items)
        for item in selected_items:
            self.list_images.takeItem(self.list_images.row(item))
        self.status_message_requested.emit(f"{count_removed} images removed.", "info")

    def _start_images_to_video_task(self):
        image_paths = [self.list_images.item(i).text() for i in range(self.list_images.count())]
        output_path = self.le_i2v_output_path.text()
        fps = self.spin_fps.value()
        duration_input = self.dspin_duration.value()
        duration_per_image = duration_input if duration_input > 0 else None

        if not image_paths:
            QMessageBox.warning(self, "Input Error", "Please add at least one image.")
            return
        if not output_path:
            QMessageBox.warning(self, "Output Error", "Please specify an output file path.")
            return

        self._set_task_buttons_enabled_state(True, "images_to_video")
        self.progress_i2v.setVisible(True)
        self.progress_i2v.setValue(0)
        self.lbl_i2v_status.setText("Starting video creation...")
        self.status_message_requested.emit("Images to video task started.", "info")
        self.video_agent.images_to_video(image_paths, output_path, fps, duration_per_image)

    def _cancel_current_task(self):
        self.video_agent.cancel_current_task()
        self.status_message_requested.emit("Task cancellation requested.", "warning")
        # UI update will primarily be handled by _on_task_error or _on_task_finished
        # (if cancellation is treated as an error or a specific type of finish)
        # For immediate feedback:
        active_task_type = None
        if self.btn_cancel_colorize.isEnabled(): active_task_type = "colorize_video"
        elif self.btn_cancel_i2v.isEnabled(): active_task_type = "images_to_video"

        if active_task_type == "colorize_video":
            self.lbl_colorize_status.setText("Cancellation requested...")
        elif active_task_type == "images_to_video":
            self.lbl_i2v_status.setText("Cancellation requested...")
        # Buttons will be reset by _on_task_finished or _on_task_error

    @Slot(int)
    def _update_progress(self, value):
        # This slot might receive progress before task_type is reliably known via worker
        # So, check which cancel button is active or which progress bar is visible
        if self.btn_cancel_colorize.isEnabled() or self.progress_colorize.isVisible():
            self.progress_colorize.setValue(value)
        elif self.btn_cancel_i2v.isEnabled() or self.progress_i2v.isVisible():
            self.progress_i2v.setValue(value)

    @Slot(str)
    def _update_status_label(self, message):
        # Try to get task_type from worker if available, else guess from UI state
        task_type = None
        if self.video_agent and self.video_agent.worker and hasattr(self.video_agent.worker, 'task_type'):
            task_type = self.video_agent.worker.task_type

        if task_type == "colorize_video":
            self.lbl_colorize_status.setText(message)
        elif task_type == "images_to_video":
            self.lbl_i2v_status.setText(message)
        else: # Fallback if task_type is unknown, update based on visible progress or active tab
            if self.progress_colorize.isVisible(): self.lbl_colorize_status.setText(message)
            elif self.progress_i2v.isVisible(): self.lbl_i2v_status.setText(message)
            else: # General status if no task seems active
                current_tab_idx = self.tab_widget.currentIndex()
                if current_tab_idx == 0: self.lbl_colorize_status.setText(message)
                elif current_tab_idx == 1: self.lbl_i2v_status.setText(message)
        self.status_message_requested.emit(f"Video task update: {message}", "info")


    @Slot(str, str, str)
    def _on_task_finished(self, output_path, task_type, message):
        QMessageBox.information(self, "Task Completed", f"{message}")
        self._set_task_buttons_enabled_state(False) # No task running now
        if task_type == "colorize_video":
            self.progress_colorize.setVisible(False)
            self.lbl_colorize_status.setText(f"Done. Output: {os.path.basename(output_path)}")
        elif task_type == "images_to_video":
            self.progress_i2v.setVisible(False)
            self.lbl_i2v_status.setText(f"Done. Output: {os.path.basename(output_path)}")
        self.status_message_requested.emit(f"'{task_type.replace('_', ' ').title()}' task completed: {message}", "success")

    @Slot(str, str)
    def _on_task_error(self, error_message, task_type):
        QMessageBox.critical(self, f"Task Error ({task_type.replace('_', ' ').title()})", str(error_message))
        self._set_task_buttons_enabled_state(False) # No task running now
        if task_type == "colorize_video":
            self.progress_colorize.setVisible(False)
            self.lbl_colorize_status.setText(f"Error.")
        elif task_type == "images_to_video":
            self.progress_i2v.setVisible(False)
            self.lbl_i2v_status.setText(f"Error.")
        self.status_message_requested.emit(f"Error in '{task_type.replace('_', ' ').title()}': {error_message}", "error")

if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    test_window = QMainWindow()
    video_page = VideoAgentPage()
    test_window.setCentralWidget(video_page)
    test_window.setWindowTitle("Video Agent Page Test")
    test_window.setGeometry(100, 100, 750, 700) # Increased height for better layout

    test_window.setStyleSheet("""
        QWidget { font-size: 10pt; }
        QGroupBox {
            font-weight: bold; margin-top: 1ex;
            border: 1px solid #CCC; border-radius: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; subcontrol-position: top left;
            padding: 2px 8px; background-color: #F0F0F0; border-radius: 3px;
        }
        QLabel#pageTitle { font-size: 16pt; font-weight: bold; margin-bottom: 15px; padding-top: 5px;}
        QLabel#statusLabel { font-style: italic; color: #333; margin-top: 5px; min-height: 22px; padding: 2px;}
        QPushButton { padding: 7px 15px; background-color: #EFEFEF; border: 1px solid #DDD; border-radius: 4px;}
        QPushButton:hover { background-color: #E0E0E0; }
        QPushButton:pressed { background-color: #D0D0D0; }
        QPushButton:disabled { background-color: #F5F5F5; color: #AAA; }
        QLineEdit, QSpinBox, QDoubleSpinBox, QListWidget { padding: 5px; border: 1px solid #CCC; border-radius: 4px;}
        QProgressBar { text-align: center; padding: 1px; border-radius: 4px; height: 22px;}
        QProgressBar::chunk { background-color: #3498DB; border-radius: 3px;}
        QTabWidget::pane { border-top: 1px solid #CCC; padding: 15px; }
        QTabBar::tab {
            padding: 10px 20px; background: #E0E0E0; border: 1px solid #CCC;
            border-bottom: none; border-top-left-radius: 5px; border-top-right-radius: 5px;
            margin-right: 2px;
        }
        QTabBar::tab:selected { background: #FFF; font-weight: bold; border-bottom: 1px solid #FFF; }
        QTabBar::tab:!selected:hover { background: #D5D5D5; }
    """)

    test_window.show()
    sys.exit(app.exec())
