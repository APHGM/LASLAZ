import sys
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QTextEdit, QProgressBar,
    QFileDialog, QGridLayout, QCheckBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# Ensure processing is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.tile_processor import ProcessParams, process_all_files


class WorkerThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal()

    def __init__(self, input_dir, out_dir, params):
        super().__init__()
        self.input_dir = input_dir
        self.out_dir = out_dir
        self.params = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        import threading
        import time
        import psutil
        import os

        # Start a background heartbeat thread
        def heartbeat():
            start_time = time.time()
            process = psutil.Process(os.getpid())
            while not self._finished_flag and not self._cancel:
                time.sleep(15)
                if self._finished_flag or self._cancel:
                    break
                elapsed = int(time.time() - start_time)
                # Get RAM usage in GB
                ram_gb = process.memory_info().rss / (1024**3)
                self.log.emit(f"    [Heartbeat] Processing for {elapsed}s... RAM Usage: {ram_gb:.1f} GB")

        self._finished_flag = False
        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()

        try:
            process_all_files(
                input_dir=self.input_dir,
                out_dir=self.out_dir,
                params=self.params,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
                cancelled_fn=lambda: self._cancel,
            )
        except Exception as e:
            self.log.emit(f"FATAL ERROR: {e}")
        finally:
            self._finished_flag = True
            self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Ground Classifier (TerraScan Style)")
        self.setMinimumWidth(600)
        self._worker: WorkerThread | None = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        # 1. I/O Group
        io_grp = QGroupBox("Input / Output")
        io_lay = QGridLayout(io_grp)
        
        io_lay.addWidget(QLabel("Input folder:"), 0, 0)
        self.in_edit = QLineEdit()
        self.in_edit.setPlaceholderText("Folder containing .laz/.las files")
        self.in_edit.textChanged.connect(self._on_input_changed)
        io_lay.addWidget(self.in_edit, 0, 1)
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self._browse_in)
        io_lay.addWidget(btn_in, 0, 2)

        io_lay.addWidget(QLabel("Output folder:"), 1, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Same as input folder (default)")
        io_lay.addWidget(self.out_edit, 1, 1)
        self.btn_out = QPushButton("Browse…")
        self.btn_out.clicked.connect(self._browse_out)
        io_lay.addWidget(self.btn_out, 1, 2)
        
        # Checkbox to enable/disable custom output folder
        self.custom_output_check = QCheckBox("Use custom output folder")
        self.custom_output_check.setChecked(False)
        self.custom_output_check.toggled.connect(self._toggle_custom_output)
        io_lay.addWidget(self.custom_output_check, 2, 0, 1, 3)
        
        layout.addWidget(io_grp)

        # 2. Ground Settings Group
        g_grp = QGroupBox("Classification Settings")
        g_lay = QGridLayout(g_grp)
        
        self.auto_check = QCheckBox("Use Auto-Adaptive Intelligence (Recommended)")
        self.auto_check.setChecked(True)
        self.auto_check.setStyleSheet("font-weight: bold; color: #2c3e50;")
        self.auto_check.toggled.connect(self._toggle_manual)
        g_lay.addWidget(self.auto_check, 0, 0, 1, 2)

        g_lay.addWidget(QLabel("Adaptive Detail Target (m):"), 1, 0)
        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(0.05, 5.0)
        self.cell_spin.setValue(0.25)
        self.cell_spin.setSingleStep(0.05)
        g_lay.addWidget(self.cell_spin, 1, 1)

        g_lay.addWidget(QLabel("Ground Thickness (m):"), 2, 0)
        self.thr_spin = QDoubleSpinBox()
        self.thr_spin.setRange(0.01, 2.0)
        self.thr_spin.setValue(0.15)
        self.thr_spin.setSingleStep(0.05)
        g_lay.addWidget(self.thr_spin, 2, 1)

        g_lay.addWidget(QLabel("Object Removal Sensitivity (m):"), 3, 0)
        self.sensitivity_spin = QDoubleSpinBox()
        self.sensitivity_spin.setRange(0.05, 5.0)
        self.sensitivity_spin.setValue(0.50)
        self.sensitivity_spin.setSingleStep(0.10)
        g_lay.addWidget(self.sensitivity_spin, 3, 1)
        
        layout.addWidget(g_grp)
        self._toggle_manual(True)

        # 3. Export Options Group
        export_grp = QGroupBox("Export Options")
        export_lay = QGridLayout(export_grp)
        
        # Format selection
        export_lay.addWidget(QLabel("Output Format:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["LAZ (Compressed)", "LAS (Uncompressed)"])
        self.format_combo.setCurrentIndex(0)  # LAZ by default
        export_lay.addWidget(self.format_combo, 0, 1)
        
        # LAS version selection
        export_lay.addWidget(QLabel("LAS Version:"), 1, 0)
        self.version_combo = QComboBox()
        self.version_combo.addItems(["LAS 1.4", "LAS 1.2"])
        self.version_combo.setCurrentIndex(0)  # 1.4 by default
        export_lay.addWidget(self.version_combo, 1, 1)
        
        layout.addWidget(export_grp)

        # 4. Progress / Log
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMinimumHeight(200)
        layout.addWidget(QLabel("Log (Auto-updates every 15s during long runs):"))
        layout.addWidget(self.log_box)

        # 5. Buttons
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Start Auto-Classification")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 10px; background-color: #ecf0f1;")
        self.run_btn.clicked.connect(self._start)
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_row)
        
        # Initialize output folder state (disabled by default, synced with input folder)
        self._toggle_custom_output(False)

    def _toggle_manual(self, checked):
        """Disable manual boxes if Auto-Adaptive is ON"""
        disabled = checked
        self.cell_spin.setDisabled(disabled)
        self.thr_spin.setDisabled(disabled)
        self.sensitivity_spin.setDisabled(disabled)

    def _on_input_changed(self, text):
        """Auto-update output folder when input folder changes (if custom output not enabled)"""
        if not self.custom_output_check.isChecked():
            self.out_edit.setText(text)

    def _toggle_custom_output(self, checked):
        """Enable/disable custom output folder selection"""
        self.out_edit.setDisabled(not checked)
        self.btn_out.setDisabled(not checked)
        # If disabling custom output, sync with input folder
        if not checked:
            self.out_edit.setText(self.in_edit.text())

    def _browse_in(self):
        d = QFileDialog.getExistingDirectory(self, "Select input folder")
        if d: self.in_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d: self.out_edit.setText(d)

    def _start(self):
        in_dir = self.in_edit.text().strip()
        custom_output = self.custom_output_check.isChecked()
        out_dir = self.out_edit.text().strip() if custom_output else in_dir
        
        if not in_dir:
            self.log_box.append("ERROR: Select input folder.")
            return
        
        if custom_output and not out_dir:
            self.log_box.append("ERROR: Select output folder or uncheck 'Use custom output folder'.")
            return
        
        # Get output format (strip "(Compressed)" or "(Uncompressed)" from display text)
        format_text = self.format_combo.currentText()
        output_format = "laz" if "LAZ" in format_text else "las"
        
        # Get LAS version (strip "LAS " from display text)
        version_text = self.version_combo.currentText()
        las_version = version_text.replace("LAS ", "")

        params = ProcessParams(
            auto_mode=self.auto_check.isChecked(),
            grid_cell_size=self.cell_spin.value(),
            ground_threshold=self.thr_spin.value(),
            object_sensitivity=self.sensitivity_spin.value(),
            output_format=output_format,
            same_location=not custom_output,
            las_version=las_version
        )

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self.log_box.append(">>> Initializing Auto-Adaptive Intelligence Engine...")

        self._worker = WorkerThread(in_dir, out_dir, params)
        self._worker.log.connect(self.log_box.append)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, done, total):
        if total:
            self.progress_bar.setValue(int(done / total * 100))

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.log_box.append("\n>>> Process finished.")


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
