import sys
import os
from pathlib import Path
import laspy

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QTextEdit, QProgressBar, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from processing.converter_worker import ConverterWorker


class ConverterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Converter (LAStools)")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        self._worker = None
        self._build_ui()
        self._files = []  # List of file paths

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(self._config_group())
        layout.addWidget(self._files_group())
        layout.addWidget(self._run_group())

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMinimumHeight(150)
        layout.addWidget(QLabel("Process Log:"))
        layout.addWidget(self.log_box)

    def _config_group(self):
        grp = QGroupBox("Configuration")
        lay = QVBoxLayout(grp)

        # Output dir
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Output folder:"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Folder to save converted point clouds")
        row2.addWidget(self.out_edit)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out)
        row2.addWidget(btn_out)
        lay.addLayout(row2)

        return grp

    def _files_group(self):
        grp = QGroupBox("Input Point Clouds")
        lay = QVBoxLayout(grp)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add Files…")
        self.add_btn.clicked.connect(self._add_files)
        btn_row.addWidget(self.add_btn)

        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(self._clear_files)
        btn_row.addWidget(self.clear_btn)
        
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["File Path", "Source Unit / Scale", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        lay.addWidget(self.table)

        return grp

    def _run_group(self):
        grp = QGroupBox("Conversion Settings")
        lay = QHBoxLayout(grp)

        lay.addWidget(QLabel("Target Scale:"))
        self.scale_combo = QComboBox()
        
        self.scale_mapping = {
            "Meters (1.0)": "1.0",
            "Decimeters (0.1)": "0.1",
            "Centimeters (0.01)": "0.01",
            "Millimeters (0.001)": "0.001",
            "Micrometers (0.000001)": "0.000001",
            "Feet (1.0)": "1.0",
            "Feet (0.1)": "0.1",
            "Feet (0.01)": "0.01",
            "Feet (0.001)": "0.001",
            "Inches (1.0)": "1.0",
            "Inches (0.1)": "0.1",
            "Inches (0.01)": "0.01",
            "Inches (0.001)": "0.001",
            "Feet to Millimeters (304.8)": "304.8",
            "Meters to Feet (3.2808)": "3.2808",
            "Feet to Meters (0.3048)": "0.3048"
        }
        
        self.scale_combo.addItems(list(self.scale_mapping.keys()))
        self.scale_combo.setCurrentText("Millimeters (0.001)")
        self.scale_combo.setEditable(True)
        self.scale_combo.setToolTip("Select a preset or type a custom numerical scale (e.g., 304.8)")
        lay.addWidget(self.scale_combo)
        
        lay.addStretch()

        self.run_btn = QPushButton("Convert All")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px 20px;")
        self.run_btn.clicked.connect(self._start_conversion)
        lay.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_conversion)
        lay.addWidget(self.cancel_btn)

        return grp

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output folder")
        if d:
            self.out_edit.setText(d)

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Point Cloud Files",
            "",
            "Point Clouds (*.las *.laz *.e57);;All Files (*)"
        )
        if files:
            for f in files:
                if f not in self._files:
                    self._files.append(f)
                    self._add_file_to_table(f)

    def _clear_files(self):
        self._files.clear()
        self.table.setRowCount(0)
        self.log_box.clear()

    def _add_file_to_table(self, file_path):
        row = self.table.rowCount()
        self.table.insertRow(row)

        path_item = QTableWidgetItem(file_path)
        self.table.setItem(row, 0, path_item)

        scale_text = "Unknown"
        # Try to read initial scale quickly for las/laz
        suffix = Path(file_path).suffix.lower()
        if suffix in ['.las', '.laz']:
            try:
                import laspy
                with laspy.open(file_path) as f:
                    scales = f.header.scales
                    scale_text = f"[{scales[0]}, {scales[1]}, {scales[2]}]"
            except Exception as e:
                scale_text = "Error reading"
        elif suffix == '.e57':
            scale_text = "Meters (Standard)"

        scale_item = QTableWidgetItem(scale_text)
        self.table.setItem(row, 1, scale_item)

        status_item = QTableWidgetItem("Ready")
        self.table.setItem(row, 2, status_item)

    def _update_file_status(self, file_path, status):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == file_path:
                item = self.table.item(row, 2)
                item.setText(status)
                if status == "Success":
                    item.setForeground(QColor("green"))
                elif status.startswith("Failed"):
                    item.setForeground(QColor("red"))
                elif status == "Processing...":
                    item.setForeground(QColor("blue"))
                break

    def _log(self, msg: str):
        self.log_box.append(msg)

    def _start_conversion(self):
        if not self._files:
            QMessageBox.warning(self, "No files", "Please add files to convert.")
            return

        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please select an output folder.")
            return

        current_text = self.scale_combo.currentText()
        target_scale_str = self.scale_mapping.get(current_text, current_text)
        try:
            target_scale = float(target_scale_str)
        except ValueError:
            QMessageBox.warning(self, "Invalid Scale", f"The scale '{target_scale_str}' is not a valid number.")
            return

        # Reset statuses
        for row in range(self.table.rowCount()):
            self.table.item(row, 2).setText("Pending")
            self.table.item(row, 2).setForeground(QColor("black"))

        self.run_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log_box.clear()

        self._worker = ConverterWorker(self._files, out_dir, target_scale)
        self._worker.log.connect(self._log)
        self._worker.file_status.connect(self._update_file_status)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel_conversion(self):
        if self._worker:
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._worker = None


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ConverterApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
