"""
Minimal GUI for Surface Material Extractor.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox,
    QSpinBox, QTextEdit, QProgressBar, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.pipeline import (
    PipelineParams, extract_surface_polygons, extract_folder,
)
from processing.material_classifier import MaterialParams


class WorkerThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal(str)

    def __init__(self, mode, source, out_dir, params):
        super().__init__()
        self.mode = mode          # "file" or "folder"
        self.source = source
        self.out_dir = out_dir
        self.params = params
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            if self.mode == "file":
                r = extract_surface_polygons(
                    self.source, self.out_dir, self.params,
                    log_fn=self.log.emit,
                    progress_fn=self.progress.emit,
                    cancelled_fn=lambda: self._cancel,
                )
                self.log.emit(f"\n✓ Done: {r.get('dxf', '')}")
            else:
                rs = extract_folder(
                    self.source, self.out_dir, self.params,
                    log_fn=self.log.emit,
                    progress_fn=self.progress.emit,
                    cancelled_fn=lambda: self._cancel,
                )
                ok = sum(1 for r in rs if r.get("status") == "processed")
                self.log.emit(f"\n✓ Done: {ok}/{len(rs)} tiles processed.")
            self.finished_ok.emit(str(self.out_dir))
        except Exception as e:
            self.log.emit(f"\nFATAL ERROR: {type(e).__name__}: {e}")
            self.finished_ok.emit("")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Surface Material Extractor  —  V01 (WIP)")
        self.setMinimumSize(820, 640)
        self.resize(1000, 720)
        self._worker: WorkerThread | None = None
        self._build_menu()
        self._build_ui()

    # ── Menu ──────────────────────────────────────────────────────────
    def _build_menu(self):
        m = self.menuBar()
        help_m = m.addMenu("&Help")
        a = help_m.addAction("About…")
        a.triggered.connect(self._show_about)

    def _show_about(self):
        text = (
            "<h2>Surface Material Extractor</h2>"
            "<p><b>Version:</b> V01 &mdash; WIP</p>"
            "<p>Extracts 2D polygons of surface materials (grass, asphalt,<br>"
            "concrete, painted markings) from classified drone photogrammetry<br>"
            "point clouds. Outputs DXF + GeoJSON + CSV for Revit/CAD workflows.</p>"
            "<hr>"
            "<p><b>Created by:</b> Arun Prashad Hariharan<br>"
            "<b>Email:</b> <a href='mailto:arunprashadh@gmail.com'>arunprashadh@gmail.com</a><br>"
            "<b>LinkedIn:</b> <a href='https://www.linkedin.com/in/arun-prashad-hariharan'>"
            "linkedin.com/in/arun-prashad-hariharan</a></p>"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("About")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.setText(text)
        msg.exec()

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        L = QVBoxLayout(root)
        L.setSpacing(10)

        L.addWidget(self._io_group())
        L.addWidget(self._material_group())
        L.addWidget(self._car_group())
        L.addWidget(self._grid_group())
        L.addWidget(self._run_group())

        L.addWidget(QLabel("Log:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMinimumHeight(180)
        L.addWidget(self.log_box, stretch=1)

    def _io_group(self):
        g = QGroupBox("Input / Output")
        gl = QGridLayout(g)
        gl.setColumnStretch(1, 1)

        gl.addWidget(QLabel("Input mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Single classified LAZ/LAS", "file")
        self.mode_combo.addItem("Folder of classified tiles", "folder")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_change)
        gl.addWidget(self.mode_combo, 0, 1, 1, 2)

        gl.addWidget(QLabel("Source:"), 1, 0)
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Classified LAS/LAZ file (or folder)")
        gl.addWidget(self.src_edit, 1, 1)
        btn = QPushButton("Browse…")
        btn.clicked.connect(self._browse_source)
        gl.addWidget(btn, 1, 2)

        gl.addWidget(QLabel("Output folder:"), 2, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Where to save DXF + GeoJSON + CSV")
        gl.addWidget(self.out_edit, 2, 1)
        btn2 = QPushButton("Browse…")
        btn2.clicked.connect(self._browse_out)
        gl.addWidget(btn2, 2, 2)

        gl.addWidget(QLabel("EPSG code:"), 3, 0)
        self.epsg_spin = QSpinBox()
        self.epsg_spin.setRange(0, 100000)
        self.epsg_spin.setValue(27700)
        self.epsg_spin.setSpecialValueText("(none)")
        gl.addWidget(self.epsg_spin, 3, 1)
        gl.addWidget(QLabel("(27700 = OSGB36 British National Grid)"), 3, 2)

        return g

    def _material_group(self):
        g = QGroupBox("Material Classification (VARI + brightness thresholds)")
        gl = QGridLayout(g)

        gl.addWidget(QLabel("VARI grass threshold:"), 0, 0)
        self.vari_spin = QDoubleSpinBox()
        self.vari_spin.setRange(-0.5, 0.9)
        self.vari_spin.setSingleStep(0.01)
        self.vari_spin.setDecimals(3)
        self.vari_spin.setValue(0.10)
        gl.addWidget(self.vari_spin, 0, 1)

        gl.addWidget(QLabel("Asphalt max intensity (0–255):"), 1, 0)
        self.asphalt_max_spin = QDoubleSpinBox()
        self.asphalt_max_spin.setRange(20, 200)
        self.asphalt_max_spin.setValue(100)
        gl.addWidget(self.asphalt_max_spin, 1, 1)

        gl.addWidget(QLabel("Concrete min intensity (0–255):"), 2, 0)
        self.concrete_min_spin = QDoubleSpinBox()
        self.concrete_min_spin.setRange(80, 250)
        self.concrete_min_spin.setValue(160)
        gl.addWidget(self.concrete_min_spin, 2, 1)

        gl.addWidget(QLabel("Painted min intensity (0–255):"), 3, 0)
        self.painted_min_spin = QDoubleSpinBox()
        self.painted_min_spin.setRange(150, 255)
        self.painted_min_spin.setValue(220)
        gl.addWidget(self.painted_min_spin, 3, 1)

        gl.addWidget(QLabel("Hard-surface max saturation (0–1):"), 4, 0)
        self.hard_sat_spin = QDoubleSpinBox()
        self.hard_sat_spin.setRange(0.05, 0.60)
        self.hard_sat_spin.setSingleStep(0.05)
        self.hard_sat_spin.setDecimals(2)
        self.hard_sat_spin.setValue(0.20)
        gl.addWidget(self.hard_sat_spin, 4, 1)

        # Class export toggles
        row = QHBoxLayout()
        row.addWidget(QLabel("Export classes:"))
        self.chk_grass    = QCheckBox("Grass");    self.chk_grass.setChecked(True)
        self.chk_asphalt  = QCheckBox("Asphalt");  self.chk_asphalt.setChecked(True)
        self.chk_concrete = QCheckBox("Concrete"); self.chk_concrete.setChecked(True)
        self.chk_painted  = QCheckBox("Painted");  self.chk_painted.setChecked(True)
        self.chk_other    = QCheckBox("Other");    self.chk_other.setChecked(False)
        for chk in (self.chk_grass, self.chk_asphalt, self.chk_concrete,
                    self.chk_painted, self.chk_other):
            row.addWidget(chk)
        row.addStretch()
        gl.addLayout(row, 5, 0, 1, 2)

        return g

    def _car_group(self):
        g = QGroupBox("Car Removal (geometric filter)")
        gl = QGridLayout(g)
        self.chk_cars = QCheckBox("Auto-remove parked cars from surface classification")
        self.chk_cars.setChecked(True)
        gl.addWidget(self.chk_cars, 0, 0, 1, 4)

        gl.addWidget(QLabel("Car nZ min (m):"), 1, 0)
        self.car_z_min = QDoubleSpinBox()
        self.car_z_min.setRange(0.1, 2.0); self.car_z_min.setSingleStep(0.05)
        self.car_z_min.setValue(0.30)
        gl.addWidget(self.car_z_min, 1, 1)

        gl.addWidget(QLabel("Car nZ max (m):"), 1, 2)
        self.car_z_max = QDoubleSpinBox()
        self.car_z_max.setRange(1.0, 4.0); self.car_z_max.setSingleStep(0.10)
        self.car_z_max.setValue(2.20)
        gl.addWidget(self.car_z_max, 1, 3)

        gl.addWidget(QLabel("Min footprint (m²):"), 2, 0)
        self.car_min_area = QDoubleSpinBox()
        self.car_min_area.setRange(0.5, 20.0); self.car_min_area.setValue(2.0)
        gl.addWidget(self.car_min_area, 2, 1)

        gl.addWidget(QLabel("Max footprint (m²):"), 2, 2)
        self.car_max_area = QDoubleSpinBox()
        self.car_max_area.setRange(10.0, 100.0); self.car_max_area.setValue(25.0)
        gl.addWidget(self.car_max_area, 2, 3)

        return g

    def _grid_group(self):
        g = QGroupBox("Rasterisation + Polygonisation")
        gl = QGridLayout(g)
        gl.addWidget(QLabel("Grid cell size (m):"), 0, 0)
        self.cell_spin = QDoubleSpinBox()
        self.cell_spin.setRange(0.05, 2.0); self.cell_spin.setSingleStep(0.05)
        self.cell_spin.setDecimals(2); self.cell_spin.setValue(0.25)
        gl.addWidget(self.cell_spin, 0, 1)

        gl.addWidget(QLabel("Min polygon area (m²):"), 1, 0)
        self.min_area_spin = QDoubleSpinBox()
        self.min_area_spin.setRange(0.1, 100.0); self.min_area_spin.setValue(1.0)
        gl.addWidget(self.min_area_spin, 1, 1)

        gl.addWidget(QLabel("Simplify tolerance (m):"), 2, 0)
        self.simplify_spin = QDoubleSpinBox()
        self.simplify_spin.setRange(0.0, 2.0); self.simplify_spin.setSingleStep(0.05)
        self.simplify_spin.setDecimals(2); self.simplify_spin.setValue(0.20)
        gl.addWidget(self.simplify_spin, 2, 1)

        return g

    def _run_group(self):
        g = QGroupBox("Run")
        gl = QVBoxLayout(g)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        gl.addWidget(self.progress_bar)
        row = QHBoxLayout()
        self.run_btn = QPushButton("Start Processing")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.run_btn.clicked.connect(self._start)
        row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        row.addWidget(self.cancel_btn)
        gl.addLayout(row)
        return g

    # ── Actions ──────────────────────────────────────────────────────
    def _on_mode_change(self, idx):
        pass

    def _browse_source(self):
        if self.mode_combo.currentData() == "file":
            f, _ = QFileDialog.getOpenFileName(
                self, "Select classified LAS/LAZ", "",
                "LAS/LAZ (*.laz *.las);;All (*.*)"
            )
            if f:
                self.src_edit.setText(f)
        else:
            d = QFileDialog.getExistingDirectory(self, "Select folder of classified tiles")
            if d:
                self.src_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_edit.setText(d)

    def _log(self, msg):
        self.log_box.append(msg)

    def _params(self) -> PipelineParams:
        mat = MaterialParams(
            vari_grass_threshold=self.vari_spin.value(),
            dark_asphalt_max=self.asphalt_max_spin.value(),
            light_concrete_min=self.concrete_min_spin.value(),
            painted_min=self.painted_min_spin.value(),
            hardsurf_max_sat=self.hard_sat_spin.value(),
        )
        epsg = self.epsg_spin.value()
        p = PipelineParams(
            cell_size_m=self.cell_spin.value(),
            min_polygon_area_m2=self.min_area_spin.value(),
            simplify_tolerance_m=self.simplify_spin.value(),
            remove_cars=self.chk_cars.isChecked(),
            car_z_min=self.car_z_min.value(),
            car_z_max=self.car_z_max.value(),
            car_footprint_min_m2=self.car_min_area.value(),
            car_footprint_max_m2=self.car_max_area.value(),
            material_params=mat,
            export_grass=self.chk_grass.isChecked(),
            export_asphalt=self.chk_asphalt.isChecked(),
            export_concrete=self.chk_concrete.isChecked(),
            export_painted=self.chk_painted.isChecked(),
            export_other=self.chk_other.isChecked(),
            epsg=None if epsg == 0 else epsg,
        )
        return p

    def _start(self):
        src = self.src_edit.text().strip()
        out = self.out_edit.text().strip()
        if not src:
            self._log("ERROR: pick a source file/folder.")
            return
        if not out:
            self._log("ERROR: pick an output folder.")
            return
        mode = self.mode_combo.currentData()
        if mode == "file" and not Path(src).is_file():
            self._log(f"ERROR: file not found: {src}"); return
        if mode == "folder" and not Path(src).is_dir():
            self._log(f"ERROR: folder not found: {src}"); return

        params = self._params()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._worker = WorkerThread(mode, src, out, params)
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, done, total):
        if total:
            self.progress_bar.setValue(int(done / total * 100))

    def _on_finished(self, out_dir):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if out_dir:
            self._log(f"Output: {out_dir}")
