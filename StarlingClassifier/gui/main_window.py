# pyright: reportOptionalMemberAccess=false
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QTextEdit, QProgressBar,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QStackedWidget
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.tile_processor import (
    ProcessParams, process_all_tiles, process_single_file
)
from processing.tiler import tile_file


class WorkerThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)

    def __init__(self, mode, source, out_dir, params,
                 tile_size=100.0, skip_tiling=False):
        """
        mode         : "folder" or "file"
        source       : tile folder path (folder mode) or LAZ file path (file mode)
        out_dir      : output folder for classified results
        params       : ProcessParams
        tile_size    : metres (only used in file mode when skip_tiling=False)
        skip_tiling  : if True (file mode only), process whole file as one
        """
        super().__init__()
        self.mode = mode
        self.source = source
        self.out_dir = out_dir
        self.params = params
        self.tile_size = tile_size
        self.skip_tiling = skip_tiling
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            # ── Single file + skip tiling: direct one-shot path ─────────
            if self.mode == "file" and self.skip_tiling:
                csv_path = process_single_file(
                    file_path=self.source,
                    out_dir=self.out_dir,
                    params=self.params,
                    log_fn=self.log.emit,
                    progress_fn=self.progress.emit,
                    cancelled_fn=lambda: self._cancel,
                )
                self.finished.emit(str(csv_path))
                return

            # ── Single file → tile first ────────────────────────────────
            if self.mode == "file":
                src = Path(self.source)
                tile_dir = src.parent / f"Tile_{src.stem}"
                self.log.emit(f"Single-file mode — tiling to: {tile_dir}")
                tile_file(
                    in_path=str(src),
                    out_dir=str(tile_dir),
                    tile_size=self.tile_size,
                    log_fn=self.log.emit,
                    progress_fn=self.progress.emit,
                    cancelled_fn=lambda: self._cancel,
                )
                if self._cancel:
                    self.finished.emit("")
                    return
                self.log.emit("Tiling complete — starting classification.\n")
                tile_dir_str = str(tile_dir)
            else:
                tile_dir_str = self.source

            csv_path = process_all_tiles(
                tile_dir=tile_dir_str,
                out_dir=self.out_dir,
                params=self.params,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
                cancelled_fn=lambda: self._cancel,
            )
            self.finished.emit(str(csv_path))
        except Exception as e:
            self.log.emit(f"FATAL ERROR: {e}")
            self.finished.emit("")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Starling Ground-Contact Classifier")
        self.setMinimumWidth(760)
        self._worker: WorkerThread | None = None
        self._build_ui()


    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(self._io_group())
        layout.addWidget(self._ground_group())
        layout.addWidget(self._bird_group())
        layout.addWidget(self._run_group())

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMinimumHeight(160)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_box)

    # ── I/O ──────────────────────────────────────────────────────────────
    def _io_group(self):
        grp = QGroupBox("Input / Output")
        outer = QVBoxLayout(grp)

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Input mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Tile folder  (pre-tiled *.laz files)", "folder")
        self.mode_combo.addItem("Single LAZ file  (auto-tile then process)", "file")
        self.mode_combo.currentIndexChanged.connect(self._on_input_mode_change)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        outer.addLayout(mode_row)

        # Stacked input panels
        self.input_stack = QStackedWidget()

        # ── Folder mode panel ──
        folder_widget = QWidget()
        fg = QGridLayout(folder_widget)
        fg.addWidget(QLabel("Tile folder:"), 0, 0)
        self.tile_edit = QLineEdit()
        self.tile_edit.setPlaceholderText("Folder containing *.laz tiles")
        fg.addWidget(self.tile_edit, 0, 1)
        btn_tile = QPushButton("Browse…")
        btn_tile.clicked.connect(self._browse_tiles)
        fg.addWidget(btn_tile, 0, 2)
        self.input_stack.addWidget(folder_widget)

        # ── Single-file mode panel ──
        file_widget = QWidget()
        ff = QGridLayout(file_widget)
        ff.addWidget(QLabel("LAZ file:"), 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Single .laz/.las file — will be tiled automatically")
        ff.addWidget(self.file_edit, 0, 1)
        btn_file = QPushButton("Browse…")
        btn_file.clicked.connect(self._browse_file)
        ff.addWidget(btn_file, 0, 2)

        ff.addWidget(QLabel("Tile size (m):"), 1, 0)
        self.tile_size_spin = QDoubleSpinBox()
        self.tile_size_spin.setRange(10.0, 500.0)
        self.tile_size_spin.setValue(100.0)
        self.tile_size_spin.setSingleStep(5.0)
        ff.addWidget(self.tile_size_spin, 1, 1)
        ff.addWidget(QLabel("(tiles written next to source file)"), 1, 2)

        # Skip-tiling checkbox
        self.skip_tile_chk = QCheckBox(
            "Skip tiling — process whole file as one  "
            "(recommended for files < ~300 M points)"
        )
        self.skip_tile_chk.setChecked(False)
        self.skip_tile_chk.toggled.connect(
            lambda on: self.tile_size_spin.setEnabled(not on)
        )
        ff.addWidget(self.skip_tile_chk, 2, 0, 1, 3)

        self.input_stack.addWidget(file_widget)

        outer.addWidget(self.input_stack)

        # Common output controls
        common = QGridLayout()
        common.addWidget(QLabel("Output folder:"), 0, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Where to save classified LAS + CSV")
        common.addWidget(self.out_edit, 0, 1)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out)
        common.addWidget(btn_out, 0, 2)

        common.addWidget(QLabel("Buffer (m):"), 1, 0)
        self.buf_spin = QDoubleSpinBox()
        self.buf_spin.setRange(1.0, 20.0)
        self.buf_spin.setValue(5.0)
        self.buf_spin.setSingleStep(1.0)
        common.addWidget(self.buf_spin, 1, 1)

        # Parallel workers
        import os
        common.addWidget(QLabel("Parallel workers:"), 1, 2)
        self.workers_spin = QSpinBox()
        cpu = os.cpu_count() or 4
        self.workers_spin.setRange(1, max(1, cpu))
        # Default: half the logical cores, capped at 6 (RAM headroom)
        self.workers_spin.setValue(min(max(1, cpu // 2), 6))
        self.workers_spin.setToolTip(
            "1 = sequential.  N = run N tiles in parallel processes.\n"
            "Tiles ≥150M points always run solo to protect RAM."
        )
        common.addWidget(self.workers_spin, 1, 3)

        common.addWidget(QLabel("Output format:"), 2, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("LAZ  (compressed)", "laz")
        self.output_format_combo.addItem("LAS  (uncompressed)", "las")
        self.output_format_combo.setCurrentIndex(0)
        common.addWidget(self.output_format_combo, 2, 1)

        common.addWidget(QLabel("LAS version:"), 3, 0)
        self.las_version_combo = QComboBox()
        self.las_version_combo.addItem("1.4", "1.4")
        self.las_version_combo.addItem("1.2", "1.2")
        self.las_version_combo.setCurrentIndex(0)
        common.addWidget(self.las_version_combo, 3, 1)

        outer.addLayout(common)

        return grp

    def _on_input_mode_change(self, idx):
        self.input_stack.setCurrentIndex(idx)

    # ── Ground ───────────────────────────────────────────────────────────
    def _ground_group(self):
        grp = QGroupBox("Ground Classification")
        outer = QVBoxLayout(grp)

        # Method selector
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItem("CSF  (Cloth Simulation Filter — recommended)", "csf")
        self.method_combo.addItem("Grid Minimum  (fallback, no CSF)", "grid")
        self.method_combo.currentIndexChanged.connect(self._on_method_change)
        method_row.addWidget(self.method_combo)
        method_row.addStretch()
        outer.addLayout(method_row)

        # Stacked panels — one per method
        self.ground_stack = QStackedWidget()

        # ── CSF panel ──
        csf_widget = QWidget()
        cg = QGridLayout(csf_widget)
        self._add_dspin(cg, 0, "Cloth resolution (m):", 0.10, 5.0, 0.30, 0.10, "csf_res")
        self._add_dspin(cg, 1, "Class threshold (m):", 0.01, 1.0, 0.05, 0.01, "csf_thr")
        self._add_spin(cg, 2, "Rigidness  (1=steep · 2=general · 3=flat):", 1, 3, 1, "csf_rig")
        self._add_spin(cg, 3, "Iterations:", 100, 2000, 500, "csf_iter")
        self._add_dspin(cg, 4, "Pre-thin voxel size (m):", 0.05, 1.0, 0.10, 0.05, "csf_vox")
        cg.addWidget(QLabel("Slope smooth:"), 5, 0)
        self.csf_slope = QCheckBox()
        self.csf_slope.setChecked(True)
        cg.addWidget(self.csf_slope, 5, 1)
        self.ground_stack.addWidget(csf_widget)

        # ── Grid panel ──
        grid_widget = QWidget()
        gg = QGridLayout(grid_widget)
        self._add_dspin(gg, 0, "Cell size (m):", 0.25, 5.0, 0.50, 0.25, "ground_cell")
        self._add_dspin(gg, 1, "Ground threshold (m):", 0.05, 2.0, 0.50, 0.05, "ground_thr")
        self._add_spin(gg, 2, "Smooth passes:", 1, 15, 5, "smooth_pass")
        self.ground_stack.addWidget(grid_widget)

        outer.addWidget(self.ground_stack)
        return grp

    def _on_method_change(self, idx):
        self.ground_stack.setCurrentIndex(idx)

    # ── Bird detection ────────────────────────────────────────────────────
    def _bird_group(self):
        grp = QGroupBox("Bird Contact Detection")
        g = QGridLayout(grp)

        self._add_dspin(g, 0, "Min height above ground (m):", 0.01, 0.20, 0.02, 0.01, "nz_min")
        self._add_dspin(g, 1, "Max height above ground (m):", 0.10, 1.00, 0.40, 0.05, "nz_max")
        self._add_dspin(g, 2, "DBSCAN eps (m):", 0.05, 2.00, 0.30, 0.05, "dbscan_eps")
        self._add_spin(g, 3, "DBSCAN min points:", 2, 100, 8, "dbscan_min")
        self._add_dspin(g, 4, "Min cluster footprint (m²):", 0.001, 0.10, 0.005, 0.001, "min_foot")
        self._add_dspin(g, 5, "Max cluster footprint (m²):", 0.10, 5.00, 1.00, 0.10, "max_foot")

        return grp

    # ── Run / progress ────────────────────────────────────────────────────
    def _run_group(self):
        grp = QGroupBox("Run")
        lay = QVBoxLayout(grp)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        lay.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Start Processing")
        self.run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.run_btn.clicked.connect(self._start)
        btn_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)
        lay.addLayout(btn_row)

        return grp

    # ── Helpers ───────────────────────────────────────────────────────────
    def _add_dspin(self, grid, row, label, lo, hi, val, step, attr):
        grid.addWidget(QLabel(label), row, 0)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        spin.setSingleStep(step)
        spin.setDecimals(3)
        grid.addWidget(spin, row, 1)
        setattr(self, f"_{attr}", spin)

    def _add_spin(self, grid, row, label, lo, hi, val, attr):
        grid.addWidget(QLabel(label), row, 0)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setValue(val)
        grid.addWidget(spin, row, 1)
        setattr(self, f"_{attr}", spin)

    def _browse_tiles(self):
        d = QFileDialog.getExistingDirectory(self, "Select tile folder")
        if d:
            self.tile_edit.setText(d)

    def _browse_file(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select LAZ file", "",
            "LAS/LAZ files (*.laz *.las);;All files (*.*)"
        )
        if f:
            self.file_edit.setText(f)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_edit.setText(d)

    def _log(self, msg: str):
        self.log_box.append(msg)

    def _build_params(self) -> ProcessParams:
        method = self.method_combo.currentData()
        return ProcessParams(
            buffer_m=self.buf_spin.value(),
            output_format=self.output_format_combo.currentData(),
            las_version=self.las_version_combo.currentData(),
            ground_method=method,
            num_workers=self.workers_spin.value(),
            # CSF params
            csf_cloth_resolution=self._csf_res.value(),
            csf_class_threshold=self._csf_thr.value(),
            csf_rigidness=self._csf_rig.value(),
            csf_iterations=self._csf_iter.value(),
            csf_slope_smooth=self.csf_slope.isChecked(),
            csf_voxel_size=self._csf_vox.value(),
            # Grid params
            ground_cell_size=self._ground_cell.value(),
            ground_threshold=self._ground_thr.value(),
            smooth_passes=self._smooth_pass.value(),
            # Bird detection
            nz_min=self._nz_min.value(),
            nz_max=self._nz_max.value(),
            dbscan_eps=self._dbscan_eps.value(),
            dbscan_min_pts=self._dbscan_min.value(),
            min_footprint_m2=self._min_foot.value(),
            max_footprint_m2=self._max_foot.value(),
        )

    def _start(self):
        mode = self.mode_combo.currentData()
        out_dir = self.out_edit.text().strip()

        if not out_dir:
            self._log("ERROR: Select an output folder.")
            return

        if mode == "folder":
            source = self.tile_edit.text().strip()
            if not source or not Path(source).is_dir():
                self._log("ERROR: Select a valid tile folder.")
                return
            tile_size = 100.0
        else:
            source = self.file_edit.text().strip()
            if not source or not Path(source).is_file():
                self._log("ERROR: Select a valid LAZ file.")
                return
            tile_size = self.tile_size_spin.value()

        params = self._build_params()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        skip_tiling = (mode == "file") and self.skip_tile_chk.isChecked()
        self._worker = WorkerThread(
            mode, source, out_dir, params,
            tile_size=tile_size, skip_tiling=skip_tiling,
        )
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, done: int, total: int):
        if total:
            self.progress_bar.setValue(int(done / total * 100))

    def _on_finished(self, csv_path: str):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if csv_path:
            self._log(f"\nDone. Results: {csv_path}")
        else:
            self._log("\nProcessing ended (see errors above).")
