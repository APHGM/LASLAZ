# pyright: reportOptionalMemberAccess=false
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton,
    QDoubleSpinBox, QSpinBox, QTextEdit, QProgressBar,
    QFileDialog, QGridLayout, QComboBox, QCheckBox, QStackedWidget,
    QScrollArea, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import sys
from pathlib import Path

from gui.collapsible import CollapsibleSection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from processing.tile_processor import (
    ProcessParams, process_all_tiles, process_single_file
)
from processing.tiler import tile_file, merge_las_files
from processing.las_io import estimate_optimal_tile_size, guess_scan_type


class WorkerThread(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(str)

    def __init__(self, mode, source, out_dir, params,
                 tile_size=100.0, skip_tiling=False,
                 multi_files: list[str] | None = None,
                 merged_name: str = ""):
        """
        mode         : "folder" | "file" | "multi"
        source       : tile folder (folder), LAZ file (file), or "" (multi — uses multi_files)
        out_dir      : output folder for classified results
        params       : ProcessParams
        tile_size    : metres (when tiling is used)
        skip_tiling  : if True, process the single/merged file as one (no tile split)
        multi_files  : list of file paths to merge (multi mode only)
        merged_name  : output filename for the merged LAZ (multi mode); blank = auto
        """
        super().__init__()
        self.mode = mode
        self.source = source
        self.out_dir = out_dir
        self.params = params
        self.tile_size = tile_size
        self.skip_tiling = skip_tiling
        self.multi_files = multi_files or []
        self.merged_name = merged_name
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _resolve_auto(self, file_path: str):
        """Auto-detect source type and/or tile size before processing.

        Modifies self.params.point_cloud_source and self.tile_size in place.
        No-op when the user has explicitly set them.
        """
        # ── Point cloud source ──
        if self.params.point_cloud_source == "auto":
            try:
                detected, reason, info = guess_scan_type(file_path)
                if detected == "unknown":
                    # Default to LiDAR when signals are inconclusive
                    self.log.emit(f"Auto-detect (source): unknown ({reason}) — "
                                  f"defaulting to LiDAR")
                    self.params.point_cloud_source = "lidar"
                else:
                    self.log.emit(f"Auto-detect (source): {detected.upper()} — "
                                  f"{reason}")
                    self.params.point_cloud_source = detected
            except Exception as e:
                self.log.emit(f"Auto-detect (source) failed ({e}); defaulting to LiDAR")
                self.params.point_cloud_source = "lidar"

        # ── Tile size ──
        # tile_size == 0 → "Auto" special value from the GUI spinbox
        if self.tile_size == 0:
            try:
                size, info = estimate_optimal_tile_size(file_path)
                self.log.emit(
                    f"Auto-detect (tile size): {size:.0f} m  "
                    f"(density {info['density_pts_m2']:,.0f} pts/m², "
                    f"{info['point_count']:,} pts over {info['area_m2']:,.0f} m²)"
                )
                self.tile_size = float(size)
            except Exception as e:
                self.log.emit(f"Auto-detect (tile size) failed ({e}); using 100 m")
                self.tile_size = 100.0

    def _process_single_or_tile(self, file_path: str):
        """Shared path for single-file modes (skip tiling OR auto-tile)."""
        # Resolve any "Auto" settings from the source file's header
        self._resolve_auto(file_path)

        if self.skip_tiling:
            csv_path = process_single_file(
                file_path=file_path,
                out_dir=self.out_dir,
                params=self.params,
                log_fn=self.log.emit,
                progress_fn=self.progress.emit,
                cancelled_fn=lambda: self._cancel,
            )
            return csv_path

        src = Path(file_path)
        tile_dir = src.parent / f"Tile_{src.stem}"
        self.log.emit(f"Tiling to: {tile_dir}")
        tile_file(
            in_path=str(src),
            out_dir=str(tile_dir),
            tile_size=self.tile_size,
            log_fn=self.log.emit,
            progress_fn=self.progress.emit,
            cancelled_fn=lambda: self._cancel,
        )
        if self._cancel:
            return ""
        self.log.emit("Tiling complete — starting classification.\n")
        return process_all_tiles(
            tile_dir=str(tile_dir),
            out_dir=self.out_dir,
            params=self.params,
            log_fn=self.log.emit,
            progress_fn=self.progress.emit,
            cancelled_fn=lambda: self._cancel,
        )

    def run(self):
        try:
            # ── Multi-file → merge → process ─────────────────────────────
            if self.mode == "multi":
                if not self.multi_files:
                    self.log.emit("ERROR: no files selected.")
                    self.finished.emit("")
                    return

                first = Path(self.multi_files[0])
                if self.merged_name.strip():
                    merged_path = first.parent / Path(self.merged_name).name
                    if not merged_path.suffix:
                        merged_path = merged_path.with_suffix(".laz")
                else:
                    merged_path = first.parent / f"{first.stem}_merged.laz"

                merge_las_files(
                    in_paths=self.multi_files,
                    out_path=merged_path,
                    log_fn=self.log.emit,
                    progress_fn=self.progress.emit,
                    cancelled_fn=lambda: self._cancel,
                )
                if self._cancel:
                    self.finished.emit("")
                    return

                self.log.emit(f"\nMerged file: {merged_path}\nStarting classification...\n")
                csv_path = self._process_single_or_tile(str(merged_path))
                self.finished.emit(str(csv_path) if csv_path else "")
                return

            # ── Single file ───────────────────────────────────────────────
            if self.mode == "file":
                csv_path = self._process_single_or_tile(self.source)
                self.finished.emit(str(csv_path) if csv_path else "")
                return

            # ── Folder mode ───────────────────────────────────────────────
            # Auto-detect source type from a sample tile if requested
            if self.params.point_cloud_source == "auto":
                sample_tiles = sorted(list(Path(self.source).glob("*.la[sz]")))
                if sample_tiles:
                    self._resolve_auto(str(sample_tiles[0]))
                    self.log.emit(f"  (Auto-detect used sample: {sample_tiles[0].name})")
                else:
                    self.log.emit("No sample tile found — defaulting to LiDAR")
                    self.params.point_cloud_source = "lidar"

            csv_path = process_all_tiles(
                tile_dir=self.source,
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
        self.setWindowTitle("Starling Ground-Contact Classifier  —  V02 (WIP)")
        self.setMinimumSize(700, 460)
        self.resize(900, 600)
        self._worker: WorkerThread | None = None
        self._build_ui()


    def _build_ui(self):
        # ── Menu bar ──
        self._build_menu()

        # ── Scrollable content area (top half) ────────────────────────────
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(12, 12, 12, 12)
        self._sec_io     = self._io_group()
        self._sec_ground = self._ground_group()
        self._sec_photo  = self._photo_group()
        self._sec_bird   = self._bird_group()
        self._sec_height = self._height_group()
        content_layout.addWidget(self._sec_io)
        content_layout.addWidget(self._sec_ground)
        content_layout.addWidget(self._sec_photo)
        content_layout.addWidget(self._sec_bird)
        content_layout.addWidget(self._sec_height)
        content_layout.addWidget(self._run_group())
        # NO stretch at bottom — content should hug the top, log takes the rest

        # Photogrammetry section hidden by default (only shown in photo mode)
        self._sec_photo.setVisible(False)

        # Wire status-badge refresh on relevant control changes
        self._wire_status_badges()
        self._refresh_status_badges()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(content)

        # ── Log box (bottom half — outside scroll so it's always visible) ─
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Consolas", 9))
        self.log_box.setMinimumHeight(120)

        log_wrap = QWidget()
        log_lay = QVBoxLayout(log_wrap)
        # Right margin matches the scroll area's vertical scrollbar width so
        # the log box aligns horizontally with the settings panels above
        # (whether the scrollbar is currently visible or reserved).
        sb_width = scroll.verticalScrollBar().sizeHint().width()
        log_lay.setContentsMargins(12, 0, 12 + sb_width, 12)
        log_lay.setSpacing(4)
        log_lay.addWidget(QLabel("Log:"))
        log_lay.addWidget(self.log_box)

        # Force the scroll area to ALWAYS reserve scrollbar space so the
        # margin math above stays correct regardless of content height
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        # Splitter so the user can drag the divider between params and log
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(scroll)
        splitter.addWidget(log_wrap)
        # Stretch factors: 0 = don't grow this pane, 1 = fill remaining space.
        # Scroll area sizes to content; log fills everything below.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # Initial split: give scroll just enough for compact/collapsed content
        splitter.setSizes([320, 500])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)

        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(splitter)

    # ── I/O ──────────────────────────────────────────────────────────────
    def _io_group(self):
        sec = CollapsibleSection("Input / Output", start_open=True)
        outer = sec.content_layout

        # Mode selector
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Input mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Tile folder  (pre-tiled *.laz files)", "folder")
        self.mode_combo.addItem("Single LAZ file  (auto-tile then process)", "file")
        self.mode_combo.addItem("Multiple LAZ files  (merge then process)", "multi")
        self.mode_combo.currentIndexChanged.connect(self._on_input_mode_change)
        mode_row.addWidget(self.mode_combo)
        mode_row.addStretch()
        outer.addLayout(mode_row)

        # ── Point cloud source (visible at top level, drives whole pipeline) ──
        pcs_row = QHBoxLayout()
        pcs_lbl = QLabel("Point cloud source:")
        pcs_lbl.setStyleSheet("font-weight: bold;")
        pcs_row.addWidget(pcs_lbl)
        self.pcs_combo = QComboBox()
        self.pcs_combo.addItem(
            "Auto-detect  (from intensity / NIR / RGB signals)", "auto"
        )
        self.pcs_combo.addItem("LiDAR / mobile scan", "lidar")
        self.pcs_combo.addItem(
            "Drone photogrammetry  (NDVI / VARI + building detection)",
            "photogrammetry"
        )
        self.pcs_combo.setToolTip(
            "Auto-detect: reads intensity / NIR / RGB and picks LiDAR or\n"
            "Photogrammetry automatically per source file.\n\n"
            "LiDAR: CSF + multi-return-friendly bird detection.\n"
            "Photogrammetry: RGB/NIR vegetation index + building detection via\n"
            "plane RANSAC; bird detection auto-disabled (SfM birds are unreliable)."
        )
        self.pcs_combo.setCurrentIndex(0)   # default = Auto-detect
        self.pcs_combo.currentIndexChanged.connect(self._on_pcs_change)
        pcs_row.addWidget(self.pcs_combo, stretch=1)
        outer.addLayout(pcs_row)

        # Stacked input panels
        self.input_stack = QStackedWidget()

        # ── Folder mode panel ──
        folder_widget = QWidget()
        fg = QGridLayout(folder_widget)
        fg.setColumnMinimumWidth(0, 110)
        fg.setColumnStretch(1, 1)
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
        ff.setColumnMinimumWidth(0, 110)
        ff.setColumnStretch(1, 1)
        ff.addWidget(QLabel("LAZ file:"), 0, 0)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Single .laz/.las file — will be tiled automatically")
        ff.addWidget(self.file_edit, 0, 1)
        btn_file = QPushButton("Browse…")
        btn_file.clicked.connect(self._browse_file)
        ff.addWidget(btn_file, 0, 2)

        ff.addWidget(QLabel("Tile size (m):"), 1, 0)
        self.tile_size_spin = QDoubleSpinBox()
        # Value 0 = "Auto (from density)" — see setSpecialValueText below
        self.tile_size_spin.setRange(0.0, 500.0)
        self.tile_size_spin.setValue(0.0)
        self.tile_size_spin.setSingleStep(5.0)
        self.tile_size_spin.setSpecialValueText("Auto (from density)")
        self.tile_size_spin.setToolTip(
            "Set to 'Auto' (leave at 0) to have the tile size computed from\n"
            "point density — aiming for ~30M points per tile."
        )
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

        # ── Multi-file mode panel ──
        multi_widget = QWidget()
        mf = QGridLayout(multi_widget)
        mf.setColumnMinimumWidth(0, 110)
        mf.setColumnStretch(1, 1)
        mf.addWidget(QLabel("LAZ files:"), 0, 0)
        self.files_list = QTextEdit()
        self.files_list.setPlaceholderText(
            "Click Browse to pick multiple LAZ/LAS files (Ctrl+click to multi-select)"
        )
        self.files_list.setMaximumHeight(80)
        self.files_list.setReadOnly(True)
        mf.addWidget(self.files_list, 0, 1)
        btn_files = QPushButton("Browse…")
        btn_files.clicked.connect(self._browse_multi_files)
        mf.addWidget(btn_files, 0, 2)

        mf.addWidget(QLabel("Merged output:"), 1, 0)
        self.merged_name_edit = QLineEdit()
        self.merged_name_edit.setPlaceholderText("e.g. site_merged.laz  (auto-named if blank)")
        mf.addWidget(self.merged_name_edit, 1, 1, 1, 2)

        self.multi_skip_tile_chk = QCheckBox(
            "Skip tiling after merge — process whole merged file as one  "
            "(recommended for total < ~300 M points)"
        )
        self.multi_skip_tile_chk.setChecked(True)   # default ON for merged mode
        mf.addWidget(self.multi_skip_tile_chk, 2, 0, 1, 3)

        mf.addWidget(QLabel("Tile size (m):"), 3, 0)
        self.multi_tile_size_spin = QDoubleSpinBox()
        self.multi_tile_size_spin.setRange(0.0, 500.0)
        self.multi_tile_size_spin.setValue(0.0)
        self.multi_tile_size_spin.setSingleStep(5.0)
        self.multi_tile_size_spin.setSpecialValueText("Auto (from density)")
        self.multi_tile_size_spin.setEnabled(False)
        mf.addWidget(self.multi_tile_size_spin, 3, 1)
        self.multi_skip_tile_chk.toggled.connect(
            lambda on: self.multi_tile_size_spin.setEnabled(not on)
        )

        self._multi_files: list[str] = []
        self.input_stack.addWidget(multi_widget)

        outer.addWidget(self.input_stack)

        # Common output controls — fixed-width labels, line edits stretch
        common = QGridLayout()
        common.setColumnMinimumWidth(0, 110)   # label column
        common.setColumnStretch(1, 1)           # field column expands
        common.setColumnMinimumWidth(2, 90)    # browse / second label
        common.setColumnStretch(3, 0)

        # Row 0 — output folder spans full width
        common.addWidget(QLabel("Output folder:"), 0, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Where to save classified LAS + CSV")
        common.addWidget(self.out_edit, 0, 1, 1, 2)
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out)
        common.addWidget(btn_out, 0, 3)

        # Row 1 — Buffer | Parallel workers  (own narrow cells)
        import os
        buf_label = QLabel("Buffer (m):")
        common.addWidget(buf_label, 1, 0)
        self.buf_spin = QDoubleSpinBox()
        self.buf_spin.setRange(1.0, 20.0)
        self.buf_spin.setValue(5.0)
        self.buf_spin.setSingleStep(1.0)
        self.buf_spin.setMaximumWidth(120)
        common.addWidget(self.buf_spin, 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)

        common.addWidget(QLabel("Parallel workers:"), 1, 2,
                         alignment=Qt.AlignmentFlag.AlignRight)
        self.workers_spin = QSpinBox()
        cpu = os.cpu_count() or 4
        self.workers_spin.setRange(1, max(1, cpu))
        self.workers_spin.setValue(min(max(1, cpu // 2), 6))
        self.workers_spin.setMaximumWidth(80)
        self.workers_spin.setToolTip(
            "1 = sequential.  N = run N tiles in parallel processes.\n"
            "Tiles ≥150M points always run solo to protect RAM."
        )
        common.addWidget(self.workers_spin, 1, 3,
                         alignment=Qt.AlignmentFlag.AlignLeft)

        # Row 2 — Output format
        common.addWidget(QLabel("Output format:"), 2, 0)
        self.output_format_combo = QComboBox()
        self.output_format_combo.addItem("LAZ  (compressed)", "laz")
        self.output_format_combo.addItem("LAS  (uncompressed)", "las")
        self.output_format_combo.setCurrentIndex(0)
        common.addWidget(self.output_format_combo, 2, 1, 1, 3)

        # Row 3 — LAS version
        common.addWidget(QLabel("LAS version:"), 3, 0)
        self.las_version_combo = QComboBox()
        self.las_version_combo.addItem("Match source  (recommended)", "match")
        self.las_version_combo.addItem("1.4", "1.4")
        self.las_version_combo.addItem("1.2", "1.2")
        self.las_version_combo.setCurrentIndex(0)
        common.addWidget(self.las_version_combo, 3, 1, 1, 3)

        # Row 4 — Merge output tiles into single LAZ
        self.merge_output_chk = QCheckBox(
            "Merge output tiles into single LAZ  "
            "(recommended for single-file and multi-merge modes)"
        )
        self.merge_output_chk.setChecked(True)
        self.merge_output_chk.setToolTip(
            "After classification finishes, all *_classified.laz tiles are stream-merged\n"
            "into one final LAZ. Individual tile files are moved to a _shards_* subfolder\n"
            "so the top-level output folder shows one clean classified file.\n\n"
            "Uncheck if you want to keep the per-tile outputs as-is."
        )
        common.addWidget(self.merge_output_chk, 4, 0, 1, 4)

        outer.addLayout(common)
        return sec

    def _on_input_mode_change(self, idx):
        self.input_stack.setCurrentIndex(idx)

    # ── Ground ───────────────────────────────────────────────────────────
    def _ground_group(self):
        sec = CollapsibleSection("Ground Classification", start_open=False)
        outer = sec.content_layout

        # (Point cloud source moved to Input/Output section for visibility)

        # Source selector — compute fresh OR use existing class 2 from source
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self.ground_src_combo = QComboBox()
        self.ground_src_combo.addItem(
            "Compute fresh  (run CSF / Grid Minimum)", "compute"
        )
        self.ground_src_combo.addItem(
            "Use existing classification 2 from source LAS", "from_file"
        )
        self.ground_src_combo.currentIndexChanged.connect(self._on_ground_src_change)
        src_row.addWidget(self.ground_src_combo)
        src_row.addStretch()
        outer.addLayout(src_row)

        # Method selector — only relevant when computing fresh
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
        return sec

    def _on_method_change(self, idx):
        self.ground_stack.setCurrentIndex(idx)

    def _on_ground_src_change(self, idx):
        # When using existing classification, hide method + param panels
        from_file = (self.ground_src_combo.currentData() == "from_file")
        self.method_combo.setEnabled(not from_file)
        self.ground_stack.setEnabled(not from_file)

    def _on_pcs_change(self, idx):
        """Toggle sections and apply safe defaults based on the chosen source."""
        current = self.pcs_combo.currentData()
        photo = (current == "photogrammetry")
        auto  = (current == "auto")

        # Photogrammetry panel: visible when photo is explicitly chosen,
        # or when auto (detection may resolve to photo — user can still tune)
        if hasattr(self, "_sec_photo"):
            self._sec_photo.setVisible(photo or auto)

        # Bird detection: greyed only when photo is explicitly chosen
        if hasattr(self, "_sec_bird"):
            self._sec_bird.setEnabled(not photo)
            if photo:
                title = "Bird Contact Detection  (disabled in photogrammetry mode)"
            elif auto:
                title = "Bird Contact Detection  (skipped if auto detects photogrammetry)"
            else:
                title = "Bird Contact Detection"
            self._sec_bird.toggle_btn.setText(title)

        # Auto-apply defaults only for explicit LiDAR / photogrammetry —
        # 'auto' leaves user's settings alone until runtime detection resolves.
        if photo:
            if hasattr(self, "workers_spin"):
                self.workers_spin.setValue(min(2, self.workers_spin.maximum()))
            if hasattr(self, "veg_index_combo"):
                for i in range(self.veg_index_combo.count()):
                    if self.veg_index_combo.itemData(i) == "vari":
                        self.veg_index_combo.setCurrentIndex(i)
                        break
            if hasattr(self, "_vari_thr"):
                self._vari_thr.setValue(0.10)
            self._log("Drone photogrammetry defaults applied: "
                      "workers=2, index=VARI, VARI threshold=0.10")
        elif current == "lidar":
            if hasattr(self, "workers_spin"):
                import os
                cpu = os.cpu_count() or 4
                self.workers_spin.setValue(min(max(1, cpu // 2), 6))
            if hasattr(self, "veg_index_combo"):
                for i in range(self.veg_index_combo.count()):
                    if self.veg_index_combo.itemData(i) == "auto":
                        self.veg_index_combo.setCurrentIndex(i)
                        break

        self._refresh_status_badges()

    # ── Bird detection ────────────────────────────────────────────────────
    def _bird_group(self):
        sec = CollapsibleSection("Bird Contact Detection", start_open=False)
        grid_w = QWidget()
        g = QGridLayout(grid_w)
        g.setContentsMargins(0, 0, 0, 0)

        self._add_dspin(g, 0, "Min height above ground (m):", 0.01, 0.20, 0.02, 0.01, "nz_min")
        self._add_dspin(g, 1, "Max height above ground (m):", 0.10, 1.00, 0.40, 0.05, "nz_max")
        self._add_dspin(g, 2, "DBSCAN eps (m):", 0.05, 2.00, 0.30, 0.05, "dbscan_eps")
        self._add_spin(g, 3, "DBSCAN min points:", 2, 100, 8, "dbscan_min")
        self._add_dspin(g, 4, "Min cluster footprint (m²):", 0.001, 0.10, 0.005, 0.001, "min_foot")
        self._add_dspin(g, 5, "Max cluster footprint (m²):", 0.10, 5.00, 1.00, 0.10, "max_foot")

        sec.content_layout.addWidget(grid_w)
        return sec

    # ── Menu bar (v2) ────────────────────────────────────────────────────
    def _build_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("&Help")

        about_act = help_menu.addAction("About Starling Classifier…")
        about_act.triggered.connect(self._show_about)

        docs_act = help_menu.addAction("Open README…")
        docs_act.triggered.connect(self._open_readme)

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox
        from PyQt6.QtCore import Qt
        text = (
            "<h2>Starling Ground-Contact Classifier</h2>"
            "<p><b>Version:</b> V02 &mdash; WIP<br>"
            "<b>Build channel:</b> Photogrammetry Edition</p>"
            "<hr>"
            "<p><b>Created by:</b> Arun Prashad Hariharan<br>"
            "<b>Email:</b> <a href='mailto:arunprashadh@gmail.com'>"
            "arunprashadh@gmail.com</a><br>"
            "<b>LinkedIn:</b> "
            "<a href='https://www.linkedin.com/in/arun-prashad-hariharan'>"
            "linkedin.com/in/arun-prashad-hariharan</a></p>"
            "<hr>"
            "<p style='font-size:10px;color:#888;'>"
            "Classifies LiDAR and drone photogrammetry point clouds into<br>"
            "ground / vegetation / buildings / bird contacts using CSF,<br>"
            "NDVI/VARI vegetation indices, and open3d plane RANSAC.</p>"
        )
        msg = QMessageBox(self)
        msg.setWindowTitle("About")
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg.setText(text)
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _open_readme(self):
        from pathlib import Path
        import webbrowser
        readme = Path(__file__).resolve().parents[1] / "README.md"
        if readme.exists():
            webbrowser.open(readme.as_uri())
        else:
            self._log(f"README.md not found at {readme}")

    # ── Photogrammetry section (v2) ──────────────────────────────────────
    def _photo_group(self):
        sec = CollapsibleSection(
            "Photogrammetry Pre-Classification  (vegetation index + building detection)",
            start_open=True,
        )
        outer = sec.content_layout

        # Vegetation index selection
        vi_row = QHBoxLayout()
        vi_row.addWidget(QLabel("Vegetation index:"))
        self.veg_index_combo = QComboBox()
        self.veg_index_combo.addItem("Auto  (NDVI if NIR, else VARI)", "auto")
        self.veg_index_combo.addItem("NDVI  (requires NIR channel)", "ndvi")
        self.veg_index_combo.addItem("VARI  (RGB-only, atmosphere-resistant)", "vari")
        self.veg_index_combo.addItem("ExG  (RGB-only, simple green-excess)", "exg")
        vi_row.addWidget(self.veg_index_combo)
        vi_row.addStretch()
        outer.addLayout(vi_row)

        thr_w = QWidget()
        g = QGridLayout(thr_w)
        g.setContentsMargins(0, 0, 0, 0)
        self._add_dspin(g, 0, "NDVI threshold:",  0.05, 0.60, 0.20, 0.01, "ndvi_thr")
        self._add_dspin(g, 1, "VARI threshold:",  0.05, 0.60, 0.15, 0.01, "vari_thr")
        self._add_dspin(g, 2, "ExG threshold:",   1.0, 100.0, 15.0, 1.0, "exg_thr")
        outer.addWidget(thr_w)

        # Building detection
        self.building_chk = QCheckBox(
            "Detect buildings via planar RANSAC  (class 6, rooftops)"
        )
        self.building_chk.setChecked(True)
        outer.addWidget(self.building_chk)

        bg_w = QWidget()
        bg = QGridLayout(bg_w)
        bg.setContentsMargins(0, 0, 0, 0)
        self._add_dspin(bg, 0, "Min planar area (m²):",       5.0, 200.0, 20.0, 5.0, "bldg_area")
        self._add_dspin(bg, 1, "Min height above ground (m):", 1.0, 10.0, 2.5, 0.5, "bldg_hagl")
        self._add_dspin(bg, 2, "Plane RANSAC threshold (m):",  0.03, 0.50, 0.10, 0.01, "bldg_plane_thr")
        self._add_dspin(bg, 3, "Cluster eps (m):",             0.3, 3.0, 1.0, 0.1, "bldg_cluster_eps")
        self._add_spin (bg, 4, "Min cluster points:",          50, 5000, 300, "bldg_min_pts")
        outer.addWidget(bg_w)

        return sec

    # ── Height classification (TerraScan-style) ──────────────────────────
    def _height_group(self):
        sec = CollapsibleSection(
            "Height Classification  (vegetation + low noise + model keypoints)",
            start_open=False,
        )
        outer = sec.content_layout

        # Master toggle
        self.veg_enable_chk = QCheckBox(
            "Classify vegetation by height above ground  (classes 3 / 4 / 5 / 7)"
        )
        self.veg_enable_chk.setChecked(True)
        outer.addWidget(self.veg_enable_chk)

        g = QGridLayout()
        self._add_dspin(g, 0, "Low veg starts at nZ (m):",   0.02, 1.0, 0.10, 0.01, "veg_low_min")
        self._add_dspin(g, 1, "Low / Med boundary nZ (m):",  0.20, 5.0, 1.00, 0.10, "veg_low_max")
        self._add_dspin(g, 2, "Med / High boundary nZ (m):", 1.00, 20.0, 3.00, 0.50, "veg_med_max")
        self._add_dspin(g, 3, "Low noise threshold nZ (m):", -2.0, -0.01, -0.10, 0.01, "noise_thr")
        outer.addLayout(g)

        # Model keypoints
        self.modelkey_chk = QCheckBox(
            "Also classify model keypoints  (class 8 — thinned ground for TIN building)"
        )
        self.modelkey_chk.setChecked(False)
        outer.addWidget(self.modelkey_chk)

        g2 = QGridLayout()
        self._add_dspin(g2, 0, "Model keypoint grid step (m):", 1.0, 50.0, 8.0, 1.0, "modelkey_step")
        outer.addLayout(g2)

        return sec

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

    def _browse_multi_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select multiple LAZ / LAS files (Ctrl+click)", "",
            "LAS/LAZ files (*.laz *.las);;All files (*.*)"
        )
        if files:
            self._multi_files = files
            self.files_list.setPlainText("\n".join(files))
            self._log(f"{len(files)} file(s) selected for merging.")

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
            ground_source=self.ground_src_combo.currentData(),
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
            # Height classification
            classify_vegetation=self.veg_enable_chk.isChecked(),
            veg_low_min=self._veg_low_min.value(),
            veg_low_max=self._veg_low_max.value(),
            veg_med_max=self._veg_med_max.value(),
            noise_below_ground=self._noise_thr.value(),
            classify_model_keys=self.modelkey_chk.isChecked(),
            modelkey_step_m=self._modelkey_step.value(),
            merge_output_to_single=self.merge_output_chk.isChecked(),
            # ── v2 — Photogrammetry ─────────────────────────────────────
            point_cloud_source=self.pcs_combo.currentData(),
            vegetation_index=self.veg_index_combo.currentData(),
            ndvi_threshold=self._ndvi_thr.value(),
            vari_threshold=self._vari_thr.value(),
            exg_threshold=self._exg_thr.value(),
            detect_buildings_flag=self.building_chk.isChecked(),
            building_min_area_m2=self._bldg_area.value(),
            building_min_height_agl=self._bldg_hagl.value(),
            building_plane_threshold=self._bldg_plane_thr.value(),
            building_cluster_eps=self._bldg_cluster_eps.value(),
            building_min_cluster_pts=self._bldg_min_pts.value(),
        )

    # ── Status badges on collapsed section headers ────────────────────────
    def _wire_status_badges(self):
        """Connect every control whose change should refresh the badges."""
        # I/O
        self.mode_combo.currentIndexChanged.connect(self._refresh_status_badges)
        self.workers_spin.valueChanged.connect(self._refresh_status_badges)
        self.output_format_combo.currentIndexChanged.connect(self._refresh_status_badges)
        self.skip_tile_chk.toggled.connect(self._refresh_status_badges)
        self.merge_output_chk.toggled.connect(self._refresh_status_badges)
        # Ground
        self.ground_src_combo.currentIndexChanged.connect(self._refresh_status_badges)
        self.method_combo.currentIndexChanged.connect(self._refresh_status_badges)
        # Bird (always on — but show key thresholds)
        self._nz_min.valueChanged.connect(self._refresh_status_badges)
        self._nz_max.valueChanged.connect(self._refresh_status_badges)
        # Height
        self.veg_enable_chk.toggled.connect(self._refresh_status_badges)
        self.modelkey_chk.toggled.connect(self._refresh_status_badges)
        # Photogrammetry (v2)
        self.pcs_combo.currentIndexChanged.connect(self._refresh_status_badges)
        self.veg_index_combo.currentIndexChanged.connect(self._refresh_status_badges)
        self.building_chk.toggled.connect(self._refresh_status_badges)

    def _refresh_status_badges(self):
        """Rebuild each section's right-hand summary text."""
        # ── Input / Output ─────────────────────────────────────────────
        mc = self.mode_combo.currentData()
        if mc == "folder":
            mode = "Tile folder"
        elif mc == "file":
            mode = "Single LAZ"
            if self.skip_tile_chk.isChecked():
                mode += " (no tile)"
        else:   # multi
            n = len(self._multi_files) if self._multi_files else 0
            mode = f"Multi-merge ({n} files)"
            if self.multi_skip_tile_chk.isChecked():
                mode += " (no tile)"
        fmt = self.output_format_combo.currentData().upper()
        workers = self.workers_spin.value()
        worker_txt = "1 worker" if workers == 1 else f"{workers} workers"
        merge_txt = " · merge→1" if self.merge_output_chk.isChecked() else " · tiles kept"
        self._sec_io.set_status(f"{mode} · {fmt} · {worker_txt}{merge_txt}", "#8cf")

        # ── Ground Classification ──────────────────────────────────────
        pcs = self.pcs_combo.currentData() if hasattr(self, "pcs_combo") else "lidar"
        src = self.ground_src_combo.currentData()
        pcs_tag = " · Photo" if pcs == "photogrammetry" else " · LiDAR"
        if src == "from_file":
            self._sec_ground.set_status("Use existing class 2" + pcs_tag, "#fc8")
        else:
            method = self.method_combo.currentData().upper()
            self._sec_ground.set_status(f"Compute fresh · {method}{pcs_tag}", "#8fc")

        # ── Photogrammetry (only meaningful in photo mode) ─────────────
        if hasattr(self, "_sec_photo"):
            if pcs == "photogrammetry":
                idx = self.veg_index_combo.currentData().upper()
                bldg = "Bldg ON" if self.building_chk.isChecked() else "Bldg OFF"
                self._sec_photo.set_status(f"{idx} · {bldg}", "#8fc")
            else:
                self._sec_photo.set_status("(LiDAR mode)", "#666")

        # ── Bird Contact Detection ─────────────────────────────────────
        if pcs == "photogrammetry":
            self._sec_bird.set_status("Disabled (photogrammetry)", "#666")
        else:
            self._sec_bird.set_status(
                f"nZ {self._nz_min.value():.2f}–{self._nz_max.value():.2f} m · ON",
                "#8fc",
            )

        # ── Height Classification ──────────────────────────────────────
        veg_on = self.veg_enable_chk.isChecked()
        key_on = self.modelkey_chk.isChecked()
        if not veg_on and not key_on:
            self._sec_height.set_status("OFF", "#888")
        else:
            parts = []
            parts.append("Veg ON" if veg_on else "Veg OFF")
            parts.append("Keys ON" if key_on else "Keys OFF")
            colour = "#8fc" if veg_on else "#fc8"
            self._sec_height.set_status(" · ".join(parts), colour)

    def _start(self):
        mode = self.mode_combo.currentData()
        out_dir = self.out_edit.text().strip()

        if not out_dir:
            self._log("ERROR: Select an output folder.")
            return

        source = ""
        tile_size = 100.0
        skip_tiling = False
        multi_files: list[str] = []
        merged_name = ""

        if mode == "folder":
            source = self.tile_edit.text().strip()
            if not source or not Path(source).is_dir():
                self._log("ERROR: Select a valid tile folder.")
                return
        elif mode == "file":
            source = self.file_edit.text().strip()
            if not source or not Path(source).is_file():
                self._log("ERROR: Select a valid LAZ file.")
                return
            tile_size = self.tile_size_spin.value()
            skip_tiling = self.skip_tile_chk.isChecked()
        elif mode == "multi":
            multi_files = list(self._multi_files)
            if not multi_files:
                self._log("ERROR: Select multiple LAZ files via Browse.")
                return
            for f in multi_files:
                if not Path(f).is_file():
                    self._log(f"ERROR: file not found: {f}")
                    return
            tile_size = self.multi_tile_size_spin.value()
            skip_tiling = self.multi_skip_tile_chk.isChecked()
            merged_name = self.merged_name_edit.text().strip()
        else:
            self._log(f"ERROR: unknown mode {mode!r}")
            return

        params = self._build_params()
        self.progress_bar.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self._worker = WorkerThread(
            mode, source, out_dir, params,
            tile_size=tile_size,
            skip_tiling=skip_tiling,
            multi_files=multi_files,
            merged_name=merged_name,
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
