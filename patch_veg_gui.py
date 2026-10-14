"""
patch_veg_gui.py — GUI wrapper for the vegetation patch script.
Builds to a standalone EXE via PyInstaller (no Python required on target).
"""

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)


# ─── core logic (same as patch_veg.py, inlined so EXE is self-contained) ─────

def _fill_nan(grid):
    from scipy.ndimage import distance_transform_edt
    valid = ~np.isnan(grid)
    if not valid.any():
        return np.zeros_like(grid)
    _, idx = distance_transform_edt(~valid, return_indices=True)
    out = grid.copy()
    out[~valid] = grid[idx[0][~valid], idx[1][~valid]]
    return out


def _classify_heights(cls, nz, veg_low_min, veg_low_max, veg_med_max,
                      noise_below, ground_thr, protect=(2, 20)):
    cls = cls.copy()
    protected = np.zeros(len(cls), dtype=bool)
    for c in protect:
        protected |= cls == c
    cand = ~protected
    finite = np.isfinite(nz)
    c = cand & finite
    cls[c & (nz <= noise_below)]                           = 7
    cls[c & (nz >= veg_low_min) & (nz < veg_low_max)]     = 3
    cls[c & (nz >= veg_low_max) & (nz < veg_med_max)]     = 4
    cls[c & (nz >= veg_med_max)]                           = 5
    return cls


def _patch_one(args):
    import laspy
    (src, dst, veg_low_min, veg_low_max, veg_med_max,
     noise_below, ground_thr, chunk_size, cell_size) = args

    src, dst = Path(src), Path(dst)
    t0 = time.time()

    with laspy.open(str(src)) as reader:
        h = reader.header
        xmin, ymin = float(h.mins[0]), float(h.mins[1])
        xmax, ymax = float(h.maxs[0]), float(h.maxs[1])
        src_fmt  = h.point_format.id
        src_ver  = h.version
        src_scl  = h.scales
        src_off  = h.offsets
        try:
            src_vlrs = list(h.vlrs)
        except Exception:
            src_vlrs = []

        ncols = int(math.ceil((xmax - xmin) / cell_size)) + 2
        nrows = int(math.ceil((ymax - ymin) / cell_size)) + 2
        z_sum   = np.zeros((nrows, ncols), dtype=np.float64)
        z_count = np.zeros((nrows, ncols), dtype=np.int32)

        for chunk in reader.chunk_iterator(chunk_size):
            mask = np.asarray(chunk.classification, dtype=np.uint8) == 2
            if not mask.any():
                continue
            xs = np.asarray(chunk.x, dtype=np.float64)[mask]
            ys = np.asarray(chunk.y, dtype=np.float64)[mask]
            zs = np.asarray(chunk.z, dtype=np.float64)[mask]
            ci = np.clip(((xs - xmin) / cell_size).astype(np.int32), 0, ncols - 1)
            ri = np.clip(((ys - ymin) / cell_size).astype(np.int32), 0, nrows - 1)
            np.add.at(z_sum,   (ri, ci), zs)
            np.add.at(z_count, (ri, ci), 1)

    if int((z_count > 0).sum()) == 0:
        raise RuntimeError("No class-2 (ground) points found")

    z_grid = np.where(z_count > 0, z_sum / np.maximum(z_count, 1), np.nan)
    if np.isnan(z_grid).any():
        z_grid = _fill_nan(z_grid)

    def _nz(xs, ys, zs):
        ci = np.clip(((xs - xmin) / cell_size).astype(np.int32), 0, ncols - 1)
        ri = np.clip(((ys - ymin) / cell_size).astype(np.int32), 0, nrows - 1)
        return zs - z_grid[ri, ci]

    ver_str = f"{src_ver.major}.{src_ver.minor}"
    new_hdr = laspy.LasHeader(point_format=src_fmt, version=ver_str)
    new_hdr.scales  = src_scl
    new_hdr.offsets = src_off
    try:
        new_hdr.vlrs.extend(src_vlrs)
    except Exception:
        pass

    do_compress = dst.suffix.lower() == ".laz"
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 7: 0, 20: 0}
    n_pts  = 0
    tmp    = dst.with_suffix(".tmp_patch")

    try:
        with laspy.open(str(src)) as reader, \
             open(str(tmp), "wb") as _fout, \
             laspy.LasWriter(_fout, header=new_hdr, do_compress=do_compress) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                xs  = np.asarray(chunk.x, dtype=np.float64)
                ys  = np.asarray(chunk.y, dtype=np.float64)
                zs  = np.asarray(chunk.z, dtype=np.float64)
                nz  = _nz(xs, ys, zs)
                cls = np.asarray(chunk.classification, dtype=np.uint8)
                cls = _classify_heights(cls, nz, veg_low_min, veg_low_max,
                                        veg_med_max, noise_below, ground_thr)
                chunk.classification = cls
                writer.write_points(chunk)
                n_pts += len(xs)
                for c in counts:
                    counts[c] += int((cls == c).sum())

        if dst.exists():
            dst.unlink()
        tmp.rename(dst)

    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    veg   = counts[3] + counts[4] + counts[5]
    noise = counts[7]
    return str(src), time.time() - t0, n_pts, veg, noise, counts


def _find_files(paths, pattern):
    found = []
    seen  = set()
    for p in paths:
        p = Path(p)
        if p.is_file():
            items = [p]
        elif p.is_dir():
            items = sorted(p.rglob(pattern))
        else:
            continue
        for f in items:
            key = f.resolve()
            if key not in seen:
                seen.add(key)
                found.append(f)
    return found


# ─── worker thread ────────────────────────────────────────────────────────────

class PatchWorker(QThread):
    log      = pyqtSignal(str)
    progress = pyqtSignal(int, int)   # (done, total)
    finished = pyqtSignal(int, int)   # (ok, err)

    def __init__(self, files, dst_mode, pattern,
                 veg_low_min, veg_low_max, veg_med_max,
                 noise_below, ground_thr, cell_size,
                 chunk_size, workers):
        super().__init__()
        self.files       = files          # list of Path
        self.dst_mode    = dst_mode       # "overwrite" or "alongside"
        self.pattern     = pattern
        self.veg_low_min = veg_low_min
        self.veg_low_max = veg_low_max
        self.veg_med_max = veg_med_max
        self.noise_below = noise_below
        self.ground_thr  = ground_thr
        self.cell_size   = cell_size
        self.chunk_size  = chunk_size
        self.workers     = workers
        self._cancel     = False

    def cancel(self):
        self._cancel = True

    def run(self):
        all_files = _find_files(self.files, self.pattern)
        if not all_files:
            self.log.emit("No matching files found.")
            self.finished.emit(0, 0)
            return

        self.log.emit(f"Found {len(all_files)} file(s) to patch.\n")
        tasks = []
        for src in all_files:
            if self.dst_mode == "overwrite":
                dst = src
            else:
                stem = src.stem
                if stem.endswith("_classified"):
                    stem = stem[:-len("_classified")]
                dst = src.with_name(stem + "_classified_vegpatch" + src.suffix)
            tasks.append((
                str(src), str(dst),
                self.veg_low_min, self.veg_low_max, self.veg_med_max,
                self.noise_below, self.ground_thr,
                self.chunk_size, self.cell_size,
            ))

        ok = err = 0
        total = len(tasks)
        done  = 0

        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            fmap = {pool.submit(_patch_one, t): t[0] for t in tasks}
            for fut in as_completed(fmap):
                if self._cancel:
                    pool.shutdown(wait=False, cancel_futures=True)
                    self.log.emit("\n⚠ Cancelled by user.")
                    break
                src = fmap[fut]
                done += 1
                try:
                    src_path, elapsed, n_pts, veg, noise, counts = fut.result()
                    ok += 1
                    name = Path(src_path).name
                    self.log.emit(
                        f"  ✓ {name}\n"
                        f"      {n_pts:,} pts   {elapsed/60:.1f} min\n"
                        f"      ground={counts[2]:,}  "
                        f"low_veg={counts[3]:,}  med_veg={counts[4]:,}  "
                        f"high_veg={counts[5]:,}  noise={counts[7]:,}  "
                        f"bird={counts[20]:,}"
                    )
                except Exception as e:
                    err += 1
                    self.log.emit(f"  ✗ {Path(src).name}  ERROR: {e}")
                self.progress.emit(done, total)

        self.log.emit(
            f"\n{'='*55}\n"
            f"Finished: {ok} patched, {err} failed — "
            f"{total} file(s) total."
        )
        self.finished.emit(ok, err)


# ─── drag-drop file list ──────────────────────────────────────────────────────

class DropList(QListWidget):
    """QListWidget that accepts dropped files and folders."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(120)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if path and path not in self._all_texts():
                self.addItem(path)
        e.acceptProposedAction()

    def _all_texts(self):
        return [self.item(i).text() for i in range(self.count())]

    def paths(self):
        return [Path(self.item(i).text()) for i in range(self.count())]


# ─── main window ─────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Starling — Patch Vegetation Classes")
        self.resize(860, 700)
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(splitter)

        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(6)

        # ── Input files/folders ───────────────────────────────────────────
        grp_files = QGroupBox("Input — LAZ / LAS files or folders (drag & drop supported)")
        gf_lay = QVBoxLayout(grp_files)

        self._file_list = DropList()
        gf_lay.addWidget(self._file_list)

        btn_row = QHBoxLayout()
        add_file_btn = QPushButton("Add file(s)…")
        add_file_btn.clicked.connect(self._add_files)
        add_dir_btn  = QPushButton("Add folder…")
        add_dir_btn.clicked.connect(self._add_folder)
        remove_btn   = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected)
        clear_btn    = QPushButton("Clear all")
        clear_btn.clicked.connect(self._file_list.clear)
        for b in (add_file_btn, add_dir_btn, remove_btn, clear_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        gf_lay.addLayout(btn_row)

        # pattern hint
        pat_row = QHBoxLayout()
        pat_row.addWidget(QLabel("File pattern (for folders):"))
        self._pattern_edit = QLabel("*_classified.laz")
        pat_row.addWidget(self._pattern_edit)
        pat_row.addStretch()
        gf_lay.addLayout(pat_row)

        top_lay.addWidget(grp_files)

        # ── Options ───────────────────────────────────────────────────────
        grp_opt = QGroupBox("Classification thresholds")
        opt_lay = QHBoxLayout(grp_opt)

        def _dspin(val, lo, hi, step=0.05, decimals=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setValue(val)
            s.setSuffix(" m")
            s.setFixedWidth(90)
            return s

        def _labeled(label, widget):
            w = QWidget()
            l = QVBoxLayout(w)
            l.setContentsMargins(0, 0, 0, 0)
            l.setSpacing(2)
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size:10px; color:#888;")
            l.addWidget(lbl)
            l.addWidget(widget)
            return w

        self._low_min   = _dspin(0.10, 0.00, 5.00)
        self._low_max   = _dspin(1.00, 0.10, 10.0)
        self._med_max   = _dspin(3.00, 0.50, 30.0)
        self._noise_thr = _dspin(-0.10, -5.0, 0.00)
        self._gnd_thr   = _dspin(0.05,  0.01, 1.00)
        self._cell_size = _dspin(1.00,  0.25, 10.0, step=0.25)

        opt_lay.addWidget(_labeled("Low-veg\nstarts (nZ≥)", self._low_min))
        opt_lay.addWidget(_labeled("Low/med\nboundary",     self._low_max))
        opt_lay.addWidget(_labeled("Med/high\nboundary",    self._med_max))

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color:#555;")
        opt_lay.addWidget(sep)

        opt_lay.addWidget(_labeled("Noise\n(nZ≤)",          self._noise_thr))
        opt_lay.addWidget(_labeled("Ground\nband (±)",       self._gnd_thr))
        opt_lay.addWidget(_labeled("Ground\ngrid cell",      self._cell_size))
        opt_lay.addStretch()

        top_lay.addWidget(grp_opt)

        # ── Output + parallelism ──────────────────────────────────────────
        grp_run = QGroupBox("Output & execution")
        run_lay = QHBoxLayout(grp_run)

        self._overwrite_chk = QCheckBox("Overwrite input files in-place")
        self._overwrite_chk.setToolTip(
            "Checked: patches are written back to the same file.\n"
            "Unchecked (safer): writes *_classified_vegpatch.laz alongside.")
        run_lay.addWidget(self._overwrite_chk)

        run_lay.addSpacing(20)
        run_lay.addWidget(QLabel("Workers:"))
        self._workers_spin = QSpinBox()
        self._workers_spin.setRange(1, 32)
        self._workers_spin.setValue(4)
        self._workers_spin.setFixedWidth(55)
        run_lay.addWidget(self._workers_spin)

        run_lay.addSpacing(10)
        run_lay.addWidget(QLabel("Chunk (M pts):"))
        self._chunk_spin = QSpinBox()
        self._chunk_spin.setRange(1, 50)
        self._chunk_spin.setValue(5)
        self._chunk_spin.setFixedWidth(55)
        run_lay.addWidget(self._chunk_spin)
        run_lay.addStretch()

        top_lay.addWidget(grp_run)

        # ── Run / Cancel ──────────────────────────────────────────────────
        act_row = QHBoxLayout()
        self._run_btn    = QPushButton("▶  Patch Vegetation Classes")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setStyleSheet(
            "QPushButton{background:#2a7a2a;color:#fff;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#3a9a3a;}"
            "QPushButton:disabled{background:#555;color:#888;}")
        self._run_btn.clicked.connect(self._run)

        self._cancel_btn = QPushButton("■  Cancel")
        self._cancel_btn.setFixedHeight(36)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet(
            "QPushButton{background:#7a2a2a;color:#fff;font-weight:bold;border-radius:4px;}"
            "QPushButton:hover{background:#9a3a3a;}"
            "QPushButton:disabled{background:#555;color:#888;}")
        self._cancel_btn.clicked.connect(self._cancel)

        act_row.addWidget(self._run_btn)
        act_row.addWidget(self._cancel_btn)
        top_lay.addLayout(act_row)

        # ── Progress ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        top_lay.addWidget(self._progress)

        splitter.addWidget(top)

        # ── Log ───────────────────────────────────────────────────────────
        log_widget = QWidget()
        log_lay = QVBoxLayout(log_widget)
        log_lay.setContentsMargins(0, 0, 0, 0)
        log_lay.setSpacing(3)

        log_hdr = QHBoxLayout()
        log_hdr.addWidget(QLabel("Log"))
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setFixedWidth(60)
        clear_log_btn.clicked.connect(lambda: self._log_box.clear())
        log_hdr.addWidget(clear_log_btn)
        log_hdr.addStretch()
        log_lay.addLayout(log_hdr)

        self._log_box = QPlainTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setFont(QFont("Consolas", 9))
        self._log_box.setStyleSheet(
            "QPlainTextEdit{background:#1a1a1a;color:#d4d4d4;"
            "border:1px solid #333;border-radius:4px;}")
        log_lay.addWidget(self._log_box)
        splitter.addWidget(log_widget)
        splitter.setSizes([420, 280])

    # ── file/folder pickers ───────────────────────────────────────────────────

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select LAZ/LAS files", "",
            "Point cloud files (*.laz *.las);;All files (*)")
        existing = [self._file_list.item(i).text()
                    for i in range(self._file_list.count())]
        for p in paths:
            if p not in existing:
                self._file_list.addItem(p)

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            existing = [self._file_list.item(i).text()
                        for i in range(self._file_list.count())]
            if d not in existing:
                self._file_list.addItem(d)

    def _remove_selected(self):
        for item in self._file_list.selectedItems():
            self._file_list.takeItem(self._file_list.row(item))

    # ── run / cancel ──────────────────────────────────────────────────────────

    def _run(self):
        paths = self._file_list.paths()
        if not paths:
            self._log("No files or folders added.")
            return

        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._progress.setValue(0)
        self._progress.setMaximum(1)

        self._log(
            f"Starting patch — {len(paths)} source path(s)\n"
            f"  veg: low ≥{self._low_min.value():.2f} m  "
            f"med ≥{self._low_max.value():.2f} m  "
            f"high ≥{self._med_max.value():.2f} m\n"
            f"  noise ≤{self._noise_thr.value():.2f} m  "
            f"ground ±{self._gnd_thr.value():.2f} m  "
            f"cell {self._cell_size.value():.2f} m\n"
            f"  output: {'overwrite in-place' if self._overwrite_chk.isChecked() else 'write alongside'}\n"
            f"  workers: {self._workers_spin.value()}   "
            f"chunk: {self._chunk_spin.value()}M pts\n"
        )

        self._worker = PatchWorker(
            files        = paths,
            dst_mode     = "overwrite" if self._overwrite_chk.isChecked() else "alongside",
            pattern      = "*_classified.laz",
            veg_low_min  = self._low_min.value(),
            veg_low_max  = self._low_max.value(),
            veg_med_max  = self._med_max.value(),
            noise_below  = self._noise_thr.value(),
            ground_thr   = self._gnd_thr.value(),
            cell_size    = self._cell_size.value(),
            chunk_size   = self._chunk_spin.value() * 1_000_000,
            workers      = self._workers_spin.value(),
        )
        self._worker.log.connect(self._log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _cancel(self):
        if self._worker:
            self._worker.cancel()
        self._cancel_btn.setEnabled(False)

    def _on_progress(self, done, total):
        self._progress.setMaximum(total)
        self._progress.setValue(done)
        self._progress.setFormat(f"{done} / {total} files")

    def _on_finished(self, ok, err):
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        color = "#2a7a2a" if err == 0 else "#7a4a00"
        self._progress.setStyleSheet(f"QProgressBar::chunk{{background:{color};}}")

    def _log(self, msg: str):
        self._log_box.appendPlainText(msg)
        self._log_box.verticalScrollBar().setValue(
            self._log_box.verticalScrollBar().maximum())


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor(45, 45, 45))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor(212, 212, 212))
    pal.setColor(QPalette.ColorRole.Base,            QColor(30, 30, 30))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(40, 40, 40))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(50, 50, 50))
    pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(212, 212, 212))
    pal.setColor(QPalette.ColorRole.Text,            QColor(212, 212, 212))
    pal.setColor(QPalette.ColorRole.Button,          QColor(55, 55, 55))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor(212, 212, 212))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor(42, 122, 42))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
