import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import laspy
import numpy as np
from sklearn.neighbors import KDTree

DEFAULT_CHUNK_SIZE = 1_000_000
MAX_SLOPE_EDGE_POINTS = 1_000_000


def _log(message):
    print(message, flush=True)


def _has_nearby_kept(cell_map, cell, point, spacing):
    """Check kept points in this cell and adjacent cells."""
    spacing2 = spacing * spacing
    cx, cy, cz = cell

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nearby = cell_map.get((cx + dx, cy + dy, cz + dz))
                if nearby is None:
                    continue
                for kept_point in nearby:
                    diff = kept_point - point
                    if float(np.dot(diff, diff)) < spacing2:
                        return True
    return False

def refine_grid_slope_edges(input_file, output_file, spacing=0.025, k=16,
                             linearity_thresh=0.6,
                             normal_dev_thresh=15.0,   # degrees
                             grad_thresh=0.3,          # Z-change per unit XY
                             log_func=_log):
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if input_path == output_path:
        raise ValueError("Output file must be different from the input file.")

    if not output_path.suffix:
        output_path = output_path.with_suffix(".las")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_func(f"Reading input       : {input_path}")
    las = laspy.read(input_file)
    pts = np.column_stack((las.x, las.y, las.z))
    N = len(pts)
    log_func(f"Input points       : {N:,}")

    log_func("Building neighbour index ...")
    tree = KDTree(pts)
    log_func("Finding local neighbours ...")
    _, idx = tree.query(pts, k=k)
    neighbors = pts[idx]  # (N, k, 3)

    log_func("Calculating edge signals ...")
    centroid = neighbors.mean(axis=1, keepdims=True)
    centered = neighbors - centroid
    cov = np.einsum('ijk,ijl->ikl', centered, centered) / k
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order: λ₀ ≤ λ₁ ≤ λ₂
    lam0, lam1, lam2 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]

    # --- Signal 1: Linearity — high where points form a line (edges, backs) ---
    linearity = (lam2 - lam1) / (lam2 + 1e-10)

    # --- Signal 2: Normal deviation — slope-break detection ---
    # Normal = eigenvector for smallest eigenvalue
    normals = eigvecs[:, :, 0]  # (N, 3)
    # Ensure normals point upward for consistency
    normals *= np.sign(normals[:, 2:3] + 1e-10)
    neighbor_normals = normals[idx]         # (N, k, 3)
    dot = np.einsum('ij,ikj->ik', normals, neighbor_normals).clip(-1, 1)
    mean_angle_deg = np.degrees(np.arccos(np.abs(dot)).mean(axis=1))

    # --- Signal 3: Height gradient — steep local faces ---
    dz = np.abs(neighbors[:, :, 2] - pts[:, 2:3])   # (N, k)
    dxy = np.linalg.norm(neighbors[:, :, :2] - pts[:, :2].reshape(N, 1, 2), axis=2) + 1e-10
    grad = (dz / dxy).mean(axis=1)

    # --- Combine: a point is an edge if ANY signal fires ---
    is_edge = (
        (linearity    > linearity_thresh)  |
        (mean_angle_deg > normal_dev_thresh) |
        (grad         > grad_thresh)
    )

    # --- Starling local-rule subsampling ---
    edge_idx    = np.where(is_edge)[0]
    non_edge_idx = np.where(~is_edge)[0]
    rng   = np.random.default_rng(seed=42)
    order = np.concatenate([edge_idx, rng.permutation(non_edge_idx)])

    kept     = np.zeros(N, dtype=bool)
    cell_map = {}
    origin = pts.min(axis=0)
    progress_step = max(1, min(100_000, len(order) // 20 or 1))

    log_func("Subsampling points ...")
    for n_done, i in enumerate(order, start=1):
        point = pts[i]
        cell = tuple(np.floor((point - origin) / spacing).astype(np.int64))

        if not is_edge[i] and _has_nearby_kept(cell_map, cell, point, spacing):
            if n_done % progress_step == 0:
                log_func(f"  Subsample progress: {n_done:,}/{N:,} checked, {kept.sum():,} kept")
            continue

        kept[i] = True
        cell_map.setdefault(cell, []).append(point)

        if n_done % progress_step == 0:
            log_func(f"  Subsample progress: {n_done:,}/{N:,} checked, {kept.sum():,} kept")

    final_indices = np.where(kept)[0]

    log_func(f"Edge points kept   : {is_edge.sum():,}")
    log_func(f"  by linearity     : {(linearity > linearity_thresh).sum():,}")
    log_func(f"  by slope break   : {(mean_angle_deg > normal_dev_thresh).sum():,}")
    log_func(f"  by height grad   : {(grad > grad_thresh).sum():,}")
    log_func(f"Output points      : {len(final_indices):,}")

    log_func(f"Writing output     : {output_path}")
    new_las = laspy.LasData(header=las.header)
    new_las.points = las.points[final_indices]
    new_las.write(str(output_path))

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise OSError(f"Output was not created: {output_path}")

    log_func(f"Saved output       : {output_path}")
    return output_path


def refine_grid_streaming(input_file, output_file, spacing=0.025,
                          chunk_size=DEFAULT_CHUNK_SIZE,
                          log_func=_log):
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if input_path == output_path:
        raise ValueError("Output file must be different from the input file.")

    if not output_path.suffix:
        output_path = output_path.with_suffix(".las")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_cells = set()
    total_read = 0
    total_written = 0

    log_func(f"Reading input       : {input_path}")
    log_func("Mode                : streaming XY grid")
    log_func(f"Chunk size          : {chunk_size:,}")

    with laspy.open(str(input_path)) as reader:
        header = reader.header.copy()
        point_count = reader.header.point_count
        min_x, min_y = reader.header.mins[:2]
        max_x, max_y = reader.header.maxs[:2]
        approx_x_cells = int(np.floor((max_x - min_x) / spacing)) + 1
        approx_y_cells = int(np.floor((max_y - min_y) / spacing)) + 1
        log_func(f"Input points       : {point_count:,}")
        log_func(f"Approx XY cells    : {approx_x_cells:,} x {approx_y_cells:,}")

        with laspy.open(str(output_path), mode="w", header=header) as writer:
            for chunk in reader.chunk_iterator(chunk_size):
                total_read += len(chunk)
                cells = np.floor(
                    np.column_stack((chunk.x, chunk.y)) / spacing
                ).astype(np.int64)

                keep_mask = np.zeros(len(chunk), dtype=bool)
                for row_index, cell in enumerate(map(tuple, cells)):
                    if cell in kept_cells:
                        continue
                    kept_cells.add(cell)
                    keep_mask[row_index] = True

                if keep_mask.any():
                    kept_chunk = chunk[keep_mask]
                    writer.write_points(kept_chunk)
                    total_written += len(kept_chunk)

                log_func(
                    f"  Stream progress: {total_read:,}/{point_count:,} read, "
                    f"{total_written:,} kept"
                )

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise OSError(f"Output was not created: {output_path}")

    log_func(f"Output points      : {total_written:,}")
    log_func(f"Saved output       : {output_path}")
    return output_path

def _find_las_files(input_folder):
    folder = Path(input_folder)
    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".las", ".laz"}
    ]
    return sorted(files, key=lambda path: path.name.lower())


def _default_output_path(input_file, output_folder):
    return Path(output_folder) / f"{input_file.stem}_grid.las"


class BatchGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LAS/LAZ Grid Batch Processor")
        self.root.minsize(780, 520)
        self.is_running = False

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.spacing = tk.StringVar(value="0.025")
        self.mode = tk.StringVar(value="streaming")
        self.chunk_size = tk.StringVar(value=str(DEFAULT_CHUNK_SIZE))

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main.columnconfigure(1, weight=1)
        main.rowconfigure(6, weight=1)

        ttk.Label(main, text="Input folder").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(main, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(main, text="Browse", command=self._browse_input).grid(row=0, column=2, pady=(0, 8))

        ttk.Label(main, text="Output folder").grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(main, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(main, text="Browse", command=self._browse_output).grid(row=1, column=2, pady=(0, 8))

        ttk.Label(main, text="Grid spacing (metres)").grid(row=2, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(main, textvariable=self.spacing, width=16).grid(row=2, column=1, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(main, text="Mode").grid(row=3, column=0, sticky="w", pady=(0, 8))
        mode_frame = ttk.Frame(main)
        mode_frame.grid(row=3, column=1, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        ttk.Radiobutton(
            mode_frame,
            text="Streaming grid for large files",
            variable=self.mode,
            value="streaming",
        ).grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Radiobutton(
            mode_frame,
            text="Slope-edge refinement for smaller files",
            variable=self.mode,
            value="slope",
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(main, text="Chunk size").grid(row=4, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(main, textvariable=self.chunk_size, width=16).grid(row=4, column=1, sticky="w", padx=8, pady=(0, 8))

        actions = ttk.Frame(main)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 10))
        actions.columnconfigure(1, weight=1)
        self.run_button = ttk.Button(actions, text="Run Folder", command=self._start_batch)
        self.run_button.grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(actions, mode="determinate")
        self.progress.grid(row=0, column=1, sticky="ew", padx=10)
        self.status = ttk.Label(actions, text="Ready")
        self.status.grid(row=0, column=2, sticky="e")

        log_frame = ttk.Frame(main)
        log_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=18, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select folder containing LAS/LAZ files")
        if folder:
            self.input_dir.set(folder)
            if not self.output_dir.get():
                self.output_dir.set(folder)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir.set(folder)

    def _append_log(self, message):
        print(message, flush=True)
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _thread_log(self, message):
        self.root.after(0, self._append_log, message)

    def _set_running(self, running):
        self.is_running = running
        self.run_button.configure(state="disabled" if running else "normal")

    def _start_batch(self):
        if self.is_running:
            return

        input_folder = self.input_dir.get().strip()
        output_folder = self.output_dir.get().strip()

        if not input_folder:
            messagebox.showerror("Missing input folder", "Select a folder containing LAS/LAZ files.")
            return
        if not output_folder:
            messagebox.showerror("Missing output folder", "Select a folder for output LAS files.")
            return

        try:
            spacing = float(self.spacing.get())
        except ValueError:
            messagebox.showerror("Invalid spacing", "Grid spacing must be a number.")
            return

        if spacing <= 0:
            messagebox.showerror("Invalid spacing", "Grid spacing must be greater than zero.")
            return

        try:
            chunk_size = int(self.chunk_size.get())
        except ValueError:
            messagebox.showerror("Invalid chunk size", "Chunk size must be a whole number.")
            return

        if chunk_size < 1:
            messagebox.showerror("Invalid chunk size", "Chunk size must be greater than zero.")
            return

        input_path = Path(input_folder)
        output_path = Path(output_folder)

        if not input_path.exists() or not input_path.is_dir():
            messagebox.showerror("Invalid input folder", "The selected input folder does not exist.")
            return

        files = _find_las_files(input_path)
        if input_path.resolve() == output_path.resolve():
            files = [path for path in files if not path.stem.lower().endswith("_grid")]

        if not files:
            messagebox.showinfo("No files found", "No LAS/LAZ files were found in the selected folder.")
            return

        self.progress.configure(value=0, maximum=len(files))
        self.status.configure(text=f"0/{len(files)}")
        self._set_running(True)
        self._append_log("=" * 72)
        self._append_log(f"Input folder  : {input_path}")
        self._append_log(f"Output folder : {output_path}")
        self._append_log(f"Files found   : {len(files)}")
        self._append_log(f"Spacing       : {spacing}")
        self._append_log(f"Mode          : {self.mode.get()}")

        worker = threading.Thread(
            target=self._run_batch,
            args=(files, output_path, spacing, self.mode.get(), chunk_size),
            daemon=True,
        )
        worker.start()

    def _run_batch(self, files, output_folder, spacing, mode, chunk_size):
        output_folder.mkdir(parents=True, exist_ok=True)
        successes = []
        failures = []

        for index, input_file in enumerate(files, start=1):
            output_file = _default_output_path(input_file, output_folder)
            self._thread_log("")
            self._thread_log(f"[{index}/{len(files)}] Processing {input_file.name}")

            try:
                if mode == "slope":
                    with laspy.open(str(input_file)) as reader:
                        point_count = reader.header.point_count
                    if point_count > MAX_SLOPE_EDGE_POINTS:
                        raise MemoryError(
                            "Slope-edge mode is limited to "
                            f"{MAX_SLOPE_EDGE_POINTS:,} points in this GUI. "
                            "Use Streaming grid mode for larger files."
                        )
                    saved_path = refine_grid_slope_edges(
                        input_file,
                        output_file,
                        spacing=spacing,
                        log_func=self._thread_log,
                    )
                else:
                    saved_path = refine_grid_streaming(
                        input_file,
                        output_file,
                        spacing=spacing,
                        chunk_size=chunk_size,
                        log_func=self._thread_log,
                    )
            except Exception as exc:
                failures.append((input_file, exc))
                self._thread_log(f"ERROR: {input_file.name}: {exc}")
            else:
                successes.append(saved_path)

            self.root.after(0, self.progress.configure, {"value": index})
            self.root.after(0, self.status.configure, {"text": f"{index}/{len(files)}"})

        self.root.after(0, self._batch_finished, successes, failures)

    def _batch_finished(self, successes, failures):
        self._set_running(False)
        self._append_log("")
        self._append_log("=" * 72)
        self._append_log(f"Complete. Successful: {len(successes)} | Failed: {len(failures)}")
        for input_file, exc in failures:
            self._append_log(f"Failed: {input_file.name} - {exc}")

        if failures:
            messagebox.showwarning(
                "Batch complete with errors",
                f"Processed {len(successes)} file(s).\nFailed {len(failures)} file(s). See the log for details.",
            )
        else:
            messagebox.showinfo("Batch complete", f"Processed {len(successes)} file(s).")


def run_gui():
    root = tk.Tk()
    BatchGridApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
