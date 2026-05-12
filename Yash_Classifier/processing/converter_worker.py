import os
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import laspy

class ConverterWorker(QThread):
    log = pyqtSignal(str)
    progress = pyqtSignal(int, int) # done, total
    file_status = pyqtSignal(str, str) # file_path, status
    finished = pyqtSignal()

    def __init__(self, files, out_dir, target_scale):
        super().__init__()
        self.files = files
        self.out_dir = out_dir
        self.target_scale = target_scale
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def process_las_laz(self, input_path, output_path):
        self.log.emit(f"Reading {input_path.name}...")
        las = laspy.read(str(input_path))
        
        if self._cancel: return False

        self.log.emit(f"Applying scale multiplier: {self.target_scale}...")
        
        x = las.x * self.target_scale
        y = las.y * self.target_scale
        z = las.z * self.target_scale

        if self._cancel: return False

        las.header.offsets = [np.min(x), np.min(y), np.min(z)]
        
        las.x = x
        las.y = y
        las.z = z

        if self._cancel: return False

        self.log.emit(f"Writing to {output_path.name}...")
        las.write(str(output_path))
        return True

    def process_e57(self, input_path, output_path):
        import pye57
        self.log.emit(f"Opening E57 file {input_path.name}...")
        e57 = pye57.E57(str(input_path))
        
        header = laspy.LasHeader(point_format=2, version="1.2")
        # Initialize generic offsets, we will fix them per scan if possible, 
        # but laspy writer doesn't allow changing offsets mid-write. 
        # For typical coordinates, 0.0 offset is fine, but to be robust, 
        # let's pre-calculate global minimums by reading a chunk or just use 0.0.
        header.offsets = [0.0, 0.0, 0.0]
        header.scales = [0.001, 0.001, 0.001]
        
        # Track bounding box manually
        x_min, x_max = float('inf'), float('-inf')
        y_min, y_max = float('inf'), float('-inf')
        z_min, z_max = float('inf'), float('-inf')
        
        writer = laspy.open(str(output_path), mode="w", header=header)
        
        try:
            for i in range(e57.scan_count):
                if self._cancel: return False
                self.log.emit(f"Reading scan {i+1} of {e57.scan_count}...")
                
                data = e57.read_scan(i, intensity=True, colors=True, ignore_missing_fields=True)
                
                if "cartesianX" not in data:
                    self.log.emit(f"Warning: Scan {i+1} has no cartesianX. Skipping.")
                    continue
                
                # Apply E57 Pose (Rotation and Translation) to bring local coordinates to global
                scan_header = e57.get_header(i)
                R = scan_header.rotation_matrix
                T = scan_header.translation
                
                pts = np.vstack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
                global_pts = R.dot(pts) + T.reshape(3, 1)
                
                x = global_pts[0] * self.target_scale
                y = global_pts[1] * self.target_scale
                z = global_pts[2] * self.target_scale
                
                # Update bounding box
                x_min, x_max = min(x_min, np.min(x)), max(x_max, np.max(x))
                y_min, y_max = min(y_min, np.min(y)), max(y_max, np.max(y))
                z_min, z_max = min(z_min, np.min(z)), max(z_max, np.max(z))
                
                las_data = laspy.LasData(header)
                # It's important to assign x, y, z as arrays
                las_data.x = x
                las_data.y = y
                las_data.z = z
                
                if "intensity" in data:
                    intens = data["intensity"]
                    if np.max(intens) <= 1.0:
                        intens = intens * 65535
                    las_data.intensity = intens.astype(np.uint16)
                
                if "colorRed" in data:
                    r, g, b = data["colorRed"], data["colorGreen"], data["colorBlue"]
                    if np.max(r) <= 255:
                        r, g, b = r * 256, g * 256, b * 256
                    las_data.red = r.astype(np.uint16)
                    las_data.green = g.astype(np.uint16)
                    las_data.blue = b.astype(np.uint16)
                
                writer.write_points(las_data.points)
        finally:
            writer.close()
            
        return True

    def run(self):
        out_path_dir = Path(self.out_dir)
        out_path_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(self.files)
        self.progress.emit(0, total_files)

        for i, file_path in enumerate(self.files):
            if self._cancel:
                break

            p = Path(file_path)
            self.file_status.emit(file_path, "Processing...")
            self.log.emit(f"\n--- Processing {p.name} ---")
            
            final_out = out_path_dir / f"{p.stem}_converted.laz"
            success = False
            
            try:
                if p.suffix.lower() == '.e57':
                    success = self.process_e57(p, final_out)
                elif p.suffix.lower() in ['.las', '.laz']:
                    success = self.process_las_laz(p, final_out)
                else:
                    self.log.emit(f"Unsupported file format: {p.suffix}")
                    success = False

                if success and final_out.exists() and not self._cancel:
                    self.file_status.emit(file_path, "Success")
                    self.log.emit(f"Successfully created: {final_out}")
                elif not self._cancel:
                    self.file_status.emit(file_path, "Failed")
            except Exception as e:
                self.log.emit(f"Error processing {p.name}: {str(e)}")
                self.file_status.emit(file_path, "Error")

            self.progress.emit(i + 1, total_files)

        if self._cancel:
            self.log.emit("\nBatch process cancelled.")
        else:
            self.log.emit("\nBatch process finished.")
            
        self.finished.emit()
