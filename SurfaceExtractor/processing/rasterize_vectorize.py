"""
Convert per-point material labels → 2D grid → polygons.

Pipeline:
  1. Rasterise points onto a NxM grid; each cell gets the DOMINANT class
     (mode of the class labels of all points falling in the cell).
  2. For each class, generate a binary mask.
  3. Apply morphological opening (remove speckle) + closing (fill small holes).
  4. Find contours (OpenCV findContours) → convert to Shapely polygons.
  5. Simplify with Douglas-Peucker (Shapely's simplify).
  6. Drop polygons with area below min_area.
"""

import numpy as np


def rasterise_points(
    xy: np.ndarray,
    materials: np.ndarray,
    cell_size: float = 0.25,
    n_classes: int = 6,
) -> tuple[np.ndarray, float, float]:
    """
    Bin points into a 2D grid, take dominant class per cell.

    Returns:
        class_raster: 2D int8 array [rows, cols], values 0..n_classes-1
                      Empty cells = 0 (UNKNOWN)
        x_origin, y_origin: world coords of raster[0, 0] cell CENTRE
    """
    if len(xy) == 0:
        return np.zeros((1, 1), dtype=np.int8), 0.0, 0.0

    x_min = xy[:, 0].min()
    y_min = xy[:, 1].min()

    col = ((xy[:, 0] - x_min) / cell_size).astype(np.int32)
    row = ((xy[:, 1] - y_min) / cell_size).astype(np.int32)
    n_cols = int(col.max()) + 1
    n_rows = int(row.max()) + 1

    # Vote grid: for each cell, count occurrences of each class
    # Shape (rows, cols, n_classes) — memory-safe when scene is <= 400x400 tile at 0.25m
    # For huge scenes this may be reworked with hash-based aggregation
    votes = np.zeros((n_rows, n_cols, n_classes), dtype=np.int32)
    np.add.at(votes, (row, col, materials.astype(np.int32)), 1)

    class_raster = votes.argmax(axis=2).astype(np.int8)
    # Cells with no points at all → keep UNKNOWN (already 0)
    cell_has_pts = votes.sum(axis=2) > 0
    class_raster[~cell_has_pts] = 0

    x_origin = x_min + cell_size / 2.0
    y_origin = y_min + cell_size / 2.0

    return class_raster, x_origin, y_origin


def clean_class_mask(mask: np.ndarray, open_iter: int = 1, close_iter: int = 2) -> np.ndarray:
    """
    Morphological opening (remove speckle) + closing (fill holes) on a bool mask.
    """
    import cv2
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    if open_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if close_iter > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    return m.astype(bool)


def _raster_to_world(
    contour: np.ndarray,
    x_origin: float, y_origin: float, cell_size: float,
    n_rows: int,
) -> list[tuple[float, float]]:
    """
    Convert an OpenCV contour (pixel coords, (col, row)) to world XY.
    OpenCV pixel origin is TOP-LEFT with y going DOWN — flip y to match world.
    """
    world = []
    for pt in contour:
        c, r = pt[0]
        x = x_origin + c * cell_size
        # Flip row so that raster[0,0] (top-left in cv2) corresponds to
        # y_origin (which is the min y in world space)
        y = y_origin + (n_rows - 1 - r) * cell_size
        world.append((x, y))
    return world


def extract_polygons(
    class_raster: np.ndarray,
    x_origin: float, y_origin: float,
    cell_size: float,
    class_codes: list[int],
    min_area_m2: float = 1.0,
    simplify_tol_m: float = 0.20,
    open_iter: int = 1,
    close_iter: int = 2,
) -> dict[int, list]:
    """
    For each class in class_codes, extract simplified polygons.

    Returns dict {class_code: [shapely.geometry.Polygon, ...]}.
    """
    import cv2
    from shapely.geometry import Polygon
    from shapely.validation import make_valid

    n_rows = class_raster.shape[0]
    # OpenCV expects images with y going DOWN; our raster has row=0 at Y_MIN.
    # Flip vertically ONCE for polygon extraction then adjust in _raster_to_world.
    result: dict[int, list] = {code: [] for code in class_codes}

    for code in class_codes:
        mask = (class_raster == code)
        if not mask.any():
            continue
        mask_clean = clean_class_mask(mask, open_iter, close_iter)
        if not mask_clean.any():
            continue

        # Flip so OpenCV sees rows going TOP → BOTTOM = high Y → low Y
        flipped = np.flipud(mask_clean.astype(np.uint8))

        # RETR_CCOMP so we get outer boundaries + holes as pairs; keep only outer here.
        contours, hierarchy = cv2.findContours(
            flipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            if len(contour) < 3:
                continue
            # OpenCV contour is (n,1,2) with columns col,row (both in flipped space)
            world_pts = []
            for pt in contour:
                c, r_flip = pt[0]
                r_orig = n_rows - 1 - r_flip
                x = x_origin + c * cell_size
                y = y_origin + r_orig * cell_size
                world_pts.append((x, y))
            if len(world_pts) < 3:
                continue
            try:
                poly = Polygon(world_pts)
                poly = make_valid(poly)
                if poly.is_empty:
                    continue
                # make_valid may return MultiPolygon / GeometryCollection
                if poly.geom_type == "Polygon":
                    polys = [poly]
                elif poly.geom_type == "MultiPolygon":
                    polys = list(poly.geoms)
                else:
                    polys = [g for g in poly.geoms if g.geom_type == "Polygon"]
                for p in polys:
                    if p.area < min_area_m2:
                        continue
                    p_simple = p.simplify(simplify_tol_m, preserve_topology=True)
                    if p_simple.is_empty or p_simple.area < min_area_m2:
                        continue
                    result[code].append(p_simple)
            except Exception:
                continue
    return result
