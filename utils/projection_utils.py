"""
Projection helpers (3D -> 2D) and mapping heuristics.
"""

import numpy as np
import math
from matplotlib.path import Path as MplPath

def project_3d_to_2d(positions3d, projection):
    positions3d = np.array(positions3d, dtype=float)
    if positions3d.ndim == 1:
        positions3d = positions3d.reshape(1, 3)
    if projection == "x,-z":
        return np.column_stack([positions3d[:, 0], -positions3d[:, 2]])
    if projection == "x,z":
        return np.column_stack([positions3d[:, 0], positions3d[:, 2]])
    if projection == "-x,-z":
        return np.column_stack([-positions3d[:, 0], -positions3d[:, 2]])
    if projection == "-x,z":
        return np.column_stack([-positions3d[:, 0], positions3d[:, 2]])
    if projection == "y,-z":
        return np.column_stack([positions3d[:, 1], -positions3d[:, 2]])
    raise ValueError("Unknown projection: " + projection)

def count_points_inside_polygons(points2d, spaces):
    pts = np.asarray(points2d, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    cnt = 0
    for p in pts:
        if np.isnan(p[0]) or np.isnan(p[1]): continue
        for sp in spaces:
            try:
                poly = sp.get("poly")
                if poly is None: continue
                if MplPath(np.asarray(poly, dtype=float)).contains_point((p[0], p[1])):
                    cnt += 1
                    break
            except Exception:
                continue
    return cnt

def auto_map_and_choose(proj2, spaces, floor_min, floor_max, compass=None):
    proj2 = np.asarray(proj2, dtype=float)
    if proj2.ndim == 1:
        proj2 = proj2.reshape(1, 2)
    floor_min = np.asarray(floor_min, dtype=float); floor_max = np.asarray(floor_max, dtype=float)
    floor_center = (floor_min + floor_max) / 2.0
    floor_size = floor_max - floor_min
    if floor_size[0] == 0 or floor_size[1] == 0:
        raise RuntimeError("Invalid floor bounds: zero size")
    valid_mask = ~np.isnan(proj2[:, 0])
    if valid_mask.sum() == 0:
        raise RuntimeError("No valid projected points to map")
    vpts = proj2[valid_mask]
    cam_min = vpts.min(axis=0); cam_max = vpts.max(axis=0)
    cam_span = cam_max - cam_min
    cam_span[cam_span == 0] = 1.0
    scale_x = (floor_size[0] * 0.9) / cam_span[0]; scale_y = (floor_size[1] * 0.9) / cam_span[1]
    scale = (scale_x + scale_y) / 2.0
    cam_center = (cam_min + cam_max) / 2.0
    mapped = (proj2 - cam_center) * scale + floor_center

    rotations = [0.0]
    if compass is not None:
        try:
            rotations.extend([float(compass), -float(compass)])
        except Exception:
            pass

    best_score = -1; best_rot = 0.0; best_mapped = None
    for rot in rotations:
        theta = math.radians(rot)
        Rr = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        rel = mapped - floor_center
        mapped_rot = (rel @ Rr.T) + floor_center
        sc = count_points_inside_polygons(mapped_rot, spaces)
        if sc > best_score or (sc == best_score and abs(rot) < abs(best_rot)):
            best_score = sc; best_rot = rot; best_mapped = mapped_rot
    if best_mapped is None:
        best_mapped = mapped; best_rot = 0.0; best_score = 0
    return np.asarray(best_mapped, dtype=float), float(scale), float(best_rot), int(best_score)
