#!/usr/bin/env python3
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

def umeyama_2d(src, dst, with_scaling=True):
    src = np.array(src, dtype=float); dst = np.array(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 2
    N = src.shape[0]
    mu_src = src.mean(axis=0); mu_dst = dst.mean(axis=0)
    src_c = src - mu_src; dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    if with_scaling:
        var_src = (src_c**2).sum() / N
        s = 1.0 / var_src * np.trace(np.diag(D) @ S)
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t

def apply_similarity_to_points(points2d, s, R, t):
    pts = np.array(points2d, dtype=float)
    return (s * (R @ pts.T)).T + t

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

def interp_missing(mapped):
    m = np.array(mapped, dtype=float).copy()
    n = m.shape[0]
    if n == 0: return m
    for dim in (0, 1):
        arr = m[:, dim]; isn = np.isnan(arr)
        if isn.any():
            good_idx = np.where(~isn)[0]
            if good_idx.size > 0:
                interp_all = np.interp(np.arange(n), good_idx, arr[good_idx])
                m[:, dim] = interp_all
            else:
                m[:, dim] = 0.0
    return m
