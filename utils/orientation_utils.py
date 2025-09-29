"""
Utilities for handling camera orientation and reprojection.
"""

import numpy as np
import math

def choose_best_rotation_and_sign(cam_pos, cam_rot, neighbor_pos):
    travel = neighbor_pos - cam_pos
    n = np.linalg.norm(travel)
    if n < 1e-6:
        return cam_rot, 1.0, 0.0
    travel_n = travel / n
    best_score = -2.0
    best_R = cam_rot
    best_sign = 1.0
    for use_transpose in (False, True):
        Rtry = cam_rot.T if use_transpose else cam_rot
        for sign_try in (1.0, -1.0):
            fw = Rtry @ np.array([0.0, 0.0, sign_try])
            fw_n = fw / (np.linalg.norm(fw) + 1e-12)
            score = float(np.dot(fw_n, travel_n))
            if score > best_score:
                best_score = score
                best_R = Rtry.copy()
                best_sign = sign_try
    return best_R, best_sign, best_score

def compute_yaw_from_direction(d_world_dir):
    dx = float(d_world_dir[0]); dz = float(d_world_dir[2])
    ang = math.degrees(math.atan2(dx, -dz))
    ang = (ang + 360.0) % 360.0
    return ang

def reproject_error(p_obj_world, cam_pos3d, cam_rot, intrinsics_tuple, u, v, frame_size=None):
    try:
        R = np.array(cam_rot, dtype=float)
        p_cam = R.T @ (np.array(p_obj_world, dtype=float) - np.array(cam_pos3d, dtype=float))
        if intrinsics_tuple is None:
            fw, fh = (frame_size if frame_size is not None else (640, 480))
            fx = fy = 0.8 * max(fw, fh); cx = fw / 2.0; cy = fh / 2.0
        else:
            fx, fy, cx, cy, K_w, K_h = intrinsics_tuple
            if frame_size is not None and K_w is not None and K_h is not None:
                frame_w, frame_h = frame_size
                try:
                    K_w = int(K_w); K_h = int(K_h)
                except Exception:
                    K_w = None; K_h = None
                if K_w is not None and K_h is not None and (K_w != frame_w or K_h != frame_h):
                    sx = float(frame_w) / float(K_w); sy = float(frame_h) / float(K_h)
                    fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
        if p_cam[2] == 0:
            return 1e9
        u_est = (p_cam[0] / p_cam[2]) * fx + cx
        v_est = (p_cam[1] / p_cam[2]) * fy + cy
        err = math.hypot(u - u_est, v - v_est)
        return err
    except Exception:
        return 1e9