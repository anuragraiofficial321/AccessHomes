# utils/reprojection_utils.py
import numpy as np
import math

def _rescale_intrinsics_for_frame(intr, frame_w, frame_h):
    """
    Given intr = (fx,fy,cx,cy,K_w,K_h), rescale fx,fy,cx,cy if K_w/K_h provided and differs
    """
    if intr is None:
        return None
    fx, fy, cx, cy, K_w, K_h = intr
    try:
        if K_w is not None and K_h is not None:
            K_w_i = int(K_w); K_h_i = int(K_h)
        else:
            K_w_i = None; K_h_i = None
    except Exception:
        K_w_i = None; K_h_i = None

    if K_w_i is not None and K_h_i is not None and (K_w_i != frame_w or K_h_i != frame_h):
        sx = float(frame_w) / float(K_w_i); sy = float(frame_h) / float(K_h_i)
        return (fx * sx, fy * sy, cx * sx, cy * sy, K_w, K_h)
    return (fx, fy, cx, cy, K_w, K_h)

def reproject_point_world_to_image(p_obj_world, cam_pos3d, cam_rot, intrinsics_tuple, frame_size=None, z_sign=1.0):
    """
    Project a 3D world point into image u_est,v_est using given camera pose & intrinsics.
    Returns (u_est, v_est).
    """
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
                K_w_i = int(K_w); K_h_i = int(K_h)
            except Exception:
                K_w_i = None; K_h_i = None
            if K_w_i is not None and K_h_i is not None and (K_w_i != frame_w or K_h_i != frame_h):
                sx = float(frame_w) / float(K_w_i); sy = float(frame_h) / float(K_h_i)
                fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
    if p_cam[2] == 0:
        return None, None
    u_est = (p_cam[0] / p_cam[2]) * fx + cx
    v_est = (p_cam[1] / p_cam[2]) * fy + cy
    return float(u_est), float(v_est)

def reprojection_error_for_depth(depth_m, cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, chosen_proj, s_map, R_map, t_map, frame_size, z_sign=1.0):
    """
    Given a depth (meters) along the ray from camera through image point, compute reprojection distance:
    - compute p_world from depth and direction -> project into image -> distance from original u,v
    - returns scalar error (Euclidean pixels)
    """
    # compute ray direction in camera coords
    x1, y1, x2, y2 = bbox_xyxy
    u = (x1 + x2) * 0.5; v = (y1 + y2) * 0.5
    # intrinsics scaling handled in reproject_point_world_to_image
    if intrinsics_tuple is None:
        fw, fh = (frame_size if frame_size is not None else (640, 480))
        fx = fy = 0.8 * max(fw, fh); cx = fw / 2.0; cy = fh / 2.0
    else:
        fx, fy, cx, cy, K_w, K_h = intrinsics_tuple
        if frame_size is not None and K_w is not None and K_h is not None:
            frame_w, frame_h = frame_size
            try:
                K_w_i = int(K_w); K_h_i = int(K_h)
            except Exception:
                K_w_i = None; K_h_i = None
            if K_w_i is not None and K_h_i is not None and (K_w_i != frame_w or K_h_i != frame_h):
                sx = float(frame_w) / float(K_w_i); sy = float(frame_h) / float(K_h_i)
                fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy

    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, float(z_sign)], dtype=float)
    n = np.linalg.norm(d_cam)
    if n == 0:
        return 1e9
    d_cam_n = d_cam / n
    R = np.array(cam_rot, dtype=float)
    d_world = R @ d_cam_n
    d_world_norm = np.linalg.norm(d_world)
    if d_world_norm == 0:
        return 1e9
    d_world_dir = d_world / d_world_norm
    p_obj_world = np.array(cam_pos3d, dtype=float) + float(depth_m) * d_world_dir

    # project back into image
    u_est, v_est = reproject_point_world_to_image(p_obj_world, cam_pos3d, cam_rot, intrinsics_tuple, frame_size=frame_size, z_sign=z_sign)
    if u_est is None:
        return 1e9
    err = math.hypot(u_est - u, v_est - v)
    return err, p_obj_world

def choose_best_orientation_by_reproj(cam_pos3d, cam_rot_raw, intrinsics_tuple, bbox_xyxy, depth_guess, frame_size):
    """
    Try R vs R.T and z_sign +/-1; return (best_R, best_z_sign, best_err, best_p_world)
    depth_guess is a coarse distance (may be None)
    """
    best = (None, 1.0, 1e9, None)
    if cam_rot_raw is None:
        return (None, 1.0, 1e9, None)
    for use_transpose in (False, True):
        Rtry = cam_rot_raw.T if use_transpose else cam_rot_raw
        for zsign in (1.0, -1.0):
            # if we have a depth_guess, compute an error; if not, use a default depth (1.5m)
            d_guess = depth_guess if (depth_guess is not None) else 1.5
            err, p_world = reprojection_error_for_depth(d_guess, cam_pos3d, Rtry, intrinsics_tuple, bbox_xyxy, None, None, None, frame_size, z_sign=zsign)
            if err < best[2]:
                best = (Rtry, zsign, err, p_world)
    return best  # (R, z_sign, err, p_world)

def depth_refine_binary_search(cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, frame_size, z_sign=1.0, depth_min=0.2, depth_max=10.0, tol=1e-2, max_iter=30):
    """
    1-D depth minimization: binary search on depth to reduce reprojection error.
    Returns (best_depth, best_err, p_world)
    """
    lo = float(depth_min); hi = float(depth_max)
    best_depth = None; best_err = 1e9; best_p = None
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        err_mid, p_mid = reprojection_error_for_depth(mid, cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, None, None, None, frame_size, z_sign=z_sign)
        # evaluate neighbors for gradient direction
        err_lo, _ = reprojection_error_for_depth(max(depth_min, mid * 0.9), cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, None, None, None, frame_size, z_sign=z_sign)
        err_hi, _ = reprojection_error_for_depth(min(depth_max, mid * 1.1), cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, None, None, None, frame_size, z_sign=z_sign)

        if err_mid < best_err:
            best_err = err_mid; best_depth = mid; best_p = p_mid

        # adjust search direction by comparing err_lo and err_hi
        if err_lo < err_hi:
            hi = mid
        else:
            lo = mid

        if abs(hi - lo) < tol:
            break

    return best_depth, best_err, best_p
