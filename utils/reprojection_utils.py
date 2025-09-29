"""
Reprojection helpers: compute object world position given depth & camera.
"""

import numpy as np
import math

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

def compute_object_world_and_mapped(cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, depth_val,
                                   chosen_proj, s_map, R_map, t_map, frame_size=None, z_sign=1.0):
    try:
        if cam_pos3d is None or cam_rot is None or depth_val is None:
            return None, None, None, None
        x1, y1, x2, y2 = bbox_xyxy
        u = (x1 + x2) * 0.5; v = (y1 + y2) * 0.5
        if intrinsics_tuple is None:
            fw, fh = (frame_size if frame_size is not None else (640, 480))
            fx = fy = 0.8 * max(fw, fh); cx = fw / 2.0; cy = fh / 2.0
        else:
            fx, fy, cx, cy, K_img_w, K_img_h = intrinsics_tuple
            if frame_size is not None and K_img_w is not None and K_img_h is not None:
                frame_w, frame_h = frame_size
                try:
                    K_w = int(K_img_w); K_h = int(K_img_h)
                except Exception:
                    K_w = None; K_h = None
                if K_w is not None and K_h is not None and (K_w != frame_w or K_h != frame_h):
                    sx = float(frame_w) / float(K_w); sy = float(frame_h) / float(K_h)
                    fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, float(z_sign)], dtype=float)
        n = np.linalg.norm(d_cam)
        d_cam_n = d_cam / n if n != 0 else d_cam
        R = np.array(cam_rot, dtype=float)
        d_world = R @ d_cam_n
        d_world_norm = np.linalg.norm(d_world)
        if d_world_norm == 0:
            return None, None, None, None
        d_world_dir = d_world / d_world_norm
        p_obj_world = np.array(cam_pos3d, dtype=float) + float(depth_val) * d_world_dir
        from utils.projection_utils import project_3d_to_2d
        obj_proj2 = project_3d_to_2d(p_obj_world.reshape(1, 3), chosen_proj)
        obj_mapped = (s_map * (R_map @ obj_proj2.T)).T + t_map
        obj_x = float(obj_mapped[0, 0]); obj_y = float(obj_mapped[0, 1])
        dx = float(d_world_dir[0]); dz = float(d_world_dir[2])
        ang = math.degrees(math.atan2(dx, -dz))
        ang = (ang + 360.0) % 360.0
        return obj_x, obj_y, ang, p_obj_world
    except Exception as e:
        print("compute_object_world_and_mapped error:", e)
        return None, None, None, None
