#!/usr/bin/env python3
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
