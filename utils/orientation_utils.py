import numpy as np
import math

def _safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n

def choose_best_rotation_and_sign(cam_pos, cam_rot, neighbor_pos=None, neighbor_positions=None):
    """
    Robustly choose whether to use cam_rot or cam_rot.T and whether the camera's forward
    axis should be +z (sign=1) or -z (sign=-1).

    Parameters
    ----------
    cam_pos : array-like (3,)    camera position in world coords
    cam_rot : array-like (3,3)   camera rotation (mapping camera->world or world->camera; we test both)
    neighbor_pos : array-like (3,) optional
        A single neighbor position (as your code does). If provided, it's used to get travel vector.
    neighbor_positions : iterable of positions (optional)
        If provided, we average travel vectors across neighbors to reduce noise.
        Example: pass [positions3d[i-1], positions3d[i+1]] or more.

    Returns
    -------
    (best_R, best_sign, best_score)
    - best_R: chosen 3x3 rotation to use (camera->world)
    - best_sign: +1.0 or -1.0 for the forward z sign (applied in the projection code)
    - best_score: scalar score (higher is better)
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    cam_rot = np.asarray(cam_rot, dtype=float)

    # Build travel direction (try neighbor_positions first if present, else neighbor_pos)
    trav = None
    if neighbor_positions:
        vecs = []
        for npos in neighbor_positions:
            try:
                v = np.asarray(npos, dtype=float) - cam_pos
                nv = _safe_normalize(v)
                if nv is not None:
                    vecs.append(nv)
            except Exception:
                pass
        if vecs:
            trav = np.mean(np.vstack(vecs), axis=0)
    if trav is None and neighbor_pos is not None:
        v = np.asarray(neighbor_pos, dtype=float) - cam_pos
        trav = v
    if trav is None:
        # no motion info -> return cam_rot, sign=1 with neutral score 0
        return cam_rot, 1.0, 0.0

    travel_n = _safe_normalize(trav)
    if travel_n is None:
        return cam_rot, 1.0, 0.0

    # lateral direction in horizontal (world X-Z) plane perpendicular to travel,
    # used to disambiguate left/right. travel_n is 3D; project onto X-Z plane.
    # If travel is near vertical, lateral_dir may be poorly defined -> skip lateral term.
    travel_xz = np.array([travel_n[0], travel_n[2]])
    lateral_dir = None
    if np.linalg.norm(travel_xz) > 1e-6:
        # rotate travel_xz by +90 degrees to get lateral in X-Z plane
        lx, lz = travel_xz
        lateral_xz = np.array([-lz, lx])
        # build 3D lateral vector (y=0)
        lateral_dir = _safe_normalize(np.array([lateral_xz[0], 0.0, lateral_xz[1]]))

    best_score = -9e9
    best_R = cam_rot
    best_sign = 1.0

    # try both transpose options
    for use_transpose in (False, True):
        Rtry = cam_rot.T if use_transpose else cam_rot
        # define camera basis vectors in world coords
        # camera forward (camera z axis in camera coords is [0,0,1])
        for sign_try in (1.0, -1.0):
            # forward vector candidate (camera pointing direction in world coords)
            fw = Rtry @ np.array([0.0, 0.0, sign_try])
            fw_n = _safe_normalize(fw)
            if fw_n is None:
                continue
            # right vector candidate (camera x axis)
            right = Rtry @ np.array([1.0, 0.0, 0.0])
            right_n = _safe_normalize(right)

            # primary score: alignment of forward with travel direction (prefer values near +1)
            score_fw = float(np.dot(fw_n, travel_n))

            # secondary: lateral consistency. If lateral_dir is available, check sign of right vector
            score_lat = 0.0
            if lateral_dir is not None and right_n is not None:
                # dot(right, lateral_dir) > 0 means camera-right aligns with lateral_dir
                dot_r = float(np.dot(right_n, lateral_dir))
                # We want consistent lateral sign. Use absolute magnitude as small bonus,
                # and penalize if sign inconsistent with forward alignment (rare)
                score_lat = 0.3 * dot_r  # tuned small weight to avoid overpowering forward alignment

            # prefer Rtry with less roll: reward if up vector y component is positive (camera up aligned with world up)
            up = Rtry @ np.array([0.0, 1.0, 0.0])
            up_n = _safe_normalize(up)
            score_up = 0.0
            if up_n is not None:
                # reward uprightness (y component close to +1)
                score_up = 0.15 * float(up_n[1])

            combined = score_fw + score_lat + score_up

            # small tie-breaker: prefer transpose choices that didn't flip axes (prevents unnecessary flips)
            if combined > best_score:
                best_score = combined
                best_R = Rtry.copy()
                best_sign = sign_try

    return best_R, best_sign, best_score


def compute_yaw_from_direction(d_world_dir):
    """
    Compute yaw in degrees from a 3D world direction vector.

    We define yaw so that:
      - it measures rotation around world 'y' (up) axis,
      - 0 degrees points to +Z in world coordinates,
      - positive angles rotate toward +X (i.e. right-handed).

    This matches many conventions used in the rest of this project where projection uses x,-z etc.
    If your mapping convention differs, adjust the returned angle accordingly in the plotting code.
    """
    v = np.asarray(d_world_dir, dtype=float)
    nv = _safe_normalize(np.array([v[0], 0.0, v[2]]))
    if nv is None:
        return None
    # yaw = atan2(x, z) but to get 0->+Z and positive to +X:
    yaw_rad = math.atan2(nv[0], nv[2])
    yaw_deg = (math.degrees(yaw_rad) + 360.0) % 360.0
    return float(yaw_deg)
