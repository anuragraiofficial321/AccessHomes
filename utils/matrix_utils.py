#!/usr/bin/env python3
import numpy as np

def parse_homogeneous_matrix(raw):
    """
    Parse common matrix formats (flat len16, 3x4 flat len12, 4x4 nested).
    Return (pos3, R3x3) where R maps camera coords -> world coords.
    """
    try:
        arr = np.array(raw, dtype=float)
    except Exception:
        return None, None

    if arr.size == 16:
        try:
            M = arr.reshape(4, 4)
        except Exception:
            return None, None
    elif arr.size == 12:
        M = np.vstack((arr.reshape(3, 4), np.array([0.0, 0.0, 0.0, 1.0])))
    elif arr.shape == (4, 4):
        M = arr
    else:
        return None, None

    def orth_error(R):
        return float(np.linalg.norm(R @ R.T - np.eye(3)))

    R_A = M[:3, :3].astype(float); t_A = M[:3, 3].astype(float); err_A = orth_error(R_A)
    R_B = R_A.T; t_B = M[3, :3].astype(float); err_B = orth_error(R_B)

    magA = float(np.linalg.norm(t_A)); magB = float(np.linalg.norm(t_B))
    if err_A + 1e-6 < err_B or (abs(err_A - err_B) < 1e-6 and magA >= magB):
        R = R_A; t = t_A
    else:
        R = R_B; t = t_B

    try:
        U, S, Vt = np.linalg.svd(R)
        R_clean = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt
        R = R_clean
    except Exception:
        pass

    return t.astype(float), R.astype(float)
