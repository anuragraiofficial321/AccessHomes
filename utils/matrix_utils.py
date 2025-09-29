"""
Matrix parsing and Umeyama similarity helper functions.
"""

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
