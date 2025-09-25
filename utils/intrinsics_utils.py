#!/usr/bin/env python3
import numpy as np

def get_intrinsics_from_meta(meta_entry):
    if not isinstance(meta_entry, dict): return None
    def parse_K(K):
        try:
            arr = np.array(K, dtype=float)
            if arr.shape == (3, 3):
                fx = float(arr[0,0]); fy = float(arr[1,1]); cx = float(arr[0,2]); cy = float(arr[1,2])
                return fx, fy, cx, cy
            if arr.size >= 9:
                arr2 = arr.flatten()[:9].reshape(3,3)
                fx = float(arr2[0,0]); fy = float(arr2[1,1]); cx = float(arr2[0,2]); cy = float(arr2[1,2])
                return fx, fy, cx, cy
        except Exception:
            return None
        return None

    K = meta_entry.get("cameraIntrinsics", None)
    raw = None
    if K is None:
        raw = meta_entry.get("raw") if isinstance(meta_entry.get("raw"), dict) else None
        if raw is not None:
            K = raw.get("cameraIntrinsics", None)

    parsed = None
    if K is not None:
        parsed = parse_K(K)
    if parsed is None:
        for k, v in meta_entry.items():
            if isinstance(v, list) and len(v) in (9, 12, 16):
                maybe = parse_K(v)
                if maybe is not None: parsed = maybe; break
        if parsed is None and raw is not None:
            for k, v in raw.items():
                if isinstance(v, list) and len(v) in (9, 12, 16):
                    maybe = parse_K(v)
                    if maybe is not None: parsed = maybe; break

    if parsed is None: return None
    fx, fy, cx, cy = parsed
    K_img_w = None; K_img_h = None

    def try_int(x):
        try: return int(np.asarray(x).item())
        except Exception: return None

    for key in ("imageWidth", "image_width", "width", "imageWidthPx", "imageWidthPixels"):
        if key in meta_entry:
            K_img_w = try_int(meta_entry[key]); break
    for key in ("imageHeight", "image_height", "height", "imageHeightPx", "imageHeightPixels"):
        if key in meta_entry:
            K_img_h = try_int(meta_entry[key]); break
    if raw is not None and (K_img_w is None or K_img_h is None):
        for key in ("imageWidth", "image_width", "width", "imageWidthPx"):
            if key in raw and K_img_w is None:
                K_img_w = try_int(raw[key]); break
        for key in ("imageHeight", "image_height", "height", "imageHeightPx"):
            if key in raw and K_img_h is None:
                K_img_h = try_int(raw[key]); break

    return (float(fx), float(fy), float(cx), float(cy), K_img_w, K_img_h)

#-------------------------Calculating the real-world size of an object from its bounding box and depth-------------------------#
def compute_real_size_from_bbox(bbox_xyxy, depth_val, intrinsics_tuple, frame_size=None):
    """
    Compute real-world width & height (in meters) of an object bounding box.

    bbox_xyxy: [x1,y1,x2,y2] in pixels
    depth_val: estimated object depth (meters) from ZoeDepth/ARKit
    intrinsics_tuple: (fx, fy, cx, cy, img_w, img_h) from get_intrinsics_from_meta
    frame_size: (w,h) of current video frame
    """
    if intrinsics_tuple is None or depth_val is None:
        return None, None

    fx, fy, cx, cy, K_w, K_h = intrinsics_tuple
    if frame_size is not None and K_w is not None and K_h is not None:
        fw, fh = frame_size
        if fw != K_w or fh != K_h:
            sx = float(fw) / float(K_w); sy = float(fh) / float(K_h)
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

    x1, y1, x2, y2 = bbox_xyxy
    # left/right center points at object depth
    uL, uR = x1, x2
    vC = (y1 + y2) * 0.5
    Xl = (uL - cx) / fx * depth_val
    Xr = (uR - cx) / fx * depth_val
    width_m = abs(Xr - Xl)

    # top/bottom center points
    uC = (x1 + x2) * 0.5
    vT, vB = y1, y2
    Yt = (vT - cy) / fy * depth_val
    Yb = (vB - cy) / fy * depth_val
    height_m = abs(Yb - Yt)

    return width_m, height_m
