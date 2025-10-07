"""
Plotting helper for floorplans with detections (plot_floorplan).
Updated: fixed furniture icons are warped and bounded to polygons and properly aligned
with the floorplan plotting coordinate system (fixed vertical flip bug).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import cv2
from utils.intrinsics_utils import get_intrinsics_from_meta, compute_real_size_from_bbox

M_TO_FT = 3.280839895013123

def _load_icon_rgba(path, remove_white=True, white_thresh=245):
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Cannot load image: {path}")
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
    if arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
    if remove_white:
        b, g, r, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
        mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
        a[mask] = 0
        arr[:, :, 3] = a
    rgba = arr[:, :, [2, 1, 0, 3]]
    return rgba

def plot_floorplan(found_first, spaces, fixed_furniture, floor_min, floor_max, mapped_plot,
                   out_path=None, title=None, class_image_json=None,
                   rotate_icons=True, remove_white_bg=True, white_threshold=5,
                   show_arrows=True, min_icon_px=24,
                   rejected_list=None, meta=None):
    """
    Plot floorplan with:
      - Colored rooms (by room class/name)
      - Doors (brown lines)
      - Fixed furniture: PNG icons warped/bounded to furniture polygon (from CubiCasa) and aligned
        to the floorplan coordinate system (fixes vertical-flip alignment bug).
      - Icons for detected objects (keeps previous behavior)
      - Optional arrows from camera to object
      - Detected object frame numbers + real-world size (W x H x Depth) in feet plotted near the object
    """
    try:
        if class_image_json is None:
            raise ValueError("class_image_json is required")

        jm_path = Path(class_image_json)
        if not jm_path.exists():
            raise FileNotFoundError(f"class_image_json not found: {class_image_json}")
        with open(jm_path, "r") as f:
            jm_raw = json.load(f)

        # Build class_map (normalize to lowercase keys)
        class_map = {}
        for k, v in jm_raw.items():
            key = str(k).lower()
            if isinstance(v, dict):
                class_map[key] = {
                    "path": v.get("path"),
                    "width": v.get("width"),
                    "height": v.get("height"),
                    "rotate": bool(v.get("rotate", False))
                }
            else:
                class_map[key] = {"path": v, "width": None, "height": None, "rotate": False}

        # helper to load icon into RGBA (matplotlib-friendly)
        def _load_icon_rgba_inner(path, remove_white=remove_white_bg, white_thresh=white_threshold):
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError(f"Cannot load image: {path}")
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
            if arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
            if remove_white:
                b, g, r, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
                mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
                a[mask] = 0
                arr[:, :, 3] = a
            # convert B G R A -> R G B A for internal numpy use
            rgba = arr[:, :, [2, 1, 0, 3]]
            return rgba

        # Simple color map for room types (fallback uses a cycle)
        room_color_map = {
            "bedroom": "#c7e9c0",
            "bedroom master": "#b3e2b8",
            "primary bedroom": "#b3e2b8",
            "livingroom": "#ffd9b3",
            "living room": "#ffd9b3",
            "kitchen": "#d1e7ff",
            "dining": "#f5e6ff",
            "dining area": "#f5e6ff",
            "bath": "#f7c6d9",
            "fullbath": "#f7c6d9",
            "balcony": "#e6e6e6",
            "bathroom": "#f7c6d9"
        }
        default_colors = ["#e8f6f3", "#fff3b0", "#f0d9ff", "#dff0d8", "#fbe7e6"]

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        fig, ax = plt.subplots(figsize=(12, 10))

        # ---------------- Rooms (colored by class/name) ----------------
        for idx, sp in enumerate((spaces or [])):
            try:
                poly = np.asarray(sp["poly"], dtype=float)
                rname = (sp.get("name") or sp.get("class") or "").strip()
                rkey = rname.lower()
                face = None
                if rkey in room_color_map:
                    face = room_color_map[rkey]
                else:
                    for k, v in room_color_map.items():
                        if k in rkey:
                            face = v
                            break
                if face is None:
                    face = default_colors[idx % len(default_colors)]
                patch = MplPolygon(poly, closed=True, fill=True, alpha=0.35,
                                   facecolor=face, edgecolor="black", linewidth=0.8)
                ax.add_patch(patch)
                cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                ax.text(cx, cy, str(rname or sp.get("id") or ""), fontsize=8, ha="center", va="center")
                for door in (sp.get("doors") or []):
                    loc = door.get("location") if isinstance(door, dict) else None
                    if loc and isinstance(loc, dict) and loc.get("type") == "LineString":
                        coords = np.asarray(loc.get("coordinates", []), dtype=float)
                        if coords.shape[0] >= 2:
                            ax.plot(coords[:, 0], coords[:, 1],
                                    linewidth=3.0, color="sienna", solid_capstyle='butt', zorder=30)
            except Exception:
                continue

        # ---------------- Axis setup (pre-calc px_per_unit used for icon scaling/warping) ----------------
        pad_x = max(1.0, abs(floor_max[0] - floor_min[0])) * 0.05
        pad_y = max(1.0, abs(floor_max[1] - floor_min[1])) * 0.05
        ax.set_xlim(float(floor_min[0]) - pad_x, float(floor_max[0]) + pad_x)
        ax.set_ylim(float(floor_min[1]) - pad_y, float(floor_max[1]) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        plt.gca().invert_yaxis()   # keep this, we will account for it in pixel mapping
        fig.canvas.draw()

        def _px_per_unit():
            cx = (floor_min[0] + floor_max[0]) / 2.0
            cy = (floor_min[1] + floor_max[1]) / 2.0
            p1 = ax.transData.transform((cx, cy))
            p2 = ax.transData.transform((cx + 1.0, cy))
            return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

        px_per_unit = max(1.0, _px_per_unit())

        # ---------------- Fixed furniture (use boundingPolygon when available and warp PNG into it) ----------------
        # Canvas pixel dimensions that correspond to data extents (one pixel per data * px_per_unit)
        data_w_units = float(floor_max[0] - floor_min[0])
        data_h_units = float(floor_max[1] - floor_min[1])
        canvas_w_px = max(2, int(round(data_w_units * px_per_unit)))
        canvas_h_px = max(2, int(round(data_h_units * px_per_unit)))

        def _ensure_quad(poly):
            """Return a 4-corner polygon (float32). If poly does not have 4 points, return minAreaRect box."""
            pts = np.array(poly, dtype=np.float32)
            if pts.shape[0] == 4:
                return pts
            rect = cv2.minAreaRect(pts)
            box = cv2.boxPoints(rect)
            return np.array(box, dtype=np.float32)

        def _order_points_tl_tr_br_bl(pts):
            """
            Ensure points are ordered [TL, TR, BR, BL] for pixel-image coords (x right, y down).
            pts: (4,2) array (x,y) in pixel-space coordinates.
            """
            pts = np.array(pts, dtype=np.float32)
            if pts.shape[0] != 4:
                # fallback: compute minAreaRect
                rect = cv2.minAreaRect(pts)
                pts = cv2.boxPoints(rect)
            # compute sums and differences for ordering
            s = pts.sum(axis=1)
            diff = pts[:, 0] - pts[:, 1]
            tl = pts[np.argmin(s)]
            br = pts[np.argmax(s)]
            tr = pts[np.argmin(diff)]
            bl = pts[np.argmax(diff)]
            ordered = np.array([tl, tr, br, bl], dtype=np.float32)
            return ordered

        # initialize empty canvas in BGRA for OpenCV operations (rows = height)
        canvas_bgra = np.zeros((canvas_h_px, canvas_w_px, 4), dtype=np.uint8)

        # iterate fixed furniture items (prefer boundingPolygon if present)
        for ff in (fixed_furniture or []):
            try:
                # extract polygon (prefer CubiCasa boundingPolygon if available)
                bpoly = None
                if isinstance(ff, dict):
                    bp = ff.get("boundingPolygon") or ff.get("bounding_polygon") or ff.get("bbox")
                    if isinstance(bp, dict) and bp.get("coordinates"):
                        coords = bp.get("coordinates")
                        # coords might be nested: Polygon -> [ [ [x,y], ... ] ]
                        if isinstance(coords[0][0], list):
                            bpoly = np.asarray(coords[0], dtype=float)
                        else:
                            bpoly = np.asarray(coords, dtype=float)
                if bpoly is None and "poly" in ff:
                    bpoly = np.asarray(ff["poly"], dtype=float)
                if bpoly is None:
                    continue

                # Normalize key and find mapping (exact or substring)
                raw_key = str((ff.get("class") or ff.get("name") or "")).lower().strip()
                if not raw_key:
                    continue
                mapping = None
                if raw_key in class_map:
                    mapping = class_map[raw_key]
                else:
                    for kmap in class_map.keys():
                        if kmap in raw_key or raw_key in kmap:
                            mapping = class_map[kmap]
                            break
                if mapping is None:
                    continue

                img_path = mapping.get("path")
                if not img_path:
                    continue
                pth = Path(img_path)
                if not pth.exists():
                    pth = Path.cwd() / img_path
                    if not pth.exists():
                        print(f"[plot_floorplan] icon file not found for '{raw_key}': {img_path}")
                        continue

                # load icon RGBA (R,G,B,A)
                icon_rgba = _load_icon_rgba_inner(pth)
                icon_h, icon_w = icon_rgba.shape[:2]

                # source rectangle corners (TL,TR,BR,BL) in icon pixel coords
                src_pts = np.array([[0, 0], [icon_w - 1, 0], [icon_w - 1, icon_h - 1], [0, icon_h - 1]], dtype=np.float32)

                # destination polygon in data units -> we need to map to canvas pixel coordinates
                dst_data = _ensure_quad(bpoly)  # returns 4 points in data units (x,y)
                dst_pix = []
                for (x_d, y_d) in dst_data:
                    # map data coords -> pixel coords (origin = bottom-left)
                    px = (float(x_d) - float(floor_min[0])) * px_per_unit
                    py = (float(y_d) - float(floor_min[1])) * px_per_unit
                    # Now **flip Y** to convert bottom-left origin -> top-left origin used by image arrays
                    py_flipped = (canvas_h_px - 1) - py
                    dst_pix.append([px, py_flipped])
                dst_pix = np.array(dst_pix, dtype=np.float32)

                # order dst points TL,TR,BR,BL (important!)
                dst_ordered = _order_points_tl_tr_br_bl(dst_pix)

                # compute homography from src -> dst
                Hmat = None
                try:
                    Hmat, status = cv2.findHomography(src_pts, dst_ordered)
                except Exception:
                    Hmat = None

                if Hmat is None:
                    # fallback: scale/rescale to bounding box (preserve aspect ratio using mapping.width/height if present)
                    minxy = dst_pix.min(axis=0)
                    maxxy = dst_pix.max(axis=0)
                    minx, miny = int(round(minxy[0])), int(round(minxy[1]))
                    maxx, maxy = int(round(maxxy[0])), int(round(maxxy[1]))
                    tgt_w = max(1, maxx - minx)
                    tgt_h = max(1, maxy - miny)
                    m_w = mapping.get("width"); m_h = mapping.get("height")
                    if m_w or m_h:
                        if m_w:
                            tgt_w = max(1, int(round(float(m_w) * px_per_unit)))
                            scale = tgt_w / float(icon_w) if icon_w else 1.0
                            tgt_h = max(1, int(round(icon_h * scale)))
                        elif m_h:
                            tgt_h = max(1, int(round(float(m_h) * px_per_unit)))
                            scale = tgt_h / float(icon_h) if icon_h else 1.0
                            tgt_w = max(1, int(round(icon_w * scale)))
                    tgt_w = max(min_icon_px, tgt_w)
                    tgt_h = max(min_icon_px, tgt_h)
                    # convert RGBA->BGRA for cv2
                    icon_bgra = icon_rgba[:, :, [2, 1, 0, 3]]
                    resized = cv2.resize(icon_bgra, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                    x0 = min(max(0, minx), canvas_w_px - 1)
                    y0 = min(max(0, miny), canvas_h_px - 1)
                    x1 = min(canvas_w_px, x0 + resized.shape[1])
                    y1 = min(canvas_h_px, y0 + resized.shape[0])
                    if x0 >= x1 or y0 >= y1:
                        continue
                    existing = canvas_bgra[y0:y1, x0:x1, :].astype(np.float32) / 255.0
                    top = resized[0:(y1-y0), 0:(x1-x0), :].astype(np.float32) / 255.0
                    alpha_top = top[:, :, 3:4]
                    alpha_exist = existing[:, :, 3:4]
                    out_rgb = top[:, :, :3] * alpha_top + existing[:, :, :3] * (1 - alpha_top)
                    out_alpha = alpha_top + alpha_exist * (1 - alpha_top)
                    blended = np.concatenate([out_rgb, out_alpha], axis=2)
                    canvas_bgra[y0:y1, x0:x1, :] = (blended * 255.0).clip(0,255).astype(np.uint8)
                    continue

                # warp the icon into the canvas pixel coordinates
                icon_bgra = icon_rgba[:, :, [2, 1, 0, 3]]  # convert R G B A -> B G R A for cv2
                warped = cv2.warpPerspective(icon_bgra, Hmat, (canvas_w_px, canvas_h_px),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=(0,0,0,0))  # BGRA

                # alpha blend warped onto canvas_bgra
                canvas_f = canvas_bgra.astype(np.float32) / 255.0
                warped_f = warped.astype(np.float32) / 255.0
                alpha_top = warped_f[:, :, 3:4]
                alpha_canvas = canvas_f[:, :, 3:4]
                out_rgb = warped_f[:, :, :3] * alpha_top + canvas_f[:, :, :3] * (1 - alpha_top)
                out_alpha = alpha_top + alpha_canvas * (1 - alpha_top)
                canvas_f[:, :, :3] = out_rgb
                canvas_f[:, :, 3:4] = out_alpha
                canvas_bgra = (canvas_f * 255.0).clip(0, 255).astype(np.uint8)

            except Exception as e:
                # keep running - log to console so you can debug that item
                print("Fixed furniture icon failed:", e)
                continue

        # Place the finished pixel-canvas into matplotlib axes using imshow with extent mapped to data coords
        try:
            # convert BGRA -> RGBA for matplotlib
            canvas_rgba = canvas_bgra[:, :, [2,1,0,3]]
            xmin = float(floor_min[0]); ymin = float(floor_min[1])
            xmax = float(floor_max[0]); ymax = float(floor_max[1])
            # origin='upper' because we flipped pixel Y earlier (canvas top-left == data ymax)
            ax.imshow(canvas_rgba, extent=[xmin, xmax, ymin, ymax], origin="upper", zorder=120)
        except Exception as e:
            print("Failed to place furniture canvas on axes:", e)

        # ---------------- helper: find intrinsics robustly ----------------
        def _find_intrinsics_for_detection(frame_idx, arkit_idx=None):
            if meta is None:
                return None, None
            try:
                if frame_idx is not None and 0 <= int(frame_idx) < len(meta):
                    intr = get_intrinsics_from_meta(meta[int(frame_idx)])
                    if intr is not None:
                        return intr, int(frame_idx)
            except Exception:
                pass
            try:
                if arkit_idx is not None and 0 <= int(arkit_idx) < len(meta):
                    intr = get_intrinsics_from_meta(meta[int(arkit_idx)])
                    if intr is not None:
                        return intr, int(arkit_idx)
            except Exception:
                pass
            try:
                candidates = []
                for i, m in enumerate(meta):
                    try:
                        if get_intrinsics_from_meta(m) is not None:
                            candidates.append(i)
                    except Exception:
                        continue
                if candidates:
                    if frame_idx is None:
                        chosen_idx = candidates[0]
                    else:
                        chosen_idx = min(candidates, key=lambda x: abs(x - int(frame_idx)))
                    intr = get_intrinsics_from_meta(meta[chosen_idx])
                    if intr is not None:
                        return intr, int(chosen_idx)
            except Exception:
                pass
            return None, None

        # ---------------- Icons & annotations for detected objects (unchanged behavior) ----------------
        for cls_l, info in sorted((found_first or {}).items()):
            try:
                mapping = class_map.get(cls_l.lower())
                img_path = None
                if mapping and mapping.get("path"):
                    pth = Path(mapping["path"])
                    if pth.exists():
                        img_path = pth
                objx, objy = info.get("object_x"), info.get("object_y")
                camx, camy = info.get("mapped_x"), info.get("mapped_y")
                if objx is None or objy is None:
                    continue

                # Place icon if available
                if img_path is not None:
                    try:
                        rgba = _load_icon_rgba(img_path)
                        src_h, src_w = rgba.shape[:2]
                        desired_w, desired_h = mapping.get("width"), mapping.get("height")
                        tgt_w, tgt_h = src_w, src_h
                        if desired_w and desired_h:
                            tgt_w = int(round(desired_w * px_per_unit))
                            tgt_h = int(round(desired_h * px_per_unit))
                        elif desired_w:
                            tgt_w = int(round(desired_w * px_per_unit))
                            scale = tgt_w / src_w if src_w != 0 else 1.0
                            tgt_h = int(round(src_h * scale))
                        elif desired_h:
                            tgt_h = int(round(desired_h * px_per_unit))
                            scale = tgt_h / src_h if src_h != 0 else 1.0
                            tgt_w = int(round(src_w * scale))
                        tgt_w = max(min_icon_px, int(max(1, tgt_w)))
                        tgt_h = max(min_icon_px, int(max(1, tgt_h)))
                        resized = cv2.resize(rgba, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                        img_for_mat = resized.astype(np.float32) / 255.0
                        ab = AnnotationBbox(OffsetImage(img_for_mat, zoom=1.0), (float(objx), float(objy)),
                                            frameon=False, pad=0.0, zorder=200)
                        ax.add_artist(ab)
                    except Exception:
                        pass

                # camera point + arrow + frame label
                if show_arrows and camx is not None and camy is not None:
                    ax.scatter([camx], [camy], s=80, marker="x", c="red", zorder=210)
                    try:
                        ax.text(camx + 4, camy + 4, f"Cam vf{int(info.get('video_frame_index', -1))}",
                                fontsize=8, color="darkred")
                    except Exception:
                        ax.text(camx + 4, camy + 4, f"Cam vf?", fontsize=8, color="darkred")
                    dx, dy = objx - camx, objy - camy
                    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
                        dx += 1e-3
                    ax.arrow(camx, camy, dx, dy, head_width=6, length_includes_head=True,
                             fc="k", ec="k", zorder=205)

                # compute sizes: width & height from bbox & intrinsics (in meters) then convert to feet
                depth_m = None
                if info.get("distance_m") is not None:
                    try:
                        depth_m = float(info.get("distance_m"))
                    except Exception:
                        depth_m = None
                if depth_m is None and info.get("distance_ft") is not None:
                    try:
                        depth_m = float(info.get("distance_ft")) / float(M_TO_FT)
                    except Exception:
                        depth_m = None

                w_ft_txt = "W:N/A"
                h_ft_txt = "H:N/A"
                d_ft_txt = "D:N/A"
                bbox = info.get("bbox")
                frame_idx = info.get("video_frame_index")
                arkit_idx = info.get("arkit_index") or info.get("arkit_idx") or info.get("arkitIndex") or None

                intr = None
                intr_source_idx = None
                if bbox is not None and depth_m is not None and meta is not None:
                    try:
                        intr, intr_source_idx = _find_intrinsics_for_detection(frame_idx, arkit_idx)
                        if intr is not None:
                            try:
                                w_m, h_m = compute_real_size_from_bbox(bbox, depth_m, intr, frame_size=None)
                                if (w_m is not None) and (h_m is not None) and (w_m > 0 and h_m > 0):
                                    w_ft = float(w_m) * float(M_TO_FT)
                                    h_ft = float(h_m) * float(M_TO_FT)
                                    w_ft_txt = f"W:{w_ft:.1f}ft"
                                    h_ft_txt = f"H:{h_ft:.1f}ft"
                            except Exception:
                                pass
                    except Exception:
                        pass

                if depth_m is not None:
                    try:
                        d_ft = float(depth_m) * float(M_TO_FT)
                        d_ft_txt = f"D:{d_ft:.1f}ft"
                    except Exception:
                        d_ft_txt = "D:N/A"

                frame_txt = f"Frame {int(frame_idx)}" if frame_idx is not None else "Frame ?"
                lines = [str(cls_l), frame_txt, d_ft_txt, w_ft_txt, h_ft_txt]
                ann_txt = "\n".join(lines)
                text_offset_x = 8.0
                text_offset_y = -8.0
                ax.text(objx + text_offset_x, objy + text_offset_y, ann_txt,
                        fontsize=8, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
                        zorder=250)
            except Exception:
                continue

        # ---------------- Finalize & save ----------------
        plt.title(title or "Floor plan with colored rooms and detections")
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=200)
            print("Saved overlay image:", out_path)
        plt.close(fig)
    except Exception as e:
        print("plot_floorplan failed:", e)






# # """
# # Plotting helper for floorplans with detections (plot_floorplan).
# # The implementation mirrors your script (keeps behavior identical).
# # """

# # import json
# # from pathlib import Path
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.patches import Polygon as MplPolygon
# # from matplotlib.path import Path as MplPath
# # from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# # import cv2
# # from utils.intrinsics_utils import get_intrinsics_from_meta, compute_real_size_from_bbox
# # from utils.projection_utils import project_3d_to_2d

# # M_TO_FT = 3.280839895013123

# # def _load_icon_rgba(path, remove_white=True, white_thresh=245):
# #     arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
# #     if arr is None:
# #         raise RuntimeError(f"Cannot load image: {path}")
# #     if arr.ndim == 2:
# #         arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
# #     if arr.shape[2] == 3:
# #         arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
# #     if remove_white:
# #         b, g, r, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
# #         mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
# #         a[mask] = 0
# #         arr[:, :, 3] = a
# #     rgba = arr[:, :, [2, 1, 0, 3]]
# #     return rgba

# # # def plot_floorplan(found_first, spaces, fixed_furniture, floor_min, floor_max, mapped_plot,
# # #                    out_path=None, title=None, class_image_json=None,
# # #                    rotate_icons=True, remove_white_bg=True, white_threshold=245,
# # #                    show_arrows=True, min_icon_px=24,
# # #                    rejected_list=None, meta=None):
# # #     """
# # #     Plot floorplan with:
# # #       - Colored rooms (by room class/name)
# # #       - Doors (brown lines)
# # #       - Fixed furniture (hatched polygons + name + size) AND place PNG icons on fixed furniture
# # #         when a mapping exists in class_image_json (this is the added behavior you requested)
# # #       - Icons for detected objects (keeps previous behavior)
# # #       - Optional arrows from camera to object
# # #       - Detected object frame numbers + real-world size (W x H x Depth) in feet plotted near the object

# # #     NOTE: This function is the updated replacement for your original plot_floorplan and must be
# # #     used in place of it. It retains all previous functionality but adds PNG-placement for fixed
# # #     furniture that matches an icon mapping.
# # #     """
# # #     try:
# # #         if class_image_json is None:
# # #             raise ValueError("class_image_json is required")

# # #         jm_path = Path(class_image_json)
# # #         if not jm_path.exists():
# # #             raise FileNotFoundError(f"class_image_json not found: {class_image_json}")
# # #         with open(jm_path, "r") as f:
# # #             jm_raw = json.load(f)

# # #         # Build class_map (normalize to lowercase keys)
# # #         class_map = {}
# # #         for k, v in jm_raw.items():
# # #             key = str(k).lower()
# # #             if isinstance(v, dict):
# # #                 class_map[key] = {
# # #                     "path": v.get("path"),
# # #                     "width": v.get("width"),
# # #                     "height": v.get("height"),
# # #                     "rotate": bool(v.get("rotate", False))
# # #                 }
# # #             else:
# # #                 # allow string path but keep width/height none (legacy)
# # #                 class_map[key] = {"path": v, "width": None, "height": None, "rotate": False}

# # #         # helper to load icon into RGBA (matplotlib-friendly)
# # #         def _load_icon_rgba(path, remove_white=remove_white_bg, white_thresh=white_threshold):
# # #             arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
# # #             if arr is None:
# # #                 raise RuntimeError(f"Cannot load image: {path}")
# # #             if arr.ndim == 2:
# # #                 arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
# # #             if arr.shape[2] == 3:
# # #                 arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
# # #             # remove near-white -> transparent
# # #             if remove_white:
# # #                 b, g, r, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
# # #                 mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
# # #                 a[mask] = 0
# # #                 arr[:, :, 3] = a
# # #             # convert B G R A -> R G B A
# # #             rgba = arr[:, :, [2, 1, 0, 3]]
# # #             return rgba

# # #         # Simple color map for room types (fallback uses a cycle)
# # #         room_color_map = {
# # #             "bedroom": "#c7e9c0",
# # #             "bedroom master": "#b3e2b8",
# # #             "primary bedroom": "#b3e2b8",
# # #             "livingroom": "#ffd9b3",
# # #             "living room": "#ffd9b3",
# # #             "kitchen": "#d1e7ff",
# # #             "dining": "#f5e6ff",
# # #             "dining area": "#f5e6ff",
# # #             "bath": "#f7c6d9",
# # #             "fullbath": "#f7c6d9",
# # #             "balcony": "#e6e6e6",
# # #             "bathroom": "#f7c6d9"
# # #         }
# # #         default_colors = ["#e8f6f3", "#fff3b0", "#f0d9ff", "#dff0d8", "#fbe7e6"]

# # #         from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# # #         fig, ax = plt.subplots(figsize=(12, 10))

# # #         # ---------------- Rooms (colored by class/name) ----------------
# # #         for idx, sp in enumerate((spaces or [])):
# # #             try:
# # #                 poly = np.asarray(sp["poly"], dtype=float)
# # #                 rname = (sp.get("name") or sp.get("class") or "").strip()
# # #                 rkey = rname.lower()
# # #                 face = None
# # #                 if rkey in room_color_map:
# # #                     face = room_color_map[rkey]
# # #                 else:
# # #                     for k, v in room_color_map.items():
# # #                         if k in rkey:
# # #                             face = v
# # #                             break
# # #                 if face is None:
# # #                     face = default_colors[idx % len(default_colors)]
# # #                 patch = MplPolygon(poly, closed=True, fill=True, alpha=0.35,
# # #                                    facecolor=face, edgecolor="black", linewidth=0.8)
# # #                 ax.add_patch(patch)
# # #                 cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
# # #                 ax.text(cx, cy, str(rname or sp.get("id") or ""), fontsize=8, ha="center", va="center")
# # #                 for door in (sp.get("doors") or []):
# # #                     loc = door.get("location") if isinstance(door, dict) else None
# # #                     if loc and isinstance(loc, dict) and loc.get("type") == "LineString":
# # #                         coords = np.asarray(loc.get("coordinates", []), dtype=float)
# # #                         if coords.shape[0] >= 2:
# # #                             ax.plot(coords[:, 0], coords[:, 1],
# # #                                     linewidth=3.0, color="sienna", solid_capstyle='butt', zorder=30)
# # #             except Exception:
# # #                 continue

# # #         # ---------------- Axis setup (pre-calc px_per_unit used for icon scaling) ----------------
# # #         pad_x = max(1.0, abs(floor_max[0] - floor_min[0])) * 0.05
# # #         pad_y = max(1.0, abs(floor_max[1] - floor_min[1])) * 0.05
# # #         ax.set_xlim(float(floor_min[0]) - pad_x, float(floor_max[0]) + pad_x)
# # #         ax.set_ylim(float(floor_min[1]) - pad_y, float(floor_max[1]) + pad_y)
# # #         ax.set_aspect("equal", adjustable="box")
# # #         plt.gca().invert_yaxis()
# # #         fig.canvas.draw()

# # #         def _px_per_unit():
# # #             cx = (floor_min[0] + floor_max[0]) / 2.0
# # #             cy = (floor_min[1] + floor_max[1]) / 2.0
# # #             p1 = ax.transData.transform((cx, cy))
# # #             p2 = ax.transData.transform((cx + 1.0, cy))
# # #             return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

# # #         px_per_unit = max(1.0, _px_per_unit())

# # #         # ---------------- Fixed furniture (draw polygons AND place PNG if mapping exists) ----------------
# # #         for ff in (fixed_furniture or []):
# # #             try:
# # #                 fpoly = np.asarray(ff["poly"], dtype=float)
# # #                 patch = MplPolygon(fpoly, closed=True, fill=True,
# # #                                    facecolor="none", edgecolor="saddlebrown",
# # #                                    linewidth=1.0, hatch="////", zorder=22)
# # #                 ax.add_patch(patch)
# # #                 cx, cy = np.mean(fpoly[:, 0]), np.mean(fpoly[:, 1])
# # #                 name = ff.get("name") or ff.get("class") or "furniture"
# # #                 w = ff.get("width"); h = ff.get("height"); d = ff.get("depth")
# # #                 dims = []
# # #                 if w: dims.append(f"W={w}")
# # #                 if h: dims.append(f"H={h}")
# # #                 if d: dims.append(f"D={d}")
# # #                 label = f"{name}\n{' '.join(dims)}" if dims else name
# # #                 ax.text(cx, cy, label, fontsize=7, ha="center", va="center", color="saddlebrown")

# # #                 # NEW: try to place an icon on the fixed furniture if mapping exists
# # #                 try:
# # #                     ff_key = str((ff.get("class") or ff.get("name") or "").lower()).strip()
# # #                     if ff_key and ff_key in class_map:
# # #                         mapping = class_map[ff_key]
# # #                         img_path = mapping.get("path")
# # #                         if img_path:
# # #                             pth = Path(img_path)
# # #                             if pth.exists():
# # #                                 rgba = _load_icon_rgba(pth)
# # #                                 src_h, src_w = rgba.shape[:2]

# # #                                 # Determine desired pixel size:
# # #                                 # Prefer mapping width/height (in floor units) if provided, otherwise use bbox of polygon
# # #                                 desired_w_units = mapping.get("width")
# # #                                 desired_h_units = mapping.get("height")
# # #                                 if desired_w_units is None or desired_h_units is None:
# # #                                     # fallback: bounding box of polygon in floor units
# # #                                     minx, miny = fpoly.min(axis=0)
# # #                                     maxx, maxy = fpoly.max(axis=0)
# # #                                     bbox_w_units = float(maxx - minx) if (maxx - minx) > 0 else None
# # #                                     bbox_h_units = float(maxy - miny) if (maxy - miny) > 0 else None
# # #                                     if desired_w_units is None and bbox_w_units is not None:
# # #                                         desired_w_units = bbox_w_units
# # #                                     if desired_h_units is None and bbox_h_units is not None:
# # #                                         desired_h_units = bbox_h_units

# # #                                 # Convert units -> pixels
# # #                                 tgt_w = src_w
# # #                                 tgt_h = src_h
# # #                                 try:
# # #                                     if desired_w_units:
# # #                                         tgt_w = int(round(desired_w_units * px_per_unit))
# # #                                         # preserve aspect ratio
# # #                                         scale = tgt_w / float(src_w) if src_w != 0 else 1.0
# # #                                         tgt_h = int(round(src_h * scale))
# # #                                     elif desired_h_units:
# # #                                         tgt_h = int(round(desired_h_units * px_per_unit))
# # #                                         scale = tgt_h / float(src_h) if src_h != 0 else 1.0
# # #                                         tgt_w = int(round(src_w * scale))
# # #                                 except Exception:
# # #                                     tgt_w, tgt_h = src_w, src_h

# # #                                 # enforce minimum pixel size so icons are visible
# # #                                 tgt_w = max(min_icon_px, int(max(1, tgt_w)))
# # #                                 tgt_h = max(min_icon_px, int(max(1, tgt_h)))

# # #                                 # resize and place
# # #                                 resized = cv2.resize(rgba, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
# # #                                 img_for_mat = resized.astype(np.float32) / 255.0
# # #                                 ab = AnnotationBbox(OffsetImage(img_for_mat, zoom=1.0), (float(cx), float(cy)),
# # #                                                     frameon=False, pad=0.0, zorder=120)
# # #                                 ax.add_artist(ab)
# # #                 except Exception:
# # #                     # do not let icon placement on fixed furniture break plotting
# # #                     pass

# # #             except Exception:
# # #                 continue

# # #         # ---------------- helper: find intrinsics robustly ----------------
# # #         def _find_intrinsics_for_detection(frame_idx, arkit_idx=None):
# # #             if meta is None:
# # #                 return None, None
# # #             try:
# # #                 if frame_idx is not None and 0 <= int(frame_idx) < len(meta):
# # #                     intr = get_intrinsics_from_meta(meta[int(frame_idx)])
# # #                     if intr is not None:
# # #                         return intr, int(frame_idx)
# # #             except Exception:
# # #                 pass
# # #             try:
# # #                 if arkit_idx is not None and 0 <= int(arkit_idx) < len(meta):
# # #                     intr = get_intrinsics_from_meta(meta[int(arkit_idx)])
# # #                     if intr is not None:
# # #                         return intr, int(arkit_idx)
# # #             except Exception:
# # #                 pass
# # #             try:
# # #                 candidates = []
# # #                 for i, m in enumerate(meta):
# # #                     try:
# # #                         if get_intrinsics_from_meta(m) is not None:
# # #                             candidates.append(i)
# # #                     except Exception:
# # #                         continue
# # #                 if candidates:
# # #                     if frame_idx is None:
# # #                         chosen_idx = candidates[0]
# # #                     else:
# # #                         chosen_idx = min(candidates, key=lambda x: abs(x - int(frame_idx)))
# # #                     intr = get_intrinsics_from_meta(meta[chosen_idx])
# # #                     if intr is not None:
# # #                         return intr, int(chosen_idx)
# # #             except Exception:
# # #                 pass
# # #             return None, None

# # #         # ---------------- Icons & annotations for detected objects (unchanged behavior) ----------------
# # #         for cls_l, info in sorted((found_first or {}).items()):
# # #             try:
# # #                 mapping = class_map.get(cls_l.lower())
# # #                 img_path = None
# # #                 if mapping and mapping.get("path"):
# # #                     pth = Path(mapping["path"])
# # #                     if pth.exists():
# # #                         img_path = pth
# # #                 objx, objy = info.get("object_x"), info.get("object_y")
# # #                 camx, camy = info.get("mapped_x"), info.get("mapped_y")
# # #                 if objx is None or objy is None:
# # #                     continue

# # #                 # Place icon if available
# # #                 if img_path is not None:
# # #                     try:
# # #                         rgba = _load_icon_rgba(img_path)
# # #                         src_h, src_w = rgba.shape[:2]
# # #                         desired_w, desired_h = mapping.get("width"), mapping.get("height")
# # #                         tgt_w, tgt_h = src_w, src_h
# # #                         if desired_w and desired_h:
# # #                             tgt_w = int(round(desired_w * px_per_unit))
# # #                             tgt_h = int(round(desired_h * px_per_unit))
# # #                         elif desired_w:
# # #                             tgt_w = int(round(desired_w * px_per_unit))
# # #                             scale = tgt_w / src_w if src_w != 0 else 1.0
# # #                             tgt_h = int(round(src_h * scale))
# # #                         elif desired_h:
# # #                             tgt_h = int(round(desired_h * px_per_unit))
# # #                             scale = tgt_h / src_h if src_h != 0 else 1.0
# # #                             tgt_w = int(round(src_w * scale))
# # #                         tgt_w = max(min_icon_px, int(max(1, tgt_w)))
# # #                         tgt_h = max(min_icon_px, int(max(1, tgt_h)))
# # #                         resized = cv2.resize(rgba, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
# # #                         img_for_mat = resized.astype(np.float32) / 255.0
# # #                         ab = AnnotationBbox(OffsetImage(img_for_mat, zoom=1.0), (float(objx), float(objy)),
# # #                                             frameon=False, pad=0.0, zorder=200)
# # #                         ax.add_artist(ab)
# # #                     except Exception:
# # #                         pass

# # #                 # camera point + arrow + frame label
# # #                 if show_arrows and camx is not None and camy is not None:
# # #                     ax.scatter([camx], [camy], s=80, marker="x", c="red", zorder=210)
# # #                     try:
# # #                         ax.text(camx + 4, camy + 4, f"Cam vf{int(info.get('video_frame_index', -1))}",
# # #                                 fontsize=8, color="darkred")
# # #                     except Exception:
# # #                         ax.text(camx + 4, camy + 4, f"Cam vf?", fontsize=8, color="darkred")
# # #                     dx, dy = objx - camx, objy - camy
# # #                     if abs(dx) < 1e-8 and abs(dy) < 1e-8:
# # #                         dx += 1e-3
# # #                     ax.arrow(camx, camy, dx, dy, head_width=6, length_includes_head=True,
# # #                              fc="k", ec="k", zorder=205)

# # #                 # compute sizes: width & height from bbox & intrinsics (in meters) then convert to feet
# # #                 depth_m = None
# # #                 if info.get("distance_m") is not None:
# # #                     try:
# # #                         depth_m = float(info.get("distance_m"))
# # #                     except Exception:
# # #                         depth_m = None
# # #                 if depth_m is None and info.get("distance_ft") is not None:
# # #                     try:
# # #                         depth_m = float(info.get("distance_ft")) / float(M_TO_FT)
# # #                     except Exception:
# # #                         depth_m = None

# # #                 w_ft_txt = "W:N/A"
# # #                 h_ft_txt = "H:N/A"
# # #                 d_ft_txt = "D:N/A"
# # #                 bbox = info.get("bbox")
# # #                 frame_idx = info.get("video_frame_index")
# # #                 arkit_idx = info.get("arkit_index") or info.get("arkit_idx") or info.get("arkitIndex") or None

# # #                 intr = None
# # #                 intr_source_idx = None
# # #                 if bbox is not None and depth_m is not None and meta is not None:
# # #                     try:
# # #                         intr, intr_source_idx = _find_intrinsics_for_detection(frame_idx, arkit_idx)
# # #                         if intr is not None:
# # #                             try:
# # #                                 w_m, h_m = compute_real_size_from_bbox(bbox, depth_m, intr, frame_size=None)
# # #                                 if (w_m is not None) and (h_m is not None) and (w_m > 0 and h_m > 0):
# # #                                     w_ft = float(w_m) * float(M_TO_FT)
# # #                                     h_ft = float(h_m) * float(M_TO_FT)
# # #                                     w_ft_txt = f"W:{w_ft:.1f}ft"
# # #                                     h_ft_txt = f"H:{h_ft:.1f}ft"
# # #                             except Exception:
# # #                                 pass
# # #                     except Exception:
# # #                         pass

# # #                 if depth_m is not None:
# # #                     try:
# # #                         d_ft = float(depth_m) * float(M_TO_FT)
# # #                         d_ft_txt = f"D:{d_ft:.1f}ft"
# # #                     except Exception:
# # #                         d_ft_txt = "D:N/A"

# # #                 frame_txt = f"Frame {int(frame_idx)}" if frame_idx is not None else "Frame ?"
# # #                 lines = [str(cls_l), frame_txt, d_ft_txt, w_ft_txt, h_ft_txt]
# # #                 ann_txt = "\n".join(lines)
# # #                 text_offset_x = 8.0
# # #                 text_offset_y = -8.0
# # #                 ax.text(objx + text_offset_x, objy + text_offset_y, ann_txt,
# # #                         fontsize=8, ha="left", va="bottom",
# # #                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
# # #                         zorder=250)
# # #             except Exception:
# # #                 continue

# # #         # ---------------- Finalize & save ----------------
# # #         plt.title(title or "Floor plan with colored rooms and detections")
# # #         if out_path:
# # #             out_path = Path(out_path)
# # #             out_path.parent.mkdir(parents=True, exist_ok=True)
# # #             plt.savefig(str(out_path), dpi=200)
# # #             print("Saved overlay image:", out_path)
# # #         plt.close(fig)
# # #     except Exception as e:
# # #         print("plot_floorplan failed:", e)


# # def plot_floorplan(found_first, spaces, fixed_furniture, floor_min, floor_max, mapped_plot,
# #                    out_path=None, title=None, class_image_json=None,
# #                    rotate_icons=True, remove_white_bg=True, white_threshold=245,
# #                    show_arrows=True, min_icon_px=24,
# #                    rejected_list=None, meta=None):
# #     """
# #     Plot floorplan with:
# #       - Colored rooms (by room class/name)
# #       - Doors (brown lines)
# #       - Fixed furniture (REPLACED: place PNG icons bounded/warped to the furniture bounding polygon from CubiCasa)
# #       - Icons for detected objects (keeps previous behavior)
# #       - Optional arrows from camera to object
# #       - Detected object frame numbers + real-world size (W x H x Depth) in feet plotted near the object
# #     """
# #     try:
# #         if class_image_json is None:
# #             raise ValueError("class_image_json is required")

# #         jm_path = Path(class_image_json)
# #         if not jm_path.exists():
# #             raise FileNotFoundError(f"class_image_json not found: {class_image_json}")
# #         with open(jm_path, "r") as f:
# #             jm_raw = json.load(f)

# #         # Build class_map (normalize to lowercase keys)
# #         class_map = {}
# #         for k, v in jm_raw.items():
# #             key = str(k).lower()
# #             if isinstance(v, dict):
# #                 class_map[key] = {
# #                     "path": v.get("path"),
# #                     "width": v.get("width"),
# #                     "height": v.get("height"),
# #                     "rotate": bool(v.get("rotate", False))
# #                 }
# #             else:
# #                 class_map[key] = {"path": v, "width": None, "height": None, "rotate": False}

# #         # helper to load icon into RGBA (matplotlib-friendly)
# #         def _load_icon_rgba_inner(path, remove_white=remove_white_bg, white_thresh=white_threshold):
# #             arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
# #             if arr is None:
# #                 raise RuntimeError(f"Cannot load image: {path}")
# #             if arr.ndim == 2:
# #                 arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
# #             if arr.shape[2] == 3:
# #                 arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
# #             if remove_white:
# #                 b, g, r, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
# #                 mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
# #                 a[mask] = 0
# #                 arr[:, :, 3] = a
# #             # convert B G R A -> R G B A for consistent numpy mat use (we will convert back to BGRA for cv2)
# #             rgba = arr[:, :, [2, 1, 0, 3]]
# #             return rgba

# #         # Room color map + plotting (unchanged)
# #         room_color_map = {
# #             "bedroom": "#c7e9c0",
# #             "bedroom master": "#b3e2b8",
# #             "primary bedroom": "#b3e2b8",
# #             "livingroom": "#ffd9b3",
# #             "living room": "#ffd9b3",
# #             "kitchen": "#d1e7ff",
# #             "dining": "#f5e6ff",
# #             "dining area": "#f5e6ff",
# #             "bath": "#f7c6d9",
# #             "fullbath": "#f7c6d9",
# #             "balcony": "#e6e6e6",
# #             "bathroom": "#f7c6d9"
# #         }
# #         default_colors = ["#e8f6f3", "#fff3b0", "#f0d9ff", "#dff0d8", "#fbe7e6"]

# #         from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# #         fig, ax = plt.subplots(figsize=(12, 10))

# #         # ---------------- Rooms (colored by class/name) ----------------
# #         for idx, sp in enumerate((spaces or [])):
# #             try:
# #                 poly = np.asarray(sp["poly"], dtype=float)
# #                 rname = (sp.get("name") or sp.get("class") or "").strip()
# #                 rkey = rname.lower()
# #                 face = None
# #                 if rkey in room_color_map:
# #                     face = room_color_map[rkey]
# #                 else:
# #                     for k, v in room_color_map.items():
# #                         if k in rkey:
# #                             face = v
# #                             break
# #                 if face is None:
# #                     face = default_colors[idx % len(default_colors)]
# #                 patch = MplPolygon(poly, closed=True, fill=True, alpha=0.35,
# #                                    facecolor=face, edgecolor="black", linewidth=0.8)
# #                 ax.add_patch(patch)
# #                 cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
# #                 ax.text(cx, cy, str(rname or sp.get("id") or ""), fontsize=8, ha="center", va="center")
# #                 for door in (sp.get("doors") or []):
# #                     loc = door.get("location") if isinstance(door, dict) else None
# #                     if loc and isinstance(loc, dict) and loc.get("type") == "LineString":
# #                         coords = np.asarray(loc.get("coordinates", []), dtype=float)
# #                         if coords.shape[0] >= 2:
# #                             ax.plot(coords[:, 0], coords[:, 1],
# #                                     linewidth=3.0, color="sienna", solid_capstyle='butt', zorder=30)
# #             except Exception:
# #                 continue

# #         # ---------------- Axis setup (pre-calc px_per_unit used for icon scaling/warping) ----------------
# #         pad_x = max(1.0, abs(floor_max[0] - floor_min[0])) * 0.05
# #         pad_y = max(1.0, abs(floor_max[1] - floor_min[1])) * 0.05
# #         ax.set_xlim(float(floor_min[0]) - pad_x, float(floor_max[0]) + pad_x)
# #         ax.set_ylim(float(floor_min[1]) - pad_y, float(floor_max[1]) + pad_y)
# #         ax.set_aspect("equal", adjustable="box")
# #         plt.gca().invert_yaxis()
# #         fig.canvas.draw()

# #         def _px_per_unit():
# #             cx = (floor_min[0] + floor_max[0]) / 2.0
# #             cy = (floor_min[1] + floor_max[1]) / 2.0
# #             p1 = ax.transData.transform((cx, cy))
# #             p2 = ax.transData.transform((cx + 1.0, cy))
# #             return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

# #         px_per_unit = max(1.0, _px_per_unit())

# #         # ---------------- Fixed furniture (use boundingPolygon when available and warp PNG into it) ----------------
# #         # Compute canvas pixel size corresponding to data extents
# #         data_w_units = float(floor_max[0] - floor_min[0])
# #         data_h_units = float(floor_max[1] - floor_min[1])
# #         canvas_w_px = max(2, int(round(data_w_units * px_per_unit)))
# #         canvas_h_px = max(2, int(round(data_h_units * px_per_unit)))

# #         def _ensure_quad(poly):
# #             """Return a 4-corner polygon (float32). If poly has not 4 points, fallback to minAreaRect."""
# #             pts = np.array(poly, dtype=np.float32)
# #             if pts.shape[0] == 4:
# #                 return pts
# #             rect = cv2.minAreaRect(pts)
# #             box = cv2.boxPoints(rect)
# #             return np.array(box, dtype=np.float32)

# #         # initialize empty canvas in BGRA for OpenCV operations
# #         canvas_bgra = np.zeros((canvas_h_px, canvas_w_px, 4), dtype=np.uint8)

# #         # iterate fixed furniture items (prefer boundingPolygon if present)
# #         for ff in (fixed_furniture or []):
# #             try:
# #                 # Prefer CubiCasa boundingPolygon -> coordinates; fall back to ff["poly"]
# #                 bpoly = None
# #                 if isinstance(ff, dict):
# #                     bp = ff.get("boundingPolygon") or ff.get("bounding_polygon") or ff.get("bbox")
# #                     if isinstance(bp, dict) and bp.get("coordinates"):
# #                         # coords may be nested (Polygon -> [ [ [x,y], ... ] ])
# #                         coords = bp.get("coordinates")
# #                         # handle both nested and flat
# #                         if isinstance(coords[0][0], list) or isinstance(coords[0][0], (float, int)):
# #                             # flatten one level if needed
# #                             if isinstance(coords[0][0], list):
# #                                 bpoly = np.asarray(coords[0], dtype=float)
# #                             else:
# #                                 bpoly = np.asarray(coords, dtype=float)
# #                 if bpoly is None and "poly" in ff:
# #                     bpoly = np.asarray(ff["poly"], dtype=float)
# #                 if bpoly is None:
# #                     continue

# #                 # Normalize class key and find mapping (flexible substring matching)
# #                 raw_key = str((ff.get("class") or ff.get("name") or "")).lower().strip()
# #                 if not raw_key:
# #                     continue
# #                 mapping = None
# #                 if raw_key in class_map:
# #                     mapping = class_map[raw_key]
# #                 else:
# #                     for kmap in class_map.keys():
# #                         if kmap in raw_key or raw_key in kmap:
# #                             mapping = class_map[kmap]
# #                             break
# #                 if mapping is None:
# #                     # no icon mapping found for this furniture
# #                     continue

# #                 img_path = mapping.get("path")
# #                 if not img_path:
# #                     continue
# #                 pth = Path(img_path)
# #                 if not pth.exists():
# #                     # try resolving relative to cwd
# #                     pth = Path.cwd() / img_path
# #                     if not pth.exists():
# #                         print(f"[plot_floorplan] icon file not found for '{raw_key}': {img_path}")
# #                         continue

# #                 # load icon RGBA (R G B A)
# #                 icon_rgba = _load_icon_rgba_inner(pth)
# #                 icon_h, icon_w = icon_rgba.shape[:2]

# #                 # prepare src (icon) and dst (polygon) points
# #                 src_pts = np.array([[0, 0], [icon_w - 1, 0], [icon_w - 1, icon_h - 1], [0, icon_h - 1]], dtype=np.float32)
# #                 dst_pts_data = _ensure_quad(bpoly)  # data coordinates (units)
# #                 # convert data coords -> pixel coords on canvas (origin = floor_min)
# #                 dst_pts_pix = []
# #                 for (x_d, y_d) in dst_pts_data:
# #                     px_x = (float(x_d) - float(floor_min[0])) * px_per_unit
# #                     px_y = (float(y_d) - float(floor_min[1])) * px_per_unit
# #                     # clamp to canvas
# #                     px_x = float(px_x); px_y = float(px_y)
# #                     dst_pts_pix.append([px_x, px_y])
# #                 dst_pts_pix = np.array(dst_pts_pix, dtype=np.float32)

# #                 # compute homography (icon -> polygon in canvas pixel space)
# #                 Hmat = None
# #                 try:
# #                     Hmat, status = cv2.findHomography(src_pts, dst_pts_pix)
# #                 except Exception:
# #                     Hmat = None

# #                 if Hmat is None:
# #                     # fallback: scale to bounding box of polygon
# #                     minx, miny = dst_pts_pix.min(axis=0)
# #                     maxx, maxy = dst_pts_pix.max(axis=0)
# #                     tgt_w = max(1, int(round(maxx - minx)))
# #                     tgt_h = max(1, int(round(maxy - miny)))
# #                     # preserve aspect ratio using mapping.width/height if provided (in data units)
# #                     m_w_units = mapping.get("width")
# #                     m_h_units = mapping.get("height")
# #                     if m_w_units or m_h_units:
# #                         if m_w_units:
# #                             tgt_w = max(1, int(round(m_w_units * px_per_unit)))
# #                             scale = tgt_w / float(icon_w) if icon_w else 1.0
# #                             tgt_h = max(1, int(round(icon_h * scale)))
# #                         elif m_h_units:
# #                             tgt_h = max(1, int(round(m_h_units * px_per_unit)))
# #                             scale = tgt_h / float(icon_h) if icon_h else 1.0
# #                             tgt_w = max(1, int(round(icon_w * scale)))
# #                     # ensure min icon pixels
# #                     tgt_w = max(min_icon_px, tgt_w)
# #                     tgt_h = max(min_icon_px, tgt_h)
# #                     # convert RGBA (R G B A) -> BGRA for cv2
# #                     icon_bgra = icon_rgba[:, :, [2, 1, 0, 3]]
# #                     resized = cv2.resize(icon_bgra, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
# #                     x0 = int(round(minx)); y0 = int(round(miny))
# #                     x1 = min(canvas_w_px, x0 + resized.shape[1])
# #                     y1 = min(canvas_h_px, y0 + resized.shape[0])
# #                     if x0 < 0 or y0 < 0 or x0 >= canvas_w_px or y0 >= canvas_h_px:
# #                         # out of canvas - skip
# #                         continue
# #                     # blend
# #                     existing = canvas_bgra[y0:y1, x0:x1, :].astype(np.float32) / 255.0
# #                     top = resized[0:(y1-y0), 0:(x1-x0), :].astype(np.float32) / 255.0
# #                     alpha_top = top[:, :, 3:4]
# #                     alpha_exist = existing[:, :, 3:4]
# #                     out_rgb = top[:, :, :3] * alpha_top + existing[:, :, :3] * (1 - alpha_top)
# #                     out_alpha = alpha_top + alpha_exist * (1 - alpha_top)
# #                     blended = np.concatenate([out_rgb, out_alpha], axis=2)
# #                     canvas_bgra[y0:y1, x0:x1, :] = (blended * 255.0).clip(0,255).astype(np.uint8)
# #                     continue

# #                 # warp with homography into canvas pixel space (canvas size: canvas_w_px x canvas_h_px)
# #                 icon_bgra = icon_rgba[:, :, [2, 1, 0, 3]]  # convert R G B A -> B G R A
# #                 warped = cv2.warpPerspective(icon_bgra, Hmat, (canvas_w_px, canvas_h_px),
# #                                              flags=cv2.INTER_LINEAR,
# #                                              borderMode=cv2.BORDER_CONSTANT,
# #                                              borderValue=(0,0,0,0))  # BGRA

# #                 # alpha blend warped onto canvas_bgra
# #                 canvas_f = canvas_bgra.astype(np.float32) / 255.0
# #                 warped_f = warped.astype(np.float32) / 255.0
# #                 alpha_top = warped_f[:, :, 3:4]
# #                 alpha_canvas = canvas_f[:, :, 3:4]
# #                 out_rgb = warped_f[:, :, :3] * alpha_top + canvas_f[:, :, :3] * (1 - alpha_top)
# #                 out_alpha = alpha_top + alpha_canvas * (1 - alpha_top)
# #                 canvas_f[:, :, :3] = out_rgb
# #                 canvas_f[:, :, 3:4] = out_alpha
# #                 canvas_bgra = (canvas_f * 255.0).clip(0, 255).astype(np.uint8)

# #             except Exception as e:
# #                 print("Fixed furniture icon failed:", e)
# #                 continue

# #         # Place the finished pixel-canvas into matplotlib axes using imshow with extent mapped to data coords
# #         try:
# #             # convert BGRA -> RGBA for matplotlib
# #             canvas_rgba = canvas_bgra[:, :, [2,1,0,3]]
# #             xmin = float(floor_min[0])
# #             ymin = float(floor_min[1])
# #             xmax = float(floor_max[0])
# #             ymax = float(floor_max[1])
# #             # Because we inverted the y-axis above, origin='upper' keeps alignment correct
# #             ax.imshow(canvas_rgba, extent=[xmin, xmax, ymin, ymax], origin="upper", zorder=120)
# #         except Exception as e:
# #             print("Failed to place furniture canvas on axes:", e)

# #         # ---------------- helper: find intrinsics robustly ----------------
# #         def _find_intrinsics_for_detection(frame_idx, arkit_idx=None):
# #             if meta is None:
# #                 return None, None
# #             try:
# #                 if frame_idx is not None and 0 <= int(frame_idx) < len(meta):
# #                     intr = get_intrinsics_from_meta(meta[int(frame_idx)])
# #                     if intr is not None:
# #                         return intr, int(frame_idx)
# #             except Exception:
# #                 pass
# #             try:
# #                 if arkit_idx is not None and 0 <= int(arkit_idx) < len(meta):
# #                     intr = get_intrinsics_from_meta(meta[int(arkit_idx)])
# #                     if intr is not None:
# #                         return intr, int(arkit_idx)
# #             except Exception:
# #                 pass
# #             try:
# #                 candidates = []
# #                 for i, m in enumerate(meta):
# #                     try:
# #                         if get_intrinsics_from_meta(m) is not None:
# #                             candidates.append(i)
# #                     except Exception:
# #                         continue
# #                 if candidates:
# #                     if frame_idx is None:
# #                         chosen_idx = candidates[0]
# #                     else:
# #                         chosen_idx = min(candidates, key=lambda x: abs(x - int(frame_idx)))
# #                     intr = get_intrinsics_from_meta(meta[chosen_idx])
# #                     if intr is not None:
# #                         return intr, int(chosen_idx)
# #             except Exception:
# #                 pass
# #             return None, None

# #         # ---------------- Icons & annotations for detected objects (unchanged behavior) ----------------
# #         for cls_l, info in sorted((found_first or {}).items()):
# #             try:
# #                 mapping = class_map.get(cls_l.lower())
# #                 img_path = None
# #                 if mapping and mapping.get("path"):
# #                     pth = Path(mapping["path"])
# #                     if pth.exists():
# #                         img_path = pth
# #                 objx, objy = info.get("object_x"), info.get("object_y")
# #                 camx, camy = info.get("mapped_x"), info.get("mapped_y")
# #                 if objx is None or objy is None:
# #                     continue

# #                 # Place icon if available
# #                 if img_path is not None:
# #                     try:
# #                         rgba = _load_icon_rgba(img_path)
# #                         src_h, src_w = rgba.shape[:2]
# #                         desired_w, desired_h = mapping.get("width"), mapping.get("height")
# #                         tgt_w, tgt_h = src_w, src_h
# #                         if desired_w and desired_h:
# #                             tgt_w = int(round(desired_w * px_per_unit))
# #                             tgt_h = int(round(desired_h * px_per_unit))
# #                         elif desired_w:
# #                             tgt_w = int(round(desired_w * px_per_unit))
# #                             scale = tgt_w / src_w if src_w != 0 else 1.0
# #                             tgt_h = int(round(src_h * scale))
# #                         elif desired_h:
# #                             tgt_h = int(round(desired_h * px_per_unit))
# #                             scale = tgt_h / src_h if src_h != 0 else 1.0
# #                             tgt_w = int(round(src_w * scale))
# #                         tgt_w = max(min_icon_px, int(max(1, tgt_w)))
# #                         tgt_h = max(min_icon_px, int(max(1, tgt_h)))
# #                         resized = cv2.resize(rgba, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
# #                         img_for_mat = resized.astype(np.float32) / 255.0
# #                         ab = AnnotationBbox(OffsetImage(img_for_mat, zoom=1.0), (float(objx), float(objy)),
# #                                             frameon=False, pad=0.0, zorder=200)
# #                         ax.add_artist(ab)
# #                     except Exception:
# #                         pass

# #                 # camera point + arrow + frame label
# #                 if show_arrows and camx is not None and camy is not None:
# #                     ax.scatter([camx], [camy], s=80, marker="x", c="red", zorder=210)
# #                     try:
# #                         ax.text(camx + 4, camy + 4, f"Cam vf{int(info.get('video_frame_index', -1))}",
# #                                 fontsize=8, color="darkred")
# #                     except Exception:
# #                         ax.text(camx + 4, camy + 4, f"Cam vf?", fontsize=8, color="darkred")
# #                     dx, dy = objx - camx, objy - camy
# #                     if abs(dx) < 1e-8 and abs(dy) < 1e-8:
# #                         dx += 1e-3
# #                     ax.arrow(camx, camy, dx, dy, head_width=6, length_includes_head=True,
# #                              fc="k", ec="k", zorder=205)

# #                 # compute sizes: width & height from bbox & intrinsics (in meters) then convert to feet
# #                 depth_m = None
# #                 if info.get("distance_m") is not None:
# #                     try:
# #                         depth_m = float(info.get("distance_m"))
# #                     except Exception:
# #                         depth_m = None
# #                 if depth_m is None and info.get("distance_ft") is not None:
# #                     try:
# #                         depth_m = float(info.get("distance_ft")) / float(M_TO_FT)
# #                     except Exception:
# #                         depth_m = None

# #                 w_ft_txt = "W:N/A"
# #                 h_ft_txt = "H:N/A"
# #                 d_ft_txt = "D:N/A"
# #                 bbox = info.get("bbox")
# #                 frame_idx = info.get("video_frame_index")
# #                 arkit_idx = info.get("arkit_index") or info.get("arkit_idx") or info.get("arkitIndex") or None

# #                 intr = None
# #                 intr_source_idx = None
# #                 if bbox is not None and depth_m is not None and meta is not None:
# #                     try:
# #                         intr, intr_source_idx = _find_intrinsics_for_detection(frame_idx, arkit_idx)
# #                         if intr is not None:
# #                             try:
# #                                 w_m, h_m = compute_real_size_from_bbox(bbox, depth_m, intr, frame_size=None)
# #                                 if (w_m is not None) and (h_m is not None) and (w_m > 0 and h_m > 0):
# #                                     w_ft = float(w_m) * float(M_TO_FT)
# #                                     h_ft = float(h_m) * float(M_TO_FT)
# #                                     w_ft_txt = f"W:{w_ft:.1f}ft"
# #                                     h_ft_txt = f"H:{h_ft:.1f}ft"
# #                             except Exception:
# #                                 pass
# #                     except Exception:
# #                         pass

# #                 if depth_m is not None:
# #                     try:
# #                         d_ft = float(depth_m) * float(M_TO_FT)
# #                         d_ft_txt = f"D:{d_ft:.1f}ft"
# #                     except Exception:
# #                         d_ft_txt = "D:N/A"

# #                 frame_txt = f"Frame {int(frame_idx)}" if frame_idx is not None else "Frame ?"
# #                 lines = [str(cls_l), frame_txt, d_ft_txt, w_ft_txt, h_ft_txt]
# #                 ann_txt = "\n".join(lines)
# #                 text_offset_x = 8.0
# #                 text_offset_y = -8.0
# #                 ax.text(objx + text_offset_x, objy + text_offset_y, ann_txt,
# #                         fontsize=8, ha="left", va="bottom",
# #                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
# #                         zorder=250)
# #             except Exception:
# #                 continue

# #         # ---------------- Finalize & save ----------------
# #         plt.title(title or "Floor plan with colored rooms and detections")
# #         if out_path:
# #             out_path = Path(out_path)
# #             out_path.parent.mkdir(parents=True, exist_ok=True)
# #             plt.savefig(str(out_path), dpi=200)
# #             print("Saved overlay image:", out_path)
# #         plt.close(fig)
# #     except Exception as e:
# #         print("plot_floorplan failed:", e)
