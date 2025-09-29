"""
Plotting helper for floorplans with detections (plot_floorplan).
The implementation mirrors your script (keeps behavior identical).
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import cv2
from utils.intrinsics_utils import get_intrinsics_from_meta, compute_real_size_from_bbox
from utils.projection_utils import project_3d_to_2d

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
                   rotate_icons=True, remove_white_bg=True, white_threshold=245,
                   show_arrows=True, min_icon_px=24,
                   rejected_list=None, meta=None):
    try:
        if class_image_json is None:
            raise ValueError("class_image_json is required")
        jm_path = Path(class_image_json)
        if not jm_path.exists():
            raise FileNotFoundError(f"class_image_json not found: {class_image_json}")
        with open(jm_path, "r") as f:
            jm_raw = json.load(f)
        class_map = {}
        for k, v in jm_raw.items():
            key = str(k).lower()
            if isinstance(v, dict):
                class_map[key] = {"path": v.get("path"), "width": v.get("width"), "height": v.get("height"), "rotate": bool(v.get("rotate", False))}
            else:
                class_map[key] = {"path": v, "width": None, "height": None, "rotate": False}

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

        fig, ax = plt.subplots(figsize=(12, 10))

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

        for ff in (fixed_furniture or []):
            try:
                fpoly = np.asarray(ff["poly"], dtype=float)
                patch = MplPolygon(fpoly, closed=True, fill=True,
                                   facecolor="none", edgecolor="saddlebrown",
                                   linewidth=1.0, hatch="////", zorder=22)
                ax.add_patch(patch)
                cx, cy = np.mean(fpoly[:, 0]), np.mean(fpoly[:, 1])
                name = ff.get("name") or ff.get("class") or "furniture"
                w = ff.get("width"); h = ff.get("height"); d = ff.get("depth")
                dims = []
                if w: dims.append(f"W={w}")
                if h: dims.append(f"H={h}")
                if d: dims.append(f"D={d}")
                label = f"{name}\n{' '.join(dims)}" if dims else name
                ax.text(cx, cy, label, fontsize=7, ha="center", va="center", color="saddlebrown")
            except Exception:
                continue

        pad_x = max(1.0, abs(floor_max[0] - floor_min[0])) * 0.05
        pad_y = max(1.0, abs(floor_max[1] - floor_min[1])) * 0.05
        ax.set_xlim(float(floor_min[0]) - pad_x, float(floor_max[0]) + pad_x)
        ax.set_ylim(float(floor_min[1]) - pad_y, float(floor_max[1]) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        plt.gca().invert_yaxis()
        fig.canvas.draw()

        def _px_per_unit():
            cx = (floor_min[0] + floor_max[0]) / 2.0
            cy = (floor_min[1] + floor_max[1]) / 2.0
            p1 = ax.transData.transform((cx, cy))
            p2 = ax.transData.transform((cx + 1.0, cy))
            return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

        px_per_unit = max(1.0, _px_per_unit())

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
                if arkit_idx is None:
                    arkit_idx = None
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

                if img_path is not None:
                    try:
                        rgba = _load_icon_rgba(img_path, remove_white=remove_white_bg, white_thresh=white_threshold)
                        src_h, src_w = rgba.shape[:2]
                        desired_w, desired_h = mapping.get("width"), mapping.get("height")
                        tgt_w, tgt_h = src_w, src_h
                        if desired_w and desired_h:
                            tgt_w = int(round(desired_w * px_per_unit))
                            tgt_h = int(round(desired_h * px_per_unit))
                        elif desired_w:
                            tgt_w = int(round(desired_w * px_per_unit))
                            scale = tgt_w / src_w
                            tgt_h = int(round(src_h * scale))
                        elif desired_h:
                            tgt_h = int(round(desired_h * px_per_unit))
                            scale = tgt_h / src_h
                            tgt_w = int(round(src_w * scale))
                        tgt_w = max(min_icon_px, tgt_w)
                        tgt_h = max(min_icon_px, tgt_h)
                        resized = cv2.resize(rgba, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                        img_for_mat = resized.astype(np.float32) / 255.0
                        ab = AnnotationBbox(OffsetImage(img_for_mat, zoom=1.0), (float(objx), float(objy)),
                                            frameon=False, pad=0.0, zorder=100)
                        ax.add_artist(ab)
                    except Exception:
                        pass

                if show_arrows and camx is not None and camy is not None:
                    ax.scatter([camx], [camy], s=80, marker="x", c="red", zorder=120)
                    try:
                        ax.text(camx + 4, camy + 4, f"Cam vf{int(info.get('video_frame_index', -1))}",
                                fontsize=8, color="darkred")
                    except Exception:
                        ax.text(camx + 4, camy + 4, f"Cam vf?", fontsize=8, color="darkred")
                    dx, dy = objx - camx, objy - camy
                    if abs(dx) < 1e-8 and abs(dy) < 1e-8:
                        dx += 1e-3
                    ax.arrow(camx, camy, dx, dy, head_width=6, length_includes_head=True,
                             fc="k", ec="k", zorder=110)

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
                        if intr is None:
                            print(f"DEBUG: intrinsics not found for class {cls_l} arkit_idx={arkit_idx} frame_idx={frame_idx}")
                        else:
                            print(f"DEBUG: using intrinsics from meta[{intr_source_idx}] for class {cls_l} (frame_idx={frame_idx})")
                            try:
                                w_m, h_m = compute_real_size_from_bbox(bbox, depth_m, intr, frame_size=None)
                                if (w_m is not None) and (h_m is not None) and (w_m > 0 and h_m > 0):
                                    w_ft = float(w_m) * float(M_TO_FT)
                                    h_ft = float(h_m) * float(M_TO_FT)
                                    w_ft_txt = f"W:{w_ft:.1f}ft"
                                    h_ft_txt = f"H:{h_ft:.1f}ft"
                            except Exception as e:
                                print(f"DEBUG: compute_real_size_from_bbox failed for {cls_l}: {e}")
                    except Exception as e:
                        print("DEBUG: intrinsics lookup error:", e)
                        intr = None

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
                        zorder=200)
            except Exception:
                continue

        plt.title(title or "Floor plan with colored rooms and detections")
        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=200)
            print("Saved overlay image:", out_path)
        plt.close(fig)
    except Exception as e:
        print("plot_floorplan failed:", e)
