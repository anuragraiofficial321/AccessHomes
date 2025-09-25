import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
import matplotlib.font_manager as fm
import numpy as np
import cv2
import json
from pathlib import Path
from config.config import (VERBOSE, M_TO_FT) 

def plot_floorplan(found_first, spaces, fixed_furniture, floor_min, floor_max, mapped_plot,
                   out_path=None, title=None, class_image_json=None,
                   rotate_icons=True, remove_white_bg=True, white_threshold=245,
                   show_arrows=True):
    """
    PNG-only plotting with OpenCV + optional camera arrows.

    - Reads PNGs with cv2 (BGRA → RGBA).
    - Stretches icons to exact width & height if both provided.
    - Removes near-white backgrounds.
    - If show_arrows=True: plots camera location and arrow to object.
    - Annotates each detected object with Depth, Width, Height (feet).
    """

    try:
        if class_image_json is None:
            raise ValueError("class_image_json is required")

        # Load mapping
        jm_path = Path(class_image_json)
        if not jm_path.exists():
            raise FileNotFoundError(f"class_image_json not found: {class_image_json}")
        with open(jm_path, "r") as f:
            jm_raw = json.load(f)

        class_map = {}
        for k, v in jm_raw.items():
            key = str(k).lower()
            if isinstance(v, dict):
                p = v.get("path")
                w = float(v.get("width")) if v.get("width") is not None else None
                h = float(v.get("height")) if v.get("height") is not None else None
                rotate_flag = bool(v.get("rotate", False))
                class_map[key] = {"path": str(p), "width": w, "height": h, "rotate": rotate_flag}
            elif isinstance(v, str):
                class_map[key] = {"path": str(v), "width": None, "height": None, "rotate": False}

        # Helper: cv2 load + clean
        def _load_and_clean_rgba_cv2(pth, white_thresh=white_threshold, remove_white=remove_white_bg):
            arr = cv2.imread(str(pth), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise RuntimeError(f"cv2 failed to load image: {pth}")
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGRA)
            if arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)

            # remove near-white → transparent
            r, g, b, a = arr[:, :, 2], arr[:, :, 1], arr[:, :, 0], arr[:, :, 3]
            if remove_white:
                white_mask = (r >= white_thresh) & (g >= white_thresh) & (b >= white_thresh)
                a[white_mask] = 0
                arr[:, :, 3] = a

            # crop transparent border
            coords = np.argwhere(a > 0)
            if coords.size == 0:
                return np.zeros((1, 1, 4), dtype=np.uint8)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            cropped = arr[y0:y1, x0:x1].copy()

            return cv2.cvtColor(cropped, cv2.COLOR_BGRA2RGBA)

        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        fig, ax = plt.subplots(figsize=(12, 10))

        # spaces
        for sp in spaces:
            try:
                poly = np.asarray(sp["poly"], dtype=float)
                patch = MplPolygon(poly, closed=True, fill=True, alpha=0.25, edgecolor="black")
                ax.add_patch(patch)

                # optional: also show space names
                if "name" in sp and sp["name"]:
                    cx, cy = poly.mean(axis=0)
                    ax.text(cx, cy, str(sp["name"]), fontsize=9, ha="center", va="center", color="black")
            except Exception:
                continue

        # fixed furniture with names
        for ff in (fixed_furniture or []):
            try:
                fpoly = np.asarray(ff["poly"], dtype=float)
                patch = MplPolygon(fpoly, closed=True, fill=True, facecolor="none",
                                   edgecolor="saddlebrown", linewidth=1.0, hatch="////", zorder=22)
                ax.add_patch(patch)

                # add furniture name in the center if available
                if "name" in ff and ff["name"]:
                    cx, cy = fpoly.mean(axis=0)
                    ax.text(cx, cy, str(ff["name"]), fontsize=8, ha="center", va="center",
                            color="saddlebrown", fontweight="bold")
            except Exception as e:
                if VERBOSE:
                    print("plot_floorplan fixed furniture error:", e)
                continue

        # px per unit
        fig.canvas.draw()
        def _px_per_unit():
            cx = (floor_min[0] + floor_max[0]) / 2.0
            cy = (floor_min[1] + floor_max[1]) / 2.0
            p1 = ax.transData.transform((cx, cy))
            p2 = ax.transData.transform((cx + 1.0, cy))
            return np.hypot(p2[0] - p1[0], p2[1] - p1[1])
        px_per_unit = _px_per_unit()

        placed_any = False

        # iterate objects
        for cls_l, info in sorted((found_first or {}).items()):
            try:
                mapping = class_map.get(str(cls_l).lower())
                if not mapping:
                    # still show text for detections even if no icon mapping is present
                    mapping = None
                else:
                    img_path = Path(mapping["path"])
                    if not img_path.exists() or img_path.suffix.lower() != ".png":
                        mapping = None

                objx, objy = info.get("object_x"), info.get("object_y")
                camx, camy = info.get("mapped_x"), info.get("mapped_y")
                if objx is None or objy is None:
                    continue

                # place icon if mapping exists
                if mapping is not None:
                    try:
                        rgba = _load_and_clean_rgba_cv2(Path(mapping["path"]))
                        src_h, src_w = rgba.shape[:2]

                        # resize/stretch
                        desired_w, desired_h = mapping["width"], mapping["height"]
                        tgt_w, tgt_h = src_w, src_h
                        if desired_w and desired_h:
                            tgt_w = max(1, int(round(desired_w * px_per_unit)))
                            tgt_h = max(1, int(round(desired_h * px_per_unit)))
                        elif desired_w:
                            tgt_w = max(1, int(round(desired_w * px_per_unit)))
                            scale = tgt_w / src_w
                            tgt_h = max(1, int(round(src_h * scale)))
                        elif desired_h:
                            tgt_h = max(1, int(round(desired_h * px_per_unit)))
                            scale = tgt_h / src_h
                            tgt_w = max(1, int(round(src_w * scale)))

                        bgra = rgba[:, :, [2, 1, 0, 3]]
                        resized_bgra = cv2.resize(bgra, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
                        final_img = resized_bgra[:, :, [2, 1, 0, 3]]

                        # rotation
                        if rotate_icons and mapping.get("rotate", False):
                            yaw = info.get("object_yaw_deg")
                            if yaw is not None:
                                (h, w) = final_img.shape[:2]
                                bgra2 = final_img[:, :, [2, 1, 0, 3]]
                                M = cv2.getRotationMatrix2D((w/2, h/2), -float(yaw), 1.0)
                                warped = cv2.warpAffine(bgra2, M, (w, h), flags=cv2.INTER_CUBIC,
                                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
                                final_img = warped[:, :, [2, 1, 0, 3]]

                        ab = AnnotationBbox(OffsetImage(final_img, zoom=1.0),
                                            (float(objx), float(objy)), frameon=False,
                                            box_alignment=(0.5, 0.5), pad=0.0, zorder=100)
                        ax.add_artist(ab)
                        placed_any = True
                    except Exception as e:
                        if VERBOSE:
                            print("plot_floorplan icon placement error:", e)

                # --- camera + arrow if requested ---
                if show_arrows and camx is not None and camy is not None:
                    ax.scatter([camx], [camy], s=120, marker="x", c="red", zorder=20)
                    ax.text(camx + 4, camy + 4, f"Cam vf{info.get('video_frame_index', '?')}", fontsize=8)
                    dx, dy = objx - camx, objy - camy
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        dx += 1e-3
                    ax.arrow(camx, camy, dx, dy, head_width=6, length_includes_head=True,
                             fc="k", ec="k", zorder=18)

                # --- annotate size & depth near object (in feet) ---
                dist_ft = info.get("distance_ft")
                # prefer sizes already saved in feet; if stored in meters, convert
                w_m = info.get("real_width_m")
                h_m = info.get("real_height_m")
                w_ft = None; h_ft = None
                if w_m is not None:
                    try:
                        w_ft = float(w_m) * float(M_TO_FT)
                    except Exception:
                        w_ft = None
                if h_m is not None:
                    try:
                        h_ft = float(h_m) * float(M_TO_FT)
                    except Exception:
                        h_ft = None

                # Some pipelines might have saved widths in feet already (rare); if distance_ft missing, fallback to converting distance_m
                if dist_ft is None and info.get("distance_m") is not None:
                    try:
                        dist_ft = float(info.get("distance_m")) * float(M_TO_FT)
                    except Exception:
                        dist_ft = None

                # build text lines
                lines = []
                if dist_ft is not None:
                    lines.append(f"D:{dist_ft:.1f}ft")
                else:
                    lines.append("D:N/A")
                if w_ft is not None:
                    lines.append(f"W:{w_ft:.1f}ft")
                else:
                    lines.append("W:N/A")
                if h_ft is not None:
                    lines.append(f"H:{h_ft:.1f}ft")
                else:
                    lines.append("H:N/A")

                txt = "  ".join(lines)

                # place annotation a bit above the icon / object point
                text_offset_x = 6.0
                text_offset_y = -6.0
                ax.text(objx + text_offset_x, objy + text_offset_y, txt,
                        fontsize=8, ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
                        zorder=200)

            except Exception as e:
                if VERBOSE:
                    print("plot_floorplan error:", e)
                continue

        # finalize
        pad_x = max(1.0, abs(floor_max[0]-floor_min[0]))*0.05
        pad_y = max(1.0, abs(floor_max[1]-floor_min[1]))*0.05
        ax.set_xlim(float(floor_min[0]) - pad_x, float(floor_max[0]) + pad_x)
        ax.set_ylim(float(floor_min[1]) - pad_y, float(floor_max[1]) + pad_y)
        ax.set_aspect("equal", adjustable="box")
        plt.gca().invert_yaxis()
        plt.title(title or "Floor plan — PNG")
        plt.tight_layout()

        if out_path:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_path), dpi=200)
            print("Saved overlay image:", out_path)
        plt.close(fig)

    except Exception as e:
        print("Warning: plot_floorplan failed:", e)
