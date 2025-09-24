#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

def _detect_emoji_font():
    """
    Search common emoji font names in matplotlib font manager list.
    Return font name if found, else None.
    """
    candidates = ["Noto Color Emoji", "Segoe UI Emoji", "Apple Color Emoji", "EmojiOne Color", "Twemoji Mozilla", "Symbola"]
    tt = fm.fontManager.ttflist
    available_names = {f.name: f.fname for f in tt}
    for c in candidates:
        # direct name match
        if c in available_names:
            return c
    # fallback: check lowercase substring in file names
    for c in candidates:
        for f in tt:
            if c.lower().replace(" ", "") in Path(f.fname).name.lower().replace(" ", ""):
                return f.name
    return None

def plot_floorplan(found_first, spaces, fixed_furniture, floor_min, floor_max, mapped_plot,
                   use_emojis=True, out_path=None, title=None, VERBOSE=True, TARGET_CLASSES=None):
    """
    Robust plotting of floorplan and first-seen detections.
    - Automatically detects an emoji-capable font on the system. If not found,
      emojis are disabled (avoids glyph missing warnings).
    - Draws arrows from camera mapped point -> object mapped point.
    """
    try:
        # decide emoji support
        emoji_font = None
        if use_emojis:
            emoji_font = _detect_emoji_font()
            if emoji_font:
                try:
                    plt.rcParams['font.family'] = emoji_font
                    if VERBOSE:
                        print(f"plot_floorplan: using emoji-capable font '{emoji_font}'")
                except Exception:
                    if VERBOSE:
                        print("plot_floorplan: failed to set emoji font — falling back to plain labels")
                    use_emojis = False
            else:
                if VERBOSE:
                    print("plot_floorplan: no emoji font found on system — falling back to plain labels")
                use_emojis = False

        fig, ax = plt.subplots(figsize=(12, 10))

        # draw spaces
        for sp in spaces:
            try:
                poly = np.asarray(sp["poly"], dtype=float)
                patch = MplPolygon(poly, closed=True, fill=True, alpha=0.25, edgecolor='black')
                ax.add_patch(patch)
                c = poly.mean(axis=0)
                ax.text(c[0], c[1], str(sp.get("name", "")), fontsize=9, ha='center', va='center')
            except Exception:
                continue

        # draw fixed furniture
        for ff in (fixed_furniture or []):
            try:
                fpoly = np.asarray(ff['poly'], dtype=float)
                patch = MplPolygon(fpoly, closed=True, fill=True, facecolor='none',
                                   edgecolor='saddlebrown', linewidth=1.2, hatch='////', zorder=22)
                ax.add_patch(patch)
                cent = fpoly.mean(axis=0)
                ax.text(cent[0], cent[1], ff.get('class', ''), fontsize=7, ha='center', va='center',
                        zorder=23, color='saddlebrown', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
            except Exception:
                continue

        # marker / emoji maps
        class_to_marker = {"couch": "s", "person": "o", "table": "P", "sink": "X", "chair": "D", "bed": "^", "tv": "*"}
        class_to_emoji = {"couch": "🛋️", "person": "👤", "table": "🪑", "sink": "🚰", "chair": "💺", "bed": "🛏️", "tv": "📺"}

        markers = []
        # Sort keys so output is deterministic
        for cls_l, info in sorted((found_first or {}).items()):
            try:
                cls = info.get('class_name', cls_l)
                # camera mapped location (should always exist)
                cam_mx = info.get('mapped_x')
                cam_my = info.get('mapped_y')
                if cam_mx is None or cam_my is None:
                    # skip entries without camera mapped coords
                    if VERBOSE:
                        print(f"plot_floorplan: skipping {cls} because camera mapped coords missing")
                    continue

                ax.scatter([cam_mx], [cam_my], s=120, marker='x', zorder=20)
                ax.text(cam_mx + 4, cam_my + 4, f"Cam vf{info.get('video_frame_index', '?')}", fontsize=8)

                objx = info.get('object_x'); objy = info.get('object_y')
                if objx is None or objy is None:
                    # we still plotted camera point; show that object position is unknown
                    if VERBOSE:
                        print(f"plot_floorplan: object world not computed for {cls} (no arrow drawn)")
                    continue

                # distance label if available
                dist_label = ""
                if info.get("distance_ft") is not None:
                    try:
                        dist_label = f" ({info['distance_ft']:.1f}ft)"
                    except Exception:
                        dist_label = ""

                marker = class_to_marker.get(cls_l, 'o')
                ax.scatter([objx], [objy], s=220, marker=marker, zorder=25)

                # choose emoji or plain text
                emoji = class_to_emoji.get(cls_l, "") if use_emojis else ""
                label_text = f"{emoji} {cls}{dist_label}" if use_emojis else f"{cls}{dist_label}"

                # draw label
                ax.text(objx + 6, objy + 6, label_text, fontsize=9, fontweight='bold')

                # arrow from camera to object (ensure visible even if identical coords)
                dx = objx - cam_mx; dy = objy - cam_my
                # small jitter if zero to ensure arrow draws
                if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    dx += 1e-3
                ax.arrow(cam_mx, cam_my, dx, dy, head_width=6, length_includes_head=True, fc='k', ec='k', zorder=18)

                # legend entry (use a Patch with marker-like text)
                legend_label = f"{cls} {emoji}" if use_emojis else cls
                markers.append(Patch(label=legend_label))
            except Exception as e:
                if VERBOSE:
                    print("plot_floorplan: exception while plotting a found_first entry:", e)
                continue

        if markers:
            try:
                ax.legend(handles=markers, loc='lower right', fontsize=9)
            except Exception:
                pass

        ax.set_xlim(floor_min[0] - 10, floor_max[0] + 10)
        ax.set_ylim(floor_min[1] - 10, floor_max[1] + 10)
        ax.set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        if title is None:
            title = f"Floor plan — first-seen (found {len(found_first or {})} of {len(TARGET_CLASSES) if TARGET_CLASSES else '?'})"
        plt.title(title)
        plt.tight_layout()

        if out_path is not None:
            try:
                out_path = Path(out_path)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(out_path), dpi=200)
                print("Saved overlay image:", out_path)
            except Exception as e:
                print("Warning: failed to save plot image:", e)
        plt.close(fig)
    except Exception as e:
        print("Warning: plot_floorplan failed:", e)
