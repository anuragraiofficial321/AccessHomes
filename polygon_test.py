#!/usr/bin/env python3
"""
test_fixed_furniture_plot.py

Standalone test to visualize fixed furniture polygons from /mnt/data/floor_plan.json
and draw per-class PNG icons on top of each fixed furniture item.

Creates:
 - /mnt/data/class_image_sample.json
 - /mnt/data/icons/<class>.png
 - /mnt/data/test_fixed_furniture_output.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import cv2
from PIL import Image, ImageDraw, ImageFont

# --- CONFIG: paths ---
BASE = Path("input_data")
FLOOR_JSON = BASE / "floor_plan.json"          # your uploaded file
ICONS_DIR = BASE / "icons"
CLASS_IMAGE_JSON = "class_image.json"
OUT_PNG = BASE / "test_fixed_furniture_output.png"

# --- Utilities ---
def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --- Extract fixed furniture from floor_plan.json ---
def extract_fixed_furniture(floor_json):
    furn = []
    floors = floor_json.get("floors", []) or []
    if not floors:
        return furn
    floor0 = floors[0]
    for sp in floor0.get("spaces", []):
        ff_list = sp.get("fixedFurniture", []) or []
        space_name = sp.get("name") or sp.get("class") or sp.get("id")
        for f in ff_list:
            bp = f.get("boundingPolygon") or {}
            coords = None
            if bp:
                coords = bp.get("coordinates")
            if not coords:
                continue
            # coords often nested: either coords = [[ [x,y], ... ]] or [ [x,y], ... ]
            poly = None
            try:
                if isinstance(coords[0][0], list):
                    poly = np.array(coords[0], dtype=float)
                else:
                    poly = np.array(coords, dtype=float)
            except Exception:
                try:
                    poly = np.array(coords, dtype=float)
                except Exception:
                    poly = None
            if poly is None:
                continue
            furn.append({
                "class": (f.get("class") or f.get("type") or "furniture"),
                "poly": poly,
                "space": space_name
            })
    return furn

# --- Create simple placeholder PNG icons ---
def create_placeholder_icon(path: Path, text: str, size=(128,128)):
    """Create a colored rectangle with the class name text and transparent background"""
    # create RGBA image
    img = Image.new("RGBA", size, (255,255,255,0))
    draw = ImageDraw.Draw(img)

    # colored rectangle background
    bg = (220, 220, 240, 255)
    rect = (8, 8, size[0]-8, size[1]-8)
    draw.rectangle(rect, fill=bg)

    # pick font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # --- text measurement: Pillow ≥10 uses textbbox ---
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except AttributeError:
        # fallback for old versions of Pillow
        w, h = font.getsize(text)

    # center text
    draw.text(((size[0]-w)/2, (size[1]-h)/2), text, fill=(30,30,30,255), font=font)

    img.save(str(path), format="PNG")


# --- Plot function focused on fixed furniture + icons ---
def plot_fixed_furniture_with_icons(fixed_furniture, floor_json, class_map, out_path: Path):
    # compute floor bounds from boundary polygons (simple aggregate)
    floors = floor_json.get("floors", []) or []
    if not floors:
        raise RuntimeError("No floors found in floor_plan.json")
    floor0 = floors[0]
    spaces = floor0.get("spaces", []) or []
    all_pts = []
    for sp in spaces:
        bp = sp.get("boundaryPolygon") or {}
        coords = bp.get("coordinates") if bp else None
        if not coords:
            continue
        try:
            if isinstance(coords[0][0], list):
                poly = np.array(coords[0], dtype=float)
            else:
                poly = np.array(coords, dtype=float)
            all_pts.append(poly)
        except Exception:
            continue
    if not all_pts:
        raise RuntimeError("No boundary polygons found to compute bounds.")
    all_pts = np.vstack(all_pts)
    floor_min = all_pts.min(axis=0)
    floor_max = all_pts.max(axis=0)

    # setup plot
    fig, ax = plt.subplots(figsize=(10,8))

    # draw room boundaries lightly
    for sp in spaces:
        bp = sp.get("boundaryPolygon") or {}
        coords = bp.get("coordinates") if bp else None
        if not coords:
            continue
        try:
            poly = np.array(coords[0], dtype=float) if isinstance(coords[0][0], list) else np.array(coords, dtype=float)
            patch = MplPolygon(poly, closed=True, fill=True, alpha=0.12, edgecolor="black", linewidth=0.7)
            ax.add_patch(patch)
            cx, cy = poly.mean(axis=0)
            ax.text(cx, cy, (sp.get("name") or sp.get("class") or ""), fontsize=7, ha="center", va="center")
        except Exception:
            continue

    # draw fixed furniture polygons and icons
    for ff in fixed_furniture:
        try:
            poly = np.asarray(ff["poly"], dtype=float)
            patch = MplPolygon(poly, closed=True, fill=False, edgecolor="saddlebrown", linewidth=1.0, hatch="////", zorder=5)
            ax.add_patch(patch)
            cx, cy = float(poly[:,0].mean()), float(poly[:,1].mean())
            ax.text(cx, cy, ff.get("class"), fontsize=7, ha="center", va="center", color="saddlebrown")

            # place icon if mapping exists
            cls_key = str(ff.get("class","")).lower()
            mapping = class_map.get(cls_key)
            if mapping:
                pth = Path(mapping.get("path"))
                if pth.exists():
                    # load RGBA via cv2
                    arr = cv2.imread(str(pth), cv2.IMREAD_UNCHANGED)
                    if arr is not None:
                        # convert B G R A -> R G B A and normalize for matplotlib
                        if arr.shape[2] == 3:
                            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2BGRA)
                        rgba = arr[:, :, [2,1,0,3]].astype(np.float32) / 255.0

                        # choose target size: try mapping width/height (in floor units) else use polygon bbox
                        px_per_unit = _px_per_unit(ax, floor_min, floor_max)
                        desired_w_units = mapping.get("width")
                        desired_h_units = mapping.get("height")
                        if desired_w_units is None or desired_h_units is None:
                            minx, miny = poly.min(axis=0)
                            maxx, maxy = poly.max(axis=0)
                            bbox_w = float(maxx - minx)
                            bbox_h = float(maxy - miny)
                            if desired_w_units is None:
                                desired_w_units = bbox_w
                            if desired_h_units is None:
                                desired_h_units = bbox_h

                        tgt_w = int(max(24, round(desired_w_units * px_per_unit))) if desired_w_units else 48
                        tgt_h = int(max(24, round(desired_h_units * px_per_unit))) if desired_h_units else 48

                        resized = cv2.resize((rgba*255).astype(np.uint8), (tgt_w, tgt_h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                        ab = AnnotationBbox(OffsetImage(resized, zoom=1.0), (cx, cy), frameon=False, pad=0.0, zorder=20)
                        ax.add_artist(ab)
        except Exception as e:
            print("Warning: failed to plot fixed furniture entry:", e)
            continue

    # finalize axes
    pad_x = max(1.0, abs(floor_max[0]-floor_min[0])) * 0.05
    pad_y = max(1.0, abs(floor_max[1]-floor_min[1])) * 0.05
    ax.set_xlim(float(floor_min[0]) - pad_x, float(floor_max[0]) + pad_x)
    ax.set_ylim(float(floor_min[1]) - pad_y, float(floor_max[1]) + pad_y)
    ax.set_aspect("equal", adjustable="box")
    plt.gca().invert_yaxis()
    plt.title("Fixed furniture + icons (test)")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    print("Saved test output to:", out_path)

def _px_per_unit(ax, floor_min, floor_max):
    # helper to estimate pixels per 1 floor-unit
    cx = (floor_min[0] + floor_max[0]) / 2.0
    cy = (floor_min[1] + floor_max[1]) / 2.0
    p1 = ax.transData.transform((cx, cy))
    p2 = ax.transData.transform((cx + 1.0, cy))
    return max(1.0, ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) ** 0.5)

# --- MAIN execution ---
def main():
    if not FLOOR_JSON.exists():
        print("ERROR: floor_plan.json not found at", FLOOR_JSON)
        return

    floor_json = load_json(FLOOR_JSON)
    fixed = extract_fixed_furniture(floor_json)
    print("Found fixed furniture items:", len(fixed))
    for i, f in enumerate(fixed[:5]):
        print(i, f["class"], "poly points:", f["poly"].shape[0])

    # create icons dir and placeholder icons
    ensure_dir(ICONS_DIR)
    # choose unique classes found
    classes = sorted({(f["class"] or "furniture") for f in fixed})
    class_map = {}
    for cls in classes:
        key = str(cls).lower()
        icon_path = ICONS_DIR / f"{key}.png"
        if not icon_path.exists():
            create_placeholder_icon(icon_path, text=cls.replace(" ", "\n"), size=(128,128))
        # sample mapping: no explicit width/height so code will fallback to polygon bbox
        class_map[key] = {"path": str(icon_path), "width": None, "height": None, "rotate": False}

    # write sample class_image JSON (optional)
    with open(CLASS_IMAGE_JSON, "w") as f:
        json.dump(class_map, f, indent=2)
    print("Wrote sample class_image JSON to:", CLASS_IMAGE_JSON)

    # plot
    plot_fixed_furniture_with_icons(fixed, floor_json, class_map, OUT_PNG)

if __name__ == "__main__":
    main()
