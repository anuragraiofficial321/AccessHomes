"""
Floor polygon loader functions (load_floor_polygons, load_fixed_furniture)
"""

import numpy as np
from pathlib import Path
from utils.json_utils import load_json

def load_floor_polygons(floor_path):
    floor = load_json(floor_path)
    spaces = []
    fixed_furniture = []
    floors_list = floor.get('floors', []) or []
    if not floors_list:
        raise RuntimeError("No floors found in floor_plan.json")
    floor0 = floors_list[0]
    for s in floor0.get('spaces', []):
        bp = s.get('boundaryPolygon') or {}
        if bp:
            coords = bp.get('coordinates')
            if coords:
                poly = coords[0] if isinstance(coords[0][0], list) else coords
                poly_arr = np.array(poly, dtype=float)
                spaces.append({"id": s.get('id'), "name": s.get('name') or s.get('class') or s.get('id'), "poly": poly_arr, **({"doors": s.get("doors")} if s.get("doors") else {})})
        ff = s.get('fixedFurniture', []) or []
        for f in ff:
            b = f.get('boundingPolygon') or {}
            if b:
                coords_f = b.get('coordinates')
                if coords_f:
                    polyf = coords_f[0] if isinstance(coords_f[0][0], list) else coords_f
                    try:
                        polyf_arr = np.array(polyf, dtype=float)
                        fixed_furniture.append({"class": f.get('class') or f.get('type') or "furniture", "poly": polyf_arr, "space_id": s.get('id')})
                    except Exception:
                        continue
    top_ff = floor.get('fixedFurniture', []) or []
    for f in top_ff:
        b = f.get('boundingPolygon') or {}
        if b:
            coords_f = b.get('coordinates')
            if coords_f:
                polyf = coords_f[0] if isinstance(coords_f[0][0], list) else coords_f
                try:
                    polyf_arr = np.array(polyf, dtype=float)
                    fixed_furniture.append({"class": f.get('class') or f.get('type') or "furniture", "poly": polyf_arr, "space_id": None})
                except Exception:
                    continue
    if not spaces:
        raise RuntimeError("No spaces/polygons found in floor_plan.json")
    all_pts = np.vstack([sp["poly"] for sp in spaces])
    floor_min = all_pts.min(axis=0); floor_max = all_pts.max(axis=0)
    return spaces, floor_min, floor_max, fixed_furniture

def load_fixed_furniture(floor_path):
    floor = load_json(floor_path)
    furn = []
    if 'floors' not in floor or not floor['floors']:
        return furn
    floor0 = floor['floors'][0]
    for s in floor0.get('spaces', []):
        space_name = s.get('name') or s.get('class') or s.get('id')
        ff_list = s.get('fixedFurniture') or s.get('fixed_furniture') or []
        for f in ff_list:
            bp = f.get('boundingPolygon') or f.get('bounding_polygon') or None
            poly = None
            if bp:
                coords = bp.get('coordinates')
                if coords:
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
            furn.append({'class': f.get('class') or f.get('type') or 'fixedFurniture', 'poly': poly, 'space': space_name})
    return furn
