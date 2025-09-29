#!/usr/bin/env python3
"""
Entry point that wires everything together. Behavior preserved from the original monolithic script.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from config.config import *
from config.logger import setup_logger
from utils.json_utils import load_json, find_dicts_with_key
from utils.matrix_utils import parse_homogeneous_matrix, umeyama_2d, apply_similarity_to_points, interp_missing
from utils.projection_utils import project_3d_to_2d, auto_map_and_choose, count_points_inside_polygons
from floor.floor_loader import load_floor_polygons, load_fixed_furniture
from video.video_processing import process_video_first_per_class, build_arkit_frame_index_map
from detectors.yolo_detector import YoloDetector
from detectors.zoe_depth import init_zoe, ZoeDepthForDepthEstimation
from floor.plotter import plot_floorplan

logger = setup_logger()

def extract_positions_list(arkit_path):
    ark = load_json(arkit_path)
    cand_dicts = find_dicts_with_key(ark, "cameraTransform")
    if not cand_dicts:
        cand_dicts = find_dicts_with_key(ark, "transform") + find_dicts_with_key(ark, "matrix") + find_dicts_with_key(ark, "camera_transform")
    entries = []
    for d in cand_dicts:
        rawmat = d.get("cameraTransform") or d.get("transform") or d.get("matrix") or None
        if rawmat is None:
            for k, v in d.items():
                if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                    if len(v) in (16, 12):
                        rawmat = v
                        break
        if rawmat is None:
            continue
        pos, rot = parse_homogeneous_matrix(rawmat)
        if pos is None:
            continue
        entries.append({"pos": pos, "rot": rot, "method": "parsed", "raw": d,
                        "frameNumber": d.get("frameNumber"), "frameTimestamp": d.get("frameTimestamp")})
    if not entries:
        def find_xy(obj):
            out = []
            if isinstance(obj, dict):
                if 'x' in obj and 'y' in obj:
                    try:
                        out.append({"pos": np.array([float(obj['x']), 0.0, float(obj['y'])])})
                    except: pass
                for v in obj.values(): out += find_xy(v)
            elif isinstance(obj, list):
                for it in obj: out += find_xy(it)
            return out
        ark_xy = find_xy(ark)
        if ark_xy:
            entries = [{"pos": it["pos"], "rot": None, "method": "xy_direct", "raw": None, "frameNumber": None, "frameTimestamp": None} for it in ark_xy]

    if not entries:
        raise RuntimeError("No camera transforms or x/y data found in arkitData.json")

    positions = np.vstack([e["pos"] for e in entries])
    meta = [{"frameNumber": e.get("frameNumber"), "frameTimestamp": e.get("frameTimestamp"),
             "method": e.get("method"), "raw": e.get("raw"), "rot": e.get("rot")} for e in entries]
    return positions, meta

def compute_global_orientation_votes(positions3d, meta):
    votes = []; confidences = []
    for i in range(len(positions3d)):
        cam_rot = None
        try:
            if isinstance(meta[i], dict):
                cam_rot = meta[i].get("rot", None)
            if cam_rot is None:
                raw = meta[i].get("raw") if isinstance(meta[i].get("raw"), dict) else None
                if raw is not None:
                    rawmat = raw.get("cameraTransform") or raw.get("transform") or raw.get("matrix") or None
                    if rawmat is None:
                        for k, v in raw.items():
                            if isinstance(v, list) and len(v) in (12, 16):
                                rawmat = v; break
                    if rawmat is not None:
                        p, R = parse_homogeneous_matrix(rawmat)
                        if R is not None:
                            cam_rot = R
            if cam_rot is not None:
                cam_rot = np.array(cam_rot, dtype=float)
                if cam_rot.shape != (3, 3):
                    if cam_rot.T.shape == (3, 3):
                        cam_rot = cam_rot.T
                    else:
                        cam_rot = None
        except Exception:
            cam_rot = None

        if cam_rot is None: continue
        neighbor_idx = None
        if i+1 < len(positions3d): neighbor_idx = i+1
        elif i-1 >= 0: neighbor_idx = i-1
        if neighbor_idx is None: continue
        from utils.orientation_utils import choose_best_rotation_and_sign
        Ruse, zsign, score = choose_best_rotation_and_sign(positions3d[i], cam_rot, positions3d[neighbor_idx])
        votes.append((Ruse is cam_rot.T, zsign)); confidences.append(score)

    if votes:
        from collections import Counter
        cnt = Counter(votes)
        most, count = cnt.most_common(1)[0]
        global_use_transpose, global_z_sign = most
        logger.info(f"Global orientation chosen by majority vote: use_transpose={global_use_transpose} global_z_sign={global_z_sign} (votes {len(votes)}); mean_conf={np.mean(confidences):.3f}")
    else:
        global_use_transpose, global_z_sign = None, None
        logger.info("Global orientation vote: insufficient data -> leaving per-frame auto-detect active")
    return global_use_transpose, global_z_sign

def main():
    logger.info("Loading and extracting ARKit positions...")
    positions3d, meta = extract_positions_list(ARKIT_PATH)
    logger.info(f"Extracted {len(positions3d)} frames (3D positions).")

    if CONVERT_M_TO_FT:
        positions3d = positions3d * M_TO_FT
        logger.info("Converted ARKit positions meters -> feet using factor %s", M_TO_FT)

    proj2_all = project_3d_to_2d(positions3d, PROJECTION)
    logger.info("Projected to 2D (projection=%s), sample: %s", PROJECTION, proj2_all[:6].tolist())

    spaces, floor_min, floor_max, fixed_furniture = load_floor_polygons(FLOOR_PATH)
    logger.info("Floor bounds: %s %s", floor_min, floor_max)

    extra_fixed = load_fixed_furniture(FLOOR_PATH)
    if extra_fixed:
        if isinstance(fixed_furniture, list):
            fixed_furniture.extend(extra_fixed)
        else:
            fixed_furniture = extra_fixed
    logger.info("Loaded %d fixed furniture items", len(fixed_furniture) if fixed_furniture is not None else 0)

    s_map, R_map, t_map = 1.0, np.eye(2), np.array([0.0, 0.0])
    mapped_all = interp_missing(proj2_all)
    chosen_method = "none"; rotation_used = 0.0; chosen_score = 0; chosen_proj = PROJECTION

    if CONTROL_POINTS and len(CONTROL_POINTS) >= 2:
        logger.info("Using CONTROL_POINTS for exact alignment (Umeyama similarity).")
        src = []; dst = []
        for (fi, fx, fy) in CONTROL_POINTS:
            if fi < 0 or fi >= len(proj2_all):
                raise ValueError(f"Control point frame index {fi} out of range (0..{len(proj2_all)-1})")
            src.append(proj2_all[fi]); dst.append([fx, fy])
        src = np.vstack(src); dst = np.vstack(dst)
        s_map, R_map, t_map = umeyama_2d(src, dst, with_scaling=True)
        mapped_all = apply_similarity_to_points(proj2_all, s_map, R_map, t_map)
        chosen_method = f"umeyama_{len(CONTROL_POINTS)}pts"
        rotation_used = 0.0
        chosen_score = count_points_inside_polygons(mapped_all, spaces)
        chosen_proj = PROJECTION
    else:
        logger.info("No control points provided — using automatic mapping heuristic.")
        floor_json = load_json(FLOOR_PATH)
        compass = floor_json.get("compassHeading", None)
        proj_options = ["x,-z", "x,z", "-x,-z", "-x,z", "y,-z"]
        best_score = -1; best_choice = None
        for proj in proj_options:
            p2 = project_3d_to_2d(positions3d, proj)
            try:
                mapped_candidate, scale, rot_deg, score = auto_map_and_choose(p2, spaces, floor_min, floor_max, compass=compass)
            except Exception as e:
                logger.warning("proj %s failed: %s", proj, e)
                continue
            logger.info("proj %s -> score %d (rot %.1f, scale approx %.2f)", proj, score, rot_deg, scale)
            if score > best_score:
                best_score = score
                best_choice = {"proj": proj, "mapped": mapped_candidate, "scale": scale, "rot_deg": rot_deg, "p2": p2}

        if best_choice is None:
            logger.warning("Automatic mapping failed for all projections — using center-fit fallback.")
            proj2_all = project_3d_to_2d(positions3d, PROJECTION)
            mapped_all = (proj2_all - proj2_all.mean(axis=0)) * 10.0 + (floor_min + floor_max) / 2.0
            s_map, R_map, t_map = 1.0, np.eye(2), np.array([0.0, 0.0])
            chosen_method = "fallback_center"; rotation_used = 0.0; chosen_score = 0; chosen_proj = PROJECTION
        else:
            proj2_all = best_choice["p2"]
            mapped_all = best_choice["mapped"]
            chosen_method = f"auto_proj_{best_choice['proj']}"
            rotation_used = best_choice["rot_deg"]
            chosen_score = best_score
            chosen_proj = best_choice['proj']
            logger.info("Auto-chosen projection: %s rotation: %.1f score: %d", chosen_proj, rotation_used, chosen_score)
            try:
                s_map, R_map, t_map = umeyama_2d(proj2_all, mapped_all, with_scaling=True)
            except Exception as e:
                logger.warning("Failed to compute umeyama similarity; falling back. Error: %s", e)
                s_map, R_map, t_map = 1.0, np.eye(2), np.array([0.0, 0.0])

    mapped_plot = interp_missing(mapped_all)
    logger.info("mapped_plot sample: %s", mapped_plot[:5].tolist())

    global_use_transpose, global_z_sign = compute_global_orientation_votes(positions3d, meta)

    logger.info("Initializing YOLO detector (this may load weights and take a moment)...")
    detector = None
    try:
        detector = YoloDetector(model_path=YOLO_MODEL, device=YOLO_DEVICE, conf=YOLO_CONF, iou=YOLO_IOU)
    except Exception as e:
        logger.warning("YOLO detector not available or failed to initialize: %s", e)
        logger.warning("If you want detection, install ultralytics and provide YOLO_MODEL weights at %s", YOLO_MODEL)
        return {
            "positions3d": positions3d,
            "meta": meta,
            "proj2_all": proj2_all,
            "mapped_plot": mapped_plot,
            "spaces": spaces,
            "floor_min": floor_min,
            "floor_max": floor_max,
            "fixed_furniture": fixed_furniture,
            "found_first": {},
            "mapping": {"method": chosen_method, "proj": chosen_proj, "rot_deg": rotation_used, "score": chosen_score}
        }

    try:
        if ZoeDepthForDepthEstimation is not None:
            init_zoe(ZOE_MODEL_NAME)
    except Exception as e:
        logger.warning("Zoe init warning: %s", e)

    found_first = {}
    if VIDEO_PATH.exists():
        found_first = process_video_first_per_class(VIDEO_PATH, detector, meta, mapped_plot, positions3d,
                                                    s_map, R_map, t_map, chosen_proj,
                                                    TARGET_CLASSES, spaces=spaces, save_detected=True,
                                                    debug_reproject=DEBUG_REPROJECT, reproject_csv=OUT_REPRO_CSV,
                                                    global_use_transpose=global_use_transpose, global_z_sign=global_z_sign,
                                                    room_margin=0.5, verbose=True)
        logger.info("First-seen classes collected: %d", len(found_first))
    else:
        logger.warning("Video not found at %s -> skipping detection", VIDEO_PATH)

    if found_first:
        rows = []
        for cls_l, info in found_first.items():
            rows.append({
                "class": info['class_name'],
                "video_frame_index": info['video_frame_index'],
                "arkit_index": info['arkit_index'],
                "cam_mapped_x": info['mapped_x'],
                "cam_mapped_y": info['mapped_y'],
                "object_mapped_x": info.get('object_x'),
                "object_mapped_y": info.get('object_y'),
                "object_yaw_deg": info.get('object_yaw_deg'),
                "conf": info['conf'],
                "crop_path": info['crop_path'],
                "annotated_frame": info['annotated_frame'],
                "depth_model_units": info.get("depth_model_units"),
                "distance_m": info.get('distance_m'),
                "distance_ft": info.get('distance_ft'),
            })
        df = pd.DataFrame(rows)
        OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUT_CSV, index=False)
        logger.info("Saved detections CSV: %s", OUT_CSV)
    else:
        logger.info("No first-seen detections to save to CSV.")

    return {
        "positions3d": positions3d,
        "meta": meta,
        "proj2_all": proj2_all,
        "mapped_plot": mapped_plot,
        "spaces": spaces,
        "floor_min": floor_min,
        "floor_max": floor_max,
        "fixed_furniture": fixed_furniture,
        "found_first": found_first,
        "mapping": {"method": chosen_method, "proj": chosen_proj, "rot_deg": rotation_used, "score": chosen_score}
    }

if __name__ == "__main__":
    result = main()
    if result is None:
        print("Main returned None — exiting.")
        sys.exit(0)

    mapped_plot = result.get("mapped_plot")
    spaces = result.get("spaces")
    fixed_furniture = result.get("fixed_furniture", [])
    floor_min = result.get("floor_min")
    floor_max = result.get("floor_max")
    found_first = result.get("found_first", {})

    if spaces is None or floor_min is None or floor_max is None:
        print("Missing floor polygons or bounds — cannot plot.")
        sys.exit(0)

    CLASS_IMAGE_JSON = CLASS_IMAGE_JSON  # from config

    def _validate_class_images(json_path):
        try:
            p = Path(json_path)
            if not p.exists():
                print(f"PNG mapping JSON not found: {json_path}")
                return False
            jm = load_json(json_path)
            if not isinstance(jm, dict):
                print("PNG mapping JSON should be an object/dictionary of class->path or class->dict")
                return False
            missing = []; not_png = []; invalid = []
            for k, v in jm.items():
                if isinstance(v, str):
                    invalid.append(f"{k}: mapping is string-only; width/height required")
                    continue
                if isinstance(v, dict):
                    pval = v.get("path") or v.get("png") or v.get("file")
                    if not pval:
                        invalid.append(f"{k}: missing 'path' in dict")
                        continue
                    pth = Path(pval)
                    if not pth.exists():
                        missing.append(str(pth))
                    elif pth.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        not_png.append(str(pth))
                    if v.get("width") is None and v.get("height") is None:
                        invalid.append(f"{k}: missing width/height in mapping (recommended)")
                else:
                    invalid.append(f"{k}: unsupported mapping type {type(v)}")
            if invalid:
                print("PNG mapping has invalid entries:", invalid)
            if missing:
                print("PNG mapping contains missing files:", missing)
            if not_png:
                print("PNG mapping contains non-image files:", not_png)
            return (len(missing) == 0 and len(not_png) == 0)
        except Exception as e:
            print("Validator error:", e)
            return False

    if not _validate_class_images(CLASS_IMAGE_JSON):
        print("Image mapping validation failed. Fix the JSON or image files and rerun.")
    else:
        rej_csv = Path("input_data") / "rejected_detections_room_mismatch.csv"
        rejected_list = None
        if rej_csv.exists():
            try:
                rejected_list = pd.read_csv(str(rej_csv)).to_dict(orient='records')
                print(f"Loaded {len(rejected_list)} rejected detections for visualization.")
            except Exception as e:
                print("Warning: failed to load rejected CSV for plotting:", e)
                rejected_list = None

        try:
            plot_floorplan(found_first, spaces, fixed_furniture, floor_min, floor_max, mapped_plot,
                           out_path=Path("output_data") / "floor_plan_with_pngs.png",
                           title=f"Floor plan — PNG-only — method {result.get('mapping',{}).get('method','?')}",
                           class_image_json=str(CLASS_IMAGE_JSON),
                           rotate_icons=True,
                           remove_white_bg=True,
                           white_threshold=245,
                           rejected_list=rejected_list,
                           meta=result.get("meta"))
        except Exception as e:
            print("Warning: plotting failed:", e)

    print("Done.")
