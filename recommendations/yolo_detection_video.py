#!/usr/bin/env python3
"""
yolo_video_to_csv.py

Usage:
    python yolo_video_to_csv.py

Edit the VIDEO_PATH variable below (or pass via CLI) to point to your static video file.
Outputs:
 - output/detections.csv        : detailed detection rows
 - output/annotated_frames/     : optional annotated frames (jpg)
 - output/summary.csv           : per-frame summary (num_detections, top_confidence)
"""

import os
import csv
import math
from pathlib import Path
from datetime import timedelta

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO  # ultralytics package (yolov8)

# ---------------------------
# USER SETTINGS (change these)
# ---------------------------
VIDEO_PATH = "/home/anuragrai/Desktop/AccessHomes/recommendations/video.mp4"    # <-- change to your static path
MODEL = "models/yolo11m.pt"                   # smallest YOLOv8; change to yolov8s.pt/yolov8m.pt or custom model path
OUTPUT_DIR = "recommendations_output"
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, "annotated_frames")
SAVE_ANNOTATED = True                  # set False to skip saving annotated frames (faster)
PROCESS_EVERY_NTH_FRAME = 1            # 1 = every frame, 30 = every 30th frame
CONF_THRESHOLD = 0.50                 # skip detections below this confidence
# ---------------------------

def ensure_dirs():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if SAVE_ANNOTATED:
        Path(ANNOTATED_DIR).mkdir(parents=True, exist_ok=True)

def frame_timestamp(frame_idx, fps):
    """
    Return timestamp (seconds and formatted string) given frame index and fps.
    We compute seconds as frame_idx / fps (floating).
    """
    seconds = frame_idx / fps
    # Format as H:MM:SS.mmm
    td = timedelta(seconds=seconds)
    hms = str(td)
    return seconds, hms

def run_video_detection(video_path, model_path, out_dir,
                        save_annotated=True,
                        every_nth=1,
                        conf_thresh=0.25):
    ensure_dirs()

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 30.0  # fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video opened. FPS: {fps:.2f}, Total frames: {total_frames}, Resolution: {width}x{height}")

    # Prepare CSV write
    detections_columns = [
        "frame_idx", "timestamp_s", "timestamp_hms",
        "class_id", "class_name", "confidence",
        "x_min", "y_min", "x_max", "y_max",
        "bbox_width", "bbox_height", "area",
        "frame_width", "frame_height", "video_path"
    ]
    detections_csv_path = os.path.join(out_dir, "detections.csv")
    summary_csv_path = os.path.join(out_dir, "summary.csv")

    # We'll accumulate rows then save with pandas (safer)
    detection_rows = []
    summary_rows = []

    # Iterate frames
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Frames", unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # process every Nth frame only
        if frame_idx % every_nth == 0:
            seconds, hms = frame_timestamp(frame_idx, fps)

            # Ultralytics model accept PIL/ndarray BGR or RGB, it handles different types.
            # We run inference and get results
            results = model.predict(frame, imgsz=640, conf=conf_thresh, verbose=False)  # returns list of Results

            # results is a list; since we give single frame, results[0] is the object
            r = results[0]

            # r.boxes contains bbox info (xyxy), r.boxes.conf, r.boxes.cls
            boxes = getattr(r, "boxes", None)
            names = model.names  # dict id->name

            num_dets = 0
            top_conf = 0.0

            if boxes is not None and len(boxes) > 0:
                # boxes.xyxy: tensor Nx4 (x1,y1,x2,y2)
                # boxes.conf: N
                # boxes.cls: N
                for i in range(len(boxes)):
                    xyxy = boxes[i].xyxy.tolist()[0] if hasattr(boxes[i].xyxy, "tolist") else boxes[i].xyxy
                    # xyxy may be nested, handle carefully
                    if isinstance(xyxy[0], list) or isinstance(xyxy[0], tuple):
                        xyxy = [v for sub in xyxy for v in sub]

                    x_min, y_min, x_max, y_max = [float(v) for v in xyxy]
                    w_box = x_max - x_min
                    h_box = y_max - y_min
                    area = w_box * h_box
                    conf = float(boxes[i].conf.tolist()[0]) if hasattr(boxes[i].conf, "tolist") else float(boxes[i].conf)
                    cls_id = int(boxes[i].cls.tolist()[0]) if hasattr(boxes[i].cls, "tolist") else int(boxes[i].cls)

                    # skip low confidence (additional filter)
                    if conf < conf_thresh:
                        continue

                    class_name = names.get(cls_id, str(cls_id))
                    detection_rows.append({
                        "frame_idx": frame_idx,
                        "timestamp_s": seconds,
                        "timestamp_hms": hms,
                        "class_id": cls_id,
                        "class_name": class_name,
                        "confidence": conf,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "bbox_width": w_box,
                        "bbox_height": h_box,
                        "area": area,
                        "frame_width": width,
                        "frame_height": height,
                        "video_path": str(video_path)
                    })
                    num_dets += 1
                    if conf > top_conf:
                        top_conf = conf

            # Save annotated frame if requested
            if save_annotated:
                # draw boxes on frame copy
                annotated = frame.copy()
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        # same extraction
                        xyxy = boxes[i].xyxy.tolist()[0] if hasattr(boxes[i].xyxy, "tolist") else boxes[i].xyxy
                        if isinstance(xyxy[0], list) or isinstance(xyxy[0], tuple):
                            xyxy = [v for sub in xyxy for v in sub]
                        x_min, y_min, x_max, y_max = [int(float(v)) for v in xyxy]

                        conf = float(boxes[i].conf.tolist()[0]) if hasattr(boxes[i].conf, "tolist") else float(boxes[i].conf)
                        cls_id = int(boxes[i].cls.tolist()[0]) if hasattr(boxes[i].cls, "tolist") else int(boxes[i].cls)
                        if conf < conf_thresh:
                            continue
                        label = f"{model.names.get(cls_id, cls_id)} {conf:.2f}"
                        # draw rectangle and put text
                        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        # text background
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated, (x_min, y_min - th - 6), (x_min + tw, y_min), (0, 255, 0), -1)
                        cv2.putText(annotated, label, (x_min, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                annotated_path = os.path.join(ANNOTATED_DIR, f"frame_{frame_idx:06d}.jpg")
                cv2.imwrite(annotated_path, annotated)

            summary_rows.append({
                "frame_idx": frame_idx,
                "timestamp_s": seconds,
                "timestamp_hms": hms,
                "num_detections": len([r for r in detection_rows if r["frame_idx"] == frame_idx]),
                "top_confidence": top_conf
            })

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Save detections.csv
    if len(detection_rows) == 0:
        print("No detections saved (maybe thresholds too high). Writing empty CSV headers.")
        pd.DataFrame(columns=detections_columns).to_csv(detections_csv_path, index=False)
    else:
        df = pd.DataFrame(detection_rows)
        # reorder cols
        df = df[detections_columns]
        df.to_csv(detections_csv_path, index=False)
        print(f"Wrote detections to {detections_csv_path} (rows: {len(df)})")

    # Save summary
    if len(summary_rows) > 0:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(summary_csv_path, index=False)
        print(f"Wrote summary to {summary_csv_path}")

    print("Done.")

if __name__ == "__main__":
    # quick check that video exists
    if not os.path.exists(VIDEO_PATH):
        raise SystemExit(f"Video not found: {VIDEO_PATH}\nPlease edit VIDEO_PATH in the script or pass a valid path.")
    run_video_detection(
        video_path=VIDEO_PATH,
        model_path=MODEL,
        out_dir=OUTPUT_DIR,
        save_annotated=SAVE_ANNOTATED,
        every_nth=PROCESS_EVERY_NTH_FRAME,
        conf_thresh=CONF_THRESHOLD
    )
