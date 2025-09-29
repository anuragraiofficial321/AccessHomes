"""
Configuration file to control paths, devices, and parameters.
"""

from pathlib import Path
import cv2

# --------------------- CONFIG ---------------------
DATA_DIR = Path("input_data")
ARKIT_PATH = DATA_DIR / "arkitData.json"
FLOOR_PATH = DATA_DIR / "floor_plan.json"
VIDEO_PATH = DATA_DIR / "video.mp4"

OUT_DATA_DIR = Path("output_data")
OUT_PNG = OUT_DATA_DIR / "floor_plan_first_detections_only_with_distance.png"
OUT_CSV = OUT_DATA_DIR / "first_seen_detections.csv"
OUT_DEBUG_CSV = OUT_DATA_DIR / "first_seen_detections_debug.csv"
OUT_REPRO_CSV = OUT_DATA_DIR / "reproject_debug.csv"

YOLO_MODEL = "models/yolo11m.pt"
YOLO_DEVICE = "cpu"
YOLO_CONF = 0.90
YOLO_IOU = 0.45

ZOE_MODEL_NAME = "Intel/zoedepth-nyu-kitti"
ZOE_DEVICE = "cuda" if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

TARGET_CLASSES = ["couch", "sink", "chair", "bed", "tv","person","table","toilet","potted plant","dining table","laptop","microwave","oven","refrigerator"]  # edit as needed, or set to [] to keep all classes
CLASS_IMAGE_JSON = "class_image.json" 

PROJECTION = "x,-z"
CONVERT_M_TO_FT = False
M_TO_FT = 3.280839895013123

CONTROL_POINTS = []

DEBUG_REPROJECT = True
SAVE_DETECTED_CROPS = True
VERBOSE = False
