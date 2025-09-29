"""
ZoeDepth helpers (init & depth extraction). Mirrors behavior from the big script.
"""

from pathlib import Path
import cv2
import numpy as np

try:
    from transformers import ZoeDepthForDepthEstimation, AutoImageProcessor
    import torch
    from PIL import Image
    _zoe_processor = None; _zoe_model = None
    _zoe_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    ZoeDepthForDepthEstimation = None; AutoImageProcessor = None; torch = None; Image = None
    _zoe_processor = None; _zoe_model = None
    print("ZoeDepth imports failed:", e)

def init_zoe(model_name="Intel/zoedepth-nyu-kitti", device=None):
    global _zoe_processor, _zoe_model, _zoe_device
    if ZoeDepthForDepthEstimation is None:
        print("ZoeDepth classes not available; cannot init Zoe.")
        return False
    device = device or _zoe_device
    try:
        _zoe_processor = AutoImageProcessor.from_pretrained(model_name)
        _zoe_model = ZoeDepthForDepthEstimation.from_pretrained(model_name).to(device)
        _zoe_device = device
        print("Zoe model loaded:", model_name, "on", _zoe_device)
        return True
    except Exception as e:
        print("Failed to init Zoe:", e)
        _zoe_processor = None; _zoe_model = None
        return False

def get_zoe_depth_map(image_rgb):
    global _zoe_processor, _zoe_model, _zoe_device, torch
    if _zoe_processor is None or _zoe_model is None:
        raise RuntimeError("Zoe not initialized. Call init_zoe()")
    pil = Image.fromarray(image_rgb)
    inputs = _zoe_processor(images=pil, return_tensors="pt").to(_zoe_device)
    with torch.no_grad():
        outputs = _zoe_model(**inputs)
    depth_map = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
    if depth_map.shape != image_rgb.shape[:2]:
        import cv2
        depth_map = cv2.resize(depth_map, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_map
