#!/usr/bin/env python3
"""
detectors/zoe_depth.py

Robust ZoeDepth wrapper:
- Initializes processor/model with device handling and use_fast attempt.
- Handles processor returning BatchFeature or unexpected structures.
- Returns depth map resized to input image and as float32.
- Raises clear RuntimeError messages on failure.
- Provides a simple heuristic fallback estimator (estimate_depth_fallback).
"""

from PIL import Image
import cv2
import numpy as np
import os

# Try to import transformers + torch; if unavailable we keep placeholders
try:
    from transformers import ZoeDepthForDepthEstimation, AutoImageProcessor
    import torch
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    ZoeDepthForDepthEstimation = None
    AutoImageProcessor = None
    torch = None
    TRANSFORMERS_AVAILABLE = False
    # don't print loudly here; caller may prefer quiet failure

_zoe_processor = None
_zoe_model = None
_zoe_device = None
_zoe_model_name = None

def _choose_device(preferred="cuda"):
    """Return a torch.device if torch is available, else None."""
    global torch
    if torch is None:
        return None
    try:
        if preferred and preferred.startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")

def init_zoe(model_name="Intel/zoedepth-nyu-kitti", device=None, use_fast=True):
    """
    Initialize Zoe model and processor.
    - model_name: model id or path
    - device: 'cpu', 'cuda', torch.device, or None (auto)
    - use_fast: try to use fast tokenizer/processor option if supported
    Returns True on success, False on failure.
    """
    global _zoe_processor, _zoe_model, _zoe_device, _zoe_model_name, AutoImageProcessor, ZoeDepthForDepthEstimation, torch, TRANSFORMERS_AVAILABLE
    _zoe_model_name = model_name
    if not TRANSFORMERS_AVAILABLE:
        print("ZoeDepth init: transformers/torch not available. Install them to use Zoe.")
        return False

    try:
        # Resolve device
        if device is None:
            _zoe_device = _choose_device(preferred="cuda")
        else:
            if isinstance(device, str):
                if torch is None:
                    _zoe_device = None
                else:
                    _zoe_device = torch.device(device)
            else:
                _zoe_device = device

        # Load processor with use_fast if supported
        try:
            _zoe_processor = AutoImageProcessor.from_pretrained(model_name, use_fast=use_fast)
        except TypeError:
            # older/newer versions may not accept use_fast kwarg
            _zoe_processor = AutoImageProcessor.from_pretrained(model_name)

        _zoe_model = ZoeDepthForDepthEstimation.from_pretrained(model_name).to(_zoe_device) if _zoe_device is not None else ZoeDepthForDepthEstimation.from_pretrained(model_name)
        print("Zoe model loaded:", model_name, "on", _zoe_device)
        return True
    except Exception as e:
        print("Failed to init ZoeDepth:", e)
        _zoe_processor = None
        _zoe_model = None
        _zoe_device = None
        return False

def get_zoe_depth_map(image_rgb):
    """
    Run Zoe to produce a depth map.
    - image_rgb: HxWx3 uint8 RGB numpy array
    Returns: depth_map (float32 HxW) or raises RuntimeError.
    """
    global _zoe_processor, _zoe_model, _zoe_device, torch, TRANSFORMERS_AVAILABLE
    if not TRANSFORMERS_AVAILABLE or _zoe_processor is None or _zoe_model is None:
        raise RuntimeError("Zoe not initialized. Call init_zoe() and ensure transformers/torch are installed.")

    try:
        # Convert to PIL image
        pil = Image.fromarray(image_rgb)

        # The processor can sometimes expect a list or single image; try both safely.
        inputs = None
        try:
            inputs = _zoe_processor(images=pil, return_tensors="pt")
        except Exception:
            try:
                inputs = _zoe_processor([pil], return_tensors="pt")
            except Exception as e:
                raise RuntimeError("Processor failed to prepare inputs: " + str(e))

        if inputs is None:
            raise RuntimeError("Processor returned None inputs")

        # Move tensors to device if applicable
        try:
            for k, v in list(inputs.items()):
                if hasattr(v, "to"):
                    inputs[k] = v.to(_zoe_device)
        except Exception:
            # non-fatal; continue
            pass

        # Forward pass
        with torch.no_grad():
            outputs = _zoe_model(**inputs)

        # Extract depth - typical attribute is predicted_depth
        depth_map = None
        if hasattr(outputs, "predicted_depth"):
            depth_map = outputs.predicted_depth.squeeze().cpu().numpy().astype("float32")
        else:
            # fallback: look for any tensor-like attribute
            for v in outputs.__dict__.values():
                try:
                    if hasattr(v, "cpu"):
                        arr = v.squeeze().cpu().numpy().astype("float32")
                        if arr.ndim == 2:
                            depth_map = arr
                            break
                except Exception:
                    continue

        if depth_map is None:
            raise RuntimeError("Zoe returned no predicted_depth in outputs")

        # Resize to original image size if needed
        H, W = image_rgb.shape[:2]
        if depth_map.shape != (H, W):
            depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Ensure float32 and finite values
        depth_map = depth_map.astype("float32")
        depth_map[~np.isfinite(depth_map)] = np.nan
        return depth_map

    except Exception as e:
        raise RuntimeError("get_zoe_depth_map failed: " + str(e))

# --- Simple fallback estimator (heuristic) ---
def estimate_depth_fallback(bbox, frame_shape):
    """
    Heuristic fallback: map bbox height proportion to an approximate depth (meters).
    - bbox: (x1,y1,x2,y2)
    - frame_shape: (H,W) or HxWxC numpy shape
    Returns: depth in meters (float)
    """
    if isinstance(frame_shape, (tuple, list)) and len(frame_shape) >= 2:
        H = int(frame_shape[0])
    else:
        # assume frame is an array
        H = int(frame_shape.shape[0])

    x1, y1, x2, y2 = bbox
    h = max(1, (y2 - y1))
    # Tunable parameters: near (when bbox fills frame), far (when tiny)
    near_depth = 0.6
    far_depth = 6.0
    frac = np.clip(h / float(H), 1e-3, 1.0)
    depth = far_depth * (1.0 - frac) + near_depth * frac
    return float(depth)
