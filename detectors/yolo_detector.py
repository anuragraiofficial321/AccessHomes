"""
YoloDetector wrapper module. Keeps behavior identical to the monolithic script.
"""

from pathlib import Path
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

class YoloDetector:
    def __init__(self, model_path="models/yolo11n.pt", device="cpu", conf=0.6, iou=0.45):
        if YOLO is None:
            raise RuntimeError("ultralytics.YOLO not available: install ultralytics to use YOLO detection")
        self.model = YOLO(model_path)
        self.conf = conf; self.iou = iou; self.device = device
        try:
            self.model.fuse()
        except Exception:
            pass

    def detect_frame(self, frame_bgr):
        img = frame_bgr[..., ::-1]
        results = self.model.predict(source=[img], conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        dets = []
        if not results: return dets
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None: return dets
        for box in r.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
            except Exception:
                try:
                    xyxy = box.xyxy.tolist()[0]
                    conf = float(box.conf.tolist()[0])
                    cls_id = int(box.cls.tolist()[0])
                except Exception:
                    continue
            cls_name = self.model.names.get(cls_id, str(cls_id))
            dets.append({'class_name': cls_name, 'class_id': cls_id, 'conf': conf, 'xyxy': xyxy})
        return dets
