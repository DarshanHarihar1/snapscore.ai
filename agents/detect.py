import os
import json
import uuid
import logging
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image

from transformers import pipeline

class DetectionAgent:
    def __init__(self, cfg: Dict[str, Any]):
        d = cfg['detect']
        self.allowed = set(d.get('allowed_classes', []))
        self.use_hf = bool(d.get('use_hf', False))

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.use_hf:
            model_id = d['hf_model']
            device   = int(d.get('device', -1))
            self.logger.info(f"Loading HF object-detection pipeline → {model_id}")
            self.detector = pipeline(
                task="object-detection",
                model=model_id,
                device=device
            )
        else:
            from ultralytics import YOLO
            self.logger.info(f"Loading YOLOv8 weights → {d['yolov8_weights']}")
            self.model = YOLO(d['yolov8_weights'])
            self.imgsz = int(d.get('imgsz', 640))
            self.conf  = float(d.get('conf_threshold', 0.5))
            self.iou   = float(d.get('iou_threshold', 0.45))
            self.classes = d.get('classes') or None

    def run(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect objects in the image and save results to JSON.
        Args:
            image_path: path to the input image
        Returns:
            List[Dict[str, Any]]: list of detected objects with their details
        """
        detections: List[Dict[str, Any]] = []

        if self.use_hf:
            img = Image.open(image_path).convert("RGB")
            results = self.detector(img)
            for r in results:
                label = r["label"]
                if self.allowed and label not in self.allowed:
                    continue
                det = {
                    "id":    str(uuid.uuid4()),
                    "class": label,
                    "score": float(r["score"]),
                    "bbox":  [
                        float(r["box"]["xmin"]),
                        float(r["box"]["ymin"]),
                        float(r["box"]["xmax"]),
                        float(r["box"]["ymax"])
                    ]
                }
                detections.append(det)
        else:
            res = self.model(
                image_path,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes
            )[0]
            boxes = res.boxes.data.cpu().numpy()
            for x1, y1, x2, y2, conf, cls_id in boxes:
                cls_name = self.model.names[int(cls_id)]
                if self.allowed and cls_name not in self.allowed:
                    continue
                detections.append({
                    "id":    str(uuid.uuid4()),
                    "class": cls_name,
                    "score": float(conf),
                    "bbox":  [float(x1), float(y1), float(x2), float(y2)]
                })

        out_dir = os.path.join("data", "detections")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        with open(os.path.join(out_dir, f"{base}.json"), "w") as fp:
            json.dump({"detections": detections}, fp, indent=2)

        self.logger.info(f"Detected {len(detections)} allowed objects in '{image_path}'")
        return detections

    def annotate(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        out_path: str = None
    ) -> str:
        """
        Draws bbox+label on the image and saves to data/annotated/.
        Returns the saved path.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cls = det["class"]
            score = det["score"]
            # box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            # label
            text = f"{cls} {score:.2f}"
            cv2.putText(
                img, text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 1
            )

        if not out_path:
            ann_dir = os.path.join("data", "annotated")
            os.makedirs(ann_dir, exist_ok=True)
            base = os.path.basename(image_path)
            out_path = os.path.join(ann_dir, base)

        cv2.imwrite(out_path, img)
        self.logger.info(f"Saved annotated image to {out_path}")
        return out_path

    def execute(self, image_path: str) -> List[Dict[str, Any]]:
        return self.run(image_path)
