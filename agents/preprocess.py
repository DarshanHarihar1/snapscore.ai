# agents/preprocess.py

import cv2
from PIL import Image
import numpy as np
import os
import logging
from typing import Dict, Any

class PreprocessingAgent:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        self.max_w, self.max_h = cfg['preprocess']['max_size']
        self.pad_color = tuple(cfg['preprocess']['pad_color'])

        enh = cfg['preprocess']['enhancement']
        self.enh_type   = enh.get('type', 'none')
        self.clip_limit = enh.get('clip_limit', 2.0)
        self.tile_grid  = tuple(enh.get('tile_grid_size', [8,8]))
        self.conditional = enh.get('conditional', False)

        thr = enh.get('thresholds', {})
        self.bright_min   = thr.get('brightness_min', 0)
        self.bright_max   = thr.get('brightness_max', 255)
        self.contrast_std = thr.get('contrast_std_min', 0)

        self.logger.info(
            f"Preprocessor cfg: max_size=({self.max_w},{self.max_h}), "
            f"enhancement={self.enh_type}, conditional={self.conditional}"
        )

    def should_enhance(self, img: np.ndarray) -> bool:
        """Return True if brightness/contrast outside acceptable range."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_b = float(gray.mean())
        std_c  = float(gray.std())
        self.logger.info(
            f"Image stats → brightness={mean_b:.1f}, contrast_std={std_c:.1f}"
        )

        # brightness out of [min, max]?
        if mean_b < self.bright_min or mean_b > self.bright_max:
            self.logger.info("Brightness outside bounds, will enhance.")
            return True

        # contrast below threshold?
        if std_c < self.contrast_std:
            self.logger.info("Contrast too low, will enhance.")
            return True

        self.logger.info("Image quality OK, skipping enhancement.")
        return False

    def run(self, image_path: str) -> str:
        """
        Preprocess image: resize, pad, and enhance if needed.
        Args:
            image_path: path to the input image
        Returns:
            str: path to the cleaned image
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at {image_path}")

        h, w = img.shape[:2]
        scale = min(self.max_w / w, self.max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self.logger.info(f"Resized to ({new_w},{new_h})")

        dw, dh = self.max_w - new_w, self.max_h - new_h
        top, bottom = dh // 2, dh - dh//2
        left, right = dw // 2, dw - dw//2
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=self.pad_color
        )
        self.logger.info(f"Padded to {img.shape[:2]} (top/bot={top}/{bottom}, left/right={left}/{right})")

        do_enh = (self.enh_type == 'clahe' and
                  (not self.conditional or self.should_enhance(img)))

        if do_enh:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=self.tile_grid
            )
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            self.logger.info("Applied CLAHE.")

        fname = os.path.basename(image_path)
        out_dir = os.path.join("data", "cleaned")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, img)
        self.logger.info(f"Saved cleaned image to {out_path}")
        return out_path

    def execute(self, image_path: str) -> str:
        """
        Unified entrypoint for orchestrator:
        takes raw image path, returns cleaned image path.
        """
        return self.run(image_path)
