# utils/image_utils.py

import cv2
import os
import uuid
from typing import List, Union

def crop_image(
    image_path: str,
    bbox: List[Union[int, float]],
    out_dir: str = "data/crops"
) -> str:
    """
    Crops the region defined by bbox from image_path and
    writes it to out_dir with a unique filename.
    Returns the path to the cropped image.
    """
    # 1. Load
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at: {image_path}")

    # 2. Unpack & clamp
    x1, y1, x2, y2 = map(int, bbox)
    h, w = img.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    # 3. Crop
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop for bbox {bbox} on image {image_path}")

    # 4. Write to disk
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    fname = f"{base}_{uuid.uuid4().hex}.jpg"
    out_path = os.path.join(out_dir, fname)
    cv2.imwrite(out_path, crop)

    return out_path
