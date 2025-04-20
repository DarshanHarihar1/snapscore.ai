# agents/querygen.py

import os
import json
import logging
import re
from typing import Dict, Any
from jinja2 import Template
from PIL import Image
import webcolors

# limited mapping of hex to common color names
NAMED_COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'blue':  (0, 0, 255),
    'green': (0, 128, 0),
    'brown': (165, 42, 42),
    'grey':  (128, 128, 128)
}

class QueryGenAgent:
    def __init__(self, cfg: Dict[str, Any]):
        q = cfg.get('querygen', {})
        self.templates = [Template(t) for t in q.get('templates', [])]
        self.use_clip = bool(q.get('use_clip', False))

        self.logger = logging.getLogger(self.__class__.__name__)
        if self.use_clip:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            model_id = q.get('clip_model')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Loading CLIP for embeddings → {model_id}")
            self.clip = CLIPModel.from_pretrained(model_id).to(self.device)
            self.proc = CLIPProcessor.from_pretrained(model_id)

    def _hex_to_common_name(self, hexcode: str) -> str:
        r, g, b = tuple(int(hexcode.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        nearest_name, min_dist = None, float('inf')
        for name, (cr, cg, cb) in NAMED_COLORS.items():
            dist = (r - cr)**2 + (g - cg)**2 + (b - cb)**2
            if dist < min_dist:
                min_dist, nearest_name = dist, name
        return nearest_name or hexcode

    def run(self, metadata: Dict[str, Any], crop_path: str) -> Dict[str, Any]:
        """
        Generate text queries based on metadata and optional CLIP embeddings.
        Args:
            metadata: dict with image metadata (e.g., color
                and dominant_color)
            crop_path: path to the cropped image
        Returns:
            dict: generated text queries and optional CLIP embeddings
        """
        for key in ('color', 'dominant_color'):
            val = metadata.get(key)
            if isinstance(val, str) and re.match(r'^#([0-9A-Fa-f]{6})$', val):
                try:
                    metadata[key] = webcolors.hex_to_name(val, spec='css3')
                except Exception:
                    metadata[key] = self._hex_to_common_name(val)

        text_queries = []
        for tpl in self.templates:
            try:
                q = tpl.render(**metadata).strip()
                if q:
                    text_queries.append(" ".join(q.split()))
            except Exception as e:
                self.logger.error(f"Template render error: {e}")

        gender = metadata.get('gender')
        if isinstance(gender, str) and gender.strip():
            gender_lower = gender.lower()
            merged = []
            for query in text_queries:
                if query.lower().startswith(gender_lower + ' '):
                    merged.append(query)
                else:
                    merged.append(f"{gender} {query}")
            text_queries = merged

        result = {"text_queries": text_queries}

        if self.use_clip:
            import torch
            img = Image.open(crop_path).convert("RGB")
            inputs = self.proc(images=img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.clip.get_image_features(**inputs)
            result["clip_embedding"] = emb[0].cpu().tolist()

        out_dir = os.path.join("data", "queries")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(crop_path))[0]
        with open(os.path.join(out_dir, f"{base}.json"), "w") as fp:
            json.dump(result, fp, indent=2)

        self.logger.info(f"Generated queries for {crop_path}: {text_queries}")
        return result

    def execute(self, metadata: Dict[str, Any], crop_path: str) -> Dict[str, Any]:
        return self.run(metadata, crop_path)
