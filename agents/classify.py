# agents/classify.py

import os
import json
import uuid
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any

import torch
from transformers import CLIPProcessor, CLIPModel
import easyocr

class ClassificationAgent:
    """
    Takes a single-item crop and returns metadata including:
      - category (and confidence)
      - gender (and confidence)
      - dominant_color (hex)
      - pattern: 'solid' or 'patterned'
      - logo_text: any OCR'd text
    """

    def __init__(self, cfg: Dict[str, Any]):
        d = cfg['classify']
        # categories to score
        self.categories = d.get('categories', [])
        self.top_k = int(d.get('top_k', 1))
        # optional gender labels for zero-shot
        self.gender_labels = d.get('gender_labels', ['men', 'women', 'unisex'])

        # device for torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load CLIP for zero-shot classification
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(self.device)
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # color‑quantization
        self.num_clusters = int(d['color'].get('clusters', 3))

        # pattern detection threshold
        self.edge_thresh = float(d['pattern'].get('edge_thresh', 0.02))

        # OCR reader
        langs = d.get('ocr', {}).get('languages', ['en'])
        self.reader = easyocr.Reader(langs, gpu=torch.cuda.is_available())

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"ClassificationAgent loaded on {self.device}")

    def run(self, crop_path: str) -> Dict[str, Any]:
        # 1. Load images
        pil_img = Image.open(crop_path).convert('RGB')
        cv_img  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 2. Category Zero-shot via CLIP
        inputs = self.processor(
            text=self.categories,
            images=pil_img,
            return_tensors='pt',
            padding=True
        )
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
        image_feats = self.clip.get_image_features(**{'pixel_values': inputs['pixel_values']})
        text_feats  = self.clip.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        image_feats = image_feats / image_feats.norm(p=2, dim=-1, keepdim=True)
        text_feats  = text_feats  / text_feats.norm(p=2, dim=-1, keepdim=True)
        sims = (image_feats @ text_feats.T)[0]
        cat_idx = sims.topk(self.top_k).indices[0].item()
        category   = self.categories[cat_idx]
        cat_score  = float(sims[cat_idx].cpu())

        # 3. Gender Zero-shot via CLIP
        g_inputs = self.processor(
            text=self.gender_labels,
            images=pil_img,
            return_tensors='pt',
            padding=True
        )
        for k, v in g_inputs.items():
            g_inputs[k] = v.to(self.device)
        gender_text_feats = self.clip.get_text_features(
            input_ids=g_inputs['input_ids'],
            attention_mask=g_inputs['attention_mask']
        )
        gender_text_feats = gender_text_feats / gender_text_feats.norm(p=2, dim=-1, keepdim=True)
        g_sims = (image_feats @ gender_text_feats.T)[0]
        gender_idx = g_sims.argmax().item()
        gender      = self.gender_labels[gender_idx]
        gender_score = float(g_sims[gender_idx].cpu())

        # 4. Dominant color via K-means
        Z = cv_img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            Z, self.num_clusters, None,
            criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        counts = np.bincount(labels.flatten())
        dom = centers[np.argmax(counts)].astype(int)
        b, g, r = dom.tolist()
        dominant_color = '#{:02x}{:02x}{:02x}'.format(r, g, b)

        # 5. Pattern detection
        gray  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        pattern = 'patterned' if edges.mean() > self.edge_thresh else 'solid'

        # 6. OCR
        texts = self.reader.readtext(np.array(pil_img), detail=0)
        logo_text = ' '.join(texts) if texts else ''

        # assemble metadata
        meta = {
            'id':             str(uuid.uuid4()),
            'category':       category,
            'category_score': cat_score,
            'gender':         gender,
            'gender_score':   gender_score,
            'dominant_color': dominant_color,
            'logo_text':      logo_text
        }

        # optional: dump to disk
        out_dir = os.path.join('data', 'attributes')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(crop_path))[0]
        with open(os.path.join(out_dir, f"{base}.json"), 'w') as fp:
            json.dump(meta, fp, indent=2)

        self.logger.info(f"Classified {crop_path} → {meta}")
        return meta

    def execute(self, crop_path: str) -> Dict[str, Any]:
        return self.run(crop_path)



# if __name__ == "__main__":
#     import yaml, sys, logging
#     logging.basicConfig(level=logging.INFO)
#     cfg = yaml.safe_load(open("config.yml"))
#     agent = ClassificationAgent(cfg)
#     crop = sys.argv[1] if len(sys.argv)>1 else "data/crops/test_0.jpg"
#     out = agent.execute(crop)
#     print(out)
