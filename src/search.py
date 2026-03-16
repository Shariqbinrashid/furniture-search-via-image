"""
Core search logic: multi-model support, TTA, lazy engine caching.
Supports CLIP (base/large), SigLIP, and DINOv2.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCT_INDEX_PATH = os.path.join(BASE_DIR, "data", "product_index.json")

NO_MATCH_THRESHOLD = 0.20
TOP_K = 5

MODELS = {
    "clip-base": {
        "hf_id": "openai/clip-vit-base-patch32",
        "type": "clip",
        "dim": 512,
        "label": "CLIP ViT-B/32 (fast)",
    },
    "clip-large": {
        "hf_id": "openai/clip-vit-large-patch14",
        "type": "clip",
        "dim": 768,
        "label": "CLIP ViT-L/14 (better)",
    },
    "siglip": {
        "hf_id": "google/siglip-base-patch16-224",
        "type": "siglip",
        "dim": 768,
        "label": "SigLIP B/16",
    },
    "dinov2": {
        "hf_id": "facebook/dinov2-base",
        "type": "dinov2",
        "dim": 768,
        "label": "DINOv2 B/14",
    },
}

_engine_cache: dict = {}


def get_engine(model_key: str) -> "ImageSearchEngine":
    if model_key not in _engine_cache:
        _engine_cache[model_key] = ImageSearchEngine(model_key)
    return _engine_cache[model_key]


def _tta_crops(image: Image.Image) -> list:
    """5 crops: center + 4 corners at 85% scale."""
    w, h = image.size
    cw, ch = int(w * 0.85), int(h * 0.85)
    return [
        image.crop(((w - cw) // 2, (h - ch) // 2, (w + cw) // 2, (h + ch) // 2)),
        image.crop((0, 0, cw, ch)),
        image.crop((w - cw, 0, w, ch)),
        image.crop((0, h - ch, cw, h)),
        image.crop((w - cw, h - ch, w, h)),
    ]


class ImageSearchEngine:
    def __init__(self, model_key: str):
        assert model_key in MODELS, f"Unknown model: {model_key}"
        self.model_key = model_key
        config = MODELS[model_key]
        model_type = config["type"]
        hf_id = config["hf_id"]

        print(f"Loading {config['label']} ...")

        if model_type == "clip":
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(hf_id)
            self.processor = CLIPProcessor.from_pretrained(hf_id)
        elif model_type == "siglip":
            from transformers import SiglipModel, SiglipProcessor
            self.model = SiglipModel.from_pretrained(hf_id)
            self.processor = SiglipProcessor.from_pretrained(hf_id)
        elif model_type == "dinov2":
            from transformers import AutoImageProcessor, AutoModel
            self.model = AutoModel.from_pretrained(hf_id)
            self.processor = AutoImageProcessor.from_pretrained(hf_id)

        self.model.eval()
        self.model_type = model_type

        embeddings_path = os.path.join(BASE_DIR, "data", f"embeddings_{model_key}.npy")
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"Embeddings not found: {embeddings_path}\n"
                f"Run: venv/bin/python src/build_index.py --model {model_key}"
            )

        print(f"Loading embeddings from {embeddings_path} ...")
        self.embeddings = np.load(embeddings_path)

        with open(PRODUCT_INDEX_PATH, encoding="utf-8") as f:
            self.product_index = json.load(f)

        print(f"Ready. {len(self.embeddings)} products indexed with {config['label']}.")

    def _extract_features(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            if self.model_type in ("clip", "siglip"):
                features = self.model.get_image_features(**inputs)
                raw = features.pooler_output if hasattr(features, "pooler_output") else features
            elif self.model_type == "dinov2":
                outputs = self.model(**inputs)
                raw = outputs.last_hidden_state[:, 0, :]  # CLS token
        vec = raw.squeeze().float().numpy()
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec.astype(np.float32)

    def embed_query(self, image: Image.Image, use_tta: bool = True) -> np.ndarray:
        crops = _tta_crops(image) if use_tta else [image]
        vecs = np.stack([self._extract_features(c) for c in crops], axis=0)
        pooled = vecs.mean(axis=0)
        norm = np.linalg.norm(pooled)
        return (pooled / norm).astype(np.float32) if norm > 0 else pooled

    def search(self, image: Image.Image, top_k: int = TOP_K, use_tta: bool = True):
        query_vec = self.embed_query(image, use_tta=use_tta)
        scores = self.embeddings @ query_vec  # cosine similarity (pre-normalized)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < NO_MATCH_THRESHOLD:
                break
            product = self.product_index[str(idx)]
            results.append({
                "name": product["name"],
                "price": product["price"],
                "score": score,
                "images": product["images"],
            })

        return results
