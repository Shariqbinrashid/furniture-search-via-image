"""
Core search logic: load embeddings, embed a query image, return top-K products.
Uses numpy cosine similarity — fast enough for 240 products, no FAISS needed.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRODUCT_INDEX_PATH = os.path.join(BASE_DIR, "data", "product_index.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")

MODEL_NAME = "openai/clip-vit-base-patch32"
NO_MATCH_THRESHOLD = 0.20
TOP_K = 3


class ImageSearchEngine:
    def __init__(self):
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()

        print("Loading embeddings...")
        self.embeddings = np.load(EMBEDDINGS_PATH)  # (240, 512)

        with open(PRODUCT_INDEX_PATH, encoding="utf-8") as f:
            self.product_index = json.load(f)

        print(f"Ready. {len(self.embeddings)} products indexed.")

    def embed_query(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        # transformers v5 returns BaseModelOutputWithPooling; v4 returns tensor
        raw = features.pooler_output if hasattr(features, "pooler_output") else features
        vec = raw.squeeze().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec  # (512,)

    def search(self, image: Image.Image, top_k: int = TOP_K):
        query_vec = self.embed_query(image)  # (512,)

        # Cosine similarity: embeddings are pre-normalized, query is normalized
        # dot product == cosine similarity
        scores = self.embeddings @ query_vec  # (240,)

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
