"""
Step 2: Generate CLIP embeddings for all products and save to disk.
Reads product_registry.json, processes all images, mean-pools per product,
writes embeddings.npy and product_index.json.

No FAISS needed — at 240 products, numpy cosine similarity is instantaneous.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "product_registry.json")
PRODUCT_INDEX_PATH = os.path.join(BASE_DIR, "data", "product_index.json")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_clip():
    print(f"Loading CLIP model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()
    return model, processor


def embed_images(image_paths: list, model, processor) -> np.ndarray | None:
    """Embed a list of image paths. Returns (N, 512) float32 array."""
    vectors = []
    for path in image_paths:
        full_path = os.path.join(BASE_DIR, path)
        try:
            image = Image.open(full_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            # transformers v5 returns BaseModelOutputWithPooling; v4 returns tensor
            raw = features.pooler_output if hasattr(features, "pooler_output") else features
            vec = raw.squeeze().numpy().astype(np.float32)
            vectors.append(vec)
        except Exception as e:
            print(f"  Warning: skipping {path} — {e}")

    return np.stack(vectors, axis=0) if vectors else None


def mean_pool_normalize(arr: np.ndarray) -> np.ndarray:
    """Mean-pool N vectors into 1, then L2-normalize."""
    pooled = arr.mean(axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled / norm).astype(np.float32) if norm > 0 else pooled


def build_index():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    model, processor = load_clip()

    product_metadata = {}
    all_embeddings = []
    total = len(registry)

    for i, (product_name, info) in enumerate(registry.items()):
        print(f"[{i+1}/{total}] {product_name} ({len(info['images'])} images)")

        arr = embed_images(info["images"], model, processor)
        if arr is None:
            print(f"  Skipping — no valid images")
            continue

        embedding = mean_pool_normalize(arr)
        idx = len(all_embeddings)

        all_embeddings.append(embedding)
        product_metadata[str(idx)] = {
            "name": product_name,
            "price": info["price"],
            "images": info["images"],
        }

    embeddings_matrix = np.stack(all_embeddings, axis=0)  # (240, 512)
    np.save(EMBEDDINGS_PATH, embeddings_matrix)

    with open(PRODUCT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(product_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_embeddings)} products embedded.")
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    print(f"Saved to: {EMBEDDINGS_PATH}")
    print(f"Saved product map to: {PRODUCT_INDEX_PATH}")


if __name__ == "__main__":
    build_index()
