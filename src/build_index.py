"""
Step 2: Generate embeddings for all products and save to disk.
Reads product_registry.json, processes all images, mean-pools per product.

Usage:
    venv/bin/python src/build_index.py --model clip-base
    venv/bin/python src/build_index.py --model clip-large
    venv/bin/python src/build_index.py --model siglip
    venv/bin/python src/build_index.py --model dinov2
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(BASE_DIR, "data", "product_registry.json")
PRODUCT_INDEX_PATH = os.path.join(BASE_DIR, "data", "product_index.json")

MODELS = {
    "clip-base": {
        "hf_id": "openai/clip-vit-base-patch32",
        "type": "clip",
        "label": "CLIP ViT-B/32",
    },
    "clip-large": {
        "hf_id": "openai/clip-vit-large-patch14",
        "type": "clip",
        "label": "CLIP ViT-L/14",
    },
    "siglip": {
        "hf_id": "google/siglip-base-patch16-224",
        "type": "siglip",
        "label": "SigLIP B/16",
    },
    "dinov2": {
        "hf_id": "facebook/dinov2-base",
        "type": "dinov2",
        "label": "DINOv2 B/14",
    },
}


def load_model(model_key: str):
    config = MODELS[model_key]
    hf_id = config["hf_id"]
    model_type = config["type"]
    print(f"Loading {config['label']} ...")

    if model_type == "clip":
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(hf_id)
        processor = CLIPProcessor.from_pretrained(hf_id)
    elif model_type == "siglip":
        from transformers import SiglipModel, SiglipProcessor
        model = SiglipModel.from_pretrained(hf_id)
        processor = SiglipProcessor.from_pretrained(hf_id)
    elif model_type == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(hf_id)
        processor = AutoImageProcessor.from_pretrained(hf_id)

    model.eval()
    return model, processor, model_type


def extract_features(model_key, model, processor, model_type, image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        if model_type in ("clip", "siglip"):
            features = model.get_image_features(**inputs)
            raw = features.pooler_output if hasattr(features, "pooler_output") else features
        elif model_type == "dinov2":
            outputs = model(**inputs)
            raw = outputs.last_hidden_state[:, 0, :]
    return raw.squeeze().float().numpy()


def embed_images(image_paths, model_key, model, processor, model_type):
    vectors = []
    for path in image_paths:
        full_path = os.path.join(BASE_DIR, path)
        try:
            image = Image.open(full_path).convert("RGB")
            vec = extract_features(model_key, model, processor, model_type, image)
            vectors.append(vec)
        except Exception as e:
            print(f"  Warning: skipping {path} — {e}")
    return np.stack(vectors, axis=0) if vectors else None


def mean_pool_normalize(arr: np.ndarray) -> np.ndarray:
    pooled = arr.mean(axis=0)
    norm = np.linalg.norm(pooled)
    return (pooled / norm).astype(np.float32) if norm > 0 else pooled.astype(np.float32)


def build_index(model_key: str):
    embeddings_path = os.path.join(BASE_DIR, "data", f"embeddings_{model_key}.npy")

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    model, processor, model_type = load_model(model_key)

    product_metadata = {}
    all_embeddings = []
    total = len(registry)

    for i, (product_name, info) in enumerate(registry.items()):
        print(f"[{i+1}/{total}] {product_name} ({len(info['images'])} images)")
        arr = embed_images(info["images"], model_key, model, processor, model_type)
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

    embeddings_matrix = np.stack(all_embeddings, axis=0)
    np.save(embeddings_path, embeddings_matrix)

    # Always write product_index.json (same content regardless of model)
    with open(PRODUCT_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(product_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_embeddings)} products embedded.")
    print(f"Shape: {embeddings_matrix.shape}")
    print(f"Saved embeddings: {embeddings_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="clip-base",
        help="Which model to build embeddings for",
    )
    args = parser.parse_args()
    build_index(args.model)
