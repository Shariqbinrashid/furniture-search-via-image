# Furniture Reverse Image Search — POC

## Project Overview
Reverse image search POC for a furniture catalog using CLIP embeddings and cosine similarity.
Upload a furniture photo → returns top 3 visually similar products with name, price, and score.

## Stack
- Python 3.13
- CLIP ViT-B/32 (openai/clip-vit-base-patch32 via HuggingFace transformers)
- NumPy cosine similarity (no FAISS — not needed at 240 products)
- Gradio 6.x for UI
- Deployed on Hugging Face Spaces

## Known Issues / Gotchas
- transformers v5 breaking change: `get_image_features()` returns `BaseModelOutputWithPooling`, not a tensor. Use `.pooler_output` attribute.
- faiss-cpu has no pre-built wheel for Python 3.13 on macOS — use numpy dot product instead.
- macOS filesystem stores folder names in NFD unicode; CSV is NFC. Fix: `unicodedata.normalize('NFC', name)`.
- `Squiggle Chair/Table` in CSV → folder on disk is `Squiggle ChairTable`. Handled via NAME_CORRECTIONS dict in build_registry.py.
- Always run scripts from project root, not from src/.
- venv python: `venv/bin/python`

## Commands
```bash
# One-time setup
python -m venv venv
venv/bin/pip install -r requirements.txt

# Build data files (run once)
venv/bin/python src/build_registry.py
venv/bin/python src/build_index.py

# Run locally
venv/bin/python src/app.py
```

## Project Structure
```
poc-image/
├── src/
│   ├── build_registry.py   # Step 1: parse CSV → product_registry.json
│   ├── build_index.py      # Step 2: CLIP embeddings → embeddings.npy
│   ├── search.py           # Core search logic
│   └── app.py              # Gradio UI (local)
├── data/                   # Generated files (gitignored)
│   ├── product_registry.json
│   ├── embeddings.npy
│   └── product_index.json
├── Product Assets/         # Raw images (gitignored)
├── The Essential Products.csv
├── furniture-search/       # HF Spaces deployment repo (gitignored)
└── requirements.txt
```

## Deployment
Live on HuggingFace Spaces: https://huggingface.co/spaces/shariqbinrashid/furniture-search
Deployment repo: `furniture-search/` — single `app.py` with all logic bundled.
