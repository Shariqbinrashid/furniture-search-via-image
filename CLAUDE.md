# Furniture Reverse Image Search — POC

## Project Overview
Reverse image search POC for a furniture catalog using vision model embeddings and cosine similarity.
Upload a furniture photo → returns top 5 visually similar products with name, price, and cosine score.

## Stack
- Python 3.13
- Multi-model: CLIP ViT-B/32, CLIP ViT-L/14, SigLIP B/16, DINOv2 B/14 (all via HuggingFace transformers)
- NumPy cosine similarity (no FAISS — not needed at 240 products)
- Query-time TTA: 5 crops (center + 4 corners) mean-pooled for better accuracy
- Optional background removal via `rembg` (off by default — hurts glass/transparent surfaces)
- Gradio 6.x for UI
- Deployed on Hugging Face Spaces

## Known Issues / Gotchas
- transformers v5 breaking change: `get_image_features()` returns `BaseModelOutputWithPooling`, not a tensor. Use `.pooler_output` attribute.
- faiss-cpu has no pre-built wheel for Python 3.13 on macOS — use numpy dot product instead.
- macOS filesystem stores folder names in NFD unicode; CSV is NFC. Fix: `unicodedata.normalize('NFC', name)`.
- `Squiggle Chair/Table` in CSV → folder on disk is `Squiggle ChairTable`. Handled via NAME_CORRECTIONS dict in build_registry.py.
- SigLIP requires `sentencepiece` and `protobuf` packages — both in requirements.txt.
- Always run scripts from project root, not from src/.
- venv python: `venv/bin/python`
- Embeddings are model-specific: `embeddings_{model_key}.npy`. Each model must be built separately.

## Commands
```bash
# One-time setup
python -m venv venv
venv/bin/pip install -r requirements.txt

# Build data files (run once per model)
venv/bin/python src/build_registry.py
venv/bin/python src/build_index.py --model clip-base
venv/bin/python src/build_index.py --model clip-large
venv/bin/python src/build_index.py --model siglip
venv/bin/python src/build_index.py --model dinov2

# Run locally
venv/bin/python src/app.py
```

## Models
| Key | HF ID | Dim | Notes |
|---|---|---|---|
| `clip-base` | openai/clip-vit-base-patch32 | 512 | Fast, default |
| `clip-large` | openai/clip-vit-large-patch14 | 768 | Better accuracy |
| `siglip` | google/siglip-base-patch16-224 | 768 | Good retrieval |
| `dinov2` | facebook/dinov2-base | 768 | Best visual/color similarity |

## Project Structure
```
poc-image/
├── src/
│   ├── build_registry.py   # Step 1: parse CSV → product_registry.json
│   ├── build_index.py      # Step 2: embeddings → embeddings_{model}.npy (--model flag)
│   ├── search.py           # Core search: multi-model engine cache, TTA, lazy loading
│   └── app.py              # Gradio UI (local) — settings hidden in ⚙️ accordion
├── data/                   # Generated files (gitignored)
│   ├── product_registry.json
│   ├── product_index.json
│   ├── embeddings_clip-base.npy
│   ├── embeddings_clip-large.npy
│   ├── embeddings_siglip.npy
│   ├── embeddings_dinov2.npy
│   └── search_results.csv  # Auto-logged on each search
├── Product Assets/         # Raw images (gitignored)
├── The Essential Products.csv
├── furniture-search/       # HF Spaces deployment repo (gitignored)
└── requirements.txt
```

## UI Features
- Top 5 matches with 4-decimal cosine scores
- `⚙️ Developer Settings` accordion (collapsed by default):
  - Model dropdown — switches model, lazy-loads on first use
  - Background removal checkbox — uses rembg, off by default
- CSV download button (local: persists; HF Spaces: session only)

## Deployment
Live on HuggingFace Spaces: https://huggingface.co/spaces/shariqbinrashid/furniture-search
Deployment repo: `furniture-search/` — single `app.py` with all logic bundled (no imports from src/).
