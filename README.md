# Furniture Reverse Image Search — POC

A reverse image search system for a furniture catalog. Upload any furniture photo and instantly find the top 3 visually similar products.

**Live Demo → [huggingface.co/spaces/shariqbinrashid/furniture-search](https://huggingface.co/spaces/shariqbinrashid/furniture-search)**

---

## How It Works

1. All catalog images are embedded with **CLIP** (OpenAI's vision model) into 512-dimensional vectors
2. Multiple images per product are **mean-pooled** into a single representative vector
3. At query time, the uploaded image is embedded and compared to all product vectors using **cosine similarity**
4. Top 3 matches above a confidence threshold are returned

```
Upload image → CLIP encoder → cosine similarity → Top 3 products
```

---

## Dataset

| Metric | Value |
|---|---|
| Total products | 240 |
| Total images | 1,514 |
| Avg images per product | 6.3 |
| Max images per product | 104 |
| Image formats | JPG, PNG, WebP |
| Price range | $48 – $16,500 |

---

## Tech Stack

| Component | Tool |
|---|---|
| Image embeddings | CLIP ViT-B/32 (HuggingFace) |
| Vector search | NumPy cosine similarity |
| UI | Gradio |
| Hosting | Hugging Face Spaces (free) |

---

## Project Structure

```
poc-image/
├── src/
│   ├── build_registry.py   # Parse CSV → product_registry.json
│   ├── build_index.py      # Generate CLIP embeddings → embeddings.npy
│   ├── search.py           # Core search logic
│   └── app.py              # Gradio UI
├── The Essential Products.csv
└── requirements.txt
```

---

## Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build index (one-time, ~10-15 min on CPU)
python src/build_registry.py
python src/build_index.py

# 4. Run the app
python src/app.py
# Opens at http://localhost:7860
```

> **Note:** The `data/` folder (embeddings) and `Product Assets/` (images) are not committed to this repo. Run the build scripts to generate them locally.

---

## Best Products to Test With

These have the most images and most distinctive shapes — ideal for demo:

- **Wishbone Chair** — iconic Y-back silhouette
- **Egg Chair** — rounded pod design
- **Womb Chair** — curved shell form
- **Fireside Chair** — 59 catalog images, strongest embedding
- **Crescent Sofa** — distinctive curved sofa
- **Modular Patio Set** — great for outdoor furniture queries
