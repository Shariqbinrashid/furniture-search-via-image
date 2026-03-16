"""
Gradio UI for the furniture reverse image search POC.
"""

import csv
import hashlib
import os
from datetime import datetime

import gradio as gr
from PIL import Image

from search import MODELS, TOP_K, get_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "search_results.csv")

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


def remove_background(image: Image.Image) -> Image.Image:
    result = rembg_remove(image)
    # Paste onto white background so model doesn't see transparency artifacts
    bg = Image.new("RGB", result.size, (255, 255, 255))
    bg.paste(result, mask=result.split()[3] if result.mode == "RGBA" else None)
    return bg


def log_to_csv(model_key: str, query_hash: str, results: list):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "query_hash", "rank", "product_name", "price", "cosine_score"])
        for i, r in enumerate(results):
            writer.writerow([
                datetime.now().isoformat(),
                model_key,
                query_hash,
                i + 1,
                r["name"],
                r["price"],
                f"{r['score']:.4f}",
            ])


def search(query_image, model_key, use_bg_removal):
    if query_image is None:
        empty = [None] * TOP_K + [""] * TOP_K
        return *empty, "Please upload an image.", None

    pil_image = Image.fromarray(query_image).convert("RGB")

    # Compute hash for CSV logging (before bg removal)
    img_bytes = pil_image.tobytes()
    query_hash = hashlib.md5(img_bytes).hexdigest()[:8]

    if use_bg_removal:
        if not REMBG_AVAILABLE:
            status_msg = "rembg not installed — skipping background removal."
        else:
            pil_image = remove_background(pil_image)
            status_msg = None
    else:
        status_msg = None

    try:
        engine = get_engine(model_key)
    except FileNotFoundError as e:
        empty = [None] * TOP_K + [""] * TOP_K
        return *empty, str(e), None

    results = engine.search(pil_image, top_k=TOP_K)

    if not results:
        empty = [None] * TOP_K + [""] * TOP_K
        return *empty, "No match found above threshold.", None

    log_to_csv(model_key, query_hash, results)

    # Pad to TOP_K slots
    while len(results) < TOP_K:
        results.append(None)

    def make_label(r):
        if r is None:
            return ""
        return f"{r['name']}\n${r['price']:,.2f}  |  Score: {r['score']:.4f}"

    def load_image(r):
        if r is None or not r["images"]:
            return None
        path = os.path.join(BASE_DIR, r["images"][0])
        return Image.open(path).convert("RGB") if os.path.exists(path) else None

    images = [load_image(r) for r in results]
    labels = [make_label(r) for r in results]

    n_found = sum(1 for r in results if r)
    model_label = MODELS[model_key]["label"]
    bg_note = " (BG removed)" if use_bg_removal and REMBG_AVAILABLE else ""
    status = status_msg or f"Top {n_found} result(s) — {model_label}{bg_note}  |  query: {query_hash}"

    csv_output = CSV_PATH if os.path.exists(CSV_PATH) else None

    return *images, *labels, status, csv_output


MODEL_CHOICES = [(v["label"], k) for k, v in MODELS.items()]

with gr.Blocks(title="Furniture Image Search") as demo:
    gr.Markdown("# Furniture Reverse Image Search")
    gr.Markdown("Upload a photo of a furniture item to find visually similar products.")

    with gr.Row():
        query_input = gr.Image(label="Upload Query Image", type="numpy", height=300)

    with gr.Accordion("⚙️ Developer Settings", open=False):
        model_dropdown = gr.Dropdown(
            choices=MODEL_CHOICES,
            value="clip-base",
            label="Model",
            info="Switch model (first use downloads & builds embeddings)",
        )
        bg_removal_checkbox = gr.Checkbox(
            value=False,
            label="Remove background (enable for real-world/in-room photos)",
            info="Uses rembg. Avoid for clean product shots — can hurt glass/transparent surfaces.",
        )

    search_btn = gr.Button("Search", variant="primary")
    status_box = gr.Textbox(label="Status", interactive=False)

    gr.Markdown(f"### Top {TOP_K} Matches")
    with gr.Row():
        out_imgs = []
        out_labels = []
        for i in range(TOP_K):
            with gr.Column():
                out_imgs.append(gr.Image(label=f"Match {i+1}", height=220))
                out_labels.append(gr.Textbox(label="", interactive=False, lines=2))

    csv_download = gr.File(label="Download Results CSV (local only)", interactive=False)

    search_btn.click(
        fn=search,
        inputs=[query_input, model_dropdown, bg_removal_checkbox],
        outputs=[*out_imgs, *out_labels, status_box, csv_download],
    )

if __name__ == "__main__":
    demo.launch(share=True)
