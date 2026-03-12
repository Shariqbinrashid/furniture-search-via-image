"""
Gradio UI for the furniture reverse image search POC.
"""

import os

import gradio as gr
from PIL import Image

from search import ImageSearchEngine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

engine = ImageSearchEngine()


def search(query_image):
    if query_image is None:
        return [], [], [], "Please upload an image."

    pil_image = Image.fromarray(query_image)
    results = engine.search(pil_image)

    if not results:
        return None, None, None, "No match found in the catalog."

    # Pad to always return 3 slots
    while len(results) < 3:
        results.append(None)

    def make_label(r):
        if r is None:
            return ""
        return f"{r['name']}\n${r['price']:,.2f}  |  Score: {r['score']:.2f}"

    def load_image(r):
        if r is None or not r["images"]:
            return None
        path = os.path.join(BASE_DIR, r["images"][0])
        return Image.open(path).convert("RGB") if os.path.exists(path) else None

    img1 = load_image(results[0])
    img2 = load_image(results[1])
    img3 = load_image(results[2])

    label1 = make_label(results[0])
    label2 = make_label(results[1])
    label3 = make_label(results[2])

    status = f"Top {sum(1 for r in results if r)} result(s) found."
    return img1, img2, img3, label1, label2, label3, status


with gr.Blocks(title="Furniture Image Search") as demo:
    gr.Markdown("# Furniture Reverse Image Search")
    gr.Markdown("Upload a photo of a furniture item to find visually similar products in the catalog.")

    with gr.Row():
        query_input = gr.Image(label="Upload Query Image", type="numpy", height=300)

    search_btn = gr.Button("Search", variant="primary")

    status_box = gr.Textbox(label="Status", interactive=False)

    gr.Markdown("### Top 3 Matches")
    with gr.Row():
        with gr.Column():
            out_img1 = gr.Image(label="Match 1", height=250)
            out_label1 = gr.Textbox(label="", interactive=False)
        with gr.Column():
            out_img2 = gr.Image(label="Match 2", height=250)
            out_label2 = gr.Textbox(label="", interactive=False)
        with gr.Column():
            out_img3 = gr.Image(label="Match 3", height=250)
            out_label3 = gr.Textbox(label="", interactive=False)

    search_btn.click(
        fn=search,
        inputs=[query_input],
        outputs=[out_img1, out_img2, out_img3, out_label1, out_label2, out_label3, status_box],
    )

if __name__ == "__main__":
    demo.launch(share=True)
