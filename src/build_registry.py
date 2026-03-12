"""
Step 1: Parse CSV and build product_registry.json
Maps each product to its resolved image paths on disk.
"""

import csv
import json
import os
import unicodedata

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "The Essential Products.csv")
ASSETS_PATH = os.path.join(BASE_DIR, "Product Assets")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "product_registry.json")

# Folder name on disk differs from CSV product name for these
NAME_CORRECTIONS = {
    "Squiggle Chair/Table": "Squiggle ChairTable",
}


def normalize(name: str) -> str:
    return unicodedata.normalize("NFC", name.strip())


def resolve_folder(product_name: str) -> str:
    corrected = NAME_CORRECTIONS.get(product_name, product_name)
    return os.path.join(ASSETS_PATH, corrected)


def build_registry():
    registry = {}
    missing = []

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            product_name = normalize(row["Product Name"])
            price = float(row["Product Price"])
            image_filenames = [s.strip() for s in row["Image File Names"].split(",")]

            folder = resolve_folder(product_name)
            image_paths = []

            for filename in image_filenames:
                full_path = os.path.join(folder, filename)
                if os.path.exists(full_path):
                    # Store path relative to BASE_DIR for portability
                    rel_path = os.path.relpath(full_path, BASE_DIR)
                    image_paths.append(rel_path)
                else:
                    missing.append(f"{product_name} -> {filename}")

            registry[product_name] = {
                "price": price,
                "images": image_paths,
            }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Registry built: {len(registry)} products")
    print(f"Total images resolved: {sum(len(v['images']) for v in registry.values())}")
    if missing:
        print(f"Missing files ({len(missing)}):")
        for m in missing:
            print(f"  {m}")
    else:
        print("No missing files.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_registry()
