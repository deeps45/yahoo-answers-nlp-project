"""
Extract all image outputs from the main notebook and save to assets/.
Usage: python scripts/extract_figures.py
"""
import nbformat
import os
import base64

NOTEBOOK = "main_notebook.ipynb"
ASSETS_DIR = "assets"

os.makedirs(ASSETS_DIR, exist_ok=True)
with open(NOTEBOOK, "r") as f:
    nb = nbformat.read(f, as_version=4)

img_count = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code" and "outputs" in cell:
        for output in cell["outputs"]:
            if output.output_type == "display_data" and "image/png" in output.get("data", {}):
                img_data = output["data"]["image/png"]
                img_bytes = base64.b64decode(img_data)
                img_path = os.path.join(ASSETS_DIR, f"figure_{img_count+1:02d}_cell_{i+1}.png")
                with open(img_path, "wb") as img_file:
                    img_file.write(img_bytes)
                print(f"[INFO] Saved {img_path}")
                img_count += 1
print(f"[INFO] Extracted {img_count} images to {ASSETS_DIR}/")