"""
Verify repo structure and key dependencies for Yahoo Answers NLP Project.
Usage: python scripts/verify_setup.py
"""
import os
import sys

REQUIRED_FILES = [
    "main_notebook.ipynb",
    "requirements.txt",
    "README.md",
    "assets",
    "checkpoints",
    "data"
]

missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    print("[ERROR] Missing required files/folders:", ", ".join(missing))
    sys.exit(1)
else:
    print("[INFO] All required files/folders are present.")

try:
    import pandas, numpy, sklearn, torch, transformers
    print("[INFO] Key Python packages are installed.")
except ImportError as e:
    print(f"[ERROR] Missing package: {e.name}")
    sys.exit(1)

print("[INFO] Repo structure and dependencies look good!")