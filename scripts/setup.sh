#!/bin/bash
# One-step environment setup for Yahoo Answers NLP Project
set -e

# Create venv if not exists
echo "[INFO] Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install requirements
echo "[INFO] Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[INFO] Setup complete. Activate with: source .venv/bin/activate"