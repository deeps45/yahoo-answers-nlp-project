#!/bin/bash
# Download Yahoo Answers dataset to data/ folder
set -e
mkdir -p data
cd data

echo "[INFO] Downloading Yahoo Answers dataset..."
# Example: direct link from the recommended repo (user may need to update if link changes)
wget -O yahoo_answers_csv.tar.gz "https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset/releases/download/data/yahoo_answers_csv.tar.gz"

echo "[INFO] Extracting..."
tar -xzf yahoo_answers_csv.tar.gz

echo "[INFO] Data download and extraction complete. Files in data/:"
ls -lh *.csv