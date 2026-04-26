# Yahoo Answers Dataset Download Instructions

This project uses the **Yahoo Answers Topic Classification** dataset, which is too large to commit directly to GitHub.

## Source Links

- Dataset repo: https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset
- Google Drive folder: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ

## File to Download

Download this archive from the Drive folder:

- `yahoo_answers_csv.tar.gz`

## Where to Place It

Put the downloaded archive inside this `data/` folder:

- `data/yahoo_answers_csv.tar.gz`

## Extract on macOS / Linux (zsh)

Run from the repository root:

```bash
mkdir -p data
cd data
tar -xzf yahoo_answers_csv.tar.gz
```

## Expected Structure After Extraction

Depending on the archive layout, you  have :

- `data/yahoo_answers_csv/train.csv` and `data/yahoo_answers_csv/test.csv`

As the notebook expects `data/train.csv` and `data/test.csv`, move the files as needed.

## Notes

- Do **not** commit raw dataset files to GitHub.
- Keep only instructions and lightweight metadata in `data/`.
