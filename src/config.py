from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = RAW_DIR / "ml-100k"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_ZIP = RAW_DIR / "ml-100k.zip"
