from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import EXTRACTED_DIR, ML100K_URL, ML100K_ZIP, RAW_DIR


RATINGS_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
ITEM_COLUMNS = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "imdb_url",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


class MovieLens100KLoader:
    def __init__(self, extracted_dir: Path = EXTRACTED_DIR) -> None:
        self.extracted_dir = extracted_dir

    def ensure_dataset(self) -> None:
        if self.extracted_dir.exists():
            return

        RAW_DIR.mkdir(parents=True, exist_ok=True)

        if not ML100K_ZIP.exists():
            response = requests.get(ML100K_URL, timeout=60)
            response.raise_for_status()
            ML100K_ZIP.write_bytes(response.content)

        with zipfile.ZipFile(ML100K_ZIP, "r") as zip_ref:
            zip_ref.extractall(RAW_DIR)

    def load_items(self) -> pd.DataFrame:
        self.ensure_dataset()
        items_path = self.extracted_dir / "u.item"
        items_df = pd.read_csv(
            items_path,
            sep="|",
            names=ITEM_COLUMNS,
            encoding="latin-1",
            engine="python",
        )
        return items_df

    def load_fold(self, fold_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if fold_idx not in {1, 2, 3, 4, 5}:
            raise ValueError("fold_idx must be one of {1,2,3,4,5}")

        self.ensure_dataset()
        train_path = self.extracted_dir / f"u{fold_idx}.base"
        test_path = self.extracted_dir / f"u{fold_idx}.test"

        train_df = pd.read_csv(train_path, sep="\t", names=RATINGS_COLUMNS)
        test_df = pd.read_csv(test_path, sep="\t", names=RATINGS_COLUMNS)

        return train_df, test_df

    def load_full_ratings(self) -> pd.DataFrame:
        self.ensure_dataset()
        ratings_path = self.extracted_dir / "u.data"
        ratings_df = pd.read_csv(ratings_path, sep="\t", names=RATINGS_COLUMNS)
        return ratings_df


def summarize_fold(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_pairs = set(zip(train_df["user_id"], train_df["item_id"]))
    test_pairs = set(zip(test_df["user_id"], test_df["item_id"]))
    overlap = len(train_pairs.intersection(test_pairs))

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_users": train_df["user_id"].nunique(),
        "test_users": test_df["user_id"].nunique(),
        "train_items": train_df["item_id"].nunique(),
        "test_items": test_df["item_id"].nunique(),
        "train_test_pair_overlap": overlap,
        "train_rating_mean": float(train_df["rating"].mean()),
        "test_rating_mean": float(test_df["rating"].mean()),
    }


def create_random_split(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
