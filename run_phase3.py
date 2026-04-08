from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from src.cf import EnhancedItemBasedCF, ItemBasedCF
from src.data import MovieLens100KLoader, create_random_split


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    print("=== PHASE 3: Improve CF ===")

    loader = MovieLens100KLoader()
    ratings_df = loader.load_full_ratings()
    train_df, test_df = create_random_split(ratings_df, test_size=0.2, random_state=42)

    y_true = test_df["rating"].values

    baseline_cf = ItemBasedCF(k=40)
    baseline_cf.fit(train_df)
    baseline_preds = baseline_cf.predict(test_df)
    baseline_rmse = rmse(y_true, baseline_preds)

    improved_cf = EnhancedItemBasedCF(
        k=20,
        similarity="cosine",
        min_common=2,
        shrinkage=0.0,
    )
    improved_cf.fit(train_df)
    improved_preds = improved_cf.predict(test_df)
    improved_rmse = rmse(y_true, improved_preds)

    print("Improved config        : cosine similarity, k=20, min_common=2, shrinkage=0")
    print(f"Baseline Item-CF RMSE : {baseline_rmse:.4f}")
    print(f"Improved Item-CF RMSE : {improved_rmse:.4f}")
    print(f"RMSE Improvement       : {baseline_rmse - improved_rmse:.4f}")

    preview = test_df[["user_id", "item_id", "rating"]].copy().head(10)
    preview["baseline_pred"] = baseline_preds[:10]
    preview["improved_pred"] = improved_preds[:10]

    print("\nSample prediction comparison (first 10 test rows):")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
