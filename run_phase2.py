from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from src.cf import ItemBasedCF
from src.data import MovieLens100KLoader, create_random_split


def main() -> None:
    print("=== PHASE 2: Baseline CF Model (Item-based CF) ===")

    loader = MovieLens100KLoader()
    ratings_df = loader.load_full_ratings()
    train_df, test_df = create_random_split(ratings_df, test_size=0.2, random_state=42)

    model = ItemBasedCF(k=40)
    model.fit(train_df)

    preds = model.predict(test_df)
    rmse = float(np.sqrt(mean_squared_error(test_df["rating"].values, preds)))

    print(f"Train size: {len(train_df)}")
    print(f"Test size : {len(test_df)}")
    print(f"RMSE      : {rmse:.4f}")

    preview = test_df[["user_id", "item_id", "rating"]].copy().head(10)
    preview["pred_rating"] = preds[:10]

    print("\nSample predictions (first 10 test rows):")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
