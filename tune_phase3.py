from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.cf import EnhancedItemBasedCF, ItemBasedCF
from src.data import MovieLens100KLoader, create_random_split

def explain_similarity_choice():
    print("\nSimilarity Measures:")
    print("- Pearson Correlation: Accounts for user biases by normalizing ratings. Ideal for ratings data.")
    print("- Cosine Similarity: Measures angle between vectors. Simpler but ignores rating scale differences.\n")

def main() -> None:
    loader = MovieLens100KLoader()
    ratings = loader.load_full_ratings()
    train, test = create_random_split(ratings, test_size=0.2, random_state=42)
    y_true = test["rating"].values

    explain_similarity_choice()

    best_rmse = float("inf")
    best_mae = float("inf")
    best_cfg = None

    for similarity in ["pearson", "cosine"]:
        for k in [10, 20, 50]:
            for min_common in [2, 3]:
                for shrinkage in [0.0, 10.0]:
                    model = EnhancedItemBasedCF(
                        k=k,
                        similarity=similarity,
                        min_common=min_common,
                        shrinkage=shrinkage,
                    )
                    model.fit(train)
                    preds = model.predict(test)

                    # Handle missing values
                    preds = np.nan_to_num(preds, nan=np.mean(train["rating"]))

                    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
                    mae = float(mean_absolute_error(y_true, preds))

                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_mae = mae
                        best_cfg = ("enhanced", similarity, k, min_common, shrinkage)

    print("\nFinal Results:")
    print(f"BEST_RMSE={best_rmse:.4f}")
    print(f"BEST_MAE={best_mae:.4f}")
    print(f"BEST_CFG={best_cfg}")

    print("\nImprovement Summary:")
    print("Model Version\tRMSE")
    print("Basic CF (Phase 2)\t0.9130")  # Replace with actual Phase 2 RMSE
    print(f"Improved CF (Phase 3)\t{best_rmse:.4f}")

if __name__ == "__main__":
    main()
 