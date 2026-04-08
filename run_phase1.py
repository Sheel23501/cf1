from __future__ import annotations

from src.data import MovieLens100KLoader, create_random_split


def main() -> None:
    loader = MovieLens100KLoader()

    print("=== PHASE 1: Data Setup (MovieLens 100K) ===")

    ratings_df = loader.load_full_ratings()
    items_df = loader.load_items()

    print("\nFirst 5 rows:")
    print(ratings_df.head(5).to_string(index=False))

    print("\nDataset stats:")
    print(f"Number of users   : {ratings_df['user_id'].nunique()}")
    print(f"Number of movies  : {items_df['item_id'].nunique()}")
    print(f"Number of ratings : {len(ratings_df)}")

    train_df, test_df = create_random_split(ratings_df, test_size=0.2, random_state=42)

    print("\nTrain/Test split (80/20):")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")
    print(f"Train rating mean: {train_df['rating'].mean():.4f}")
    print(f"Test rating mean : {test_df['rating'].mean():.4f}")


if __name__ == "__main__":
    main()
