"""
Data Loader for DiffRec Project.

Handles loading, binarizing, and splitting datasets for implicit feedback
collaborative filtering. Supports Amazon-Book and Yelp datasets.

Binarization: Any rating >= threshold → 1 (interacted), else → 0 (not interacted)
Split: 80% train / 10% validation / 10% test (per-user random split)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple, Dict, Set
from collections import defaultdict


class ImplicitDataLoader:
    """Load and preprocess datasets for implicit feedback recommendation."""
    
    def __init__(self, data_dir: str, dataset_name: str, min_interactions: int = 5):
        """
        Args:
            data_dir: Root data directory (e.g., "data/")
            dataset_name: "amazon-book" or "yelp"
            min_interactions: Minimum interactions per user/item to keep (k-core filtering)
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.min_interactions = min_interactions
        self.processed_dir = os.path.join(data_dir, dataset_name, "processed")
        
        self.n_users = 0
        self.n_items = 0
        self.user_map: Dict[int, int] = {}  # original_id → indexed_id
        self.item_map: Dict[int, int] = {}
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw rating data from file.
        
        Expected format: user_id, item_id, rating, [timestamp]
        Supports CSV, TSV, and space-separated files.
        
        Args:
            filepath: Path to the raw data file
            
        Returns:
            DataFrame with columns [user_id, item_id, rating]
        """
        # Try different separators
        for sep in ["\t", ",", " ", "::"]:
            try:
                df = pd.read_csv(
                    filepath, 
                    sep=sep, 
                    header=None,
                    names=["user_id", "item_id", "rating", "timestamp"][:4],
                    engine="python",
                    nrows=5,
                )
                if len(df.columns) >= 3:
                    df = pd.read_csv(
                        filepath,
                        sep=sep,
                        header=None,
                        engine="python",
                    )
                    break
            except Exception:
                continue
        
        # Use first 3 columns: user, item, rating
        df = df.iloc[:, :3]
        df.columns = ["user_id", "item_id", "rating"]
        
        print(f"Loaded {len(df)} raw interactions")
        print(f"  Users: {df['user_id'].nunique()}")
        print(f"  Items: {df['item_id'].nunique()}")
        print(f"  Rating range: [{df['rating'].min()}, {df['rating'].max()}]")
        
        return df
    
    def binarize(self, df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
        """
        Binarize ratings: rating >= threshold → 1, else drop.
        
        For implicit feedback, we only keep positive interactions.
        
        Args:
            df: Raw DataFrame with rating column
            threshold: Minimum rating to consider as positive interaction
            
        Returns:
            DataFrame with only positive interactions (rating = 1)
        """
        # Keep only positive interactions
        df_binary = df[df["rating"] >= threshold].copy()
        df_binary["rating"] = 1
        
        # Remove duplicates (same user-item pair)
        df_binary = df_binary.drop_duplicates(subset=["user_id", "item_id"])
        
        print(f"\nAfter binarization (threshold={threshold}):")
        print(f"  Positive interactions: {len(df_binary)}")
        print(f"  Users: {df_binary['user_id'].nunique()}")
        print(f"  Items: {df_binary['item_id'].nunique()}")
        
        return df_binary
    
    def k_core_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iteratively remove users/items with fewer than k interactions.
        This ensures every user and item has enough data to learn from.
        
        Args:
            df: Binary interaction DataFrame
            
        Returns:
            Filtered DataFrame
        """
        print(f"\nApplying {self.min_interactions}-core filtering...")
        
        while True:
            prev_len = len(df)
            
            # Remove users with too few interactions
            user_counts = df["user_id"].value_counts()
            valid_users = user_counts[user_counts >= self.min_interactions].index
            df = df[df["user_id"].isin(valid_users)]
            
            # Remove items with too few interactions
            item_counts = df["item_id"].value_counts()
            valid_items = item_counts[item_counts >= self.min_interactions].index
            df = df[df["item_id"].isin(valid_items)]
            
            if len(df) == prev_len:
                break
        
        print(f"  After filtering: {len(df)} interactions")
        print(f"  Users: {df['user_id'].nunique()}, Items: {df['item_id'].nunique()}")
        
        return df
    
    def create_id_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Re-index user and item IDs to contiguous integers starting from 0.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            DataFrame with re-indexed IDs
        """
        unique_users = sorted(df["user_id"].unique())
        unique_items = sorted(df["item_id"].unique())
        
        self.user_map = {old: new for new, old in enumerate(unique_users)}
        self.item_map = {old: new for new, old in enumerate(unique_items)}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        df = df.copy()
        df["user_id"] = df["user_id"].map(self.user_map)
        df["item_id"] = df["item_id"].map(self.item_map)
        
        print(f"\nRe-indexed: {self.n_users} users, {self.n_items} items")
        
        return df
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        val_ratio: float = 0.1, 
        test_ratio: float = 0.1, 
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Per-user random split into train/validation/test.
        
        For each user, randomly hold out val_ratio of interactions for validation
        and test_ratio for test. The rest goes to training.
        
        Args:
            df: Re-indexed interaction DataFrame
            val_ratio: Fraction for validation
            test_ratio: Fraction for test
            seed: Random seed for reproducibility
            
        Returns:
            (train_df, val_df, test_df)
        """
        rng = np.random.RandomState(seed)
        
        train_data, val_data, test_data = [], [], []
        
        for user_id, group in df.groupby("user_id"):
            items = group["item_id"].values.copy()
            rng.shuffle(items)
            
            n = len(items)
            n_test = max(1, int(n * test_ratio))
            n_val = max(1, int(n * val_ratio))
            
            test_items = items[:n_test]
            val_items = items[n_test:n_test + n_val]
            train_items = items[n_test + n_val:]
            
            for item in train_items:
                train_data.append((user_id, item, 1))
            for item in val_items:
                val_data.append((user_id, item, 1))
            for item in test_items:
                test_data.append((user_id, item, 1))
        
        cols = ["user_id", "item_id", "rating"]
        train_df = pd.DataFrame(train_data, columns=cols)
        val_df = pd.DataFrame(val_data, columns=cols)
        test_df = pd.DataFrame(test_data, columns=cols)
        
        print(f"\nSplit results:")
        print(f"  Train: {len(train_df)} interactions")
        print(f"  Val:   {len(val_df)} interactions")
        print(f"  Test:  {len(test_df)} interactions")
        
        return train_df, val_df, test_df
    
    def to_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Convert DataFrame to sparse user-item interaction matrix."""
        rows = df["user_id"].values
        cols = df["item_id"].values
        vals = np.ones(len(df), dtype=np.float32)
        return csr_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))
    
    def to_user_dict(self, df: pd.DataFrame) -> Dict[int, Set[int]]:
        """Convert DataFrame to {user_id: set of item_ids} dictionary."""
        user_dict = defaultdict(set)
        for _, row in df.iterrows():
            user_dict[int(row["user_id"])].add(int(row["item_id"]))
        return dict(user_dict)
    
    def save_processed(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame,
    ) -> None:
        """Save processed data splits to disk."""
        os.makedirs(self.processed_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(self.processed_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(self.processed_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(self.processed_dir, "test.csv"), index=False)
        
        # Save dataset stats
        stats = {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "density": len(train_df) / (self.n_users * self.n_items) * 100,
        }
        
        stats_path = os.path.join(self.processed_dir, "stats.txt")
        with open(stats_path, "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        
        print(f"\nSaved processed data to {self.processed_dir}/")
        print(f"  Density: {stats['density']:.4f}%")
    
    def load_processed(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously processed data splits."""
        train_df = pd.read_csv(os.path.join(self.processed_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(self.processed_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(self.processed_dir, "test.csv"))
        
        self.n_users = max(train_df["user_id"].max(), val_df["user_id"].max(), test_df["user_id"].max()) + 1
        self.n_items = max(train_df["item_id"].max(), val_df["item_id"].max(), test_df["item_id"].max()) + 1
        
        print(f"Loaded processed data: {self.n_users} users, {self.n_items} items")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def process_pipeline(self, raw_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Full pipeline: load → binarize → filter → re-index → split → save.
        
        Args:
            raw_filepath: Path to the raw data file
            
        Returns:
            (train_df, val_df, test_df)
        """
        print(f"\n{'='*60}")
        print(f"  Processing: {self.dataset_name}")
        print(f"{'='*60}")
        
        df = self.load_raw_data(raw_filepath)
        df = self.binarize(df)
        df = self.k_core_filter(df)
        df = self.create_id_mappings(df)
        train_df, val_df, test_df = self.split_data(df)
        self.save_processed(train_df, val_df, test_df)
        
        return train_df, val_df, test_df
