"""
MostPop Baseline — Recommends the most globally popular items to everyone.
This is the simplest, dumbest baseline. If LightGCN can't beat this, something is wrong.
"""

import numpy as np


class MostPopular:
    def __init__(self):
        self.item_popularity = None
        self.n_items = 0

    def fit(self, train_data, n_items):
        """
        Count how many users interacted with each item.
        
        Args:
            train_data: np.array of shape (N, 2) — [user_id, item_id]
            n_items: Total number of items
        """
        self.n_items = n_items
        self.item_popularity = np.zeros(n_items, dtype=np.float32)
        
        for _, item_id in train_data:
            self.item_popularity[int(item_id)] += 1.0

    def predict(self, user_ids, n_items):
        """
        Returns the same popularity-based scores for every user.
        
        Args:
            user_ids: list of user IDs to predict for
            n_items: total items
            
        Returns:
            (len(user_ids), n_items) score matrix
        """
        # Every user gets the exact same popularity scores
        scores = np.tile(self.item_popularity, (len(user_ids), 1))
        return scores
