"""
ItemKNN Baseline — Item-based K-Nearest Neighbors Collaborative Filtering.
Recommends items similar to what the user has interacted with.
Uses cosine similarity between item interaction vectors.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemKNN:
    def __init__(self, k=50):
        """
        Args:
            k: Number of nearest neighbor items to consider
        """
        self.k = k
        self.item_similarity = None
        self.user_item_matrix = None
    
    def fit(self, train_data, n_users, n_items):
        """
        Build item-item similarity matrix from training interactions.
        
        Args:
            train_data: np.array of shape (N, 2) — [user_id, item_id]
            n_users: Total users
            n_items: Total items
        """
        # Build sparse user-item interaction matrix
        users = train_data[:, 0].astype(int)
        items = train_data[:, 1].astype(int)
        values = np.ones(len(users), dtype=np.float32)
        
        self.user_item_matrix = csr_matrix(
            (values, (users, items)), shape=(n_users, n_items)
        )
        
        # Compute item-item cosine similarity  
        # item_matrix is (n_items, n_users) — each row is an item's user-interaction vector
        item_matrix = self.user_item_matrix.T  
        self.item_similarity = cosine_similarity(item_matrix, dense_output=False)
        
        # Zero out self-similarity (diagonal)
        self.item_similarity.setdiag(0)
        
        # Keep only top-K neighbors per item for speed
        # Convert to dense for top-k selection, then back to sparse
        sim_dense = self.item_similarity.toarray()
        for i in range(n_items):
            row = sim_dense[i]
            # Zero out everything except top-k
            if np.count_nonzero(row) > self.k:
                threshold = np.partition(row, -self.k)[-self.k]
                row[row < threshold] = 0
        
        self.item_similarity = csr_matrix(sim_dense)
        print(f"ItemKNN: Built similarity matrix with k={self.k}")
    
    def predict(self, user_ids, n_items):
        """
        For each user, score = user_interactions · item_similarity_matrix.
        
        Args:
            user_ids: list of user IDs
            n_items: total items
            
        Returns:
            (len(user_ids), n_items) score matrix
        """
        scores = np.zeros((len(user_ids), n_items), dtype=np.float32)
        
        for idx, uid in enumerate(user_ids):
            user_vec = self.user_item_matrix[uid]  # (1, n_items) sparse
            # score for each item = sum of similarities to items user has interacted with
            user_scores = user_vec.dot(self.item_similarity).toarray().flatten()
            scores[idx] = user_scores
        
        return scores
