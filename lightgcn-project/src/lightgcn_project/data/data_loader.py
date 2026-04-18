import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset


class BipartiteGraphLoader:
    def __init__(self, dataset_path, threshold=1.0):
        """
        Load an implicit-feedback recommendation dataset and build graph tensors.

        The loader expects raw interaction rows in the format:
            user_id, item_id, rating

        It then:
        1. Binarizes by threshold (rating >= threshold becomes interaction=1)
        2. Remaps user/item IDs to contiguous [0..N-1] indices
        3. Splits per-user interactions into train/val/test
        4. Builds sparse user-item matrix and normalized adjacency for LightGCN
        """
        self.dataset_path = dataset_path
        self.threshold = threshold
        
        # These are populated after `load_raw_csv`.
        self.n_users = 0
        self.n_items = 0
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.user_item_net = None  # CSR matrix of train interactions (users x items)
        
    def load_raw_csv(self, filepath, test_ratio=0.2, val_ratio=0.0, seed=42):
        """
        Read raw interactions and produce train/val/test edges.

        Splitting strategy is user-wise so each user appears in test set.
        This is important for top-K recommendation evaluation.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, sep=None, engine='python', header=None,
                         names=["user_id", "item_id", "rating"], usecols=[0, 1, 2])
        
        # Binarize
        print(f"Binarizing with threshold >= {self.threshold}")
        df = df[df['rating'] >= self.threshold].copy()
        df['rating'] = 1.0
        
        # LightGCN embedding tables expect compact integer indices.
        user_ids = df['user_id'].unique()
        item_ids = df['item_id'].unique()
        
        self.n_users = len(user_ids)
        self.n_items = len(item_ids)
        
        user2id = {u: i for i, u in enumerate(user_ids)}
        item2id = {i: j for j, i in enumerate(item_ids)}
        
        df['user_id'] = df['user_id'].map(user2id)
        df['item_id'] = df['item_id'].map(item2id)
        
        # Per-user split keeps evaluation realistic and avoids user cold-start in test.
        np.random.seed(seed)
        train_list = []
        val_list = []
        test_list = []
        
        for user_id, group in df.groupby('user_id'):
            items = group['item_id'].values.copy()
            np.random.shuffle(items)
            
            # Ensure every user contributes at least one test interaction.
            n_test = max(1, int(len(items) * test_ratio))
            test_items = items[:n_test]
            remain_items = items[n_test:]

            n_val = 0
            if val_ratio > 0 and len(remain_items) > 1:
                # Keep at least one train item if validation is enabled.
                n_val = max(1, int(len(items) * val_ratio))
                n_val = min(n_val, len(remain_items) - 1)

            val_items = remain_items[:n_val]
            train_items = remain_items[n_val:]
            
            for item in train_items:
                train_list.append([user_id, item])
            for item in val_items:
                val_list.append([user_id, item])
            for item in test_items:
                test_list.append([user_id, item])
                
        # Final edge arrays are always shape (N, 2): [user_idx, item_idx]
        self.train_data = np.array(train_list, dtype=np.int64) if train_list else np.empty((0, 2), dtype=np.int64)
        self.val_data = np.array(val_list, dtype=np.int64) if val_list else np.empty((0, 2), dtype=np.int64)
        self.test_data = np.array(test_list, dtype=np.int64) if test_list else np.empty((0, 2), dtype=np.int64)
        
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        if len(self.val_data) > 0:
            print(f"Train edges: {len(self.train_data)}, Val edges: {len(self.val_data)}, Test edges: {len(self.test_data)}")
        else:
            print(f"Train edges: {len(self.train_data)}, Test edges: {len(self.test_data)}")
        
        self._build_sparse_graph()
        
    def _build_sparse_graph(self):
        """
        Builds the normalized adjacency matrix for LightGCN.
        A = [0   R]
            [R.T 0]
        """
        print("Building bipartite sparse graph...")
        train_u = self.train_data[:, 0]
        train_i = self.train_data[:, 1]
        
        R = sp.coo_matrix((np.ones(len(train_u)), (train_u, train_i)), 
                          shape=(self.n_users, self.n_items))
        
        # Keep train-only interactions for fast masking and negative checks.
        self.user_item_net = R.tocsr()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.user_item_net.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        # This is the propagation operator used in LightGCN.
        rowsum = np.array(adj_mat.sum(axis=1))
        
        # Nodes with zero degree get normalization weight 0.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        
        self.norm_adj = self._convert_sp_mat_to_tensor(norm_adj)
        print("Graph building completed.")
        
    def _convert_sp_mat_to_tensor(self, X):
        """Convert SciPy sparse matrix to PyTorch sparse COO tensor."""
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

class BPRDataset(Dataset):
    def __init__(self, data_loader, sampling_strategy="mixed", popular_prob=0.7):
        """
        Dataset wrapper that yields (user, positive_item, negative_item) triples.

        Negative sampling options:
        - mixed: mostly popularity-aware negatives + some uniform negatives
        - uniform fallback: random item sampling
        """
        self.users = data_loader.train_data[:, 0]
        self.pos_items = data_loader.train_data[:, 1]
        self.user_item_net = data_loader.user_item_net
        self.n_items = data_loader.n_items
        self.sampling_strategy = sampling_strategy
        self.popular_prob = popular_prob

        # Popularity distribution for popularity-aware negative sampling.
        item_counts = np.bincount(self.pos_items, minlength=self.n_items).astype(np.float64)
        item_counts = np.power(item_counts + 1e-8, 0.75)
        self.item_probs = item_counts / item_counts.sum()
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        # Positive pair comes directly from train interactions.
        u = self.users[idx]
        pos_i = self.pos_items[idx]

        # Try multiple times to draw a true negative (an item this user did not interact with).
        for _ in range(20):
            if self.sampling_strategy == "mixed" and np.random.rand() < self.popular_prob:
                # Harder negatives tend to come from popular items.
                neg_i = np.random.choice(self.n_items, p=self.item_probs)
            else:
                # Uniform negatives keep diversity in sampled triples.
                neg_i = np.random.randint(0, self.n_items)

            if self.user_item_net[u, neg_i] == 0:
                return u, pos_i, neg_i

        # Guaranteed fallback if all attempts hit known positives.
        neg_i = np.random.randint(0, self.n_items)
        while self.user_item_net[u, neg_i] == 1:
            neg_i = np.random.randint(0, self.n_items)
            
        return u, pos_i, neg_i
