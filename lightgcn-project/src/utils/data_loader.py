import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class BipartiteGraphLoader:
    def __init__(self, dataset_path, threshold=1.0):
        """
        Loads dataset, binarizes ratings, and creates sparse adjacency matrix.
        Assuming dataset_path has train.txt and test.txt, or we process raw csv.
        For simplicity, we assume a raw csv: user_id, item_id, rating
        """
        self.dataset_path = dataset_path
        self.threshold = threshold
        
        # Internal states
        self.n_users = 0
        self.n_items = 0
        self.train_data = []
        self.test_data = []
        self.user_item_net = None # Sparse matrix
        
    def load_raw_csv(self, filepath, test_ratio=0.2, seed=42):
        """
        Loads raw data, binarizes, remaps IDs, and splits into Train/Test.
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, sep=None, engine='python', header=None,
                         names=["user_id", "item_id", "rating"], usecols=[0, 1, 2])
        
        # Binarize
        print(f"Binarizing with threshold >= {self.threshold}")
        df = df[df['rating'] >= self.threshold].copy()
        df['rating'] = 1.0
        
        # Remap IDs to contiguous integers
        user_ids = df['user_id'].unique()
        item_ids = df['item_id'].unique()
        
        self.n_users = len(user_ids)
        self.n_items = len(item_ids)
        
        user2id = {u: i for i, u in enumerate(user_ids)}
        item2id = {i: j for j, i in enumerate(item_ids)}
        
        df['user_id'] = df['user_id'].map(user2id)
        df['item_id'] = df['item_id'].map(item2id)
        
        # Per-user train/test split
        np.random.seed(seed)
        train_list = []
        test_list = []
        
        for user_id, group in df.groupby('user_id'):
            items = group['item_id'].values.copy()
            np.random.shuffle(items)
            
            n_test = max(1, int(len(items) * test_ratio))
            test_items = items[:n_test]
            train_items = items[n_test:]
            
            for item in train_items:
                train_list.append([user_id, item])
            for item in test_items:
                test_list.append([user_id, item])
                
        self.train_data = np.array(train_list)
        self.test_data = np.array(test_list)
        
        print(f"Users: {self.n_users}, Items: {self.n_items}")
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
        
        self.user_item_net = R.tocsr()
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.user_item_net.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize
        rowsum = np.array(adj_mat.sum(axis=1))
        
        # Avoid divide by zero
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        
        self.norm_adj = self._convert_sp_mat_to_tensor(norm_adj)
        print("Graph building completed.")
        
    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

class BPRDataset(Dataset):
    def __init__(self, data_loader):
        self.users = data_loader.train_data[:, 0]
        self.pos_items = data_loader.train_data[:, 1]
        self.user_item_net = data_loader.user_item_net
        self.n_items = data_loader.n_items
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        u = self.users[idx]
        pos_i = self.pos_items[idx]
        
        # Negative sampling (BPR Needs 1 positive, 1 negative per user)
        neg_i = np.random.randint(0, self.n_items)
        while self.user_item_net[u, neg_i] == 1:
            neg_i = np.random.randint(0, self.n_items)
            
        return u, pos_i, neg_i
