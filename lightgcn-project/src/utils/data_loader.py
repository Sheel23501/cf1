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
        self.val_data = []
        self.test_data = []
        self.user_item_net = None # Sparse matrix
        
    def load_raw_csv(self, filepath, val_ratio=0.1, test_ratio=0.2, seed=42):
        """
        Loads raw data, binarizes, remaps IDs, and splits into Train/Val/Test.
        """
        print(f"Loading data from {filepath}...")
        # Detect if it's a CSV with header or a space/tab separated file
        try:
            # Try to read the first few lines to check for header
            first_rows = pd.read_csv(filepath, nrows=5, sep=None, engine='python')
            header_exists = 'userId' in first_rows.columns or 'user_id' in first_rows.columns
            
            if header_exists:
                df = pd.read_csv(filepath, sep=',', usecols=[0, 1, 2], names=["user_id", "item_id", "rating"], header=0)
            else:
                df = pd.read_csv(filepath, sep=None, engine='python', header=None, 
                                 names=["user_id", "item_id", "rating"], usecols=[0, 1, 2])
        except Exception as e:
            print(f"Error reading file with auto-detection: {e}. Falling back to default.")
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
        
        # Per-user train/val/test split
        np.random.seed(seed)
        train_list = []
        val_list = []
        test_list = []
        
        for user_id, group in df.groupby('user_id'):
            items = group['item_id'].values.copy()
            np.random.shuffle(items)
            
            n_test = max(1, int(len(items) * test_ratio))
            n_val = max(1, int(len(items) * val_ratio))
            
            test_items = items[:n_test]
            val_items = items[n_test:n_test+n_val]
            train_items = items[n_test+n_val:]
            
            for item in train_items:
                train_list.append([user_id, item])
            for item in val_items:
                val_list.append([user_id, item])
            for item in test_items:
                test_list.append([user_id, item])
                
        self.train_data = np.array(train_list)
        self.val_data = np.array(val_list)
        self.test_data = np.array(test_list)
        
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        print(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        
        self._build_sparse_graph()
        
    def _build_sparse_graph(self):
        """
        Builds the normalized adjacency matrix for LightGCN.
        A = [0   R]
            [R.T 0]
        Uses efficient COO format to avoid OOM on large datasets.
        """
        print("Building bipartite sparse graph...")
        train_u = self.train_data[:, 0]
        train_i = self.train_data[:, 1]
        
        R = sp.coo_matrix((np.ones(len(train_u)), (train_u, train_i)), 
                          shape=(self.n_users, self.n_items))
        
        self.user_item_net = R.tocsr()
        
        # Build the large symmetric adjacency matrix directly using COO indices
        # Top-right is R, Bottom-left is R.T
        u_indices = train_u
        i_indices = train_i + self.n_users
        
        row = np.concatenate([u_indices, i_indices])
        col = np.concatenate([i_indices, u_indices])
        data = np.ones(len(row), dtype=np.float32)
        
        adj_mat = sp.coo_matrix((data, (row, col)), 
                                shape=(self.n_users + self.n_items, self.n_users + self.n_items))
        
        # Normalize: D^-1/2 * A * D^-1/2
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        self.norm_adj = self._convert_sp_mat_to_tensor(norm_adj.tocsr())
        print("Graph building completed.")
        
    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

class BPRDataset(Dataset):
    def __init__(self, data_loader, mode='train'):
        if mode == 'train':
            data = data_loader.train_data
        elif mode == 'val':
            data = data_loader.val_data
        else:
            data = data_loader.test_data
            
        self.users = data[:, 0]
        self.pos_items = data[:, 1]
        self.user_item_net = data_loader.user_item_net
        self.n_items = data_loader.n_items
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        u = self.users[idx]
        pos_i = self.pos_items[idx]
        
        # Negative sampling
        neg_i = np.random.randint(0, self.n_items)
        while self.user_item_net[u, neg_i] == 1:
            neg_i = np.random.randint(0, self.n_items)
            
        return u, pos_i, neg_i

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
