import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import BipartiteGraphLoader, BPRDataset, EarlyStopping
from src.models.lightgcn import LightGCN
from src.utils.metrics import hit_rate_at_k, ndcg_at_k

# Re-using BPRLoss and evaluate from train_lightgcn.py logic
# For a production/large project, these would be in src/utils/
class BPRLoss(nn.Module):
    def __init__(self, decay=1e-4):
        super(BPRLoss, self).__init__()
        self.decay = decay

    def forward(self, pos_scores, neg_scores, u_emb, pos_emb, neg_emb):
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        reg_loss = (1/2) * (u_emb.norm(2).pow(2) + 
                            pos_emb.norm(2).pow(2) + 
                            neg_emb.norm(2).pow(2)) / float(len(pos_scores))
        return loss + self.decay * reg_loss

def evaluate(model, data_loader, device, k=20):
    model.eval()
    test_data = data_loader.test_data
    test_dict = {}
    for u, i in test_data:
        if u not in test_dict:
            test_dict[u] = set()
        test_dict[u].add(i)
        
    users = list(test_dict.keys())
    batch_size = 1024
    hr_list = []
    ndcg_list = []
    
    with torch.no_grad():
        for i in range(0, len(users), batch_size):
            batch_users = users[i:i+batch_size]
            batch_users_t = torch.Tensor(batch_users).long().to(device)
            rating_preds = model.get_users_rating(batch_users_t).cpu().numpy()
            
            for idx, user in enumerate(batch_users):
                # Using sparse matrix directly to avoid .toarray() on huge matrices
                train_items = data_loader.user_item_net[user].indices
                rating_preds[idx][train_items] = -np.inf
                ranked_items = np.argsort(-rating_preds[idx])
                test_items = test_dict[user]
                hr_list.append(hit_rate_at_k(ranked_items, test_items, k))
                ndcg_list.append(ndcg_at_k(ranked_items, test_items, k))
    return np.mean(hr_list), np.mean(ndcg_list)

def train():
    # --- Configuration for ML-20M ---
    config = {
        'latent_dim': 64,
        'n_layers': 3,
        'lr': 0.001,
        'decay': 1e-4,
        'batch_size': 4096, # Larger batch size for larger dataset
        'epochs': 50
    }
    
    # CPU is more stable for large sparse graph operations on Mac
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    filepath = "data/ml-20m/ratings.csv"
    
    if not os.path.exists(filepath):
        print(f"Error: Data file {filepath} not found.")
        return

    print(f"Processing MovieLens 20M dataset...")
    # Threshold 4.0 keeps the density manageable on a personal machine
    loader = BipartiteGraphLoader(filepath, threshold=4.0) 
    loader.load_raw_csv(filepath)

    bpr_dataset = BPRDataset(loader, mode='train')
    bpr_loader = DataLoader(bpr_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    
    # --- 2. Initialize Model ---
    model = LightGCN(loader.n_users, loader.n_items, loader.norm_adj.to(device), config).to(device)
    bpr_loss_fn = BPRLoss(decay=config['decay'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=5)

    print(f"\n--- Starting LightGCN Training on ML-20M ---")
    
    best_ndcg = 0.0
    total_start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        # Use tqdm for a live progress bar with batch time and ETA
        pbar = tqdm.tqdm(bpr_loader, desc=f"Epoch {epoch:03d}", unit="batch")
        for batch_u, batch_pos, batch_neg in pbar:
            batch_u = batch_u.long().to(device)
            batch_pos = batch_pos.long().to(device)
            batch_neg = batch_neg.long().to(device)
            
            pos_scores, neg_scores, u_emb, pos_emb, neg_emb = model(batch_u, batch_pos, batch_neg)
            loss = bpr_loss_fn(pos_scores, neg_scores, u_emb, pos_emb, neg_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(bpr_loader)
        epoch_time = time.time() - start_time
        avg_epoch_time = (time.time() - total_start_time) / epoch
        est_remaining = avg_epoch_time * (config['epochs'] - epoch)
        
        print(f"\nEpoch {epoch:03d} Summary | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Est. Remaining: {est_remaining/60:.1f}m")        
        # Evaluate periodically (using a subset of users for speed on ML-20M)
        if epoch % 2 == 0 or epoch == 1:
            # Sample 2000 users for validation to keep it fast
            val_users = np.unique(loader.val_data[:, 0])
            if len(val_users) > 2000:
                sampled_val_users = np.random.choice(val_users, 2000, replace=False)
                # Filter validation data for these users
                mask = np.isin(loader.val_data[:, 0], sampled_val_users)
                sampled_val_data = loader.val_data[mask]
            else:
                sampled_val_data = loader.val_data
            
            tmp_loader = type('TmpLoader', (), {'test_data': sampled_val_data, 'user_item_net': loader.user_item_net})
            hr, ndcg = evaluate(model, tmp_loader, device, k=20)
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Val HR@20: {hr:.4f} | Val NDCG@20: {ndcg:.4f} | Time: {time.time()-start_time:.1f}s")
            scheduler.step(ndcg)
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                early_stopping.counter = 0
                torch.save(model.state_dict(), "best_lightgcn_ml20m.pth")
            else:
                early_stopping.counter += 1
            if early_stopping.counter >= early_stopping.patience:
                print("Early stopping triggered.")
                break
        else:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    train()
