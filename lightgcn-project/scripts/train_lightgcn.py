import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import BipartiteGraphLoader, BPRDataset
from src.models.lightgcn import LightGCN
from src.utils.metrics import hit_rate_at_k, ndcg_at_k

class BPRLoss(nn.Module):
    def __init__(self, decay=1e-4):
        """
        Bayesian Personalized Ranking (BPR) Loss.
        Optimizes for the formulation: pos_score > neg_score
        """
        super(BPRLoss, self).__init__()
        self.decay = decay

    def forward(self, pos_scores, neg_scores, u_emb, pos_emb, neg_emb):
        """
        Calculates the BPR Loss + L2 Regularization.
        """
        # BPR Loss: -ln(sigmoid(pos - neg))
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        
        # L2 Regularization (Weight Decay) to prevent overfitting
        reg_loss = (1/2) * (u_emb.norm(2).pow(2) + 
                            pos_emb.norm(2).pow(2) + 
                            neg_emb.norm(2).pow(2)) / float(len(pos_scores))
                            
        return loss + self.decay * reg_loss


def evaluate(model, data_loader, device, k=20):
    """
    Evaluates the model on the Test set calculating HR@K and NDCG@K.
    """
    model.eval()
    test_data = data_loader.test_data
    
    # Create dict of true items per user for fast lookup
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
            
            # Predict scores for all items
            rating_preds = model.get_users_rating(batch_users_t).cpu().numpy()
            
            # Mask training interactions so we don't recommend past items
            for idx, user in enumerate(batch_users):
                train_items = np.where(data_loader.user_item_net[user].toarray()[0] == 1)[0]
                rating_preds[idx][train_items] = -np.inf
                
                # Rank items 
                ranked_items = np.argsort(-rating_preds[idx])
                
                # Calculate metrics
                test_items = test_dict[user]
                hr_list.append(hit_rate_at_k(ranked_items, test_items, k))
                ndcg_list.append(ndcg_at_k(ranked_items, test_items, k))

    return np.mean(hr_list), np.mean(ndcg_list)


def train():
    # --- Configuration ---
    config = {
        'latent_dim': 64,    # Size of the embeddings
        'n_layers': 3,       # LightGCN layers (K=3 is optimal in the paper)
        'lr': 0.001,         # Learning rate
        'decay': 1e-4,       # L2 Regularization weight
        'batch_size': 2048,  # BPR batch size
        'epochs': 100        # Number of training epochs
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Update this path to wherever your movielens 1m ratings.dat or ratings.csv is stored.
    # We'll adapt it directly to your "ml-100k 4" folder
    filepath = "data/movielens-1m/ml-100k 4/u.data"
    
    if not os.path.exists(filepath):
        print(f"Warning: Data file {filepath} not found. Please ensure your MovieLens data is there.")
        return

    loader = BipartiteGraphLoader(filepath, threshold=1.0)
    loader.load_raw_csv(filepath)

    bpr_dataset = BPRDataset(loader)
    bpr_loader = DataLoader(bpr_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # --- 2. Initialize Model & Optimizer ---
    model = LightGCN(loader.n_users, loader.n_items, loader.norm_adj.to(device), config).to(device)
    bpr_loss_fn = BPRLoss(decay=config['decay'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    print(f"\n--- Starting LightGCN Training ---")
    
    # --- 3. Training Loop ---
    best_ndcg = 0.0
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for batch_u, batch_pos, batch_neg in bpr_loader:
            batch_u = batch_u.long().to(device)
            batch_pos = batch_pos.long().to(device)
            batch_neg = batch_neg.long().to(device)
            
            # Forward pass
            pos_scores, neg_scores, u_emb, pos_emb, neg_emb = model(batch_u, batch_pos, batch_neg)
            
            # Loss calculation
            loss = bpr_loss_fn(pos_scores, neg_scores, u_emb, pos_emb, neg_emb)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(bpr_loader)
        
        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            hr, ndcg = evaluate(model, loader, device, k=20)
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | HR@20: {hr:.4f} | NDCG@20: {ndcg:.4f} | Time: {time.time()-start_time:.1f}s")
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                # Optional: Save model checkpoint
                # torch.save(model.state_dict(), "best_lightgcn.pth")
        else:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")

if __name__ == "__main__":
    train()
