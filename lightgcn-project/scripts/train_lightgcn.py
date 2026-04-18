import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import sys
import os
import copy
from pathlib import Path

# Ensure we can import from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    # Allow direct script execution while importing the package under src/.
    sys.path.append(str(SRC_ROOT))

from lightgcn_project.data.data_loader import BipartiteGraphLoader, BPRDataset
from lightgcn_project.models.lightgcn import LightGCN
from lightgcn_project.evaluation.metrics import hit_rate_at_k, ndcg_at_k


def resolve_data_path():
    """Resolve dataset path from env var or standard project locations."""
    env_path = os.getenv("LIGHTGCN_DATA_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        PROJECT_ROOT / "data" / "raw" / "u.data",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            # First valid path wins.
            return candidate

    raise FileNotFoundError(
        "Dataset file not found. Set LIGHTGCN_DATA_PATH or place u.data in data/raw/u.data"
    )

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


def evaluate(model, data_loader, device, k=20, eval_data=None):
    """
    Evaluates the model on the Test set calculating HR@K and NDCG@K.
    """
    model.eval()
    test_data = data_loader.test_data if eval_data is None else eval_data
    
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
    # Centralized experiment knobs for reproducibility and easy tuning.
    config = {
        'latent_dim': 128,   # Larger embedding space improves ranking quality on ML-100K
        'n_layers': 1,       # Shallow propagation often works better on this dataset
        'lr': 0.005,         # Higher learning rate from ablation trends
        'decay': 1e-4,       # L2 Regularization weight
        'batch_size': 2048,  # BPR batch size
        'epochs': 8,         # Best value from exhaustive grid search
        'early_stop_patience': 3  # Early stop based on validation NDCG@20
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Load Data ---
    # Data resolver supports environment override and project default.
    filepath = resolve_data_path()
    
    if not os.path.exists(filepath):
        print(f"Warning: Data file {filepath} not found. Please ensure your MovieLens data is there.")
        return

    loader = BipartiteGraphLoader(str(filepath), threshold=1.0)
    loader.load_raw_csv(str(filepath), val_ratio=0.1)
    use_validation = len(loader.val_data) > 0

    bpr_dataset = BPRDataset(loader, sampling_strategy="mixed", popular_prob=0.7)
    bpr_loader = DataLoader(bpr_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # --- 2. Initialize Model & Optimizer ---
    # Build model and optimization objective for pairwise ranking.
    model = LightGCN(loader.n_users, loader.n_items, loader.norm_adj.to(device), config).to(device)
    bpr_loss_fn = BPRLoss(decay=config['decay'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    print(f"\n--- Starting LightGCN Training ---")
    
    # --- 3. Training Loop ---
    # Track best checkpoint on validation NDCG@20.
    best_ndcg = 0.0
    best_hr = 0.0
    best_epoch = 0
    stale_evals = 0
    best_state = copy.deepcopy(model.state_dict())
    
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
        
        # Evaluate periodically to monitor overfitting and trigger early stop.
        if epoch % 5 == 0 or epoch == 1:
            eval_data = loader.val_data if use_validation else loader.test_data
            hr, ndcg = evaluate(model, loader, device, k=20, eval_data=eval_data)
            split_name = "Val" if use_validation else "Test"
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | {split_name} HR@20: {hr:.4f} | {split_name} NDCG@20: {ndcg:.4f} | Time: {time.time()-start_time:.1f}s")
            
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_hr = hr
                best_epoch = epoch
                stale_evals = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale_evals += 1

            if stale_evals >= config['early_stop_patience']:
                print(f"Early stopping triggered at epoch {epoch:03d}.")
                break
        else:
            print(f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")

    # Restore best checkpoint before final test report.
    model.load_state_dict(best_state)
    test_hr, test_ndcg = evaluate(model, loader, device, k=20, eval_data=loader.test_data)
    print(f"Best checkpoint -> Epoch {best_epoch:03d} | Val HR@20: {best_hr:.4f} | Val NDCG@20: {best_ndcg:.4f}")
    print(f"Final Test -> HR@20: {test_hr:.4f} | NDCG@20: {test_ndcg:.4f}")

if __name__ == "__main__":
    train()
