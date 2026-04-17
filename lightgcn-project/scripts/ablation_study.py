"""
ablation_study.py — The most scoring section of the project.

Runs systematic experiments to answer:
1. How many GCN layers (K) are optimal? (K=1, 2, 3, 4)
2. Does embedding dimension matter? (32, 64, 128)
3. Does LightGCN's layer combination strategy matter? (mean vs weighted vs last)
4. Best hyperparameters to beat BPR-MF on ML-100K

Outputs clean tables for the final PDF submission.
"""

import sys
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import BipartiteGraphLoader, BPRDataset
from src.utils.metrics import hit_rate_at_k, ndcg_at_k
from src.models.lightgcn import LightGCN


# ============================================================
# Extended LightGCN with configurable layer combination
# ============================================================

class LightGCN_Ablation(nn.Module):
    """LightGCN with configurable layer combination strategy for ablation."""

    def __init__(self, n_users, n_items, norm_adj, config):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']
        # "mean" (paper default), "weighted", "last"
        self.combine = config.get('combine', 'mean')

        self.embedding_user = nn.Embedding(n_users, self.latent_dim)
        self.embedding_item = nn.Embedding(n_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # Learnable layer weights for "weighted" combination
        if self.combine == 'weighted':
            self.layer_weights = nn.Parameter(torch.ones(self.n_layers + 1) / (self.n_layers + 1))

    def computer(self):
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]

        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)  # (N, K+1, D)

        if self.combine == 'mean':
            # Paper default: simple average of all layer embeddings
            light_out = torch.mean(embs, dim=1)
        elif self.combine == 'weighted':
            # Learnable weighted sum
            weights = torch.softmax(self.layer_weights, dim=0)
            light_out = torch.sum(embs * weights.view(1, -1, 1), dim=1)
        elif self.combine == 'last':
            # Only use the final layer embedding (no combination)
            light_out = embs[:, -1, :]
        else:
            light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        u_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)

        u_emb_0 = self.embedding_user(users)
        pos_emb_0 = self.embedding_item(pos_items)
        neg_emb_0 = self.embedding_item(neg_items)

        return pos_scores, neg_scores, u_emb_0, pos_emb_0, neg_emb_0

    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        return torch.matmul(users_emb, all_items.t())


# ============================================================
# BPR Loss
# ============================================================

class BPRLoss(nn.Module):
    def __init__(self, decay=1e-4):
        super().__init__()
        self.decay = decay

    def forward(self, pos_scores, neg_scores, u_emb, pos_emb, neg_emb):
        loss = -torch.mean(nn.functional.logsigmoid(pos_scores - neg_scores))
        reg = (1/2) * (u_emb.norm(2).pow(2) +
                       pos_emb.norm(2).pow(2) +
                       neg_emb.norm(2).pow(2)) / float(len(pos_scores))
        return loss + self.decay * reg


# ============================================================
# Shared helpers
# ============================================================

def build_test_dict(test_data):
    d = defaultdict(set)
    for u, i in test_data:
        d[int(u)].add(int(i))
    return dict(d)


def evaluate_model(model, loader, device, k_values=[5, 10, 20]):
    model.eval()
    test_dict = build_test_dict(loader.test_data)

    results = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})

    with torch.no_grad():
        users = list(test_dict.keys())
        for i in range(0, len(users), 1024):
            batch = users[i:i+1024]
            batch_t = torch.tensor(batch).long().to(device)
            preds = model.get_users_rating(batch_t).cpu().numpy()

            for idx, user in enumerate(batch):
                train_items = np.where(loader.user_item_net[user].toarray()[0] == 1)[0]
                preds[idx][train_items] = -np.inf
                ranked = np.argsort(-preds[idx])

                test_items = test_dict[user]
                for k in k_values:
                    results[f"HR@{k}"].append(hit_rate_at_k(ranked, test_items, k))
                    results[f"NDCG@{k}"].append(ndcg_at_k(ranked, test_items, k))

    return {name: float(np.mean(vals)) for name, vals in results.items()}


def train_and_evaluate(config, loader, bpr_loader, device, epochs=100):
    """Train one LightGCN variant and return its metrics."""
    model = LightGCN_Ablation(
        loader.n_users, loader.n_items, loader.norm_adj.to(device), config
    ).to(device)

    loss_fn = BPRLoss(decay=config.get('decay', 1e-4))
    optimizer = optim.Adam(model.parameters(), lr=config.get('lr', 0.001))

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_u, batch_pos, batch_neg in bpr_loader:
            batch_u = batch_u.long().to(device)
            batch_pos = batch_pos.long().to(device)
            batch_neg = batch_neg.long().to(device)

            pos_s, neg_s, u_e, p_e, n_e = model(batch_u, batch_pos, batch_neg)
            loss = loss_fn(pos_s, neg_s, u_e, p_e, n_e)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    results = evaluate_model(model, loader, device)
    return results


# ============================================================
# Main ablation runner
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    filepath = "data/movielens-1m/ml-100k 4/u.data"
    loader = BipartiteGraphLoader(filepath, threshold=1.0)
    loader.load_raw_csv(filepath)

    bpr_dataset = BPRDataset(loader)
    bpr_loader = DataLoader(bpr_dataset, batch_size=2048, shuffle=True, drop_last=True)

    os.makedirs("results/tables", exist_ok=True)
    all_ablation_results = []

    # =========================================================
    # ABLATION 1: Number of GCN Layers (K=1, 2, 3, 4)
    # This is the CORE ablation of the LightGCN paper
    # =========================================================
    print("=" * 60)
    print("  ABLATION 1: Number of Graph Convolution Layers (K)")
    print("=" * 60)

    for n_layers in [1, 2, 3, 4]:
        config = {'latent_dim': 64, 'n_layers': n_layers, 'combine': 'mean',
                  'lr': 0.001, 'decay': 1e-4}
        print(f"\n  Training LightGCN with K={n_layers} layers...")
        t0 = time.time()
        results = train_and_evaluate(config, loader, bpr_loader, device, epochs=100)
        elapsed = time.time() - t0

        row = {'experiment': f'K={n_layers} layers', **results, 'time': f'{elapsed:.1f}s'}
        all_ablation_results.append(row)

        print(f"    HR@10={results['HR@10']:.4f}  NDCG@10={results['NDCG@10']:.4f}  "
              f"HR@20={results['HR@20']:.4f}  NDCG@20={results['NDCG@20']:.4f}  ({elapsed:.1f}s)")

    # =========================================================
    # ABLATION 2: Embedding Dimension (32, 64, 128)
    # =========================================================
    print("\n" + "=" * 60)
    print("  ABLATION 2: Embedding Dimension")
    print("=" * 60)

    for dim in [32, 64, 128]:
        config = {'latent_dim': dim, 'n_layers': 3, 'combine': 'mean',
                  'lr': 0.001, 'decay': 1e-4}
        print(f"\n  Training LightGCN with dim={dim}...")
        t0 = time.time()
        results = train_and_evaluate(config, loader, bpr_loader, device, epochs=100)
        elapsed = time.time() - t0

        row = {'experiment': f'dim={dim}', **results, 'time': f'{elapsed:.1f}s'}
        all_ablation_results.append(row)

        print(f"    HR@10={results['HR@10']:.4f}  NDCG@10={results['NDCG@10']:.4f}  "
              f"HR@20={results['HR@20']:.4f}  NDCG@20={results['NDCG@20']:.4f}  ({elapsed:.1f}s)")

    # =========================================================
    # ABLATION 3: Layer Combination Strategy
    # mean (paper) vs weighted vs last-layer-only
    # =========================================================
    print("\n" + "=" * 60)
    print("  ABLATION 3: Layer Combination Strategy")
    print("=" * 60)

    for combine in ['mean', 'weighted', 'last']:
        config = {'latent_dim': 64, 'n_layers': 3, 'combine': combine,
                  'lr': 0.001, 'decay': 1e-4}
        print(f"\n  Training LightGCN with combine={combine}...")
        t0 = time.time()
        results = train_and_evaluate(config, loader, bpr_loader, device, epochs=100)
        elapsed = time.time() - t0

        row = {'experiment': f'combine={combine}', **results, 'time': f'{elapsed:.1f}s'}
        all_ablation_results.append(row)

        print(f"    HR@10={results['HR@10']:.4f}  NDCG@10={results['NDCG@10']:.4f}  "
              f"HR@20={results['HR@20']:.4f}  NDCG@20={results['NDCG@20']:.4f}  ({elapsed:.1f}s)")

    # =========================================================
    # ABLATION 4: Learning Rate
    # =========================================================
    print("\n" + "=" * 60)
    print("  ABLATION 4: Learning Rate")
    print("=" * 60)

    for lr in [0.0005, 0.001, 0.005]:
        config = {'latent_dim': 64, 'n_layers': 3, 'combine': 'mean',
                  'lr': lr, 'decay': 1e-4}
        print(f"\n  Training LightGCN with lr={lr}...")
        t0 = time.time()
        results = train_and_evaluate(config, loader, bpr_loader, device, epochs=100)
        elapsed = time.time() - t0

        row = {'experiment': f'lr={lr}', **results, 'time': f'{elapsed:.1f}s'}
        all_ablation_results.append(row)

        print(f"    HR@10={results['HR@10']:.4f}  NDCG@10={results['NDCG@10']:.4f}  "
              f"HR@20={results['HR@20']:.4f}  NDCG@20={results['NDCG@20']:.4f}  ({elapsed:.1f}s)")

    # =========================================================
    # Print Final Ablation Table
    # =========================================================
    print("\n\n")
    print("=" * 90)
    print("  COMPLETE ABLATION RESULTS — LightGCN on MovieLens 100K")
    print("=" * 90)

    header = f"{'Experiment':<25}  {'HR@5':>7}  {'HR@10':>7}  {'HR@20':>7}  {'NDCG@5':>7}  {'NDCG@10':>8}  {'NDCG@20':>8}  {'Time':>7}"
    print(header)
    print("-" * 90)

    for row in all_ablation_results:
        line = (f"{row['experiment']:<25}  "
                f"{row['HR@5']:>7.4f}  {row['HR@10']:>7.4f}  {row['HR@20']:>7.4f}  "
                f"{row['NDCG@5']:>7.4f}  {row['NDCG@10']:>8.4f}  {row['NDCG@20']:>8.4f}  "
                f"{row['time']:>7}")
        print(line)

    print("=" * 90)

    # ---- Save to CSV ----
    csv_path = "results/tables/ablation_ml100k.csv"
    fieldnames = ['experiment', 'HR@5', 'HR@10', 'HR@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'time']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_ablation_results:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})

    print(f"\nAblation results saved to {csv_path}")


if __name__ == "__main__":
    main()
