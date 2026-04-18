"""
BPR-MF Baseline — Bayesian Personalized Ranking with Matrix Factorization.

This is THE critical baseline for LightGCN. LightGCN is essentially BPR-MF
with graph-based embedding propagation on top. If LightGCN doesn't beat BPR-MF,
the graph convolution layers aren't adding value.
"""

import torch
import torch.nn as nn
import numpy as np


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, latent_dim=64):
        """
        Standard Matrix Factorization with BPR loss.
        
        Args:
            n_users: Total users
            n_items: Total items
            latent_dim: Embedding dimension
        """
        super(BPRMF, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.latent_dim = latent_dim
        
        # User and Item embedding tables — this IS matrix factorization
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.item_embedding = nn.Embedding(n_items, latent_dim)
        
        # Keep initialization consistent with LightGCN to make comparison fair.
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, users, pos_items, neg_items):
        """
        BPR forward pass.
        
        Args:
            users: (batch_size,) tensor of user IDs
            pos_items: (batch_size,) tensor of positive item IDs
            neg_items: (batch_size,) tensor of negative item IDs
            
        Returns:
            pos_scores, neg_scores, user/item embeddings for regularization
        """
        # Look up trainable latent vectors for each triple component.
        u_emb = self.user_embedding(users)
        pos_emb = self.item_embedding(pos_items)
        neg_emb = self.item_embedding(neg_items)
        
        # Dot products produce preference scores.
        # BPR loss will enforce pos_scores > neg_scores.
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        
        return pos_scores, neg_scores, u_emb, pos_emb, neg_emb
    
    def get_all_ratings(self, users):
        """
        Get predicted scores for all items for a batch of users.
        
        Args:
            users: (batch_size,) tensor of user IDs
            
        Returns:
            (batch_size, n_items) score matrix
        """
        # Compute all item scores for a user batch in one matrix multiply.
        u_emb = self.user_embedding(users)
        all_items = self.item_embedding.weight
        ratings = torch.matmul(u_emb, all_items.t())
        return ratings
