import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, norm_adj, config):
        """
        LightGCN Model Structure.
        Args:
            n_users (int): Total number of users.
            n_items (int): Total number of items.
            norm_adj (torch.sparse.FloatTensor): The normalized adjacency matrix (User-Item Bipartite graph).
            config (dict): Dictionary containing hyperparameters like latent_dim, n_layers.
        """
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.norm_adj = norm_adj
        self.latent_dim = config['latent_dim']
        self.n_layers = config['n_layers']
        
        # Initial Embeddings E_0
        self.embedding_user = nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)
        
        # Initialize weights
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
    def computer(self):
        """
        Propagates embeddings through the graph (The actual routing and aggregation).
        E^{(k+1)} = D^{-1/2} A D^{-1/2} E^{(k)}
        Final Embeddings = E^{(0)} + E^{(1)} + ... + E^{(K)}
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        # E_0
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        
        # Propagate through layers
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
            
        # Stack and average (LightGCN paper averages the embeddings of all layers)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items
    
    def forward(self, users, pos_items, neg_items):
        """
        Forward pass for BPR Training calculating the ratings for positive and negative samples.
        """
        all_users, all_items = self.computer()
        
        u_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        # Prediction via inner product
        pos_scores = torch.mul(u_emb, pos_emb).sum(dim=1)
        neg_scores = torch.mul(u_emb, neg_emb).sum(dim=1)
        
        # We also return initial embeddings for the L2 Regularization (weight decay)
        u_emb_0 = self.embedding_user(users)
        pos_emb_0 = self.embedding_item(pos_items)
        neg_emb_0 = self.embedding_item(neg_items)
        
        return pos_scores, neg_scores, u_emb_0, pos_emb_0, neg_emb_0

    def get_users_rating(self, users):
        """
        Get all item scores for a batch of users (used for evaluation).
        """
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
