"""
LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Paper: https://arxiv.org/abs/2002.02126

Why LightGCN over vanilla GCN?
- Removes feature transformation & non-linear activation (not needed for CF)
- Only keeps neighborhood aggregation — the key op for recommendations
- Used in production at Alibaba, Pinterest, and similar at LinkedIn/Google
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCNConv(MessagePassing):
    """
    Single LightGCN propagation layer.
    
    The propagation rule (from the paper):
        e_u^(k+1) = sum_{i in N(u)} e_i^(k) / sqrt(|N(u)| * |N(i)|)
    
    This is just normalized sum aggregation — no weights, no activation.
    The magic is in the multi-layer combination at the end.
    """

    def __init__(self):
        super().__init__(aggr='add')  # sum aggregation

    def forward(self, x, edge_index):
        # Compute normalization: 1 / sqrt(deg(u) * deg(v))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """
    Full LightGCN model for user-item recommendation.
    
    Architecture:
    1. Learn initial embeddings for all users & items
    2. Propagate through K graph conv layers
    3. Final embedding = mean of all layer outputs (layer combination)
    4. Score = dot product of user & item final embeddings
    
    Args:
        num_users: total number of unique users
        num_items: total number of unique items  
        embedding_dim: latent dimension size (64-128 is standard)
        num_layers: number of GCN layers (3-4 is optimal per paper)
        dropout: dropout rate on embeddings
    """

    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_layers=3, dropout=0.1):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Learnable initial embeddings — the only parameters in LightGCN
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # K graph conv layers
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index):
        """
        Full forward pass: propagate embeddings through the user-item graph.
        
        edge_index: [2, num_interactions] — the user-item interaction graph
                    (bidirectional: user->item AND item->user edges)
        
        Returns: final user embeddings, final item embeddings
        """
        # Concatenate user and item embeddings into one node feature matrix
        # Shape: [num_users + num_items, embedding_dim]
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)

        # Apply dropout to initial embeddings
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Collect embeddings from each layer for the final combination
        all_embeddings = [x]

        for conv in self.convs:
            x = conv(x, edge_index)
            all_embeddings.append(x)

        # Layer combination: simple mean across all layers
        # This is what gives LightGCN its multi-hop reasoning ability
        final_embedding = torch.stack(all_embeddings, dim=1).mean(dim=1)

        # Split back into user and item embeddings
        user_emb = final_embedding[:self.num_users]
        item_emb = final_embedding[self.num_users:]

        return user_emb, item_emb

    def predict(self, user_ids, item_ids, edge_index):
        """Predict interaction scores for given user-item pairs."""
        user_emb, item_emb = self.forward(edge_index)
        u = user_emb[user_ids]
        i = item_emb[item_ids]
        return (u * i).sum(dim=-1)  # dot product score

    def recommend(self, user_id, edge_index, k=10, exclude_seen=None):
        """
        Get top-K item recommendations for a user.
        
        exclude_seen: set of item_ids the user already interacted with
        """
        self.eval()
        with torch.no_grad():
            user_emb, item_emb = self.forward(edge_index)
            u = user_emb[user_id].unsqueeze(0)  # [1, dim]
            scores = (u @ item_emb.T).squeeze()  # [num_items]

            if exclude_seen:
                scores[list(exclude_seen)] = -float('inf')

            top_k = torch.topk(scores, k).indices.tolist()
        return top_k
