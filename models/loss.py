"""
Bayesian Personalized Ranking (BPR) Loss
Paper: https://arxiv.org/abs/1205.2618

Why BPR for recommendation?
- Standard cross-entropy treats all unseen items as negative (wrong!)
- BPR uses PAIRWISE ranking: "user should prefer item i over item j"
- Much better signal for implicit feedback (clicks, purchases, views)

The loss:
    L_BPR = -sum log(sigmoid(score(u,i) - score(u,j))) + lambda * ||params||^2
"""

import torch
import torch.nn.functional as F


def bpr_loss(user_emb, pos_item_emb, neg_item_emb,
             user_emb_0, pos_item_emb_0, neg_item_emb_0,
             reg_lambda=1e-4):
    """
    Compute BPR loss with L2 regularization on initial (layer-0) embeddings.
    
    Regularizing layer-0 embeddings (not final) is the correct approach
    for LightGCN -- prevents overfitting the learned parameters directly.
    """
    pos_scores = (user_emb * pos_item_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

    bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

    reg = (user_emb_0.norm(2).pow(2) +
           pos_item_emb_0.norm(2).pow(2) +
           neg_item_emb_0.norm(2).pow(2)) / (2 * len(user_emb))

    total = bpr + reg_lambda * reg
    return total, bpr.item(), reg.item()


def sample_negative_items(pos_item_ids, num_items, num_neg=1):
    """Uniform negative sampling."""
    neg_items = torch.randint(0, num_items,
                              (len(pos_item_ids) * num_neg,),
                              device=pos_item_ids.device)
    return neg_items
