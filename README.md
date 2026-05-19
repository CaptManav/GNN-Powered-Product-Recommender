# 🛍️ GNN-Based Amazon Product Recommender

> **LightGCN implementation for product recommendations using Graph Neural Networks**  
> *PyTorch Geometric · BPR Loss · Recall@K · NDCG@K · Streamlit*

---

## 🧠 What is this?

A production-ready implementation of **LightGCN** (He et al., SIGIR 2020) for product recommendations on Amazon-style implicit feedback data.

Instead of traditional collaborative filtering, this models user-item interactions as a **bipartite graph** and uses GNN message passing to capture **multi-hop relationships**:

```
User A → Product X ← User B → Product Y
                               ↑
                          Recommend to User A!
```

This 2-hop reasoning is impossible with Matrix Factorization but trivial with GNNs.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                   LightGCN                      │
│                                                 │
│  Input: User/Item Embeddings [N, 64]            │
│                                                 │
│  Layer 0:  E⁰ = [E_user ║ E_item]               │
│  Layer 1:  E¹ = A_norm × E⁰     (graph conv)    │
│  Layer 2:  E² = A_norm × E¹     (graph conv)    │
│  Layer 3:  E³ = A_norm × E²     (graph conv)    │
│                                                 │
│  Final:    E = mean(E⁰, E¹, E², E³)             │
│  Score:    ŷ(u,i) = E_u · E_i                   │
│  Loss:     BPR (pairwise ranking)               │
└─────────────────────────────────────────────────┘
```

**Why LightGCN over vanilla GCN?**
- ❌ Removes weight matrices (unnecessary for CF)
- ❌ Removes non-linear activations (hurt performance empirically)  
- ✅ Keeps neighborhood aggregation (the key signal)
- ✅ 16.5% NDCG improvement over NGCF on Amazon-Book dataset

---

## 📁 Project Structure

```
gnn-recommender/
├── models/
│   ├── lightgcn.py       # LightGCN architecture (MessagePassing)
│   └── loss.py           # BPR loss + negative sampling
├── utils/
│   ├── data_loader.py    # Dataset generation, graph construction
│   └── metrics.py        # Recall@K, NDCG@K, Hit Rate@K
├── app/
│   └── streamlit_app.py  # Interactive web demo
├── notebooks/
│   └── GNN_Recommender_Walkthrough.ipynb  # Full tutorial
├── train.py              # Main training script
└── requirements.txt
```

---

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Train
python train.py

# Train with custom config
python train.py --epochs 50 --dim 128 --layers 4 --lr 0.001 --device cuda

# Launch web app
streamlit run app/streamlit_app.py

# Or run the notebook
jupyter notebook notebooks/GNN_Recommender_Walkthrough.ipynb
```

---

## 📊 Results (on synthetic Amazon Electronics data)

| Metric      | @10    | @20    |
|-------------|--------|--------|
| Recall      | ~0.08  | ~0.13  |
| NDCG        | ~0.06  | ~0.08  |
| Hit Rate    | ~0.25  | ~0.35  |

*Results on real Amazon dataset will vary. Run on Amazon Electronics subset for real benchmarks.*

---

## 🔑 Key Concepts

### 1. Implicit Feedback
Users don't give explicit ratings — we infer preferences from **interactions** (clicks, purchases, views). Ratings ≥ 4 are treated as positive interactions.

### 2. BPR Loss (Bayesian Personalized Ranking)
```
L = -log σ(score(u, i_pos) - score(u, i_neg)) + λ||Θ||²
```
Teaches the model to rank interacted items above random items.

### 3. Graph Construction
```python
# Bipartite graph: users (0..N-1) + items (N..N+M-1)
edge_index = torch.stack([
    torch.cat([users, items]),  # source
    torch.cat([items, users])   # target (bidirectional)
])
```

### 4. Evaluation Protocol
- **Leave-last-out:** each user's last item → test set
- **Recall@K:** fraction of relevant items in top-K
- **NDCG@K:** position-aware ranking quality

---

## 🏭 Real-World Usage

Companies using GNNs for recommendations:
- **Pinterest** — PinSage (GraphSAGE) for 3B+ pins
- **LinkedIn** — GNNs for job/connection recommendations
- **Uber Eats** — Restaurant recommendations
- **Alibaba** — LightGCN variant in production

---

## 📈 Extensions

| Extension | Complexity | Impact |
|-----------|-----------|--------|
| Real Amazon dataset | Low | High |
| Hard negative mining | Medium | Medium |
| Add item text features (BERT) | Medium | High |
| Contrastive learning (SGL/SimGCL) | High | High |
| Temporal dynamics | High | High |

---

## 📚 References

- [LightGCN: Simplifying and Powering GCN for Recommendation](https://arxiv.org/abs/2002.02126) - He et al., SIGIR 2020
- [BPR: Bayesian Personalized Ranking](https://arxiv.org/abs/1205.2618) - Rendle et al., UAI 2009
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)

---

*Built as a portfolio project demonstrating production-level ML engineering.*
