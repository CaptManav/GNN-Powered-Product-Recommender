"""
Streamlit Web App for GNN-based Amazon Product Recommender
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightgcn import LightGCN
from utils.data_loader import (generate_synthetic_amazon_data,
                                train_val_test_split, build_graph,
                                get_user_positive_items)

st.set_page_config(
    page_title="GNN Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

CATEGORIES = ['Electronics', 'Computers', 'Cameras', 'Phones', 'Audio']
PRODUCT_NAMES = {
    'Electronics': ['Smart TV 55"', 'Wireless Speaker', 'LED Strip Lights', 'Power Bank 20000mAh', 'USB-C Hub'],
    'Computers': ['Mechanical Keyboard', 'Gaming Mouse', 'External SSD 1TB', '27" Monitor', 'Laptop Stand'],
    'Cameras': ['Action Camera 4K', 'Ring Light', 'Camera Bag', 'SD Card 128GB', 'Tripod'],
    'Phones': ['Wireless Charger', 'Phone Case', 'Screen Protector', 'Bluetooth Earbuds', 'Car Mount'],
    'Audio': ['Noise Cancelling Headphones', 'Soundbar', 'Studio Microphone', 'Audio Interface', 'DAC Amplifier']
}


@st.cache_resource
def load_model_and_data():
    """Cache the model and data — only loads once per session."""
    df, user_map, item_map = generate_synthetic_amazon_data(seed=42)
    train_df, val_df, test_df = train_val_test_split(df)

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()

    edge_index = build_graph(train_df, num_users, num_items)
    user_pos_items = get_user_positive_items(train_df)

    model = LightGCN(num_users=num_users, num_items=num_items,
                      embedding_dim=64, num_layers=3)

    # Load pretrained weights if available, else use random init
    try:
        ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        trained = True
    except FileNotFoundError:
        trained = False

    model.eval()
    return model, edge_index, user_pos_items, num_users, num_items, train_df, df, trained


def get_item_display_name(item_idx, df):
    """Map item index to a display name."""
    cat_row = df[df['item_idx'] == item_idx]
    if len(cat_row) > 0:
        cat = cat_row.iloc[0]['category']
        products = PRODUCT_NAMES[cat]
        return f"{products[item_idx % len(products)]} ({cat})"
    return f"Product #{item_idx}"


def get_recommendations_display(model, edge_index, user_id, user_pos_items,
                                  num_items, df, k=10):
    """Get top-K recommendations with display names."""
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
        scores = (user_emb[user_id] @ item_emb.T).numpy()

    seen = user_pos_items.get(user_id, set())
    scores[list(seen)] = -np.inf

    top_k_idx = np.argsort(scores)[-k:][::-1]
    results = []
    for rank, item_idx in enumerate(top_k_idx, 1):
        results.append({
            'Rank': rank,
            'Product': get_item_display_name(int(item_idx), df),
            'Score': f"{scores[item_idx]:.3f}",
            'Item ID': int(item_idx)
        })
    return pd.DataFrame(results)


# ---- UI ----

st.title("🛍️ GNN-Powered Product Recommender")
st.markdown("*Built with LightGCN + PyTorch Geometric | Amazon-style implicit feedback*")

model, edge_index, user_pos_items, num_users, num_items, train_df, df, trained = load_model_and_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{num_users:,}")
col2.metric("Total Products", f"{num_items:,}")
col3.metric("Interactions", f"{len(train_df):,}")


if not trained:
    st.warning("⚠️ Model not trained yet. Run `python train.py` first for meaningful recommendations. Showing random-init output for demo.")

st.divider()

tab1, tab2, tab3 = st.tabs(["🎯 Get Recommendations", "📊 Model Architecture", "📈 About GNNs"])

with tab1:
    st.subheader("Personalized Recommendations")
    col_a, col_b = st.columns([1, 2])

    with col_a:
        user_id = st.number_input("User ID", min_value=0,
                                    max_value=num_users - 1, value=42)
        k = st.slider("Top-K recommendations", 5, 20, 10)

        user_history = user_pos_items.get(user_id, set())
        st.metric("User's purchase history", f"{len(user_history)} products")

        if st.button("🔍 Get Recommendations", type="primary"):
            with st.spinner("Running GNN inference..."):
                recs = get_recommendations_display(
                    model, edge_index, user_id, user_pos_items, num_items, df, k)
            st.dataframe(recs, use_container_width=True, hide_index=True)

    with col_b:
        st.subheader("User's Purchase History")
        if user_history:
            history_items = [get_item_display_name(i, df) for i in list(user_history)[:15]]
            for item in history_items:
                st.write(f"✅ {item}")
        else:
            st.info("No purchase history for this user.")

with tab2:
    st.subheader("LightGCN Architecture")
    st.code("""
LightGCN Model
=============
Input:
  - User embeddings: [num_users, 64]
  - Item embeddings: [num_items, 64]
  - Interaction graph: [2, num_edges]

Layer 0 (Initial):    E^0 = [E_user || E_item]   # concat
Layer 1:              E^1 = A_norm * E^0          # graph conv  
Layer 2:              E^2 = A_norm * E^1          # graph conv
Layer 3:              E^3 = A_norm * E^2          # graph conv

Final:                E = mean(E^0, E^1, E^2, E^3)  # layer combo

Scoring:              score(u, i) = E_u · E_i      # dot product
Loss:                 BPR: -log(sigma(pos_score - neg_score))
    """, language='text')

    st.markdown("""
    **Why LightGCN beats vanilla GCN for recommendations:**
    - ❌ Removes feature transformation matrices (not needed for CF)
    - ❌ Removes non-linear activations (hurt performance empirically)
    - ✅ Keeps only normalized neighborhood aggregation
    - ✅ Multi-layer combination captures multi-hop user-item relationships
    """)

with tab3:
    st.subheader("Why Graph Neural Networks for Recommendations?")
    st.markdown("""
    **The core insight:** User-item interactions naturally form a graph.
    
    Traditional collaborative filtering (Matrix Factorization) only looks at 
    **direct** user-item interactions. GNNs capture **higher-order** connectivity:
    
    > *"User A bought Item X. User B also bought Item X and Item Y.  
    > Therefore, User A might like Item Y."*
    
    This is 2-hop reasoning — impossible with vanilla MF, trivial with GNNs.
    
    **In production:**
    - 📌 **Pinterest** uses PinSage (GraphSAGE-based) for 3B+ pins
    - 💼 **LinkedIn** uses GNNs for job recommendations
    - 🚗 **Uber Eats** uses GNNs for restaurant recommendations
    - 🛍️ **Alibaba** uses LightGCN variant for product recommendations
    """)

st.divider()
