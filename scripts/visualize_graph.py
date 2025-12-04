#!/usr/bin/env python
"""
Graph visualization using torch and matplotlib (no NetworkX).
Loads outputs/scored_graph.pt and displays with multiple views: edge weight distribution, 
top-connected nodes, and force-directed layout.
"""
import os
import torch
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from sklearn.decomposition import PCA

artifact_path = "outputs/scored_graph.pt"
if not os.path.exists(artifact_path):
    raise SystemExit(f"Artifact not found: {artifact_path!r}. Run training to produce it.")

print(f"Loading {artifact_path}...")
state = torch.load(artifact_path, weights_only=False)

# Inspect structure
print("Keys in saved state:", state.keys() if isinstance(state, dict) else type(state))

# Attempt to extract graph data
if isinstance(state, dict):
    if "graph" in state:
        graph_data = state["graph"]
    elif "artifacts" in state:
        graph_data = state["artifacts"]
    else:
        graph_data = state
else:
    graph_data = state

# Inspect structure
print("Keys in saved state:", state.keys() if isinstance(state, dict) else type(state))

# Attempt to extract graph data
if isinstance(state, dict):
    if "graph" in state:
        graph_data = state["graph"]
    elif "artifacts" in state:
        graph_data = state["artifacts"]
    else:
        graph_data = state
else:
    graph_data = state

print("Graph data type:", type(graph_data))

# Handle GraphArtifacts object
if hasattr(graph_data, "node_features") and hasattr(graph_data, "edge_index"):
    print("Detected GraphArtifacts object")
    node_features = graph_data.node_features
    node_mapping = graph_data.node_mapping
    
    if isinstance(node_features, pd.DataFrame):
        x = node_features.drop(columns=["node"], errors="ignore").values
        node_names = node_features["node"].values
    else:
        x = np.array(node_features)
        node_names = list(node_mapping.keys())
    
    edge_index = graph_data.edge_index
    if hasattr(edge_index, "numpy"):
        edge_index = edge_index.numpy()
    
    # Extract edge weights
    edge_attr = graph_data.edge_attr
    if edge_attr is not None and len(edge_attr) > 0:
        if hasattr(edge_attr, "numpy"):
            edge_attr = edge_attr.numpy()
        amounts = edge_attr[:, 0] if edge_attr.ndim > 1 else edge_attr
    else:
        amounts = np.ones(edge_index.shape[1])
    
    # === Visualization 1: Edge weight distribution ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Edge weight histogram
    axes[0, 0].hist(amounts, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_title("Edge Weight Distribution", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Transaction Amount")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(alpha=0.3)
    
    # Subplot 2: Node degree distribution
    node_in_degree = np.bincount(edge_index[1], minlength=len(node_names))
    node_out_degree = np.bincount(edge_index[0], minlength=len(node_names))
    node_total_degree = node_in_degree + node_out_degree
    
    axes[0, 1].bar(range(len(node_names)), node_total_degree, color="coral", alpha=0.7, edgecolor="black")
    axes[0, 1].set_title("Node Degree (Top 30)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Node Index")
    axes[0, 1].set_ylabel("Total Degree")
    axes[0, 1].set_xlim(0, min(30, len(node_names)))
    axes[0, 1].grid(axis="y", alpha=0.3)
    
    # Subplot 3: Top connected nodes
    top_k = 10
    top_indices = np.argsort(node_total_degree)[-top_k:][::-1]
    top_names = [node_names[i] if i < len(node_names) else f"Node {i}" for i in top_indices]
    top_degrees = node_total_degree[top_indices]
    
    axes[1, 0].barh(range(len(top_names)), top_degrees, color="lightgreen", edgecolor="black")
    axes[1, 0].set_yticks(range(len(top_names)))
    axes[1, 0].set_yticklabels(top_names, fontsize=9)
    axes[1, 0].set_title(f"Top {top_k} Connected Nodes", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Total Degree")
    axes[1, 0].grid(axis="x", alpha=0.3)
    
    # Subplot 4: PCA layout with top nodes highlighted
    pca = PCA(n_components=2)
    pos = pca.fit_transform(x)
    
    # Normalize edge amounts for coloring
    norm = colors.Normalize(vmin=amounts.min(), vmax=amounts.max())
    cmap = cm.get_cmap("viridis")
    
    ax = axes[1, 1]
    
    # Draw edges
    for idx, (i, j) in enumerate(edge_index.T):
        color = cmap(norm(amounts[idx]))
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=color, alpha=0.2, linewidth=0.5)
    
    # Draw all nodes
    ax.scatter(pos[:, 0], pos[:, 1], alpha=0.5, s=50, c="steelblue", label="Nodes")
    
    # Highlight top nodes
    top_pos = pos[top_indices]
    ax.scatter(top_pos[:, 0], top_pos[:, 1], alpha=0.9, s=200, c="red", marker="*", 
               edgecolors="darkred", linewidth=2, label="Top nodes", zorder=5)
    
    ax.set_title("Network Layout (PCA) - Top Nodes Highlighted", fontsize=12, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=9)
    
    plt.suptitle(f"Graph Analysis: {len(node_names)} nodes, {edge_index.shape[1]} edges", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "graph_analysis.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"✓ Saved multi-view analysis: {out_png}")
    
    # === Visualization 2: Filtered high-weight subgraph ===
    threshold = np.percentile(amounts, 75)  # Top 25% edges
    high_weight_edges = np.where(amounts >= threshold)[0]
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Only draw high-weight edges
    for idx in high_weight_edges:
        i, j = edge_index[:, idx]
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                color=cmap(norm(amounts[idx])), alpha=0.6, linewidth=1.5)
    
    # Draw nodes involved in high-weight edges
    active_nodes = set(edge_index[:, high_weight_edges].flatten())
    active_pos = pos[[n for n in range(len(node_names)) if n in active_nodes]]
    ax.scatter(active_pos[:, 0], active_pos[:, 1], alpha=0.8, s=100, 
               c="orange", edgecolors="darkorange", linewidth=1.5)
    
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Transaction Amount", shrink=0.8)
    
    ax.set_title(f"High-Weight Subgraph (Top 25% of {len(high_weight_edges)} edges)", 
                 fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    
    out_png2 = os.path.join(out_dir, "graph_high_weight_subgraph.png")
    plt.savefig(out_png2, dpi=150, bbox_inches="tight")
    print(f"✓ Saved high-weight subgraph: {out_png2}")
    
    print(f"\n📊 Statistics:")
    print(f"  Nodes: {len(node_names)}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Avg edge weight: {amounts.mean():.2f}")
    print(f"  Max edge weight: {amounts.max():.2f}")
    print(f"  Min edge weight: {amounts.min():.2f}")

else:
    print("Unable to visualize: graph_data structure not recognized.")
    raise SystemExit(1)

try:
    plt.show()
except Exception:
    print("  (interactive display unavailable)")
