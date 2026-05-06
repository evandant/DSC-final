import torch
import umap
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Load data
# -----------------------
data = torch.load("geo_embeddings.pt")
X = data["embeddings"].numpy()
coords = data["coords"].numpy()

# -----------------------
# UMAP
# -----------------------
reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X)

# -----------------------
# Normalize latitude for better color scaling
# -----------------------

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10, 8), dpi=300)

lat = coords[:, 0]

scatter = plt.scatter(
    X_umap[:, 0],
    X_umap[:, 1],
    c=lat,
    cmap="viridis",
    s=6,
    alpha=0.8,
    linewidths=0
)

# Colorbar
cbar = plt.colorbar(scatter)
cbar.set_label("Latitude (°)", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Titles and labels
plt.title("UMAP Visualization of DINOv2 Embeddings", fontsize=16, pad=12)


# Clean up axes
plt.xticks([])
plt.yticks([])
plt.gca().set_frame_on(False)

# Tight layout for poster
plt.tight_layout()
plt.savefig("umap_plot.png", dpi=300, bbox_inches='tight')
plt.show()