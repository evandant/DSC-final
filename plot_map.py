import torch
import pandas as pd
import matplotlib.pyplot as plt

data = torch.load("geo_embeddings.pt")
true_coords = data["coords"]

plt.figure(figsize=(10,5))

# Scatter plot
plt.scatter(true_coords[:,1], true_coords[:,0], s=2)

# Labels + title (bold + larger)
plt.xlabel("Longitude", fontsize=14, fontweight='bold')
plt.ylabel("Latitude", fontsize=14, fontweight='bold')
plt.title("Data Locations", fontsize=16, fontweight='bold')

# Make ticks bigger + slightly bolder
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Make axis lines thicker
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Optional: thicker tick marks
ax.tick_params(width=1.5)

plt.savefig("og_map.png", dpi=300)
plt.show()