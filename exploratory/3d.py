import torch
import umap
import plotly.express as px

data = torch.load("geo_embeddings.pt")
X = data["embeddings"].numpy()
coords = data["coords"].numpy()

# 3D UMAP
reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.1)
X_umap = reducer.fit_transform(X)

# Plot colored by latitude
fig = px.scatter_3d(
    x=X_umap[:,0],
    y=X_umap[:,1],
    z=X_umap[:,2],
    color=coords[:,0],
    title="3D UMAP of Image Embeddings (colored by latitude)"
)

fig.show()