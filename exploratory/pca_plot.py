import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = torch.load("geo_embeddings.pt")
X = data["embeddings"].numpy()
coords = data["coords"].numpy()

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# plot colored by latitude
plt.scatter(X_pca[:,0], X_pca[:,1], c=coords[:,0], s=2)
plt.colorbar(label="Latitude")
plt.title("PCA of DINOv2 Embeddings")
plt.show()