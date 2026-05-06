import torch
from sklearn.neighbors import NearestNeighbors
import math
import random

# -----------------------
# Load embeddings
# -----------------------
data = torch.load("geo_embeddings.pt")
X = data["embeddings"].numpy()
y = data["coords"].numpy()

print("Loaded:", X.shape)


# -----------------------
# Haversine distance (km)
# -----------------------
def geo_distance(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# -----------------------
# KNN baseline
# -----------------------
# Fit KNN in DINOv2 embedding space using Euclidean distance.
# n_neighbors=2 so we can skip the first neighbor (which is the query point itself)
nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
nn.fit(X)

# Sample random query points and measure geographic error of nearest neighbor prediction
errors = []

for _ in range(200):
    idx = random.randint(0, len(X) - 1)

    dist, neighbors = nn.kneighbors([X[idx]])
    neighbor_idx = neighbors[0][1]  # [0] is itself, [1] is the true nearest neighbor

    pred = y[neighbor_idx]
    true = y[idx]

    error_km = geo_distance(pred, true)
    errors.append(error_km)

print("Average error (km):", sum(errors) / len(errors))
print("Median error (km):", sorted(errors)[len(errors) // 2])
