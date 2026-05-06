import torch
import math
from torch.utils.data import TensorDataset, DataLoader, Subset

# -----------------------
# Load data + model
# -----------------------
data = torch.load("geo_embeddings.pt")
x = data["embeddings"]
y = data["coords"]

# -----------------------
# Load the saved test split
# -----------------------
# Use the same indices from training to avoid evaluating on training data
split = torch.load("regression_split.pt")
test_idx = split["test_idx"]

x_test = x[test_idx]
y_test = y[test_idx]

# Load full model object saved by train.py (weights_only=False required for full model)
model = torch.load("geo_model.pt", weights_only=False)
model.eval()


# -----------------------
# Haversine distance
# -----------------------
def geo_distance(a, b):
    # Computes great-circle distance in km between two (lat, lon) points.
    # More accurate than Euclidean distance for geographic coordinates.
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    R = 6371  # Earth's radius in km
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# -----------------------
# Evaluate
# -----------------------
errors = []


with torch.no_grad():
    preds = model(x_test).cpu()

for p, t in zip(preds, y_test):
    err = geo_distance(p.tolist(), t.tolist())
    errors.append(err)

errors.sort()

print("Median error (km):", errors[len(errors) // 2])
print("Mean error (km):", sum(errors) / len(errors))
print("90th percentile (km):", errors[int(len(errors) * 0.9)])