import torch
import torch.nn as nn
import pandas as pd
import math
from torch.utils.data import TensorDataset, DataLoader, Subset

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load data
# -----------------------
data = torch.load("geo_embeddings.pt")
X = data["embeddings"]
coords = data["coords"]

df = pd.read_csv("coords_cells.csv")

# Rebuild the same class mapping used during training
classes = sorted(df["cell_id"].unique())
class_map = {c: i for i, c in enumerate(classes)}
reverse_map = {i: c for c, i in class_map.items()}  # index → cell_id (for decoding predictions)

y = torch.tensor([class_map[c] for c in df["cell_id"].values])

GRID_SIZE = 2  # must match make_labels.py

# -----------------------
# Load the saved test split
# -----------------------
# Use the same indices from training to avoid evaluating on training data
split = torch.load("data_split.pt")
test_idx = split["test_idx"]

dataset = TensorDataset(X, y, coords)
test_ds = Subset(dataset, test_idx)
test_loader = DataLoader(test_ds, batch_size=128)

# -----------------------
# Model
# -----------------------
num_classes = len(classes)

model = nn.Sequential(
    nn.Linear(384, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)
)

model.load_state_dict(torch.load("geo_classifier.pt"))
model.to(device)
model.eval()


# -----------------------
# Helpers
# -----------------------
def cell_to_latlon(cell_id):
    # Decode a cell_id back to the center lat/lon of that grid cell
    lat_bin = cell_id // 1000
    lon_bin = cell_id % 1000
    lat = lat_bin * GRID_SIZE - 90 + GRID_SIZE / 2  # center of cell
    lon = lon_bin * GRID_SIZE - 180 + GRID_SIZE / 2
    return lat, lon


def cell_to_bins(cell_id):
    return cell_id // 1000, cell_id % 1000


def geo_distance(a, b):
    # Haversine great-circle distance in km
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


# -----------------------
# Evaluation loop
# -----------------------
errors = []
# Chebyshev cell distance: max of absolute differences in lat/lon bin indices
# 0 = exact cell, 1 = adjacent cell, etc.
counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, "5+": 0}
results = []

topk = 3
correct_topk = 0
total = 0

global_idx = 0  # tracks position in test_idx for image loading

with torch.no_grad():
    for xb, yb, coords_batch in test_loader:
        xb = xb.to(device)

        preds = model(xb)
        pred_classes = preds.argmax(dim=1).cpu()
        # Top-k: check if true class appears anywhere in top-k predictions
        topk_preds = preds.topk(topk, dim=1).indices.cpu()

        for i in range(len(pred_classes)):
            pred_cell = reverse_map[pred_classes[i].item()]
            true_cell = reverse_map[yb[i].item()]

            # Convert predicted cell center to lat/lon for haversine error
            pred_latlon = cell_to_latlon(pred_cell)
            true_latlon = coords_batch[i].tolist()  # exact GPS coords

            err = geo_distance(pred_latlon, true_latlon)
            errors.append(err)

            # Cell distance (Chebyshev) — measures grid-level accuracy
            pred_lat_bin, pred_lon_bin = cell_to_bins(pred_cell)
            true_lat_bin, true_lon_bin = cell_to_bins(true_cell)
            dist = max(
                abs(pred_lat_bin - true_lat_bin),
                abs(pred_lon_bin - true_lon_bin)
            )

            if dist >= 5:
                counts["5+"] += 1
            else:
                counts[dist] += 1

            if yb[i].item() in topk_preds[i]:
                correct_topk += 1

            results.append({
                "error": err,
                "index": test_idx[global_idx],
                "true": true_latlon,
                "pred": pred_latlon
            })

            total += 1
            global_idx += 1

# -----------------------
# Metrics
# -----------------------
errors_sorted = sorted(errors)

print("Median error (km):", errors_sorted[len(errors) // 2])
print("Mean error (km):", sum(errors) / len(errors))
print("90th percentile (km):", errors_sorted[int(len(errors) * 0.9)])

# Normalize cell distance counts to fractions
for k in counts:
    counts[k] /= total

print("\nCell distance distribution:")
print(counts)

print(f"\nTop-{topk} accuracy:", correct_topk / total)

# -----------------------
# Outlier Analysis — visualize worst predictions
# -----------------------
results_sorted = sorted(results, key=lambda x: x["error"], reverse=True)

print("\nTop 10 worst predictions:")
for r in results_sorted[:10]:
    print(f"Error: {r['error']:.0f} km | True: {r['true']} | Pred: {r['pred']} | Index: {r['index']}")

import matplotlib.pyplot as plt
from PIL import Image

N = 10
fig, axes = plt.subplots(2, 5, figsize=(18, 8))

for i in range(N):
    r = results_sorted[i]
    idx = r["index"]

    try:
        img = Image.open(f"images/{idx}.png")
    except:
        continue

    ax = axes[i // 5, i % 5]
    ax.imshow(img)

    title = (
        f"Err: {r['error']:.0f} km\n"
        f"True: {r['true'][0]:.1f}, {r['true'][1]:.1f}\n"
        f"Pred: {r['pred'][0]:.1f}, {r['pred'][1]:.1f}"
    )

    ax.set_title(title, fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()