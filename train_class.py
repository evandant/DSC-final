import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

# -----------------------
# Load data
# -----------------------
data = torch.load("geo_embeddings.pt")
X = data["embeddings"]  # [N, 384] DINOv2 embeddings
coords = data["coords"]  # [N, 2]   raw lat/lon (kept for evaluation)

# Load the grid cell labels produced by make_labels.py
df = pd.read_csv("coords_cells.csv")

# Map cell_id integers to contiguous class indices (0, 1, 2, ...)
# Required because cell_ids are sparse (not all grid cells exist in the data)
classes = sorted(df["cell_id"].unique())
class_map = {c: i for i, c in enumerate(classes)}
y = torch.tensor([class_map[c] for c in df["cell_id"].values])

# Include raw coords in dataset so evaluate_class.py can compute haversine error
dataset = TensorDataset(X, y, coords)

# -----------------------
# Train / val / test split
# -----------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Fixed seed so the split is reproducible across training and evaluation runs
generator = torch.Generator().manual_seed(42)

train_ds, test_ds = random_split(
    dataset,
    [train_size, test_size],
    generator=generator
)

# Save split indices so evaluate_class.py uses the exact same test set
torch.save({
    "train_idx": train_ds.indices,
    "test_idx": test_ds.indices
}, "data_split.pt")

num_classes = len(classes)
print("Classes:", num_classes)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

# -----------------------
# Model
# -----------------------
# Same MLP architecture as train.py, but output size is num_classes
# instead of 2 — predicts a probability distribution over grid cells
model = nn.Sequential(
    nn.Linear(384, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, num_classes)
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()  # expects raw logits (no softmax needed)

# -----------------------
# Training loop
# -----------------------
for epoch in range(30):
    model.train()
    total_loss = 0

    for xb, yb, _ in train_loader:  # _ discards coords (not needed during training)
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: {total_loss / len(train_loader):.4f}")

# Save only the state dict (weights), not the full model object
torch.save(model.state_dict(), "geo_classifier.pt")