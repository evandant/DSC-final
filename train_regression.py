import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# -----------------------
# Load embeddings
# -----------------------
# geo_embeddings.pt was produced by extract.py
# X: [N, 384] DINOv2 CLS embeddings
# y: [N, 2]   (latitude, longitude) targets
data = torch.load("geo_embeddings.pt")
X = data["embeddings"]
y = data["coords"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# -----------------------
# Train / test split
# -----------------------
dataset = TensorDataset(X, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_ds, test_ds = random_split(dataset, [train_size, test_size])

torch.save({"test_idx": test_ds.indices}, "regression_split.pt")

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=128)

# -----------------------
# Model
# -----------------------
# Lightweight MLP prediction head on top of frozen DINOv2 embeddings.
# Input: 384-dim embedding → Output: (lat, lon) coordinate pair
model = nn.Sequential(
    nn.Linear(384, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 2)  # 2 outputs: latitude and longitude
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# -----------------------
# Training Setup
# -----------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # MSE over (lat, lon) — treats both dimensions equally

# -----------------------
# Training Loop
# -----------------------
for epoch in range(20):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Train Loss = {total_loss / len(train_loader):.4f}")

# -----------------------
# Evaluation
# -----------------------
model.eval()
test_loss = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        test_loss += loss_fn(pred, yb).item()

print("Test Loss:", test_loss / len(test_loader))

# Save the full model object (not just weights) for easy loading in evaluate_regression.py
torch.save(model, "geo_model.pt")
