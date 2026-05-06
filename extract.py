import torch
from torch.utils.data import DataLoader
from dataset import GeoDataset  # shared dataset class

# -----------------------
# Config
# -----------------------
IMG_DIR = "images"
CSV_PATH = "coords.csv"
OUTPUT_FILE = "geo_embeddings.pt"
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Main Extraction Logic
# -----------------------
def main():
    print("Using device:", device)

    dataset = GeoDataset(IMG_DIR, CSV_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # keep order consistent with CSV index
        num_workers=4,
        pin_memory=True  # speeds up CPU→GPU transfers
    )

    print("Images found:", len(dataset))

    # Load pretrained DINOv2 (ViT-S/14) from torch hub
    # dinov2_vits14: small ViT with 14x14 patch size → 384-dim embeddings
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2.to(device)
    dinov2.eval()

    # Freeze all weights — we only use DINOv2 as a feature extractor
    for p in dinov2.parameters():
        p.requires_grad = False

    all_embeddings = []
    all_coords = []

    with torch.no_grad():
        for i, (imgs, coords) in enumerate(loader):
            imgs = imgs.to(device)

            # Forward pass through DINOv2 → [batch, 384] CLS token embeddings
            emb = dinov2(imgs)

            all_embeddings.append(emb.cpu())
            all_coords.append(coords)

            if i % 10 == 0:
                print(f"Processed batch {i}/{len(loader)}")

    # Concatenate all batches into single tensors
    embeddings = torch.cat(all_embeddings)
    coordinates = torch.cat(all_coords)

    print("Final embedding shape:", embeddings.shape)
    print("Final coord shape:", coordinates.shape)

    # Save embeddings and coordinates together for downstream training/eval
    torch.save({
        "embeddings": embeddings,
        "coords": coordinates
    }, OUTPUT_FILE)

    print("Saved to", OUTPUT_FILE)


# Guard required for DataLoader multiprocessing on Windows
if __name__ == "__main__":
    main()
