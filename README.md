# Geographic Location Prediction with DINOv2

A deep learning pipeline for predicting geographic coordinates from street-level images using DINOv2 vision transformer embeddings.

## Overview

This project extracts image embeddings from a pretrained DINOv2 model and trains lightweight MLP heads to predict GPS coordinates. Two approaches are compared: direct coordinate regression and discretized grid-cell classification.

## Pipeline

```
images/ + coords.csv
        │
        ▼
    extract.py          → geo_embeddings.pt
        │
        ├──▶ train_regression.py           → geo_model.pt       (regression)
        │         └──▶ evaluate_regression.py
        │
        ├──▶ make_labels.py     → coords_cells.csv
        │         └──▶ train_class.py   → geo_classifier.pt  (classification)
        │                   └──▶ evaluate_class.py
        │
        └──▶ exploratory/       (visualizations & baselines)
```

## Project Structure

```
geo-project/
├── dataset.py          # GeoDataset class (shared)
├── extract.py          # Extract DINOv2 embeddings from images
├── make_labels.py      # Bin coordinates into grid cells for classification
├── train_regression.py            # Train MLP regression head (lat/lon output)
├── train_class.py      # Train MLP classifier (grid cell output)
├── evaluate_regression.py     # Evaluate regression model (haversine error)
├── evaluate_class.py   # Evaluate classifier (top-k accuracy, cell distance)
├── exploratory/
│   ├── pca_plot.py     # PCA of embedding space
│   ├── umap_plot.py    # 2D UMAP colored by latitude
│   ├── 3d.py           # 3D UMAP visualization
│   ├── plot_map.py     # Geographic scatter plot of dataset
│   ├── knn_test.py     # KNN baseline on embeddings
│   └── dino_visual.py  # DINOv2 attention map visualization
└── README.md
```

## Setup

**Requirements**

```bash
pip install torch torchvision pandas scikit-learn umap-learn matplotlib plotly pillow
```

**Data**

Place your data in the project root:
- `images/` — street-level images named `0.png`, `1.png`, ...
- `coords.csv` — CSV with `latitude` and `longitude` columns (one row per image)

## Usage

**1. Extract embeddings**
```bash
python extract.py
```
Runs all images through DINOv2 (`dinov2_vits14`) and saves 384-dim embeddings to `geo_embeddings.pt`.

**2a. Train regression model**
```bash
python train_regression.py
```
Trains an MLP to directly predict (latitude, longitude). Saves model to `geo_model.pt`.

**2b. Train classification model**
```bash
python make_labels.py   # bin coords into 2° grid cells
python train_class.py   # train classifier over cells
```
Saves model to `geo_classifier.pt` and the train/val/test split to `data_split.pt`.

**3. Evaluate**
```bash
python evaluate_regression.py      # regression: median/mean/90th percentile error in km
python evaluate_class.py    # classification: top-k accuracy, cell distance distribution
```

## Models

Both prediction heads share the same architecture, differing only in output size:

```
Linear(384 → 512) → ReLU → Linear(512 → 256) → ReLU → Linear(256 → out)
```

| Model | Output | Loss | File |
|---|---|---|---|
| Regression | 2 (lat, lon) | MSE | `geo_model.pt` |
| Classifier | N grid cells | CrossEntropy | `geo_classifier.pt` |

Currently, the grid uses 2° × 2° cells globally (~222 km per cell side).

## Evaluation Metrics

- **Haversine error** (km) — median, mean, 90th percentile
- **Cell distance** — Chebyshev distance in grid cells between predicted and true cell
- **Top-k accuracy** — whether the true cell appears in the top-k predictions

## Exploratory Analysis

Scripts in `exploratory/` visualize the embedding space and establish a KNN baseline. Run any of them after `extract.py` has been run:

```bash
python exploratory/umap_plot.py     # saves umap_plot.png
python exploratory/pca_plot.py
python exploratory/plot_map.py      # saves og_map.png
python exploratory/knn_test.py      # KNN baseline error
python exploratory/dino_visual.py   # saves dino_0.png (attention map for images/0.png)
```

## Advanced Topics

This project incorporates the following advanced topics (6.5 points total, requirement: 4):

| Topic | Points | Implementation |
|---|---|---|
| Image Processing | 2 | DINOv2 ViT feature extraction pipeline (`extract.py`) |
| Neural Net | 1.5 | MLP prediction heads (`train_regression.py`, `train_class.py`) |
| tSNE / UMAP | 1 | 2D and 3D UMAP embeddings (`umap_plot.py`, `3d.py`) |
| K-Nearest Neighbors | 1 | KNN baseline in embedding space (`knn_test.py`) |
| Principal Component Analysis | 1 | PCA of DINOv2 embeddings (`pca_plot.py`) |
| **Total** | **6.5** | |
