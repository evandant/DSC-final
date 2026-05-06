import pandas as pd

df = pd.read_csv("coords.csv")

# Size of each grid cell in degrees
GRID_SIZE = 2


def latlon_to_cell(lat, lon):
    # Shift lat/lon to positive range, then bin into grid cell indices
    # lat range: [-90, 90]  → bin range: [0, 90]
    # lon range: [-180, 180] → bin range: [0, 180]
    lat_bin = int((lat + 90) // GRID_SIZE)
    lon_bin = int((lon + 180) // GRID_SIZE)
    return lat_bin, lon_bin


cells = df.apply(lambda row: latlon_to_cell(row["latitude"], row["longitude"]), axis=1)

df["lat_bin"] = [c[0] for c in cells]
df["lon_bin"] = [c[1] for c in cells]

# Encode the 2D bin pair as a single integer ID for use as a class label.
# Multiplying lat_bin by 1000 ensures uniqueness as long as lon_bin < 1000
# (lon bins max out at 360/2 = 180, so this is safe)
df["cell_id"] = df["lat_bin"] * 1000 + df["lon_bin"]

df.to_csv("coords_cells.csv", index=False)

print("Number of classes:", df["cell_id"].nunique())