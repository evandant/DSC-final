import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt


# paths (relative to your PyCharm project root)
image_dir = "images"
csv_file = "coords.csv"

# load CSV
data = pd.read_csv(csv_file)

# pick an index to inspect
idx = 5000

# load image
img_path = os.path.join(image_dir, f"{idx}.png")
image = Image.open(img_path).convert("RGB")

# get coordinates
lat = data.iloc[idx, 0]
lon = data.iloc[idx, 1]

# display
plt.imshow(image)
plt.axis("off")
plt.title(f"Index {idx} | Lat: {lat}, Lon: {lon}")
plt.show()

print(f"Loaded image {idx}.png with coordinates: ({lat}, {lon})")

