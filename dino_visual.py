import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

IMG_SIZE = 518  # must be divisible by PATCH_SIZE for clean patch grid
PATCH_SIZE = 14  # DINOv2 ViT-S/14 patch size
NUM_HEADS = 6  # number of attention heads in ViT-S
HEAD_DIM = 384 // NUM_HEADS  # 64 dims per head


# -----------------------
# Custom attention module
# -----------------------
# DINOv2 uses memory-efficient attention by default, which doesn't expose
# intermediate attention weights. We swap it out for standard attention
# that saves the weight matrix so we can visualize it.
class StandardAttention(nn.Module):
    def __init__(self, mem_eff_attn):
        super().__init__()
        # Reuse the trained projection weights from the original module
        self.qkv = mem_eff_attn.qkv
        self.proj = mem_eff_attn.proj
        self.proj_drop = mem_eff_attn.proj_drop
        self.num_heads = NUM_HEADS
        self.scale = HEAD_DIM ** -0.5
        self.attn_weights = None  # populated after each forward pass

    def forward(self, x):
        B, N, C = x.shape
        # Compute Q, K, V and reshape for multi-head attention
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn.detach()  # (B, heads, N+1, N+1) — save for visualization

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Replace only the last block's attention — it captures the highest-level features
model.blocks[-1].attn = StandardAttention(model.blocks[-1].attn)

# -----------------------
# Run inference
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("images/0.png").convert("RGB")
x = transform(img).unsqueeze(0)  # add batch dim → (1, 3, H, W)

with torch.no_grad():
    model(x)  # forward pass; attn_weights are saved as a side effect

# -----------------------
# Extract and visualize attention
# -----------------------
attn = model.blocks[-1].attn.attn_weights[0]  # (6 heads, N+1 tokens, N+1 tokens)

h = w = IMG_SIZE // PATCH_SIZE  # patch grid size: 37×37 for 518px image

# CLS token row (index 0) shows which patches the model attends to
# Slice off the CLS token itself → remaining N entries are patch attention scores
cls_attn = attn[:, 0, 1:]  # (6, N)
cls_attn = cls_attn.reshape(NUM_HEADS, h, w)
avg_attn = cls_attn.mean(0).numpy()  # average over heads → (h, w)

# Upsample patch-resolution attention map back to image resolution for overlay
attn_tensor = torch.tensor(avg_attn).unsqueeze(0).unsqueeze(0)
attn_up = F.interpolate(attn_tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear').squeeze().numpy()

img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("DINOv2 Attention", y=1)

axes[0].imshow(img_resized)
axes[0].set_title("Original")

axes[1].imshow(avg_attn, cmap='inferno')
axes[1].set_title(f"Attention ({h}×{w})")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig("dino_0.png", dpi=300)
plt.show()
