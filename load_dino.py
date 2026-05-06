import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.to(device)
dinov2.eval()

for p in dinov2.parameters():
    p.requires_grad = False

x = torch.randn(1,3,224,224)
y = dinov2(x)

print(y.shape)