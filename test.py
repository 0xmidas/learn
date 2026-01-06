from modules.resnet import ResNetCIFAR
import torch

model = ResNetCIFAR()
x = torch.randn(1, 3, 32, 32)
out = model(x)
print(out.shape)
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
