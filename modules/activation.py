import torch
import torch.nn as nn

class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, 0)


