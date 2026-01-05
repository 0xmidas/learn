import torch
import torch.nn as nn

from modules.activation import ReLU

class Linear(nn.Module):
    def __init__(self, in_size, out_size, device=None):
        super().__init__()
        self.W: torch.Tensor = nn.Parameter(torch.randn([out_size, in_size])) # Store Transposed, for better cache access during forward pass
        self.b: torch.Tensor = nn.Parameter(torch.zeros(out_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T  + self.b # Transpose! see init

class MLP(nn.Module):
    def __init__(self, in_size: int, hidden_sizes: list[int], out_size: int, device=None):
        super().__init__()
        layers = [Linear(in_size, hidden_sizes[0], device=None)]
        for i, _ in enumerate(hidden_sizes[1:]):
            layers.append(Linear(hidden_sizes[i], hidden_sizes[i+1]))
        layers.append(Linear(hidden_sizes[-1], out_size, device=None))
        self.linear_layers: list[nn.Module] = nn.ModuleList(layers)

        self.a: nn.Module = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.linear_layers[:-1]:
            x = self.a(l(x))
        return self.linear_layers[-1](x)
    

