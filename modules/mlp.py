import torch
import torch.nn as nn

from modules.activation import ReLU


class Linear(nn.Module):
    def __init__(
        self, in_size: int, out_size: int, device: torch.device | None = None
    ) -> None:
        super().__init__()
        self.W: torch.Tensor = nn.Parameter(
            torch.randn([out_size, in_size], device=device)
        )  # Store Transposed, for better cache access during forward pass
        self.b: torch.Tensor = nn.Parameter(torch.zeros(out_size, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T + self.b  # Transpose! see init


class MLP(nn.Module):
    def __init__(
        self,
        in_size: int,
        hidden_sizes: list[int],
        out_size: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        layers = [Linear(in_size, hidden_sizes[0], device=device)]
        for i, _ in enumerate(hidden_sizes[1:]):
            layers.append(Linear(hidden_sizes[i], hidden_sizes[i + 1], device=device))
        layers.append(Linear(hidden_sizes[-1], out_size, device=device))
        self.linear_layers: nn.ModuleList = nn.ModuleList(layers)

        self.a: nn.Module = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.linear_layers) - 1):
            x = self.a(self.linear_layers[i](x))
        return self.linear_layers[-1](x)
