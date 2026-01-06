import torch
import torch.nn as nn
from einops import repeat, rearrange

class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        # TODO handle 3d input for seq
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum 
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
    
        self.beta = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        self.gamma = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))

        self.running_mean: torch.Tensor = torch.zeros(num_features, device=device, dtype=dtype)
        self.running_var: torch.Tensor = torch.zeros(num_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (Batch, Features)
        mean = torch.mean(x, dim=0)
        var = torch.var(x, dim=0, unbiased=False) 
        var_running = torch.var(x, dim=0, unbiased=True)


        if self.training and self.track_running_stats:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_running
        
            y = (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta
        else:
            y = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps) * self.gamma + self.beta

        return y 


class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None):
        super().__init__()
        self.num_features: int = num_features
        self.eps: float = eps
        self.momentum: float = momentum 
        self.affine: bool = affine
        self.track_running_stats: bool = track_running_stats
    
        self.beta = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        self.gamma = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))

        self.running_mean: torch.Tensor = torch.zeros(num_features, device=device, dtype=dtype)
        self.running_var: torch.Tensor = torch.zeros(num_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = (N, C, H, W)
        N, C, H, W = x.shape
        mean = torch.mean(x, dim=(0, 2, 3))
        var = torch.var(x, dim=(0, 2, 3), unbiased=False) 
        var_running = torch.var(x, dim=(0, 2, 3), unbiased=True)
        
        print(x.shape, mean.shape, self.running_mean.shape)

        if self.training and self.track_running_stats:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_running

            mean = rearrange(mean, "C -> C 1 1")
            var = rearrange(var, "C -> C 1 1")
        else:
            mean = rearrange(self.running_mean, "C -> C 1 1")
            var = rearrange(self.running_var, "C -> C 1 1")

        gamma = rearrange(self.gamma, "C -> C 1 1")
        beta = rearrange(self.beta, "C -> C 1 1")
        y = (x - mean) / torch.sqrt(var + self.eps) * gamma + beta


        return y 


