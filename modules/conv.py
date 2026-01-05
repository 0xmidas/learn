from einops.einops import rearrange
import math
import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, bias=True, padding_model='zeros', device=None, dtype=None):
        super().__init__()
        self.kernels = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size), dtype=dtype, device=device))
        self.kernel_size: tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.C_in: int = in_channels
        self.C_out: int = out_channels
        self.stride: tuple = (stride, stride) if isinstance(stride, int) else stride
        self.padding: tuple = (padding, padding) if isinstance(padding, int) else padding
        self.dilation: tuple = (dilation, dilation) if isinstance(dilation, int) else dilation # not supported for now
        self.bias: bool = bias
        self.groups: int = groups
        self.padding_model: str = padding_model
        self.device: torch.Device = device
        self.dtype: torch.dtype = dtype

        if bias:
            self.biases = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # first naive way
        # x = (D, C_in, H, W)
        D, C_in, H_in, W_in = x.shape 
        # TODO: handle non batch inputs
        if self.padding_model == "zeros":
            x = nn.functional.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        else:
            raise NotImplementedError

        # out = (D, C_out, H_out, W_out)
        H_out = math.floor((H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / (self.stride[0]) + 1)
        W_out = math.floor((W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / (self.stride[1]) + 1)
        out =  torch.zeros((D, self.C_out, H_out, W_out))

        for kernel_idx in range(self.C_out):
            for i in range(0, H_out, self.stride[0]):
                for j in range(0, W_out, self.stride[1]):
                    total = 0.0
                    for channel_idx in range(self.C_in):
                        kernel = self.kernels[kernel_idx, channel_idx]
                        window = x[: , channel_idx, i:i + self.kernel_size[0], j:j+self.kernel_size[1]] 
                        avg = torch.sum(window * kernel, dim=(1, 2))
                        total += avg
                    out[:, kernel_idx, i, j] = total + self.biases[kernel_idx] if self.bias else 0.0
        return out


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: tuple | int, stride: tuple | int = (0, 0), padding: tuple | int = (0, 0)):
        super().__init__()
        self.kernel_size: tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride: tuple = (stride, stride) if isinstance(stride, int) else stride 
        self.padding: tuple = (padding, padding) if isinstance(padding, int) else padding
        # TODO return indices for max unpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, C_in, H_in, W_in = x.shape 
        x = nn.functional.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        # out = (D, C_out, W_out, H_out)
        H_out = math.floor((H_in + 2 * self.padding[0] - 1) / (self.stride[0]) + 1)
        W_out = math.floor((W_in + 2 * self.padding[1] - 1) / (self.stride[1]) + 1)
        out = torch.zeros((D, C_in, H_out, W_out))

        for c in range(C_in):
            for i in range(0, H_out):
                for j in range(0, W_out):
                    window = x[: , c, i*self.stride[0]:i*self.stride[0] + self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]] 
                    window = rearrange(window, "B H W -> B (H W)")
                    max_val = torch.max(window, dim=1).values
                    out[:, c, i, j] = max_val 

        return out

