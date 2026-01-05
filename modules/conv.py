import math
import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=(0, 0), dilation=1, groups=1, bias=True, padding_model='zeros', device=None, dtype=None):
        super().__init__()
        self.kernels = nn.Parameter(torch.randn((out_channels, kernel_size, kernel_size), dtype=dtype, device=device))
        self.kernel_size: tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.C_in: int = in_channels
        self.C_out: int = out_channels
        self.stride: tuple = (stride, stride) if isinstance(stride, int) else stride
        self.padding: tuple = (padding, padding) if isinstance(padding, int) else padding
        self.dilation: tuple = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups: int = groups
        self.bias: bool = bias
        self.padding_model: str = padding_model
        self.device: torch.Device = device
        self.dtype: torch.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # first naive way
        # x = (D, C_in, H, W)
        D, C_in, H_in, W_in = x.shape 
        # TODO: handle non batch inputs
        # TODO: padding
        # out = (D, C_out, W_out, H_out)
        H_out = math.floor((H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / (self.stride[0]))
        W_out = math.floor((W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / (self.stride[1]))
        out =  torch.zeros((D, self.C_out, H_out, W_out))
        for kernel_idx in range(self.C_out):
            kernel = self.kernels[kernel_idx]
            for i in range(0, H_in, self.stride[0]):
                for j in range(0, W_in, self.stride[1]):
                    total = 0.0
                    for channel_idx in range(self.C_in):
                        window = x[: , channel_idx, i:i + self.kernel_size[0], j:j+self.kernel_size[1]] 
                        total += torch.sum(window * kernel) # check dims here
                    out[:, kernel_idx, i, j] = total

        return out

