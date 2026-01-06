from einops.einops import rearrange
import math
import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size: tuple[int, int] = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.C_in: int = in_channels
        self.C_out: int = out_channels
        self.stride: tuple[int, int] = (
            (stride, stride) if isinstance(stride, int) else stride
        )
        self.padding: tuple[int, int] = (
            (padding, padding) if isinstance(padding, int) else padding
        )
        self.dilation: tuple[int, int] = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )  # not supported for now
        self.has_bias: bool = bias
        self.groups: int = groups
        self.padding_mode: str = padding_mode

        self.k: int = math.sqrt(1 / (self.C_in * self.kernel_size[0] * self.kernel_size[1]))
        self.kernels = nn.Parameter(
            torch.rand(
                (out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]),
                dtype=dtype,
                device=device,
            ) * 2 * self.k - self.k
        )
        if bias:
            self.biases = nn.Parameter(
                torch.rand(out_channels, dtype=dtype, device=device) * 2 * self.k - self.k
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(
                x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
            )
        N, C_in, H_in, W_in = x.shape

        H_out = (H_in - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in - self.kernel_size[1]) // self.stride[1] + 1
 
        col = self.im2col(x, H_out, W_out)
        kernels_flat = rearrange(self.kernels, "C_out C_in k1 k2 -> C_out (C_in k1 k2)")

        out = kernels_flat @ col

        return out.reshape(N, self.C_out, H_out, W_out)
        

    def im2col(self, x: torch.Tensor, H_out: int, W_out: int):
        N, C, H, W = x.shape
       
        col = torch.zeros((N, C * self.kernel_size[0] * self.kernel_size[1], H_out * W_out))
        
        col_idx = 0
        for i in range(0, H - self.kernel_size[0] + 1, self.stride[0]):
            for j in range(0, W - self.kernel_size[1] + 1, self.stride[1]):
                    patch = x[:, :, i:i+self.kernel_size[0], j:j+self.kernel_size[1]]
                    col_idx = (i // self.stride[0]) * W_out + (j // self.stride[1])
                    col[:, :, col_idx] = patch.reshape(N, -1)

        return col


class MaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size: tuple[int, int] = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        # PyTorch default: stride = kernel_size if not specified
        if stride is None:
            self.stride: tuple[int, int] = self.kernel_size
        else:
            self.stride: tuple[int, int] = (stride, stride) if isinstance(stride, int) else stride
        self.padding: tuple[int, int] = (
            (padding, padding) if isinstance(padding, int) else padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H_in, W_in = x.shape
        
        x = nn.functional.pad(
            x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]),
            value=float('-inf')  # so padded values never win the max
        )
        
        _, _, H_pad, W_pad = x.shape
        H_out = (H_pad - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_pad - self.kernel_size[1]) // self.stride[1] + 1
        
        # Extract all windows: (N, C, H_out, W_out, kH, kW)
        windows = self._extract_windows(x, H_out, W_out)
        
        # Flatten kernel dims and take max: (N, C, H_out, W_out, kH*kW) -> (N, C, H_out, W_out)
        out = windows.reshape(N, C, H_out, W_out, -1).max(dim=-1).values
        
        return out

    def _extract_windows(self, x: torch.Tensor, H_out: int, W_out: int) -> torch.Tensor:
        """Extract all pooling windows into shape (N, C, H_out, W_out, kH, kW)"""
        N, C, H, W = x.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        
        # Use as_strided for zero-copy view 
        # Output shape: (N, C, H_out, W_out, kH, kW)
        shape = (N, C, H_out, W_out, kH, kW)
        strides = (
            x.stride(0),                    # batch
            x.stride(1),                    # channel  
            x.stride(2) * sH,               # output row (jump by stride)
            x.stride(3) * sW,               # output col (jump by stride)
            x.stride(2),                    # kernel row
            x.stride(3),                    # kernel col
        )
        
        return x.as_strided(shape, strides)
