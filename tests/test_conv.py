import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from modules.conv import Conv2d, MaxPool2d


class TestConv2d:
    """Compare custom Conv2d against PyTorch's nn.Conv2d."""

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randn(2, 3, 8, 8)  # (batch, channels, H, W)

    def _compare_conv(
        self, custom: Conv2d, pytorch: nn.Conv2d, x: torch.Tensor, atol=1e-5
    ):
        """Copy weights from custom to pytorch and compare outputs."""
        with torch.no_grad():
            pytorch.weight.copy_(custom.kernels)
            if custom.has_bias:
                pytorch.bias.copy_(custom.biases)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert (
            custom_out.shape == pytorch_out.shape
        ), f"Shape mismatch: {custom_out.shape} vs {pytorch_out.shape}"
        assert torch.allclose(
            custom_out, pytorch_out, atol=atol
        ), f"Output mismatch. Max diff: {(custom_out - pytorch_out).abs().max()}"

    def test_basic_conv(self, sample_input):
        """Basic 3x3 convolution."""
        custom = Conv2d(3, 16, kernel_size=3, padding=1)
        pytorch = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self._compare_conv(custom, pytorch, sample_input)

    def test_no_padding(self, sample_input):
        """Convolution without padding."""
        custom = Conv2d(3, 8, kernel_size=3, padding=0)
        pytorch = nn.Conv2d(3, 8, kernel_size=3, padding=0)
        self._compare_conv(custom, pytorch, sample_input)

    def test_different_kernel_sizes(self, sample_input):
        """Test various kernel sizes."""
        for k in [1, 2, 5]:
            custom = Conv2d(3, 4, kernel_size=k, padding=0)
            pytorch = nn.Conv2d(3, 4, kernel_size=k, padding=0)
            self._compare_conv(custom, pytorch, sample_input)

    def test_rectangular_kernel(self, sample_input):
        """Test non-square kernel."""
        custom = Conv2d(3, 4, kernel_size=(3, 5), padding=(1, 2))
        pytorch = nn.Conv2d(3, 4, kernel_size=(3, 5), padding=(1, 2))
        self._compare_conv(custom, pytorch, sample_input)

    def test_stride(self, sample_input):
        """Test strided convolution."""
        custom = Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        pytorch = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self._compare_conv(custom, pytorch, sample_input)

    def test_no_bias(self, sample_input):
        """Test convolution without bias."""
        custom = Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        pytorch = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self._compare_conv(custom, pytorch, sample_input)

    def test_single_channel(self):
        """Test single input/output channel."""
        x = torch.randn(1, 1, 5, 5)
        custom = Conv2d(1, 1, kernel_size=3, padding=1)
        pytorch = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self._compare_conv(custom, pytorch, x)


class TestMaxPool2d:
    """Compare custom MaxPool2d against PyTorch's nn.MaxPool2d."""

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randn(2, 3, 8, 8)

    def _compare_pool(
        self, custom: MaxPool2d, pytorch: nn.MaxPool2d, x: torch.Tensor, atol=1e-5
    ):
        """Compare outputs of custom vs pytorch pooling."""
        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert (
            custom_out.shape == pytorch_out.shape
        ), f"Shape mismatch: {custom_out.shape} vs {pytorch_out.shape}"
        assert torch.allclose(
            custom_out, pytorch_out, atol=atol
        ), f"Output mismatch. Max diff: {(custom_out - pytorch_out).abs().max()}"

    def test_basic_pool(self, sample_input):
        """Basic 2x2 max pooling with stride 2."""
        custom = MaxPool2d(kernel_size=2, stride=2)
        pytorch = nn.MaxPool2d(kernel_size=2, stride=2)
        self._compare_pool(custom, pytorch, sample_input)

    def test_3x3_pool(self, sample_input):
        """3x3 max pooling."""
        custom = MaxPool2d(kernel_size=3, stride=2, padding=1)
        pytorch = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self._compare_pool(custom, pytorch, sample_input)

    def test_no_overlap(self, sample_input):
        """Non-overlapping pooling (stride == kernel_size)."""
        custom = MaxPool2d(kernel_size=4, stride=4)
        pytorch = nn.MaxPool2d(kernel_size=4, stride=4)
        self._compare_pool(custom, pytorch, sample_input)

    def test_overlap(self, sample_input):
        """Overlapping pooling (stride < kernel_size)."""
        custom = MaxPool2d(kernel_size=3, stride=1, padding=1)
        pytorch = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self._compare_pool(custom, pytorch, sample_input)
