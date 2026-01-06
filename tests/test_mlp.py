import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from modules.mlp import Linear, MLP


class TestLinear:
    """Compare custom Linear against PyTorch's nn.Linear."""

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randn(4, 16)  # (batch, features)

    def _compare_linear(
        self, custom: Linear, pytorch: nn.Linear, x: torch.Tensor, atol=1e-5
    ):
        """Copy weights from custom to pytorch and compare outputs."""
        with torch.no_grad():
            pytorch.weight.copy_(custom.W)
            pytorch.bias.copy_(custom.b)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert (
            custom_out.shape == pytorch_out.shape
        ), f"Shape mismatch: {custom_out.shape} vs {pytorch_out.shape}"
        assert torch.allclose(
            custom_out, pytorch_out, atol=atol
        ), f"Output mismatch. Max diff: {(custom_out - pytorch_out).abs().max()}"

    def test_basic_linear(self, sample_input):
        """Basic linear transformation."""
        custom = Linear(16, 32)
        pytorch = nn.Linear(16, 32)
        self._compare_linear(custom, pytorch, sample_input)

    def test_single_output(self, sample_input):
        """Linear layer with single output (e.g., regression)."""
        custom = Linear(16, 1)
        pytorch = nn.Linear(16, 1)
        self._compare_linear(custom, pytorch, sample_input)

    def test_same_size(self):
        """Input and output have same dimension."""
        x = torch.randn(8, 64)
        custom = Linear(64, 64)
        pytorch = nn.Linear(64, 64)
        self._compare_linear(custom, pytorch, x)

    def test_expand(self):
        """Expand to larger dimension."""
        x = torch.randn(2, 8)
        custom = Linear(8, 128)
        pytorch = nn.Linear(8, 128)
        self._compare_linear(custom, pytorch, x)

    def test_contract(self):
        """Contract to smaller dimension."""
        x = torch.randn(2, 128)
        custom = Linear(128, 8)
        pytorch = nn.Linear(128, 8)
        self._compare_linear(custom, pytorch, x)

    def test_batched_3d(self):
        """Works with 3D input (sequence data)."""
        x = torch.randn(2, 10, 16)  # (batch, seq, features)
        custom = Linear(16, 32)
        pytorch = nn.Linear(16, 32)
        self._compare_linear(custom, pytorch, x)


class TestMLP:
    """Compare custom MLP against PyTorch equivalent."""

    def _build_pytorch_mlp(self, in_size, hidden_sizes, out_size):
        """Build equivalent PyTorch MLP."""
        layers = []
        prev_size = in_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, out_size))
        return nn.Sequential(*layers)

    def _copy_weights(self, custom: MLP, pytorch: nn.Sequential):
        """Copy weights from custom MLP to pytorch Sequential."""
        with torch.no_grad():
            linear_idx = 0
            for module in pytorch:
                if isinstance(module, nn.Linear):
                    custom_layer = custom.linear_layers[linear_idx]
                    module.weight.copy_(custom_layer.W)
                    module.bias.copy_(custom_layer.b)
                    linear_idx += 1

    def test_single_hidden(self):
        """MLP with one hidden layer."""
        x = torch.randn(4, 16)
        custom = MLP(16, [32], 8)
        pytorch = self._build_pytorch_mlp(16, [32], 8)
        self._copy_weights(custom, pytorch)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert custom_out.shape == pytorch_out.shape
        assert torch.allclose(custom_out, pytorch_out, atol=1e-5)

    def test_multiple_hidden(self):
        """MLP with multiple hidden layers."""
        x = torch.randn(4, 32)
        custom = MLP(32, [64, 32, 16], 10)
        pytorch = self._build_pytorch_mlp(32, [64, 32, 16], 10)
        self._copy_weights(custom, pytorch)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert custom_out.shape == pytorch_out.shape
        assert torch.allclose(custom_out, pytorch_out, atol=1e-5)

    def test_classification(self):
        """Typical classification setup."""
        x = torch.randn(8, 784)  # flattened MNIST
        custom = MLP(784, [256, 128], 10)
        pytorch = self._build_pytorch_mlp(784, [256, 128], 10)
        self._copy_weights(custom, pytorch)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert custom_out.shape == pytorch_out.shape
        assert torch.allclose(custom_out, pytorch_out, atol=1e-4)

    def test_wide_hidden(self):
        """MLP with wide hidden layer."""
        x = torch.randn(2, 16)
        custom = MLP(16, [512], 4)
        pytorch = self._build_pytorch_mlp(16, [512], 4)
        self._copy_weights(custom, pytorch)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert custom_out.shape == pytorch_out.shape
        assert torch.allclose(custom_out, pytorch_out, atol=1e-5)
