import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from modules.batchnorm import BatchNorm1d, BatchNorm2d


class TestBatchNorm1d:
    """Compare custom BatchNorm1d against PyTorch's nn.BatchNorm1d."""

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randn(8, 16)  # (batch, features)

    def _copy_params(self, custom: BatchNorm1d, pytorch: nn.BatchNorm1d):
        """Copy parameters from custom to pytorch."""
        with torch.no_grad():
            if pytorch.affine:
                pytorch.weight.copy_(custom.gamma)
                pytorch.bias.copy_(custom.beta)
            pytorch.running_mean.copy_(custom.running_mean)
            pytorch.running_var.copy_(custom.running_var)

    def _compare_bn(
        self, custom: BatchNorm1d, pytorch: nn.BatchNorm1d, x: torch.Tensor, atol=1e-5
    ):
        """Compare outputs of custom vs pytorch batch norm."""
        self._copy_params(custom, pytorch)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert (
            custom_out.shape == pytorch_out.shape
        ), f"Shape mismatch: {custom_out.shape} vs {pytorch_out.shape}"
        assert torch.allclose(
            custom_out, pytorch_out, atol=atol
        ), f"Output mismatch. Max diff: {(custom_out - pytorch_out).abs().max()}"

    def test_basic_train(self, sample_input):
        """Basic batch norm in training mode."""
        custom = BatchNorm1d(16)
        pytorch = nn.BatchNorm1d(16)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_basic_eval(self, sample_input):
        """Basic batch norm in eval mode (uses running stats)."""
        custom = BatchNorm1d(16)
        pytorch = nn.BatchNorm1d(16)

        # First run some data through in training mode to accumulate stats
        self._copy_params(custom, pytorch)
        custom.train()
        pytorch.train()
        for _ in range(5):
            x = torch.randn(8, 16)
            custom(x)
            pytorch(x)

        # Now test in eval mode
        custom.eval()
        pytorch.eval()
        self._copy_params(custom, pytorch)  # sync again after training
        self._compare_bn(custom, pytorch, sample_input)

    def test_running_stats_update(self):
        """Running mean/var should update during training."""
        custom = BatchNorm1d(8)
        pytorch = nn.BatchNorm1d(8)
        self._copy_params(custom, pytorch)

        custom.train()
        pytorch.train()

        # Run several batches
        for _ in range(10):
            x = torch.randn(16, 8)
            custom(x)
            pytorch(x)

        # Running stats should match
        assert torch.allclose(
            custom.running_mean, pytorch.running_mean, atol=1e-5
        ), "Running mean mismatch"
        assert torch.allclose(
            custom.running_var, pytorch.running_var, atol=1e-5
        ), "Running var mismatch"

    def test_no_affine(self, sample_input):
        """Batch norm without learnable parameters."""
        custom = BatchNorm1d(16, affine=False)
        pytorch = nn.BatchNorm1d(16, affine=False)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_custom_momentum(self, sample_input):
        """Test with different momentum value."""
        custom = BatchNorm1d(16, momentum=0.2)
        pytorch = nn.BatchNorm1d(16, momentum=0.2)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_custom_eps(self, sample_input):
        """Test with different epsilon value."""
        custom = BatchNorm1d(16, eps=1e-3)
        pytorch = nn.BatchNorm1d(16, eps=1e-3)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_single_feature(self):
        """Single feature channel."""
        x = torch.randn(32, 1)
        custom = BatchNorm1d(1)
        pytorch = nn.BatchNorm1d(1)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, x)

    def test_large_batch(self):
        """Large batch size."""
        x = torch.randn(256, 64)
        custom = BatchNorm1d(64)
        pytorch = nn.BatchNorm1d(64)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, x)

class TestBatchNorm2d:
    """Compare custom BatchNorm2d against PyTorch's nn.BatchNorm2d."""

    @pytest.fixture
    def sample_input(self):
        torch.manual_seed(42)
        return torch.randn(4, 16, 8, 8)  # (batch, channels, H, W)

    def _copy_params(self, custom: BatchNorm2d, pytorch: nn.BatchNorm2d):
        """Copy parameters from custom to pytorch."""
        with torch.no_grad():
            if pytorch.affine:
                pytorch.weight.copy_(custom.gamma)
                pytorch.bias.copy_(custom.beta)
            pytorch.running_mean.copy_(custom.running_mean)
            pytorch.running_var.copy_(custom.running_var)

    def _compare_bn(
        self, custom: BatchNorm2d, pytorch: nn.BatchNorm2d, x: torch.Tensor, atol=1e-5
    ):
        """Compare outputs of custom vs pytorch batch norm."""
        self._copy_params(custom, pytorch)

        custom_out = custom(x)
        pytorch_out = pytorch(x)

        assert (
            custom_out.shape == pytorch_out.shape
        ), f"Shape mismatch: {custom_out.shape} vs {pytorch_out.shape}"
        assert torch.allclose(
            custom_out, pytorch_out, atol=atol
        ), f"Output mismatch. Max diff: {(custom_out - pytorch_out).abs().max()}"

    def test_basic_train(self, sample_input):
        """Basic batch norm in training mode."""
        custom = BatchNorm2d(16)
        pytorch = nn.BatchNorm2d(16)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_basic_eval(self, sample_input):
        """Basic batch norm in eval mode (uses running stats)."""
        custom = BatchNorm2d(16)
        pytorch = nn.BatchNorm2d(16)

        # First run some data through in training mode to accumulate stats
        self._copy_params(custom, pytorch)
        custom.train()
        pytorch.train()
        for _ in range(5):
            x = torch.randn(4, 16, 8, 8)
            custom(x)
            pytorch(x)

        # Now test in eval mode
        custom.eval()
        pytorch.eval()
        self._copy_params(custom, pytorch)
        self._compare_bn(custom, pytorch, sample_input)

    def test_running_stats_update(self):
        """Running mean/var should update during training."""
        custom = BatchNorm2d(8)
        pytorch = nn.BatchNorm2d(8)
        self._copy_params(custom, pytorch)

        custom.train()
        pytorch.train()

        # Run several batches
        for _ in range(10):
            x = torch.randn(4, 8, 6, 6)
            custom(x)
            pytorch(x)

        # Running stats should match
        assert torch.allclose(
            custom.running_mean, pytorch.running_mean, atol=1e-5
        ), "Running mean mismatch"
        assert torch.allclose(
            custom.running_var, pytorch.running_var, atol=1e-5
        ), "Running var mismatch"

    def test_no_affine(self, sample_input):
        """Batch norm without learnable parameters."""
        custom = BatchNorm2d(16, affine=False)
        pytorch = nn.BatchNorm2d(16, affine=False)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_custom_momentum(self, sample_input):
        """Test with different momentum value."""
        custom = BatchNorm2d(16, momentum=0.2)
        pytorch = nn.BatchNorm2d(16, momentum=0.2)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_custom_eps(self, sample_input):
        """Test with different epsilon value."""
        custom = BatchNorm2d(16, eps=1e-3)
        pytorch = nn.BatchNorm2d(16, eps=1e-3)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, sample_input)

    def test_single_channel(self):
        """Single channel input."""
        x = torch.randn(8, 1, 16, 16)
        custom = BatchNorm2d(1)
        pytorch = nn.BatchNorm2d(1)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, x)

    def test_different_spatial_sizes(self):
        """Test various spatial dimensions."""
        for h, w in [(4, 4), (7, 7), (16, 8), (1, 1)]:
            x = torch.randn(2, 8, h, w)
            custom = BatchNorm2d(8)
            pytorch = nn.BatchNorm2d(8)
            custom.train()
            pytorch.train()
            self._compare_bn(custom, pytorch, x)

    def test_large_batch(self):
        """Large batch size."""
        x = torch.randn(64, 32, 4, 4)
        custom = BatchNorm2d(32)
        pytorch = nn.BatchNorm2d(32)
        custom.train()
        pytorch.train()
        self._compare_bn(custom, pytorch, x)
