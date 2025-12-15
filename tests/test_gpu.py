"""Tests to verify GPU access and basic operations."""
import pytest
import torch

# Add parent directory to path for model import
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig


class TestGPUAvailability:
    """Tests for GPU detection and availability."""

    def test_cuda_is_available(self):
        """PyTorch should detect CUDA/ROCm GPU."""
        assert torch.cuda.is_available(), "No GPU detected by PyTorch"

    def test_device_count(self):
        """Should have at least one GPU device."""
        assert torch.cuda.device_count() >= 1, "No GPU devices found"

    def test_device_name(self):
        """Should be able to get device name."""
        name = torch.cuda.get_device_name(0)
        assert len(name) > 0, "Empty device name"
        print(f"GPU Device: {name}")


class TestGPUBasicOperations:
    """Tests for basic GPU tensor operations."""

    @pytest.fixture
    def device(self):
        """Get the GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return torch.device("cuda")

    def test_tensor_to_gpu(self, device):
        """Should be able to move tensor to GPU."""
        x = torch.randn(10, 10)
        x_gpu = x.to(device)
        assert x_gpu.device.type == "cuda"

    def test_tensor_operations_on_gpu(self, device):
        """Basic tensor operations should work on GPU."""
        a = torch.randn(10, 10, device=device)
        b = torch.randn(10, 10, device=device)

        # Element-wise operations
        c = a + b
        assert c.shape == (10, 10)
        assert c.device.type == "cuda"

        # Matrix multiplication
        d = torch.matmul(a, b)
        assert d.shape == (10, 10)
        assert d.device.type == "cuda"

    def test_gradient_computation_on_gpu(self, device):
        """Gradient computation should work on GPU."""
        x = torch.randn(10, 10, device=device, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()

        assert x.grad is not None
        assert x.grad.device.type == "cuda"


class TestGPUModelOperations:
    """Tests for model operations on GPU."""

    @pytest.fixture
    def device(self):
        """Get the GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return torch.device("cuda")

    @pytest.fixture
    def small_config(self):
        """Create a small GPT config for testing."""
        return GPTConfig(
            block_size=64,
            vocab_size=26,  # A-Z for scriptio continua
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=False,
        )

    def test_model_to_gpu(self, device, small_config):
        """Model should be movable to GPU."""
        model = GPT(small_config)
        model = model.to(device)

        # Check that parameters are on GPU
        for param in model.parameters():
            assert param.device.type == "cuda"

    def test_model_forward_pass(self, device, small_config):
        """Forward pass should work on GPU."""
        model = GPT(small_config).to(device)
        model.eval()

        # Create dummy input (batch_size=2, seq_len=32)
        x = torch.randint(0, small_config.vocab_size, (2, 32), device=device)

        with torch.no_grad():
            logits, loss = model(x)

        # Without targets, model returns logits only for last position (inference optimization)
        assert logits.shape == (2, 1, small_config.vocab_size)
        assert logits.device.type == "cuda"
        assert loss is None

    def test_model_backward_pass(self, device, small_config):
        """Backward pass should work on GPU."""
        model = GPT(small_config).to(device)
        model.train()

        # Create dummy input and target
        x = torch.randint(0, small_config.vocab_size, (2, 32), device=device)
        y = torch.randint(0, small_config.vocab_size, (2, 32), device=device)

        logits, loss = model(x, y)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_model_generation(self, device, small_config):
        """Text generation should work on GPU."""
        model = GPT(small_config).to(device)
        model.eval()

        # Start with a single token
        x = torch.zeros((1, 1), dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(x, max_new_tokens=10, temperature=1.0)

        assert generated.shape[1] == 11  # 1 original + 10 new tokens
        assert generated.device.type == "cuda"


class TestGPUMemory:
    """Tests for GPU memory operations."""

    @pytest.fixture
    def device(self):
        """Get the GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        return torch.device("cuda")

    def test_memory_allocation(self, device):
        """Should be able to allocate and free GPU memory."""
        # Allocate a small tensor
        x = torch.randn(100, 100, device=device)

        # Check memory is allocated
        assert torch.cuda.memory_allocated() > 0

        # Delete and clear cache
        del x
        torch.cuda.empty_cache()

    def test_memory_stats(self, device):
        """Should be able to query memory stats."""
        x = torch.randn(100, 100, device=device)

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()

        assert allocated > 0
        assert reserved >= allocated

        del x
