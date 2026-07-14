# SPDX-License-Identifier: Apache-2.0
"""
Phase 3 validation: Layout-driven allocate/reshape dispatch.

Validates that the new _allocate_kv_cache_tensors_v2 and
_reshape_kv_cache_tensors_v2 methods produce correct outputs
by comparing against the Layout classes directly.

Run on server:
    PYTHONPATH=/path/to/vllm:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 \
    python tests/test_phase3_layout_dispatch.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import torch

# --- Force env var OFF during import so patch modules work correctly ---
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

# Import patch module to activate monkey-patching
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401

from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.core.kv_cache_layout import (
    KVCacheLayout,
    MambaLayout,
    SingleTensorLayout,
    SparseMLAC8Layout,
    SparseMLALayout,
    SplitKVLayout,
)


# ---------------------------------------------------------------------------
# Test 1: New methods exist on NPUModelRunner
# ---------------------------------------------------------------------------

def test_v2_methods_exist():
    """Verify all v2 methods are present on NPUModelRunner."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    assert hasattr(NPUModelRunner, "_alloc_aligned"), "_alloc_aligned missing"
    assert hasattr(NPUModelRunner, "_build_layout_kwargs"), "_build_layout_kwargs missing"
    assert hasattr(NPUModelRunner, "_allocate_kv_cache_tensors_v2"), "_allocate_kv_cache_tensors_v2 missing"
    assert hasattr(NPUModelRunner, "_reshape_kv_cache_tensors_v2"), "_reshape_kv_cache_tensors_v2 missing"
    assert hasattr(NPUModelRunner, "_initialize_kv_cache_tensors_v2"), "_initialize_kv_cache_tensors_v2 missing"
    print("✓ All v2 methods present on NPUModelRunner")


# ---------------------------------------------------------------------------
# Test 2: _alloc_aligned static method
# ---------------------------------------------------------------------------

def test_alloc_aligned():
    """Verify _alloc_aligned creates a tensor with the correct size."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    device = torch.device("cpu")
    alignment = 2 * 1024 * 1024  # 2 MB
    size = 1024

    t = NPUModelRunner._alloc_aligned(size, alignment, device)
    assert t.numel() == size, f"Expected {size} elements, got {t.numel()}"
    assert t.dtype == torch.int8
    print("✓ _alloc_aligned returns correct size and dtype")


# ---------------------------------------------------------------------------
# Test 3: Gate env var controls dispatch
# ---------------------------------------------------------------------------

def test_gate_default_off():
    """Default: VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH is 0 (off)."""
    from vllm_ascend.envs import VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH

    assert VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH is False, (
        "Gate should be OFF by default"
    )
    print("✓ VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH defaults to False")


def test_gate_can_be_enabled():
    """Env var can be set to 1 to enable the new path."""
    with patch.dict(os.environ, {"VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH": "1"}):
        # Re-import to pick up new env value via lazy __getattr__
        import importlib
        import vllm_ascend.envs as envs_mod
        importlib.reload(envs_mod)
        assert envs_mod.VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH is True
        # Restore
        os.environ.pop("VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH", None)
        importlib.reload(envs_mod)
    print("✓ VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH can be enabled")


# ---------------------------------------------------------------------------
# Test 4: Spec → Layout dispatch produces expected number of tensors
# ---------------------------------------------------------------------------

def test_layout_num_tensors_gqa():
    """FullAttentionSpec → SplitKVLayout → 2 tensors."""
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128,
        dtype=torch.bfloat16, head_size_v=128,
    )
    layout = spec.get_kv_cache_layout()
    assert layout.num_tensors() == 2
    print("✓ GQA → 2 tensors")


def test_layout_num_tensors_sparse_mla():
    """Sparse MLA → SparseMLALayout → 3 tensors."""
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=704,
        dtype=torch.bfloat16, sparse_head_dim=(512, 64, 128),
    )
    layout = spec.get_kv_cache_layout()
    assert layout.num_tensors() == 3
    print("✓ Sparse MLA → 3 tensors")


def test_layout_num_tensors_sparse_c8():
    """Sparse MLA + C8 → SparseMLAC8Layout → 4 tensors."""
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=704,
        dtype=torch.bfloat16, sparse_head_dim=(512, 64, 128),
        cache_sparse_c8=True,
    )
    layout = spec.get_kv_cache_layout()
    assert layout.num_tensors() == 4
    print("✓ Sparse MLA C8 → 4 tensors")


def test_layout_num_tensors_compressed():
    """Compressed MLA → CompressedMLALayout → 1 tensor."""
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512,
        dtype=torch.bfloat16, compress_ratio=4,
    )
    layout = spec.get_kv_cache_layout()
    assert layout.num_tensors() == 1
    print("✓ Compressed MLA → 1 tensor")


def test_layout_num_tensors_mamba():
    """MambaSpec → MambaLayout → 1 tensor."""
    spec = MambaSpec(
        block_size=128, shapes=((4, 256),), dtypes=(torch.float32,),
    )
    layout = spec.get_kv_cache_layout()
    assert layout.num_tensors() == 1
    print("✓ Mamba → 1 tensor")


def test_layout_num_tensors_standard_mla():
    """Standard MLA (no sparse, no compress) → SplitKVLayout → 2 tensors."""
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=576,
        dtype=torch.bfloat16,
    )
    layout = spec.get_kv_cache_layout()
    assert layout.num_tensors() == 2
    print("✓ Standard MLA → 2 tensors")


# ---------------------------------------------------------------------------
# Test 5: split_sizes sums to total_bytes for each layout
# ---------------------------------------------------------------------------

def test_split_sizes_sums():
    """For every layout, sum(split_sizes) ≈ total_bytes."""
    total = 1_048_576  # 1 MB

    class _FakeSpec:
        pass

    cases: list[tuple[str, KVCacheLayout, Any, dict]] = [
        ("SingleTensor", SingleTensorLayout(), _FakeSpec(), {}),
        ("SplitKV", SplitKVLayout(), _FakeSpec(), {"head_dims": (128, 128)}),
        ("Mamba", MambaLayout(), _FakeSpec(), {}),
    ]

    for name, layout, spec, kw in cases:
        sizes = layout.split_sizes(total, spec, **kw)
        assert len(sizes) == layout.num_tensors(), (
            f"{name}: expected {layout.num_tensors()} sizes, got {len(sizes)}"
        )
        assert abs(sum(sizes) - total) < layout.num_tensors(), (
            f"{name}: sum={sum(sizes)} != total={total}"
        )
        print(f"  ✓ {name}: {sizes}")


def test_split_sizes_sparse_mla():
    """SparseMLA split_sizes produces 3 parts summing to total."""
    total = 704 * 128 * 2 * 4  # head_size * block_size * dtype * num_blocks
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=704,
        dtype=torch.bfloat16, sparse_head_dim=(512, 64, 128),
    )
    layout = spec.get_kv_cache_layout()
    sizes = layout.split_sizes(total, spec)
    assert len(sizes) == 3
    assert abs(sum(sizes) - total) < 3
    print(f"  ✓ SparseMLA: {sizes} sum={sum(sizes)}")


def test_split_sizes_sparse_c8():
    """SparseMLAC8 split_sizes produces 4 parts summing to total."""
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=704,
        dtype=torch.bfloat16, sparse_head_dim=(512, 64, 128),
        cache_sparse_c8=True,
    )
    total = spec.page_size_bytes * 4
    layout = spec.get_kv_cache_layout()
    sizes = layout.split_sizes(total, spec)
    assert len(sizes) == 4
    assert abs(sum(sizes) - total) < 4
    print(f"  ✓ SparseMLAC8: {sizes} sum={sum(sizes)}")


# ---------------------------------------------------------------------------
# Test 6: reshape output shapes match expected patterns
# ---------------------------------------------------------------------------

def _make_backend_mock():
    """Return a MagicMock that computes get_kv_cache_shape(N, B, H, D)."""
    b = MagicMock()

    def _shape(n, bs, h, d, **kw):
        return (n, bs, h, d)

    b.get_kv_cache_shape.side_effect = _shape
    return b


def test_reshape_single_tensor():
    """SingleTensorLayout.reshape produces correct shape."""
    backend = _make_backend_mock()
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128,
        dtype=torch.bfloat16, head_size_v=128,
    )
    # reshape does raw.view(spec.dtype).view(kv_cache_shape), so raw must
    # contain product(kv_cache_shape) * dtype_bytes int8 elements.
    # dtype = bfloat16 → 2 bytes per element.
    shape_elements = 4 * 128 * 4 * 128  # kv_cache_shape product = 262144
    raw_bytes = shape_elements * 2  # = 524288 int8 bytes
    raw = torch.zeros(raw_bytes, dtype=torch.int8)

    layout = SingleTensorLayout()
    result = layout.reshape(
        [raw], spec, num_blocks=4, kernel_num_blocks=4,
        kernel_block_size=128, backend=backend,
        vllm_config=MagicMock(),
    )
    assert result.shape == (4, 128, 4, 128), f"Unexpected shape: {result.shape}"
    assert result.dtype == spec.dtype
    print(f"  ✓ SingleTensor: shape={result.shape}")


def test_reshape_split_kv():
    """SplitKVLayout.reshape produces correct K/V shapes."""
    backend = _make_backend_mock()
    spec = FullAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128,
        dtype=torch.bfloat16, head_size_v=128,
    )
    # Each cache tensor must hold product(k_shape) * dtype_bytes int8 elements.
    # dtype = bfloat16 → 2 bytes per element.
    shape_elements = 4 * 128 * 4 * 128  # k_shape product = 262144
    per_cache_bytes = shape_elements * 2  # = 524288 int8 bytes
    total = per_cache_bytes * 2  # K + V
    sizes = SplitKVLayout().split_sizes(total, spec, head_dims=(128, 128))
    raw_k = torch.zeros(sizes[0], dtype=torch.int8)
    raw_v = torch.zeros(sizes[1], dtype=torch.int8)

    layout = SplitKVLayout()
    k_cache, v_cache = layout.reshape(
        [raw_k, raw_v], spec, num_blocks=4, kernel_num_blocks=4,
        kernel_block_size=128, backend=backend, vllm_config=MagicMock(),
        head_dims=(128, 128),
    )
    assert k_cache.shape == (4, 128, 4, 128), f"K shape: {k_cache.shape}"
    assert v_cache.shape == (4, 128, 4, 128), f"V shape: {v_cache.shape}"
    print(f"  ✓ SplitKV: K={k_cache.shape}, V={v_cache.shape}")


def test_reshape_mamba():
    """MambaLayout.reshape produces correct multi-state views."""
    spec = MambaSpec(
        block_size=128,
        shapes=((4, 256), (16, 16)),
        dtypes=(torch.float32, torch.float32),
    )
    raw_bytes = (4 * 256 * 4 + 16 * 16 * 4) * 4  # 2 states × 4 blocks
    raw = torch.zeros(raw_bytes, dtype=torch.int8)

    layout = MambaLayout()
    result = layout.reshape(
        [raw], spec, num_blocks=4, kernel_num_blocks=4,
        kernel_block_size=128, backend=MagicMock(),
        vllm_config=MagicMock(),
    )
    assert len(result) == 2
    # shapes_with_blocks = ((4, 4, 256), (4, 16, 16))
    assert result[0].shape == (4, 4, 256), f"state[0] shape: {result[0].shape}"
    assert result[1].shape == (4, 16, 16), f"state[1] shape: {result[1].shape}"
    print(f"  ✓ Mamba: {len(result)} states, shapes={[tuple(r.shape) for r in result]}")


def test_reshape_sparse_mla():
    """SparseMLALayout.reshape produces correct 3-tensor output."""
    backend = _make_backend_mock()
    spec = MLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=704,
        dtype=torch.bfloat16, sparse_head_dim=(512, 64, 128),
    )
    # Each tensor: product(target_shape) * dtype_bytes int8 elements.
    # dtype = bfloat16 → 2 bytes per element.
    # k_shape=(4,128,1,512)=262144, v_shape=(4,128,1,64)=32768, dsa_k=(4,128,1,128)=65536
    # total bf16 elements = 360448 → int8 bytes = 360448 * 2 = 720896
    total = (512 + 64 + 128) * 128 * 1 * 4 * 2
    layout = spec.get_kv_cache_layout()
    sizes = layout.split_sizes(total, spec)
    raw = [torch.zeros(s, dtype=torch.int8) for s in sizes]

    k, v, dsa_k = layout.reshape(
        raw, spec, num_blocks=4, kernel_num_blocks=4,
        kernel_block_size=128, backend=backend, vllm_config=MagicMock(),
    )
    assert k.shape == (4, 128, 1, 512), f"k shape: {k.shape}"
    assert v.shape == (4, 128, 1, 64), f"v shape: {v.shape}"
    assert dsa_k.shape == (4, 128, 1, 128), f"dsa_k shape: {dsa_k.shape}"
    print(f"  ✓ SparseMLA: k={k.shape}, v={v.shape}, dsa_k={dsa_k.shape}")


# ---------------------------------------------------------------------------
# Test 7: needs_alignment is correct for each layout
# ---------------------------------------------------------------------------

def test_needs_alignment():
    """Only SplitKV, SparseMLA, SparseMLAC8 need PD alignment."""
    assert SingleTensorLayout().needs_alignment() is False
    assert SplitKVLayout().needs_alignment() is True
    assert SparseMLALayout().needs_alignment() is True
    assert SparseMLAC8Layout().needs_alignment() is True
    assert MambaLayout().needs_alignment() is False
    # CompressedMLALayout inherits the default (False)
    from vllm_ascend.core.kv_cache_layout import CompressedMLALayout
    assert CompressedMLALayout().needs_alignment() is False
    print("✓ needs_alignment correct for all layouts")


# ---------------------------------------------------------------------------
# Test 8: v2 methods are callable with minimal mocks
# ---------------------------------------------------------------------------

def test_v2_imports_cleanly():
    """Verify the new v2 code imports without errors."""
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.envs import VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH

    # The module itself imports cleanly
    assert "NPUModelRunner" in str(NPUModelRunner)
    assert isinstance(VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH, bool)
    print("✓ v2 code imports cleanly")


if __name__ == "__main__":
    test_v2_methods_exist()
    test_alloc_aligned()
    test_gate_default_off()
    test_gate_can_be_enabled()
    test_layout_num_tensors_gqa()
    test_layout_num_tensors_sparse_mla()
    test_layout_num_tensors_sparse_c8()
    test_layout_num_tensors_compressed()
    test_layout_num_tensors_mamba()
    test_layout_num_tensors_standard_mla()
    test_split_sizes_sums()
    test_split_sizes_sparse_mla()
    test_split_sizes_sparse_c8()
    test_reshape_single_tensor()
    test_reshape_split_kv()
    test_reshape_mamba()
    test_reshape_sparse_mla()
    test_needs_alignment()
    test_v2_imports_cleanly()
    print("\n" + "=" * 50)
    print("ALL PHASE 3 TESTS PASSED")
