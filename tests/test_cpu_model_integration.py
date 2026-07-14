# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPU-only integration test: Layout dispatch → split_sizes → reshape
using exact parameters from each target model family.

This test validates the FULL allocate/reshape pipeline on CPU without
requiring NPU memory, model weights, or even torch-npu.  It is designed
to run anywhere (local dev machine, CI, etc.) and gives high confidence
that the Layout-driven refactoring produces correct KV cache shapes.

Coverage by model:
  - Qwen3-8B / Qwen3-MoE       (GQA → SplitKVLayout)
  - Qwen2.5 / Llama-3.1        (GQA → SplitKVLayout)
  - Standard MLA (DS V3.1)     (SplitKVLayout with MLA dims)
  - DS V3.2 sparse MLA         (SparseMLALayout)
  - DS V3.2 sparse MLA + C8    (SparseMLAC8Layout)
  - DS V4 compressed MLA       (CompressedMLALayout)
  - Mamba / Hybrid              (MambaLayout / SingleTensorLayout)
  - Cache-only layers           (SingleTensorLayout)
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# Activate monkey-patching before importing downstream classes
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    HiddenStateCacheSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend.core.kv_cache_layout import (
    CompressedMLALayout,
    KVCacheLayout,
    MambaLayout,
    SingleTensorLayout,
    SparseMLAC8Layout,
    SparseMLALayout,
    SplitKVLayout,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _backend_mock(kernel_num_blocks, kernel_block_size, num_kv_heads, head_size):
    """Simulate FA3/Attention backend get_kv_cache_shape."""
    backend = MagicMock()
    backend.get_kv_cache_shape.return_value = (
        kernel_num_blocks, kernel_block_size, num_kv_heads, head_size,
    )
    return backend


def _compute_page_size_bytes(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    split_kv: bool = True,
) -> int:
    """Compute page_size_bytes matching Ascend platform conventions.

    For SplitKVLayout (GQA/MLA): K + V are separate pages, each page is
    ``block_size * num_kv_heads * head_size * dtype_bytes``.  The raw
    allocation is 2× that, but the *spec* page_size_bytes in vLLM is
    per-head-size and the raw buffer is allocated with split_sizes.
    """
    dtype_bytes = torch.finfo(dtype).bits // 8
    page_bytes = block_size * num_kv_heads * head_size * dtype_bytes
    return page_bytes * (2 if split_kv else 1)


def _reshape_and_check(
    layout: KVCacheLayout,
    spec,
    raw_tensors: list[torch.Tensor],
    backend: MagicMock,
    num_blocks: int,
    kernel_num_blocks: int,
    kernel_block_size: int,
    vllm_config: MagicMock,
    **kwargs,
):
    """Call layout.reshape and verify contiguity of ALL output tensors."""
    result = layout.reshape(
        raw_tensors, spec,
        num_blocks=num_blocks,
        kernel_num_blocks=kernel_num_blocks,
        kernel_block_size=kernel_block_size,
        backend=backend,
        vllm_config=vllm_config,
        **kwargs,
    )

    # Flatten and check
    def _check(t, name):
        assert t.is_contiguous(), f"{name}: tensor not contiguous"
        return t.shape

    if isinstance(result, torch.Tensor):
        return {"tensor": _check(result, "output")}
    elif isinstance(result, (tuple, list)):
        return {
            f"elem_{i}": _check(t, f"output[{i}]")
            for i, t in enumerate(result)
        }
    return {}


# ---------------------------------------------------------------------------
# Model parameter definitions
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ModelParams:
    """KV-cache-relevant parameters for a target model."""
    name: str
    num_layers: int
    num_kv_heads: int
    head_size: int
    k_dim: int
    v_dim: int
    dtype: torch.dtype = torch.bfloat16
    block_size: int = 128

    # MLA-specific
    use_mla: bool = False
    sparse_head_dim: tuple[int, int, int] | None = None  # (kv_lora, qk_rope, indexer)
    cache_sparse_c8: bool = False
    compress_ratio: int = 1
    scale_dim: int = 0
    sliding_window: int | None = None

    # Mamba
    mamba_shapes: tuple[tuple[int, ...], ...] = ()
    mamba_dtypes: tuple[torch.dtype, ...] = ()


# Known parameters for each target model family
MODELS: list[ModelParams] = [
    # ── GQA family ──────────────────────────────────────────────
    ModelParams(
        name="Qwen3-8B",
        num_layers=32,  # incl. 1 cache_only layer
        num_kv_heads=8,
        head_size=128,
        k_dim=128,
        v_dim=128,
    ),
    ModelParams(
        name="Qwen3-MoE-Instruct",
        num_layers=28,
        num_kv_heads=4,
        head_size=128,
        k_dim=128,
        v_dim=128,
    ),
    ModelParams(
        name="Qwen2.5-7B (GQA reference)",
        num_layers=28,
        num_kv_heads=4,
        head_size=128,
        k_dim=128,
        v_dim=128,
    ),

    # ── Standard MLA family ─────────────────────────────────────
    ModelParams(
        name="DS-V3.1 (MLA, no sparse)",
        num_layers=60,
        num_kv_heads=1,
        head_size=576,  # kv_lora_rank + qk_rope_head_dim
        k_dim=512,      # kv_lora_rank
        v_dim=64,       # qk_rope_head_dim
        use_mla=True,
    ),

    # ── Sparse MLA family ───────────────────────────────────────
    ModelParams(
        name="DS-V3.2 (sparse MLA, bf16)",
        num_layers=60,
        num_kv_heads=1,
        head_size=704,  # 512 + 64 + 128
        k_dim=512,
        v_dim=64,
        use_mla=True,
        sparse_head_dim=(512, 64, 128),
    ),
    ModelParams(
        name="DS-V3.2 (sparse MLA + C8)",
        num_layers=60,
        num_kv_heads=1,
        head_size=704,
        k_dim=512,
        v_dim=64,
        use_mla=True,
        sparse_head_dim=(512, 64, 128),
        cache_sparse_c8=True,
    ),

    # ── Compressed MLA family ───────────────────────────────────
    ModelParams(
        name="DS-V4 (fp8 compressed MLA)",
        num_layers=60,
        num_kv_heads=1,
        head_size=512,
        k_dim=512,
        v_dim=64,
        use_mla=True,
        compress_ratio=4,
        scale_dim=0,
    ),
    ModelParams(
        name="DS-V4 (fp8 compressed MLA + scale)",
        num_layers=60,
        num_kv_heads=1,
        head_size=512,
        k_dim=512,
        v_dim=64,
        use_mla=True,
        compress_ratio=4,
        scale_dim=8,
    ),

    # ── Sliding-window MLA ──────────────────────────────────────
    ModelParams(
        name="DS-V4 (sliding window MLA)",
        num_layers=60,
        num_kv_heads=1,
        head_size=512,
        k_dim=512,
        v_dim=64,
        use_mla=True,
        sliding_window=4096,
        compress_ratio=4,
        scale_dim=8,
    ),

    # ── Mamba family ────────────────────────────────────────────
    ModelParams(
        name="Mamba-2.8B (SSM reference)",
        num_layers=64,
        num_kv_heads=0,          # N/A for Mamba
        head_size=0,
        k_dim=0,
        v_dim=0,
        mamba_shapes=((4, 256), (16, 16)),
        mamba_dtypes=(torch.float32, torch.float32),
    ),
]


# ---------------------------------------------------------------------------
# Helper: create appropriate KVCacheSpec from ModelParams
# ---------------------------------------------------------------------------

def create_spec(m: ModelParams, **overrides) -> KVCacheSpec:
    """Create a vLLM KVCacheSpec matching the model parameters."""
    if m.mamba_shapes:
        return MambaSpec(
            block_size=m.block_size,
            shapes=m.mamba_shapes,
            dtypes=m.mamba_dtypes,
        )

    if m.sliding_window is not None:
        return SlidingWindowMLASpec(
            block_size=m.block_size,
            num_kv_heads=m.num_kv_heads,
            head_size=m.head_size,
            dtype=m.dtype,
            sliding_window=m.sliding_window,
            cache_dtype_str=None,
            compress_ratio=m.compress_ratio,
            model_version="deepseek_v4",
            **overrides,
        )

    if m.use_mla:
        extra: dict[str, Any] = {}
        if m.sparse_head_dim is not None:
            extra["sparse_head_dim"] = m.sparse_head_dim
            extra["cache_sparse_c8"] = m.cache_sparse_c8
        if m.compress_ratio > 1:
            extra["compress_ratio"] = m.compress_ratio
            extra["scale_dim"] = m.scale_dim
            extra["scale_dtype"] = torch.int8
        return MLAAttentionSpec(
            block_size=m.block_size,
            num_kv_heads=m.num_kv_heads,
            head_size=m.head_size,
            dtype=m.dtype,
            **extra,
            **overrides,
        )

    return FullAttentionSpec(
        block_size=m.block_size,
        num_kv_heads=m.num_kv_heads,
        head_size=m.head_size,
        dtype=m.dtype,
        head_size_v=m.v_dim,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Test: Spec → Layout dispatch (one per model)
# ---------------------------------------------------------------------------

class TestSpecToLayoutDispatch:
    """Every model's spec maps to the expected Layout class."""

    @pytest.mark.parametrize("m", MODELS, ids=lambda m: m.name)
    def test_layout_dispatch(self, m: ModelParams):
        spec = create_spec(m)
        layout = spec.get_kv_cache_layout()

        # Determine expected layout
        if m.mamba_shapes:
            expected = MambaLayout
        elif m.compress_ratio > 1:
            expected = CompressedMLALayout
        elif m.cache_sparse_c8:
            expected = SparseMLAC8Layout
        elif m.sparse_head_dim is not None:
            expected = SparseMLALayout
        else:
            expected = SplitKVLayout

        assert isinstance(layout, expected), (
            f"{m.name}: expected {expected.__name__}, got {type(layout).__name__}"
        )


# ---------------------------------------------------------------------------
# Test: HiddenStateCacheSpec → SingleTensorLayout
# ---------------------------------------------------------------------------

def test_cache_only_layers():
    """Hidden state cache (draft model / cache_only) → SingleTensorLayout."""
    spec = HiddenStateCacheSpec(
        block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SingleTensorLayout)


# ---------------------------------------------------------------------------
# Test: allocate + reshape for every model
# ---------------------------------------------------------------------------

class TestAllocateAndReshape:
    """End-to-end split_sizes + reshape on CPU for each model type."""

    NUM_BLOCKS = 4
    KERNEL_BLOCK_SIZE = 128

    @pytest.mark.parametrize("m", MODELS, ids=lambda m: m.name)
    def test_split_then_reshape(self, m: ModelParams):
        """Allocate raw int8 buffers, reshape them, check shapes & contiguity."""
        spec = create_spec(m)
        layout = spec.get_kv_cache_layout()

        # --- Build kwargs for split_sizes / reshape ---
        kwargs: dict[str, Any] = {}
        if isinstance(layout, SplitKVLayout):
            kwargs["head_dims"] = (m.k_dim, m.v_dim)
            kwargs["vllm_config"] = MagicMock()
            kwargs["layer_name"] = "test_layer"
        elif isinstance(layout, (SparseMLALayout, SparseMLAC8Layout)):
            # sparse_kv_cache_ratio is a property on AscendMLAAttentionSpec
            pass  # computed automatically
        elif isinstance(layout, CompressedMLALayout):
            pass

        # --- Compute total_bytes from page_size_bytes ---
        page_size_bytes = spec.page_size_bytes
        total_bytes = page_size_bytes * self.NUM_BLOCKS

        # --- split_sizes ---
        sizes = layout.split_sizes(total_bytes, spec, **kwargs)
        assert len(sizes) == layout.num_tensors(), (
            f"{m.name}: expected {layout.num_tensors()} sizes, got {len(sizes)}"
        )
        # Sum must equal total_bytes EXACTLY for page-aligned totals
        diff = total_bytes - sum(sizes)
        assert diff == 0, (
            f"{m.name}: split_sizes sum={sum(sizes)} != total={total_bytes} "
            f"(diff={diff}, sizes={sizes})"
        )

        # --- Allocate raw int8 buffers ---
        raw_tensors = [torch.zeros(s, dtype=torch.int8) for s in sizes]

        # --- reshape ---
        backend = _backend_mock(
            kernel_num_blocks=self.NUM_BLOCKS,
            kernel_block_size=self.KERNEL_BLOCK_SIZE,
            num_kv_heads=max(m.num_kv_heads, 1),
            head_size=m.head_size or max(m.mamba_shapes[0]) if m.mamba_shapes else 128,
        )
        result = layout.reshape(
            raw_tensors, spec,
            num_blocks=self.NUM_BLOCKS,
            kernel_num_blocks=self.NUM_BLOCKS,
            kernel_block_size=self.KERNEL_BLOCK_SIZE,
            backend=backend,
            vllm_config=kwargs.get("vllm_config", MagicMock()),
            **{k: v for k, v in kwargs.items() if k in ("head_dims",)},
        )

        # --- Verify output structure ---
        if isinstance(layout, MambaLayout):
            assert isinstance(result, list), f"{m.name}: MambaLayout must return list"
            assert len(result) == len(m.mamba_shapes), (
                f"{m.name}: expected {len(m.mamba_shapes)} Mamba states, "
                f"got {len(result)}"
            )
            for i, (t, shape) in enumerate(zip(result, m.mamba_shapes)):
                expected = (self.NUM_BLOCKS, *shape)
                assert t.shape == expected, (
                    f"{m.name} state[{i}]: expected {expected}, got {t.shape}"
                )
                assert t.is_contiguous(), f"{m.name} state[{i}] not contiguous"

        elif isinstance(layout, CompressedMLALayout):
            assert isinstance(result, list), f"{m.name}: CompressedMLALayout must return list"
            assert result[0].dtype == m.dtype
            assert result[0].is_contiguous()
            # num_blocks, block_size, num_kv_heads, head_size
            assert result[0].shape == (
                self.NUM_BLOCKS, m.block_size, m.num_kv_heads, m.head_size,
            ), f"{m.name}: unexpected shape {result[0].shape}"

        elif isinstance(layout, SplitKVLayout):
            assert isinstance(result, tuple), f"{m.name}: SplitKVLayout must return tuple"
            assert len(result) == 2
            k_cache, v_cache = result
            # kv_cache_shape = (N, BS, H, D)
            base = (self.NUM_BLOCKS, self.KERNEL_BLOCK_SIZE, m.num_kv_heads)
            assert k_cache.shape == base + (m.k_dim,), (
                f"{m.name} K: expected {base + (m.k_dim,)}, got {k_cache.shape}"
            )
            assert v_cache.shape == base + (m.v_dim,), (
                f"{m.name} V: expected {base + (m.v_dim,)}, got {v_cache.shape}"
            )
            assert k_cache.dtype == m.dtype
            assert v_cache.dtype == m.dtype
            assert k_cache.is_contiguous()
            assert v_cache.is_contiguous()

        elif isinstance(layout, SparseMLALayout):
            assert isinstance(result, tuple)
            assert len(result) == 3
            k, v, dsa_k = result
            base = (self.NUM_BLOCKS, self.KERNEL_BLOCK_SIZE, m.num_kv_heads)
            assert k.shape == base + (m.sparse_head_dim[0],), f"{m.name}: k {k.shape}"
            assert v.shape == base + (m.sparse_head_dim[1],), f"{m.name}: v {v.shape}"
            assert dsa_k.shape == base + (m.sparse_head_dim[2],), f"{m.name}: dsa_k {dsa_k.shape}"
            assert k.is_contiguous() and v.is_contiguous() and dsa_k.is_contiguous()

        elif isinstance(layout, SparseMLAC8Layout):
            assert isinstance(result, tuple)
            assert len(result) == 4
            k, v, dsa_k, dsa_scale = result
            base = (self.NUM_BLOCKS, self.KERNEL_BLOCK_SIZE, m.num_kv_heads)
            assert k.dtype == m.dtype
            assert v.dtype == m.dtype
            assert dsa_k.dtype == torch.int8
            assert dsa_scale.dtype == torch.float16
            assert dsa_scale.shape == base + (1,), f"{m.name}: scale {dsa_scale.shape}"

        elif isinstance(layout, SingleTensorLayout):
            assert isinstance(result, torch.Tensor)
            assert result.is_contiguous()

    def test_page_size_bytes_positive(self):
        """Every model's spec.page_size_bytes must be > 0."""
        for m in MODELS:
            if m.mamba_shapes:
                continue  # Mamba has its own page_size computation
            spec = create_spec(m)
            assert spec.page_size_bytes > 0, (
                f"{m.name}: page_size_bytes = {spec.page_size_bytes}"
            )


# ---------------------------------------------------------------------------
# Test: needs_alignment
# ---------------------------------------------------------------------------

def test_alignment_requirements():
    """PD alignment is only required for SplitKV, SparseMLA, SparseMLAC8."""
    assert SingleTensorLayout().needs_alignment() is False
    assert SplitKVLayout().needs_alignment() is True
    assert SparseMLALayout().needs_alignment() is True
    assert SparseMLAC8Layout().needs_alignment() is True
    assert MambaLayout().needs_alignment() is False
    assert CompressedMLALayout().needs_alignment() is False


# ---------------------------------------------------------------------------
# Test: num_tensors consistency
# ---------------------------------------------------------------------------

def test_num_tensors_consistency():
    """layout.num_tensors() matches len(split_sizes(...))."""
    for m in MODELS:
        spec = create_spec(m)
        layout = spec.get_kv_cache_layout()
        kwargs: dict[str, Any] = {}
        if isinstance(layout, SplitKVLayout):
            kwargs["head_dims"] = (m.k_dim, m.v_dim)

        sizes = layout.split_sizes(spec.page_size_bytes * 4, spec, **kwargs)
        assert len(sizes) == layout.num_tensors(), (
            f"{m.name} ({type(layout).__name__}): "
            f"len(split_sizes)={len(sizes)} != num_tensors={layout.num_tensors()}"
        )


# ---------------------------------------------------------------------------
# Test: Multiple block counts (stress-test shapes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_blocks", [1, 2, 8, 32, 128])
def test_variable_block_counts(num_blocks: int):
    """Reshape works correctly across a range of block counts."""
    m = MODELS[0]  # Qwen3-8B
    spec = create_spec(m)
    layout = spec.get_kv_cache_layout()

    total_bytes = spec.page_size_bytes * num_blocks
    kwargs: dict[str, Any] = {}
    if isinstance(layout, SplitKVLayout):
        kwargs["head_dims"] = (m.k_dim, m.v_dim)

    sizes = layout.split_sizes(total_bytes, spec, **kwargs)
    assert sum(sizes) == total_bytes

    raw = [torch.zeros(s, dtype=torch.int8) for s in sizes]
    backend = _backend_mock(num_blocks, 128, m.num_kv_heads, m.head_size)
    result = layout.reshape(
        raw, spec,
        num_blocks=num_blocks,
        kernel_num_blocks=num_blocks,
        kernel_block_size=128,
        backend=backend,
        vllm_config=MagicMock(),
        **kwargs,
    )

    k, v = result
    assert k.shape[0] == num_blocks, f"K: expected {num_blocks} blocks, got {k.shape[0]}"
    assert v.shape[0] == num_blocks, f"V: expected {num_blocks} blocks, got {v.shape[0]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
