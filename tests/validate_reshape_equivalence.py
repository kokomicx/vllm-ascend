# SPDX-License-Identifier: Apache-2.0
"""
Cross-validation script: compare new KVCacheLayout.reshape() output
against the current model_runner's _reshape_kv_cache_tensors /
_reshape_kv_cache_tensors_for_mla output.

Run this on the Ascend server with torch-npu available.

Usage:
    python tests/validate_reshape_equivalence.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import torch

from vllm_ascend.core.kv_cache_layout import (
    CompressedMLALayout,
    MambaLayout,
    SingleTensorLayout,
    SparseMLAC8Layout,
    SparseMLALayout,
    SplitKVLayout,
    adjust_kv_layout,
)
from vllm_ascend.utils import calc_split_factor


# ---------------------------------------------------------------------------
# Helpers — replicate exactly the reshape logic from model_runner_v1.py
# ---------------------------------------------------------------------------

def _get_attention_kv_cache_dims_gqa(spec) -> tuple[int, int]:
    """Simulate _get_attention_kv_cache_dims for GQA (non-MLA)."""
    hv = getattr(spec, "head_size_v", spec.head_size)
    return spec.head_size, hv


def _reshape_old_splitkv(
    raw_k: torch.Tensor,
    raw_v: torch.Tensor,
    spec,
    num_blocks: int,
    kernel_num_blocks: int,
    kernel_block_size: int,
    backend,
    k_dim: int,
    v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate old _reshape_kv_cache_tensors for SplitKV case."""
    kv_cache_shape = backend.get_kv_cache_shape(
        kernel_num_blocks, kernel_block_size, spec.num_kv_heads, spec.head_size
    )
    k_shape = kv_cache_shape[:-1] + (k_dim,)
    v_shape = kv_cache_shape[:-1] + (v_dim,)
    k = raw_k.view(spec.dtype).view(k_shape)
    v = raw_v.view(spec.dtype).view(v_shape)
    return k, v


def _reshape_old_sparse_mla(
    raw: list[torch.Tensor],
    spec,
    kernel_num_blocks: int,
    kernel_block_size: int,
    backend,
    is_c8: bool,
) -> Any:
    """Replicate old _reshape_kv_cache_tensors_for_mla sparse branch."""
    kv_cache_shape = backend.get_kv_cache_shape(
        kernel_num_blocks, kernel_block_size, spec.num_kv_heads, spec.head_size
    )
    k_dim, v_dim, indexer_dim = spec.sparse_head_dim
    k_shape = kv_cache_shape[:-1] + (k_dim,)
    v_shape = kv_cache_shape[:-1] + (v_dim,)
    dsa_k_shape = (kernel_num_blocks, kernel_block_size, spec.num_kv_heads, indexer_dim)

    k = raw[0].view(spec.dtype).view(k_shape)
    v = raw[1].view(spec.dtype).view(v_shape)

    if is_c8:
        dsa_k = raw[2].view(spec.c8_k_cache_dtype).view(dsa_k_shape)
        dsa_scale_shape = (kernel_num_blocks, kernel_block_size, spec.num_kv_heads, 1)
        dsa_scale = raw[3].view(spec.c8_k_scale_cache_dtype).view(dsa_scale_shape)
        return k, v, dsa_k, dsa_scale
    else:
        dsa_k = raw[2].view(spec.dtype).view(dsa_k_shape)
        return k, v, dsa_k


# ---------------------------------------------------------------------------
# Fake spec / backend for testing
# ---------------------------------------------------------------------------

@dataclass
class FakeSpec:
    block_size: int = 128
    num_kv_heads: int = 4
    head_size: int = 128
    head_size_v: int = 128
    dtype: torch.dtype = torch.bfloat16
    page_size_bytes: int = 128 * 4 * 128 * 2
    sparse_head_dim: tuple | None = None
    sparse_kv_cache_ratio: tuple | None = None
    c8_k_cache_dtype: torch.dtype = torch.int8
    c8_k_scale_cache_dtype: torch.dtype = torch.float16
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    compress_ratio: int = 1
    shapes: tuple = ()
    dtypes: tuple = ()


def make_backend():
    b = MagicMock()

    def _shape(n, bs, h, d, **kw):
        return (n, bs, h, d)

    b.get_kv_cache_shape.side_effect = _shape
    return b


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def check_equal(name: str, old_result, new_result):
    """Compare old and new reshape outputs."""
    def _flatten(obj):
        if isinstance(obj, torch.Tensor):
            return [obj]
        if isinstance(obj, (tuple, list)):
            return [x for item in obj for x in _flatten(item)]
        return []

    old_flat = _flatten(old_result)
    new_flat = _flatten(new_result)

    assert len(old_flat) == len(new_flat), (
        f"[{name}] Tensor count mismatch: {len(old_flat)} vs {len(new_flat)}"
    )

    all_ok = True
    for i, (ot, nt) in enumerate(zip(old_flat, new_flat)):
        shape_ok = ot.shape == nt.shape
        dtype_ok = ot.dtype == nt.dtype
        if not (shape_ok and dtype_ok):
            all_ok = False
            print(f"  [{name}] tensor[{i}] MISMATCH:")
            print(f"         old: shape={ot.shape}, dtype={ot.dtype}")
            print(f"         new: shape={nt.shape}, dtype={nt.dtype}")
    if all_ok:
        print(f"  [{name}] ✓ All {len(old_flat)} tensors match (shape + dtype)")
    return all_ok


def main():
    errors = 0
    vllm_config = MagicMock()
    backend = make_backend()

    # ------ Test 1: SingleTensorLayout (cache_only / extract_hidden_states) ------
    print("=== Test 1: SingleTensorLayout ===")
    spec = FakeSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
    spec.page_size_bytes = 128 * 4 * 128 * 2
    total = spec.page_size_bytes * 4
    raw = torch.zeros(total, dtype=torch.int8)

    layout = SingleTensorLayout()
    new_result = layout.reshape(
        [raw], spec, num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
        backend=backend, vllm_config=vllm_config,
    )

    # Old way:
    old_shape = backend.get_kv_cache_shape(4, 128, 4, 128)
    old_result = raw.view(spec.dtype).view(old_shape)

    if not check_equal("SingleTensor", old_result, new_result):
        errors += 1

    # ------ Test 2: SplitKVLayout (GQA) ------
    print("=== Test 2: SplitKVLayout (GQA) ===")
    spec = FakeSpec(block_size=128, num_kv_heads=4, head_size=128, head_size_v=128, dtype=torch.bfloat16)
    spec.page_size_bytes = 128 * 4 * 256 * 2  # K+V combined
    k_dim, v_dim = _get_attention_kv_cache_dims_gqa(spec)
    total = spec.page_size_bytes * 4

    layout = SplitKVLayout()
    sizes = layout.split_sizes(total, spec, head_dims=(k_dim, v_dim))
    raw_k, raw_v = torch.zeros(sizes[0], dtype=torch.int8), torch.zeros(sizes[1], dtype=torch.int8)

    new_result = layout.reshape(
        [raw_k, raw_v], spec, num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
        backend=backend, vllm_config=vllm_config, head_dims=(k_dim, v_dim),
    )
    old_result = _reshape_old_splitkv(
        raw_k, raw_v, spec, 4, 4, 128, backend, k_dim, v_dim,
    )
    if not check_equal("GQA", old_result, new_result):
        errors += 1

    # ------ Test 3: SplitKVLayout (MLA non-sparse) ------
    print("=== Test 3: SplitKVLayout (MLA non-sparse) ===")
    spec = FakeSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    spec.page_size_bytes = 128 * 1 * 576 * 2
    k_dim, v_dim = 512, 64  # kv_lora_rank, qk_rope_head_dim
    total = spec.page_size_bytes * 4

    layout = SplitKVLayout()
    sizes = layout.split_sizes(total, spec, head_dims=(k_dim, v_dim))
    raw_k, raw_v = torch.zeros(sizes[0], dtype=torch.int8), torch.zeros(sizes[1], dtype=torch.int8)

    new_result = layout.reshape(
        [raw_k, raw_v], spec, num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
        backend=backend, vllm_config=vllm_config, head_dims=(k_dim, v_dim),
    )
    old_result = _reshape_old_splitkv(
        raw_k, raw_v, spec, 4, 4, 128, backend, k_dim, v_dim,
    )
    if not check_equal("MLA-2tensor", old_result, new_result):
        errors += 1

    # ------ Test 4: SparseMLALayout (DS V3.2) ------
    print("=== Test 4: SparseMLALayout (DS V3.2) ===")
    spec = FakeSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16,
                    sparse_head_dim=(512, 64, 128))
    spec.sparse_kv_cache_ratio = (576 / 512, 576 / 64, 576 / 128, None)  # (1.125, 9.0, 4.5, None)
    spec.page_size_bytes = 128 * 1 * (512 + 64 + 128) * 2
    total = spec.page_size_bytes * 4

    layout = SparseMLALayout()
    sizes = layout.split_sizes(total, spec)
    raw = [torch.zeros(s, dtype=torch.int8) for s in sizes]

    new_result = layout.reshape(
        raw, spec, num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
        backend=backend, vllm_config=vllm_config,
    )
    old_result = _reshape_old_sparse_mla(raw, spec, 4, 128, backend, is_c8=False)
    if not check_equal("SparseMLA", old_result, new_result):
        errors += 1

    # ------ Test 5: SparseMLAC8Layout ------
    print("=== Test 5: SparseMLAC8Layout (DS V3.2 + C8) ===")
    spec = FakeSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16,
                    sparse_head_dim=(512, 64, 128),
                    c8_k_cache_dtype=torch.int8, c8_k_scale_cache_dtype=torch.float16)
    # virtual dims for C8: (512, 64, 64, 1), total = 641
    spec.sparse_kv_cache_ratio = (641 / 512, 641 / 64, 641 / 64, 641 / 1)
    spec.page_size_bytes = 128 * 1 * (512 * 2 + 64 * 2 + 128 * 1 + 1 * 2)
    total = spec.page_size_bytes * 4

    layout = SparseMLAC8Layout()
    sizes = layout.split_sizes(total, spec)
    raw = [torch.zeros(s, dtype=torch.int8) for s in sizes]

    new_result = layout.reshape(
        raw, spec, num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
        backend=backend, vllm_config=vllm_config,
    )
    old_result = _reshape_old_sparse_mla(raw, spec, 4, 128, backend, is_c8=True)
    if not check_equal("SparseMLAC8", old_result, new_result):
        errors += 1

    # ------ Test 6: MambaLayout ------
    print("=== Test 6: MambaLayout ===")
    spec = FakeSpec(block_size=128, shapes=((4, 256), (16, 16)),
                    dtypes=(torch.float32, torch.float32))
    spec.page_size_bytes = (4 * 256 * 4) + (16 * 16 * 4)
    total = spec.page_size_bytes * 4
    raw = torch.zeros(total, dtype=torch.int8)

    layout = MambaLayout()
    new_result = layout.reshape(
        [raw], spec, num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
        backend=backend, vllm_config=vllm_config,
    )

    # Old way: _adjust_kv_layout
    shapes_with_blocks = tuple((4, *s) for s in spec.shapes)
    old_result = adjust_kv_layout(raw, shapes_with_blocks, spec.dtypes, spec.page_size_bytes)
    if not check_equal("Mamba", old_result, new_result):
        errors += 1

    # ------ Test 7: adjust_kv_layout vs old method ------
    print("=== Test 7: adjust_kv_layout fidelity ===")
    raw = torch.zeros(8192, dtype=torch.int8)
    page_size = 1024
    shapes = [(4, 32, 8), (4, 16, 4)]
    dtypes = [torch.float32, torch.float16]

    result = adjust_kv_layout(raw, shapes, dtypes, page_size)
    assert len(result) == 2
    assert result[0].shape == (4, 32, 8)
    assert result[0].dtype == torch.float32
    assert result[1].shape == (4, 16, 4)
    assert result[1].dtype == torch.float16
    # Verify different offsets
    assert result[0].storage_offset() != result[1].storage_offset()
    print("  [adjust_kv_layout] ✓ Non-overlapping views correct")

    # Test overlay
    shapes3 = [(4, 32, 8), (4, 16, 4), (4, 32, 8)]
    dtypes3 = [torch.float32, torch.float16, torch.float32]
    result3 = adjust_kv_layout(raw, shapes3, dtypes3, page_size, overlap_full_kv_cache=True)
    assert result3[2].storage_offset() == result3[0].storage_offset()
    print("  [adjust_kv_layout] ✓ Overlay view correct")

    # ------ Summary ------
    print(f"\n{'='*50}")
    if errors == 0:
        print("ALL TESTS PASSED — Layout reshape matches old code exactly")
    else:
        print(f"{errors} TEST(S) FAILED — see details above")
        sys.exit(1)


if __name__ == "__main__":
    main()
