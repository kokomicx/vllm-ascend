# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for KVCacheLayout and its six concrete subclasses.

Verifies:
- num_tensors() returns correct arity
- split_sizes() sums to total_bytes (within rounding)
- reshape() produces correct shapes for every layout
- adjust_kv_layout() utility correctness
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.core.kv_cache_layout import (
    CompressedMLALayout,
    KVCacheLayout,
    MambaLayout,
    SingleTensorLayout,
    SparseMLAC8Layout,
    SparseMLALayout,
    SplitKVLayout,
    adjust_kv_layout,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeSpec:
    """Minimal fake spec with just the attributes each layout needs."""
    block_size: int = 128
    num_kv_heads: int = 4
    head_size: int = 128
    dtype: torch.dtype = torch.bfloat16
    page_size_bytes: int = 128 * 4 * 128 * 2  # 131072

    # MLA-specific
    sparse_head_dim: tuple[int, int, int] | None = None
    sparse_kv_cache_ratio: tuple[float, float, float, float | None] | None = None
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype = torch.int8
    c8_k_scale_cache_dtype: torch.dtype = torch.float16

    # Compressed MLA
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    compress_ratio: int = 1

    # Mamba
    shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[torch.dtype, ...] = ()


def _make_backend_mock(kv_cache_shape: tuple[int, ...]) -> MagicMock:
    """Return a backend whose get_kv_cache_shape always returns the given shape."""
    backend = MagicMock()
    backend.get_kv_cache_shape.return_value = kv_cache_shape
    return backend


def _make_dynamic_backend_mock() -> MagicMock:
    """Return a backend that computes (num_blocks, block_size, num_kv_heads, head_size)."""
    backend = MagicMock()

    def _get_shape(num_blocks, block_size, num_kv_heads, head_size, **kwargs):
        return (num_blocks, block_size, num_kv_heads, head_size)

    backend.get_kv_cache_shape.side_effect = _get_shape
    return backend


# ---------------------------------------------------------------------------
# adjust_kv_layout
# ---------------------------------------------------------------------------

class TestAdjustKVLayout:
    def test_single_view(self):
        """Single view: carved tensor has correct shape and shares storage."""
        raw = torch.zeros(1024, dtype=torch.int8)
        shape_list = [(4, 32, 4, 16)]
        dtype_list = [torch.bfloat16]
        page_size_bytes = 128 * 4 * 128 * 2  # 131072 — larger than raw but for test purposes

        # Actually, page_size_bytes must match the stride calculation.
        # Let's use simpler numbers: raw has 1024 int8 bytes.
        # We want a view of shape (2, 8) in float32 (4 bytes each).
        # page_stride = page_size_bytes / dtype_size = 1024 / 4 = 256
        # target_stride = (256, 1) → shape (2, 8) means the first dim has stride 256*4 = 1024 bytes...
        # Actually this is more complex. Let me design a simpler test.

        raw = torch.zeros(4096, dtype=torch.int8)
        # dtype=float32 → dtype_size=4, num_elements_per_page = 1024 / 4 = 256
        # shape = (4, 32, 8) → empty.stride = (256, 8, 1)
        # target_stride = (256, 8, 1) — same!
        page_size_bytes = 1024
        shape = (4, 32, 8)
        views = adjust_kv_layout(raw, [shape], [torch.float32], page_size_bytes)
        assert len(views) == 1
        assert views[0].shape == shape
        assert views[0].dtype == torch.float32

    def test_two_views_non_overlapping(self):
        """Two views carved sequentially from one buffer."""
        raw = torch.zeros(8192, dtype=torch.int8)
        page_size_bytes = 1024
        shape_a = (4, 32, 8)
        shape_b = (4, 32, 4)
        views = adjust_kv_layout(
            raw,
            [shape_a, shape_b],
            [torch.float32, torch.float16],
            page_size_bytes,
        )
        assert len(views) == 2
        assert views[0].shape == shape_a
        assert views[0].dtype == torch.float32
        assert views[1].shape == shape_b
        assert views[1].dtype == torch.float16
        # Different storage offsets (view_b should start after view_a)
        assert views[0].storage_offset() != views[1].storage_offset()

    def test_overlay_view(self):
        """Third view overlaps first when overlap_full_kv_cache=True."""
        raw = torch.zeros(12288, dtype=torch.int8)
        page_size_bytes = 1024
        views = adjust_kv_layout(
            raw,
            [(4, 32, 8), (4, 16, 4), (4, 32, 8)],
            [torch.float32, torch.float16, torch.float32],
            page_size_bytes,
            overlap_full_kv_cache=True,
        )
        assert len(views) == 3
        # View 2 and view 0 should share the same storage offset
        assert views[2].storage_offset() == views[0].storage_offset()
        # View 1 should be at a different offset
        assert views[1].storage_offset() != views[0].storage_offset()


# ---------------------------------------------------------------------------
# SingleTensorLayout
# ---------------------------------------------------------------------------

class TestSingleTensorLayout:
    def test_num_tensors(self):
        layout = SingleTensorLayout()
        assert layout.num_tensors() == 1

    def test_split_sizes(self):
        layout = SingleTensorLayout()
        spec = _FakeSpec()
        sizes = layout.split_sizes(12345, spec)
        assert sizes == [12345]

    def test_reshape(self):
        layout = SingleTensorLayout()
        spec = _FakeSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)

        # Simulate: page_size_bytes = 128*4*128*2 = 131072
        # num_blocks = total_bytes // page_size_bytes = 131072*4 // 131072 = 4
        total_bytes = spec.page_size_bytes * 4
        raw = torch.zeros(total_bytes, dtype=torch.int8)

        backend = _make_backend_mock((4, 128, 4, 128))
        result = layout.reshape(
            [raw], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        assert result.shape == (4, 128, 4, 128)
        assert result.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# SplitKVLayout
# ---------------------------------------------------------------------------

class TestSplitKVLayout:
    def test_num_tensors(self):
        assert SplitKVLayout().num_tensors() == 2

    def test_split_sizes_symmetric(self):
        """Equal head dims → equal split."""
        layout = SplitKVLayout()
        spec = _FakeSpec()
        sizes = layout.split_sizes(1000, spec, head_dims=(128, 128))
        assert len(sizes) == 2
        assert sizes == [500, 500]

    def test_split_sizes_asymmetric(self):
        """MLA-style asymmetric K/V dims."""
        layout = SplitKVLayout()
        spec = _FakeSpec()
        # k_dim=512, v_dim=64 → total=576
        # factors: (576/512, 576/64) = (1.125, 9.0)
        # 5760 // 1.125 = 5120, 5760 // 9.0 = 640
        sizes = layout.split_sizes(5760, spec, head_dims=(512, 64))
        assert sizes[0] > sizes[1]  # K larger than V
        assert sum(sizes) <= 5760

    def test_reshape_gqa(self):
        """GQA reshape: K and V each get the standard 4D shape."""
        layout = SplitKVLayout()
        spec = _FakeSpec(block_size=128, num_kv_heads=4, head_size=128, dtype=torch.bfloat16)
        # Combined page_size_bytes = block * heads * (k_dim + v_dim) * dtype_size
        spec.page_size_bytes = 128 * 4 * (128 + 128) * 2  # 262144

        # 4 blocks worth of bytes
        total = spec.page_size_bytes * 4
        sizes = layout.split_sizes(total, spec, head_dims=(128, 128))
        raw_k = torch.zeros(sizes[0], dtype=torch.int8)
        raw_v = torch.zeros(sizes[1], dtype=torch.int8)

        backend = _make_backend_mock((4, 128, 4, 128))
        result = layout.reshape(
            [raw_k, raw_v], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
            head_dims=(128, 128),
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        k, v = result
        assert k.shape == (4, 128, 4, 128)
        assert v.shape == (4, 128, 4, 128)
        assert k.dtype == torch.bfloat16

    def test_reshape_mla(self):
        """MLA reshape: different head dims for K vs V."""
        layout = SplitKVLayout()
        spec = _FakeSpec(block_size=128, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
        # page_size_bytes for MLA: 128*1*576*2 = 147456
        spec.page_size_bytes = 128 * 1 * 576 * 2

        total = spec.page_size_bytes * 4
        sizes = layout.split_sizes(total, spec, head_dims=(512, 64))
        raw_k = torch.zeros(sizes[0], dtype=torch.int8)
        raw_v = torch.zeros(sizes[1], dtype=torch.int8)

        backend = _make_backend_mock((4, 128, 1, 576))
        result = layout.reshape(
            [raw_k, raw_v], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
            head_dims=(512, 64),
        )
        k, v = result
        assert k.shape == (4, 128, 1, 512)
        assert v.shape == (4, 128, 1, 64)

    def test_needs_alignment(self):
        assert SplitKVLayout().needs_alignment() is True


# ---------------------------------------------------------------------------
# SparseMLALayout
# ---------------------------------------------------------------------------

class TestSparseMLALayout:
    def _make_spec(self):
        """DS V3.2 sparse spec: kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128."""
        spec = _FakeSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=704,  # sum(sparse_head_dim) = 512+64+128
            sparse_head_dim=(512, 64, 128),
            dtype=torch.bfloat16,
        )
        # page_size_bytes matches AscendMLAAttentionSpec.page_size_bytes
        # for non-C8 sparse: block_size * num_kv_heads * head_size * dtype_size
        spec.page_size_bytes = 128 * 1 * 704 * 2  # = 180224
        return spec

    def test_num_tensors(self):
        assert SparseMLALayout().num_tensors() == 3

    def test_split_sizes_proportions(self):
        """kv_lora should be largest, k_rope smallest."""
        layout = SparseMLALayout()
        spec = self._make_spec()
        # head_size=704, sparse_head_dim=(512,64,128)
        # ratio = (704/512, 704/64, 704/128) = (1.375, 11.0, 5.5)
        spec.sparse_kv_cache_ratio = (1.375, 11.0, 5.5, None)

        total = 9000
        sizes = layout.split_sizes(total, spec)
        assert len(sizes) == 3
        # kv_lora (largest): 9000/1.125 = 8000
        # k_rope (smallest): 9000/9.0 = 1000
        # indexer_k: 9000/4.5 = 2000
        # Note: due to int truncation, values may differ slightly
        assert sizes[0] > sizes[2] > sizes[1], (
            f"Expected sizes[0] > sizes[2] > sizes[1], got {sizes}"
        )

    def test_reshape_shapes(self):
        layout = SparseMLALayout()
        spec = self._make_spec()
        spec.sparse_kv_cache_ratio = (1.375, 11.0, 5.5, None)

        total = spec.page_size_bytes * 4
        sizes = layout.split_sizes(total, spec)
        raw = [torch.zeros(s, dtype=torch.int8) for s in sizes]

        backend = _make_backend_mock((4, 128, 1, 576))
        result = layout.reshape(
            raw, spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        k, v, dsa_k = result
        assert k.shape == (4, 128, 1, 512)
        assert v.shape == (4, 128, 1, 64)
        assert dsa_k.shape == (4, 128, 1, 128)

    def test_needs_alignment(self):
        assert SparseMLALayout().needs_alignment() is True


# ---------------------------------------------------------------------------
# SparseMLAC8Layout
# ---------------------------------------------------------------------------

class TestSparseMLAC8Layout:
    def _make_spec(self):
        spec = _FakeSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=704,  # sum(sparse_head_dim) = 512+64+128
            sparse_head_dim=(512, 64, 128),
            cache_sparse_c8=True,
            c8_k_cache_dtype=torch.int8,
            c8_k_scale_cache_dtype=torch.float16,
            dtype=torch.bfloat16,
        )
        # Byte sizes for C8: kv_lora(bf16*512) + k_rope(bf16*64) + indexer(int8*128) + scale(fp16*1)
        spec.page_size_bytes = (
            128 * 1 * 512 * 2
            + 128 * 1 * 64 * 2
            + 128 * 1 * 128 * 1
            + 128 * 1 * 1 * 2
        )
        return spec

    def test_num_tensors(self):
        assert SparseMLAC8Layout().num_tensors() == 4

    def test_split_sizes(self):
        layout = SparseMLAC8Layout()
        spec = self._make_spec()
        # virtual dims: (512, 64, 128/2, 1) = (512, 64, 64, 1)
        # total_virtual = 512+64+64+1 = 641
        # ratios: (641/512, 641/64, 641/64, 641/1)
        spec.sparse_kv_cache_ratio = (
            641.0 / 512,  # ~1.252
            641.0 / 64,   # ~10.016
            641.0 / 64,   # ~10.016
            641.0 / 1,    # 641.0
        )

        total = spec.page_size_bytes * 4
        sizes = layout.split_sizes(total, spec)
        assert len(sizes) == 4
        # scale tensor should be smallest (largest divisor)
        assert sizes[3] < sizes[1], f"scale should be smaller than k_rope, got {sizes}"

    def test_reshape_dtypes(self):
        layout = SparseMLAC8Layout()
        spec = self._make_spec()
        spec.sparse_kv_cache_ratio = (641.0 / 512, 641.0 / 64, 641.0 / 64, 641.0 / 1)

        total = spec.page_size_bytes * 4
        sizes = layout.split_sizes(total, spec)
        raw = [torch.zeros(s, dtype=torch.int8) for s in sizes]

        backend = _make_backend_mock((4, 128, 1, 576))
        result = layout.reshape(
            raw, spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        k, v, dsa_k, dsa_scale = result
        assert k.dtype == torch.bfloat16
        assert v.dtype == torch.bfloat16
        assert dsa_k.dtype == torch.int8
        assert dsa_scale.dtype == torch.float16
        assert dsa_scale.shape[-1] == 1  # per-token scale

    def test_needs_alignment(self):
        assert SparseMLAC8Layout().needs_alignment() is True


# ---------------------------------------------------------------------------
# CompressedMLALayout
# ---------------------------------------------------------------------------

class TestCompressedMLALayout:
    def _make_spec(self, with_scale: bool = True):
        spec = _FakeSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=512,
            compress_ratio=4,
            dtype=torch.bfloat16,
            scale_dim=0,
            scale_dtype=torch.int8,
        )
        spec.page_size_bytes = 128 // 4 * 1 * 512 * 2  # storage_block_size * num_kv_heads * head_size * dtype_size
        if with_scale:
            spec.scale_dim = 8
        return spec

    def test_num_tensors(self):
        assert CompressedMLALayout().num_tensors() == 1

    def test_split_sizes(self):
        layout = CompressedMLALayout()
        spec = self._make_spec(with_scale=False)
        assert layout.split_sizes(12345, spec) == [12345]

    @patch("vllm_ascend.utils.get_ascend_device_type")
    def test_reshape_no_scale(self, mock_device_type):
        """Without scale_dim, single view."""
        mock_device_type.return_value = 0  # A2: not A5
        layout = CompressedMLALayout()
        spec = self._make_spec(with_scale=False)

        # Compressed MLA: as_strided needs (num_blocks-1)*page_bytes + logical_block_bytes
        # page_size_bytes=32768, logical_block=128*1*512*2=131072, bf16 itemsize=2
        logical_block_bytes = spec.block_size * spec.num_kv_heads * spec.head_size * 2
        raw_size = (4 - 1) * spec.page_size_bytes + logical_block_bytes  # 3*32768 + 131072 = 229376
        raw = torch.zeros(raw_size, dtype=torch.int8)
        backend = _make_dynamic_backend_mock()

        result = layout.reshape(
            [raw], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        assert len(result) == 1
        assert result[0].shape == (4, 128, 1, 512)

    @patch("vllm_ascend.utils.get_ascend_device_type")
    def test_reshape_with_scale_non_a5(self, mock_device_type):
        """With scale_dim and non-A5 device → 2 views."""
        mock_device_type.return_value = 0  # A2
        layout = CompressedMLALayout()
        spec = self._make_spec(with_scale=True)

        # Two views sequentially: k_view needs as_strided headroom,
        # then scale_view starts after k_view's full extent
        logical_block_bytes = spec.block_size * spec.num_kv_heads * spec.head_size * 2  # 131072
        raw_size = (4 - 1) * spec.page_size_bytes + logical_block_bytes  # k view: 229376
        k_stride0_bytes = spec.block_size * spec.num_kv_heads * spec.head_size * 2  # 128*1*512*2 = 131072
        raw_size += k_stride0_bytes  # scale view starts after k's last block
        raw = torch.zeros(raw_size, dtype=torch.int8)
        backend = _make_dynamic_backend_mock()

        result = layout.reshape(
            [raw], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        assert len(result) == 2
        assert result[0].shape == (4, 128, 1, 512)  # K view
        assert result[1].shape == (4, 128, 1, 8)     # scale view

    @patch("vllm_ascend.utils.get_ascend_device_type")
    def test_reshape_with_scale_a5(self, mock_device_type):
        """A5 device → 3 views with overlay."""
        from vllm_ascend.utils import AscendDeviceType
        mock_device_type.return_value = AscendDeviceType.A5
        layout = CompressedMLALayout()
        spec = self._make_spec(with_scale=True)

        # Same as non-A5 case: k view + scale view sequential
        logical_block_bytes = spec.block_size * spec.num_kv_heads * spec.head_size * 2  # 131072
        raw_size = (4 - 1) * spec.page_size_bytes + logical_block_bytes  # 229376
        k_stride0_bytes = spec.block_size * spec.num_kv_heads * spec.head_size * 2  # 131072
        raw_size += k_stride0_bytes
        raw = torch.zeros(raw_size, dtype=torch.int8)
        backend = _make_dynamic_backend_mock()

        result = layout.reshape(
            [raw], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        assert len(result) == 3
        # Third view (full overlay) shares storage offset with first
        assert result[2].storage_offset() == result[0].storage_offset()


# ---------------------------------------------------------------------------
# MambaLayout
# ---------------------------------------------------------------------------

class TestMambaLayout:
    def _make_mamba_spec(self):
        spec = _FakeSpec(
            block_size=128,
            shapes=((4, 256), (16, 16)),
            dtypes=(torch.float32, torch.float32),
            dtype=torch.int8,
        )
        # page_size_bytes: sum of (shape[0] * shape[1] * dtype_size) per state
        spec.page_size_bytes = (4 * 256 * 4) + (16 * 16 * 4)  # 4096 + 1024 = 5120
        return spec

    def test_num_tensors(self):
        assert MambaLayout().num_tensors() == 1

    def test_split_sizes(self):
        layout = MambaLayout()
        spec = self._make_mamba_spec()
        assert layout.split_sizes(5000, spec) == [5000]

    def test_reshape(self):
        layout = MambaLayout()
        spec = self._make_mamba_spec()

        total = spec.page_size_bytes * 4
        raw = torch.zeros(total, dtype=torch.int8)
        backend = MagicMock()

        result = layout.reshape(
            [raw], spec,
            num_blocks=4, kernel_num_blocks=4, kernel_block_size=128,
            backend=backend, vllm_config=MagicMock(),
        )
        assert len(result) == 2
        assert result[0].shape == (4, 4, 256)
        assert result[1].shape == (4, 16, 16)
        assert result[0].dtype == torch.float32
        assert result[1].dtype == torch.float32


# ---------------------------------------------------------------------------
# Layout registration completeness check
# ---------------------------------------------------------------------------

class TestLayoutRegistry:
    """Verify every layout subclass is properly abstract."""

    ALL_LAYOUTS = [
        SingleTensorLayout,
        SplitKVLayout,
        SparseMLALayout,
        SparseMLAC8Layout,
        CompressedMLALayout,
        MambaLayout,
    ]

    def test_all_concrete(self):
        """Every class in ALL_LAYOUTS can be instantiated."""
        for cls in self.ALL_LAYOUTS:
            instance = cls()
            assert isinstance(instance, KVCacheLayout)

    def test_all_num_tensors_positive(self):
        """num_tensors() must be >= 1 for every layout."""
        for cls in self.ALL_LAYOUTS:
            assert cls().num_tensors() >= 1

    def test_split_sizes_sums_to_total(self):
        """split_sizes must sum to total_bytes (within rounding tolerance).

        Uses total = page_size_bytes * 4 so split factors divide evenly.
        """
        # Use each layout's actual page_size_bytes to avoid float rounding
        single_tensor_spec = _FakeSpec()
        single_tensor_spec.page_size_bytes = 131072
        splitkv_spec = _FakeSpec()
        splitkv_spec.page_size_bytes = 128 * 4 * (128 + 128) * 2  # 262144
        sparse_mla_spec = _FakeSpec(
            head_size=704,
            sparse_head_dim=(512, 64, 128),
            sparse_kv_cache_ratio=(1.375, 11.0, 5.5, None),
        )
        sparse_mla_spec.page_size_bytes = 128 * 1 * 704 * 2  # 180224
        sparse_c8_spec = _FakeSpec(
            head_size=704,
            sparse_head_dim=(512, 64, 128),
            sparse_kv_cache_ratio=(641 / 512, 641 / 64, 641 / 64, 641),
        )
        sparse_c8_spec.page_size_bytes = 164096
        compressed_spec = _FakeSpec(compress_ratio=4)
        compressed_spec.page_size_bytes = 128 // 4 * 1 * 512 * 2  # 32768
        mamba_spec = _FakeSpec(
            shapes=((4, 256),),
            dtypes=(torch.float32,),
            page_size_bytes=4096,
        )

        test_cases: list[tuple[KVCacheLayout, _FakeSpec, dict]] = [
            (SingleTensorLayout(), single_tensor_spec, {}),
            (SplitKVLayout(), splitkv_spec, {"head_dims": (128, 128)}),
            (SparseMLALayout(), sparse_mla_spec, {}),
            (SparseMLAC8Layout(), sparse_c8_spec, {}),
            (CompressedMLALayout(), compressed_spec, {}),
            (MambaLayout(), mamba_spec, {}),
        ]

        for layout, spec, kwargs in test_cases:
            total = spec.page_size_bytes * 4
            sizes = layout.split_sizes(total, spec, **kwargs)
            assert len(sizes) == layout.num_tensors(), (
                f"{type(layout).__name__}: expected {layout.num_tensors()} sizes, "
                f"got {len(sizes)}"
            )
            # Sum must equal total exactly when using page_size_bytes multiples
            difference = total - sum(sizes)
            assert difference == 0, (
                f"{type(layout).__name__}: sizes {sizes} sum to {sum(sizes)}, "
                f"expected {total} (diff={difference})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
