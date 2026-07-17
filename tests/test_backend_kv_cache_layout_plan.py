# SPDX-License-Identifier: Apache-2.0
"""Tests for backend-owned KV cache layout planning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec

from vllm_ascend.attention.kv_cache_layout import AscendKVCacheLayoutBackendMixin
from vllm_ascend.core.kv_cache_layout import (
    CompressedMLALayout,
    SingleTensorLayout,
    SplitKVLayout,
)


class _TestBackend(AscendKVCacheLayoutBackendMixin):
    """Minimal backend used to exercise plan selection without an NPU."""

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)


def _vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto"),
        quant_config=None,
    )


def test_backend_plan_owns_full_attention_layout() -> None:
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )

    plan = _TestBackend.get_kv_cache_layout_plan(
        spec,
        layer_name="model.layers.0.self_attn",
        vllm_config=_vllm_config(),
    )

    assert isinstance(plan.layout, SplitKVLayout)
    assert plan.head_dims == (128, 128)
    assert plan.num_tensors == 2
    assert sum(plan.split_sizes(spec.page_size_bytes * 4)) == spec.page_size_bytes * 4


def test_backend_plan_preserves_compressed_mla_layout() -> None:
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )

    # Compressed MLA does not need layer introspection for this decision.
    with patch.object(_TestBackend, "_is_cache_only_layer", return_value=False):
        plan = _TestBackend.get_kv_cache_layout_plan(
            spec,
            layer_name="model.layers.0.self_attn",
            vllm_config=_vllm_config(),
        )

    assert isinstance(plan.layout, CompressedMLALayout)
    assert plan.num_tensors == 1


def test_backend_plan_owns_hybrid_single_tensor_policy() -> None:
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )

    plan = _TestBackend.get_kv_cache_layout_plan(
        spec,
        layer_name="model.layers.0.self_attn",
        vllm_config=_vllm_config(),
        is_hybrid_model=True,
    )

    assert isinstance(plan.layout, SingleTensorLayout)
    assert plan.num_tensors == 1
