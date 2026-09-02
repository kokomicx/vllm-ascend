from dataclasses import replace
from typing import Any

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)


def uses_upstream_kv_cache_layout() -> bool:
    """Whether vLLM uses the post-layout-refactor KVCacheTensor schema."""
    return "layers" in getattr(KVCacheTensor, "__dataclass_fields__", {})


def customize_standard_gqa_spec(spec: AttentionSpec) -> AttentionSpec:
    """Pack K and V as two contiguous head groups for Ascend kernels."""
    if (
        not uses_upstream_kv_cache_layout()
        or getattr(spec, "state_content_bytes", None) is not None
    ):
        return spec
    if spec.kv_quant_mode != KVQuantMode.NONE:
        raise NotImplementedError(
            "The upstream-layout Ascend GQA path does not support "
            f"quantized KV cache mode {spec.kv_quant_mode.name} yet."
        )
    assert spec.head_size == spec.head_size_v, (
        "Ascend's separate K/V cache layout requires symmetric head sizes."
    )
    return replace(
        spec,
        num_head_slots=2,
        state_content_bytes=(
            spec.num_kv_heads * spec.head_size * get_dtype_size(spec.dtype)
        ),
    )


def is_standard_gqa_kv_cache_config(kv_cache_config: KVCacheConfig) -> bool:
    """Return whether the config is in the first upstream-allocation scope.

    The initial migration intentionally accepts FullAttentionSpec-only cache
    groups. Hybrid, recurrent, MLA/SFA and other custom specs stay on their
    existing Ascend allocation paths until they are migrated independently.
    """
    if not uses_upstream_kv_cache_layout() or not kv_cache_config.kv_cache_groups:
        return False

    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        specs = (
            group_spec.kv_cache_specs.values()
            if isinstance(group_spec, UniformTypeKVCacheSpecs)
            else (group_spec,)
        )
        if any(type(spec).__name__ != "FullAttentionSpec" for spec in specs):
            return False
    return True


def normalize_kernel_block_sizes(
    kernel_block_sizes: list[int] | list[list[int]],
) -> list[int] | None:
    """Convert Ascend MRV1's nested block sizes for the public allocator."""
    normalized: list[int] = []
    for size in kernel_block_sizes:
        if isinstance(size, (list, tuple)):
            if not size:
                return None
            normalized.append(int(size[0]))
        else:
            normalized.append(int(size))
    return normalized or None


def split_standard_gqa_kv_cache(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose an upstream standard GQA cache as Ascend K/V views.

    Ascend customizes the standard spec into two head slots and chooses the
    LHBNC layout. The public allocator therefore returns a logical
    [B, 2, N, H * D] tensor whose K and V regions are each contiguous.
    """
    if kv_cache.ndim != 4:
        raise ValueError(
            "Ascend GQA expects a rank-4 KV cache with logical shape "
            f"[B, 2, N, H * D], but got shape {tuple(kv_cache.shape)}."
        )

    if head_size != head_size_v:
        raise ValueError(
            "Ascend's separate contiguous K/V layout currently requires "
            f"symmetric head sizes, but got K={head_size}, V={head_size_v}."
        )
    if kv_cache.shape[1] != 2:
        raise ValueError(
            "Ascend GQA expects exactly two K/V head slots, but got "
            f"{kv_cache.shape[1]}."
        )

    expected_cache_width = num_kv_heads * head_size
    if kv_cache.shape[-1] != expected_cache_width:
        raise ValueError(
            "Ascend GQA KV cache width must equal num_kv_heads * head_size "
            f"({num_kv_heads} * {head_size} = {expected_cache_width}), but got "
            f"{kv_cache.shape[-1]}."
        )

    num_blocks, _, block_size, _ = kv_cache.shape
    target_shape = (num_blocks, block_size, num_kv_heads, head_size)
    return kv_cache[:, 0].view(target_shape), kv_cache[:, 1].view(target_shape)


def bind_standard_gqa_kv_cache(layer: Any, kv_cache: torch.Tensor) -> None:
    """Bind an upstream standard cache to a vLLM Attention layer on Ascend."""
    layer.kv_cache = split_standard_gqa_kv_cache(
        kv_cache,
        num_kv_heads=layer.num_kv_heads,
        head_size=layer.head_size,
        head_size_v=layer.head_size_v,
    )
