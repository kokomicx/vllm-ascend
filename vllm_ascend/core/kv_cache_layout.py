# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
KVCacheLayout: Physical memory layout strategy for KV cache tensors.

This module implements the second layer of the three-layer architecture:

    Spec (declares "what")
      → Layout (decides "how to store" — THIS LAYER)
        → Backend (knows "how to compute")

Each KVCacheLayout subclass encapsulates the complete physical memory
strategy for a model family, including:

- How many separate tensors to allocate
- How to partition total_bytes across those tensors
- How to reshape each raw tensor for the backend

Adding a new model variant requires only a new Layout subclass, with
zero changes to the allocate/reshape dispatch code in the model runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from vllm.utils.torch_utils import get_dtype_size
from vllm_ascend.utils import calc_split_factor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpec, MambaSpec


# ---------------------------------------------------------------------------
# Utility: carve memory views from a flat buffer via torch.as_strided
# ---------------------------------------------------------------------------

def adjust_kv_layout(
    raw_tensor: torch.Tensor,
    kv_cache_shape_list: list[tuple[int, ...]],
    kv_cache_dtype_list: list[torch.dtype],
    page_size_bytes: int,
    overlap_full_kv_cache: bool = False,
) -> list[torch.Tensor]:
    """Carve one or more reshaped views from a single flat buffer.

    Uses ``torch.as_strided`` so the resulting tensors share the same
    underlying storage — no data is copied.

    Args:
        raw_tensor: Flat int8 buffer allocated by the manager.
        kv_cache_shape_list: Target logical shape for each view.
        kv_cache_dtype_list: Target dtype for each view.
        page_size_bytes: Bytes per logical page (from the spec).
        overlap_full_kv_cache: When True, the third view (idx==2) starts
            at the same offset as the first view, producing an overlay.

    Returns:
        List of tensor views, one per entry in *kv_cache_shape_list*.
    """
    reshaped: list[torch.Tensor] = []
    base_offset_bytes = raw_tensor.storage_offset()
    storage_offset_bytes = base_offset_bytes

    for idx, (shape, dtype) in enumerate(zip(kv_cache_shape_list, kv_cache_dtype_list)):
        # Overlay view: share the start offset of the first view
        if overlap_full_kv_cache and idx == 2:
            storage_offset_bytes = base_offset_bytes

        dtype_size = get_dtype_size(dtype)
        num_elements_per_page = page_size_bytes // dtype_size
        stride_ref = torch.empty(shape).stride()
        target_stride = (num_elements_per_page, *stride_ref[1:])
        assert storage_offset_bytes % dtype_size == 0, (
            f"Storage offset {storage_offset_bytes} is not aligned to "
            f"dtype {dtype} size {dtype_size}"
        )
        view = torch.as_strided(
            raw_tensor.view(dtype),
            size=shape,
            stride=target_stride,
            storage_offset=storage_offset_bytes // dtype_size,
        )
        reshaped.append(view)
        storage_offset_bytes += stride_ref[0] * dtype_size

    return reshaped


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class KVCacheLayout(ABC):
    """Abstract physical memory layout for KV cache tensors.

    Each concrete subclass represents one strategy for laying out KV cache
    tensors in physical memory — how many separate buffers, how to split the
    total byte budget among them, and how to reshape each buffer into the
    shape the attention backend expects.

    Subclasses are **stateless singletons** — instantiate once and reuse.
    """

    @abstractmethod
    def num_tensors(self) -> int:
        """Number of physical tensors this layout produces per layer."""
        ...

    @abstractmethod
    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        """Partition *total_bytes* across the physical tensors.

        Args:
            total_bytes: Total byte budget for one layer's KV cache.
            spec: The KVCacheSpec describing the attention format.
            **kwargs: Additional context (e.g. ``head_dims``, ``vllm_config``,
                ``layer_name``, ``quant_config``).

        Returns:
            A list of *num_tensors()* byte sizes, whose sum should equal
            *total_bytes* (within rounding tolerance).
        """
        ...

    @abstractmethod
    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: AttentionSpec,
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> Any:
        """Reshape raw flat tensors into backend-compatible form.

        Args:
            raw_tensors: Flat int8 buffers (one per ``num_tensors()``).
            spec: The KVCacheSpec.
            num_blocks: Number of physical KV blocks (from total bytes).
            kernel_num_blocks: num_blocks adjusted for kernel block size.
            kernel_block_size: Block size the backend kernel expects.
            backend: Attention backend (provides ``get_kv_cache_shape``).
            vllm_config: Full vLLM configuration.
            **kwargs: Additional context (e.g. ``head_dims``, ``layer_name``).

        Returns:
            The reshaped KV cache in the form the attention backend expects.
            This is typically a single tensor, a tuple of tensors, or a list
            of tensor views.
        """
        ...

    def needs_alignment(self) -> bool:
        """Whether PD (prefill disaggregation) alignment is required."""
        return False


# ---------------------------------------------------------------------------
# Layout 1: SingleTensorLayout
# ---------------------------------------------------------------------------

class SingleTensorLayout(KVCacheLayout):
    """A single flat tensor, reshaped with a simple ``.view(dtype).view(shape)``.

    Used for:
    - ``cache_only_layers`` (extract_hidden_states in draft models)
    - Mamba / linear attention layers (via MambaLayout instead)
    - Hybrid attention-Mamba layers that share a single buffer

    This is the simplest layout — one allocation, one reshape.
    """

    def num_tensors(self) -> int:
        return 1

    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        return [total_bytes]

    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: AttentionSpec,
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> torch.Tensor:
        raw = raw_tensors[0]
        cache_dtype_str: str = kwargs.get("cache_dtype_str", "")
        kv_cache_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype_str,
        )
        return raw.view(spec.dtype).view(kv_cache_shape)


# ---------------------------------------------------------------------------
# Layout 2: SplitKVLayout
# ---------------------------------------------------------------------------

class SplitKVLayout(KVCacheLayout):
    """Two physically separate tensors — one for K, one for V.

    This is the standard Ascend layout for all GQA / SWA / standard MLA
    models. K and V must be separate physical allocations for PD RDMA
    alignment (each can be independently pinned to a 2 MB boundary).

    Used for:
    - Qwen2.5, Llama-3.1 (GQA / SWA)
    - Standard MLA models without sparse indexer

    *head_dims* must be passed via ``**kwargs`` as a ``(k_dim, v_dim)`` tuple.
    """

    def num_tensors(self) -> int:
        return 2

    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        head_dims: tuple[int, int] = kwargs["head_dims"]
        k_dim, v_dim = head_dims

        # FA-quant models may supply custom split factors
        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        layer_name: str = kwargs.get("layer_name", "")
        quant_config = kwargs.get("quant_config")

        if quant_config is not None and vllm_config is not None:
            k_factor, v_factor = quant_config.get_kv_quant_split_factor(
                layer_name, [k_dim, v_dim]
            )
        else:
            k_factor, v_factor = calc_split_factor([k_dim, v_dim])

        return [int(total_bytes // k_factor), int(total_bytes // v_factor)]

    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: AttentionSpec,
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dims: tuple[int, int] = kwargs["head_dims"]
        k_dim, v_dim = head_dims
        layer_name: str = kwargs.get("layer_name", "")
        cache_dtype_str: str = kwargs.get("cache_dtype_str", "")

        kv_cache_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
            cache_dtype_str=cache_dtype_str,
        )
        # FA3/Attention backends return (2, N, BS, H, D) with a leading 2×
        # K/V factor.  SplitKVLayout handles separate K/V tensors, so drop
        # the leading dimension (same logic as old V1 reshape).
        if kv_cache_shape[0] == 2:
            base_shape = kv_cache_shape[1:]
        else:
            base_shape = kv_cache_shape
        k_shape = base_shape[:-1] + (k_dim,)
        v_shape = base_shape[:-1] + (v_dim,)

        # Determine per-tensor dtype (FA-quant may override)
        k_dtype = v_dtype = spec.dtype
        quant_config = kwargs.get("quant_config")
        if quant_config is not None and vllm_config is not None:
            k_dtype, v_dtype = quant_config.get_kv_quant_dtype(
                layer_name, spec.dtype, vllm_config.model_config
            )

        k_cache = raw_tensors[0].view(k_dtype).view(k_shape)
        v_cache = raw_tensors[1].view(v_dtype).view(v_shape)
        return (k_cache, v_cache)

    def needs_alignment(self) -> bool:
        # PD always needs 2 MB alignment for K/V tensors
        return True


# ---------------------------------------------------------------------------
# Layout 3: SparseMLALayout
# ---------------------------------------------------------------------------

class SparseMLALayout(KVCacheLayout):
    """Three physically separate tensors for sparse MLA (no C8 quantization).

    Cache tuple semantics::

        cache[0]  →  kv_lora   (bf16)   K nope
        cache[1]  →  k_rope    (bf16)   K rope
        cache[2]  →  indexer_k  (bf16)   DSA indexer key

    Used for:
    - DeepSeek V3.2 sparse attention (without C8 quantization)

    Split ratios come from ``spec.sparse_kv_cache_ratio``, which is a
    property on ``AscendMLAAttentionSpec``.
    """

    def num_tensors(self) -> int:
        return 3

    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        r0, r1, r2, _ = spec.sparse_kv_cache_ratio  # type: ignore[attr-defined]
        return [
            int(total_bytes // r0),
            int(total_bytes // r1),
            int(total_bytes // r2),
        ]

    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: AttentionSpec,
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_dim, v_dim, indexer_dim = spec.sparse_head_dim  # type: ignore[attr-defined]

        kv_cache_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
        )
        # Drop leading 2× K/V factor (same as SplitKVLayout)
        if kv_cache_shape[0] == 2:
            base_shape = kv_cache_shape[1:]
        else:
            base_shape = kv_cache_shape
        k_shape = base_shape[:-1] + (k_dim,)
        v_shape = base_shape[:-1] + (v_dim,)
        dsa_k_shape = (
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            indexer_dim,
        )

        k_cache = raw_tensors[0].view(spec.dtype).view(k_shape)
        v_cache = raw_tensors[1].view(spec.dtype).view(v_shape)
        dsa_k_cache = raw_tensors[2].view(spec.dtype).view(dsa_k_shape)
        return (k_cache, v_cache, dsa_k_cache)

    def needs_alignment(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Layout 4: SparseMLAC8Layout
# ---------------------------------------------------------------------------

class SparseMLAC8Layout(KVCacheLayout):
    """Four physically separate tensors for sparse MLA **with** C8 quantization.

    Cache tuple semantics::

        cache[0]  →  kv_lora        (bf16)   K nope
        cache[1]  →  k_rope         (bf16)   K rope
        cache[2]  →  indexer_k       (int8)   DSA indexer key, quantized
        cache[3]  →  indexer_scale   (fp16)   Per-token quantization scale

    Used for:
    - DeepSeek V3.2 sparse attention **with** C8 quantization

    The split ratios account for dtype differences via "virtual head dims"
    computed inside ``AscendMLAAttentionSpec.sparse_kv_cache_ratio``.
    """

    def num_tensors(self) -> int:
        return 4

    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        r0, r1, r2, r3 = spec.sparse_kv_cache_ratio  # type: ignore[attr-defined]
        return [
            int(total_bytes // r0),
            int(total_bytes // r1),
            int(total_bytes // r2),
            int(total_bytes // r3),
        ]

    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: AttentionSpec,
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k_dim, v_dim, indexer_dim = spec.sparse_head_dim  # type: ignore[attr-defined]
        c8_k_dtype = spec.c8_k_cache_dtype  # type: ignore[attr-defined]
        c8_scale_dtype = spec.c8_k_scale_cache_dtype  # type: ignore[attr-defined]

        kv_cache_shape = backend.get_kv_cache_shape(
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            spec.head_size,
        )
        # Drop leading 2× K/V factor (same as SplitKVLayout)
        if kv_cache_shape[0] == 2:
            base_shape = kv_cache_shape[1:]
        else:
            base_shape = kv_cache_shape
        k_shape = base_shape[:-1] + (k_dim,)
        v_shape = base_shape[:-1] + (v_dim,)
        dsa_k_shape = (
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            indexer_dim,
        )
        dsa_k_scale_shape = (
            kernel_num_blocks,
            kernel_block_size,
            spec.num_kv_heads,
            1,
        )

        k_cache = raw_tensors[0].view(spec.dtype).view(k_shape)
        v_cache = raw_tensors[1].view(spec.dtype).view(v_shape)
        dsa_k_cache = raw_tensors[2].view(c8_k_dtype).view(dsa_k_shape)
        dsa_k_scale_cache = raw_tensors[3].view(c8_scale_dtype).view(dsa_k_scale_shape)
        return (k_cache, v_cache, dsa_k_cache, dsa_k_scale_cache)

    def needs_alignment(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Layout 5: CompressedMLALayout
# ---------------------------------------------------------------------------

class CompressedMLALayout(KVCacheLayout):
    """Single physical buffer with multiple ``as_strided`` overlay views.

    DeepSeek V4's compressed MLA cache uses one flat int8 allocation, then
    carves multiple logical views over it. Views are:

    - **View 1** (K cache): main KV cache in model dtype
    - **View 2** (scale): per-token quantization scale (when ``scale_dim != 0``)
    - **View 3** (full overlay): A5-only, combines K+scale into one view

    This is the most complex layout because a single raw tensor produces a
    *list* of views (not a tuple of independent tensors).

    Used for:
    - DeepSeek V4 ``compress_ratio > 1`` (fp8 compressed MLA)
    """

    def num_tensors(self) -> int:
        return 1

    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        return [total_bytes]

    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: AttentionSpec,
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        raw = raw_tensors[0]
        kv_cache_shape = backend.get_kv_cache_shape(
            num_blocks,
            spec.block_size,
            spec.num_kv_heads,
            spec.head_size,
        )

        kv_cache_shape_list: list[tuple[int, ...]] = [kv_cache_shape]
        kv_cache_dtype_list: list[torch.dtype] = [spec.dtype]
        overlap_full_kv_cache = False

        # Optional scale view (DS V4 fp8)
        if hasattr(spec, "scale_dim") and spec.scale_dim != 0:  # type: ignore[attr-defined]
            scale_dim: int = spec.scale_dim  # type: ignore[attr-defined]
            scale_dtype: torch.dtype = spec.scale_dtype  # type: ignore[attr-defined]

            indexer_k_shape = kv_cache_shape
            indexer_scale_shape = backend.get_kv_cache_shape(
                num_blocks,
                spec.block_size,
                spec.num_kv_heads,
                scale_dim,
            )
            # Determine whether to add A5 full-overlay view
            from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

            if get_ascend_device_type() in {AscendDeviceType.A5}:
                # A5 epilog kernel needs a combined K+scale view
                indexer_full_shape = backend.get_kv_cache_shape(
                    num_blocks,
                    spec.block_size,
                    spec.num_kv_heads,
                    spec.head_size + scale_dim * get_dtype_size(scale_dtype),
                )
                kv_cache_shape_list = [
                    indexer_k_shape,
                    indexer_scale_shape,
                    indexer_full_shape,
                ]
                kv_cache_dtype_list = [spec.dtype, scale_dtype, spec.dtype]
                overlap_full_kv_cache = True
            else:
                kv_cache_shape_list = [indexer_k_shape, indexer_scale_shape]
                kv_cache_dtype_list = [spec.dtype, scale_dtype]

        return adjust_kv_layout(
            raw,
            kv_cache_shape_list,
            kv_cache_dtype_list,
            spec.page_size_bytes,
            overlap_full_kv_cache=overlap_full_kv_cache,
        )


# ---------------------------------------------------------------------------
# Layout 6: MambaLayout
# ---------------------------------------------------------------------------

class MambaLayout(KVCacheLayout):
    """Multi-state carving from a single flat buffer.

    State Space Models (Mamba, Jamba) store multiple variable-size state
    tensors (conv state, SSM state, etc.) inside one allocation. Each state
    tensor is carved out via ``adjust_kv_layout``.

    The shapes and dtypes come from ``MambaSpec.shapes`` and
    ``MambaSpec.dtypes``.

    Used for:
    - Mamba / Jamba / linear_attn layers
    """

    def num_tensors(self) -> int:
        return 1

    def split_sizes(
        self,
        total_bytes: int,
        spec: AttentionSpec,
        **kwargs: Any,
    ) -> list[int]:
        return [total_bytes]

    def reshape(
        self,
        raw_tensors: list[torch.Tensor],
        spec: MambaSpec,  # type: ignore[override]
        num_blocks: int,
        kernel_num_blocks: int,
        kernel_block_size: int,
        backend: AttentionBackend,
        vllm_config: VllmConfig,
        **kwargs: Any,
    ) -> list[torch.Tensor]:
        raw = raw_tensors[0]
        shapes_with_blocks = tuple(
            (num_blocks, *shape) for shape in spec.shapes
        )
        return adjust_kv_layout(
            raw,
            shapes_with_blocks,
            spec.dtypes,
            spec.page_size_bytes,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "KVCacheLayout",
    "SingleTensorLayout",
    "SplitKVLayout",
    "SparseMLALayout",
    "SparseMLAC8Layout",
    "CompressedMLALayout",
    "MambaLayout",
    "adjust_kv_layout",
]
