# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-owned KV cache layout planning.

The attention backend owns the physical representation consumed by its
kernels. ``NPUModelRunner`` supplies only lifecycle context and executes the
immutable plan returned here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.models.extract_hidden_states import CacheOnlyAttentionLayer
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_layout import (
    CompressedMLALayout,
    KVCacheLayout,
    KVCacheLayoutPlan,
    MambaLayout,
    SingleTensorLayout,
    SparseMLAC8Layout,
    SparseMLALayout,
    SplitKVLayout,
)
from vllm_ascend.quantization.utils import enable_fa_quant

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.kv_cache_interface import KVCacheSpec


class AscendKVCacheLayoutBackendMixin:
    """Provide the layout-plan contract shared by Ascend attention backends."""

    @staticmethod
    def _is_cache_only_layer(
        layer_name: str,
        spec: KVCacheSpec,
        vllm_config: VllmConfig,
    ) -> bool:
        if not isinstance(spec, MLAAttentionSpec):
            return False

        attention_layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,
            [layer_name],
        )
        return isinstance(attention_layers[layer_name], CacheOnlyAttentionLayer)

    @staticmethod
    def _get_attention_head_dims(
        layer_name: str,
        spec: AttentionSpec,
        vllm_config: VllmConfig,
    ) -> tuple[int, int]:
        if not isinstance(spec, MLAAttentionSpec):
            head_size_v = getattr(spec, "head_size_v", spec.head_size)
            return spec.head_size, head_size_v

        attention_layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,
            [layer_name],
        )
        attention_layer = attention_layers[layer_name]
        if isinstance(attention_layer, MLAAttention):
            return attention_layer.kv_lora_rank, attention_layer.qk_rope_head_dim
        if isinstance(attention_layer, CacheOnlyAttentionLayer):
            return spec.head_size, spec.head_size
        raise TypeError(
            f"Expected an MLA or cache-only attention layer for {layer_name}, "
            f"got {type(attention_layer).__name__}."
        )

    @classmethod
    def _select_kv_cache_layout(
        cls,
        spec: KVCacheSpec,
        *,
        is_hybrid_model: bool,
        is_cache_only: bool,
    ) -> KVCacheLayout:
        if isinstance(spec, MambaSpec):
            return MambaLayout()
        if is_cache_only or (is_hybrid_model and isinstance(spec, AttentionSpec)):
            return SingleTensorLayout()
        if isinstance(spec, MLAAttentionSpec):
            if getattr(spec, "compress_ratio", 1) > 1:
                return CompressedMLALayout()
            if getattr(spec, "cache_sparse_c8", False):
                return SparseMLAC8Layout()
            if getattr(spec, "sparse_head_dim", None) is not None:
                return SparseMLALayout()
        if isinstance(spec, AttentionSpec):
            return SplitKVLayout()
        return SingleTensorLayout()

    @classmethod
    def get_kv_cache_layout_plan(
        cls,
        spec: KVCacheSpec,
        *,
        layer_name: str,
        vllm_config: VllmConfig,
        is_hybrid_model: bool = False,
    ) -> KVCacheLayoutPlan:
        """Return the complete backend-owned physical KV cache plan.

        ``is_hybrid_model`` is lifecycle context supplied by the runner. The
        backend decides how that context affects its physical cache contract.
        """
        is_cache_only = cls._is_cache_only_layer(layer_name, spec, vllm_config)
        layout = cls._select_kv_cache_layout(
            spec,
            is_hybrid_model=is_hybrid_model,
            is_cache_only=is_cache_only,
        )

        head_dims = None
        if isinstance(layout, SplitKVLayout):
            assert isinstance(spec, AttentionSpec)
            head_dims = cls._get_attention_head_dims(
                layer_name,
                spec,
                vllm_config,
            )

        quant_config = (
            vllm_config.quant_config if enable_fa_quant(vllm_config) else None
        )
        cache_dtype_str = None
        if isinstance(spec, AttentionSpec) and not isinstance(spec, MLAAttentionSpec):
            cache_dtype_str = vllm_config.cache_config.cache_dtype

        return KVCacheLayoutPlan(
            layout=layout,
            spec=spec,
            backend=cls,
            layer_name=layer_name,
            vllm_config=vllm_config,
            head_dims=head_dims,
            quant_config=quant_config,
            cache_dtype_str=cache_dtype_str,
        )
