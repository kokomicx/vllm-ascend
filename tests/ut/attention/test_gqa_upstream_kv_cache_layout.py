import pytest
import torch

pytest.importorskip("vllm.v1.kv_cache_layout")

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)
from vllm.v1.kv_cache_layout import KVCacheLayout  # noqa: E402
from vllm.v1.worker.utils import allocate_kv_cache  # noqa: E402

from vllm_ascend.attention.gqa_kv_cache import (  # noqa: E402
    customize_standard_gqa_spec,
    is_standard_gqa_kv_cache_config,
    split_standard_gqa_kv_cache,
)


def test_upstream_allocator_to_ascend_gqa_views():
    layer_names = ["model.layers.0.self_attn", "model.layers.1.self_attn"]
    num_blocks = 3
    spec = customize_standard_gqa_spec(
        FullAttentionSpec(
            block_size=5,
            num_kv_heads=2,
            head_size=4,
            head_size_v=4,
            dtype=torch.float32,
        )
    )
    assert spec.num_head_slots == 2
    assert spec.state_content_bytes == 32
    layer_stride = spec.page_size_bytes * num_blocks
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=layer_stride * len(layer_names),
                layers=layer_names,
                layer_stride=layer_stride,
                block_stride=spec.page_size_bytes // 2,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names, spec)],
    )

    assert is_standard_gqa_kv_cache_config(config)
    caches = allocate_kv_cache(config, torch.device("cpu"), KVCacheLayout.LHBNC)

    first = caches[layer_names[0]]
    second = caches[layer_names[1]]
    assert first.shape == (num_blocks, 2, spec.block_size, 8)
    assert first.stride() == (40, 120, 8, 1)
    assert second.data_ptr() - first.data_ptr() == layer_stride
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()

    key_cache, value_cache = split_standard_gqa_kv_cache(
        first,
        spec.num_kv_heads,
        spec.head_size,
        spec.head_size_v,
    )
    assert key_cache.shape == (num_blocks, spec.block_size, spec.num_kv_heads, 4)
    assert value_cache.shape == (num_blocks, spec.block_size, spec.num_kv_heads, 4)
    assert key_cache.is_contiguous()
    assert value_cache.is_contiguous()
    assert key_cache.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert value_cache.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
