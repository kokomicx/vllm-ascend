import pytest
import torch

from vllm_ascend.attention.gqa_kv_cache import (
    normalize_kernel_block_sizes,
    split_standard_gqa_kv_cache,
)


def test_split_standard_gqa_kv_cache_returns_contiguous_views():
    num_blocks = 3
    num_heads = 2
    block_size = 5
    head_size = 4
    width = num_heads * head_size
    raw = torch.arange(
        num_blocks * 2 * block_size * width,
        dtype=torch.float32,
    )
    kv_cache = torch.as_strided(
        raw,
        size=(num_blocks, 2, block_size, width),
        stride=(block_size * width, num_blocks * block_size * width, width, 1),
    )

    key_cache, value_cache = split_standard_gqa_kv_cache(
        kv_cache, num_heads, head_size, head_size
    )

    assert key_cache.shape == (num_blocks, block_size, num_heads, head_size)
    assert value_cache.shape == (num_blocks, block_size, num_heads, head_size)
    assert key_cache.is_contiguous()
    assert value_cache.is_contiguous()
    torch.testing.assert_close(key_cache, kv_cache[:, 0].view(key_cache.shape))
    torch.testing.assert_close(value_cache, kv_cache[:, 1].view(value_cache.shape))
    assert key_cache.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()
    assert value_cache.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()


def test_split_standard_gqa_kv_cache_rejects_wrong_rank():
    with pytest.raises(ValueError, match="rank-4"):
        split_standard_gqa_kv_cache(torch.empty(3, 2, 8), 2, 4, 4)


def test_split_standard_gqa_kv_cache_rejects_wrong_width():
    with pytest.raises(ValueError, match="num_kv_heads \\* head_size"):
        split_standard_gqa_kv_cache(torch.empty(3, 2, 5, 7), 2, 4, 4)


def test_split_standard_gqa_kv_cache_rejects_asymmetric_heads():
    with pytest.raises(ValueError, match="symmetric head sizes"):
        split_standard_gqa_kv_cache(torch.empty(3, 2, 5, 8), 2, 4, 2)


def test_normalize_kernel_block_sizes():
    assert normalize_kernel_block_sizes([[128], [64]]) == [128, 64]
    assert normalize_kernel_block_sizes([128, 64]) == [128, 64]
    assert normalize_kernel_block_sizes([]) is None
