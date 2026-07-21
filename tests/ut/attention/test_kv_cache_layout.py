# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_ascend.attention.kv_cache_layout import SplitKVLayout


def test_split_kv_layout_splits_and_reshapes_distinct_fields():
    layout = SplitKVLayout(
        key_dim=512,
        value_dim=64,
        key_dtype=torch.float16,
        value_dtype=torch.float16,
        key_split_factor=1.125,
        value_split_factor=9.0,
    )
    total_size = 2 * 16 * (512 + 64) * torch.empty((), dtype=torch.float16).element_size()

    key_size, value_size = layout.raw_tensor_sizes(total_size)
    key_raw = torch.zeros(key_size, dtype=torch.int8)
    value_raw = torch.zeros(value_size, dtype=torch.int8)
    key_cache, value_cache = layout.reshape(
        key_raw,
        value_raw,
        (2, 16, 1, 512),
        (2, 16, 1, 64),
    )

    assert key_size + value_size == total_size
    assert key_cache.shape == (2, 16, 1, 512)
    assert value_cache.shape == (2, 16, 1, 64)
    assert key_cache.dtype == torch.float16
    assert value_cache.dtype == torch.float16
