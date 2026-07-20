# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical KV-cache layouts owned by Ascend attention backends."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SplitKVLayout:
    """Layout for attention caches stored as independent K and V tensors.

    The attention backend owns the cache shape contract. This helper only
    applies that contract to the two raw byte buffers allocated by the model
    runner. It deliberately has no model-specific routing: any backend with
    two independently allocated cache fields can reuse it when its storage
    and view rules match this layout.
    """

    key_dim: int
    value_dim: int
    key_dtype: torch.dtype
    value_dtype: torch.dtype
    key_split_factor: float
    value_split_factor: float

    def raw_tensor_sizes(self, total_size: int) -> tuple[int, int]:
        """Split a combined cache size into independent raw K/V byte buffers."""
        return (
            int(total_size // self.key_split_factor),
            int(total_size // self.value_split_factor),
        )

    def reshape(
        self,
        raw_key_tensor: torch.Tensor,
        raw_value_tensor: torch.Tensor,
        kv_cache_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """View raw K/V buffers using the shape returned by the backend.

        The leading dimension of a two-field backend shape identifies K/V and
        is not part of either physical tensor. The last dimension is replaced
        with the field-specific head dimension.
        """
        assert len(kv_cache_shape) >= 2
        cache_prefix = kv_cache_shape[1:-1]
        key_shape = (*cache_prefix, self.key_dim)
        value_shape = (*cache_prefix, self.value_dim)
        return (
            raw_key_tensor.view(self.key_dtype).view(key_shape),
            raw_value_tensor.view(self.value_dtype).view(value_shape),
        )
