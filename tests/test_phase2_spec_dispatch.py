# SPDX-License-Identifier: Apache-2.0
"""
Phase 2 validation: every Spec class returns the correct Layout type
without any isinstance / if-else branching in the dispatch path.

Run on server:
    PYTHONPATH=/path/to/vllm:$PYTHONPATH ASCEND_RT_VISIBLE_DEVICES=0 \
    python tests/test_phase2_spec_dispatch.py
"""

from __future__ import annotations

import torch

# Must import the patch module FIRST so monkey-patching takes effect
# before any downstream code accesses vllm.v1.kv_cache_interface classes.
import vllm_ascend.patch.platform.patch_kv_cache_interface  # noqa: F401

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)

from vllm_ascend.core.kv_cache_layout import (
    CompressedMLALayout,
    MambaLayout,
    SingleTensorLayout,
    SparseMLAC8Layout,
    SparseMLALayout,
    SplitKVLayout,
)


def test_all_spec_classes_have_get_layout():
    """Every KVCacheSpec subclass must expose get_kv_cache_layout()."""
    spec_types_to_check = [
        FullAttentionSpec,
        MambaSpec,
        MLAAttentionSpec,
        SlidingWindowMLASpec,
    ]
    for spec_cls in spec_types_to_check:
        assert hasattr(spec_cls, "get_kv_cache_layout"), (
            f"{spec_cls.__name__} is missing get_kv_cache_layout()"
        )
    print("✓ All Spec classes have get_kv_cache_layout()")


def test_full_attention_spec():
    """FullAttentionSpec → SplitKVLayout (GQA models)."""
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
        head_size_v=128,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SplitKVLayout), f"Expected SplitKVLayout, got {type(layout).__name__}"
    print("✓ FullAttentionSpec → SplitKVLayout")


def test_standard_mla():
    """Standard MLA (no sparse, no compress) → SplitKVLayout."""
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SplitKVLayout), f"Expected SplitKVLayout, got {type(layout).__name__}"
    print("✓ Standard MLA → SplitKVLayout")


def test_sparse_mla():
    """Sparse MLA (DS V3.2) → SparseMLALayout."""
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=704,
        dtype=torch.bfloat16,
        sparse_head_dim=(512, 64, 128),
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SparseMLALayout), f"Expected SparseMLALayout, got {type(layout).__name__}"
    print("✓ Sparse MLA → SparseMLALayout")


def test_sparse_mla_c8():
    """Sparse MLA + C8 (DS V3.2+quant) → SparseMLAC8Layout."""
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=704,
        dtype=torch.bfloat16,
        sparse_head_dim=(512, 64, 128),
        cache_sparse_c8=True,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SparseMLAC8Layout), f"Expected SparseMLAC8Layout, got {type(layout).__name__}"
    print("✓ Sparse MLA C8 → SparseMLAC8Layout")


def test_compressed_mla():
    """Compressed MLA (DS V4) → CompressedMLALayout."""
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, CompressedMLALayout), (
        f"Expected CompressedMLALayout, got {type(layout).__name__}"
    )
    print("✓ Compressed MLA → CompressedMLALayout")


def test_sliding_window_mla_standard():
    """Sliding window MLA (standard) → SplitKVLayout."""
    spec = SlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=576,
        sliding_window=4096,
        dtype=torch.bfloat16,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SplitKVLayout), f"Expected SplitKVLayout, got {type(layout).__name__}"
    print("✓ SlidingWindow MLA (standard) → SplitKVLayout")


def test_sliding_window_mla_compressed():
    """Sliding window MLA (compressed) → CompressedMLALayout."""
    spec = SlidingWindowMLASpec(
        block_size=128,
        num_kv_heads=1,
        head_size=512,
        sliding_window=4096,
        compress_ratio=4,
        dtype=torch.bfloat16,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, CompressedMLALayout), (
        f"Expected CompressedMLALayout, got {type(layout).__name__}"
    )
    print("✓ SlidingWindow MLA (compressed) → CompressedMLALayout")


def test_mamba_spec():
    """MambaSpec → MambaLayout."""
    spec = MambaSpec(
        block_size=128,
        shapes=((4, 256),),
        dtypes=(torch.float32,),
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, MambaLayout), f"Expected MambaLayout, got {type(layout).__name__}"
    print("✓ MambaSpec → MambaLayout")


def test_kvcachespec_base_fallback():
    """KVCacheSpec base (e.g. HiddenStateCacheSpec) → SingleTensorLayout."""
    from vllm.v1.kv_cache_interface import HiddenStateCacheSpec
    spec = HiddenStateCacheSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.bfloat16,
    )
    layout = spec.get_kv_cache_layout()
    assert isinstance(layout, SingleTensorLayout), (
        f"Expected SingleTensorLayout, got {type(layout).__name__}"
    )
    print("✓ KVCacheSpec fallback → SingleTensorLayout")


def test_no_isinstance_in_dispatch():
    """Verify get_kv_cache_layout does NOT use isinstance on spec type (self-check)."""
    import inspect
    source = inspect.getsource(MLAAttentionSpec.get_kv_cache_layout)
    # The decision tree should only check self attributes, not isinstance(self, Xxx)
    assert "isinstance(self" not in source, (
        "get_kv_cache_layout() should NOT use isinstance! Decision is based on "
        "spec fields (sparse_head_dim, cache_sparse_c8, compress_ratio), not class type."
    )
    print("✓ No isinstance in MLA dispatch — pure field-based decision")


if __name__ == "__main__":
    test_all_spec_classes_have_get_layout()
    test_full_attention_spec()
    test_standard_mla()
    test_sparse_mla()
    test_sparse_mla_c8()
    test_compressed_mla()
    test_sliding_window_mla_standard()
    test_sliding_window_mla_compressed()
    test_mamba_spec()
    test_kvcachespec_base_fallback()
    test_no_isinstance_in_dispatch()
    print("\n" + "=" * 50)
    print("ALL PHASE 2 TESTS PASSED")
