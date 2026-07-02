"""
离线推理脚本 — KV Cache Debug 版

与 offline_inference_npu.py 功能相同, 但在关键位置插入了 breakpoint(),
方便逐步观察 KV Cache 的分配、写入、读取全过程。

用法:
    # 方式1: 直接运行 (会在第一个断点停下)
    bash scripts/run_kv_cache_debug.sh debug

    # 方式2: 命令行运行
    ASCEND_RT_VISIBLE_DEVICES=14 python3 examples/offline_inference_npu_debug.py

    # 方式3: VSCode 调试 (在 .vscode/launch.json 中配置)

断点列表 (进入 pdb 后的常用观察命令):
    BP#1 — 观察 num_blocks、kv_cache_groups、kv_cache_tensors
    BP#2 — 观察 allocate_slots 的 block 分配
    BP#3 — 观察 block_pool 物理块分配
    BP#4 — 观察 slot_mapping 计算
    BP#5 — 观察 reshape_and_cache KV 写入

pdb 常用命令:
    p var           — 打印变量
    p var.shape     — 查看 tensor 形状
    pp vars(obj)    — 查看对象属性
    n               — 下一行
    s               — 进入函数
    c               — 继续执行到下一个断点
    bt              — 调用栈
    interact        — 进入交互式 Python shell
    q               — 退出
"""

import os
import sys

# 使用本地权重，不下载
# os.environ["VLLM_USE_MODELSCOPE"] = "True"  # 注释掉，使用本地路径
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 权重路径 (根据服务器实际情况修改)
_MODEL_PATH = os.environ.get(
    "VLLM_MODEL_PATH",
    "/home/c50058674/weight/Qwen2.5-7B-Instruct"
)

# =========================================================================
# 在 import vllm 之前, 先 patch 关键函数, 插入 breakpoint()
# =========================================================================

# --- Patch 1: KVCacheManager.allocate_slots ---
# 观察: 为 request 分配了多少个 block, block_table 如何变化
_original_allocate_slots = None


def _patch_kv_cache_manager():
    """在 allocate_slots 入口插入断点"""
    global _original_allocate_slots
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    _original_allocate_slots = KVCacheManager.allocate_slots

    def _allocate_slots_with_bp(self, request, num_new_tokens, **kwargs):
        print(f"\n{'='*60}")
        print(f"[BP#2] KVCacheManager.allocate_slots()")
        print(f"  request_id:     {request.request_id}")
        print(f"  num_new_tokens: {num_new_tokens}")
        print(f"  current blocks: {len(request.block_ids) if hasattr(request, 'block_ids') else 'N/A'}")
        print(f"  block_ids:      {request.block_ids if hasattr(request, 'block_ids') else 'N/A'}")
        print(f"{'='*60}")
        print(f"  ** 在此观察 block 分配, 输入 'c' 继续执行 **")
        breakpoint()
        result = _original_allocate_slots(self, request, num_new_tokens, **kwargs)
        if hasattr(request, 'block_ids'):
            print(f"  [after] block_ids: {request.block_ids}")
            print(f"  [after] total blocks: {len(request.block_ids)}")
        return result

    KVCacheManager.allocate_slots = _allocate_slots_with_bp
    print("[PATCH] KVCacheManager.allocate_slots — BP#2 inserted")


# --- Patch 2: BlockPool.get_new_blocks ---
# 观察: 从 FreeQueue 取物理块
_original_get_new_blocks = None


def _patch_block_pool():
    """在 get_new_blocks 入口插入断点"""
    global _original_get_new_blocks
    from vllm.v1.core.block_pool import BlockPool

    _original_get_new_blocks = BlockPool.get_new_blocks

    def _get_new_blocks_with_bp(self, num_blocks):
        print(f"\n{'='*60}")
        print(f"[BP#3] BlockPool.get_new_blocks()")
        print(f"  num_blocks requested: {num_blocks}")
        print(f"  free blocks count:    {self.free_block_queue.num_free_blocks if hasattr(self.free_block_queue, 'num_free_blocks') else 'N/A'}")
        print(f"{'='*60}")
        print(f"  ** 在此观察物理块分配, 输入 'c' 继续执行 **")
        breakpoint()
        result = _original_get_new_blocks(self, num_blocks)
        print(f"  [after] allocated block_ids: {[b.block_id for b in result]}")
        return result

    BlockPool.get_new_blocks = _get_new_blocks_with_bp
    print("[PATCH] BlockPool.get_new_blocks — BP#3 inserted")


# --- Patch 3: AscendBlockTables.compute_slot_mappings ---
# 观察: slot_mapping 计算
_original_compute_slot_mappings = None


def _patch_block_tables():
    """在 compute_slot_mappings 入口插入断点"""
    global _original_compute_slot_mappings
    try:
        from vllm_ascend.worker.v2.block_table import AscendBlockTables

        if hasattr(AscendBlockTables, 'compute_slot_mappings'):
            _original_compute_slot_mappings = AscendBlockTables.compute_slot_mappings

            def _compute_slot_mappings_with_bp(self, idx_mapping, query_start_loc, positions, num_tokens_padded):
                print(f"\n{'='*60}")
                print(f"[BP#4] AscendBlockTables.compute_slot_mappings()")
                print(f"  num_groups:              {self.num_kv_cache_groups}")
                print(f"  max_num_batched_tokens:  {self.max_num_batched_tokens}")
                print(f"  block_sizes:             {self.block_sizes_tensor}")
                print(f"  slot_mappings dtype:     {self.slot_mappings.dtype}")  # 应该是 int32
                print(f"  num_tokens_padded:       {num_tokens_padded}")
                print(f"{'='*60}")
                print(f"  ** 在此观察 slot_mapping 计算, 输入 'c' 继续执行 **")
                breakpoint()
                result = _original_compute_slot_mappings(self, idx_mapping, query_start_loc, positions, num_tokens_padded)
                print(f"  [after] slot_mappings sample (first 30): {result[0, :30].tolist()}")
                return result

            AscendBlockTables.compute_slot_mappings = _compute_slot_mappings_with_bp
            print("[PATCH] AscendBlockTables.compute_slot_mappings — BP#4 inserted")
    except ImportError:
        print("[PATCH] AscendBlockTables not available (v2 runner not used), BP#4 skipped")


# --- Patch 4: AscendAttentionBackendImpl.reshape_and_cache ---
# 观察: KV Cache 写入
_original_reshape_and_cache = None


def _patch_reshape_and_cache():
    """在 reshape_and_cache 入口插入断点"""
    global _original_reshape_and_cache
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        _original_reshape_and_cache = AscendAttentionBackendImpl.reshape_and_cache

        def _reshape_and_cache_with_bp(self, query, key, value, kv_cache, attn_metadata, output):
            print(f"\n{'='*60}")
            print(f"[BP#5] AscendAttentionBackendImpl.reshape_and_cache()")
            print(f"  key.shape:              {key.shape}")
            print(f"  value.shape:            {value.shape}")
            print(f"  attn_state:             {attn_metadata.attn_state}")
            print(f"  num_actual_tokens:      {attn_metadata.num_actual_tokens}")

            if self.key_cache is not None:
                print(f"  key_cache.shape:        {self.key_cache.shape}")
                print(f"  key_cache.dtype:        {self.key_cache.dtype}")
                print(f"  value_cache.shape:      {self.value_cache.shape}")
                print(f"  value_cache.dtype:      {self.value_cache.dtype}")
                # 打印 cache 中目前是否全零
                print(f"  key_cache is all zero:  {self.key_cache.sum().item() == 0}")
            else:
                print(f"  key_cache:              None (will be set from kv_cache)")

            slots = attn_metadata.slot_mapping
            if slots is not None:
                print(f"  slot_mapping[:20]:      {slots[:20].tolist()}")
                # 反算 block_id 和 offset
                if hasattr(self, 'key_cache') and self.key_cache is not None:
                    block_size = self.key_cache.shape[1] if self.key_cache.ndim >= 2 else -1
                    block_ids = [s // block_size for s in slots[:20].tolist()] if block_size > 0 else []
                    offsets = [s % block_size for s in slots[:20].tolist()] if block_size > 0 else []
                    print(f"  → block_ids[:20]:       {block_ids}")
                    print(f"  → offsets[:20]:         {offsets}")

            print(f"{'='*60}")
            print(f"  ** 在此观察 KV Cache 写入, 输入 'c' 继续执行 **")
            print(f"  ** 写入前验证: self.key_cache[block_id, offset] 是全零 **")
            print(f"  ** 写入后验证: self.key_cache[block_id, offset] == key[token_idx] **")
            breakpoint()

            result = _original_reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output)

            # 写入后验证
            if self.key_cache is not None and slots is not None and key is not None:
                block_size = self.key_cache.shape[1]
                for i in range(min(3, len(slots))):
                    s = slots[i].item()
                    blk, off = s // block_size, s % block_size
                    if s >= 0 and blk < self.key_cache.shape[0]:
                        written = self.key_cache[blk, off, 0, :5]  # 前 5 个 head_size 元素
                        expected = key[i, 0, :5]
                        print(f"  [verify token {i}] slot={s}, block={blk}, offset={off}")
                        print(f"    written:  {written}")
                        print(f"    expected: {expected}")
            return result

        AscendAttentionBackendImpl.reshape_and_cache = _reshape_and_cache_with_bp
        print("[PATCH] AscendAttentionBackendImpl.reshape_and_cache — BP#5 inserted")
    except ImportError:
        print("[PATCH] AscendAttentionBackendImpl not available, BP#5 skipped")


# --- Patch 5: Attention forward — 观察 Prefill vs Decode 路由 ---
_original_forward_impl = None


def _patch_forward_impl():
    """在 forward_impl 入口观察 Prefill vs Decode 路由"""
    global _original_forward_impl
    try:
        from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl

        _original_forward_impl = AscendAttentionBackendImpl.forward_impl

        def _forward_impl_with_log(self, query, key, value, kv_cache, attn_metadata, output):
            num_tokens = query.shape[0]
            state_name = str(attn_metadata.attn_state).split('.')[-1] if attn_metadata.attn_state else "None"
            print(f"  [Attention.forward_impl] num_tokens={num_tokens}, state={state_name}, "
                  f"sliding_window={self.sliding_window}")
            return _original_forward_impl(self, query, key, value, kv_cache, attn_metadata, output)

        AscendAttentionBackendImpl.forward_impl = _forward_impl_with_log
        print("[PATCH] AscendAttentionBackendImpl.forward_impl — route logging inserted")
    except ImportError:
        print("[PATCH] AscendAttentionBackendImpl not available, forward_impl patch skipped")


# =========================================================================
# 应用所有 Patch (在 import LLM 之前!)
# =========================================================================
print("=" * 60)
print("  Applying KV Cache debug patches...")
print("=" * 60)

_patch_kv_cache_manager()       # BP#2
_patch_block_pool()              # BP#3
_patch_block_tables()            # BP#4 (需要 v2 model runner)
_patch_reshape_and_cache()       # BP#5
_patch_forward_impl()            # Prefill vs Decode 路由日志

print("=" * 60)
print("  Patches applied. Starting inference...")
print("=" * 60)
print("""
调试提示:
  - 每个请求的 prefill step 会触发 1 次 allocate_slots + N 层 reshape_and_cache
  - 每个 decode step 会触发 1 次 allocate_slots + N 层 reshape_and_cache
  - Prefill: num_tokens=512 (整条 prompt), Forward→FIA
  - Decode:  num_tokens=1, Forward→PA (paged attention)
  - 在 BP#5 观察 slot_mapping 可以区分:
      Prefill: slot_mapping 有 512 个 ID (批量写入)
      Decode:  slot_mapping 只有 batch_size 个 ID (增量写入)
""")
print("=" * 60)

# =========================================================================
# 正式开始推理
# =========================================================================
from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    sampling_params = SamplingParams(max_tokens=5, temperature=0.0)

    print("\n[INIT] Creating LLM instance...")
    print("  → 这会触发 KV Cache 初始化 (BP#1 可用在 model_runner_v1.py:3526)")
    print("  → 观察: kv_cache_config.num_blocks, kv_cache_groups, kv_cache_tensors")

    llm = LLM(
        model=_MODEL_PATH,  # 使用本地权重路径
        max_model_len=128,  # 故意设小, 减少 num_blocks, 方便观察
        max_num_seqs=2,
        enforce_eager=True,  # 禁用 CUDA Graph, 便于 debug
    )

    print("\n[RUN] Generating...")
    print("  → Prefill Step: allocate_slots → slot_mapping → reshape_and_cache")
    print("  → Decode Step:  allocate_slots → slot_mapping → reshape_and_cache (× max_tokens)")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n[RESULT] Prompt: {prompt!r}")
        print(f"[RESULT] Generated: {generated_text!r}")


if __name__ == "__main__":
    main()
