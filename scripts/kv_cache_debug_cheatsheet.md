# KV Cache Debug 速查卡

## 启动命令

```bash
# 方式1: 正常运行 (观察日志)
bash scripts/run_kv_cache_debug.sh normal

# 方式2: 断点调试 (逐步观察)
bash scripts/run_kv_cache_debug.sh debug

# 方式3: 详细日志 (打印所有 KV Cache tensor)
bash scripts/run_kv_cache_debug.sh verbose

# 方式4: 单元测试 (不需要 NPU / 模型)
bash scripts/run_kv_cache_debug.sh unit

# 方式5: 算子测试 (需要 NPU, 不需模型)
bash scripts/run_kv_cache_debug.sh op-test

# 切换模型
MODEL=/home/c50058674/weight/Qwen2.5-7B-Instruct bash scripts/run_kv_cache_debug.sh normal

# 切换到 Llama-3.1-8B
MODEL=/home/c50058674/weight/Meta-Llama-3.1-8B-Instruct bash scripts/run_kv_cache_debug.sh normal

# 切换到 Qwen3-8B (可能有 hybrid KV cache)
MODEL=/home/c50058674/weight/Qwen3-8B bash scripts/run_kv_cache_debug.sh normal

# 切回默认双 chip (TP=2 时需要)
ASCEND_RT_VISIBLE_DEVICES=14,15 bash scripts/run_kv_cache_debug.sh normal
```

## 断点速查

| 断点 | 位置 | 触发时机 | 观察什么 |
|------|------|---------|---------|
| **BP#1** | `model_runner_v1.py:3526` | 启动时 1 次 | `kv_cache_config.num_blocks`, `kv_cache_groups`, `kv_cache_tensors` |
| **BP#2** | `kv_cache_manager.py:236` | 每个 step | `request.block_ids` 如何增长 |
| **BP#3** | `block_pool.py:333` | 需要新 block 时 | `FreeQueue` → 取走哪几个物理 block |
| **BP#4** | `v2/block_table.py:79` | 每个 step | `slot_mappings` 值 — token→物理地址映射 |
| **BP#5** | `attention_v1.py:1315` | 每层 attention | key/value 写入 cache 前后对比 |

## pdb 常用命令

```
p var                  # 打印变量
p var.shape            # tensor 形状
p var[:, :10]          # 前 10 个元素
n                      # 下一行
s                      # 进入函数调用
c                      # 继续执行 (到下一个断点)
bt                     # 调用栈
l                      # 当前代码
pp vars(obj)           # 对象属性
interact      ·         # 进入交互 Python (可执行任意代码!)
q                      # 退出
```

## 核心 Tensor 形状速查 (Qwen2.5-7B-Instruct)

```
标准 Attention (Qwen2.5-7B):
  KV Cache tensor:  (2, num_blocks, block_size, num_kv_heads, head_size)
                    = (2, ~32, 16, 4, 128)
                       ↑  ↑     ↑   ↑  ↑
                    K/V 物理块 每块  KV  head
                                 16   头数 维度
                                  token

  key_cache:         [num_blocks, block_size, num_kv_heads, head_size]
  value_cache:       [num_blocks, block_size, num_kv_heads, head_size]
  slot_mapping:      [num_tokens]  dtype=int32 (Ascend 特有!)
  block_table:       [num_requests, max_blocks]  dtype=int32

  key (新计算):       [num_tokens, num_kv_heads, head_size]
  value (新计算):     [num_tokens, num_kv_heads, head_size]

  Each block: 2 × 16 × 4 × 128 × 2(bytes) = 32KB  (bf16)
  KV Cache (max_model_len=512): 32 blocks × 32KB = ~1MB per layer
                               28 layers = ~28MB total

Prefill vs Decode:
  Prefill: num_tokens=512 (整条 prompt) → slot_mapping 有 512 个 ID
  Decode:  num_tokens=1   (1 个新 token) → slot_mapping 只有 1 个 ID
```

## 关键计算验证公式

```python
# 1. Slot Mapping 验证
# 在 BP#4 拿到 slot_mappings 后, 在 interact 中执行:
block_table = attn_metadata.block_tables  # or self.block_table
block_size = 16
for token_i in range(min(50, len(slot_mapping))):
    expected = block_table[req_idx, token_i // block_size] * block_size + (token_i % block_size)
    actual = slot_mapping[token_i]
    assert expected == actual or actual == -1, f"Mismatch at token {token_i}"

# 2. KV Cache 写入验证
# 在 BP#5 reshape_and_cache 执行后:
for i in range(min(3, len(slots))):
    s = slots[i].item()
    blk, off = s // block_size, s % block_size
    assert torch.allclose(key_cache[blk, off], key[i])
    assert torch.allclose(value_cache[blk, off], value[i])

# 3. Block 占用内存计算
page_size = block_size * num_kv_heads * head_size * dtype_size * 2  # *2 for K+V
total_mb = page_size * num_blocks / 1024 / 1024
print(f"Each block: {page_size/1024:.1f} KB")
print(f"Total KV cache: {total_mb:.1f} MB")
```

## 服务器环境

```
NPU 7: Chip 14 (0000:81:00.0) + Chip 15 (0000:83:00.0)
HBM: 64GB each, 当前 ~3GB 已用
设备选择: ASCEND_RT_VISIBLE_DEVICES=14

推荐学习模型: Qwen2.5-7B-Instruct
  路径:   /home/c50058674/weight/Qwen2.5-7B-Instruct/
  架构:   Dense 7B, GQA (4 KV heads × 128 head_size), 28 layers
  KV:     标准 FullAttention, 无 SWA, 无量化, 无 MoE
  预估:   max_model_len=512 → num_blocks≈32 → KV Cache ≈256MB

备选: Meta-Llama-3.1-8B-Instruct  (/home/c50058674/weight/Meta-Llama-3.1-8B-Instruct/)
备选: Qwen3-8B                   (/home/c50058674/weight/Qwen3-8B/)
```

## VSCode launch.json 配置

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "vllm-ascend: KV Cache Debug",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/examples/offline_inference_npu_debug.py",
            "cwd": "${workspaceFolder}",
            "env": {
                "ASCEND_RT_VISIBLE_DEVICES": "14",
                "VLLM_MODEL_PATH": "/home/c50058674/weight/Qwen2.5-7B-Instruct",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"
            },
            "justMyCode": false
        }
    ]
}
```

## 常见问题

| 问题 | 解决 |
|------|------|
| `torch.npu` 不可用 | `ASCEND_RT_VISIBLE_DEVICES` 设错了? 试试 `unset` 再用默认 |
| Worker 进程断点不生效 | `VLLM_WORKER_MULTIPROC_METHOD=spawn` 时 Worker 独立, 在 Worker 代码加 `breakpoint()` |
| 模型下载失败 | 现在使用本地权重路径 `/home/c50058674/weight/`, 不需要下载 |
| OOM | 调低 `max_model_len` (128 → 64), 或减少 `max_num_seqs` |
| Prefill vs Decode 无法区分 | 看 `attn_metadata.attn_state`: `PrefillNoCache` / `PrefillCacheHit` / `DecodeOnly` |
