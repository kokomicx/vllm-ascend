# KV Cache Allocate / Reshape 架构分析

## 总览：三大家族

社区现有模型按 KV cache 管理方式分三类：

| 类别 | 模型代表 | 关键标志 | KVCacheSpec | Manager |
|------|----------|----------|-------------|---------|
| **① GQA (标准)** | Qwen3, Llama-3.1, Qwen2.5 | `use_mla=False` | `FullAttentionSpec` | `FullAttentionManager` |
| **② MLA (稀疏)** | DeepSeek V3.1 / V3.2 | `use_mla=True, use_sparse=True` | `AscendMLAAttentionSpec` | `FullAttentionManager` |
| **③ Compressed MLA** | DeepSeek V4 | `use_compress=True` | `MLAAttentionSpec(compress_ratio>1)` + `SlidingWindowMLASpec` | `CompressAttentionManager` + `SlidingWindowManager` |

---

## ① GQA 标准 Attention（Qwen3 / Llama / Qwen2.5）

### 触发条件

```python
# model_runner_v1.py:328-332
self.use_sparse = False   # 没有 index_topk
self.use_compress = False # 没有 compress_ratios
# model_config.use_mla = False
```

### KV Cache Spec

使用上游 vLLM 原生的 `FullAttentionSpec`：
- `block_size`: 128（Ascend 强制）
- `num_kv_heads`: 模型 KV head 数（如 Qwen2.5-7B 为 4）
- `head_size`: head 维度（如 128）
- `dtype`: bfloat16 或 float8_e4m3fn
- `page_size_bytes` = `block_size * num_kv_heads * head_size * dtype_size * 2`

### 分组方案

- 所有同规格 attention 层属于**同一个 KV cache group**
- 没有 `UniformTypeKVCacheSpecs`，直接使用单一 `FullAttentionSpec`
- `group_and_unify_kv_cache_specs()` 返回 `None`（不触发特殊分组）
- Block table: 单一 `BlockTable`，`use_hybrid_blocks=False`

### 空间申请 (`_allocate_kv_cache_tensors`)

**文件**: [model_runner_v1.py:3735-3818](vllm_ascend/worker/model_runner_v1.py#L3735-L3818)

```
上层调度器确定 num_blocks（物理块数）
↓
每个 KV cache tensor 的 size = num_blocks * page_size_bytes  (字节)
↓
按 K/V 维度比例拆分为两个 tensor:
  k_tensor_size = total_size // k_split_factor
  v_tensor_size = total_size // v_split_factor
↓
torch.zeros(..., dtype=torch.int8)  ← 分配原始 int8 内存
```

**策略**:

| 策略 | 适用场景 | 存储格式 |
|------|---------|---------|
| 单 tensor | Mamba / cache_only_layers / hybrid attn-mamba | `torch.zeros(size, dtype=torch.int8)` |
| K/V 双 tensor（无 PD） | 标准 attention，无 prefill disaggregation | `(k_tensor, v_tensor)` 各自 `int8` |
| K/V 双 tensor + 2MB 对齐 | 有 PD (RDMA 传输) | `(k_tensor, v_tensor)` + 2MB 对齐 |

### Reshape (`_reshape_kv_cache_tensors`)

**文件**: [model_runner_v1.py:4057-4125](vllm_ascend/worker/model_runner_v1.py#L4057-L4125)

```
raw int8 tensor
  ↓ .view(kv_cache_spec.dtype)     → 转为目标 dtype (bf16/fp8)
  ↓ .view(kv_cache_shape)          → 变为 4D tensor
  ↓ kv_cache = (num_blocks, block_size, num_kv_heads, head_size)
```

**最终 KV cache tensor 形状**:

```
key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
value_cache: [num_blocks, block_size, num_kv_heads, head_size]

例如 Qwen2.5-7B (block_size=128):
  key_cache:   [~32, 128, 4, 128]   dtype=bfloat16 → ~2MB per layer
  value_cache: [~32, 128, 4, 128]   dtype=bfloat16 → ~2MB per layer
```

### reshape_and_cache (写入)

**文件**: [attention_v1.py:1315-1375](vllm_ascend/attention/attention_v1.py#L1315-L1375)

```python
DeviceOperator.reshape_and_cache(
    key=key,            # [num_tokens, num_kv_heads, head_size]
    value=value,        # [num_tokens, num_kv_heads, head_size]
    key_cache=self.key_cache,    # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache=self.value_cache,
    slot_mapping=slot_mapping,   # [num_tokens] int32, 每个 token → (block_id*block_size + offset)
)
```

C8 INT8 量化变体 (`AscendC8AttentionBackendImpl`): 先 `_quantize_kv_to_int8` 再写入。

---

## ② MLA 稀疏 Attention（DeepSeek V3.1 / V3.2）

### 触发条件

```python
# model_runner_v1.py:328-332
self.use_sparse = True    # hf_text_config 有 index_topk，但没有 compress_ratios
# model_config.use_mla = True
```

**V3.1 vs V3.2 区别**: V3.2 增加了 sparse indexer 模块，KV cache 从 2-tuple 变为 3/4-tuple。

### KV Cache Spec

**文件**: [model_runner_v1.py:4363-4378](vllm_ascend/worker/model_runner_v1.py#L4363-L4378)

使用 `AscendMLAAttentionSpec`（patch 后的 `MLAAttentionSpec`）：

```
AscendMLAAttentionSpec(
    block_size=128,
    num_kv_heads=1,                    # MLA 固定为 1
    head_size=sum(sparse_head_dim),    # kv_lora_rank + qk_rope_head_dim + index_head_dim
    sparse_head_dim=(kv_lora_rank, qk_rope_head_dim, index_head_dim),
    cache_sparse_c8=True/False,        # 是否开启 C8 量化
    dtype=bfloat16,
)
```

### 分组方案

- 所有 MLA 层属于**同一个 KV cache group**
- 不使用 `group_and_unify_kv_cache_specs`（只有 MLA spec 无 SlidingWindowMLASpec）

### 空间申请 (`_allocate_kv_cache_tensors`)

**文件**: [model_runner_v1.py:3744-3818](vllm_ascend/worker/model_runner_v1.py#L3744-L3818)

按 `sparse_kv_cache_ratio` 拆分为 **3 或 4 个 tensor**：

```
total_size = num_blocks * page_size_bytes

sparse_kv_cache_ratio = (
    total_virtual / virtual_dims[0],  # kv_cache[0]: kv_lora (nope)
    total_virtual / virtual_dims[1],  # kv_cache[1]: k_rope  (rope)
    total_virtual / virtual_dims[2],  # kv_cache[2]: dsa_k   (indexer key)
    total_virtual / virtual_dims[3],  # kv_cache[3]: dsa_k_scale (C8 only)
)

k_tensor_size  = total_size // ratio[0]   (kv_lora, nope)
v_tensor_size  = total_size // ratio[1]   (k_rope)
dsa_k_tensor_size = total_size // ratio[2]  (indexer key)
dsa_k_scale_tensor_size = total_size // ratio[3]  (仅 C8)
```

### 存储格式

| 条目 | 用途 | 无 C8 dtype | C8 dtype |
|------|------|-----------|---------|
| `kv_cache[0]` | kv_lora (nope cache) | bfloat16 | bfloat16 |
| `kv_cache[1]` | k_rope (rope cache) | bfloat16 | bfloat16 |
| `kv_cache[2]` | dsa_k (indexer key cache) | bfloat16 | **int8** |
| `kv_cache[3]` | dsa_k_scale (indexer key scale) | 不存在 | **float16** |

### Reshape (`_reshape_kv_cache_tensors_for_mla`)

**文件**: [model_runner_v1.py:3969-4050](vllm_ascend/worker/model_runner_v1.py#L3969-L4050)

```
kv_lora (k_cache):
  raw_k_tensor → .view(dtype) → .view(kernel_num_blocks, kernel_block_size, num_kv_heads, kv_lora_rank)

k_rope (v_cache):
  raw_v_tensor → .view(dtype) → .view(kernel_num_blocks, kernel_block_size, num_kv_heads, qk_rope_head_dim)

dsa_k (index_cache):
  raw_dsa_k_tensor → .view(dtype) → .view(kernel_num_blocks, kernel_block_size, num_kv_heads, index_head_dim)

dsa_k_scale (C8 only):
  raw_dsa_k_scale_tensor → .view(float16) → .view(kernel_num_blocks, kernel_block_size, num_kv_heads, 1)
```

**关键**: K dim = `kv_lora_rank`, V dim = `qk_rope_head_dim`（MLA 特有，不是标准 head_size）

### reshape_and_cache (写入)

不通过 `AscendAttentionBackendImpl.reshape_and_cache`，而是通过 **DSA 专用路径**：

- [dsa_v1.py:1752](vllm_ascend/attention/dsa_v1.py#L1752): `DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)` — 将 nope + rope 写入 cache
- Indexer 的 key cache 写入在 indexer forward 中处理

---

## ③ Compressed MLA（DeepSeek V4）

### 触发条件

```python
# model_runner_v1.py:264-266
self.use_compress = (
    hf_config is not None and hasattr(hf_config, "compress_ratios")
)
```

这是你说的 "3.2 加了个 xxx" —**DeepSeek V4 的 Compressed MLA 机制**，用 `compress_ratio` 压缩 token 数来减少 KV cache 占用。

### KV Cache Spec 体系

DeepSeek V4 有多种 KV cache spec 共存：

| Spec 类型 | 层级 | compress_ratio | 说明 |
|-----------|------|----------------|------|
| `MLAAttentionSpec` | Full MLA 层 (C4) | 4 | 标准 MLA，4x 压缩 |
| `MLAAttentionSpec` | Full MLA 层 (C128) | 128 | 高压缩 MLA，128x |
| `SlidingWindowMLASpec` | SWA MLA 层 | 1 | 滑动窗口 MLA |

### 分组方案 (`group_and_unify_kv_cache_specs`)

**文件**: [patch_kv_cache_utils.py:58-89](vllm_ascend/patch/platform/patch_kv_cache_utils.py#L58-L89)

DeepSeek V4 使用 `UniformTypeKVCacheSpecs` 进行**多 page_size 统一分组**：

```
Step 1: 按 compress_ratio 分组 MLA specs
  → ratio_specs[4]  = {layer0_mla, layer1_mla, ...}   # C4 layers
  → ratio_specs[128] = {layer2_mla, layer3_mla, ...}  # C128 layers

Step 2: 按 block_size 分组 SWA specs
  → grouped_swa_mla_specs[block_size] = {swa_layer0, swa_layer1, ...}

Step 3: 生成 UniformTypeKVCacheSpecs 列表
  → [Uniform_MLA_C4, Uniform_MLA_C128, Uniform_SWA_0, Uniform_SWA_1, ...]

Step 4: _get_kv_cache_groups_uniform_groups
  → layer tuples 对齐: 保证每个 tuple 包含 (C4_layer, C128_layer, SWA_layer0, SWA_layer1, ...)
  → page_size padding: 小 SWA page 向上对齐到大 page
```

### 最终 Group 结构

```
Group 0: Full MLA (compress_ratio=4)  → KVCacheGroupSpec
Group 1: Full MLA (compress_ratio=128) → KVCacheGroupSpec
Group 2+: SWA MLA groups              → KVCacheGroupSpec (每 group 一个 page_size)
```

### 空间申请 (`_get_kv_cache_config_deepseek_v4`)

**文件**: [patch_kv_cache_utils.py:184-244](vllm_ascend/patch/platform/patch_kv_cache_utils.py#L184-L244)

DeepSeek V4 有**专用的 tensor layout 规划**：

```
输入: available_memory (可用 HBM 字节)

layer_tuple_page_bytes = sum(all_page_sizes)
num_layer_tuples = max(bucket_sizes across all groups) + len(mtp_layers)

num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)

每个 KVCacheTensor:
  size = page_size_bytes * num_blocks
  shared_by = [对应 tuple_idx 的各 group layer]
```

**关键设计**: 不同 group 中同一位置的 layer 共享底层 tensor（通过 shared_by），减少碎片。

### Block 管理 (`CompressAttentionManager`)

**文件**: [single_type_kv_cache_manager.py:26-227](vllm_ascend/core/single_type_kv_cache_manager.py#L26-L227)

所有 token 计数除以 `compress_ratio`：

```python
def get_num_blocks_to_allocate(self, request_id, num_tokens, ...):
    num_tokens //= self.compress_ratio       # ← 核心: token 压缩
    num_tokens_main_model //= self.compress_ratio
    return super().get_num_blocks_to_allocate(...)

def allocate_new_blocks(self, request_id, num_tokens, ...):
    num_tokens //= self.compress_ratio       # ← 同样压缩
    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)
    ...
```

**Admission cap**: `max_admission_blocks_per_request = cdiv(max_model_len // compress_ratio, block_size) + 1`，防止超长 prompt 耗尽 block pool。

### KV Cache 写入

使用 DSA (DeepSeek Sparse Attention) 后端 [dsa_v1.py](vllm_ascend/attention/dsa_v1.py)：

- `_mla_prolog_multistream`: 三阶段多流 CV 并行
  - Part1: q_quant + q_a_down ∥ kv_quant
  - Part2: q_norm + q_b_quant ∥ kv_matmul
  - Part3: q_b_matmul ∥ kv_norm + rope + **scatter** (写入 cache)
  - Tail: q_rms + rope

- `DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, slot_mapping)`: 将 nope+rope 写入 cache

---

## 汇总对比

### 申请空间大小计算

| 类别 | 公式 |
|------|------|
| ① GQA | `total_bytes = num_blocks * block_size * num_kv_heads * head_size * dtype_size * 2` |
| ② MLA (稀疏) | `total_bytes = num_blocks * block_size * (kv_lora_rank + qk_rope_head_dim + index_head_dim [+ scale_dim]) * dtype_size` |
| ③ Compressed MLA | `total_bytes = num_layer_tuples * sum(page_sizes) * num_blocks` |

其中 `num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)`

### Tensor 拆分方式

| 类别 | K/V 拆分方式 | Tensor 数量 |
|------|-------------|------------|
| ① GQA | `calc_split_factor([k_dim, v_dim])` — 按 head_dim 比例 | 2 (k, v) |
| ② MLA (稀疏) | `sparse_kv_cache_ratio` — 按各 cache 条目字节占比 | 3 或 4 (nope, rope, dsa_k, [scale]) |
| ③ Compressed MLA | `_get_kv_cache_config_deepseek_v4` — 按 layer_tuple 和 page_size | N (每 tuple 每 page size 一个) |

### 存储布局

| 类别 | K Cache 维度 | V Cache 维度 | 特殊条目 |
|------|-------------|-------------|---------|
| ① GQA | `[blocks, bs, kv_heads, head_size]` | `[blocks, bs, kv_heads, head_size]` | — |
| ② MLA (稀疏) | `[blocks, bs, 1, kv_lora_rank]` | `[blocks, bs, 1, qk_rope_head_dim]` | dsa_k: `[blocks, bs, 1, index_head_dim]` |
| ③ Compressed MLA | 同 MLA | 同 MLA | 多 page_size 混存 |

### reshape_and_cache 写入路径

| 类别 | 写入函数 | 文件 |
|------|---------|------|
| ① GQA | `DeviceOperator.reshape_and_cache` | [attention_v1.py:1307](vllm_ascend/attention/attention_v1.py#L1307) |
| ① GQA (C8) | `C8BackendImpl._reshape_and_cache` | [attention_v1.py:1488](vllm_ascend/attention/attention_v1.py#L1488) |
| ② MLA (稀疏) | `DeviceOperator.dsa_kv_compress_scatter` | [dsa_v1.py:1752](vllm_ascend/attention/dsa_v1.py#L1752) |
| ③ Compressed MLA | `DeviceOperator.dsa_kv_compress_scatter` (multistream) | [dsa_v1.py:1683](vllm_ascend/attention/dsa_v1.py#L1683) |

### 核心代码路径

```
initialize_kv_cache()                          [model_runner_v1.py:3526]
├── _allocate_kv_cache_tensors()              [model_runner_v1.py:3682]
│   ├── ① GQA: K/V 双 tensor split
│   ├── ② MLA: 3/4-tuple split by sparse_kv_cache_ratio
│   └── ③ Compressed: 复用 MLA 路径 (单 tensor)
│
├── _reshape_kv_cache_tensors()               [model_runner_v1.py:4057]
│   └── ① GQA: raw int8 → .view(dtype) → .view(4D shape)
│
├── _reshape_kv_cache_tensors_for_mla()       [model_runner_v1.py:3862]
│   ├── ② MLA: raw int8 → .view(dtype) → K/V split reshape
│   └── ③ Compressed: _adjust_kv_layout() + multi-shape layout
│
└── bind_kv_cache()                            [model_runner_v1.py:3632]
```

### Manager 初始化

```python
# single_type_kv_cache_manager.py:229-275
def get_manager_for_kv_cache_spec(kv_cache_spec, ...):
    manager_class = spec_manager_map[type(kv_cache_spec)]
    if isinstance(kv_cache_spec, MLAAttentionSpec) and kv_cache_spec.compress_ratio > 1:
        manager_class = CompressAttentionManager   # ③ Compressed MLA
        # 设置 admission cap
    elif isinstance(kv_cache_spec, (SlidingWindowSpec, ChunkedLocalAttentionSpec)):
        # 设置 recycling cap
    return manager_class(kv_cache_spec, **kwargs)
```

---

## Block Table (物理→逻辑映射)

**文件**: [block_table.py](vllm_ascend/worker/block_table.py)

### Hybrid Blocks 机制

当 scheduler 的 `block_size` ≠ kernel 的 `block_size` 时启用：

```python
# block_table.py:54-80
self.blocks_per_phys_block = physical_block_size // logical_block_size
self.use_hybrid_blocks = blocks_per_phys_block > 1

# 例如: physical=128, kernel=64 → blocks_per_phys_block=2
# 物理 block 1 → 逻辑 block [2, 3]
```

### MultiGroupBlockTable

多个 KV cache group 时（③ Compressed MLA），每个 group 有独立的 `BlockTable`：

```python
# block_table.py:305-380
class MultiGroupBlockTable:
    self.block_tables = [
        BlockTable(block_size, ...) for block_size in block_sizes
    ]
```

---

## 关键标志位速查

| 标志 | 含义 | 设置位置 |
|------|------|---------|
| `self.use_mla` | 是否 MLA 模型 | `model_config.use_mla` |
| `self.use_sparse` | DS V3.2 稀疏 attention | [model_runner_v1.py:328](vllm_ascend/worker/model_runner_v1.py#L328) |
| `self.use_compress` | DS V4 压缩 MLA | [model_runner_v1.py:264](vllm_ascend/worker/model_runner_v1.py#L264) |
| `self.use_sparse_c8_indexer` | DS V3.2 C8 量化 indexer | [model_runner_v1.py:340](vllm_ascend/worker/model_runner_v1.py#L340) |
| `self.use_hybrid_blocks` | 多 group 混合 blocks | [model_runner_v1.py:3541](vllm_ascend/worker/model_runner_v1.py#L3541) |
