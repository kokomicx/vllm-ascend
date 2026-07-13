# NPU Operator Dependency Inventory — KV Cache Management

> **目的**：梳理 vLLM-Ascend 中每个模型类型涉及的所有 NPU 算子依赖，带上 layout 描述符和 stride/连续性要求，便于与算子侧逐一对齐。
>
> **生成日期**：2026-07-13
> **基于分支**: `feature/layout-refactor-phase3`
> **对齐对象**：社区 vLLM（CUDA kernel 行为）vs vLLM-Ascend（NPU kernel 行为）

---

## 1. 模型类型 ↔ Backend ↔ Layout 映射

| 编号 | 模型类型 | 典型模型 | Backend | Layout 类 | KV Cache Tensor 数 |
|------|---------|---------|---------|-----------|-------------------|
| **A** | GQA | Qwen3 MoE, Qwen2.5, Llama-3.1 | `AscendAttentionBackend` | `SplitKVLayout` | **2** (K + V) |
| **B** | 标准 MLA (Dense) | DeepSeek V3.1 | `AscendMLABackend` | `SplitKVLayout` | **2** (K + V) |
| **C** | Sparse MLA (SFA) | DeepSeek V3.2, GLM5.1 | `AscendSFABackend` | `SparseMLALayout` / `SparseMLAC8Layout` | **3** (K + V + Indexer_K) 或 **4** (+ Scale) |
| **D** | Compress MLA (DSA) | DeepSeek V4 | `AscendDSABackend` | `CompressedMLALayout` | **4~7** (Compress_K + SWA_K + State + Indexer_K + Scale + ...) |
| **E** | Hybrid | Qwen3.5 | Mamba backend + GQA/MLA backend | `MambaLayout` + `SplitKVLayout` | **1** (mamba) + **2** (attention) |

---

## 2. 算子依赖矩阵

### 2.1 模型类型 A: GQA（AscendAttentionBackend）

#### 2.1.1 Execute KV: 写入 KV Cache

| 算子 | `npu_scatter_pa_kv_cache_vllm` (V1) / `npu_scatter_pa_kv_cache` (V2) |
|------|---------------------------------------------------------------------|
| 入口 | `torch.ops._C_ascend.npu_scatter_pa_kv_cache_vllm` / `torch_npu.npu_scatter_pa_kv_cache` |
| 文件 | [device_op.py:36](vllm_ascend/device/device_op.py#L36) / [device_op.py:565](vllm_ascend/device/device_op.py#L565) |
| 输入 | `key: (num_tokens, num_kv_heads, head_size)`, `value: (num_tokens, num_kv_heads, head_size)` |
| KV Cache | `key_cache: (num_blocks, block_size, num_kv_heads, head_size)` — layout: **PA_BSND** implicit |
| | `value_cache: (num_blocks, block_size, num_kv_heads, head_size)` — layout: **PA_BSND** implicit |
| Stride 要求 | **K 和 V 必须物理分离**，各自连续（`is_contiguous()=True`） |
| 对比 CUDA | CUDA kernel 内部可以通过 stride 访问单个 `(2,N,B,H,D)` tensor 中的 K/V 部分 |

#### 2.1.2 Prefill Attention

| 算子 | `npu_fused_infer_attention_score` / `npu_fused_infer_attention_score_v2` |
|------|-----------------------------------------------------------------------|
| 入口 | `torch_npu.npu_fused_infer_attention_score` / `torch_npu.npu_fused_infer_attention_score_v2` |
| 文件 | [attention_v1.py:1148-1220](vllm_ascend/attention/attention_v1.py#L1148-L1220) |
| Query | `(num_tokens, num_heads, head_size)` — layout: **TND** (`input_layout="TND"`) |
| Key | `key_cache: (num_blocks, block_size, num_kv_heads, head_size)` |
| Value | `value_cache: (num_blocks, block_size, num_kv_heads, head_size)` |
| Num KV Heads | `self.num_kv_heads` (e.g., Qwen3: 8) |
| 变体 | `sparse_mode=0` (非 causal), `sparse_mode=3` (causal), `sparse_mode=4` (sliding window) |
| 变体 sink | `npu_fused_infer_attention_score_v2` + `learnable_sink` 参数 |
| Stride 要求 | K/V cache 各自连续；Query 需要连续 |

#### 2.1.3 Decode Attention (Paged Attention)

| 算子 | `_npu_paged_attention` |
|------|----------------------|
| 入口 | `torch_npu._npu_paged_attention` |
| 文件 | [attention_v1.py:1229-1240](vllm_ascend/attention/attention_v1.py#L1229-L1240) |
| Query | `(num_tokens, num_heads, head_size)` |
| Key Cache | `key_cache: (num_blocks, block_size, num_kv_heads, head_size)` |
| Value Cache | `value_cache: (num_blocks, block_size, num_kv_heads, head_size)` |
| 参数 | `block_table`, `context_lens` |
| Stride 要求 | K/V cache 各自物理连续（PagedAttention 内部按 page 寻址） |

#### 2.1.4 KV Cache Load (Gather)

| 算子 | `npu_gather_pa_kv_cache_vllm` |
|------|-------------------------------|
| 入口 | `torch.ops._C_ascend.npu_gather_pa_kv_cache_vllm` |
| 文件 | [device_op.py:202](vllm_ascend/device/device_op.py#L202) |
| 用途 | 从分页 KV cache 中 gather 连续的 K/V tensor（用于 prefill 序列拼接） |
| Stride 要求 | K/V cache 各自连续 |

---

### 2.2 模型类型 B: 标准 MLA (Dense)（AscendMLABackend）

#### 2.2.1 MLA Preprocess + KV Write（合并算子）

| 算子 | `npu_mla_prolog_v2` (A3) / `npu_mla_prolog_v3` (其他设备) |
|------|---------------------------------------------------------|
| 入口 | `torch_npu.npu_mla_prolog_v2` / `torch_npu.npu_mla_prolog_v3` |
| 文件 | [device_op.py:225](vllm_ascend/device/device_op.py#L225) / [device_op.py:869](vllm_ascend/device/device_op.py#L869) |
| 功能 | **融合算子**: Q/KV 低秩分解 + KV cache 写入 + RoPE + RMSNorm，一次性完成 |
| 输入 | `hidden_states: (num_tokens, hidden_dim)` |
| Output | `decode_q_nope, decode_q_pe, decode_k_nope, decode_k_pe, dequant_scale_q_nope` |
| Stride 要求 | 输入 kv_cache 连续 |

#### 2.2.2 MLA Preprocess (无融合)

| 算子 | `mla_preprocess` (fallback 路径) |
|------|-------------------------------|
| 入口 | `torch.ops._C_ascend.mla_preprocess` |
| 文件 | [device_op.py:257](vllm_ascend/device/device_op.py#L257) |
| 条件 | `num_tokens > MLAPO_MAX_SUPPORTED_TOKENS(1024)` 或非量化模型 |

#### 2.2.3 MLA Attention Score (Prefill & Decode)

| 算子 | `npu_fused_infer_attention_score` / `npu_fused_infer_attention_score_v2` |
|------|-----------------------------------------------------------------------|
| 入口 | 同 2.1.2 GQA，但参数不同 |
| 文件 | [mla_v1.py:874](vllm_ascend/attention/mla_v1.py#L874) / [mla_v1.py:1230](vllm_ascend/attention/mla_v1.py#L1230) / [mla_v1.py:1582](vllm_ascend/attention/mla_v1.py#L1582) |
| 与 GQA 区别 | MLA 下 `num_kv_heads=1`, `head_size=kv_lora_rank(=512)` |
| Stride 要求 | K/V cache 各自连续；Query 需要连续 |

---

### 2.3 模型类型 C: Sparse MLA / SFA（AscendSFABackend）

#### 2.3.1 Sparse Indexer（Top-K 选择）

| 算子 | `npu_lightning_indexer` / `npu_lightning_indexer_quant` (C8) |
|------|-------------------------------------------------------------|
| 入口 | `torch.ops._C_ascend.npu_lightning_indexer` (bf16) / `torch.ops._C_ascend.npu_lightning_indexer_quant` (C8) / `torch_npu.npu_lightning_indexer` |
| 文件 | [sfa_v1.py:994-1035](vllm_ascend/attention/sfa_v1.py#L994-L1035) |
| Query | `q_li: (num_tokens, index_n_heads, index_head_dim)` — layout: **TND** (`layout_query="TND"`) |
| Key (从 KV Cache) | `kv_cache[2]: (num_blocks, block_size, 1, index_head_dim)` — layout: **PA_BSND** (`layout_key="PA_BSND"`) |
| Key (C8) | `kv_cache[2]: (num_blocks, block_size, 1, index_head_dim)` int8 + `kv_cache[3]: (num_blocks, block_size, 1, 1)` quant_scale fp16 |
| Output | `topk_indices: (num_tokens, topk)` — 稀疏索引 |
| 参数 | `sparse_count=2048`, `sparse_mode=3` |
| Stride 要求 | **Indexer_K 必须独立 tensor**，物理连续；C8 变体 Indexer_K 和 Scale 必须各自独立 |

#### 2.3.2 Sparse Flash Attention（真正的 attention 计算）

| 算子 | `npu_sparse_flash_attention` |
|------|----------------------------|
| 入口 | `torch.ops._C_ascend.npu_sparse_flash_attention` |
| 文件 | [sfa_v1.py:1058-1082](vllm_ascend/attention/sfa_v1.py#L1058-L1082) |
| Query | `ql_nope: (num_tokens, nope_head_dim)` — layout: **TND** (`layout_query="TND"`) |
| Query RoPE | `q_pe: (num_tokens, rope_head_dim)` |
| Key | `kv_cache[0]: (num_blocks, block_size, 1, kv_lora_rank)` — layout: **PA_BSND** (`layout_kv="PA_BSND"`) |
| Value | 复用 Key tensor（MLA 中 K=V 同源，`kv_cache[0]`） |
| Key RoPE | `kv_cache[1]: (num_blocks, block_size, 1, qk_rope_head_dim)` |
| 稀疏索引 | `topk_indices: (num_tokens, topk)` |
| 参数 | `sparse_block_size=1`, `sparse_mode=3`, `attention_mode=2` |
| Stride 要求 | **K、K_rope、V 必须各自独立 tensor**，物理连续 |

#### 2.3.3 MLAPO（MLA Prefetch Optimized, 可选）

| 算子 | `sfa_preprocess_with_mlapo` |
|------|----------------------------|
| 入口 | 经 `DeviceOperator.sfa_preprocess_with_mlapo` 分发 |
| 文件 | [sfa_v1.py:1110-1119](vllm_ascend/attention/sfa_v1.py#L1110-L1119) |
| 条件 | `enable_mlapo=True` 且 `num_input_tokens <= 1024` |
| 功能 | 融合 Q/KV 预处理 + KV cache 写入 |

---

### 2.4 模型类型 D: Compress MLA / DSA（AscendDSABackend）

#### 2.4.1 KV Compression Scatter（写入压缩 KV Cache）

| 算子 | `npu_scatter_nd_update_v2` |
|------|--------------------------|
| 入口 | `torch.ops._C_ascend.npu_scatter_nd_update_v2` |
| 文件 | [device_op.py:344](vllm_ascend/device/device_op.py#L344) |
| 功能 | 将压缩后的 KV 写入 cache（通用 scatter，非 attention 专用） |
| Stride 要求 | Cache 连续 |

#### 2.4.2 Sparse Attention Metadata Builder

| 算子 | `npu_sparse_attn_sharedkv_metadata` |
|------|-----------------------------------|
| 入口 | `torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata` |
| 文件 | [dsa_v1.py:746-808](vllm_ascend/attention/dsa_v1.py#L746-L808) |
| 功能 | 构建 DSA Sparse Attention 的元数据（根据 `cmp_ratio` 不同参数不同） |
| 变体 | `cmp_ratio=1` (无压缩 SWA), `cmp_ratio=4` (4x 压缩), `cmp_ratio=128` (128x 压缩) |
| Layout | `layout_q="TND"`, `layout_kv="PA_ND"` |
| Stride 要求 | 输入 tensor 连续 |

#### 2.4.3 Sparse Attention (Shared KV)

| 算子 | `npu_sparse_attn_sharedkv` |
|------|---------------------------|
| 入口 | `torch.ops._C_ascend.npu_sparse_attn_sharedkv` |
| 文件 | [dsa_v1.py:2093-2137](vllm_ascend/attention/dsa_v1.py#L2093-L2137) (prefill), [dsa_v1.py:2366-2423](vllm_ascend/attention/dsa_v1.py#L2366-L2423) (decode) |
| Query | `q: (num_tokens, num_heads, head_dim)` — layout: **TND** (`layout_q="TND"`) |
| Ori KV (SWA) | `swa_kv_cache: (num_blocks, 128, 1, head_dim)` — layout: **PA_ND** (`layout_kv="PA_ND"`) |
| Cmp KV (Compressed) | `compress_kv_cache: (num_compressed_blocks, block_size, 1, head_dim)` — layout: **PA_ND** |
| Cmp Sparse Indices | `compress_topk_idxs` (topk 索引) |
| 参数 | `cmp_ratio`, `ori_mask_mode=4`, `cmp_mask_mode=3`, `ori_win_left/right` |
| Stride 要求 | SWA KV cache 和 Compress KV cache 各自独立连续 |

#### 2.4.4 Quantized Lightning Indexer（压缩后 Top-K 选择）

| 算子 | `npu_quant_lightning_indexer` |
|------|------------------------------|
| 入口 | `torch.ops._C_ascend.npu_quant_lightning_indexer` |
| 文件 | [dsa_v1.py:2064-2084](vllm_ascend/attention/dsa_v1.py#L2064-L2084) (prefill), [dsa_v1.py:2337-2357](vllm_ascend/attention/dsa_v1.py#L2337-L2357) (decode) |
| Query | int8 quantized `q_quant` + `q_scale` float16 |
| Key | `indexer_k_cache` (int8) |
| Scale | `indexer_scale_cache` (float16/fp32) |
| Layout | `layout_query="TND"`, `layout_key="PA_BSND"` |
| 参数 | `sparse_count=512`, `cmp_ratio=4` |
| Stride 要求 | Indexer_K 和 Indexer_Scale 各自独立连续 |

#### 2.4.5 QLI Metadata Builder

| 算子 | `npu_quant_lightning_indexer_metadata` |
|------|---------------------------------------|
| 入口 | `torch.ops._C_ascend.npu_quant_lightning_indexer_metadata` |
| 文件 | [dsa_v1.py:810](vllm_ascend/attention/dsa_v1.py#L810) |
| Layout | `layout_query="TND"`, `layout_key="PA_BSND"` |

#### 2.4.6 KV Cache Load / Compression

| 算子 | `dsa_kv_compress_scatter` |
|------|--------------------------|
| 入口 | 经 `DeviceOperator.dsa_kv_compress_scatter` 分发 |
| 文件 | [device_op.py:342](vllm_ascend/device/device_op.py#L342) / [dsa_v1.py:2323](vllm_ascend/attention/dsa_v1.py#L2323) |
| 功能 | 将压缩后的 KV 写入 Compress KV Cache |

#### 2.4.7 Dynamic Quant

| 算子 | `npu_dynamic_quant` / `npu_dynamic_mx_quant` |
|------|---------------------------------------------|
| 入口 | `torch_npu.npu_dynamic_quant` |
| 文件 | [device_op.py:111](vllm_ascend/device/device_op.py#L111) |
| 用途 | W8A8 量化下的动态量化 |
| 变体 | int8 (A3), float8_e4m3fn (A5) |

---

### 2.5 模型类型 E: Hybrid（Qwen3.5 — Mamba + GQA Attention）

#### 2.5.1 GDN (Gated DeltaNet) Attention（Mamba 部分）

| 算子 | `AscendGatedDeltaNetAttention` (Triton kernel) |
|------|-----------------------------------------------|
| 文件 | [patch_qwen3_5.py](vllm_ascend/patch/worker/patch_qwen3_5.py) |
| 功能 | Linear attention（Mamba），非 paged attention |
| Stride 要求 | 单 tensor 连续 |

#### 2.5.2 GQA Attention（注意力部分）

| 算子 | 复用 2.1 的所有算子（GQA 路径） |
|------|------------------------------|
| Layout | `SplitKVLayout` — K/V 分离 |

---

## 3. Layout 描述符汇总

所有 Ascend NPU attention 算子都通过 **字符串描述符** 通知 kernel 如何解析输入 tensor 的维度语义：

| 描述符 | 全称 | Shape 语义 | 使用场景 |
|--------|------|-----------|---------|
| **TND** | Tokens × Num_heads × head_Dim | `(T, N, D)` | Query 输入（prefill/decode） |
| **PA_BSND** | Block_num × block_Size × Num_kv_heads × head_Dim | `(B, S, N, D)` | KV Cache 按 block 分页，SFA Indexer Key |
| **PA_ND** | Block_num × Num_kv_heads × head_Dim | `(B, N, D)` | DSA Sparse Attention 的 KV Cache（没有显式 block_size 维度） |
| **BSND** | Batch × Seq × Num_heads × head_Dim | `(B, S, N, D)` | 连续序列（非分页场景） |
| (implicit) | 无显式描述符 | 由算子签名隐式决定 | `npu_scatter_pa_kv_cache`、`npu_paged_attention` 等 |

### 3.1 描述符维度槽位映射

```
PA_BSND:  (num_blocks,  block_size,  num_kv_heads,  head_dim)
              │             │             │             │
             "B"           "S"           "N"           "D"
           block_num    block_Size    k_heads      head_Dim

PA_ND:    (num_blocks,  num_kv_heads,  head_dim)
              │             │             │
             "B"           "N"           "D"
           block_num     k_heads      head_Dim

TND:      (num_tokens,  num_heads,  head_dim)
              │             │           │
             "T"           "N"         "D"
           tokens      heads       head_Dim
```

### 3.2 为什么 Ascend 必须保留 `num_kv_heads=1`

即使 MLA 的 KV heads 永远为 1，Ascend 也不能像 CUDA 那样省略它，因为 **PA_BSND 是固定 4 槽位的描述符**。如果省略：

```
shape = (num_blocks, block_size, 512)
按 PA_BSND: B=num_blocks, S=block_size, N=512, D=???
→ head_dim 被误解析为 num_kv_heads，维度全错位
```

CUDA kernel 的维度语义通过 C++ 编译时常量硬编码，不需要运行时描述符。

---

## 4. Stride / 连续性要求矩阵

| 算子 | K Cache | V Cache | Indexer K | Indexer Scale | Compress KV | Query | 特殊要求 |
|------|---------|---------|-----------|---------------|-------------|-------|---------|
| `npu_scatter_pa_kv_cache` | 连续 ✅ | 连续 ✅ | N/A | N/A | N/A | 连续 ✅ | K/V 物理分离 |
| `npu_fused_infer_attention_score` | 连续 ✅ | 连续 ✅ | N/A | N/A | N/A | 连续 ✅ | `actual_seq_lengths` 最后元素=num_tokens |
| `_npu_paged_attention` | 连续 ✅ | 连续 ✅ | N/A | N/A | N/A | 连续 ✅ | |
| `npu_mla_prolog_v2/v3` | 连续 ✅ | 连续 ✅ | N/A | N/A | N/A | 连续 ✅ | 输入 num_tokens ≤ 1024 |
| `npu_sparse_flash_attention` | 连续 ✅ | 连续 ✅ (同K) | N/A | N/A | N/A | 连续 ✅ | K 和 K_rope 独立 tensor |
| `npu_lightning_indexer` | N/A | N/A | 连续 ✅ | N/A (bf16) | N/A | 连续 ✅ | |
| `npu_lightning_indexer_quant` | N/A | N/A | 连续 ✅ | 连续 ✅ | N/A | 连续 ✅ | Scale 必须独立 |
| `npu_quant_lightning_indexer` | N/A | N/A | 连续 ✅ | 连续 ✅ | N/A | 连续 ✅ | DSA C4/C128 |
| `npu_sparse_attn_sharedkv` | N/A | N/A | N/A | N/A | 连续 ✅ | 连续 ✅ | SWA + Cmp KV 各自独立 |
| `npu_sparse_attn_sharedkv_metadata` | N/A | N/A | N/A | N/A | N/A | N/A | 仅元数据 |
| `npu_scatter_nd_update_v2` | N/A | N/A | 连续 ✅ | 连续 ✅ | 连续 ✅ | N/A | 通用 scatter |
| `npu_gather_pa_kv_cache_vllm` | 连续 ✅ | 连续 ✅ | N/A | N/A | N/A | N/A | 连续读取 |

### 4.1 连续性问题的影响

```
                    上游 CUDA                           Ascend NPU
                   ──────────                         ──────────
K/V 交织:     (2, N, B, H, D) 单 tensor           ✗ 必须分离为 2 tensor
Stride 访问:  kernel 内部指针运算随意 cross       ✗ kernel 要求连续内存
Overlay:      as_strided view 可直接传 kernel     ✗ 需要"固化"为连续 tensor
Padding:      page_size_padded → as_strided      与上游一样（block 间 padding）
```

---

## 5. 按模型类型的依赖拓扑图

### 5.1 GQA（Qwen3 MoE）

```
hidden_states
    │
    ├─→ [QKV Projection] ──→ q, k, v
    │                            │
    │                    ┌───────┴────────┐
    │                    │                │
    │              [npu_scatter_pa    [npu_fused_infer
    │               _kv_cache]         _attention_score]  ← 算子 2.1.2
    │               ← 算子 2.1.1         │ layout: TND
    │                    │              │
    │                    ▼              ▼
    │              KV Cache (K+V)    attention_output
    │                                     │
    │                              [o_proj] → 下一层
```

### 5.2 Sparse MLA / SFA（DeepSeek V3.2 / GLM5.1）

```
hidden_states
    │
    ├─→ [Q/KV Low-rank Proj] ──→ q_lora, kv_lora
    │         │
    │         ├─→ [RoPE] ──→ q_pe, k_pe
    │         │                    │
    │         │            [npu_scatter_pa_kv_cache]
    │         │            ← 写入 k_cache(v_lora), k_rope
    │         │                    │
    │         ├─→ [Indexer] ──→ q_li ──→ [npu_lightning_indexer]  ← 算子 2.3.1
    │         │                         │  layout_key="PA_BSND"
    │         │                         │  ← 读取 indexer_k_cache
    │         │                         ▼
    │         │                    topk_indices
    │         │                         │
    │         └─→ ql_nope ──→ [npu_sparse_flash_attention]  ← 算子 2.3.2
    │                           │ layout_kv="PA_BSND"
    │                           │ q_pe + k_rope (RoPE 在 kernel 内完成)
    │                           │ ← 读取 kv_cache[0] (K), kv_cache[1] (K_rope)
    │                           ▼
    │                      attention_output
    │                           │
    └───────────────────── [o_proj] → 下一层
```

### 5.3 Compress MLA / DSA（DeepSeek V4）

```
hidden_states
    │
    ├─→ [CV Linear (Cube+Vector)]
    │         │
    │         ├─→ q_quant (Main Stream)       kv_quant (Aux Stream)
    │         │         │                          │
    │         │         ▼                          ▼
    │         │   [q_a_down]                  [kv_matmul]
    │         │         │                          │
    │         │         ▼                          │
    │         │   [q_norm + q_b_quant]        [kv_norm + RoPE]
    │         │         │                          │
    │         │         ▼                          ▼
    │         │   [q_b_matmul]               [npu_scatter_nd_update_v2]
    │         │         │                     ← 算子 2.4.1 (scatter 到 swa_kv_cache)
    │         │         │                          │
    │         │         │                    [compressor] → 压缩 KV
    │         │         │                          │
    │         │         │               [npu_scatter_nd_update_v2]
    │         │         │               ← 算子 2.4.6 (scatter 到 compress_kv_cache)
    │         │         │                          │
    │         │         ├─→ [Indexer_q] ──→ [npu_quant_lightning_indexer] ← 算子 2.4.4
    │         │         │                         │ layout_key="PA_BSND"
    │         │         │                         ▼
    │         │         │                    compress_topk_idxs
    │         │         │                         │
    │         │         └─→ q ──→ [npu_sparse_attn_sharedkv] ← 算子 2.4.3
    │         │                           │ layout_q="TND"
    │         │                           │ layout_kv="PA_ND"
    │         │                           │ ← ori_kv (SWA), cmp_kv (Compressed)
    │         │                           │ ← metadata (算子 2.4.2)
    │         │                           ▼
    │         │                      attention_output
    │         │                           │
    └─────────┴─────────────────── [o_proj] → 下一层
```

---

## 6. 与算子侧对齐的关键问题

### 6.1 需要确认的问题清单

| # | 问题 | 涉及算子 | 优先级 |
|---|------|---------|--------|
| 1 | **Stride 支持**: 是否能支持非连续 tensor（`as_strided` views）作为 KV Cache 输入？ | 全部 attention 算子 | 🔴 高 |
| 2 | **K/V 交织**: 未来是否支持类似 CUDA `(2,N,B,H,D)` 的单 tensor 布局，内部通过 stride 切分 K/V？ | `npu_fused_infer_attention_score`<br>`_npu_paged_attention` | 🔴 高 |
| 3 | **PA_BSND 槽位灵活性**: 槽位含义是否固定（B,S,N,D）？能否支持省略 `N` 维度（num_kv_heads=1 时）？ | 全部 PA_BSND 算子 | 🟡 中 |
| 4 | **block_size 支持范围**: DSA 支持 `[2,4,8,16,32,64,128]`，其他算子是否也能支持更小 block_size？ | `npu_fused_infer_attention_score` 等 | 🟡 中 |
| 5 | **as_strided overlay**: 多 view 共享同一物理 buffer（`storage_offset` 不同），kernel 是否接受？ | 全部 KV Cache 写入/读取算子 | 🟡 中 |
| 6 | **page_size_padded**: block 间 padding（stride 放大），kernel 是否接受？ | `npu_fused_infer_attention_score`<br>`_npu_paged_attention` | 🟢 低 |
| 7 | **Dynamic Quant 精度**: int8 (A3) vs float8_e4m3fn (A5)，量化行为是否一致？ | `npu_dynamic_quant` | 🟢 低 |

### 6.2 为什么 stride 支持是关键瓶颈

当前架构的核心限制（也是 Phase 2-3 重构要解决的问题）：

```
上游 vLLM CUDA:
  KVCacheManager 分配 (2, N, B, H, D) 一个大 tensor
  → model_runner.view().view() 一步完成 reshape
  → CUDA kernel 内部用 offset 切 K/V → 零额外开销

vLLM-Ascend NPU (现状):
  model_runner 必须自己 split total_bytes 为 2~4 个独立 tensor
  → 每个 tensor 独立 allocate + reshape
  → 600 行 if-else 分支
  → 原因: NPU kernel 不接受 stride/non-contiguous

vLLM-Ascend NPU (如果算子支持 stride):
  KVCacheManager 分配 1 个大 tensor（和上游一样）
  → Layout.reshape 用 as_strided 做 K/V view（零拷贝）
  → NPU kernel 接受 as_strided views → model_runner 回归上游简洁模式
```

---

## 7. 附加信息

### 7.1 量化变体

| 量化类型 | 设备 | K/V dtype | Indexer dtype | Scale dtype | 额外字段 |
|---------|------|-----------|---------------|-------------|---------|
| bf16 (无量化) | A3 | bf16 | bf16 | N/A | N/A |
| C8 int8 | A3 | bf16 | int8 | float16 | `cache_sparse_c8=True` |
| C8 fp8 (CKV merged) | A5 | float8_e4m3fn (CKV合并) | float8_e4m3fn | float32 | `qk_rope_head_dim=0` (合并标志) |
| W8A8 Dynamic | A3/A5 | bf16 | int8/fp8 | float16/fp32 | 仅 QKV 线性层量化 |

### 7.2 相关文件索引

| 文件 | 角色 |
|------|------|
| [device_op.py](vllm_ascend/device/device_op.py) | 算子入口适配层（BaseDeviceAdaptor + A5DeviceAdaptor） |
| [attention_v1.py](vllm_ascend/attention/attention_v1.py) | GQA Backend（AscendAttentionBackend） |
| [mla_v1.py](vllm_ascend/attention/mla_v1.py) | 标准 MLA Backend（AscendMLABackend） |
| [sfa_v1.py](vllm_ascend/attention/sfa_v1.py) | Sparse MLA Backend（AscendSFABackend） |
| [dsa_v1.py](vllm_ascend/attention/dsa_v1.py) | Compress MLA Backend（AscendDSABackend） |
| [kv_cache_layout.py](vllm_ascend/core/kv_cache_layout.py) | Layout 抽象层（Phase 2-3 — 封装 KV Cache 物理布局） |
| [model_runner_v1.py](vllm_ascend/worker/model_runner_v1.py) | allocate/reshape 入口（Phase 2-3: V2 路径 + feature gate） |
| [patch_kv_cache_interface.py](vllm_ascend/patch/platform/patch_kv_cache_interface.py) | AscendMLAAttentionSpec + get_kv_cache_layout() monkey-patch |
| `aclnnSparseFlashAttention.md` (csrc) | SFA 算子官方文档 |

### 7.3 上游 CUDA 等价

| Ascend NPU 算子 | 上游 CUDA 等价 | 差异 |
|----------------|---------------|------|
| `npu_sparse_flash_attention` | `flash_mla_sparse_fwd` | CUDA 单 tensor，内部 offset 拆分 K=V |
| `npu_fused_infer_attention_score` | `flash_attn_varlen_func` | CUDA 支持单 `(2,N,B,H,D)` tensor |
| `npu_lightning_indexer` | (模型内部的 indexer 模块) | CUDA 在模型 forward 里完成，不涉及 KV cache |
| `npu_quant_lightning_indexer` | (同上) | CUDA 在模型 forward 里完成 |
| `npu_sparse_attn_sharedkv` | (DSA 专有，无 CUDA 等价) | Ascend 自定义 DSA kernel |
