# DCP / PCP（Context Parallelism）学习指南



---

## 一、概述

### 1.1 什么是 Context Parallel (CP)？

**Context Parallel (CP)** 是一种沿 **序列维度（sequence dimension）** 将计算并行分布到多个设备上的策略。

随着 LLM 支持的上下文长度不断增长（128K、256K 甚至更长），单张卡显存无法容纳整个序列的 KV Cache。为了解决这个问题，vllm-ascend 引入了两种互补的 CP 模式：

| 模式 | 全称 | 核心目标 |
|------|------|---------|
| **DCP** | Decode Context Parallelism（解码上下文并行） | 消除 KV Cache 重复存储 |
| **PCP** | Prefill Context Parallelism（预填充上下文并行） | 加速 Prefill 阶段计算 |

两者可以**组合使用**，形成 **DCP + PCP 双级交织**。

### 1.2 为什么需要 CP？

```
单卡场景（无 CP）:
┌─────────────┐
│   GPU 0     │
│             │
│  KV Cache:  │
│  [T0...T7]  │  ← 完整序列，单卡存储
│  QKV 计算    │
└─────────────┘

多卡 + DCP 场景:
┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │
│  KV Cache:  │  │  KV Cache:  │
│  [T0,T2,T4] │  │  [T1,T3,T5] │  ← 序列分片，消除冗余
└─────────────┘  └─────────────┘

多卡 + PCP 场景:
┌─────────────┐  ┌─────────────┐
│   GPU 0     │  │   GPU 1     │
│  Prefill:   │  │  Prefill:   │
│  前一半序列  │  │  后一半序列  │  ← 序列切分，并行计算
│  KV 分片存储 │  │  KV 分片存储 │
└─────────────┘  └─────────────┘
```

---

## 二、DCP — Decode Context Parallelism 详解

### 2.1 核心目标

DCP 的**唯一目标**：消除 KV Cache 的冗余存储。

在标准的 Tensor Parallelism (TP) 中，每个 TP rank 都持有**完整**的 KV Cache 副本，导致显存浪费。DCP **复用 TP 的通信域**，将 KV Cache 沿序列维度分片存储，使 TP 域内各设备不再持有冗余副本。

> **关键点**：DCP **不需要额外设备**，它复用 TP 通信域。

### 2.2 设备拓扑

```
DCP 复用 TP 通信域的示例（DCP2 + TP4）：

   TP Group 0          TP Group 1
┌──────┬──────┐    ┌──────┬──────┐
│GPU 0 │GPU 1 │    │GPU 2 │GPU 3 │
│DCP 0 │DCP 1 │    │DCP 0 │DCP 1 │
└──────┴──────┘    └──────┴──────┘

DCP size = 2：在同一个 TP group 内，两张卡分别存储序列的不同分片
TP size = 4：四张卡分别持有不同的模型参数分片
```

### 2.3 KV Cache 分片存储

DCP 下 KV Cache 以 **token-interleave（令牌交织）** 方式存储：

```
假设 dcp_world_size=2，cp_kv_cache_interleave_size=1

Token 序列：[T0, T1, T2, T3, T4, T5, T6, T7]
              │   │   │   │   │   │   │   │
Rank 0:      [T0,     T2,     T4,     T6    ]  ← 偶数位置 token 的 KV cache
Rank 1:      [    T1,     T3,     T5,     T7]  ← 奇数位置 token 的 KV cache

每个 rank 只存储自己负责的 token 的 KV cache
其他 token 位置上填 PAD_SLOT_ID（-1）
```

### 2.4 对 block_size 的影响

DCP 会**扩展**虚拟的 `block_size`，使调度器以更大的粒度分配 block：

```python
# UnitaryKVCacheCoordinator.__init__()
if dcp_world_size > 1:
    self.block_size *= dcp_world_size  # block_size 扩展
if pcp_world_size > 1:
    self.block_size *= pcp_world_size

# 示例：原始 block_size=16，dcp_size=2，pcp_size=1
# → 实际 block_size = 16 * 2 * 1 = 32
```

### 2.5 对 slot_mapping 的影响

`slot_mapping` 计算时需要判断每个 token 属于哪个 CP rank：

```python
# block_table.py - compute_slot_mapping
total_cp_world_size = self.pcp_world_size * self.dcp_world_size
total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

virtual_block_off = position % (block_size * total_cp_world_size)
interleave_idx = virtual_block_off // cp_kv_cache_interleave_size

if interleave_idx % total_cp_world_size != total_cp_rank:
    slot_mapping[token_i] = PAD_SLOT_ID  # -1，非本地 token
```

### 2.6 DCP 下的 Prefill 阶段

DCP 对 **Chunked Prefill** 的计算逻辑有显著影响，GQA 和 MLA 后端采用不同策略：

#### GQA 后端 — AllGather Q 方案

```
┌─────────────────────────────────────┐
│  1. all_gather Q 沿 head 维度       │
│     (因为 DCP 组内 Q heads 不同)     │
│  2. 用本地 KV cache 计算 attention   │
│  3. cp_lse_ag_out_rs:                │
│     聚合 attn_output 和 attn_lse    │
│     在线更新 softmax 结果            │
│     reduce-scatter 输出              │
└─────────────────────────────────────┘
```

#### MLA 后端 — AllGather KV 方案

```
┌─────────────────────────────────────┐
│  1. all_gather Context KV cache     │
│     （聚合完整的 KV 值）             │
│  2. reorg_kvcache:                  │
│     重排 KV cache，确保同一请求的     │
│     KV cache 连续存储               │
│  3. 用完整 KV + 当前 chunk Q 计算    │
└─────────────────────────────────────┘
```

### 2.7 DCP 下的 Decode 阶段

Decode 阶段的逻辑与 GQA 的 Chunked Prefill 一致：

1. 沿 Q 的 head 维度执行 **all-gather**，确保 DCP 组内一致
2. 用本地 KV cache 计算结果
3. 通过 `cp_lse_ag_out_rs` 更新结果

---

## 三、PCP — Prefill Context Parallelism 详解

### 3.1 核心目标

PCP 的**核心目标**是加速 **Prefill 阶段**的计算。通过将长序列切分到多个设备上并行计算 attention，大幅降低 Prefill 延迟。

> **关键点**：PCP 使用**专用通信域**（dedicated communication domains），需要**额外设备**（扩展 world size）。

### 3.2 设备拓扑

```
PCP 使用专用通信域的示例（PCP2 + DCP2 + TP4）：

      PCP Group 0              PCP Group 1
┌──────┬──────┬──────┐  ┌──────┬──────┬──────┐
│GPU 0 │GPU 1 │ ...  │  │GPU 4 │GPU 5 │ ...  │
└──────┴──────┴──────┘  └──────┴──────┴──────┘

PCP size = 2：序列切分到两个 PCP group 上并行 prefill
DCP size = 2：每个 PCP group 内复用 TP 通信域做 KV cache 分片
TP size  = 4：模型参数在 4 张卡上分片
总卡数 = pcp_size × dcp_size × tp_size 张卡
```

> 上图简化了排布，实际 `PCP2, DCP2, TP4` 共需要 `2×2×4 = 16` 张卡。

### 3.3 Head-Tail 序列切分策略



PCP 使用 **Head-Tail 风格**的序列切分，以平衡各设备的计算负载：

```
原始序列（已 pad）：[T0, T1, T2, T3, T4, T5, T6, T7]
                      ↓ 分成 2*pcp_size = 4 等份
Chunk 0: [T0, T1]     ← head
Chunk 1: [T2, T3]
Chunk 2: [T4, T5]
Chunk 3: [T6, T7]     ← tail
                      ↓ 交错合并
PCP Rank 0: Chunk 0 + Chunk 3 = [T0, T1, T6, T7]  ← "头尾配对"
PCP Rank 1: Chunk 1 + Chunk 2 = [T2, T3, T4, T5]

这样每个 rank 的 tokens 来自序列两端而非连续区间，
虽然损失了 locality，但换来了负载均衡。
```

**为什么要用 Head-Tail？**

- 计算 Attention 时，序列前半部和后半部的 token 具有不同数量的 visible context
- 如果简单均分（rank0 处理 T0-T3，rank1 处理 T4-T7），rank0 的 token 看到更少的上下文，计算量更小
- Head-Tail 配对可以使每个 rank 的计算量大致均衡

### 3.4 `update_tokens_for_pcp` 核心逻辑

该函数是 PCP 切分的核心入口，位于 [pcp_utils.py](../../vllm_ascend/worker/pcp_utils.py) 的 `PCPManager.update_tokens_for_pcp()`。

以 `tokens=[1, 5, 8]`，`pcp_world_size=2` 为例：

```
步骤 1: Pad 到 2×pcp_size 的倍数
  num_padded = ceil(tokens / 4) × 4
  [1,5,8] → [4, 8, 8]  （注意 decode 不 pad，这里简化）

步骤 2: 计算每个 rank 应处理的 token 数
  pcp_tokens = num_padded // pcp_world_size
  Rank 0: [2, 4, 4]
  Rank 1: [2, 4, 4]

步骤 3: Head-Tail 切分
  chunk_sizes = pcp_tokens // 2
  prefill: 每 req 分 head 和 tail 两部分
  decode:  不分 chunk，直接复制到所有 rank

步骤 4: 计算 position
  Rank 0 的 positions: [0, 0, 1, 6, 7, 0, 1, 6, 7]
  Rank 1 的 positions: [0, 2, 3, 4, 5, 2, 3, 4, 5]

步骤 5: 生成 allgather_restore_idx
  用于 allgather 后恢复原始顺序
```

**Decode 请求的特殊处理**：

PCP 不切分 decode 请求。对于 decode 请求，会将 tokens **复制**到所有 PCP rank（而非切分）。这意味着每个 PCP rank 都处理完整的 decode tokens。

```python
# Decode 请求不 pad，而是复制
num_padded_scheduled_tokens[:self.num_decode_reqs] = (
    num_scheduled_tokens[:self.num_decode_reqs] * self.pcp_world_size
)
```

### 3.5 PCP 下的 Prefill 阶段

在 PCP 的纯 Prefill 阶段（不含 Chunked Prefill）：

```
┌──────────────────────────────────────────────┐
│  GPU 0（Rank 0）         GPU 1（Rank 1）      │
│  ┌─────────────┐         ┌─────────────┐     │
│  │ 本地 QKV 计算 │         │ 本地 QKV 计算 │     │
│  │ (仅部分序列)  │         │ (仅部分序列)  │     │
│  └──────┬──────┘         └──────┬──────┘     │
│         │ all_gather KV          │            │
│         │◄──────────────────────►│            │
│         ▼                        ▼            │
│  ┌─────────────┐         ┌─────────────┐     │
│  │ 完整 KV 做   │         │ 完整 KV 做   │     │
│  │ Attention    │         │ Attention    │     │
│  └─────────────┘         └─────────────┘     │
│                                              │
│  ⚠️ 只聚合当前层的 KV，用完即丢弃              │
│     避免峰值显存过高                          │
└──────────────────────────────────────────────┘
```

> **为什么不使用 Ring Attention？**
>
> 开发团队评估后认为 Ring Attention 虽然峰值显存更低且能实现 compute-communication overlap，但开发复杂度高，overlap 带来的收益有限。当前选择 all-gather KV 实现。

### 3.6 PCP 下的 Decode 阶段

Decode 阶段相对简单：

1. DCP 的 **all-to-all 通信**交换 output 和 LSE
2. 在 PCP 组内进行一次 **allgather**
3. 进行输出更新（output update）

```
┌────────────────────────────────────────┐
│ DCP all-to-all（交换 output + LSE）     │
│         ↓                              │
│ PCP allgather（组内聚合）               │
│         ↓                              │
│ Output Update（更新最终输出）            │
└────────────────────────────────────────┘
```

### 3.7 PCP 下的 Chunked Prefill

Chunked Prefill 有三种可选方案：

| 方案 | 描述 | 优缺点 |
|------|------|--------|
| **AllGatherQ** | 聚合 Q，保持 KV 分片 | GQA 后端使用，与 decode 逻辑一致 |
| **AllGatherKV** | 聚合 KV，保持 Q 不变 | MLA 后端使用，与 prefill 逻辑一致；长序列时峰值显存可能较高 |
| **Ring-Attn** | 环形传递做增量计算 | 峰值显存低，但实现复杂；未采用 |

vllm-ascend 的实现：
- **GQA 后端** → AllGatherQ 方案
- **MLA 后端** → AllGatherKV 方案（分段处理，控制峰值显存）

---

## 四、DCP + PCP 双级交织

### 4.1 层级关系

```python
# cp_size 和 cp_rank 的计算
cp_size = pcp_size * dcp_size
cp_rank = pcp_rank * dcp_size + dcp_rank
```

```
示例：PCP2 × DCP2

cp_rank 排布：
        DCP 0    DCP 1
PCP 0    0        1
PCP 1    2        3

cp_size = 2 × 2 = 4
```

### 4.2 交织存储格式

在双级交织下，token 的 KV cache 按照 `cp_kv_cache_interleave_size` 粒度在 cp_rank 之间交错存储：

```
假设 dcp_size=2, pcp_size=2, interleave_size=1

Token 序列：[T0, T1, T2, T3, T4, T5, T6, T7]

cp_rank=0 (PCP0, DCP0): [T0,         T4        ]
cp_rank=1 (PCP0, DCP1): [    T1,         T5    ]
cp_rank=2 (PCP1, DCP0): [        T2,         T6]
cp_rank=3 (PCP1, DCP1): [            T3,         T7]

每个 cp_rank 只存储 interleave_size 粒度的 KV cache 分片
```

### 4.3 Virtual Block 概念

为了统一管理跨 CP rank 的 KV cache 分片，引入了 **Virtual Block** 概念：

```
┌──────────────────────────────────────────┐
│           Virtual Block                   │
│  ┌──────────┬──────────┬──────────┬─────┐ │
│  │CP Rank 0 │CP Rank 1 │CP Rank 2 │ ... │ │
│  │(物理块A) │(物理块B) │(物理块C) │     │ │
│  └──────────┴──────────┴──────────┴─────┘ │
│                                           │
│  virtual_block_size = block_size × cp_size │
└──────────────────────────────────────────┘
```

**Token 到 KV Cache 位置的映射公式**：

```python
# 对于 token x:
virtual_block_idx = x // virtual_block_size                          # 虚拟 block 编号
offset_in_virtual = x % virtual_block_size                           # 虚拟 block 内偏移
local_block_idx = offset_in_virtual // cp_kv_cache_interleave_size   # 本地 block 编号
target_rank = local_block_idx % cp_size                              # 目标设备
offset_in_local_block = (local_block_idx // cp_size) * interleave_size \
                        + offset_in_virtual % interleave_size        # 本地 block 内偏移
```

### 4.4 约束条件

```python
# block_size 必须能被 interleave_size 整除
assert block_size % cp_kv_cache_interleave_size == 0

# HybridKVCacheCoordinator 当前不支持 DCP/PCP
assert dcp_world_size == 1  # 如果使用 HybridKVCacheCoordinator

# 当前 hybrid block 与 CP 互斥
if use_hybrid_blocks:
    assert pcp_world_size == 1 and dcp_world_size == 1
```

---

## 五、关键代码路径

### 5.1 Attention 接口切换

启用 CP 后，Attention 后端会自动切换到 CP 专用实现：

```python
# AscendAttentionBackend
@staticmethod
def get_impl_cls():
    if enable_cp():
        return AscendAttentionCPImpl       # CP 专用实现
    return AscendAttentionBackendImpl      # 标准实现

@staticmethod
def get_builder_cls():
    if enable_cp():
        return AscendAttentionCPMetadataBuilder  # CP 专用 metadata builder
    return AscendAttentionMetadataBuilder
```

### 5.2 完整处理流程

```
SchedulerOutput
      │
      ▼
┌─────────────────────────────────┐
│ ① CPU 端计算 pre-PCP slot_mapping│  ← block_table.py
│   (使用 PCP split 之前的位置)      │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ ② update_tokens_for_pcp()       │  ← pcp_utils.py
│   - Head-Tail 切分              │
│   - 更新 num_scheduled_tokens    │
│   - 生成 positions_pcp          │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ ③ 重新计算 slot_mapping           │  ← block_table.py
│   (使用 PCP split 之后的位置)      │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ ④ Attention 前向计算              │  ← attention_cp.py / mla_cp.py
│   - AllGather Q / AllGather KV  │
│   - Online Softmax 更新          │
│   - Reduce-Scatter 输出          │
└──────────────┬──────────────────┘
               ▼
┌─────────────────────────────────┐
│ ⑤ AllGather restore + Unpad     │  ← pcp_utils.py
│   恢复原始序列顺序                │
└─────────────────────────────────┘
```

---

## 六、关键数据结构

### 6.1 AscendMetadata（Attention Metadata）

```python
@dataclass
class AscendMetadata:
    # 基础属性
    attn_state: AscendAttentionState  # PrefillNoCache / PrefillCacheHit /
                                      # DecodeOnly / ChunkedPrefill / SpecDecoding
    num_actual_tokens: int            # 实际 token 数（不含 padding）
    num_decode_tokens: int            # decode token 数
    num_prefills: int                 # prefill 请求数
    num_decodes: int                  # decode 请求数

    # 序列信息
    seq_lens: torch.Tensor            # [num_reqs] 每个请求的序列长度
    query_start_loc: torch.Tensor     # [num_reqs+1] 累计 token 位置

    # KV Cache 相关
    block_tables: torch.Tensor        # [num_reqs, max_blocks_per_req]
    slot_mapping: torch.Tensor        # [num_tokens]

    # CP 相关
    prefill: AscendMetadataForPrefill | None
    decode_meta: AscendMetadataForDecode | None
```

### 6.2 PCPManager 核心缓冲区

```python
class PCPManager:
    # 恢复索引：allgather 后恢复原始顺序
    pcp_allgather_restore_idx: CpuGpuBuffer   # [max_tokens]

    # Unpad mask：标记 padded buffer 中哪些是真实 token
    pcp_unpad_mask_cpu: np.ndarray            # [max_tokens]

    # 每个请求的 pad 数量
    num_pcp_pads_cpu: np.ndarray              # [max_num_reqs]

    # 每个请求在当前 rank 上的 token 数
    pcp_tokens: np.ndarray                    # [max_num_reqs]
```

---

## 七、相关源码文件索引

| 文件 | 功能 |
|------|------|
| [context_parallel.md](context_parallel.md) | CP 设计文档（英文） |
| [vllm_ascend/worker/block_table.py](../../vllm_ascend/worker/block_table.py) | `slot_mapping` 计算（含 CP offset 逻辑） |
| [vllm_ascend/worker/pcp_utils.py](../../vllm_ascend/worker/pcp_utils.py) | PCPManager：Head-Tail 切分、allgather 恢复 |
| [vllm_ascend/worker/model_runner_v1.py](../../vllm_ascend/worker/model_runner_v1.py) | ModelRunner：调度 CP 相关预处理 |
| [vllm_ascend/attention/context_parallel/attention_cp.py](../../vllm_ascend/attention/context_parallel/attention_cp.py) | GQA 后端的 CP attention 实现 |
| [vllm_ascend/attention/context_parallel/mla_cp.py](../../vllm_ascend/attention/context_parallel/mla_cp.py) | MLA 后端的 CP attention 实现 |
| [vllm_ascend/attention/context_parallel/common_cp.py](../../vllm_ascend/attention/context_parallel/common_cp.py) | CP 公共工具（metadata、LSE 更新等） |
| [tests/ut/worker/test_pcp_manager.py](../../tests/ut/worker/test_pcp_manager.py) | PCPManager 单元测试 |
| [tests/ut/worker/a2/test_block_table.py](../../tests/ut/worker/a2/test_block_table.py) | BlockTable CP 单元测试 |
| [tests/ut/attention/a2/test_attention_cp.py](../../tests/ut/attention/a2/test_attention_cp.py) | CP Attention 单元测试 |

---

## 八、使用配置

### 8.1 启动参数

```bash
# 启用 DCP（复用 TP 通信域，无需额外设备）
--dcp-size 2

# 启用 PCP（需要额外设备）
--pcp-size 2

# 设置 CP KV cache 交织粒度（默认 1，即 token-interleave）
--cp-kv-cache-interleave-size 1
```

### 8.2 环境变量

```bash
# 启用 Context Parallel
VLLM_ASCEND_ENABLE_CP=1
```

### 8.3 约束条件

- `block_size % cp_kv_cache_interleave_size == 0`
- HybridKVCacheCoordinator 不支持 DCP/PCP（`assert dcp_world_size == 1`）
- Hybrid Blocks 与 CP 互斥（`assert pcp_world_size == 1 and dcp_world_size == 1`）
- PCP 与 Eagle3 推测解码叠加，当 prefill 的 scheduled tokens < `1 + num_speculative_tokens` 时可能出错

---

## 九、与上游 vLLM (CUDA) 的差异

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) |
|------|------------------|-------------------|
| **CP 模式** | 仅 DCP | **DCP + PCP 双级交织** |
| **交织粒度** | 固定（基础实现） | `cp_kv_cache_interleave_size` 可配置 |
| **接口切换** | 编译期/配置切换 | `get_impl_cls()` / `get_builder_cls()` 运行时动态切换 |
| **slot_mapping 处理** | 单次计算 | PCP pre-split + post-split 两阶段计算 |
| **Attn State 枚举** | DECODE / PREFILL / CHUNKED_PREFILL（3 种） | **5 种**：加 PrefillNoCache / PrefillCacheHit / SpecDecoding |
| **Chunked Prefill** | 基础 GQA/MLA 路径 | 分段处理 + AllGatherQ (GQA) / AllGatherKV (MLA) |

---

## 十、总结

```
                              Context Parallel (CP)
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
    ┌─────────▼─────────┐                       ┌───────────▼───────────┐
    │   DCP (Decode CP) │                       │   PCP (Prefill CP)    │
    ├───────────────────┤                       ├───────────────────────┤
    │ 目标: 消除冗余     │                       │ 目标: 加速 Prefill     │
    │ 通信: 复用 TP 域   │                       │ 通信: 专用通信域       │
    │ 设备: 无需额外     │                       │ 设备: 需要额外设备     │
    │ 影响: Decode+CP    │                       │ 影响: Prefill+Decode  │
    │ 策略: Token 交织   │                       │ 策略: Head-Tail 切分  │
    └───────────────────┘                       └───────────────────────┘
              │                                             │
              └──────────────────┬──────────────────────────┘
                                 │
                      cp_size = pcp_size × dcp_size
                      cp_rank = pcp_rank × dcp_size + dcp_rank
                                 │
                    ┌─────────────▼─────────────┐
                    │  DCP + PCP 双级交织存储     │
                    │  Virtual Block 统一管理     │
                    │  interleave_size 可配粒度   │
                    └───────────────────────────┘
```

**学习路径建议**：

1. 先读 [context_parallel.md](context_parallel.md) 理解设计思路
2. 阅读 [pcp_utils.py](../../vllm_ascend/worker/pcp_utils.py) 的 `update_tokens_for_pcp()` —— PCP 切分的核心
3. 阅读 [block_table.py](../../vllm_ascend/worker/block_table.py) 的 `compute_slot_mapping()` —— 理解 CP 下的 KV cache 寻址
4. 阅读 [attention_cp.py](../../vllm_ascend/attention/context_parallel/attention_cp.py) —— 理解 GQA 后端在 CP 下的 attention 计算
5. 阅读 [mla_cp.py](../../vllm_ascend/attention/context_parallel/mla_cp.py) —— 理解 MLA 后端在 CP 下的特殊处理
