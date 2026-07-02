# vLLM KV Cache 流程梳理（完善版）

> **完善说明**：在你的原始笔记基础上，我从 vllm 代码中补充了细节、修正了一些描述，并在最后一章新增了高级特性。

---

## 一、背景知识

### 什么是 KV Cache

在 Transformer 的自注意力机制中，每个 token 都需要与之前所有 token 进行注意力计算。如果每次生成新 token 都重新计算所有的 Key 和 Value 向量，计算量会非常大。KV Cache 的核心思想是缓存已计算的 Key 和 Value 向量，使得后续生成无需对已处理的 Token 重新进行矩阵运算，从而直接复用特征，达到时间换空间的效果。

### 传统 KV Cache 的问题

传统实现为每个请求预分配固定大小的连续内存：

```
┌──────────────────────────────────────────────────────────┐
│ Request A: [████████████____________]                      │
│            ← 预分配 max_seq_len    实际使用      预留但未使用 │
├──────────────────────────────────────────────────────────┤
│ Request B: [████_______________]                            │ ← 短请求浪费更多
├──────────────────────────────────────────────────────────┤
│ Request C: [████████████████████████████████]               │ ← 长请求可能超出
└──────────────────────────────────────────────────────────┘
```

**问题：**
1. **内存浪费**：预分配但未使用的内存高达 60-80%
2. **批大小受限**：因碎片化无法充分利用 GPU 内存
3. **不灵活**：无法适应变长序列

### PagedAttention

vLLM KV Cache 以 PagedAttention 为基础进行构建，分为逻辑层与物理层，该方式类似于操作系统的虚拟内存（Virtual Memory）管理。借鉴操作系统虚拟内存的分页机制，将 KV Cache 切分为固定大小的块（Block）进行管理。

PagedAttention 的核心逻辑是将 Attention 运算中的 KV 值按照虚拟映射的方式管理起来，如下图所示。图中有两个请求 Request A 和 B，它们各自拥有自己的逻辑块（Logical KV Blocks），通过对应的映射表（Block Table）找到每个词在物理块（Physical KV Blocks）中的位置。

**这种方式的优势：**
1. 能够充分利用显存，降低显存碎片化问题；
2. 减少物理显存的反复申请/释放操作，提升效率；

目前在 V1 版本中 KV Cache 管理还融合了前缀树的特点，更好地适配了 Prefix Cache 功能。整体的架构如下图所示，分为逻辑层和物理层。逻辑层由 KV Manager 管理、物理层由 Model Runner 处理；Scheduler（调度器）作为信息传递的桥梁，衔接了逻辑层与物理层。Cache 的管理元素包括：池（Pool）、表（Table）、层（Layer）、块（Block）和槽（Slot）。

- **slot**：最小管理单元，每个 token 占一个 slot；
- **block**：请求分配的基本单位，一个 block 包含多个 slot；
- **pool**：逻辑层 block 的管理集合，通过链表将 block 数据组织起来；
- **table**：管理请求与数据的映射表，一个 table 可包含多个请求的信息，位于物理层；
- **layer**：一个整体的 tensor，拆分成多个 blocks 使用，对应 attention 的一个层，所有请求共用；

---

## 二、vLLM KV Cache 流程梳理

一次推理请求从头到尾，KV Cache 经历的完整流程主要有：

1. 系统初始化与显存分配
2. 请求接入与逻辑调度
3. 实际计算（PagedAttention）
4. 显存置换、抢占和回收

### 2.0 KV Cache 数据结构

介绍 vLLM 中 KV Cache 管理涉及到的核心数据结构。

#### 2.0.1 KVCacheBlock

```python
# vllm/v1/core/kv_cache_utils.py:118-155
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int              # 块ID，0 ~ num_gpu_blocks-1
    ref_cnt: int = 0           # 引用计数（>0说明有请求在用）
    _block_hash: bytes | None  # 块哈希（前缀缓存用），存的是 BlockHashWithGroupId
    prev_free_block / next_free_block  # 空闲链表的前后指针
    is_null: bool = False      # 是否为占位空块
```

- 每个 block 承载 `block_size`（默认 16/128）个 token 的 KV cache
- `ref_cnt` 用于追踪多少请求引用了该 block（前缀缓存会共享 block）
- 空闲 block 组成双向链表（`FreeKVCacheBlockQueue`），LRU 顺序
- `_block_hash` 实际类型是 `BlockHashWithGroupId`（block hash + group id），用在多 group 的混合注意力场景

**Block 生命周期：**

```
[空闲] → get_new_blocks() → [使用中 ref_cnt=1]
       → touch() → [共享中 ref_cnt+=1]
       → free_blocks() → ref_cnt-- → (ref_cnt==0?) → [回到空闲队列]
       → _maybe_evict_cached_block() → reset_hash() → 清除缓存
```

#### 2.0.2 BlockPool

维护当前还有多少个空闲物理块，以及如何以 $O(1)$ 的极速把空物理块交出去或收回来。

```python
# vllm/v1/core/block_pool.py:131-175
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # 1. 创建所有 KVCacheBlock 对象
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # 2. 全部放入空闲队列
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # 3. 前缀缓存 HashMap
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        # 4. null_block：SWA 占位块，ref_cnt 不维护
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        # 5. KV Cache Events 事件队列
        self.kv_event_queue: list[KVCacheEvent] = []
        # 6. 可选的 metrics 收集器
        self.metrics_collector: KVCacheMetricsCollector | None = None
```

**FreeKVCacheBlockQueue**：使用双向链表进行管理，核心优化：
- 使用 fake head 和 fake tail 节点减少分支判断
- 支持 $O(1)$ 时间从队列中间移除 block（标准 deque 需要 $O(n)$）
- 初始按 block_id 排序，分配再释放后按 LRU 顺序排列：最近最少使用的在前面

**BlockHashToBlockMap**：前缀缓存的哈希映射表
- `{block_hash: KVCacheBlock | dict[int, KVCacheBlock]}`
- 大多数情况一个 hash 对应一个 block
- 当存在重复 block 时不 dedup（保证 block table 是 append-only 的）

**null_block**：当滑动窗口注意力（SWA）淘汰旧 block 时，BlockTable 中对应的位置不能简单地删掉（否则会导致逻辑索引错误），而是用 null_block 占位。这个 block 永远不会被前缀缓存，也永远不会被真正释放。

#### 2.0.3 BlockTable

BlockTable 是 CPU 侧的"请求→物理块ID 映射表"，它把 Scheduler 分配好的逻辑块序列转化为 GPU Attention Kernel 可直接寻址的二维矩阵。

```
BlockTable = [max_num_reqs, max_num_blocks_per_req] 的 int32 矩阵
行 (row)    → 一个请求 (req_index)
列 (col)    → 该请求的第几个 block
值          → 物理 block_id（在 GPU KV Cache 显存中的索引）
```

LLM 有两套实现：

| | BlockTable（基础版） | BlockTables（GPU 优化版） |
|---|---|---|
| **文件** | `vllm/v1/worker/block_table.py` | `vllm/v1/worker/gpu/block_table.py` |
| **粒度** | 单个 KV cache group | 所有 KV cache groups |
| **GPU 写入** | CpuGpuBuffer — 逐请求 copy_to_gpu | StagedWriteTensor — 批量 staging + Triton kernel |
| **num_blocks 跟踪** | num_blocks_per_row (numpy) | UvaBackedTensor (UVA 零拷贝) |
| **slot_mapping** | 1D Triton kernel, 单个 group | 2D Triton kernel, 所有 groups 并行 |
| **行操作** | `append_row` / `add_row` / `clear_row` / `move_row` / `swap_row` | `append_block_ids` (staged) / `apply_staged_writes` |
| **使用者** | 非 GPU worker / 旧版 | GPU model_runner（主流） |

**StagedWriteTensor 机制**（GPU 优化版的核心）：
- 请求的 block_table 更新不是逐条写入 GPU，而是先累积在 CPU 侧的 staging buffer
- 当累积满（或 step 结束时），通过 Triton kernel 批量写入 GPU
- 极大减少了 CPU→GPU 的数据传输次数

```
┌───────────────────────────────────────────────────────────┐
│                   Scheduler (CPU)                        │
│  决定: 哪些请求需要多少 block, 是否有 prefix cache hit    │
│  产出: block_ids (逻辑块的物理 ID 列表)                   │
└──────────────────────┬───────────────────────────────────┘
                       │ block_ids
                       ▼
┌───────────────────────────────────────────────────────────┐
│              BlockTable (CPU↔GPU 桥梁)                   │
│  功能1: 存储 ─── 将 block_ids 按请求组织成 2D 矩阵       │
│  功能2: 同步 ─── 批量将增量写入 GPU (StagedWrite)        │
│  功能3: 重排 ─── 按 batch 顺序重排 (gather)              │
│  功能4: 映射 ─── position → slot_id (compute_slot)       │
└──────────────────────┬───────────────────────────────────┘
                       │ block_table_tensor + slot_mapping
                       ▼
┌───────────────────────────────────────────────────────────┐
│           PagedAttention Kernel (GPU)                    │
│  输入: block_table[req][block_idx] → 物理 block_id       │
│  输入: slot_mapping[token_idx]    → KV cache slot_id     │
│  读写: KV_cache[block_id * block_size + offset]          │
└───────────────────────────────────────────────────────────┘
```

#### 2.0.4 KVCacheCoordinator

在 vLLM V0 版本中，主要依赖单一的 BlockSpaceManager，但随着大模型架构的异构化演进，V1 版本对 KV Cache 进行了彻底的重构，引入了 Coordinator 和 BlockPool 的解耦设计。

**为什么需要 Coordinator？** 因为某些模型（如 DeepSeek、Gemma-2）有多种不同类型的注意力层（MLA + Full Attention、Full + SWA），需要不同的 KV Cache 管理策略。

**三种 Coordinator 实现：**

| Coordinator | 适用场景 | 算法 |
|---|---|---|
| `KVCacheCoordinatorNoPrefixCache` | caching 关闭 | 没有开启前缀缓存功能，直接返回空 |
| `UnitaryKVCacheCoordinator` | 单 group（90% 的模型） | 适用于绝大多数经典大模型，如 Llama 3、DeepSeek-V3/R1、Mistral 等。所有层的 attention 机制相同。从左到右线性扫描，遇到第一个 miss 即停止 |
| `HybridKVCacheCoordinator` | 多 group（混合 attention） | 为混合注意力模型设计，迭代定点算法，每种 attention type 分别查，取交集 |

**混合注意力模型的例子：**
- **Mistral / Gemma-2**：一部分层使用全局注意力（Full Attention），另一部分层使用滑动窗口注意力（SWA）
- **Jamba**：混合了 Mamba（线性状态，不需要传统 KV Cache）和 Transformer Attention 层

每种 KV cache group 都有独立的 `SingleTypeKVCacheManager`（如 `FullAttentionManager`、`MLAAttentionManager`、`SlidingWindowManager`），但共享同一个 `BlockPool`。

**HybridKVCacheCoordinator 的挑战**：不同 group 的 block_size 可能不同。前缀缓存长度必须是所有 block_size 的 LCM（最小公倍数），否则部分 block 不完整，无法被缓存。

#### 2.0.5 SingleTypeKVCacheManager：单一类型管理器

这是抽象基类，定义了每种注意力类型对 KV Cache 的管理行为：

```python
# vllm/v1/core/single_type_kv_cache_manager.py:31-88
class SingleTypeKVCacheManager(ABC):
    def __init__(self, kv_cache_spec, block_pool, ...):
        self.block_size = kv_cache_spec.block_size        # 该 group 的块大小
        self.block_pool = block_pool                       # 共享的 BlockPool
        self.kv_cache_group_id = kv_cache_group_id         # group 编号

        # 核心映射：req_id → 该请求持有的 KVCacheBlock 列表
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # 每个请求已被缓存的 block 数量
        self.num_cached_block: dict[str, int] = {}

        # 本轮分配的新 block_ids（供 scheduler 取用）
        self.new_block_ids: list[int] = []
```

**Manager 类型映射表：**

| KVCacheSpec | Manager |
|---|---|
| `FullAttentionSpec` | `FullAttentionManager` |
| `TQFullAttentionSpec` | `FullAttentionManager` |
| `MLAAttentionSpec` | `FullAttentionManager` |
| `HiddenStateCacheSpec` | `FullAttentionManager` |
| `SlidingWindowSpec` | `SlidingWindowManager` |
| `SlidingWindowMLASpec` | `SlidingWindowManager` |
| `ChunkedLocalAttentionSpec` | `ChunkedLocalAttentionManager` |
| `MambaSpec` | `MambaManager` |
| `CrossAttentionSpec` | `CrossAttentionManager` |
| `SinkFullAttentionSpec` | `SinkFullAttentionManager` |

```
Scheduler.allocate_slots()
  └→ Coordinator.allocate_new_blocks()
       └→ for each manager:
            ├→ get_num_blocks_to_allocate()   # 1. 计算需要多少块
            ├→ allocate_new_computed_blocks() # 2. 处理 prefix cache 命中
            └→ allocate_new_blocks()          # 3. 不够就从 BlockPool 拿新块
```

#### 2.0.6 KVCacheBlocks

Scheduler 与 Coordinator 之间的接口封装：

```python
# vllm/v1/core/kv_cache_manager.py:24-100
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[i][j] = i-th kv_cache_group 的 j-th block

    def get_block_ids(self) -> tuple[list[int], ...]:
        """将 KVCacheBlock 列表转为纯 block_id 列表"""
        return tuple([blk.block_id for blk in group] for group in self.blocks)
```

它封装了 Coordinator 的分配结果，向 Scheduler 隐藏内部数据结构（KVCacheBlock 对象），只暴露 block_id。

#### 2.0.7 KVCacheSpec 层次体系

`vllm/v1/kv_cache_interface.py` 定义了完整的 KV Cache 规格层次：

```
KVCacheSpec (基类)
├── AttentionSpec (KV attention 基类)
│   ├── FullAttentionSpec
│   │   ├── TQFullAttentionSpec (TurboQuant)
│   │   ├── MLAAttentionSpec (DeepSeek MLA)
│   │   ├── SinkFullAttentionSpec (带 sink token)
│   │   └── EncoderOnlyAttentionSpec
│   ├── SlidingWindowSpec
│   │   └── SlidingWindowMLASpec
│   ├── ChunkedLocalAttentionSpec
│   └── CrossAttentionSpec
├── MambaSpec (SSM)
└── UniformTypeKVCacheSpecs (同质层容器)
```

---

### 2.1 系统初始化与显存分配

传统的 Hugging Face Transformers 的推理方式是"按需分配、动态增长"，也就是每生成一个 Token，KV Cache 就变大一点，PyTorch 就会在底层重新分配一块更大的显存，把旧数据拷过去，导致严重显存碎片化和申请/释放显存的性能消耗。

**vLLM 思路**：既然大模型推理的最大瓶颈是显存容量，那干脆在一开始，就把 GPU 上所有可用的"闲置显存"全部一把抓过来，划分成 block 供后续使用。

发生时机：vLLM 引擎启动时（此时没有任何用户请求）。

核心动作：
- 通过一次空跑试探出 GPU 剩余的极限可用显存
- 根据模型层数和配置，计算出能切分出多少个物理 Block
- 调用底层的 `_allocate_kv_cache`，用 `torch.zeros` 占用一整块物理显存，并塑形为 5D 张量
- 最终结果：建好了一个巨大的、空荡荡的"显存停车场"，等待请求接入

#### 阶段 1：每个 Worker 声明自己的 KV Cache 需求

引擎启动后，每个 worker 调用 `get_kv_cache_spec()`，遍历模型的所有 Attention Layer，收集各层需要什么样的 KV Cache。

```python
# vllm/v1/worker/gpu/attn_utils.py - get_kv_cache_spec()
def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
    kv_cache_spec: dict[str, KVCacheSpec] = {}
    # 遍历模型中所有 Attention Layer
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for layer_name, attn_module in attn_layers.items():
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

结果示例（Llama-8B）：
```json
{
  "model.layers.0.self_attn": FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=bfloat16),
  "model.layers.1.self_attn": FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=bfloat16),
  ...（32 层都是相同的 FullAttentionSpec）
}
```

#### 阶段 2：显存 Profiling — 算出来能给 KV Cache 多少显存

Worker 通过一次空跑来测量模型权重 + 激活值占了多少显存，剩下的才是 KV Cache 能用的。

```python
# vllm/v1/worker/gpu_worker.py - Worker.determine_available_memory()
@torch.inference_mode()
def determine_available_memory(self) -> int:
    if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
        self.model_runner.profile_run()
        return kv_cache_memory_bytes
    # 核心：用 memory_profiling 上下文做一次空跑
    with memory_profiling(
        self.init_snapshot,              # 模型加载前的内存快照
        weights_memory=int(self.model_runner.model_memory_usage),
    ) as profile_result:
        self.model_runner.profile_run()  # ★ 跑一次空推理，激活值峰值由此得出
    # 计算非 KV Cache 的内存占用
    profile_result.non_kv_cache_memory = (
        profile_result.non_torch_increase      # 非 PyTorch 分配（如 cuBLAS workspace）
        + profile_result.torch_peak_increase   # PyTorch 峰值 - 初始 PyTorch 占用
        + profile_result.weights_memory        # 模型权重
    )
    # ★★★ 核心公式：KV Cache 可用显存 = 请求的总内存 - 非 KV Cache 的内存
    self.available_kv_cache_memory_bytes = (
        self.requested_memory                  # 总 GPU 显存 × gpu_memory_utilization
        - profile_result.non_kv_cache_memory   # 权重 + 激活 + 其他
        - cudagraph_memory_estimate            # CUDA Graph 回放所需显存
    )
    return int(self.available_kv_cache_memory_bytes)
```

图解：

```
┌───────────────────────────────────────────────────────────┐
│          GPU 总显存 (如 80 GB)                       │
├───────────────────────────────────────────────────────────┤
│  requested_memory = total × gpu_memory_utilization   │
│  (如 80 × 0.9 = 72 GB)                              │
├──────────────────────────┬────────────────────────────────┤
│   non_kv_cache_memory    │  available_kv_cache_      │
│   - 模型权重              │  memory_bytes            │
│   - 激活值峰值            │  ← 给 KV Cache 的显存     │
│   - CUDA Graph           │                          │
│   - 其他 runtime         │                          │
├──────────────────────────┼────────────────────────────────┤
│   e.g. 30 GB             │  e.g. 42 GB              │
└──────────────────────────┴────────────────────────────────┘
```

#### 阶段 3：Config 生成 — 把可用显存换算成 Block 数

有了可用显存，下一步是计算能分配多少个 Block，以及如何在不同层之间分配。

```python
# vllm/v1/core/kv_cache_utils.py - get_kv_cache_configs()
def get_kv_cache_configs(
    vllm_config, kv_cache_specs, available_memory
) -> list[KVCacheConfig]:
    # ① 合并所有 Worker 的 KV Cache Spec（处理 PP 的情况）
    merged_kv_cache_specs = {}
    for kv_cache_spec_one_worker in kv_cache_specs:
        for layer_name, layer_spec in kv_cache_spec_one_worker.items():
            merged_kv_cache_specs[layer_name] = layer_spec

    # ② 把相同 spec 的 layer 归为一个 group
    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)

    # ③ 为每个 Worker 生成 KVCacheConfig（考虑 PP 按 worker 投影）
    for projected_groups, ..., available_memory_one_worker in zip(...):
        kv_cache_configs.append(
            get_kv_cache_config_from_groups(
                vllm_config, projected_groups, available_memory_one_worker
            )
        )

    # ④ 所有 Worker 取最小的 num_blocks（保证一致性）
    min_num_blocks = min(cfg.num_blocks for cfg in kv_cache_configs)
    for cfg in kv_cache_configs:
        cfg.num_blocks = min_num_blocks  # 统一缩小到最小值
        for tensor in cfg.kv_cache_tensors:
            tensor.size = tensor.size // num_blocks_old * min_num_blocks  # 等比例缩小
```

**Step ④ 是关键**：多 worker（TP/PP）场景下，不同 worker 可能有不同的可用显存。最终取最小的 num_blocks，然后所有 worker 等比缩小，确保 block table 长度在所有 worker 上一致。

**计算 Block 数：**

```python
# vllm/v1/core/kv_cache_utils.py - get_num_blocks()
def get_num_blocks(vllm_config, num_layers, available_memory, page_size):
    # ★ 公式：可用显存 ÷ 每页大小 ÷ 层数
    num_blocks = int(available_memory // page_size // num_layers)
    return num_blocks
```

以 Llama-8B, 16 tokens/block, 32 layers, bf16 为例：

```
page_size_bytes = 2 × 16 × 8 × 128 × 2         # 2(K+V) × block_size × kv_heads × head_dim × bf16
                = 2 × 16 × 8 × 128 × 2
                = 65,536 bytes = 64 KiB

available_memory = 42 GB = 42 × 1024³ bytes
num_blocks = 42 × 1024³ // 65536 // 32
           = 688,128 // 32
           = 21,504 blocks per layer

total_blocks = 21,504  (block table 长度统一)
每个 layer 的 tensor size = 21,504 × 64 KiB = 1.3125 GiB
```

1 个 Block 的内部结构：

```
========================================================================
                      1 个 Block (物理内存块, 总大小: 64 KiB)
========================================================================
[ 内存切分 ]: 划分为 16 个连续的 Slot (槽位)，用于存放 16 个 Token 的 KV 缓存
------------------------------------------------------------------------
内存地址偏移   |  关联数据 (Token)  |  内部张量结构 (Tensor T)
------------------------------------------------------------------------
Slot 0      |  Token 1        | [ Key 张量 (2 KB) ]  +  [ Value 张量 (2 KB) ]
Slot 1      |  Token 2        | [ Key 张量 (2 KB) ]  +  [ Value 张量 (2 KB) ]
Slot 2      |  Token 3        | [ Key 张量 (2 KB) ]  +  [ Value 张量 (2 KB) ]
...         |  ...            | ...
Slot 15     |  Token 16       | [ Key 张量 (2 KB) ]  +  [ Value 张量 (2 KB) ]
========================================================================
[ 调度状态 ]: 满载 (Capacity Reached)。
[ 触发动作 ]: 当模型生成第 17 个 Token 时，内存管理器 (Block Allocator)
             将向 GPU 显存池申请分配并映射一个新的 Block。
========================================================================
```

生成的 KVCacheConfig：

```python
KVCacheConfig(
    num_blocks=21504,           # 所有层共享同一个 block table 长度
    kv_cache_tensors=[          # 每个 layer 一个 tensor (或共享)
        KVCacheTensor(size=1409286144, shared_by=["model.layers.0.self_attn"]),
        KVCacheTensor(size=1409286144, shared_by=["model.layers.1.self_attn"]),
        # ... 32 个
    ],
    kv_cache_groups=[           # 所有层在同一个 group（因为 spec 相同）
        KVCacheGroupSpec(
            layer_names=["model.layers.0.self_attn", ..., "model.layers.31.self_attn"],
            kv_cache_spec=FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=bfloat16)
        )
    ]
)
```

#### 阶段 4：Worker 分配 GPU Tensor

Config 传回 Worker 后，Worker 在 GPU 上真正分配 KV Cache 的物理内存。

```python
# vllm/v1/worker/gpu_worker.py - Worker.initialize_from_config()
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
    # ★ 调用 ModelRunner 分配 KV Cache
    self.model_runner.initialize_kv_cache(kv_cache_config)

# vllm/v1/worker/gpu_model_runner.py - initialize_kv_cache()
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False):
    # ① 配置隔离与架构兼容 (深拷贝防污染)
    kv_cache_config = deepcopy(kv_cache_config)
    self.may_add_encoder_only_layers_to_kv_cache_config()
    self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)

    # ② 初始化底层的注意力算子后端
    self.initialize_attn_backend(kv_cache_config, is_profiling=is_profiling)

    # ③ ★ 核心对齐：计算底层 Kernel 真正支持的 Block Size
    # (例如 vLLM 调度器设定的 block_size 是 256，但底层算子只支持 64，这里会将其逻辑切分为 4 个 64)
    kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)

    # ④ 构建元数据管理器
    self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

    # ⑤ ★ 真正分配显存
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)

    # ⑥ 注册前沿加速特性
    if self.speculative_config:
        self.drafter.validate_same_kv_cache_group(kv_cache_config)
    if has_kv_transfer_group() and not is_profiling:
        kv_transfer_group.register_kv_caches(kv_caches)
```

**Step ③ `prepare_kernel_block_sizes` 是精髓设计** —— Scheduler 侧的 block_size（逻辑）和 GPU Kernel 支持的 block_size（物理）可以不同。例如调度器按 256 tokens/block 管理，但 FlashInfer kernel 只支持 64，系统会自动将一个逻辑块映射到 4 个物理子块。

```python
# vllm/v1/worker/gpu/attn_utils.py - init_kv_cache()
def init_kv_cache(runner_kv_caches, forward_context, kv_cache_config,
                  attn_groups, device, cache_dtype, kernel_block_sizes, vllm_config):
    # ① 分配 raw tensor（int8 类型，后面会 reshape）
    kv_cache_raw_tensors = _allocate_kv_cache(
        kv_cache_config, shared_kv_cache_layers, device
    )
    # ② reshape 成 attention backend 需要的形状
    kv_caches = _reshape_kv_cache(
        attn_groups, kv_cache_raw_tensors,
        kernel_block_sizes, cache_dtype, shared_kv_cache_layers
    )
    # ③ 绑定到 forward_context，forward 时使用
    bind_kv_cache(kv_caches, forward_context, runner_kv_caches)
    return kv_caches

# vllm/v1/worker/gpu/attn_utils.py - _allocate_kv_cache()
def _allocate_kv_cache(kv_cache_config, shared_layers, device):
    kv_cache_raw_tensors = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        # ★ 在 GPU 上分配一段连续的 int8 内存
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor
    return kv_cache_raw_tensors
```

KV Cache Tensor 的 reshape 过程：

```
分配时 (raw tensor):
┌────────────────────────────────────────────────────────────┐
│  torch.zeros(size=1.3 GiB, dtype=torch.int8)  # 连续内存  │
└────────────────────────────────────────────────────────────┘

reshape 后（给 FlashInfer kernel 用）:
Shape: (num_blocks, 2, num_kv_heads, block_size, head_size)
     : (21504,    2,     8,          16,         128)
       ↑         ↑      ↑           ↑           ↑
     block 数   K/V   kv_heads  tokens/block  head_dim
```

#### 阶段 5：Scheduler 侧初始化 BlockPool 和 KVCacheManager

Scheduler 也收到同一个 KVCacheConfig，用它来初始化 BlockPool：

```python
# vllm/v1/core/kv_cache_coordinator.py - KVCacheCoordinator.__init__()
def __init__(self, kv_cache_config, ...):
    # ★ 创建 BlockPool：num_blocks 个 KVCacheBlock
    self.block_pool = BlockPool(
        num_gpu_blocks=kv_cache_config.num_blocks,  # 21504 个 block
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
        ...
    )

# vllm/v1/core/block_pool.py - BlockPool.__init__()
def __init__(self, num_gpu_blocks, ...):
    # ★ 创建所有 block 的元数据
    self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
    # ★ 全部放入 free queue
    self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
    # ★ 创建 prefix cache 的 hash→block 映射
    self.cached_block_hash_to_block = BlockHashToBlockMap()
```

---

### 2.2 请求接入与逻辑调度

在这个阶段，系统刚刚接到了用户的 Prompt，还没开始进行任何的 GPU 矩阵运算。这一阶段的优化目标是：
1. 最大化复用已有的 KV Cache（降低 TTFT）
2. 调度前确保显存够用（避免 OOM）

发生时机：用户发来一段 Prompt 请求时。

核心动作：分为两个子步骤——先查 Prefix Cache 有没有现成的 block，再分配新的 block。

#### 阶段 1：Prefix Cache 查找

收到一个新 Request 时，先根据 token id 按 `block_size`（如 16）切块，计算出每个逻辑块的 Hash 值。然后查 `BlockHashToBlockMap` 看它的前缀 token 是否已经被之前的请求计算过。

**最多命中 `prompt_length - 1`**，最后一个 token 必须重新计算以获取 logits。

```python
# kv_cache_manager.py
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    # 跳过 prefix cache 查找的条件：
    # 1. caching 被禁用
    # 2. request 标记为 skip_reading_prefix_cache（如需要 prompt logprobs、pooling 模型）
    if not self.enable_caching or request.skip_reading_prefix_cache:
        return self.empty_kv_cache_blocks, 0

    # ★ 关键：最多命中 prompt_length - 1，最后一个 token 必须重新计算以获取 logits
    max_cache_hit_length = request.num_tokens - 1

    # ★ 调用 Coordinator 查找最长前缀命中
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

`request.block_hashes` 的来源：Request 对象在创建时就已经预计算了每个 block 的 hash 值（基于 token IDs）。

**UnitaryKVCacheCoordinator 的查找策略（从左到右线性扫描）：**

```python
# single_type_kv_cache_manager.py - FullAttentionManager.find_longest_cache_hit()
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                            block_pool, kv_cache_spec, use_eagle, ...):
    computed_blocks = tuple([] for _ in range(len(kv_cache_group_ids)))
    block_size = kv_cache_spec.block_size
    max_num_blocks = max_length // block_size

    # ★ 从左到右逐个查 block hash
    for block_hash in itertools.islice(block_hashes, max_num_blocks):
        if cached_block := block_pool.get_cached_block(
            block_hash, kv_cache_group_ids
        ):
            # 命中！加入结果列表
            for computed, cached in zip(computed_blocks, cached_block):
                computed.append(cached)
        else:
            break  # ★ 第一次 miss 就停止（前缀必须连续命中）

    # EAGLE 特殊处理：丢弃最后一个命中的 block（需要重算以获取 hidden states）
    if use_eagle and computed_blocks[0]:
        for computed in computed_blocks:
            computed.pop()
    return computed_blocks
```

**HybridKVCacheCoordinator 的查找策略（迭代定点算法）：**

混合注意力模型的 Prefix Cache 查找更复杂，因为不同 Attention 类型可能有不同的 block_size 和缓存行为：

```python
# kv_cache_coordinator.py - HybridKVCacheCoordinator.find_longest_cache_hit()
def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
    # 迭代定点算法：每种 attention type 检查当前候选长度
    # - 如果某类型接受 → 继续下一类型
    # - 如果某类型缩短了命中长度 → 重新从第一个类型开始检查
    # 收敛条件：长度单调递减且有下界 0

    while True:
        curr_hit_length = hit_length
        for idx, (spec, group_ids, manager_cls) in enumerate(self.attention_groups):
            # Full attention 是 downward-closed：只需查一次，后续迭代直接裁剪
            if isinstance(spec, FullAttentionSpec) and cached_blocks is not None:
                curr_hit_length = curr_hit_length // spec.block_size * spec.block_size
                continue

            hit_blocks = manager_cls.find_longest_cache_hit(...)
            _new_hit_length = len(hit_blocks[0]) * spec.block_size
            curr_hit_length = _new_hit_length

        if curr_hit_length >= hit_length:
            break  # 所有类型达成一致
        hit_length = curr_hit_length

    return hit_blocks_by_group, hit_length
```

**核心设计约束：** 前缀缓存命中长度必须是所有 block_size 的 LCM，保证不同 group 的 block 边界对齐。

**查找结果示例：**

```
block_hashes  = [H0,   H1,   H2,   H3,   H4,   ...]
                       ↓ 逐个查 BlockHashToBlockMap
                 H0 → hit!  block_7   (ref_cnt 从 0→1, 从 free_queue 移除)
                 H1 → hit!  block_12  (ref_cnt 从 0→1)
                 H2 → MISS! → 停止
结果: computed_blocks = ( [block_7, block_12], )
      num_new_computed_tokens = 2 × 16 = 32
```

#### 阶段 2：Block 分配

Prefix Cache 命中之后，接下来为那些未命中缓存的新 Token 申请真正的物理 block。

**完整的 `allocate_slots()` 流程：**

```python
# kv_cache_manager.py
def allocate_slots(self, request, num_new_tokens, ...):
    # ════════════════════════════════════════════════
    # ① 可选：全序列准入检查（防止 chunked prefill 过度接纳请求）
    # ════════════════════════════════════════════════
    if full_sequence_must_fit:
        full_num_tokens = min(request.num_tokens, self.max_model_len)
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id, full_num_tokens, new_computed_block_list, ...
        )
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # ★ 全序列放不下，拒绝调度

    # ════════════════════════════════════════════════
    # ② 释放滑动窗口外的旧 block（SWA 模型；Full Attention 跳过）
    # ════════════════════════════════════════════════
    self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    # ════════════════════════════════════════════════
    # ③ 检查 free block 够不够
    # ════════════════════════════════════════════════
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # ★ 空间不够，等下一步

    # ════════════════════════════════════════════════
    # ④ 把 prefix cache 命中的 block 挂到 request 上
    #   （包括外部 connector 缓存的 block）
    # ════════════════════════════════════════════════
    self.coordinator.allocate_new_computed_blocks(
        request_id, new_computed_block_list,
        num_local_computed_tokens, num_external_computed_tokens
    )

    # ════════════════════════════════════════════════
    # ⑤ 从 free queue 拿新 block
    # ════════════════════════════════════════════════
    new_blocks = self.coordinator.allocate_new_blocks(
        request_id, num_tokens_need_slot, num_tokens_main_model
    )

    # ════════════════════════════════════════════════
    # ⑥ 把满的 block 写入 prefix cache
    #    (只缓存已验证的 token，排除可能被拒绝的 draft token)
    # ════════════════════════════════════════════════
    num_tokens_to_cache = min(
        total_computed_tokens + num_new_tokens,
        request.num_tokens,  # 只缓存已验证的 token
    )
    self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return self.create_kv_cache_blocks(new_blocks)
```

**对于步骤⑤，从 BlockPool 拿新 block：**

```python
# single_type_kv_cache_manager.py - allocate_new_blocks()
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_new_blocks = num_required_blocks - len(req_blocks)
    if num_new_blocks <= 0:
        return []
    # ★ 从 free queue 拿 num_new_blocks 个 block
    new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
    req_blocks.extend(new_blocks)  # 挂到 request 的 block 列表
    return new_blocks

# block_pool.py - BlockPool.get_new_blocks()
def get_new_blocks(self, num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)  # ★ 从 free queue 头部拿
    for block in ret:
        self._maybe_evict_cached_block(block)  # 如果 block 有旧 hash，清除
        block.ref_cnt += 1                     # 引用计数 = 1
    return ret
```

**Block 布局示意图（含投机解码和外部缓存的扩展视图）：**

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
                                                  |   < 待 计 算 >     |
----------------------------------------------------------------------
                                  |            < 待 分 配 (底层显存) >    |
----------------------------------------------------------------------
                                  | < 待 加 入 > |
                                  | < 前缀缓存 > |
----------------------------------------------------------------------
| <----------- 来自 vLLM 或 Connector 的前缀缓存 (已计算) Token -----------> |
| <--------------- 如果超出滑动窗口(Sliding Window)，可安全移除 -----------> |
----------------------------------------------------------------------
| <-------- vLLM 本地已缓存 --------> | vLLM 未缓存, |
|                                     | 但已被外部   |
| 引用计数(ref_cnt) | 引用计数        | Connector    |
| 已增加            | 尚未增加        | 缓存         |
----------------------------------------------------------------------

缩写：
comp      = request.num_computed_tokens
new_comp  = num_new_computed_tokens (prefix cache 新命中)
ext_comp  = num_external_computed_tokens (外部 connector 缓存，如 P/D 分离)
new       = num_new_tokens (含未验证的 draft token)
lookahead = num_lookahead_tokens (投机解码的 lookahead token)
```

**Request 的 block 列表演化（block_size=16, 共 48 tokens）：**

```
─────────────────────────────────────────────────────────────
Step 1 (prefill, 48 tokens):
  prefix 命中: [block_7,  block_12]               ← 2 blocks (32 tokens)
  新分配:      [block_99, block_42, block_55]     ← 3 blocks (48 tokens total)
  最终列表:    [block_7,  block_12, block_99, block_42, block_55]

Step 2 (decode, +1 token):
  最终列表:    [block_7, block_12, block_99, block_42, block_55]
              (无需新分配，因为 49 tokens 仍在 5 个 block 的容量内)

Step 3 (decode, 又多了 17 tokens, 总共 66 tokens):
  新分配:      [block_3]                          ← 1 block
  最终列表:    [block_7, block_12, block_99, block_42, block_55, block_3]
```

**对于步骤② SWA 跳过 block：**

```python
# single_type_kv_cache_manager.py - remove_skipped_blocks()
def remove_skipped_blocks(self, request_id, total_computed_tokens):
    num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
    # FullAttentionManager.get_num_skipped_tokens() 直接返回 0
    # SlidingWindowManager 返回 sliding_window 外的 token 数
    if num_skipped_tokens <= 0:
        return  # ★ Full Attention 模型永远不会进入这里

    blocks = self.req_to_blocks[request_id]
    num_skipped_blocks = num_skipped_tokens // self.block_size
    for i in range(num_skipped_blocks - 1, -1, -1):
        if blocks[i] == self._null_block:
            break
        removed_blocks.append(blocks[i])
        blocks[i] = self._null_block   # ★ 替换为 null_block（占位符）
    self.block_pool.free_blocks(removed_blocks)  # ★ 实际 block 归还 free queue
```

**关于 `num_tokens_main_model` 参数**：这个参数用于投机解码场景。主模型（target model）的 token 数 = `num_tokens - num_lookahead_tokens`。投机解码的 draft token（lookahead）需要分配 KV Cache slot，但它们可能被拒绝——因此在步骤⑥缓存时，只缓存到 `request.num_tokens`（已验证的 token），不缓存 draft token。

**关于 `num_external_computed_tokens` 参数**：用于 P/D（Prefill/Decode）分离场景。外部 connector（如 KV Transfer）可能已经缓存了部分 token 的 KV Cache，这些 token 不是由本机计算的，但需要分配本地 block 来接收从远程节点传输过来的 KV 数据。`delay_cache_blocks=True` 时跳过立即缓存，等待 KV transfer 完成后再处理。

---

### 2.3 模型执行

ModelRunner 把 Scheduler 传来的抽象决策（哪些请求要跑、跑几个 Token），具象化为底层的输入张量（input_ids、positions），将连续的逻辑 Token 序列翻译成离散的物理显存地址。

**发生时机**：每次 Scheduler step 中，ModelRunner 执行模型 forward。

**核心动作**：构造 `block_table` 和 `slot_mapping`，传给 Attention Backend，kernel 通过查表实现逻辑地址 → 物理地址的映射。

#### 完整 execute_model 流程：

```python
# vllm/worker/gpu_model_runner.py - GPUModelRunnerBase.execute_model()
def execute_model(self, scheduler_output, intermediate_tensors=None):

    # ★ 1. 状态防重入校验
    if self.execute_model_state is not None:
        raise RuntimeError("State error: sample_tokens() must be called...")
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    with self.synchronize_input_prep():
        # ★ 2. 空载拦截
        if not num_scheduled_tokens:
            return self.kv_connector_no_forward(...)

        # ★ 3. 提取请求元数据 & 更新内部状态
        logits_indices, _ = self._prepare_inputs(scheduler_output, ...)
        #  内部状态更新包括：
        #  - 清除 finished_req_ids 的缓存状态
        #  - 清零 new_block_ids_to_zero 对应的 GPU 内存
        #  - 添加 scheduled_new_reqs 的新请求状态
        #  - 更新 scheduled_cached_reqs 的增量状态

        # ★ 4. CUDA Graph 策略决定
        (cudagraph_mode, batch_desc, ...) = self._determine_batch_execution_and_padding(...)
        num_tokens_padded = batch_desc.num_tokens

        # ★ 5. 显存物理映射 (Slot Mappings)
        _, slot_mappings = self._get_slot_mappings(...)

        # ★ 6. 构建注意力元数据 (Attention Metadata)
        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            slot_mappings=slot_mappings_by_group,
            ...
        )

        # ★ 7. 张量组装 (Preprocess)
        input_ids, inputs_embeds, positions, ... = self._preprocess(...)

    # ★ 8. 物理前向传播 (The Actual Forward Pass)
    with set_forward_context(
        attn_metadata,
        cudagraph_runtime_mode=cudagraph_mode,
        slot_mapping=slot_mappings, ...
    ):
        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = model_output

    # ★ 9. 流水线并行(PP)路由
    if not get_pp_group().is_last_rank:
        return hidden_states  # 非最后一张卡，只发隐状态

    # ★ 10. 计算词表概率 (Logits)
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states)

    # ★ 11. 状态打包与交接
    self.execute_model_state = ExecuteModelState(
        scheduler_output, logits, hidden_states, slot_mappings, ...
    )
    return None
```

#### block_table 和 slot_mapping 的关系

- **block_table**（控制面）：以请求（Request）为单位，记录每个句子占用了哪些 GPU 物理块。维度 `[num_reqs, max_blocks_per_req]`
- **slot_mapping**（数据面）：以 Token 为单位，直接扁平化地指明当前 Batch 里每一个输入 Token 的 KV 应该写入到显存的什么精确位置。维度 `[num_tokens]`

#### slot_mapping 的计算过程

当 `execute_model` 启动并进入输入准备阶段时：

```python
# 例子：request 当前有 2 个 block，第 20 个 token 的 slot 计算
block_idx = 20 // 16  # 得到逻辑块索引 1
offset = 20 % 16      # 得到块内偏移 4
block_id = block_table[0][block_idx]  # 查表得到物理块 ID 是 12
slot = block_id * block_size + offset # 12 * 16 + 4 = 196
# slot_mapping 向量的对应位置被填入 196
```

在 GPU 优化版（`BlockTables`），这个计算通过 Triton kernel `_compute_slot_mappings_kernel` 直接在 GPU 上并行完成，支持：
- 多 group 并行计算
- Context Parallelism（DCP）的 offset 调整

#### attn_metadata 的构建

生成好 slot_mapping 向量和 block_table 矩阵后，`execute_model` 把它们塞进 `attn_metadata`（注意力元数据）结构体中：

```python
# build_attn_metadata() 为每个 KV cache group 构建 AttentionMetadata
# 内容包含:
CommonAttentionMetadata:
    query_start_loc      # 每个请求的 query 起始位置
    seq_lens             # 每个请求的序列长度
    block_table_tensor   # GPU 上的 block table 张量
    slot_mapping         # GPU 上的 slot mapping 张量
    num_reqs             # batch 中的请求数
    num_actual_tokens    # 实际 token 数（不含 padding）
    max_query_len        # 最大 query 长度
    max_seq_len          # 最大序列长度
```

使用 `set_forward_context(attn_metadata, slot_mapping=slot_mappings, ...)` 将映射表和显存指针做成全局上下文。底层模型的每一个 Attention 层都能直接从这个上下文中免传参读取到当前的 slot_mapping。

#### Attention 计算中的读写

当模型前向传播走到具体的 Attention 算子时，block_table 和 slot_mapping 将分别负责"写（Append）"和"读（Read）"两个动作：

**写（Append）**：当前 Token 进入 Attention 层后，通过线性层计算出了它自己的 $Q, K, V$ 张量。此时底层算子读取 slot_mapping：
- 算子看到当前 Token 对应的 slot 值是 196
- 直接将新算出来的 $K$ 和 $V$ 向量，写入到预先分好的大矩阵的第 196 个槽位

**读（PagedAttention）**：写完之后，当前 Token 需要和历史所有的 Token 做矩阵乘法（计算 Attention Score）：
- 由于历史的 KV Cache 散落在物理内存的不同 Block 里
- 算子此时会激活 PagedAttention 机制，去读取该请求对应的 block_table 一整行：`[7, 12, 99, 42, 55, 3]`
- Kernel 会启动并行线程，根据这 6 个物理块的 ID，跨越不连续的显存空间，把历史 KV 统统捞出来，和当前的 $Q$ 做高效的 Attention 计算

```
execute_model(scheduler_output, intermediate_tensors)
  │
  ├─ ① 清理上一次的 execute_model_state（否则报错）
  │
  ├─ ② [可选] 处理 KV Connector 的抢占
  │     get_kv_transfer_group().handle_preemptions(...)
  │
  ├─ ③ _update_states(scheduler_output)
  │     - 清除 finished_req_ids 的缓存状态
  │     - 清零 new_block_ids_to_zero 对应的 GPU 内存
  │     - 添加 scheduled_new_reqs 的新请求状态
  │     - 更新 scheduled_cached_reqs 的增量状态
  │     - 移除不在本 step 调度的旧请求
  │
  ├─ ④ 如果 total_num_scheduled_tokens == 0:
  │     返回 EMPTY_MODEL_RUNNER_OUTPUT（没有 token 要计算）
  │
  ├─ ⑤ _prepare_inputs(scheduler_output)
  │     - 构建 input_ids, positions, query_start_loc
  │     - 构建 block_table 的 slot_mapping（逻辑 token → 物理 KV Cache 地址）
  │     - 计算 logits_indices（哪些 token 位置要采样）
  │     - 构建 spec_decode_metadata
  │
  ├─ ⑥ _build_attention_metadata()
  │     - 为每个 KV cache group 构建 AttentionMetadata
  │     - 包含 block_tables, slot_mappings, seq_lens 等
  │
  ├─ ⑦ _preprocess(scheduler_output)
  │     - 运行多模态 encoder（如有）
  │     - 处理 prompt_embeds
  │     - 处理 PP 的 intermediate_tensors
  │
  ├─ ⑧ model.forward(input_ids, positions, ...)
  │     ★ 实际模型推理，Attention 层通过 block_table 读写 KV Cache
  │
  ├─ ⑨ 如果不是最后一个 PP 阶段:
  │     返回 IntermediateTensors（传给下一个 PP 阶段）
  │
  ├─ ⑩ 如果是 Pooling 模型:
  │     返回 _pool() 的结果
  │
  └─ ⑪ 生成模型 + 同步调度:
        compute_logits → 保存到 self.execute_model_state → 返回 None
        （外部后续调用 sample_tokens() 完成采样并返回 ModelRunnerOutput）
```

---

### 2.4 显存置换、抢占和回收

这一阶段目标是：
1. 维持 Prefix Cache 的高命中率
2. 高效回收

#### 2.4.1 显存置换（Eviction）— 懒清除策略

**发生时机**：当 Free Queue 里弹出一个 Block，准备分配给新来的 Token 时。

```python
# block_pool.py
def get_new_blocks(self, num_blocks):
    ret = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        self._maybe_evict_cached_block(block)  # ★ 关键：懒清除
        block.ref_cnt += 1
    return ret

def _maybe_evict_cached_block(self, block):
    block_hash = block.block_hash
    if block_hash is None:
        return False  # block 没有 hash，无需 evict
    # ★ 从 BlockHashToBlockMap 中移除
    if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
        return False
    block.reset_hash()   # block._block_hash = None
    return True
```

**注意**：eviction 只清理 hash 映射，不释放 GPU 显存——因为 block 马上要分配给新请求使用。GPU 显存只是被"覆写"，而非"释放再分配"。

#### 2.4.2 Request 完成 → 释放 Block

**发生时机**：一个 Request 生成到了 `<EOS>` 或者达到了 `max_tokens`，彻底结束生命周期。

**逆序释放的原因**：在长文本推理中，尾部的 Block 通常是该 Request 独有的生成内容（Decode 阶段），而头部的 Block 极大概率是共享的 System Prompt（Prefill 阶段）。逆序释放能够最快地将独有资源剥离出来。

```python
# kv_cache_manager.py
def free(self, request):
    self.coordinator.free(request.request_id)

# single_type_kv_cache_manager.py
def free(self, request_id):
    req_blocks = self.req_to_blocks.pop(request_id, [])
    # ★ 逆序释放：尾部的 block 先被 free
    ordered_blocks = reversed(req_blocks)
    self.block_pool.free_blocks(ordered_blocks)
    self.num_cached_block.pop(request_id, None)

# block_pool.py
def free_blocks(self, ordered_blocks):
    for block in ordered_blocks:
        block.ref_cnt -= 1  # ★ 引用计数 -1
    # ★ ref_cnt == 0 且不是 null_block → 放回 free queue
    self.free_block_queue.append_n([
        block for block in ordered_blocks
        if block.ref_cnt == 0 and not block.is_null
    ])
```

**释放后的三种可能：**

```
┌────────────────────────────────────────────────────────────────┐
│ 情况 1: ref_cnt > 0                                          │
│   → 还有其他请求在共享这个 block（prefix cache 命中）          │
│   → 不放入 free queue，block 仍在使用中                       │
│                                                              │
│ 情况 2: ref_cnt == 0, block_hash 有值                        │
│   → 放回 free queue，但保留 block_hash                        │
│   → 成为 "eviction candidate"：可以被新 prefix cache 命中      │
│   → 也可以被新请求 evict（覆写 GPU 内存）                      │
│                                                              │
│ 情况 3: ref_cnt == 0, block_hash 为 None                     │
│   → 放回 free queue，一个干净的 free block                    │
│   → 下次直接分配，无需 evict                                  │
└────────────────────────────────────────────────────────────────┘
```

#### 2.4.3 SWA 旧 Block 回收

**发生时机**：针对带有 Sliding Window Attention 的模型，在每次 `allocate_slots` 时触发。

流程：
1. 计算当前 Sequence 长度与 Window 的差集，算出有几个旧 Block "过期"了
2. 把 Request 内部记录的物理 Block 抽出来，调用 `free_blocks`（走上面的状态机逻辑，减引用计数，丢回 Free Queue）
3. 在 Request 原本的位置塞入 `self._null_block`。这保证了逻辑位置坐标不断裂

```python
# sliding window = 32, block_size = 16
# 目前已 computed 48 tokens, 只保留最后 32 tokens
# → 前 16 tokens (1 block) 被释放
num_skipped_tokens = 48 - 32 = 16
num_skipped_blocks = 16 // 16 = 1
# 第 0 个 block 被替换为 null_block 并归还 free queue
blocks[0] = self._null_block
self.block_pool.free_blocks([old_block_7])
```

#### 2.4.4 抢占（Preemption）

**发生时机**：当系统负载极高，连 Free Queue 都被彻底榨干时。

vLLM V1 目前**不支持传统意义上的抢占**（不把 KV Cache 从 GPU 搬到 CPU）。

**V1 架构为什么放弃了传统的 Swap 抢占：**
- **Chunked Prefill 的引入**：让显存的消耗变得平滑可控。长 prompt 被切分为多个 chunk 逐步处理，每个 chunk 处理完后释放中间激活值，避免了显存峰值
- **等待机制（Wait）**：在 V1 中，如果 `allocate_slots` 发现显存真的不够了，它不会去折腾复杂的 Swap。它会直接返回 `None`。调度器看到 `None`，就会让这个请求在这一步"空转"或者挂起，让其他快要结束的请求先跑完

---

## 三、vLLM KV Cache 高级特性

### 3.1 KV Cache 量化（FP8/INT8/NVFP4）

支持的 `kv_cache_dtype`：

| 模式 | KVQuantMode | 描述 |
|---|---|---|
| `"auto"` | NONE | 使用模型默认 dtype |
| `"fp8"` / `"fp8_e4m3"` / `"fp8_e5m2"` | FP8_PER_TENSOR | 每 tensor 一个 scale |
| `"fp8_per_token_head"` | FP8_PER_TOKEN_HEAD | 逐 token-head 的 FP8 动态 scale（Flash Attention 3 使用） |
| `"int8_per_token_head"` | INT8_PER_TOKEN_HEAD | 逐 token-head 的 INT8 动态 scale |
| `"nvfp4"` | NVFP4 | NVIDIA Blackwell FP4 格式，打包 FP4 数据 + FP8 block scales |
| `"fp8_ds_mla"` | — | DeepSeek V3.2/V4 MLA 专用格式（584B/token） |

量化通过 `kv_cache_dtype` 参数启用，可以将 KV Cache 内存占用降低 **2-4 倍**。量化发生在 GPU tensor 的 reshape 阶段 —— `_reshape_kv_cache()` 会根据 `kv_cache_dtype` 为 K/V 分配不同大小的 tensor 区域，并在 attention kernel 中做反量化。

### 3.2 KV Cache Offload

KV Offload 的本质是在"容量（Capacity）"与"带宽（Bandwidth）"之间做妥协。随着 1M 甚至 10M 超长上下文模型的普及，单机 GPU HBM 已经不可能装下完整的 KV Cache。核心思想是将 KV Cache 从昂贵的 GPU 显存卸载到更便宜的 CPU 内存甚至远端存储中。

```python
# vllm/v1/kv_offload/base.py
class OffloadingManager(ABC):
    def lookup(self, key, req_context) -> bool | None:      # 检查块是否已卸载
    def prepare_load(self, keys, req_context) -> LoadStoreSpec:  # 准备从 CPU 加载
    def prepare_store(self, keys, req_context) -> PrepareStoreOutput | None:  # 准备卸载到 CPU
    def complete_load(self, keys, req_context):              # 加载完成
    def complete_store(self, keys, req_context, success):    # 卸载完成
```

#### 方案一：CPU Offloading（CPUOffloadingSpec）

- 将 KV Cache 直接卸载到 CPU 内存（通过 mmap 共享内存区域）
- 配置 `cpu_bytes_to_use` 指定 CPU 端缓存大小
- Worker 端 `CpuGpuOffloadingHandlers` 负责 GPU ↔ CPU 的数据搬运
- 支持 **LRU/ARC** 淘汰策略，通过 `eviction_policy` 参数选择
- 通过 `store_threshold` 参数控制块被访问多少次后才卸载到 CPU（热块保护机制）

#### 方案二：Tiering Offloading（TieringOffloadingSpec）

多级缓存层次：GPU → CPU（主层）→ 二级层（Storage/Network）

**关键设计原则：**
1. **Always offload to all tiers** — 写入主层时自动级联到所有二级层
2. **Primary tier is the gateway** — 二级层不能直接访问 GPU
3. **Staged promotion** — 二级层数据必须先提升到主层才能被 GPU 使用
4. **Transparent retry** — `lookup()` 返回 `None` 表示"数据正在提升，稍后再试"

```python
class SecondaryTierManager(ABC):
    def lookup(self, key, req_context) -> bool | None:    # 检查二级层
    def submit_store(self, key, data, req_context):       # 异步写入
    def submit_load(self, key, req_context):              # 异步读取
    def get_finished(self) -> list:                       # 获取完成的操作
    def touch(self, key, req_context):                    # 更新访问时间
```

### 3.3 Encoder Cache（多模态编码器缓存）

`vllm/v1/core/encoder_cache_manager.py` 管理多模态模型（如 LLaVA、Qwen-VL）的 Encoder 输出缓存：

- **缓存粒度**：以单个多模态输入 item（如图片）为单位，而非 encoder token
- **缓存 key**：`mm_hash`（多模态数据的哈希值）
- **淘汰策略**：LRU eviction — 最老的未被引用的缓存条目优先淘汰
- **内存管理**：细粒度的 slot 级管理，支持 chunked 多模态处理

```python
class EncoderCacheManager:
    cache_size: int                   # 总缓存容量（以 encoder embedding 计）
    num_free_slots: int               # 当前可用容量
    num_freeable_slots: int           # 可立即回收的容量（零引用的缓存条目）
    cached: dict[str, set[str]]       # mm_hash → 引用该缓存的 request_id 集合
    freeable: OrderedDict[str, int]   # LRU 序的可回收条目
```

**设计要点**：Encoder Cache 的淘汰发生在分配时（lazy eviction），而不是定期清理，与 KV Cache 的 block pool 保持一致的设计理念。

### 3.4 投机解码（Speculative Decoding）KV Cache 管理

投机解码场景下，KV Cache 管理需要处理 draft token 的特殊性：

**1. EAGLE/MTP lookahead token 的处理：**

```python
# allocate_slots 中:
num_tokens_need_slot = min(
    num_tokens_main_model + num_lookahead_tokens, self.max_model_len
)
# lookahead token 也需要分配 KV Cache slot

# 但缓存时排除 lookahead:
num_tokens_to_cache = min(
    total_computed_tokens + num_new_tokens,
    request.num_tokens,  # ← 只缓存到已验证的 token
)
```

**2. EAGLE 的 Prefix Cache 特殊处理：**

EAGLE 在 prefix cache 查找时会丢弃最后一个命中的 block，因为 EAGLE 需要该位置的 hidden states 来生成 draft tokens，所以必须重新计算：

```python
if use_eagle and computed_blocks[0]:
    for computed in computed_blocks:
        computed.pop()  # 丢弃最后一个命中 block
```

**3. EAGLE 的 Prefix Cache 查找扩展：**

在 HybridKVCacheCoordinator 中，EAGLE group 需要比 Full Attention 多匹配一个 block（因为最后一个要丢弃）：

```python
if use_eagle:
    _max_length = min(
        curr_hit_length + spec.block_size, max_cache_hit_length
    )
```

### 3.5 Chunked Prefill 的 KV Cache 交互

Chunked Prefill 将长 prompt 切分为多个 chunk 逐步处理，对 KV Cache 管理有几个关键影响：

**1. 全序列准入检查（Admission Gate）：**

```python
# allocate_slots 中的 full_sequence_must_fit 参数
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        ..., full_num_tokens, ...
    )
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # ★ 全序列放不下，拒绝调度，防止 OOM
```

这防止了"先接纳了请求的前几个 chunk，处理到一半发现显存不够"的尴尬局面。

**2. Recycling-aware Admission Cap：**

对于 SWA 和 ChunkedLocalAttention 模型，`max_admission_blocks_per_request` 限制了每个请求的最大 block 数。`remove_skipped_blocks` 在每个 chunk 的分配前运行，确保 `sum(peak_real_held) <= pool_size`。

```python
if apply_admission_cap and self._max_admission_blocks_per_request is not None:
    # 回收型 spec 在此处限制 per-request 分配上限
    num_required_blocks = min(
        num_required_blocks, self._max_admission_blocks_per_request
    )
```

### 3.6 KV Cache Events & Tracing 系统

vLLM V1 提供了一套完整的 KV Cache 事件系统（`vllm/distributed/kv_events.py`），用于外部监控和 KV Cache 共享：

**事件类型：**

| 事件 | 描述 |
|---|---|
| `BlockStored` | Block 被写入 prefix cache。包含 `block_hashes`、`parent_block_hash`、`token_ids`、`block_size`、`lora_name`、`extra_keys`、`group_idx`、`kv_cache_spec_kind`、`kv_cache_spec_sliding_window` |
| `BlockRemoved` | Block 从 prefix cache 中被移除。包含 `block_hashes`、`group_idx` |
| `AllBlocksCleared` | 所有 prefix cache 被重置（如 RLHF 权重更新后） |

**事件生产**：Scheduler 侧的 `BlockPool` 在 `cache_full_blocks()` 和 `_maybe_evict_cached_block()` 中发出事件。

**事件消费**：支持 ZMQ Publisher/Subscriber 模式：
- `ZmqEventPublisher`：通过 ZMQ PUB socket 发布事件，支持 replay buffer（`buffer_steps=10000`）
- `KVEventAggregator`：跨 worker 聚合事件，只返回所有 worker 都发出的事件（去重）
- `KVConnectorKVEvents`：外部 connector 的抽象接口

**启用方式**：通过 `KVEventsConfig` 配置，设置 `enable_kv_cache_events=True`。

### 3.7 Common Prefix Block 检测

`get_num_common_prefix_blocks()` 用于检测所有运行中请求的公共前缀块数：

```python
def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
    # 选一个运行中的请求，遍历其 blocks
    # 一个 block 是 common prefix block 的条件：
    # ref_cnt == 当前持有 KV Cache 的请求总数
    # （即所有请求都共享这个 block）
```

**用途**：
- **CUDA Graph 捕获优化**：当存在公共前缀时，可以使用更激进的 CUDA Graph 捕获策略
- **P/D 分离**：公共前缀 block 可以安全地在节点间共享，不需要每次传输
- **监控**：衡量系统 prompt 的复用程度

**注意**：存在已分配 KV Cache 但当前 step 未被调度的请求时，可能会错误返回 0（无法区分"不共享"和"有未被调度的请求不共享"）。

### 3.8 DCP/PCP（Context Parallelism）KV Cache 处理

Decoder Context Parallelism（DCP）和 Pipeline Context Parallelism（PCP）允许将超长序列的 KV Cache 分布在多个 worker 上：

**DCP 对 block_size 的影响：**

```python
# UnitaryKVCacheCoordinator.__init__()
if dcp_world_size > 1:
    self.block_size *= dcp_world_size  # block_size 扩展
if pcp_world_size > 1:
    self.block_size *= pcp_world_size
```

**DCP 对 slot_mapping 的影响：**

```python
# block_table.py - compute_slot_mapping()
# DCP 场景下，每个 rank 只负责一段 token range
# slot_mapping 计算需要带上 context_parallel_offset
offset = cp_rank * num_tokens_per_rank
```

**约束**：HybridKVCacheCoordinator 当前不支持 DCP/PCP（`assert dcp_world_size == 1`）。

### 3.9 CUDA Graph 与 KV Cache 集成

CUDA Graph 捕获整个模型 forward 的执行图进行回放，绕过了 kernel launch overhead。KV Cache 系统需要特殊处理：

**1. CUDA Graph 对 block_table 的要求：**

- CUDA Graph 回放时，输入张量的指针地址必须不变
- `UvaBackedTensor`（UVA 零拷贝 tensor）满足这个要求：CPU 侧修改，GPU 侧立即可见，指针不变
- 这就是为什么 GPU 优化版 `BlockTables` 使用 `UvaBackedTensor` 的根本原因

**2. Padding 对齐：**

```python
# _determine_batch_execution_and_padding()
# 决定是否需要用 padding 把 batch size 对齐到 CUDA Graph 的固定大小
num_tokens_padded = batch_desc.num_tokens
```

**3. CUDA Graph 的 warmup 与 KV Cache 的关系：**

- 在 profiling 阶段，`initialize_kv_cache` 的 `is_profiling=True` 路径会分配 profiling 专用的 KV Cache
- 正式初始化时（`is_profiling=False`），会释放 profiling 的 KV Cache 并重新分配正式的

### 3.10 Sleep/Wake 模式与 KV Cache 持久化

vLLM 支持模型休眠（Sleep）模式，将模型权重卸载到 CPU 以释放 GPU 显存，同时保留 KV Cache 状态。Wake 时重新加载权重，KV Cache 继续可用：

```python
# gpu_worker.py
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    # 处理 sleep 模式：
    # - block_table_layout_tensors 在 CuMem wake-up 后需要重新初始化
    self.model_runner.init_block_table_layout_tensors()
```

**注意事项**：
- KV Cache 的 GPU tensor 在 sleep 期间保留（不能释放）
- Wake 后，UVA tensor 的指针可能需要重建
- 前缀缓存的 hash map（CPU 侧）在 sleep 期间保留

### 3.11 KV Cache Metrics 可观测性

`vllm/v1/core/kv_cache_metrics.py` 提供了基于采样的 block 生命周期指标收集：

```python
class KVCacheMetricsCollector:
    sample_rate: float = 0.01           # 采样率 1%
    block_metrics: dict[int, BlockMetricsState]  # block_id → 指标

class BlockMetricsState:
    birth_time_ns: int                  # 创建时间
    last_access_ns: int                 # 最后访问时间
    access_history: deque[int]          # 最近 4 次访问时间

class KVCacheEvictionEvent:
    lifetime_seconds: float             # block 存活时间
    idle_seconds: float                 # 空闲时间（自上次访问）
    reuse_gaps_seconds: tuple[float, ...]  # 相邻访问之间的间隔
```

**用途**：
- 衡量 Prefix Cache 的有效性（命中率、复用间隔）
- 判断 eviction 策略是否合理（是否过早淘汰了热 block）
- 为容量规划提供数据支持

### 3.12 异构注意力架构总结

vLLM V1 的 KV Cache 框架原生支持在同一模型中混合多种注意力机制：

```python
class KVCacheSpecKind(str, Enum):
    FULL_ATTENTION = "full_attention"          # 全局注意力
    MLA_ATTENTION = "mla_attention"            # 多潜头注意力 (DeepSeek)
    SLIDING_WINDOW = "sliding_window"          # 滑动窗口注意力
    SLIDING_WINDOW_MLA = "sliding_window_mla"  # 带 MLA 的滑动窗口
    MAMBA = "mamba"                            # Mamba SSM（无传统 KV Cache）
    CROSS_ATTENTION = "cross_attention"        # 交叉注意力（编码器-解码器）
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"  # 分块局部注意力
    SINK_FULL_ATTENTION = "sink_full_attention"          # 带 sink token 的全局注意力
    ENCODER_ONLY_ATTENTION = "encoder_only_attention"    # 纯编码器注意力
```

新模型在不断推出，KV Cache 管理也在持续迭代，下一步的方向：
- **多节点 KV Cache 池化** — `kv_transfer` 目录下的 KV Connector 框架支持跨节点 KV 传输
- **KV Cache Events/Tracing** — `KVEventsConfig` 支持 KV Cache 事件的监控和跟踪
- **模型适配解耦** — 通过 `KVCacheSpec` 抽象层使新 Attention 架构的接入成本最小
- **更细粒度的量化** — NVFP4、per-token-head 量化持续演进
- **多级 Offload** — Tiering Offloading 支持 GPU → CPU → Storage/Network 三级缓存

