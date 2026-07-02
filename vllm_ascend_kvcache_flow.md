# vllm-ascend KV Cache 流程梳理（完整版）

> 基于 [vllm-ascend](https://github.com/vllm-project/vllm-ascend) 源码分析，对照上游 [vllm](https://github.com/vllm-project/vllm) GPU 实现。

---

## 〇、背景：GPU vs NPU 架构差异

### GPU (CUDA)
GPU 拥有海量且灵活的流处理器（CUDA Cores）。访存极其自由，几千个线程可以同时向显存中不同的、离散的物理地址发起读取请求。

### 昇腾 NPU (Ascend)
昇腾 NPU 的核心是矩阵乘加单元（Cube）和向量计算单元（Vector）。Cube 算力很强，但极其依赖 **DMA（直接内存访问）** 将数据整块、连续地从 HBM 搬运到 L1/L0 缓存中。**Cube 极度讨厌离散访存**。

### 对 KV Cache 的意义（vllm-ascend 的核心痛点）
**PagedAttention 本质上就是离散访存**，这天生与 NPU 的 Cube 架构不合。为了解决这个问题，NPU 底层的 PagedAttention 算子（通常融合在 FlashAttention 变体中）在执行前，往往需要通过底层的 DMA / Vector 单元做极其复杂的内部搬移，把离散的 Block 临时"拼接"成相对连续的数据，再喂给 Cube 计算。

---

## 一、架构总览

### 1.1 vllm-ascend 整体定位
vllm-ascend 是**华为昇腾 NPU 对 vLLM 的适配层**，设计理念是**最大化复用上游 vLLM 框架**，仅在 NPU 硬件特性不兼容处做定制。

### 1.2 KV Cache 核心组件地图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Scheduler 层                                   │
│  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐ │
│  │ KVCacheManager   │  │ RecomputeScheduler │  │ DynamicBatch      │ │
│  │ (完全复用上游)     │  │ (KV Transfer感知)  │  │ Scheduler         │ │
│  └────────┬─────────┘  └────────┬───────────┘  └────────┬─────────┘ │
│           │       Block 分配/释放 │                       │           │
├───────────┼─────────────────────┼───────────────────────┼───────────┤
│           ▼                     ▼                       ▼           │
│                      Worker 层 (Model Runner)                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  NPUInputBatch → MultiGroupBlockTable → BlockTable           │   │
│  │    ├─ CpuGpuBuffer (block_table): [max_reqs, max_blocks]     │   │
│  │    ├─ CpuGpuBuffer (slot_mapping): [max_tokens]              │   │
│  │    └─ num_blocks_per_row: [max_reqs] (numpy)                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               Attention Backend                                │   │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ reshape_and_cache   │  │ forward_impl                  │   │   │
│  │  │ (NPU DeviceOp)      │  │  ├─ PagedAttention (PA)       │   │   │
│  │  │                     │  │  └─ FusedInferAttention (FIA) │   │   │
│  │  └─────────────────────┘  └──────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         高级特性                                               │   │
│  │  ├─ KV Transfer (PD 分离, RDMA, 2MB 对齐)                     │   │
│  │  ├─ C8 INT8 量化 (per-tensor scale+offset, FIA 内反量化)      │   │
│  │  ├─ Hamming Sparse KV Compression (hashk topK 筛选)           │   │
│  │  ├─ ACLGraph (NPU 原生图捕获与更新)                            │   │
│  │  ├─ Sleep Mode (权重卸载/恢复, CaMemAllocator)                 │   │
│  │  └─ Context Parallel (DCP + PCP 双级交织)                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、阶段 1：系统初始化与显存预分配

### 2.1 平台配置入口：`check_and_update_config()`

**文件**: `vllm_ascend/platform.py` → `NPUPlatform.check_and_update_config()`

```python
@classmethod
def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
    # 步骤 1：自动检测量化方案（ascend / compressed-tensors）
    maybe_auto_detect_quantization(vllm_config)

    # 步骤 2：修正 NPU 不兼容的配置（如 cudagraph 模式降级）
    cls._fix_incompatible_config(vllm_config)

    # 步骤 3：初始化 Ascend 特有配置（xlite、fusion、EPLB、profiling_chunk 等）
    ascend_config = init_ascend_config(vllm_config)

    # 步骤 4：验证 KV Transfer 配置（PD 分离架构）
    if vllm_config.kv_transfer_config is not None:
        check_kv_extra_config(vllm_config)  # 验证 TP/DP 一致性
        vllm_config.kv_transfer_config.engine_id = f"{...}-{uuid4().hex}"

    # 步骤 5：稀疏 Attention 模型 → cache_dtype 设为 model dtype
    if model_config and hasattr(model_config.hf_text_config, "index_topk"):
        vllm_config.cache_config.cache_dtype = str(model_config.dtype)

    # ⭐ 步骤 6：强制 block_size = 128
    refresh_block_size(vllm_config)
```

### 2.2 Block Size 强制策略：`refresh_block_size()`

**文件**: `vllm_ascend/utils.py`

```python
def refresh_block_size(vllm_config):
    cache_config = vllm_config.cache_config
    scheduler_config = vllm_config.scheduler_config
    model_config = vllm_config.model_config

    # ① 未设置时默认 128
    if cache_config.block_size is None:
        cache_config.block_size = 128

    # ② Hybrid 模型（如 Mamba+Attention）：由模型自身逻辑决定，不强制
    if model_config.is_hybrid:
        return

    # ③ prefix caching 或 chunked prefill 启用时 → 强制 128
    if cache_config.block_size != 128:
        if cache_config.enable_prefix_caching or scheduler_config.enable_chunked_prefill:
            cache_config.block_size = 128

    # ④ xlite graph 启用时 → 强制 128
    if ascend_config.xlite_graph_config.enabled and cache_config.block_size > 128:
        cache_config.block_size = 128
```

**核心原因**: NPU 的 PagedAttention / FIA 算子对 `block_size=128` 有最优的硬件支持和 DMA 搬移效率。这不是任意选择，而是 NPU 硬件架构的强约束。

### 2.3 Model Runner 初始化：`NPUInputBatch` 和 `MultiGroupBlockTable` 创建

**文件**: `vllm_ascend/worker/model_runner_v1.py` → `NPUModelRunner.__init__()`

```python
# 首次创建 NPUInputBatch（此时真实 block_sizes 尚不明确，使用临时值）
self.input_batch = NPUInputBatch(
    max_num_reqs=...,
    max_model_len=...,
    max_num_batched_tokens=...,
    device=self.device,
    pin_memory=...,
    vocab_size=...,
    block_sizes=[self.block_size],                        # 临时值
    kernel_block_sizes=[[self.cache_config.block_size]],  # 临时值
    max_num_blocks_per_req=...,
    num_speculative_tokens=...,
    cp_kv_cache_interleave_size=...,
)
```

**`NPUInputBatch.__init__()` 内部**:
```python
# 创建 MultiGroupBlockTable
self.block_table = MultiGroupBlockTable(
    max_num_reqs=...,
    max_model_len=...,
    max_num_batched_tokens=...,
    block_sizes=block_sizes,           # 每个 KV cache group 的 block_size
    max_num_blocks=max_num_blocks_per_req,
    kernel_sizes=kernel_block_sizes,   # attention backend 确定的内核 block 大小
    num_speculative_tokens=...,
    cp_kv_cache_interleave_size=...,
)
```

**`MultiGroupBlockTable.__init__()` 内部**:
- 对每个 KV cache group 创建独立的 `BlockTable` 实例
- 每个 BlockTable 包含：
  - `CpuGpuBuffer(block_table)`: `[max_reqs, max_blocks_per_req]` int32
  - `CpuGpuBuffer(slot_mapping)`: `[max_tokens]` int32
  - `num_blocks_per_row`: `[max_reqs]` int32 (numpy)

### 2.4 模型加载 + BlockTable 重建（第二次）

**文件**: `vllm_ascend/worker/model_runner_v1.py` → `initialize_kv_cache()`

```python
def initialize_kv_cache(self, kv_cache_config: KVCacheConfig):
    kv_cache_config = deepcopy(kv_cache_config)  # ⭐ 深拷贝，因为会修改

    # ① 初始化 attention backend → 确定 kernel block sizes
    self.initialize_attn_backend(kv_cache_config)
    # → AscendAttentionBackend.get_supported_kernel_block_sizes() 返回 [128]

    self.use_hybrid_blocks = len(self.attn_groups) > 1

    # ② ⭐ 如果真实 block_sizes ≠ 临时值 → 重建 NPUInputBatch
    self.may_reinitialize_input_batch(kv_cache_config)
    # → 提取真实 block_sizes（来自 kv_cache_config）
    # → 提取 kernel_block_sizes（来自 attention backend）
    # → 重建 NPUInputBatch → 重建 MultiGroupBlockTable → 重建 BlockTable

    # ③ ⭐ 实际分配显存
    kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)
    # → _allocate_kv_cache_tensors() → torch.zeros(...) 分配
    # → _reshape_kv_cache_tensors() → view 为目标 shape
    # → bind_kv_cache() → 绑定到每层 Attention 的 key_cache/value_cache
```

### 2.5 显存分配策略：`_allocate_kv_cache_tensors()`

**文件**: `vllm_ascend/worker/model_runner_v1.py` → `_allocate_kv_cache_tensors()`

#### 策略一：Linear Attention / Mamba Hybrid（无 PD 分离）
```
单一 tensor，不分 K/V：
  tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=npu)
  所有 shared_by 内的 layer 共享同一个 tensor
```

#### 策略二：标准 Attention（无 PD 分离）
```
K/V 独立分配：
  k_tensor = torch.zeros(k_tensor_size, dtype=torch.int8, device=npu)
  v_tensor = torch.zeros(v_tensor_size, dtype=torch.int8, device=npu)
  split_factor = calc_split_factor([k_dim, v_dim])
  kv_cache_raw_tensors[layer_name] = (k_tensor, v_tensor)
```

#### 策略三：标准 Attention（有 PD 分离，kv_transfer_config 不为 None）
```
K/V 独立分配 + 2MB 地址对齐（支持 RDMA 传输）：
  k_tensor = torch.zeros(k_tensor_size + 2MB, dtype=torch.int8, device=npu)
  k_tensor = _align_memory(k_tensor, 2MB)[:k_tensor_size]
  v_tensor 同理
```

#### 策略四：DeepSeek V3.2 Sparse Attention（无 C8 量化）
```
三路拆分：
  kv_cache_raw_tensors[layer_name] = (
      k_tensor,        # dense K
      v_tensor,        # dense V
      dsa_k_tensor,    # sparse indexer K
  )
```

#### 策略五：DeepSeek V3.2 Sparse Attention（有 C8 量化）
```
四路拆分：
  kv_cache_raw_tensors[layer_name] = (
      k_tensor,            # dense K
      v_tensor,            # dense V
      dsa_k_tensor,        # sparse indexer K (INT8)
      dsa_k_scale_tensor,  # sparse C8 quant scale (FP16)
  )
```

### 2.6 显存重塑：`_reshape_kv_cache_tensors()`

```python
# 标准 Attention (FA / PA / FIA)
k_cache = raw_k_tensor.view(k_cache_dtype).view(num_blocks, block_size, num_kv_heads, head_size)
v_cache = raw_v_tensor.view(v_cache_dtype).view(num_blocks, block_size, num_kv_heads, head_size)

# MLA (Multi-head Latent Attention)
k_cache = raw_k_tensor.view(dtype).view(num_blocks, block_size, num_kv_heads, kv_lora_rank)  # nope
v_cache = raw_v_tensor.view(dtype).view(num_blocks, block_size, num_kv_heads, qk_rope_head_dim)  # rope

# Hybrid Blocks
kv_cache_shape = (num_blocks * block_size_chunk, block_size, num_kv_heads, head_size)

# Sparse C8
dsa_k_cache   = raw_dsa_k_tensor.view(torch.int8).view(num_blocks, block_size, num_kv_heads, index_head_dim)
dsa_k_scale   = raw_dsa_k_scale_tensor.view(torch.float16).view(num_blocks, block_size, num_kv_heads, 1)
```

### 2.7 KV Cache 绑定：`bind_kv_cache()`

将 reshape 后的 tensor 绑定到 forward_context 中每个 Attention 层的 `key_cache` / `value_cache` 属性上，使得 Attention 前向计算时可以直接访问对应的 KV cache tensor。

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **Block Size** | 默认 16，可灵活配置（8/16/32/...） | 强制 128（prefix caching / chunked prefill / xlite 启用时） | NPU 硬件约束，128 是 DMA 搬移最优粒度 |
| **Hybrid Block** | 不支持 | 物理 block (128) → 逻辑 block 拆分 | 新增能力，用于混合 block_size 场景 |
| **KV 存储 dtype** | fp16 / fp8 | int8（存储），view 为实际 dtype | 底层存储统一 int8，提高兼容性 |
| **K/V 分离** | V2 为 unified tensor | V1 始终 K/V 独立分配 | 为 PD 分离（RDMA）做准备，需独立地址对齐 |
| **PD 分离 2MB 对齐** | 不支持 | 2MB 地址对齐 | 支持跨节点 RDMA 传输 KV cache |
| **Sparse KV Cache** | 不支持 | DeepSeek V3.2 三路/四路拆分 | 新增 dsa_k_tensor + dsa_k_scale_tensor |
| **BlockTable 重建** | 一次初始化 | 两次初始化（临时→真实） | NPUInputBatch/BlockTable 先创建后重建 |
| **ACLGraph** | CUDA Graph | ACLGraph（NPU 原生） | 平台绑定差异 |
| **CaMemAllocator** | 不支持 | 支持 Sleep Mode 内存池管理 | 权重卸载/恢复的内存管理 |

---

## 三、阶段 2：请求接入与逻辑调度

### 3.1 Scheduler 侧：Block 分配（**完全复用上游**）

vllm-ascend 的 Scheduler 层面**完全复用上游 vLLM 的 KVCacheManager**，无任何修改：

```python
# RecomputeScheduler / SchedulerDynamicBatch 均继承上游
# 直接使用上游 KVCacheManager：
new_blocks = self.kv_cache_manager.allocate_slots(
    request, num_new_tokens, num_lookahead_tokens=...
)
# 返回 KVCacheBlocks：每个 KV cache group 一组 block_ids
```

**vllm-ascend 差异**：Scheduler 层面的 Block 分配/释放逻辑**零差异**。唯一的差异在于 `RecomputeScheduler` 是 KV Transfer（PD 分离）场景下的定制调度器，它需要感知 P/D 节点角色来协调 KV cache 的跨节点传输时机。

### 3.2 Worker 侧：Block Table 数据填充

**文件**: `vllm_ascend/worker/model_runner_v1.py` → `_prepare_inputs()`

```python
# ⭐ 第一步：Block Table CPU→GPU（提前启动，与后续 CPU 工作重叠）
self.input_batch.block_table.commit_block_table(num_reqs)
```

#### BlockTable 核心数据结构

**文件**: `vllm_ascend/worker/block_table.py`

```
BlockTable
├── block_table: CpuGpuBuffer [max_reqs, max_blocks_per_req] int32
│   ├── .np (numpy) — CPU 侧，numpy 数组
│   └── .gpu (torch.Tensor) — GPU 侧，torch tensor
├── slot_mapping: CpuGpuBuffer [max_tokens + 2 * pcp_size * max_reqs] int32
│   ├── .np (numpy)
│   └── .gpu (torch.Tensor)
├── num_blocks_per_row: np.ndarray [max_reqs] int32 — CPU 侧跟踪每行已填充块数
├── physical_block_size: int — 物理 block 大小
├── logical_block_size: int — 逻辑 block 大小（kernel block size）
├── blocks_per_phys_block: int — 物理→逻辑转换比例
└── use_hybrid_blocks: bool — 是否启用混合 block
```

#### BlockTable 行操作方法

```python
# 1. 请求首次分配 → add_row（先清空再写入）
def add_row(self, block_ids: list[int], row_idx: int) -> None:
    self.num_blocks_per_row[row_idx] = 0     # 计数归零
    self.append_row(block_ids, row_idx)

# 2. 增量追加（chunked prefill 后续分配）
def append_row(self, block_ids, row_idx: int) -> None:
    if self.use_hybrid_blocks:
        # ⭐ 物理块 → 逻辑块拆分：
        #   block_ids=[3] (physical block_size=128, logical_block_size=64)
        #   → [6, 7] (blocks_per_phys_block=2)
        block_ids = self._convert_physical_to_logical_blocks(block_ids)
    start = self.num_blocks_per_row[row_idx]
    self.block_table.np[row_idx, start:start+n] = block_ids

# 3. 请求完成/被抢占 → clear_row
def clear_row(self, row_idx: int) -> None:
    num_blocks = self.num_blocks_per_row[row_idx]
    self.block_table.np[row_idx, :num_blocks] = 0  # 清零
    self.num_blocks_per_row[row_idx] = 0

# 4. 请求删除后 compact → move_row
def move_row(self, src: int, tgt: int) -> None:
    n = self.num_blocks_per_row[src]
    self.block_table.np[tgt, :n] = self.block_table.np[src, :n]
    self.num_blocks_per_row[tgt] = n

# 5. 抢占时交换两行 → swap_row
def swap_row(self, src: int, tgt: int) -> None:
    self.num_blocks_per_row[src], self.num_blocks_per_row[tgt] = ...
    self.block_table.np[[src, tgt]] = self.block_table.np[[tgt, src]]
```

#### commit_block_table: numpy → GPU

```python
def commit_block_table(self, num_reqs: int) -> None:
    self.block_table.copy_to_gpu(num_reqs)  # 只复制前 num_reqs 行
```

### 3.3 Slot Mapping 计算

**文件**: `vllm_ascend/worker/block_table.py` → `compute_slot_mapping()`

```python
def compute_slot_mapping(self, num_reqs, query_start_loc, positions):
    num_tokens = positions.shape[0]
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank

    # GPU kernel：将 (request → token_position) 映射为 (block_id × block_size + offset)
    _compute_slot_mapping_kernel[(num_reqs + 1,)](...)  # 复用上游 CUDA kernel
```

**Kernel 内部逻辑**（伪代码）：
```
对于 token i（属于 request r，位置为 P）：

  ① block_idx   = P // block_size
  ② block_id    = block_table[r * stride + block_idx]   // GPU 侧读取
  ③ block_off   = P % block_size
  ④ slot = block_id * block_size + block_off

  // === CP 交织模式 ===
  if total_cp_world_size > 1:
      virtual_block_off = P % (block_size * total_cp_world_size)
      interleave_idx = virtual_block_off // cp_kv_cache_interleave_size
      if interleave_idx % total_cp_world_size != total_cp_rank:
          slot = PAD_SLOT_ID (-1)  // 非本地 token 设为 PAD
    
  slot_mapping[i] = slot
```

**CP 交织含义**：在 DCP+PCP 模式下，token 按照 `cp_kv_cache_interleave_size` 粒度在 cp_rank 之间交错存储。例如 interleave_size=1 时：
- token 0 → rank 0
- token 1 → rank 1
- token 2 → rank 0
- token 3 → rank 1

每个 rank 只存储自己负责的 token，非本地 token 的 slot_mapping 设为 `-1 (PAD_SLOT_ID)`。

#### draft model 的单独路径：`compute_slot_mapping_draft()`

```python
def compute_slot_mapping_draft(self, req_indices, positions):
    # draft model 的 token 可能跨多个 request 交错排列
    # E.g., req_indices = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # 需要独立的映射逻辑，支持 virtual_block_size（CP 模式）
    if self.dcp_world_size * self.pcp_world_size > 1:
        virtual_block_size = self.block_size * self.dcp_world_size * self.pcp_world_size
        logical_block_idx = positions // virtual_block_size
        block_table_indices = req_indices * max_num_blocks_per_req * blocks_per_phys_block + logical_block_idx
        block_numbers = self.block_table.np.ravel()[block_table_indices]
        mask = (virtual_offsets // interleave_size % cp_world_size) == current_rank
        slot_mapping = np.where(mask, block_numbers * block_size + block_offsets, -1)
```

### 3.4 `_prepare_inputs()` 中的完整调度流程

```python
def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    num_reqs = self.input_batch.num_reqs

    # ① commit block_table（提前到最开始，与后续 CPU 工作重叠）
    self.input_batch.block_table.commit_block_table(num_reqs)

    # ② 确定 attention state
    attn_state = self._build_attn_state(num_reqs, num_scheduled_tokens, num_valid_tokens)
    # → PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding

    # ③ 计算 positions
    positions_np = num_computed_tokens[req_indices] + query_pos[0:cu_num_tokens[-1]]

    # ④ CP 模式：先计算 pre-PCP slot_mapping（在 PCP split 之前）
    if self.pcp_size > 1:
        self.input_batch.block_table.compute_slot_mapping(num_reqs, pre_pcp_qsl, pre_pcp_positions)

    # ⑤ PCP split（更新 num_scheduled_tokens 等）
    if self.pcp_size > 1:
        num_scheduled_tokens = self.pcp_manager.update_tokens_for_pcp(...)

    # ⑥ 重新计算 positions + slot_mapping（PCP split 之后）
    # ...
    self.input_batch.block_table.compute_slot_mapping(num_reqs, query_start_loc, positions)
```

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **Scheduler / KVCacheManager** | 标准实现 | **完全复用**，无修改 | 零差异 |
| **BlockTable 数据载体** | V2: `StagedWriteTensor` / V1: `torch.Tensor` | `CpuGpuBuffer`（CPU numpy + GPU torch 双缓冲） | 数据结构差异，双缓冲更高效 |
| **Block ID 存储精度** | int64 | **int32** | 类型降级，NPU 算子偏好 int32 |
| **Hybrid Block 转换** | 无 | `_convert_physical_to_logical_blocks()` 物理→逻辑拆分 | 新增能力 |
| **commit 时机** | V2: staged write, lazy commit | V1: `_prepare_inputs` 最先 commit，与后续 CPU 工作重叠 | 调度策略差异 |
| **Slot Mapping Kernel** | Numba CUDA / Triton | 复用上游 CUDA kernel（V1） | 平台兼容 |
| **CP 交织存储** | 标准 CP（distributed） | DCP+PCP 双级交织，`cp_kv_cache_interleave_size` 粒度交错 | 新增能力 |
| **Speculative Draft Slot** | 标准处理 | `compute_slot_mapping_draft()` 单独处理（含 CP 的 `virtual_block_size`） | 追加路径 |
| **Attn State 枚举** | DECODE / PREFILL / CHUNKED_PREFILL | **5 种**：PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding | 更细粒度，用于精确选择 attention 路径 |
| **PAD_SLOT_ID** | `_PAD_SLOT_ID` (int64) | `-1` (int32) | 类型差异 |

---

## 四、阶段 3：实际计算（PagedAttention）

### 4.1 `reshape_and_cache`：K/V 写入

**文件**: `vllm_ascend/attention/attention_v1.py` → `AscendAttentionBackendImpl.reshape_and_cache()`

```python
def reshape_and_cache(self, query, key, value, kv_cache, attn_metadata, output):
    if self.key_cache is None:
        self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
    
    slots = attn_metadata.slot_mapping  # [num_tokens] int32

    DeviceOperator.reshape_and_cache(
        key=key[:num_actual_tokens],         # [num_tokens, num_kv_heads, head_size]
        value=value[:num_actual_tokens],
        key_cache=self.key_cache,            # [num_blocks, 128, num_kv_heads, head_size]
        value_cache=self.value_cache,
        slot_mapping=slots[:num_actual_tokens],  # int32
    )
```

**底层 NPU 算子写入逻辑**（`DeviceOperator.reshape_and_cache`）：
```
对于每个 token i：
  slot = slot_mapping[i]
  if slot != -1 (PAD):  # 跳过 padding 和非本地 CP token
      block_id = slot // block_size
      offset   = slot % block_size
      key_cache[block_id, offset, :, :] = key[i]     # 写入对应 block 的对应 slot
      value_cache[block_id, offset, :, :] = value[i]
```

**PD 分离特殊处理**：
```python
if self.is_kv_producer:
    # KV Producer 端记录 reshape_and_cache 完成事件
    attn_metadata.reshape_cache_event = torch.npu.Event()
    attn_metadata.reshape_cache_event.record()
    # 该事件供 KV transfer 层等待，确保 KV 数据写入完成后再传输
```

### 4.2 Attention 计算：双路径选择

**文件**: `vllm_ascend/attention/attention_v1.py` → `AscendAttentionBackendImpl.forward_impl()`

```python
def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
    num_tokens = query.shape[0]

    if (attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        and using_paged_attention(num_tokens, self.vllm_config)
        and self.sliding_window is None):
        
        # 路径 A：PagedAttention（decode 专用）
        output = self.forward_paged_attention(query, attn_metadata, output)
    else:
        # 路径 B：FusedInferAttention（prefill / 混合 batch）
        output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)
    return output
```

#### PA 触发条件：`using_paged_attention()`

```python
def using_paged_attention(num_tokens: int, vllm_config) -> bool:
    # 条件 1：用户手动指定 pa_shape_list
    if pa_shape_list:
        return num_tokens in pa_shape_list
    # 条件 2：num_tokens ≤ SEQ_LEN_WITH_MAX_PA_WORKSPACE (6144)
    return num_tokens <= SEQ_LEN_WITH_MAX_PA_WORKSPACE
```

加上 forward_impl 中的条件：**DecodeOnly + 非 sliding window + 非 A5 设备**。

#### 路径 A：PagedAttention（`_npu_paged_attention`）

```python
def forward_paged_attention(self, query, attn_metadata, output):
    torch_npu._npu_paged_attention(
        query=query,                     # [num_reqs, num_heads, head_size]
        key_cache=self.key_cache,        # [num_blocks, 128, num_kv_heads, head_size]
        value_cache=self.value_cache,
        num_kv_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale_value=self.scale,
        block_table=attn_metadata.block_tables,  # [num_reqs, max_blocks] int32
        context_lens=attn_metadata.seq_lens,     # [num_reqs]
        out=output,
    )
```

**PA 算子特点**：
- 传统 PagedAttention 实现，通过 `block_table` 索引离散的 KV block
- NPU 底层通过 DMA 将离散 block 临时拼接为连续数据，再喂给 Cube 计算
- 在 decode 小 batch 场景下比 FIA 更优

#### 路径 B：FusedInferAttention（`npu_fused_infer_attention_score`）

```python
def forward_fused_infer_attention(self, query, key, value, attn_metadata, output, kv_cache):
    key, value, block_size, block_table, actual_seq_lengths_kv = \
        self._get_fia_params(key, value, attn_metadata, kv_cache)

    # Hamming Sparse 处理（prefill 阶段写入 hashk，decode 阶段 topK 筛选）
    if self.enable_hamming_sparse and attn_state != DecodeOnly:
        reshape_and_cache_kvcomp(...)   # 写入 hashk_cache
    elif self.enable_hamming_sparse:
        block_table, actual_seq_lengths_kv = get_kvcomp_decode_params(...)  # topK 筛选

    # 选择 layout 和 extra_args
    input_layout = "TND"      # Token-Head-Dim (默认)
    sparse_mode = 3           # 3=causal, 0=no mask
    extra_args = {}

    if self.enable_c8_quant:
        # C8 INT8 KV Cache 反量化参数
        extra_args = {
            "key_antiquant_scale": layer._c8_k_aq_scale,
            "key_antiquant_offset": layer._c8_k_aq_offset,
            "value_antiquant_scale": layer._c8_v_aq_scale,
            "value_antiquant_offset": layer._c8_v_aq_offset,
            "key_antiquant_mode": 0,
            "value_antiquant_mode": 0,
        }
        input_layout = "BNSD"   # C8 模式下必须用 BNSD layout
        sparse_mode = 0         # causal mask 在 FIA 内部处理

    # 调用 NPU FIA 算子
    torch_npu.npu_fused_infer_attention_score(
        query=query,
        key=key,              # ← key_cache 视图
        value=value,          # ← value_cache 视图
        block_table=block_table,
        input_layout=input_layout,
        block_size=block_size,
        actual_seq_lengths=actual_seq_lengths_q,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        num_key_value_heads=self.num_kv_heads,
        num_heads=self.num_heads,
        scale=self.scale,
        sparse_mode=sparse_mode,
        **extra_args,
    )
```

**FIA 算子特点**：
- NPU 原生融合 attention 算子（融合了 QK 计算、softmax、PV 计算）
- 支持 `block_table` 索引的 Paged KV Cache
- 通过 `input_layout` 支持 TND（Token-Head-Dim）和 BNSD（Batch-Num_heads-Seq-Dim）
- C8 量化模式下在内部分完成 INT8 → FP16 反量化

### 4.3 C8 INT8 KV Cache 反量化详解

**量化存储**：KV cache 以 INT8 精度存储在 HBM 中
- `k_cache_scale` / `k_cache_offset`：per-channel 量化参数
- `v_cache_scale` / `v_cache_offset`：per-channel 量化参数

**反量化时机**：在 FIA 算子内部，读取 INT8 KV cache 后即时反量化为 FP16，再参与 attention 计算。

```python
# AscendC8AttentionBackendImpl._prepare_c8_scales()
def _prepare_c8_scales(self, layer, device):
    # 将 per-channel scale/offset 转换为 BNSD 格式供 FIA 使用
    # BNSD = (1, num_kv_heads, 1, head_size)
    layer._c8_k_aq_scale  = layer._c8_k_scale.view(1, H, 1, D).contiguous()
    layer._c8_k_aq_offset = layer._c8_k_offset.view(1, H, 1, D).contiguous()
    layer._c8_v_aq_scale  = layer._c8_v_scale.view(1, H, 1, D).contiguous()
    layer._c8_v_aq_offset = layer._c8_v_offset.view(1, H, 1, D).contiguous()
```

### 4.4 Hamming Sparse KV Compression

**用途**：DeepSeek V3.2 等稀疏 attention 模型的 KV cache 压缩。

**Prefill 阶段**：
```python
def reshape_and_cache_kvcomp(kvcomp_meta, layer_index, key):
    # ① 计算 key 的 hash
    hashk = hash_encoder.compute_hash(key[:num_tokens])
    # hashk shape: [num_tokens, num_kv_heads, hash_bits // 8]

    # ② 存储到 hashk_cache（类似 KV cache 的 block 结构）
    torch.ops._C_ascend.npu_reshape_and_cache_bnsd(
        hashk, hashk_cache, slot_mapping, ...
    )
```

**Decode 阶段**：
```python
def get_kvcomp_decode_params(layer_index, kvcomp_meta, query, key, block_table, seq_lens):
    # ① 计算 query 的 hash
    hashq = hash_encoder.compute_hash(query)

    # ② Hamming 距离 topK 筛选
    new_block_table = torch.ops._C_ascend.npu_hamming_dist_top_k(
        hashq, hashk_cache, ...
        topk_for_hamming_full,  # 每个 layer 的 topK 比例
        seq_lens_gpu,
        chunk_sizes,
        sink, recent,           # sink/recent token 保留策略
        block_table,            # 原始 block_table
        output_block_table,     # 输出：筛选后的 block_table
    )

    # ③ 重新计算 seq_lens（基于筛选后的 block_table）
    new_seq_lens = compute_seq_lens(new_block_table, chunk_size, top_k_ratio)

    return new_block_table, new_seq_lens
```

### 4.5 ACLGraph 捕获与更新

**捕获阶段**（`full_graph_fia()` / `full_graph_pa()`）：

```python
def full_graph_fia(self, query, key, value, attn_metadata, output, layer=None):
    # ① 记录 layer 级别的 attn_params / handles / events
    stream = torch_npu.npu.current_stream()
    event = torch.npu.ExternalEvent()
    event.wait(stream)
    
    # ② graph_task_group_begin → FIA 执行 → graph_task_group_end
    torch.npu.graph_task_group_begin(stream)
    torch_npu.npu_fused_infer_attention_score.out(...)
    handle = torch.npu.graph_task_group_end(stream)
    
    # ③ 保存参数引用供 update 阶段使用
    graph_params.attn_params[num_tokens].append(attn_params)
    graph_params.handles[num_tokens].append(handle)
    graph_params.events[num_tokens].append(event)
```

**更新阶段**（`update_graph_params()`）：

```python
@staticmethod
def update_graph_params(update_stream, forward_context, num_tokens, ...):
    if using_paged_attention(num_tokens, vllm_config):
        # PA 更新：更新 seq_lens 和 workspace
        for key, param, handle, event in zip(...):
            seq_lens = forward_context.attn_metadata[key].seq_lens
            workspace = torch_npu._npu_paged_attention_get_workspace(...)
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(...)
            torch.npu.graph_task_update_end(update_stream)
    else:
        # FIA 更新：更新 seq_lens, block_tables, query 长度
        for key, param, handle, event in zip(...):
            seq_lens = attn_metadata[key].seq_lens_list
            actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
            block_tables = attn_metadata[key].block_tables
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(...)
            torch.npu.graph_task_update_end(update_stream)
```

### 4.6 完整 Forward 流程

```
AscendAttentionBackendImpl.forward()
│
├─① reshape_and_cache()
│   └─ DeviceOperator.reshape_and_cache()  ← K/V 写入 KV cache
│      └─ (如果是 KV Producer) record reshape_cache_event
│
├─② (如果是 pooling model) _forward_encoder_attention()
│   └─ npu_fusion_attention()  ← 无 block_table 的标准 attention
│
└─③ forward_impl()
    ├─ DecodeOnly + PA 条件满足
    │   └─ forward_paged_attention()
    │       ├─ 正常模式：torch_npu._npu_paged_attention()
    │       └─ Graph 捕获模式：full_graph_pa()
    │
    └─ 否则
        └─ forward_fused_infer_attention()
            ├─ (Hamming Sparse Prefill) reshape_and_cache_kvcomp()
            ├─ (Hamming Sparse Decode) get_kvcomp_decode_params()
            ├─ (C8 Quant) _forward_c8_decode() / _forward_c8_chunked_prefill()
            ├─ 正常模式：torch_npu.npu_fused_infer_attention_score()
            └─ Graph 捕获模式：full_graph_fia()
```

### 4.7 Attention State 状态机

vllm-ascend 使用 5 种 attention 状态（`AscendAttentionState`）：

```
PrefillNoCache  →  PrefillCacheHit  →  ChunkedPrefill  →  DecodeOnly
                                                              ↓
                                                        SpecDecoding
```

| 状态 | 含义 | 触发条件 |
|------|------|----------|
| `PrefillNoCache` | 首次 prefill，cache 为空 | 新请求的首次 prefill |
| `PrefillCacheHit` | Prefill 命中已有 cache | chunked prefill 非首次 chunk |
| `ChunkedPrefill` | Splitfuse 混合 batch | scheduler.enable_chunked_prefill |
| `DecodeOnly` | 纯 decode | 所有请求 num_tokens == 1 |
| `SpecDecoding` | 投机解码 | speculative_config 启用 |

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **reshape_and_cache** | CUDA kernel（`reshape_and_cache_flash`） | NPU `DeviceOperator.reshape_and_cache`（slot_mapping int32） | 算子级平台绑定 |
| **Attention 主算子** | FlashAttention-2 / FlashInfer | `npu_fused_infer_attention_score` (FIA) 或 `_npu_paged_attention` (PA) | 算子级平台绑定 |
| **PA 触发条件** | decode + 混合条件 | decode + FULL_DECODE_ONLY graph + 非 speculative + 非 A5 + num_tokens ≤ 6144 | 条件更严格 |
| **FIA Layout** | N/A | TND（默认）/ **BNSD**（C8 量化时切换） | NPU 特有 |
| **KV Cache dtype** | fp16 / fp8 | int8 C8，**per-tensor scale+offset** 反量化在 FIA 内部完成 | 量化方案差异 |
| **Graph 捕获** | CUDA Graph | **ACLGraph**（NPU 原生），`graph_task_update_begin/end` 更新 batch 参数 | 平台绑定差异 |
| **Graph 参数结构** | 相对简单 | 每个 token 数量独立维护 **attn_params/handles/events/workspaces** | 实现复杂度更高 |
| **KV 压缩** | 无 | **Hamming Sparse** → hashk_cache 写入（prefill）+ 读取（decode topK 选择） | 新增能力 |
| **Attn State** | DECODE / PREFILL / CHUNKED_PREFILL | **5 种**：PrefillNoCache / PrefillCacheHit / DecodeOnly / ChunkedPrefill / SpecDecoding | 更细粒度 |
| **slot_mapping 类型** | int64 fill(-1) | **int32** fill(-1)，CP 模式下非本地 token 也设为 PAD | 类型+语义差异 |
| **Sliding Window** | 标准 | 额外 `_forward_fia_slidingwindow()` 路径，使用 `npu_fused_infer_attention_score_v2` | 路径追加 |
| **Cross Attention** | 标准 | slot_mapping override 为 int32（规避上游 bug） | 兼容性修复 |

---

## 五、阶段 4：KV Transfer（PD 分离架构）

### 5.1 架构概述

**PD 分离（Prefill-Decode Disaggregation）**：
- **P 节点（KV Producer）**：执行 prefill，产生 KV cache，通过高速网络发送给 D 节点
- **D 节点（KV Consumer）**：接收 KV cache，执行 decode

**核心组件目录**：`vllm_ascend/distributed/kv_transfer/`

```
KV Transfer 层
├── ascend_multi_connector.py       # 多连接器聚合管理
├── kv_p2p/                         # P2P 直连传输（同机）
├── kv_pool/
│   └── ascend_store/
│       └── pool_worker.py          # 连接池 Worker（Mooncake / 自定义 backend）
└── utils/                          # 工具函数
```

### 5.2 显存对齐策略

**2MB 对齐的必要性**：RDMA 传输要求内存页对齐，2MB 是大页（Huge Page）标准大小。

```python
# _allocate_kv_cache_tensors() 中的 PD 分离分配路径
if self.vllm_config.kv_transfer_config is not None:
    # K/V 各自独立分配 + 2MB 地址对齐
    k_tensor = torch.zeros(k_tensor_size + alignment, dtype=torch.int8, device=npu)
    k_tensor = self._align_memory(k_tensor, alignment)[:k_tensor_size]
    v_tensor = torch.zeros(v_tensor_size + alignment, dtype=torch.int8, device=npu)
    v_tensor = self._align_memory(v_tensor, alignment)[:v_tensor_size]
```

### 5.3 PD TP Ratio 计算

**文件**: `vllm_ascend/ascend_config.py`

```python
# 从 kv_transfer_config 读取 P/D 节点的 TP 配置
prefill_tp_size = kv_transfer_config.get_from_extra_config("prefill", {"tp_size": 1})["tp_size"]
decode_tp_size  = kv_transfer_config.get_from_extra_config("decode", {"tp_size": 1})["tp_size"]

# 计算 ratio
self.pd_tp_ratio = prefill_tp_size // decode_tp_size
self.pd_head_ratio = prefill_tp_size // decode_tp_size  # KV head 维度的 ratio
```

### 5.4 KV Transfer 生命周期

```
P 节点 (Producer)                          D 节点 (Consumer)
─────────────────                          ─────────────────
① prefill + reshape_and_cache
   写入 K/V 到本地 KV cache
   
② record reshape_cache_event              ③ RecomputeScheduler 调度
   (标记 KV 数据就绪)                         (等待 KV transfer 完成)
   
④ Connector.send()                        ⑤ Connector.recv()
   (通过 RDMA/P2P 发送 K/V block)            (接收 K/V block 到本地 cache)
                                              
                                          ⑥ decode (使用接收到的 KV cache)
```

### 5.5 RecomputeScheduler

**文件**: `vllm_ascend/core/recompute_scheduler.py`

RecomputeScheduler 是 KV Transfer 场景下的定制调度器：
- **KV Consumer 端**：等待 KV transfer 完成后才将请求加入 decode batch
- **KV Producer 端**：prefill 完成后触发 KV 发送
- 支持异步调度模式（`AsyncRecomputeScheduler`）

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **PD 分离** | 实验性支持 | **完整支持**，含 2MB 对齐、多种 Connector | 更成熟的生产级实现 |
| **K/V 分离分配** | V2 unified tensor | **始终 K/V 独立** | 为 RDMA 独立传输做准备 |
| **地址对齐** | 无 | **2MB 大页对齐** | RDMA 传输要求 |
| **RecomputeScheduler** | 无 | **新增**，KV transfer 感知调度 | PD 分离专属 |
| **Connector 类型** | 基础 | Mooncake / P2P / 自定义 backend | 多种传输后端 |

---

## 六、阶段 5：显存置换、抢占和回收

### 6.1 Block 释放：Scheduler 侧（**完全复用上游**）

```python
# vllm-ascend 的 Scheduler 完全复用上游 KVCacheManager.free()
self.kv_cache_manager.free(request)
# 物理 block_ids 回到 free list，可供后续请求分配
```

**vllm-ascend 零差异**：Scheduler 层面的 Block 分配/释放逻辑无任何修改。

### 6.2 Worker 侧：BlockTable 清理

```python
# 请求完成或被抢占后
self.input_batch.block_table.clear_row(row_idx)
# → block_table.np[row_idx, :num_blocks] = 0
# → num_blocks_per_row[row_idx] = 0

# compact（压缩空洞）
self.input_batch.block_table.move_row(src, tgt)
# → 将 src 行数据移动到 tgt 行

# 抢占时交换
self.input_batch.block_table.swap_row(src, tgt)
# → 交换两行的 block_table 数据
```

### 6.3 KV Cache Block 物理交换：`swap_blocks` / `copy_blocks`

**文件**: `vllm_ascend/attention/attention_v1.py`

```python
@staticmethod
def swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst):
    # 跨 device/跨 cache 的 block 交换（用于 offload / preemption）
    src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
    dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
    src_indices = src_to_dst[:, 0]
    dst_indices = src_to_dst[:, 1]

    # 直接 tensor 拷贝（GPU→GPU 或 CPU→GPU）
    dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
    dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

@staticmethod
def copy_blocks(kv_caches, src_to_dists):
    # 同一 cache 内的 block 拷贝（用于 defragmentation）
    src_indices = src_to_dists[:, 0]
    dst_indices = src_to_dists[:, 1]

    for kv_cache in kv_caches:
        key_caches = kv_cache[0]
        value_caches = kv_cache[1]
        key_caches[dst_indices] = key_caches[src_indices]
        value_caches[dst_indices] = value_caches[src_indices]
```

### 6.4 抢占完整流程

```
Scheduler 检测显存压力
│
├─ 选择被抢占的 request
│
├─ KVCacheManager 确定 src_to_dst 映射
│   ├─ swap_to_cpu: src_to_dst = [(gpu_block_id, cpu_block_id), ...]
│   └─ swap_to_gpu: src_to_dst = [(cpu_block_id, gpu_block_id), ...]
│
├─ AttentionBackend.swap_blocks()
│   └─ 实际执行 GPU↔CPU 的 KV cache block 拷贝
│
├─ BlockTable.swap_row()
│   └─ 更新 BlockTable 中的 block_id 映射
│
└─ 下一轮 _prepare_inputs()
    └─ 重新 compute_slot_mapping（使用更新后的 block_table）
```

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **KVCacheManager** | 标准实现 | **完全复用** | 零差异 |
| **swap_blocks** | CUDA copy / D2D | `torch.Tensor.to(device)` 标准拷贝 | 实现方式一致 |
| **copy_blocks** | CUDA copy | `torch.Tensor[...] = ...` 标准拷贝 | 实现方式一致 |
| **BlockTable 行操作** | 类似 | `clear_row` / `move_row` / `swap_row` | 语义一致，实现细节不同 |
| **抢占策略** | 上游控制 | 上游控制（算子层面无定制） | 无差异 |

---

## 七、阶段 6：Sleep Mode（权重卸载/恢复）

### 7.1 概述

Sleep Mode 允许模型权重在空闲时从 NPU 显存卸载到 CPU 内存，需要时再恢复，从而腾出显存供其他实例使用。

### 7.2 初始化

```python
# worker.py: NPUWorker.__init__()
if vllm_config.model_config.enable_sleep_mode:
    self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

# platform.py
def is_sleep_mode_available(self) -> bool:
    return True  # NPU 平台支持 Sleep Mode
```

### 7.3 模型加载（使用 CaMemAllocator 内存池）

```python
# worker.py: load_model()
if self.vllm_config.model_config.enable_sleep_mode:
    allocator = CaMemAllocator.get_instance()
    assert allocator.get_current_usage() == 0
    context = allocator.use_memory_pool(tag="weights")  # 使用 "weights" 标签的内存池
else:
    context = nullcontext()

with context:
    self.model_runner.load_model()
```

### 7.4 KV Cache 分配（使用 CaMemAllocator 内存池）

```python
# worker.py: initialize_from_config()
if self.vllm_config.model_config.enable_sleep_mode:
    allocator = CaMemAllocator.get_instance()
    context = allocator.use_memory_pool(tag="kv_cache")  # 使用 "kv_cache" 标签的内存池
else:
    context = nullcontext()

with context:
    self.model_runner.initialize_kv_cache(kv_cache_config)
```

### 7.5 Sleep / Wake 流程

```python
# worker.py: sleep()
def sleep(self, level: int = 1) -> None:
    free_bytes_before_sleep = torch.npu.mem_get_info()[0]
    
    if level == 2:
        # Level 2: 保存所有 buffer 到 CPU，卸载所有权重
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone() 
            for name, buffer in model.named_buffers()
        }
        # ... 触发 CaMemAllocator 的内存释放

# worker.py: wake()
def wake(self, ...) -> None:
    # 恢复权重和 buffer 到 NPU
    # ... 从 CaMemAllocator 内存池重新分配并加载
```

**Sleep Level 语义**：
- **Level 1**：轻度睡眠，保留权重在显存中，仅释放 KV cache 等临时分配
- **Level 2**：深度睡眠，权重 + KV cache 全部卸载到 CPU，最大化释放 NPU 显存

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **Sleep Mode** | CUDA IPC / 显存卸载 | **CaMemAllocator** 内存池管理 | 平台绑定差异，NPU 使用华为自研内存分配器 |
| **内存池标签** | 无 | `"weights"` / `"kv_cache"` 标签隔离 | 更精细的内存池管理 |
| **Buffer 保存** | 类似 | `named_buffers()` → `.cpu().clone()` | 实现方式一致 |
| **Level 分级** | 类似概念 | Level 1 / Level 2 | 语义一致 |

---

## 八、阶段 7：Context Parallel（DCP + PCP 双级交织）

### 8.1 概述

vllm-ascend 支持两种 Context Parallel 模式：
- **DCP（Decode Context Parallel）**：decode 阶段的 KV cache 分布式存储
- **PCP（Prefill Context Parallel）**：prefill 阶段的序列维度切分

两者可以组合使用，形成 **DCP + PCP 双级交织**。

### 8.2 交织存储格式

```
假设 dcp_world_size=2, pcp_world_size=1, cp_kv_cache_interleave_size=1

Token 序列：[T0, T1, T2, T3, T4, T5, T6, T7]
              │   │   │   │   │   │   │   │
Rank 0:      [T0,     T2,     T4,     T6    ]  ← 偶数位置
Rank 1:      [    T1,     T3,     T5,     T7]  ← 奇数位置

每个 rank 只存储自己负责的 token 的 KV cache
```

### 8.3 Slot Mapping 中的 CP 处理

```python
# BlockTable.compute_slot_mapping() 中的 CP 逻辑
virtual_block_off = position % (block_size * dcp_world_size * pcp_world_size)
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank

interleave_idx = virtual_block_off // cp_kv_cache_interleave_size
if interleave_idx % (dcp_world_size * pcp_world_size) != total_cp_rank:
    slot_mapping[token_i] = PAD_SLOT_ID  # -1，非本地 token
```

### 8.4 CP 接口切换

```python
# AscendAttentionBackend 根据 CP 启用状态切换接口实现
@staticmethod
def get_impl_cls():
    if enable_cp():
        return AscendAttentionCPImpl      # CP 专用实现
    return AscendAttentionBackendImpl     # 标准实现

@staticmethod
def get_builder_cls():
    if enable_cp():
        return AscendAttentionCPMetadataBuilder  # CP 专用 metadata builder
    return AscendAttentionMetadataBuilder
```

---

### 🔄 本阶段 vllm-ascend vs 上游 vLLM 差异总结

| 维度 | 上游 vLLM (CUDA) | vllm-ascend (NPU) | 差异说明 |
|------|------------------|-------------------|----------|
| **CP 模式** | DCP only | **DCP + PCP 双级交织** | 新增 PCP（Prefill Context Parallel） |
| **交织粒度** | 基础 | `cp_kv_cache_interleave_size` 可配置 | 更灵活的交织粒度控制 |
| **CP 接口切换** | 标准 | `get_impl_cls()` / `get_builder_cls()` 动态切换 | 架构更模块化 |
| **slot_mapping 处理** | 基础 | PCP pre-split slot_mapping + post-split 重新计算 | 分阶段处理 |

---

## 九、关键数据结构速查

### 9.1 AscendMetadata（Attention Metadata）

```python
@dataclass
class AscendMetadata:
    # 基础属性
    attn_state: AscendAttentionState  # 当前 attention 状态
    num_actual_tokens: int            # 实际 token 数（不含 padding）
    num_decode_tokens: int            # decode token 数
    num_prefills: int                 # prefill 请求数
    num_decodes: int                  # decode 请求数

    # 序列信息
    seq_lens: torch.Tensor            # [num_reqs] 每个请求的序列长度
    seq_lens_cpu: torch.Tensor        # CPU 侧镜像
    seq_lens_list: list[int]          # Python list 格式
    actual_seq_lengths_q: list[int]   # 每个请求的 query 长度

    query_start_loc: torch.Tensor     # [num_reqs+1] 累计 token 位置

    # KV Cache 相关
    block_tables: torch.Tensor        # [num_reqs, max_blocks_per_req] int32
    slot_mapping: torch.Tensor        # [num_tokens] int32

    # CP 相关
    prefill: AscendMetadataForPrefill | None
    decode_meta: AscendMetadataForDecode | None

    # 高级特性
    causal: bool = True
    kvcomp_metadata: KVCompMetaData | None  # Hamming Sparse 元数据
    reshape_cache_event: torch.npu.Event    # PD 分离 reshape 完成事件
```

### 9.2 BlockTable 尺寸计算

```
max_num_blocks_per_req = ceil(max_model_len / block_size)

逻辑 table 大小:
  非 hybrid: max_num_blocks_per_req
  hybrid:    max_num_blocks_per_req * blocks_per_phys_block

CP 模式下 (dcp * pcp > 1):
  block_table 大小 *= (1 + num_speculative_tokens)  ← 为 spec decode 复制
  slot_mapping 大小 += 2 * pcp_world_size * max_num_reqs
```

---

## 十、完整调用链路总结

```
┌─ 启动 ─────────────────────────────────────────────────────────────┐
│ ① check_and_update_config()                                        │
│    ├─ refresh_block_size() → block_size = 128                      │
│    ├─ init_ascend_config() → 读取所有 AscendConfig                 │
│    └─ 验证 kv_transfer_config                                     │
│                                                                    │
│ ② NPUModelRunner.__init__()                                       │
│    └─ 创建 NPUInputBatch → MultiGroupBlockTable → BlockTable[]    │
│                                                                    │
│ ③ initialize_kv_cache(kv_cache_config)                             │
│    ├─ may_reinitialize_input_batch() → 重建 BlockTable (真实参数)  │
│    ├─ _allocate_kv_cache_tensors() → torch.zeros() 分配显存        │
│    ├─ _reshape_kv_cache_tensors() → view 为目标 shape              │
│    └─ bind_kv_cache() → 绑定到 Attention 层                        │
└────────────────────────────────────────────────────────────────────┘

┌─ 每轮推理 ─────────────────────────────────────────────────────────┐
│ ④ Scheduler.schedule()                                            │
│    └─ KVCacheManager.allocate_slots() → 分配 block_ids             │
│                                                                    │
│ ⑤ _prepare_inputs()                                                │
│    ├─ commit_block_table() → numpy → GPU                          │
│    ├─ _build_attn_state() → PrefillNoCache / DecodeOnly / ...     │
│    ├─ 计算 positions                                              │
│    ├─ compute_slot_mapping() → GPU kernel                         │
│    └─ 构建 AscendMetadata                                          │
│                                                                    │
│ ⑥ model.forward()                                                 │
│    └─ 每层 Attention.forward()                                    │
│        ├─ reshape_and_cache() → NPU DeviceOp 写入 K/V             │
│        └─ forward_impl()                                          │
│            ├─ PA: torch_npu._npu_paged_attention()                │
│            └─ FIA: torch_npu.npu_fused_infer_attention_score()    │
│                                                                    │
│ ⑦ Scheduler.free(request)                                         │
│    └─ KVCacheManager.free() → block_ids 回到 free list            │
└────────────────────────────────────────────────────────────────────┘
```

---

## 十一、总结：vllm-ascend 对上游 vLLM 的继承与扩展

| 层级 | 继承上游 | 定制扩展 |
|------|----------|----------|
| **Scheduler** | KVCacheManager（allocate/free）**完全复用** | RecomputeScheduler（KV Transfer 感知调度） |
| **BlockTable** | 核心概念（block_table, slot_mapping） | `CpuGpuBuffer` 双缓冲、int32 类型、Hybrid Block 转换、CP 交织存储 |
| **Attention** | 双路径思想（PA / FIA） | NPU 原生算子（`_npu_paged_attention` / `npu_fused_infer_attention_score`）、C8 INT8 量化、Hamming Sparse、ACLGraph |
| **显存管理** | KVCacheManager 分配策略 | `_allocate_kv_cache_tensors()`：K/V 独立分配、2MB RDMA 对齐、Sparse KV 多路拆分 |
| **Graph** | CUDA Graph 思想 | ACLGraph：NPU 原生图捕获、`graph_task_update_begin/end`、per-token batch 参数管理 |
| **高级特性** | Sleep Mode 概念 | CaMemAllocator 内存池、DCP+PCP 双级交织、PD 分离完整支持 |

---

## 附录：关键文件索引

| 文件 | 作用 |
|------|------|
| `vllm_ascend/platform.py` | 平台配置修正（含 block_size 强制） |
| `vllm_ascend/utils.py` | `refresh_block_size()` |
| `vllm_ascend/ascend_config.py` | AscendConfig 完整配置对象 |
| `vllm_ascend/worker/model_runner_v1.py` | NPU Model Runner（`NPUModelRunner`） |
| `vllm_ascend/worker/npu_input_batch.py` | NPU InputBatch（`NPUInputBatch`） |
| `vllm_ascend/worker/block_table.py` | BlockTable + MultiGroupBlockTable |
| `vllm_ascend/attention/attention_v1.py` | Ascend Attention Backend |
| `vllm_ascend/attention/utils.py` | `AscendCommonAttentionMetadata`, `using_paged_attention()` |
| `vllm_ascend/attention/kvcomp_attn/attention_utils.py` | KV Compression（Hamming Sparse） |
| `vllm_ascend/worker/kvcomp_utils.py` | KVCompMetaData 初始化 |
| `vllm_ascend/worker/worker.py` | `NPUWorker`（含 sleep/wake 流程） |
| `vllm_ascend/core/recompute_scheduler.py` | RecomputeScheduler（PD 分离调度器） |
| `vllm_ascend/distributed/kv_transfer/` | KV Transfer 基础设施 |
| `vllm_ascend/compilation/acl_graph.py` | ACLGraph 图捕获/更新 |
| `vllm_ascend/_310p/model_runner_310p.py` | 310P 专用 Model Runner |
