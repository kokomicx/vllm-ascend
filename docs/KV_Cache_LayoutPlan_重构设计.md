# KV Cache Layout Plan 重构设计

## 1. 这次重构要解决什么问题

这次改动不是为了改变 KV Cache 的算法或宣称性能提升，而是一次**保持现有语义的职责重构**。目标是把“某种 attention 算子需要怎样的 KV Cache 物理布局”收口到 attention backend，让 `Model Runner` 只负责通用的生命周期工作。

可以用一句话向导师说明：

> 我把 KV Cache 的“布局决策和 shape/view 契约”下沉到 backend；Model Runner 只执行 backend 给出的 Layout Plan，不再知道模型是 GQA、MLA、Sparse MLA、Compressed MLA 还是 Mamba。

这里的“物理布局”包括：

- 一个逻辑 KV Cache 需要分配几个物理 tensor；
- 总字节数怎样按 K、V、latent、RoPE、indexer、scale 等字段切分；
- 每个 raw buffer 应 reshape 成什么 shape、dtype、stride 或 overlay view；
- 是否需要为 PD/RDMA 做 2 MB 对齐；
- 量化和 `cache_dtype` 如何影响上述结果。

## 2. 改造前的问题

改造前，`vllm_ascend/worker/model_runner_v1.py` 同时承担了两类职责：

1. **通用运行时职责**：读取 `KVCacheConfig`、申请 buffer、记录跨层共享、绑定到模型层、执行 hybrid 后处理；
2. **backend 专属职责**：根据 `AttentionSpec`、MLA/Sparse/Compressed/Mamba、cache-only、量化和模型层内部维度来选择 Layout，并在 allocate 和 reshape 两个位置重复判断。

这会造成三个风险。

| 风险 | 具体表现 | 后果 |
| --- | --- | --- |
| 决策分散 | allocate 与 reshape 分别选择 layout | 两处逻辑稍有不一致就会分配正确、reshape 错误，或反过来 |
| Runner 知道太多模型细节 | Runner 读取 MLA layer 的 `kv_lora_rank`、`qk_rope_head_dim`，判断 sparse/C8/compress | 新模型适配会持续扩大 `if/elif` 树 |
| raw container 反推语义 | reshape 根据 raw 是 tensor 还是 tuple 改选 layout | raw 的外部形态不能可靠代表 backend 的真实物理契约 |

已发现的典型 P0 就来自第三点：`CompressedMLALayout` 按设计只分配一个 raw tensor，但旧 gate=1 reshape 路径把“单 tensor”理解为普通 `SingleTensorLayout`，导致 compressed MLA 所需的 `as_strided` scale/overlay view 被跳过。

## 3. 目标架构

```text
KVCacheSpec + layer_name + VllmConfig + runtime context
                         │
                         ▼
              Attention Backend
          get_kv_cache_layout_plan(...)
                         │
                         ▼
              KVCacheLayoutPlan
  { layout, split sizes, dtype, shape/view, alignment }
                         │
                         ▼
                NPUModelRunner
       allocate raw buffers → Plan.reshape → bind
                         │
                         ▼
                 attention / indexer kernel
```

核心边界是：

- **Spec** 表达“逻辑 cache 是什么”；
- **Backend / Plan** 表达“本 backend 的 kernel 要怎样存、怎样看”；
- **Runner** 表达“什么时候申请、如何复用、如何绑定”。

## 4. 代码落点与职责

### 4.1 `vllm_ascend/core/kv_cache_layout.py`

保留并集中各种可复用的物理布局策略：

| Layout | 适用语义 | 输出形态 |
| --- | --- | --- |
| `SingleTensorLayout` | cache-only、hybrid attention buffer | 1 个 tensor |
| `SplitKVLayout` | GQA、标准 MLA | `(K, V)` 两个 tensor |
| `SparseMLALayout` | bf16 Sparse MLA/SFA | `(latent, rope, indexer_k)` |
| `SparseMLAC8Layout` | C8 Sparse MLA | `(latent, rope, indexer_k_int8, scale)` |
| `CompressedMLALayout` | compressed MLA | 单 raw buffer 上的多个 `as_strided` view |
| `MambaLayout` | Mamba / linear attention state | 单 raw buffer 切出多个 state view |

新加入的 `KVCacheLayoutPlan` 是不可变执行计划。它将下面信息绑定在一起：

- 已选中的 `KVCacheLayout`；
- 当前 `spec` 与消费它的 `backend`；
- `layer_name`、`vllm_config`；
- K/V head dimensions；
- FA quant 上下文；
- 非 MLA attention 的 `cache_dtype_str`。

因此，Runner 不再传递零散的 `kwargs`，而只调用：

```python
sizes = plan.split_sizes(total_bytes)
cache = plan.reshape(raw_tensors, ...)
```

### 4.2 `vllm_ascend/attention/kv_cache_layout.py`

新增 `AscendKVCacheLayoutBackendMixin`，它为 Ascend 的 attention backend 提供统一接口：

```python
backend.get_kv_cache_layout_plan(
    spec,
    layer_name=layer_name,
    vllm_config=vllm_config,
    is_hybrid_model=is_hybrid_model,
)
```

Plan 的选择规则位于 backend 侧：

1. `MambaSpec` → `MambaLayout`；
2. cache-only 或 hybrid attention → `SingleTensorLayout`；
3. MLA 且 `compress_ratio > 1` → `CompressedMLALayout`；
4. MLA 且 C8 sparse → `SparseMLAC8Layout`；
5. MLA 且有 `sparse_head_dim` → `SparseMLALayout`；
6. 其他 `AttentionSpec` → `SplitKVLayout`。

同时，backend 在需要时读取模型层，以取得标准 MLA 的 `(kv_lora_rank, qk_rope_head_dim)`；这属于算子/模型结构契约，不应由通用 Runner 决定。

`AscendAttentionBackend`、`AscendMLABackend`、`AscendDSABackend`、`AscendSFABackend` 和 `AscendFABackend` 均继承该 mixin。对于上游 Mamba backend，在 `patch_kv_cache_interface.py` 增加了同一接口的兼容 provider，使 hybrid 模型也能走统一 Plan 流程。

### 4.3 `vllm_ascend/worker/model_runner_v1.py`

gate=1 路径现在只保留以下动作：

1. 获取每个 layer 对应的 attention backend；
2. 调用 backend 获得每层 `KVCacheLayoutPlan`；
3. 对 `shared_by` 的 layer 校验 Plan 的 tensor 数和 split sizes 一致；
4. 根据 Plan 分配一个或多个 raw `int8` buffer，需要时做 2 MB 对齐；
5. 在 reshape 阶段使用**同一份** Plan；
6. 执行既有的 hybrid 后处理、跨层共享和模型绑定。

因此 Runner 已移除以下布局决策：

- `_build_layout_kwargs()`；
- MLA/Sparse/C8/Compressed/Mamba/cache-only 的分支；
- `raw_is_tuple` 的 layout 反推；
- Runner 内读取 MLA layer 维度的行为。

## 5. 为什么 Plan 能解决 compressed MLA 问题

改造前的逻辑相当于：

```text
allocate: spec → CompressedMLALayout → 一个 raw tensor
reshape : raw 是一个 tensor → SingleTensorLayout
```

两阶段的判据不同，因而选择不一致。

改造后的逻辑是：

```text
backend: spec → 一个 CompressedMLALayout Plan
allocate: 使用该 Plan 的 split_sizes
reshape : 使用该 Plan 的 reshape
```

同一份不可变 Plan 贯穿 allocate 和 reshape。raw buffer 的外在容器形式不再参与 layout 选择，因此不会覆盖 compressed MLA 的特殊 view 语义。

## 6. 与导师交流时可以强调的设计取舍

### 这不是“把代码换文件”

重点不是单纯把 `if/elif` 从 Runner 搬到别处，而是建立一个明确的契约：**谁消费 KV Cache，谁定义它的物理表示**。这让 allocation 和 reshape 绑定到同一对象，避免两处重复选择导致的错配。

### 为什么不是让 Spec 直接承担全部逻辑

Spec 描述的是逻辑 cache，而最终 shape、dtype、连续性、stride 和 kernel view 都与 backend/算子有关；同一个逻辑 spec 在不同 backend 上可以有不同物理要求。Plan 放在 backend 侧更符合上游 attention backend 的职责边界。

### 为什么仍保留 `KVCacheLayout` 策略类

Plan 负责“为某层选择并绑定上下文”，Layout 负责“实现一种可复用的物理布局算法”。二者分工如下：

```text
Backend Plan：选择哪个 Layout，给它什么上下文
Layout       ：如何切字节、如何生成 tensor/view
Runner       ：何时申请、如何共享、如何绑定
```

这避免每个 backend 重复实现 split/reshape，又避免 Runner 理解模型特例。

### 为什么 gate 不立即默认打开

这是高风险内存布局代码。现阶段保持 `VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH` 默认关闭：gate=0 是旧路径，gate=1 是 Plan 路径。只有真实模型的 metadata、dtype、contiguous 以及生成 token ID A/B 均通过后，才讨论默认开启与删除旧路径。

## 7. 验证现状与下一步

### 已有正确性证据

- GQA：Qwen3-30B-A3B，gate=0/1 的 KV metadata 与 8 个生成 token ID 一致；
- Hybrid：Qwen3.5-2B，24 层 metadata 与 8 个 token ID 一致；
- 标准 MLA：DeepSeek-V2-Lite-W8A8，27 层 metadata 与 8 个 token ID 一致；
- Phase 3 原有单测在服务器曾通过 `21 passed`。

### 本次新增验证

`tests/test_backend_kv_cache_layout_plan.py` 覆盖：

- Full Attention 生成 `SplitKVLayoutPlan`；
- compressed MLA Plan 保持 `CompressedMLALayout`；
- hybrid attention 的单 tensor 策略由 backend 产生。

服务器侧应优先执行：

```bash
python -m pytest \
  tests/test_backend_kv_cache_layout_plan.py \
  tests/test_phase3_layout_dispatch.py \
  -q
```

之后重跑 GQA、Hybrid、标准 MLA 的 gate=0/1 真实 token parity。Sparse MLA 的真实模型 GLM-5-W4A8 仍被多进程 safetensors loader 初始化异常阻塞；该异常发生在 KV Cache 初始化之前，不能据此否定或证明 Layout Plan 的正确性。

## 8. 给导师的 30 秒版本

> 原先 Model Runner 同时决定 KV Cache 的布局、分配和 reshape，模型类型与量化分支在 allocate 和 reshape 两边重复，容易出现不一致。现在我将布局选择和 shape/view 契约下沉到 attention backend：backend 为每层返回不可变 Layout Plan，Plan 固定 buffer 数、字节切分、dtype/shape/view 与对齐要求；Runner 只负责按 Plan 分配、调用和绑定。这样 compressed MLA 不会再因单 raw tensor 被错误降级，后续新增 Sparse/C8/新模型时主要扩展 backend Plan/Layout，而不继续膨胀 Model Runner 的条件树。当前保留 gate 进行真实模型 token parity 回归验证。
