# 课题：面向 vllm-ascend 的多级 KV Cache 分层卸载

> **课题编号**：3.1 | **方向**：KV Cache 池化与卸载 | **难度**：⭐⭐⭐⭐ | **预计周期**：3-4 个月

---

## 一、课题内容

### 1.1 课题概述

本课题旨在为 vllm-ascend 实现**多级 KV Cache 分层卸载**（Multi-tier KV Cache Offloading），将现有的 NPU ↔ CPU 两层卸载架构扩展为 **NPU → CPU → NVMe 三级存储层次**，使昇腾 NPU 上的大语言模型推理能够支持超出 NPU HBM 容量的超长序列（128K → 1M+ tokens）。

### 1.2 技术内涵

本课题的核心技术工作包括四个层面：

| 层面 | 内容 | 说明 |
|------|------|------|
| **存储后端** | NVMe SSD 存储层设计与实现 | 在 NVMe 设备上管理 KV cache block 的读写，支持异步 I/O |
| **调度策略** | 三级缓存淘汰与提升策略 | 设计 block 在 NPU→CPU→NVMe 三层之间的迁移决策算法 |
| **传输管道** | 异步多流数据传输管道 | NPU↔CPU 通过 `swap_blocks_batch` 搬运，CPU↔NVMe 通过 `libaio`/`io_uring` 搬运 |
| **预取机制** | 面向 Attention 模式的预取 | 基于 Attention 计算 pattern 预测下一个需要的 block，提前从 NVMe 提升到 CPU |

### 1.3 架构全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ascend NPU (HBM)                              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │ Block 0 │  │ Block 1 │  │ Block 2 │  │ Block N │  ← 热数据      │
│  │ (active)│  │ (active)│  │ (free)  │  │ (free)  │     当前 batch  │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘     正在使用    │
└───────┬──────────────┬──────────────────────────────────────────────┘
        │ D2H stream   │ H2D stream
        │ (swap_blocks │ (swap_blocks
        │  _batch)     │  _batch)
        ▼              ▲
┌─────────────────────────────────────────────────────────────────────┐
│                       CPU Memory (DDR)                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │ Block 0  │ │ Block 1  │ │ Block 2  │ │ Block M  │ ← 温数据      │
│  │ (cached) │ │ (cached) │ │ (cached) │ │ (cached) │    已完成的    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘    请求缓存    │
│                    │                                  ▲              │
│         Cascading  │ (submit_store)     Promotion     │              │
│         Primary→   │                     Secondary→   │ (submit_load)│
│         Secondary  ▼                     Primary      │              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │               NVMe SSD Storage (Tertiary Tier)                │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │ Seg 0    │ │ Seg 1    │ │ Seg 2    │ │ Seg K    │ ← 冷数据│   │
│  │  │ (LRU)    │ │ (LRU)    │ │ (LRU)    │ │ (LRU)    │   历史   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   会话   │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

数据流：
  降级 (evict): NPU ──D2H──► CPU ──cascade──► NVMe
  提升 (promote): NVMe ──load──► CPU ──H2D──► NPU
```

### 1.4 与现有代码的关系

| 现有组件 | 位置 | 本课题中的角色 |
|----------|------|---------------|
| `CpuNpuOffloadingHandler` | `vllm_ascend/kv_offload/cpu_npu.py` | **复用**：NPU↔CPU 搬运层，无需修改 |
| `NPUOffloadingSpec` | `vllm_ascend/kv_offload/npu.py` | **扩展**：增加 `secondary_tiers` 配置解析 |
| 上游 `TieringOffloadingSpec` | `vllm/v1/kv_offload/tiering/spec.py` | **参考蓝图**：三级调度的核心设计 |
| 上游 `TieringOffloadingManager` | `vllm/v1/kv_offload/tiering/manager.py` | **参考蓝图**：三层编排逻辑 |
| 上游 `SecondaryTierManager` (ABC) | `vllm/v1/kv_offload/tiering/base.py` | **实现接口**：需实现 `AscendNVMeSecondaryTier` |
| 上游 `CpuGpuOffloadingHandlers` | `vllm/v1/kv_offload/cpu/gpu_worker.py` | **参考**：CPU↔GPU 搬运模式，对应 Ascend 已有的 NPU↔CPU |

---

## 二、必要性分析

### 2.1 长序列推理是 LLM 服务的核心痛点

当前大语言模型正朝着超长上下文方向发展：

| 模型 | 最大上下文长度 | KV Cache 占用 (FP16, 单卡估算) |
|------|---------------|-------------------------------|
| DeepSeek V3.2 | 128K tokens | ~45 GB (MLA 压缩后) |
| DeepSeek V4 | 1M tokens | ~180 GB (MLA + compress_ratio=4) |
| Qwen3 | 256K tokens | ~80 GB (GQA) |
| Llama 4 | 256K tokens | ~120 GB (GQA) |

昇腾 Atlas 单卡 HBM 通常为 64GB。即使经过 MLA 压缩和 INT8 量化，单个请求的 KV cache 仍可能超出 HBM 容量。**多级卸载是突破 HBM 容量墙、支持超长序列推理的必由之路**。

### 2.2 现有两层卸载的局限性

vllm-ascend 当前已具备 NPU ↔ CPU 两层卸载能力（`CpuNpuOffloadingHandler` + `CPUKVCacheManager`），但存在以下不足：

1. **CPU 内存有限**：单节点 CPU 内存通常 512GB-1TB，在并发服务场景下仍可能成为瓶颈。以 DeepSeek V4 1M 上下文、16 并发为例，峰值 KV cache 需求可达 ~3TB，远超 CPU 内存容量。
2. **无持久化能力**：进程退出后 CPU 上的 KV cache 全部丢失。对于 Prefix Caching 场景（如系统提示词），每次重启都需要重新计算。
3. **缺少自动分层**：当前两层卸载的策略较为简单（LRU 驱逐到 CPU），缺乏精细化的多级温度感知调度。

### 2.3 上游 vLLM 已提供参考架构

上游 vLLM 社区已在 V1 架构中实现了 `TieringOffloadingSpec` + `TieringOffloadingManager`，定义了清晰的三级卸载架构：

```
TieringOffloadingManager
├── Primary Tier: CPUOffloadingManager (LRU/ARC)
│   ├── GPU ↔ CPU 传输（通过 OffloadingHandler）
│   └── 作为 GPU 和 Secondary Tier 之间的网关
└── Secondary Tiers: [SecondaryTierManager, ...]
    └── 仅通过 Primary Tier 间接访问 GPU
```

这为本课题提供了经过社区评审的架构设计参考，降低了"从零设计"的风险。

### 2.4 学术价值与工业价值

- **学术价值**：多级 KV cache 分层调度是一个活跃的研究方向。如何为不同模型（MLA vs GQA vs Hybrid）的 Attention pattern 设计最优的预取策略，是一个具有发表价值的问题。
- **工业价值**：昇腾生态中尚无开箱即用的多级 KV cache 卸载方案。本课题的产出可直接集成到 vllm-ascend 主线，为所有昇腾用户提供长序列推理能力。
- **团队价值**：本课题是导师三个目标中"串通其余模块，结合池化和 offload，提升长序列性能"的直接落地。

### 2.5 与碎片整理（课题 2.2）的关系

课题 2.2（碎片整理）和本课题是**互补关系**而非替代关系：

- 碎片整理解决的是 **HBM 利用率问题**（碎片导致空闲 block 无法被大请求使用）
- 多级卸载解决的是 **HBM 容量问题**（即使碎片为 0，总容量也不够）

碎片整理更适合作为第二个课题——当对 BlockPool 生命周期管理建立肌肉记忆后，再回到碎片问题会事半功倍。

---

## 三、预期目标

### 3.1 功能目标

| 编号 | 目标 | 量化指标 | 验证方式 |
|------|------|---------|---------|
| F1 | 实现 NVMe 后端存储层 | 支持 `libaio`/`io_uring` 异步 I/O，吞吐 ≥ 2GB/s | `fio` 基准测试 |
| F2 | 实现三级分层调度 | NPU→CPU→NVMe 自动降级 + NVMe→CPU→NPU 按需提升 | 端到端功能测试 |
| F3 | 适配 Ascend 自定义算子 | `swap_blocks_batch` (已有) + NVMe I/O 的 NPU Stream 并发 | 多流并发无死锁 |
| F4 | 支持 Hybrid Attention 模型 | DeepSeek V4 (MLA + Mamba 混合层) 的正确卸载/恢复 | 混合模型一致性测试 |
| F5 | Prefix Cache 持久化 | 系统提示词的 KV cache 在重启后可复用 (NVMe 层) | 重启后 prefix cache hit rate 不变 |

### 3.2 性能目标

| 编号 | 指标 | 基线 (两层 NPU-CPU) | 目标 (三级 NPU-CPU-NVMe) | 测试条件 |
|------|------|-------------------|------------------------|---------|
| P1 | 最大支持上下文长度 | 128K (CPU 容量限制) | 1M+ (NVMe 扩展) | DeepSeek V4, 8×Atlas |
| P2 | 吞吐 (token/s) | 基准值 T₀ | ≥ 0.85×T₀（NVMe 引入 <15% 吞吐下降）| 128K 上下文, batch_size=8 |
| P3 | 首 token 延迟 (TTFT) | 基准值 L₀ | ≤ 1.3×L₀（三级提升延迟可控）| Cache miss 场景 |
| P4 | Prefix Cache 命中率退化 | 0%（两层，CPU 不丢） | ≤ 5%（NVMe 的额外 cache miss）| 相同 workload 对比 |
| P5 | NVMe 读写带宽利用率 | N/A | ≥ 70% 设备理论带宽 | 顺序读写 block |

### 3.3 可交付成果

1. **代码**：`AscendTieringOffloadingSpec` + `AscendNVMeSecondaryTier` + 相关适配代码，约 1500-2000 行
2. **测试**：单元测试 + 集成测试 + 性能 benchmark 脚本
3. **文档**：设计文档（架构决策、接口定义）+ 用户指南（配置参数、调优建议）
4. **技术报告**：实习结题报告，包含方案设计、实现细节、性能数据和经验总结

---

## 四、实施计划

### 4.1 总体路线图

```
Month 1                Month 2                Month 3                Month 4
│                      │                      │                      │
├─ Phase 1 ───────────┤                      │                      │
│  现有基础设施调研     │                      │                      │
│  + 上游代码精读       │                      │                      │
│                      ├─ Phase 2 ───────────┤                      │
│                      │  NVMe 后端实现       │                      │
│                      │  + 单元测试          │                      │
│                      │                      ├─ Phase 3 ───────────┤
│                      │                      │  三级调度 + 集成     │
│                      │                      │                      ├─ Phase 4 ───┤
│                      │                      │                      │  预取 + 优化 │
```

### 4.2 Phase 1：现有基础设施调研与上游代码精读（第 1-3 周）

**目标**：完全理解现有两层卸载和上游三级卸载的全部代码路径。

| 周次 | 任务 | 产出 |
|------|------|------|
| W1 | 精读 `CpuNpuOffloadingHandler` (`cpu_npu.py`)、`NPUOffloadingSpec` (`npu.py`) | 画出 NPU↔CPU 搬运的完整时序图 |
| W1 | 精读上游 `TieringOffloadingManager` (`manager.py`) 全部方法 | 理解 lookup/store/load/cascade/promotion 五个核心流程 |
| W2 | 精读上游 `SecondaryTierManager` 接口 (`base.py`) 和 `ExampleSecondaryTier` | 理解接口契约和最少实现要求 |
| W2 | 精读上游 `TieringOffloadingSpec` (`spec.py`) 和 `factory.py` | 理解配置解析和组件创建流程 |
| W2 | 精读上游 `CpuGpuOffloadingHandlers` (`gpu_worker.py`) | 对比 Ascend 版的异同，确认复用/适配点 |
| W3 | 研究 NVMe I/O 方案：`libaio` vs `io_uring` 在 Python 中的绑定方式 | 技术选型决策文档 |
| W3 | 跑通现有 CPU offloading 的端到端流程，记录关键日志和性能基线 | 基线 benchmark 数据 |
| W3 | 编写 Phase 1 调研报告，明确 Phase 2 的接口设计和测试策略 | 设计文档 v1 |

**关键风险**：
- 上游 `TieringOffloadingManager` 中有 `_maybe_process_finished_jobs()` 的 once-per-step 门控机制和 `_flush_pending_promotions()` 的延迟提交机制，这两个设计模式需要彻底理解，否则会出现状态机错误。
- `SharedOffloadRegion` 使用 mmap 在 Scheduler 和 Worker 之间共享 CPU 内存，Ascend 环境下需验证 mmap 兼容性。

### 4.3 Phase 2：NVMe 后端存储层实现（第 4-7 周）

**目标**：实现 `AscendNVMeSecondaryTier`，使其通过 `SecondaryTierManager` 接口的所有契约测试。

| 周次 | 任务 | 产出 |
|------|------|------|
| W4 | 设计 NVMe 上的 block 存储布局（复用 CPU 的 `int8` flat buffer 格式） | 存储格式规范 |
| W4 | 实现 `NVMeBlockStore`：基于 `io_uring` 的异步 block 读写 | `nvme_store.py` |
| W5 | 实现 `AscendNVMeSecondaryTier.lookup()` — 内存中的 block hash 索引 | 带 LRU 的哈希索引 |
| W5 | 实现 `AscendNVMeSecondaryTier.submit_store()` — CPU→NVMe 异步写入 | 级联存储路径 |
| W6 | 实现 `AscendNVMeSecondaryTier.submit_load()` — NVMe→CPU 异步读取 | 提升加载路径 |
| W6 | 实现 `AscendNVMeSecondaryTier.get_finished()` — 异步 I/O 完成轮询 | 事件通知机制 |
| W7 | 编写 NVMe 后端的独立单元测试（使用 tmpfs 模拟 NVMe 设备） | 测试覆盖率 ≥ 85% |
| W7 | 性能微基准：单 block 读写延迟、批量 block 读写吞吐 | NVMe 性能报告 |
| W7 | 实现 `AscendNVMeSecondaryTier.shutdown()` 和 `get_tier_type()` | 资源清理 |

**关键设计决策**：

```
NVMe Block 存储布局:
┌─────────────────────────────────────────────────────┐
│                    NVMe File                         │
├──────────┬──────────┬──────────┬────────────────────┤
│ Block 0  │ Block 1  │ Block 2  │ ... Block N        │
│ (fixed   │ (fixed   │ (fixed   │ (fixed             │
│  size)   │  size)   │  size)   │  size)             │
└──────────┴──────────┴──────────┴────────────────────┘
每 block = page_size_bytes（与 CPU 层一致，减少序列化开销）
```

**关键风险**：
- `io_uring` 在 Python 中的绑定：`python-io_uring` 或 `liburing` 的 ctypes 封装。如果稳定性不足，降级方案为 `libaio`（通过 `ctypes` 调用）。
- NVMe 文件大小管理：采用固定大小的 sparse file 预分配，避免运行时 `ftruncate`。

### 4.4 Phase 3：三级调度逻辑与集成（第 8-11 周）

**目标**：实现 `AscendTieringOffloadingSpec`，端到端跑通三级卸载。

| 周次 | 任务 | 产出 |
|------|------|------|
| W8 | 实现 `AscendTieringOffloadingSpec`，继承/改编上游 `TieringOffloadingSpec` | `ascend_tiering_spec.py` |
| W8 | 适配 `SharedOffloadRegion` 到 Ascend 环境（若需要） | 环境适配 |
| W9 | 实现 NPU↔CPU↔NVMe 的完整降级路径 (cascade) | NPU→CPU→NVMe 端到端 |
| W9 | 实现 NVMe→CPU→NPU 的完整提升路径 (promotion) | NVMe→CPU→NPU 端到端 |
| W10 | 集成测试：单请求的 store→evict→lookup→load 完整生命周期 | 功能正确性验证 |
| W10 | 集成测试：并发多请求、不同序列长度、prefix cache 命中/未命中 | 并发正确性验证 |
| W10 | 适配 Hybrid Attention 模型（DeepSeek V4）：处理不同层类型的 block | 混合模型支持 |
| W11 | 系统级测试：真实模型 (DeepSeek V3.2/V4) + 真实 workload | 端到端可用 |
| W11 | Bug 修复与边缘情况处理（OOM、NVMe 满、进程崩溃恢复） | 鲁棒性增强 |

**关键设计决策 —— Ascend 与上游的差异点**：

1. **block_size 固定为 128**：Ascend NPU 为 DMA 效率强制 block_size=128，而上游 GPU 可配置为 16/32/64。这意味着 Ascend 侧每个 block 更大（128 tokens × MLA 压缩后约 16-32KB），对 NVMe I/O 更友好（更大的 I/O 单元 = 更高的吞吐）。
2. **Sparse C8 量化格式**：`AscendMLAAttentionSpec.cache_sparse_c8=True` 时，KV cache 存储格式为 (bf16, bf16, int8, fp16)，与上游的单一 bf16 格式不同。在 CPU↔NVMe 传输时无需做格式转换（直接搬运 int8 字节），但需注意 offset 计算。
3. **`swap_blocks_batch` 自定义算子**：Ascend 已通过 `torch.ops._C_ascend.swap_blocks_batch` 实现了批量 NPU↔CPU 搬运，Phase 3 直接复用。

**关键风险**：
- **状态机正确性**：`TieringOffloadingManager` 的状态管理涉及 `ref_cnt`（保护传输中的 block）、`PendingPromotion` 延迟提交、`_maybe_process_finished_jobs()` 门控。必须编写白盒测试覆盖所有状态转换。
- **NVMe 延迟**：NVMe 随机读取延迟约 100μs，顺序读带宽约 3-7 GB/s。一个 128-token 的 block 约 16-32KB，单 block 读取延迟可能成为瓶颈。需要批量预取来摊薄延迟。

### 4.5 Phase 4：性能优化与预取机制（第 12-14 周）

**目标**：性能达标（吞吐下降 < 15%，支持 1M+ 上下文）。

| 周次 | 任务 | 产出 |
|------|------|------|
| W12 | 实现 **block 级预取**：在 Attention 计算当前 block 时，后台从 NVMe 加载下一个 block | 预取管道 |
| W12 | 实现 **批量 I/O 合并**：合并连续的 block 为一个 I/O 请求 | I/O 批量化 |
| W13 | 多流并发优化：NPU compute stream、D2H stream、H2D stream、NVMe I/O 四流并发 | 流水线调度 |
| W13 | 性能 benchmark：不同上下文长度 (32K/128K/256K/512K/1M)、不同 batch_size | 性能报告 |
| W14 | 调优：LRU vs ARC 淘汰策略对比、NVMe 预取窗口大小调优 | 调优指南 |
| W14 | 文档完善 + 技术报告撰写 | 最终交付 |

**预取策略设计**：

```
┌─────────────────────────────────────────────────────────┐
│                  Prefetch Pipeline                       │
│                                                          │
│  Time ─────────────────────────────────────────────►     │
│                                                          │
│  Compute:  [Block N-1] [Block N]   [Block N+1] [Block N+2]
│  H2D:                [Block N+1]             [Block N+2]
│  NVMe→CPU:         [Block N+2] [Block N+3]              │
│                                                          │
│  Prefetch window = 2 blocks ahead                        │
└─────────────────────────────────────────────────────────┘
```

对于 Decode 阶段（逐 token 生成），Attention 计算呈现严格的顺序访问模式，预取准确率接近 100%。对于 Prefill 阶段（并行处理所有 token），所有 block 一次性加载，预取意义不大——但 Prefill 本身的 compute 密度高，I/O 等待可以被计算掩盖。

### 4.6 里程碑与评审节点

| 节点 | 时间 | 评审内容 |
|------|------|---------|
| M1 | W3 结束 | Phase 1 调研报告 + 设计文档 v1 |
| M2 | W7 结束 | NVMe 后端独立可测 + 性能微基准 |
| M3 | W11 结束 | 端到端三级卸载功能跑通 |
| M4 | W14 结束 | 性能达标 + 最终交付 |

---

## 五、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| `io_uring` Python 绑定不稳定 | 中 | 高 | 降级方案：`libaio`（更成熟但性能稍低） |
| 上游 `TieringOffloadingManager` 接口变动 | 低 | 中 | 锁定上游 commit，Phase 1 即确认版本 |
| NVMe 延迟导致 TTFT 超标 | 中 | 中 | 增大预取窗口 + 增大 CPU 层容量（减少 miss） |
| Hybrid 模型层类型差异导致 block 格式不一致 | 中 | 高 | Phase 3 专门安排一周处理混合模型适配 |
| `SharedOffloadRegion` mmap 在 Ascend 环境不兼容 | 低 | 高 | 降级方案：使用普通 `torch.zeros(pin_memory=True)` |

---

## 六、总结

本课题以**上游 vLLM 的三级卸载架构为蓝本**，在 vllm-ascend 现有两层卸载（NPU↔CPU）的坚实基础上，新增 **NVMe 第三级存储层**，实现 KV cache 在 NPU HBM → CPU DDR → NVMe SSD 之间的自动分层调度。

- **技术难度**适中：上游提供了清晰的架构蓝图，现有 `CpuNpuOffloadingHandler` 提供了稳定的 NPU↔CPU 搬运基础，核心增量在于 NVMe 后端的工程实现和三层状态机的正确编排
- **产出效果**直观：从"最大 128K 上下文"到"支持 1M 上下文"，一句话就能说清楚价值
- **技能栈**完整：异步 I/O、缓存淘汰策略、预取算法、多流并发——这些技能在系统方向具有广泛的通用性
- **与导师目标**高度对齐：直接对应第 3 个目标——"以 KV cache 为基础，串通其余模块，结合池化和 offload，提升 vllm-ascend 在长序列上的性能"
