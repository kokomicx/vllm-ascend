# KV Cache 管理优化 — 任务看板

> **Q3 重点推进：KV Cache 管理的系统化梳理与治理**
>
> 牵头：曾彦嘉、赵闯 | 协助：曹梦晴、郭小辉、温捷
>
> 最后更新：2026-07-13

---

## 一、任务总览

| # | 任务 | 完成时间 | 状态 |
|---|------|---------|------|
| 1 | 模型 KV Cache 管理方式梳理与差异分析 | 7月15日 | ✅ 基本完成 |
| 2 | 算子依赖梳理与对齐 | 7月22日 | 🔄 进行中 |
| 3 | 历史债务清理 | 7月30日 | 🔄 进行中（代码就绪，待测试验证） |

---

## 二、模型覆盖范围

| 模型类型 | 典型模型 | Backend | Layout 类 | 优先级 |
|---------|---------|---------|-----------|--------|
| Hybrid | Qwen3.5 | Mamba + GQA | `MambaLayout` + `SplitKVLayout` | 1️⃣ |
| SFA | GLM5.1 | `AscendSFABackend` | `SparseMLALayout` / `SparseMLAC8Layout` | 2️⃣ |
| GQA | Qwen3 MoE | `AscendAttentionBackend` | `SplitKVLayout` | 3️⃣ |
| MLA | DeepSeek 3.1 | `AscendMLABackend` | `SplitKVLayout` | 4️⃣ |
| MLA | DeepSeek V3.2 | `AscendSFABackend` | `SparseMLALayout` | 4️⃣ |
| MLA | DeepSeek V4 | `AscendDSABackend` | `CompressedMLALayout` | 4️⃣ |

---

## 三、已完成的工作

### 3.1 Task 1: 模型差异分析 ✅

| 产出物 | 路径 | 状态 |
|--------|------|------|
| KV Cache 管理分析文档（3000+ 行） | `docs/kv_cache_management_analysis.md` | ✅ |
| Operator 依赖清单 | `docs/operator_dependency_inventory.md` | ✅ |
| 技术分享 PPT | `docs/kv_cache_presentation.pptx` | ✅ |
| 演讲稿 V2 | `docs/PPT_speech_script_v2.md` | ✅ |

**覆盖维度**：

- [x] 架构设计对比（上游两层 vs Ascend 两层 + 隐含分支）
- [x] Backend 层对比（17 个上游 Backend vs 4 个 Ascend Backend）
- [x] Spec 层对比（`FullAttentionSpec` / `MLAAttentionSpec` vs `AscendMLAAttentionSpec`）
- [x] Allocate 管线对比（上游 ~30 行 vs Ascend ~600 行）
- [x] Reshape 管线对比（上游 view().view() vs Ascend 15+ if-else 分支）
- [x] 算子地址空间限制分析（§7.6）
- [x] bf16 vs fp8 精度维度分析（§7.7）
- [x] 对齐路线图（Phase 1→2→3，§10.5.3）

### 3.2 Phase 2-3 Layout-driven 代码重构 ✅

| 组件 | 文件 | 行数 |
|------|------|------|
| Layout 抽象基类 + 6 子类 | `kv_cache_layout.py` | 624 行（新增） |
| Monkey-patch 分发 | `patch_kv_cache_interface.py` | 320 行（新增 + 修改） |
| model_runner V2 分配/重塑 | `model_runner_v1.py` | +250 行（新增，feature gate 保护） |
| 弃用旧 Spec | `kv_cache_interface.py` | -259 行（删除） |
| model_runner 代码清理 | `model_runner_v1.py` | -600+ 行（删除 A5 分支等） |

**6 个 Layout 子类**：

| Layout | 模型 | KV Cache 布局 |
|--------|------|--------------|
| `SingleTensorLayout` | Draft models, cache_only | 单 tensor |
| `SplitKVLayout` | GQA, 标准 MLA | K + V 两 tensor |
| `SparseMLALayout` | DS V3.2 Sparse (bf16) | K + V + Indexer_K 三 tensor |
| `SparseMLAC8Layout` | DS V3.2 Sparse (C8) | K + V + Indexer_K + Scale 四 tensor |
| `CompressedMLALayout` | DS V4 | K + V + Scale + Compress 等多 tensor |
| `MambaLayout` | Mamba, Qwen3.5 Hybrid | 单 tensor |

**Feature gate 策略**：

```python
# 环境变量控制，默认关闭，旧代码全部保留
VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=0  # 走旧路径（现有逻辑）
VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=1  # 走新路径（Layout-driven）
```

**测试**：

- `tests/test_phase2_spec_dispatch.py` — 11 个测试（Spec → Layout 分发）
- `tests/test_phase3_layout_dispatch.py` — 18 个测试（Layout → allocate/reshape）

---

## 四、待完成的工作

### 4.1 Task 2: 算子依赖对齐（7月22日 截止）

- [ ] **与算子侧对齐会议**：拿 `docs/operator_dependency_inventory.md` §6 的问题清单逐条确认
- [ ] **确认 stride 支持计划**：各算子是否 / 何时支持非连续 tensor 输入
- [ ] **确认 K/V 交织计划**：是否计划支持类似 CUDA 的单 tensor + offset 布局
- [ ] **确认 block_size 支持范围**：各算子是否支持更小的 block_size（当前仅 DSA 支持 [2,4,8,...]）
- [ ] **输出对齐结论文档**：每个算子的确认状态（已支持 / 规划中 / 不支持）

### 4.2 Task 3: 历史债务清理（7月30日 截止）

#### 4.2.1 按优先级逐步测试 & 开启 Feature Gate

| 步骤 | 内容 | 验证内容 | 状态 |
|------|------|---------|------|
| 1 | Qwen3.5 Hybrid 测试 | `VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=1` 跑 correctness + performance | ⬜ |
| 2 | GLM5.1 SFA 测试 | 同上 | ⬜ |
| 3 | GQA（Qwen3 MoE / Qwen2.5）测试 | 同上 | ⬜ |
| 4 | MLA（DS V3.1 / V3.2 / V4）测试 | 同上 | ⬜ |

#### 4.2.2 待清理的文件（验证通过后）

| 文件 | 清理方式 | 优先级 |
|------|---------|--------|
| `model_runner_v1.py` 旧 allocate/reshape 分支 | 删除 else 分支，保留 V2 路径 | 1️⃣ |
| `patch_kv_cache_interface.py` 中的旧字段定义 | 精简为仅 Layout 分发 monkey-patch | 2️⃣ |
| `patch_mla_prefill_backend.py` | 确认 Ascend 不需要后删除或保留 no-op | 3️⃣ |
| `kv_cache_interface.py` (已删除) | 确认无引用后彻底移除 | ✅ Done |
| `patch_qwen3_5.py` 中与 Layout 重复的逻辑 | 评估是否可以合并 | 3️⃣ |

#### 4.2.3 长期规划（Q4 及以后）

```
Phase 3 (Q3): 内部重构          Phase 4 (Q4+): 社区合入
─────────────────────────      ────────────────────────
✓ Layout 6 子类完成             → 提议 KVCacheLayout 作为上游硬件无关抽象
✓ model_runner 缩减到 ~135 行   → 提议 Spec 字段通用化
✓ 15+ if-else → 多态分发       → 删除 model_runner patch
  (feature gate off)            → 推动算子侧 stride 支持
                                → 全面回归上游 allocate/reshape
```

### 4.3 技术债务追踪

| 债务项 | 影响范围 | 计划清理时间 |
|--------|---------|------------|
| `model_runner_v1.py` 600 行 if-else | allocate/reshape | Q3 测试通过后 |
| A5 设备特殊分支残留 | `sparse_head_dim` 计算等处 | Q3 |
| `_get_attention_kv_cache_dims` 绕过 Backend 直接查 layer | reshape 路径 | Q4（等 Layout 完全接管） |
| `as_strided` overlay 不能直接传 kernel | reshape 需"固化" | 等算子侧 stride 支持 |
| 54 个 patch 文件中有 ~6 个与 KV Cache 直接相关 | 架构整洁度 | Q3-Q4 逐一清理 |

---

## 五、关键里程碑

```
7月13日  ✅ Task 1 文档完成，Phase 2-3 代码就绪
7月15日  📌 Task 1 交付节点
7月22日  📌 Task 2 交付节点（算子对齐）
7月30日  📌 Task 3 交付节点（债务清理）
   ↓
Q4       📌 Phase 4：社区合入 RFC
```

---

## 六、风险与阻塞项

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| GLM5.1 测试环境不确定 | 🔴 高 | 确认是否有测试集群 / 权限 |
| 算子侧 stride 支持排期未定 | 🟡 中 | Task 2 对齐会议中明确时间表 |
| Feature gate 开启后可能遇到 hidden bug | 🟡 中 | 逐模型开启，保留回退能力 |
| 分支合入 main 的冲突风险 | 🟢 低 | 及时 rebase，func revert 保护 |

---

## 七、相关链接

| 资源 | 路径 |
|------|------|
| 分析文档 | `docs/kv_cache_management_analysis.md` |
| 算子依赖清单 | `docs/operator_dependency_inventory.md` |
| Layout 代码 | `vllm_ascend/core/kv_cache_layout.py` |
| Model Runner V2 | `vllm_ascend/worker/model_runner_v1.py` L3597-L3860 |
| Patch 入口 | `vllm_ascend/patch/platform/patch_kv_cache_interface.py` |
| Feature Gate | `vllm_ascend/envs.py` L119-120 |
| Phase 2 测试 | `tests/test_phase2_spec_dispatch.py` |
| Phase 3 测试 | `tests/test_phase3_layout_dispatch.py` |
| Git 分支 | `feature/layout-refactor-phase3` |
