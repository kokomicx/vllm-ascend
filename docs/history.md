# 对话历史摘要

> 供切换对话时使用。将本文内容提供给新模型，即可继续之前的工作。
>
> 最后更新：2026-07-13

---

## 项目背景

**vLLM-Ascend KV Cache 管理优化**（Q3 重点），牵头：曾彦嘉、赵闯。

三个并行任务：

| # | 任务 | 截止 | 状态 |
|---|------|------|------|
| 1 | 模型 KV Cache 管理方式梳理与差异分析 | 7/15 | ✅ 基本完成 |
| 2 | 算子依赖梳理与对齐 | 7/22 | 🔄 进行中 |
| 3 | 历史债务清理 + Phase 2-3 重构测试验证 | 7/30 | 🔄 代码就绪，待测试 |

代码在 `feature/layout-refactor-phase3` 分支，未合入 main。

---

## 已完成的工作

### 文档产出（Task 1）

| 文件 | 说明 |
|------|------|
| `docs/KV_Cache_管理方式梳理与差异分析.md` | 主分析文档（最终版：客观记录风格，无主观口吻） |
| `docs/operator_dependency_inventory.md` | 算子依赖清单 |
| `docs/task.md` | 任务看板 |
| `docs/kv_cache_presentation.pptx` | 技术分享 PPT |
| `docs/PPT_speech_script_v2.md` | 演讲稿 |

### Phase 2-3 代码重构（代码已就绪，待服务器验证）

**核心改动**：用多态 `KVCacheLayout` 体系替代 model_runner 中 600+ 行 if-else。

| 文件 | 行数 | 说明 |
|------|------|------|
| `vllm_ascend/core/kv_cache_layout.py` | 625 行（新增） | 1 个抽象基类 + 6 个子类 |
| `vllm_ascend/patch/platform/patch_kv_cache_interface.py` | 321 行 | Monkey-patch Spec → Layout 分发 |
| `vllm_ascend/worker/model_runner_v1.py` L3597-3860 | +250 / -600 行 | V2 allocate/reshape，feature gate 保护 |

**6 个 Layout 子类**：

| Layout | 模型 | Tensor 数 |
|--------|------|-----------|
| `SingleTensorLayout` | Draft models, cache_only | 1 |
| `SplitKVLayout` | GQA, 标准 MLA | 2 (K+V) |
| `SparseMLALayout` | DS V3.2 bf16 | 3 |
| `SparseMLAC8Layout` | DS V3.2 C8 量化 | 4 |
| `CompressedMLALayout` | DS V4 fp8 | 1 (as_strided overlay) |
| `MambaLayout` | Mamba, Hybrid | 1 (多状态 carve) |

**Feature gate**：`VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH` 环境变量，默认 0（走旧路径），设为 1 启用新路径。

**已有单元测试**：
- `tests/test_phase2_spec_dispatch.py` — 11 个测试（Spec → Layout）
- `tests/test_phase3_layout_dispatch.py` — 18 个测试（Layout → allocate/reshape）
- `tests/test_kv_cache_layout.py` — pytest 类组织，完整覆盖

### 验证脚本（本次对话产出）

为服务器端正确性验证创建了三个文件：

| 文件 | 作用 |
|------|------|
| `tests/e2e/test_layout_correctness.py` | 加载模型 → 生成 → dump KV cache shape/dtype/contiguous 到 JSON |
| `tests/e2e/compare_kv_cache_shapes.py` | 加载两份 JSON snapshot，逐层对比 |
| `tests/e2e/verify_layout_refactor.sh` | 一键脚本：单测 → 旧路径 dump → 新路径 dump → 对比 |

---

## 关键决策记录

1. **文档风格**：保持客观记录风格，不加主观口吻（"说实话"、"我猜"等均删除）。
2. **Feature gate 策略**：新代码和旧代码并存，默认关闭，逐模型开启验证。
3. **验证方法**：新旧路径跑同一个模型，对比 KV cache shape/dtype/tensor 数量是否完全一致。
4. **测试优先级**：Qwen3 MoE (GQA) → Qwen3.5 (Hybrid) → GLM5.1 (SFA) → DS V3.1 → DS V3.2 → DS V4。

---

## 当前状态 & 下一步

### 待做：在 NPU 服务器上验证重构代码

**第一步：推送代码到服务器**

```bash
# 本地
git add -A && git commit -s -m "feat: Layout-driven KV cache refactoring + e2e verification scripts"
git push myfork feature/layout-refactor-phase3

# 服务器
git fetch myfork && git checkout feature/layout-refactor-phase3
```

**第二步：按优先级逐模型验证**

```bash
# 一键验证（推荐先跑 GQA 模型）
bash tests/e2e/verify_layout_refactor.sh /path/to/Qwen3-MoE-Instruct

# 或分步：
VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=0 ASCEND_RT_VISIBLE_DEVICES=0 \
  python tests/e2e/test_layout_correctness.py --model /path/to/model --output /tmp/old.json

VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=1 ASCEND_RT_VISIBLE_DEVICES=0 \
  python tests/e2e/test_layout_correctness.py --model /path/to/model --output /tmp/new.json

python tests/e2e/compare_kv_cache_shapes.py /tmp/old.json /tmp/new.json
```

**验证通过标准**：
- KV cache shape/dtype/tensor 数量新旧完全一致
- 生成文本非空（模型正常运行）
- 所有 tensor 连续（算子要求）

**第三步：验证通过后清理旧代码**

验证通过后，删除 `model_runner_v1.py` 中 else 分支（旧 if-else 树），默认走 Layout-driven 路径。

### 待做：Task 2 算子对齐（7/22 截止）

- 与算子侧会议，逐条确认 `docs/operator_dependency_inventory.md` §6 的问题清单
- 确认 stride 支持计划、K/V 交织计划、block_size 支持范围

---

## 关键文件索引

```
vllm-ascend/
├── docs/
│   ├── task.md                          # 任务看板
│   ├── KV_Cache_管理方式梳理与差异分析.md  # 主分析文档
│   ├── operator_dependency_inventory.md  # 算子依赖清单
│   └── history.md                        # 本文件
├── vllm_ascend/
│   ├── core/kv_cache_layout.py           # Layout 体系（新增）
│   ├── patch/platform/patch_kv_cache_interface.py  # Spec monkey-patch
│   ├── worker/model_runner_v1.py         # V2 allocate/reshape (L3597-3860)
│   └── envs.py                           # Feature gate 定义 (L119-121)
├── tests/
│   ├── test_phase2_spec_dispatch.py      # Phase 2 单测
│   ├── test_phase3_layout_dispatch.py    # Phase 3 单测
│   ├── test_kv_cache_layout.py           # Layout 完整单测
│   ├── validate_reshape_equivalence.py   # 新旧 reshape 等价性
│   └── e2e/
│       ├── test_layout_correctness.py    # E2E 正确性测试（新）
│       ├── compare_kv_cache_shapes.py    # A/B 对比脚本（新）
│       └── verify_layout_refactor.sh     # 一键验证脚本（新）
└── Git: feature/layout-refactor-phase3
```

---

## 给新模型的指引

如果你是被用户通过本文档引入的新对话模型，请：

1. 阅读 `docs/task.md` 了解全局任务
2. 阅读 `docs/KV_Cache_管理方式梳理与差异分析.md` 了解技术背景
3. 检查当前 git 状态：`git log --oneline -5` 和 `git status`
4. 询问用户：**"当前验证到哪个模型了？有没有遇到报错？"**
5. 如果验证已通过：下一步是清理旧代码（Task 3 代码清理）和算子对齐（Task 2）
6. 如果验证遇到问题：对比新旧路径的 JSON snapshot，定位 shape/dtype 差异
