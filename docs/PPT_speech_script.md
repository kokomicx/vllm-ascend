# vLLM-Ascend KV Cache 管理优化方案 — 演讲稿

## 开场 (Slide 1-2, ~2 分钟)

各位好，今天我汇报的主题是 **vLLM-Ascend 的 KV Cache 管理方式梳理与差异分析**。

这份汇报的核心目标是回答一个问题：**为什么上游 vLLM 的 KV Cache 管线只有 ~60 行代码、零分支，而 Ascend 对应的代码膨胀到了 ~600 行、15 个以上的 if-else 分支？**

我们的分析路径是：先建立上游 vLLM 的端到端管线共识——Spec → Backend → Allocate → Reshape 四个阶段是怎么运作的——然后逐阶段对比 Ascend 的实现，找出每一层的差距。最后把这些差距归纳为几个根本原因，并提出一条从内部重构到社区合入的三步对齐路线。

覆盖范围包括 GQA、标准 MLA、Sparse MLA、Compress MLA、Hybrid 五种模型类型，以及 bf16 / int8 C8 / fp8 C8 三种精度。

---

## Part 1: 社区 vLLM 端到端管线 (Slide 3-8, ~5 分钟)

先看上游是怎么做的。

### Slide 3-4: 四阶段管线全景

上游 vLLM 的 KV Cache 管线可以归纳为四个阶段，每个阶段职责清晰：

- **第一阶段：Spec 创建**。每一层通过 `get_kv_cache_spec()` 返回一个 frozen dataclass——`KVCacheSpec` 的子类实例。这个对象声明了该层的 KV cache 长什么样：block_size、num_kv_heads、head_size、dtype、page_size_bytes。关键点是：上游有完整的 Spec 子类层次体系——FullAttentionSpec、MLAAttentionSpec、SlidingWindowMLASpec、MambaSpec——**一种 attention 类型对应一种 Spec 子类**。model_runner 拿到 Spec 后，不需要问"你是什么模型"——类型本身就编码了全部语义。这就是我们反复提到的"类型即语义"原则。

- **第二阶段：Backend 返回 Shape**。`get_kv_cache_shape()` 方法返回完整的 shape tuple。比如 GQA 返回 `(2, N, B, H, D)`，MLA 返回 `(N, B, 576)`。**一个 shape 回答全部存储问题**——这个 tensor 有几维、每维什么意思、总共多大。model_runner 拿到 shape 后只需要 `view()` 一下。

- **第三阶段：全局规划**。`get_kv_cache_configs()` 负责分组决策（哪些层共享 block table）、算块数（available_memory / page_size_bytes）、出配置（KVCacheConfig）。关键是 `KVCacheTensor.shared_by` 机制——多层共享同一个物理 buffer，比如 Qwen3.5 里一个 attention 层和三个 linear attention 层共享同一块 int8 buffer。

- **第四阶段：Allocate + Reshape**。这两个步骤加起来大约 60 行，零分支。Allocate 就是 `torch.zeros(size, dtype=int8)`，Reshape 就是 `raw.view(dtype).view(shape)`。as_strided 仅用于 page padding（一个通用运行时需求），跟模型架构无关。

**核心原则：model_runner 不做模型类型的判断。它只是机械执行——拿到 Spec → 拿到 shape → 分配 → reshape → 结束。**

### Slide 5-6: 分组与空间计算

社区的分组方案是一套决策级联——从简到繁依次尝试。最简单的情况：所有层相同，1 个 group 全包含。最复杂的情况：DS V4 的三层结构（Compressor + Indexer + SWA），需要按 page_size 分桶。

空间计算的核心是 `page_size_bytes`——每个 Spec 子类独立实现，零 if-else 分支。从这个值出发，除以 available_memory 得到 num_blocks，组装成 KVCacheConfig。Worker 侧只需 `torch.zeros(tensor.size, dtype=int8)`——30 行，0 分支。

### Slide 7-8: 内部数据组织

上游所有模型类型都产生 1 个 tensor/层。MLA 的 kv_lora(512) + k_rope(64) 在同一个 tensor 里，kernel 内部通过 offset 区分。fp8 的字节级自定义布局也封装在同一个 blob 里。**上游的 GPU kernel 接受打包的 blob，内部解析 struct 字节布局。host 侧完全不需要知道内部结构。**

---

## Part 2: Ascend 管线——差距在哪 (Slide 9-13, ~8 分钟)

现在我们来看 Ascend 的同一条管线。核心发现：**上游的四个阶段在 Ascend 侧全部出现了信息丢失——每一步都只能完成 1/N 的工作，剩余部分泄漏到了 model_runner。**

### Slide 9: Spec 层——"类型擦除"

上游 1 种模型 → 1 种 Spec 子类。Ascend 所有 MLA 变体挤进**同一个** `AscendMLAAttentionSpec`。新增了 6 个 NPU 特有字段：scale_dim、scale_dtype、sparse_head_dim、cache_sparse_c8、c8_k_cache_dtype、c8_k_scale_cache_dtype。

问题是：`isinstance(spec, AscendMLAAttentionSpec)` 只能告诉你"这是 MLA"，但区分不了"这是哪种 MLA"。5 种变体的差异被降级为布尔字段的排列组合，下游必须用 if-else 恢复这些已经丢失的类型信息。**这就是类型擦除——Spec 层本身没有解释自己的能力。**

### Slide 10: Backend 层——"1/N 信息"

上游 `get_kv_cache_shape()` 返回 `(N, B, 576)`——完整可用。Ascend 同名方法也返回 `(N, B, 1, 576)`——但这只是"参考 shape"。Sparse MLA 实际需要 3~4 个独立 tensor（k_nope、k_rope、dsa_k、dsa_k_scale），但接口只能返回 1 个 shape。**剩余 N-1 个 tensor 的 shape 信息被迫泄漏到 model_runner。**

为什么会这样？根本原因是接口设计假设了"1 种 attention → 1 个 tensor"。上游恰好满足这个假设（GPU kernel 接受 blob 内部分拆），但 Ascend NPU 需要 host 侧预先拆分为独立 tensor，单返回值语义空间不够。

### Slide 11-12: Allocate + Reshape——决策树爆炸

因为 Spec 不编码类型、Backend 只返回 1/N 信息，model_runner 被迫自己接管一切：

```
检查 use_compress? use_sparse? use_mla? use_hybrid_blocks? C8? A5? A3? cache_only?
→ Allocate: ~185 行，判断拆几个 tensor + 各自多大
→ Reshape: ~350 行，判断每个 tensor 的 dtype + shape
→ 15+ if-else 分支
```

以 DS V3.2 Sparse MLA 为例追踪完整链路：① Spec 万能类 → ② Backend 返回参考 shape → ③ model_runner 绕过 Backend 直连 attention layer 掏 kv_lora_rank → ④ C8 × A5/A3 × device 三维分支。这个案例清楚地说明：**这不是代码风格问题，是三层设计缺陷的必然结果。**

### Slide 13: as_strided 的职责错配

同一个工具 `as_strided`，在上游只服务 page padding（通用运行时需求），在 Ascend 承载了模型语义——compress ratio、KV merge、scale overlay、epilog kernel 叠加视图。最极端的案例是 DS V4 Compress MLA：allocate 最简单（1 个 int8 buffer），reshape 最复杂（3 个 as_strided overlay views）。这是经典的"复杂度倒挂"——上游 Spec/Layout 没有告诉 reshape "这块 buffer 怎么解读"，reshape 只能靠自己推导。

---

## Part 3: 模型类型差异 + 精度维度 (Slide 14-20, ~5 分钟)

### Slide 14: 五种模型类型对比

从 GQA（差距 ★）到 Hybrid（差距 ★★★★★），差距不是线性叠加——每增加一个模型特性（sparse / C8 / compress / hybrid），Ascend pipeline 就多一层 if-else。上游通过 Spec 子类 + Backend 多态吸收差异；Ascend 把这部分职责全部下沉到了 model_runner。

### Slide 15-17: Qwen3.5 案例

Qwen3.5 是一个特别有说服力的案例——因为它是唯一不需要 `AscendMLAAttentionSpec` 的模型，直接复用上游的 `FullAttentionSpec` + `MambaSpec`。结果是 Spec 和 Backend 几乎无差异，差异全部集中在 Allocate 和 Reshape。

上游用一个 `as_strided_` 完成 `(2,N,B,H,D) → (N,2,B,H,D)` 的零拷贝重排。Ascend 做同样的事情需要 strip conv_padding → 手动算 K 区 offset → 手动算 V 区 offset → split → 组装 tuple。

### Slide 18-20: 精度维度

精度是与模型类型、硬件设备正交的第三维度。同一个 DS V3.2 在三种精度下：tensor 数从 3→4→3（非单调！），每 token 字节数从 1408B→1282B→644B。

上游切换精度：Backend 内部的 shape 最后一维从 576 变成 656，其余完全透明。Ascend 切换精度：需要同步修改 5 个不同位置（page_size_bytes、sparse_kv_cache_ratio、Allocate 分支、Reshape k_shape、Reshape v_shape）。

根因是：GPU kernel 接受打包 blob 内部解析，NPU kernel 需要 host 侧预先拆分且每个 tensor 的 dtype 必须正确。

---

## Part 4: 根因 + 对齐方案 (Slide 21-26, ~5 分钟)

### Slide 22: 六大根因，归结为三个源头

我们把差异归结为六个具体根因，进一步归纳为**三个根本源头**：

1. **硬件约束**：NPU 算子不支持 stride 访问——这是根因中的根因（涵盖 K/V 分离、Sparse 多 tensor、Compress 交织、A3/A5 差异）
2. **Spec 设计**：类型不编码语义——isinstance() 无区分力，下游用 if-else 恢复丢失的类型信息
3. **接口缺陷**：get_kv_cache_shape() 单返回值无法表达多 tensor 布局

三个源头互相放大。如果 NPU 支持 stride，Spec 设计缺陷的影响会小得多。如果 Spec 用了正确的子类拆分，硬件约束的后果也会更可控。两者叠加，再加上接口不足，才导致了 600 行、15+ 分支的 model_runner。

### Slide 23: 差距分析 + 三步路线

差距从分组（★）→ 分配（★★★）→ 内部布局（★★★★★）逐级递增。

已对齐部分属于"规划层"——在 tensor 分配之前，与 NPU 硬件无关，可直接复用上游。待解决部分属于"执行层"——直接依赖 NPU 硬件特性，需要 Layout 层封装。

三步路线：
- **Phase 1 (Q3)**：内部重构——KVCacheLayout 6 子类 + Spec 4~5 子类拆分，model_runner ~135 行
- **Phase 2 (Q4)**：上游 RFC——提议 KVCacheLayout 为硬件无关抽象，Spec 字段通用化
- **Phase 3 (Q1+)**：社区合入 + 清理——删除 model_runner patch，删除 Ascend 万能类

### Slide 24: Layout 重构方案

从 600 行 if-else 到 6 个子类的多态分发：

```python
for layer_name, spec in kv_cache_specs.items():
    layout = spec.get_layout()        # 多态选择
    shapes = layout.get_kv_cache_shape(N, B, H, D)  # list[shape]
    sizes  = layout.split_sizes(total_bytes, spec)   # list[int]
    raw_tensors = [torch.zeros(s, dtype=int8) for s in sizes]
    kv_caches[name] = layout.reshape(raw_tensors, spec)
```

model_runner 不再有 if-else。SparseMLALayout(precision='fp8_c8', device='A5') 在构造时消化全部精度和硬件差异。

### Slide 26: 总结

三句话概括：

1. **三大差异**：算子约束（NPU 不支持 stride）+ 类型擦除（万能类 if-else）+ 接口不足（1/N 信息）
2. **三大对策**：KVCacheLayout 多态 + Spec 子类拆分 + Backend 接口扩展
3. **三步路线**：内部重构（Q3）→ 上游 RFC（Q4）→ 社区合入（Q1+）

最终目标：**model_runner 回归上游的简洁——不检查 layer_type，不判断 C8，不区分 A5/A3，不知道精度。** 从 ~600 行到 ~60 行，从 15+ 分支到零分支。

---

## 结束语

这份汇报的核心信息可以用一句话概括：**上游用类型系统和 Backend 多态实现了 model_runner 对模型差异的零感知；Ascend 目前的三层设计缺陷（Spec 万能类 + Backend 接口不足 + Layout 缺失）将这些差异全部下沉到了 model_runner。我们的方案是通过 Spec 子类拆分 + KVCacheLayout 多态 + 接口扩展，分三步让 Ascend 回归上游的设计原则。**

谢谢，欢迎提问。
