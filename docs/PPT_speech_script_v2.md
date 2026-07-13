# vLLM-Ascend KV Cache 管理优化方案 — 演讲稿（逐页版）

> 对应 PPT：`vllm-ascend-KV cache管理优化方案.pptx`（28 页）
> 预估时长：25~30 分钟

---

## Slide 1 · 封面

各位好，今天我汇报的主题是 **vLLM-Ascend 的 KV Cache 管理方式梳理与差异分析**。

这份汇报覆盖五种模型类型——GQA、标准 MLA、Sparse MLA、Compress MLA、Hybrid——以及 bf16 / int8 C8 / fp8 C8 三种精度。核心目标是回答一个问题：**为什么上游 vLLM 的 KV Cache 管线只有约 60 行、零分支，而 Ascend 的同一条管线膨胀到了约 600 行、15 个以上的 if-else？**

---

## Slide 2 · 汇报路线图

汇报分六个部分。总体思路是"总体→局部→根因→方案"：

先在 Part 1 建立上游 vLLM 端到端管线的共识——Spec、Backend、Allocate、Reshape 四个阶段各自做什么——然后 Part 2 到 Part 5 逐阶段对比 Ascend 的实现，找出每一层的差距。Part 6 把差异归纳为根因，给出从内部重构到社区合入的三步对齐路线。

**每一页都回答同一个问题：上游怎么做？Ascend 怎么做？差距在哪？**

---

## Part 1 · 总体架构（Slides 3-9）

### Slide 3 · 端到端管线全景

先看宏观数字。

上游 vLLM 的 KV Cache 管线大约 60 行代码，model_runner 中零个 if-else 分支——所有模型类型走同一条路径。

Ascend 的同一条管线膨胀到了约 600 行，15 个以上的条件分支，覆盖 5 种模型类型。这个差距不是"代码风格"能解释的——它指向了更深层的设计问题。

---

### Slide 4 · 上游：四阶段管线，Backend 说了算

上游 vLLM 的 KV Cache 管线分四个阶段：

**第一阶段：Spec 创建。** 每一层调用 `get_kv_cache_spec()` 返回一个 frozen dataclass——KVCacheSpec 的子类实例。这个对象声明了 block_size、num_kv_heads、head_size、dtype 和 page_size_bytes。关键在于上游有一套完整的 Spec 子类体系：FullAttentionSpec、MLAAttentionSpec、SlidingWindowMLASpec、MambaSpec——一种 attention 类型对应一种 Spec 子类。这就是"类型即语义"——`isinstance()` 本身就编码了全部信息。

**第二阶段：Backend 返回 Shape。** `get_kv_cache_shape()` 返回完整的 shape——GQA 返回 `(2, N, B, H, D)`，MLA 返回 `(N, B, 576)`。一个 shape 回答全部存储问题。model_runner 拿到 shape 后只需要 `view()` 一下。

**第三阶段：全局规划。** `get_kv_cache_configs()` 统一调度三个子步骤：分组决策（哪些层共享 block table）→ 空间计算（available_memory / page_size_bytes = num_blocks）→ 组装 KVCacheConfig。

**第四阶段：Allocate + Reshape。** 加起来约 60 行，零分支。Allocate 只管总字节数——`torch.zeros(total_bytes, dtype=int8)`——不管是 K 还是 V 还是 conv_state，对分配器来说都一样。Reshape 三步机械操作：`raw → view(dtype) → view(shape) → permute()`，所有模型共用同一套逻辑。特殊类型如 Mamba 和 MLA 用 `as_strided` 改 stride 不拷贝数据，不产生新分支。

核心原则：**Backend 是 KV Cache shape 的唯一决定者。model_runner 只做机械执行——拿到 raw tensor → view(dtype) → view(shape) → permute() → 结束。它不关心 buffer 里面是什么。**

---

### Slide 5 · Ascend：为什么变成了 600 行 if-else

现在看 Ascend 的同一管线。核心发现：**上游的四个阶段在 Ascend 侧全部出现了信息丢失。**

**Spec 层**：上游 1 种模型 → 1 种 Spec 子类。Ascend 所有 MLA 变体挤进同一个 `AscendMLAAttentionSpec`。新增 6 个 NPU 特有字段——scale_dim、sparse_head_dim、cache_sparse_c8 等。`isinstance()` 只能区分"是不是 MLA"，区分不了"是哪种 MLA"。

**Backend 层**：上游 `get_kv_cache_shape()` 返回完整可用 shape。Ascend 同名方法返回 `(N, B, 1, 576)`——但这只是"参考 shape"。Sparse MLA 实际需要 3 到 4 个独立 tensor，接口只返回了其中 1 个的等价 shape。剩余 N-1 个 tensor 的信息泄漏到了 model_runner。

**Allocate + Reshape 层**：model_runner 被迫接管一切——检查 use_sparse? use_compress? use_hybrid_blocks? C8? A5? A3? cache_only? 七到八个标志位——Allocate 约 185 行判断拆几个 tensor 每个多大，Reshape 约 350 行判断每个 tensor 的 dtype 和 shape。15 个以上 if-else，每新增一种模型就要改 3 到 5 个分支点。

**一句话总结核心差距：上游 Backend 一个方法回答了全部存储问题；Ascend Backend 的同名方法只回答了 1/N 的信息，剩下 N-1/N 泄漏到了 model_runner 的 15 个以上 if-else 分支里。**

底部分析了两个根因：一是 NPU 算子不支持 stride——DMA Engine 只接受 {base_addr, length}，host 侧必须预拆分为连续 tensor。二是 Backend 接口语义不足——单返回值 `tuple[int,...]` 无法表达 2 到 4 个 tensor 的布局，拆分信息只能泄漏到外部。

---

### Slide 6 · 社区 KV Cache 管理（一）：分层分组方案

现在暂停对比，先深入理解上游是怎么做到如此简洁的。这一页讲分组方案。

`get_kv_cache_groups()` 的核心思路是"决策级联"——从最简单的情况开始尝试，匹配不上再降级到更复杂的情况。四种情况按复杂度递增：Uniform（所有层完全相同，1 个 group 搞定）→ Uniform Type（同类型但维度不同，GCD 对齐）→ DS V4（三层结构按 page_size 分桶）→ 通用 Hybrid（min_num_layers 分组 + padding 补齐）。

关键设计理念：大部分模型走情况 1 或 2——90% 的场景代码简洁高效。只有真正复杂的模型才触发后面的逻辑。对比 Ascend：所有模型类型的判断平铺在同一个 if-else 里，GQA 和 DS V4 混在一起——这就是 30 行和 600 行的差距。

---

### Slide 7 · 社区 KV Cache 管理（二）：空间分配与大小计算

一切从 `page_size_bytes` 开始——一个 block 占多少字节。不同 Spec 子类独立实现，无 if-else 分支。从 page_size_bytes 出发，除以 available_memory 得到 num_blocks，组装成 KVCacheConfig。

`KVCacheTensor.shared_by` 是实现多层共享的关键机制。以 Qwen3.5 为例：物理上只分配一块 buffer，4 个层（1 个 attn + 3 个 mamba）共用同一个 tensor 对象，各自读取 buffer 中自己需要的区域——attn 层用 K/V 区域，mamba 层用 conv_state 和 ssm_state 区域。互不重叠，零拷贝。

Worker 侧只需要 `torch.zeros(tensor.size, dtype=int8)`——30 行，0 分支。

---

### Slide 8 · 社区 KV Cache 管理（三）：内部数据组织

上游所有模型类型遵循一条原则：**单 tensor，Kernel 自解析。** 一个 layer 一个 tensor，不同类型的组件打包在同一块 buffer 里。GQA 的 K 和 V 在同一 tensor 的相邻区域，MLA 的 kv_lora 和 k_rope 在同一 tensor 的相邻区域，fp8 的多几个 padding 字节还是同一个 tensor。

分工非常清晰：**kernel 负责拆**——传入一个 int8 blob，内部按 struct 布局解析各字段。**host 只管传**——`raw.view(dtype).view(shape)` → 传进去 → 结束。host 侧完全零感知内部结构——精度切换只改 shape 最后一维，其余完全透明。

---

### Slide 9 · 过渡页：社区三原则

在进入逐层对比之前，记住这三条原则。Ascend 的差异不是"实现细节不同"，而是"原则层面的偏离"。

**原则 1：Backend 说了算。** KV Cache shape 的唯一决定者是 Backend，model_runner 不关心内部怎么拆分。

**原则 2：单 Tensor 原则。** 所有模型类型都产生 1 个 tensor 或一个 buffer 加 as_strided views。Host 只管分配，不关心内部数据组织。

**原则 3：model_runner 零感知。** model_runner 不检查 layer_type，不判断 use_sparse、C8、A5/A3。拿到 Spec → 拿到 shape → 分配 → reshape → 结束。约 60 行，零分支，新增模型不改 model_runner。

接下来逐层对比 Ascend，看这三条原则在每一层是如何被打破的。

---

## Part 2-4 · Spec / Backend / Allocate & Reshape（Slides 10-14）

### Slide 10 · Spec 层对比

上游 Spec 层遵循"类型即语义"——每种 attention 类型有独立的 Spec 子类。Frozen dataclass，不可变，声明式。`page_size_bytes` 每个子类独立实现，无分支。

Ascend 所有 MLA 变体挤进同一个 `AscendMLAAttentionSpec`。怎么区分标准 MLA 和 Sparse MLA？只能看 `sparse_head_dim` 有没有值。怎么区分 C8 和 非 C8？只能看 `cache_sparse_c8` 布尔字段。Spec 层本身没有解释自己的能力——`isinstance()` 区分不了，字段排列组合代替了类型层次。

代码对比非常直观：上游 `FlashMLASparseBackend` 的 `get_kv_cache_spec()` 返回一个 `MLAAttentionSpec`——7 个字段，零 NPU 概念。Ascend 的同一层返回 `AscendMLAAttentionSpec`——十几个字段，包含 sparse_head_dim、cache_sparse_c8、scale_dim 等 NPU 特有概念。

**核心差距：上游类型名编码了所有语义；Ascend 只能通过字段值间接推断行为——model_runner 被迫替代 Spec 做解释。**

---

### Slide 11 · Backend 层对比

上游 `get_kv_cache_shape()` 返回即就绪。FlashAttention → `(2,N,B,H,D)`，FlashMLA → `(N,B,576)`——一个 shape 等于一个 tensor 的全部信息。

Ascend 四个 Backend，每个返回的都只是"参考 shape"——`(N,B,1,576)`。Sparse MLA 实际需要 3 到 4 个 tensor（k_nope、k_rope、dsa_k、dsa_k_scale），接口只返回了 1 个 shape。model_runner 被迫绕过 Backend，直接查 attention layer 属性，自己算 split ratio，拆 tensor，算各自 shape，组装 tuple——这一套下来就是约 600 行。

**根本原因：接口设计假设了"1 attention = 1 tensor"。上游恰好满足（GPU kernel 接受 blob 内部分拆），Ascend NPU 不满足（需要 host 预拆分为独立 tensor），单返回值语义空间不够。**

---

### Slide 12 · Allocate 管线对比

上游 Allocate 30 行，统一循环——`torch.zeros(tensor.size, dtype=int8)` + `shared_by` 分发——所有模型通用。

Ascend Allocate 约 185 行，一棵深度 3 的决策树：先判断是 hybrid/mamba 还是 compress 还是 standard attention；standard 里再判断是 sparse 还是普通 MLA；sparse 里再判断是 C8+A5 还是 C8+A3 还是没有 C8。**每次新增模型类型，都要在这棵决策树里找到所有分支点，手动添加新的分支。**

---

### Slide 13 · 实战拆解：DS V3.2 Sparse MLA 完整链路

这是整场汇报最核心的一页——拿一个具体模型，从 Spec 一路追踪到 Reshape，让每一个信息丢失点都暴露出来。

**① Spec 层**：`AscendMLAAttentionSpec` 万能类。`isinstance()` 不区分 5 种 MLA 变体。⚠ 信息丢失：类型擦除。

**② Backend 层**：`get_kv_cache_shape()` 返回 `(N, B, 1, 576)`，只能表达 1/N 信息。实际需要 3 到 4 个 tensor 的不同 shape。⚠ 信息丢失：multi-tensor shape。

**③ Allocate 层**：model_runner 被迫绕过 Backend，直接查 `attn_layer.kv_lora_rank` 和 `attn_layer.qk_rope_head_dim`。⚠ 信息丢失：分层抽象被破坏。

**④ Reshape 层**：C8 × A5/A3 × device 三维分支，约 350 行，15 个以上 if-else。⚠ 信息丢失：代码可维护性。

右侧展示了上游同一条链路的正确形态：`MLAAttentionSpec` 1 类型 7 字段 → `get_kv_cache_shape()` 返回完整 shape → reshape 一行 `view(bf16).view(N,B,576)` → 结束。

底部根因链把四层串了起来：**Spec 万能类 → Backend 接口不足 → Layout 缺失 → 三者叠加 → model_runner 被迫接管所有上游放弃的职责。这不是"方法写太长"的风格问题，是三层设计缺陷的必然结果。**

---

### Slide 14 · Reshape 管线：as_strided 的职责错配

同一个工具 `as_strided`，在上游和 Ascend 服务于截然不同的语义层级。

**上游**：`as_strided` 仅在 `page_size_padded` 时触发——这是一个通用运行时参数，与模型架构无关。所有模型类型共用同一逻辑。

**Ascend**：`as_strided` 服务于模型语义——compress ratio、KV merge、scale overlay、epilog kernel 叠加视图——触发条件变成了模型类型（`use_compress` / `use_sparse` / `hybrid`）。换一个模型或 NPU 架构，`as_strided` 的逻辑就要改。

最极端的案例是 DS V4 Compress MLA：allocate 最简单——1 个 int8 buffer，不拆分；reshape 最复杂——3 个 `as_strided` overlay views（K、scale、K+scale 叠加）。这是经典的"复杂度倒挂"——上游没有告诉 reshape"这块 buffer 怎么解读"，reshape 只能自己推导 compress ratio、overlay stride、scale dtype。

---

## Part 5 · 模型类型差异 + Qwen3.5 案例（Slides 15-18）

### Slide 15 · 五种模型类型完整对比

这张表的核心信息是：**每一列都同时展示上游基线和 Ascend 现状**。从 GQA 到 Hybrid，差距从一颗星递增到五颗星。

上游基线：5 种模型走同一条简洁路径——Spec 子类、Backend 完整 shape、Allocate 单 tensor、Reshape 一行 view。整个 pipeline 约 100 行，model_runner 不感知模型差异。

Ascend 现状：每种模型一个独立分支，复杂度递增。Spec 万能类 if-else 恢复类型、Backend 返回 1/N 信息、Allocate 2 到 4 tensor 三维分支、Reshape 每个模型独立逻辑。model_runner 约 600 行，每加一种模型改一次。

**关键洞察：差距不是线性叠加——每增加一个模型特性（sparse / C8 / compress / hybrid），pipeline 就多一层 if-else。上游通过 Spec 子类加 Backend 多态吸收模型差异；Ascend 把这部分职责全部下沉到了 model_runner。**

---

### Slide 16 · Qwen3.5 案例分隔页

Qwen3.5 是一个特别有说服力的案例——因为它是唯一不需要 `AscendMLAAttentionSpec` 的模型，直接复用上游的 `FullAttentionSpec` 和 `MambaSpec`。32 层中 8 层 full_attention 加 24 层 linear_attention。

---

### Slide 17 · Qwen3.5 上游管线

上游怎么处理这种混合架构？核心原则：**model_runner 从不检查 layer_type。**

Spec 层使用两种上游原生类型——`FullAttentionSpec`（GQA 路径）和 `MambaSpec`（Mamba 路径）。两者的父类只有 `KVCacheSpec`，没有公共 Attention 基类——但类型系统优雅地处理了这种差异。

分组方案：4 groups，每组 8 层（1 attn + 3 linear），共享同一块 buffer。attn 用 K/V 区域，mamba 用 conv+ssm 区域。

Allocate 统一循环，30 行，零分支——没有 `if layer.is_attn` 也没有 `elif layer.is_mamba`。

Reshape 用 `as_strided_` 零拷贝重排——`(2,N,B,H,D)` 变为 `(N,2,B,H,D)`，只调整 stride 不拷贝数据，与 mamba 的 block-dim=0 布局对齐。约 5 行。

**关键信息：上游 model_runner 不!检!查! layer_type。所有模型特有逻辑封装在 Spec 子类和 Backend 内部。**

---

### Slide 18 · Qwen3.5 Ascend 管线 + 对比

Ascend 管线呈现一个有趣的模式：**差异集中在管线的后半段。**

Spec 和 Backend 几乎无差异——直接复用上游的 `FullAttentionSpec` 和 `MambaSpec`，`get_kv_cache_shape()` 返回的 `(2,N,B,4,256)` 与上游完全相同。Qwen3.5 是唯一不需要"二次解读"shape 的 Ascend 模型。

但 Allocate 和 Reshape 差异集中爆发——约 100 行 hybrid 专用代码。Reshape 需要手动 strip mamba conv 前缀、手动算 K 区 offset、手动算 V 区 offset——上游一个 `as_strided_` 解决的事情，Ascend 需要四步手动操作。

底部的 Buffer Layout 对比非常直观：上游单 tensor 内 K 和 V 自然交织，`as_strided_` 零拷贝改变逻辑视图。Ascend 物理上分成四个区域——conv_state、K blocks、V blocks、ssm_state——需要手动 strip、手动 split，每一步都要自己算 offset。

---

## Part 6 · 精度维度（Slides 19-21）

### Slide 19 · 精度维度分隔页

精度是与模型类型、硬件设备完全正交的第三维度。同一个 DS V3.2 在三种精度下表现完全不同。

---

### Slide 20 · 精度维度总览

同一个 DS V3.2 Sparse MLA 在三种精度下：tensor 数量 3→4→3 非单调变化。每 token 字节数从 1408 字节降到 1282 再到 644。这些差异不是模型造成的、不是硬件造成的——纯粹是精度选择造成的。

**精度是独立变量。**

---

### Slide 21 · 上游 vs Ascend 精度处理

上游切换精度：Backend 内部 shape 最后一维从 576 变成 656——其余一切不变——tensor 数量不变、dtype 不变、分支不变。`cache_dtype_str` 一个字段封装全部精度差异。

Ascend 切换精度：需要同步修改 5 个不同位置——page_size_bytes、sparse_kv_cache_ratio、Allocate 分支、Reshape k_shape、Reshape v_shape。这 5 处分散在 model_runner 不同函数里，每加一种精度都要找到所有分支点手动保持一致。

根因分析：GPU kernel 接受打包 blob 内部解析，NPU kernel 需要每个子 component 都是独立且 dtype 正确的 tensor。精度差异在 GPU 侧被 Backend 内部消化，在 NPU 侧泄漏到了整个 pipeline。

目标方案：`Layout(precision, device)` 在构造时接受精度和硬件参数，内部消化全部差异，model_runner 回归"不感知精度"。

---

## Part 7 · 根因 + 对齐（Slides 22-28）

### Slide 22 · 根因 + 对齐分隔页

前面五部分是诊断——我们找到了每一层的差距。现在是方案——把差距归纳为根因，给出可执行的路线。

---

### Slide 23 · 六大根因，三个源头

六个具体根因进一步归纳为**三个根本源头**：

**硬件约束**：NPU 算子不支持 stride 访问，host 侧必须预拆分 tensor。涵盖 K/V 物理分离、Compress 交织、A3/A5 差异等根因 1、2、4、5。

**Spec 设计**：类型不编码语义。`isinstance()` 无区分力，下游用 if-else 恢复丢失的类型信息。根因 3。

**接口缺陷**：`get_kv_cache_shape()` 单返回值无法表达多 tensor 布局。根因 6。

三个源头互相放大。如果 NPU 支持 stride，Spec 设计缺陷的影响会小得多。如果 Spec 用了正确的子类拆分，硬件约束的后果也会更可控。两者叠加，再加上接口不足，才导致了 600 行、15 个以上分支的 model_runner。

解决方案也在这一页——Layout 重构（多态分发替代 if-else，KVCacheLayout 6 个子类）和社区对齐（5 大任务加上三步路线）。

---

### Slide 24 · 差距分析 + 三步路线

差距从分组（★ 最小）到分配（★★★ 关键）到内部布局（★★★★★ 根本）逐级递增。

已对齐部分属于"规划层"——在 tensor 分配之前，与 NPU 硬件无关，可直接复用上游。待解决部分属于"执行层"——直接依赖 NPU 硬件特性，需要 Layout 层封装。

三步路线：
- **Phase 1（Q3）内部重构**：KVCacheLayout 6 子类加 Spec 4 到 5 子类拆分。关键交付包括 StandardKVLayout、SparseMLALayout、CompressedMLALayout、HybridLayout、CacheOnlyLayout。
- **Phase 2（Q4）上游 RFC**：向社区提议 KVCacheLayout 为硬件无关抽象。关键交付是 RFC 文档和原型代码。
- **Phase 3（Q1+）社区合入加清理**：删除 model_runner patch、删除 AscendMLAAttentionSpec NPU 特有字段。关键交付是社区 PR 合入和代码清理。

---

### Slide 25 · Layout 重构方案

左边是当前的 if-else 分发——约 600 行，use_compress、use_sparse、use_hybrid_blocks、C8、A5——每层嵌套都是新的分支维度。

右边是目标的多态分发。KVCacheLayout 抽象基类定义三个方法：`get_kv_cache_shape()` 返回 `list[shape]` 而不是单个 shape、`split_sizes()` 返回每个 sub-tensor 的字节数、`reshape()` 返回 dtype 正确的 tensor 列表。

model_runner 的核心逻辑简化为 6 行伪代码：`spec.get_layout()` → `layout.get_kv_cache_shape()` → `layout.split_sizes()` → `torch.zeros()` → `layout.reshape()`。不再有 if-else。Layout 子类内部消化所有差异。

---

### Slide 26 · 已对齐机制 + 待覆盖 Gap

规划层——分组决策、block 管理、哈希计算、配置生成——与 NPU 硬件无关，Ascend 直接使用上游代码，对齐度接近 100%。

执行层——Allocate 和 Reshape——差异巨大。上游拿到 KVCacheConfig → `torch.zeros` → `view` → done。Ascend 拿到 KVCacheConfig → 检查 5 个标志 → 拆 tensor → 算各自 shape/dtype → view → 组装 tuple。这正是 KVCacheLayout 要封装的层次。

DS V4 目前还有 patch 覆盖两个函数：`resolve_kv_cache_block_sizes()` 和 `group_and_unify_kv_cache_specs()`。上游已支持 DS V4，这些 patch 未来可通过 Layout 重构合并或删除。

---

### Slide 27 · 总结

最后的总结可以用四句话概括：

**三大差异**：算子约束（NPU 不支持 stride）→ 类型擦除（万能类 if-else）→ 接口不足（1/N 信息）

**三大对策**：KVCacheLayout 多态 → Spec 子类拆分 → Backend 接口扩展

**三步路线**：Q3 内部重构 → Q4 上游 RFC → Q1+ 社区合入

**最终目标**：model_runner 回归上游的简洁——**不检查 layer_type，不判断 C8，不区分 A5/A3，不知道精度。** 从约 600 行到约 60 行，从 15 个以上分支到零分支，从 6 条路径到 1 条通用路径，每个 Layout 子类独立可测。

---

## 结束语

这份汇报的核心信息可以用一句话概括：**上游用类型系统和 Backend 多态实现了 model_runner 对模型差异的零感知；Ascend 目前的三层设计缺陷——Spec 万能类、Backend 接口不足、Layout 缺失——将这些差异全部下沉到了 model_runner。方案是通过 Spec 子类拆分、KVCacheLayout 多态、接口扩展，分三步让 Ascend 回归上游的设计原则。**

谢谢，欢迎提问。
