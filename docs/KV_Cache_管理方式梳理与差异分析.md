# KV Cache 管理方式梳理与差异分析

覆盖模型：GQA（Qwen3 MoE）、MLA（DeepSeek V3.1 / V3.2 / V4）、SFA（GLM5.1）、Hybrid（Qwen3.5）

对比维度：架构设计、Backend 分发机制、Spec 扩展体系、Allocate 管线、Reshape 管线、按模型类型差异分析

---

## 1. 背景与问题

### 1.1 现状

vLLM-Ascend 的 KV Cache 管理当前存在以下问题：

- **与社区差异大**：社区 vLLM 使用统一的 `KVCacheSpec` + `KVCacheManager` 体系，Ascend 侧有大量 patch 和自定义逻辑。当前方向是通过 KVCacheLayout 抽象层封装差异，model_runner 回归与上游一致的简洁模式。
- **model_runner 分支爆炸**：`_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()` 合计约 600 行、15+ 个 if-else 分支，每个新模型类型需要改动 3-5 处。方向是用多态分发替代 if-else，新增模型只需创建/选择 Layout 子类。
- **算子侧对不连续地址空间支持不完整**：导致 K/V 必须物理分离、Sparse MLA 必须在 host 侧拆分为多个独立 tensor。拆分逻辑封装在 Layout 子类中，对上层 model_runner 透明。

### 1.2 架构总览

上游架构：

```
社区 vLLM（两层）：
  Attention.get_kv_cache_spec()  →  KVCacheSpec（声明"是什么"）
       ↓
  AttentionBackend（知道"怎么算"）
       ↓
  model_runner 机械执行（~60 行，无分支）
```

Ascend 架构：

```
vLLM-Ascend（两层 + 隐含分支）：
  MLAAttention.get_kv_cache_spec()  →  AscendMLAAttentionSpec（扩展 6 个 Ascend 字段）
       ↓
  AscendAttentionBackend（NPU 后端，4 类）
       ↓
  model_runner："翻译官"（~600 行，15+ 分支）
  需要读取 Spec 字段 + 全局标志（use_sparse/use_compress/...）
  自行做分配/重塑决策
```

核心差距：上游 Backend 的 `get_kv_cache_shape()` 一个方法回答全部存储问题；Ascend Backend 的同名方法只回答 1/N 的信息（一个 tensor 的 shape），剩余 N-1/N 泄漏到 model_runner 的 if-else 中。

```
        现状                                  目标
   model_runner 知道太多底层细节        model_runner 只知道 Layout 接口

   if sparse:                              layout = spec.get_layout()
      拆 3 个 tensor                       tensors = layout.allocate()
   elif mla:                               layout.reshape()
      拆 2 个 tensor
   elif compress:
      as_strided ...
   ...600 行 if-else                       ...~50 行
```

---

## 2. AttentionBackend 层对比

Backend 是 KV Cache shape 的唯一决定者。每一维的含义、tensor 的数量、K/V 的存储方式，都由 Backend 决定——model_runner 的职责是机械执行，不做模型类型的判断。

上游和 Ascend 的关键差异：上游 Backend 的 `get_kv_cache_shape()` 一个返回值编码了全部布局信息（几个 tensor、各自什么 shape）；Ascend Backend 的同名方法只返回一个 tensor 的 shape，其余 tensor 的数量和 shape 需要 model_runner 自行推断。

上游 `get_kv_cache_shape()`：

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks, block_size, num_kv_heads, head_size,
) -> tuple[int, ...]:   # 返回一个 shape，对应一个 tensor
```

这个签名在设计上就是 1 tensor → 1 shape 的语义。上游恰好只需要 1 个 shape，因为 kernel 在内部解决一切。但 Ascend 的 4 种 Backend 分别需要：

| Backend 类型 | 对应类 | Tensor 数量 | 存储内容 |
| :--- | :--- | :--- | :--- |
| GQA | `AscendAttentionBackend` | 2 | K + V 独立存放 |
| 标准 MLA | `AscendMLABackend` | 2 | k_nope（潜向量）+ k_pe（RoPE 位置信息） |
| Sparse MLA | `AscendSFABackend` | 3 ~ 4 | k_nope + k_pe + k_li（稀疏索引）+ 量化 scale |
| Compress MLA | `AscendDSABackend` | 1（带多重视图） | 1 个完整 int8 Buffer，as_strided 映射为 2~3 个逻辑视图 |

一个 `tuple[int, ...]` 返回类型物理上无法表达这些。Backend 是 static method，拿不到模型参数（如 kv_lora_rank / qk_rope_head_dim），存在传参断层。

对齐方向：

```python
# 现在（信息不够）                      # 目标（信息完整）
@staticmethod                           def get_kv_cache_layout(
def get_kv_cache_shape(                     self,  # 可以访问实例属性
    num_blocks, block_size,                 num_blocks, block_size,
    num_kv_heads, head_size,                num_kv_heads, head_size,
) -> tuple[int, ...]:                   ) -> KVCacheLayout:
    return (N, B, 1, 512)  # 只说了1/N       return MLALayout(
                                                  k_nope=(N, B, 1, 512),
                                                  k_rope=(N, B, 1, 64),
                                              )
```

即：1) 改返回值类型（从单个 shape → 完整多 tensor 布局），2) 打破 static method 限制（Layout 由 Impl 或 Spec 生成），3) model_runner 归一化为 `layout.allocate()` / `layout.reshape()`。

### 2.1 GQA Backend — 差异最小

#### 上游 `FlashAttentionBackend`

```python
# vllm/v1/attention/backends/flash_attn.py
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    if block_size % 16 != 0:
        raise ValueError("Block size must be a multiple of 16.")
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

shape 各维的含义：

```
(2, num_blocks, block_size, num_kv_heads, head_size)
 ↑    ↑           ↑           ↑              ↑
 K/V  物理 block  每个 block   每层 KV        Qwen3: 128
 交织  数量       里的 token   head 数        Llama: 128
```

dim=0 的那个 `2` 是核心设计——K 和 V 在同一个 tensor 里交织存储。CUDA kernel 内部用指针偏移来区分 K 和 V 区域。所以 model_runner 只需要一行：

```python
kv_cache = raw_tensor.view(dtype).view(2, N, B, H, 128)
# 就完了，Backend 说了算
```

#### Ascend `AscendAttentionBackend`

shape 本身跟上游一样 `(2, N, B, H, 128)`，但 model_runner 的用法不同——因为 NPU kernel 要求 K 和 V 各是独立连续的 tensor：

```python
# Ascend 这边必须拆成两个独立 tensor
k_raw, v_raw = kv_cache_raw_tensors[layer_name]  # 分配时就已经是两个独立 int8 buffer

k_shape = kv_cache_shape[1:]       # 去掉 dim=0 的 "2" → (N, B, H, 128)
v_shape = (*kv_cache_shape[1:-1], head_size_v)

k_cache = k_raw.view(dtype).view(k_shape)   # 独立 K tensor
v_cache = v_raw.view(dtype).view(v_shape)   # 独立 V tensor
```

差异本质不在 shape 本身，而在于 CUDA kernel 可以在一个 buffer 内用偏移区分 K/V，但 NPU kernel 需要 host 侧预先拆成两块物理独立的内存。

### 2.2 标准 MLA Backend — 576 这个数丢了内部结构

#### 上游 `MLACommonBackend`

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):
    return (num_blocks, block_size, head_size)
    #      (N,         B,          576)
```

MLA 的 KV cache 不同于 GQA——GQA 每个 head 各存一份 K/V，MLA 则将 attention weight 吸收到潜在空间，num_kv_heads=1，head_size 较大（576 = 512 + 64）：

```
(N, B, 576) → [kv_lora(512维) | k_rope(64维)]
 同一个 tensor 内拼接，kernel 内部按 offset 切分
```

#### Ascend `AscendMLABackend`

返回的是 `(N, B, 1, 576)`——比上游多了 `num_kv_heads=1` 这个维度。然后 model_runner 还得进一步拆成两个 tensor：

```python
k_dim, v_dim = self._get_attention_kv_cache_dims(layer_name, current_kv_cache_spec)
# k_dim = kv_lora_rank = 512
# v_dim = qk_rope_head_dim = 64

k_shape = (N, B, 1, 512)   # nope_cache
v_shape = (N, B, 1, 64)    # rope_cache
```

Backend 不知道 kv_lora_rank 和 qk_rope_head_dim 各是多少，只知道 head_size=576。model_runner 必须绕过 Backend，直接访问 layer 对象的 kv_lora_rank 和 qk_rope_head_dim 属性。

### 2.3 Sparse MLA Backend

#### 上游 `FlashMLASparseBackend`（DS V3.2）

上游 shape 的最后一维不是 head_size，而是固定字节数：

```python
# V3.2 fp8: 返回 (N, B, 656)，不是 576！
# V4 fp8:  返回 (N, B, 584)，也不是 576！
```

这些字节数是 kernel 自定义的私有格式：

```
DS V3.2 fp8: 656 bytes/token
  ├── NoPE (fp8): 512 bytes
  ├── Scale (fp32): 16 bytes
  └── RoPE (bf16): 128 bytes

DS V4 fp8: 584 bytes/token
  ├── NoPE (fp8): 448 bytes
  ├── RoPE (bf16): 128 bytes
  └── Scale (ue8m0): 8 bytes (7+1pad)
```

上游的 model_runner 完全不需要知道这些字节的内部结构——直接把整个 buffer 以 uint8 的形式传给 kernel，kernel 内部自己解析。

#### Ascend `AscendSFABackend`

返回的是 `(N, B, 1, 576)`，跟标准 MLA 一模一样。但实际上 NPU kernel 需要 3~4 个独立 tensor：

```
kv_cache[0]: kv_lora (bf16)     → (N, B, 1, 512)
kv_cache[1]: k_rope (bf16)      → (N, B, 1, 64)
kv_cache[2]: indexer_k (int8/fp8) → (N, B, 1, 128)
kv_cache[3]: indexer_scale (fp16) → (N, B, 1, 1) [只有 C8 量化时才有]
```

model_runner 自行推断 tensor 数量和类型：

```python
if self.use_sparse:
    if current_sparse_c8 and A5:
        # CKV 合并，3 tensors
    elif current_sparse_c8:
        # A3，4 tensors
    else:
        # 无 C8，3 tensors
```

拆分为多 tensor 的原因：1) 各部分 dtype 不同（bf16 / int8 / fp16），无法放入单一 tensor；2) NPU 算子签名固定，每个参数有独立 dtype；3) 访问模式不同（attention kernel vs indexer）。

### 2.4 Compress MLA Backend — as_strided 方案

DS V4 的 `AscendDSABackend` 跟前面的又不一样——它是唯一提供了额外 `get_scale_shape()` 方法的 Backend：

```python
@staticmethod
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (num_blocks, block_size, num_kv_heads, head_size)

@staticmethod
def get_scale_shape(num_blocks, block_size, scale_size):
    return (num_blocks, block_size, scale_size)
```

但 model_runner 仍需知道何时调用 `get_scale_shape()` 以及如何组合这些 shape——这比 Sparse MLA 好一些，但仍未达到上游"一个返回值说完一切"的标准。

### 2.5 Ascend Backend 分发机制

Backend 按 `(use_mla, use_sparse, use_compress)` 三元组分发，而非模型名称：

```python
backend_map = {
    (True,  False, False): AscendMLABackend,       # MLA, 无 sparse, 无 compress
    (False, False, False): AscendAttentionBackend,  # GQA / 标准 Attention
    (True,  True,  False): AscendSFABackend,        # MLA + sparse (DS V3.2, GLM5.1)
    (True,  False, True):  AscendDSABackend,        # MLA + compress (DS V4)
}
```

上游给每个 MLA 变体一个 Backend 子类，Ascend 用一个 Backend 对应一种能力组合。新模型接入时一般复用现有 Backend，调整 spec 参数就行。

### 2.6 `get_kv_cache_shape()` 签名不统一的问题

这个细节也体现了 API 缺乏统一抽象——同一个方法名，参数名不一样：

```python
AscendAttentionBackend: (num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="")
AscendMLABackend:      (num_blocks, block_size, num_kv_heads, head_size, cache_type="")
AscendSFABackend:      (num_blocks, block_size, num_kv_heads, head_size, cache_type="")
AscendDSABackend:      (num_blocks, block_size, num_kv_heads, head_size)  # 没有 cache_type/dtype_str
```

`cache_dtype_str` vs `cache_type` vs 无参数——体现 API 缺乏统一抽象。

### 2.7 Backend 层对照表

| 模型 | 上游 get_shape | Ascend get_shape | 上游用法 | Ascend 用法 |
|------|--------------|-----------------|---------|-----------|
| GQA | (2,N,B,H,128) | (2,N,B,H,128) 相同 | view(2,N,B,H,128) | 拆成 K=(N,B,H,128), V=(N,B,H,128) |
| 标准 MLA | (N,B,576) | (N,B,1,576) 多个 head 维度 | view(N,B,576) | 拆成 nope=(N,B,1,512), rope=(N,B,1,64) |
| Sparse MLA | (N,B,656) fp8 | (N,B,1,576) | view(uint8).view(N,B,656) | 拆 3~4 个 tensor |
| Compress MLA | (N,B,584) fp8 | (N,B,1,hd) + get_scale_shape() | view(uint8).view(N,B,584) | as_strided 2~3 views |

规律：上游 Backend 返回的 shape 就是最终 tensor 的 shape，一个 view() 足够。Ascend Backend 返回的是"参考 shape"，model_runner 需自行决定拆成几个 tensor、各自什么 dtype。

---

## 3. Spec 扩展机制对比

KVCacheSpec 是 frozen dataclass，回答 KV cache 分配前需确认的元信息：block_size、num_kv_heads、head_size、dtype、page_size_bytes 等。

### 3.1 上游 Spec：类型即语义

上游 9 个 Spec 类，每个类的名称直接表达 attention 类型：

```
KVCacheSpec (基类: block_size)
├── AttentionSpec (+num_kv_heads, head_size, dtype)
│   ├── FullAttentionSpec (+head_size_v, sliding_window)
│   │   ├── MLAAttentionSpec (+cache_dtype_str, compress_ratio, ...)
│   │   │   └── HiddenStateCacheSpec
│   │   ├── SinkFullAttentionSpec (+sink_len)
│   │   └── TQFullAttentionSpec
│   ├── SlidingWindowSpec
│   │   └── SlidingWindowMLASpec
│   ├── ChunkedLocalAttentionSpec
│   ├── EncoderOnlyAttentionSpec
│   └── CrossAttentionSpec
├── MambaSpec (+shapes, dtypes, mamba_type)
└── UniformTypeKVCacheSpecs
```

`isinstance()` 即告知 model_runner 全部信息——拿到 `MLAAttentionSpec` 就是标准 MLA，无需检查任何字段。

### 3.2 Ascend Spec：单一类型承载多种语义

Ascend 只有一个 `AscendMLAAttentionSpec`，服务于 DS V3.1（标准 MLA）、DS V3.2（Sparse MLA）、DS V4（Compress MLA）、GLM5.1（SFA）、Qwen3.5（Hybrid）五种模型。

Ascend 新增 6 个字段：

```python
class AscendMLAAttentionSpec(MLAAttentionSpec):
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    sparse_head_dim: tuple[int, ...] | None = None   # (kv_lora_rank, qk_rope_head_dim, index_head_dim)
    cache_sparse_c8: bool = False
    c8_k_cache_dtype: torch.dtype
    c8_k_scale_cache_dtype: torch.dtype
```

同一类，不同模型填充不同字段：

| 字段 | DS V3.1 | DS V3.2 Sparse bf16 | DS V3.2 Sparse C8 | DS V4 | GLM5.1 | Qwen3.5 |
|------|:--:|:--:|:--:|:--:|:--:|:--:|
| sparse_head_dim | None | (512,64,128) | (512,64,128) | None | (512,64,128) | None |
| cache_sparse_c8 | False | False | True | False | False | False |
| scale_dim | 0 | 0 | 0 | 0/1 | 0 | 0 |
| compress_ratio | 1 | 1 | 1 | 4/1 | 1 | 1 |

`AscendMLAAttentionSpec` 的类型名不能区分具体模型——需检查 6 个字段组合。这是"类型擦除"问题：创建 Spec 时已知的类型信息被降级为布尔字段，下游必须重新检查字段以恢复类型知识。

硬件约束与 Spec 设计缺陷互相放大：如果 NPU 完全支持 stride 和单 tensor 内部分解，Spec 类型不区分的影响很小；如果每个模型变体有独立 Spec 子类，硬件约束的后果也更可控。两个源头叠加，使 model_runner 从理论上的 ~100 行膨胀到 600 行。

形成原因：这是渐进式开发的自然结果——v1 只支持 DS V3.1（一个类够用），v2 加 sparse（加 Optional 字段），v3 加 C8（再加字段），v4 加 A5，v5 加 DS V4——每一步最小改动，累积为当前状态。

`AscendSlidingWindowMLASpec` 已按正确的子类方式拆分（无 `cache_sparse_c8`、无 `sparse_head_dim`），是纯 DS V4 SWA MLA 子类，证明该拆分方式在技术上可行，只是尚未推广到所有 MLA 变体。

### 3.3 Spec 创建路径对比

上游 model_runner 只是收集 layer 返回的 spec，不做改写。Ascend 有三条创建路径：

1. **v1 worker 路径**：compress 信任 layer 返回 / GQA 不改写 / Sparse 不信任 layer 自行构建 / 标准 MLA 改写 upstream spec / CacheOnly 重建
2. **v2 worker 路径**：改写为 Ascend 类型
3. **DS V4 模型专属覆盖**：绕过 model_runner，直接返回 Ascend 类型

此外 `patch/__init__.py` 直接 monkey-patch 上游的 `get_kv_cache_spec`。三条路径互不感知，维护复杂度较高。

---

## 4. 分配（Allocate）管线对比

Allocate 阶段任务：根据 KVCacheConfig 创建原始 int8 buffer 并分配给各 layer。

上游用 5 行代码完成；Ascend 需 ~185 行、6 层 if-else。差异根因与 §2 一致：Ascend 需要物理分离的 K/V tensor，且不同模型需要不同数量（1/2/3/4 个）的独立 buffer。

### 4.1 上游：5 行，零分支

```python
def _allocate_kv_cache_tensors(self, kv_cache_config):
    kv_cache_raw_tensors = {}
    for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
        tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=self.device)
        for layer_name in kv_cache_tensor.shared_by:
            kv_cache_raw_tensors[layer_name] = tensor
    return kv_cache_raw_tensors
```

上游能如此简单的原因：1) `KVCacheTensor` 已决定物理布局；2) 每个 layer 只需一个 buffer；3) 不需要对齐。

### 4.2 Ascend：185 行，6 层 if-else

`_allocate_kv_cache_tensors()` 顶层结构：

```
_allocate_kv_cache_tensors()
  ├─ linear_attn / hybrid / cache_only / HiddenStateCache?
  │    → 单 tensor（跟上差不多）
  ├─ use_compress?
  │    → 单 tensor（reshape 时再 as_strided）
  └─ use_sparse?
       ├─ A5 + C8   → 3 tensors (ckv, qli, qli_scale)
       ├─ A3 + C8   → 4 tensors (k, v, qli, qli_scale)
       ├─ 非 C8     → 3 tensors (k, v, qli)
       └─ 非 sparse → 2 tensors (k, v)
                        ├─ FA-quant? → get_kv_quant_split_factor
                        └─ else      → calc_split_factor([k_dim, v_dim])
```

拆分逻辑通过 `sparse_kv_cache_ratio`（Sparse MLA）或 `calc_split_factor`（标准 MLA/GQA）计算子 tensor 字节大小。Ascend PD（Prefill Disaggregation）模式下需 2MB 对齐（RDMA 要求）。

`calc_split_factor` 含义：如 DS V3.1 的 k_dim=512, v_dim=64，total=576，返回 `[576/512, 576/64]` = `[1.125, 9.0]`，即 K 和 V 按最后一维大小比例拆分总字节预算。

---

## 5. 重塑（Reshape）管线对比

Reshape 阶段任务：将 allocate 阶段分配的 raw int8 buffer，按 Backend 声明的 shape 和 dtype 重新解释（view / as_strided），生成 kernel 可直接使用的 KV cache tensor。

上游用统一 `.view(dtype).view(shape)` 模式，所有 layer 走同一代码路径。Ascend 需 ~350 行、7 个分支。

### 5.1 上游：统一模式，Backend 决定一切

Backend 的 `get_kv_cache_shape()` 返回的 shape 即最终 tensor shape。model_runner 无分支：`isinstance(spec, AttentionSpec)` → `backend.get_kv_cache_shape()` → `.view(dtype).view(shape)`。

### 5.2 Ascend：350 行，7 个分支

`_reshape_kv_cache_tensors()` 顶层结构：

```
_reshape_kv_cache_tensors()
  ├─ 分支 1: compress → _adjust_kv_layout() as_strided overlay (2~3 views)
  ├─ 分支 2: sparse → unpack 3~4 tensors, 各自 view
  ├─ 分支 3: hybrid attn+mamba → 切片 padding, strip K/V 区域
  ├─ 分支 4: cache_only/HiddenState → .view(dtype) + as_strided(page_size_padded)
  ├─ 分支 5: 标准 K/V 分离 → GQA 直接 view / MLA 查 layer 后 view
  └─ 分支 6: Mamba → 物理切片 raw[start:end].view(dtype).view(shape)
```

#### 分支 1：Compress MLA — as_strided overlay

DS V4 的 compress MLA 将 K cache 和 scale 按 block 粒度交织存放（block_0: [K|scale], block_1: [K|scale], ...），as_strided 是其自然访问方式。A5 设备上额外包含叠加的 K+scale 组合视图。

#### 分支 2：Sparse MLA — 多 tensor unpack

按设备类型和 C8 开关 unpack 3~4 个 raw tensor。A5 CKV 的 k_shape 最后一维为 `kv_lora_rank + qk_rope_head_dim * 2 + 4 * 4`（kv_lora + k_rope + scale 元数据合并存储）。

#### 分支 3：Hybrid attn+mamba — padding strip

Qwen3.5 的 attention 需要 K/V 分离而 mamba 需要连续内存，因此分配偏大的单 tensor（含 padding），reshape 时切片跳过 padding。

### 5.3 Reshape 核心差异对照

| 维度 | 上游 | Ascend |
|------|------|--------|
| reshape 方式 | 统一 .view(dtype).view(shape) | view / as_strided / 切片 strip 三种 |
| MLA tensor 形状 | (N, B, 576) 单 tensor | (N, B, 1, 512) + (N, B, 1, 64) 双 tensor |
| Sparse MLA | (N, B, 656) uint8 view | unpack 3~4 个 raw tensor,各自 view |
| Compress MLA | .view(dtype).view(shape) | as_strided 2~3 overlay views |
| Hybrid | _update_hybrid_attention_mamba_layout | 额外 conv_block_padding strip |

---

## 6. 按模型类型的完整差异分析

以下按模型类型横向展开，串联 Spec → Backend → Allocate → Reshape 四个阶段的差异。

### 6.1 GQA：Qwen3 MoE / Llama

差异最小的模型类型——Ascend 不扩展 Spec 类，直接复用上游 `FullAttentionSpec`，所有差异集中在 K/V 物理分离。

完整链路：`FullAttentionSpec` → `AscendAttentionBackend`（shape 与上游相同 `(2,N,B,H,128)`）→ Allocate 用 `calc_split_factor([128,128])` 拆 K/V 各 2MB 对齐 → Reshape 丢弃 dim=0 的 `2`，分别 view 成独立 K/V tensor。

### 6.2 标准 MLA：DeepSeek V3.1

从此处开始出现 Ascend 特有差异。`AscendMLAAttentionSpec`(sparse_head_dim=None) → `AscendMLABackend`(shape: `(N,B,1,576)`) → Allocate 时查 layer 得到 kv_lora_rank=512、qk_rope_head_dim=64 → Reshape 拆为 k_shape=(N,B,1,512)、v_shape=(N,B,1,64)。

核心 gap：head_size=576 在 Backend→Spec→model_runner 传递中丢失 512+64 拆分信息，model_runner 必须绕过 Backend 和 Spec 直接访问 MLAAttention layer 对象。

### 6.3 Sparse MLA：DS V3.2 / GLM5.1

最复杂的模型类型。`AscendMLAAttentionSpec`(sparse_head_dim=(512,64,128)) → `AscendSFABackend`(shape 仍为 `(N,B,1,576)`) → Allocate 用 sparse_kv_cache_ratio 按设备/C8 拆 3~4 tensor → Reshape unpack 后各自 view。

GLM5.1（SFA）与 DS V3.2 的 KV cache 结构完全相同，差异仅在 kernel 内部的 RoPE 计算方式和 indexer API 选择——体现在 `AscendSFABackend` impl 层，非 spec/allocate/reshape 层。

### 6.4 Compress MLA：DeepSeek V4

引入 compress_ratio 进行时间维度压缩。Allocate 阶段最简单（单 int8 buffer），Reshape 阶段最复杂（as_strided overlay + A5 全视图）。Backend 提供两个 shape 方法（`get_kv_cache_shape` + `get_scale_shape`），为唯一特例。

### 6.5 Hybrid：Qwen3.5

attention + mamba 混合架构引入了独特问题：attention 需要 K/V 分离的 2 tensor，mamba 需要连续单 tensor，二者共享同一物理 buffer。解决方式：分配偏大单 tensor，reshape 时切片适配。

Qwen3.5 是 Spec 差异最小（直接用上游 `FullAttentionSpec` + `MambaSpec`）但 Reshape 差异很大的模型——证明单纯扩展 Spec 无法覆盖 Backend 的 K/V 物理分离需求和 hybrid buffer layout 管理。

---

## 7. 差异根因分析

### 7.1 K/V 物理分离

原因：PD（PreFill-Decode）分离架构。Prefill 节点 KV cache 需通过 RDMA 传到 Decode 节点，K/V 独立传输更高效（MLA: kv_lora 512 维 vs k_rope 64 维）。2MB 对齐为 RDMA 要求。

### 7.2 Sparse MLA 多 Tensor 拆分

原因：NPU kernel（`aclnnSparseFlashAttention`）需要 3-4 个独立 tensor 作为输入。上游 CUDA kernel 在内部处理单 tensor 的分解。

### 7.3 Compressed MLA as_strided Overlay

原因：A5 设备 epilog kernel 需要叠加的 K+scale 组合视图，as_strided 避免内存复制。

### 7.4 Hybrid Padding

原因：attention 需要 K/V 分离，mamba 需要连续内存——两种需求互斥，通过分配偏大单 tensor + reshape 时切片解决。

### 7.5 Device Variants（A3 vs A5）

A5 使用 fp8 CKV（Combined KV），A3 使用 bf16 分离的 kv_lora + k_rope。影响 Sparse MLA reshape（3 tensor vs 4 tensor）。

### 7.6 算子地址空间限制

NPU attention 算子要求输入 tensor 物理连续，不支持 stride 访问。约束连锁反应：

- K/V 必须物理分离 → allocate/reshape 逻辑复杂化
- Sparse MLA 必须在 host 侧预拆 → 比上游多 2-3 倍 buffer 管理
- as_strided overlay 无法直接传 kernel → reshape 时需"固化"

这是 model_runner 分支爆炸的最根本技术约束。如果 NPU kernel 能像 CUDA kernel 处理 stride 和单 tensor 内部分解，Ascend model_runner 可直接使用上游的 allocate/reshape 实现。

### 7.7 bf16 vs fp8 精度维度分析

精度是与模型类型和硬件设备交叉叠加的正交维度。以 DS V3.2 为例：

| 阶段 | bf16（非 C8） | int8 C8（A3） | fp8 C8（A5） |
|------|:---|:---|:---|
| tensor 数 | 3 | 4 | 3 |
| k dtype | bf16 | bf16 | float8_e4m3fn |
| dsa_k dtype | bf16 | int8 | float8_e4m3fn |
| k_shape | (N,B,1,512) | (N,B,1,512) | (N,B,1,512+128+16) CKV 合并 |
| v_shape | (N,B,1,64) | (N,B,1,64) | None（已合入 CKV） |

上游做法：一个 `cache_dtype_str` 字段——`"auto"` → bf16，`"fp8_ds_mla"` → fp8（656/584 字节）。model_runner 不检查此字段，仅传递给 Backend。精度差异完全封装在 Backend 的 `get_kv_cache_shape()` 内部。

Ascend 无法做到上游那样，因为精度改变不仅改变 shape 最后一维，还改变了 tensor 数量和 dtype——每个子 component 需为独立、dtype 正确的 tensor。上游 CUDA kernel 在内部解析打包格式（struct 字节布局），Ascend NPU kernel 不支持此模式。

---

## 8. 代码行数对比

| 方法 | 上游行数 | Ascend 行数 | Ascend 主要分支 |
|------|---------|-----------|------|
| allocate | ~30 | ~180 | hybrid/mamba/cache_only → compress → sparse_C8_A5/C8_A3/非C8 → standard |
| reshape | ~30 | ~300 | compress → sparse_C8_A5/C8_A3/非C8 → hybrid → cache_only → standard → mamba |
| 辅助方法 | — | ~70 | _adjust_kv_layout / _get_attention_kv_cache_dims / _allocate_int8_cache_tensor |
| **合计** | **~60** | **~550** | **15+ 分支** |

分支按类型分：设备差异 3 处、模型类型 8 处、特殊层 3 处、量化 3 处。

---

## 9. 社区 vLLM KV Cache 管理方案

### 9.1 整体架构

```
Layer → Attention.get_kv_cache_spec() → KVCacheSpec（声明需求）
                                                ↓
get_kv_cache_configs() → KVCacheConfig（全局规划：分组 → 分桶 → 算块数）
                                                ↓
worker 侧 → _allocate_kv_cache() → _reshape_kv_cache() → kv_caches
```

核心数据结构：KVCacheSpec（声明层的物理需求）、KVCacheTensor（描述一个要分配的物理 tensor）、KVCacheGroupSpec（一组共享 block table 的层）、KVCacheConfig（全局配置）。

### 9.2 分层分组方案

上游通过 `get_kv_cache_groups()` 实现了一个决策级联：所有层 spec 相同就走 uniform spec → 同类型不同 hidden_size 走 uniform type → DS V4 走专属分组 → 通用 hybrid 按 spec 类型分组。

hybrid 分组的关键约束：同一 group 内所有层同类型、跨 group 有相同 page_size、每类 group 层数相等（或 padding 补齐）。

### 9.3 内部数据组织

上游所有模型类型均产生 1 个 tensor 或 1 个 int8 buffer + 多个 as_strided view：

| 模型类型 | 每层 Tensor 数 | 内部数据拆分方式 |
|---------|:----:|---------|
| GQA | 1 | K/V 在 dim=0 交织 |
| MLA 标准 | 1 | kv_lora[:512] + k_rope[512:]（kernel 内） |
| MLA fp8 | 1 | 字节布局，kernel 内部按 struct 解析 |
| Mamba | 1 buffer + N views | 多 state tensor 共享 buffer |

model_runner 不需要知道内部数据拆分方式——K/V 关系、kv_lora/k_rope 切分，全部由 Backend 内部处理。

---

## 10. 精选案例：Qwen3.5 Hybrid — 完整链路拆解

Qwen3.5 同时包含 GQA attention 和 Mamba linear attention，覆盖 hybrid 模型的核心挑战。

### 10.1 上游的完整管线

Qwen3.5 的 attention 层用 `FullAttentionSpec`（head_dim=256, num_kv_heads=4），linear attention 层用 `MambaSpec`（conv_state + ssm_state）。Backend 分别是 `FlashAttentionBackend` 和 `GDNAttentionBackend`。

分组时按 Spec 类型分：8 个 full_attn 层 + 24 个 linear_attn 层 → 4 groups（1 组 full_attn × 8 + 3 组 linear_attn × 8）。Allocate 是标准的 30 行 0 分支。Reshape 时 full_attention 走 `.view(bf16).view(2,N,B,4,256)`，linear_attention 走 `as_strided`。Hybrid layout 调整通过 `as_strided_` 做 re-stride。

上游 model_runner 从不检查 `layer_type` 是 full_attention 还是 linear_attention——它只知道 `FullAttentionSpec → view()`、`MambaSpec → as_strided()`。

### 10.2 Ascend 的完整管线

Ascend 的 Qwen3.5 没有模型专属文件——差异全通过 monkey-patch 和 model_runner 的 if-else 实现。

Spec 完全沿用上游（Qwen3.5 是唯一不用 `AscendMLAAttentionSpec` 的模型），但需要给 attention spec 加 `page_size_padded` 来对齐 mamba 的 page_size。

Allocate 走单 tensor 路径（hybrid_with_attn_and_mamba=True），分配一个大的 int8 buffer 让 attention 和 mamba 共享。Reshape 是 Ascend 最复杂的路径——strip conv_padding → 切 K/V 区域 → 分别 view。Mamba state 用物理切片 `raw[start_idx:target_idx]`（而不是上游的 as_strided）。

### 10.3 共享 Buffer 的物理布局

Ascend hybrid buffer 实际排布：`[conv_state padding | K 区域 | V 区域 | ssm_state]`。上游排布：`[K blocks | V blocks]`。

### 10.4 Qwen3.5 的启示

Qwen3.5 是 Spec 差异最小（直接用上游原生 Spec）但 Reshape 差异很大的模型——差异不在 Spec 层，而在 Backend 的 K/V 物理分离需求和 hybrid buffer layout 管理。单纯扩展 Spec 不能解决所有问题。

---

## 11. Layout 重构方向

### 11.1 当前状态

`feature/layout-refactor-phase3` 分支已完成显著重构（12 files, +2080 / -3357 lines）。核心思路：将 15+ 个 if-else 分支替换为 6 个 `KVCacheLayout` 子类的多态分发。

### 11.2 目标架构

```python
class KVCacheLayout(ABC):
    @abstractmethod
    def get_kv_cache_shape(self, ...) -> list[tuple[int, ...]]: ...
    @abstractmethod
    def split_sizes(self, total_bytes, spec) -> list[int]: ...
    @abstractmethod
    def reshape(self, raw_tensors, spec) -> list[Tensor]: ...

# 6 个子类
class SplitKVLayout(KVCacheLayout):       # GQA: K/V 分离，2 tensors
class SparseMLALayout(KVCacheLayout):     # DS V3.2 / GLM5.1: 3-4 tensors
class SparseMLAC8Layout(KVCacheLayout):   # DS V3.2 C8
class CompressedMLALayout(KVCacheLayout): # DS V4: as_strided overlay
class MambaLayout(KVCacheLayout):        # Mamba: 多 state
class SingleTensorLayout(KVCacheLayout):  # cache_only / draft
```

### 11.3 重构前后对比

```
旧代码: ~600 行 if-else, 15+ 分支，散在 model_runner_v1.py
新代码: ~135 行 model_runner 调度 + 625 行 Layout 类（独立可测试）

旧方式加新模型: 找到所有 if-else 分支 → 加分支 → 测试所有旧模型
新方式加新模型: 选/建 Layout 子类 → 实现 3 个方法 → 独立测试新 Layout
```

当前使用 `VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH` 环境变量做 feature gate，默认关闭（走旧路径），设为 1 走新 Layout-driven 路径。

---

## 12. 与社区对齐规划

### 12.1 差异本质

1. 上游 1 tensor/层，kernel 内部拆分；Ascend N tensors/层，host 侧预拆（算子 stride 限制）
2. 上游 `Backend.get_kv_cache_shape()` 返回完整信息；Ascend 同名方法返回 1/N 信息，剩余泄漏到 model_runner
3. 上游每种 attention 类型 = 一种 Spec 子类（类型即语义）；Ascend 所有 MLA 变体挤进一个 `AscendMLAAttentionSpec`（类型擦除）

### 12.2 对齐步骤

**Phase 1（进行中）**：内部重构 — Spec 子类拆分 + Layout 多态 → model_runner ~135 行

**Phase 2**：向上游提 RFC — 提议 KVCacheLayout 作为硬件无关抽象 → Spec 字段通用化

**Phase 3**：社区合入 → 删除 patch → 回归上游模式

### 12.3 已对齐无需处理

以下机制 Ascend 已与上游对齐：`KVCacheSpec` 基类、`KVCacheConfig`/`KVCacheTensor` 数据结构、`get_kv_cache_groups()` 决策级联（仅 DS V4 有 patch）、`BlockPool`/`SingleTypeKVCacheManager`/`KVCacheCoordinator` 体系、Prefix caching、`shared_by` 多层共享机制。

