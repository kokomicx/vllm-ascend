#!/bin/bash
#=============================================================================
# vllm-ascend KV Cache Debug 启动脚本
# 适用环境: 8x Ascend910 服务器, NPU 7 (Chip 14/15) 空闲
#
# 用法:
#   bash scripts/run_kv_cache_debug.sh <mode>
#
# mode:
#   normal    - 正常运行，观察日志输出
#   debug     - 插入 breakpoint() 断点调试
#   verbose   - 开启 HMA_DEBUG 日志，打印 KV Cache tensor 细节
#   unit      - 运行单元测试（无需模型权重）
#=============================================================================

set -e

#=============================================================================
# 1. NPU 设备选择 — 只用 NPU 7 (Chip 14/15)
#=============================================================================
# 单 Chip 模式 (推荐学习用, 简单清晰)
export ASCEND_RT_VISIBLE_DEVICES=14
# 双 Chip 模式 (TP=2 时需要)
# export ASCEND_RT_VISIBLE_DEVICES=14,15

#=============================================================================
# 2. 基础环境变量
#=============================================================================
# 使用本地权重，不下载
unset VLLM_USE_MODELSCOPE
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:False  # 关闭 expandable segments, 方便观察显存

#=============================================================================
# 3. 权重根目录 & 模型选择
#=============================================================================
WEIGHT_ROOT=${WEIGHT_ROOT:-/home/c50058674/weight}

# 可用模型 (本地路径, 无需下载)
# ⭐ 首选: Qwen2.5-7B-Instruct — Dense 7B, 标准 FullAttention, 无量化, 28层
#          KV Cache: GQA 4 KV heads × 128 head_size = 标准 shape
MODEL_QWEN25_7B="${WEIGHT_ROOT}/Qwen2.5-7B-Instruct"
# 备选1: Meta-Llama-3.1-8B-Instruct — Dense 8B, 标准 FullAttention
MODEL_LLAMA_8B="${WEIGHT_ROOT}/Meta-Llama-3.1-8B-Instruct"
# 备选2: Qwen3-8B — Dense 8B (可能有 SWA 层, hybrid KV cache)
MODEL_QWEN3_8B="${WEIGHT_ROOT}/Qwen3-8B"

# 默认使用 Qwen2.5-7B-Instruct (标准架构, 最适合学习)
MODEL=${MODEL:-$MODEL_QWEN25_7B}

#=============================================================================
# 4. KV Cache 参数 (为方便观察而调小)
#=============================================================================
# max-model-len 调小 → num_blocks 少 → 更容易观察 block 分配
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}
# 单条 prompt 最大 token 数
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
# 最多同时调度的 token 数
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512}

echo "============================================"
echo " vllm-ascend KV Cache Debug Launcher"
echo "============================================"
echo " NPU Chips:         $ASCEND_RT_VISIBLE_DEVICES"
echo " Model:             $MODEL"
echo " Max Model Len:     $MAX_MODEL_LEN"
echo " Mode:              ${1:-normal}"
echo "============================================"

#=============================================================================
# 5. 健康检查 — 确认 NPU 可用
#=============================================================================
check_npu() {
    echo ""
    echo "[CHECK] Verifying NPU availability..."

    if ! command -v npu-smi &> /dev/null; then
        echo "[WARN] npu-smi not found, skip health check"
        return
    fi

    # 只看我们使用的 chip
    for chip_id in ${ASCEND_RT_VISIBLE_DEVICES//,/ }; do
        echo -n "  Chip $chip_id: "
        npu-smi info -i $chip_id -t health 2>/dev/null | grep -q "OK" && echo "OK" || echo "FAIL"
    done

    # Python 检查
    python3 -c "
import torch
print(f'  torch.npu.is_available(): {torch.npu.is_available()}')
print(f'  torch.npu.device_count():  {torch.npu.device_count()}')
if torch.npu.is_available():
    print(f'  device 0 name:            {torch.npu.get_device_name(0)}')
    props = torch.npu.get_device_properties(0)
    print(f'  total memory:              {props.total_memory / 1024**3:.1f} GB')
"
    echo ""
}

#=============================================================================
# 6. 运行模式
#=============================================================================

MODE=${1:-normal}

case $MODE in

normal)
    # 正常运行模式
    check_npu
    echo "[RUN] Starting normal inference with local model..."
    echo "  Model: $MODEL"
    # 使用环境变量将本地模型路径传给 Python 脚本
    VLLM_MODEL_PATH="$MODEL" python3 -u -c "
import os, sys
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

model_path = os.environ.get('VLLM_MODEL_PATH', '/home/c50058674/weight/Qwen2.5-7B-Instruct')
print(f'[INIT] Loading model from: {model_path}')

from vllm import LLM, SamplingParams

prompts = [
    'Hello, my name is',
    'The capital of France is',
]

sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
llm = LLM(
    model=model_path,
    max_model_len=128,
    max_num_seqs=2,
    enforce_eager=True,
)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f'Prompt: {output.prompt!r}')
    print(f'Generated: {output.outputs[0].text!r}')
    print()
"
    ;;

debug)
    # 断点调试模式
    # 在代码中已插入 breakpoint() 的位置会自动停下
    # 进入 pdb 后常用命令:
    #   p tensor.shape      — 查看 tensor 形状
    #   p tensor             — 打印 tensor 值
    #   n                    — 下一行
    #   s                    — 进入函数
    #   c                    — 继续执行
    #   bt                   — 查看调用栈
    #   interact             — 进入交互式 Python
    check_npu
    echo "[RUN] Starting with breakpoint debugging..."
    echo "  Tips: 进入 pdb 后:"
    echo "    p self.key_cache.shape     — KV cache tensor 形状"
    echo "    p slot_mapping[:50]        — 前 50 个 slot mapping"
    echo "    p request.block_ids        — 当前 request 的 block table"
    echo "    interact                   — 进入交互 Python"
    echo ""
    python3 -u examples/offline_inference_npu_debug.py
    ;;

verbose)
    # 详细日志模式
    check_npu
    export HMA_DEBUG=1
    export VLLM_LOGGING_LEVEL=DEBUG
    echo "[RUN] Starting with verbose KV Cache logging (HMA_DEBUG=1)..."
    python3 -u examples/offline_inference_npu.py 2>&1 | tee kv_cache_debug.log
    echo ""
    echo "[DONE] Log saved to kv_cache_debug.log"
    echo "  grep 'HMA_RESHAPE_AND_CACHE' kv_cache_debug.log  — KV Cache 写入日志"
    echo "  grep 'HMA_FORWARD_FIRST_CACHE' kv_cache_debug.log — KV Cache 首次绑定"
    ;;

unit)
    # 单元测试模式（不需要模型权重！）
    echo "[RUN] Running unit tests (no model needed)..."
    echo ""
    echo "=== Test 1: Block Table / Slot Mapping ==="
    python3 -m pytest tests/ut/worker/a2/test_block_table.py -v 2>&1 | tail -20
    echo ""
    echo "=== Test 2: Attention Forward (mocked) ==="
    python3 -m pytest tests/ut/attention/a2/test_attention_v1.py -v 2>&1 | tail -20
    ;;

op-test)
    # 底层算子测试（合成数据，不加载模型）
    check_npu
    echo "[RUN] Running low-level KV cache op tests..."
    echo ""
    echo "=== Test: transpose_kv_cache_by_block ==="
    python3 -m pytest tests/e2e/nightly/single_node/ops/singlecard_ops/test_transpose_kv_cache_by_block.py -v -s
    ;;

*)
    echo "Usage: bash scripts/run_kv_cache_debug.sh <mode>"
    echo ""
    echo "Available modes:"
    echo "  normal   — 正常运行推理"
    echo "  debug    — 带 breakpoint() 断点调试"
    echo "  verbose  — 开启 HMA_DEBUG 日志, 打印 KV Cache tensor 细节"
    echo "  unit     — 运行单元测试 (不需要模型权重)"
    echo "  op-test  — 运行底层算子测试 (合成数据, 不需要模型)"
    exit 1
    ;;

esac
