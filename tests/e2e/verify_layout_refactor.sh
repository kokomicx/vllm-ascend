#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# One-click verification for Layout-driven KV cache refactoring (Phase 2-3).
#
# Runs the full verification pipeline for a given model:
#   1. Unit tests (CPU — Phase 2 Spec dispatch + Phase 3 Layout dispatch)
#   2. Old-path KV cache snapshot (VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=0)
#   3. New-path KV cache snapshot (VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=1)
#   4. A/B comparison of the two snapshots
#
# Usage:
#   bash tests/e2e/verify_layout_refactor.sh /path/to/model [--max-model-len 2048] [--skip-unit-tests]
#
# Prerequisites:
#   - NPU server with torch-npu installed
#   - vLLM installed and on PYTHONPATH
#   - Model weights accessible at the given path
#
# Exit codes: 0 = all checks passed, 1 = mismatch or test failure.

set -euo pipefail

# ---------------------------------------------------------------------------
# Helpers (source common.sh for colours if available)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/common.sh" ]]; then
    # shellcheck source=./common.sh
    source "$SCRIPT_DIR/common.sh"
else
    _cyan()  { echo -e "\e[96m$*\e[0m"; }
    _red()   { echo -e "\e[31m$*\e[0m"; }
    _info()  { _cyan "Info: $*"; }
    _err()   { _red "Error: $*" && exit 1; }
fi

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
SKIP_UNIT_TESTS=0
MAX_MODEL_LEN=2048
MODEL=""
TMPDIR="${TMPDIR:-/tmp}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-unit-tests)
            SKIP_UNIT_TESTS=1 ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"; shift ;;
        --tmpdir)
            TMPDIR="$2"; shift ;;
        -*)
            _err "Unknown flag: $1" ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
            else
                _err "Only one model can be specified (got: $MODEL and $1)"
            fi ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    _err "Usage: $0 <model_path> [--max-model-len N] [--skip-unit-tests]"
fi

MODEL_NAME="$(basename "$MODEL")"
OLD_JSON="$TMPDIR/kv_cache_${MODEL_NAME}_gate0.json"
NEW_JSON="$TMPDIR/kv_cache_${MODEL_NAME}_gate1.json"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

echo ""
echo "============================================================"
echo " Layout-driven KV Cache Refactoring — Verification Pipeline"
echo "============================================================"
echo ""
_info "Model          : $MODEL"
_info "Max model len  : $MAX_MODEL_LEN"
_info "NPU device     : $ASCEND_RT_VISIBLE_DEVICES"
_info "Snapshots      : $OLD_JSON  /  $NEW_JSON"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Unit tests
# ---------------------------------------------------------------------------
if [[ "$SKIP_UNIT_TESTS" -eq 0 ]]; then
    _info "[1/4] Running Phase 2 Spec-dispatch unit tests ..."
    python "$SCRIPT_DIR/../test_phase2_spec_dispatch.py" || _err "Phase 2 tests failed"

    _info "[1/4] Running Phase 3 Layout-dispatch unit tests ..."
    python "$SCRIPT_DIR/../test_phase3_layout_dispatch.py" || _err "Phase 3 tests failed"

    _info "[1/4] Running KVCacheLayout unit tests ..."
    pytest -q "$SCRIPT_DIR/../test_kv_cache_layout.py" || _err "Layout unit tests failed"

    echo "   [PASS] All unit tests passed"
else
    _info "[1/4] Skipping unit tests (--skip-unit-tests)"
fi

# ---------------------------------------------------------------------------
# Step 2: Old-path snapshot (gate=0)
# ---------------------------------------------------------------------------
_info "[2/4] Capturing KV cache snapshot — OLD path (gate=0) ..."

export VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=0
python "$SCRIPT_DIR/test_layout_correctness.py" \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --output "$OLD_JSON" \
    || _err "Old-path snapshot failed"

_info "   Old-path snapshot saved to $OLD_JSON"

# ---------------------------------------------------------------------------
# Step 3: New-path snapshot (gate=1)
# ---------------------------------------------------------------------------
_info "[3/4] Capturing KV cache snapshot — NEW path (gate=1) ..."

export VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=1
python "$SCRIPT_DIR/test_layout_correctness.py" \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --output "$NEW_JSON" \
    || _err "New-path snapshot failed"

_info "   New-path snapshot saved to $NEW_JSON"

# ---------------------------------------------------------------------------
# Step 4: Compare
# ---------------------------------------------------------------------------
_info "[4/4] Comparing KV cache shapes ..."

python "$SCRIPT_DIR/compare_kv_cache_shapes.py" "$OLD_JSON" "$NEW_JSON" \
    || _err "A/B comparison FAILED — see discrepancies above"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo " [PASS] ALL CHECKS PASSED"
echo "============================================================"
echo ""
_info "Model          : $MODEL_NAME"
_info "Layers         : $(python -c "import json; print(json.load(open('$OLD_JSON'))['num_layers'])")"
_info "Old gate       : $(python -c "import json; print(json.load(open('$OLD_JSON'))['gate_enabled'])")"
_info "New gate       : $(python -c "import json; print(json.load(open('$NEW_JSON'))['gate_enabled'])")"
_info "Old generated  : $(python -c "import json; print(repr(json.load(open('$OLD_JSON'))['generated_text']))")"
_info "New generated  : $(python -c "import json; print(repr(json.load(open('$NEW_JSON'))['generated_text']))")"
_info "Snapshots kept : $OLD_JSON  /  $NEW_JSON"
echo ""

# Clean up snapshots on success?  Keep them for manual inspection.
# rm -f "$OLD_JSON" "$NEW_JSON"
