# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end correctness test for Layout-driven KV cache refactoring.

Compares KV cache tensor shapes and generation outputs between the old
code path (gate=0) and the new Layout-driven code path (gate=1).

Uses the ``VLLM_ASCEND_DUMP_KV_CACHE`` env var to trigger a KV cache
metadata dump from *inside* the engine-core process.  This works
regardless of multiprocess architecture and does not require accessing
internal engine objects from the main process.

Usage (run twice — once per gate value, then compare the JSON outputs):

    # Old path
    VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=0 ASCEND_RT_VISIBLE_DEVICES=0 \\
    python tests/e2e/test_layout_correctness.py \\
        --model /path/to/model --output /tmp/kv_old.json

    # New path
    VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH=1 ASCEND_RT_VISIBLE_DEVICES=0 \\
    python tests/e2e/test_layout_correctness.py \\
        --model /path/to/model --output /tmp/kv_new.json

    # Compare
    python tests/e2e/compare_kv_cache_shapes.py /tmp/kv_old.json /tmp/kv_new.json

For batch verification across all model types, use verify_layout_refactor.sh.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any

import pytest
import torch
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# generate + capture (env-var-driven dump)
# ---------------------------------------------------------------------------

def generate_and_capture(
    model: str,
    max_model_len: int,
    no_generate: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.30,
) -> dict[str, Any]:
    """Load a model, optionally run one short generation, capture KV cache metadata.

    The KV cache metadata is captured via ``VLLM_ASCEND_DUMP_KV_CACHE``,
    which causes ``NPUModelRunner.initialize_kv_cache_tensors`` to write a
    JSON snapshot from inside the engine-core process.  This avoids the need
    to traverse engine internals from the main process.

    When ``no_generate=True``, skips the forward pass and captures KV cache
    shapes right after model initialisation.
    """
    # Use a temp file for the engine-core dump — the engine-core process
    # writes JSON to this path, then we read it back in the main process.
    dump_fd, dump_path = tempfile.mkstemp(suffix=".json", prefix="kv_dump_")
    os.close(dump_fd)
    os.environ["VLLM_ASCEND_DUMP_KV_CACHE"] = dump_path

    try:
        llm = LLM(
            model=model,
            max_model_len=max_model_len,
            max_num_seqs=4,
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )

        if no_generate:
            generated_text = "(skipped — --no-generate)"
        else:
            prompts = ["Hello, how are you?"]
            sampling_params = SamplingParams(max_tokens=8, temperature=0.0, seed=42)
            outputs = llm.generate(prompts, sampling_params)
            generated_text = outputs[0].outputs[0].text

        # Read the snapshot dumped by the engine-core process
        with open(dump_path) as f:
            snapshot = json.load(f)

        snapshot["generated_text"] = generated_text
        snapshot["model"] = model
        snapshot["max_model_len"] = max_model_len

        # Clean up
        del llm
        torch.cuda.empty_cache()

        return snapshot
    finally:
        # Clean up env var and temp file
        os.environ.pop("VLLM_ASCEND_DUMP_KV_CACHE", None)
        try:
            os.unlink(dump_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# pytest fixtures & test
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--model", required=True, help="Path or HF name of the model")
    parser.addoption("--output", default=None, help="Path for JSON snapshot output")
    parser.addoption("--max-model-len", type=int, default=2048)


@pytest.fixture(scope="module")
def snapshot(request):
    """Run one generation and return the KV cache snapshot dict."""
    model = request.config.getoption("--model")
    max_model_len = request.config.getoption("--max-model-len")
    return generate_and_capture(model, max_model_len)


@pytest.mark.skipif(
    os.environ.get("SKIP_NPU_TESTS", "0") == "1",
    reason="SKIP_NPU_TESTS=1",
)
def test_gate_consistency(snapshot: dict[str, Any], request):
    """Verify the gate env var matches the snapshot metadata."""
    gate_in_snapshot = snapshot["gate_enabled"]
    gate_in_env = os.environ.get("VLLM_ASCEND_USE_KV_LAYOUT_DISPATCH", "0")
    assert gate_in_snapshot == gate_in_env, (
        f"Snapshot recorded gate={gate_in_snapshot} but env has gate={gate_in_env}"
    )


@pytest.mark.skipif(
    os.environ.get("SKIP_NPU_TESTS", "0") == "1",
    reason="SKIP_NPU_TESTS=1",
)
def test_kv_cache_not_empty(snapshot: dict[str, Any]):
    """Sanity: KV cache has at least one layer."""
    assert snapshot["num_layers"] > 0, "KV cache has zero layers"


@pytest.mark.skipif(
    os.environ.get("SKIP_NPU_TESTS", "0") == "1",
    reason="SKIP_NPU_TESTS=1",
)
def test_kv_cache_contiguous(snapshot: dict[str, Any]):
    """Every tensor in the KV cache must be contiguous (operator requirement)."""
    non_contiguous: list[str] = []

    def _check(name: str, entry: dict):
        if entry.get("container") == "tensor":
            if not entry.get("is_contiguous", True):
                non_contiguous.append(name)
        elif "entries" in entry:
            for i, sub in enumerate(entry["entries"]):
                _check(f"{name}[{i}]", sub)

    for layer_name, entry in snapshot["layers"].items():
        _check(layer_name, entry)

    assert len(non_contiguous) == 0, (
        f"Found {len(non_contiguous)} non-contiguous tensors: {non_contiguous}"
    )


@pytest.mark.skipif(
    os.environ.get("SKIP_NPU_TESTS", "0") == "1",
    reason="SKIP_NPU_TESTS=1",
)
def test_generated_text_non_empty(snapshot: dict[str, Any]):
    """The model produced at least one token of output."""
    assert len(snapshot.get("generated_text", "")) > 0, (
        "Generated text is empty — model may have failed to run"
    )


# ---------------------------------------------------------------------------
# CLI entry-point (runs without pytest when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow direct execution for quick smoke-testing
    import argparse

    parser = argparse.ArgumentParser(description="Layout-driven KV cache correctness test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation (forward pass) — only capture KV cache shapes "
        "after model init.",
    )
    args = parser.parse_args()

    snapshot = generate_and_capture(
        args.model,
        args.max_model_len,
        no_generate=args.no_generate,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    output_path = args.output
    if output_path is None:
        gate = snapshot["gate_enabled"]
        model_name = os.path.basename(args.model.rstrip("/\\"))
        output_path = f"kv_cache_snapshot_{model_name}_gate{gate}.json"

    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"\nSnapshot saved to {output_path}")
    print(f"  Gate enabled : {snapshot['gate_enabled']}")
    print(f"  Num layers   : {snapshot['num_layers']}")
    print(f"  Generated    : {snapshot['generated_text']!r}")
    for name, entry in snapshot["layers"].items():
        if entry["container"] == "tuple":
            shapes = [e["shape"] for e in entry["entries"]]
            print(f"  {name}: tuple({entry['num_tensors']}) → {shapes}")
        elif entry["container"] == "list":
            shapes = [e["shape"] for e in entry["entries"]]
            print(f"  {name}: list({entry['num_tensors']}) → {shapes}")
        else:
            print(f"  {name}: {entry['shape']}")
