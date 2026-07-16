# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Compare two KV cache snapshots produced by test_layout_correctness.py.

Checks:
  1. Layer count matches between old and new paths
  2. Container type (tensor / tuple / list) is the same
  3. Number of sub-tensors matches (for tuple/list containers)
  4. Shape and dtype of every tensor match
  5. All tensors are contiguous

Usage:
    python tests/e2e/compare_kv_cache_shapes.py kv_old.json kv_new.json [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

class ComparisonError(Exception):
    """Raised when a mismatch is found between two snapshots."""


def _flatten_entries(name: str, entry: dict) -> list[tuple[str, list[int], str, bool]]:
    """Flatten nested container entries into (name, shape, dtype, contiguous) tuples."""
    flat: list[tuple[str, list[int], str, bool]] = []

    def _walk(prefix: str, e: dict):
        if e.get("container") == "tensor":
            flat.append((prefix, e["shape"], e["dtype"], e.get("is_contiguous", True)))
        elif "entries" in e:
            for i, sub in enumerate(e["entries"]):
                _walk(f"{prefix}[{i}]", sub)
        else:
            flat.append((prefix, [], str(e.get("type", "unknown")), True))

    _walk(name, entry)
    return flat


def _compare_generated_token_ids(
    old: dict[str, Any],
    new: dict[str, Any],
    require_generated_token_ids: bool,
    errors: list[str],
) -> None:
    """Append an error when generated token IDs are required or differ."""
    old_ids = old.get("generated_token_ids")
    new_ids = new.get("generated_token_ids")

    if old_ids is None and new_ids is None:
        if require_generated_token_ids:
            errors.append(
                "Generated token IDs are missing from both snapshots; "
                "rerun without --no-generate."
            )
        return

    if old_ids is None or new_ids is None:
        errors.append(
            "Generated token IDs are present in only one snapshot "
            f"(old={old_ids is not None}, new={new_ids is not None})."
        )
        return

    if not isinstance(old_ids, list) or not isinstance(new_ids, list):
        errors.append("Generated token IDs must be JSON lists.")
        return

    if old_ids != new_ids:
        mismatch_index = next(
            (
                index
                for index, (old_id, new_id) in enumerate(zip(old_ids, new_ids))
                if old_id != new_id
            ),
            min(len(old_ids), len(new_ids)),
        )
        errors.append(
            "Generated token IDs differ at index "
            f"{mismatch_index} (old={old_ids}, new={new_ids})."
        )


def compare_snapshots(
    old_path: str,
    new_path: str,
    strict: bool = False,
    require_generated_token_ids: bool = False,
) -> int:
    """Compare two snapshot JSON files.  Returns 0 on success, 1 on mismatch."""
    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    errors: list[str] = []

    # --- Layer count ---
    if old["num_layers"] != new["num_layers"]:
        errors.append(
            f"Layer count: old={old['num_layers']}, new={new['num_layers']}"
        )

    old_layers = old.get("layers", {})
    new_layers = new.get("layers", {})

    common = set(old_layers) & set(new_layers)
    only_old = set(old_layers) - set(new_layers)
    only_new = set(new_layers) - set(old_layers)

    for name in sorted(only_old):
        errors.append(f"Layer {name}: exists in old but missing in new")
    for name in sorted(only_new):
        errors.append(f"Layer {name}: exists in new but missing in old")

    for name in sorted(common):
        old_flat = _flatten_entries(name, old_layers[name])
        new_flat = _flatten_entries(name, new_layers[name])

        if len(old_flat) != len(new_flat):
            errors.append(
                f"{name}: tensor count mismatch "
                f"(old={len(old_flat)}, new={len(new_flat)})"
            )
            continue

        for (o_name, o_shape, o_dtype, o_contig), (n_name, n_shape, n_dtype, n_contig) in zip(old_flat, new_flat):
            # Names derived from container structure should match
            if o_name != n_name and strict:
                errors.append(f"{name}: sub-tensor naming differs ({o_name} vs {n_name})")

            if o_shape != n_shape:
                errors.append(
                    f"{o_name}: shape old={o_shape} new={n_shape}"
                )
            if o_dtype != n_dtype:
                errors.append(
                    f"{o_name}: dtype old={o_dtype} new={n_dtype}"
                )
            if o_contig != n_contig:
                errors.append(
                    f"{o_name}: contiguous old={o_contig} new={n_contig}"
                )

    _compare_generated_token_ids(
        old,
        new,
        require_generated_token_ids=require_generated_token_ids,
        errors=errors,
    )

    # --- Report ---
    if errors:
        print(f"\n[FAIL] MISMATCH found ({len(errors)} issue(s)):")
        for e in errors:
            print(f"    {e}")
        return 1
    else:
        print(f"\n[PASS] All {len(common)} layers match (shape + dtype + contiguous)")
        print(f"  Old gate: {old.get('gate_enabled', '?')}")
        print(f"  New gate: {new.get('gate_enabled', '?')}")
        print(f"  Old model: {old.get('model', '?')}")
        print(f"  New model: {new.get('model', '?')}")
        print(f"  Old generated: {old.get('generated_text', '')!r}")
        print(f"  New generated: {new.get('generated_text', '')!r}")
        token_ids = old.get("generated_token_ids")
        if token_ids is None:
            print("  Generated token IDs: skipped")
        else:
            print(f"  Generated token IDs: identical ({len(token_ids)} tokens)")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare two KV cache shape snapshots from test_layout_correctness.py"
    )
    parser.add_argument("old_json", help="Snapshot from old path (gate=0)")
    parser.add_argument("new_json", help="Snapshot from new path (gate=1)")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also fail on sub-tensor naming differences",
    )
    parser.add_argument(
        "--require-generated-token-ids",
        action="store_true",
        help="Fail unless both snapshots contain identical generated token IDs",
    )
    args = parser.parse_args()

    try:
        code = compare_snapshots(
            args.old_json,
            args.new_json,
            strict=args.strict,
            require_generated_token_ids=args.require_generated_token_ids,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        code = 2
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON — {e}", file=sys.stderr)
        code = 2

    sys.exit(code)


if __name__ == "__main__":
    main()
