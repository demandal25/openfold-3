#!/usr/bin/env python
# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Overlay the parity settings onto an existing OF3 runner yaml.

Hand-editing two configs is how the two arms end up differing in something
nobody wrote down. This applies one fixed overlay, so the only intended
differences between arms are the seed and the dataset root, and prints a hash of
the parity-relevant subset that both legs must agree on.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import yaml

#: Applied to every parity arm. Kernel flags off on both train and eval: Triton
#: triangle kernels default to True for eval and False for train, so a run left
#: at defaults uses different kernels in its validation loop than in training.
OVERLAY = {
    "pl_trainer_args": {
        "deterministic": True,
        # Pinned: several micro-batches per optimizer step complicate the
        # trajectory keying for no benefit at batch_size 1.
        "accumulate_grad_batches": 1,
        "precision": "bf16-mixed",
        "num_nodes": 1,
        "deepspeed_config_path": None,
        "mpi_plugin": False,
    },
    "model_update": {
        "custom": {
            "settings": {
                "parity_probe": {
                    "enabled": True,
                    "every_n_steps": 1,
                    "abort_on_nonfinite": True,
                },
                "debug": {"log_grad_norm": True, "log_extra_grad_metrics": True},
                # The trainer arg above is forced to 1 under per-sample
                # clipping (on by default); this is the knob that then applies.
                "manual_optimization": {"accumulate_grad_batches": 1},
                "memory": {
                    "train": {
                        "use_deepspeed_evo_attention": False,
                        "use_cueq_triangle_kernels": False,
                        "use_triton_triangle_kernels": False,
                        "use_lma": False,
                    },
                    "eval": {
                        "use_deepspeed_evo_attention": False,
                        "use_cueq_triangle_kernels": False,
                        "use_triton_triangle_kernels": False,
                        "use_lma": False,
                    },
                },
            }
        }
    },
    "experiment_settings": {
        # An explicit path, never "last": the bare form is a full Lightning
        # resume, restoring optimizer moments, scheduler position and loop
        # counters, which silently makes an arm incomparable.
        "restart_checkpoint_path": None,
    },
}

#: Everything except the intended per-arm differences is fingerprinted.
#: Excluding whole top-level blocks hid real mismatches: dataset_configs
#: carries crop size and dataset weights, which are parity-critical and are
#: not the same thing as dataset_paths.
FINGERPRINT_EXCLUDE_TOP = ("dataset_paths",)
FINGERPRINT_EXCLUDE_NESTED = (
    ("experiment_settings", "seed"),
    ("experiment_settings", "output_dir"),
)


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursive dict merge; overlay wins. Lists are replaced, not extended."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def fingerprint(config: dict) -> str:
    subset = copy.deepcopy(config)
    for key in FINGERPRINT_EXCLUDE_TOP:
        subset.pop(key, None)
    for outer, inner in FINGERPRINT_EXCLUDE_NESTED:
        if isinstance(subset.get(outer), dict):
            subset[outer].pop(inner, None)
    return hashlib.sha256(
        json.dumps(subset, sort_keys=True, default=str).encode()
    ).hexdigest()


def build(source: dict, *, seed: int, devices: int, num_workers: int) -> dict:
    config = deep_merge(source, OVERLAY)
    config.setdefault("experiment_settings", {})["seed"] = seed
    config.setdefault("pl_trainer_args", {})["devices"] = devices
    config.setdefault("data_module_args", {})["num_workers"] = num_workers
    return config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="runner yaml to start from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--devices", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="must match the other arm exactly: worker seeds derive from worker_id",
    )
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: no such file: {args.source}", file=sys.stderr)
        return 2
    source = yaml.safe_load(args.source.read_text())
    if not isinstance(source, dict):
        print(f"error: {args.source} is not a yaml mapping", file=sys.stderr)
        return 2

    config = build(
        source, seed=args.seed, devices=args.devices, num_workers=args.num_workers
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(config, sort_keys=True))

    print(f"wrote {args.out}")
    print(f"parity fingerprint: {fingerprint(config)}")
    print("  both arms must print the same fingerprint; only seed and dataset")
    print("  paths may differ between them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
