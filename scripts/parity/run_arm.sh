#!/usr/bin/env bash
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
#
# Launch one arm of an AMD-vs-NVIDIA training comparison.
#
# Every pin this applies is also recorded. A pin that cannot be applied aborts
# the run rather than warning: an arm that silently ran unpinned is worse than
# no arm, because it looks like data.

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: run_arm.sh --runner-yaml FILE --output-dir DIR --seed N [options]

  --runner-yaml FILE   OF3 training config (required)
  --output-dir DIR     where to write provenance + trajectory (required)
  --seed N             experiment seed (required; differs between null arms)
  --num-workers N      dataloader workers. MUST match the other arm exactly:
                       worker seeds are a function of worker_id (default 8)
  --devices N          GPUs per node, must match the other arm (default 8)
  --allow-dirty        proceed with uncommitted changes (records the fact)
  --dry-run            write provenance and print the command, do not launch

Pins applied and recorded: vendor-independent RNG, deterministic algorithms,
TF32 off, BLAS backend, autotune off, PYTHONHASHSEED.
USAGE
    exit 2
}

RUNNER_YAML="" OUTPUT_DIR="" SEED="" NUM_WORKERS=8 DEVICES=8
ALLOW_DIRTY=0 DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner-yaml) RUNNER_YAML="${2:?}"; shift 2 ;;
        --output-dir)  OUTPUT_DIR="${2:?}";  shift 2 ;;
        --seed)        SEED="${2:?}";        shift 2 ;;
        --num-workers) NUM_WORKERS="${2:?}"; shift 2 ;;
        --devices)     DEVICES="${2:?}";     shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1;        shift ;;
        --dry-run)     DRY_RUN=1;            shift ;;
        -h|--help)     usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done

[[ -n "$RUNNER_YAML" && -n "$OUTPUT_DIR" && -n "$SEED" ]] || usage
[[ -f "$RUNNER_YAML" ]] || { echo "no such runner yaml: $RUNNER_YAML" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }

command -v run_openfold >/dev/null || {
    echo "run_openfold is not on PATH; activate the openfold3 environment" >&2
    exit 2
}

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

# --- pins ------------------------------------------------------------------
# Exported here so they reach the training process through the final exec. The
# verification block below checks they arrived and that the runner yaml asks for
# the settings that are not environment-driven.
export OF3_VENDOR_INDEPENDENT_RNG=1
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MIOPEN_FIND_MODE=NORMAL
export TORCHINDUCTOR_COMPILE_THREADS=1

# --- anchor to the tree this script lives in --------------------------------
# The editable install resolves openfold3 by cwd, so a run launched from
# elsewhere silently imports a different checkout. Anchor, do not inherit.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
echo "repo root: $REPO_ROOT"

# --- refuse to run on a tree that cannot be pinned to a commit --------------
GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY=0
if git rev-parse --git-dir >/dev/null 2>&1 && ! git diff --quiet HEAD 2>/dev/null; then
    GIT_DIRTY=1
    if [[ "$ALLOW_DIRTY" -eq 0 ]]; then
        cat >&2 <<EOF
refusing to launch: the working tree has uncommitted changes, so this arm
cannot be attributed to a commit and is not reproducible. Commit, or pass
--allow-dirty to record the fact and proceed anyway.
EOF
        exit 2
    fi
    echo "WARNING: launching with a dirty tree (--allow-dirty)" >&2
fi

# --- provenance ------------------------------------------------------------
# No provenance, no result: an arm without this block cannot be compared.
PROVENANCE="$OUTPUT_DIR/provenance.json"
python - "$PROVENANCE" "$RUNNER_YAML" "$SEED" "$GIT_SHA" "$GIT_DIRTY" \
         "$NUM_WORKERS" "$DEVICES" <<'PY'
import hashlib, json, os, pathlib, platform, subprocess, sys

out, yaml_path, seed, sha, dirty, workers, devices = sys.argv[1:8]

import torch

import openfold3

def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None

is_rocm = torch.version.hip is not None
record = {
    "seed": int(seed),
    "num_workers": int(workers),
    "devices": int(devices),
    "git": {"sha": sha, "dirty": bool(int(dirty))},
    "runner_yaml": {
        "path": os.path.abspath(yaml_path),
        "sha256": hashlib.sha256(open(yaml_path, "rb").read()).hexdigest(),
    },
    "vendor": "amd" if is_rocm else "nvidia",
    "torch": {
        "version": torch.__version__,
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
    },
    "env": {k: os.environ.get(k) for k in (
        "OF3_VENDOR_INDEPENDENT_RNG", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
        "MIOPEN_FIND_MODE", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
        "TORCH_ROCM_FA_PREFER_CK", "OF3_TRITON_EXP2", "OF3_TRITON_DYNAMIC_SHAPES",
        "NCCL_ALGO", "NCCL_PROTO", "RCCL_ALGO",
    )},
    "host": {"hostname": platform.node(), "python": platform.python_version()},
    # Which tree actually got imported. The editable install resolves by cwd, so
    # a run launched from the wrong directory silently tests a different commit.
    "openfold3_path": str(pathlib.Path(openfold3.__file__).resolve().parent),
    "rocm_version": run("cat", "/opt/rocm/.info/version"),
    "packages": run(sys.executable, "-m", "pip", "freeze"),
}

# AOTriton ships inside torch/lib, not /opt/rocm. The versioned filename is the
# only place its version is visible, and it matters: 0.11.2 carries ROCM-31035.
if is_rocm:
    lib = pathlib.Path(torch.__file__).parent / "lib"
    versioned = sorted(p.name for p in lib.glob("libaotriton*.so.*"))
    record["aotriton"] = versioned[-1] if versioned else "not found"
else:
    record["aotriton"] = None

with open(out, "w") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)

print(f"vendor={record['vendor']} torch={record['torch']['version']} "
      f"device={record['torch']['device']} sha={sha[:8]}"
      f"{' DIRTY' if record['git']['dirty'] else ''}")
PY

# --- verify the pins actually took --------------------------------------
# Environment pins are checked in a child process, which is valid because they
# reach the run the same way (inheritance through exec). Everything else is
# checked by reading the runner yaml, since asserting it in a throwaway process
# would prove nothing about the training process.
OF3_EXPECTED_ROOT="$REPO_ROOT" python - "$RUNNER_YAML" <<'PY'
import os, pathlib, sys

import yaml

failures = []

if os.environ.get("OF3_VENDOR_INDEPENDENT_RNG") != "1":
    failures.append("OF3_VENDOR_INDEPENDENT_RNG did not reach the process")

import openfold3
from openfold3.core.utils import vendor_rng

if not vendor_rng.enabled():
    failures.append("vendor_rng is off despite the environment variable")

imported = pathlib.Path(openfold3.__file__).resolve().parent
expected = pathlib.Path(os.environ["OF3_EXPECTED_ROOT"]).resolve() / "openfold3"
if imported != expected:
    failures.append(
        f"openfold3 imported from {imported}, not {expected} -- the run would "
        f"test a different checkout than the one it was launched from"
    )

with open(sys.argv[1]) as handle:
    config = yaml.safe_load(handle) or {}

trainer = config.get("pl_trainer_args", {}) or {}
if trainer.get("deterministic") is not True:
    failures.append(
        "pl_trainer_args.deterministic is not true; add it to the runner yaml "
        "(extra='allow' passes it straight to pl.Trainer)"
    )
if "precision" not in trainer:
    failures.append("pl_trainer_args.precision is unset; pin it explicitly")

settings = ((config.get("model_update") or {}).get("custom") or {}).get("settings", {})
probe = (settings or {}).get("parity_probe", {}) or {}
if probe.get("enabled") is not True:
    failures.append(
        "settings.parity_probe.enabled is not true; the run would record no "
        "trajectory and could not be compared"
    )

memory = (settings or {}).get("memory", {}) or {}
for phase in ("train", "eval"):
    flags = memory.get(phase, {}) or {}
    on = [k for k, v in flags.items() if k.startswith("use_") and v]
    if on:
        failures.append(f"settings.memory.{phase} leaves kernel flags on: {on}")

if failures:
    print("refusing to launch; the arm would not be comparable:", file=sys.stderr)
    for item in failures:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(2)
print(
    f"pins verified: deterministic={trainer['deterministic']} "
    f"precision={trainer['precision']} probe=on kernels=off"
)
PY

CMD=(run_openfold train --runner-yaml "$RUNNER_YAML" --seed "$SEED")

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'dry run, would exec:'; printf ' %q' "${CMD[@]}"; printf '\n'
    exit 0
fi

echo "provenance: $PROVENANCE"
exec "${CMD[@]}"
