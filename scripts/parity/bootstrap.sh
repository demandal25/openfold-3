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
# Stand the parity harness up on a fresh machine.
#
#   bootstrap.sh probe     what is here: GPUs, ROCm, scheduler, fabric, disk
#   bootstrap.sh install   build the environment and fetch the subset dataset
#   bootstrap.sh verify [N]  two same-seed arms on N devices (default 1);
#                            they must come back IDENTICAL. The verdict only
#                            covers the topology and precision it ran at, so
#                            pass the arm's device count before relying on it.
#
# `probe` is read-only and safe to run first. Nothing else assumes a scheduler,
# a shared filesystem, or a pre-built environment -- those are discovered.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${OF3_PARITY_SCRATCH:-${REPO_ROOT}/tmp/parity}"

say() { printf '\n== %s ==\n' "$1"; }
val() { printf '  %-26s %s\n' "$1" "${2:-<none>}"; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

probe_gpus() {
    say "GPUs"
    if have rocm-smi; then
        val "rocm-smi" "$(command -v rocm-smi)"
        val "gfx arch" "$(rocm-smi --showhw 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1)"
        val "product" "$(rocm-smi --showproductname 2>/dev/null | grep -oE 'MI[0-9]+[A-Z]*' | head -1)"
        # Unique GPU[n] indices: --showid prints several property lines per GPU,
        # so a plain line count reports ~5x the real number.
        val "visible to rocm-smi" "$(rocm-smi --showid 2>/dev/null | grep -oE '^GPU\[[0-9]+\]' | sort -u | wc -l)"
    else
        val "rocm-smi" "NOT FOUND"
    fi
    val "physical (PCI)" "$(lspci -d 1002: 2>/dev/null | grep -ci 'Processing accelerators')"
    val "kfd nodes" "$(ls /sys/class/kfd/kfd/topology/nodes/ 2>/dev/null | wc -l)"
    val "ROCR_VISIBLE_DEVICES" "${ROCR_VISIBLE_DEVICES:-}"
    val "HIP_VISIBLE_DEVICES" "${HIP_VISIBLE_DEVICES:-}"
    val "ROCm version" "$(cat /opt/rocm/.info/version 2>/dev/null)"
    have nvidia-smi && val "nvidia-smi" "PRESENT (mixed box?)"
}

probe_scheduler() {
    say "Scheduler"
    for tool in srun sbatch sinfo squeue kubectl mpirun; do
        have "$tool" && val "$tool" "$(command -v "$tool")"
    done
    val "inside SLURM job" "${SLURM_JOB_ID:-no}"
    val "SLURM_NNODES" "${SLURM_NNODES:-}"
    val "SLURM_JOB_NODELIST" "${SLURM_JOB_NODELIST:-}"
    if have sinfo; then
        echo "  idle GPU nodes:"
        sinfo -h -o "    %N %G %t %P" 2>/dev/null | grep -i "gfx" | grep -w idle | head -6
    fi
}

probe_fabric() {
    say "Interconnect"
    local py=python3
    [[ -x "$REPO_ROOT/.pixi/envs/openfold3-rocm7/bin/python" ]] \
        && py="$REPO_ROOT/.pixi/envs/openfold3-rocm7/bin/python"
    val "RDMA devices" "$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')"
    val "RCCL in torch" "$("$py" -c 'import torch,pathlib;print(" ".join(p.name for p in (pathlib.Path(torch.__file__).parent/"lib").glob("librccl*")) or "none")' 2>/dev/null)"
    val "xGMI links" "$(rocm-smi --showtopo 2>/dev/null | grep -ci xgmi)"
    val "hostname" "$(hostname -f 2>/dev/null || hostname)"
}

probe_host() {
    say "Host"
    val "OS" "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
    val "kernel" "$(uname -r)"
    val "cores (online)" "$(nproc --all 2>/dev/null)"
    val "cores (usable)" "$(nproc)"
    val "RAM (GB)" "$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
    say "Storage"
    df -h --output=target,size,avail,pcent -x tmpfs -x devtmpfs 2>/dev/null | head -12 | sed 's/^/  /'
    echo "  shared filesystems (nfs/lustre/gpfs):"
    mount 2>/dev/null | grep -E "type (nfs|nfs4|lustre|gpfs|ceph)" | awk '{print "    " $3 " (" $5 ")"}' | head -6
}

probe_software() {
    say "Software"
    for tool in git python python3 pixi conda micromamba uv aws docker podman; do
        have "$tool" && val "$tool" "$(command -v "$tool")"
    done
    # Prefer the project environment if it is already built; a bare system
    # python reporting "torch NOT INSTALLED" on a working box is misleading.
    local py=python3
    [[ -x "$REPO_ROOT/.pixi/envs/openfold3-rocm7/bin/python" ]] \
        && py="$REPO_ROOT/.pixi/envs/openfold3-rocm7/bin/python"
    if have "$py" || [[ -x "$py" ]]; then
        val "python probed" "$py"
        val "version" "$("$py" --version 2>&1)"
        "$py" - <<'PY' 2>/dev/null || val "torch" "NOT INSTALLED"
import torch, pathlib
print(f"  {'torch':<26} {torch.__version__}")
print(f"  {'torch.version.hip':<26} {torch.version.hip}")
print(f"  {'cuda available':<26} {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {'device count':<26} {torch.cuda.device_count()}")
    print(f"  {'device 0':<26} {torch.cuda.get_device_name(0)}")
lib = pathlib.Path(torch.__file__).parent / "lib"
ao = sorted(p.name for p in lib.glob("libaotriton*.so.*"))
print(f"  {'aotriton':<26} {ao[-1] if ao else 'not found'}")
PY
    fi
    have run_openfold && val "run_openfold" "$(command -v run_openfold)"
}

probe_network() {
    say "Network reachability"
    # Probe the paths actually used, and report the HTTP code rather than
    # pass/fail: download.pytorch.org answers 403 on / and 200 on /whl/, so a
    # bare -f check calls a working index unreachable.
    for target in https://pypi.org/simple/ \
                  https://github.com \
                  https://download.pytorch.org/whl/ \
                  https://repo.radeon.com \
                  https://openfold3-data.s3.amazonaws.com; do
        local code
        code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$target" 2>/dev/null)"
        local label
        label="$(echo "$target" | sed 's|https://||')"
        case "$code" in
            2*|3*) val "$label" "reachable (HTTP $code)" ;;
            000|"") val "$label" "UNREACHABLE (no response)" ;;
            *)     val "$label" "responds, HTTP $code" ;;
        esac
    done
}

cmd_probe() {
    echo "parity harness -- environment probe"
    echo "repo: $REPO_ROOT"
    probe_host
    probe_gpus
    probe_software
    probe_scheduler
    probe_fabric
    probe_network
    say "Next"
    echo "  Read the above, then: bootstrap.sh install"
    echo "  Anything marked NOT FOUND or UNREACHABLE decides how install proceeds."
}

# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------

cmd_install() {
    say "Installing"
    cd "$REPO_ROOT" || exit 2

    if have pixi; then
        echo "  pixi found; building the openfold3-rocm7 environment"
        pixi install -e openfold3-rocm7 || {
            echo "  pixi install failed -- see the probe's network section" >&2
            exit 2
        }
        local bin="$REPO_ROOT/.pixi/envs/openfold3-rocm7/bin"
        [[ -x "$bin/python" ]] || { echo "  no python in $bin" >&2; exit 2; }
        echo "  environment at $bin"
    else
        echo "  pixi not found. Install it first:" >&2
        echo "    curl -fsSL https://pixi.sh/install.sh | bash" >&2
        exit 2
    fi

    say "Dataset"
    if [[ -d "$REPO_ROOT/datasets/pdb_training_set" ]]; then
        echo "  subset already present"
    else
        echo "  fetching the 8-train/4-val subset (public S3, unsigned)"
        pixi run -e openfold3-rocm7 setup-pdb-subset || {
            echo "  subset download failed" >&2
            exit 2
        }
    fi

    say "Done"
    echo "  next: bootstrap.sh verify"
}

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

cmd_verify() {
    # Defaults to one device for a quick gate; pass the real arm topology to
    # make the verdict cover it. Precision is never overridden -- a machine
    # deterministic in fp32 can still be non-deterministic in bf16.
    local devices="${1:-1}"
    local bin="$REPO_ROOT/.pixi/envs/openfold3-rocm7/bin"
    [[ -x "$bin/python" ]] || { echo "environment missing; run install first" >&2; exit 2; }
    export PATH="$bin:$PATH"
    mkdir -p "$SCRATCH"

    local src="$REPO_ROOT/datasets/train_pdb_subset.yaml"
    [[ -f "$src" ]] || { echo "no subset config at $src; run install first" >&2; exit 2; }

    # The probe refuses to append to an existing trajectory, so a second
    # verify would die looking like a training failure. Named literally.
    rm -rf "$SCRATCH/v1" "$SCRATCH/v2" "$SCRATCH/v1_prov" "$SCRATCH/v2_prov"

    say "Building two identical arms"
    "$bin/python" "$REPO_ROOT/scripts/parity/make_runner_yaml.py" \
        "$src" --out "$SCRATCH/verify.yml" --seed 42 --devices "$devices" \
        --num-workers 2 || exit 2
    "$bin/python" - "$SCRATCH" <<'PY' || exit 2
import sys, yaml
scratch = sys.argv[1]
cfg = yaml.safe_load(open(f"{scratch}/verify.yml"))
cfg["data_module_args"]["epoch_len"] = 4
cfg["pl_trainer_args"].update(
    {"max_epochs": 1, "limit_val_batches": 0, "num_sanity_val_steps": 0}
)
for name in ("v1", "v2"):
    cfg["experiment_settings"]["output_dir"] = f"{scratch}/{name}"
    yaml.safe_dump(cfg, open(f"{scratch}/{name}.yml", "w"), sort_keys=True)
print("  two arms written, same seed, 4 steps each")
PY

    for arm in v1 v2; do
        say "Arm $arm"
        "$REPO_ROOT/scripts/parity/run_arm.sh" \
            --runner-yaml "$SCRATCH/$arm.yml" --output-dir "$SCRATCH/${arm}_prov" \
            --seed 42 --devices "$devices" --num-workers 2 --allow-dirty \
            > "$SCRATCH/$arm.log" 2>&1
        local code=$?
        if [[ $code -ne 0 ]]; then
            echo "  arm $arm failed (exit $code); tail of $SCRATCH/$arm.log:" >&2
            tail -20 "$SCRATCH/$arm.log" >&2
            exit 2
        fi
        echo "  ok, $(wc -l < "$SCRATCH/$arm/logs/parity/trajectory_rank0.jsonl") rows"
    done

    say "Determinism check"
    # --fail-above 0: without it a purely numeric divergence still exits 0,
    # and this gate exists to catch exactly that.
    "$bin/python" "$REPO_ROOT/scripts/parity/compare_trajectories.py" \
        --cross "$SCRATCH/v1/logs/parity" "$SCRATCH/v2/logs/parity" \
        --fail-above 0
    local verdict=$?

    say "Verdict"
    local gpu
    gpu="$("$bin/python" -c 'import torch; print(torch.cuda.get_device_name(0))' \
        2>/dev/null || echo "unknown GPU")"
    local precision
    precision="$("$bin/python" -c \
        'import sys,yaml; print((yaml.safe_load(open(sys.argv[1])) or {})["pl_trainer_args"]["precision"])' \
        "$SCRATCH/v1.yml" 2>/dev/null || echo unknown)"
    if [[ $verdict -eq 2 ]]; then
        cat <<'EOF'
  INCONCLUSIVE -- the trajectories could not be compared at all (missing,
  truncated, or no steps in common). This is not a determinism result. Check
  the arm logs above before drawing any conclusion.
EOF
    elif [[ $verdict -eq 0 ]]; then
        cat <<EOF
  IDENTICAL -- $gpu is bit-deterministic at precision=$precision on $devices
  device(s). Cross-vendor differences measured under the same settings are
  arithmetic, not noise. The verdict covers this GPU, precision and device
  count only -- rerun with the arm topology before relying on it.
EOF
    else
        cat <<'EOF'
  NOT IDENTICAL -- two same-seed runs differ on this machine. Do not run a
  comparison until this is explained: every cross-vendor number would be
  contaminated by the same unexplained variation. Check the probe output for a
  different ROCm/torch/AOTriton stack, and compare provenance.json between the
  two arms.
EOF
    fi
    echo
    echo "  provenance: $SCRATCH/v1_prov/provenance.json"
    return $verdict
}

case "${1:-}" in
    probe)   cmd_probe ;;
    install) cmd_install ;;
    verify)  shift; cmd_verify "$@" ;;
    *) sed -n '16,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
esac
