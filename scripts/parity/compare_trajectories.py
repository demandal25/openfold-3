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

"""Compare two training trajectories step by step.

Two runs launched with the same seed share their initial weights, their recycle
schedule and their sample order, so the only thing left to differ is arithmetic.
Measured on MI355X: two same-seed runs are bit-identical, 0.0 at every step. The
comparison is therefore a direct diff, not a statistical test -- no envelope, no
threshold.

What this reports is where the two runs stopped agreeing, how fast the gap
grows, and whether the cause was arithmetic or a desync. A desync means the two
runs did different work, which makes the numbers incomparable rather than merely
different, so it is checked first.

Pass ``--reference`` to plot a second pair alongside for scale -- the natural
one is the same vendor with a different BLAS backend, which shows how far a
legitimate implementation swap moves the trajectory.

Exit codes: 0 compared, 1 actionable finding (non-finite, desync, or over
``--fail-above``), 2 unusable input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-12


class InputError(Exception):
    """The trajectories cannot be compared at all."""


@dataclass
class Trajectory:
    """One arm's records, indexed for comparison."""

    label: str
    #: step -> (module, ordinal) -> row
    activations: dict[int, dict[tuple[str, int], dict]] = field(default_factory=dict)
    grad_summary: dict[int, dict] = field(default_factory=dict)
    batch_ids: dict[int, list[str]] = field(default_factory=dict)
    losses: dict[int, list[float]] = field(default_factory=dict)
    firings: dict[int, dict[str, int]] = field(default_factory=dict)
    nonfinite_steps: list[int] = field(default_factory=list)

    @property
    def steps(self) -> set[int]:
        return set(self.activations) | set(self.grad_summary)


def load(path: Path, label: str) -> Trajectory:
    """Read a trajectory from a JSONL file or a run directory."""
    if path.is_dir():
        candidates = sorted(path.glob("trajectory_rank*.jsonl"))
        if not candidates:
            raise InputError(f"no trajectory_rank*.jsonl under {path}")
        # Rank 0 only: other ranks see different samples by construction.
        path = candidates[0]
    if not path.is_file():
        raise InputError(f"not a file or directory: {path}")

    traj = Trajectory(label=label)
    bad_lines = 0
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            kind, step = row["kind"], int(row["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # A run killed mid-write leaves a truncated final line; tolerate a
            # few, but a file that is mostly garbage is not a trajectory.
            bad_lines += 1
            if bad_lines > 10:
                raise InputError(
                    f"{path}: {bad_lines} unparseable lines by line {lineno}"
                ) from None
            continue

        if row.get("nonfinite"):
            traj.nonfinite_steps.append(step)
        if kind == "activation":
            traj.activations.setdefault(step, {})[
                (row["module"], int(row["ordinal"]))
            ] = row
        elif kind == "grad_summary":
            traj.grad_summary[step] = row
        elif kind == "batch":
            traj.batch_ids[step] = [str(i) for i in row.get("ids", [])]
        elif kind == "loss":
            traj.losses.setdefault(step, []).append(float(row["value"]))
        elif kind == "firings":
            traj.firings[step] = dict(row.get("counts", {}))

    if not traj.steps:
        raise InputError(f"{path}: no comparable records")
    return traj


def rel_diff(a: float, b: float) -> float:
    """Relative difference, symmetric and safe at zero."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b), EPS)


def step_divergence(
    left: Trajectory, right: Trajectory, step: int, fields=("mean", "std", "absmax")
) -> tuple[float, str | None]:
    """Largest relative difference at *step*, and the module responsible."""
    worst, culprit = 0.0, None

    lhs, rhs = left.activations.get(step, {}), right.activations.get(step, {})
    for key in lhs.keys() & rhs.keys():
        for name in fields:
            if name in lhs[key] and name in rhs[key]:
                diff = rel_diff(lhs[key][name], rhs[key][name])
                if diff > worst:
                    worst, culprit = diff, f"{key[0]}[{key[1]}].{name}"

    lg, rg = left.grad_summary.get(step), right.grad_summary.get(step)
    if lg and rg:
        for name in ("total_norm", "absmax"):
            if name in lg and name in rg:
                diff = rel_diff(lg[name], rg[name])
                if diff > worst:
                    worst, culprit = diff, f"grad_summary.{name}"

    return worst, culprit


def classify(left: Trajectory, right: Trajectory, step: int) -> tuple[str, str]:
    """Why the two arms differ at *step*: desync or arithmetic.

    Checked before any numeric comparison, because a desync means the runs did
    different work and the numbers are not comparable at all.
    """
    lb, rb = left.batch_ids.get(step), right.batch_ids.get(step)
    if lb is not None and rb is not None and lb != rb:
        return "data-desync", f"different samples: {lb} vs {rb}"

    lf, rf = left.firings.get(step), right.firings.get(step)
    if lf and rf and lf != rf:
        differing = [k for k in set(lf) | set(rf) if lf.get(k) != rf.get(k)]
        example = sorted(differing)[0]
        return (
            "execution-desync",
            f"{len(differing)} modules fired a different number of times "
            f"(recycle count?), e.g. {example}: {lf.get(example)} vs "
            f"{rf.get(example)}",
        )

    # Fallback for trajectories without firings rows: compare the recorded
    # activation keys directly. Weaker, because capping truncates them.
    lhs, rhs = left.activations.get(step, {}), right.activations.get(step, {})
    if not (lf and rf) and lhs and rhs and lhs.keys() != rhs.keys():
        only_left = sorted(str(k) for k in lhs.keys() - rhs.keys())[:3]
        only_right = sorted(str(k) for k in rhs.keys() - lhs.keys())[:3]
        return (
            "execution-desync",
            f"different module firings (recycle count?): {left.label} only "
            f"{only_left}, {right.label} only {only_right}",
        )

    ll, rl = left.losses.get(step, []), right.losses.get(step, [])
    if len(ll) != len(rl):
        return "execution-desync", f"different backward count: {len(ll)} vs {len(rl)}"

    return "numeric", "same samples and same module firings; values differ"


def curve(left: Trajectory, right: Trajectory) -> dict[int, tuple[float, str | None]]:
    shared = sorted(left.steps & right.steps)
    return {s: step_divergence(left, right, s) for s in shared}


@dataclass
class Report:
    steps_compared: int = 0
    first_divergent_step: int | None = None
    first_divergent_module: str | None = None
    kind: str | None = None
    detail: str | None = None
    max_divergence: float = 0.0
    max_divergence_step: int | None = None
    max_divergence_module: str | None = None
    final_divergence: float = 0.0
    growing: bool = False
    nonfinite_step: int | None = None
    nonfinite_arm: str | None = None
    actionable: bool = False


def summarise(
    left: Trajectory, right: Trajectory, cross: dict[int, tuple[float, str | None]]
) -> Report:
    report = Report(steps_compared=len(cross))

    for traj in (left, right):
        if traj.nonfinite_steps:
            report.nonfinite_step = min(traj.nonfinite_steps)
            report.nonfinite_arm = traj.label
            report.kind = "nonfinite"
            report.actionable = True
            return report

    ordered = sorted(cross)
    for step in ordered:
        value, module = cross[step]
        if value > 0 and report.first_divergent_step is None:
            report.first_divergent_step = step
            report.first_divergent_module = module
            report.kind, report.detail = classify(left, right, step)
            report.actionable = report.kind.endswith("desync")
        if value > report.max_divergence:
            report.max_divergence = value
            report.max_divergence_step = step
            report.max_divergence_module = module

    if ordered:
        report.final_divergence = cross[ordered[-1]][0]
        half = len(ordered) // 2 or 1
        early = sum(cross[s][0] for s in ordered[:half]) / half
        late = sum(cross[s][0] for s in ordered[-half:]) / half
        report.growing = late > early
    return report


def growth_table(
    cross: dict[int, tuple[float, str | None]],
    reference: dict[int, tuple[float, str | None]] | None = None,
    *,
    buckets: int = 12,
) -> list[dict]:
    """Bucketed divergence curve. For cumulative instability the shape is the
    answer, so this is the primary output rather than a decoration."""
    steps = sorted(cross)
    if not steps:
        return []
    size = max(1, len(steps) // buckets)
    rows = []
    for start in range(0, len(steps), size):
        window = steps[start : start + size]
        row = {
            "from_step": window[0],
            "to_step": window[-1],
            "cross": sum(cross[s][0] for s in window) / len(window),
        }
        if reference:
            values = [reference[s][0] for s in window if s in reference]
            row["reference"] = sum(values) / len(values) if values else None
        rows.append(row)
    return rows


def _fmt_growth(rows: list[dict]) -> str:
    if not rows:
        return "  (no steps to chart)"
    has_ref = any(r.get("reference") is not None for r in rows)
    header = "  steps             divergence" + ("   reference" if has_ref else "")
    lines = ["", header, "  " + "-" * (46 if has_ref else 34)]
    peak = max(r["cross"] for r in rows) or 1.0
    for row in rows:
        bar = "#" * round(24 * row["cross"] / peak) if row["cross"] else ""
        ref = ""
        if has_ref:
            value = row.get("reference")
            ref = f"  {value:.2e}" if value is not None else "         -"
        span = f"{row['from_step']:>6}-{row['to_step']:<8}"
        lines.append(f"  {span} {row['cross']:.2e}{ref}  {bar}")
    return "\n".join(lines)


def _fmt(report: Report, left: Trajectory, right: Trajectory) -> str:
    if report.kind == "nonfinite":
        return (
            f"NON-FINITE  {report.nonfinite_arm} recorded a non-finite value at "
            f"step {report.nonfinite_step}"
        )
    if report.first_divergent_step is None:
        return f"IDENTICAL  {report.steps_compared} steps compared, no difference"

    lines = [
        f"DIFFERS  over {report.steps_compared} compared steps",
        f"  first difference : step {report.first_divergent_step}"
        f"  ({report.first_divergent_module})",
        f"  cause            : {report.kind} -- {report.detail}",
        f"  largest          : {report.max_divergence:.3e} at step "
        f"{report.max_divergence_step} ({report.max_divergence_module})",
        f"  final            : {report.final_divergence:.3e}",
        f"  trend            : {'growing' if report.growing else 'flat or shrinking'}",
    ]
    if report.kind == "numeric":
        lines.append(
            "  note             : same-seed runs share init, recycle schedule and "
            "samples, so this is arithmetic"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cross",
        nargs=2,
        required=True,
        metavar=("ARM_A", "ARM_B"),
        help="the two arms to compare, e.g. the AMD and NVIDIA runs",
    )
    parser.add_argument(
        "--reference",
        nargs=2,
        metavar=("ARM_A", "ARM_B"),
        help="a second pair plotted alongside for scale. The useful one is the "
        "same vendor with a different BLAS backend: it shows how far a "
        "legitimate implementation swap moves the trajectory.",
    )
    parser.add_argument(
        "--fail-above",
        type=float,
        metavar="X",
        help="exit 1 if the divergence exceeds X. Off by default: the magnitude "
        "that matters is a judgement, not a constant.",
    )
    parser.add_argument("--after", type=int, default=0, help="ignore steps below this")
    parser.add_argument("--json", type=Path, help="write the report here")
    parser.add_argument(
        "--watch",
        type=float,
        metavar="SECONDS",
        help="re-read and re-report on an interval against a live run",
    )
    args = parser.parse_args(argv)

    if args.after < 0:
        parser.error("--after must be >= 0")
    if args.watch is not None and args.watch <= 0:
        parser.error("--watch must be a positive number of seconds")

    def once(quiet_errors: bool) -> int | None:
        try:
            left = load(Path(args.cross[0]), "cross-A")
            right = load(Path(args.cross[1]), "cross-B")
            cross = {s: v for s, v in curve(left, right).items() if s >= args.after}
            if not cross:
                raise InputError(
                    f"no steps in common at or after {args.after}: "
                    f"{sorted(left.steps)[:3]}... vs {sorted(right.steps)[:3]}..."
                )
            reference = None
            if args.reference:
                ref_left = load(Path(args.reference[0]), "ref-A")
                ref_right = load(Path(args.reference[1]), "ref-B")
                reference = curve(ref_left, ref_right)
        except InputError as exc:
            # A live run may not have written enough yet.
            if quiet_errors:
                print(f"waiting: {exc}", file=sys.stderr)
                return None
            print(f"error: {exc}", file=sys.stderr)
            return 2

        report = summarise(left, right, cross)
        if args.fail_above is not None and report.max_divergence > args.fail_above:
            report.actionable = True

        print(_fmt(report, left, right))
        print(_fmt_growth(growth_table(cross, reference)))

        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(vars(report), indent=2, sort_keys=True))

        return 1 if report.actionable else 0

    if args.watch is None:
        return once(quiet_errors=False)

    try:
        while True:
            print(f"--- {time.strftime('%H:%M:%S')} ---")
            code = once(quiet_errors=True)
            if code == 1:
                return 1
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\nwatch interrupted", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
