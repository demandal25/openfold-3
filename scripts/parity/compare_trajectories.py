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

"""Compare two training trajectories against an intra-vendor null.

Two bf16 runs diverge at epsilon on step one and amplify chaotically even when
both are healthy, so "AMD and NVIDIA differ" is not a finding on its own. What
makes it one is the cross-vendor divergence sitting *above* the divergence
between two same-vendor runs that differ only by seed.

    cross = |AMD seed A - NVIDIA seed A|      null = |AMD seed A - AMD seed B|

A cross curve inside the null envelope is parity. Above it, this reports the
first step and module that separated, and whether the cause was numeric or an
RNG/data desync -- a module name alone is not a diagnosis.

Exit codes: 0 parity, 1 divergence, 2 unusable input or an inconclusive run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

EPS = 1e-12
#: Floor on the null curve. Without it, a step where the two same-vendor runs
#: happen to agree exactly makes any cross-vendor difference look infinite.
NULL_FLOOR = 1e-7


class InputError(Exception):
    """The trajectories cannot be compared at all."""


@dataclass
class Trajectory:
    """One arm's records, indexed for comparison."""

    label: str
    #: step -> (module, ordinal) -> {field: value}
    activations: dict[int, dict[tuple[str, int], dict]] = field(default_factory=dict)
    #: step -> {field: value}
    grad_summary: dict[int, dict] = field(default_factory=dict)
    #: step -> list of sample ids
    batch_ids: dict[int, list[str]] = field(default_factory=dict)
    #: step -> list of loss values, by ordinal
    losses: dict[int, list[float]] = field(default_factory=dict)
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
            key = (row["module"], int(row["ordinal"]))
            traj.activations.setdefault(step, {})[key] = row
        elif kind == "grad_summary":
            traj.grad_summary[step] = row
        elif kind == "batch":
            traj.batch_ids[step] = [str(i) for i in row.get("ids", [])]
        elif kind == "loss":
            traj.losses.setdefault(step, []).append(float(row["value"]))

    if not traj.steps:
        raise InputError(f"{path}: no comparable records")
    return traj


def rel_diff(a: float, b: float) -> float:
    """Relative difference, symmetric and safe at zero."""
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    denom = max(abs(a), abs(b), EPS)
    return abs(a - b) / denom


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

    Checked before any numeric comparison, because a desync makes the numbers
    incomparable rather than merely different.
    """
    lb, rb = left.batch_ids.get(step), right.batch_ids.get(step)
    if lb is not None and rb is not None and lb != rb:
        return "data-desync", f"different samples: {lb} vs {rb}"

    lhs, rhs = left.activations.get(step, {}), right.activations.get(step, {})
    if lhs and rhs and lhs.keys() != rhs.keys():
        only_left = sorted(str(k) for k in lhs.keys() - rhs.keys())[:3]
        only_right = sorted(str(k) for k in rhs.keys() - lhs.keys())[:3]
        return (
            "execution-desync",
            f"different module firings (recycle count?): "
            f"{left.label} only {only_left}, {right.label} only {only_right}",
        )

    ll, rl = left.losses.get(step, []), right.losses.get(step, [])
    if len(ll) != len(rl):
        return (
            "execution-desync",
            f"different backward count: {len(ll)} vs {len(rl)}",
        )

    return "numeric", "same samples and same module firings; values differ"


def curve(left: Trajectory, right: Trajectory) -> dict[int, tuple[float, str | None]]:
    shared = sorted(left.steps & right.steps)
    return {s: step_divergence(left, right, s) for s in shared}


@dataclass
class Verdict:
    diverged: bool
    inconclusive: bool = False
    first_step: int | None = None
    module: str | None = None
    kind: str | None = None
    detail: str | None = None
    cross_value: float | None = None
    null_value: float | None = None
    reason: str = ""


def find_divergence(
    cross: dict[int, tuple[float, str | None]],
    null: dict[int, tuple[float, str | None]],
    *,
    factor: float,
    consecutive: int,
    after: int,
) -> tuple[list[int], dict[int, float], bool]:
    """Steps where cross exceeds ``factor`` x null, the ratio series, and whether
    an exceeding run was still open when the data ran out."""
    ratios: dict[int, float] = {}
    for step, (value, _) in cross.items():
        if step < after:
            continue
        null_value = max(null.get(step, (0.0, None))[0], NULL_FLOOR)
        ratios[step] = value / null_value

    exceeding = sorted(s for s, r in ratios.items() if r > factor)
    if not exceeding:
        return [], ratios, False

    # A run of `consecutive` steps that are consecutive *among compared steps*,
    # so thinning with every_n_steps does not defeat the rule.
    ordered = sorted(ratios)
    index = {s: i for i, s in enumerate(ordered)}
    run: list[int] = []
    longest: list[int] = []
    for step in exceeding:
        run = run + [step] if run and index[step] == index[run[-1]] + 1 else [step]
        if len(run) > len(longest):
            longest = list(run)
        if len(run) >= consecutive:
            return run, ratios, False

    # A run still rising when the data ended is not a pass. Reporting PARITY for
    # a large divergence that simply had too few steps left to satisfy the rule
    # is the worst failure this tool could have.
    truncated = bool(longest) and index[longest[-1]] == len(ordered) - 1
    return [], ratios, truncated


def compare(
    cross_pair: tuple[Trajectory, Trajectory],
    null_pair: tuple[Trajectory, Trajectory] | None,
    *,
    factor: float,
    consecutive: int,
    after: int,
) -> Verdict:
    left, right = cross_pair
    cross = curve(left, right)
    if not cross:
        raise InputError(
            f"no steps in common between {left.label} and {right.label}: "
            f"{sorted(left.steps)[:3]}... vs {sorted(right.steps)[:3]}..."
        )

    for traj in (left, right):
        if traj.nonfinite_steps:
            step = min(traj.nonfinite_steps)
            return Verdict(
                diverged=True,
                first_step=step,
                kind="nonfinite",
                detail=f"{traj.label} recorded a non-finite value",
                reason=f"non-finite value in {traj.label} at step {step}",
            )

    null = curve(*null_pair) if null_pair else {}
    if null_pair and not null:
        raise InputError("null arms share no steps; cannot build an envelope")

    run, ratios, truncated = find_divergence(
        cross, null, factor=factor, consecutive=consecutive, after=after
    )
    if not run and truncated:
        worst = max(ratios.items(), key=lambda kv: kv[1])
        return Verdict(
            diverged=False,
            inconclusive=True,
            first_step=worst[0],
            kind="inconclusive",
            reason=(
                f"cross was still above {factor}x null at the last compared step "
                f"(peak {worst[1]:.1f}x at step {worst[0]}) but the run ended before "
                f"{consecutive} consecutive steps -- extend the run or lower "
                f"--consecutive"
            ),
        )
    if not run:
        worst = max(ratios.items(), key=lambda kv: kv[1], default=(None, 0.0))
        return Verdict(
            diverged=False,
            reason=(
                f"cross stays within {factor}x the null envelope "
                f"(peak {worst[1]:.2f}x at step {worst[0]})"
            ),
        )

    step = run[0]
    kind, detail = classify(left, right, step)
    return Verdict(
        diverged=True,
        first_step=step,
        module=cross[step][1],
        kind=kind,
        detail=detail,
        cross_value=cross[step][0],
        null_value=null.get(step, (0.0, None))[0],
        reason=(
            f"cross exceeded {factor}x null for {len(run)} consecutive compared "
            f"steps from step {step}"
        ),
    )


def _fmt(verdict: Verdict, factor: float) -> str:
    if verdict.inconclusive:
        return f"INCONCLUSIVE  {verdict.reason}"
    if not verdict.diverged:
        return f"PARITY  {verdict.reason}"
    lines = [f"DIVERGED  {verdict.reason}", f"  first step : {verdict.first_step}"]
    if verdict.module:
        lines.append(f"  module     : {verdict.module}")
    lines.append(f"  kind       : {verdict.kind}")
    if verdict.detail:
        lines.append(f"  detail     : {verdict.detail}")
    if verdict.cross_value is not None:
        ratio = verdict.cross_value / max(verdict.null_value or 0.0, NULL_FLOOR)
        lines.append(
            f"  cross={verdict.cross_value:.3e}  null={verdict.null_value:.3e}  "
            f"ratio={ratio:.1f}x  (threshold {factor}x)"
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
        "--null",
        nargs=2,
        metavar=("ARM_A", "ARM_B"),
        help="two same-vendor, different-seed runs defining the noise floor. "
        "Without it every difference is reported against a fixed floor, which "
        "is a weaker claim -- say so if you report it.",
    )
    parser.add_argument("--factor", type=float, default=3.0)
    parser.add_argument("--consecutive", type=int, default=10)
    parser.add_argument(
        "--after", type=int, default=100, help="ignore steps below this"
    )
    parser.add_argument("--json", type=Path, help="write the verdict here")
    args = parser.parse_args(argv)

    if args.factor <= 0 or args.consecutive < 1 or args.after < 0:
        parser.error("--factor must be > 0, --consecutive >= 1, --after >= 0")

    try:
        cross_pair = (
            load(Path(args.cross[0]), "cross-A"),
            load(Path(args.cross[1]), "cross-B"),
        )
        null_pair = (
            (load(Path(args.null[0]), "null-A"), load(Path(args.null[1]), "null-B"))
            if args.null
            else None
        )
        verdict = compare(
            cross_pair,
            null_pair,
            factor=args.factor,
            consecutive=args.consecutive,
            after=args.after,
        )
    except InputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if null_pair is None:
        print("warning: no --null arms; the envelope is a fixed floor", file=sys.stderr)
    print(_fmt(verdict, args.factor))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(vars(verdict), indent=2, sort_keys=True))

    if verdict.inconclusive:
        return 2
    return 1 if verdict.diverged else 0


if __name__ == "__main__":
    raise SystemExit(main())
