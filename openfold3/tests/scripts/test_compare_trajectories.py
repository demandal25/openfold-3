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

"""Tests for ``scripts/parity/compare_trajectories.py``.

Built on synthetic trajectories so the answer is known. The pair that matters is
:func:`test_clean_pair_is_parity` and :func:`test_injected_step_is_found` — a
detector that has only ever been shown to stay quiet is not a detector.
"""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "scripts/parity/compare_trajectories.py"
)
_spec = importlib.util.spec_from_file_location("compare_trajectories", _MODULE_PATH)
ct = importlib.util.module_from_spec(_spec)
# Registered before exec: @dataclass resolves annotations via sys.modules, and
# on 3.14 an unregistered module makes it fail with an opaque AttributeError.
sys.modules["compare_trajectories"] = ct
_spec.loader.exec_module(ct)


MODULES = ["model.pairformer_stack.blocks.0", "model.pairformer_stack.blocks.1"]


def write_trajectory(
    path: Path,
    *,
    steps: int = 200,
    seed: int = 0,
    jitter: float = 1e-3,
    inject_at: int | None = None,
    inject_scale: float = 10.0,
    inject_module: str = MODULES[1],
    ids=("7ohe", "7kud"),
    recycles: int = 2,
    nonfinite_at: int | None = None,
) -> Path:
    """Write a plausible trajectory. ``jitter`` stands in for bf16 run-to-run noise."""
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:

        def emit(row):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

        for step in range(steps):
            emit(
                {
                    "kind": "batch",
                    "step": step,
                    "batch_idx": step,
                    "ids": list(ids),
                }
            )
            for module in MODULES:
                for ordinal in range(recycles):
                    scale = 1.0
                    if (
                        inject_at is not None
                        and step >= inject_at
                        and module == inject_module
                    ):
                        scale = inject_scale
                    base = 1.0 + 0.001 * step
                    row = {
                        "kind": "activation",
                        "step": step,
                        "module": module,
                        "ordinal": ordinal,
                        "shape": [4, 4],
                        "mean": base * scale * (1 + rng.uniform(-jitter, jitter)),
                        "std": 0.5 * (1 + rng.uniform(-jitter, jitter)),
                        "absmax": 3.0 * scale * (1 + rng.uniform(-jitter, jitter)),
                        "nonfinite": 0,
                    }
                    if nonfinite_at is not None and step == nonfinite_at:
                        row["nonfinite"] = 7
                    emit(row)
            emit({"kind": "loss", "step": step, "ordinal": 0, "value": 1.0})
            emit(
                {
                    "kind": "grad_summary",
                    "step": step,
                    "clipped": True,
                    "total_norm": 2.0 * (1 + rng.uniform(-jitter, jitter)),
                    "absmax": 0.4,
                    "nonfinite": 0,
                    "n_params": 12,
                }
            )
    return path


@pytest.fixture
def arms(tmp_path):
    """Four arms: two same-vendor (the null) and two cross-vendor."""
    return {
        "null_a": write_trajectory(tmp_path / "null_a.jsonl", seed=1),
        "null_b": write_trajectory(tmp_path / "null_b.jsonl", seed=2),
        "cross_a": write_trajectory(tmp_path / "cross_a.jsonl", seed=3),
        "cross_b": write_trajectory(tmp_path / "cross_b.jsonl", seed=4),
    }


def _run(cross, null=None, extra=()):
    argv = ["--cross", str(cross[0]), str(cross[1])]
    if null:
        argv += ["--null", str(null[0]), str(null[1])]
    return ct.main(argv + list(extra))


# ---------------------------------------------------------------------------
# The A/B that makes this a detector
# ---------------------------------------------------------------------------


def test_clean_pair_is_parity(arms):
    assert (
        _run((arms["cross_a"], arms["cross_b"]), (arms["null_a"], arms["null_b"])) == 0
    )


def test_injected_step_is_found(tmp_path, arms):
    """Same noise as the clean pair, plus a 10x shift from step 120."""
    bad = write_trajectory(
        tmp_path / "bad.jsonl", seed=4, inject_at=120, inject_scale=10.0
    )
    out = tmp_path / "verdict.json"
    code = _run(
        (arms["cross_a"], bad), (arms["null_a"], arms["null_b"]), ["--json", str(out)]
    )
    assert code == 1
    verdict = json.loads(out.read_text())
    assert verdict["first_step"] == 120
    assert verdict["kind"] == "numeric"
    assert "blocks.1" in verdict["module"]


def test_a_shift_below_the_threshold_is_not_reported(tmp_path, arms):
    """2x the null noise must not fire at the default 3x threshold."""
    subtle = write_trajectory(
        tmp_path / "subtle.jsonl", seed=4, inject_at=120, inject_scale=1.0000001
    )
    assert _run((arms["cross_a"], subtle), (arms["null_a"], arms["null_b"])) == 0


# ---------------------------------------------------------------------------
# Classification: a module name alone is not a diagnosis
# ---------------------------------------------------------------------------


def test_different_samples_are_reported_as_a_data_desync(tmp_path, arms):
    other = write_trajectory(
        tmp_path / "other.jsonl", seed=4, inject_at=120, ids=("1abc", "2def")
    )
    out = tmp_path / "v.json"
    code = _run(
        (arms["cross_a"], other), (arms["null_a"], arms["null_b"]), ["--json", str(out)]
    )
    assert code == 1
    assert json.loads(out.read_text())["kind"] == "data-desync"


def test_different_recycle_counts_are_reported_as_an_execution_desync(tmp_path, arms):
    fewer = write_trajectory(
        tmp_path / "fewer.jsonl", seed=4, inject_at=120, recycles=1
    )
    out = tmp_path / "v.json"
    code = _run(
        (arms["cross_a"], fewer), (arms["null_a"], arms["null_b"]), ["--json", str(out)]
    )
    assert code == 1
    assert json.loads(out.read_text())["kind"] == "execution-desync"


def test_non_finite_short_circuits_everything(tmp_path, arms):
    blown = write_trajectory(tmp_path / "blown.jsonl", seed=4, nonfinite_at=57)
    out = tmp_path / "v.json"
    code = _run(
        (arms["cross_a"], blown), (arms["null_a"], arms["null_b"]), ["--json", str(out)]
    )
    assert code == 1
    verdict = json.loads(out.read_text())
    assert verdict["kind"] == "nonfinite"
    assert verdict["first_step"] == 57


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def test_after_suppresses_early_steps(tmp_path, arms):
    early = write_trajectory(tmp_path / "early.jsonl", seed=4, inject_at=5)
    null = (arms["null_a"], arms["null_b"])
    assert _run((arms["cross_a"], early), null, ["--after", "100"]) == 1


def test_a_run_that_ends_mid_divergence_is_inconclusive_not_parity(tmp_path, arms):
    """5 steps left and --consecutive 10: the rule cannot be satisfied.

    Reporting PARITY here would be the worst failure this tool could have -- a
    large, still-rising divergence read as a pass.
    """
    early = write_trajectory(tmp_path / "early.jsonl", seed=4, inject_at=5)
    out = tmp_path / "v.json"
    code = _run(
        (arms["cross_a"], early),
        (arms["null_a"], arms["null_b"]),
        ["--after", "195", "--json", str(out)],
    )
    assert code == 2
    verdict = json.loads(out.read_text())
    assert verdict["inconclusive"] is True
    assert verdict["diverged"] is False


def test_a_long_enough_tail_still_fires(tmp_path, arms):
    """Control for the test above: the same data with room for the rule."""
    early = write_trajectory(tmp_path / "early2.jsonl", seed=4, inject_at=5)
    assert (
        _run(
            (arms["cross_a"], early),
            (arms["null_a"], arms["null_b"]),
            ["--after", "150"],
        )
        == 1
    )


def test_consecutive_requires_a_sustained_run(tmp_path, arms):
    """One spiking step is noise; the rule needs a run."""
    spike = write_trajectory(tmp_path / "spike.jsonl", seed=4)
    rows = [json.loads(line) for line in spike.read_text().splitlines()]
    for row in rows:
        if row["kind"] == "activation" and row["step"] == 150:
            row["mean"] *= 50
    spike.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    assert _run((arms["cross_a"], spike), (arms["null_a"], arms["null_b"])) == 0


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_missing_file_exits_two(tmp_path, arms):
    assert _run((arms["cross_a"], tmp_path / "nope.jsonl")) == 2


def test_empty_file_exits_two(tmp_path, arms):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert _run((arms["cross_a"], empty)) == 2


def test_mostly_garbage_exits_two(tmp_path, arms):
    junk = tmp_path / "junk.jsonl"
    junk.write_text("\n".join("not json" for _ in range(50)))
    assert _run((arms["cross_a"], junk)) == 2


def test_a_truncated_final_line_is_tolerated(tmp_path, arms):
    """A run killed mid-write must still be readable."""
    truncated = write_trajectory(tmp_path / "trunc.jsonl", seed=4)
    text = truncated.read_text()
    truncated.write_text(text[: len(text) - 20])
    assert _run((arms["cross_a"], truncated), (arms["null_a"], arms["null_b"])) == 0


def test_disjoint_step_ranges_exit_two(tmp_path, arms):
    shifted = tmp_path / "shifted.jsonl"
    rows = [json.loads(line) for line in arms["cross_b"].read_text().splitlines()]
    for row in rows:
        row["step"] += 10_000
    shifted.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    assert _run((arms["cross_a"], shifted)) == 2


def test_run_directory_is_accepted(tmp_path, arms):
    run_dir = tmp_path / "run"
    write_trajectory(run_dir / "trajectory_rank0.jsonl", seed=3)
    assert _run((run_dir, arms["cross_b"]), (arms["null_a"], arms["null_b"])) == 0


def test_rejects_nonsense_thresholds(arms):
    with pytest.raises(SystemExit):
        _run((arms["cross_a"], arms["cross_b"]), extra=["--factor", "0"])


# ---------------------------------------------------------------------------
# --growth and --watch
# ---------------------------------------------------------------------------


def test_growth_table_shows_a_rising_ratio(tmp_path, arms):
    """For the cumulative question the shape is the answer, not the pass/fail."""
    rows = [json.loads(line) for line in arms["cross_b"].read_text().splitlines()]
    for row in rows:
        if row["kind"] == "activation":
            # divergence that grows with step, rather than a step change
            row["mean"] *= 1.0 + row["step"] / 200.0
    creeping = tmp_path / "creeping.jsonl"
    creeping.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

    cross = ct.curve(ct.load(arms["cross_a"], "a"), ct.load(creeping, "b"))
    null = ct.curve(ct.load(arms["null_a"], "na"), ct.load(arms["null_b"], "nb"))
    table = ct.growth_table(cross, null, buckets=5)

    assert len(table) >= 5
    assert table[-1]["ratio"] > table[0]["ratio"], "a growing divergence read as flat"
    assert "rising" in ct._fmt_growth(table)


def test_growth_table_on_a_clean_pair_is_flat(tmp_path, arms):
    cross = ct.curve(ct.load(arms["cross_a"], "a"), ct.load(arms["cross_b"], "b"))
    null = ct.curve(ct.load(arms["null_a"], "na"), ct.load(arms["null_b"], "nb"))
    table = ct.growth_table(cross, null, buckets=5)
    ratios = [row["ratio"] for row in table]
    assert max(ratios) / max(min(ratios), 1e-9) < 5, (
        f"clean pair looks like a trend: {ratios}"
    )


def test_growth_table_is_empty_without_shared_steps():
    assert ct.growth_table({}, {}) == []
    assert "no steps" in ct._fmt_growth([])


def test_growth_flag_prints_the_table(tmp_path, arms, capsys):
    _run(
        (arms["cross_a"], arms["cross_b"]),
        (arms["null_a"], arms["null_b"]),
        ["--growth"],
    )
    out = capsys.readouterr().out
    assert "ratio" in out and "trend:" in out


def test_watch_returns_on_divergence(tmp_path, arms):
    """Watch mode must stop as soon as there is something to act on."""
    bad = write_trajectory(tmp_path / "bad.jsonl", seed=4, inject_at=120)
    assert (
        _run(
            (arms["cross_a"], bad),
            (arms["null_a"], arms["null_b"]),
            ["--watch", "0.01"],
        )
        == 1
    )


def test_watch_rejects_a_non_positive_interval(arms):
    with pytest.raises(SystemExit):
        _run((arms["cross_a"], arms["cross_b"]), extra=["--watch", "0"])


def test_watch_waits_instead_of_erroring_on_a_missing_file(tmp_path, arms, capsys):
    """A live run may not have written anything yet; that is not an error."""
    import threading

    late = tmp_path / "late.jsonl"

    def write_later():
        import time as _t

        _t.sleep(0.4)
        write_trajectory(late, seed=4, inject_at=120)

    thread = threading.Thread(target=write_later)
    thread.start()
    try:
        code = _run(
            (arms["cross_a"], late),
            (arms["null_a"], arms["null_b"]),
            ["--watch", "0.2"],
        )
    finally:
        thread.join()
    assert code == 1
    assert "waiting:" in capsys.readouterr().err
