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
:func:`test_identical_arms_report_identical` and
:func:`test_injected_step_is_found` — a detector that has only ever been shown
to stay quiet is not a detector.
"""

import importlib.util
import json
import random
import sys
import time
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
                    "kind": "firings",
                    "step": step,
                    "counts": {m: recycles for m in MODULES},
                }
            )
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


def _run(cross, extra=()):
    argv = ["--cross", str(cross[0]), str(cross[1])]
    return ct.main(argv + list(extra))


# ---------------------------------------------------------------------------
# The A/B that makes this a detector
# ---------------------------------------------------------------------------


def test_identical_arms_report_identical(tmp_path):
    """Same seed means bit-identical, which is what we measured on MI355X."""
    a = write_trajectory(tmp_path / "a.jsonl", seed=1, jitter=0.0)
    b = write_trajectory(tmp_path / "b.jsonl", seed=1, jitter=0.0)
    out = tmp_path / "r.json"
    assert _run((a, b), extra=["--json", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["first_divergent_step"] is None
    assert report["max_divergence"] == 0.0


def test_injected_step_is_found(tmp_path, arms):
    bad = write_trajectory(tmp_path / "bad.jsonl", seed=3, inject_at=120)
    out = tmp_path / "r.json"
    assert _run((arms["cross_a"], bad), extra=["--json", str(out)]) in (0, 1)
    report = json.loads(out.read_text())
    assert report["max_divergence_step"] >= 120
    assert "blocks.1" in report["max_divergence_module"]
    assert report["growing"] is True


def test_fail_above_is_off_by_default_and_works_when_set(tmp_path, arms):
    bad = write_trajectory(tmp_path / "bad.jsonl", seed=3, inject_at=120)
    assert _run((arms["cross_a"], bad)) == 0
    assert _run((arms["cross_a"], bad), extra=["--fail-above", "0.1"]) == 1


# ---------------------------------------------------------------------------
# Classification: a module name alone is not a diagnosis
# ---------------------------------------------------------------------------


def test_different_samples_are_a_data_desync(tmp_path, arms):
    other = write_trajectory(
        tmp_path / "o.jsonl", seed=3, inject_at=10, ids=("1abc", "2def")
    )
    out = tmp_path / "r.json"
    assert _run((arms["cross_a"], other), extra=["--json", str(out)]) == 1
    assert json.loads(out.read_text())["kind"] == "data-desync"


def test_different_firing_counts_are_an_execution_desync(tmp_path, arms):
    fewer = write_trajectory(tmp_path / "f.jsonl", seed=3, inject_at=10, recycles=1)
    out = tmp_path / "r.json"
    assert _run((arms["cross_a"], fewer), extra=["--json", str(out)]) == 1
    report = json.loads(out.read_text())
    assert report["kind"] == "execution-desync"
    assert "firing" in report["detail"] or "fired" in report["detail"]


def test_non_finite_short_circuits_everything(tmp_path, arms):
    blown = write_trajectory(tmp_path / "b.jsonl", seed=3, nonfinite_at=57)
    out = tmp_path / "r.json"
    assert _run((arms["cross_a"], blown), extra=["--json", str(out)]) == 1
    report = json.loads(out.read_text())
    assert report["kind"] == "nonfinite"
    assert report["nonfinite_step"] == 57


# ---------------------------------------------------------------------------
# The growth curve, which is the primary output
# ---------------------------------------------------------------------------


def test_growth_curve_rises_for_a_growing_divergence(tmp_path, arms):
    rows = [json.loads(line) for line in arms["cross_b"].read_text().splitlines()]
    for row in rows:
        if row["kind"] == "activation":
            row["mean"] *= 1.0 + row["step"] / 200.0
    creeping = tmp_path / "c.jsonl"
    creeping.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    table = ct.growth_table(
        ct.curve(ct.load(arms["cross_a"], "a"), ct.load(creeping, "b"))
    )
    assert table[-1]["cross"] > table[0]["cross"]
    assert "#" in ct._fmt_growth(table)


def test_growth_curve_flat_for_a_clean_pair(tmp_path):
    a = write_trajectory(tmp_path / "a.jsonl", seed=1)
    b = write_trajectory(tmp_path / "b.jsonl", seed=2)
    table = ct.growth_table(ct.curve(ct.load(a, "a"), ct.load(b, "b")))
    values = [r["cross"] for r in table]
    assert max(values) / max(min(values), 1e-12) < 10, values


def test_reference_arm_is_charted_alongside(tmp_path, arms, capsys):
    _run(
        (arms["cross_a"], arms["cross_b"]),
        extra=["--reference", str(arms["null_a"]), str(arms["null_b"])],
    )
    assert "reference" in capsys.readouterr().out


def test_growth_table_is_empty_without_steps():
    assert ct.growth_table({}) == []
    assert "no steps" in ct._fmt_growth([])


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_missing_file_exits_two(tmp_path, arms):
    assert _run((arms["cross_a"], tmp_path / "nope.jsonl")) == 2


def test_empty_file_exits_two(tmp_path, arms):
    empty = tmp_path / "e.jsonl"
    empty.write_text("")
    assert _run((arms["cross_a"], empty)) == 2


def test_mostly_garbage_exits_two(tmp_path, arms):
    junk = tmp_path / "j.jsonl"
    junk.write_text("\n".join("not json" for _ in range(50)))
    assert _run((arms["cross_a"], junk)) == 2


def test_truncated_final_line_is_tolerated(tmp_path, arms):
    trunc = tmp_path / "t.jsonl"
    text = write_trajectory(trunc, seed=3).read_text()
    trunc.write_text(text[: len(text) - 20])
    assert _run((arms["cross_a"], trunc)) == 0


def test_disjoint_steps_exit_two(tmp_path, arms):
    shifted = tmp_path / "s.jsonl"
    rows = [json.loads(line) for line in arms["cross_b"].read_text().splitlines()]
    for row in rows:
        row["step"] += 10_000
    shifted.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    assert _run((arms["cross_a"], shifted)) == 2


def test_run_directory_is_accepted(tmp_path, arms):
    run_dir = tmp_path / "run"
    write_trajectory(run_dir / "trajectory_rank0.jsonl", seed=3)
    assert _run((run_dir, arms["cross_b"])) == 0


def test_after_filters_steps(tmp_path, arms):
    assert _run((arms["cross_a"], arms["cross_b"]), extra=["--after", "10000"]) == 2


def test_rejects_bad_arguments(arms):
    with pytest.raises(SystemExit):
        _run((arms["cross_a"], arms["cross_b"]), extra=["--watch", "0"])
    with pytest.raises(SystemExit):
        _run((arms["cross_a"], arms["cross_b"]), extra=["--after", "-1"])


# ---------------------------------------------------------------------------
# --watch
# ---------------------------------------------------------------------------


def test_watch_returns_on_an_actionable_finding(tmp_path, arms):
    blown = write_trajectory(tmp_path / "b.jsonl", seed=3, nonfinite_at=57)
    assert _run((arms["cross_a"], blown), extra=["--watch", "0.01"]) == 1


def test_watch_waits_instead_of_erroring_on_a_missing_file(tmp_path, arms, capsys):
    import threading

    late = tmp_path / "late.jsonl"

    def write_later():
        time.sleep(0.4)
        write_trajectory(late, seed=3, nonfinite_at=57)

    thread = threading.Thread(target=write_later)
    thread.start()
    try:
        code = _run((arms["cross_a"], late), extra=["--watch", "0.2"])
    finally:
        thread.join()
    assert code == 1
    assert "waiting:" in capsys.readouterr().err
