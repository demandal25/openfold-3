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

"""Tests for :mod:`openfold3.core.utils.parity_probe`.

Driven through a real ``pl.Trainer`` in **manual optimization**, because that is
what OF3 uses (per-sample gradient clipping forces it) and it is the mode in
which ``on_before_optimizer_step`` firing is not obvious.
"""

import json

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from openfold3.core.utils.parity_probe import (
    DEFAULT_MODULE_PATTERNS,
    NonFiniteValue,
    ParityProbeCallback,
    _stats,
    check_patterns_match,
)


class _ManualModule(pl.LightningModule):
    """Minimal stand-in for OF3's manual-optimization training step."""

    def __init__(self, blow_up_at: int | None = None):
        super().__init__()
        self.automatic_optimization = False
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
        self.blow_up_at = blow_up_at

    def forward(self, x):
        for block in self.blocks:
            x = block(x).tanh()
        return x

    def training_step(self, batch, batch_idx):
        (inputs,) = batch
        opt = self.optimizers()
        opt.zero_grad()
        if self.blow_up_at is not None and self.global_step == self.blow_up_at:
            with torch.no_grad():
                self.blocks[1].weight.fill_(float("inf"))
        loss = self(inputs).pow(2).sum()
        self.manual_backward(loss)
        opt.step()
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-3)


def _run(tmp_path, *, steps=4, blow_up_at=None, **probe_kwargs):
    probe = ParityProbeCallback(
        output_dir=tmp_path, module_patterns=(r"blocks\.\d+$",), **probe_kwargs
    )
    data = DataLoader(TensorDataset(torch.randn(steps * 2, 4)), batch_size=2)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[probe],
    )
    trainer.fit(_ManualModule(blow_up_at=blow_up_at), data)
    return probe


def _rows(tmp_path):
    path = tmp_path / "trajectory_rank0.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _by_kind(rows, kind):
    return [r for r in rows if r["kind"] == kind]


# ---------------------------------------------------------------------------
# It records at all, under manual optimization
# ---------------------------------------------------------------------------


def test_stats_survives_an_empty_tensor():
    """A zero-length activation must not take the run down from inside a hook."""
    stats = _stats(torch.empty(0, 4))
    assert stats == {"shape": [0, 4], "nonfinite": 0}


def test_records_activations_losses_and_gradients(tmp_path):
    torch.manual_seed(0)
    _run(tmp_path)
    rows = _rows(tmp_path)

    assert _by_kind(rows, "activation"), "no activation rows"
    assert _by_kind(rows, "loss"), "no loss rows"
    assert _by_kind(rows, "grad_summary"), (
        "on_before_optimizer_step did not fire under manual optimization"
    )
    assert {"blocks.0", "blocks.1", "blocks.2"} == {
        r["module"] for r in _by_kind(rows, "activation")
    }


def test_grad_summary_is_emitted_once_per_optimizer_step(tmp_path):
    torch.manual_seed(0)
    _run(tmp_path, steps=4)
    summaries = _by_kind(_rows(tmp_path), "grad_summary")
    assert len(summaries) == 4
    assert [s["step"] for s in summaries] == [0, 1, 2, 3]
    assert all(s["total_norm"] > 0 for s in summaries)
    assert all(s["clipped"] is True for s in summaries)


def test_ordinals_are_dense_and_restart_each_step(tmp_path):
    torch.manual_seed(0)
    _run(tmp_path)
    per_step = {}
    for row in _by_kind(_rows(tmp_path), "activation"):
        per_step.setdefault((row["step"], row["module"]), []).append(row["ordinal"])
    assert per_step
    for key, ordinals in per_step.items():
        assert ordinals == list(range(len(ordinals))), key


# ---------------------------------------------------------------------------
# every_n_steps
# ---------------------------------------------------------------------------


def test_every_n_steps_thins_detail_but_not_the_summary(tmp_path):
    torch.manual_seed(0)
    _run(tmp_path, steps=4, every_n_steps=2)
    rows = _rows(tmp_path)
    assert {r["step"] for r in _by_kind(rows, "activation")} == {0, 2}
    # The cheap summary and batch identity must survive the thinning, or a spike
    # between probe steps is invisible.
    assert {r["step"] for r in _by_kind(rows, "grad_summary")} == {0, 1, 2, 3}


def test_rejects_non_positive_every_n_steps(tmp_path):
    with pytest.raises(ValueError, match="every_n_steps"):
        ParityProbeCallback(output_dir=tmp_path, every_n_steps=0)


# ---------------------------------------------------------------------------
# The tripwire
# ---------------------------------------------------------------------------


def test_aborts_on_non_finite_activation(tmp_path):
    torch.manual_seed(0)
    with pytest.raises(NonFiniteValue, match="non-finite"):
        _run(tmp_path, steps=4, blow_up_at=2)
    rows = _rows(tmp_path)
    bad = [r for r in rows if r.get("nonfinite", 0)]
    assert bad, "tripwire fired but nothing was recorded"
    assert bad[0]["step"] == 2, f"blamed the wrong step: {bad[0]}"
    assert bad[0]["module"] == "blocks.1", f"blamed the wrong module: {bad[0]}"


def test_can_be_told_not_to_abort(tmp_path):
    """The run continues and the non-finite value is still on the record."""
    torch.manual_seed(0)
    _run(tmp_path, steps=4, blow_up_at=2, abort_on_nonfinite=False)
    rows = _rows(tmp_path)
    assert any(r.get("nonfinite", 0) for r in rows)
    assert max(r["step"] for r in _by_kind(rows, "grad_summary")) == 3


# ---------------------------------------------------------------------------
# The validation leak
# ---------------------------------------------------------------------------


class _WithValidation(_ManualModule):
    def validation_step(self, batch, batch_idx):
        (inputs,) = batch
        return self(inputs).pow(2).sum()


def test_validation_forwards_are_not_recorded(tmp_path):
    """Validation must not inflate a training step's firing counts.

    Found on the real model: the probe recorded the validation loop's forwards
    and attributed them to the preceding training step, ~20x the real count.
    """
    probe = ParityProbeCallback(output_dir=tmp_path, module_patterns=(r"blocks\.\d+$",))
    train = DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)
    val = DataLoader(TensorDataset(torch.randn(8, 4)), batch_size=2)
    torch.manual_seed(0)
    pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[probe],
    ).fit(_WithValidation(), train, val)

    rows = _rows(tmp_path)
    firings = _by_kind(rows, "firings")
    assert firings, "no firings rows"
    # One forward per module per training batch, and nothing from the 4
    # validation batches that follow.
    for row in firings:
        assert set(row["counts"].values()) == {1}, row["counts"]

    recorded = {}
    for row in _by_kind(rows, "activation"):
        recorded.setdefault(row["step"], []).append(row["module"])
    for step, modules in recorded.items():
        assert len(modules) == len(set(modules)), (
            f"step {step} recorded a module twice; validation leaked in"
        )


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def test_hooks_are_removed_on_teardown(tmp_path):
    torch.manual_seed(0)
    probe = _run(tmp_path)
    assert probe._handles == []
    assert probe._file is None


def test_unmatched_patterns_warn_but_still_record_gradients(tmp_path, caplog):
    torch.manual_seed(0)
    probe = ParityProbeCallback(output_dir=tmp_path, module_patterns=(r"nope$",))
    data = DataLoader(TensorDataset(torch.randn(4, 4)), batch_size=2)
    pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[probe],
    ).fit(_ManualModule(), data)
    rows = _rows(tmp_path)
    assert not _by_kind(rows, "activation")
    assert _by_kind(rows, "grad_summary")
    assert any("no module matched" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    "tensor_factory",
    [
        pytest.param(lambda: torch.randn(1000) * 3 + 2, id="plain"),
        pytest.param(lambda: torch.tensor([4.25]), id="single-element"),
        pytest.param(
            lambda: torch.cat([torch.randn(100), torch.tensor([float("nan")])]),
            id="with-nan",
        ),
        pytest.param(lambda: (torch.randn(1000) * 3).bfloat16(), id="bf16"),
    ],
)
def test_stats_matches_a_masked_reference(tensor_factory):
    """The packed single-sync reduction must agree with plain masked torch ops."""
    torch.manual_seed(0)
    tensor = tensor_factory()
    stats = _stats(tensor)

    flat = tensor.detach().float().reshape(-1)
    good = flat[torch.isfinite(flat)]
    assert stats["mean"] == pytest.approx(float(good.mean()), abs=2e-4)
    assert stats["std"] == pytest.approx(
        float(good.std()) if good.numel() > 1 else 0.0, abs=2e-4
    )
    assert stats["absmax"] == pytest.approx(float(good.abs().max()), abs=2e-4)
    assert stats["nonfinite"] == int((~torch.isfinite(flat)).sum())


def test_stats_returns_none_for_integer_tensors():
    assert _stats(torch.arange(10)) is None


def test_stats_reports_an_all_non_finite_tensor_without_moments():
    stats = _stats(torch.full((10,), float("nan")))
    assert stats["nonfinite"] == 10
    assert "mean" not in stats


# ---------------------------------------------------------------------------
# The patterns must match the model they are written for
# ---------------------------------------------------------------------------


def test_default_patterns_match_the_real_model():
    """Guards against silent no-match, which costs one warning and all the data.

    ``msa_module_stack`` did not exist; the module is ``msa_module``.
    """
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

    config = OF3ProjectEntry().get_model_config_with_presets()
    config.architecture.pairformer.no_blocks = 3
    config.architecture.diffusion_module.diffusion_transformer.no_blocks = 3
    model = OpenFold3AllAtom(config)

    counts = check_patterns_match(model, DEFAULT_MODULE_PATTERNS)
    unmatched = [p for p, n in counts.items() if n == 0]
    assert not unmatched, f"patterns match nothing: {unmatched}"


def test_trunk_and_confidence_head_stacks_are_not_conflated():
    """``aux_heads.pairformer_embedding`` has its own separately-sized stack."""
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
    from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom

    config = OF3ProjectEntry().get_model_config_with_presets()
    config.architecture.pairformer.no_blocks = 3
    model = OpenFold3AllAtom(config)

    trunk = check_patterns_match(model, [r"^model\.pairformer_stack\.blocks\.\d+$"])
    assert next(iter(trunk.values())) == 3, (
        "trunk pattern is picking up another stack's blocks"
    )


# ---------------------------------------------------------------------------
# Review findings
# ---------------------------------------------------------------------------


def test_refuses_to_append_to_an_existing_trajectory(tmp_path):
    """A restart into the same dir would splice two runs into one file, and the
    loader keeps only the last row per key."""
    (tmp_path / "trajectory_rank0.jsonl").write_text('{"kind":"loss","step":0}\n')
    probe = ParityProbeCallback(output_dir=tmp_path)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    with pytest.raises(FileExistsError, match="already holds a trajectory"):
        probe.setup(trainer, _ManualModule(), stage="fit")


def test_an_empty_existing_file_is_not_an_obstacle(tmp_path):
    (tmp_path / "trajectory_rank0.jsonl").write_text("")
    probe = ParityProbeCallback(output_dir=tmp_path)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    probe.setup(trainer, _ManualModule(), stage="fit")
    probe.teardown(trainer, None, stage="fit")


def test_warns_when_no_parameter_has_a_gradient(tmp_path, caplog):
    """Under DeepSpeed/ZeRO param.grad is None; silence would leave the whole
    gradient half of the trajectory missing with no signal."""
    probe = ParityProbeCallback(output_dir=tmp_path)
    module = _ManualModule()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    probe.setup(trainer, module, stage="fit")
    try:
        probe.on_before_optimizer_step(trainer, module, None)
        probe.on_before_optimizer_step(trainer, module, None)
    finally:
        probe.teardown(trainer, module, stage="fit")
    warnings = [r for r in caplog.records if "no parameter has .grad" in r.message]
    assert len(warnings) == 1, "should warn once, not every step"


def test_rows_carry_batch_idx_so_accumulation_cannot_collide(tmp_path):
    """Micro-batches share a global_step; without batch_idx their rows overwrite."""
    torch.manual_seed(0)
    _run(tmp_path, steps=2)
    rows = _rows(tmp_path)
    assert all("batch_idx" in r for r in rows), "a row is missing batch_idx"


def test_a_non_finite_past_the_firing_cap_still_aborts(tmp_path):
    """A capped firing is still checked: it is never written, so only the
    device-side flag can see a non-finite value there."""
    probe = ParityProbeCallback(output_dir=tmp_path, max_firings_per_module=1)
    probe._probing = True
    probe._in_train_batch = True
    hook = probe._make_hook("blocks.0")

    hook(None, None, torch.ones(4))  # ordinal 0, recorded, finite
    assert probe._nonfinite_flag is None
    hook(None, None, torch.full((4,), float("inf")))  # ordinal 1, capped
    assert probe._nonfinite_flag is not None
    assert bool(probe._nonfinite_flag)

    module = torch.nn.Linear(2, 2)
    with pytest.raises(NonFiniteValue):
        probe._abort_if_nonfinite(module)


def test_every_trajectory_row_is_strict_json(tmp_path):
    """Python's json.loads accepts bare Infinity/NaN; nothing else does."""
    try:
        _run(tmp_path, steps=3, blow_up_at=1, abort_on_nonfinite=False)
    except NonFiniteValue:  # pragma: no cover - abort is off
        pass

    def _reject(token):
        raise AssertionError(f"non-JSON constant in trajectory: {token}")

    decoder = json.JSONDecoder(parse_constant=_reject)
    path = tmp_path / "trajectory_rank0.jsonl"
    rows = [decoder.decode(line) for line in path.read_text().splitlines()]
    assert rows
    blown = [r for r in rows if r.get("nonfinite_fields")]
    assert blown, "expected at least one row to have carried a non-finite float"
    for row in blown:
        for field in row["nonfinite_fields"]:
            assert row[field] is None
