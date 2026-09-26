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

"""Tests for ``scripts/parity/make_runner_yaml.py``.

The fingerprint is the point: it must be blind to the things that are *meant* to
differ between arms (seed, dataset paths) and sensitive to everything else.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "scripts/parity/make_runner_yaml.py"
)
_spec = importlib.util.spec_from_file_location("make_runner_yaml", _MODULE_PATH)
mry = importlib.util.module_from_spec(_spec)
sys.modules["make_runner_yaml"] = mry
_spec.loader.exec_module(mry)


BASE = {
    "experiment_settings": {
        "mode": "train",
        "seed": 1,
        "restart_checkpoint_path": "last",
    },
    "data_module_args": {"batch_size": 1, "num_workers": 4, "epoch_len": 32},
    "pl_trainer_args": {"devices": 1, "precision": "32-true", "max_epochs": 2},
    "model_update": {
        "presets": ["train"],
        "custom": {
            "settings": {
                "memory": {
                    "train": {"use_deepspeed_evo_attention": True},
                    "eval": {"use_triton_triangle_kernels": True},
                }
            }
        },
    },
    "dataset_paths": {
        "weighted-pdb": {"dataset_cache_file": "/data/site-a/cache.json"}
    },
}


def build(**kwargs):
    params = {"seed": 42, "devices": 8, "num_workers": 8}
    params.update(kwargs)
    return mry.build(BASE, **params)


# ---------------------------------------------------------------------------
# The overlay
# ---------------------------------------------------------------------------


def test_overlay_turns_every_kernel_flag_off_on_both_phases():
    """Triton defaults True for eval and False for train, so a run left at
    defaults uses different kernels in validation than in training."""
    memory = build()["model_update"]["custom"]["settings"]["memory"]
    for phase in ("train", "eval"):
        on = [k for k, v in memory[phase].items() if k.startswith("use_") and v]
        assert not on, f"{phase} still has {on}"


def test_overlay_enables_the_probe_and_determinism():
    config = build()
    settings = config["model_update"]["custom"]["settings"]
    assert settings["parity_probe"]["enabled"] is True
    assert config["pl_trainer_args"]["deterministic"] is True


def test_overlay_clears_restart_checkpoint_path():
    """``last`` is a full Lightning resume: optimizer moments, scheduler
    position and loop counters, which makes an arm incomparable."""
    assert BASE["experiment_settings"]["restart_checkpoint_path"] == "last"
    assert build()["experiment_settings"]["restart_checkpoint_path"] is None


def test_overlay_does_not_mutate_the_source():
    before = BASE["model_update"]["custom"]["settings"]["memory"]["eval"].copy()
    build()
    assert BASE["model_update"]["custom"]["settings"]["memory"]["eval"] == before


def test_unrelated_keys_are_preserved():
    config = build()
    assert config["dataset_paths"] == BASE["dataset_paths"]
    assert config["data_module_args"]["epoch_len"] == 32
    assert config["model_update"]["presets"] == ["train"]


def test_seed_devices_and_workers_are_applied():
    config = build(seed=7, devices=4, num_workers=2)
    assert config["experiment_settings"]["seed"] == 7
    assert config["pl_trainer_args"]["devices"] == 4
    assert config["data_module_args"]["num_workers"] == 2


# ---------------------------------------------------------------------------
# The fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_ignores_the_seed():
    """Null arms differ only by seed and must still fingerprint identically."""
    assert mry.fingerprint(build(seed=42)) == mry.fingerprint(build(seed=7))


def test_fingerprint_ignores_dataset_paths():
    """The two sites hold their own copies under different roots."""
    other = {
        **BASE,
        "dataset_paths": {"weighted-pdb": {"dataset_cache_file": "/b.json"}},
    }
    assert mry.fingerprint(build()) == mry.fingerprint(
        mry.build(other, seed=42, devices=8, num_workers=8)
    )


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"devices": 4}, "global batch = devices x num_nodes"),
        ({"num_workers": 2}, "worker seeds derive from worker_id"),
    ],
)
def test_fingerprint_catches_arm_mismatches(kwargs, why):
    assert mry.fingerprint(build()) != mry.fingerprint(build(**kwargs)), why


def test_fingerprint_catches_a_kernel_flag_difference():
    sneaky = {**BASE}
    base_fp = mry.fingerprint(build())
    config = mry.build(sneaky, seed=42, devices=8, num_workers=8)
    config["model_update"]["custom"]["settings"]["memory"]["train"]["use_lma"] = True
    assert mry.fingerprint(config) != base_fp


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_a_loadable_yaml(tmp_path, capsys):
    source = tmp_path / "src.yml"
    source.write_text(yaml.safe_dump(BASE))
    out = tmp_path / "nested" / "out.yml"
    assert mry.main([str(source), "--out", str(out), "--seed", "42"]) == 0
    written = yaml.safe_load(out.read_text())
    assert written["experiment_settings"]["seed"] == 42
    assert "parity fingerprint:" in capsys.readouterr().out


def test_cli_rejects_a_missing_source(tmp_path):
    assert (
        mry.main(
            [
                str(tmp_path / "nope.yml"),
                "--out",
                str(tmp_path / "o.yml"),
                "--seed",
                "1",
            ]
        )
        == 2
    )


def test_cli_rejects_a_non_mapping_source(tmp_path):
    source = tmp_path / "list.yml"
    source.write_text("- a\n- b\n")
    assert mry.main([str(source), "--out", str(tmp_path / "o.yml"), "--seed", "1"]) == 2


# ---------------------------------------------------------------------------
# Review finding: the fingerprint was blind to whole config blocks
# ---------------------------------------------------------------------------


def test_fingerprint_catches_a_dataset_weight_difference():
    """dataset_configs carries crop size and dataset weights, and is not the
    same thing as dataset_paths."""
    base_fp = mry.fingerprint(build())
    other = mry.build(BASE, seed=42, devices=8, num_workers=8)
    other["dataset_configs"] = {"train": {"weighted-pdb": {"weight": 0.7}}}
    assert mry.fingerprint(other) != base_fp


def test_fingerprint_catches_a_crop_size_difference():
    base = mry.build(BASE, seed=42, devices=8, num_workers=8)
    base["dataset_configs"] = {
        "train": {"weighted-pdb": {"config": {"crop": {"token_budget": 384}}}}
    }
    other = mry.build(BASE, seed=42, devices=8, num_workers=8)
    other["dataset_configs"] = {
        "train": {"weighted-pdb": {"config": {"crop": {"token_budget": 256}}}}
    }
    assert mry.fingerprint(base) != mry.fingerprint(other)


def test_fingerprint_catches_a_resume_setting_difference():
    base = mry.build(BASE, seed=42, devices=8, num_workers=8)
    other = mry.build(BASE, seed=42, devices=8, num_workers=8)
    other["experiment_settings"]["preemption_safe_resume"] = True
    assert mry.fingerprint(base) != mry.fingerprint(other)


def test_fingerprint_still_ignores_output_dir():
    """Each arm writes somewhere different; that is not a mismatch."""
    base = mry.build(BASE, seed=42, devices=8, num_workers=8)
    other = mry.build(BASE, seed=7, devices=8, num_workers=8)
    base["experiment_settings"]["output_dir"] = "/runs/a"
    other["experiment_settings"]["output_dir"] = "/runs/b"
    assert mry.fingerprint(base) == mry.fingerprint(other)


def test_overlay_pins_gradient_accumulation():
    """Several micro-batches per step share one global_step and collide."""
    assert build()["pl_trainer_args"]["accumulate_grad_batches"] == 1
