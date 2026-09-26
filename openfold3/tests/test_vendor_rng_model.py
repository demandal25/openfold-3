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

"""Whole-model check that no device RNG is consumed in vendor-independent mode.

The unit tests in ``core/utils/test_vendor_rng.py`` cover the helpers. This one
covers *completeness*: a full training step must leave the device RNG state
untouched, so a stochastic site that was never converted fails here rather than
surfacing later as an unexplained AMD-vs-NVIDIA trajectory divergence.

Scope limit: ``random_of3_features`` omits some ground-truth keys, so permutation
alignment falls through to its "turning off losses" path — as it does for
``test_of3_model.py`` too. Trunk dropout, diffusion noise and augmentation are
still exercised; the loss-side backward is not.
"""

import pytest
import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils import vendor_rng
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
from openfold3.tests.utils.data_utils import random_of3_features

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU to have a device RNG to watch"
)


def _training_step(device: str) -> None:
    """One reduced-size training forward + backward."""
    project_entry = OF3ProjectEntry()
    config = project_entry.get_model_config_with_presets()
    config.settings.blocks_per_ckpt = 1
    config.settings.ckpt_intermediate_steps = True
    config.architecture.pairformer.no_blocks = 2
    config.architecture.diffusion_module.diffusion_transformer.no_blocks = 2
    config.architecture.loss_module.diffusion.chunk_size = 16

    model = OpenFold3AllAtom(config).to(device=device, dtype=torch.float32)
    loss_fn = OpenFold3Loss(config=config.architecture.loss_module)

    batch = random_of3_features(
        batch_size=2, n_token=18, n_msa=10, n_templ=3, is_eval=False
    )
    batch = OF3DeepSpeedPrecision(precision="32-true").convert_input(batch)
    batch = tensor_tree_map(lambda t: t.to(device=torch.device(device)), batch)

    batch, outputs = model(batch=batch)
    loss_fn(batch=batch, output=outputs).backward()


def test_training_step_consumes_no_device_rng(monkeypatch):
    """A converted stochastic site leaves the device RNG alone; a missed one does not."""
    monkeypatch.setattr(vendor_rng, "_enabled", True)
    torch.manual_seed(0)
    before = torch.cuda.get_rng_state()
    _training_step("cuda")
    assert torch.equal(before, torch.cuda.get_rng_state()), (
        "a stochastic site still draws from the device RNG; find it with "
        "`grep -rn 'torch.rand' openfold3/core openfold3/projects`"
    )


def test_training_step_does_consume_device_rng_when_disabled(monkeypatch):
    """Negative control: without the flag the device RNG is what gets consumed.

    Without this, the test above would pass just as well on a model that had no
    stochasticity left at all.
    """
    monkeypatch.setattr(vendor_rng, "_enabled", False)
    torch.manual_seed(0)
    before = torch.cuda.get_rng_state()
    _training_step("cuda")
    assert not torch.equal(before, torch.cuda.get_rng_state())
