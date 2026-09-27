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

"""Tests for :mod:`openfold3.core.utils.vendor_rng`.

The load-bearing test here is
:func:`test_dropout_mask_survives_checkpoint_recompute`. ``torch.utils.checkpoint``
rewinds only ``torch.get_rng_state()`` — the *default* CPU generator — so routing
these draws through a private ``torch.Generator`` would hand the recompute pass a
different dropout mask and corrupt gradients silently.
"""

import pytest
import torch
import torch.utils.checkpoint as cp
from torch import nn

from openfold3.core.model.primitives.dropout import DropoutRowwise
from openfold3.core.utils import vendor_rng

#: ROCm reports its devices as ``cuda``, so this covers both vendors.
ACCEL = "cuda" if torch.cuda.is_available() else None
requires_accelerator = pytest.mark.skipif(
    ACCEL is None, reason="needs a GPU to be a cross-device test"
)


@pytest.fixture
def parity_enabled():
    """Enable vendor-independent draws for one test, then restore."""
    original = vendor_rng._enabled
    vendor_rng.set_enabled(True)
    yield
    vendor_rng._enabled = original


@pytest.fixture
def parity_disabled():
    original = vendor_rng._enabled
    vendor_rng.set_enabled(False)
    yield
    vendor_rng._enabled = original


# ---------------------------------------------------------------------------
# enabled() / set_enabled()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("YES", True),
        ("  1  ", True),
        ("0", False),
        ("false", False),
        ("", False),
        # An allow-list, so every other spelling of "off" stays off. A deny-list
        # read each of these as enabled and silently rerouted every draw to CPU.
        ("no", False),
        ("off", False),
        ("FALSE", False),
        ("disabled", False),
    ],
)
def test_enabled_reads_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv(vendor_rng._ENV_VAR, value)
    monkeypatch.setattr(vendor_rng, "_enabled", None)
    assert vendor_rng.enabled() is expected


def test_enabled_defaults_off(monkeypatch):
    monkeypatch.delenv(vendor_rng._ENV_VAR, raising=False)
    monkeypatch.setattr(vendor_rng, "_enabled", None)
    assert vendor_rng.enabled() is False


# ---------------------------------------------------------------------------
# dropout_mask
# ---------------------------------------------------------------------------


def test_disabled_dropout_mask_matches_functional_dropout(parity_disabled):
    """The default training path must keep its exact numerics and RNG draws."""
    shape = (1, 8, 16)
    torch.manual_seed(0)
    ours = vendor_rng.dropout_mask(shape, 0.25, device="cpu", dtype=torch.float32)
    torch.manual_seed(0)
    reference = nn.functional.dropout(torch.ones(shape), p=0.25, training=True)
    assert torch.equal(ours, reference)


@pytest.mark.parametrize("p", [0.0, 0.15, 0.25, 1.0])
def test_dropout_mask_rate_and_scale(parity_enabled, p):
    torch.manual_seed(0)
    mask = vendor_rng.dropout_mask((256, 512), p, device="cpu", dtype=torch.float32)
    dropped = float((mask == 0).float().mean())
    assert dropped == pytest.approx(p, abs=0.01)
    if p < 1.0:
        assert float(mask.max()) == pytest.approx(1.0 / (1.0 - p))
    else:
        assert float(mask.abs().max()) == 0.0


def test_dropout_mask_rejects_out_of_range_p(parity_enabled):
    with pytest.raises(ValueError, match="must be in"):
        vendor_rng.dropout_mask((4,), 1.5, device="cpu")


def test_dropout_mask_is_reproducible_from_the_cpu_seed(parity_enabled):
    torch.manual_seed(7)
    first = vendor_rng.dropout_mask((64, 64), 0.25, device="cpu")
    torch.manual_seed(7)
    second = vendor_rng.dropout_mask((64, 64), 0.25, device="cpu")
    torch.manual_seed(8)
    other = vendor_rng.dropout_mask((64, 64), 0.25, device="cpu")
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


@requires_accelerator
def test_enabled_draws_do_not_touch_the_device_rng(parity_enabled):
    """Device RNG must stay untouched, or the two legs desynchronise anyway."""
    torch.manual_seed(0)
    before = torch.cuda.get_rng_state()
    vendor_rng.dropout_mask((128, 128), 0.25, device=ACCEL)
    vendor_rng.randperm(1024, device=ACCEL)
    vendor_rng.randn((64, 64), device=ACCEL)
    assert torch.equal(before, torch.cuda.get_rng_state())


@requires_accelerator
def test_disabled_dropout_does_touch_the_device_rng(parity_disabled):
    """Negative control: without the flag the device RNG is what gets consumed."""
    torch.manual_seed(0)
    before = torch.cuda.get_rng_state()
    vendor_rng.dropout_mask((128, 128), 0.25, device=ACCEL)
    assert not torch.equal(before, torch.cuda.get_rng_state())


# ---------------------------------------------------------------------------
# The checkpoint-recompute trap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_dropout_mask_survives_checkpoint_recompute(parity_enabled, use_reentrant):
    """Forward and recompute must draw the same mask.

    A private ``torch.Generator`` fails this in both reentrant modes, which is why
    :mod:`vendor_rng` draws from the default CPU generator instead.
    """
    device = ACCEL or "cpu"
    dropout = DropoutRowwise(0.25).to(device).train()
    first, second = nn.Linear(64, 64).to(device), nn.Linear(64, 64).to(device)
    masks = []

    def block(x):
        hidden = first(x).tanh()
        out = dropout(hidden)
        masks.append((out / hidden.clamp(min=1e-9)).detach().clone())
        return second(out).tanh()

    torch.manual_seed(3)
    inputs = torch.randn(1, 64, 64, device=device, requires_grad=True)
    cp.checkpoint(block, inputs, use_reentrant=use_reentrant).sum().backward()

    assert len(masks) == 2, (
        "block did not recompute; the test is not exercising the trap"
    )
    assert torch.equal(masks[0], masks[1])


# ---------------------------------------------------------------------------
# Passthrough helpers
# ---------------------------------------------------------------------------


@requires_accelerator
@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("randperm", lambda dev: vendor_rng.randperm(128, device=dev)),
        ("randint", lambda dev: vendor_rng.randint(0, 7, (32,), device=dev)),
        ("rand", lambda dev: vendor_rng.rand((32, 4), device=dev)),
        ("randn", lambda dev: vendor_rng.randn((32, 4), device=dev)),
        (
            "randn_like",
            lambda dev: vendor_rng.randn_like(torch.zeros(32, 4, device=dev)),
        ),
    ],
)
def test_helpers_land_on_the_requested_device(parity_enabled, name, call):
    assert call(ACCEL).device.type == torch.device(ACCEL).type


@pytest.mark.parametrize(
    ("ours", "reference"),
    [
        (lambda: vendor_rng.randperm(128, device="cpu"), lambda: torch.randperm(128)),
        (
            lambda: vendor_rng.randint(0, 7, (32,), device="cpu"),
            lambda: torch.randint(0, 7, (32,)),
        ),
        (lambda: vendor_rng.rand((32, 4), device="cpu"), lambda: torch.rand((32, 4))),
        (lambda: vendor_rng.randn((32, 4), device="cpu"), lambda: torch.randn((32, 4))),
    ],
)
def test_disabled_helpers_are_plain_torch_calls(parity_disabled, ours, reference):
    torch.manual_seed(11)
    got = ours()
    torch.manual_seed(11)
    assert torch.equal(got, reference())


def test_randn_honours_dtype(parity_enabled):
    assert (
        vendor_rng.randn((8,), device="cpu", dtype=torch.bfloat16).dtype
        == torch.bfloat16
    )
    assert (
        vendor_rng.randn_like(torch.zeros(8, dtype=torch.bfloat16)).dtype
        == torch.bfloat16
    )


@pytest.mark.parametrize("fn", ["randn_like", "rand_like"])
@pytest.mark.parametrize("device", ["cpu", 0, None])
def test_like_wrappers_accept_an_explicit_device(parity_enabled, fn, device):
    """torch.*_like takes device=, including the falsy-but-valid index 0.

    ``device=0`` is cuda:0 in torch, so `or` would have silently sent it to the
    source tensor's device instead.
    """
    if device == 0 and not torch.cuda.is_available():
        pytest.skip("device index 0 needs a GPU")
    kwargs = {} if device is None else {"device": device}
    out = getattr(vendor_rng, fn)(torch.zeros(4, 3), **kwargs)
    assert out.shape == (4, 3)
    expected = torch.zeros(1).device if device is None else torch.device(device)
    assert out.device == torch.empty(0, device=expected).device
