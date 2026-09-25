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

"""Vendor-independent RNG draws, for cross-vendor comparison runs.

Device RNG is not portable: CUDA and ROCm disagree on ``randperm`` (different
sort backends) and on per-call offset advancement, so two runs of the same model
on the two vendors diverge before any arithmetic happens. When enabled, these
helpers draw on CPU and copy to the device instead.

The draws deliberately use the *default* CPU generator rather than a private
one: ``torch.utils.checkpoint`` saves and restores ``torch.get_rng_state()``, so
only the default generator is rewound when a checkpointed block is recomputed
during backward. A private generator would yield different dropout masks on the
recompute pass and silently corrupt gradients.

Off by default; enable with ``OF3_VENDOR_INDEPENDENT_RNG=1`` or
:func:`set_enabled`. Costs a host-to-device copy per draw.
"""

from __future__ import annotations

import os

import torch

_ENV_VAR = "OF3_VENDOR_INDEPENDENT_RNG"
_enabled: bool | None = None


def enabled() -> bool:
    """Whether draws are routed through the CPU."""
    global _enabled
    if _enabled is None:
        _enabled = os.environ.get(_ENV_VAR, "0") not in ("", "0", "false", "False")
    return _enabled


def set_enabled(value: bool) -> None:
    """Override the environment setting. Intended for tests and parity runners."""
    global _enabled
    _enabled = value


def _device_of(device: torch.device | str | None) -> torch.device:
    return torch.device("cpu" if device is None else device)


def randperm(n: int, *, device=None, **kwargs) -> torch.Tensor:
    """``torch.randperm``, drawn on CPU when enabled.

    The highest-priority site: CUDA and ROCm use different sort backends here,
    so this diverges even with identical inputs and seeds.
    """
    if not enabled():
        return torch.randperm(n, device=device, **kwargs)
    return torch.randperm(n, device="cpu", **kwargs).to(_device_of(device))


def randint(low: int, high: int, size, *, device=None, **kwargs) -> torch.Tensor:
    """``torch.randint``, drawn on CPU when enabled."""
    if not enabled():
        return torch.randint(low, high, size, device=device, **kwargs)
    return torch.randint(low, high, size, device="cpu", **kwargs).to(_device_of(device))


def rand(size, *, device=None, **kwargs) -> torch.Tensor:
    """``torch.rand``, drawn on CPU when enabled."""
    if not enabled():
        return torch.rand(size, device=device, **kwargs)
    return torch.rand(size, device="cpu", **kwargs).to(_device_of(device))


def randn(size, *, device=None, dtype=None, **kwargs) -> torch.Tensor:
    """``torch.randn``, drawn on CPU when enabled.

    Drawn directly in ``dtype`` so the enabled and disabled paths differ only in
    where the numbers come from, not in rounding.
    """
    if not enabled():
        return torch.randn(size, device=device, dtype=dtype, **kwargs)
    out = torch.randn(size, device="cpu", dtype=dtype, **kwargs)
    return out.to(_device_of(device))


def randn_like(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """``torch.randn_like``, drawn on CPU when enabled."""
    if not enabled():
        return torch.randn_like(tensor, **kwargs)
    dtype = kwargs.pop("dtype", tensor.dtype)
    out = torch.randn(tensor.shape, device="cpu", dtype=dtype, **kwargs)
    return out.to(tensor.device)


def dropout_mask(shape, p: float, *, device=None, dtype=None) -> torch.Tensor:
    """Scaled inverted-dropout keep-mask, equal in distribution to
    ``nn.Dropout(p)`` applied to a tensor of ones.

    When disabled this *is* ``F.dropout`` on the target device, so the default
    training path keeps its exact numerics and RNG consumption.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"dropout probability must be in [0, 1], got {p}")
    device = _device_of(device)
    dtype = dtype or torch.float32
    if not enabled():
        ones = torch.ones(shape, device=device, dtype=dtype)
        return torch.nn.functional.dropout(ones, p=p, training=True)
    keep = 1.0 - p
    mask = torch.empty(shape, device="cpu", dtype=dtype)
    # p == 1 zeroes the mask rather than dividing by zero, as F.dropout does.
    if keep == 0.0:
        mask.zero_()
    else:
        mask.bernoulli_(keep).div_(keep)
    return mask.to(device)
