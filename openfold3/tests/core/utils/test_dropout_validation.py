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

"""Dropout rate validation, at construction rather than at the first forward."""

import pytest

from openfold3.core.model.primitives.dropout import (
    Dropout,
    DropoutColumnwise,
    DropoutRowwise,
)


@pytest.mark.parametrize("cls", [Dropout, DropoutRowwise, DropoutColumnwise])
@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan")])
def test_invalid_rate_is_rejected_at_construction(cls, rate):
    kwargs = {"batch_dim": -3} if cls is Dropout else {}
    with pytest.raises(ValueError, match="must be in"):
        cls(rate, **kwargs)


@pytest.mark.parametrize("rate", [0.0, 0.15, 0.25, 1.0])
def test_valid_rates_are_accepted(rate):
    assert DropoutRowwise(rate).r == rate
