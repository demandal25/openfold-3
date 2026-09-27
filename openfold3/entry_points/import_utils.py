# Copyright 2026 AlQuraishi Laboratory
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

"""
Manage imports run_openfold.py
"""
# ruff: noqa: F821
# ruff: noqa: F401


def _enable_tf32():
    import torch

    torch_versions = torch.__version__.split(".")
    torch_major_version = int(torch_versions[0])
    torch_minor_version = int(torch_versions[1])
    if torch_major_version > 1 or (
        torch_major_version == 1 and torch_minor_version >= 12
    ):
        # Gives a large speedup on Ampere-class GPUs
        torch.set_float32_matmul_precision("high")


def _configure_torch_backend():
    """Apply backend settings.

    ``OF3_BLAS_LIBRARY`` overrides the ROCm BLAS choice, for attributing a
    numerical difference to the GEMM backend. Unset keeps the default.
    """
    import os

    import torch

    # Empty means unset: `export OF3_BLAS_LIBRARY=` must not abort the run.
    library = (os.environ.get("OF3_BLAS_LIBRARY") or "").strip().lower() or None
    # torch's own set; "ck" is the ROCm alternative this knob exists to try.
    known = ("default", "cublas", "hipblas", "cublaslt", "hipblaslt", "ck")
    if library is not None and library not in known:
        raise ValueError(
            f"OF3_BLAS_LIBRARY={library!r} is not one of {', '.join(known)}"
        )
    if not torch.cuda.is_available():
        return
    if library is not None and torch.version.hip is None and library.startswith("hip"):
        raise ValueError(
            f"OF3_BLAS_LIBRARY={library!r} is a ROCm backend but this is a CUDA "
            f"build; use cublas or cublaslt on the NVIDIA leg"
        )
    if torch.version.hip is not None:
        # Force the cuBLAS backend on AMD/ROCm to match the numerics of
        # NVIDIA-trained models.
        torch.backends.cuda.preferred_blas_library(library or "cublas")
    elif library is not None:
        # Honoured on CUDA too, so the same control arm can run on both legs.
        torch.backends.cuda.preferred_blas_library(library)
