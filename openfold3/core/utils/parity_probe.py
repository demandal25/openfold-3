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

"""Per-step trajectory records, for comparing two training runs step by step.

End-of-run metrics cannot localise a divergence. This writes one JSONL row per
observation so two runs can be aligned and the *first* disagreeing step and
module identified.

Every activation row carries ``(step, module, ordinal)``, where ordinal counts
firings of that module within the step. That is not bookkeeping: ``num_recycles``
is drawn per step, the MSA embedder sits inside the recycle loop, the diffusion
rollout adds forwards, and activation checkpointing re-runs blocks during
backward — so a module fires a variable number of times per step. Two runs whose
firing counts differ are *already* divergent, and the ordinal makes that visible
instead of silently shifting every later comparison.

Deliberately no forward/backward label: OF3's training step interleaves them
(``runner.py`` calls ``manual_backward`` per sample inside the loop), so a phase
flag driven by Lightning hooks would be wrong after the first sample. The
ordinal alone is unambiguous.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch

logger = logging.getLogger(__name__)

#: Modules worth watching by default: the trunk stacks where instability shows
#: up first, plus the diffusion path. Matched with ``re.search`` against the
#: dotted module name.
DEFAULT_MODULE_PATTERNS = (
    r"pairformer_stack\.blocks\.\d+$",
    r"msa_module_stack\.blocks\.\d+$",
    r"template_embedder$",
    r"diffusion_module$",
    r"diffusion_transformer\.blocks\.\d+$",
)


class NonFiniteValue(RuntimeError):
    """Raised at the first non-finite activation or gradient."""


def _stats(tensor: torch.Tensor) -> dict | None:
    """Cheap summary of a tensor; ``None`` for non-float tensors.

    Reduced in fp32 so bf16 sums do not saturate.
    """
    if not torch.is_floating_point(tensor):
        return None
    flat = tensor.detach().float().reshape(-1)
    finite = torch.isfinite(flat)
    n_bad = int((~finite).sum())
    if n_bad == flat.numel():
        return {"shape": list(tensor.shape), "nonfinite": n_bad}
    good = flat[finite]
    return {
        "shape": list(tensor.shape),
        "mean": float(good.mean()),
        "std": float(good.std()) if good.numel() > 1 else 0.0,
        "absmax": float(good.abs().max()),
        "nonfinite": n_bad,
    }


class ParityProbeCallback(pl.Callback):
    """Write a per-step trajectory record for cross-vendor comparison.

    Args:
        output_dir: Directory for ``trajectory_rank<N>.jsonl``.
        module_patterns: Regexes matched against dotted module names.
        every_n_steps: How often to record the expensive per-module and
            per-parameter detail. A cheap global gradient summary and the batch
            identity are recorded on *every* step regardless, so a loss spike or
            NaN is never missed between probe steps.
        abort_on_nonfinite: Stop at the first non-finite value rather than let it
            propagate into a run that is no longer comparable.
    """

    def __init__(
        self,
        output_dir: str | Path,
        module_patterns: tuple[str, ...] = DEFAULT_MODULE_PATTERNS,
        every_n_steps: int = 1,
        abort_on_nonfinite: bool = True,
    ):
        super().__init__()
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be >= 1, got {every_n_steps}")
        self.output_dir = Path(output_dir)
        self.patterns = [re.compile(p) for p in module_patterns]
        self.every_n_steps = every_n_steps
        self.abort_on_nonfinite = abort_on_nonfinite

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._file = None
        self._ordinals: dict[str, int] = defaultdict(int)
        self._step = 0
        self._probing = False

    # -- lifecycle ---------------------------------------------------------

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage=None):
        if stage != "fit" or self._file is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"trajectory_rank{trainer.global_rank}.jsonl"
        # Line-buffered, so a run killed by a hang still leaves a usable trace.
        self._file = path.open("a", buffering=1)
        self._register_hooks(pl_module)
        logger.info(
            "ParityProbe: %d modules matched, writing %s", len(self._handles), path
        )
        if not self._handles:
            logger.warning(
                "ParityProbe: no module matched %s; only gradients and losses "
                "will be recorded",
                [p.pattern for p in self.patterns],
            )

    def teardown(self, trainer, pl_module, stage=None):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._file is not None:
            self._file.close()
            self._file = None

    def _register_hooks(self, pl_module: pl.LightningModule) -> None:
        for name, module in pl_module.named_modules():
            if any(p.search(name) for p in self.patterns):
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    # -- recording ---------------------------------------------------------

    def _emit(self, **row) -> None:
        if self._file is not None:
            self._file.write(json.dumps(row, sort_keys=True) + "\n")

    def _check_finite(self, stats: dict, what: str, name: str) -> None:
        if not stats.get("nonfinite", 0):
            return
        logger.error(
            "ParityProbe: %d non-finite values in %s of %r at step %d",
            stats["nonfinite"],
            what,
            name,
            self._step,
        )
        if self.abort_on_nonfinite:
            if self._file is not None:
                self._file.flush()
            raise NonFiniteValue(
                f"{stats['nonfinite']} non-finite values in {what} of {name!r} "
                f"at step {self._step}; trace in {self.output_dir}"
            )

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not self._probing:
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            stats = _stats(tensor)
            if stats is None:
                return
            ordinal = self._ordinals[name]
            self._ordinals[name] = ordinal + 1
            self._emit(
                kind="activation",
                step=self._step,
                module=name,
                ordinal=ordinal,
                **stats,
            )
            self._check_finite(stats, "activation", name)

        return hook

    # -- hooks -------------------------------------------------------------

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._step = int(trainer.global_step)
        self._ordinals.clear()
        self._probing = self._step % self.every_n_steps == 0
        # Sample identity, every step: this is what separates an RNG/data desync
        # from a numeric divergence when two runs disagree.
        ids = batch.get("pdb_id") if isinstance(batch, dict) else None
        if ids is not None:
            self._emit(
                kind="batch",
                step=self._step,
                batch_idx=int(batch_idx),
                ids=[str(i) for i in ids],
            )

    def on_before_backward(self, trainer, pl_module, loss):
        # Fires once per sample: OF3 calls manual_backward inside its own loop.
        ordinal = self._ordinals["__loss__"]
        self._ordinals["__loss__"] = ordinal + 1
        self._emit(
            kind="loss", step=self._step, ordinal=ordinal, value=float(loss.detach())
        )

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """Record gradients. They are already clipped, synced and averaged here.

        OF3 runs ``grad_manager.clip_and_accumulate`` and ``sync_and_average_grads``
        before ``opt.step()``; the unclipped norms come from the grad manager's
        own logging, not from this hook.
        """
        named_grads = [
            (name, param.grad)
            for name, param in pl_module.named_parameters()
            if param.grad is not None and torch.is_floating_point(param.grad)
        ]
        if not named_grads:
            return

        # Accumulate on device and sync once: a per-parameter .item() here would
        # be hundreds of host syncs per step on a ~600M-parameter model.
        total_sq = max_abs = n_bad = None
        for _name, grad in named_grads:
            flat = grad.detach().float().reshape(-1)
            finite = torch.isfinite(flat)
            safe = torch.where(finite, flat, torch.zeros_like(flat))
            sq, mx, bad = safe.dot(safe), safe.abs().max(), (~finite).sum()
            total_sq = sq if total_sq is None else total_sq + sq
            max_abs = mx if max_abs is None else torch.maximum(max_abs, mx)
            n_bad = bad if n_bad is None else n_bad + bad

        n_bad = int(n_bad)
        self._emit(
            kind="grad_summary",
            step=self._step,
            clipped=True,
            total_norm=float(total_sq) ** 0.5,
            absmax=float(max_abs),
            nonfinite=n_bad,
            n_params=len(named_grads),
        )

        # Per-parameter detail only when probing, or when the summary says
        # something went non-finite and we need to name the parameter.
        if self._probing or n_bad:
            for name, grad in named_grads:
                stats = _stats(grad)
                if stats is None:
                    continue
                self._emit(
                    kind="grad", step=self._step, param=name, clipped=True, **stats
                )
                self._check_finite(stats, "gradient", name)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._file is not None:
            self._file.flush()
