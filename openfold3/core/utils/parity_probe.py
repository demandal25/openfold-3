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

Rows are keyed ``(step, batch_idx, module, ordinal)``. A module fires a variable
number of times per step -- recycles, diffusion rollout, checkpoint recompute --
so the ordinal is what keeps two runs aligned.
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

#: Trunk stacks where instability shows up first, plus the diffusion path and
#: the confidence head's own stack. Anchored: an unanchored pairformer pattern
#: also matches the confidence head's separate stack.
DEFAULT_MODULE_PATTERNS = (
    r"^model\.pairformer_stack\.blocks\.\d+$",
    r"^model\.msa_module\.blocks\.\d+$",
    r"^model\.template_embedder\.template_pair_stack\.blocks\.\d+$",
    r"^model\.diffusion_module\.diffusion_transformer\.blocks\.\d+$",
    r"^model\.aux_heads\.pairformer_embedding\.pairformer_stack\.blocks\.\d+$",
)


def check_patterns_match(module: torch.nn.Module, patterns) -> dict[str, int]:
    """Return per-pattern match counts against ``module.named_modules()``.

    A pattern that matches nothing is the failure mode this exists to catch: the
    probe would log one warning and then record no activations at all.
    """
    names = [name for name, _ in module.named_modules()]
    return {p: sum(bool(re.search(p, n)) for n in names) for p in patterns}


class NonFiniteValue(RuntimeError):
    """Raised at the first non-finite activation or gradient."""


def _stats(tensor: torch.Tensor) -> dict | None:
    """Cheap summary of a tensor; ``None`` for non-float tensors.

    Reduced in fp32 so bf16 sums do not saturate, and stacked into one tensor so
    the whole summary costs a single host sync rather than four — the probe runs
    on ~80 modules per recycle.
    """
    if not torch.is_floating_point(tensor):
        return None
    flat = tensor.detach().float().reshape(-1)
    # max() on an empty tensor raises, and a probe must not kill the run.
    if flat.numel() == 0:
        return {"shape": list(tensor.shape), "nonfinite": 0}
    finite = torch.isfinite(flat)
    n_finite = finite.sum()
    # Non-finite entries are zeroed rather than indexed out: boolean masking
    # would force a host sync of its own to size the result.
    safe = torch.where(finite, flat, torch.zeros_like(flat))
    count = n_finite.clamp(min=1)
    mean = safe.sum() / count
    var = ((safe - mean) * finite).pow(2).sum() / (count - 1).clamp(min=1)
    packed = torch.stack(
        [
            mean,
            var.sqrt(),
            safe.abs().max(),
            (flat.numel() - n_finite).float(),
            n_finite.float(),
        ]
    ).tolist()

    shape = list(tensor.shape)
    if packed[4] == 0:  # everything was non-finite; the moments are meaningless
        return {"shape": shape, "nonfinite": int(packed[3])}
    return {
        "shape": shape,
        "mean": packed[0],
        "std": packed[1],
        "absmax": packed[2],
        "nonfinite": int(packed[3]),
    }


class ParityProbeCallback(pl.Callback):
    """Write a per-step trajectory record for cross-vendor comparison.

    ``every_n_steps`` thins the per-module and per-parameter detail only; the
    gradient summary and batch identity are recorded every step.
    """

    def __init__(
        self,
        output_dir: str | Path,
        module_patterns: tuple[str, ...] = DEFAULT_MODULE_PATTERNS,
        every_n_steps: int = 1,
        abort_on_nonfinite: bool = True,
        max_firings_per_module: int = 16,
    ):
        super().__init__()
        if every_n_steps < 1:
            raise ValueError(f"every_n_steps must be >= 1, got {every_n_steps}")
        if max_firings_per_module < 0:
            raise ValueError("max_firings_per_module must be >= 0 (0 = unlimited)")
        self.output_dir = Path(output_dir)
        self.patterns = [re.compile(p) for p in module_patterns]
        self.every_n_steps = every_n_steps
        self.abort_on_nonfinite = abort_on_nonfinite
        self.max_firings_per_module = max_firings_per_module

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._file = None
        self._ordinals: dict[str, int] = defaultdict(int)
        self._step = 0
        self._batch_idx = 0
        self._probing = False
        self._warned_no_grads = False
        #: True only between on_train_batch_start and on_train_batch_end;
        #: otherwise validation forwards land on the preceding training step.
        self._in_train_batch = False

    # -- lifecycle ---------------------------------------------------------

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage=None):
        if stage != "fit" or self._file is not None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"trajectory_rank{trainer.global_rank}.jsonl"
        if path.exists() and path.stat().st_size:
            raise FileExistsError(
                f"{path} already holds a trajectory. Appending would splice two "
                f"runs into one file and the loader keeps only the last row per "
                f"key. Use a fresh output_dir, or move the old file aside."
            )
        # Line-buffered, so a run killed by a hang still leaves a usable trace.
        self._file = path.open("w", buffering=1)
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
            row.setdefault("batch_idx", self._batch_idx)
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
            if not (self._probing and self._in_train_batch):
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            stats = _stats(tensor)
            if stats is None:
                return
            ordinal = self._ordinals[name]
            self._ordinals[name] = ordinal + 1
            # Beyond the cap the module is still counted but not recorded: the
            # diffusion transformer fires ~842x per step under rollout, and the
            # onset of a divergence is in the first few, not the last.
            capped = self.max_firings_per_module
            if capped and ordinal >= capped:
                return
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
        # Keyed on batch_idx too: under gradient accumulation several
        # micro-batches share one global_step, and ordinals restart per
        # batch, so records would overwrite each other on load.
        self._batch_idx = int(batch_idx)
        self._ordinals.clear()
        self._probing = self._step % self.every_n_steps == 0
        self._in_train_batch = True
        # Sample identity, every step: this is what separates an RNG/data desync
        # from a numeric divergence when two runs disagree.
        ids = batch.get("pdb_id") if isinstance(batch, dict) else None
        if ids is not None:
            self._emit(kind="batch", step=self._step, ids=[str(i) for i in ids])

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
            if not self._warned_no_grads:
                self._warned_no_grads = True
                logger.warning(
                    "ParityProbe: no parameter has .grad at the optimizer step, "
                    "so no gradients will be recorded. Under DeepSpeed/ZeRO the "
                    "grads are partitioned and .grad is None; this harness pins "
                    "DDP, so a run reaching here is misconfigured."
                )
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
        # Full firing counts, even for modules capped above. Two runs whose
        # counts differ are already divergent, and this is what makes that
        # visible after capping.
        self._in_train_batch = False
        firings = {k: v for k, v in self._ordinals.items() if not k.startswith("__")}
        if firings:
            self._emit(kind="firings", step=self._step, counts=firings)
        if self._file is not None:
            self._file.flush()
