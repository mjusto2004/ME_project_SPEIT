"""Base class shared by all models."""

from __future__ import annotations

import argparse
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:
    from ..datasets import Dataset, DatasetSpec


class IncompatibleError(ValueError):
    """Raised when a model cannot be used with a given dataset."""


SCHEDULERS = ("none", "cosine", "plateau")


class LRSchedule:
    """Learning rate of :meth:`SurrogateModel._fit_loop` (see there for the kinds).

    Kept outside ``torch.optim.lr_scheduler`` so that every kind is a few
    readable lines: :meth:`apply` sets the rate before each batch,
    :meth:`end_of_epoch` reacts to the validation result.
    """

    def __init__(self, kind, lr, epochs, steps_per_epoch, warmup=0, min_lr=0.0,
                 plateau_patience=10, plateau_factor=0.5):
        if kind not in SCHEDULERS:
            raise ValueError(f"Unknown scheduler '{kind}'. Available: {list(SCHEDULERS)}")
        if not 0 <= min_lr <= lr:
            raise ValueError("min_lr must be between 0 and lr.")
        if warmup < 0 or (warmup > 0 and warmup >= epochs):
            raise ValueError("warmup must be >= 0 and smaller than the number of epochs.")
        if not 0 < plateau_factor < 1 or plateau_patience < 1:
            raise ValueError("plateau_factor must be in (0, 1) and plateau_patience >= 1.")
        self.kind, self.base, self.min_lr = kind, lr, min_lr
        self.steps_per_epoch = max(steps_per_epoch, 1)
        self.total = epochs * self.steps_per_epoch
        self.warmup = warmup * self.steps_per_epoch
        self.plateau_patience, self.plateau_factor = plateau_patience, plateau_factor
        self.step, self.stale = 0, 0
        self.lr = self._cosine(0) if kind == "cosine" else lr

    def _cosine(self, step: int) -> float:
        if step < self.warmup:
            return self.base * (step + 1) / self.warmup
        progress = (step - self.warmup) / max(self.total - self.warmup - 1, 1)  # last batch: min_lr
        return self.min_lr + (self.base - self.min_lr) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def next_lr(self) -> float:
        """Rate the next batch will use."""
        return self._cosine(self.step) if self.kind == "cosine" else self.lr

    def apply(self, optimizer) -> None:
        """Set the rate of the next batch."""
        if self.kind == "cosine":
            self.lr = self._cosine(self.step)
        self.step += 1
        for group in optimizer.param_groups:
            group["lr"] = self.lr

    def end_of_epoch(self, improved: bool) -> bool:
        """Update the plateau schedule. ``True`` means stop: the rate is at its minimum and still no progress."""
        if self.kind != "plateau":
            return False
        self.stale = 0 if improved else self.stale + 1
        if self.stale < self.plateau_patience:
            return False
        if self.lr <= self.min_lr:
            return True
        self.lr, self.stale = max(self.lr * self.plateau_factor, self.min_lr), 0
        return False

    def describe(self) -> str:
        if self.kind == "cosine":
            return f"cosine to {self.min_lr:g}, warmup {self.warmup // self.steps_per_epoch} epochs"
        if self.kind == "plateau":
            return (f"plateau: x{self.plateau_factor:g} after {self.plateau_patience} epochs "
                    f"without progress, min {self.min_lr:g}")
        return "constant"


class SurrogateModel(nn.Module, ABC):
    """Maps ``n_inputs`` parameters to ``n_outputs`` values.

    Every model must implement two methods (the class cannot be instantiated
    otherwise):

    * :meth:`forward`: the prediction;
    * :meth:`fit`: trains the model (usually by calling :meth:`_fit_loop`).

    Subclasses are registered with :func:`~me_project_speit.models.register_model`
    and must accept ``(n_inputs, n_outputs, **kwargs)``.

    Compatibility with a dataset is checked by :meth:`check_compatible`. Its
    default implementation verifies that the dataset class is one of
    :attr:`supported_datasets` (or a subclass of one); models with extra
    requirements (fixed mesh, coordinate inputs, ...) extend it and call
    ``super().check_compatible``.
    """

    name: str = ""  # set by @register_model
    init_kwargs: dict  # set by build_model, stored in checkpoints to rebuild the model

    #: Dataset classes the model can learn from (subclasses are accepted too);
    #: ``None`` accepts any dataset.
    supported_datasets: "tuple[type[Dataset], ...] | None" = None

    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Predict the outputs from one batch of inputs (what a loader yields before the target)."""

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        epochs: int,
        lr: float,
        patience: int,
        device: torch.device,
        y_std: torch.Tensor,
        on_epoch: Callable[..., None] | None = None,
        on_best: Callable[[int], None] | None = None,
        **fit_kwargs,
    ) -> int:
        """Train the model; returns the best epoch (lowest validation MSE).

        ``on_epoch(epoch, train_mse, val_mse, val_mae, val_rmse[, lr])`` is called
        after every epoch and ``on_best(epoch)`` each time the validation MSE
        improves (the caller uses it to save a checkpoint). ``y_std`` converts
        normalized errors back to real units. ``fit_kwargs`` are the model's own
        training options (see :meth:`add_fit_arguments`). Training stops after ``patience``
        epochs without improvement. Models with a standard loop simply call
        :meth:`_fit_loop`.
        """

    def set_scaling(self, x_mean, x_std, y_mean, y_std) -> None:
        """Called by ``train`` before :meth:`fit` with the normalisation statistics.

        Does nothing by default. Override it when the model must work on raw
        (physical) values instead of the normalised ones the loaders give.
        """

    def evaluate(self, loader: DataLoader, device: torch.device, y_std: torch.Tensor):
        """Normalized MSE and real-unit MAE / RMSE over ``loader``."""
        self.eval()
        sq_norm = abs_real = sq_real = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                *inputs, target = (t.to(device, non_blocking=True) for t in batch)
                difference = self(*inputs) - target
                real_difference = difference * y_std
                sq_norm += difference.square().sum().item()
                abs_real += real_difference.abs().sum().item()
                sq_real += real_difference.square().sum().item()
                count += target.numel()
        return sq_norm / count, abs_real / count, math.sqrt(sq_real / count)

    def _fit_loop(
        self, train_loader, val_loader, *, epochs, lr, patience, device, y_std,
        on_epoch=None, on_best=None, weight_decay: float = 0.0,
        scheduler: str = "none", warmup: int = 0, min_lr: float = 0.0,
        plateau_patience: int = 10, plateau_factor: float = 0.5,
    ) -> int:
        """Standard training: Adam (with optional ``weight_decay``) + MSE on batches shaped
        ``(*inputs, target)``, early stopping, optional learning-rate schedule.

        ``scheduler``:

        - ``"none"``: constant ``lr``;
        - ``"cosine"``: linear warmup over ``warmup`` epochs, then cosine decay from
          ``lr`` to ``min_lr`` at the last epoch (updated at every batch). It reaches
          ``min_lr`` only if early stopping does not stop the run first;
        - ``"plateau"``: the rate is multiplied by ``plateau_factor`` after
          ``plateau_patience`` epochs without a better validation MSE, never below
          ``min_lr``; training stops when it is at ``min_lr`` and still not improving.
        """
        schedule = LRSchedule(scheduler, lr, epochs, len(train_loader), warmup, min_lr,
                              plateau_patience, plateau_factor)
        optimizer = torch.optim.Adam(self.parameters(), lr=schedule.lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        best_val, best_epoch, stale = float("inf"), 0, 0

        for epoch in range(1, epochs + 1):
            self.train()
            total, count = 0.0, 0
            epoch_lr = schedule.next_lr()  # rate of the first batch (cosine changes it at every batch)
            # leave=False: the bar is erased from the terminal when the epoch is over.
            bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False, unit="batch")
            for batch in bar:
                *inputs, target = (t.to(device, non_blocking=True) for t in batch)
                schedule.apply(optimizer)
                optimizer.zero_grad()
                loss = criterion(self(*inputs), target)
                loss.backward()
                optimizer.step()
                total += loss.item() * target.numel()
                count += target.numel()
                bar.set_postfix(loss=f"{total / count:.4f}")
            train_mse = total / count

            val_mse, val_mae, val_rmse = self.evaluate(val_loader, device, y_std)
            if not math.isfinite(val_mse):
                raise RuntimeError("Non-finite validation loss.")
            if on_epoch:
                on_epoch(epoch, train_mse, val_mse, val_mae, val_rmse, epoch_lr)

            if val_mse < best_val:
                best_val, best_epoch, stale = val_mse, epoch, 0
                if on_best:
                    on_best(epoch)
            else:
                stale += 1

            # Every epoch leaves one line (tqdm.write keeps the terminal tidy next to the bar).
            star = " *" if stale == 0 else ""  # * = new best validation MSE
            tqdm.write(
                f"Epoch {epoch:3d} | Train MSE: {train_mse:.5f} | "
                f"Val MSE: {val_mse:.5f} | Val MAE: {val_mae:.3f} | LR: {epoch_lr:.2e}{star}"
            )
            if stale >= patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break
            if schedule.end_of_epoch(improved=stale == 0):
                tqdm.write(f"Learning rate at its minimum ({min_lr:g}) and no progress: stopping at epoch {epoch}")
                break
        return best_epoch

    @classmethod
    def add_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Override to expose constructor arguments as options of ``train``.

        Default: none. Use the constructor argument name as ``dest``
        (``--n-heads`` -> ``n_heads``, the default of argparse) and leave the
        argparse default at ``None``, meaning "not given: use the constructor's
        default".
        """

    @classmethod
    def kwargs_from_cli(cls, args: argparse.Namespace) -> dict:
        """Constructor keyword arguments given on the command line.

        Default: every constructor parameter (except ``n_inputs`` and
        ``n_outputs``) present in ``args`` with a value other than ``None``.
        """
        parameters = list(inspect.signature(cls.__init__).parameters)[1:]  # without self
        names = [p for p in parameters if p not in ("n_inputs", "n_outputs")]
        return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}

    @classmethod
    def add_fit_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Override to expose extra training options (e.g. ``--weight-decay``) on ``train``.

        They reach :meth:`fit` as keyword arguments, next to ``epochs``, ``lr``
        and ``patience``. Same rules as :meth:`add_cli_arguments`: ``dest`` is
        the keyword name and the argparse default stays ``None``.
        """

    @classmethod
    def fit_kwargs_from_cli(cls, args: argparse.Namespace) -> dict:
        """Keyword arguments for :meth:`fit` given on the command line.

        Default: every option declared by :meth:`add_fit_arguments` whose value
        is not ``None``.
        """
        probe = argparse.ArgumentParser(add_help=False)
        cls.add_fit_arguments(probe)
        names = [a.dest for a in probe._actions if a.dest != "help"]
        return {n: getattr(args, n) for n in names if getattr(args, n, None) is not None}

    @classmethod
    def from_spec(cls, spec: "DatasetSpec", **kwargs) -> "SurrogateModel":
        """Build the model for a dataset. Default: ``cls(spec.n_inputs, spec.n_outputs, **kwargs)``.

        Override when the model needs more of the spec (e.g. the input names).
        ``kwargs`` are the options stored in checkpoints; what is taken from the
        spec is not stored, since the checkpoint keeps the spec.
        """
        return cls(spec.n_inputs, spec.n_outputs, **kwargs)

    @classmethod
    def check_compatible(cls, spec: "DatasetSpec") -> None:
        """Raise :class:`IncompatibleError` if ``spec`` cannot be handled."""
        if cls.supported_datasets is None:
            return
        from ..datasets import get_dataset_class

        dataset_cls = get_dataset_class(spec.kind)
        if not issubclass(dataset_cls, cls.supported_datasets):
            wanted = [c.__name__ for c in cls.supported_datasets]
            raise IncompatibleError(
                f"needs a dataset of type {wanted}, got {dataset_cls.__name__} (kind '{spec.kind}')"
            )
