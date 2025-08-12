"""Minimal generic trainer supporting unsupervised HMM training.

Design goals:
- Extremely small / readable.
- Works with any `BaseModel` whose forward returns per-sample log-likelihood or predictions.
- Default loss (if none provided): negative mean of model output (assumed log-likelihood).
- Supports optional validation, gradient clipping, and simple metric logging.

Example (HMM):
	from torch.utils.data import DataLoader, TensorDataset
	import torch
	from scr.models.hmm import HMM
	from scr.training.base_trainer import BaseTrainer

	x = torch.randn(128, 100, 10)  # 128 sequences
	ds = TensorDataset(x)  # each item -> (tensor,)
	dl = DataLoader(ds, batch_size=16, shuffle=True)
	model = HMM(num_states=4, obs_dim=10)
	optim = torch.optim.Adam(model.parameters(), lr=1e-2)
	trainer = BaseTrainer(model, optim, dl)
	trainer.fit(epochs=5)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional
import time

import torch
from torch.utils.data import DataLoader


LossFn = Callable[[torch.Tensor, list[torch.Tensor] | None], torch.Tensor]


@dataclass
class TrainState:
	epoch: int = 0
	global_step: int = 0


class BaseTrainer:
	def __init__(
		self,
		model: torch.nn.Module,
		optimizer: torch.optim.Optimizer,
		train_loader: DataLoader,
		val_loader: DataLoader | None = None,
		device: str | torch.device | None = None,
		loss_fn: LossFn | None = None,
		grad_clip: float | None = None,
		log_interval: int = 50,
		verbose: bool = True,
		log_each_step: bool = False,
	) -> None:
		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = torch.device(device)
		self.model = model.to(self.device)
		self.optimizer = optimizer
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.grad_clip = grad_clip
		self.log_interval = log_interval
		self.state = TrainState()
		self.verbose = verbose
		self._total_epochs: int | None = None  # filled in fit()
		self.log_each_step = log_each_step

		if loss_fn is None:
			# Assume model(x) returns per-sample log-likelihood -> maximize => minimize -mean
			def _default_loss(out: torch.Tensor, _: list[torch.Tensor] | None) -> torch.Tensor:
				return -out.mean()
			loss_fn = _default_loss
		self.loss_fn = loss_fn

	def _move(self, batch):
		if isinstance(batch, torch.Tensor):
			return batch.to(self.device)
		if isinstance(batch, (list, tuple)):
			return [self._move(b) for b in batch]
		return batch

	def _unpack_inputs(self, batch):
		# Accept forms: x ; (x,) ; (x, y) (ignore y for unsupervised) ; dict
		if isinstance(batch, torch.Tensor):
			return batch
		if isinstance(batch, (list, tuple)):
			return batch[0]
		if isinstance(batch, dict):
			# Try common keys
			for k in ("x", "data", "inputs"):
				if k in batch:
					return batch[k]
		raise ValueError("Cannot infer input tensor from batch structure.")

	def train_epoch(self) -> dict:
		self.model.train()
		total_loss = 0.0
		total_samples = 0
		start_time = time.time()
		try:
			num_batches = len(self.train_loader)  # type: ignore[arg-type]
		except Exception:
			num_batches = None
		if self.verbose:
			if num_batches:
				print(f"Epoch {self.state.epoch}/{self._total_epochs or '?'} - {num_batches} batches")
			else:
				print(f"Epoch {self.state.epoch}/{self._total_epochs or '?'}")
		for i, batch in enumerate(self.train_loader, start=1):
			batch = self._move(batch)
			x = self._unpack_inputs(batch)
			# Ensure batch dimension
			if x.dim() == 2:
				x = x.unsqueeze(0)
			out = self.model(x)
			loss = self.loss_fn(out, batch if isinstance(batch, list) else None)
			self.optimizer.zero_grad()
			loss.backward()
			if self.grad_clip is not None:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
			self.optimizer.step()

			bs = x.shape[0]
			total_loss += loss.item() * bs
			total_samples += bs
			self.state.global_step += 1

			if self.verbose and (self.log_each_step or i % self.log_interval == 0 or i == 1):
				avg_loss = total_loss / max(1, total_samples)
				if num_batches:
					pct = (i / num_batches) * 100
					print(f"  [Batch {i}/{num_batches} ({pct:5.1f}%)] loss={loss.item():.4f} avg={avg_loss:.4f}")
				else:
					print(f"  [Batch {i}] loss={loss.item():.4f}")

		elapsed = time.time() - start_time
		metrics = {
			"train_loss": total_loss / max(1, total_samples),
			"time_s": elapsed,
		}
		if self.verbose:
			print(f"Epoch {self.state.epoch} done in {elapsed:.2f}s | loss={metrics['train_loss']:.4f}")
		return metrics

	@torch.no_grad()
	def evaluate(self) -> dict:
		if self.val_loader is None:
			return {}
		self.model.eval()
		total_loss = 0.0
		total_samples = 0
		for batch in self.val_loader:
			batch = self._move(batch)
			x = self._unpack_inputs(batch)
			if x.dim() == 2:
				x = x.unsqueeze(0)
			out = self.model(x)
			loss = self.loss_fn(out, batch if isinstance(batch, list) else None)
			bs = x.shape[0]
			total_loss += loss.item() * bs
			total_samples += bs
		return {"val_loss": total_loss / max(1, total_samples)}

	def fit(self, epochs: int) -> list[dict]:
		history: list[dict] = []
		self._total_epochs = epochs
		for e in range(1, epochs + 1):
			self.state.epoch = e
			train_metrics = self.train_epoch()
			val_metrics = self.evaluate()
			metrics = {"epoch": e, **train_metrics, **val_metrics}
			history.append(metrics)
			if self.verbose:
				msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k,v in metrics.items())
				print(msg)
		return history

__all__ = ["BaseTrainer"]

