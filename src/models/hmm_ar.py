"""Autoregressive Hidden Markov Model (AR‑HMM) with state‑specific univariate AR
emissions (order defined by an arbitrary lag set).

Redesign (Phase 1) features:
		* Vectorised emission log-probability (no Python loop over lags per step).
		* Flexible input standardisation: accepts (B,T,1), (T,), (T,1), (B,T) and upgrades
			to canonical (B,T,1). (Keeps behaviour isolated from base HMM.)
		* Explicit prefix masking for first ``max(lags)`` frames (optional). When
			``normalize_time=True`` we can normalise by *valid* frames instead of T
			via ``drop_prefix_from_normalization``.
		* Ridge regularisation helper via ``regularization_loss``.
		* Warm start via least squares from provided state paths (unchanged API,
			internal logic clarified / lightly refactored).

Scope / simplifications:
		* Univariate only (obs_dim fixed to 1). Multivariate VAR could extend by
			reshaping lag stacks and learning per‑state matrices.
		* No variable‑length padding mask yet (must pass equal length sequences).
		* Innovation variance is scalar per state (log variance parameter).

Shape conventions
-----------------
Input x (canonical): (B, T, 1)
		Accepted shortcuts: (T,), (T,1) -> (1,T,1); (B,T) -> (B,T,1)
Output forward(x): (B,) log p(x) (optionally normalised)

Emission model
--------------
For state s with coefficients a_s (L,) over lags L_i in ``self.lags``:
		pred_{t,s} = Σ_j a_{s,j} x_{t - lag_j}
Residual r_{t,s} = x_t − pred_{t,s}; likelihood assumes N(0, σ_s^2)
Frames with insufficient history (t < max_lag) either:
		* contribute zero log-prob (masked) and optionally removed from normalisation,
			or
		* (future option) could use partial lag sets (not implemented for clarity now).
"""
from __future__ import annotations

from typing import Iterable, Sequence
import math
import torch
from torch import Tensor
import torch.nn as nn

from .base_model import BaseModel
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader


def exponential_lags(max_lag: int | None = None, num: int | None = None) -> list[int]:
	"""Return exponential lag list: 1,2,4,... up to constraint.

	Specify either ``max_lag`` (include powers-of-two <= max_lag) or ``num`` (number
	of lags). If both provided, stop when either condition reached.
	"""
	if max_lag is None and num is None:
		raise ValueError("Provide max_lag or num (or both).")
	lags: list[int] = []
	v = 1
	while True:
		if max_lag is not None and v > max_lag:
			break
		lags.append(v)
		if num is not None and len(lags) >= num:
			break
		v *= 2
	return lags


def fit_ar_ls(x: Tensor, lags: Sequence[int], ridge: float = 0.0) -> tuple[Tensor, Tensor]:
	"""Closed-form ridge least-squares AR fit for a single 1D sequence.

	Parameters
	----------
	x : (T,) tensor
	lags : list of positive integers
	ridge : λ (adds λI to normal equations)
	Returns
	-------
	coeffs : (L,)
	var : () noise variance estimate
	"""
	if x.dim() != 1:
		raise ValueError("x must be 1D tensor")
	Ls = list(lags)
	mL = max(Ls)
	if x.numel() <= mL:
		raise ValueError("Sequence too short for largest lag")
	rows = []
	ys = []
	for t in range(mL, x.numel()):
		rows.append([x[t - l].item() for l in Ls])
		ys.append(x[t].item())
	X = torch.tensor(rows, dtype=x.dtype, device=x.device)  # (N,L)
	y = torch.tensor(ys, dtype=x.dtype, device=x.device)    # (N,)
	XtX = X.T @ X
	if ridge > 0:
		XtX = XtX + ridge * torch.eye(XtX.size(0), device=X.device, dtype=X.dtype)
	Xty = X.T @ y
	coeffs = torch.linalg.solve(XtX, Xty)  # (L,)
	resid = y - X @ coeffs
	var = resid.pow(2).mean().clamp_min(1e-12)
	return coeffs, var


class HMMAR(BaseModel):
	"""Autoregressive HMM (univariate) with state‑specific AR(p) emissions.

	Parameters
	----------
	lags : Sequence[int]
		Positive lag values (duplicates removed, sorted).
	normalize_time : bool
		If True, forward returns average log-likelihood per DENOM where
		DENOM = T or (T - max_lag) depending on ``drop_prefix_from_normalization``.
	ridge : float
		L2 penalty factor applied to AR coefficients (exposed via
		``regularization_loss`` – not added automatically to forward).
	ignore_prefix : bool
		If True, first ``max_lag`` frames have zero emission contribution
		(masked out) because full lag context unavailable.
	drop_prefix_from_normalization : bool
		When both ``normalize_time`` and ``ignore_prefix`` are True, divide by
		number of *valid* frames (T - max_lag) instead of T.
	"""

	def __init__(
		self,
		data_loader: DataLoader,
		config: GlobalConfig,
		device: torch.device,
		lags: Iterable[int] = (1, 2, 4, 8),
		normalize_time: bool = False,
		ridge: float = 0.0,
		ignore_prefix: bool = True,
		drop_prefix_from_normalization: bool = True,
	) -> None:
		super().__init__(data_loader, config, device)
		
		# Validate that this is univariate data
		if self.num_features != 1:
			raise ValueError(f"HMMAR expects univariate data (num_features=1), got {self.num_features}")
		
		lags = sorted({int(l) for l in lags if l > 0})
		if not lags:
			raise ValueError("Provide at least one positive lag")
		self.lags: list[int] = lags
		self.max_lag: int = max(lags)
		
		# AR parameters
		default_dtype = torch.get_default_dtype()
		self.coeffs = nn.Parameter(torch.zeros(self.num_states, len(lags), dtype=default_dtype))  # (S,L)
		self.log_var = nn.Parameter(torch.zeros(self.num_states, dtype=default_dtype))            # (S,)
		
		# HMM transition parameters
		self.initial_logits = nn.Parameter(torch.zeros(self.num_states, dtype=default_dtype))                 # (S,)
		self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states, dtype=default_dtype))  # (S,S)
		
		self.normalize_time = normalize_time
		self.ridge = float(ridge)
		self.ignore_prefix = bool(ignore_prefix)
		self.drop_prefix_from_normalization = bool(drop_prefix_from_normalization)
		
		# Constant buffer for numerical expressions
		self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=default_dtype))
		
		self.__initialize_weights()
		self.to(self.device)

	# ---------------------------- Input handling ----------------------------
	def _standardize_input(self, x: Tensor) -> Tensor:  # type: ignore[override]
		"""Accept (B,T,1), (T,), (T,1), (B,T) -> (B,T,1)."""
		if x.dim() == 3:
			if x.shape[2] == 1:
				return x
			raise ValueError("HMMAR expects last dim==1 for univariate data")
		if x.dim() == 2:
			# (B,T) -> (B,T,1) OR (T,1) -> (1,T,1)
			if x.shape[1] == 1:  # (T,1) ambiguous; treat as (T,1)
				return x.unsqueeze(0) if x.shape[0] != 1 else x.unsqueeze(0)
			return x.unsqueeze(-1)
		if x.dim() == 1:
			return x.unsqueeze(0).unsqueeze(-1)  # (1,T,1)
		raise ValueError("Unsupported input rank for HMMAR (expected 1,2,3 dims)")

	# ----------------------- Emission log probability -----------------------
	def emission_log_prob(self, x: Tensor) -> Tensor:  # type: ignore[override]
		if x.dim() != 3 or x.shape[2] != 1:
			raise ValueError("Expected standardized shape (B,T,1)")
		B, T, _ = x.shape
		S = self.num_states
		L = len(self.lags)
		if T <= self.max_lag:
			# Not enough context for any full lag row -> return zeros (neutral)
			return x.new_zeros(B, T, S)
		# Build lag stack: (B,T,L) with zeros for unavailable history
		lag_stack = x.new_zeros(B, T, L, dtype=x.dtype)
		x_flat = x[:, :, 0]  # (B,T)
		for j, lag in enumerate(self.lags):
			if lag < T:
				lag_stack[:, lag:, j] = x_flat[:, :-lag]
		# Prediction: (B,T,S) = (B,T,L) @ (L,S)
		# Ensure dtype compatibility
		coeffs = self.coeffs.to(dtype=lag_stack.dtype)
		pred = torch.einsum("btl,sl->bts", lag_stack, coeffs)  # (B,T,S)
		resid = x_flat.unsqueeze(-1) - pred  # (B,T,S)
		log_var = self.log_var.view(1, 1, S).to(dtype=x.dtype)
		inv_var = torch.exp(-log_var)
		log_2pi = self._log_2pi.to(dtype=x.dtype)
		lp = -0.5 * (resid.pow(2) * inv_var + log_var + log_2pi)  # (B,T,S)
		if self.ignore_prefix:
			if self.max_lag > 0:
				lp[:, : self.max_lag, :] = 0.0
		return lp

	# ------------------------------ Forward pass ----------------------------
	def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
		x_std = self._standardize_input(x.to(self.device))  # (B,T,1)
		log_pi = torch.log_softmax(self.initial_logits, dim=-1)
		log_A = torch.log_softmax(self.transition_logits, dim=-1)
		log_emiss = self.emission_log_prob(x_std)  # (B,T,S)
		logp = self._forward_algorithm(log_emiss, log_pi, log_A)  # (B,)
		if self.normalize_time:
			denom = x_std.shape[1]
			if self.ignore_prefix and self.drop_prefix_from_normalization:
				denom = max(1, denom - self.max_lag)
			logp = logp / denom
		return logp

	def _forward_algorithm(self, log_emiss: Tensor, log_pi: Tensor, log_A: Tensor) -> Tensor:
		"""Run forward algorithm.

		Parameters
		----------
		log_emiss : (B,T,S)
		log_pi : (S,)
		log_A : (S,S)
		Returns
		-------
		log_likelihood : (B,)
		"""
		B, T, S = log_emiss.shape
		# alpha_0
		alpha = log_pi.unsqueeze(0) + log_emiss[:, 0, :]  # (B,S)
		for t in range(1, T):
			# (B,S,S): previous alpha_j + log_A_{j->i}
			prev = alpha.unsqueeze(2) + log_A.unsqueeze(0)
			alpha = log_emiss[:, t, :] + torch.logsumexp(prev, dim=1)
		return torch.logsumexp(alpha, dim=1)  # (B,)

	# --------------------------- Regularization -----------------------------
	def regularization_loss(self) -> Tensor:
		if self.ridge <= 0:
			return torch.zeros((), device=self.coeffs.device)
		return self.ridge * self.coeffs.pow(2).sum()

	# --------------------------- Abstract method implementations -------------
	@torch.no_grad()
	def predict(self, x: Tensor) -> Tensor:
		"""Viterbi decoding for most likely state sequence."""
		return self._decode_viterbi(x)

	@torch.no_grad()
	def _decode_viterbi(self, x: Tensor) -> Tensor:
		"""Most likely state sequence (Viterbi path).

		Expects input of shape (B,T,1); returns (B,T) int64.
		"""
		x = self._standardize_input(x.to(self.device))
		B, T, _ = x.shape
		log_pi = torch.log_softmax(self.initial_logits, dim=-1)
		log_A = torch.log_softmax(self.transition_logits, dim=-1)
		log_emiss = self.emission_log_prob(x)  # (B,T,S)
		
		backptr = x.new_zeros((B, T, self.num_states), dtype=torch.long)
		delta = log_pi.unsqueeze(0) + log_emiss[:, 0, :]  # (B,S)
		
		for t in range(1, T):
			scores = delta.unsqueeze(2) + log_A.unsqueeze(0)  # (B,S,S)
			delta, idx = torch.max(scores, dim=1)            # (B,S)
			delta = delta + log_emiss[:, t, :]
			backptr[:, t, :] = idx
		
		last = torch.argmax(delta, dim=1)  # (B,)
		path = x.new_zeros((B, T), dtype=torch.long)
		path[:, -1] = last
		
		for t in range(T - 2, -1, -1):
			path[:, t] = backptr[torch.arange(B), t + 1, path[:, t + 1]]
		
		return path

	def prepare_for_training(self):
		self.train()

	def prepare_for_inference(self):
		self.eval()

	def reset(self):
		self.__initialize_weights()

	def __str__(self) -> str:
		return (
			f"HMMAR(states={self.num_states}, lags={self.lags}, max_lag={self.max_lag}, "
			f"normalize_time={self.normalize_time}, ignore_prefix={self.ignore_prefix}, "
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge})"
		)

	# ------------------------------ Warm start ------------------------------
	@torch.no_grad()
	def __initialize_weights(self) -> None:
		"""Initialize AR-HMM parameters."""
		# Initialize transition parameters uniformly
		self.initial_logits.zero_()
		self.transition_logits.zero_()
		
		# Initialize AR coefficients to small random values
		torch.nn.init.normal_(self.coeffs, mean=0.0, std=0.1)
		
		# Initialize log variance to reasonable values (log(1.0) = 0)
		self.log_var.zero_()

	@torch.no_grad()
	def warm_start_from_paths(
		self,
		sequences: list[Tensor],
		paths: list[Tensor],
		min_samples_per_state: int = 16,
		ridge_ls: float = 1e-3,
	) -> None:
		"""Initialise AR parameters via per‑state ridge least squares.

		For each state k we form design rows using only timesteps where
		(z_t == k) AND t >= max_lag so all lagged samples available.
		States with insufficient rows remain at initial zeros.
		"""
		if len(sequences) != len(paths):
			raise ValueError("sequences and paths must have same length")
		S = self.num_states
		Ls = self.lags
		mL = self.max_lag
		for k in range(S):
			design_rows: list[Tensor] = []
			targets: list[Tensor] = []
			for x, z in zip(sequences, paths):
				if x.dim() == 2 and x.shape[1] == 1:
					x = x[:, 0]
				if x.dim() != 1 or z.dim() != 1:
					raise ValueError("Each sequence/path must be 1D or (T,1)")
				if x.numel() != z.numel():
					raise ValueError("Mismatched sequence/path lengths")
				T = x.numel()
				if T <= mL:
					continue
				for t in range(mL, T):
					if z[t] != k:
						continue
					design_rows.append(torch.stack([x[t - l] for l in Ls]))
					targets.append(x[t])
			if not design_rows:
				continue
			X = torch.stack(design_rows)  # (N,L)
			y = torch.stack(targets)      # (N,)
			if X.size(0) < min_samples_per_state:
				continue
			XtX = X.T @ X
			if ridge_ls > 0:
				XtX = XtX + ridge_ls * torch.eye(X.size(1), device=X.device, dtype=X.dtype)
			Xty = X.T @ y
			try:
				a = torch.linalg.solve(XtX, Xty)
			except RuntimeError:
				continue
			resid = y - (X @ a)
			var = resid.pow(2).mean().clamp_min(1e-12)
			self.coeffs.data[k].copy_(a.to(self.coeffs.dtype))
			self.log_var.data[k] = var.log()

	# ----------------------------- Representation ---------------------------
	def extra_repr(self) -> str:  # type: ignore[override]
		return (
			f"states={self.num_states}, lags={self.lags}, max_lag={self.max_lag}, "
			f"normalize_time={self.normalize_time}, ignore_prefix={self.ignore_prefix}, "
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge}"
		)

__all__ = ["HMMAR", "exponential_lags", "fit_ar_ls"]

