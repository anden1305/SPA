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

from .hmm import HMM


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


class HMMAR(HMM):
	"""Autoregressive HMM (univariate) with state‑specific AR(p) emissions.

	Parameters
	----------
	num_states : int
		Number of latent states.
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
		num_states: int,
		lags: Iterable[int],
		device: str | torch.device | None = None,
		normalize_time: bool = False,
		ridge: float = 0.0,
		ignore_prefix: bool = True,
		drop_prefix_from_normalization: bool = True,
	) -> None:
		super().__init__(
			num_states=num_states,
			obs_dim=1,                 # univariate for simplicity
			covariance_type="meanonly",  # emissions overridden
			device=device,
			normalize_time=normalize_time,
		)
		lags = sorted({int(l) for l in lags if l > 0})
		if not lags:
			raise ValueError("Provide at least one positive lag")
		self.lags: list[int] = lags
		self.max_lag: int = max(lags)
		self.coeffs = nn.Parameter(torch.zeros(self.num_states, len(lags)))  # (S,L)
		self.log_var = nn.Parameter(torch.zeros(self.num_states))            # (S,)
		self.ridge = float(ridge)
		self.ignore_prefix = bool(ignore_prefix)
		self.drop_prefix_from_normalization = bool(drop_prefix_from_normalization)

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
		lag_stack = x.new_zeros(B, T, L)
		x_flat = x[:, :, 0]  # (B,T)
		for j, lag in enumerate(self.lags):
			if lag < T:
				lag_stack[:, lag:, j] = x_flat[:, :-lag]
		# Prediction: (B,T,S) = (B,T,L) @ (L,S)
		pred = torch.einsum("btl,sl->bts", lag_stack, self.coeffs)  # (B,T,S)
		resid = x_flat.unsqueeze(-1) - pred  # (B,T,S)
		log_var = self.log_var.view(1, 1, S)
		inv_var = torch.exp(-log_var)
		lp = -0.5 * (resid.pow(2) * inv_var + log_var + math.log(2 * math.pi))  # (B,T,S)
		if self.ignore_prefix:
			if self.max_lag > 0:
				lp[:, : self.max_lag, :] = 0.0
		return lp

	# ------------------------------ Forward pass ----------------------------
	def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
		x_std = self._standardize_input(x.to(self.device))  # (B,T,1)
		log_pi = self.log_initial()
		log_A = self.log_transition()
		log_emiss = self.emission_log_prob(x_std)  # (B,T,S)
		logp = self._forward_algorithm(log_emiss, log_pi, log_A)  # (B,)
		if self.normalize_time:
			denom = x_std.shape[1]
			if self.ignore_prefix and self.drop_prefix_from_normalization:
				denom = max(1, denom - self.max_lag)
			logp = logp / denom
		return logp

	# --------------------------- Regularization -----------------------------
	def regularization_loss(self) -> Tensor:
		if self.ridge <= 0:
			return torch.zeros((), device=self.coeffs.device)
		return self.ridge * self.coeffs.pow(2).sum()

	# ------------------------------ Warm start ------------------------------
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

