"""Multivariate Autoregressive Hidden Markov Model (MAR‑HMM) with state‑specific 
multivariate Vector AutoRegressive (VAR) emissions.

Redesign (Phase 1) features:
		* Vectorised emission log-probability (no Python loop over lags per step).
		* Flexible input standardisation: accepts (B,T,D), (T,D), (T,), (T,1), (B,T) and upgrades
			to canonical (B,T,D). (Keeps behaviour isolated from base HMM.)
		* Explicit prefix masking for first ``max(lags)`` frames (optional). When
			``normalize_time=True`` we can normalise by *valid* frames instead of T
			via ``drop_prefix_from_normalization``.
		* Ridge regularisation helper via ``regularization_loss``.
		* Warm start via least squares from provided state paths (unchanged API,
			internal logic clarified / lightly refactored).

Scope / features:
		* Multivariate Vector AutoRegression (VAR) with diagonal covariance per state.
		* Supports arbitrary observation dimensions D.
		* No variable‑length padding mask yet (must pass equal length sequences).
		* Each state has a D x (D*L) coefficient matrix for VAR(p) modeling.

Shape conventions
-----------------
Input x (canonical): (B, T, D)
		Accepted shortcuts: (T,D) -> (1,T,D); (T,) -> (1,T,1) if D=1; (B,T) -> (B,T,1) if D=1
Output forward(x): (B,) log p(x) (optionally normalised)

Emission model
--------------
For state s with coefficient matrix A_s (D, D*L) over lags L_i in ``self.lags``:
		pred_{t,s} = A_s @ [x_{t-lag_1}; x_{t-lag_2}; ...; x_{t-lag_L}]
Residual r_{t,s} = x_t − pred_{t,s}; likelihood assumes N(0, Σ_s) with diagonal Σ_s
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


class MARHMM(BaseModel):
	"""Multivariate Autoregressive HMM with state‑specific AR(p) emissions.

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

	def __init__(self,
			  	 data_loader: DataLoader,
				 config: GlobalConfig,
				 device: torch.device) -> None:
		super().__init__(data_loader, config, device)
		
		# MARHMM supports multivariate data
		self.obs_dim = self.num_features
		
		# Read MARHMM-specific parameters from config.model.params
		params = config.model.params
		lags = params.get("lags", [1, 2, 4, 8])
		normalize_time = params.get("normalize_time", False)
		ridge = params.get("ridge", 0.0)
		ignore_prefix = params.get("ignore_prefix", True)
		drop_prefix_from_normalization = params.get("drop_prefix_from_normalization", True)
		
		lags = sorted({int(l) for l in lags if l > 0})
		if not lags:
			raise ValueError("Provide at least one positive lag")
		self.lags: list[int] = lags
		self.max_lag: int = max(lags)
		
		# Multivariate AR parameters: each state has a D x (D*L) coefficient matrix
		# where D is observation dimension and L is number of lags
		default_dtype = torch.get_default_dtype()
		self.coeffs = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, self.obs_dim * len(lags), dtype=default_dtype))  # (S,D,D*L)
		self.log_var = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, dtype=default_dtype))  # (S,D) - diagonal covariance
		
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
		"""Accept (B,T,D), (T,D), (T,), (T,1), (B,T) -> (B,T,D)."""
		if x.dim() == 3:
			return x  # Already (B,T,D)
		if x.dim() == 2:
			if x.shape[1] == self.obs_dim:
				return x.unsqueeze(0)  # (T,D) -> (1,T,D)
			elif x.shape[1] == 1 and self.obs_dim == 1:
				return x.unsqueeze(0)  # (T,1) -> (1,T,1)
			else:
				return x.unsqueeze(-1)  # (B,T) -> (B,T,1)
		if x.dim() == 1:
			if self.obs_dim == 1:
				return x.unsqueeze(0).unsqueeze(-1)  # (T,) -> (1,T,1)
			else:
				raise ValueError(f"Cannot convert 1D input to {self.obs_dim}D observations")
		raise ValueError("Unsupported input rank for MARHMM (expected 1,2,3 dims)")

	# ----------------------- Emission log probability -----------------------
	def emission_log_prob(self, x: Tensor) -> Tensor:  # type: ignore[override]
		if x.dim() != 3 or x.shape[2] != self.obs_dim:
			raise ValueError(f"Expected standardized shape (B,T,{self.obs_dim})")
		B, T, D = x.shape
		S = self.num_states
		L = len(self.lags)
		
		if T <= self.max_lag:
			# Not enough context for any full lag row -> return zeros (neutral)
			return x.new_zeros(B, T, S)
		
		# Build lag stack: (B,T,D*L) with zeros for unavailable history
		lag_stack = x.new_zeros(B, T, D * L, dtype=x.dtype)
		
		for j, lag in enumerate(self.lags):
			if lag < T:
				start_idx = j * D
				end_idx = (j + 1) * D
				lag_stack[:, lag:, start_idx:end_idx] = x[:, :-lag, :]  # Shift by lag
		
		# Multivariate prediction: (B,T,D) = (B,T,D*L) @ (D*L,D) for each state
		# We need to reshape coeffs from (S,D,D*L) to (S,D*L,D) for matrix multiplication
		coeffs = self.coeffs.transpose(-2, -1).to(dtype=lag_stack.dtype)  # (S,D*L,D)
		
		# Compute predictions for all states: (B,T,S,D)
		pred = torch.einsum("btf,sfd->btsd", lag_stack, coeffs)  # (B,T,S,D)
		
		# Residuals: (B,T,S,D)
		resid = x.unsqueeze(2) - pred  # (B,T,1,D) - (B,T,S,D) -> (B,T,S,D)
		
		# Log variances: (S,D) -> (1,1,S,D)
		log_var = self.log_var.view(1, 1, S, D).to(dtype=x.dtype)
		inv_var = torch.exp(-log_var)
		
		# Multivariate Gaussian log probability (assuming diagonal covariance)
		log_2pi = self._log_2pi.to(dtype=x.dtype)
		
		# Add numerical stability - clamp variances to avoid extreme values
		inv_var = torch.clamp(inv_var, min=1e-6, max=1e6)
		log_var = torch.clamp(log_var, min=-10, max=10)
		
		lp = -0.5 * (resid.pow(2) * inv_var + log_var + log_2pi)  # (B,T,S,D)
		lp = lp.sum(dim=-1)  # Sum over dimensions: (B,T,S)
		
		# Additional numerical stability for the final log probabilities
		lp = torch.clamp(lp, min=-100, max=50)
		
		if self.ignore_prefix:
			if self.max_lag > 0:
				lp[:, : self.max_lag, :] = 0.0
		return lp

	# ------------------------------ Forward pass ----------------------------
	def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
		x_std = self._standardize_input(x.to(self.device))  # (B,T,D)
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

		Expects input of shape (B,T,D); returns (B,T) int64.
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
			f"MARHMM(states={self.num_states}, obs_dim={self.obs_dim}, lags={self.lags}, max_lag={self.max_lag}, "
			f"normalize_time={self.normalize_time}, ignore_prefix={self.ignore_prefix}, "
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge})"
		)

	# ------------------------------ Warm start ------------------------------
	@torch.no_grad()
	def __initialize_weights(self) -> None:
		"""Initialize Multivariate AR-HMM parameters."""
		# Initialize transition parameters uniformly
		self.initial_logits.zero_()
		self.transition_logits.zero_()
		
		# Initialize AR coefficients to small random values
		# For multivariate case: (S, D, D*L) - scale by dimension to maintain stability
		scale = 0.1 / math.sqrt(self.obs_dim)
		torch.nn.init.normal_(self.coeffs, mean=0.0, std=scale)
		
		# Initialize log variance to reasonable values for high-dimensional data
		# For multivariate case: (S, D) - diagonal covariance
		# Start with log(1.0) = 0 but could be adjusted for high-dim data
		self.log_var.fill_(0.0)

	@torch.no_grad()
	def warm_start_from_paths(
		self,
		sequences: list[Tensor],
		paths: list[Tensor],
		min_samples_per_state: int = 16,
		ridge_ls: float = 1e-3,
	) -> None:
		"""Initialise Multivariate AR parameters via per‑state ridge least squares.

		For each state k we form design rows using only timesteps where
		(z_t == k) AND t >= max_lag so all lagged samples available.
		States with insufficient rows remain at initial zeros.
		"""
		if len(sequences) != len(paths):
			raise ValueError("sequences and paths must have same length")
		S = self.num_states
		D = self.obs_dim
		Ls = self.lags
		mL = self.max_lag
		
		for k in range(S):
			design_rows: list[Tensor] = []
			targets: list[Tensor] = []
			
			for x, z in zip(sequences, paths):
				# Standardize input to (T, D)
				if x.dim() == 3:
					if x.shape[0] == 1:
						x = x.squeeze(0)  # (1,T,D) -> (T,D)
				elif x.dim() == 2 and x.shape[1] == 1 and D > 1:
					raise ValueError("Cannot fit multivariate model on univariate data")
				elif x.dim() == 1 and D == 1:
					x = x.unsqueeze(-1)  # (T,) -> (T,1)
				
				if x.dim() != 2 or x.shape[1] != D:
					raise ValueError(f"Each sequence must have shape (T,{D})")
				if z.dim() != 1:
					raise ValueError("Each path must be 1D")
				if x.shape[0] != z.numel():
					raise ValueError("Mismatched sequence/path lengths")
				
				T = x.shape[0]
				if T <= mL:
					continue
				
				for t in range(mL, T):
					if z[t] != k:
						continue
					# Build lag features: concatenate [x_{t-lag1}, x_{t-lag2}, ...]
					lag_features = []
					for lag in Ls:
						lag_features.append(x[t - lag])  # (D,)
					design_row = torch.cat(lag_features)  # (D*L,)
					design_rows.append(design_row)
					targets.append(x[t])  # (D,)
			
			if not design_rows:
				continue
			
			X = torch.stack(design_rows)  # (N, D*L)
			Y = torch.stack(targets)      # (N, D)
			
			if X.size(0) < min_samples_per_state:
				continue
			
			# Solve for each output dimension separately
			for d in range(D):
				y = Y[:, d]  # (N,)
				XtX = X.T @ X  # (D*L, D*L)
				if ridge_ls > 0:
					XtX = XtX + ridge_ls * torch.eye(X.size(1), device=X.device, dtype=X.dtype)
				Xty = X.T @ y  # (D*L,)
				
				try:
					a = torch.linalg.solve(XtX, Xty)  # (D*L,)
				except RuntimeError:
					continue
				
				# Reshape coefficients for state k, dimension d
				self.coeffs.data[k, d, :].copy_(a.to(self.coeffs.dtype))
				
				# Compute residuals and variance
				resid = y - (X @ a)
				var = resid.pow(2).mean().clamp_min(1e-12)
				self.log_var.data[k, d] = var.log()

	# ----------------------------- Representation ---------------------------
	def extra_repr(self) -> str:  # type: ignore[override]
		return (
			f"states={self.num_states}, obs_dim={self.obs_dim}, lags={self.lags}, max_lag={self.max_lag}, "
			f"normalize_time={self.normalize_time}, ignore_prefix={self.ignore_prefix}, "
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge}"
		)

__all__ = ["MARHMM", "exponential_lags", "fit_ar_ls"]

