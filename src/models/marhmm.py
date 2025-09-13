from __future__ import annotations

from typing import Sequence
import math
import torch
from torch import Tensor
import torch.nn as nn

from .base_model import BaseModel
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.helpers.exponential_lags import exponential_lags
from src.helpers.fit_ar_ls import fit_ar_ls
from src.initializations.random_uniform import init_random_uniform

class MARHMM(BaseModel):
	"""Multivariate Autoregressive HMM with state-specific AR(p) emissions."""

	def __init__(self,
			 	 data_loader: DataLoader,
				 config: GlobalConfig,
				 device: torch.device) -> None:
		super().__init__(data_loader, config, device)
		self.__initialize_parameters()
		self.__initialize_weights()
		self.to(self.device)

	def __initialize_parameters(self) -> None:
		"""Allocate parameters & read config (aligned with HMM style)."""
		# Seed
		self.seed = self.global_config.seed
		torch.manual_seed(int(self.seed))
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(int(self.seed))
		# Dimensions
		self.obs_dim = self.num_features
		# Config
		params = self.global_config.model.params
		lags = params.get("lags", [1, 2, 4, 8])
		lags = sorted({int(l) for l in lags if l > 0})
		if not lags:
			raise ValueError("Provide at least one positive lag")
		self.lags: list[int] = lags
		self.max_lag: int = max(lags)
		self.normalize_time = bool(params.get("normalize_time", True))
		self.ridge = float(params.get("ridge", 0.0))  # L2 coeff penalty
		self.var_reg = float(params.get("var_reg", 0.0))  # variance stabiliser
		# Optional sticky transition prior
		self.sticky_coef = float(params.get("sticky_coef", 0.0))
		self.sticky_kappa = float(params.get("sticky_kappa", 0.9))
		self.ignore_prefix = bool(params.get("ignore_prefix", True))
		self.drop_prefix_from_normalization = bool(params.get("drop_prefix_from_normalization", True))
		# Parameters
		default_dtype = torch.get_default_dtype()
		self.coeffs = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, self.obs_dim * len(lags), dtype=default_dtype))
		# State-specific intercept for each observed dimension
		self.bias = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, dtype=default_dtype))
		self.log_var = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, dtype=default_dtype))
		self.initial_logits = nn.Parameter(torch.zeros(self.num_states, dtype=default_dtype))
		self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states, dtype=default_dtype))
		self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=default_dtype))

	def forward(self, x: Tensor) -> Tensor:  
		x_std = self.__validate_input(x)
		log_pi = torch.log_softmax(self.initial_logits, dim=-1)
		log_A = torch.log_softmax(self.transition_logits, dim=-1)
		
		# Add small random perturbation to avoid completely uniform distributions
		if self.training:
			eps = 1e-4
			log_pi = log_pi + torch.randn_like(log_pi) * eps
			log_A = log_A + torch.randn_like(log_A) * eps
		
		log_emiss = self.__emission_log_prob(x_std)
		logp = self.__forward_algorithm(log_emiss, log_pi, log_A)
		return self.negative_log_likelihood(x_std, logp)

	def negative_log_likelihood(self, x: Tensor, logp: Tensor) -> Tensor:
		"""Return a scalar NLL suitable for minimization by gradient descent.

		- Normalizes by effective time steps (optionally excluding prefix)
		  and by feature dimension to keep magnitudes comparable across configs.
		- Returns the mean across the batch.
		"""
		T = x.shape[1]
		F = x.shape[2]
		if self.normalize_time:
			denom_t = T
			if self.ignore_prefix and self.drop_prefix_from_normalization:
				denom_t = max(1, T - self.max_lag)
			logp = logp / denom_t
		# Also normalize by feature dimension as in the plain HMM for stability
		logp = logp / max(1, F)
		nll = -logp
		return nll.mean()

	def __emission_log_prob(self, x: Tensor) -> Tensor:  
		B, T, D = x.shape
		S = self.num_states
		L = len(self.lags)
		if T <= self.max_lag:
			return x.new_zeros(B, T, S)
		
		# Build lagged features
		lag_stack = x.new_zeros(B, T, D * L, dtype=x.dtype)
		for j, lag in enumerate(self.lags):
			if lag < T:
				start_idx = j * D
				end_idx = (j + 1) * D
				lag_stack[:, lag:, start_idx:end_idx] = x[:, :-lag, :]
		
		# AR predictions for each state
		coeffs = self.coeffs.transpose(-2, -1).to(dtype=lag_stack.dtype)
		pred = torch.einsum("btf,sfd->btsd", lag_stack, coeffs)
		# Add state-specific intercept term
		pred = pred + self.bias.view(1, 1, S, D)
		resid = x.unsqueeze(2) - pred
		
		# Emission probabilities with better numerical stability
		log_var = self.log_var.view(1, 1, S, D).to(dtype=x.dtype)
		# Clamp log_var to prevent extreme values
		log_var = torch.clamp(log_var, min=-10, max=10)
		
		# Use more stable computation
		log_2pi = self._log_2pi.to(dtype=x.dtype)
		lp = -0.5 * (resid.pow(2) * torch.exp(-log_var) + log_var + log_2pi)
		lp = lp.sum(dim=-1)
		
		# More conservative clamping to prevent numerical issues
		lp = torch.clamp(lp, min=-100, max=10)
		
		if self.ignore_prefix and self.max_lag > 0:
			lp[:, : self.max_lag, :] = 0.0
		return lp

	def __forward_algorithm(self, log_emiss: Tensor, log_pi: Tensor, log_A: Tensor) -> Tensor:
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

	def regularization_loss(self) -> Tensor:
		"""Composite regularization: ridge on coeffs + variance stabilisation.

		Variance term (optional via var_reg) discourages extremely small or large
		variances which can cause state collapse or numerical issues.
		"""
		reg = torch.zeros((), device=self.coeffs.device)
		# L2 on AR coefficients discourages unbounded growth that can lead to near-deterministic emissions
		if self.ridge > 0:
			reg = reg + self.ridge * self.coeffs.pow(2).sum()
		# Variance regularization to avoid collapse to extremely small or huge variances
		var_reg = float(getattr(self, "var_reg", 0.0))
		if var_reg > 0:
			var = torch.exp(self.log_var)  # (S,D)
			min_var = 1e-3
			max_var = 1e3
			small_pen = torch.clamp(min_var - var, min=0).div(min_var).pow(2)
			large_pen = torch.clamp(var - max_var, min=0).div(max_var).pow(2)
			reg = reg + var_reg * (small_pen.sum() + large_pen.sum())
		else:
			# Light penalty against too-small variances by default
			var = torch.exp(self.log_var)
			reg = reg + 1e-3 * torch.clamp(1e-4 - var, min=0).pow(2).sum()
		# Sticky transitions: encourage self-transitions via KL(A || A_prior)
		sticky_coef = float(getattr(self, "sticky_coef", 0.0))
		if sticky_coef > 0:
			A = torch.softmax(self.transition_logits, dim=-1)
			S = A.shape[-1]
			kappa = float(self.sticky_kappa)
			off = (1.0 - kappa) / max(1, S - 1)
			A_prior = torch.full_like(A, off)
			idx = torch.arange(S, device=A.device)
			A_prior[idx, idx] = kappa
			kl = A * (torch.log(torch.clamp(A, min=1e-12)) - torch.log(torch.clamp(A_prior, min=1e-12)))
			reg = reg + sticky_coef * kl.sum()
		return reg

	@torch.no_grad()
	def predict(self, x: Tensor) -> Tensor:
		"""Viterbi decoding for most likely state sequence."""
		return self.__decode_viterbi(x)

	@torch.no_grad()
	def __decode_viterbi(self, x: Tensor) -> Tensor:
		"""Most likely state sequence (Viterbi path).

		Expects input of shape (B,T,D); returns (B,T) int64.
		"""
		x = self.__validate_input(x)
		B, T, _ = x.shape
		log_pi = torch.log_softmax(self.initial_logits, dim=-1)
		log_A = torch.log_softmax(self.transition_logits, dim=-1)
		log_emiss = self.__emission_log_prob(x)  # (B,T,S)
		
		backptr = x.new_zeros((B, T, self.num_states), dtype=torch.long)
		delta = log_pi.unsqueeze(0) + log_emiss[:, 0, :]  # (B,S)
		
		for t in range(1, T):
			scores = delta.unsqueeze(2) + log_A.unsqueeze(0)
			delta, idx = torch.max(scores, dim=1)
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

	@torch.no_grad()
	def __initialize_weights(self, coeff_std: float = 0.03, jitter_std: float = 0.05, var_init: float = 1.0) -> None:
		"""Parameter initialization with support for different strategies."""
		strategy = getattr(self.global_config.model, 'init_strategy', 'default').lower()
		
		if strategy == "random_uniform":
			init_random_uniform(self, coeff_std=coeff_std, jitter_std=jitter_std, var_init=var_init)
		else:
			raise ValueError(f"Unknown initialization strategy: {strategy}")

	def __validate_input(self, x: Tensor) -> Tensor:
		"""Strict validator: require (B,T,D) with D==obs_dim; no reshaping."""
		if x.dim() != 3:
			raise ValueError(f"Expected input rank 3 (B,T,D); got shape {tuple(x.shape)}")
		if x.shape[2] != self.obs_dim:
			raise ValueError(f"Feature dim mismatch: got {x.shape[2]}, expected {self.obs_dim}")
		return x.to(self.device) 

	def __str__(self) -> str:
		return (
			f"MARHMM(states={self.num_states}, obs_dim={self.obs_dim}, lags={self.lags}, max_lag={self.max_lag}, "
			f"normalize_time={self.normalize_time}, ignore_prefix={self.ignore_prefix}, "
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge}, "
			f"sticky_coef={self.sticky_coef}, sticky_kappa={self.sticky_kappa})"
		)

__all__ = ["MARHMM", "exponential_lags", "fit_ar_ls"]

