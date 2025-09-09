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
		self.normalize_time = bool(params.get("normalize_time", False))
		self.ridge = float(params.get("ridge", 0.0))  # L2 coeff penalty
		self.var_reg = float(params.get("var_reg", 0.0))  # variance stabiliser
		self.ignore_prefix = bool(params.get("ignore_prefix", True))
		self.drop_prefix_from_normalization = bool(params.get("drop_prefix_from_normalization", True))
		# Parameters
		default_dtype = torch.get_default_dtype()
		self.coeffs = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, self.obs_dim * len(lags), dtype=default_dtype))
		self.log_var = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, dtype=default_dtype))
		self.initial_logits = nn.Parameter(torch.zeros(self.num_states, dtype=default_dtype))
		self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states, dtype=default_dtype))
		self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=default_dtype))

	def forward(self, x: Tensor) -> Tensor:  
		x_std = self.__validate_input(x)
		log_pi = torch.log_softmax(self.initial_logits, dim=-1)
		log_A = torch.log_softmax(self.transition_logits, dim=-1)
		log_emiss = self.__emission_log_prob(x_std)
		logp = self.__forward_algorithm(log_emiss, log_pi, log_A)
		if self.normalize_time:
			denom = x_std.shape[1]
			if self.ignore_prefix and self.drop_prefix_from_normalization:
				denom = max(1, denom - self.max_lag)
			logp = logp / denom
		return logp

	def __emission_log_prob(self, x: Tensor) -> Tensor:  
		B, T, D = x.shape
		S = self.num_states
		L = len(self.lags)
		if T <= self.max_lag:
			return x.new_zeros(B, T, S)
		lag_stack = x.new_zeros(B, T, D * L, dtype=x.dtype)
		for j, lag in enumerate(self.lags):
			if lag < T:
				start_idx = j * D
				end_idx = (j + 1) * D
				lag_stack[:, lag:, start_idx:end_idx] = x[:, :-lag, :]
		coeffs = self.coeffs.transpose(-2, -1).to(dtype=lag_stack.dtype)
		pred = torch.einsum("btf,sfd->btsd", lag_stack, coeffs)
		resid = x.unsqueeze(2) - pred
		log_var = self.log_var.view(1, 1, S, D).to(dtype=x.dtype)
		inv_var = torch.exp(-log_var)
		log_2pi = self._log_2pi.to(dtype=x.dtype)
		inv_var = torch.clamp(inv_var, min=1e-8, max=1e8)
		log_var = torch.clamp(log_var, min=-15, max=15)
		lp = -0.5 * (resid.pow(2) * inv_var + log_var + log_2pi)
		lp = lp.sum(dim=-1)
		lp = torch.clamp(lp, min=-200, max=50)
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
		if self.ridge > 0:
			reg = reg + self.ridge * self.coeffs.pow(2).sum()
		if getattr(self, "var_reg", 0.0) > 0:
			var = torch.exp(self.log_var)  # (S,D)
			min_var = 1e-6
			max_var = 1e2
			small_pen = torch.clamp(min_var - var, min=0).div(min_var).pow(2)
			large_pen = torch.clamp(var - max_var, min=0).div(max_var).pow(2)
			reg = reg + self.var_reg * (small_pen.sum() + large_pen.sum())
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
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge})"
		)

	@torch.no_grad()
	def __initialize_weights(self) -> None:
		"""Parameter initialisation with slight noise & self-transition bias."""
		with torch.no_grad():
			# Small noise so gradients flow (break symmetry)
			self.initial_logits.data.normal_(mean=0.0, std=1e-4)
			self.transition_logits.data.normal_(mean=0.0, std=1e-4)
			# Encourage persistence
			bias = 0.05
			self.transition_logits.data += torch.eye(self.num_states, device=self.transition_logits.device) * bias
			scale = 0.1 / math.sqrt(self.obs_dim)
			torch.nn.init.normal_(self.coeffs, mean=0.0, std=scale)
			self.log_var.fill_(0.0)

	def extra_repr(self) -> str:  
		return (
			f"states={self.num_states}, obs_dim={self.obs_dim}, lags={self.lags}, max_lag={self.max_lag}, "
			f"normalize_time={self.normalize_time}, ignore_prefix={self.ignore_prefix}, "
			f"drop_prefix_norm={self.drop_prefix_from_normalization}, ridge={self.ridge}"
		)

__all__ = ["MARHMM", "exponential_lags", "fit_ar_ls"]

