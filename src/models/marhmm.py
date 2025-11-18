from __future__ import annotations

from typing import Sequence
import math
import torch
from torch import Tensor
import torch.nn as nn

from src.data.data_loader_collection import DataLoaderCollection

from .base_model import BaseModel
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.helpers.exponential_lags import exponential_lags
from src.helpers.fit_ar_ls import fit_ar_ls
from src.initializations.random_uniform import init_random_uniform
from src.initializations.random_dirichlet import init_random_dirichlet
from src.initializations.random_separated import init_random_separated
from src.initializations.kmeans import init_kmeans
from src.initializations.kmeans_pca import init_kmeans_pca
from src.initializations.apply_noise import apply_noise_and_bias

class MARHMM(BaseModel):
	"""Multivariate Autoregressive HMM with state-specific AR(p) emissions."""
	
	def __init__(self,
			 	 data_loader: DataLoaderCollection,
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
		# Covariance type
		self.covariance_type = self.global_config.model.covariance_type
		self.jitter = float(1e-5)
		# Config
		params = self.global_config.model.params
		lags = params.get("lags", [1, 2, 4, 8])
		lags = sorted({int(l) for l in lags if l > 0})
		if not lags:
			raise ValueError("Provide at least one positive lag")
		self.lags: list[int] = lags
		self.max_lag: int = max(lags)
		self.ridge = float(params.get("ridge", 0.0))  # L2 coeff penalty
		self.var_reg = float(params.get("var_reg", 0.0))  # variance stabiliser
		# Optional sticky transition prior
		self.sticky_coef = float(params.get("sticky_coef", 0.0))
		self.sticky_kappa = float(params.get("sticky_kappa", 0.9))
		# Parameters
		default_dtype = torch.get_default_dtype()
		self.coeffs = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, self.obs_dim * len(lags), dtype=default_dtype))
		# State-specific intercept for each observed dimension
		self.bias = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, dtype=default_dtype))
		# Covariance parameters
		if self.covariance_type == "diag":
			self.log_var = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, dtype=default_dtype))
		elif self.covariance_type == "full":
			# Unconstrained lower-triangular raw parameters for Cholesky factors L (S,D,D)
			# Diagonals passed through softplus for positive constraint
			self.emission_cholesky_raw = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, self.obs_dim, dtype=default_dtype))
		else:
			raise ValueError(f"Unsupported covariance_type for MARHMM: {self.covariance_type}. MARHMM supports 'diag' or 'full'.")
		self.initial_logits = nn.Parameter(torch.zeros(self.num_states, dtype=default_dtype))
		self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states, dtype=default_dtype))
		self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=default_dtype))

	def forward(self, x: Tensor) -> Tensor:  
		x_std = self.__validate_input(x)
		log_pi = torch.log_softmax(self.initial_logits, dim=-1)
		log_A = torch.log_softmax(self.transition_logits, dim=-1)
		
		log_emiss = self.__emission_log_prob(x_std)
		logp = self.__forward_algorithm(log_emiss, log_pi, log_A)
		return self.negative_log_likelihood(x_std, logp)

	def negative_log_likelihood(self, x: Tensor, logp: Tensor) -> Tensor:
		"""Return a scalar NLL suitable for minimization by gradient descent.

		- Normalizes by effective time steps 
		  and by feature dimension to keep magnitudes comparable across configs.
		- Returns the mean across the batch.
		"""
		B = x.shape[0]
		T = x.shape[1]
		D = x.shape[2]
		logp = logp / B
		logp = logp / T
		logp = logp / D
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
		
		# Compute log probabilities based on covariance type
		if self.covariance_type == "diag":
			# Diagonal covariance case
			log_var = self.log_var.view(1, 1, S, D).to(dtype=x.dtype)
			# Clamp log_var to prevent extreme values
			log_var = torch.clamp(log_var, min=-10, max=10)
			
			# Use more stable computation
			log_2pi = self._log_2pi.to(dtype=x.dtype)
			lp = -0.5 * (resid.pow(2) * torch.exp(-log_var) + log_var + log_2pi)
			lp = lp.sum(dim=-1)
		else:  # full covariance
			# Full covariance with Cholesky factorization
			L_chol = self.__full_cov_cholesky()  # (S,D,D)
			# Precompute log det Σ_s = 2 * sum(log(diag(L_s)))
			log_det = 2 * torch.log(torch.diagonal(L_chol, dim1=1, dim2=2)).sum(-1)  # (S,)
			log_2pi = self._log_2pi.to(dtype=x.dtype)
			
			# Loop over states for clarity
			log_probs = []
			for s in range(S):
				Ls = L_chol[s]  # (D,D)
				# Flatten (B*T,D) for solve
				resid_flat = resid[:, :, s, :].reshape(B * T, D).T  # (D, B*T)
				# Solve L y = resid^T -> y
				y = torch.linalg.solve_triangular(Ls, resid_flat, upper=False)  # (D, B*T)
				m_dist2 = (y.pow(2).sum(0)).reshape(B, T)  # (B,T)
				lp_s = -0.5 * (m_dist2 + log_det[s] + D * log_2pi)  # (B,T)
				log_probs.append(lp_s.unsqueeze(-1))  # (B,T,1)
			lp = torch.cat(log_probs, dim=-1)  # (B,T,S)
		
		if self.max_lag > 0:
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

	def __full_cov_cholesky(self) -> Tensor:
		"""Return lower-triangular Cholesky factors L (S,D,D) with positive diag."""
		raw = self.emission_cholesky_raw
		# Mask upper triangle
		tril_mask = torch.tril(torch.ones_like(raw)).bool()
		L = torch.zeros_like(raw)
		L[tril_mask] = raw[tril_mask]
		# Stabilise diagonals: softplus + jitter
		diag_idx = torch.arange(self.obs_dim, device=L.device)
		diag = L[:, diag_idx, diag_idx]
		L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(diag) + self.jitter
		return L

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
			if self.covariance_type == "diag":
				var = torch.exp(self.log_var)  # (S,D)
				min_var = 1e-3
				max_var = 1e3
				small_pen = torch.clamp(min_var - var, min=0).div(min_var).pow(2)
				large_pen = torch.clamp(var - max_var, min=0).div(max_var).pow(2)
				reg = reg + var_reg * (small_pen.sum() + large_pen.sum())
			elif self.covariance_type == "full":
				# Regularize Cholesky diagonal entries
				L = self.__full_cov_cholesky()
				diag = torch.diagonal(L, dim1=1, dim2=2)
				min_diag = 1e-3
				max_diag = 1e3
				small_pen = torch.clamp(min_diag - diag, min=0).div(min_diag).pow(2)
				large_pen = torch.clamp(diag - max_diag, min=0).div(max_diag).pow(2)
				reg = reg + var_reg * (small_pen.sum() + large_pen.sum())
		else:
			# Light penalty against too-small variances by default
			if self.covariance_type == "diag":
				var = torch.exp(self.log_var)
				reg = reg + 1e-3 * torch.clamp(1e-4 - var, min=0).pow(2).sum()
			elif self.covariance_type == "full":
				L = self.__full_cov_cholesky()
				diag = torch.diagonal(L, dim1=1, dim2=2)
				reg = reg + 1e-3 * torch.clamp(1e-4 - diag, min=0).pow(2).sum()
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
	def __initialize_weights(
		self,
		coeff_std: float = 0.05,
		jitter_std: float = 0.05,
		var_init: float = 0.05,
		kmeans_iters: int = 150,
		estimate_transitions: bool = True,
		mean_std: float = 0.01,
		cov_noise_std: float = 0.01,
		init_logits_std: float = 0.01,
		self_transition_bias: float = 0.01,
		jitter_std_separated: float = 0.05,
		ar_noise_std_init: float = 0.0001,
		spread: float = 0.05,
	) -> None:
		"""Parameter initialization with support for different strategies.

		Default hyperparameters act as a stable baseline:
		- coeff_std=1.0: unit-scale AR coefficient magnitude similar to random_uniform success case.
		- jitter_std=0.0: avoid extra noise; training will shape dynamics.
		- mean_std=1.0: same scale for random_dirichlet means for parity with coeff_std.
		- cov_noise_std=0.02: light emission variance/logvar perturbation when noisy init is requested.
		- init_logits_std=0.1: mild spread to break symmetry without inducing extreme priors.
		- self_transition_bias=0.05: slight persistence encouragement; small enough to not lock states.
		- jitter_std_separated=0.05: small jitter to avoid exact overlaps in random_separated.
		- spread=2.0: separation between states for random_separated while keeping features in reasonable range.
		Adjust per dataset scale if features are pre-normalized or heavily scaled.
		"""

		if self.global_config.model.features:
			coeff_std = 0.4
			var_init = 0.2
			jitter_std_separated = 0.5
			spread = 0.5

		strategy = getattr(self.global_config.model, 'init_strategy', 'default').lower()
		init_noisy = getattr(self.global_config.model, 'init_noisy', False)
		
		# Handle aliasing for noisy versions
		if strategy == "kmeans_pca_noisy":
			strategy = "kmeans_pca"
			init_noisy = True

		data = None
		if strategy in {"kmeans", "kmeans_pca"}:
			data, _ = self.data_loader.get_all_data()
			data = self.__validate_input(data)

		if strategy == "random_uniform":
			init_random_uniform(self, coeff_std=coeff_std, jitter_std=jitter_std, var_init=var_init)
		elif strategy == "random_dirichlet":
			init_random_dirichlet(self, mean_std=mean_std, self_transition_bias=self_transition_bias)
		elif strategy == "random_separated":
			init_random_separated(self, spread=spread, jitter_std=jitter_std_separated)
		elif strategy == "kmeans":
			init_kmeans(self, data, kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
		elif strategy == "kmeans_pca":
			init_kmeans_pca(self, data, kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
		else:
			raise ValueError(f"Unknown initialization strategy: {strategy}")

		if init_noisy:
			apply_noise_and_bias(
				self,
				mean_std=mean_std,
				cov_noise_std=cov_noise_std,
				init_logits_std=init_logits_std,
				self_transition_bias=self_transition_bias,
				ar_noise_std=ar_noise_std_init,
			)

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
			f"ridge={self.ridge}, "
			f"sticky_coef={self.sticky_coef}, sticky_kappa={self.sticky_kappa})"
		)

__all__ = ["MARHMM", "exponential_lags", "fit_ar_ls"]

