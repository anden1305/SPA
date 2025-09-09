"""Hidden Markov Model with Gaussian emissions (mean-only, diagonal, or full covariance).

- Minimal, readable implementation for sequence modeling.
- Supports batch input of shape (B, T, D): batch size, sequence length, feature dimension.
- Clean separation of log-probability computation and inference algorithms.
- Initialization via random or k-means strategies.
- Assumes all sequences have the same length.
"""
from __future__ import annotations

import math
from pyexpat import model
import torch
from torch import Tensor
import torch.nn as nn
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from .base_model import BaseModel
from src.initializations.random_uniform import init_random_uniform
from src.initializations.random_dirichlet import init_random_dirichlet
from src.initializations.random_separated import init_random_separated
from src.initializations.kmeans import init_kmeans
from src.initializations.kmeans_pca import init_kmeans_pca
from src.initializations.apply_noise import apply_noise_and_bias

class HMM(BaseModel):
    def __init__(self,
                 data_loader: DataLoader,
                 config: GlobalConfig,
                 device: torch.device) -> None:
        super().__init__(data_loader, config, device)
        
        self.__initialize_parameters()
        self.__initialize_weights()

    def __initialize_parameters(self) -> None:
        self.seed = self.global_config.seed
        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))
            
        self.covariance_type = "diag"
        self.jitter = float(1e-5)

        # Parameterization (logits for simplex params; mean/logvar for Gaussians)
        self.initial_logits = nn.Parameter(torch.zeros(self.num_states))                 # (S,)
        self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states))  # (S,S)
        self.emission_mean = nn.Parameter(torch.zeros(self.num_states, self.num_features))    # (S,D)
        if self.covariance_type == "diag":
            self.emission_logvar = nn.Parameter(torch.zeros(self.num_states, self.num_features))  # (S,D)
        elif self.covariance_type == "full":
            # Unconstrained lower‑triangular raw parameters for Cholesky factors L (S,D,D)
            # We store a full matrix then mask to lower tri; diagonals passed through softplus.
            self.emission_cholesky_raw = nn.Parameter(torch.zeros(self.num_states, self.num_features, self.num_features))
        else:  # meanonly -> no covariance parameters (implicit Identity)
            self.register_buffer("_identity_cov", torch.eye(self.num_features))
        # Constant buffer for numerical expressions (avoids repeated Python math calls)
        self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=torch.get_default_dtype()))
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Per-sequence log-likelihood.

        Returns (B,) log p(x). Expects input of shape (B,T,D).
        """
        x = self.__validate_input(x)
        log_pi = torch.log_softmax(self.initial_logits, dim=-1)
        log_A = torch.log_softmax(self.transition_logits, dim=-1)
        log_emiss = self.__emission_log_prob(x)
        logp = self.__forward_algorithm(log_emiss, log_pi, log_A)
        return logp
    
    def __emission_log_prob(self, x: Tensor) -> Tensor:
        """Return log p(x_t | z_t) for all states.

        Parameters
        ----------
        x : Tensor, shape (B,T,D)

        Returns
        -------
        Tensor, shape (B,T,S)
        """
        B, T, D = x.shape
        S = self.num_states
        # Broadcast diff: (B,T,1,D) - (S,D) -> (B,T,S,D)
        diff = x.unsqueeze(2) - self.emission_mean  # (B,T,S,D)

        if self.covariance_type == "diag":
            logvar = self.emission_logvar  # (S,D)
            inv_var = torch.exp(-logvar)  # positive
            log_prob = -0.5 * (diff.pow(2) * inv_var + logvar + self._log_2pi)
            return log_prob.sum(-1)  # (B,T,S)
        elif self.covariance_type == "meanonly":
            # Identity covariance -> log N(x | mu, I)
            quad = diff.pow(2).sum(-1)  # (B,T,S)
            const = D * self._log_2pi
            return -0.5 * (quad + const)
        else:  # full
            L = self.__full_cov_cholesky()  # (S,D,D)
            # Precompute log det Σ_s = 2 * sum(log(diag(L_s)))
            log_det = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)  # (S,)
            # We'll loop over states (S typically small) for clarity
            log_probs = []
            for s in range(S):
                Ls = L[s]  # (D,D)
                # Flatten (B*T,D) for solve
                d_flat = diff[:, :, s, :].reshape(B * T, D).T  # (D, B*T)
                # Solve L y = diff^T -> y
                y = torch.linalg.solve_triangular(Ls, d_flat, upper=False)  # (D, B*T)
                m_dist2 = (y.pow(2).sum(0)).reshape(B, T)  # (B,T)
                lp = -0.5 * (m_dist2 + log_det[s] + D * self._log_2pi)  # (B,T)
                log_probs.append(lp.unsqueeze(-1))  # (B,T,1)
            return torch.cat(log_probs, dim=-1)  # (B,T,S)

    def __full_cov_cholesky(self) -> Tensor:
        """Return lower‑triangular Cholesky factors L (S,D,D) with positive diag."""
        raw = self.emission_cholesky_raw
        # Mask upper triangle
        tril_mask = torch.tril(torch.ones_like(raw)).bool()
        L = torch.zeros_like(raw)
        L[tril_mask] = raw[tril_mask]
        # Stabilise diagonals: softplus + jitter
        diag_idx = torch.arange(self.num_features, device=L.device)
        diag = L[:, diag_idx, diag_idx]
        L[:, diag_idx, diag_idx] = torch.nn.functional.softplus(diag) + self.jitter
        return L

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

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Alias returning Viterbi path for compatibility with sklearn-like API."""
        return self.__decode_viterbi(x)

    @torch.no_grad()
    def __decode_viterbi(self, x: Tensor) -> Tensor:
        """Most likely state sequence (Viterbi path).

        Expects input of shape (B,T,D); returns (B,T) int64.
        """
        x = self.__validate_input(x)
        B, T, D = x.shape
        log_pi = torch.log_softmax(self.initial_logits, dim=-1)
        log_A = torch.log_softmax(self.transition_logits, dim=-1)
        log_emiss = self.__emission_log_prob(x)  # (B,T,S)
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

    def regularization_loss(self) -> torch.Tensor:
        """Regularize emission variances to avoid too small or too large values."""
        
        max_var_threshold = 10.0
        min_variance_threshold = 1
        factor = 10
        
        reg_loss = torch.zeros((), device=self.device)
        if self.covariance_type == "diag":
            variance = torch.exp(self.emission_logvar)
            min_var_penalty = torch.clamp(min_variance_threshold - variance, min=0.0).pow(2)
            max_var_penalty = torch.clamp(variance - max_var_threshold, min=0.0).pow(2)
            reg_loss = (min_var_penalty.sum() + max_var_penalty.sum()) * factor
        elif self.covariance_type == "full":
            L = self.__full_cov_cholesky()
            diag = torch.diagonal(L, dim1=1, dim2=2)
            min_diag_penalty = torch.clamp(min_variance_threshold - diag, min=0.0).pow(2)
            max_diag_penalty = torch.clamp(diag - max_var_threshold, min=0.0).pow(2)
            reg_loss = (min_diag_penalty.sum() + max_diag_penalty.sum()) * factor
        return reg_loss

    def __validate_input(self, x: Tensor) -> Tensor:
        """Validate that x has shape (B,T,D) with D == self.num_features.
        Returns the tensor moved to the model device.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of rank 3 (B,T,D); got shape {tuple(x.shape)}")
        if x.shape[2] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: got D={x.shape[2]}, expected {self.num_features} (obs_dim)."
            )
        return x.to(self.device) 

    def __str__(self):
        return f"HMM(num_states={self.num_states}, num_features={self.num_features})"

    def reset(self):
        self.__initialize_weights()

    @torch.no_grad()
    def __initialize_weights(
        self,
        kmeans_iters: int = 150,
        estimate_transitions: bool = True,
        mean_std: float = 0.05,
        cov_noise_std: float = 0.05,
        init_logits_std: float = 0.05,
        self_transition_bias: float = 0.05,
        spread: float = 0.05,
        jitter_std: float = 0.05,
    ) -> None:
        """Short dispatcher for HMM weight initialization.

        strategy in {random_uniform, random_dirichlet, random_separated, kmeans, kmeans_pca}.
        - random_uniform: fully random means, unit covariance, uniform pi/A.
        - random_dirichlet: fully random means, unit covariance, Dirichlet-sampled pi/A.
        - random_separated: separated means (approx. orthogonal), unit covariance, uniform pi/A.
        - kmeans: k-means for emissions (and optionally pi/A).
        - kmeans_pca: search small PCA subspaces (3D among top 4 PCs) for best k-means clustering (by Calinski-Harabasz score).
        Data is optional (not needed for random*).
        """
        strategy = self.global_config.model.init_strategy.lower()
        if strategy in {"kmeans", "kmeans_pca"}:
            data, _ = self.data_loader.get_all_data()
            data = self.__validate_input(data)

        if strategy == "random_uniform":
            init_random_uniform(self, coeff_std=mean_std, jitter_std=jitter_std)
        elif strategy == "random_dirichlet":
            init_random_dirichlet(self, mean_std=mean_std, alpha=1.0, self_transition_bias=self_transition_bias)
        elif strategy == "random_separated":
            init_random_separated(self, spread=spread, jitter_std=jitter_std)
        elif strategy == "kmeans":
            init_kmeans(self, data, kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
        elif strategy == "kmeans_pca":
            init_kmeans_pca(self, data, kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
        else:
            raise ValueError("strategy must be one of {'random','random_separated','random_uniform','random_dirichlet','kmeans','kmeans_noisy','kmeans_pca','sticky_em_warmstart'}")

        init_noisy = self.global_config.model.init_noisy
        if init_noisy:
            apply_noise_and_bias(self, mean_std=mean_std, cov_noise_std=cov_noise_std, init_logits_std=init_logits_std, self_transition_bias=self_transition_bias)
        self.clear_param_grads()
     
    def __validate_input(self, x: Tensor) -> Tensor:
        """Validate that x has shape (B,T,D) with D == self.num_features.
        Returns the tensor moved to the model device.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of rank 3 (B,T,D); got shape {tuple(x.shape)}")
        if x.shape[2] != self.num_features:
            raise ValueError(
                f"Feature dimension mismatch: got D={x.shape[2]}, expected {self.num_features} (obs_dim)."
            )
        return x.to(self.device) 

    def __str__(self):
        return f"HMM(num_states={self.num_states}, num_features={self.num_features})"
   
__all__ = ["HMM"]
