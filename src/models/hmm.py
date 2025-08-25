"""Hidden Markov Model with Gaussian emissions (mean-only, diagonal, or full covariance).

- Minimal, readable implementation for sequence modeling.
- Supports batch input of shape (B, T, D): batch size, sequence length, feature dimension.
- Clean separation of log-probability computation and inference algorithms.
- Initialization via random or k-means strategies.
- Assumes all sequences have the same length.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Sequence
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from .base_model import MLModel

class HMM(MLModel):
    def __init__(self,
                 data_loader: DataLoader,
                 config: GlobalConfig,
                 device: torch.device) -> None:
        super().__init__(data_loader, config, device)
        self.__initialize_parameters()
        self.__initialise_weights()

    def __initialize_parameters(self) -> None:
        self.seed = self.global_config.seed
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

    # ----------------------- Initialization helpers -----------------------
    @torch.no_grad()
    def __initialise_weights(
        self,
        kmeans_iters: int = 15,
        estimate_transitions: bool = True,
        mean_std: float = 1.5,
        cov_noise_std: float = 1.5,
        init_logits_std: float = 0.5,
        self_transition_bias: float = 0.5,
        spread: float = 2.0,
        jitter_std: float = 0.05,
    ) -> None:
        """Short dispatcher for HMM weight initialization.

        strategy in {random, kmeans, kmeans_noisy}.
        - random: simple separated means with unit variance, uniform pi/A.
        - kmeans: k-means for emissions (and optionally pi/A).
        - kmeans_noisy: run k-means, then add small random noise to avoid plateaus.
        Data is optional (not needed for random).
        """
        s = self.global_config.model.init_strategy.lower()
        if s == "random":
            self.__init_random(spread=spread, jitter_std=jitter_std)
        elif s == "kmeans":
            self.__init_kmeans(kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
            self.__clear_param_grads()
        elif s == "kmeans_noisy":
            self.__init_kmeans(kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
            self.__apply_noise_and_bias(
                mean_std=mean_std,
                cov_noise_std=cov_noise_std,
                init_logits_std=init_logits_std,
                self_transition_bias=self_transition_bias,
            )
            self.__clear_param_grads()
        else:
            raise ValueError("strategy must be one of {'random','kmeans','kmeans_noisy'}")

    def __init_random(
        self,
        spread: float = 2.0,
        jitter_std: float = 0.05,
    ) -> None:
        """Very simple random init with clearly separated means.

        Strategy:
        1. Place each state's mean along a different (cycled) feature dimension
           at evenly spaced positions in [-spread, spread]. Add small Gaussian
           jitter (jitter_std) to all dimensions to avoid exact symmetry.
        2. Set diagonal variances to 1 (logvar=0) or full cov to identity.
        3. Use uniform initial state and transition distributions (logits=0).
        This yields distinguishable emissions without complex heuristics.
        """
        S, D = self.num_states, self.num_features
        device = self.emission_mean.device
        # Construct S distinct direction vectors in R^D
        if D >= S:
            # Use QR for orthogonal directions (columns of Q)
            R = torch.randn(D, S, device=device)
            Q, _ = torch.linalg.qr(R, mode='reduced')  # (D,S)
            dirs = Q.T  # (S,D)
        else:
            # More states than dimensions: sample and normalise
            R = torch.randn(S, D, device=device)
            dirs = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)
        means = spread * dirs
        means = means + jitter_std * torch.randn_like(means)
        self.emission_mean.copy_(means)
        # 2. Covariance
        if self.covariance_type == "diag":
            self.emission_logvar.zero_()
        elif self.covariance_type == "full":
            raw = torch.zeros_like(self.emission_cholesky_raw)
            # diagonals: inverse softplus(1.0) ~ log(exp(1)-1)
            diag_raw_val = math.log(math.e - 1.0)
            for s in range(S):
                raw[s].diagonal().fill_(diag_raw_val)
            self.emission_cholesky_raw.copy_(raw)
        # meanonly: implicit identity
        # 3. Uniform pi/A -> logits already zero
        self.initial_logits.zero_()
        self.transition_logits.zero_()

        # Diagnostics: report separation to help debug single-state collapse
        with torch.no_grad():
            means = self.emission_mean
            # Pairwise distances
            pdist = torch.cdist(means, means, p=2)
            # Ignore diagonal for stats
            off_diag = pdist[~torch.eye(S, dtype=torch.bool, device=means.device)]
            mean_dist = off_diag.mean().item() if off_diag.numel() else 0.0
            min_dist = off_diag.min().item() if off_diag.numel() else 0.0
            max_dist = off_diag.max().item() if off_diag.numel() else 0.0
            feat_std = means.std(0)
            avg_feat_std = feat_std.mean().item()
            trans_row = torch.softmax(self.transition_logits, dim=-1)[0]
            print(f"[HMM random init] pairwise_dist min/mean/max = {min_dist:.3f}/{mean_dist:.3f}/{max_dist:.3f}; avg_feature_std={avg_feat_std:.3f}")
            print(f"[HMM random init] first transition row (uniform expected): {trans_row.cpu().numpy()}")

        self.__clear_param_grads()
        
    @torch.no_grad()
    def __init_kmeans(self, kmeans_iters: int, estimate_transitions: bool) -> None:
        data, _ = self.data_loader.get_all_data()
        data = self.__validate_input(data)
        B, T, D = data.shape
        S = self.num_states
        if D != self.num_features:
            raise ValueError("obs_dim mismatch")
        device = self.device
        flat = data.reshape(-1, D).to(device)

        perm = torch.randperm(flat.size(0), device=device)
        means = flat[perm[:S]].clone()
        iters = max(0, int(kmeans_iters))
        for _ in range(iters):
            assign = torch.cdist(flat, means).argmin(-1)  # (N,)
            counts = torch.bincount(assign, minlength=S).clamp_min(1)
            new_means = torch.zeros_like(means)
            new_means.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), flat)
            means = new_means / counts.unsqueeze(1)
        if iters == 0:
            assign = torch.cdist(flat, means).argmin(-1)

        resid = flat - means[assign]
        var = resid.pow(2).mean(0).clamp_min(1e-6)  # (D,)
        self.emission_mean.copy_(means)
        if self.covariance_type == "diag":
            self.emission_logvar.copy_(var.log().expand(S, D))
        elif self.covariance_type == "full":
            target_std = var.sqrt().clamp_min(1e-6)
            raw = torch.zeros_like(self.emission_cholesky_raw)
            diag_inv_softplus = torch.log(torch.exp(target_std - self.jitter) - 1.0)
            for s in range(S):
                raw[s].fill_(0.)
                raw[s].diagonal().copy_(diag_inv_softplus)
            self.emission_cholesky_raw.copy_(raw)

        if estimate_transitions:
            d2_bt = (data.to(device).unsqueeze(2) - means.view(1, 1, S, D)).pow(2).sum(-1)  # (B,T,S)
            z = d2_bt.argmin(-1)  # (B,T)
            pi_counts = torch.bincount(z[:, 0], minlength=S).float() + 1e-3
            pi = pi_counts / pi_counts.sum()
            prev = z[:, :-1].reshape(-1)
            nxt = z[:, 1:].reshape(-1)
            joint_idx = prev * S + nxt
            trans_counts = torch.bincount(joint_idx, minlength=S * S).float().reshape(S, S) + 1e-3
            A = trans_counts / trans_counts.sum(-1, keepdim=True)
        else:
            pi = torch.full((S,), 1.0 / S, device=device)
            A = torch.full((S, S), 1.0 / S, device=device)

        self.initial_logits.copy_(pi.clamp_min(1e-12).log())
        self.transition_logits.copy_(A.clamp_min(1e-12).log())

    @torch.no_grad()
    def __apply_noise_and_bias(
        self,
        *,
        mean_std: float,
        cov_noise_std: float,
        init_logits_std: float,
        self_transition_bias: float,
    ) -> None:
        """Apply small Gaussian noise to emissions and logits, and add optional self-transition bias."""
        if mean_std > 0:
            self.emission_mean.add_(mean_std * torch.randn_like(self.emission_mean))
        if cov_noise_std > 0:
            if self.covariance_type == "diag":
                self.emission_logvar.add_(cov_noise_std * torch.randn_like(self.emission_logvar))
            elif self.covariance_type == "full":
                noise = cov_noise_std * torch.randn_like(self.emission_cholesky_raw)
                tril_mask = torch.tril(torch.ones_like(self.emission_cholesky_raw)).bool()
                self.emission_cholesky_raw[tril_mask] = self.emission_cholesky_raw[tril_mask] + noise[tril_mask]
        if init_logits_std > 0:
            self.initial_logits.add_(init_logits_std * torch.randn_like(self.initial_logits))
            self.transition_logits.add_(init_logits_std * torch.randn_like(self.transition_logits))
        if self_transition_bias != 0.0:
            self.transition_logits.diagonal().add_(float(self_transition_bias))

    @torch.no_grad()
    def __clear_param_grads(self) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

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
