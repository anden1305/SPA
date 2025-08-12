"""Hidden Markov Model (Gaussian emissions w/ diagonal covariance).

Design goals:
    * Small, readable, minimal dependencies.
    * Clean separation of responsibilities (helpers for log probs / algorithms).
    * Works as a drop‑in `BaseModel` (forward -> per-sample log-likelihood).

Current scope / simplifications:
    * All sequences assumed same length (no padding mask logic yet).
    * Diagonal Gaussian emissions (state-specific mean + log-variance).
    * Uniform prior / transitions at init (logits=0) unless modified.

Forward(x) returns: Tensor of shape (batch,) with log p(x).

Example
-------
        from scr.models.hmm import HMM
        import torch
        model = HMM(num_states=4, obs_dim=10)
        x = torch.randn(8, 100, 10)  # (B,T,D)
        logp = model(x)              # (B,)
        loss = -logp.mean()
        loss.backward()

Future extensions (not implemented to keep it simple):
    * Variable length handling via mask (log-sum ignoring padded steps).
    * Batched Baum-Welch (EM) style updates or posterior decoding.
    * Different emission families (categorical, mixture, etc.).
"""
from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn as nn

from .base_model import BaseModel


class HMM(BaseModel):
    """Discrete-time HMM with diagonal Gaussian emissions.

    Parameters
    ----------
    num_states : int
        Number of latent states (S).
    obs_dim : int
        Observation dimensionality (D).
    device : str | torch.device | None
        Device to place parameters. Auto-selects CUDA if available.
    """

    def __init__(
        self,
        num_states: int,
        obs_dim: int,
        device: str | torch.device | None = None,
        normalize_time: bool = False,
    ) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.obs_dim = int(obs_dim)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.normalize_time = normalize_time  # if True, forward returns average log-prob per time step

        # Parameterization (logits for simplex params; mean/logvar for Gaussians)
        self.initial_logits = nn.Parameter(torch.zeros(self.num_states))                 # (S,)
        self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states))  # (S,S)
        self.emission_mean = nn.Parameter(torch.zeros(self.num_states, self.obs_dim))    # (S,D)
        self.emission_logvar = nn.Parameter(torch.zeros(self.num_states, self.obs_dim))  # (S,D)

        self.to(self.device)

    # ------------------------- Helper accessors -------------------------
    def log_initial(self) -> Tensor:
        return torch.log_softmax(self.initial_logits, dim=-1)  # (S,)

    def log_transition(self) -> Tensor:
        return torch.log_softmax(self.transition_logits, dim=-1)  # (S,S)

    def emission_log_prob(self, x: Tensor) -> Tensor:
        """Return log p(x_t | z_t) for all states.

        Parameters
        ----------
        x : Tensor, shape (B,T,D)

        Returns
        -------
        Tensor, shape (B,T,S)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B,T,D); got {tuple(x.shape)}")
        if x.shape[2] != self.obs_dim:
            raise ValueError("Input obs_dim mismatch.")
        # Broadcast diff: (B,T,1,D) - (S,D) -> (B,T,S,D)
        diff = x.unsqueeze(2) - self.emission_mean  # (B,T,S,D)
        logvar = self.emission_logvar  # (S,D)
        # (S,D) -> (1,1,S,D) for broadcast in next ops
        inv_var = torch.exp(-logvar)  # positive
        # Gaussian log-density per feature then sum over D
        log_prob = -0.5 * (diff.pow(2) * inv_var + logvar + math.log(2 * math.pi))
        return log_prob.sum(-1)  # (B,T,S)

    # ----------------------- Core algorithms ------------------------
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

    # ---------------------------- API --------------------------------
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Per-sequence log-likelihood.

        x : (B,T,D)
        Returns: (B,) log p(x)
        """
        x = x.to(self.device)
        log_pi = self.log_initial()
        log_A = self.log_transition()
        log_emiss = self.emission_log_prob(x)
        logp = self._forward_algorithm(log_emiss, log_pi, log_A)
        if self.normalize_time:
            logp = logp / x.shape[1]
        return logp

    @torch.no_grad()
    def decode_viterbi(self, x: Tensor) -> Tensor:
        """Most likely state sequence (Viterbi).

        x : (B,T,D)
        Returns: (B,T) int64 state indices
        """
        x = x.to(self.device)
        B, T, D = x.shape
        if D != self.obs_dim:
            raise ValueError("Input obs_dim mismatch.")
        log_pi = self.log_initial()
        log_A = self.log_transition()
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

    # ----------------------- Initialization helpers -----------------------
    @torch.no_grad()
    def reset_parameters(
        self,
        data: Tensor,
        kmeans_iters: int = 15,
        estimate_transitions: bool = True,
    ) -> None:
        """Data-driven (k-means) reinitialization.

        Parameters
        ----------
        data : Tensor
            Shape (T,D) or (B,T,D). Used to initialize means, variances, and (optionally) transitions.
        kmeans_iters : int
            Number of refinement iterations for the simple k-means.
        estimate_transitions : bool
            If True and data has temporal order, build empirical pi and A from cluster sequence.
        """
        S, D = self.num_states, self.obs_dim
        if data.dim() == 3:  # (B,T,D)
            B, T, _ = data.shape
            flat = data.reshape(-1, D).to(self.device)
        elif data.dim() == 2:  # (T,D)
            T = data.shape[0]
            flat = data.to(self.device)
            B = 1
            data = data.unsqueeze(0)  # (1,T,D) for transition estimation
        else:
            raise ValueError("data must have shape (T,D) or (B,T,D)")

        # K-means init
        idx = torch.randperm(flat.size(0), device=self.device)[:S]
        means = flat[idx].clone()
        for _ in range(max(1, kmeans_iters)):
            d2 = (flat[:, None, :] - means[None, :, :]).pow(2).sum(-1)
            assign = d2.argmin(dim=1)
            for s in range(S):
                mask = assign == s
                if mask.any():
                    means[s] = flat[mask].mean(0)

        # Variance (shared across states -> replicate)
        d2_final = (flat - means[assign]).pow(2)
        var = d2_final.mean(0).clamp_min(1e-6)  # (D,)
        self.emission_mean.copy_(means)
        self.emission_logvar.copy_(var.log().expand(S, D))

        # Transitions / initial distribution
        if estimate_transitions:
            # Recompute assignment respecting time order per sequence
            # For efficiency, reuse means: assign each time step to nearest mean
            with torch.no_grad():
                data_bt = data.to(self.device)  # (B,T,D)
                d2_bt = (data_bt.unsqueeze(2) - means.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)  # (B,T,S)
                z = d2_bt.argmin(dim=-1)  # (B,T)
            # Initial distribution
            pi_counts = torch.bincount(z[:, 0], minlength=S).float() + 1e-3
            pi = pi_counts / pi_counts.sum()
            # Transition counts
            trans_counts = torch.zeros(S, S, device=self.device)
            for b in range(B):
                prev = z[b, :-1]
                nxt = z[b, 1:]
                trans_counts.index_put_((prev, nxt), torch.ones_like(prev, dtype=trans_counts.dtype), accumulate=True)
            trans_counts += 1e-3  # smoothing
            A = trans_counts / trans_counts.sum(-1, keepdim=True)
        else:
            pi = torch.full((S,), 1.0 / S, device=self.device)
            A = torch.full((S, S), 1.0 / S, device=self.device)

        self.initial_logits.copy_(pi.clamp_min(1e-12).log())
        self.transition_logits.copy_(A.clamp_min(1e-12).log())

        # Clear any stale grads
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

__all__ = ["HMM"]
