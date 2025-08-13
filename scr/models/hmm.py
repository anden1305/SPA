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
        """K-means based reinit of emissions (+ optional empirical pi/A).

        data: (T,D) or (B,T,D)
        kmeans_iters: refinement steps (0 -> only random init)
        estimate_transitions: if True, build pi/A from cluster path(s)
        """
        if data.dim() == 2:  # (T,D)
            data = data.unsqueeze(0)
        if data.dim() != 3:
            raise ValueError("data must have shape (T,D) or (B,T,D)")
        B, T, D = data.shape
        S = self.num_states
        if D != self.obs_dim:
            raise ValueError("obs_dim mismatch")
        device = self.device
        flat = data.reshape(-1, D).to(device)

        # -------- K-means (simple, vectorized) --------
        perm = torch.randperm(flat.size(0), device=device)
        means = flat[perm[:S]].clone()
        iters = max(0, int(kmeans_iters))
        for _ in range(iters):
            # assignment
            assign = torch.cdist(flat, means).argmin(-1)  # (N,)
            # update means (scatter-add then divide by counts)
            counts = torch.bincount(assign, minlength=S).clamp_min(1)
            new_means = torch.zeros_like(means)
            new_means.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), flat)
            means = new_means / counts.unsqueeze(1)
        if iters == 0:
            # still need assignments for variance
            assign = torch.cdist(flat, means).argmin(-1)

        # shared variance, then replicate
        resid = flat - means[assign]
        var = resid.pow(2).mean(0).clamp_min(1e-6)  # (D,)
        self.emission_mean.copy_(means)
        self.emission_logvar.copy_(var.log().expand(S, D))

        # -------- Transitions / initial --------
        if estimate_transitions:
            # per time-step assignment (B,T)
            d2_bt = (data.to(device).unsqueeze(2) - means.view(1, 1, S, D)).pow(2).sum(-1)  # (B,T,S)
            z = d2_bt.argmin(-1)  # (B,T)
            # initial distribution
            pi_counts = torch.bincount(z[:, 0], minlength=S).float() + 1e-3
            pi = pi_counts / pi_counts.sum()
            # transitions (vectorized pair counting)
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

        for p in self.parameters():  # clear stale grads
            if p.grad is not None:
                p.grad.zero_()

__all__ = ["HMM"]
