"""Hidden Markov Model (Gaussian emissions; mean-only / diag / full covariance).

Design goals:
    * Small, readable, minimal dependencies.
    * Clean separation of responsibilities (helpers for log probs / algorithms).
    * Works as a drop‑in `BaseModel` (forward -> per-sample log-likelihood).

Current scope / simplifications:
    * All sequences assumed same length (no padding mask logic yet).
    * Gaussian emissions: covariance type in {meanonly (I), diag, full}.
    * Uniform prior / transitions at init (logits=0) unless modified.

Forward(x) returns: Tensor of shape (batch,) with log p(x).

Required input shape
--------------------
Canonical shape: (B, T, D)
    B = batch size (independent sequences)
    T = sequence length (ordered time windows / frames)
    D = flattened feature dimension = Channels * Per‑channel spectral features (C×F).

Pragmatic compatibility helpers (auto‑reshape):
    * (T, D) -> unsqueezed to (1, T, D)
    * (B, D, T) -> transposed to (B, T, D) if D==obs_dim and T != obs_dim
    * (C, T) with C==obs_dim -> treated as (1, T, C) after transpose
Any other unexpected rank raises ValueError. This keeps core logic minimal while
allowing simple dataset loaders that output channel-first sequences.

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


    Input: N_samples x Number of frequency bins

"""
from __future__ import annotations

import math
import torch
from torch import Tensor
import torch.nn as nn

from .base_model import BaseModel


class HMM(BaseModel):
    """Discrete‑time Hidden Markov Model with Gaussian emissions.

    Supports three emission covariance parameterisations:
    - ``diag`` (default): state‑specific diagonal covariance (original behaviour)
    - ``full``: state‑specific full covariance via unconstrained Cholesky factors
    - ``meanonly``: fixed identity covariance (learn only means) – useful as a
      baseline to test whether covariance structure matters.

    Forward returns per‑sequence log p(x) (optionally normalised by time if
    ``normalize_time=True``). Training can therefore proceed by maximising this
    marginal log‑likelihood directly with any gradient optimiser (no EM needed).

    Parameters
    ----------
    num_states : int
        Number of latent states (S).
    obs_dim : int
        Observation dimensionality (D).
    covariance_type : {'diag','full','meanonly'}
        Emission covariance parameterisation.
    device : str | torch.device | None
        Device to place parameters. Auto‑selects CUDA if available.
    jitter : float
        Small positive value added to covariance diagonals for numerical
        stability (ignored for meanonly which uses Identity).
    normalize_time : bool
        If True, divide log p(x) by sequence length T (average per time step).
    """

    def __init__(
        self,
        num_states: int,
        obs_dim: int,
        covariance_type: str = "diag",
        device: str | torch.device | None = None,
        jitter: float = 1e-5,
        normalize_time: bool = False,
    ) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.obs_dim = int(obs_dim)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.normalize_time = normalize_time  # if True, forward returns average log-prob per time step
        self.covariance_type = covariance_type.lower()
        if self.covariance_type not in {"diag", "full", "meanonly"}:
            raise ValueError("covariance_type must be one of {'diag','full','meanonly'}")
        self.jitter = float(jitter)

        # Parameterization (logits for simplex params; mean/logvar for Gaussians)
        self.initial_logits = nn.Parameter(torch.zeros(self.num_states))                 # (S,)
        self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states))  # (S,S)
        self.emission_mean = nn.Parameter(torch.zeros(self.num_states, self.obs_dim))    # (S,D)
        if self.covariance_type == "diag":
            self.emission_logvar = nn.Parameter(torch.zeros(self.num_states, self.obs_dim))  # (S,D)
        elif self.covariance_type == "full":
            # Unconstrained lower‑triangular raw parameters for Cholesky factors L (S,D,D)
            # We store a full matrix then mask to lower tri; diagonals passed through softplus.
            self.emission_cholesky_raw = nn.Parameter(torch.zeros(self.num_states, self.obs_dim, self.obs_dim))
        else:  # meanonly -> no covariance parameters (implicit Identity)
            self.register_buffer("_identity_cov", torch.eye(self.obs_dim))
        # Constant buffer for numerical expressions (avoids repeated Python math calls)
        self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=torch.get_default_dtype()))
        self.to(self.device)

    # ------------------------- Helper accessors -------------------------
    def log_initial(self) -> Tensor:
        return torch.log_softmax(self.initial_logits, dim=-1)  # (S,)

    def log_transition(self) -> Tensor:
        return torch.log_softmax(self.transition_logits, dim=-1)  # (S,S)

    def _full_cov_cholesky(self) -> Tensor:
        """Return lower‑triangular Cholesky factors L (S,D,D) with positive diag."""
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
            raise ValueError(f"emission_log_prob expects (B,T,D) after any flattening; got {tuple(x.shape)}")
        if x.shape[2] != self.obs_dim:
            raise ValueError(f"Input feature dimension {x.shape[2]} != model obs_dim {self.obs_dim}. If you passed (B,T,C,F) ensure obs_dim=C*F when constructing the model.")
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
            L = self._full_cov_cholesky()  # (S,D,D)
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
    def _standardize_input(self, x: Tensor) -> Tensor:
        """Convert allowed shapes to (B,T,D).

        Allowed:
            (B,T,D) -> unchanged
            (T,D)   -> (1,T,D)
            (D,T)   -> (1,T,D) if D==obs_dim
            (B,D,T) -> (B,T,D) if D==obs_dim
        """
        if x.dim() == 3:
            B, A, B_or_D = x.shape
            # Case already (B,T,D)
            if x.shape[2] == self.obs_dim:
                return x
            # Case (B,D,T)
            if x.shape[1] == self.obs_dim:
                return x.transpose(1, 2)
            raise ValueError(f"3D input must have one dimension equal to obs_dim={self.obs_dim}; got {tuple(x.shape)}")
        if x.dim() == 2:
            if x.shape[1] == self.obs_dim:  # (T,D)
                return x.unsqueeze(0)
            if x.shape[0] == self.obs_dim:  # (D,T)
                return x.transpose(0,1).unsqueeze(0)
            raise ValueError(f"2D input ambiguous shape {tuple(x.shape)} (obs_dim={self.obs_dim})")
        raise ValueError(f"Unsupported input rank {x.dim()} (expected 2 or 3)")

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Per-sequence log-likelihood.

        Returns (B,) log p(x). Accepts canonical (B,T,D) and a few convenience
        variants documented in `_standardize_input`.
        """
        x = self._standardize_input(x.to(self.device))
        log_pi = self.log_initial()
        log_A = self.log_transition()
        log_emiss = self.emission_log_prob(x)
        logp = self._forward_algorithm(log_emiss, log_pi, log_A)
        if self.normalize_time:
            logp = logp / x.shape[1]
        return logp

    # ---------------- Convenience API aliases ----------------
    @torch.no_grad()
    def log_prob(self, x: Tensor) -> Tensor:
        """Alias for forward (evaluates log p(x))."""
        return self.forward(x)

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Alias returning Viterbi path for compatibility with sklearn-like API."""
        return self.decode_viterbi(x)

    def extra_repr(self) -> str:  # type: ignore[override]
        return (
            f"states={self.num_states}, obs_dim={self.obs_dim}, cov='{self.covariance_type}', "
            f"normalize_time={self.normalize_time}"
        )

    @torch.no_grad()
    def decode_viterbi(self, x: Tensor) -> Tensor:
        """Most likely state sequence (Viterbi path).

        Accepts same flexible shapes as forward; returns (B,T) int64.
        """
        x = self._standardize_input(x.to(self.device))
        B, T, D = x.shape
        if D != self.obs_dim:
            raise ValueError(f"Input feature dimension {D} != model obs_dim {self.obs_dim}.")
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
        # log_pi = self.log_initial()
        # log_A = self.log_transition()
        # log_emiss = self.emission_log_prob(x)  # (B,T,S)

        # backptr = x.new_zeros((B, T, self.num_states), dtype=torch.long)
        # delta = log_pi.unsqueeze(0) + log_emiss[:, 0, :]  # (B,S)
        # for t in range(1, T):
        #     scores = delta.unsqueeze(2) + log_A.unsqueeze(0)  # (B,S,S)
        #     delta, idx = torch.max(scores, dim=1)            # (B,S)
        #     delta = delta + log_emiss[:, t, :]
        #     backptr[:, t, :] = idx
        # last = torch.argmax(delta, dim=1)  # (B,)
        # path = x.new_zeros((B, T), dtype=torch.long)
        # path[:, -1] = last
        # for t in range(T - 2, -1, -1):
        #     path[:, t] = backptr[torch.arange(B), t + 1, path[:, t + 1]]
        # return path

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
        if self.covariance_type == "diag":
            self.emission_logvar.copy_(var.log().expand(S, D))
        elif self.covariance_type == "full":
            # Initialise to diagonal with var -> set raw so that softplus(raw)=sqrt(var)
            target_std = var.sqrt().clamp_min(1e-6)
            with torch.no_grad():
                raw = torch.zeros_like(self.emission_cholesky_raw)
                diag_inv_softplus = torch.log(torch.exp(target_std - self.jitter) - 1.0)
                for s in range(S):
                    raw[s].fill_(0.)
                    raw[s].diagonal().copy_(diag_inv_softplus)
                self.emission_cholesky_raw.copy_(raw)
        # meanonly: nothing

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
