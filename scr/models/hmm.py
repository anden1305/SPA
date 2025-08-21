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
from typing import Optional, Sequence
import itertools

from scr.config.config import GlobalConfig
from scr.data.data_loader import DataLoader

from .base_model import MLModel


class HMM(MLModel):
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

    def __init__(self,
                 data_loader: DataLoader,
                 config: GlobalConfig) -> None:
        super().__init__(data_loader, config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
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
        diag_idx = torch.arange(self.num_features, device=L.device)
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
        if x.shape[2] != self.num_features:
            raise ValueError(f"Input feature dimension {x.shape[2]} != model obs_dim {self.num_features}. If you passed (B,T,C,F) ensure obs_dim=C*F when constructing the model.")
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
            if x.shape[2] == self.num_features:
                return x
            # Case (B,D,T)
            if x.shape[1] == self.num_features:
                return x.transpose(1, 2)
            raise ValueError(f"3D input must have one dimension equal to obs_dim={self.num_features}; got {tuple(x.shape)}")
        if x.dim() == 2:
            if x.shape[1] == self.num_features:  # (T,D)
                return x.unsqueeze(0)
            if x.shape[0] == self.num_features:  # (D,T)
                return x.transpose(0,1).unsqueeze(0)
            raise ValueError(f"2D input ambiguous shape {tuple(x.shape)} (obs_dim={self.num_features})")
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
            f"states={self.num_states}, obs_dim={self.num_features}, cov='{self.covariance_type}', "
            f"normalize_time={self.normalize_time}"
        )

    @torch.no_grad()
    def decode_viterbi(self, x: Tensor) -> Tensor:
        """Most likely state sequence (Viterbi path).

        Accepts same flexible shapes as forward; returns (B,T) int64.
        """
        x = self._standardize_input(x.to(self.device))
        B, T, D = x.shape
        if D != self.num_features:
            raise ValueError(f"Input feature dimension {D} != model obs_dim {self.num_features}.")
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
        if D != self.num_features:
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

    @torch.no_grad()
    def reset_parameters_random(
        self,
        data: Tensor | None = None,
        kmeans_iters: int = 15,
        estimate_transitions: bool = True,
        mean_std: float = 1e-2,
        cov_noise_std: float = 1e-3,
        init_logits_std: float = 0.0,
        self_transition_bias: float = 0.0,
    ) -> None:
        """K-means init followed by small random noise (symmetry breaking).

        If ``data`` is provided, run ``reset_parameters(data, ...)`` to get a
        sensible K-means initialisation for emissions (and optionally pi/A),
        then add small random noise to parameters to avoid degenerate plateaus.
        If ``data`` is None, fall back to a pure random jitter initialisation
        close to Identity covariance.

        Parameters
        ----------
        data : Tensor | None
            Training batch or dataset slice shaped (T,D) or (B,T,D) used for K-means.
        kmeans_iters : int
            Number of K-means refinement iterations.
        estimate_transitions : bool
            Whether to estimate initial and transition probabilities from K-means labels.
        mean_std : float
            Std-dev of additive noise for emission means after initialisation.
        cov_noise_std : float
            Additive noise magnitude for covariance parameters (logvar or cholesky raw).
        init_logits_std : float
            Std-dev of additive noise for initial/transition logits after initialisation.
        self_transition_bias : float
            Optional extra bias added to transition logits diagonal after init (default 0.0).
        """
        # 1) Base initialisation
        if data is not None:
            # Use existing K-means initialiser
            self.reset_parameters(data, kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
        else:
            # Fallback: initialise near Identity with small random differences
            self.emission_mean.normal_(mean=0.0, std=mean_std)
            if self.covariance_type == "diag":
                self.emission_logvar.zero_()
                if cov_noise_std > 0:
                    self.emission_logvar.add_(cov_noise_std * torch.randn_like(self.emission_logvar))
            elif self.covariance_type == "full":
                raw = torch.zeros_like(self.emission_cholesky_raw)
                if cov_noise_std > 0:
                    tril_mask = torch.tril(torch.ones_like(raw)).bool()
                    noise = cov_noise_std * torch.randn_like(raw)
                    raw[tril_mask] = noise[tril_mask]
                self.emission_cholesky_raw.copy_(raw)
            # logits near 0
            if init_logits_std > 0:
                self.initial_logits.normal_(mean=0.0, std=init_logits_std)
                self.transition_logits.normal_(mean=0.0, std=init_logits_std)
            if self_transition_bias != 0.0:
                self.transition_logits.diagonal().add_(float(self_transition_bias))

        # 2) Add small random noise to break symmetry (applies to both branches)
        # Emission means
        if mean_std > 0:
            self.emission_mean.add_(mean_std * torch.randn_like(self.emission_mean))

        # Covariance params
        if cov_noise_std > 0:
            if self.covariance_type == "diag":
                self.emission_logvar.add_(cov_noise_std * torch.randn_like(self.emission_logvar))
            elif self.covariance_type == "full":
                noise = cov_noise_std * torch.randn_like(self.emission_cholesky_raw)
                tril_mask = torch.tril(torch.ones_like(self.emission_cholesky_raw)).bool()
                self.emission_cholesky_raw[tril_mask] = self.emission_cholesky_raw[tril_mask] + noise[tril_mask]
            # meanonly: nothing

        # Logits (small unbiased jitter, optionally with diagonal bias)
        if init_logits_std > 0:
            self.initial_logits.add_(init_logits_std * torch.randn_like(self.initial_logits))
            self.transition_logits.add_(init_logits_std * torch.randn_like(self.transition_logits))
        if self_transition_bias != 0.0:
            self.transition_logits.diagonal().add_(float(self_transition_bias))

        # Clear any stale grads
        for p in self.parameters():
            if p.grad is not None:
                p.grad.zero_()

    # ----------------------- Forward-Backward (Posterior) -----------------------
    ## OPTIONAL - for testing math
    @torch.no_grad()
    def forward_backward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute posterior state probabilities (gamma) and log-likelihood.

        Returns
        -------
        gamma : (T,S) posterior p(z_t | x)
        logp  : scalar log p(x)
        """
        x_std = self._standardize_input(x.to(self.device))  # (B,T,D) or (1,T,D)
        if x_std.shape[0] != 1:
            raise ValueError("forward_backward expects a single sequence (T,D) or (1,T,D)")
        _, T, _ = x_std.shape
        log_pi = self.log_initial()               # (S,)
        log_A = self.log_transition()             # (S,S)
        log_emiss = self.emission_log_prob(x_std) # (1,T,S)
        log_emiss = log_emiss.squeeze(0)          # (T,S)
        S = self.num_states
        # Forward
        alpha = torch.empty((T, S), device=self.device)
        alpha[0] = log_pi + log_emiss[0]
        for t in range(1, T):
            # shape (S,S): prev_alpha_j + log_A_{j->i}
            scores = alpha[t-1].unsqueeze(1) + log_A  # (S,S)
            alpha[t] = log_emiss[t] + torch.logsumexp(scores, dim=0)
        logp = torch.logsumexp(alpha[-1], dim=0)
        # Backward
        beta = torch.empty((T, S), device=self.device)
        beta[-1].zero_()
        for t in range(T-2, -1, -1):
            # (S,S): log_A_{i->j} + log_emiss_{t+1,j} + beta_{t+1,j}
            scores = log_A + (log_emiss[t+1] + beta[t+1]).unsqueeze(0)
            beta[t] = torch.logsumexp(scores, dim=1)
        # Posterior
        gamma = alpha + beta - logp  # log gamma
        gamma = torch.softmax(gamma, dim=-1)  # numerically stable normalization
        return gamma, logp

    # ----------------------- Brute Force Likelihood (Tiny) -----------------------
    ## OPTIONAL - for testing math
    @torch.no_grad()
    def brute_force_log_prob(self, x: Tensor, max_T: int = 8) -> Tensor:
        """Exact log p(x) by enumerating all state sequences (for tiny T).

        Useful for correctness verification. Only feasible for very small T.
        """
        x_std = self._standardize_input(x.to(self.device))  # (1,T,D) or (B,T,D)
        if x_std.shape[0] != 1:
            raise ValueError("brute_force_log_prob expects a single sequence")
        _, T, _ = x_std.shape
        if T > max_T:
            raise ValueError(f"Sequence length T={T} too large for brute force (max_T={max_T})")
        log_pi = self.log_initial()             # (S,)
        log_A = self.log_transition()           # (S,S)
        log_emiss = self.emission_log_prob(x_std).squeeze(0)  # (T,S)
        S = self.num_states
        paths = []
        for states in itertools.product(range(S), repeat=T):
            lp = log_pi[states[0]] + log_emiss[0, states[0]]
            for t in range(1, T):
                lp = lp + log_A[states[t-1], states[t]] + log_emiss[t, states[t]]
            paths.append(lp)
        return torch.logsumexp(torch.stack(paths), dim=0)

    # ----------------------- Tiny Verification Bundle -----------------------
    ## OPTIONAL - for testing math
    @torch.no_grad()
    def verify_tiny(self, x: Tensor, atol: float = 1e-5, verbose: bool = True) -> dict:
        """Run a set of internal consistency checks on a tiny sequence.

        Checks:
            - forward vs brute-force likelihood
            - forward vs forward_backward log-likelihood
            - Viterbi path joint prob <= log-likelihood
            - gamma rows sum to 1
        Returns dictionary of metrics / differences.
        """
        bf = self.brute_force_log_prob(x)
        fw = self.forward(x).squeeze(0)
        gamma, fb = self.forward_backward(x)
        # Viterbi path joint prob
        path = self.decode_viterbi(x).squeeze(0)
        x_std = self._standardize_input(x.to(self.device))
        log_pi = self.log_initial()
        log_A = self.log_transition()
        log_emiss = self.emission_log_prob(x_std).squeeze(0)  # (T,S)
        joint = log_pi[path[0]] + log_emiss[0, path[0]]
        for t in range(1, path.numel()):
            joint = joint + log_A[path[t-1], path[t]] + log_emiss[t, path[t]]
        max_diff = (fw - joint).item()
        gamma_row_sums = gamma.sum(-1)
        metrics = {
            "logp_forward": fw.item(),
            "logp_bruteforce": bf.item(),
            "abs_diff_forward_bruteforce": abs(fw.item() - bf.item()),
            "logp_forward_backward": fb.item(),
            "abs_diff_forward_fb": abs(fw.item() - fb.item()),
            "viterbi_joint": joint.item(),
            "forward_minus_viterbi_joint": max_diff,
            "gamma_row_sums_min": gamma_row_sums.min().item(),
            "gamma_row_sums_max": gamma_row_sums.max().item(),
        }
        if verbose:
            print("[HMM verify_tiny]")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            if metrics["abs_diff_forward_bruteforce"] > atol:
                print(f"  WARNING: forward vs brute-force diff {metrics['abs_diff_forward_bruteforce']:.3e} > atol={atol}")
            if metrics["abs_diff_forward_fb"] > atol:
                print(f"  WARNING: forward vs forward_backward diff {metrics['abs_diff_forward_fb']:.3e} > atol={atol}")
        return metrics

    # ----------------------- Lightweight Plot Helper -----------------------
    def plot_pca_scatter(
        self,
        x: Tensor,
        true_labels: Optional[Sequence[int]] = None,
        pred_labels: Optional[Sequence[int]] = None,
        subsample: int = 5000,
        out_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate a simple 2D PCA scatter of the (T,D) sequence.

        Parameters
        ----------
        x : Tensor | (T,D) or (B,T,D)
            Input sequence(s); if batched, will be flattened across batch.
        true_labels : optional sequence length T
        pred_labels : optional sequence length T
        subsample : int
            Uniformly subsample at most this many time points (0 => no limit).
        out_path : str | None
            If provided, saves figure to this path (parent dirs must exist).

        Returns
        -------
        out_path if saved, else None.
        """
        try:
            import numpy as _np
            import matplotlib.pyplot as _plt
        except Exception:
            raise RuntimeError("matplotlib and numpy required for plot_pca_scatter")

        x_std = self._standardize_input(x.to(self.device))  # (B,T,D)
        B, T, D = x_std.shape
        X = x_std.reshape(B * T, D).detach().cpu().numpy()  # (N,D)

        if true_labels is not None:
            true_arr = _np.asarray(true_labels).ravel()
        else:
            true_arr = None
        if pred_labels is not None:
            pred_arr = _np.asarray(pred_labels).ravel()
        else:
            pred_arr = None

        N = X.shape[0]
        if subsample and subsample > 0 and N > subsample:
            idx = _np.linspace(0, N - 1, subsample).astype(int)
            Xs = X[idx]
            if true_arr is not None and true_arr.shape[0] == N:
                true_arr = true_arr[idx]
            if pred_arr is not None and pred_arr.shape[0] == N:
                pred_arr = pred_arr[idx]
        else:
            Xs = X

        # PCA via SVD
        Xc = Xs - Xs.mean(0, keepdims=True)
        U, Svals, _ = _np.linalg.svd(Xc, full_matrices=False)
        proj = (U[:, :2] * Svals[:2]) if Svals.size >= 2 else _np.pad(U[:, :1] * Svals[:1], ((0,0),(0,1)))

        def _color_map(labels):
            if labels is None:
                return None
            palette = _np.array(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
            uniq = _np.unique(labels)
            lut = {u: palette[i % len(palette)] for i, u in enumerate(sorted(uniq))}
            return _np.array([lut[v] for v in labels])

        colors_true = _color_map(true_arr)
        colors_pred = _color_map(pred_arr)

        ncols = 1 + int(pred_arr is not None and true_arr is not None)
        fig, axes = _plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharex=True, sharey=True)
        if ncols == 1:
            axes = [axes]
        if true_arr is not None and pred_arr is not None:
            axes[0].scatter(proj[:, 0], proj[:, 1], c=colors_true, s=6, alpha=0.8, edgecolors='none')
            axes[0].set_title('True Labels (PCA)')
            axes[1].scatter(proj[:, 0], proj[:, 1], c=colors_pred, s=6, alpha=0.8, edgecolors='none')
            axes[1].set_title('Predicted Labels (PCA)')
        else:
            axes[0].scatter(proj[:, 0], proj[:, 1], c=colors_pred or colors_true or '#1f77b4', s=6, alpha=0.8, edgecolors='none')
            axes[0].set_title('PCA Scatter')
        for ax in axes:
            ax.set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        _plt.tight_layout()
        if out_path is not None:
            _plt.savefig(out_path, dpi=150)
            _plt.close(fig)
            return out_path
        return None
    
    def prepare_for_training(self, data: Tensor):
        self.train()
        self.reset_parameters_random()

    def __str__(self):
        return f"HMM(num_states={self.num_states}, num_features={self.num_features})"

__all__ = ["HMM"]
