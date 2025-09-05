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

    # ----------------------- Initialization helpers -----------------------
    @torch.no_grad()
    def __initialise_weights(
        self,
        kmeans_iters: int = 150,
        estimate_transitions: bool = True,
        mean_std: float = 1.5,
        cov_noise_std: float = 1.5,
        init_logits_std: float = 0.5,
        self_transition_bias: float = 0.5,
        spread: float = 2.0,
        jitter_std: float = 0.05,
    ) -> None:
        """Short dispatcher for HMM weight initialization.

        strategy in {random, random_separated, random_uniform, random_dirichlet, kmeans, kmeans_noisy, kmeans_pca}.
        - random or random_separated: separated means (approx. orthogonal), unit covariance, uniform pi/A.
        - random_uniform: fully random means, unit covariance, uniform pi/A.
        - random_dirichlet: fully random means, unit covariance, Dirichlet-sampled pi/A.
        - kmeans: k-means for emissions (and optionally pi/A).
        - kmeans_noisy: run k-means, then add small random noise to avoid plateaus.
        - kmeans_pca: search small PCA subspaces (1–3 dims among top-L PCs), run k-means, pick best by CH score.
        - sticky_em_warmstart: run a few EM iterations with high self-transition bias to get a reasonable starting point.
        Data is optional (not needed for random*).
        """
        s = self.global_config.model.init_strategy.lower()
        if s in ("random", "random_separated"):
            self.__init_random_separated(spread=spread, jitter_std=jitter_std)
        elif s == "random_uniform":
            self.__init_random_uniform(mean_std=mean_std, jitter_std=jitter_std)
        elif s == "random_dirichlet":
            self.__init_random_dirichlet(mean_std=mean_std, alpha=1.0, self_transition_bias=self_transition_bias)
        elif s == "kmeans":
            self.__init_kmeans(kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
        elif s == "kmeans_noisy":
            self.__init_kmeans(kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
            self.__apply_noise_and_bias(mean_std=mean_std, cov_noise_std=cov_noise_std, init_logits_std=init_logits_std, self_transition_bias=self_transition_bias)
        elif s == "kmeans_pca":
            self.__init_kmeans_pca(kmeans_iters=kmeans_iters, estimate_transitions=estimate_transitions)
        elif s == "sticky_em_warmstart":
            self.__init_sticky_em_warmstart(n_iters=3, kappa=3.0, min_var=1e-6)
        else:
            raise ValueError("strategy must be one of {'random','random_separated','random_uniform','random_dirichlet','kmeans','kmeans_noisy','kmeans_pca','sticky_em_warmstart'}")
        self.__clear_param_grads()

    def __init_random_separated(self, spread: float = 2.0, jitter_std: float = 0.05,) -> None:
        """Structured random initialization that separates state means.

          Compared to a fully random init (e.g., sampling means i.i.d. from N(0, I)),
          this method intentionally spreads means across distinct directions to avoid
          overlapping emissions and poor early training dynamics.

          Steps:
          1) Build S approximately orthogonal direction vectors in R^D and scale by
              `spread`. Add small Gaussian jitter (jitter_std) in all dims to break
              symmetry.
          2) Set covariance to unit variance: diag logvars=0 or full-cov L with
              softplus(diag) ~ 1 (+ self.jitter for stability).
          3) Initialize initial/transition logits uniformly (zeros).
        """
        S, D = self.num_states, self.num_features
        device = self.emission_mean.device
        # Construct S distinct direction vectors in R^D
        if D >= S:
            R = torch.randn(D, S, device=device)
            Q, _ = torch.linalg.qr(R, mode='reduced')  # (D,S)
            dirs = Q.T  # (S,D)
        else:
            R = torch.randn(S, D, device=device)
            dirs = R / R.norm(dim=1, keepdim=True).clamp_min(1e-8)
        means = spread * dirs
        means = means + jitter_std * torch.randn(means.shape, device=device)
        self.emission_mean.copy_(means)
        # 2. Covariance
        if self.covariance_type == "diag":
            self.emission_logvar.zero_()
        elif self.covariance_type == "full":
            raw = torch.zeros_like(self.emission_cholesky_raw)
            target = max(1.0 - float(self.jitter), 1e-6)
            diag_raw_val = torch.log(torch.expm1(torch.tensor(target, device=device)))
            idx = torch.arange(D, device=device)
            raw[:, idx, idx] = diag_raw_val
            self.emission_cholesky_raw.copy_(raw)
        # 3. Uniform pi/A
        self.initial_logits.zero_()
        self.transition_logits.zero_()

    def __init_random_uniform(self, mean_std: float = 1.0, jitter_std: float = 0.0) -> None:
        """Fully random means, unit covariance, uniform π and A."""
        S, D = self.num_states, self.num_features
        device = self.emission_mean.device
        # Means ~ N(0, mean_std^2 I)
        means = mean_std * torch.randn(S, D, device=device)
        if jitter_std > 0:
            means = means + jitter_std * torch.randn_like(means)
        self.emission_mean.copy_(means)
        # Covariance ~ Identity
        if self.covariance_type == "diag":
            self.emission_logvar.zero_()
        elif self.covariance_type == "full":
            raw = torch.zeros_like(self.emission_cholesky_raw)
            target = max(1.0 - float(self.jitter), 1e-6)
            diag_raw_val = torch.log(torch.expm1(torch.tensor(target, device=device)))
            idx = torch.arange(D, device=device)
            raw[:, idx, idx] = diag_raw_val
            self.emission_cholesky_raw.copy_(raw)
        # Uniform π and A
        self.initial_logits.zero_()
        self.transition_logits.zero_()

    def __init_random_dirichlet(
        self,
        mean_std: float = 1.0,
        alpha: float = 1.0,
        self_transition_bias: float = 0.0,
    ) -> None:
        """Fully random means, unit covariance, Dirichlet-sampled π and A."""
        S, D = self.num_states, self.num_features
        device = self.emission_mean.device
        dtype = self.initial_logits.dtype

        # Means ~ N(0, mean_std^2 I)
        means = mean_std * torch.randn(S, D, device=device)
        self.emission_mean.copy_(means)

        # Covariance ~ Identity
        if self.covariance_type == "diag":
            self.emission_logvar.zero_()
        elif self.covariance_type == "full":
            raw = torch.zeros_like(self.emission_cholesky_raw)
            target = max(1.0 - float(self.jitter), 1e-6)
            diag_raw_val = torch.log(torch.expm1(torch.tensor(target, device=device)))
            idx = torch.arange(D, device=device)
            raw[:, idx, idx] = diag_raw_val
            self.emission_cholesky_raw.copy_(raw)

        # π ~ Dirichlet(α)
        alpha_vec = torch.full((S,), float(alpha), device=device, dtype=dtype)
        pi = torch.distributions.Dirichlet(alpha_vec).sample()
        # A rows ~ Dirichlet(α)
        A = torch.stack(
            [torch.distributions.Dirichlet(alpha_vec).sample() for _ in range(S)],
            dim=0
        )

        # Store as logits (training uses log_softmax later)
        self.initial_logits.copy_(pi.clamp_min(1e-12).log())
        self.transition_logits.copy_(A.clamp_min(1e-12).log())

        # Optional self-transition bias in logits space
        if self_transition_bias != 0.0:
            self.transition_logits.diagonal().add_(float(self_transition_bias))


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

        # Deterministic seeding for initial centers
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
            noise = torch.randn(self.emission_mean.shape, device=self.emission_mean.device)
            self.emission_mean.add_(mean_std * noise)
        if cov_noise_std > 0:
            if self.covariance_type == "diag":
                noise = torch.randn(self.emission_logvar.shape, device=self.emission_logvar.device)
                self.emission_logvar.add_(cov_noise_std * noise)
            elif self.covariance_type == "full":
                noise = cov_noise_std * torch.randn(self.emission_cholesky_raw.shape, device=self.emission_cholesky_raw.device)
                tril_mask = torch.tril(torch.ones_like(self.emission_cholesky_raw)).bool()
                self.emission_cholesky_raw[tril_mask] = self.emission_cholesky_raw[tril_mask] + noise[tril_mask]
        if init_logits_std > 0:
            self.initial_logits.add_(init_logits_std * torch.randn(self.initial_logits.shape, device=self.initial_logits.device))
            self.transition_logits.add_(init_logits_std * torch.randn(self.transition_logits.shape, device=self.transition_logits.device))
        if self_transition_bias != 0.0:
            self.transition_logits.diagonal().add_(float(self_transition_bias))

    @torch.no_grad()
    def __init_kmeans_pca(
        self,
        *,
        kmeans_iters: int,
        estimate_transitions: bool,
        top_l: int = 4,
        max_dim: int = 3,
        whiten: bool = True,
    ) -> None:
        """Initialize emissions via k-means in the best PCA subspace (auto-selected).

        Procedure:
        1) Flatten frames X (N,D), center, compute compact SVD -> PCs V and singular values S.
        2) Consider all combinations of 1..max_dim PCs drawn from the top 'top_l' PCs.
        3) For each subspace: project (and optionally whiten), run k-means, score by Calinski–Harabasz (CH).
        4) Pick the best subspace by CH; lift assignments back to original space to compute means/covariances.
        5) Estimate π and A from per-time hard labels (with smoothing).
        
        Parameters
        ----------
        kmeans_iters : int
            Number of k-means iterations per candidate subspace
        estimate_transitions : bool
            Whether to estimate initial/transition logits from data
        top_l : int
            Number of top PCA components to consider for subspace search, i.e. the pool of PCs
            from which to draw combinations. Must be at least max_dim.
        max_dim : int
            Maximum PCA subspace dimension to consider (up to 3). This controls the largest
            number of PCs to combine when building candidate subspaces.
        whiten : bool
            Whether to whiten PCA projections before k-means. Whiten means to scale
            each PC by 1/singular_value, so that the resulting features have unit variance. 
            Produces more balanced clustering because distances in K-means aren't dominated
            by high-variance PCs.
        """
        # Allow config to override defaults without cluttering dispatcher
        cfg_model = getattr(self.global_config, "model", None)
        if cfg_model is not None:
            try:
                top_l = int(getattr(cfg_model, "init_pca_top_l", top_l))
            except Exception:
                pass
            try:
                max_dim = int(getattr(cfg_model, "init_pca_max_dim", max_dim))
            except Exception:
                pass
            try:
                whiten = bool(getattr(cfg_model, "init_pca_whiten", whiten))
            except Exception:
                pass

        data, _ = self.data_loader.get_all_data()
        data = self.__validate_input(data)
        B, T, D = data.shape
        S = self.num_states
        device = self.device
        X = data.reshape(-1, D).to(device)  # (N,D)
        N = X.size(0)
        if N < S:
            raise ValueError("Not enough frames for k-means init in PCA space.")

        # Center and compute compact SVD
        mean_x = X.mean(0, keepdim=True)  # (1,D)
        Xc = X - mean_x
        # U:(N,r), Svals:(r,), Vh:(r,D)
        U, Svals, Vh = torch.linalg.svd(Xc, full_matrices=False)
        V = Vh.transpose(0, 1)  # (D,r)
        r = V.size(1)
        L = max(1, min(int(top_l), r))
        mmax = max(1, min(int(max_dim), L))

        # Build candidate index tuples of sizes 1..mmax from [0..L-1]
        # Use an inline combinations builder for clarity (avoid global imports)
        candidates = []  # list[list[int]]
        rng = list(range(L))
        # size 1
        for i in rng:
            candidates.append([i])
        # size 2
        if mmax >= 2:
            for i in rng:
                for j in range(i + 1, L):
                    candidates.append([i, j])
        # size 3
        if mmax >= 3:
            for i in rng:
                for j in range(i + 1, L):
                    for k in range(j + 1, L):
                        candidates.append([i, j, k])

        def kpp_init(Z: torch.Tensor, K: int) -> torch.Tensor:
            # k-means++ init
            n, d = Z.shape
            idx0 = torch.randint(0, n, (1,), device=Z.device)
            centers = [Z[idx0]]
            d2 = torch.cdist(Z, centers[0]).pow(2).squeeze(1)
            for _ in range(1, K):
                probs = (d2 / d2.sum().clamp_min(1e-12)).clamp_min(1e-12)
                next_idx = torch.multinomial(probs, 1)
                centers.append(Z[next_idx])
                d2 = torch.minimum(d2, torch.cdist(Z, centers[-1]).pow(2).squeeze(1))
            return torch.cat(centers, dim=0)

        def ch_score(Z: torch.Tensor, assign: torch.Tensor, centers: torch.Tensor) -> float:
            # Calinski–Harabasz score
            # Z: (N,m); assign: (N,); centers: (K,m)
            K = centers.size(0)
            N = Z.size(0)
            mu = Z.mean(dim=0, keepdim=True)  # (1,m)
            # within scatter
            W = (Z - centers[assign]).pow(2).sum()
            # between scatter
            # counts
            counts = torch.bincount(assign, minlength=K).float().unsqueeze(1)  # (K,1)
            B = (counts * (centers - mu).pow(2)).sum()
            # CH = (B/(K-1)) / (W/(N-K))
            # Guard small denominators
            if K <= 1 or N <= K:
                return float('-inf')
            ch = (B / (K - 1 + 1e-12)) / (W / (N - K + 1e-12) + 1e-12)
            return float(ch.item())

        best = {
            "score": float('-inf'),
            "dims": None,
            "means_z": None,
        }

        K = S
        iters = max(1, int(kmeans_iters))
        for dims in candidates:
            Vsub = V[:, dims]  # (D,m)
            Z = Xc @ Vsub      # (N,m)
            if whiten:
                scale = Svals[dims].clamp_min(1e-8)
                Z = Z / scale

            # k-means in Z
            means_z = kpp_init(Z, K).clone()
            for _ in range(iters):
                # Assign
                assign = torch.cdist(Z, means_z).argmin(-1)
                counts = torch.bincount(assign, minlength=K)
                # Re-seed empties to farthest points
                empty = (counts == 0).nonzero(as_tuple=False).flatten()
                if empty.numel() > 0:
                    d2 = torch.min(torch.cdist(Z, means_z).pow(2), dim=1).values
                    far_idx = torch.topk(d2, k=empty.numel(), largest=True).indices
                    for j, e in enumerate(empty):
                        means_z[e] = Z[far_idx[j]]
                    assign = torch.cdist(Z, means_z).argmin(-1)
                    counts = torch.bincount(assign, minlength=K)
                # Update centers
                new_means = torch.zeros_like(means_z)
                new_means.scatter_add_(0, assign.view(-1, 1).expand(-1, means_z.size(1)), Z)
                means_z = new_means / counts.clamp_min(1).view(-1, 1)

            # Final assign and CH score
            assign = torch.cdist(Z, means_z).argmin(-1)
            score = ch_score(Z, assign, means_z)
            if score > best["score"]:
                best = {"score": score, "dims": dims, "means_z": means_z.clone()}

        # Use best subspace
        assert best["dims"] is not None and best["means_z"] is not None, "PCA auto-selection failed."
        dims = best["dims"]  # type: ignore[assignment]
        means_z = best["means_z"]  # (K,m)
        
        # TODO: CLEAN UP
        # Minimal, gated logging of the chosen PCA subspace (1-based PC indices)
        verbose = bool(getattr(self.global_config, "verbose", False))
        pcs_1based = [int(d) + 1 for d in dims]
        try:
            score_val = float(best.get("score", float("nan")))
        except Exception:
            score_val = float("nan")
        if verbose:
            print(f"[HMM:init] PCA k-means init using PCs {pcs_1based} (whiten={whiten}, CH={score_val:.3f})")
        # Persist for debugging/inspection
        
        try:
            self._pca_init_info = {"dims": dims, "dims_1based": pcs_1based, "score": score_val, "whiten": bool(whiten)}
        except Exception:
            pass
        Vsub = V[:, dims]
        Z = Xc @ Vsub
        if whiten:
            scale = Svals[dims].clamp_min(1e-8)
            Z = Z / scale
        assign = torch.cdist(Z, means_z).argmin(-1)  # (N,)

        # Original-space means and variances from assignments
        D = X.size(1)
        mu = torch.zeros(S, D, device=device)
        counts = torch.bincount(assign, minlength=S).clamp_min(1)
        mu.scatter_add_(0, assign.view(-1, 1).expand(-1, D), X)
        mu = mu / counts.view(-1, 1)
        self.emission_mean.copy_(mu)

        if self.covariance_type == "diag":
            resid = X - mu[assign]
            var = torch.zeros(S, D, device=device)
            var.scatter_add_(0, assign.view(-1, 1).expand(-1, D), resid.pow(2))
            var = (var / counts.view(-1, 1)).clamp_min(1e-6)
            self.emission_logvar.copy_(var.log())
        elif self.covariance_type == "full":
            # Estimate sample covariance per cluster, map to raw Cholesky params
            raw = torch.zeros_like(self.emission_cholesky_raw)
            I = torch.eye(D, device=device, dtype=X.dtype)
            for k in range(S):
                mask = (assign == k)
                nk = int(mask.sum().item())
                if nk <= 1:
                    # Fallback: diagonal from pooled variance of cluster
                    Xk = X[mask] if nk > 0 else X
                    var_k = Xk.var(dim=0, unbiased=False).clamp_min(1e-6)
                    Ck = torch.diag(var_k) + self.jitter * I
                else:
                    Xk = X[mask]
                    # Centered covariance
                    Xk_c = Xk - Xk.mean(0, keepdim=True)
                    # cov = (Xk_c^T Xk_c)/(nk-1)
                    Ck = (Xk_c.T @ Xk_c) / max(nk - 1, 1)
                    Ck = Ck + self.jitter * I
                # Cholesky and map to raw: off-diag copy, diag uses inverse softplus on (Ldiag - jitter)
                try:
                    Lk = torch.linalg.cholesky(Ck)
                except RuntimeError:
                    # Ensure PSD via small ridge if needed
                    Lk = torch.linalg.cholesky(Ck + 1e-4 * I)
                # Fill lower triangle
                raw[k].copy_(torch.tril(Lk))
                # Correct diagonals to be pre-softplus values: softplus(raw_diag) + jitter = L_diag
                diag_L = torch.diagonal(Lk)
                target = (diag_L - self.jitter).clamp_min(1e-8)
                diag_raw = torch.log(torch.expm1(target))
                raw[k].diagonal().copy_(diag_raw)
            self.emission_cholesky_raw.copy_(raw)
        else:
            # meanonly -> nothing to set for covariance
            pass

        # Transitions and initial distribution from per-time assignments
        if estimate_transitions:
            z = assign.view(B, T)  # (B,T), same flattening order as reshape(-1, D)
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
    def __init_sticky_em_warmstart(self, n_iters: int = 10, kappa: float = 3.0, min_var: float = 1e-6):
        """Run a few EM iterations with stickiness (κ added to self-transitions)."""
        data, _ = self.data_loader.get_all_data()
        x = self.__validate_input(data)                 # (B,T,D)
        B, T, D = x.shape
        S = self.num_states
        device = self.device

        for _ in range(n_iters):
            # ----- E-step: forward-backward -> gammas and xis -----
            log_pi = torch.log_softmax(self.initial_logits, dim=-1)        # (S,)
            log_A  = torch.log_softmax(self.transition_logits, dim=-1)     # (S,S)
            log_em = self.__emission_log_prob(x)                           # (B,T,S)

            # forward
            log_alpha = x.new_empty(B, T, S)
            log_alpha[:, 0, :] = log_pi + log_em[:, 0, :]
            for t in range(1, T):
                # alpha_t(i) = log_em + logsumexp_j alpha_{t-1}(j) + logA_{j->i}
                prev = log_alpha[:, t-1, :].unsqueeze(2) + log_A.unsqueeze(0)    # (B,S,S)
                log_alpha[:, t, :] = log_em[:, t, :] + torch.logsumexp(prev, dim=1)

            # backward
            log_beta = x.new_zeros(B, T, S)
            for t in range(T - 2, -1, -1):
                nxt = log_beta[:, t+1, :].unsqueeze(1) + log_em[:, t+1, :].unsqueeze(1) + log_A.unsqueeze(0)  # (B,S,S)
                log_beta[:, t, :] = torch.logsumexp(nxt, dim=2)

            # posteriors
            log_gamma = log_alpha + log_beta
            log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=2, keepdim=True)
            gamma = log_gamma.exp()                                         # (B,T,S)

            # pairwise posteriors ξ_t(i,j)
            xi = x.new_empty(B, T - 1, S, S)
            for t in range(T - 1):
                term = (log_alpha[:, t, :].unsqueeze(2) + log_A.unsqueeze(0)
                        + log_em[:, t+1, :].unsqueeze(1) + log_beta[:, t+1, :].unsqueeze(1))  # (B,S,S)
                term = term - torch.logsumexp(term.reshape(B, -1), dim=1, keepdim=True).unsqueeze(-1)
                xi[:, t, :, :] = term.exp()

            # ----- M-step: emissions -----
            # weights per state
            w = gamma.sum(dim=1)                                           # (B,S) -> per sequence
            N_s = w.sum(dim=0).clamp_min(1e-8)                              # (S,)

            # means
            x_expanded = x.unsqueeze(2)                                     # (B,T,1,D)
            mu_num = (gamma.unsqueeze(-1) * x_expanded).sum(dim=(0,1))      # (S,D)
            mu = mu_num / N_s.unsqueeze(1)
            self.emission_mean.copy_(mu)

            # covariances
            if self.covariance_type == "diag":
                diff2 = (x_expanded - mu.unsqueeze(0).unsqueeze(0)).pow(2)  # (B,T,S,D)
                var_num = (gamma.unsqueeze(-1) * diff2).sum(dim=(0,1))      # (S,D)
                var = (var_num / N_s.unsqueeze(1)).clamp_min(min_var)
                self.emission_logvar.copy_(var.log())
            elif self.covariance_type == "full":
                # simple weighted covariance (could be optimized if D large)
                raw = torch.zeros_like(self.emission_cholesky_raw)
                Xbt = x.reshape(B*T, D)
                G = gamma.reshape(B*T, S)                                   # (N,S)
                for s in range(S):
                    mu_s = mu[s]
                    diff = Xbt - mu_s
                    w_s = G[:, s].unsqueeze(1)
                    C = (w_s * diff).T @ diff / N_s[s]
                    C = C + self.jitter * torch.eye(D, device=device)
                    raw[s] = torch.linalg.cholesky(C)
                self.emission_cholesky_raw.copy_(raw)

            # ----- M-step: π and A with stickiness κ -----
            pi = gamma[:, 0, :].sum(0) + 1e-3
            pi = pi / pi.sum()

            A_num = xi.sum(dim=(0,1)) + kappa * torch.eye(S, device=device) + 1e-6
            A_den = A_num.sum(dim=1, keepdim=True)
            A = A_num / A_den

            self.initial_logits.copy_(pi.clamp_min(1e-12).log())
            self.transition_logits.copy_(A.clamp_min(1e-12).log())

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
