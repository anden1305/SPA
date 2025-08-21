import math
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from .base_model import BaseModel

class HMM(BaseModel):
    
    def __init__(
        self,
        num_states: int,
        num_features: int,
        covariance_type: str = "diag",
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.__initialize_arguments(num_states, num_features, covariance_type, device)
        self.__initialize_parameters()


    # ------------------------- Initialization -------------------------
    
    def __initialize_arguments(self, num_states: int, num_features: int, covariance_type: str, device: Optional[str | torch.device] = None) -> None:
        self.num_states = int(num_states)
        self.num_features = int(num_features)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.covariance_type = covariance_type.lower()
        if self.covariance_type not in {"diag", "full", "meanonly"}:
            raise ValueError("covariance_type must be one of {'diag','full','meanonly'}")

    def __initialize_parameters(self):
        self.initial_logits = nn.Parameter(torch.zeros(self.num_states))
        self.transition_logits = nn.Parameter(torch.zeros(self.num_states, self.num_states))
        self.emission_mean = nn.Parameter(torch.zeros(self.num_states, self.num_features))
        if self.covariance_type == "diag":
            self.emission_logvar = nn.Parameter(torch.zeros(self.num_states, self.num_features))
        elif self.covariance_type == "full":
            self.emission_cholesky_raw = nn.Parameter(torch.zeros(self.num_states, self.num_features, self.num_features))
        else:
            self.register_buffer("_identity_cov", torch.eye(self.num_features))
        self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi), dtype=torch.get_default_dtype()))
        self.to(self.device)



    # ------------------------- Helper accessors -------------------------
    
    def __log_initial(self) -> Tensor:
        return torch.log_softmax(self.initial_logits, dim=-1)

    def __log_transition(self) -> Tensor:
        return torch.log_softmax(self.transition_logits, dim=-1)

    def __full_cov_cholesky(self) -> Tensor:
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

    def __emission_log_prob(self, x: Tensor) -> Tensor:
        
        if x.dim() != 3:
            raise ValueError(f"emission_log_prob expects (B,T,D) after any flattening; got {tuple(x.shape)}")
        if x.shape[2] != self.num_features:
            raise ValueError(f"Input feature dimension {x.shape[2]} != model obs_dim {self.num_features}. If you passed (B,T,C,F) ensure obs_dim=C*F when constructing the model.")
        B, T, D = x.shape
        S = self.num_states
        diff = x.unsqueeze(2) - self.emission_mean

        if self.covariance_type == "diag":
            logvar = self.emission_logvar
            inv_var = torch.exp(-logvar)
            log_prob = -0.5 * (diff.pow(2) * inv_var + logvar + self._log_2pi)
            return log_prob.sum(-1)
        elif self.covariance_type == "meanonly":
            quad = diff.pow(2).sum(-1)
            const = D * self._log_2pi
            return -0.5 * (quad + const)
        else:
            L = self.__full_cov_cholesky()
            log_det = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
            log_probs = []
            for s in range(S):
                Ls = L[s]
                d_flat = diff[:, :, s, :].reshape(B * T, D).T
                y = torch.linalg.solve_triangular(Ls, d_flat, upper=False)
                m_dist2 = (y.pow(2).sum(0)).reshape(B, T)
                lp = -0.5 * (m_dist2 + log_det[s] + D * self._log_2pi)
                log_probs.append(lp.unsqueeze(-1))
            return torch.cat(log_probs, dim=-1)



    # ----------------------- Core algorithms ------------------------
    
    def __forward_algorithm(self, log_emiss: Tensor, log_pi: Tensor, log_A: Tensor) -> Tensor:
        B, T, S = log_emiss.shape
        alpha = log_pi.unsqueeze(0) + log_emiss[:, 0, :]
        for t in range(1, T):
            prev = alpha.unsqueeze(2) + log_A.unsqueeze(0)
            alpha = log_emiss[:, t, :] + torch.logsumexp(prev, dim=1)
        return torch.logsumexp(alpha, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        log_pi = self.__log_initial()
        log_A = self.__log_transition()
        log_emiss = self.__emission_log_prob(x)
        logp = self.__forward_algorithm(log_emiss, log_pi, log_A)
        return logp


    # ------------------------- Convenience API aliases -------------------------

    @torch.no_grad()
    def log_prob(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        return self.decode_viterbi(x)

    def extra_repr(self) -> str:
        return (
            f"states={self.num_states}, features={self.num_features}, cov='{self.covariance_type}'"
        )

    @torch.no_grad()
    def decode_viterbi(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        log_pi = self.__log_initial()
        log_A = self.__log_transition()
        log_emiss = self.__emission_log_prob(x)
        backptr = x.new_zeros((B, T, self.num_states), dtype=torch.long)
        delta = log_pi.unsqueeze(0) + log_emiss[:, 0, :]
        for t in range(1, T):
            scores = delta.unsqueeze(2) + log_A.unsqueeze(0)
            delta, idx = torch.max(scores, dim=1)
            delta = delta + log_emiss[:, t, :]
            backptr[:, t, :] = idx
        last = torch.argmax(delta, dim=1)
        path = x.new_zeros((B, T), dtype=torch.long)
        path[:, -1] = last
        for t in range(T - 2, -1, -1):
            path[:, t] = backptr[torch.arange(B), t + 1, path[:, t + 1]]
        return path