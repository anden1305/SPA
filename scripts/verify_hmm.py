"""Minimal HMM correctness verification script.

Run to produce a short printable report you can show your professor.

Checks:
 1. Forward log-likelihood vs brute-force enumeration (tiny sequence).
 2. Forward vs forward-backward log-likelihood consistency.
 3. Viterbi joint path probability <= full log-likelihood.
 4. Gamma posterior rows sum to ~1.
 5. Optional finite-difference gradient sanity check (one parameter).
 6. Optional synthetic recovery (generate from known params, fit, compute mean error).

Usage (examples):
  python scripts/verify_hmm.py                 # run core tiny checks
  python scripts/verify_hmm.py --grad-check    # include finite diff gradient test
  python scripts/verify_hmm.py --recover       # include synthetic recovery demo

All computations are tiny and quick.
"""
from __future__ import annotations
import argparse, math, itertools
import torch
from pathlib import Path
import sys, pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))
from scr.models.hmm import HMM

def finite_difference_grad(model: HMM, x: torch.Tensor, param: torch.Tensor, idx: tuple[int,...], eps: float=1e-4):
    with torch.no_grad():
        orig = param[idx].item()
        param[idx] = orig + eps
    lp_plus = model(x).item()
    with torch.no_grad():
        param[idx] = orig - eps
    lp_minus = model(x).item()
    with torch.no_grad():
        param[idx] = orig
    num = (lp_plus - lp_minus)/(2*eps)
    # Autograd
    model.zero_grad(set_to_none=True)
    model(x).backward()
    auto = param.grad[idx].item()
    rel_err = abs(num-auto)/(abs(auto)+1e-8)
    return num, auto, rel_err

def synthetic_generate(S=3, T=200, D=2, seed=0):
    """Generate a toy HMM sequence with uniform transitions & unit variance.

    Returns
    -------
    x : (T,D) observations
    states : (T,) int64 latent states
    means : (S,D) emission means used
    logvars : (S,D) emission log-variances (zeros -> unit variance)
    """
    g = torch.Generator().manual_seed(seed)
    # Uniform initial & transition probabilities
    pi_probs = torch.full((S,), 1.0 / S)
    A_probs = torch.full((S, S), 1.0 / S)
    means = torch.randn(S, D, generator=g)
    logvars = torch.zeros(S, D)
    states_list: list[int] = []
    # Sample initial state
    z = torch.multinomial(pi_probs, 1, generator=g).item()
    states_list.append(z)
    for _ in range(T - 1):
        z = torch.multinomial(A_probs[z], 1, generator=g).item()
        states_list.append(z)
    states = torch.tensor(states_list, dtype=torch.long)
    # Sample emissions (unit variance)
    x = torch.randn(T, D, generator=g) + means[states]
    return x, states, means, logvars

def fit_and_recover(S=3, T=200, D=2, steps=1000, lr=1e-2, seed=1):
    x, z_true, means_true, logvars_true = synthetic_generate(S=S, T=T, D=D, seed=seed)
    model = HMM(num_states=S, obs_dim=D, covariance_type='diag', normalize_time=False, device='cpu')
    model.reset_parameters_random(x, mean_std=0.1, cov_noise_std=0.0, init_logits_std=0.01)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = -model(x).mean()
        loss.backward()
        opt.step()
    # Align states by Hungarian on mean distances
    with torch.no_grad():
        from math import inf
        cost = torch.cdist(model.emission_mean, means_true)  # (S,S)
        # Naive Hungarian (S small) via permutation search
        best = None; best_cost = float('inf')
        for perm in itertools.permutations(range(S)):
            c = sum(cost[i, p].item() for i,p in enumerate(perm))
            if c < best_cost:
                best_cost = c; best = perm
        perm = torch.tensor(best)
        mean_err = torch.mean((model.emission_mean - means_true[perm])**2).sqrt().item()
    return mean_err, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--grad-check', action='store_true')
    ap.add_argument('--recover', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    # Tiny sequence (T small enough for enumeration)
    S, T, D = 3, 5, 2
    model = HMM(num_states=S, obs_dim=D, covariance_type='diag', normalize_time=False, device='cpu')
    # Use random init then small noise to avoid symmetry
    x = torch.randn(T, D)
    model.reset_parameters_random(x, mean_std=0.2, cov_noise_std=0.0, init_logits_std=0.05)
    report = model.verify_tiny(x, atol=1e-5, verbose=False)
    print("=== Tiny Consistency Report ===")
    for k in sorted(report):
        print(f"{k}: {report[k]:.6f}")
    # Simple pass/fail summary
    ok = (report['abs_diff_forward_bruteforce'] < 1e-5 and report['abs_diff_forward_fb'] < 1e-5 and report['forward_minus_viterbi_joint'] >= -1e-6 and report['gamma_row_sums_min'] > 0.999 and report['gamma_row_sums_max'] < 1.001)
    print(f"PASS_TINY_CHECKS: {ok}")

    if args.grad_check:
        print("\n=== Gradient Finite Difference ===")
        p = model.emission_mean
        idx = (0,0)
        num, auto, rel = finite_difference_grad(model, x, p, idx)
        print(f"param emission_mean{idx}: num={num:.6e} auto={auto:.6e} rel_err={rel:.3e}")

    if args.recover:
        print("\n=== Synthetic Recovery (parameter RMS error) ===")
        mean_err, _ = fit_and_recover()
        print(f"Recovered mean RMS error: {mean_err:.4f}")

if __name__ == '__main__':
    main()
