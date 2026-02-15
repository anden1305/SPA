import numpy as np

nmi = [0.3083, 0.2697, 0.1179, 0.3195, 0.3409, 0.1196, 0.3582, 0.2875, 0.2495]
likelihood = [0.7018, 0.7112, 0.683, 0.7107, 0.613, 0.6146, 0.5952, 0.5961, 0.5647, 0.5646, 0.5035, 0.504, 0.7208, 0.633, 0.5348, 0.5358, 0.4566, 0.6468]

nmi_mean, nmi_sem = np.mean(nmi), np.std(nmi, ddof=1) / np.sqrt(len(nmi))
lik_mean, lik_sem = np.mean(likelihood), np.std(likelihood, ddof=1) / np.sqrt(len(likelihood))

print(f"NMI: {nmi_mean:.4f} ± {nmi_sem:.4f}")
print(f"Likelihood: {lik_mean:.4f} ± {lik_sem:.4f}")
