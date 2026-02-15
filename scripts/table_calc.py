import numpy as np
from scipy import stats

# nmi_values = [0.2258, 0.3136, 0.4544, 0.1997, 0.4677, 0.1238, 0.3232, 0.3874, 0.3094]
# nmi_values = [0.3195, 0.3409]
nmi_values = [0.0227, 0.0225, 0.0222, 0.0227, 0.0131, 0.0129, 0.0222, 0.0122, 0.0121]

likelihood_values = [0.99997, 0.99997, 0.99996, 0.99999, 0.99998, 0.99997, 0.99997, 0.99996, 0.99999, 0.99998]

mean_nmi = np.mean(nmi_values)
sem_nmi = stats.sem(nmi_values)

mean_likelihood = np.mean(likelihood_values)
sem_likelihood = stats.sem(likelihood_values)

print(f"NMI: Mean = {mean_nmi:.4f}, SEM = {sem_nmi:.4f}")
print(f"Likelihood: Mean = {mean_likelihood:.6f}, SEM = {sem_likelihood:.6f}")