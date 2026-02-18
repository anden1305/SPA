import numpy as np

data = np.load("SPA/results/substages/reliability/substages_vae_15_reliability_3_validate/plots/results.npz")

y_hat = data['y_hat']

np.savetxt("y_hat_3.csv", y_hat, delimiter=",")