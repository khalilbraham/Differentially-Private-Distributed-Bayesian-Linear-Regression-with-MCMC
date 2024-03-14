import numpy as np
from closest_psd import closest_psd

def bayes_fixedS_fast(S_obs, Z_obs, DP_params, hyperparams):
    var_Z = hyperparams['var_Z']
    C = hyperparams['C']
    d = S_obs[0].shape[0]
    J = len(S_obs)
    bound_y = DP_params['bound_y']
    var_y = bound_y / 3

    Sigma_inv = np.linalg.inv(C)
    Sigma_mean_post_theta = 0

    for j in range(J):
        # find the nearest psd matrix and construct a pd matrix close to S_obs
        S0 = closest_psd(S_obs[j])
        S0 = (S0 + S0.T) / 2

        Sigma_inv = Sigma_inv + S0 @ np.linalg.inv(S0 * var_y + np.eye(d) * var_Z) @ S0
        Sigma_mean_post_theta = Sigma_mean_post_theta + S0 @ np.linalg.inv(S0 * var_y + np.eye(d) * var_Z) @ Z_obs[j]

    mu_theta = np.linalg.solve(Sigma_inv, Sigma_mean_post_theta)
    cov_theta = np.linalg.inv(Sigma_inv)

    return mu_theta, cov_theta