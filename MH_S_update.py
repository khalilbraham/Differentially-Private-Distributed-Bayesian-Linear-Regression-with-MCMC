import numpy as np
from scipy.stats import wishart
from scipy.linalg import cholesky

def MH_S_update(S, N, S_obs, Z_obs, theta, Sigma, var_y, var_S, var_Z, a):
    # prepare the indices of the upper diagonal
    d = len(theta)
    upper_tria_ind = np.triu_indices(d)

    # calculate the log posterior for the initial S
    cov_Z = S * var_y + np.eye(d) * var_Z
    log_det_cov_Z = 2 * np.sum(np.log(np.diag(cholesky(cov_Z))))
    log_det_S = 2 * np.sum(np.log(np.diag(cholesky(S))))
    log_Z_S = -0.5 * (log_det_cov_Z + np.dot((Z_obs - np.dot(S, theta)).T, np.linalg.solve(cov_Z, (Z_obs - np.dot(S, theta)))))
    log_S_g_Sigma = log_det_S * ((N - d - 1) / 2) - np.trace(np.linalg.solve(Sigma, S)) / 2
    log_S_noise = -0.5 * np.sum((S_obs[upper_tria_ind] - S[upper_tria_ind]) ** 2) / var_S
    log_pi = log_Z_S + log_S_noise + log_S_g_Sigma

    # Proposal
    S_prop = wishart.rvs(df=a, scale=S/a)

    # calculate the acceptance ratio
    cov_Z_prop = S_prop * var_y + np.eye(d) * var_Z
    log_det_cov_Z_prop = 2 * np.sum(np.log(np.diag(cholesky(cov_Z_prop))))
    log_det_S_prop = 2 * np.sum(np.log(np.diag(cholesky(S_prop))))
    log_Z_S_prop = -0.5 * (log_det_cov_Z_prop + np.dot((Z_obs - np.dot(S_prop, theta)).T, np.linalg.solve(cov_Z_prop, (Z_obs - np.dot(S_prop, theta)))))
    log_S_prop_noise = -0.5 * np.sum((S_obs[upper_tria_ind] - S_prop[upper_tria_ind]) ** 2) / var_S
    log_S_prop_g_Sigma = log_det_S_prop * ((N - d - 1) / 2) - np.trace(np.linalg.solve(Sigma, S_prop)) / 2
    log_pi_prop = log_Z_S_prop + log_S_prop_noise + log_S_prop_g_Sigma

    # log proposal
    log_prop = (log_det_S - log_det_S_prop) * (a - (d + 1) / 2) + (np.trace(np.linalg.solve(S, S_prop)) - np.trace(np.linalg.solve(S_prop, S))) * (a / 2)

    # calculate the log-acceptance rate
    log_r = log_pi_prop - log_pi + log_prop
    log_r = min(log_r, 0)
    decision = np.random.rand() < np.exp(log_r)

    if decision:
        S = S_prop

    return S, decision