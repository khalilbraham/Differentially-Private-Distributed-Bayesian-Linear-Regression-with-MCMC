import numpy as np
from scipy.stats import invwishart, multivariate_normal
from MH_S_update import MH_S_update

def MCMC_normalX(S_obs, Z_obs, init_vars, hyperparams, prop_params, N_node, K):
    # hyperparameters
    m = hyperparams['m']
    C = hyperparams['C']
    a = hyperparams['a']
    b = hyperparams['b']
    var_S = hyperparams['var_S']
    var_Z = hyperparams['var_Z']
    Lambda = hyperparams['Lambda']
    kappa = hyperparams['kappa']

    # proposal parameters
    sigma_q_y = prop_params['sigma_q_y']
    a_MH = prop_params['a_MH']

    J = len(N_node)
    d = len(m)

    # initial variables
    theta = init_vars['theta']
    var_y = init_vars['var_y']
    S = init_vars['S']
    S_total = np.zeros((d, d))
    for j in range(J):
        S_total += S[j]

    # initialize the arrays
    theta_vec = np.zeros((d, K))
    var_y_vec = np.zeros(K)

    S_vec = [np.zeros((d, d, K)) for _ in range(J)]
    Sigma_vec = np.zeros((d, d, K))
    decision_rate_vecs = np.zeros(K)

    update_a_burn_in = K // 10
    S_acc = np.zeros(J)
    a_MH_vec = np.zeros((K, J))
    alpha_ast = 0.1

    # Calculate the inverses needed in the iterations
    C_inv = np.linalg.inv(C)

    for i in range(K):
        gamma_s = (i+1) ** (-0.6)
        gamma_a = (i+1) ** (-0.6)

        # update Sigma
        Sigma = invwishart.rvs(df=kappa + sum(N_node), scale=S_total + Lambda)
        decision_vec = np.zeros(J)
        S_total = np.zeros((d, d))

        # update Ss
        Sigma_inv = 0
        Sigma_mean_post_theta = 0

        for j in range(J):
            # wishart proposal
            S[j], decision = MH_S_update(S[j], N_node[j], S_obs[j], Z_obs[j], theta, Sigma, var_y, var_S, var_Z, a_MH[j])

            # update a_MH
            S_acc[j] = (1 - gamma_s) * S_acc[j] + gamma_s * decision
            if i > update_a_burn_in:
                a_MH[j] = np.exp(np.log(a_MH[j]) - gamma_a * (S_acc[j] - alpha_ast))
            decision_vec[j] = decision
        

        S_total = np.sum(S[j] for j in range(J))

        # update theta
        Sigma_inv = np.sum(S[j].T @ np.linalg.inv(S[j] * var_y + np.eye(d) * var_Z) @ S[j] for j in range(J)) + C_inv
        Sigma_mean_post_theta = np.sum(S[j].T @ np.linalg.inv(S[j] * var_y + np.eye(d) * var_Z) @ Z_obs[j] for j in range(J))

        Sigma_post_theta = np.linalg.inv(Sigma_inv)
        Sigma_post_theta = (Sigma_post_theta + Sigma_post_theta.T) / 2
        mean_post_theta = Sigma_post_theta @ (Sigma_mean_post_theta + C_inv @ m)
        theta = multivariate_normal.rvs(mean=mean_post_theta, cov=Sigma_post_theta)

        # update var_y
        var_y_prop = var_y + sigma_q_y * np.random.randn()

        if var_y_prop > 0:
            # log-likelihood
            log_Z_S = 0
            log_Z_S_prop = 0
            for j in range(J):
                cov_Z = S[j] * var_y + np.eye(d) * var_Z
                cov_Z_prop = S[j] * var_y_prop + np.eye(d) * var_Z

                log_det_cov_Z = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(cov_Z))))
                log_det_cov_Z_prop = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(cov_Z_prop))))

                u = Z_obs[j] - S[j] @ theta
                log_Z_S -= 0.5 * (log_det_cov_Z + u.T @ np.linalg.solve(cov_Z, u))
                log_Z_S_prop -= 0.5 * (log_det_cov_Z_prop + u.T @ np.linalg.solve(cov_Z_prop, u))

            # prior density
            log_prior = -(a + 1) * np.log(var_y) - b / var_y
            log_prior_prop = -(a + 1) * np.log(var_y_prop) - b / var_y_prop

            log_r = log_Z_S_prop - log_Z_S + log_prior_prop - log_prior

            decision = np.random.rand() < np.exp(log_r)
            if decision == 1:
                var_y = var_y_prop

        # store the variables
        theta_vec[:, i] = theta
        var_y_vec[i] = var_y

        Sigma_vec[:, :, i] = Sigma
        for j in range(J):
            S_vec[j][:, :, i] = S[j]
        a_MH_vec[i, :] = a_MH
        decision_rate_vecs[i] = np.mean(decision_vec)

    # store the outputs
    outputs = {'theta_vec': theta_vec, 'var_y_vec': var_y_vec}
    outputs['a_MH_vec'] = a_MH_vec
    outputs['S_vec'] = S_vec
    outputs['acceptance_rate_vec'] = decision_rate_vecs
    outputs['Sigma_vec'] = Sigma_vec

    return outputs