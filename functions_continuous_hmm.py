
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def forward(loga, logb, T, M, logpi):
    scaling = np.zeros(T)
    logalpha = np.empty((T, M))
    logalpha[0, :] = logpi[:] + logb[0, :]
    scaling[0] = np.logaddexp.reduce(logalpha[0, :])
    logalpha[0, :] -= scaling[0]
    for t in range(1, T):
        for j in range(M):
            logterms = [loga[i, j] + logalpha[t - 1, i] for i in range(M)]
            logalpha[t, j] = np.logaddexp.reduce(logterms) + logb[t, j]
        scaling[t] = np.logaddexp.reduce(logalpha[t, :])
        logalpha[t, :] -= scaling[t]
        
    return logalpha, scaling

def backward(loga, logb, T, M):
    logbeta = np.empty((T, M))
    logbeta[T - 1, :] = 0
    for t in range(T - 2, -1, -1):
        for i in range(M):
            logterms = [loga[i, j] + logb[t + 1, j] + logbeta[t + 1, j] for j in range(M)]
            logbeta[t, i] = np.logaddexp.reduce(logterms)

    return logbeta

def compute_gamma(logalpha, logbeta, T, M):
    loggamma = np.empty((T, M))
    for t in range(T):
        for i in range(M):
            loggamma[t, i] = logalpha[t, i] + logbeta[t, i] - np.logaddexp.reduce([
                logalpha[t, j] + logbeta[t, j] for j in range(M)
            ])

    return loggamma


def compute_xi(logalpha, logbeta, loga, logb, T, M):
    xi = np.empty((T - 1, M, M))
    for t in range(T - 1):
        logterms = []
        for i in range(M):
            for j in range(M):
                logterms.append(
                    logalpha[t, i] + loga[i, j] + logb[t + 1, j] + logbeta[t + 1, j]
                )
        log_denom = np.logaddexp.reduce(logterms)

        for i in range(M):
            for j in range(M):
                xi[t, i, j] = (
                    logalpha[t, i] + loga[i, j] + logb[t + 1, j] + logbeta[t + 1, j]
                    - log_denom
                )
    return xi

def compute_a(loggamma, logxi, T, M):
    loga = np.empty((M, M))
    for i in range(M):
        for j in range(M):
            loga[i, j] = (
                np.logaddexp.reduce([logxi[t, i, j] for t in range(T - 1)])
                - np.logaddexp.reduce([loggamma[t, i] for t in range(T - 1)])
            )
    return loga

def compute_b(mus, Sigmas, dataset, T, M):
    """
    mus: M x N mean returns (over time) per regime and asset
    Sigmas: M x N x N the covariance matrix per regime
    dataset: T x N matrix of returns

    returns: logB, T x M, where logB[t,k] = log p(r_t | s_t = k)
    """
    logb = np.empty((T, M))
    for t in range(T):
        for k in range(M):
            if np.isnan(mus[k, :]).any() or np.isnan(Sigmas[k, :, :]).any():
                raise AssertionError()
            logb[t, k] = multivariate_normal.logpdf(dataset[t, :], mus[k, :], Sigmas[k, :, :])
    return logb


def compute_mus_sigmas(loggamma, dataset, T, M, N, eps=1e-02):
    mus = np.empty((M, N))
    Sigmas = np.empty((M, N, N))
    for m in range(M):
        exp_gammas = np.empty(T)

        # compute mus
        exp_gammas = np.exp(loggamma[:, m])
        den = exp_gammas.sum()
        if den < eps:
            # state got no responsibility -> reinitialize it
            mus[m, :] = np.mean(dataset, axis=0) + np.random.normal(0, 0.01, size=N)
            Sigmas[m, :, :] = np.cov(dataset.T) + eps * np.eye(N)
            continue
        
        mus[m, :] = np.sum(exp_gammas[:, None] * dataset, axis=0) / den

        # compute Sigmas
        diffs = dataset - mus[m, :]
        Sigmas[m, :, :] = (diffs.T @ (diffs * exp_gammas[:, None])) / den

        # stabilize
        Sigmas[m, :, :] = (Sigmas[m, :, :] + Sigmas[m, :, :].T) / 2
        Sigmas[m, :, :] += eps * np.eye(N)

    return mus, Sigmas
    

def permute_model(pi, A, mus, Sigmas, perm):
    pi_r = np.zeros_like(pi)
    A_r = np.zeros_like(A)
    mus_r = np.zeros_like(mus)
    Sigmas_r = np.zeros_like(Sigmas)
    # perm[est_i] = true_i
    for est_i, true_i in enumerate(perm):
        pi_r[true_i] = pi[est_i]
    for est_i, true_i in enumerate(perm):
        for est_j, true_j in enumerate(perm):
            A_r[true_i, true_j] = A[est_i, est_j]
    for est_i, true_i in enumerate(perm):
        mus_r[true_i] = mus[est_i]
        Sigmas_r[true_i] = Sigmas[est_i]
    return pi_r, A_r, mus_r, Sigmas_r

def kl_gaussian(mu0, S0, mu1, S1, eps=1e-8):
    # KL(N0 || N1)
    d = mu0.shape[0]
    # regularize
    S0 = S0 + eps * np.eye(d)
    S1 = S1 + eps * np.eye(d)
    invS1 = np.linalg.inv(S1)
    diff = (mu1 - mu0).reshape(-1, 1)
    term_trace = np.trace(invS1 @ S0)
    term_quad = float(diff.T @ invS1 @ diff)
    # sign and natural log of determinant
    sign0, logdet0 = np.linalg.slogdet(S0)
    sign1, logdet1 = np.linalg.slogdet(S1)
    logdet_ratio = logdet1 - logdet0
    # print('KL GAUSSIAN, term_trace:', term_trace)
    # print('KL GAUSSIAN, term_quad:', term_quad)
    # print('KL GAUSSIAN, logdet_ratio:', logdet_ratio)
    return 0.5 * (term_trace + term_quad - d + logdet_ratio)

def sym_kl(mu0, S0, mu1, S1, eps=1e-8):
    return kl_gaussian(mu0, S0, mu1, S1, eps) + kl_gaussian(mu1, S1, mu0, S0, eps)

def match_states_by_gaussians(mus_est, Sigmas_est, mus_true, Sigmas_true, eps=1e-8):
    M = mus_est.shape[0]
    cost = np.zeros((M, M))
    # print('MU EST'); print(mus_est)
    # print('SIGMAS EST'); print(Sigma_est)
    # print('MU TRUE'); print(mus_true)
    # print('SIGMAS TRUE'); print(Sigmas_true)
    for i in range(M):
        for j in range(M):
            cost[i, j] = sym_kl(mus_est[i], Sigmas_est[i], mus_true[j], Sigmas_true[j], eps)
    print('COST MATRIX:'); print(cost)
    row_ind, col_ind = linear_sum_assignment(cost)
    # build perm such that perm[est_i] = true_j
    perm = np.zeros(M, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[r] = c
    return perm


def viterbi(logpi, logb, loga, T, M):
    logdelta = np.zeros((T, M))
    psi = np.zeros((T, M))
    # logdelta[t, j] = log probability of the best path ending in state j at time t
    # psi[t, j] = state that maximizes prob of having been there, over all seq
    # ended in j at time t
    logdelta[0, :] = logpi + logb[0, :]
    for t in range(1, T):
        for j in range(M):
            seq_probs = logdelta[t - 1, :] + loga[:, j]
            logdelta[t, j] = np.max(seq_probs) + logb[t, j]
            psi[t, j] = np.argmax(seq_probs)

    states = np.zeros(T, dtype=int) # holds most likely states for each time
    states[-1] = np.argmax(logdelta[-1, :])
    p = np.max(logdelta[-1, :]) # highest probability over all states (ended in T)

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
        # S_{T-1}=argmax over all j of (logdelta[T - 1, j])

    return states, p