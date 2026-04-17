import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def forward_obs(loga, logb, T, M, logpi, obs):
    logalpha = np.empty((T, M))
    scaling = np.zeros(T)

    # Initialization
    for j in range(M):
        logalpha[0, j] = logpi[j] + logb[j, obs[0]]
    scaling[0] = np.logaddexp.reduce(logalpha[0, :])
    logalpha[0, :] -= scaling[0]

    # Recursion
    for t in range(1, T):
        for j in range(M):
            terms = [loga[i, j] + logalpha[t-1, i] for i in range(M)]
            logalpha[t, j] = np.logaddexp.reduce(terms) + logb[j, obs[t]]
        scaling[t] = np.logaddexp.reduce(logalpha[t, :])
        logalpha[t, :] -= scaling[t]

    return logalpha, scaling

def backward_obs(loga, logb, T, M, obs):
    logbeta = np.empty((T, M))
    logbeta[T - 1, :] = 0
    for t in range(T - 2, -1, -1):
        for i in range(M):
            logterms = [loga[i, j] + logb[j, obs[t + 1]] + logbeta[t + 1, j] for j in range(M)]
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

def compute_xi_obs(logalpha, logbeta, loga, logb, obs, T, M):
    xi = np.empty((T-1, M, M))  # T-1!
    for t in range(T-1):
        # Marginal denominator (same as gamma[t])
        log_denom = np.logaddexp.reduce([logalpha[t, j] + logbeta[t, j] for j in range(M)])

        for i in range(M):
            for j in range(M):
                xi[t, i, j] = logalpha[t, i] + loga[i, j] + logb[j, obs[t+1]] + logbeta[t+1, j] - log_denom
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

def compute_b_obs(loggamma, obs, T, M, N):
    logb = np.empty((N, M))
    for i in range(M):
        den = loggamma[:, i]
        for j in range(N):
            num = [loggamma[t, i] for t in range(T) if obs[t] == j]
            if num:
                logb[j, i] = np.logaddexp.reduce(num) - np.logaddexp.reduce(den)
            else:
                logb[j, i] = -np.logaddexp.reduce(den)

    return logb


def match_states_by_B(B_est, B_true):
    # cost = L2 distance between emission distributions (rows)
    cost = cdist(B_est, B_true, metric='euclidean')  # shape (N_est, N_true)
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind[i] -> col_ind[i], we want a permutation array perm where perm[est_index] = true_index
    perm = np.empty(B_est.shape[0], dtype=int)
    perm[row_ind] = col_ind
    return perm

def permute_model(pi, A, B, perm):
    # perm maps estimated-index -> true-index
    # We return model reordered so index i now corresponds to true index perm[i].
    # To compare, we need inverse perm that gives mapping: new_index -> old_index
    # Simpler: build arrays aligned to true indices
    N = len(perm)
    pi_reordered = np.zeros_like(pi)
    A_reordered = np.zeros_like(A)
    B_reordered = np.zeros_like(B)
    for est_i, true_i in enumerate(perm):
        pi_reordered[true_i] = pi[est_i]
    for est_i, true_i in enumerate(perm):
        for est_j, true_j in enumerate(perm):
            A_reordered[true_i, true_j] = A[est_i, est_j]
    for est_i, true_i in enumerate(perm):
        B_reordered[true_i] = B[est_i]
    return pi_reordered, A_reordered, B_reordered

def viterbi(logpi, logb, loga, obs, T):

    N = loga.shape[0]
    T = len(obs)

    # initialization
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    delta[0] = logpi + logb[:, obs[0]]

    # recursion
    for t in range(1, T):
        for j in range(N):
            seq_probs = delta[t-1] + loga[:, j]
            psi[t, j] = np.argmax(seq_probs)
            delta[t, j] = np.max(seq_probs) + logb[j, obs[t]]

    # termination
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])

    # backtrack
    for t in range(T-2, -1, -1):
        states[t] = psi[t+1, states[t+1]]

    return states, np.max(delta[-1])