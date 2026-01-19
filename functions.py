
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

def forward(loga, logb, T, M, logpi):
    scaling = np.zeros(T)
    logalpha = np.empty((T, M))
    logalpha[0, :] = logpi[:] + logb[0, :]
    scaling[0] = np.logaddexp.reduce(logalpha[0, :])
    for t in range(1, T):
        for j in range(M):
            logterms = [loga[i, j] + logalpha[t - 1, i] for i in range(M)]
            logalpha[t, j] = np.logaddexp.reduce(logterms) + logb[t, j]
        scaling[t] = np.logaddexp.reduce(logalpha[t, :])
        logalpha[t, :] -= scaling[t]
        
    return logalpha, scaling

def backward(loga, logb, T, M, scaling):
    logbeta = np.empty((T, M))
    logbeta[T - 1, :] = 0
    for t in range(T - 2, -1, -1):
        for i in range(M):
            logterms = [loga[i, j] + logb[t + 1, i] + logbeta[t + 1, j] for j in range(M)]
            logbeta[t, i] = np.logaddexp.reduce(logterms)
        logbeta[t, :] -= scaling[t]

    return logbeta

def backward_obs(loga, logb, T, M, scaling, obs):
    logbeta = np.empty((T, M))
    logbeta[T - 1, :] = 0
    for t in range(T - 2, -1, -1):
        for i in range(M):
            logterms = [loga[i, j] + logb[i, obs[t + 1]] + logbeta[t + 1, j] for j in range(M)]
            logbeta[t, i] = np.logaddexp.reduce(logterms)
        logbeta[t, :] -= scaling[t]

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
    xi = np.empty((T, M, M))
    for t in range(T - 1):
        for i in range(M):
            for j in range(M):
                logterms = []
                for k in range(M):
                    for l in range(M):
                        logterms.append(
                            logalpha[t, k] + loga[k, l] + logb[l, obs[t + 1]] + logbeta[t + 1, l]
                        )
                xi[t, i, j] = (
                    logalpha[t, i] + loga[i, j] + logb[j, obs[t + 1]] + logbeta[t + 1, j]
                    - np.logaddexp.reduce(logterms)
                )
    return xi

def compute_xi(logalpha, logbeta, loga, logb, T, M):
    xi = np.empty((T, M, M))
    for t in range(T - 1):
        for i in range(M):
            for j in range(M):
                logterms = []
                for k in range(M):
                    for l in range(M):
                        logterms.append(
                            logalpha[t, k] + loga[k, l] + logb[t + 1, l] + logbeta[t + 1, l]
                        )
                xi[t, i, j] = (
                    logalpha[t, i] + loga[i, j] + logb[t + 1, j] + logbeta[t + 1, j]
                    - np.logaddexp.reduce(logterms)
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
    print('RECEIVED:')
    print(loggamma.shape)
    print(dataset.shape)
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

def match_states_by_B(B_est, B_true):
    # cost = L2 distance between emission distributions (rows)
    cost = cdist(B_est, B_true, metric='euclidean')  # shape (N_est, N_true)
    row_ind, col_ind = linear_sum_assignment(cost)
    # row_ind[i] -> col_ind[i], we want a permutation array perm where perm[est_index] = true_index
    perm = np.empty(B_est.shape[0], dtype=int)
    perm[row_ind] = col_ind
    return perm

def permute_model_disc(pi, A, B, perm):
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

def viterbi_disc(logpi, logb, loga, obs, T):

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



def k_means(data, k):
    # data: np-array of vectors (points), shape (n_vectors, vector_dim)
    # k: amount of clusters

    # data size: (n_points, vector_dim)

    center_indices = np.random.randint(0, len(data), size=(1,k)) # (1, k)
    centers = np.array([data[i] for i in center_indices]).reshape((k, data.shape[1])) # (k, vector_dim)
    
    clusters = [[] for _ in range(k)] # (k, n_vectors_to_cluster, vector_dim)
    
    while True:
        new_clusters = clusters
        """
        for every point: determine dist to every cluster
        categorize points based on dist (take smallest)
        redetermine clusters
        """

        labels = [] # length n_vectors

        # determine clusters
        for vector in data:
            # vector: (1, vector_dim)

            # distances for vector to every cluster vector
            # (1, k)

            distances = np.array([np.linalg.norm(vector - core) for core in centers])
            idx = np.argmin(distances)
            new_clusters[idx].append(vector) # list of length k
            labels.append(int(idx))

        if new_clusters == clusters:
            return clusters, centers, labels

        # determine new centers
        for k_idx, cluster in enumerate(clusters):
            # cluster: list of vectors
            avg_vector = np.zeros(data.shape[1])
            for vec in cluster:
                # vec: (vector_dim)
                avg_vector += vec
            avg_vector /= len(cluster)
            centers[k_idx,:] = avg_vector


def generate_volatility(V0, xi, kappa, theta, N, dt):
    V = np.empty(N)
    V[0] = V0
    for i in range(1, N):
        dw_v = np.sqrt(dt) * np.random.normal()
        sigma = max(0, V[i-1] + kappa * (theta - V[i-1]) * dt + xi * np.sqrt(V[i-1]) * dw_v)
        V[i] = sigma
    return V

def generate_prices(vola_values, S0, mu, dt, N):
    S = np.empty(N)
    S[0] = S0
    for i in range(1, N):
        dw_s = np.sqrt(dt) * np.random.normal()
        s = S[i-1] * np.exp((mu - 0.5 * vola_values[i]) * dt + np.sqrt(vola_values[i]) * dw_s)
        S[i] = s
    return S
