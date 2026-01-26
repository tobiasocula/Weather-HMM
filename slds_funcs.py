import numpy as np
from scipy.stats import multivariate_normal

def kalman_filter_regime(y, F, Q, H, R, x0, P0):
    T = y.shape[0]
    n = x0.shape[0]
    M = y.shape[1]  # Dimension of observation

    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))
    loglik = np.zeros(T)

    x_prev = x0
    P_prev = P0

    # Ensure P0 is positive definite
    P_prev = 0.5 * (P_prev + P_prev.T)
    eigenvalues, eigenvectors = np.linalg.eigh(P_prev)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    P_prev = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Ensure Q is positive definite
    Q = 0.5 * (Q + Q.T)
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    Q = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Ensure R is positive definite
    R = 0.5 * (R + R.T)
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    R = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    for t in range(T):
        # Predict
        x_p = F @ x_prev
        P_p = F @ P_prev @ F.T + Q

        # Ensure P_p is symmetric and positive definite
        P_p = 0.5 * (P_p + P_p.T)
        eigenvalues, eigenvectors = np.linalg.eigh(P_p)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        P_p = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Innovation
        y_t = y[t]
        y_pred = H @ x_p
        S = H @ P_p @ H.T + R

        # Symmetrize S
        S = 0.5 * (S + S.T)

        # Eigenvalue correction to ensure positive definiteness
        eigenvalues, eigenvectors = np.linalg.eigh(S)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        S = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Clip extreme values in S
        S = np.clip(S, -1e6, 1e6)

        # Kalman gain (use pseudoinverse)
        K = P_p @ H.T @ np.linalg.inv(S)

        # Update
        innov = y_t - y_pred
        x_f = x_p + K @ innov
        P_f = (np.eye(n) - K @ H) @ P_p

        # Ensure P_f is symmetric and positive definite
        P_f = 0.5 * (P_f + P_f.T)
        eigenvalues, eigenvectors = np.linalg.eigh(P_f)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        P_f = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Log-likelihood contribution
        try:
            loglik[t] = multivariate_normal.logpdf(y_t, mean=y_pred, cov=S, allow_singular=True)
        except:
            raise
            # print("Warning: Log-likelihood computation failed!")
            # loglik[t] = -np.inf  # Assign a default value

        x_pred[t], P_pred[t] = x_p, P_p
        x_filt[t], P_filt[t] = x_f, P_f

        x_prev, P_prev = x_f, P_f

    return x_filt, P_filt, x_pred, P_pred, loglik


def forward_backward_regimes(loglik, A, pi):
    """
    loglik: (K x T) log p(y_t | s_t=k)
    A: (K x K) regime transition matrix
    pi: (K,) initial regime probs

    Returns:
    gamma: (T x K)
    xi: (T-1 x K x K)
    """
    K, T = loglik.shape

    logA = np.log(A + 1e-12)
    logpi = np.log(pi + 1e-12)

    # Forward pass (log-alpha)
    logalpha = np.zeros((T, K))
    logalpha[0] = logpi + loglik[:, 0]

    for t in range(1, T):
        for j in range(K):
            logalpha[t, j] = np.logaddexp.reduce(
                logalpha[t-1] + logA[:, j]
            ) + loglik[j, t]

    # Backward pass (log-beta)
    logbeta = np.zeros((T, K))
    logbeta[-1] = 0.0

    for t in range(T - 2, -1, -1):
        for i in range(K):
            logbeta[t, i] = np.logaddexp.reduce(
                logA[i] + loglik[:, t+1] + logbeta[t+1]
            )

    # Gamma
    loggamma = logalpha + logbeta
    loggamma -= np.logaddexp.reduce(loggamma, axis=1, keepdims=True)
    gamma = np.exp(loggamma)

    # Xi
    xi = np.zeros((T-1, K, K))
    for t in range(T-1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (
                    logalpha[t, i]
                    + logA[i, j]
                    + loglik[j, t+1]
                    + logbeta[t+1, j]
                )
        xi[t] = np.exp(xi[t] - np.logaddexp.reduce(xi[t]))

    return gamma, xi

def rts_smoother(x_filt, P_filt, x_pred, P_pred, F):
    """
    x_filt: (T x N) is x_{t|t}
    P_filt: (T x N x N) is P_{t|t}
    x_pred: (T x N) is x_{t|t-1}
    P_pred: (T x N x N) is P_{t|t-1}
    F: (N x N)

    returns:
    x_smooth: (T x N) is  x_{t|T}
    P_smooth: (T x N x N) is P_{t|T}
    """

    T, N = x_filt.shape

    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)

    # Initialize at final time
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]

    smoother_gains = np.empty((T, N, N))

    print('in RTS smoother, gotten P_pred:'); print(P_pred)

    # Backward recursion
    for t in range(T - 2, -1, -1):
        P_t = P_filt[t]
        P_tp1_pred = P_pred[t + 1]

        # Smoother gain
        C_t = P_t @ F.T @ np.linalg.inv(P_tp1_pred)
        smoother_gains[t] = C_t

        # Smoothed mean
        x_smooth[t] = x_filt[t] + C_t @ (x_smooth[t+1] - x_pred[t+1])

        # Smoothed covariance
        P_smooth[t] = P_filt[t] + C_t @ (P_smooth[t+1] - P_pred[t+1]) @ C_t.T

    return x_smooth, P_smooth, smoother_gains

def compute_expectations(x_smooth, P_smooth, C):
    """
    x_smooth: (T x N)
    P_smooth: (T x N x N)
    C: (T-1 x N x N) smoother gains

    Returns:
    Ex:  (T x N)
    Exx: (T x N x N)
    Exx_tm1: (T-1 x N x N)
    """
    T, N = x_smooth.shape

    Ex = x_smooth
    Exx = np.zeros((T, N, N))
    Exx_tm1 = np.zeros((T - 1, N, N))

    for t in range(T):
        Exx[t] = P_smooth[t] + np.outer(x_smooth[t], x_smooth[t])

    for t in range(1, T):
        Exx_tm1[t-1] = C[t-1] @ P_smooth[t] + np.outer(x_smooth[t], x_smooth[t-1])

    return Ex, Exx, Exx_tm1


def estimate_slds_params(Exx_tm1, Exx, Ex, y, gamma, regime):

    """
    Exx_tm1: (T-1 x N x N)
    Exx: (T x N x N)
    Ex: (T x N)
    y: observation data (T x M)
    gamma: (T x K)
    regime: int between 0 and K-1
    """
    T = Exx.shape[0]

    # estimate F    
    num, den = np.zeros_like(Exx[0]), np.zeros_like(Exx[0])
    for t in range(T - 1):
        num += gamma[t, regime] * Exx_tm1[t]
        den += gamma[t, regime] * Exx[t]
    F = num @ np.linalg.inv(den)

    # estimate Q
    Q = np.zeros_like(F)
    for t in range(T - 1):
        Q += gamma[t, regime] * (Exx[t+1]
              - F @ Exx_tm1[t]
              - Exx_tm1[t].T @ F.T
              + F @ Exx[t] @ F.T)
    Q = Q / np.sum(gamma[:-1, regime])

    # estimate H
    num, den = np.zeros_like(Exx[0]), np.zeros_like(Exx[0])
    for t in range(T):
        num += gamma[t, regime] *  np.outer(y[t], Ex[t])
        den += gamma[t, regime] * Exx[t]
    H = num @ np.linalg.inv(den)


    # estimate R
    M = y[0].shape[0]
    R = np.zeros((M, M))
    for t in range(T):
        R += gamma[t, regime] * (np.outer(y[t], y[t])
              - H @ np.outer(Ex[t], y[t])
              - np.outer(y[t], Ex[t]) @ H.T
              + H @ Exx[t] @ H.T)
    R /= np.sum(gamma[:, regime])

    return F, Q, H, R