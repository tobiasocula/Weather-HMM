import numpy as np

def kalman_filter(y, F, Q, H, R, x0, P0):
    """
    let:
    N = dimensionality of hidden state vector
    M = dimensionality of observation vector
    T = number of datapoints

    y: observation data (shape T x M)
    F: state transition matrix (shape N x N)
    Q: process noise (shape N x N)
    H: mapping state -> observation (shape M x N)
    R: measurement noise (shape M x N)

    x0: initial guess of state vector
    P0: initial guess of covariance of state
    """

    T = y.shape[0]
    n = x0.shape[0]

    x_pred = np.zeros((T, n))
    P_pred = np.zeros((T, n, n))
    x_filt = np.zeros((T, n))
    P_filt = np.zeros((T, n, n))

    x_prev = x0
    P_prev = P0

    for t in range(T):
        # --- Predict --- (analogue to forward algorithm)
        # predicted values (before observation)
        x_p = F @ x_prev
        P_p = F @ P_prev @ F.T + Q
 
        # Update (analogue to backward algorithm)
        # updated values (after observation)
        y_t = y[t]
        S = H @ P_p @ H.T + R
        K = P_p @ H.T @ np.linalg.inv(S)

        x_f = x_p + K @ (y_t - H @ x_p) # updated state mean (after observing y_t)
        P_f = (np.eye(n) - K @ H) @ P_p # updated uncertainty (after observing y_t)

        # new predicted values (update)s
        x_pred[t], P_pred[t] = x_p, P_p
        x_filt[t], P_filt[t] = x_f, P_f

        x_prev, P_prev = x_f, P_f

    return x_filt, P_filt, x_pred, P_pred

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

def estimate_lds_params(Exx_tm1, Exx, Ex, y):

    """
    Exx_tm1: (T-1 x N x N)
    Exx: (T x N x N)
    Ex: (T x N)
    y: observation data (T x M)
    """

    # estimate F
    num = Exx_tm1.sum(axis=0)
    den = Exx[:-1].sum(axis=0)
    F = num @ np.linalg.inv(den)

    # estimate Q
    T = Exx.shape[0]
    Q = np.zeros_like(F)
    for t in range(T - 1):
        Q += (Exx[t+1]
              - F @ Exx_tm1[t]
              - Exx_tm1[t].T @ F.T
              + F @ Exx[t] @ F.T)
    Q = Q / (T - 1)

    # estimate H
    num = np.sum([np.outer(y[t], Ex[t]) for t in range(len(y))], axis=0)
    den = Exx.sum(axis=0)
    H = num @ np.linalg.inv(den)

    # estimate R
    M = y[0].shape[0]
    R = np.zeros((M, M))
    for t in range(T):
        R += (np.outer(y[t], y[t])
              - H @ np.outer(Ex[t], y[t])
              - np.outer(y[t], Ex[t]) @ H.T
              + H @ Exx[t] @ H.T)
    R /= T

    return F, Q, H, R