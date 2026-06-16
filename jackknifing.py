import numpy as np

def jackknife_weighted_mean_cov_fast2(X1, w1, X2, w2):
    '''
    Input:
    
        X1: (n_samples, d1)
        w1: (n_samples,)
        X2: (n_samples, d2)
        w2: (n_samples,)

    Output:
    
        mean1: (d1,)
        mean2: (d2,)
        cov11: (d1, d1) covariance of mean1
        cov22: (d2, d2) covariance of mean2
        cov12: (d1, d2) cross-covariance between mean1 and mean2
    '''
    nsamp1 = X1.shape[0]
    nsamp2 = X2.shape[0]
    if nsamp1 != nsamp2:
        w1 = w1[:min(nsamp1, nsamp2)]
        w2 = w2[:min(nsamp1, nsamp2)]
        X1 = X1[:min(nsamp1, nsamp2)]
        X2 = X2[:min(nsamp1, nsamp2)]
        
    n = X1.shape[0]
    d1 = X1.shape[1]
    d2 = X2.shape[1]

    # Normalize weights once
    w1 = w1 / np.sum(w1)
    w2 = w2 / np.sum(w2)

    # Compute full means
    mean1 = X1.T @ w1  # shape (d1,)
    mean2 = X2.T @ w2  # shape (d2,)

    # Precompute full weighted sums
    S1 = X1.T * w1     # (d1, n)
    S2 = X2.T * w2     # (d2, n)

    # Leave-one-out weighted sums (efficient)
    total_w1 = np.sum(w1)
    total_w2 = np.sum(w2)

    sum1 = np.sum(S1, axis=1, keepdims=True) - S1  # (d1, n)
    sum2 = np.sum(S2, axis=1, keepdims=True) - S2  # (d2, n)

    w1_loo = total_w1 - w1  # (n,)
    w2_loo = total_w2 - w2  # (n,)

    # LOO means: shape (n, d1) and (n, d2)
    jk_mean1 = (sum1 / w1_loo).T  # (n, d1)
    jk_mean2 = (sum2 / w2_loo).T  # (n, d2)

    # Mean of jackknife means
    mean_jk1 = np.mean(jk_mean1, axis=0)
    mean_jk2 = np.mean(jk_mean2, axis=0)

    # Demeaned jackknife means
    diff1 = jk_mean1 - mean_jk1  # (n, d1)
    diff2 = jk_mean2 - mean_jk2  # (n, d2)

    # Jackknife covariance of the mean
    cov11 = (n - 1) / n * diff1.T @ diff1  # (d1, d1)
    cov22 = (n - 1) / n * diff2.T @ diff2  # (d2, d2)
    cov12 = (n - 1) / n * diff1.T @ diff2  # (d1, d2)

    return mean1, mean2, cov11, cov22, cov12

def jackknife_weighted_mean_cov_fast_3(X1, w1, X2, w2, X3, w3):
    """
    Input:
    
        X1: (n_samples, d1)
        w1: (n_samples,)
        X2: (n_samples, d2)
        w2: (n_samples,)
        X3: (n_samples, d3)
        w3: (n_samples,)

    Output:
    
        mean1: (d1,)
        mean2: (d2,)
        mean3: (d3,)

        cov11: (d1, d1)
        cov22: (d2, d2)
        cov33: (d3, d3)

        cov12: (d1, d2)
        cov13: (d1, d3)
        cov23: (d2, d3)
    """
    n = X1.shape[0]
    d1 = X1.shape[1]
    d2 = X2.shape[1]
    d3 = X3.shape[1]

    # Normalize weights
    w1 = w1 / np.sum(w1)
    w2 = w2 / np.sum(w2)
    w3 = w3 / np.sum(w3)

    # Full weighted means
    mean1 = X1.T @ w1
    mean2 = X2.T @ w2
    mean3 = X3.T @ w3

    # Precompute weighted per-sample contributions
    S1 = X1.T * w1      # (d1, n)
    S2 = X2.T * w2      # (d2, n)
    S3 = X3.T * w3      # (d3, n)

    total_w1 = np.sum(w1)
    total_w2 = np.sum(w2)
    total_w3 = np.sum(w3)

    # Leave-one-out sums
    sum1 = np.sum(S1, axis=1, keepdims=True) - S1   # (d1, n)
    sum2 = np.sum(S2, axis=1, keepdims=True) - S2   # (d2, n)
    sum3 = np.sum(S3, axis=1, keepdims=True) - S3   # (d3, n)

    w1_loo = total_w1 - w1   # (n,)
    w2_loo = total_w2 - w2   # (n,)
    w3_loo = total_w3 - w3   # (n,)

    # Leave-one-out means
    jk_mean1 = (sum1 / w1_loo).T   # (n, d1)
    jk_mean2 = (sum2 / w2_loo).T   # (n, d2)
    jk_mean3 = (sum3 / w3_loo).T   # (n, d3)

    # Means of jackknife samples
    mean_jk1 = np.mean(jk_mean1, axis=0)
    mean_jk2 = np.mean(jk_mean2, axis=0)
    mean_jk3 = np.mean(jk_mean3, axis=0)

    # Demeaned jackknife samples
    diff1 = jk_mean1 - mean_jk1
    diff2 = jk_mean2 - mean_jk2
    diff3 = jk_mean3 - mean_jk3

    # Covariances
    factor = (n - 1) / n

    cov11 = factor * diff1.T @ diff1
    cov22 = factor * diff2.T @ diff2
    cov33 = factor * diff3.T @ diff3

    cov12 = factor * diff1.T @ diff2
    cov13 = factor * diff1.T @ diff3
    cov23 = factor * diff2.T @ diff3

    return (mean1, mean2, mean3,
            cov11, cov22, cov33,
            cov12, cov13, cov23)