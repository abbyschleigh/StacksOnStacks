import numpy as np
from multiprocessing import Pool, cpu_count
from theory_90v150 import compute_binned_profiles
import sys

kind = str(sys.argv[1])
ddir = 'abbys_data'
odir = 'chains'

# ======================================================
# 1. Load data
# ======================================================
data = np.genfromtxt(f'{ddir}/{kind}_d_vector.txt')
cov  = np.genfromtxt(f'{ddir}/{kind}_fixed_covariance.txt')
cov_inv = np.linalg.inv(cov)

# ======================================================
# 2. Draw parameters from priors
# ======================================================
Nsamp = int(1e7)
a_cib = np.random.uniform(0, 0.2, Nsamp)
a_tsz = np.random.uniform(0, 0.3, Nsamp)
r_cib = np.random.uniform(2, 6, Nsamp)
sigma_cib = np.random.uniform(0, 2, Nsamp)
sigma_tsz = np.random.uniform(0, 2, Nsamp)

params = np.array([a_cib, a_tsz, r_cib, sigma_cib, sigma_tsz]).T  # shape: (Nsamp, 5)

# ======================================================
# 3. Define worker function
# ======================================================
def evaluate_likelihood(theta):
    a_cib, a_tsz, r_cib, sigma_cib, sigma_tsz = theta
    profile = compute_binned_profiles(a_cib, a_tsz, r_cib, sigma_cib, sigma_tsz)
    diff = profile - data
    return 0.5 * diff @ cov_inv @ diff

# ======================================================
# 4. Parallel evaluation
# ======================================================
if __name__ == '__main__': 
    ncpu = cpu_count() # or set manually 
    with Pool(ncpu) as pool: 
        mlogL = np.array(pool.map(evaluate_likelihood, params)) 
    output = np.zeros((params.shape[0],params.shape[1]+1))
    output[:,0] = mlogL
    output[:,1:] = params
    np.savetxt(f'{odir}/chains_{kind}.txt', output, header = 'mlogL, a_cib, a_tsz, r_cib, sigma_cib, sigma_tsz') 