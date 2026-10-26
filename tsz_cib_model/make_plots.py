import numpy as np
import matplotlib.pyplot as plt
from theory_90v150 import rbin,compute_binned_profiles

ddir = 'abbys_data'
odir = 'chains'
fdir = 'figures'

for kind in ['highAGNLUM','lowAGNLUM','highSFRM','lowSFRM']:
    # load data and chain
    data  = np.loadtxt(f'{ddir}/{kind}_d_vector.txt')
    cov   = np.loadtxt(f'{ddir}/{kind}_fixed_covariance.txt')
    chain = np.loadtxt(f'{odir}/chains_{kind}.txt')
    # find the best-fit
    bfidx = np.argmin(chain[:,0])
    best_fit_chi2 = 2*chain[bfidx,0]
    best_fit_params = chain[bfidx,1:]
    # make plot
    # data
    error90 = np.diag(cov)[:9]**0.5
    error150 = np.diag(cov)[9:]**0.5
    r = 1/np.pi/rbin**2
    plt.errorbar(rbin,r*data[:9],yerr=r*error90,label=r'90',c='C0',ls='',marker='o')
    plt.errorbar(rbin,r*data[9:],yerr=r*error150,label=r'150',c='C1',ls='',marker='o')
    # model
    a_cib,a_tsz,r_cib,sigma_cib,sigma_tsz = best_fit_params
    plt.plot(rbin,r*compute_binned_profiles(0*a_cib,a_tsz,r_cib,sigma_cib,sigma_tsz)[:9],c='C0',ls='--',lw=0.5)
    plt.plot(rbin,r*compute_binned_profiles(0*a_cib,a_tsz,r_cib,sigma_cib,sigma_tsz)[9:],c='C1',ls='--',lw=0.5)
    plt.plot(rbin,r*compute_binned_profiles(a_cib,0*a_tsz,r_cib,sigma_cib,sigma_tsz)[:9],c='C0',ls='dotted',lw=0.5)
    plt.plot(rbin,r*compute_binned_profiles(a_cib,0*a_tsz,r_cib,sigma_cib,sigma_tsz)[9:],c='C1',ls='dotted',lw=0.5)
    plt.plot(rbin,r*compute_binned_profiles(a_cib,a_tsz,r_cib,sigma_cib,sigma_tsz)[:9],c='C0')
    plt.plot(rbin,r*compute_binned_profiles(a_cib,a_tsz,r_cib,sigma_cib,sigma_tsz)[9:],c='C1')
    plt.plot([],[],c='k',lw=0.5,ls='--',label='tSZ only')
    plt.plot([],[],c='k',lw=0.5,ls='dotted',label='CIB only')
    plt.plot([],[],c='k',lw=0.5,ls='--',label='tSZ + CIB best-fit')
    plt.legend()
    plt.xlabel(r'r [arcmin]')
    plt.ylabel(r'CAP filtered profile')
    plt.title(kind+' '+r'$(\chi^2_\mathrm{min}=$'+str(np.round(best_fit_chi2,1))+')')
    plt.savefig(f'{fdir}/{kind}_data_and_best_fits.png')
    plt.show()
    print(best_fit_params)