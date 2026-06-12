import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from astropy.constants import G

def vr_from_mass(m, z, overdensity=200):
    #shout out https://en.wikipedia.org/wiki/Virial_mass, there are papers for it though
    cosmo = FlatLambdaCDM(H0=67.7, Om0=0.31)

    rho_crit = cosmo.critical_density(z)
    
    m_halo = np.array(m) * 100 * u.solMass
    r_cubed = (3 * m_halo) / (4 * np.pi * overdensity * rho_crit)
    r = r_cubed**(1/3) # kpc
    
    # theta = R / d_A
    arcsec_per_kpc = cosmo.arcsec_per_kpc_proper(z) # 1/d_A
    angular_radius_arcsec = r * arcsec_per_kpc
    
    return np.array(angular_radius_arcsec.to(u.arcmin)) # now we're in arcmin

def mh2ms(M_h, z):
    # shout out https://iopscience.iop.org/article/10.1088/0004-637X/770/1/57/pdf
    a = 1.0 / (1.0 + z)
    nu = np.exp(-4.0 * (a**2))
    
    M1 = 10**(11.514 + (-1.793 * (a - 1.0) + -0.251 * z) * nu)
    epsilon = 10**(-1.777 + (-0.006 * (a - 1.0) + 0.0 * z) * nu + -0.119 * (a - 1.0))
    alpha = -1.412 + (0.731 * (a - 1.0)) * nu
    delta = 3.508 + (2.608 * (a - 1.0) + 0.0 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.0) + 0.279 * z) * nu

    def f(x):
        term1 = -np.log10(10**(alpha * x) + 1.0)
        
        log10_1_exp_x = np.log10(1 + np.exp(x))

        term2 = delta * (log10_1_exp_x**gamma) / (1.0 + np.exp(10**(-x)))
        return term1 + term2

    x = np.log10(M_h / M1)
    log10_M_star = np.log10(epsilon * M1) + f(x) - f(0)
    
    result = 10**log10_M_star
    return result

def ms2mh(M_star, z):
    if M_star.shape >= (100,):
        dim=len(M_star)*3
    else:
        dim=200
    mh_grid = 10**np.linspace(10, 20, dim)
    ms_grid = mh2ms(mh_grid, z)
    
    log10_ms_grid = np.log10(ms_grid)
    log10_mh_grid = np.linspace(10, 20, dim)
    
    log10_ms_targets = np.log10(M_star)
    
    return 10**np.interp(log10_ms_targets, log10_ms_grid, log10_mh_grid)