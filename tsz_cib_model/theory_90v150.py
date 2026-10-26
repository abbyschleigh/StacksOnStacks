import numpy as np

# frequencies and beams for our data (FWHM = full-with half max)
nu1   = 98
nu2   = 150
fwhm1 = 1.42
fwhm2 = 2.07
def fwhm_to_sigma(fwhm): return fwhm / (2 * np.sqrt(2 * np.log(2))) # convert fwhm to gaussian sigma
sig1 = fwhm_to_sigma(fwhm1)
sig2 = fwhm_to_sigma(fwhm2)

# compute the binning matrix 
# (i.e. if f(r) is the underlying continuous profile, dot it with bin_mat to get the measured profile)
# here I'm dividing by the ring area for each radius
r = np.linspace(0,15,500); dr=r[1]-r[0]
rbin = np.linspace(2,6,9,endpoint=True)
bin_mat = np.zeros((len(rbin),len(r)))
for i in range(bin_mat.shape[0]):
    window = np.zeros(bin_mat.shape[1])
    r0,r1 = 1,rbin[i]
    r2 = (2*rbin[i]**2-r0**2)**0.5
    window[np.where((r>r0)&(r<=r1))] = 1.
    window[np.where((r>r1)&(r<=r2))] = -1.
    window *= 2*np.pi*r*dr
    bin_mat[i,:] = window

def compute_binned_profiles(a_cib,a_tsz,r_cib,sigma_cib,sigma_tsz):
    """
    Returns an array of length 18 (90 + 150 GHz profile)

    a_cib: amplitude of CIB contribution at 90 GHz
    a_tsz: amplitude of tSZ contribution at 90 GHz
    r_cib: rescale CIB contribution at 150 GHz by some fudge factor
    sigma_[tsz,cib]: width of Gaussian for CIB and tSZ
    """
    # convolving with the beam is easy for a Gaussian signal: sigma_signal**2 -> sigma_signal**2 + sigma_beam**2
    profile1 = a_cib * np.exp(-r**2/2/(sigma_cib**2+sig1**2)**2)
    profile1-= a_tsz * np.exp(-r**2/2/(sigma_tsz**2+sig1**2)**2)
    profile2 = a_cib * r_cib * np.exp(-r**2/2/(sigma_cib**2+sig2**2)**2)
    profile2-= 0.62 * a_tsz * np.exp(-r**2/2/(sigma_tsz**2+sig2**2)**2) # 0.62 comes from the tSZ frequency scaling
    return np.concatenate([ np.dot(bin_mat,profile1) , np.dot(bin_mat,profile2)])