import numpy as np
from numba import jit
from scipy import special


@jit(nopython=True)
def Lagrangian_tophat_mass_inR(R,Omega_M):
    return 3.086e13/(2.*6.674*1.989) * Omega_M * R**3.

@jit(nopython=True)
def Lagrangian_tophat_radius_ofM(M,Omega_M):
    return (M*(2.*6.674*1.989)/(3.086e13 * Omega_M)) ** (1./3.)

@jit(nopython=True)
def derivative_Lagrangian_tophat_mass_inR(R,Omega_M):
    return 3.086e13/(2.*6.674*1.989) * Omega_M * R**2.*3.

@jit(nopython=True)
def f_ST_nu_unnorm(nu, norm, p, q):
    return norm * (1. + (q * nu * nu)**(-p)) * q**0.5 * np.exp(-q * nu * nu / 2.)

def f_ST_nu(nu, p, q):
    norm = np.sqrt(2./np.pi) / (1. + special.gamma (0.5 - p) / (2. ** p * np.sqrt(np.pi)))
    return f_ST_nu_unnorm(nu, norm, p, q)

@jit(nopython=True)
def f_Tinker(sigma, norm, a, b, c):
    return norm * ((b / sigma) ** a + 1) *  np.exp(-c / (sigma * sigma))

@jit(nopython=True)
def f_Tinker_5params(sigma, norm, p0, p1, p2, p3):
    return norm * ((p1 / sigma) ** p0 + sigma ** (-p2)) *  np.exp(-p3 / (sigma * sigma))


def f_Tinker_normalized(sigma, p0, p1, p2, p3):
    norm = 2. / (p1 ** p0 * p3**(-0.5 * p0) * special.gamma (0.5 * p0) + p3**(-0.5 * p2) * special.gamma (0.5 * p2))
    return norm * f_Tinker_5params(sigma, norm, p0, p1, p2, p3)



#@jit(nopython=True)
def f_Musso_S(S, Gamma2, B, dB_dS):
    beta_star = -2.*dB_dS * S**0.5 +  B / S**0.5
    return np.exp(-0.5 * B**2 / S) / (8.*np.pi)**0.5 * beta_star / S * (
        0.5 + 0.5 * special.erf(Gamma2**0.5 * beta_star / 2**0.5) + 
        np.exp(-0.5 * Gamma2 * beta_star**2) / ((2.*np.pi * Gamma2)**0.5 * beta_star))




def f_Musso_fixed_nu(nu, Gamma2):
    return np.exp(-0.5 * nu**2) / (8.*np.pi)**0.5 * (
        0.5 + 0.5 * special.erf(Gamma2**0.5 * nu / 2**0.5 )+ 
        np.exp(-0.5 * Gamma2 * nu**2) / (nu*(2.*np.pi * Gamma2)**0.5))


def f_S_MB_approx(s,W,B,dB_ds):

    LDD = s * W- 0.25
    D_BP = 0.5 * B / s - dB_ds
    AB2S = 0.5*s/LDD * D_BP**2
    
    I2a_S = LDD**0.5 / (4*np.pi*s) * np.exp(-0.5 / LDD * (s * dB_ds**2 + W * B**2 - B * dB_ds))
    I2b_2_S_noExp = 0.5 * D_BP * (special.erf(AB2S**0.5)+1) / (2.*np.pi*s)**0.5
        
        
    return I2a_S + (I2b_2_S_noExp + LDD**0.5/(4*np.pi*s) * np.exp(-AB2S)) * np.exp(-0.5*B**2 / s)


