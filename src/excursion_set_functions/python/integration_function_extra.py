import numpy as np
from numba import jit, prange
from numba.core import types
from numba.typed import Dict
#from interpolation.splines import UCGrid, CGrid, nodes, eval_linear, filter_cubic, eval_cubic
import pandas
from scipy import integrate, interpolate, optimize, signal
import scipy.special as sp
from multiprocessing import Pool, Process, Manager
from multiprocessing.pool import ThreadPool
from .integration import *


#int_array = types.int64[::1]
#int_Narray = types.int64[:,:1]
float_arrayDim1 = types.float64[::1]
#float_Narray = types.float64[:,::1]
#float_Narray_A = types.float64[:,:1]


@jit(nopython=True,cache=True)
def top_hat_k_array(r, k):
    x = r * k
    return 3. * (np.sin(x) - x * np.cos(x)) / (x ** 3.)



@jit(nopython=True,cache=True)
def dx_TOPHAT_Taylor(x):
    return -x/5. +x**3/70. - x**5/2520. +x**7/166320.


@jit(nopython=True,cache=True)
def dr_top_hat_rk_HR(r, k,X0=1e-1):
    x = r * k
    OUT = np.zeros(len(x))
    mask = (x <= X0)
    OUT[mask] = dx_TOPHAT_Taylor(x[mask])
    mask[:] = ~mask
    OUT[mask] = 3. * ((x[mask]**2 - 3.) * np.sin(x[mask]) + 3. * x[mask] * np.cos(x[mask])) / (x[mask] ** 4.)
    return OUT * k

@jit(nopython=True,cache=True)
def j0(x):
    return np.sin(x)/x


@jit(nopython=True,cache=True)
def j2(x):
    return ((3. - x**2) * np.sin(x) - 3.* x * np.cos(x)) / x**3



@jit(nopython=True,cache=True)
def IntegrationSmallK_correlation(Pk0,k0,n,R1,R2,r_corr,Omax=8):
    ALPHA = np.zeros(9)
    ALPHA[0] = 1.
    ALPHA[2] = -r_corr ** 2 / Factorial(3) * a_fac(2)
    ALPHA[4] = r_corr ** 4 / Factorial(5) * (a_fac(4) * (R1**4 + R2 **4) +
                                         a_fac(2) ** 2 * R1 ** 2 * R2 **2)
    ALPHA[6] = -r_corr ** 6 / Factorial(7) * (a_fac(6) * (R1 ** 6 + R2 ** 6) +
                                         a_fac(2) * a_fac(4) * R1 ** 2 * R2 ** 2 * (R1 ** 4 + R2 ** 4))
    ALPHA[8] = r_corr ** 8 / Factorial(9) * (a_fac(8) * (R1 ** 8 + R2 ** 8) +
                                         a_fac(2) * a_fac(6) * R1 ** 2 * R2 ** 2 * (R1 ** 6 + R2 ** 6) +
                                         a_fac(4) ** 2 * R1 ** 4 * R2 ** 4)
    return Pk0 / (2.*np.pi**2) * np.sum(ALPHA [:Omax+1]* k0 ** (np.arange(Omax+1) + 3) / (np.arange(Omax+1) + 3 + n))
    #return Pk0 / (2.*np.pi**2) * np.sum(ALPHA[:Omax+1] * k0 ** 3 / (np.arange(Omax+1) + 3 + n))




def findZeros(R1,R2,kMax):
    F0 = lambda X: np.tan(X) - X
    xZtan = np.arange(int(kMax * max(R1,R2) / np.pi) +1) * np.pi + np.pi / 2.
    Ztan = np.zeros(len(xZtan)-1)
    for i in range(0,len(xZtan)-1):
        Ztan[i] = optimize.brentq(F0,xZtan[i] + 1e-8,xZtan[i+1] - 1e-8)
    if R1 == R2:
        return np.sort(Ztan) / R1
    else:
        Ztan1 = Ztan / max(R1, R2)
        Ztan2 = Ztan[Ztan < kMax * min(R1,R2)] / min(R1,R2)
        AllZ = np.zeros(len(Ztan1) + len(Ztan2))
        AllZ[:len(Ztan1)] = Ztan1
        AllZ[len(Ztan1):] = Ztan2
        return np.sort(AllZ)

def findZeros_xMax(xMax):
    F0 = lambda X: np.tan(X) - X
    xZtan = np.arange(int(xMax / np.pi) + 1) * np.pi + np.pi / 2.
    Ztan = np.zeros(len(xZtan)-1)
    for i in range(0,len(xZtan)-1):
        Ztan[i] = optimize.brentq(F0,xZtan[i] + 1e-8,xZtan[i+1] - 1e-8)
    return np.sort(Ztan)

@jit(nopython=True,cache=True)
def findZeros_fromAllZeros(all0,R1,R2,kMax):
    if R1 == R2:
        out = np.sort(all0) / R1
    else:
        Ztan1 = all0 / max(R1, R2)
        Ztan2 = all0[all0 < kMax * min(R1, R2)] / min(R1, R2)
        AllZ = np.zeros(len(Ztan1) + len(Ztan2))
        AllZ[:len(Ztan1)] = Ztan1
        AllZ[len(Ztan1):] = Ztan2
        out = np.sort(AllZ)
    return out[out<kMax]


@jit(nopython=True,cache=True)
def findZeros_j0_kr(kMax,r):
    return np.arange(1,int(kMax*r/np.pi))*np.pi/r

@jit(nopython=True,cache=True)
def findZeros_fromAllZeros_r_corr(allTHzeros,j0zeros,R1,R2,kMax):
    out_TH = findZeros_fromAllZeros(allTHzeros,R1,R2,kMax)
    out = np.zeros(len(j0zeros)+len(out_TH))
    out[:len(out_TH)] = out_TH
    out[len(out_TH):] = j0zeros
    return np.sort(out)

@jit(nopython=True,cache=True)
def findZeros_fromTHandj0(allTHzeros,j0zeros,R1,R2,r_corr,kMax):
    if r_corr == 0.:
        return findZeros_fromAllZeros(allTHzeros,R1,R2,kMax)
    else:
        return findZeros_fromAllZeros_r_corr(allTHzeros,j0zeros,R1,R2,kMax)


@jit(nopython=True,cache=True)
def Karray_building(k,kZeros,NkPerZeros):
    Nk1 = len(k)
    if k[-1] == kZeros[-1]:
        Nklast = 0
    else:
        Nklast = int(round((k[-1] - kZeros[-1]) / (kZeros[-1] - kZeros[-2]) * NkPerZeros+1))
        if Nklast < 4:
            Nklast = 4
    Kout = np.zeros(Nk1 + (len(kZeros)-1)*NkPerZeros + Nklast)
    Kout[:Nk1] = 10**np.linspace(np.log10(k[0]), np.log10(kZeros[0]), Nk1)#np.logspace(np.log10(k[0]), np.log10(kZeros[0]), Nk1)
    n0 = Nk1
    for i in range(0,len(kZeros)-1):
        Kout[n0:n0+NkPerZeros] = np.linspace(kZeros[i], kZeros[i+1], NkPerZeros+1)[1:]
        n0 += NkPerZeros
    if Nklast > 0:
        Kout[n0:] = np.linspace(kZeros[i+1], k[-1], Nklast+1)[1:]
    return Kout

@jit(nopython=True,cache=True)
def Gaussian_kernel(x,kmax):
    return np.exp(-(x/kmax)**4)

def C_ij_TopHat_zeros_CORESplineIntegration(
        UnfiltInterp, k, R1, R2, r_corr, xZeros, j0_Kzeros, SmallK_args, NkPerZeros, kKern):
    if r_corr == 0:
        if kKern > 0:
            Func = lambda x: UnfiltInterp(x) * top_hat_rk_HR(R1, x) * top_hat_rk_HR(R2, x) * Gaussian_kernel(x,kKern)
        else:
            Func = lambda x: UnfiltInterp(x) * top_hat_rk_HR(R1, x) * top_hat_rk_HR(R2, x)
    else:
        if kKern > 0:
            Func = lambda x: UnfiltInterp(x) * top_hat_rk_HR(R1, x) * top_hat_rk_HR(R2, x) * j0(x * r_corr)* Gaussian_kernel(x,kKern)
        else:
            Func = lambda x: UnfiltInterp(x) * top_hat_rk_HR(R1, x) * top_hat_rk_HR(R2, x) * j0(x * r_corr)
    kZeros = findZeros_fromTHandj0(xZeros, j0_Kzeros, R1, R2, r_corr, k[-1])
    #kZeros = findZeros_fromAllZeros(xZeros,R1,R2,k[-1])
    if SmallK_args == 0:
        INT0 = 0
    else:
        Pk0, k0, n, r_corr, OMAX = SmallK_args
        if r_corr == 0:
            INT0 = IntegrationSmallK(Pk0, k0, n, R1, R2, Omax=OMAX)
        else:
            INT0 = IntegrationSmallK_correlation(Pk0, k0, n, R1, R2, r_corr, Omax=OMAX)
    #if kZeros[0] > k[0]:
    Kinterp = Karray_building(k,kZeros,NkPerZeros)
    return INT0 + float(interpolate.CubicSpline(Kinterp, Func(Kinterp)).integrate(Kinterp[0], Kinterp[-1]))

def C_ij_TopHat_SplineIntegration_CALC(
        i, k, R, r_corr, UnfiltInterp, xZeros, j0_Kzeros, SmallK_args, NkPerZeros, kKern):
    out = np.zeros(i + 1)
    for j in range(0, i + 1):
        out[j] = C_ij_TopHat_zeros_CORESplineIntegration(
            UnfiltInterp, k, R[i], R[j], r_corr, xZeros, j0_Kzeros, SmallK_args, NkPerZeros, kKern)
    return out

def C_ij_TopHat_SplineIntegration_DICT(
        C_ijdict_element, i, k, R, r_corr, UnfiltInterp, xZeros, j0_Kzeros, SmallK_args, NkPerZeros, kKern):
    C_ijdict_element[i] = C_ij_TopHat_SplineIntegration_CALC(
        i, k, R, r_corr, UnfiltInterp, xZeros, j0_Kzeros, SmallK_args, NkPerZeros, kKern)

def C_ij_TopHat_zeros_SplineIntegration(
        Pk, k, Rin, NkPerZeros=30, kmaxKern=0.,
        outType='numpy array',r_corr=0.,SmallKprecision=False,n=0.96,OmaxSmallK=8):
    R = -np.sort(-Rin)
    InterpFunction = interpolate.CubicSpline(k, k ** 2 * Pk / (2. * np.pi**2))
    #if r_corr == 0:
    #    Unfilt = InterpFunction
    #else:
    #    Unfilt = lambda x: InterpFunction(x) * j0(x * r_corr)
    j0_Kzeros = findZeros_j0_kr(k[-1], r_corr)
    if SmallKprecision:
        SmallK_args = Pk[0], k[0], n, r_corr, OmaxSmallK
    else:
        SmallK_args = 0
    kRzeros = findZeros_xMax(k[-1] * R[0])
    #def FforPool():
    #    return C_ij_TopHat_SplineIntegration_DICT(d, i, k, R, r_corr, Unfilt,kRzeros,j0_Kzeros,SmallK_args,NkPerZeros)
    manager = Manager()
    d = manager.dict()
    pool = Pool()
    for i in range(0, len(R)):
        pool.apply_async(C_ij_TopHat_SplineIntegration_DICT,
                         args=(d, i, k, R, r_corr, InterpFunction,kRzeros,j0_Kzeros,SmallK_args,NkPerZeros,kmaxKern))
    pool.close()
    pool.join()
    if outType == 'base':
        return d
    elif outType == 'numpy array':
        Cij_matr = np.zeros((len(R),len(R)))
        for i in range(0, len(R)):
            Cij_matr[i, :i + 1] = d[i]
        for i in range(0, len(R)):
            Cij_matr[i, i + 1:] = Cij_matr[i + 1:, i]
        return Cij_matr
    elif outType == 'numba dict':
        C_ij = Dict.empty(types.int64, float_arrayDim1)
        for i in range(0, len(R)):
            C_ij[i] = d[i]
            del d[i]
        return C_ij
    else:
        print('possible outType: base, numpy array, numba dict.\noutType not in list, base type is returned.')
        return d


@jit(nopython=True)
def Cprimeij_TopHat_bruteforce_TopHat_MAIN_lowR(
        Pk, k, R1, R2, OMAX, n):
    
    coeffs = cubic_spline_coeffs(k, Pk * k**2 * dr_top_hat_rk_HR(R1, k) * top_hat_rk_HR(R2, k))
    if OMAX <= 0:
        INT0 = 0
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    INT0 += np.sum((coeffs[:,0] + 
                    coeffs[:,1] / 2. +
                    coeffs[:,2] / 3. + 
                    coeffs[:,3] / 4.) * (k[1:] -  k[:-1]))
    return INT0 / (2. * np.pi ** 2)

@jit(nopython=True,parallel=True)
def Cprimeij_TopHat_bruteforce(Pk, k, R, NkPerZeros=30, n=0.96, OmaxSmallK=-1):

    Coeffs = cubic_spline_coeffs(k,Pk)
    #explicit_from_implicit_coeffs(Coeffs,k)
    k_HR = np.logspace(np.log10(k[0]),np.log10(k[-1]),(len(k)-1)*NkPerZeros+1)
    Pk_HR = get_values(k_HR,k,Coeffs)

    Cprimeij_out = np.zeros((len(R),len(R)))
    for i in range(len(R)):
        for j in prange(len(R)):
            #IDchange = find_id_change(k,(R[i] * R[j]) ** -0.5)
            Cprimeij_out[i,j] = Cprimeij_TopHat_bruteforce_TopHat_MAIN_lowR(Pk_HR, k_HR, R[i], R[j], OmaxSmallK, n)
    
    return Cprimeij_out



@jit(nopython=True)
def dSdR_TopHat_bruteforce_MAIN(
        Pk, k, R,  OMAX, n):
    
    coeffs = cubic_spline_coeffs(k, Pk * k**2 * dr_square_top_hat_rk_HR(R,k))
    if OMAX <= 0:
        INT0 = 0
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R, R, Omax=OMAX)

    INT0 += np.sum((coeffs[:,0] + 
                    coeffs[:,1] / 2. +
                    coeffs[:,2] / 3. + 
                    coeffs[:,3] / 4.) * (k[1:] -  k[:-1]))
    return INT0 / (2. * np.pi ** 2)




@jit(nopython=True,parallel=True)
def dSdR_TopHat_bruteforce(Pk, k, R, NkPerZeros=30, n=0.96, OmaxSmallK=-1):

    Coeffs = cubic_spline_coeffs(k,Pk)
    #explicit_from_implicit_coeffs(Coeffs,k)
    k_HR = np.logspace(np.log10(k[0]),np.log10(k[-1]),(len(k)-1)*NkPerZeros+1)
    Pk_HR = get_values(k_HR,k,Coeffs)

    dSdR_out = np.zeros(len(R),)
    for i in prange(len(R)):
        dSdR_out[i] = dSdR_TopHat_bruteforce_MAIN(Pk_HR, k_HR, R[i], OmaxSmallK, n)
    
    return dSdR_out


def di_C_ij_TopHat_MAIN(Coeffs, x, a, b):
    
    Si_diff, Ci_diff = sp.sici((a - b) * x)
    Si_sum, Ci_sum = sp.sici((a + b) * x)
    cos_diff = np.cos((a - b) * x)
    cos_sum = np.cos((a + b) * x)
    sin_diff = np.sin((a - b) * x)
    sin_sum = np.sin((a + b) * x)

    #Coeff_da_3 = 0.5 * (((a*b*x**2 - 1. ) / (a+b) + b / (a+b) ** 2 - 2.*a*b/(a+b)**3) * cos_sum + \
    #                   ((a*b*x**2 + 1. ) / (a-b) + b / (a-b) ** 2 - 2.*a*b/(a-b)**3) * cos_diff + \
    #                   (b*x / (a+b) - 2.*a*b*x/(a+b)**2 - x) * sin_sum + \
    #                   (b*x / (a-b) - 2.*a*b*x/(a-b)**2 + x) * sin_diff)
    
    Coeff_da_3 = 0.5 * ((a*b*x**2 / (a+b) - (a**2 + 3*a*b)/(a+b)**3) * cos_sum +
                       (a*b*x**2 / (a-b) + (a**2 - 3*a*b)/(a-b)**3) * cos_diff -
                       (a**2 + 3*a*b) / (a+b)**2 * x * sin_sum +
                       (a**2 - 3*a*b) / (a-b)**2 * x * sin_diff)
    
    Coeff_da_2 = 0.5 * ((cos_diff / (a-b) + cos_sum/(a+b))*a*b*x - sin_sum * (1.-b**2/(a+b)**2) + sin_diff * (1.-b**2/(a-b)**2))

    Coeff_da_1 = 0.5 * a*(Ci_diff-Ci_sum) + 0.5*a*b*(cos_diff/(a-b) + cos_sum/(a+b))

    Coeff_da_0 = 0.5 * a**2 * (Si_sum - Si_diff) + 0.5*a/x * (cos_sum - cos_diff)
    

    INT0 = 0.
    INT0 += np.sum((Coeff_da_3[1:] - Coeff_da_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_da_2[1:] - Coeff_da_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_da_1[1:] - Coeff_da_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_da_0[1:] - Coeff_da_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / (a**3*b**3)

    return INT0

    
def didj_C_ij_TopHat_MAIN(Coeffs, x, a, b):
    #INT0 = 0.
    #if SmallK_args == 0:
    #    INT0 = 0
    #else:
    #    INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    Si_diff, Ci_diff = sp.sici((a - b) * x)
    Si_sum, Ci_sum = sp.sici((a + b) * x)
    cos_diff = np.cos((a - b) * x)
    cos_sum = np.cos((a + b) * x)
    sin_diff = np.sin((a - b) * x)
    sin_sum = np.sin((a + b) * x)

    Coeff_tot_3 = 0.5*((6.*a*b*x / (a+b)**3 - a*b*x**3 / (a+b)) * sin_sum -
                       (6.*a*b*x / (a-b)**3 - a*b*x**3 / (a-b)) * sin_diff + 
                       (-3.*a*b*x**2 / (a+b)**2 + 6.*a*b/(a+b)**4) * cos_sum + 
                       (3.*a*b*x**2 / (a-b)**2 + 6.*a*b/(a-b)**4) * cos_diff)
    Coeff_tot_3_ff = lambda a,b,x: 0.5 * (Ci_diff - Ci_sum + a*b*x*(sin_diff / (a-b) + sin_sum / (a+b)) + 
                                       (a*b / (a-b)**2 - 1) * cos_diff + (a*b / (a+b)**2 + 1) * cos_sum)
    
    Coeff_da_3 = ((a*b*x**2 - 1. ) / (a+b) + b / (a+b) ** 2 - 2.*a*b/(a+b)**3) * cos_sum + \
                 ((a*b*x**2 + 1. ) / (a-b) + b / (a-b) ** 2 - 2.*a*b/(a-b)**3) * cos_diff + \
                 (b*x / (a+b) - 2.*a*b*x/(a+b)**2 - x) * sin_sum + (b*x / (a-b) - 2.*a*b*x/(a-b)**2 + x) * sin_diff
    def dd(a,b,x,h=1e-3):
        d1 = (Coeff_tot_3_ff(a+h,b+h,x) -Coeff_tot_3_ff(a-h,b+h,x)) / (2.*h)
        d2 = (Coeff_tot_3_ff(a+h,b-h,x) -Coeff_tot_3_ff(a-h,b-h,x)) / (2.*h)
        return  (d1 - d2) / (2. * h)
    
    Coeff_tot_2 = 0.5 * ((cos_sum - cos_diff) / x + a*b / (a-b) * sin_diff + a*b / (a+b) * sin_sum)
    Coeff_tot_1 = 0.25 * ((a**2 + b**2) * (Ci_diff - Ci_sum) - ((a-b) * sin_diff - (a+b) * sin_sum) / x - (cos_diff - cos_sum) / x**2 )
    Coeff_tot_0 = (-(a**3 - b**3) * Si_diff + (a**3 + b**3) * Si_sum - (a-b) * sin_diff/x**2 + (a+b) * sin_sum/x**2 -
                    ((a**2+b**2+a*b)/x + 1./x**3) * cos_diff + ((a**2+b**2-a*b)/x + 1./x**3) * cos_sum) / 6.
    
    #check_3 = T1_C3 / (a**3*b**3) + T2_C3 / (a**2*b**2) - T3_C3 / (a**3*b**2) - T4_C3 / (a**2*b**3)
    #check_2 = T1_C2 / (a**3*b**3) + T2_C2 / (a**2*b**2) - T3_C2 / (a**3*b**2) - T4_C2 / (a**2*b**3)
    #check_1 = T1_C1 / (a**3*b**3) + T2_C1 / (a**2*b**2) - T3_C1 / (a**3*b**2) - T4_C1 / (a**2*b**3)
    #check_0 = T1_C0 / (a**3*b**3) + T2_C0 / (a**2*b**2) - T3_C0 / (a**3*b**2) - T4_C0 / (a**2*b**3)

    INT0 = 0.
    INT0 += np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / (a**3*b**3)

    return INT0


def W_ij_TopHat_MAIN(Coeffs, x, a, b):
    #INT0 = 0.
    #if SmallK_args == 0:
    #    INT0 = 0
    #else:
    #    INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    Si_diff, Ci_diff = sp.sici((a - b) * x)
    Si_sum, Ci_sum = sp.sici((a + b) * x)
    cos_diff = np.cos((a - b) * x)
    cos_sum = np.cos((a + b) * x)
    sin_diff = np.sin((a - b) * x)
    sin_sum = np.sin((a + b) * x)
    
    #def CC3(a,b):
    #    Si_diff, Ci_diff = sp.sici((a - b) * x)
    #    Si_sum, Ci_sum = sp.sici((a + b) * x)
    #    cos_diff = np.cos((a - b) * x)
    #    cos_sum = np.cos((a + b) * x)
    #    sin_diff = np.sin((a - b) * x)
    #    sin_sum = np.sin((a + b) * x)
    #    return 0.5 * (Ci_diff - Ci_sum + a*b*x*(sin_diff / (a-b) + sin_sum / (a+b)) + 
    #                                   (a*b / (a-b)**2 - 1) * cos_diff + (a*b / (a+b)**2 + 1) * cos_sum)/ (a*b)**3
    #def db_CC3(a,b,db):
    #    return (CC3(a,b+db) - CC3(a,b-db)) / (2.*db)
    #def dadb_CC3(a,b,da=1e-6,db=1e-6):
    #    return (db_CC3(a+da,b,db) - db_CC3(a-da,b,db)) / (2.*da)

    Coeff_tot_3 = 0.5 * (9. * Ci_diff - 9 * Ci_sum + 
                         (-3.*(a*b*x**2)*(a-b)**2*(a**2 - 3.*a*b + b**2) + #15. * a**4 * b**2 * x**2 + 15. * a**2 * b**4 *x**2 - 24. * a**3 * b**3 * x**2 - 3. * a**5 * b * x**2 - 3. *a * b**5 * x**2 + 
                          57.*a*b*(a**2 + b**2) - 96. * (a*b)**2 - 12.*a**4 - 12.*b**4) * cos_diff / (a - b)**4 + 
                         (-3.*(a*b*x**2)*(a+b)**2*(a**2 + 3.*a*b + b**2) + 
                          57.*a*b*(a**2 + b**2) + 96. * (a*b)**2 + 12.*a**4 + 12.*b**4) * cos_sum / (a + b)**4 + 
                         x * (a*b * (21. + a*b*x**2) - 3.*(a**4 + b**4) / (a - b)**2) * sin_diff / (a - b)+ 
                         x * (a*b * (21. - a*b*x**2) + 3.*(a**4 + b**4) / (a + b)**2) * sin_sum / (a + b))
    Coeff_tot_2 = 0.5 * (-(9./x + a * b* x * (3.*a**2 + 3.*b**2 - 8.*a*b) / (a-b)**2) * cos_diff + 
                         (9./x - a * b* x * (3.*a**2 + 3.*b**2 + 8.*a*b) / (a+b)**2) * cos_sum + 
                         ((18.*a**3*b + 18.*a*b**3 - 32.*a**2*b**2 - 3.*a**4 - 3.*b**4) / (a-b)**3 + (a*b)**2 * x**2/(a-b)) * sin_diff + 
                         ((18.*a**3*b + 18.*a*b**3 + 32.*a**2*b**2 + 3.*a**4 + 3.*b**4) / (a+b)**3 - (a*b)**2 * x**2/(a+b)) * sin_sum)
    Coeff_tot_1 = 0.25 * (3. * (a**2 + b**2) * (Ci_diff - Ci_sum) + 
                          (-9./x**2 + (14.*a**2*b**2 - 6.*a**3*b - 6.*a*b**3) / (a-b)**2) * cos_diff + 
                          (9./x**2 - (14.*a**2*b**2 + 6.*a**3*b + 6.*a*b**3) / (a+b)**2) * cos_sum -
                          (9.*(a-b)/x - 2.*a**2*b**2*x /(a-b)) * sin_diff +
                          (9.*(a+b)/x - 2.*a**2*b**2*x /(a+b)) * sin_sum)
    Coeff_tot_0 = 0.5 * (-3. * (a*b/x + 1./x**3) * cos_diff + 3.*(-a*b/x + 1./x**3) * cos_sum + 
                         ((a*b)**2/(a-b) - 3.*(a-b) / x**2) * sin_diff - ((a*b)**2/(a+b) - 3.*(a+b)/x**2) * sin_sum)

    INT0 = 0.
    INT0 += np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / (a*b)**4

    return INT0


def W_ij_TopHat_MAIN_lowR(
        Pk, k, R1, R2, IDchange, OMAX, n):
    
    coeffs = cubic_spline_coeffs(k, 9. * Pk * k**2 * j2(R1*k) * j2(R2*k))
    if OMAX <= 0:
        INT0 = 0
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    return INT0 / (2. * np.pi ** 2)


def W_ii_TopHat_MAIN(
        Coeffs, x, a):
    #INT0 = 0.
    #if SmallK_args == 0:
    #    INT0 = 0
    #else:
    #    INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)
    Si_2ax, Ci_2ax = sp.sici(2. * a * x)
    cos_2ax = np.cos(2. * a * x)
    sin_2ax = np.sin(2. * a * x)
    lnx = np.log(x)
   
    T1_C3 = (21. * (lnx - Ci_2ax) + 6.5 * cos_2ax + a*x*sin_2ax) / a**8

    T2_C3 = ((5.25 - 5.5*(a*x)**2)*cos_2ax + (10.5 * (a*x) - (a*x)**3)*sin_2ax + 5.*(a*x)**2) / a**8

    T3_C3 = (((a*x)**2 - 10.5) * cos_2ax - 6.*a*x*sin_2ax) / a**8

    Coeff_tot_3 = (21. * (lnx - Ci_2ax) + (32.75 - 7.5 * (a*x)**2) * cos_2ax + (23.5 * a * x - (a*x)**3) * sin_2ax + 5. * (a*x)**2)
    Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x) + 0.5 * a ** 2 * x + 0.25 * a * sin_2ax)
    Coeff_tot_1 = (0.5 * a**2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x + 0.25 * (cos_2ax - 1.) / x**2)
    Coeff_tot_0 = (a**3 * Si_2ax / 3. + (cos_2ax - 3.) * a**2 / (6. * x) + a * sin_2ax / (3. * x**2) + (cos_2ax - 1.) / (6. * x**3))


    T1 = ((42.-4.*(a*x)**2)*np.sin(a*x)**2 - 24.*a*x*np.sin(a*x)*np.cos(a*x) + 2.*(a*x)**2)/a**8
    T2 = ((20.-4.*(a*x)**2)*np.cos(a*x)**2 + 16.*a*x*np.sin(a*x)*np.cos(a*x) + 2.*(a*x)**2) / a**6
    T3 = ((15. - 2.*(a*x)**2.) * np.sin(2.*a*x) - 10.*a*x * np.cos(2.*a*x))/ a**7

    check_3 = T1_C3 + T3_C3 - 2*T2_C3
    
    check_3 = T1 / x  + T2*x - 2*T3
    check_3 = T1 / x  + T2*x - 2*T3

    INT0 = 0.
    INT0 += np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / a ** 6
    

    return INT0


def W_ij_TopHat(Pk, k, R, n=0.96, OmaxSmallK=-1):

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    #dk  = k[1:] - k[:-1]
    #x0_dx = k[:-1] / dx

    Wij_out = np.zeros((len(R),len(R)))
    for i in range(len(R)):
        for j in range(i):
            IDchange = find_id_change(k,(R[i] * R[j]) ** -0.5)
            Wij_out[i,j] = W_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n)
            Wij_out[i,j] += W_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j])
            Wij_out[j,i] = Wij_out[i,j]
        IDchange = IDchange = find_id_change(k,1./R[i])
        Wij_out[i,i] = W_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[i], IDchange, OmaxSmallK, n)
        Wij_out[i,i] += W_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i])
        
    #ij_val = []
    #for i in range(len(R)):
    #    for j in range(i):
    #        ij_val.append((i,j))

    #pool = ThreadPool()
    #for i,j in ij_val:
    #    pool.apply_async(C_ij_apply_async, args=(i,j,Cij_out,R,k,Pk,Coeffs,OmaxSmallK, n))
    # close the thread pool
    #pool.close()
    # wait for all tasks to finish
    #pool.join()
    
    return Wij_out


def sigma2_TopHat_zeros_SplineIntegration(
        Pk, k, R, NkPerZeros=30, kmaxKern=0.,r_corr=0.,SmallKprecision=False,n=0.96,OmaxSmallK=8):
    InterpFunction = interpolate.CubicSpline(k, k ** 2 * Pk / (2. * np.pi**2))
    j0_Kzeros = findZeros_j0_kr(k[-1], r_corr)
    if SmallKprecision:
        SmallK_args = Pk[0], k[0], n, r_corr, OmaxSmallK
    else:
        SmallK_args = 0
    kRzeros = findZeros_xMax(k[-1] * R[0])
    OUT = np.zeros(len(R))
    for i in range(0,len(R)):
        OUT[i] = C_ij_TopHat_zeros_CORESplineIntegration(
            InterpFunction, k, R[i], R[i], r_corr, kRzeros, j0_Kzeros, SmallK_args, NkPerZeros, kmaxKern)
    return OUT


def sigma2_N_TopHat(
        Pk, k, R, N, NkPerZeros=30, kmaxKern=0.,r_corr=0.,SmallKprecision=False,n=0.96,OmaxSmallK=8):
    InterpFunction = interpolate.CubicSpline(k, k ** (2 + 2*N) * Pk / (2. * np.pi**2))
    j0_Kzeros = findZeros_j0_kr(k[-1], r_corr)
    if SmallKprecision:
        SmallK_args = Pk[0], k[0], n, r_corr, OmaxSmallK
    else:
        SmallK_args = 0
    kRzeros = findZeros_xMax(k[-1] * np.max(R))
    OUT = np.zeros(len(R))
    for i in range(0,len(R)):
        OUT[i] = C_ij_TopHat_zeros_CORESplineIntegration(
            InterpFunction, k, R[i], R[i], r_corr, kRzeros, j0_Kzeros, SmallK_args, NkPerZeros, kmaxKern)
    return OUT



def C_ij_Gauss_CORESplineIntegration(
        UnfiltInterp, k, R1, R2, r_corr, SmallK_args, NkPerZeros, kKern):
    if r_corr == 0:
        if kKern > 0:
            Func = lambda x: UnfiltInterp(x) * np.exp(-0.5*R1*R1*x*x) * np.exp(-0.5*R2*R2*x*x) * Gaussian_kernel(x,kKern)
        else:
            Func = lambda x: UnfiltInterp(x) * np.exp(-0.5*R1*R1*x*x) * np.exp(-0.5*R2*R2*x*x)
        Kinterp = np.logspace(k[0],10./R2,10*NkPerZeros)
        Kinterp = Kinterp[Kinterp <= k[-1]]
    else:
        if kKern > 0:
            Func = lambda x: UnfiltInterp(x) * np.exp(-0.5*R1*R1*x*x) * np.exp(-0.5*R2*R2*x*x) * j0(x * r_corr)* Gaussian_kernel(x,kKern)
        else:
            Func = lambda x: UnfiltInterp(x) * np.exp(-0.5*R1*R1*x*x) * np.exp(-0.5*R2*R2*x*x) * j0(x * r_corr)
        kZeros = findZeros_fromTHandj0(np.zeros(0), j0_Kzeros, R1, R2, r_corr, k[-1])
        Kinterp = Karray_building(k,kZeros,NkPerZeros)
    #kZeros = findZeros_fromAllZeros(xZeros,R1,R2,k[-1])
    if SmallK_args == 0:
        INT0 = 0
    else:
        Pk0, k0, n, r_corr, OMAX = SmallK_args
        if r_corr == 0:
            INT0 = IntegrationSmallK(Pk0, k0, n, R1, R2, Omax=OMAX)
        else:
            INT0 = IntegrationSmallK_correlation(Pk0, k0, n, R1, R2, r_corr, Omax=OMAX)
    #if kZeros[0] > k[0]:
    return INT0 + float(interpolate.CubicSpline(Kinterp, Func(Kinterp)).integrate(Kinterp[0], Kinterp[-1]))


def sigma2_N_Gaussian(
        Pk, k, R, N, NkPerZeros=30, kmaxKern=0.,r_corr=0.,SmallKprecision=False,n=0.96,OmaxSmallK=8):
    InterpFunction = interpolate.CubicSpline(k, k ** (2 + 2*N) * Pk / (2. * np.pi**2))
    #j0_Kzeros = findZeros_j0_kr(k[-1], r_corr)
    if SmallKprecision:
        SmallK_args = Pk[0], k[0], n, r_corr, OmaxSmallK
    else:
        SmallK_args = 0
    OUT = np.zeros(len(R))
    for i in range(0,len(R)):
        OUT[i] = C_ij_Gauss_CORESplineIntegration(
            InterpFunction, k, R[i], R[i], r_corr, SmallK_args, NkPerZeros, kmaxKern)
    return OUT

def sigma2_sharpK_SplineIntegration(Pk, k, k_integr, logarithmic=True, SmallKprecision=False,n=0.96):
    if logarithmic:
        k_integration = np.log(k_integr)
        k0 = np.log(k[0])
        InterpFunction = interpolate.CubicSpline(np.log(k), k ** 3 * Pk / (2. * np.pi ** 2))
    else:
        k_integration = k_integr
        k0 = k[0]
        InterpFunction = interpolate.CubicSpline(k, k ** 2 * Pk / (2. * np.pi**2))
    OUT = np.zeros(len(k_integration))
    ind_off = 0
    if SmallKprecision:
        if k_integr[0] > k[0]:
            offset = Pk[0] * k[0] ** 3 / (2 * np.pi ** 2 * (3 + n)) + InterpFunction.integrate(k0,k_integration[0])
        elif k_integr[0] == k[0]:
            offset = Pk[0] * k[0] ** 3 / (2 * np.pi ** 2 * (3 + n))
        else:
            while k_integr[ind_off] < k[0]:
                OUT[ind_off] = Pk[0] * k_integr[ind_off] ** (3 + n) / k[0] ** n / (2 * np.pi ** 2 * (3 + n))
                ind_off += 1
            if logarithmic:
                offset = Pk[0] * k[0] ** -n / (2 * np.pi ** 2 * (3 + n)) + InterpFunction.integrate(k0,k_integration[ind_off])
            else:
                offset = Pk[0] * k[0] ** -n / (2 * np.pi ** 2 * (3 + n)) + InterpFunction.integrate(k0,k_integration[ind_off])
    else:
        offset = 0
    for i in range(ind_off,len(k_integration)):
        OUT[i] = InterpFunction.integrate(k_integration[ind_off],k_integration[i]) + offset
    return OUT


def C_ij_TopHat_SplineIntegration_R1R2(
        Pk, k, R1, R2, NkPerZeros, kmaxKern=0., r_corr=0,SmallKprecision=False,n=0.96,OmaxSmallK=8):
    InterpFunction = interpolate.CubicSpline(k, k ** 2 * Pk / (2. * np.pi**2))
    if r_corr == 0:
        Unfilt = InterpFunction
    else:
        Unfilt = lambda x: InterpFunction(x) * j0(x * r_corr)
    j0_Kzeros = findZeros_j0_kr(k[-1], r_corr)
    if SmallKprecision:
        SmallK_args = Pk[0], k[0], n, r_corr, OmaxSmallK
    else:
        SmallK_args = 0
    return C_ij_TopHat_zeros_CORESplineIntegration(
        Unfilt, k, R1, R2, r_corr, findZeros_xMax(k[-1] * max(R1,R2)), j0_Kzeros, SmallK_args, NkPerZeros,kmaxKern)

def cholensky_decomposition_matrix_from_Cij(Cij,Cijtype='numpy_narray',out_type='linearized'):
    if Cijtype == 'numpy_narray':
        lenR = Cij.shape[0]
        L_ij = dict()
        for i in range(0,lenR):
            L_ij[i] = np.zeros(i+1)
        for j in range(0,lenR):
            L_ij[j][j] = (Cij[j,j] - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, lenR):
                L_ij[i][j] = (Cij[i,j] - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    elif Cijtype == 'dict':
        L_ij = Dict.empty(types.int64, float_array)
        for i in Cij.keys():
            L_ij[i] = np.zeros(i+1)
        lenR = i + 1
        for j in range(0,lenR):
            L_ij[j][j] = (Cij[j][j] - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, lenR):
                L_ij[i][j] = (Cij[i][j] - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    else:
        return -1
    if out_type == 'linearized':
        L_ij_lin = np.zeros(int((lenR+1)*lenR/2))
        j0 = 0
        for i in range(lenR):
            for j in range(0, i + 1):
                L_ij_lin[j0 + j] = L_ij[i][j]
            j0 += i + 1
        return L_ij_lin
    return L_ij


def Cij_for_2ptcorr_dict(i,j,Cij_0,Cij_r,CijDim):
    if i < CijDim:
        return Cij_0[i][j]
    else:
        if j < CijDim:
            return Cij_r[i-CijDim][j]
        else:
            return Cij_0[i-CijDim][j-CijDim]
        
        

def Cij_for_2ptcorr_numpy_array(i,j,Cij_0,Cij_r,CijDim):
    if i < CijDim:
        return Cij_0[i,j]
    else:
        if j < CijDim:
            return Cij_r[i-CijDim,j]
        else:
            return Cij_0[i-CijDim,j-CijDim]

def cholensky_decomposition_matrix_blocks_from_Cij_Cijr(
        Cij_0,Cij_r,Cijtype='numpy array',out_type='linearized'):
    if Cijtype == 'dict':
        for i in Cij_0.keys():
            i0 = i
        for i in Cij_0.keys():
            i1 = i
        if i0 != i1:
            return 'Error. dimension of Cij_0 is different from Cij_r'
        CijDim = i0+1
        L_ij = Dict.empty(types.int64, float_arrayDim1)
        for i in range(0,2*CijDim):
            L_ij[i] = np.zeros(i+1)
        for j in range(0,2*CijDim):
            L_ij[j][j] = (Cij_for_2ptcorr_dict(j,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, 2*CijDim):
                L_ij[i][j] = (Cij_for_2ptcorr_dict(i,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    else:
        L_ij = Dict.empty(types.int64, float_arrayDim1)
        CijDim = Cij_0.shape[0]
        if CijDim != Cij_r.shape[0]:
            return 'Error. dimension of Cij_0 is different from Cij_r'

        for i in range(0,2*CijDim):
            L_ij[i] = np.zeros(i+1)
        for j in range(0,2*CijDim):
            L_ij[j][j] = (Cij_for_2ptcorr_numpy_array(j,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, 2*CijDim):
                L_ij[i][j] = (Cij_for_2ptcorr_numpy_array(i,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    L_ij_0 = Dict.empty(types.int64, float_arrayDim1)
    L_ij_r = Dict.empty(types.int64, float_arrayDim1)
    for i in range(0, CijDim):
        L_ij_0[i] = L_ij[i]
        L_ij_r[i] = L_ij[i+CijDim]
    if out_type == 'linearized':
        L_ij0_lin = np.zeros(int((CijDim+1)*CijDim/2))
        j0 = 0
        for i in range(CijDim):
            for j in range(0, i + 1):
                L_ij0_lin[j0 + j] = L_ij[i][j]
            j0 += i + 1

        L_ijr_lin = np.zeros(int((CijDim+1)*CijDim/2))
        j0 = 0
        for i in range(CijDim):
            for j in range(0, i + 1):
                L_ijr_lin[j0 + j] = L_ij_r[i][j+CijDim]
            j0 += i + 1
    
        L_ijr0 = np.zeros(CijDim*CijDim) #np.zeros((CijDim,CijDim))
        progr = 0
        for i in range(CijDim):
            #L_ijr0[i,:] = L_ij_r[i][:CijDim]
            L_ijr0[progr:progr+CijDim] = L_ij_r[i][:CijDim]
            progr += CijDim
        return L_ij0_lin, L_ijr0, L_ijr_lin
    return L_ij_0, L_ij_r


def Cholensky_decomposition_matrix_for2PTcorr_from_Cij_OLD(Cij_0,Cij_r,Cijtype='numpy array'):
    if Cijtype == 'dict':
        for i in Cij_0.keys():
            i0 = i
        for i in Cij_0.keys():
            i1 = i
        if i0 != i1:
            return 'Error. dimension of Cij_0 is different from Cij_r'
        CijDim = i0+1
        L_ij = Dict.empty(types.int64, float_arrayDim1)
        for i in range(0,2*CijDim):
            L_ij[i] = np.zeros(i+1)
        for j in range(0,2*CijDim):
            L_ij[j][j] = (Cij_for_2ptcorr_dict(j,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, 2*CijDim):
                L_ij[i][j] = (Cij_for_2ptcorr_dict(i,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    else:
        L_ij = Dict.empty(types.int64, float_arrayDim1)
        CijDim = Cij_0.shape[0]
        if CijDim != Cij_r.shape[0]:
            return 'Error. dimension of Cij_0 is different from Cij_r'

        for i in range(0,2*CijDim):
            L_ij[i] = np.zeros(i+1)
        for j in range(0,2*CijDim):
            L_ij[j][j] = (Cij_for_2ptcorr_numpy_array(j,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[j][:j] ** 2)) ** 0.5
            for i in range(j+1, 2*CijDim):
                L_ij[i][j] = (Cij_for_2ptcorr_numpy_array(i,j,Cij_0,Cij_r,CijDim) - np.sum(L_ij[i][:j] * L_ij[j][:j])) / L_ij[j][j]
    L_ij_0 = Dict.empty(types.int64, float_arrayDim1)
    L_ij_r = Dict.empty(types.int64, float_arrayDim1)
    for i in range(0, CijDim):
        L_ij_0[i] = L_ij[i]
        L_ij_r[i] = L_ij[i+CijDim]
    return L_ij_0, L_ij_r


def S2sk_from_Pk(kS2, K, Pk, n=0.96):
    k0 = 1e-5
    LargeK = interpolate.CubicSpline(K, K ** 2 * Pk / (2. * np.pi ** 2.))
    A = LargeK(k0)
    # SmallK  = lambda x: A * (x / k0) ** (n + 2.)
    S2sk = np.zeros_like(kS2)
    for i in range(0, len(kS2)):
        if kS2[i] <= k0:
            S2sk[i] = A * k0 * (kS2[i] / k0) ** (n + 3.) / (n + 3.)
        else:
            S2sk[i] = A * k0 / (n + 3.) + LargeK.integrate(k0, kS2[i])
    return S2sk

def K_Pk_camb(sig8,filePATH,kmaxKern=0.,NkPerZeros=51,n_spectral=0.96,SmallK=True,OmaxSk=8):
    camb = pandas.read_csv(filePATH, sep='\s+', header=0).values
    Norm = sig8 ** 2. / C_ij_TopHat_SplineIntegration_R1R2(
        camb[:, 1], camb[:, 0], 8., 8., NkPerZeros, kmaxKern=kmaxKern,
        r_corr=0.,SmallKprecision=SmallK,n=n_spectral,OmaxSmallK=OmaxSk)
    return camb[:, 0], Norm * camb[:, 1]

def find_Pk_norm(sig8_z0,filePATH_z0,kmaxKern=0.,NkPerZeros=51,n_spectral=0.96,SmallK=True,OmaxSk=8):
    camb = pandas.read_csv(filePATH_z0, sep='\s+', header=0).values
    return sig8_z0 ** 2. / C_ij_TopHat_SplineIntegration_R1R2(
        camb[:, 1], camb[:, 0], 8., 8., NkPerZeros, kmaxKern=kmaxKern,
        r_corr=0.,SmallKprecision=SmallK,n=n_spectral,OmaxSmallK=OmaxSk)


def K_Pk_camb_ginvenNorm(filePATH,Norm):
    camb = pandas.read_csv(filePATH, sep='\s+', header=0).values
    return camb[:, 0], Norm * camb[:, 1]
