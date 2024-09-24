import numpy as np
from numba import jit, prange
from numba.core import types
#from numba.typed import Dict
#from interpolation.splines import UCGrid, CGrid, nodes, eval_linear, filter_cubic, eval_cubic
#import pandas
#from scipy import integrate, interpolate, optimize, signal
import scipy.special as sp
#from multiprocessing import Pool, Process, Manager
#from multiprocessing.pool import ThreadPool
from .spline import *


@jit(nopython=True,cache=True)
def TOPHAT_Taylor(x):
    return 1. - x**2/10. +x**4/280. - x**6/15120. +x**8/1330560.

@jit(nopython=True,cache=True)
def dx_squareTOPHAT_Taylor(x):
    return 2. * (-x/5. +x**3/70. - x**5/2520. +x**7/166320.) * (1. - x**2/10. +x**4/280. - x**6/15120. +x**8/1330560.)

@jit(nopython=True,cache=True)
def top_hat_rk_HR(r, k,X0=1e-1):
    x = r * k
    OUT = np.zeros(len(x))
    OUT[x <= X0] = TOPHAT_Taylor(x[x <= X0])
    OUT[x > X0] = 3. * (np.sin(x[x > X0]) - x[x > X0] * np.cos(x[x > X0])) / (x[x > X0] ** 3.)
    return OUT


@jit(nopython=True,cache=True)
def dr_square_top_hat_rk_HR(r, k,X0=1e-1):
    x = r * k
    x = r * k
    OUT = np.zeros(len(x))
    mask = (x <= X0)
    OUT[mask] = dx_squareTOPHAT_Taylor(x[mask])
    mask[:] = ~mask
    OUT[mask] = 18. * ((x[mask]**2-3.) * np.sin(x[mask])**2 - 3. * x[mask]**2 * np.cos(x[mask])**2 + (6.*x[mask] - x[mask]**3) * np.sin(x[mask]) * np.cos(x[mask])) / x[mask] ** 7
    return OUT * k

@jit(nopython=True,cache=True)
def Factorial(N):
    if N <= 1:
        return 1
    return np.prod(np.arange(N)+1)

@jit(nopython=True,cache=True)
def a_fac(pow):
    if pow % 2 != 0:
        return 0.
    elif pow == 0:
        return 1.
    N = pow + 3
    return 3. * (N-1) / Factorial(N) * (-1)**((N + 1) / 2)

@jit(nopython=True,cache=True)
def IntegrationSmallK(Pk0,k0,n,R1,R2,Omax=8):
    ALPHA = np.zeros(9)
    ALPHA[0] = 1.
    ALPHA[2] = a_fac(2)
    ALPHA[4] = a_fac(4) * (R1**4 + R2 **4) + \
               a_fac(2) ** 2 * R1 ** 2 * R2 **2
    ALPHA[6] = a_fac(6) * (R1 ** 6 + R2 ** 6) + \
               a_fac(2) * a_fac(4) * R1 ** 2 * R2 ** 2 * (R1 ** 4 + R2 ** 4)
    ALPHA[8] = a_fac(8) * (R1 ** 8 + R2 ** 8) + \
               a_fac(2) * a_fac(6) * R1 ** 2 * R2 ** 2 * (R1 ** 6 + R2 ** 6) + \
               a_fac(4) ** 2 * R1 ** 4 * R2 ** 4
    return Pk0 / (2.*np.pi**2) * np.sum(ALPHA[:Omax+1] * k0 ** (np.arange(Omax+1) + 3) / (np.arange(Omax+1) + 3 + n))
    #return Pk0 / (2.*np.pi**2) * np.sum(ALPHA[:Omax+1] * k0 ** 3 / (np.arange(Omax+1) + 3 + n))





##########
def C_ij_TopHat_MAIN(Coeffs, x, a, b):
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

    '''
    T1_C3 = 0.5 * (Ci_diff - Ci_sum)
    T1_C2 = 0.5 * ((b-a) * Si_diff + (a+b) * Si_sum - cos_diff / x + cos_sum / x)
    T1_C1 = 0.25*(-(a-b) ** 2 * Ci_diff + (a+b) ** 2 * Ci_sum - cos_diff / x ** 2 + cos_sum / x ** 2 + (a-b) * sin_diff / x - (a+b) * sin_sum / x )
    T1_C0  = ((a-b)**3 * Si_diff + ((a-b)**2*x**2-2.) * cos_diff/ x**3 + (a-b) * sin_diff / x ** 2 - 
              (a+b)**3 * Si_sum - ((a+b)**2*x**2-2.) * cos_sum / x**3 - (a+b) * sin_sum / x**2) / 12
    
    T2_C3 = 0.5 * (x * sin_diff / (a-b) + cos_diff / (a-b) ** 2 + 
                   x * sin_sum / (a+b) + cos_sum / (a+b) ** 2)
    T2_C2 = 0.5 * (sin_diff / (a-b) + sin_sum / (a+b))
    T2_C1 = 0.5 * (Ci_diff + Ci_sum)
    T2_C0 = -0.5 * ((a-b) * Si_diff + (a+b) * Si_sum + cos_diff / x + cos_sum / x)

    T3_C3 = -0.5 * (cos_diff / (a-b) + cos_sum / (a+b))
    T3_C2 = 0.5 * (Si_diff + Si_sum)
    T3_C1 = 0.5 * ((a-b) * Ci_diff - sin_diff / x + 
                   (a+b) * Ci_sum - sin_sum / x)
    T3_C0 = 0.25 * (-(a-b)**2 * Si_diff - (a-b) * cos_diff / x - sin_diff / x ** 2 -
                     (a+b)**2 * Si_sum - (a+b) * cos_sum / x - sin_sum / x ** 2)

    T4_C3 = 0.5 * (cos_diff / (a-b) - cos_sum / (a+b))
    T4_C2 = 0.5 * (-Si_diff + Si_sum)
    T4_C1 = 0.5 * ((b-a) * Ci_diff + sin_diff / x + 
                   (a+b) * Ci_sum - sin_sum / x)
    T4_C0 = 0.25 * ((a-b)**2 * Si_diff + (a-b) * cos_diff / x + sin_diff / x ** 2 -
                    (a+b)**2 * Si_sum - (a+b) * cos_sum / x - sin_sum / x ** 2)
    '''

    Coeff_tot_3 = 0.5 * (Ci_diff - Ci_sum + a*b*x*(sin_diff / (a-b) + sin_sum / (a+b)) + 
                         (a*b / (a-b)**2 - 1) * cos_diff + (a*b / (a+b)**2 + 1) * cos_sum)
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





    

def C_ii_TopHat_MAIN(
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

    #T1_C3 = 0.5 * (lnx - Ci_2ax)
    #T1_C2 = a * Si_2ax + (cos_2ax - 1.) / (2. * x)
    #T1_C1 = a * a * Ci_2ax - a * sin_2ax / (2. * x) + (cos_2ax - 1.) / (4. * x * x)
    #T1_C0  = -2. * a*a*a / 3. * Si_2ax - (a * a/(3. * x) - 1. / (6. * x**3)) * cos_2ax - a * sin_2ax / (6. * x**2) - 1. / (6. * x**3)
    
    #T2_C3 = cos_2ax / (8. * a*a) + x * sin_2ax / (4. * a) + 0.25 * x ** 2
    #T2_C2 = 0.5 * x + sin_2ax / (4. * a)
    #T2_C1 = 0.5 * Ci_2ax + 0.5 * lnx
    #T2_C0 = -a * Si_2ax - (cos_2ax + 1.) / (2. * x)

    #T3_C3 = -cos_2ax / (2. * a)
    #T3_C2 = Si_2ax
    #T3_C1 = 2. * a * Ci_2ax - sin_2ax / x
    #T3_C0 = -2. * a ** 2 * Si_2ax - sin_2ax / (2. * x**2) - a * cos_2ax / x
    
    
    Coeff_tot_3 = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x * sin_2ax + 0.25 * a**2 * x**2)
    Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x) + 0.5 * a ** 2 * x + 0.25 * a * sin_2ax)
    Coeff_tot_1 = (0.5 * a**2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x + 0.25 * (cos_2ax - 1.) / x**2)
    Coeff_tot_0 = (a**3 * Si_2ax / 3. + (cos_2ax - 3.) * a**2 / (6. * x) + a * sin_2ax / (3. * x**2) + (cos_2ax - 1.) / (6. * x**3))


    #check_3 = T1_C3 / a**6 + T2_C3 / a**4 - T3_C3 / a**5
    #check_2 = T1_C2 / a**6 + T2_C2 / a**4 - T3_C2 / a**5
    #check_1 = T1_C1 / a**6 + T2_C1 / a**4 - T3_C1 / a**5
    #check_0 = T1_C0 / a**6 + T2_C0 / a**4 - T3_C0 / a**5

    INT0 = 0.
    INT0 += np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * Coeffs[:,3])
    INT0 += np.sum((Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * Coeffs[:,2])
    INT0 += np.sum((Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * Coeffs[:,1])
    INT0 += np.sum((Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * Coeffs[:,0])
    INT0 *= 9. / (2. * np.pi**2) / a ** 6
    

    return INT0


def C_ij_TopHat_MAIN_lowR(
        Pk, k, R1, R2, IDchange, OMAX, n):
    
    coeffs = cubic_spline_coeffs(k, Pk * k**2 * top_hat_rk_HR(R1, k) * top_hat_rk_HR(R2, k))
    if OMAX <= 0:
        INT0 = 0.
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, Omax=OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    return INT0 / (2. * np.pi ** 2)


def C_ii_TopHat_MAIN_lowR(
        Pk, k, R, IDchange, OMAX, n):
    
    coeffs = cubic_spline_coeffs(k, Pk * k**2 * top_hat_rk_HR(R, k)**2)
    if OMAX <= 0:
        INT0 = 0
    else:
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R, R, Omax=OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    return INT0 / (2. * np.pi ** 2)

@jit(nopython=True)
def find_id_change(x_array,x_max):
    ind = 0
    len_arr_mn1 = len(x_array) - 1
    while (x_array[ind] < x_max) & (ind < len_arr_mn1):
        ind += 1
    return ind


def C_ij_apply_async(i,j,Cij_out,R,k,Pk,Coeffs,OmaxSmallK, n):
    IDchange = find_id_change(k,(R[i] * R[j]) ** -0.5)
    Cij_out[i,j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n)
    Cij_out[i,j] += C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j])
    Cij_out[j,i] = Cij_out[i,j]



def C_ij_TopHat(Pk, k, R, n=0.96, OmaxSmallK=-1):

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    #dk  = k[1:] - k[:-1]
    #x0_dx = k[:-1] / dx

    Cij_out = np.zeros((len(R),len(R)))
    for i in range(len(R)):
        for j in range(i):
            IDchange = find_id_change(k,(R[i] * R[j]) ** -0.5)
            Cij_out[i,j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n)
            Cij_out[i,j] += C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j])
            #print(i,j,' id:',IDchange,'t1:',C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n),'t2:',C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i],R[j]),'tot,',Cij_out[i,j])
            Cij_out[j,i] = Cij_out[i,j]
        IDchange = find_id_change(k,1./R[i])
        Cij_out[i,i] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n)
        Cij_out[i,i] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i])
        #print(i,' id:',IDchange,'t1:',C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n),'t2:',C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]),'tot,',Cij_out[i,i])

        
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
    
    return Cij_out
    

def sigma2_TopHat(
        Pk, k, R, n=0.96,OmaxSmallK=-1):

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    OUT = np.zeros(len(R))
    for i in range(0,len(R)):
        IDchange = find_id_change(k,1./R[i])
        OUT[i] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n)
        OUT[i] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i])
    return OUT

def sigma2_2_TopHat_numdiff(
        Pk, k, R, n=0.96,OmaxSmallK=-1,dRperc=5e-3):

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    grid_R1R2 = np.zeros((len(R),3))
    for i in range(0,len(R)):
        j=0 #--
        IDchange = find_id_change(k,1./R[i])
        grid_R1R2[i,j] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1.-dRperc), IDchange, OmaxSmallK, n)
        grid_R1R2[i,j] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1.-dRperc))
        j=1 #+-
        grid_R1R2[i,j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i]*(1.+dRperc), R[i]*(1.-dRperc), IDchange, OmaxSmallK, n)
        grid_R1R2[i,j] += C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1.+dRperc), R[i]*(1.-dRperc))
        j=2 #++
        grid_R1R2[i,j] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1.+dRperc), IDchange, OmaxSmallK, n)
        grid_R1R2[i,j] += C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1.+dRperc))
    #grid_dR1R2 = np.zeros(grid_R1R2.shape)
    #grid_dR1R2[:,0] = (grid_R1R2[:,2] - grid_R1R2[:,1]) / (2. * R*dRperc)
    #grid_dR1R2[:,1] = (grid_R1R2[:,1] - grid_R1R2[:,0]) / (2. * R*dRperc)
    OUT = 0.25 * (grid_R1R2[:,2] - 2. * grid_R1R2[:,1] + grid_R1R2[:,0]) / (R*dRperc)**2
    return OUT



def sigma2_d2_2_TopHat_numdiff(
        Pk, k, R, accuracy=1, n=0.96,OmaxSmallK=-1,dRperc=5e-3):

    Coeffs = cubic_spline_coeffs(k,Pk)
    explicit_from_implicit_coeffs(Coeffs,k)

    OUT = np.zeros(len(R))
    if accuracy < 1:
        accuracy = 1
    if accuracy == 1:
        c_deriv = np.array([1.,-2.,1.])
    elif accuracy == 2:
        c_deriv = np.array([-1./12.,4./3.,-5./2.,4./3.,-1./12.])
    elif accuracy == 3:
        c_deriv = np.array([1./90., -3./20., 3./2.,-49./18.,3./2., -3./20., 1./90.])
    else:
        accuracy = 4
        c_deriv = np.array([-1./560., 8./315.,-1./5., 8./5., -205./72., 8./5.,-1./5., 8./315., -1./560.])

    nderiv = 2 * accuracy + 1
    for i in range(0,len(R)):
        IDchange = find_id_change(k,1./R[i])
        for jj in range(nderiv):
            j_h = jj - accuracy
            OUT[i] += c_deriv[jj]**2 * (
                C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1. + j_h * dRperc), IDchange, OmaxSmallK, n) + 
                C_ii_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i]*(1. + j_h * dRperc)))
            for kk in range(jj):
                k_h = kk - accuracy
                OUT[i] += 2. * c_deriv[jj] * c_deriv[kk] * (
                    C_ij_TopHat_MAIN_lowR(Pk, k, R[i] * (1. + k_h * dRperc), R[i] * (1. + j_h * dRperc), IDchange, OmaxSmallK, n) + 
                    C_ij_TopHat_MAIN(Coeffs[IDchange:,:], k[IDchange:], R[i] * (1. + k_h * dRperc), R[i] * (1. + j_h * dRperc)))
        OUT[i] /= (R[i]*dRperc)**4
    return OUT








def dSdR_TopHat_MAIN_lowR(Pk,k, R, IDchange, OMAX, n):
    

    Pk_k2_w1w2 = Pk * k * k * dr_square_top_hat_rk_HR(R,k)
    
    coeffs = cubic_spline_coeffs(k, Pk_k2_w1w2)

    
    INT0 = 0.
    if (OMAX > 0):
        INT0 += IntegrationSmallK(Pk[0], k[0], n, R, R, OMAX)

    INT0 += np.sum((coeffs[:IDchange,0] + 
                    coeffs[:IDchange,1] / 2. +
                    coeffs[:IDchange,2] / 3. + 
                    coeffs[:IDchange,3] / 4.) * (k[1:IDchange+1] -  k[:IDchange]))
    #for i in range(IDchange):
    #    INT0 += (coeffs[i][0] + 
    #             coeffs[i][1] / 2. +
    #             coeffs[i][2] / 3. + 
    #             coeffs[i][3] / 4.) * (k[i+1] -  k[i])
    
    
    INT0 /= 2. * np.pi * np.pi
    
    return INT0


def dSdR_TopHat_MAIN(coeffs, x, a, IDchange):

    a2 = a*a
    a3 = a*a*a
    a6 = a3*a3

    Si_2ax, Ci_2ax = sp.sici(2. * a * x[IDchange:])
    lnx = np.log(x[IDchange:])
    sin_2ax = np.sin(2. * a * x[IDchange:])
    cos_2ax = np.cos(2. * a * x[IDchange:])

    Coeff_tot_3 = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x[IDchange:] * sin_2ax + 0.25 * a2 * x[IDchange:]*x[IDchange:])
    Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x[IDchange:]) + 0.5 * a2 * x[IDchange:] + 0.25 * a * sin_2ax)
    Coeff_tot_1 = (0.5 * a2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x[IDchange:] + 0.25 * (cos_2ax - 1.) / (x[IDchange:]*x[IDchange:]))
    Coeff_tot_0 = (a3 * Si_2ax / 3. + (cos_2ax - 3.) * a2 / (6. * x[IDchange:]) + a * sin_2ax / (3. * x[IDchange:]*x[IDchange:]) + (cos_2ax - 1.) / (6. * x[IDchange:]*x[IDchange:]*x[IDchange:]))

    da_Coeff_tot_3 = 0.5 * (a*x[IDchange:]*x[IDchange:] - 1./a) * cos_2ax - x[IDchange:]*sin_2ax + 0.5*a*x[IDchange:]*x[IDchange:]
    da_Coeff_tot_2 = 0.5 * a*x[IDchange:] * cos_2ax - 0.75 * sin_2ax + a*x[IDchange:]
    da_Coeff_tot_1 = a * (lnx - Ci_2ax + 0.5*cos_2ax)
    da_Coeff_tot_0 = a2 * Si_2ax  + a * (cos_2ax - 1.) / x[IDchange:]

    INT0 = np.sum((da_Coeff_tot_3[1:] - da_Coeff_tot_3[:-1]) * coeffs[IDchange:,3] + 
                  (da_Coeff_tot_2[1:] - da_Coeff_tot_2[:-1]) * coeffs[IDchange:,2] + 
                  (da_Coeff_tot_1[1:] - da_Coeff_tot_1[:-1]) * coeffs[IDchange:,1] + 
                  (da_Coeff_tot_0[1:] - da_Coeff_tot_0[:-1]) * coeffs[IDchange:,0])
    INT0 -= 6./a * np.sum((Coeff_tot_3[1:] - Coeff_tot_3[:-1]) * coeffs[IDchange:,3] +
                          (Coeff_tot_2[1:] - Coeff_tot_2[:-1]) * coeffs[IDchange:,2] + 
                          (Coeff_tot_1[1:] - Coeff_tot_1[:-1]) * coeffs[IDchange:,1] + 
                          (Coeff_tot_0[1:] - Coeff_tot_0[:-1]) * coeffs[IDchange:,0])    
    INT0 *= 9. / (2. * np.pi * np.pi) / a6

    return INT0



def dSdR_TopHat(Pk, k, R, n=0.96, OmaxSmallK=-1):

    len_R = R.shape[0]

    coeffs = cubic_spline_coeffs(k, Pk)    
    explicit_from_implicit_coeffs(coeffs, k)


    OUT = np.empty(len_R)


    for i in range(len_R):
        IDchange = find_id_change(k,1./R[i])
        OUT[i] = dSdR_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n)
        OUT[i] += dSdR_TopHat_MAIN(coeffs, k, R[i],IDchange)


    return OUT
