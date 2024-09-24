import numpy as np
from scipy import optimize
from numba import jit
from numba.core import types
from numba.typed import Dict



def delta_lin_underdensity(D_nl):
    Root = lambda x: D_nl - ((9./2.) * ((np.sinh(x) - x) ** 2.)/(np.cosh(x)-1.) ** 3. - 1.)
    eta = optimize.brenth(Root,1e-6,30)
    DELTA_lin = -(3./20.) * (6. * (np.sinh(eta)-eta))**(2./3.)
    #Radius_ratio = (1.+D_nl)**(-1./3.)
    return DELTA_lin#, Radius_ratio


def delta_NL_underdensity(DELTA_lin):
    Root = lambda eta: DELTA_lin + (3./20.) * (6. * (np.sinh(eta)-eta))**(2./3.)
    x = optimize.brenth(Root,1e-6,30)
    D_nl = ((9./2.) * ((np.sinh(x) - x) ** 2.)/(np.cosh(x)-1.) ** 3. - 1.)
    #Radius_ratio = (1.+D_nl)**(-1./3.)
    return D_nl#, Radius_ratio


def delta_lin_overdensity(D_nl):
    Root = lambda x: D_nl - ((9./2.) * ((x - np.sin(x)) ** 2.)/(1. - np.cos(x)) ** 3. - 1.)
    eta = optimize.brenth(Root,1e-6,np.pi * 2-1e-6)
    DELTA_lin = (3./20.) * (6. * (eta - np.sin(eta)))**(2./3.)
    #Radius_ratio = (1.+D_nl)**(-1./3.)
    return DELTA_lin


def delta_NL_overdensity(DELTA_lin):
    Root = lambda eta: DELTA_lin - (3./20.) * (6. * (eta - np.sin(eta)))**(2./3.)
    x = optimize.brenth(Root,1e-6,1e6)
    D_nl = ((9./2.) * ((x - np.sin(x)) ** 2.)/(1. - np.cos(x)) ** 3. - 1.)
    #Radius_ratio = (1.+D_nl)**(-1./3.)
    return D_nl

def delta_NL_from_lin(DELTA_lin):
    OUT = np.zeros(len(DELTA_lin))
    for i in range(len(DELTA_lin)):
        if DELTA_lin[i] < 0.:
            OUT[i] = delta_NL_underdensity(DELTA_lin[i])
        elif DELTA_lin[i] > 0.:
            OUT[i] = delta_NL_overdensity(DELTA_lin[i])
    return OUT



@jit(nopython=True)
def find_CL_brutal_force(sample,range_array):
    xarr = np.sort(sample)
    len_arr = len(xarr)
    CL_out = np.zeros((len(range_array),2))
    for ii in range(len(range_array)):
        len_int = int(round(len_arr * range_array[ii]))
        ind_out = 0
        delta = xarr[-1] - xarr[0]
        saveind = False
        for j in range(0,len_arr-len_int):
            delta_tmp = xarr[j+len_int] - xarr[j]
            saveind = delta_tmp < delta
            ind_out = j * int(saveind) + ind_out * int(~saveind)
            delta = min(delta,delta_tmp)
        CL_out[ii,0] = xarr[ind_out]
        CL_out[ii,1] = xarr[ind_out + len_int]
    return CL_out

