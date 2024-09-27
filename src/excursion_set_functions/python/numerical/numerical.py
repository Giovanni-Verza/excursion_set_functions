import numpy as np
from numba import jit, prange, get_num_threads
try:
    from numba import get_thread_id
except:
    from numba.np.ufunc.parallel import _get_thread_id as get_thread_id
from numba.core import types
from numba.typed import Dict

float_array = types.float64[::1]
float_array2D = types.float64[:,::1]
int_array = types.int64[::1]
int_array2D = types.int64[:,::1]





@jit(nopython=True)
def first_crossing_perCore_scalar_barrier_single_numba(F_ij_reshaped, N_paths, N_Rfilt, delta_c):
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    RAND = np.empty(N_Rfilt)
    progr_ind_out = 0
    FiltPath = 0.
    bool_cond = True
    for nn in range(0,N_paths):
        progr_ind_out = 0
        FiltPath = 0.
        i=0
        bool_cond = True
        while bool_cond:
            RAND[i] = np.random.normal()
            FiltPath = 0.
            for s in range(0,i+1):
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s]
            bool_cond = (FiltPath < delta_c)
            progr_ind_out += i + 1
            i += 1
            bool_cond &= (i < N_Rfilt)
        NumCrossing[i-1] += (FiltPath >= delta_c)
    return NumCrossing

@jit(nopython=True,parallel=True)
def first_crossing_scalar_barrier_single_numba(F_ij_reshaped, N_paths, delta_c, nCPU):
    N_Rfilt = int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2))
    Cross_perCore = Dict.empty(types.int64, int_array)
    for nn in range(0, nCPU):
        Cross_perCore[nn] = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in prange(0,nCPU):
        N_paths_core = int(N_paths / nCPU)
        N_paths_core += nn < (N_paths % nCPU)
        Cross_perCore[nn][:] = first_crossing_perCore_scalar_barrier_single_numba(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c)

    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in range(0,nCPU):
        NumCrossing += Cross_perCore[nn]
    return NumCrossing





@jit(nopython=True)
def first_crossing_perCore_array_barrier_single_numba(F_ij_reshaped, N_paths, N_Rfilt, delta_c):
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    RAND = np.empty(N_Rfilt)
    progr_ind_out = 0
    FiltPath = 0.
    bool_cond = True
    for nn in range(0,N_paths):
        progr_ind_out = 0
        FiltPath = 0.
        i=0
        bool_cond = True
        while bool_cond:
            RAND[i] = np.random.normal()
            FiltPath = 0.
            for s in range(0,i+1):
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s]
            bool_cond = (FiltPath < delta_c[i])
            progr_ind_out += i + 1
            i += 1
            bool_cond &= (i < N_Rfilt)
        NumCrossing[i-1] += (FiltPath >= delta_c[i-1])
    return NumCrossing

@jit(nopython=True,parallel=True)
def first_crossing_array_barrier_single_numba(F_ij_reshaped, N_paths, delta_c, nCPU):
    N_Rfilt = int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2))
    Cross_perCore = Dict.empty(types.int64, int_array)
    for nn in range(0, nCPU):
        Cross_perCore[nn] = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in prange(0,nCPU):
        N_paths_core = int(N_paths / nCPU)
        N_paths_core += nn < (N_paths % nCPU)
        Cross_perCore[nn][:] = first_crossing_perCore_array_barrier_single_numba(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c)

    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    for nn in range(0,nCPU):
        NumCrossing += Cross_perCore[nn]
    return NumCrossing



def first_crossing_single_barrier(F_ij_reshaped, N_paths, delta_c, nCPU=-1):
    if (nCPU < 0) | (nCPU > get_num_threads()):
        nCPU = get_num_threads()
    if np.isscalar(delta_c):
        return first_crossing_scalar_barrier_single_numba(F_ij_reshaped, N_paths, delta_c,nCPU)
    return first_crossing_array_barrier_single_numba(F_ij_reshaped, N_paths, delta_c,nCPU)


@jit(nopython=True)
def histo_profile(histo,FiltPath,N_Rfilt,binsize,offset,nbins):
    for i in range(N_Rfilt):
        ind = int((FiltPath[i] - offset) / binsize)
        if (ind < nbins) & (ind >= 0):
            histo[ind,i] += 1



@jit(nopython=True)
def first_crossing_profile_perCore_array_barrier_single(
    F_ij_reshaped, N_paths, N_Rfilt, delta_c,
    hist_dict,binsize,offset,nbins):
    
    FiltPath = np.zeros(N_Rfilt)
    FiltPath_mean = np.zeros((N_Rfilt,N_Rfilt))
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    RAND = np.empty(N_Rfilt)
    #IDcross = np.zeros(N_paths, dtype=np.int_)
    #RANDmatr = np.zeros((N_Rfilt,N_paths), dtype=np.float_)
    Nx=0
    uncrossed = True
    progr = 0
    for nn in range(0,N_paths):
        progr_ind_out = 0
        #FiltPath = 0.
        i=0
        uncrossed = True
        while uncrossed & (i < N_Rfilt):
            RAND[i] = np.random.normal()
            FiltPath[i] = 0.
            for s in range(0,i+1):
                FiltPath[i] += F_ij_reshaped[progr_ind_out + s] * RAND[s]
            uncrossed = (FiltPath[i] < delta_c[i])
            progr_ind_out += i + 1
            i += 1
            #uncrossed &= (i < N_Rfilt)
        if not uncrossed:
            j=i
            NumCrossing[i-1] += 1
            while (j < N_Rfilt):
                RAND[j] = np.random.normal()
                FiltPath[j] = 0.
                for s in range(0,j+1):
                    FiltPath[j] += F_ij_reshaped[progr_ind_out + s] * RAND[s]
                progr_ind_out += j + 1
                j += 1
            FiltPath_mean[:,i-1] += FiltPath
            histo_profile(hist_dict[i-1],FiltPath,N_Rfilt,binsize,offset,nbins)
            
    return NumCrossing, FiltPath_mean



hist_dict_type = types.DictType(types.int64, int_array2D)
@jit(nopython=True, parallel=True)
def first_crossing_profile_array_barrier_single(
    F_ij_reshaped, N_paths, delta_c,delta_min,delta_max,nbins,nCPU):
    offset = delta_min
    binsize = (delta_max - delta_min) / nbins
    histo_bins = np.linspace(delta_min,delta_max,nbins+1)
    
    #nCPU = get_num_threads()
    N_Rfilt = int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2))

    NumCrossing_perCore = Dict.empty(types.int64, int_array)
    FiltPath_mean_perCore = Dict.empty(types.int64, float_array2D)
    hist_dict_perCore = Dict.empty(types.int64, hist_dict_type)
    for nn in range(0, nCPU):
        NumCrossing_perCore[nn] = np.zeros(N_Rfilt, dtype=np.int_)
        FiltPath_mean_perCore[nn] = np.zeros((N_Rfilt,N_Rfilt), dtype=np.float_)
        hist_dict_perCore[nn] = Dict.empty(types.int64, int_array2D)
        for i in range(N_Rfilt):
            hist_dict_perCore[nn][i] = np.zeros((nbins,N_Rfilt), dtype=np.int_)
    for nn in prange(0, nCPU):
        N_paths_core = int(N_paths / nCPU)
        N_paths_core += nn < (N_paths % nCPU)
        NumCrossing_perCore[nn][:], FiltPath_mean_perCore[nn][:,:] = \
            first_crossing_profile_perCore_array_barrier_single(
                F_ij_reshaped, N_paths_core, N_Rfilt, delta_c,
                hist_dict_perCore[nn],binsize,offset,nbins)
        
    NumCrossing = np.zeros(N_Rfilt, dtype=np.int_)
    FiltPath_mean = np.zeros((N_Rfilt,N_Rfilt), dtype=np.float_)
    hist_dict = Dict.empty(types.int64, int_array2D)
    for i in range(N_Rfilt):
        hist_dict[i] = np.zeros((nbins,N_Rfilt), dtype=np.int_)
    for nn in range(0, nCPU):
        NumCrossing += NumCrossing_perCore[nn]
        FiltPath_mean += FiltPath_mean_perCore[nn]
        for i in range(N_Rfilt):
            hist_dict[i] += hist_dict_perCore[nn][i]
    for i in range(N_Rfilt):
        if NumCrossing[i] > 0:
            FiltPath_mean[:,i] /= NumCrossing[i]
    return NumCrossing, FiltPath_mean, histo_bins, hist_dict



def first_crossing_profile_single_barrier(F_ij_reshaped, N_paths, delta_c,delta_min,delta_max,nbins,nCPU=-1):
    if (nCPU < 0) | (nCPU > get_num_threads()):
        nCPU = get_num_threads()
    if np.isscalar(delta_c):
        return first_crossing_scalar_barrier_single_numba(
            F_ij_reshaped, N_paths, np.fill(int(round((np.sqrt(8 * len(F_ij_reshaped) + 1) - 1) / 2)),delta_c),delta_min,delta_max,nbins,nCPU)
    return first_crossing_profile_array_barrier_single(F_ij_reshaped, N_paths, delta_c,delta_min,delta_max,nbins,nCPU)