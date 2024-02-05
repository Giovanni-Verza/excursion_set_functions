import numpy as np
from numba import jit, prange
from numba.core import types
from numba.typed import Dict

int_array = types.int64[::1]
int_rang2array = types.int64[:,::1]
int_rang3array = types.int64[:,:,::1]
float_arrayDim1 = types.float64[::1]
float_rang2array = types.float64[:,::1]
float_rang2array_A = types.float64[:,:]

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
        L_ij = Dict.empty(types.int64, float_arrayDim1)
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


def first_crossing_from_multiplicity_func(NumCross):
    FirstCrossing = np.copy(NumCross)
    for i in range(1,len(NumCross)):
        FirstCrossing[i] += FirstCrossing[i-1]
    return FirstCrossing