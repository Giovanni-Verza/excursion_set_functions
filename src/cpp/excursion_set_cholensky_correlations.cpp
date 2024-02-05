#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <thread> 
#include <map>
#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;

namespace py = pybind11;




int first_crossing_perCore_single_barrier_for_correlations(
    vector<int> NumCrossing, vector<int> IDcross, vector<vector<double>> RAND,
    vector<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
    //py::array_t<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    double FiltPath;
    int i, Ncross, progr_ind_out;
    bool uncrossed;
    Ncross = 0;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        uncrossed = true;
        i=0;
        while ((i < N_Rfilt) && (uncrossed)) {
            RAND[Ncross][i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[Ncross][s];
            }
            uncrossed = FiltPath < delta_c;
            progr_ind_out += i + 1;
            i += 1;
        }
        IDcross[Ncross] = i-1;
        NumCrossing[i-1] += !uncrossed;
        while (i < N_Rfilt) {
            RAND[Ncross][i] = dist(e2);
            i += 1;
        }
        Ncross += !uncrossed;
    }
    return Ncross;
}


py::tuple first_crossing_single_barrier_for_correlations(
    vector<double> F_ij_reshaped, int N_paths, double delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
    #else
        nCPU = 1;
    #endif

    int N_Rfilt = round((sqrt(8 * F_ij_reshaped.size() + 1) - 1) / 2);

    vector<int> Nx_tot(nCPU);
 
    vector<vector<int>> Cross_perCore(nCPU, vector<int>(N_Rfilt, 0));
    vector<vector<int>> IDcross(nCPU, vector<int>(N_paths));
    vector<vector<vector<double>>> RAND(nCPU, vector<vector<double>> (N_paths, vector<double> (N_Rfilt)));
    
    //int N_check = 0;
    #pragma omp parallel for num_threads( nCPU )
    for (int nn = 0; nn < nCPU; nn++) {
        // #pragma atomic write
        int N_paths_core = N_paths / nCPU;
        N_paths_core += nn < (N_paths % nCPU);
        Nx_tot[nn] = first_crossing_perCore_single_barrier_for_correlations(
            Cross_perCore[nn],IDcross[nn],RAND[nn],F_ij_reshaped, N_paths_core, N_Rfilt, delta_c);
        //#pragma atomic write
        //N_check += N_paths_core;
    }

    int Ntot_out = 0;
    for (int nn = 0; nn < nCPU; nn++) {
        Ntot_out += Nx_tot[nn];
    }
    //vector<int> NumCrossing = vector<int>(N_Rfilt);
    py::array_t<int> NumCrossing = py::array_t<int>(N_Rfilt);
    py::buffer_info buf_NumCrossing = NumCrossing.request();
    int *ptr_NumCrossing = (int *) buf_NumCrossing.ptr;

    for (int i = 0; i < N_Rfilt; i++) {
        ptr_NumCrossing[i] = 0;
        for (int nn = 0; nn < nCPU; nn++) {
            ptr_NumCrossing[i] += Cross_perCore[nn][i];
        }
    }

    py::array_t<int> IDcross_out = py::array_t<int>(Ntot_out);
    py::buffer_info buf_IDcross = IDcross_out.request();
    int *ptr_IDcross = (int *) buf_IDcross.ptr;

    int Nprogr = 0;
    for (int nn = 0; nn < nCPU; nn++) {
        for (int i = 0; i < Nx_tot[nn]; i++) {
            ptr_IDcross[Nprogr + i] += IDcross[nn][i];
        }
        Nprogr += Nx_tot[nn];
    }

    py::array_t<double> RAND_out = py::array_t<double>({Ntot_out,N_Rfilt});
    //py::array_t<py::array_t<double>> RAND_out = py::array_t<double>(Ntot_out, py::array_t<double>(N_Rfilt));
    //py::array_t<py::array_t<double>> RAND_out(Ntot_out, py::array_t<double>(N_Rfilt));
    py::buffer_info buf_RAND = RAND_out.request();
    double *ptr_RAND = (double *) buf_RAND.ptr;

    Nprogr = 0;
    for (int nn = 0; nn < nCPU; nn++) {
        for (int i = 0; i < Nx_tot[nn]; i++) {
            int idx = (Nprogr + i) * N_Rfilt;
            for (int r = 0; r < N_Rfilt; r++) {
                ptr_RAND[idx + r] += RAND[nn][i][r];
            }
        }
        Nprogr += Nx_tot[nn];
    }
 
    RAND_out.resize({Ntot_out,N_Rfilt});
    py::tuple tup = py::make_tuple(NumCrossing, IDcross_out, RAND_out);

    return tup;
}




void init_ex_set_cholenski_corr(py::module_ &m) {
    m.def("first_crossing_single_barrier_for_correlations", &first_crossing_single_barrier_for_correlations);
}