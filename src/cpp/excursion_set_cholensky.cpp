#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <thread> 
#include <map>

#if defined(_OPENMP)
    #include <omp.h>
#endif

#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;

namespace py = pybind11;




vector<int> first_crossing_perCore_single_barrier(
    vector<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
    //py::array_t<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
 
    vector<int> NumCrossing(N_Rfilt, 0);
    vector<double> RAND(N_Rfilt);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    int progr_ind_out;
    double FiltPath;
    int i;
    bool uncrossed;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        for (int i = 0; i < N_Rfilt; i++) {
            RAND[i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            if (FiltPath >= delta_c) {
                NumCrossing[i] += 1;
                break;
            }
            progr_ind_out += i + 1; //progr_ind_in;
        }
        /*
        uncrossed = true;
        i=0;
        while ((i < N_Rfilt) && (uncrossed)) {
            RAND[i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            uncrossed = FiltPath < delta_c;
            //if(FiltPath >= delta_c) {
            //    NumCrossing[i] += 1;
            //    break;
            //}
            //NumCrossing[i] += !uncrossed;
            progr_ind_out += i + 1; //progr_ind_in;
            i += 1;
        }
        NumCrossing[i-1] += !uncrossed;
        */
    }
    return NumCrossing;
}



py::array_t<int> first_crossing_single_barrier(
    vector<double> F_ij_reshaped, int N_paths, double delta_c, int nCPU) {

    #if defined(_OPENMP)
        if ((nCPU < 1) || (nCPU > omp_get_max_threads())) {
            nCPU = omp_get_max_threads();
        }
    #else
        nCPU = 1;
    #endif

    int N_Rfilt = round((sqrt(8 * F_ij_reshaped.size() + 1) - 1) / 2);

    map<int, int* > ptr_Cross_perCore;
    vector< vector<int> > Cross_perCore(nCPU, vector<int>(N_Rfilt));
    
    #pragma omp parallel for num_threads( nCPU )
    for (int nn = 0; nn < nCPU; nn++) {
        int N_paths_core = N_paths / nCPU;
        N_paths_core += nn < (N_paths % nCPU);
        Cross_perCore[nn] = first_crossing_perCore_single_barrier(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c);
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
    
    return NumCrossing;
}



vector<int> first_crossing_perCore_double_barrier(
    vector<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_v, double delta_c) {
    //py::array_t<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
 
    vector<int> NumCrossing(N_Rfilt, 0);
    vector<double> RAND(N_Rfilt);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    int progr_ind_out;
    double FiltPath;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        for (int i = 0; i < N_Rfilt; i++) {
            RAND[i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            if((FiltPath >= delta_c) || (FiltPath <= delta_v)) {
                NumCrossing[i] += FiltPath <= delta_v;
                break;
            }
            progr_ind_out += i + 1; //progr_ind_in;
        }
    }
    return NumCrossing;
}



py::array_t<int> first_crossing_double_barrier(
    vector<double> F_ij_reshaped, int N_paths, double delta_v, double delta_c, int nCPU) {

    #if defined(_OPENMP)
        if ((nCPU < 1) || (nCPU > omp_get_max_threads())) {
            nCPU = omp_get_max_threads();
        }
    #else
        nCPU = 1;
    #endif

    int N_Rfilt = round((sqrt(8 * F_ij_reshaped.size() + 1) - 1) / 2);

    map<int, int* > ptr_Cross_perCore;
    vector< vector<int> > Cross_perCore(nCPU, vector<int>(N_Rfilt));
    
    //int N_check = 0;
    #pragma omp parallel for num_threads( nCPU )
    for (int nn = 0; nn < nCPU; nn++) {
        // #pragma atomic write
        int N_paths_core = N_paths / nCPU;
        N_paths_core += nn < (N_paths % nCPU);
        Cross_perCore[nn] = first_crossing_perCore_double_barrier(F_ij_reshaped, N_paths_core, N_Rfilt, delta_v, delta_c);
        //#pragma atomic write
        //N_check += N_paths_core;
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
    
    return NumCrossing;
}






vector<int> first_crossing_perCore_array_barrier_single_while_loop(
    vector<double> F_ij_reshaped, int N_paths, int N_Rfilt, vector<double> delta_c) {
    //py::array_t<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
 
    vector<int> NumCrossing(N_Rfilt, 0);
    vector<double> RAND(N_Rfilt);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    int progr_ind_out;
    double FiltPath;
    int i;
    bool uncrossed;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        uncrossed = true;
        i=0;
        while ((i < N_Rfilt) && (uncrossed)) {
            RAND[i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            uncrossed = FiltPath < delta_c[i];
            //if(FiltPath >= delta_c) {
            //    NumCrossing[i] += 1;
            //    break;
            //}
            //NumCrossing[i] += !uncrossed;
            progr_ind_out += i + 1; //progr_ind_in;
            i += 1;
        }
        NumCrossing[i-1] += !uncrossed;
    }
    return NumCrossing;
}


vector<long long int> first_crossing_perCore_array_barrier_single(
    vector<double> F_ij_reshaped, long long int N_paths, int N_Rfilt, vector<double> delta_c) {
 
    vector<long long int> NumCrossing(N_Rfilt, 0);
    vector<double> RAND(N_Rfilt);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    int progr_ind_out;
    double FiltPath;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        for (int i = 0; i < N_Rfilt; i++) {
            RAND[i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            if(FiltPath >= delta_c[i]) {
                NumCrossing[i] += 1;
                break;
            }
            progr_ind_out += i + 1; //progr_ind_in;
        }
    }

    /*
    ///////////////////////////////////////////////
    //////// inner while loop ~10% slower /////////
    ///////////////////////////////////////////////

    int progr_ind_out;
    double FiltPath;
    int i;
    bool uncrossed;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        uncrossed = true;
        i=0;
        while ((i < N_Rfilt) && (uncrossed)) {
            RAND[i] = dist(e2);
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            uncrossed = FiltPath < delta_c[i];
            progr_ind_out += i + 1; 
            i += 1;
        }
        NumCrossing[i-1] += !uncrossed;
    }
    */
    return NumCrossing;
}


py::array_t<long long int> first_crossing_array_barrier_single(
    vector<double> F_ij_reshaped, long long int N_paths, vector<double> delta_c, int nCPU) {

    #if defined(_OPENMP)
        if ((nCPU < 1) || (nCPU > omp_get_max_threads())) {
            nCPU = omp_get_max_threads();
        }
    #else
        nCPU = 1;
    #endif
    int N_Rfilt = round((sqrt(8 * F_ij_reshaped.size() + 1) - 1) / 2);

    map<int, long long int* > ptr_Cross_perCore;
    vector< vector<long long int> > Cross_perCore(nCPU, vector<long long int>(N_Rfilt));
    
    //int N_check = 0;
    #pragma omp parallel for num_threads( nCPU )
    for (int nn = 0; nn < nCPU; nn++) {
        long long int N_paths_core = N_paths / nCPU;
        N_paths_core += nn < (N_paths % nCPU);
        Cross_perCore[nn] = first_crossing_perCore_array_barrier_single(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c);
    }

    
    //vector<int> NumCrossing = vector<int>(N_Rfilt);
    py::array_t<long long int> NumCrossing = py::array_t<long long int>(N_Rfilt);
    py::buffer_info buf_NumCrossing = NumCrossing.request();
    long long int *ptr_NumCrossing = (long long int *) buf_NumCrossing.ptr;

    for (int i = 0; i < N_Rfilt; i++) {
        ptr_NumCrossing[i] = 0;
        for (int nn = 0; nn < nCPU; nn++) {
            ptr_NumCrossing[i] += Cross_perCore[nn][i];
        }
    }
    
    return NumCrossing;
}





void init_ex_set_cholenski(py::module_ &m) {
    m.def("first_crossing_single_barrier", &first_crossing_single_barrier,
          py::arg("F_ij_reshaped"), py::arg("N_paths"), py::arg("delta_c"), py::arg("nCPU")=-1, 
          R"pbdoc(
            Fisrt crossing of N_paths realizations of random walks with constant threshold
        )pbdoc");
    m.def("first_crossing_single_barrier", &first_crossing_array_barrier_single,
          py::arg("F_ij_reshaped"), py::arg("N_paths"), py::arg("delta_c"), py::arg("nCPU")=-1, 
          R"pbdoc(
            Fisrt crossing of N_paths realizations of random walks with a scale dependent threshold
        )pbdoc");
    m.def("first_crossing_double_barrier", &first_crossing_double_barrier,
          py::arg("F_ij_reshaped"), py::arg("N_paths"), py::arg("delta_c"), py::arg("delta_v"), py::arg("nCPU")=-1, R"pbdoc(
            Fisrt crossing of N_paths realizations of random walks with scale dependent double barrier
        )pbdoc");
}