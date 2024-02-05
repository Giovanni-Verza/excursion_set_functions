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

vector<int> first_crossing_perCore_single_barrier_old(
    vector<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
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
            if(FiltPath >= delta_c) {
                NumCrossing[i] += 1;
                break;
            }
            progr_ind_out += i + 1; //progr_ind_in;
        }
    }
    return NumCrossing;
}


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
    }
    return NumCrossing;
}


vector<int> first_crossing_perCore_single_barrier_2(
    vector<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
    //py::array_t<double> F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c) {
 
    vector<int> NumCrossing(N_Rfilt, 0);
    vector<double> RAND(N_Rfilt);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    int progr_ind_out;
    double FiltPath;
    bool crossed;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        crossed = false;
        for (int i=0; i< N_Rfilt;i++) {
            RAND[i] = dist(e2);
            //progr_ind_in = 0;
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_reshaped[progr_ind_out + s] * RAND[s];
                //progr_ind_in += 1;
            }
            crossed = FiltPath >= delta_c;
            //if(FiltPath >= delta_c) {
            //    NumCrossing[i] += 1;
            //    break;
            //}
            NumCrossing[i] += crossed;
            progr_ind_out += i + 1; //progr_ind_in;
            i += N_Rfilt*crossed;
        }
        //NumCrossing[i-1] += !uncrossed;
    }
    return NumCrossing;
}


py::array_t<int> first_crossing_single_barrier(
    vector<double> F_ij_reshaped, int N_paths, double delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
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
        Cross_perCore[nn] = first_crossing_perCore_single_barrier(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c);
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

py::array_t<int> first_crossing_single_barrier_old(
    vector<double> F_ij_reshaped, int N_paths, double delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
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
        Cross_perCore[nn] = first_crossing_perCore_single_barrier_old(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c);
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


py::array_t<int> first_crossing_single_barrier_2(
    vector<double> F_ij_reshaped, int N_paths, double delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
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
        Cross_perCore[nn] = first_crossing_perCore_single_barrier_2(F_ij_reshaped, N_paths_core, N_Rfilt, delta_c);
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
    vector<double> F_ij_reshaped, int N_paths, double delta_v, double delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
    #else
        nCPU = 1;
    #endif

    //cout << omp_get_max_threads() << endl;

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
    return NumCrossing;
}


py::array_t<long long int> first_crossing_array_barrier_single(
    vector<double> F_ij_reshaped, long long int N_paths, vector<double> delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
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
    m.def("first_crossing_single_barrier", &first_crossing_single_barrier);
    m.def("first_crossing_single_barrier_old", &first_crossing_single_barrier_old);
    m.def("first_crossing_single_barrier_2", &first_crossing_single_barrier_2);
    m.def("first_crossing_double_barrier", &first_crossing_double_barrier);
    m.def("first_crossing_array_barrier_single", &first_crossing_array_barrier_single);
}















/*

vector<int> first_crossing_from_multiplicity_func(vector<int> &NumCrossing) {
    vector<int> FirstCrossing(NumCrossing);

    for (int i = NumCrossing.size() - 1; i >=0; i--) {
        for (int j = 0; j < i; ++j) {
            FirstCrossing[i] += FirstCrossing[j];
        }
    }
    return FirstCrossing;
}

vector<double> first_crossing_from_multiplicity_func(vector<double> &NumCrossing) {
    vector<double> FirstCrossing(NumCrossing);
    
    for (int i = NumCrossing.size() - 1; i >=0; i--) {
        for (int j = 0; j < i; ++j) {
            FirstCrossing[i] += FirstCrossing[j];
        }
    }
    return FirstCrossing;
}

vector<int> first_crossing_perCore_single_barrier_noiseDirect(
    vector<double> F_ij_cosmo, vector<double> F_ij_noise, int N_paths, int N_Rfilt, double delta_c) {
 
    vector<int> NumCrossing(N_Rfilt, 0);
    vector<double> RAND_C(N_Rfilt);
    vector<double> RAND_N(N_Rfilt);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> dist(0., 1.);
   
    int progr_ind_out;
    double FiltPath;
    for (int nn = 0; nn < N_paths; nn++) {
        progr_ind_out = 0;
        for (int i = 0; i < N_Rfilt; i++) {
            RAND_C[i] = dist(e2);
            RAND_N[i] = dist(e2);
            FiltPath = 0.;
            for (int s = 0; s < i+1; s++) {
                FiltPath += F_ij_cosmo[progr_ind_out + s] * RAND_C[s] + F_ij_noise[progr_ind_out + s] * RAND_N[s];
            }
            if(FiltPath >= delta_c) {
                NumCrossing[i] += 1;
                break;
            }
            progr_ind_out += i + 1; 
        }
    }

    return NumCrossing;
}


vector<int> multiplicity_func_Cholensky_perCore_doubleBarr(
    vector<double> &F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c, double delta_v) {

    vector<int> NumCrossing;
    NumCrossing.assign(N_Rfilt, 0);
    vector<double> RAND;
    RAND.assign(N_Rfilt, 0);
   
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
            if(FiltPath >= delta_c) {
                break;
            }
            if(FiltPath <= delta_v) {
                NumCrossing[i] += 1;
                break;
            }
            progr_ind_out += i + 1; //progr_ind_in;
        }
    }

   return NumCrossing;
}








py::array_t<int> first_crossing_single_barrier_noiseDirect(
    vector<double> F_ij_cosmo, vector<double> F_ij_noise, int N_paths, double delta_c) {
    //py::buffer_info buf_Fij = F_ij_reshaped.request();

    int nCPU;
    #if defined(_OPENMP)
        nCPU = omp_get_max_threads();
    #else
        nCPU = 1;
    #endif

    int N_Rfilt = round((sqrt(8 * F_ij_cosmo.size() + 1) - 1) / 2);

    map<int, int* > ptr_Cross_perCore;
    vector< vector<int> > Cross_perCore(nCPU, vector<int>(N_Rfilt));
    
    #pragma omp parallel for num_threads( nCPU )
    for (int nn = 0; nn < nCPU; nn++) {
        int N_paths_core = N_paths / nCPU;
        N_paths_core += nn < (N_paths % nCPU);
        Cross_perCore[nn] = first_crossing_perCore_single_barrier_noiseDirect(
            F_ij_cosmo, F_ij_noise, N_paths_core, N_Rfilt, delta_c);
    }


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


vector<int> multiplicity_func_Cholensky_doubleBarr(
    vector<double> &F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c, double delta_v, int nCPU) {

    map<int, int* > ptr_Cross_perCore;
    vector< vector<int> > Cross_perCore(nCPU, vector<int>(N_Rfilt));

    #pragma omp parallel num_threads( nCPU )
    {

        #pragma omp for
        for (int nn = 0; nn < nCPU; nn++) {
            #pragma atomic write
            Cross_perCore[nn] = multiplicity_func_Cholensky_perCore_doubleBarr(
                F_ij_reshaped, N_paths, N_Rfilt, delta_c, delta_v);
        }
    }


    vector<int> NumCrossing = vector<int>(N_Rfilt);

    for (int i = 0; i < N_Rfilt; i++) {
        NumCrossing[i] = 0;
        for (int nn = 0; nn < nCPU; nn++) {
            NumCrossing[i] += Cross_perCore[nn][i];
        }
    }
    
    return NumCrossing;
}

vector<int> multiplicity_func_sharpk_perCore_singleBarr(
    vector<double> &sqrtD_S, int N_paths, double delta_c) {
 
    int Nsteps = sqrtD_S.size();
    vector<int> NumCrossing;
    NumCrossing.assign(Nsteps+1, 0);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> gauss_dist(0., 1.);
    uniform_real_distribution<double> unif_dist(0.0,1.0);
   
    double FiltPath, FiltPath_prev;
    for (int nn = 0; nn < N_paths; nn++) {
        FiltPath = 0.;
        FiltPath_prev = 0.;
        for (int i = 0; i < Nsteps; i++) {
            FiltPath += sqrtD_S[i] * gauss_dist(e2);
            //cout << "[" << FiltPath_prev << "," << FiltPath << "]," << endl;
            if (exp(-2. * (delta_c - FiltPath) * (delta_c - FiltPath_prev) / (sqrtD_S[i] * sqrtD_S[i])) >= unif_dist(e2)) {
                NumCrossing[i+1] += 1;
                break;
            }            
            FiltPath_prev = FiltPath;
        }
    }
    return NumCrossing;
}

vector<int> multiplicity_func_sharpk_perCore_doubleBarr(
    vector<double> &sqrtD_S, int N_paths, double delta_c, double delta_v) {
 
    int Nsteps = sqrtD_S.size();
    vector<int> NumCrossing;
    NumCrossing.assign(Nsteps+1, 0);
   
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<double> gauss_dist(0., 1.);
    uniform_real_distribution<double> unif_dist(0.0,1.0);
   
    double FiltPath, FiltPath_prev, rand;
    for (int nn = 0; nn < N_paths; nn++) {
        FiltPath = 0.;
        FiltPath_prev = 0.;
        for (int i = 0; i < Nsteps; i++) {
            FiltPath += sqrtD_S[i] * gauss_dist(e2);
            //cout << "[" << FiltPath_prev << "," << FiltPath << "]," << endl;
            rand = unif_dist(e2);
            if (exp(-2. * (delta_c - FiltPath) * (delta_c - FiltPath_prev) / (sqrtD_S[i] * sqrtD_S[i])) >= rand) {
                break;
            }   
            if (exp(-2. * (delta_v - FiltPath) * (delta_v - FiltPath_prev) / (sqrtD_S[i] * sqrtD_S[i])) >= rand) {
                NumCrossing[i+1] += 1;
                break;
            }           
            FiltPath_prev = FiltPath;
        }
    }
    return NumCrossing;
}

vector<int> multiplicity_func_sharpk_singleBarr(
    vector<double> &Sig2, int N_paths, double delta_c, int nCPU) {

    int Nsteps = Sig2.size() - 1;
    vector<double> sqrtD_S;
    sqrtD_S.assign(Nsteps, 0);
    for (int i = 0; i < Nsteps; i++) {
        sqrtD_S[i] = sqrt(Sig2[i+1] - Sig2[i]);
    }
      
    map<int, int* > ptr_Cross_perCore;
    vector< vector<int> > Cross_perCore(nCPU, vector<int>(Nsteps +1 ));

    #pragma omp parallel num_threads( nCPU )
    {
        #pragma omp for
        for (int nn = 0; nn < nCPU; nn++) {
            #pragma atomic write
            Cross_perCore[nn] = multiplicity_func_sharpk_perCore_singleBarr(sqrtD_S, N_paths, delta_c);
        }
    }

    vector<int> NumCrossing = vector<int>(Nsteps+1);

    for (int i = 0; i < Nsteps+1; i++) {
        NumCrossing[i] = 0;
        for (int nn = 0; nn < nCPU; nn++) {
            NumCrossing[i] += Cross_perCore[nn][i];
        }
    }
    
    return NumCrossing;
}


vector<int> multiplicity_func_sharpk_doubleBarr(
    vector<double> &Sig2, int N_paths, double delta_c, double delta_v, int nCPU) {

    int Nsteps = Sig2.size() - 1;
    vector<double> sqrtD_S;
    sqrtD_S.assign(Nsteps, 0);
    for (int i = 0; i < Nsteps; i++) {
        sqrtD_S[i] = sqrt(Sig2[i+1] - Sig2[i]);
    }
      
    map<int, int* > ptr_Cross_perCore;
    vector< vector<int> > Cross_perCore(nCPU, vector<int>(Nsteps +1 ));

    #pragma omp parallel num_threads( nCPU )
    {
        #pragma omp for
        for (int nn = 0; nn < nCPU; nn++) {
            #pragma atomic write
            Cross_perCore[nn] = multiplicity_func_sharpk_perCore_doubleBarr(sqrtD_S, N_paths, delta_c, delta_v);
        }
    }

    vector<int> NumCrossing = vector<int>(Nsteps+1);

    for (int i = 0; i < Nsteps+1; i++) {
        NumCrossing[i] = 0;
        for (int nn = 0; nn < nCPU; nn++) {
            NumCrossing[i] += Cross_perCore[nn][i];
        }
    }
    
    return NumCrossing;
}





py::array_t<int> multiplicity_func_Cholensky_doubleBarr_pycast(
    vector<double> &F_ij_reshaped, int N_paths, int N_Rfilt, double delta_c, double delta_v, int nCPU) {

    vector<int> std_vec = multiplicity_func_Cholensky_doubleBarr(
        F_ij_reshaped, N_paths, N_Rfilt, delta_c, delta_v, nCPU);

    py::array_t<int> np_arr = py::array_t<int>(N_Rfilt);
    py::buffer_info buf_np_arr = np_arr.request();
    int *ptr_np_arr = (int *) buf_np_arr.ptr;

    for (int i = 0; i < N_Rfilt; i++) {
        ptr_np_arr[i] = std_vec[i];
    }
    return np_arr;
}

py::array_t<int> multiplicity_func_sharpk_singleBarr_pycast(
     vector<double> &Sig2, int N_paths, double delta_c, int nCPU) {

    vector<int> std_vec = multiplicity_func_sharpk_singleBarr(
        Sig2, N_paths, delta_c, nCPU);
    
    int Nstep = Sig2.size();

    py::array_t<int> np_arr = py::array_t<int>(Nstep);
    py::buffer_info buf_np_arr = np_arr.request();
    int *ptr_np_arr = (int *) buf_np_arr.ptr;

    for (int i = 0; i < Nstep; i++) {
        ptr_np_arr[i] = std_vec[i];
    }
    return np_arr;
}

py::array_t<int> multiplicity_func_sharpk_doubleBarr_pycast(
     vector<double> &Sig2, int N_paths, double delta_c, double delta_v, int nCPU) {

    vector<int> std_vec = multiplicity_func_sharpk_doubleBarr(
        Sig2, N_paths, delta_c, delta_v, nCPU);
    
    int Nstep = Sig2.size();

    py::array_t<int> np_arr = py::array_t<int>(Nstep);
    py::buffer_info buf_np_arr = np_arr.request();
    int *ptr_np_arr = (int *) buf_np_arr.ptr;

    for (int i = 0; i < Nstep; i++) {
        ptr_np_arr[i] = std_vec[i];
    }
    return np_arr;
}


//PYBIND11_MODULE(excursion_set_functions, m) {
//    m.doc() = "pybind11 example plugin"; // optional module docstring

    //m.def("first_crossing_from_multiplicity_func", py::overload_cast<py::array_t<int> > (&first_crossing_from_multiplicity_func));

    //m.def("first_crossing_from_multiplicity_func", &first_crossing_from_multiplicity_func_pycast);

    //m.def("first_crossing_single_barrier", &first_crossing_single_barrier);

//    m.def("first_crossing_single_barrier_noiseDirect", &first_crossing_single_barrier_noiseDirect);

    //m.def("multiplicity_func_Cholensky_doubleBarr", &multiplicity_func_Cholensky_doubleBarr_pycast);

    //m.def("multiplicity_func_sharpk_singleBarr", &multiplicity_func_sharpk_singleBarr_pycast);

    //m.def("multiplicity_func_sharpk_doubleBarr", &multiplicity_func_sharpk_doubleBarr_pycast);

//}
*/