#define _USE_MATH_DEFINES
#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <thread> 
#include <map>
//#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;




py::array_t<double> Lagrangian_tophat_mass_inR(vector<double> &R,double Omega_M) {


    int len_R = R.size();
    py::array_t<double> OUT = py::array_t<double>(len_R);
    py::buffer_info buf_OUT = OUT.request();
    double *ptr_OUT = (double *) buf_OUT.ptr;

    for (int i=0;i<len_R;i++) {
        ptr_OUT[i] = 3.086e13/(2.*6.674*1.989) * Omega_M * R[i] * R[i] * R[i];
    }

    return OUT;
}





py::array_t<double> f_double_barrier_lnsigma(vector<double> sigma, double delta_v, double delta_c) {
    int len_arr = sigma.size();
    py::array_t<double> np_arr = py::array_t<double>(len_arr);
    py::buffer_info buf_np_arr = np_arr.request();
    double *ptr_np_arr = (double *) buf_np_arr.ptr;
    int *mask1 = new int[len_arr];
    int *mask2 = new int[len_arr];

    vector<double> x = sigma;
    double D = abs(delta_v) / (delta_c - delta_v);
    //int index_switch = 0;
    int progr1 = 0;
    int progr2 = 0;
    bool condition;
    for (int i=0; i<len_arr; i++) {
        x[i] *= D/abs(delta_v);
        //index_switch += x[i] > 0.276;
        condition = x[i] > 0.276;
        mask1[progr1] = i;
        mask2[progr2] = i;
        progr1 += (!condition);
        progr2 += condition;
    }
    //cout << index_switch << endl;
    

    //for (int i=0; i<index_switch; i++) {
    //    ptr_np_arr[i]=0;
    //}
    //for (int i=index_switch; i<len_arr; i++) {
    for (int i=0; i<progr1; i++) {
        ptr_np_arr[mask1[i]] = sqrt(2./M_PI)/sigma[mask1[i]] * abs(delta_v) * 
                               exp(-(delta_v*delta_v/(2.*sigma[mask1[i]]*sigma[mask1[i]])));
    }


    double x2;
    double pi2 = M_PI * M_PI;
    for (int i=0; i<progr2; i++) {
        ptr_np_arr[mask2[i]]=0;
    }
    for (int j=1; j<5; j++) {
        for (int i=0; i<progr2; i++) {
            x2 = x[mask2[i]] * x[mask2[i]];
            ptr_np_arr[mask2[i]] += 2 * j * M_PI * x2 * sin(j * M_PI * D) * exp(-j * j * pi2 * x2 / 2);
        }
    }

    delete[] mask2;
    delete[] mask1;

    //for (int i=0; i < len_out; i++) {
    //    ptr_np_arr[i*3] = ln_a;
    //    ptr_np_arr[i*3+1] = fout[id_out];
    //    ptr_np_arr[i*3+2] = exp(ln_Dz[id_out]);
    //    id_out += (stride_sample + (i < stride_res));
    //    ln_a += (stride_sample + (i < stride_res)) * dlna;
    //}
    //ptr_np_arr[(len_out-1)*3] = log_a_f;
    //np_arr.resize({len_out,3});

    return np_arr;
}


py::array_t<double> f_ST_nu(vector<double> nu, double p, double q) {
    int len_arr = nu.size();
    py::array_t<double> np_arr = py::array_t<double>(len_arr);
    py::buffer_info buf_np_arr = np_arr.request();
    double *ptr_np_arr = (double *) buf_np_arr.ptr;
    double norm = sqrt(2./M_PI) / (1. + tgamma (0.5 - p) / (pow(2.,p) * sqrt(M_PI)));
    double sqrt_q = sqrt(q);
    for (int i=0; i<len_arr; i++) {
        ptr_np_arr[i] = norm * (1. + pow(q * nu[i] * nu[i],-p)) * sqrt_q * exp(-q * nu[i] * nu[i] / 2.);
    }
    return np_arr;
}


py::array_t<double> f_ST_nu_unnorm(vector<double> nu, double norm, double p, double q) {
    int len_arr = nu.size();
    py::array_t<double> np_arr = py::array_t<double>(len_arr);
    py::buffer_info buf_np_arr = np_arr.request();
    double *ptr_np_arr = (double *) buf_np_arr.ptr;
    //double norm = sqrt(2./M_PI) / (1. + tgamma (0.5 - p) / (pow(2.,p) * sqrt(M_PI)));
    double sqrt_q = sqrt(q);
    for (int i=0; i<len_arr; i++) {
        ptr_np_arr[i] = norm * (1. + pow(q * nu[i] * nu[i],-p)) * sqrt_q * exp(-q * nu[i] * nu[i] / 2.);
    }
    return np_arr;
}



py::array_t<double> f_Tinker(vector<double> sigma, double norm, double a, double b, double c) {
    int len_arr = sigma.size();
    py::array_t<double> np_arr = py::array_t<double>(len_arr);
    py::buffer_info buf_np_arr = np_arr.request();
    double *ptr_np_arr = (double *) buf_np_arr.ptr;
    for (int i=0; i<len_arr; i++) {
        ptr_np_arr[i] = norm * (pow(b / sigma[i], a) + 1) *  exp(-c / (sigma[i] * sigma[i]));
    }
    return np_arr;
}


py::array_t<double> f_Tinker_normalized(vector<double> sigma, double p0, double p1, double p2, double p3) {
    double norm = 2. / (pow(p1,p0) * pow(p3,-0.5 * p0) * tgamma (0.5 * p0) + pow(p3,-0.5 * p2) * tgamma (0.5 * p2));
    int len_arr = sigma.size();
    py::array_t<double> np_arr = py::array_t<double>(len_arr);
    py::buffer_info buf_np_arr = np_arr.request();
    double *ptr_np_arr = (double *) buf_np_arr.ptr;
    for (int i=0; i<len_arr; i++) {
        ptr_np_arr[i] = norm * (pow(p1 / sigma[i], p0) + pow(sigma[i], -p2)) *  exp(-p3 / (sigma[i] * sigma[i]));
    }
    return np_arr;
}


py::array_t<double> f_Tinker_5params(vector<double> sigma, double norm, double p0, double p1, double p2, double p3) {
    //double norm = 2. / (pow(p1,p0) * pow(p3,-0.5 * p0) * tgamma (0.5 * p0) + pow(p3,-0.5 * p2) * tgamma (0.5 * p2));
    int len_arr = sigma.size();
    py::array_t<double> np_arr = py::array_t<double>(len_arr);
    py::buffer_info buf_np_arr = np_arr.request();
    double *ptr_np_arr = (double *) buf_np_arr.ptr;
    for (int i=0; i<len_arr; i++) {
        ptr_np_arr[i] = norm * (pow(p1 / sigma[i], p0) + pow(sigma[i], -p2)) *  exp(-p3 / (sigma[i] * sigma[i]));
    }
    return np_arr;
}




py::array_t<double> f_S_MB_approx(vector<double> &s,vector<double> &W,vector<double> &B,vector<double> &dB_ds) {

    int len_s = s.size();
    py::array_t<double> OUT = py::array_t<double>(len_s);
    py::buffer_info buf_OUT = OUT.request();
    double *ptr_OUT = (double *) buf_OUT.ptr;

    double LDD, D_BP, AB2S, I2a_S, I2b_2_S_noExp;
    
    for (int i=0; i<len_s; i++) {
        LDD = s[i] * W[i] - 0.25;
        D_BP = 0.5 * B[i] / s[i] - dB_ds[i];
        AB2S = 0.5 * s[i] / LDD * D_BP * D_BP;
        
        I2a_S = sqrt(LDD) / (4*M_PI*s[i]) * exp(-0.5 / LDD * (s[i] * dB_ds[i]*dB_ds[i] + W[i]*B[i]*B[i] - B[i]*dB_ds[i]));
        I2b_2_S_noExp = 0.5 * D_BP * (erf(sqrt(AB2S))+1) / sqrt(2.*M_PI*s[i]);
        ptr_OUT[i] = I2a_S + (I2b_2_S_noExp + sqrt(LDD)/(4*M_PI*s[i]) * exp(-AB2S)) * exp(-0.5*B[i]*B[i] / s[i]);
    }
        
    return OUT;
}


void init_ex_set_analytical(py::module_ &m) {
    m.def("Lagrangian_tophat_mass_inR", &Lagrangian_tophat_mass_inR);
    m.def("f_double_barrier_lnsigma", &f_double_barrier_lnsigma);
    m.def("f_ST_nu", &f_ST_nu);
    m.def("f_ST_nu_unnorm", &f_ST_nu_unnorm);
    m.def("f_Tinker", &f_Tinker);
    m.def("f_Tinker_normalized", &f_Tinker_normalized);
    m.def("f_Tinker_5params", &f_Tinker_5params);
    m.def("f_S_MB_approx", &f_S_MB_approx);
}


