#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <map>
#include <bits/stdc++.h>
//#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
using namespace std;

namespace py = pybind11;




void tridiagonal_elements_for_k_not_a_knot(vector<double> x, vector<double> y, double abcd[][4]) {
    //a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    //n = len(x)
    //a = np.zeros(n-1)
    //b = np.zeros(n)
    //c = np.zeros(n-1)
    //d = np.zeros(n)
    int len_x = x.size();
    double f_first, f_last;

    f_first = - 1. / ((x[2] - x[1]) * (x[2] - x[1]));
    f_last = - 1. / ((x[len_x-2] - x[len_x-3]) * (x[len_x-2] - x[len_x-3]));

    //cout << "f_first=" << f_first << ", f_last=" << f_last << endl; 

    abcd[0][1] = 1. / ((x[1] - x[0]) * (x[1] - x[0]));
    abcd[0][2] = 1. / ((x[1] - x[0]) * (x[1] - x[0])) - 1. / ((x[2] - x[1]) * (x[2] - x[1]));
    abcd[0][3] = 2. * ((y[1] - y[0]) / ((x[1] - x[0]) * (x[1] - x[0]) * (x[1] - x[0]))
                -(y[2] - y[1]) / ((x[2] - x[1]) * (x[2] - x[1]) * (x[2] - x[1])));

    abcd[len_x-2][0] = 1. / ((x[len_x-1] - x[len_x-2]) * (x[len_x-1] - x[len_x-2])) - 
                       1. / ((x[len_x-2] - x[len_x-3]) * (x[len_x-2] - x[len_x-3]));
    abcd[len_x-1][1] = 1. / ((x[len_x-1] - x[len_x-2]) * (x[len_x-1] - x[len_x-2]));
    abcd[len_x-1][3] = 2. * ((y[len_x-1] - y[len_x-2]) / 
                             ((x[len_x-1] - x[len_x-2]) * (x[len_x-1] - x[len_x-2]) * (x[len_x-1] - x[len_x-2])) - 
                             (y[len_x-2] - y[len_x-3]) / 
                             ((x[len_x-2] - x[len_x-3]) * (x[len_x-2] - x[len_x-3]) * (x[len_x-2] - x[len_x-3])));

    for (int i=1; i<len_x-1; i++) {
        abcd[i-1][0] = 1 / (x[i] - x[i-1]);
        abcd[i][1] = 2 / (x[i] - x[i-1]) + 2 / (x[i+1] - x[i]);
        abcd[i][2] = 1 / (x[i+1] - x[i]);
        abcd[i][3] = 3 * ((y[i] - y[i-1]) / ((x[i] - x[i-1]) * (x[i] - x[i-1])) + 
                          (y[i+1] - y[i]) / ((x[i+1] - x[i]) * (x[i+1] - x[i])));
    }
    abcd[0][1] += -f_first * abcd[0][0] / abcd[1][2];
    abcd[0][2] += -f_first * abcd[1][1] / abcd[1][2];
    abcd[0][3] += -f_first * abcd[1][3] / abcd[1][2];

    abcd[len_x-2][0] += -f_last * abcd[len_x-2][1] / abcd[len_x-3][0];
    abcd[len_x-1][1] += -f_last * abcd[len_x-2][2] / abcd[len_x-3][0];
    abcd[len_x-1][3] += -f_last * abcd[len_x-2][3] / abcd[len_x-3][0];
}


void solve_tridiagonal_system(double abcd[][4], double k[],int len_x) {
    //a = Lower Diag, b = Main Diag, c = Upper Diag, d = solution vector
    //w= np.zeros(n-1)
    //g= np.zeros(n)
    //p = np.zeros(n)
    double *w = new double[len_x];
    double *g = new double[len_x];
    //double *k = new double[len_x];

    w[0] = abcd[0][2] / abcd[0][1];
    g[0] = abcd[0][3] / abcd[0][1];

    for (int i=1; i<len_x-1; i++) {
        w[i] = abcd[i][2] / (abcd[i][1] - abcd[i-1][0] * w[i-1]);
    }
    for (int i=1; i<len_x; i++) {
        g[i] = (abcd[i][3] - abcd[i-1][0] * g[i-1]) / (abcd[i][1] - abcd[i-1][0] * w[i-1]);
    }

    k[len_x-1] = g[len_x-1];

    //for i in range(n-1,0,-1):
    for (int i=len_x-1; i>0; i--) {
        k[i-1] = g[i-1] - w[i-1] * k[i];
    }

    delete[] w;
    delete[] g;
    //delete[] k;
}

void coeffs_from_k(vector<double> x, vector<double> y, double k[], double coeffs[][4],int len_x) {
    //double *c1 = new double[len_x-1];
    //double *c2 = new double[len_x-1];
    //for (int i=0; i<len_x-1, i++) {
    //    c1[i] = k[i] * (x[i+1] - x[i]) - (y[i+1] - y[i]);
    //    c2[i] = -k[i] * (x[i+1] - x[i]) + (y[i+1] - y[i]);
    //}
    
    double c1, c2;
    for (int i=0; i<len_x-1; i++) {
        c1 =  k[i]  *  (x[i+1] - x[i]) - (y[i+1] - y[i]);
        c2 = -k[i+1] * (x[i+1] - x[i]) + (y[i+1] - y[i]);
        coeffs[i][0] = y[i];
        coeffs[i][1] = c1 + y[i+1] - y[i];
        coeffs[i][2] = c2 - 2 * c1;
        coeffs[i][3] = c1 - c2;
    }
}

template<typename T>
vector<int> argsort(const vector<T>& v) {
    vector<int> result(v.size());
    iota(begin(result), end(result), 0);
    sort(begin(result), end(result),
            [&v](const auto & lhs, const auto & rhs)
            {
                return v[lhs] < v[rhs];
            }
    );
    return result;
}




/*void permute(T A[], size_t perm_indexes[], int n) {

    for (int i = 0; i < n; i++) {
        int next = i;
 
        while (perm_indexes[next] >= 0) {
 
            swap(A[i], A[perm_indexes[next]]);
            int temp = perm_indexes[next];
            next = temp;
        }
    }
}*/

//template<typename T>
void permute(vector<double>& A, vector<int>& perm_indexes, int n) {

    for (int i = 0; i < n; i++) {
        int next = i;
        
        while (perm_indexes[next] >= 0) {
            
            swap(A[i], A[perm_indexes[next]]);
            int temp = perm_indexes[next];
            
            perm_indexes[next] -= n;
            next = temp;
        }
    }
}



//template<typename T>
void permute(py::array_t<double>& np_arr, vector<int>& perm_indexes, int n) {
    py::buffer_info buf_np_arr = np_arr.request();
    double *A = (double *) buf_np_arr.ptr;

    for (int i = 0; i < n; i++) {
        int next = i;
        
        while (perm_indexes[next] >= 0) {
            
            swap(A[i], A[perm_indexes[next]]);
            int temp = perm_indexes[next];
            
            perm_indexes[next] -= n;
            next = temp;
        }
    }
}


class cubic_spline{
    private:
        double (*coeffs)[4];
    public:
        vector<double> x;
        int len_x;
        //double (*coeffs)[4];
        cubic_spline(vector<double> _x, vector<double> y) {     
            //int (*ijk_in_sphere)[3] = new int[max_num_vox_for_sphere][3];
            x = _x;
            len_x = x.size();
            coeffs = new double[len_x-1][4];
            double (*abcd)[4] = new double[len_x][4];
            double *k = new double[len_x];
            //cout << "abcd:"  << endl;
            tridiagonal_elements_for_k_not_a_knot(x, y, abcd);
            //for (int i=0; i<len_x; i++) {
            //    cout << abcd[i][0] << "    " << abcd[i][1] << "    " 
            //         << abcd[i][2] << "    " << abcd[i][3] << "    " << endl;
            //}
            solve_tridiagonal_system(abcd, k, len_x);
            //cout << "k:"  << endl;
            //for (int i=0; i<len_x; i++) {
            //    cout << k[i] << endl;
            //}
            //cout << "coeffs:"  << endl;
            coeffs_from_k(x, y, k, coeffs, len_x);
            //for (int i=0; i<len_x-1; i++) {
            //    cout << coeffs[i][0] << "    " << coeffs[i][1] << "    " 
            //         << coeffs[i][2] << "    " << coeffs[i][3] << "    " << endl;
            //}
            delete[] k;
            delete[] abcd;
        }
        
        ~cubic_spline() {
            delete[] coeffs;
        }


        py::array_t<double> get_values_sorted(vector<double> x_eval) {
            int len_out = x_eval.size();
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            double delta_x, t;
            int i_out=0;
            int len_x_mn2 = len_x - 2;

            //i_out = 0;
            for (int i=0; i<len_out; i++) {
                while ((x_eval[i] >= x[i_out+1]  && (i_out < len_x_mn2))) {
                    i_out += 1;
                }
                
                delta_x = x[i_out+1] - x[i_out];
                t = (x_eval[i] - x[i_out]) / delta_x;
                y_out[i] = coeffs[i_out][0] + 
                            coeffs[i_out][1] * t + 
                            coeffs[i_out][2] * t * t + 
                            coeffs[i_out][3] * t * t * t;
                            
            }

            //return y_out;
            return np_arr;
        } 



        py::array_t<double> get_values(vector<double> x_eval) {
            int len_out = x_eval.size();
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            double delta_x, t;
            int i_out=0;
            int len_x_mn2 = len_x - 2;

            vector<int> sort_ind = argsort(x_eval);

            //i_out = 0;
            //for (int i=0; i<len_out; i++) {
            for(int i : sort_ind)  {
                while ((x_eval[i] >= x[i_out+1]  && (i_out < len_x_mn2))) {
                    i_out += 1;
                }
                
                delta_x = x[i_out+1] - x[i_out];
                t = (x_eval[i] - x[i_out]) / delta_x;
                y_out[i] = coeffs[i_out][0] + 
                            coeffs[i_out][1] * t + 
                            coeffs[i_out][2] * t * t + 
                            coeffs[i_out][3] * t * t * t;
                            
            }

            //return y_out;
            return np_arr;
        } 

        

        py::array_t<double> get_integral(vector<double> x_eval) {
            int len_out = x_eval.size()-1;
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            //vector<double> y_out(len_out);
            double delta_x, t;
            int i_out = 0;
            int len_x_mn2 = len_x - 2;

            //i_out = 0;
            while ((x_eval[0] >= x[i_out+1]) && (i_out < len_x_mn2)) {
                i_out += 1;
            }
            delta_x = x[i_out+1] - x[i_out];
            t = (x_eval[0] - x[i_out]) / delta_x;
            for (int i=0; i<len_out; i++) {
                y_out[i] = -(coeffs[i_out][0] * t +
                             coeffs[i_out][1] * t * t / 2. +
                             coeffs[i_out][2] * t * t * t / 3. +
                             coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;
                while ((x_eval[i+1] >= x[i_out+1]) && (i_out < len_x_mn2)) {
                    y_out[i] += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                                 coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                    i_out += 1;
                    delta_x = x[i_out+1] - x[i_out];
                }
                
                t = (x_eval[i+1] - x[i_out]) / delta_x;
                y_out[i] += (coeffs[i_out][0] * t + 
                             coeffs[i_out][1] * t * t / 2. + 
                             coeffs[i_out][2] * t * t * t / 3. + 
                             coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;
            }

            //return y_out;
            return np_arr;
        }    


        double get_integral(double x1, double x2) {
            double delta_x, t, integr_out;
            int i_out = 0;
            int len_x_mn2 = len_x - 2;

            //i_out = 0;
            while ((x1 >= x[i_out+1]) && (i_out < len_x_mn2)) {
                i_out += 1;
            }
            delta_x = x[i_out+1] - x[i_out];
            t = (x1 - x[i_out]) / delta_x;
            
            integr_out = -(coeffs[i_out][0] * t +
                            coeffs[i_out][1] * t * t / 2. +
                            coeffs[i_out][2] * t * t * t / 3. +
                            coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;

            while ((x2 >= x[i_out+1]) && (i_out < len_x_mn2)) {
                integr_out += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                i_out += 1;
                delta_x = x[i_out+1] - x[i_out];
            }
            
            t = (x2 - x[i_out]) / delta_x;
            integr_out += (coeffs[i_out][0] * t + 
                           coeffs[i_out][1] * t * t / 2. + 
                           coeffs[i_out][2] * t * t * t / 3. + 
                           coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;

            return integr_out;
        }    


        py::array_t<double> get_integral_sorted(double x1, vector<double> x_eval) {
            int len_out = x_eval.size();
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            //vector<double> y_out(len_out);
            double integr_offset,integr_incremental, delta_x, t;
            int i_out = 0;
            int len_x_mn2 = len_x - 2;

            integr_offset = 0;
            while ((x1 >= x[i_out+1]) && (i_out < len_x_mn2)) {
                delta_x = x[i_out+1] - x[i_out];
                integr_offset += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                i_out += 1;
            }
            delta_x = x[i_out+1] - x[i_out];
            t = (x1 - x[i_out]) / delta_x;
            
            integr_offset += (coeffs[i_out][0] * t + 
                          coeffs[i_out][1] * t * t / 2. + 
                          coeffs[i_out][2] * t * t * t / 3. + 
                          coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;

            i_out = 0;
            integr_incremental = 0;
            while ((x1 >= x[i_out+1]) && (i_out < len_x_mn2)) {
                delta_x = x[i_out+1] - x[i_out];
                integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                i_out += 1;
            }
            //delta_x = x[i_out+1] - x[i_out];
            //t = (x_eval[0] - x[i_out]) / delta_x;
            for (int i=0; i<len_out; i++) {
                //y_out[i] = -(coeffs[i_out][0] * t +
                //             coeffs[i_out][1] * t * t / 2. +
                //             coeffs[i_out][2] * t * t * t / 3. +
                //             coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;
                while ((x_eval[i] >= x[i_out+1]) && (i_out < len_x_mn2)) {
                    integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                                 coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                    i_out += 1;
                    delta_x = x[i_out+1] - x[i_out];
                }
                
                t = (x_eval[i] - x[i_out]) / delta_x;
                y_out[i] =  integr_incremental + 
                            (coeffs[i_out][0] * t + 
                             coeffs[i_out][1] * t * t / 2. + 
                             coeffs[i_out][2] * t * t * t / 3. + 
                             coeffs[i_out][3] * t * t * t * t / 4.) * delta_x - integr_offset;
            }

            //return y_out;
            return np_arr;
        }    


        py::array_t<double> get_integral(double x1, vector<double> x_eval) {
            int len_out = x_eval.size();
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            //vector<double> y_out(len_out);
            double integr_offset,integr_incremental, delta_x, t;
            int i_out = 0;
            int len_x_mn2 = len_x - 2;

            integr_offset = 0;
            while ((x1 >= x[i_out+1]) && (i_out < len_x_mn2)) {
                delta_x = x[i_out+1] - x[i_out];
                integr_offset += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                i_out += 1;
            }
            delta_x = x[i_out+1] - x[i_out];
            t = (x1 - x[i_out]) / delta_x;
            
            integr_offset += (coeffs[i_out][0] * t + 
                          coeffs[i_out][1] * t * t / 2. + 
                          coeffs[i_out][2] * t * t * t / 3. + 
                          coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;

            i_out = 0;
            integr_incremental = 0;
            while ((x1 >= x[i_out+1]) && (i_out < len_x_mn2)) {
                delta_x = x[i_out+1] - x[i_out];
                integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                               coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                i_out += 1;
            }
            //delta_x = x[i_out+1] - x[i_out];
            //t = (x_eval[0] - x[i_out]) / delta_x;

            vector<int> sort_ind = argsort(x_eval);

            //i_out = 0;
            //for (int i=0; i<len_out; i++) {
            for(int i : sort_ind)  {
            //for (int i=0; i<len_out; i++) {
                //y_out[i] = -(coeffs[i_out][0] * t +
                //             coeffs[i_out][1] * t * t / 2. +
                //             coeffs[i_out][2] * t * t * t / 3. +
                //             coeffs[i_out][3] * t * t * t * t / 4.) * delta_x;
                while ((x_eval[i] >= x[i_out+1]) && (i_out < len_x_mn2)) {
                    integr_incremental += (coeffs[i_out][0] + coeffs[i_out][1] / 2. +
                                 coeffs[i_out][2] / 3. + coeffs[i_out][3] / 4.) * delta_x;
                    i_out += 1;
                    delta_x = x[i_out+1] - x[i_out];
                }
                
                t = (x_eval[i] - x[i_out]) / delta_x;
                y_out[i] =  integr_incremental + 
                            (coeffs[i_out][0] * t + 
                             coeffs[i_out][1] * t * t / 2. + 
                             coeffs[i_out][2] * t * t * t / 3. + 
                             coeffs[i_out][3] * t * t * t * t / 4.) * delta_x - integr_offset;
            }

            //return y_out;
            return np_arr;
        }  


        
        py::array_t<double> get_derivative_sorted(vector<double> x_eval) {
            int len_out = x_eval.size();
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            //vector<double> y_out(len_out);
            double delta_x, t;
            int len_x_mn2 = len_x - 2;
            int i_out=0;

            //i_out = 0; 
            for (int i=0; i<len_out; i++) {
                while ((x_eval[i] >= x[i_out+1]) && (i_out < len_x_mn2)) {
                    i_out += 1;
                }
                delta_x = x[i_out+1] - x[i_out];
                t = (x_eval[i] - x[i_out]) / delta_x;
                y_out[i] = (coeffs[i_out][1] + 
                            coeffs[i_out][2] * t * 2. + 
                            coeffs[i_out][3] * t * t * 3.) / delta_x;
            }

            //return y_out;
            return np_arr;
        }    



        py::array_t<double> get_derivative(vector<double> x_eval) {
            int len_out = x_eval.size();
            py::array_t<double> np_arr = py::array_t<double>(len_out);
            py::buffer_info buf_np_arr = np_arr.request();
            double *y_out = (double *) buf_np_arr.ptr;
            //vector<double> y_out(len_out);
            double delta_x, t;
            int len_x_mn2 = len_x - 2;
            int i_out=0;

            vector<int> sort_ind = argsort(x_eval);
            for(int i : sort_ind)  {
                while ((x_eval[i] >= x[i_out+1]) && (i_out < len_x_mn2)) {
                    i_out += 1;
                }
                delta_x = x[i_out+1] - x[i_out];
                t = (x_eval[i] - x[i_out]) / delta_x;
                y_out[i] = (coeffs[i_out][1] + 
                            coeffs[i_out][2] * t * 2. + 
                            coeffs[i_out][3] * t * t * 3.) / delta_x;
            }

            return np_arr;
        }    



};




void init_spline(py::module_ &m) {
    py::class_<cubic_spline>(m, "cubic_spline")
        .def(py::init<vector<double>, vector<double> >())
        .def("get_values", &cubic_spline::get_values)
        .def("get_values_sorted", &cubic_spline::get_values_sorted)
        .def("get_integral", py::overload_cast<double, double>(&cubic_spline::get_integral))
        .def("get_integral", py::overload_cast<vector<double>>(&cubic_spline::get_integral))
        .def("get_integral", py::overload_cast<double, vector<double>>(&cubic_spline::get_integral))
        .def("get_integral_sorted", &cubic_spline::get_integral_sorted)
        .def("get_derivative", &cubic_spline::get_derivative)
        .def("get_derivative_sorted", &cubic_spline::get_derivative_sorted);
}

/*PYBIND11_MODULE(excursion_set_functions, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("test_sum", &test_sum);

    py::class_<cubic_spline>(m, "cubic_spline")
        .def(py::init<vector<double>, vector<double> >())
        .def("get_values", &cubic_spline::get_values);
}*/