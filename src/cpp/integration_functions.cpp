#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <map>
#include <omp.h>
#include <math.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "specialfunctions.h"
//#include <gsl/gsl_sf_expint.h>
//#include <gsl/gsl_sf_trig.h>
//#include <gsl/gsl_sf_bessel.h>

// /usr/include/gsl/
using namespace std;

namespace py = pybind11;






void tridiagonal_elements_for_k_not_a_knot_tmp(vector<double> &x, vector<double> &y, double abcd[][4]) {
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


void solve_tridiagonal_system_tmp(double abcd[][4], double k[],int len_x) {
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

void coeffs_from_k_tmp(vector<double> &x, vector<double> &y, double k[], double coeffs[][4],int len_x) {
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

vector<double> get_values_from_coeffs(vector<double> &x_eval, vector<double> &x, double coeffs[][4]) {
    int len_out = x_eval.size();
    int len_x = x.size();
    vector<double> y_out(len_out);
    double delta_x, t;
    int i_out=0;
    int len_x_mn2 = len_x - 2;

    //i_out = 0;
    //for (int i=0; i<len_out; i++) {
    for(int i=0; i < len_out; i++)  {
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
    return y_out;
} 


vector<double> top_hat_rk_HR(double r, vector<double> &k) {
    double X0=1e-1;
    int len_x = k.size();
    double x;
    vector<double> OUT(len_x);

    double x2, x4, x6, x8;

    int i = 0;
    x = r * k[i];
    while (x < X0) {
        x2 = x*x;
        x4 = x2*x2;
        x6 = x4*x2;
        x8 = x6*x2;
        OUT[i] = 1. - x2 / 10. + x4 / 280. - x6 / 15120. + x8 / 1330560.;
        i += 1;
        x = r * k[i];
    }

    for (int j=i; j<len_x; j++) {
        x = r * k[j];
        // OUT[j] = 3. * gsl_sf_bessel_j1(x) / x;
        // OUT[j] = 3. * alglib::besselj1(x) / x;
        OUT[j] = 3. * (sin(x) - x * cos(x)) / (x * x * x);
        // OUT[j] = 3. * (np.sin(x[x > X0]) - x[x > X0] * np.cos(x[x > X0])) / (x[x > X0] ** 3.);
    }
    return OUT;
}


long long int Factorial(int N){
    long long int OUT = 1;
    if (N > 1) {
        for (int i=2;i<=N;i++) {
            OUT *= i;
        }
    }
    return OUT;
}

double a_fac(int exponent) {
    double OUT;
    if ((exponent % 2) != 0) {
        OUT = 0.;
    } else if (exponent == 0) {
        OUT = 1.;
    } else {
        int N = exponent + 3;
        OUT = 3. * (N-1) / Factorial(N) * pow(-1,(N + 1) / 2);
    }   
    return OUT;
}


double IntegrationSmallK(double Pk0,double k0, double n, double R1, double R2, int Omax) {

    int progr = 0;
    double  ALPHA = 1.;
    double OUT = ALPHA * pow(k0, ((progr + 3) / (progr + 3 + n)));
    // 2
    progr += 2;
    ALPHA = a_fac(2);
    OUT += ALPHA * pow(k0, (progr + 3) / (progr + 3 + n));

    double R1_2 = R1 * R1;
    double R2_2 = R2 * R2;
    double R1_4 = R1_2 * R1_2;
    double R2_4 = R2_2 * R2_2;
    progr += 2;
    ALPHA = a_fac(4) * (R1_4 + R2_4) + a_fac(2) * a_fac(2) * R1_2 * R2_2;
    OUT += ALPHA * pow(k0, ((progr + 3) / (progr + 3 + n)));

    double R1_6 = R1_4 * R1_2;
    double R2_6 = R2_4 * R2_2;
    progr += 2;
    ALPHA = a_fac(6) * (R1_6 + R2_6) + a_fac(2) * a_fac(4) * R1_2 * R2_2 * (R1_4 + R2_4);
    OUT += ALPHA * pow(k0, ((progr + 3) / (progr + 3 + n)));

    double R1_8 = R1_6 * R1_2;
    double R2_8 = R2_6 * R2_2;
    progr += 2;
    ALPHA = a_fac(8) * (R1_8 + R2_8) +
            a_fac(2) * a_fac(6) * R1_2 * R2_2 * (R1_6 + R2_6) + 
            a_fac(4) * a_fac(4) * R1_4 * R2_4;
    OUT += ALPHA * pow(k0, ((progr + 3) / (progr + 3 + n)));

    return Pk0 / (2. * M_PI * M_PI) * OUT;
}




void explicit_from_implicit_coeffs(double coeffs[][4], vector<double> &x) {
    int len_x = x.size();
    double dx;
    for (int i=0; i< len_x-1; i++) {
        dx = x[i+1] - x[i];
        coeffs[i][1] /= dx;
        coeffs[i][2] /= dx * dx;
        coeffs[i][3] /= dx * dx * dx;
        
        coeffs[i][0] += -coeffs[i][1] * x[i] + coeffs[i][2] * x[i] * x[i] - coeffs[i][3] * x[i] * x[i] * x[i];
        coeffs[i][1] += -2. * coeffs[i][2] * x[i] + 3. * coeffs[i][3] * x[i] * x[i];
        coeffs[i][2] += -3. * coeffs[i][3] * x[i];
    }

}



double C_ii_TopHat_MAIN(double coeffs[][4], vector<double> &x, double a, int IDchange) {

    double a2 = a*a;
    double a3 = a*a*a;
    double a6 = a3*a3;
    double Si_2ax, Ci_2ax, cos_2ax, sin_2ax, lnx;
    double Coeff_tot_3, Coeff_tot_2, Coeff_tot_1, Coeff_tot_0;
    double Coeff_tot_3_prev, Coeff_tot_2_prev, Coeff_tot_1_prev, Coeff_tot_0_prev;
    
    double INT0 = 0;
    //int i = IDchange;
    int len_x = x.size();



    alglib::sinecosineintegrals(2. * a * x[IDchange], Si_2ax, Ci_2ax);
    //Si_2ax = gsl_sf_Si(2. * a * x[IDchange]);
    //Ci_2ax = gsl_sf_Ci(2. * a * x[IDchange]);
    lnx = log(x[IDchange]);
    sin_2ax = sin(2. * a * x[IDchange]);
    cos_2ax = cos(2. * a * x[IDchange]);

    Coeff_tot_3_prev = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x[IDchange] * sin_2ax + 0.25 * a2 * x[IDchange]*x[IDchange]);
    Coeff_tot_2_prev = ((cos_2ax - 1.) / (2. * x[IDchange]) + 0.5 * a2 * x[IDchange] + 0.25 * a * sin_2ax);
    Coeff_tot_1_prev = (0.5 * a2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x[IDchange] + 0.25 * (cos_2ax - 1.) / (x[IDchange]*x[IDchange]));
    Coeff_tot_0_prev = (a3 * Si_2ax / 3. + (cos_2ax - 3.) * a2 / (6. * x[IDchange]) + a * sin_2ax / (3. * x[IDchange]*x[IDchange]) + (cos_2ax - 1.) / (6. * x[IDchange]*x[IDchange]*x[IDchange]));
    // #pragma omp parallel for num_threads( nCPU )
    for (int i = IDchange + 1; i < len_x; i++) {

        alglib::sinecosineintegrals(2. * a * x[i], Si_2ax, Ci_2ax);
        //Si_2ax = gsl_sf_Si(2. * a * x[i]);
        //Ci_2ax = gsl_sf_Ci(2. * a * x[i]);
        lnx = log(x[i]);
        sin_2ax = sin(2. * a * x[i]);
        cos_2ax = cos(2. * a * x[i]);

        Coeff_tot_3 = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x[i] * sin_2ax + 0.25 * a2 * x[i]*x[i]);
        Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x[i]) + 0.5 * a2 * x[i] + 0.25 * a * sin_2ax);
        Coeff_tot_1 = (0.5 * a2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x[i] + 0.25 * (cos_2ax - 1.) / (x[i]*x[i]));
        Coeff_tot_0 = (a3 * Si_2ax / 3. + (cos_2ax - 3.) * a2 / (6. * x[i]) + a * sin_2ax / (3. * x[i]*x[i]) + (cos_2ax - 1.) / (6. * x[i]*x[i]*x[i]));


        INT0 += (Coeff_tot_3 - Coeff_tot_3_prev) * coeffs[i-1][3] + 
                (Coeff_tot_2 - Coeff_tot_2_prev) * coeffs[i-1][2] + 
                (Coeff_tot_1 - Coeff_tot_1_prev) * coeffs[i-1][1] + 
                (Coeff_tot_0 - Coeff_tot_0_prev) * coeffs[i-1][0];

        Coeff_tot_3_prev = Coeff_tot_3;
        Coeff_tot_2_prev = Coeff_tot_2;
        Coeff_tot_1_prev = Coeff_tot_1;
        Coeff_tot_0_prev = Coeff_tot_0;

    }
    
    INT0 *= 9. / (2. * M_PI * M_PI) / a6;

    return INT0;
}




double C_ij_TopHat_MAIN(double coeffs[][4], vector<double> &x, double a, double b, int IDchange) {

    double ab_diff = a - b;
    double ab_sum = a + b;
    double ab_diff_2 = ab_diff * ab_diff;
    double ab_sum_2 = ab_sum * ab_sum;
    double a2 = a * a;
    double b2 = b * b;
    double a3 = a * a * a;
    double b3 = b * b * b;
    double ab_prod = a * b;

    double Si_diff, Ci_diff, Si_sum, Ci_sum;
    double cos_diff, cos_sum, sin_diff, sin_sum;
    double Coeff_tot_3, Coeff_tot_2, Coeff_tot_1, Coeff_tot_0;
    double Coeff_tot_3_prev, Coeff_tot_2_prev, Coeff_tot_1_prev, Coeff_tot_0_prev;

    double INT0 = 0;
    double x2, x3;

    int i = IDchange; 
    int len_x = x.size();

    x2 = x[i] * x[i];
    x3 = x[i] * x2;
    alglib::sinecosineintegrals(ab_diff * x[i], Si_diff, Ci_diff);
    alglib::sinecosineintegrals(ab_sum * x[i], Si_sum, Ci_sum);
    //Si_diff = gsl_sf_Si(ab_diff * x[i]);
    //Si_sum = gsl_sf_Si(ab_sum * x[i]);
    //Ci_diff = gsl_sf_Ci(ab_diff * x[i]);
    //Ci_sum = gsl_sf_Ci(ab_sum * x[i]);
    
    sin_diff = sin(ab_diff * x[i]);
    sin_sum = sin(ab_sum * x[i]);
    cos_diff = cos(ab_diff * x[i]);
    cos_sum = cos(ab_sum * x[i]);



    Coeff_tot_3_prev = 0.5 * (Ci_diff - Ci_sum + ab_prod*x[i]*(sin_diff / ab_diff + sin_sum / ab_sum) + 
                              (ab_prod / ab_diff_2 - 1) * cos_diff + (ab_prod / ab_sum_2 + 1) * cos_sum);
    Coeff_tot_2_prev = 0.5 * ((cos_sum - cos_diff) / x[i] + ab_prod / ab_diff * sin_diff + ab_prod / ab_sum * sin_sum);
    Coeff_tot_1_prev = 0.25 * ((a2 + b2) * (Ci_diff - Ci_sum) - (ab_diff * sin_diff - ab_sum * sin_sum) / x[i] - (cos_diff - cos_sum) / x2 );
    Coeff_tot_0_prev = (-(a3 - b3) * Si_diff + (a3 + b3) * Si_sum - ab_diff * sin_diff/x2 + ab_sum * sin_sum/x2 -
                        ((a2+b2+ab_prod)/x[i] + 1./x3) * cos_diff + ((a2+b2-ab_prod)/x[i] + 1./x3) * cos_sum) / 6.;

    for (int i = IDchange + 1; i < len_x; i++) {

        x2 = x[i] * x[i];
        x3 = x[i] * x2;
        alglib::sinecosineintegrals(ab_diff * x[i], Si_diff, Ci_diff);
        alglib::sinecosineintegrals(ab_sum * x[i], Si_sum, Ci_sum);
        //Si_diff = gsl_sf_Si(ab_diff * x[i]);
        //Si_sum = gsl_sf_Si(ab_sum * x[i]);
        //Ci_diff = gsl_sf_Ci(ab_diff * x[i]);
        //Ci_sum = gsl_sf_Ci(ab_sum * x[i]);
        
        sin_diff = sin(ab_diff * x[i]);
        sin_sum = sin(ab_sum * x[i]);
        cos_diff = cos(ab_diff * x[i]);
        cos_sum = cos(ab_sum * x[i]);



        Coeff_tot_3 = 0.5 * (Ci_diff - Ci_sum + a*b*x[i]*(sin_diff / ab_diff + sin_sum / ab_sum) + 
                             (a*b / ab_diff_2 - 1) * cos_diff + (a*b / ab_sum_2 + 1) * cos_sum);
        Coeff_tot_2 = 0.5 * ((cos_sum - cos_diff) / x[i] + a*b / ab_diff * sin_diff + a*b / ab_sum * sin_sum);
        Coeff_tot_1 = 0.25 * ((a2 + b2) * (Ci_diff - Ci_sum) - (ab_diff * sin_diff - ab_sum * sin_sum) / x[i] - (cos_diff - cos_sum) / x2 );
        Coeff_tot_0 = (-(a3 - b3) * Si_diff + (a3 + b3) * Si_sum - ab_diff * sin_diff/x2 + ab_sum * sin_sum/x2 -
                        ((a2+b2+ab_prod)/x[i] + 1./x3) * cos_diff + ((a2+b2-ab_prod)/x[i] + 1./x3) * cos_sum) / 6.;
        
        INT0 += (Coeff_tot_3 - Coeff_tot_3_prev) * coeffs[i-1][3] + 
                (Coeff_tot_2 - Coeff_tot_2_prev) * coeffs[i-1][2] + 
                (Coeff_tot_1 - Coeff_tot_1_prev) * coeffs[i-1][1] + 
                (Coeff_tot_0 - Coeff_tot_0_prev) * coeffs[i-1][0];

        Coeff_tot_3_prev = Coeff_tot_3;
        Coeff_tot_2_prev = Coeff_tot_2;
        Coeff_tot_1_prev = Coeff_tot_1;
        Coeff_tot_0_prev = Coeff_tot_0;
        
    }
    
    INT0 *= 9. / (2. * M_PI * M_PI) / (a3 * b3);

    return INT0;

}


double C_ij_TopHat_MAIN_lowR(
        vector<double> &Pk, vector<double> &k, double R1, double R2, int IDchange, int OMAX, double n) {

    int len_x = Pk.size();
    double INT0;

    vector<double> Pk_k2_w1w2(len_x);
    vector<double> W1 = top_hat_rk_HR(R1, k);
    vector<double> W2 = top_hat_rk_HR(R2, k);
    for (int i = 0; i < len_x; i++) {
        Pk_k2_w1w2[i] = Pk[i] * k[i] * k[i] * W1[i] * W2[i];
    }

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    
    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk_k2_w1w2, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk_k2_w1w2, xx, coeffs, len_x);

    delete[] xx;
    delete[] abcd;

    if (OMAX <= 0) {
        INT0 = 0;
    } else {
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R1, R2, OMAX);
    }

    for (int i = 0; i < IDchange; i++) {
        INT0 += (coeffs[i][0] + 
                 coeffs[i][1] / 2. +
                 coeffs[i][2] / 3. + 
                 coeffs[i][3] / 4.) * (k[i+1] -  k[i]);
    }
    
    INT0 /= 2. * M_PI * M_PI;
    
    delete[] coeffs;
    
    return INT0;
}

double C_ii_TopHat_MAIN_lowR(
        vector<double> &Pk, vector<double> &k, double R, int IDchange, int OMAX, double n) {
    

    int len_x = Pk.size();
    double INT0;

    vector<double> Pk_k2_w1w2(len_x);
    vector<double> W1 = top_hat_rk_HR(R, k);
    for (int i = 0; i < len_x; i++) {
        Pk_k2_w1w2[i] = Pk[i] * k[i] * k[i] * W1[i] * W1[i];
    }

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    
    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk_k2_w1w2, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk_k2_w1w2, xx, coeffs, len_x);

    delete[] xx;
    delete[] abcd;


    if (OMAX <= 0) {
        INT0 = 0;
    } else {
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R, R, OMAX);
    }

    for (int i = 0; i < IDchange; i++) {
        INT0 += (coeffs[i][0] + 
                 coeffs[i][1] / 2. +
                 coeffs[i][2] / 3. + 
                 coeffs[i][3] / 4.) * (k[i+1] -  k[i]);
    }
    
    INT0 /= 2. * M_PI * M_PI;
    
    delete[] coeffs;
    
    return INT0;
}



int find_id_change(vector<double> &x_array,double x_max) {
    int ind = 0;
    int len_arr_mn1 = x_array.size() - 1;
    while ((x_array[ind] < x_max) && (ind < len_arr_mn1)) {
        ind += 1;
    }
    return ind;
}



py::array_t<double> C_ij_TopHat(vector<double> &Pk, vector<double> &k, vector<double> &R, double n, int OmaxSmallK) {


    int len_x = Pk.size();
    int len_R = R.size();

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    

    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk, xx, coeffs, len_x);

    explicit_from_implicit_coeffs(coeffs, k);

    delete[] xx;
    delete[] abcd;


    py::array_t<double> Cij_out = py::array_t<double>(len_R*len_R);

    py::buffer_info buf_Cij = Cij_out.request();


    double *ptr_Cij = (double *) buf_Cij.ptr;

    for (int i=0; i < len_R; i++){
        for (int j=0; j < i; j++) {
            int IDchange = find_id_change(k,1./sqrt(R[i] * R[j]));
            //cout << i << "," << j << "," << i*len_R + j << endl;
            ptr_Cij[i*len_R + j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i], R[j], IDchange, OmaxSmallK, n);
            //cout << "   IDchange: " << IDchange << ", sqrt:" << 1./sqrt(R[i] * R[j]) << ", term1:" << ptr_Cij[i*len_R + j] << ", term2:" <<C_ij_TopHat_MAIN(coeffs, k, R[i], R[j],  IDchange);
            ptr_Cij[i*len_R + j] += C_ij_TopHat_MAIN(coeffs, k, R[i], R[j],  IDchange);
            //cout << "," << ptr_Cij[i*len_R + j] << endl;
            ptr_Cij[i + j*len_R] = ptr_Cij[i*len_R + j];
        }

        int IDchange = find_id_change(k,1./R[i]);
        // cout << i  << "," << i*len_R + i << endl;
        ptr_Cij[i*len_R + i] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n);
        // cout << "   IDchange: " << IDchange << ", sqrt:" << 1./R[i] << ", term1:" << ptr_Cij[i*len_R + i] << ", term2:" <<C_ii_TopHat_MAIN(coeffs, k, R[i],IDchange);
        ptr_Cij[i*len_R + i] += C_ii_TopHat_MAIN(coeffs, k, R[i],IDchange);
        // cout << "," << ptr_Cij[i*len_R + i] << endl;
        
    }
    delete[] coeffs;

    Cij_out.resize({len_R,len_R});
    return Cij_out;
}




py::array_t<double> sigma2_2_TopHat_numdiff(
        vector<double> &Pk, vector<double> &k, vector<double> &R, double n, int OmaxSmallK,double dRperc) {

    int len_x = Pk.size();
    int len_R = R.size();

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    

    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk, xx, coeffs, len_x);

    explicit_from_implicit_coeffs(coeffs, k);

    delete[] xx;
    delete[] abcd;


    py::array_t<double> OUT = py::array_t<double>(len_R);

    py::buffer_info buf_OUT = OUT.request();

    double *ptr_OUT = (double *) buf_OUT.ptr;

    int j;
    double *grid_R1R2 = new double[3];
    int IDchange;
    for (int i=0; i<len_R; i++) {
        j=0; //--
        IDchange = find_id_change(k,1./R[i]);
        grid_R1R2[j] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1.-dRperc), IDchange, OmaxSmallK, n);
        grid_R1R2[j] += C_ii_TopHat_MAIN(coeffs, k, R[i]*(1.-dRperc),IDchange);
        j=1; //+-
        grid_R1R2[j] = C_ij_TopHat_MAIN_lowR(Pk, k, R[i]*(1.+dRperc), R[i]*(1.-dRperc), IDchange, OmaxSmallK, n);
        grid_R1R2[j] += C_ij_TopHat_MAIN(coeffs, k, R[i]*(1.+dRperc), R[i]*(1.-dRperc),IDchange);
        j=2; //++
        grid_R1R2[j] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i]*(1.+dRperc), IDchange, OmaxSmallK, n);
        grid_R1R2[j] += C_ii_TopHat_MAIN(coeffs, k, R[i]*(1.+dRperc),IDchange);

        ptr_OUT[i] = 0.25 * (grid_R1R2[2] - 2. * grid_R1R2[1] + grid_R1R2[0]) / (R[i]*R[i]*dRperc*dRperc);
    }
    return OUT;

}


py::array_t<double> sigma2_TopHat(vector<double> &Pk, vector<double> &k, vector<double> &R, double n, int OmaxSmallK) {

    int len_x = Pk.size();
    int len_R = R.size();

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    

    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk, xx, coeffs, len_x);


    explicit_from_implicit_coeffs(coeffs, k);


    py::array_t<double> OUT = py::array_t<double>(len_R);

    py::buffer_info buf_OUT = OUT.request();


    double *ptr_OUT = (double *) buf_OUT.ptr;


    for (int i=0; i < len_R; i++) {
        int IDchange = find_id_change(k,1./R[i]);
        ptr_OUT[i] = C_ii_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n);
        ptr_OUT[i] += C_ii_TopHat_MAIN(coeffs, k, R[i],IDchange);
    }    

    delete[] coeffs;
    return OUT;
}





vector<double> dr_square_top_hat_rk_HR(double r, vector<double> &k) {
    
    double X0=1e-1;
    int len_x = k.size();
    double x;
    vector<double> OUT(len_x);

    double sin_x, cos_x, x2, x4, x6, x8;

    int i = 0;
    x = r * k[i];
    while (x < X0) {
        x2 = x*x;
        x4 = x2*x2;
        x6 = x4*x2;
        x8 = x6*x2;
        OUT[i] = 2. * k[i] * x * (-1 / 5. + x2/70. - x4/2520. +x6/166320.) * (1. - x2/10. +x4/280. - x6/15120. +x8/1330560.);

        i += 1;
        x = r * k[i];
    }


    for (int j=i; j<len_x; j++) {
        x = r * k[j];
        x2 = x*x;
        sin_x = sin(x);
        cos_x = cos(x);
        // OUT[j] = 18. * ((x**2-3.) * sin(x)**2 - 3. * x * x * cos(x)**2 + (6.*x - x*x*x) * sin(x) * cos(x)) / (x * x * x * x * x * x * x);
        OUT[j] = 18. * k[j] * ((x2 - 3.) * sin_x * sin_x - 3. * x2 * cos_x * cos_x + 
                               (6. - x2) * x * sin_x * cos_x) / (x2 * x2 * x2 * x);
    }

    return OUT;
}




double dSdR_TopHat_MAIN(double coeffs[][4], vector<double> &x, double a, int IDchange) {

    double a2 = a*a;
    double a3 = a*a*a;
    double a6 = a3*a3;
    double Si_2ax, Ci_2ax, cos_2ax, sin_2ax, lnx;

    double Coeff_tot_3, Coeff_tot_2, Coeff_tot_1, Coeff_tot_0;
    double Coeff_tot_3_prev, Coeff_tot_2_prev, Coeff_tot_1_prev, Coeff_tot_0_prev;
    double da_Coeff_tot_3, da_Coeff_tot_2, da_Coeff_tot_1, da_Coeff_tot_0;
    double da_Coeff_tot_3_prev, da_Coeff_tot_2_prev, da_Coeff_tot_1_prev, da_Coeff_tot_0_prev;
    
    double INT0 = 0;
    //int i = IDchange;
    int len_x = x.size();



    alglib::sinecosineintegrals(2. * a * x[IDchange], Si_2ax, Ci_2ax);
    //Si_2ax = gsl_sf_Si(2. * a * x[IDchange]);
    //Ci_2ax = gsl_sf_Ci(2. * a * x[IDchange]);
    lnx = log(x[IDchange]);
    sin_2ax = sin(2. * a * x[IDchange]);
    cos_2ax = cos(2. * a * x[IDchange]);

    Coeff_tot_3_prev = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x[IDchange] * sin_2ax + 0.25 * a2 * x[IDchange]*x[IDchange]);
    Coeff_tot_2_prev = ((cos_2ax - 1.) / (2. * x[IDchange]) + 0.5 * a2 * x[IDchange] + 0.25 * a * sin_2ax);
    Coeff_tot_1_prev = (0.5 * a2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x[IDchange] + 0.25 * (cos_2ax - 1.) / (x[IDchange]*x[IDchange]));
    Coeff_tot_0_prev = (a3 * Si_2ax / 3. + (cos_2ax - 3.) * a2 / (6. * x[IDchange]) + a * sin_2ax / (3. * x[IDchange]*x[IDchange]) + (cos_2ax - 1.) / (6. * x[IDchange]*x[IDchange]*x[IDchange]));

    da_Coeff_tot_3_prev = 0.5 * (a*x[IDchange]*x[IDchange] - 1./a) * cos_2ax - x[IDchange]*sin_2ax + 0.5*a*x[IDchange]*x[IDchange];
    da_Coeff_tot_2_prev = 0.5 * a*x[IDchange] * cos_2ax - 0.75 * sin_2ax + a*x[IDchange];
    da_Coeff_tot_1_prev = a * (lnx - Ci_2ax + 0.5*cos_2ax);
    da_Coeff_tot_0_prev = a2 * Si_2ax  + a * (cos_2ax - 1.) / x[IDchange];

    // #pragma omp parallel for num_threads( nCPU )
    for (int i = IDchange + 1; i < len_x; i++) {

        alglib::sinecosineintegrals(2. * a * x[i], Si_2ax, Ci_2ax);
        lnx = log(x[i]);
        sin_2ax = sin(2. * a * x[i]);
        cos_2ax = cos(2. * a * x[i]);

        // standard part
        Coeff_tot_3 = (0.5 * lnx - 0.5 * Ci_2ax + 5. / 8. * cos_2ax + 0.25 * a * x[i] * sin_2ax + 0.25 * a2 * x[i]*x[i]);
        Coeff_tot_2 = ((cos_2ax - 1.) / (2. * x[i]) + 0.5 * a2 * x[i] + 0.25 * a * sin_2ax);
        Coeff_tot_1 = (0.5 * a2 * (lnx - Ci_2ax) + 0.5 * a * sin_2ax / x[i] + 0.25 * (cos_2ax - 1.) / (x[i]*x[i]));
        Coeff_tot_0 = (a3 * Si_2ax / 3. + (cos_2ax - 3.) * a2 / (6. * x[i]) + a * sin_2ax / (3. * x[i]*x[i]) + (cos_2ax - 1.) / (6. * x[i]*x[i]*x[i]));


        INT0 -= 6./a * ((Coeff_tot_3 - Coeff_tot_3_prev) * coeffs[i-1][3] +
                        (Coeff_tot_2 - Coeff_tot_2_prev) * coeffs[i-1][2] + 
                        (Coeff_tot_1 - Coeff_tot_1_prev) * coeffs[i-1][1] + 
                        (Coeff_tot_0 - Coeff_tot_0_prev) * coeffs[i-1][0]);

        Coeff_tot_3_prev = Coeff_tot_3;
        Coeff_tot_2_prev = Coeff_tot_2;
        Coeff_tot_1_prev = Coeff_tot_1;
        Coeff_tot_0_prev = Coeff_tot_0;


        // derivative part
        da_Coeff_tot_3 = 0.5 * (a*x[i]*x[i] - 1./a) * cos_2ax - x[i]*sin_2ax + 0.5*a*x[i]*x[i];
        da_Coeff_tot_2 = 0.5 * a*x[i] * cos_2ax - 0.75 * sin_2ax + a*x[i];
        da_Coeff_tot_1 = a * (lnx - Ci_2ax + 0.5*cos_2ax);
        da_Coeff_tot_0 = a2 * Si_2ax  + a * (cos_2ax - 1.) / x[i];

        INT0 += (da_Coeff_tot_3 - da_Coeff_tot_3_prev) * coeffs[i-1][3] + 
                (da_Coeff_tot_2 - da_Coeff_tot_2_prev) * coeffs[i-1][2] + 
                (da_Coeff_tot_1 - da_Coeff_tot_1_prev) * coeffs[i-1][1] + 
                (da_Coeff_tot_0 - da_Coeff_tot_0_prev) * coeffs[i-1][0];


        da_Coeff_tot_3_prev = da_Coeff_tot_3;
        da_Coeff_tot_2_prev = da_Coeff_tot_2;
        da_Coeff_tot_1_prev = da_Coeff_tot_1;
        da_Coeff_tot_0_prev = da_Coeff_tot_0;

    }
    
    INT0 *= 9. / (2. * M_PI * M_PI) / a6;

    return INT0;
}



double dSdR_TopHat_MAIN_lowR(
        vector<double> &Pk, vector<double> &k, double R, int IDchange, int OMAX, double n) {
    

    int len_x = Pk.size();
    double INT0;

    vector<double> Pk_k2_w1w2(len_x);
    vector<double> dr_W2 = dr_square_top_hat_rk_HR(R,k);
    for (int i = 0; i < len_x; i++) {
        Pk_k2_w1w2[i] = Pk[i] * k[i] * k[i] * dr_W2[i];
    }

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    
    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk_k2_w1w2, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk_k2_w1w2, xx, coeffs, len_x);

    delete[] xx;
    delete[] abcd;


    if (OMAX <= 0) {
        INT0 = 0;
    } else {
        INT0 = IntegrationSmallK(Pk[0], k[0], n, R, R, OMAX);
    }

    for (int i = 0; i < IDchange; i++) {
        INT0 += (coeffs[i][0] + 
                 coeffs[i][1] / 2. +
                 coeffs[i][2] / 3. + 
                 coeffs[i][3] / 4.) * (k[i+1] -  k[i]);
    }
    
    INT0 /= 2. * M_PI * M_PI;
    
    delete[] coeffs;
    
    return INT0;
}

py::array_t<double> dSdR_TopHat(vector<double> &Pk, vector<double> &k, vector<double> &R, double n, int OmaxSmallK) {

    int len_x = Pk.size();
    int len_R = R.size();

    double (*coeffs)[4] = new double[len_x-1][4];
    double (*abcd)[4] = new double[len_x][4];
    double *xx = new double[len_x];
    

    tridiagonal_elements_for_k_not_a_knot_tmp(k, Pk, abcd);
    solve_tridiagonal_system_tmp(abcd, xx, len_x);
    coeffs_from_k_tmp(k, Pk, xx, coeffs, len_x);


    explicit_from_implicit_coeffs(coeffs, k);


    py::array_t<double> OUT = py::array_t<double>(len_R);

    py::buffer_info buf_OUT = OUT.request();


    double *ptr_OUT = (double *) buf_OUT.ptr;


    for (int i=0; i < len_R; i++) {
        int IDchange = find_id_change(k,1./R[i]);
        ptr_OUT[i] = dSdR_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n);
        ptr_OUT[i] += dSdR_TopHat_MAIN(coeffs, k, R[i],IDchange);
        //cout << "i: " << i << ", IDchange: " << IDchange << ", " << dSdR_TopHat_MAIN_lowR(Pk, k, R[i], IDchange, OmaxSmallK, n) << ", " << dSdR_TopHat_MAIN(coeffs, k, R[i],IDchange) << endl;
    }    

    delete[] coeffs;
    return OUT;
}




void init_ex_set_integration(py::module_ &m) {
    m.def("C_ij_TopHat", &C_ij_TopHat,
          py::arg("Pk"), py::arg("k"), py::arg("R"), py::arg("n")=0.96, py::arg("OmaxSmallK")=-1);
    m.def("sigma2_TopHat", &sigma2_TopHat,
          py::arg("Pk"), py::arg("k"), py::arg("R"), py::arg("n")=0.96, py::arg("OmaxSmallK")=-1);
    m.def("sigma2_2_TopHat_numdiff", &sigma2_2_TopHat_numdiff,
          py::arg("Pk"), py::arg("k"), py::arg("R"), py::arg("n")=0.96, py::arg("OmaxSmallK")=-1, py::arg("dRperc")=5e-3);
    m.def("dSdR_TopHat", &dSdR_TopHat,
          py::arg("Pk"), py::arg("k"), py::arg("R"), py::arg("n")=0.96, py::arg("OmaxSmallK")=-1);
    m.def("top_hat_rk_HR", &top_hat_rk_HR);
    //m.def("dr_square_top_hat_rk_HR", &dr_square_top_hat_rk_HR);
}