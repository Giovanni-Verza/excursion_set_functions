#include <pybind11/pybind11.h>


namespace py = pybind11;


void init_ex_set_analytical(py::module_ &m);

void init_spline(py::module_ &m);

void init_ex_set_cholenski(py::module_ &m);

void init_ex_set_cholenski_corr(py::module_ &m);

void init_ex_set_integration(py::module_ &m);

PYBIND11_MODULE(excursion_set_functions, m) {
    init_ex_set_analytical(m);
    init_spline(m);
    init_ex_set_cholenski(m);
    init_ex_set_cholenski_corr(m);
    init_ex_set_integration(m);
}