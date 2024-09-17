#include <pybind11/pybind11.h>


namespace py = pybind11;

//namespace module_a {
void init_ex_set_analytical(py::module_ &m);

//}

void init_spline(py::module_ &m);

void init_ex_set_cholenski(py::module_ &m);

//void init_ex_set_cholenski_corr(py::module_ &m);

void init_ex_set_integration(py::module_ &m);


PYBIND11_MODULE(excursion_set_functions, m) {
    auto m_analytical = m.def_submodule("analytical", "Analytical multiplicity functions");
    init_ex_set_analytical(m_analytical);
    
    auto m_numerical = m.def_submodule("numerical", "Numerical excursion set function, with Cholenski decomposition");
    init_ex_set_cholenski(m_numerical);
    //init_ex_set_cholenski_corr(m_numerical);

    auto m_spline = m.def_submodule("spline", "Spline functions");
    init_spline(m_spline);

    auto m_integration = m.def_submodule("integration", "Integration functions");
    init_ex_set_integration(m_integration);
}

//PYBIND11_MODULE(analytical, m) {
//    m.attr("__name__") = "excursion_set_functions.analytical";
//    init_ex_set_analytical(m);
//}