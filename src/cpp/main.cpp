#include <pybind11/pybind11.h>
//#include <omp.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

//namespace module_a {
void init_ex_set_analytical(py::module_ &m);

//}

void init_spline(py::module_ &m);

void init_ex_set_cholenski(py::module_ &m);

//void init_ex_set_cholenski_corr(py::module_ &m);

void init_ex_set_integration(py::module_ &m);


PYBIND11_MODULE(_core, m) {

    m.doc() = R"pbdoc(
        excursion_set_functions, C++ and python package to compute excursion set related quantities.
        -----------------------

        .. currentmodule:: excursion_set_functions

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";


    //m.def("get_max_threads", &omp_get_max_threads, "Returns max number of threads");
    //m.def("set_num_threads", &omp_set_num_threads, "Set number of threads");
    //m.def("omp_get_num_threads", &omp_get_num_threads, "Get number of threads");

    auto m_analytical = m.def_submodule("analytical", R"pbdoc(
            Analytical multiplicity functions
        )pbdoc");
    init_ex_set_analytical(m_analytical);
    
    auto m_numerical = m.def_submodule("numerical", R"pbdoc(
            Numerical excursion set function, with Cholenski decomposition
        )pbdoc");
    init_ex_set_cholenski(m_numerical);
    ////init_ex_set_cholenski_corr(m_numerical);

    auto m_spline = m.def_submodule("spline", R"pbdoc(
            Spline functions
        )pbdoc");
    init_spline(m_spline);

    auto m_integration = m.def_submodule("integration", R"pbdoc(
            Integration functions
        )pbdoc");
    init_ex_set_integration(m_integration);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

//PYBIND11_MODULE(analytical, m) {
//    m.attr("__name__") = "excursion_set_functions.analytical";
//    init_ex_set_analytical(m);
//}