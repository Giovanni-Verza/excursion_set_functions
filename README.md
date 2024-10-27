# excursion_set_functions

C++/python module to compute excursion-set related quantities, such as analytical and numerical multiplicity functions, Lagrangian void density profiles, various integation moments and covariance of the power spectrum. This module it has been use for computations in [Verza et al. 2024][]

## Installation

The module can be installed by running

`pip install .`

Note: on OSX there are some difficulties to use the native clang compiler. Please use a brew installed compiler like GCC.

`brew install gcc`
and e.g.

```
export CC=/opt/homebrew/bin/gcc-14
export CXX=/opt/homebrew/bin/gcc-14
pip install . 
```

where the actual path and version depend on the path and version installed by brew. In most of the cases it would be enough to specify the verions only, e.g. `export CC=gcc-14`. 

## Dependencies

The excursion_set_functions make use of

- `pybind11`

for the C++ part, and

- `numpy`
- `numba`
- `scipy`

for the pure python part. 
In addition, the C++ functions make use of [ALGLIB][] package and are parallelizad with OpenMP. The modeule is  built with [pybind11][] and [scikit-build-core][]. The module has been tested for Python 3.7+. 




## Getting started

The directory `examples` contains a Jupyter notebook for gettting started. 

The module is composed by 4 c++ modules:  

- `analytical` contains analytical multiplicity functions
- `numerical` contains functions that compute numerical multiplicity functions via Montecarlo techniques
- `integration` contains function to compute various integration of the powerspectrum
- `spiline` contains spline functions.

Beyound these 4 modules there are the `utilities` and `python` submodules. The `utitlities` submodule contains some useful functions, not strictly related to the excursion set. The `python` submodule contains the same main 4 c++ modules, but written in pure python and, whenever possible, accelerated with `numba`. In general the c++ modules performance is always better than the pure python ones. This is always true on Linux systems, however we noted that on OSX the numba-python version of the `numerical` module may be slightly more performant.


[pybind11]: https://pybind11.readthedocs.io
[scikit-build-core]: https://scikit-build-core.readthedocs.io
[ALGLIB]: https://www.alglib.net/
[Verza et al. 2024]: https://arxiv.org/abs/2401.14451
