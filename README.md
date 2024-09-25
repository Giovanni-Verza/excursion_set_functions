# excursion_set_functions

C++/python module to compute excursion-set related quantities, such as analytical and numerical multiplicity functions, Lagrangian void density profiles, various integation moments and covariance of the power spectrum. This module it has been use for computations in [Verza et al. 2024][]

## Installation

The module can be installed by running

`pip install .`

## Dependencies

Tested for Python 3.7+. The excursion_set_functions make use of

- `pybind11`

for the C++ part, and

- `numpy`
- `numba`
- `scipy`

for the pure python part. 
In addition, the C++ functions make use of [ALGLIB][] package and are parallelizad with OpenMP. The modeule is  built with [pybind11][] and [scikit-build-core][]



## Dependencies

The directory `examples` contains a Jupyter notebook for gettting started.


[pybind11]: https://pybind11.readthedocs.io
[scikit-build-core]: https://scikit-build-core.readthedocs.io
[ALGLIB]: https://www.alglib.net/
[Verza et al. 2024]: https://arxiv.org/abs/2401.14451
