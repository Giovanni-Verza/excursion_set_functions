# excursion_set_functions

C++/python module for excursion set related quantities

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
In addition, the C++ functions make use of [ALGLIB] package and are parallelizad with OpenMP. The modeule is  built with [pybind11][] and [scikit-build-core][]



## Dependencies

The directory `examples` contains a Jupyter notebook for gettting started.


[pybind11]: https://pybind11.readthedocs.io
[scikit-build-core]: https://scikit-build-core.readthedocs.io
[ALGLIB]: https://www.alglib.net/
