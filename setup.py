from setuptools import setup, find_packages
from glob import glob
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import sys

__version__ = "0.0.1"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

file_list_cpp = []
for ff in sorted(glob("src/cpp/*.cpp")):
    if ('OLD_' in ff) | ('_OLD' in ff):
        #print(ff)
        pass
    else:
        file_list_cpp.append(ff)

ext_modules = [
    Pybind11Extension("excursion_set_functions",file_list_cpp,
        #["src/cpp/excursion_set_cholensky.cpp",
        #"src/cpp/excursion_set_cholensky_correlations.cpp",
        #"src/cpp/excursion_set_analytical.cpp",
        #"src/cpp/spline_functions.cpp",
        #"src/cpp/integration_functions.cpp",
        #"src/cpp/excursion_set_functions.cpp"],
        # Example: passing in the version to the compiled code
        define_macros = [('VERSION_INFO', __version__)],
        extra_compile_args=['-fopenmp','-I/home/giovanni/Desktop/Excursion_Set/excursion_set_functions/src/cpp/'],#,*sorted(glob("src/cpp/*.h"))], #'-O3'],
        #libraries = ['omp'],
        #extra_link_args=['-fopenmp','-O2']
        ),
]

print(find_packages()+['excursion_set.python'])

setup(
    name="excursion_set",
    version=__version__,
    author="Giovanni Verza",
    author_email="giova.verza@gmail.com",
    url="",
    description="Excursion-set functions with Cholensky decomposition",
    long_description="",
    #py_modules=['excursion_set'],
    package_dir={'./':'excursion_set'},
    packages=find_packages(), #include=['excursion_set','excursion_set.*']),#['excursion_set']
    ext_modules=ext_modules,
    #extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
