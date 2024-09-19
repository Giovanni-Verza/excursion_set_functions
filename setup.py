from setuptools import setup, find_packages
from glob import glob
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import subprocess

import os
import sys

__version__ = "0.2"

# The main interface is through Pybind11Extension.
# * You can add cxx_std=11/14/17, and then build_ext can be removed.
# * You can set include_pybind11=false to add the include directory yourself,
#   say from a submodule.
#
# Note:
#   Sort input source files if you glob sources to ensure bit-for-bit
#   reproducible builds (https://github.com/pybind/python_example/pull/53)

PWD = os.path.dirname(os.path.realpath(__file__)) + '/'
print('PWD:', PWD,flush=True)

OPENMP = False

extra_compile_args = []
extra_link_args = []


if OPENMP:
    extra_compile_args.append('-fopenmp')
    extra_link_args.append('-fopenmp')
    cpp_file_path =  PWD + 'src/cpp/'
else:
    cpp_file_path = PWD + 'src/cpp_noopenmp/'
    if not os.path.exists(cpp_file_path):
        os.makedirs(cpp_file_path)
    subprocess.run('cp ' + PWD + 'src/cpp/* ' + cpp_file_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for FF in os.listdir(cpp_file_path):
        if 'excursion_set' in cpp_file_path:
            # Read in the file
            print('    FF:',FF,flush=True)
            with open(cpp_file_path + FF, 'r') as file:
                filedata = file.read()

            # Replace the target string
            filedata = filedata.replace('#include <omp.h>', '//#include <omp.h>')

            # Write the file out again
            with open(cpp_file_path + FF, 'w') as file:
                file.write(filedata)

extra_compile_args.append('-I'+cpp_file_path)

file_list_cpp = []
for ff in sorted(glob(cpp_file_path + '*.cpp')):
    if ('OLD_' in ff) | ('_OLD' in ff):
        #print(ff)
        pass
    else:
        file_list_cpp.append(ff)

ext_modules = [
    Pybind11Extension("excursion_set_functions",
                      file_list_cpp,
                      define_macros = [('VERSION_INFO', __version__)],
                      extra_compile_args=extra_compile_args,#,*sorted(glob("src/cpp/*.h"))], #'-O3'],
                      extra_link_args=extra_link_args,
                      # libraries = ['omp'],
                      # extra_link_args=['-fopenmp','-O2']
                      # ),
    )
]
asd = 1
print(find_packages()+['excursion_set.python'])

setup(
    name="excursion_set_functions",
    version=__version__,
    author="Giovanni Verza",
    author_email="giova.verza@gmail.com",
    url="",
    description="Excursion-set functions with Cholensky decomposition",
    long_description="",
    #py_modules=['excursion_set'],
    package_dir={'./':'excursion_set_functions'},
    packages=find_packages(), #include=['excursion_set','excursion_set.*']),#['excursion_set']
    ext_modules=ext_modules,
    #extras_require={"test": "pytest"},
    # Currently, build_ext only provides an optional "highest supported C++
    # level" feature, but in the future it may provide more features.
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
