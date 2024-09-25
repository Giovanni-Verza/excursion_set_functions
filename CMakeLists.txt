# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Find the module development requirements
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA)

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

set(SOURCES "src/cpp/main.cpp" 
            "src/cpp/spline_functions.cpp"
            "src/cpp/excursion_set_cholensky.cpp"
            "src/cpp/excursion_set_analytical.cpp"
            "src/cpp/integration_functions.cpp")
file(GLOB ALGLIB_SOURCES "external/alglib/*.cpp")
file(GLOB ALGLIB_LIB "external/alglib/*.h")

python_add_library(_core MODULE ${SOURCES} ${ALGLIB_SOURCES} WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PRIVATE ${ALGLIB_LIB})
target_include_directories(_core PRIVATE src/cpp/)
target_include_directories(_core PRIVATE external/alglib/)
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
#if(CUDA_FOUND)
#    message("CUDA is available. Version: ${CUDA_VERSION}")
#    #enable_language(CUDA)
#    find_package(OpenACC REQUIRED)
#    target_link_libraries(_core PRIVATE OpenACC::OpenACC_CXX)
#else()
#    message("CUDA is not available.")
#endif()


# Passing in the version
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION excursion_set_functions)

