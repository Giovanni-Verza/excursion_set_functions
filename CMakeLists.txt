# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

#file(GLOB SOURCES src/cpp/*.cpp)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
## MAIN:
set(SOURCES "src/cpp/main.cpp" 
            "src/cpp/spline_functions.cpp"
            "src/cpp/excursion_set_cholensky.cpp"
            "src/cpp/excursion_set_analytical.cpp"
            "src/cpp/integration_functions.cpp")
file(GLOB SOURCES_ALG "src/cpp/alglib/*.cpp")
file(GLOB SOURCES_H "src/cpp/alglib/*.h")
#python_add_library(_core MODULE src/cpp/main.cpp WITH_SOABI)
#pybind11_add_module(_core MODULE ${SOURCES})
python_add_library(_core MODULE ${SOURCES} ${SOURCES_ALG} WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)
target_link_libraries(_core PRIVATE ${SOURCES_H})
target_include_directories(_core PRIVATE src/cpp/)
target_include_directories(_core PRIVATE src/cpp/alglib/)

#file(GLOB SOURCES src/*.cpp ${skiplist_SOURCE_DIR}/src/cpp/SkipList.cpp)
#file(GLOB SOURCES src/cpp/*.cpp)
#add_library(_core SHARED ${SOURCES})

#python_add_library(_core.spline MODULE src/cpp/spline_functions.cpp WITH_SOABI)
#target_link_libraries(_core.spline PRIVATE pybind11::headers)
#target_link_libraries(_core PRIVATE src/cpp/spline_functions.cpp)
#target_include_directories(_core PRIVATE src/cpp/)

#pybind11_add_module(spline src/cpp/spline_functions.cpp)
#set_target_properties(spline PROPERTIES OUTPUT_NAME _core/spline)

#target_link_libraries(spline PRIVATE pybind11::headers)
#target_link_libraries(spline PRIVATE src/cpp/spline_functions.cpp)
#pybind11_add_module(spline src/cpp/spline_functions.cpp)
#python_add_library(_core.spline MODULE src/cpp/spline_functions.cpp WITH_SOABI)
#target_link_libraries(_core.spline PRIVATE pybind11::headers)
#target_link_libraries(_core PRIVATE src/cpp/)
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/src/cpp/")
target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
#add_executable(spline src/cpp/spline_functions.cpp)

# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION excursion_set_functions)
