cmake_minimum_required(VERSION 3.14)
project(NumCppProject LANGUAGES CXX)

# Changed from Debug to Release for optimal performance
set(CMAKE_BUILD_TYPE Release)

# Setting C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Find OpenBLAS using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENBLAS REQUIRED openblas)

# Fetch dependencies
include(FetchContent)
FetchContent_Declare(
    NumCpp
    GIT_REPOSITORY https://github.com/dpilger26/NumCpp.git
    GIT_TAG        Version_2.12.1
)
FetchContent_MakeAvailable(NumCpp)

find_package(pybind11 CONFIG REQUIRED)

# Build the Python module
pybind11_add_module(ensemble_backend src/bindings.cpp src/EnsembleFunctions.cpp)

# 2. Maximum optimization flags tailored for your CPU
target_compile_options(ensemble_backend PRIVATE -O3 -march=native -DNDEBUG)

# 3. Tell NumCpp to use OpenBLAS
target_compile_definitions(ensemble_backend PRIVATE NUMCPP_USE_OPENBLAS)

# 4. Link everything up
target_include_directories(ensemble_backend PRIVATE ${OPENBLAS_INCLUDE_DIRS})
target_link_libraries(ensemble_backend PRIVATE 
    NumCpp::NumCpp
    ${OPENBLAS_LIBRARIES}
)

set_target_properties(ensemble_backend PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# # Build the standalone executable
# add_executable(ensembleFunctions src/main.cpp src/EnsembleFunctions.cpp)
# target_link_libraries(ensembleFunctions PRIVATE 
#    NumCpp::NumCpp
# )
