cmake_minimum_required(VERSION 3.14)
project(NumCppProject LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# Setting C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
target_link_libraries(ensemble_backend PRIVATE 
    NumCpp::NumCpp
)

# Add debug symbols, ASan, and keep frame pointers for readable stack traces
target_compile_options(ensemble_backend PRIVATE -g -fsanitize=address -fno-omit-frame-pointer)
# Tell the linker to use ASan
target_link_options(ensemble_backend PRIVATE -fsanitize=address)

set_target_properties(ensemble_backend PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# # Build the standalone executable
# add_executable(ensembleFunctions src/main.cpp src/EnsembleFunctions.cpp)
# target_link_libraries(ensembleFunctions PRIVATE 
#     NumCpp::NumCpp
# )
