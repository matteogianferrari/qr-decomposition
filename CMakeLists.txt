cmake_minimum_required(VERSION 3.18.0)

# Defines the project name and the version, and specify the languages used in the project.
project(qrd VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu" "src/*.c")

# Add executable
add_executable(qrd main.cpp ${SOURCES})

# Specify the CUDA architectures
set_target_properties(qrd PROPERTIES CUDA_ARCHITECTURES "89")

# Link CUDA libraries
target_link_libraries(qrd PUBLIC ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})

# Attempt to find the OpenMP package.
find_package(OpenMP)
if(OpenMP_C_FOUND)
    # Link the OpenMP libraries for the C language to the 'qrd' executable.
    target_link_libraries(qrd PUBLIC OpenMP::OpenMP_C -lm -fopenmp -foffload=nvptx-none -foffload=-lm)
endif()
