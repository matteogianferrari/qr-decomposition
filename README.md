# Parallel QR Decomposition

![Status](https://img.shields.io/badge/Status-Completed-green)

Welcome to the "Parallel QR Decomposition" repository! This project is the union of 2 assignments assigned in the High Perfomance Computing course, for my Computer Science master degree. The primary objective is to implement different methods to exploit caches usage, the multicore CPU and the GPU architectures, on the Gram-Schmidt QR Decomposition algorithm and measure the performance of the different implementations.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction

This project focuses on a critical aspect of High Performance Computing – the comprehension of the theory behind the usage of caches, the parallelization techniques applied on the multicore CPU, the parallelization offloading the computation on a target GPU, and the parallelization on the GPU using CUDA programming.

The software was developed on a Linux platform using Visual Studio Code as the integrated development environment and CMake as the build system.

## Installation

To use the software in this repository, you can simply clone the repository to your local machine:

```bash
git clone https://github.com/matteogianferrari/ParallelQRDecomposition.git
```

## Usage

To build and run the software, make sure you have the following dependencies installed on your Linux machine:

- C++ compiler (my version: gcc (Debian 12.2.0-14) 12.2.0; includes OpenMP)
- CMake (my version: cmake 3.25.1) 
- Make (my version: GNU Make 4.3)
- CUDA Toolkit (my version: release 12.3, V12.3.52)

Then, follow these steps:

 1. Navigate to the root directory of the repository.
 2. Create a build directory: `mkdir build && cd build`.
 3. Generate the build files using CMake: `cmake -DCMAKE_BUILD_TYPE=Release ..`.
 4. Build the software using Make: `make`.
 5. Run the software: `./cluster_extraction <dataset_directory_path>`.

Feel free to explore the code and experiment with it in your preferred development environment.