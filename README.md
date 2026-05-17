# Cache-Aware Matrix Optimization in C

A performance project exploring cache-aware optimizations for matrix operations in C, including blocked matrix multiplication and blocked transposition.

## Overview

This project investigates how memory locality and cache behavior affect the real-world performance of matrix operations. While standard matrix multiplication has fixed asymptotic complexity, implementation details such as traversal order, blocking strategy, and memory layout significantly influence runtime efficiency on modern hardware.

The project compares:
- Naive matrix multiplication
- Blocked (tiled) matrix multiplication
- Naive matrix transposition
- Blocked transposition
- Compiler optimization effects

## Features

- Matrix multiplication implemented in C
- Blocked/tiled multiplication for improved cache locality
- Blocked matrix transposition
- Runtime benchmarking utilities
- Configurable matrix sizes and block sizes
- Performance comparison between optimization strategies

---

## Technical Concepts Explored

  - Cache Locality
  - Blocking / Tiling
  - Cache hit frequency
  - Cache reading patterns for matrices
