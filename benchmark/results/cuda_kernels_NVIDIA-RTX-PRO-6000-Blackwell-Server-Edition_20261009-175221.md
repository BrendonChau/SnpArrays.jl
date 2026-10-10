# `cuda_kernels.jl` on NVIDIA RTX PRO 6000 Blackwell Server Edition

## Environment

Simulated `/u/scratch/b/bhchau/tiled_work/lk.bed`.

| Fact | Value |
| --- | --- |
| GPU | `NVIDIA RTX PRO 6000 Blackwell Server Edition` |
| GPU memory | `94.972 GiB` |
| CUDA.jl | `6.4.2` |
| CUDA runtime | `13.4.0` |
| threads | `16` |
| Sys.CPU_NAME | `znver4` |
| VERSION | `1.12.7` |
| pkgdir | `/u/home/b/bhchau/projects/SnpArrays.jl` |
| m, n | `m = 8192, n = 49152` |

## Timings

Centered, scaled, mean-imputed. The last column is the relative difference from the `mul!` row of the same product.

| Type | Product | k | Variant | min ms | median ms | GFMA/s | rel. diff |
|---|---|---:|---|---:|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 0.130 | 0.133 | 3094.67 | 0.0e+00 |
| Float32 | Aᵀ*y | 1 | mul! | 0.144 | 0.144 | 2802.43 | 0.0e+00 |
| Float32 | A*X | 8 | mul! (tiled) | 0.317 | 0.353 | 10160.83 | 0.0e+00 |
| Float32 | A*X | 8 | CuBLAS dense | 1.097 | 1.100 | 2935.30 | 6.7e-07 |
| Float32 | A*X | 8 | CuBLAS dense TF32 | 1.089 | 1.090 | 2957.99 | 3.0e-04 |
| Float32 | Aᵀ*X | 8 | mul! (tensor-core) | 0.450 | 0.500 | 7161.59 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense | 1.098 | 1.104 | 2933.17 | 2.8e-07 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense TF32 | 1.110 | 1.112 | 2902.97 | 2.9e-04 |
| Float32 | A*X | 32 | mul! (tensor-core) | 0.569 | 0.767 | 22654.06 | 0.0e+00 |
| Float32 | A*X | 32 | tiled | 0.715 | 0.723 | 18010.97 | 6.5e-07 |
| Float32 | A*X | 32 | CuBLAS dense | 1.110 | 1.113 | 11610.53 | 1.1e-06 |
| Float32 | A*X | 32 | CuBLAS dense TF32 | 1.097 | 1.103 | 11748.75 | 3.1e-04 |
| Float32 | Aᵀ*X | 32 | mul! (tensor-core) | 0.526 | 0.529 | 24511.67 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense | 1.625 | 1.626 | 7930.46 | 4.7e-07 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense TF32 | 1.118 | 1.123 | 11528.75 | 2.9e-04 |
| Float32 | A*X | 128 | mul! (tensor-core) | 1.550 | 2.004 | 33246.21 | 0.0e+00 |
| Float32 | A*X | 128 | tiled | 3.779 | 3.875 | 13639.67 | 1.2e-06 |
| Float32 | A*X | 128 | CuBLAS dense | 1.461 | 1.464 | 35269.41 | 1.0e-06 |
| Float32 | A*X | 128 | CuBLAS dense TF32 | 1.122 | 1.126 | 45938.76 | 3.0e-04 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 1.383 | 1.391 | 37263.72 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense | 2.153 | 2.159 | 23942.15 | 9.6e-07 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense TF32 | 1.122 | 1.126 | 45920.42 | 2.9e-04 |
| Float64 | A*x | 1 | mul! | 0.685 | 0.689 | 587.99 | 0.0e+00 |
| Float64 | Aᵀ*y | 1 | mul! | 0.579 | 0.581 | 694.96 | 0.0e+00 |
| Float64 | A*X | 8 | mul! (tiled) | 3.854 | 3.907 | 835.75 | 0.0e+00 |
| Float64 | A*X | 8 | CuBLAS dense | 3.845 | 3.848 | 837.86 | 1.2e-15 |
| Float64 | Aᵀ*X | 8 | mul! (tiled) | 5.004 | 5.007 | 643.76 | 0.0e+00 |
| Float64 | Aᵀ*X | 8 | CuBLAS dense | 19.203 | 19.219 | 167.75 | 2.2e-15 |
| Float64 | A*X | 32 | mul! (tiled) | 15.377 | 15.420 | 837.95 | 0.0e+00 |
| Float64 | A*X | 32 | CuBLAS dense | 15.202 | 15.213 | 847.59 | 1.4e-15 |
| Float64 | Aᵀ*X | 32 | mul! (tiled) | 16.465 | 16.514 | 782.54 | 0.0e+00 |
| Float64 | Aᵀ*X | 32 | CuBLAS dense | 20.611 | 24.674 | 625.15 | 3.1e-15 |
| Float64 | A*X | 128 | mul! (tiled) | 65.520 | 65.574 | 786.63 | 0.0e+00 |
| Float64 | A*X | 128 | CuBLAS dense | 73.963 | 73.968 | 696.83 | 5.5e-15 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 65.944 | 65.986 | 781.57 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | CuBLAS dense | 82.444 | 82.455 | 625.14 | 2.9e-15 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.31679308 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903707 |

## Peak RSS

5587 MiB
