# `cuda_kernels.jl` on NVIDIA RTX A6000

## Environment

Reusing `/u/scratch/b/bhchau/tiled_work/lk.bed`.

| Fact | Value |
| --- | --- |
| GPU | `NVIDIA RTX A6000` |
| GPU memory | `47.529 GiB` |
| CUDA.jl | `6.4.2` |
| CUDA runtime | `12.4.0` |
| threads | `8` |
| Sys.CPU_NAME | `znver3` |
| VERSION | `1.12.7` |
| pkgdir | `/u/home/b/bhchau/projects/SnpArrays.jl` |
| m, n | `m = 8192, n = 49152` |

## Timings

Centered, scaled, mean-imputed. The last column is the relative difference from the `mul!` row of the same product.

| Type | Product | k | Variant | min ms | median ms | GFMA/s | rel. diff |
|---|---|---:|---|---:|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 0.346 | 0.349 | 1163.68 | 0.0e+00 |
| Float32 | Aᵀ*y | 1 | mul! | 0.382 | 0.384 | 1054.82 | 0.0e+00 |
| Float32 | A*X | 8 | mul! (tiled) | 0.916 | 0.926 | 3515.39 | 0.0e+00 |
| Float32 | A*X | 8 | CuBLAS dense | 2.319 | 2.339 | 1388.84 | 1.2e-06 |
| Float32 | A*X | 8 | CuBLAS dense TF32 | 2.356 | 2.371 | 1367.11 | 3.0e-04 |
| Float32 | Aᵀ*X | 8 | mul! (tensor-core) | 1.106 | 1.108 | 2912.71 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense | 2.421 | 2.431 | 1330.68 | 6.6e-07 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense TF32 | 2.350 | 2.369 | 1370.69 | 2.9e-04 |
| Float32 | A*X | 32 | mul! (tensor-core) | 1.306 | 1.386 | 9868.95 | 0.0e+00 |
| Float32 | A*X | 32 | tiled | 1.819 | 1.824 | 7084.97 | 9.3e-07 |
| Float32 | A*X | 32 | CuBLAS dense | 2.360 | 2.365 | 5458.96 | 9.5e-07 |
| Float32 | A*X | 32 | CuBLAS dense TF32 | 2.388 | 2.395 | 5395.76 | 3.1e-04 |
| Float32 | Aᵀ*X | 32 | mul! (tensor-core) | 1.211 | 1.238 | 10637.85 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense | 2.551 | 2.560 | 5051.35 | 4.9e-07 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense TF32 | 2.382 | 2.396 | 5409.75 | 2.9e-04 |
| Float32 | A*X | 128 | mul! (tensor-core) | 3.692 | 3.731 | 13961.62 | 0.0e+00 |
| Float32 | A*X | 128 | tiled | 8.539 | 8.647 | 6035.74 | 1.7e-06 |
| Float32 | A*X | 128 | CuBLAS dense | 4.981 | 5.004 | 10347.79 | 1.5e-06 |
| Float32 | A*X | 128 | CuBLAS dense TF32 | 2.380 | 2.425 | 21657.34 | 3.4e-04 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 3.359 | 3.373 | 15345.01 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense | 4.937 | 4.958 | 10438.53 | 6.4e-07 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense TF32 | 2.434 | 2.442 | 21174.44 | 2.9e-04 |
| Float64 | A*x | 1 | mul! | 1.720 | 1.727 | 234.06 | 0.0e+00 |
| Float64 | Aᵀ*y | 1 | mul! | 1.540 | 1.543 | 261.45 | 0.0e+00 |
| Float64 | A*X | 8 | mul! (tiled) | 10.428 | 10.483 | 308.89 | 0.0e+00 |
| Float64 | A*X | 8 | CuBLAS dense | 53.903 | 57.020 | 59.76 | 3.8e-15 |
| Float64 | Aᵀ*X | 8 | mul! (tiled) | 12.757 | 12.763 | 252.51 | 0.0e+00 |
| Float64 | Aᵀ*X | 8 | CuBLAS dense | 48.764 | 48.794 | 66.06 | 2.8e-15 |
| Float64 | A*X | 32 | mul! (tiled) | 41.118 | 41.134 | 313.37 | 0.0e+00 |
| Float64 | A*X | 32 | CuBLAS dense | 41.278 | 41.314 | 312.15 | 2.7e-15 |
| Float64 | Aᵀ*X | 32 | mul! (tiled) | 44.761 | 44.782 | 287.86 | 0.0e+00 |
| Float64 | Aᵀ*X | 32 | CuBLAS dense | 51.329 | 51.333 | 251.03 | 2.9e-15 |
| Float64 | A*X | 128 | mul! (tiled) | 178.541 | 178.642 | 288.67 | 0.0e+00 |
| Float64 | A*X | 128 | CuBLAS dense | 200.048 | 200.051 | 257.64 | 3.8e-15 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 178.432 | 178.469 | 288.85 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | CuBLAS dense | 189.898 | 189.907 | 271.41 | 0.0e+00 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.31679308 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903711 |

## Peak RSS

4820 MiB
