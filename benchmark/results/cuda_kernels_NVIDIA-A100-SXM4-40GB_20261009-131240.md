# `cuda_kernels.jl` on NVIDIA A100-SXM4-40GB

## Environment

Simulated `/u/scratch/b/bhchau/.claude-tmp-code-julia-g15/claude-22840/lk.bed`.

| Fact | Value |
| --- | --- |
| GPU | `NVIDIA A100-SXM4-40GB` |
| GPU memory | `39.494 GiB` |
| CUDA.jl | `6.4.0` |
| CUDA runtime | `13.4.0` |
| threads | `16` |
| Sys.CPU_NAME | `znver2` |
| VERSION | `1.12.7` |
| pkgdir | `/u/home/b/bhchau/projects/SnpArrays.jl` |
| m, n | `m = 8192, n = 49152` |

## Timings

Centered, scaled, mean-imputed. The last column is the relative difference from the `mul!` row of the same product.

| Type | Product | k | Variant | min ms | median ms | GFMA/s | rel. diff |
|---|---|---:|---|---:|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 0.452 | 0.455 | 891.65 | 0.0e+00 |
| Float32 | Aᵀ*y | 1 | mul! | 0.523 | 0.525 | 769.50 | 0.0e+00 |
| Float32 | A*X | 8 | mul! (tiled) | 0.992 | 1.035 | 3246.37 | 0.0e+00 |
| Float32 | A*X | 8 | CuBLAS dense | 1.289 | 1.303 | 2498.59 | 1.4e-06 |
| Float32 | A*X | 8 | CuBLAS dense TF32 | 1.160 | 1.162 | 2776.46 | 3.1e-04 |
| Float32 | Aᵀ*X | 8 | mul! (tensor-core) | 0.882 | 0.972 | 3653.57 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense | 2.355 | 2.362 | 1367.71 | 1.2e-06 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense TF32 | 1.225 | 1.229 | 2630.21 | 2.9e-04 |
| Float32 | A*X | 32 | mul! (tensor-core) | 0.966 | 1.052 | 13343.49 | 0.0e+00 |
| Float32 | A*X | 32 | tiled | 2.534 | 2.535 | 5084.00 | 8.4e-07 |
| Float32 | A*X | 32 | CuBLAS dense | 2.004 | 2.006 | 6429.69 | 1.2e-06 |
| Float32 | A*X | 32 | CuBLAS dense TF32 | 1.178 | 1.182 | 10941.66 | 3.1e-04 |
| Float32 | Aᵀ*X | 32 | mul! (tensor-core) | 0.929 | 1.025 | 13873.11 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense | 2.322 | 2.328 | 5548.02 | 1.6e-06 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense TF32 | 1.192 | 1.205 | 10810.06 | 2.9e-04 |
| Float32 | A*X | 128 | mul! (tensor-core) | 3.144 | 3.262 | 16394.67 | 0.0e+00 |
| Float32 | A*X | 128 | tiled | 11.183 | 11.198 | 4608.70 | 1.5e-06 |
| Float32 | A*X | 128 | CuBLAS dense | 7.864 | 7.879 | 6553.60 | 2.0e-06 |
| Float32 | A*X | 128 | CuBLAS dense TF32 | 1.337 | 1.343 | 38538.78 | 3.0e-04 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 2.308 | 2.310 | 22329.92 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense | 6.254 | 6.262 | 8241.63 | 1.6e-06 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense TF32 | 1.271 | 1.272 | 40557.33 | 2.9e-04 |
| Float64 | A*x | 1 | mul! | 0.557 | 0.559 | 722.82 | 0.0e+00 |
| Float64 | Aᵀ*y | 1 | mul! | 0.762 | 0.765 | 528.52 | 0.0e+00 |
| Float64 | A*X | 8 | mul! (tiled) | 1.530 | 1.533 | 2105.57 | 0.0e+00 |
| Float64 | A*X | 8 | CuBLAS dense | 2.299 | 2.311 | 1401.22 | 3.5e-15 |
| Float64 | Aᵀ*X | 8 | mul! (tiled) | 2.291 | 2.294 | 1406.23 | 0.0e+00 |
| Float64 | Aᵀ*X | 8 | CuBLAS dense | 2.445 | 2.451 | 1317.31 | 2.8e-15 |
| Float64 | A*X | 32 | mul! (tiled) | 5.107 | 5.158 | 2523.14 | 0.0e+00 |
| Float64 | A*X | 32 | CuBLAS dense | 2.390 | 2.396 | 5391.14 | 3.5e-15 |
| Float64 | Aᵀ*X | 32 | mul! (tiled) | 4.671 | 4.674 | 2758.20 | 0.0e+00 |
| Float64 | Aᵀ*X | 32 | CuBLAS dense | 2.902 | 2.914 | 4439.98 | 1.9e-15 |
| Float64 | A*X | 128 | mul! (tiled) | 22.365 | 22.400 | 2304.46 | 0.0e+00 |
| Float64 | A*X | 128 | CuBLAS dense | 5.497 | 5.500 | 9376.24 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 16.590 | 16.687 | 3106.70 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | CuBLAS dense | 5.663 | 5.699 | 9101.56 | 2.9e-15 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.31679308 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903711 |

## Peak RSS

5335 MiB
