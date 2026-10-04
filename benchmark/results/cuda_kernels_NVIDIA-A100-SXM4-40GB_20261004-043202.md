# `cuda_kernels.jl` on NVIDIA A100-SXM4-40GB

## Environment

Reusing `/u/scratch/b/bhchau/.claude-tmp-code-julia-g15/claude-22840/lk.bed`.

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
| Float32 | A*x | 1 | mul! | 0.468 | 0.470 | 860.43 | 0.0e+00 |
| Float32 | Aᵀ*y | 1 | mul! | 0.522 | 0.523 | 771.01 | 0.0e+00 |
| Float32 | A*X | 8 | mul! (tiled) | 1.014 | 1.050 | 3177.50 | 0.0e+00 |
| Float32 | A*X | 8 | CuBLAS dense | 1.325 | 1.335 | 2431.01 | 1.4e-06 |
| Float32 | A*X | 8 | CuBLAS dense TF32 | 1.187 | 1.195 | 2714.17 | 3.1e-04 |
| Float32 | Aᵀ*X | 8 | mul! (tensor-core) | 0.939 | 1.023 | 3430.46 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense | 2.261 | 2.266 | 1424.70 | 1.2e-06 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense TF32 | 1.238 | 1.248 | 2601.93 | 2.9e-04 |
| Float32 | A*X | 32 | mul! (tensor-core) | 1.092 | 1.111 | 11803.86 | 0.0e+00 |
| Float32 | A*X | 32 | tiled | 2.554 | 2.641 | 5045.27 | 8.4e-07 |
| Float32 | A*X | 32 | CuBLAS dense | 2.008 | 2.008 | 6416.58 | 1.2e-06 |
| Float32 | A*X | 32 | CuBLAS dense TF32 | 1.177 | 1.184 | 10951.19 | 3.1e-04 |
| Float32 | Aᵀ*X | 32 | mul! (tensor-core) | 1.085 | 1.138 | 11870.67 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense | 2.286 | 2.288 | 5637.51 | 1.6e-06 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense TF32 | 1.220 | 1.221 | 10565.00 | 2.9e-04 |
| Float32 | A*X | 128 | mul! (tensor-core) | 3.383 | 3.400 | 15233.55 | 0.0e+00 |
| Float32 | A*X | 128 | tiled | 11.206 | 11.265 | 4599.44 | 1.5e-06 |
| Float32 | A*X | 128 | CuBLAS dense | 7.858 | 7.887 | 6558.72 | 2.0e-06 |
| Float32 | A*X | 128 | CuBLAS dense TF32 | 1.299 | 1.313 | 39662.45 | 3.0e-04 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 3.288 | 3.302 | 15674.76 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense | 6.196 | 6.200 | 8317.91 | 1.6e-06 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense TF32 | 1.293 | 1.294 | 39850.87 | 2.9e-04 |
| Float64 | A*x | 1 | mul! | 0.565 | 0.570 | 712.35 | 0.0e+00 |
| Float64 | Aᵀ*y | 1 | mul! | 0.749 | 0.750 | 537.92 | 0.0e+00 |
| Float64 | A*X | 8 | mul! (tiled) | 1.217 | 1.218 | 2647.92 | 0.0e+00 |
| Float64 | A*X | 8 | CuBLAS dense | 2.324 | 2.328 | 1385.78 | 3.5e-15 |
| Float64 | Aᵀ*X | 8 | mul! (tiled) | 1.810 | 1.814 | 1779.26 | 0.0e+00 |
| Float64 | Aᵀ*X | 8 | CuBLAS dense | 2.405 | 2.419 | 1339.18 | 2.8e-15 |
| Float64 | A*X | 32 | mul! (tiled) | 3.737 | 3.767 | 3448.32 | 0.0e+00 |
| Float64 | A*X | 32 | CuBLAS dense | 2.380 | 2.382 | 5414.33 | 3.5e-15 |
| Float64 | Aᵀ*X | 32 | mul! (tiled) | 3.680 | 3.686 | 3501.09 | 0.0e+00 |
| Float64 | Aᵀ*X | 32 | CuBLAS dense | 2.901 | 2.907 | 4441.55 | 1.9e-15 |
| Float64 | A*X | 128 | mul! (tiled) | 18.317 | 18.323 | 2813.71 | 0.0e+00 |
| Float64 | A*X | 128 | CuBLAS dense | 5.537 | 5.538 | 9308.61 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 16.696 | 16.843 | 3086.88 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | CuBLAS dense | 5.725 | 5.730 | 9002.26 | 2.9e-15 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.31679308 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903711 |

## Peak RSS

32693 MiB
