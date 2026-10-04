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
| Float32 | A*x | 1 | mul! | 0.240 | 0.241 | 1680.41 | 0.0e+00 |
| Float32 | A*x | 1 | direct | 0.468 | 0.470 | 860.43 | 6.0e-07 |
| Float32 | Aᵀ*y | 1 | mul! | 0.523 | 0.525 | 769.50 | 0.0e+00 |
| Float32 | A*X | 8 | mul! (lookup) | 1.258 | 1.316 | 2559.58 | 0.0e+00 |
| Float32 | A*X | 8 | tiled | 1.046 | 1.052 | 3081.03 | 8.3e-07 |
| Float32 | A*X | 8 | CuBLAS dense | 1.339 | 1.342 | 2404.99 | 1.3e-06 |
| Float32 | A*X | 8 | CuBLAS dense TF32 | 1.202 | 1.204 | 2679.50 | 3.1e-04 |
| Float32 | Aᵀ*X | 8 | mul! (tensor-core) | 0.940 | 1.040 | 3426.72 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense | 2.190 | 2.252 | 1470.65 | 1.2e-06 |
| Float32 | Aᵀ*X | 8 | CuBLAS dense TF32 | 1.238 | 1.240 | 2601.93 | 2.9e-04 |
| Float32 | A*X | 32 | mul! (tensor-core) | 1.118 | 1.121 | 11522.81 | 0.0e+00 |
| Float32 | A*X | 32 | lookup | 3.107 | 3.178 | 4147.30 | 4.4e-07 |
| Float32 | A*X | 32 | tiled | 2.561 | 2.644 | 5031.15 | 8.4e-07 |
| Float32 | A*X | 32 | CuBLAS dense | 2.009 | 2.010 | 6413.31 | 1.2e-06 |
| Float32 | A*X | 32 | CuBLAS dense TF32 | 1.175 | 1.185 | 10970.28 | 3.1e-04 |
| Float32 | Aᵀ*X | 32 | mul! (tensor-core) | 1.078 | 1.140 | 11949.58 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense | 2.285 | 2.286 | 5640.03 | 1.6e-06 |
| Float32 | Aᵀ*X | 32 | CuBLAS dense TF32 | 1.213 | 1.224 | 10618.49 | 2.9e-04 |
| Float32 | A*X | 128 | mul! (tensor-core) | 3.397 | 3.406 | 15173.85 | 0.0e+00 |
| Float32 | A*X | 128 | lookup | 10.743 | 10.774 | 4797.60 | 8.3e-07 |
| Float32 | A*X | 128 | tiled | 9.104 | 9.122 | 5660.97 | 1.5e-06 |
| Float32 | A*X | 128 | CuBLAS dense | 6.219 | 6.226 | 8287.77 | 2.0e-06 |
| Float32 | A*X | 128 | CuBLAS dense TF32 | 1.314 | 1.337 | 39229.66 | 3.0e-04 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 2.642 | 2.656 | 19508.39 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense | 6.095 | 6.141 | 8456.26 | 1.6e-06 |
| Float32 | Aᵀ*X | 128 | CuBLAS dense TF32 | 1.292 | 1.296 | 39882.45 | 2.9e-04 |
| Float64 | A*x | 1 | mul! | 0.329 | 0.332 | 1224.97 | 0.0e+00 |
| Float64 | A*x | 1 | direct | 0.550 | 0.552 | 732.25 | 1.1e-15 |
| Float64 | Aᵀ*y | 1 | mul! | 0.746 | 0.748 | 539.39 | 0.0e+00 |
| Float64 | A*X | 8 | mul! (lookup) | 1.745 | 1.752 | 1846.08 | 0.0e+00 |
| Float64 | A*X | 8 | tiled | 1.203 | 1.215 | 2677.22 | 1.5e-15 |
| Float64 | A*X | 8 | CuBLAS dense | 2.297 | 2.301 | 1402.46 | 3.3e-15 |
| Float64 | Aᵀ*X | 8 | mul! (tiled) | 1.806 | 1.808 | 1783.29 | 0.0e+00 |
| Float64 | Aᵀ*X | 8 | CuBLAS dense | 2.373 | 2.379 | 1357.67 | 2.8e-15 |
| Float64 | A*X | 32 | mul! (lookup) | 6.009 | 6.055 | 2144.33 | 0.0e+00 |
| Float64 | A*X | 32 | tiled | 3.740 | 3.747 | 3445.49 | 1.6e-15 |
| Float64 | A*X | 32 | CuBLAS dense | 2.344 | 2.386 | 5497.12 | 3.3e-15 |
| Float64 | Aᵀ*X | 32 | mul! (tiled) | 3.678 | 3.684 | 3503.04 | 0.0e+00 |
| Float64 | Aᵀ*X | 32 | CuBLAS dense | 2.888 | 2.889 | 4462.03 | 1.9e-15 |
| Float64 | A*X | 128 | mul! (lookup) | 26.110 | 26.123 | 1973.94 | 0.0e+00 |
| Float64 | A*X | 128 | tiled | 18.340 | 18.484 | 2810.25 | 3.3e-15 |
| Float64 | A*X | 128 | CuBLAS dense | 5.504 | 5.507 | 9364.03 | 3.3e-15 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 16.381 | 16.480 | 3146.32 | 0.0e+00 |
| Float64 | Aᵀ*X | 128 | CuBLAS dense | 5.675 | 5.676 | 9081.86 | 2.9e-15 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.31679308 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903711 |

## Peak RSS

8650 MiB
