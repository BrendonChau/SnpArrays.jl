# `metal_kernels.jl` on Apple M1 Max

## Environment

Reusing `/tmp/claude-501/metalbench/shapeA.bed`.

| Fact | Value |
| --- | --- |
| GPU | `Apple M1 Max` |
| RAM | `32.000 GiB` |
| Metal.jl | `1.10.1` |
| macOS | `26.6.2` |
| threads | `1` |
| Sys.CPU_NAME | `apple-m1` |
| VERSION | `1.12.7` |
| pkgdir | `/Users/brendonchau/projects/SnpArrays.jl` |
| m, n | `m = 4096, n = 32768` |

## Timings

Centered, scaled, mean-imputed. Times are wall times of `Metal.@sync` calls. The last column is the relative difference from the `mul!` row of the same product.

| Type | Product | k | Variant | min ms | median ms | GFMA/s | rel. diff |
|---|---|---:|---|---:|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 0.841 | 0.858 | 159.59 | 0.0e+00 |
| Float32 | A*x | 1 | MPS dense | 1.778 | 2.352 | 75.47 | 9.8e-07 |
| Float32 | Aᵀ*y | 1 | mul! | 1.051 | 1.222 | 127.73 | 0.0e+00 |
| Float32 | Aᵀ*y | 1 | MPS dense | 1.824 | 1.949 | 73.59 | 1.2e-06 |
| Float32 | A*X | 8 | mul! (tiled) | 1.988 | 2.560 | 540.08 | 0.0e+00 |
| Float32 | A*X | 8 | simdgroup | 4.441 | 4.458 | 241.75 | 1.6e-06 |
| Float32 | A*X | 8 | MPS dense | 2.691 | 3.436 | 398.96 | 1.2e-06 |
| Float32 | Aᵀ*X | 8 | mul! (tiled) | 2.697 | 2.718 | 398.08 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | simdgroup | 4.437 | 4.533 | 242.02 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | MPS dense | 3.880 | 3.915 | 276.77 | 1.2e-06 |
| Float32 | A*X | 32 | mul! (tiled) | 4.363 | 4.456 | 984.33 | 0.0e+00 |
| Float32 | A*X | 32 | simdgroup | 4.630 | 4.753 | 927.64 | 1.5e-06 |
| Float32 | A*X | 32 | MPS dense | 1.715 | 1.793 | 2504.60 | 1.5e-06 |
| Float32 | Aᵀ*X | 32 | mul! (simdgroup) | 4.557 | 4.605 | 942.56 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | tiled | 7.155 | 7.266 | 600.25 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | MPS dense | 3.589 | 3.661 | 1196.81 | 0.0e+00 |
| Float32 | A*X | 128 | mul! (simdgroup) | 9.030 | 9.072 | 1902.43 | 0.0e+00 |
| Float32 | A*X | 128 | tiled | 15.769 | 16.056 | 1089.49 | 0.0e+00 |
| Float32 | A*X | 128 | MPS dense | 5.561 | 5.787 | 3089.51 | 3.1e-06 |
| Float32 | Aᵀ*X | 128 | mul! (simdgroup) | 9.072 | 9.325 | 1893.69 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | tiled | 28.720 | 29.225 | 598.19 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | MPS dense | 7.929 | 7.992 | 2166.60 | 0.0e+00 |

## Peak RSS

1885 MiB
