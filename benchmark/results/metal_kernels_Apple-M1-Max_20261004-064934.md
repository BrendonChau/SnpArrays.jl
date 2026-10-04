# `metal_kernels.jl` on Apple M1 Max

## Environment

Reusing `/tmp/claude-501/lk_default.bed`.

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
| m, n | `m = 8192, n = 49152` |

## Timings

Centered, scaled, mean-imputed. Times are wall times of `Metal.@sync` calls. The last column is the relative difference from the `mul!` row of the same product.

| Type | Product | k | Variant | min ms | median ms | GFMA/s | rel. diff |
|---|---|---:|---|---:|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 1.456 | 1.471 | 276.47 | 0.0e+00 |
| Float32 | A*x | 1 | MPS dense | 4.561 | 4.711 | 88.28 | 1.7e-06 |
| Float32 | Aᵀ*y | 1 | mul! | 2.040 | 2.086 | 197.40 | 0.0e+00 |
| Float32 | Aᵀ*y | 1 | MPS dense | 4.674 | 4.726 | 86.14 | 1.6e-06 |
| Float32 | A*X | 8 | mul! (tiled) | 5.289 | 5.347 | 608.99 | 0.0e+00 |
| Float32 | A*X | 8 | simdgroup | 12.584 | 12.620 | 255.98 | 2.6e-06 |
| Float32 | A*X | 8 | MPS dense | 15.934 | 16.089 | 202.16 | 2.1e-06 |
| Float32 | Aᵀ*X | 8 | mul! (tiled) | 7.576 | 7.697 | 425.20 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | simdgroup | 12.713 | 12.730 | 253.37 | 0.0e+00 |
| Float32 | Aᵀ*X | 8 | MPS dense | 11.170 | 11.180 | 288.39 | 1.6e-06 |
| Float32 | A*X | 32 | mul! (tiled) | 12.133 | 12.248 | 1062.01 | 0.0e+00 |
| Float32 | A*X | 32 | simdgroup | 12.922 | 12.968 | 997.16 | 2.7e-06 |
| Float32 | A*X | 32 | MPS dense | 4.572 | 4.605 | 2818.35 | 2.7e-06 |
| Float32 | Aᵀ*X | 32 | mul! (simdgroup) | 12.971 | 13.002 | 993.35 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | tiled | 21.011 | 21.431 | 613.25 | 0.0e+00 |
| Float32 | Aᵀ*X | 32 | MPS dense | 14.050 | 14.306 | 917.06 | 0.0e+00 |
| Float32 | A*X | 128 | mul! (simdgroup) | 25.648 | 25.666 | 2009.47 | 0.0e+00 |
| Float32 | A*X | 128 | tiled | 46.716 | 46.751 | 1103.25 | 0.0e+00 |
| Float32 | A*X | 128 | MPS dense | 15.057 | 15.179 | 3422.93 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | mul! (simdgroup) | 26.278 | 26.297 | 1961.31 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | tiled | 85.304 | 86.200 | 604.18 | 0.0e+00 |
| Float32 | Aᵀ*X | 128 | MPS dense | 23.100 | 23.157 | 2231.11 | 0.0e+00 |

## Peak RSS

3145 MiB
