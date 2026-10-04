# `linalg_kernels.jl` on znver2

## Environment

Reusing `/u/scratch/b/bhchau/.claude-tmp-code-julia-g15/claude-22840/lk.bed`.

| Fact | Value |
| --- | --- |
| SnpArrays | `fork` |
| threads | `32` |
| Sys.CPU_NAME | `znver2` |
| Sys.ARCH | `x86_64` |
| VERSION | `1.12.7` |
| pkgdir | `/u/home/b/bhchau/projects/SnpArrays.jl` |
| m, n | `m = 8192, n = 49152` |
| VECTOR_BYTES | `32` |
| _register_tile_shape(Float32) | `(6, 16)` |
| _register_tile_shape(Float64) | `(6, 8)` |

## Tile sizes

| Type | Product | k | row_step | column_step | rhs_step | register tile |
|---|---|---:|---:|---:|---:|---|
| Float32 | A*x | 1 | 256 | 49152 | 1 | none |
| Float32 | Aᵀ*y | 1 | 4096 | 384 | 1 | none |
| Float32 | A*X | 8 | 256 | 512 | 8 | 6 x 16 |
| Float32 | Aᵀ*X | 8 | 4096 | 384 | 8 | 6 x 16 |
| Float32 | A*X | 32 | 256 | 512 | 32 | 6 x 16 |
| Float32 | Aᵀ*X | 32 | 2048 | 384 | 32 | 6 x 16 |
| Float32 | A*X | 128 | 256 | 512 | 128 | 6 x 16 |
| Float32 | Aᵀ*X | 128 | 512 | 384 | 128 | 6 x 16 |
| Float64 | A*x | 1 | 256 | 49152 | 1 | none |
| Float64 | Aᵀ*y | 1 | 2048 | 384 | 1 | none |
| Float64 | A*X | 8 | 256 | 512 | 8 | 6 x 8 |
| Float64 | Aᵀ*X | 8 | 4096 | 384 | 8 | 6 x 8 |
| Float64 | A*X | 32 | 256 | 512 | 32 | 6 x 8 |
| Float64 | Aᵀ*X | 32 | 1024 | 384 | 32 | 6 x 8 |
| Float64 | A*X | 128 | 256 | 256 | 128 | 6 x 8 |
| Float64 | Aᵀ*X | 128 | 256 | 384 | 128 | 6 x 8 |

## Timings

| Type | Product | k | Variant | min ms | median ms | GFMA/s |
|---|---|---:|---|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 10.888 | 10.980 | 36.98 |
| Float32 | Aᵀ*y | 1 | mul! | 4.330 | 4.540 | 93.00 |
| Float32 | A*X | 8 | mul! (lookup) | 62.261 | 68.459 | 51.74 |
| Float32 | A*X | 8 | tiled MR=1 | 33.807 | 39.104 | 95.28 |
| Float32 | A*X | 8 | tiled MR=2 | 22.087 | 29.644 | 145.84 |
| Float32 | A*X | 8 | tiled MR=4 | 15.978 | 25.958 | 201.60 |
| Float32 | A*X | 8 | tiled MR=6 (default) | 17.201 | 17.917 | 187.27 |
| Float32 | A*X | 8 | tiled MR=8 | 20.125 | 21.493 | 160.06 |
| Float32 | Aᵀ*X | 8 | mul! (register-tiled) | 8.929 | 9.033 | 360.74 |
| Float32 | Aᵀ*X | 8 | tiled MR=1 | 23.323 | 24.127 | 138.11 |
| Float32 | Aᵀ*X | 8 | tiled MR=2 | 10.557 | 10.622 | 305.13 |
| Float32 | Aᵀ*X | 8 | tiled MR=4 | 8.782 | 9.360 | 366.81 |
| Float32 | Aᵀ*X | 8 | tiled MR=6 (default) | 7.550 | 8.218 | 426.67 |
| Float32 | Aᵀ*X | 8 | tiled MR=8 | 7.854 | 8.036 | 410.13 |
| Float32 | A*X | 32 | mul! (lookup) | 89.618 | 95.302 | 143.78 |
| Float32 | A*X | 32 | tiled MR=1 | 95.315 | 100.800 | 135.18 |
| Float32 | A*X | 32 | tiled MR=2 | 74.331 | 76.125 | 173.35 |
| Float32 | A*X | 32 | tiled MR=4 | 70.578 | 78.514 | 182.56 |
| Float32 | A*X | 32 | tiled MR=6 (default) | 77.126 | 77.560 | 167.06 |
| Float32 | A*X | 32 | tiled MR=8 | 84.535 | 86.726 | 152.42 |
| Float32 | Aᵀ*X | 32 | mul! (register-tiled) | 29.976 | 36.063 | 429.84 |
| Float32 | Aᵀ*X | 32 | tiled MR=1 | 43.862 | 45.553 | 293.76 |
| Float32 | Aᵀ*X | 32 | tiled MR=2 | 27.142 | 28.048 | 474.72 |
| Float32 | Aᵀ*X | 32 | tiled MR=4 | 25.266 | 27.654 | 509.96 |
| Float32 | Aᵀ*X | 32 | tiled MR=6 (default) | 30.225 | 36.051 | 426.30 |
| Float32 | Aᵀ*X | 32 | tiled MR=8 | 35.720 | 41.036 | 360.72 |
| Float32 | A*X | 128 | mul! (lookup) | 249.130 | 268.976 | 206.88 |
| Float32 | A*X | 128 | tiled MR=1 | 455.722 | 475.461 | 113.09 |
| Float32 | A*X | 128 | tiled MR=2 | 376.322 | 377.809 | 136.96 |
| Float32 | A*X | 128 | tiled MR=4 | 348.598 | 355.891 | 147.85 |
| Float32 | A*X | 128 | tiled MR=6 (default) | 380.500 | 398.705 | 135.45 |
| Float32 | A*X | 128 | tiled MR=8 | 414.335 | 443.767 | 124.39 |
| Float32 | Aᵀ*X | 128 | mul! (register-tiled) | 185.582 | 196.484 | 277.72 |
| Float32 | Aᵀ*X | 128 | tiled MR=1 | 216.593 | 226.597 | 237.96 |
| Float32 | Aᵀ*X | 128 | tiled MR=2 | 160.847 | 173.039 | 320.43 |
| Float32 | Aᵀ*X | 128 | tiled MR=4 | 156.387 | 165.607 | 329.57 |
| Float32 | Aᵀ*X | 128 | tiled MR=6 (default) | 189.240 | 195.451 | 272.35 |
| Float32 | Aᵀ*X | 128 | tiled MR=8 | 204.596 | 214.447 | 251.91 |
| Float64 | A*x | 1 | mul! | 16.343 | 16.422 | 24.64 |
| Float64 | Aᵀ*y | 1 | mul! | 9.370 | 9.827 | 42.97 |
| Float64 | A*X | 8 | mul! (lookup) | 53.984 | 64.055 | 59.67 |
| Float64 | A*X | 8 | tiled MR=1 | 35.714 | 40.202 | 90.20 |
| Float64 | A*X | 8 | tiled MR=2 | 20.856 | 21.274 | 154.45 |
| Float64 | A*X | 8 | tiled MR=4 | 28.077 | 28.662 | 114.73 |
| Float64 | A*X | 8 | tiled MR=6 (default) | 20.573 | 22.024 | 156.58 |
| Float64 | A*X | 8 | tiled MR=8 | 25.240 | 27.756 | 127.62 |
| Float64 | Aᵀ*X | 8 | mul! (register-tiled) | 12.800 | 13.109 | 251.66 |
| Float64 | Aᵀ*X | 8 | tiled MR=1 | 20.492 | 20.678 | 157.19 |
| Float64 | Aᵀ*X | 8 | tiled MR=2 | 11.442 | 11.893 | 281.53 |
| Float64 | Aᵀ*X | 8 | tiled MR=4 | 9.640 | 9.960 | 334.15 |
| Float64 | Aᵀ*X | 8 | tiled MR=6 (default) | 15.258 | 15.380 | 211.11 |
| Float64 | Aᵀ*X | 8 | tiled MR=8 | 15.100 | 15.365 | 213.32 |
| Float64 | A*X | 32 | mul! (lookup) | 107.786 | 115.329 | 119.54 |
| Float64 | A*X | 32 | tiled MR=1 | 140.266 | 154.244 | 91.86 |
| Float64 | A*X | 32 | tiled MR=2 | 83.638 | 92.672 | 154.06 |
| Float64 | A*X | 32 | tiled MR=4 | 69.615 | 71.017 | 185.09 |
| Float64 | A*X | 32 | tiled MR=6 (default) | 81.209 | 82.209 | 158.66 |
| Float64 | A*X | 32 | tiled MR=8 | 100.748 | 101.341 | 127.89 |
| Float64 | Aᵀ*X | 32 | mul! (register-tiled) | 51.970 | 52.417 | 247.93 |
| Float64 | Aᵀ*X | 32 | tiled MR=1 | 82.781 | 84.528 | 155.65 |
| Float64 | Aᵀ*X | 32 | tiled MR=2 | 46.803 | 46.823 | 275.30 |
| Float64 | Aᵀ*X | 32 | tiled MR=4 | 38.368 | 38.682 | 335.83 |
| Float64 | Aᵀ*X | 32 | tiled MR=6 (default) | 51.970 | 53.081 | 247.93 |
| Float64 | Aᵀ*X | 32 | tiled MR=8 | 60.391 | 60.765 | 213.36 |
| Float64 | A*X | 128 | mul! (lookup) | 381.673 | 397.911 | 135.04 |
| Float64 | A*X | 128 | tiled MR=1 | 429.741 | 432.542 | 119.93 |
| Float64 | A*X | 128 | tiled MR=2 | 307.583 | 309.351 | 167.56 |
| Float64 | A*X | 128 | tiled MR=4 | 243.818 | 248.117 | 211.39 |
| Float64 | A*X | 128 | tiled MR=6 (default) | 288.419 | 293.329 | 178.70 |
| Float64 | A*X | 128 | tiled MR=8 | 397.612 | 407.025 | 129.62 |
| Float64 | Aᵀ*X | 128 | mul! (register-tiled) | 221.400 | 240.859 | 232.79 |
| Float64 | Aᵀ*X | 128 | tiled MR=1 | 341.283 | 345.733 | 151.02 |
| Float64 | Aᵀ*X | 128 | tiled MR=2 | 199.743 | 204.280 | 258.03 |
| Float64 | Aᵀ*X | 128 | tiled MR=4 | 164.320 | 171.640 | 313.65 |
| Float64 | Aᵀ*X | 128 | tiled MR=6 (default) | 220.857 | 223.316 | 233.36 |
| Float64 | Aᵀ*X | 128 | tiled MR=8 | 250.093 | 267.985 | 206.08 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.3167932 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903711 |

## Generated code

Generated code for `Val{6}`, `Val{2}`, `Val{8}`.

| File | Lines with vfmadd or fmla |
| --- | ---: |
| `linalg_kernels_znver2_fork_20261004-021931_AX.s` | 12 |
| `linalg_kernels_znver2_fork_20261004-021931_AtX.s` | 24 |

## Peak RSS

1510 MiB
