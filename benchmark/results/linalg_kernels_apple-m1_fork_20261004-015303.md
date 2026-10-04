# `linalg_kernels.jl` on apple-m1

## Environment

Reusing `/tmp/claude-501/lk_default.bed`.

| Fact | Value |
| --- | --- |
| SnpArrays | `fork` |
| threads | `4` |
| Sys.CPU_NAME | `apple-m1` |
| Sys.ARCH | `aarch64` |
| VERSION | `1.12.7` |
| pkgdir | `/Users/brendonchau/projects/SnpArrays.jl` |
| m, n | `m = 8192, n = 49152` |
| VECTOR_BYTES | `32` |
| _register_tile_shape(Float32) | `(6, 16)` |
| _register_tile_shape(Float64) | `(6, 8)` |

## Tile sizes

| Type | Product | k | row_step | column_step | rhs_step | register tile |
|---|---|---:|---:|---:|---:|---|
| Float32 | A*x | 1 | 512 | 49152 | 1 | none |
| Float32 | Aᵀ*y | 1 | 4096 | 2048 | 1 | none |
| Float32 | A*X | 8 | 512 | 512 | 8 | 6 x 16 |
| Float32 | Aᵀ*X | 8 | 4096 | 2048 | 8 | 6 x 16 |
| Float32 | A*X | 32 | 512 | 512 | 32 | 6 x 16 |
| Float32 | Aᵀ*X | 32 | 2048 | 2048 | 32 | 6 x 16 |
| Float32 | A*X | 128 | 512 | 512 | 128 | 6 x 16 |
| Float32 | Aᵀ*X | 128 | 512 | 2048 | 128 | 6 x 16 |
| Float64 | A*x | 1 | 512 | 49152 | 1 | none |
| Float64 | Aᵀ*y | 1 | 2048 | 2048 | 1 | none |
| Float64 | A*X | 8 | 512 | 512 | 8 | 6 x 8 |
| Float64 | Aᵀ*X | 8 | 4096 | 2048 | 8 | 6 x 8 |
| Float64 | A*X | 32 | 512 | 512 | 32 | 6 x 8 |
| Float64 | Aᵀ*X | 32 | 1024 | 2048 | 32 | 6 x 8 |
| Float64 | A*X | 128 | 512 | 256 | 128 | 6 x 8 |
| Float64 | Aᵀ*X | 128 | 256 | 2048 | 128 | 6 x 8 |

## Timings

| Type | Product | k | Variant | min ms | median ms | GFMA/s |
|---|---|---:|---|---:|---:|---:|
| Float32 | A*x | 1 | mul! | 32.729 | 33.727 | 12.30 |
| Float32 | Aᵀ*y | 1 | mul! | 18.782 | 19.060 | 21.44 |
| Float32 | A*X | 8 | mul! (lookup) | 73.005 | 76.111 | 44.12 |
| Float32 | A*X | 8 | tiled MR=1 | 131.247 | 132.233 | 24.54 |
| Float32 | A*X | 8 | tiled MR=2 | 71.047 | 71.708 | 45.34 |
| Float32 | A*X | 8 | tiled MR=4 | 53.030 | 53.192 | 60.74 |
| Float32 | A*X | 8 | tiled MR=6 (default) | 42.682 | 43.651 | 75.47 |
| Float32 | A*X | 8 | tiled MR=8 | 40.087 | 40.258 | 80.36 |
| Float32 | Aᵀ*X | 8 | mul! (register-tiled) | 38.678 | 39.293 | 83.28 |
| Float32 | Aᵀ*X | 8 | tiled MR=1 | 131.113 | 131.545 | 24.57 |
| Float32 | Aᵀ*X | 8 | tiled MR=2 | 66.474 | 67.012 | 48.46 |
| Float32 | Aᵀ*X | 8 | tiled MR=4 | 42.024 | 42.409 | 76.65 |
| Float32 | Aᵀ*X | 8 | tiled MR=6 (default) | 38.738 | 39.497 | 83.16 |
| Float32 | Aᵀ*X | 8 | tiled MR=8 | 37.909 | 38.094 | 84.97 |
| Float32 | A*X | 32 | mul! (lookup) | 113.897 | 115.824 | 113.13 |
| Float32 | A*X | 32 | tiled MR=1 | 281.772 | 292.287 | 45.73 |
| Float32 | A*X | 32 | tiled MR=2 | 168.485 | 170.359 | 76.48 |
| Float32 | A*X | 32 | tiled MR=4 | 141.207 | 143.045 | 91.25 |
| Float32 | A*X | 32 | tiled MR=6 (default) | 131.365 | 131.805 | 98.08 |
| Float32 | A*X | 32 | tiled MR=8 | 193.512 | 195.446 | 66.58 |
| Float32 | Aᵀ*X | 32 | mul! (register-tiled) | 102.029 | 102.728 | 126.29 |
| Float32 | Aᵀ*X | 32 | tiled MR=1 | 267.327 | 267.946 | 48.20 |
| Float32 | Aᵀ*X | 32 | tiled MR=2 | 142.738 | 143.170 | 90.27 |
| Float32 | Aᵀ*X | 32 | tiled MR=4 | 109.213 | 109.507 | 117.98 |
| Float32 | Aᵀ*X | 32 | tiled MR=6 (default) | 102.043 | 102.340 | 126.27 |
| Float32 | Aᵀ*X | 32 | tiled MR=8 | 121.270 | 122.395 | 106.25 |
| Float32 | A*X | 128 | mul! (lookup) | 474.112 | 476.048 | 108.71 |
| Float32 | A*X | 128 | tiled MR=1 | 1125.729 | 1135.164 | 45.78 |
| Float32 | A*X | 128 | tiled MR=2 | 676.090 | 679.003 | 76.23 |
| Float32 | A*X | 128 | tiled MR=4 | 558.397 | 563.498 | 92.30 |
| Float32 | A*X | 128 | tiled MR=6 (default) | 513.147 | 527.053 | 100.44 |
| Float32 | A*X | 128 | tiled MR=8 | 778.053 | 779.029 | 66.24 |
| Float32 | Aᵀ*X | 128 | mul! (register-tiled) | 483.953 | 487.084 | 106.50 |
| Float32 | Aᵀ*X | 128 | tiled MR=1 | 1091.372 | 1092.140 | 47.22 |
| Float32 | Aᵀ*X | 128 | tiled MR=2 | 609.332 | 611.143 | 84.58 |
| Float32 | Aᵀ*X | 128 | tiled MR=4 | 500.974 | 501.960 | 102.88 |
| Float32 | Aᵀ*X | 128 | tiled MR=6 (default) | 485.303 | 486.956 | 106.20 |
| Float32 | Aᵀ*X | 128 | tiled MR=8 | 581.985 | 582.724 | 88.56 |
| Float64 | A*x | 1 | mul! | 49.162 | 49.520 | 8.19 |
| Float64 | Aᵀ*y | 1 | mul! | 37.495 | 37.628 | 10.74 |
| Float64 | A*X | 8 | mul! (lookup) | 73.449 | 74.149 | 43.86 |
| Float64 | A*X | 8 | tiled MR=1 | 134.262 | 137.458 | 23.99 |
| Float64 | A*X | 8 | tiled MR=2 | 85.349 | 88.771 | 37.74 |
| Float64 | A*X | 8 | tiled MR=4 | 61.389 | 62.435 | 52.47 |
| Float64 | A*X | 8 | tiled MR=6 (default) | 56.302 | 56.639 | 57.21 |
| Float64 | A*X | 8 | tiled MR=8 | 84.165 | 84.672 | 38.27 |
| Float64 | Aᵀ*X | 8 | mul! (register-tiled) | 47.472 | 47.729 | 67.86 |
| Float64 | Aᵀ*X | 8 | tiled MR=1 | 131.827 | 133.058 | 24.44 |
| Float64 | Aᵀ*X | 8 | tiled MR=2 | 69.541 | 70.009 | 46.32 |
| Float64 | Aᵀ*X | 8 | tiled MR=4 | 51.345 | 51.466 | 62.74 |
| Float64 | Aᵀ*X | 8 | tiled MR=6 (default) | 47.209 | 47.708 | 68.23 |
| Float64 | Aᵀ*X | 8 | tiled MR=8 | 62.669 | 63.436 | 51.40 |
| Float64 | A*X | 32 | mul! (lookup) | 213.102 | 215.242 | 60.46 |
| Float64 | A*X | 32 | tiled MR=1 | 536.683 | 546.400 | 24.01 |
| Float64 | A*X | 32 | tiled MR=2 | 348.682 | 356.197 | 36.95 |
| Float64 | A*X | 32 | tiled MR=4 | 236.458 | 238.102 | 54.49 |
| Float64 | A*X | 32 | tiled MR=6 (default) | 202.917 | 208.451 | 63.50 |
| Float64 | A*X | 32 | tiled MR=8 | 329.983 | 341.776 | 39.05 |
| Float64 | Aᵀ*X | 32 | mul! (register-tiled) | 190.393 | 191.133 | 67.68 |
| Float64 | Aᵀ*X | 32 | tiled MR=1 | 517.599 | 519.044 | 24.89 |
| Float64 | Aᵀ*X | 32 | tiled MR=2 | 275.139 | 275.715 | 46.83 |
| Float64 | Aᵀ*X | 32 | tiled MR=4 | 206.976 | 207.546 | 62.25 |
| Float64 | Aᵀ*X | 32 | tiled MR=6 (default) | 190.380 | 191.106 | 67.68 |
| Float64 | Aᵀ*X | 32 | tiled MR=8 | 250.240 | 250.941 | 51.49 |
| Float64 | A*X | 128 | mul! (lookup) | 929.050 | 930.341 | 55.48 |
| Float64 | A*X | 128 | tiled MR=1 | 2059.240 | 2069.276 | 25.03 |
| Float64 | A*X | 128 | tiled MR=2 | 1347.238 | 1353.675 | 38.26 |
| Float64 | A*X | 128 | tiled MR=4 | 953.132 | 957.610 | 54.07 |
| Float64 | A*X | 128 | tiled MR=6 (default) | 837.754 | 845.334 | 61.52 |
| Float64 | A*X | 128 | tiled MR=8 | 1328.306 | 1334.304 | 38.80 |
| Float64 | Aᵀ*X | 128 | mul! (register-tiled) | 785.046 | 788.425 | 65.65 |
| Float64 | Aᵀ*X | 128 | tiled MR=1 | 1947.646 | 1950.471 | 26.46 |
| Float64 | Aᵀ*X | 128 | tiled MR=2 | 1096.386 | 1102.089 | 47.01 |
| Float64 | Aᵀ*X | 128 | tiled MR=4 | 849.574 | 850.226 | 60.67 |
| Float64 | Aᵀ*X | 128 | tiled MR=6 (default) | 784.842 | 785.875 | 65.67 |
| Float64 | Aᵀ*X | 128 | tiled MR=8 | 1024.653 | 1025.872 | 50.30 |

## Cross-package check

Float64, k = 8.

| Product | sum(abs2) |
| --- | ---: |
| `A*x` | 405890623.3167932 |
| `Aᵀ*y` | 402993804.33265293 |
| `A*X` | 3199276206.4847851 |
| `Aᵀ*X` | 3219362549.5903707 |

## Generated code

Generated code for `Val{6}`, `Val{2}`, `Val{8}`.

| File | Lines with vfmadd or fmla |
| --- | ---: |
| `linalg_kernels_apple-m1_fork_20261004-015303_AX.s` | 24 |
| `linalg_kernels_apple-m1_fork_20261004-015303_AtX.s` | 48 |

## Peak RSS

1341 MiB
