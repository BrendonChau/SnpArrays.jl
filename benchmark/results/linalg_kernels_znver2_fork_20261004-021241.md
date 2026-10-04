# `linalg_kernels.jl` on znver2

## Environment

Simulated 8192 x 49152 genotypes in 2.184 s (1.843e+08 genotypes/s).

| Fact | Value |
| --- | --- |
| SnpArrays | `fork` |
| threads | `4` |
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
| Float32 | A*x | 1 | mul! | 57.436 | 57.989 | 7.01 |
| Float32 | Aᵀ*y | 1 | mul! | 32.456 | 32.558 | 12.41 |
| Float32 | A*X | 8 | mul! (lookup) | 214.203 | 214.283 | 15.04 |
| Float32 | A*X | 8 | tiled MR=1 | 182.159 | 182.658 | 17.68 |
| Float32 | A*X | 8 | tiled MR=2 | 127.526 | 130.315 | 25.26 |
| Float32 | A*X | 8 | tiled MR=4 | 115.300 | 115.712 | 27.94 |
| Float32 | A*X | 8 | tiled MR=6 (default) | 120.133 | 120.217 | 26.81 |
| Float32 | A*X | 8 | tiled MR=8 | 147.321 | 147.925 | 21.87 |
| Float32 | Aᵀ*X | 8 | mul! (register-tiled) | 52.546 | 52.681 | 61.30 |
| Float32 | Aᵀ*X | 8 | tiled MR=1 | 151.689 | 151.970 | 21.24 |
| Float32 | Aᵀ*X | 8 | tiled MR=2 | 77.925 | 78.367 | 41.34 |
| Float32 | Aᵀ*X | 8 | tiled MR=4 | 56.491 | 56.655 | 57.02 |
| Float32 | Aᵀ*X | 8 | tiled MR=6 (default) | 52.638 | 52.924 | 61.20 |
| Float32 | Aᵀ*X | 8 | tiled MR=8 | 53.779 | 53.958 | 59.90 |
| Float32 | A*X | 32 | mul! (lookup) | 261.809 | 262.671 | 49.21 |
| Float32 | A*X | 32 | tiled MR=1 | 509.719 | 512.809 | 25.28 |
| Float32 | A*X | 32 | tiled MR=2 | 409.696 | 410.508 | 31.45 |
| Float32 | A*X | 32 | tiled MR=4 | 382.333 | 384.924 | 33.70 |
| Float32 | A*X | 32 | tiled MR=6 (default) | 420.688 | 422.736 | 30.63 |
| Float32 | A*X | 32 | tiled MR=8 | 507.864 | 509.452 | 25.37 |
| Float32 | Aᵀ*X | 32 | mul! (register-tiled) | 197.032 | 197.786 | 65.39 |
| Float32 | Aᵀ*X | 32 | tiled MR=1 | 320.687 | 321.149 | 40.18 |
| Float32 | Aᵀ*X | 32 | tiled MR=2 | 175.342 | 176.068 | 73.48 |
| Float32 | Aᵀ*X | 32 | tiled MR=4 | 147.340 | 147.714 | 87.45 |
| Float32 | Aᵀ*X | 32 | tiled MR=6 (default) | 197.651 | 197.984 | 65.19 |
| Float32 | Aᵀ*X | 32 | tiled MR=8 | 231.972 | 233.357 | 55.55 |
| Float32 | A*X | 128 | mul! (lookup) | 1063.926 | 1064.215 | 48.44 |
| Float32 | A*X | 128 | tiled MR=1 | 2383.495 | 2393.563 | 21.62 |
| Float32 | A*X | 128 | tiled MR=2 | 1967.396 | 1977.194 | 26.20 |
| Float32 | A*X | 128 | tiled MR=4 | 1824.624 | 1843.873 | 28.25 |
| Float32 | A*X | 128 | tiled MR=6 (default) | 1998.747 | 2006.778 | 25.79 |
| Float32 | A*X | 128 | tiled MR=8 | 2335.631 | 2341.453 | 22.07 |
| Float32 | Aᵀ*X | 128 | mul! (register-tiled) | 1329.016 | 1330.441 | 38.78 |
| Float32 | Aᵀ*X | 128 | tiled MR=1 | 1636.011 | 1639.496 | 31.50 |
| Float32 | Aᵀ*X | 128 | tiled MR=2 | 1186.630 | 1188.725 | 43.43 |
| Float32 | Aᵀ*X | 128 | tiled MR=4 | 1118.913 | 1120.195 | 46.06 |
| Float32 | Aᵀ*X | 128 | tiled MR=6 (default) | 1325.005 | 1328.789 | 38.90 |
| Float32 | Aᵀ*X | 128 | tiled MR=8 | 1504.214 | 1504.527 | 34.26 |
| Float64 | A*x | 1 | mul! | 96.345 | 96.490 | 4.18 |
| Float64 | Aᵀ*y | 1 | mul! | 71.908 | 72.281 | 5.60 |
| Float64 | A*X | 8 | mul! (lookup) | 212.571 | 213.646 | 15.15 |
| Float64 | A*X | 8 | tiled MR=1 | 200.272 | 200.743 | 16.08 |
| Float64 | A*X | 8 | tiled MR=2 | 134.178 | 134.976 | 24.01 |
| Float64 | A*X | 8 | tiled MR=4 | 113.211 | 114.095 | 28.45 |
| Float64 | A*X | 8 | tiled MR=6 (default) | 136.400 | 137.333 | 23.62 |
| Float64 | A*X | 8 | tiled MR=8 | 182.246 | 182.741 | 17.68 |
| Float64 | Aᵀ*X | 8 | mul! (register-tiled) | 85.228 | 85.425 | 37.80 |
| Float64 | Aᵀ*X | 8 | tiled MR=1 | 152.517 | 152.635 | 21.12 |
| Float64 | Aᵀ*X | 8 | tiled MR=2 | 77.850 | 78.259 | 41.38 |
| Float64 | Aᵀ*X | 8 | tiled MR=4 | 59.588 | 59.737 | 54.06 |
| Float64 | Aᵀ*X | 8 | tiled MR=6 (default) | 85.256 | 85.374 | 37.78 |
| Float64 | Aᵀ*X | 8 | tiled MR=8 | 102.543 | 102.930 | 31.41 |
| Float64 | A*X | 32 | mul! (lookup) | 314.634 | 319.110 | 40.95 |
| Float64 | A*X | 32 | tiled MR=1 | 796.224 | 798.771 | 16.18 |
| Float64 | A*X | 32 | tiled MR=2 | 529.640 | 533.552 | 24.33 |
| Float64 | A*X | 32 | tiled MR=4 | 444.517 | 449.727 | 28.99 |
| Float64 | A*X | 32 | tiled MR=6 (default) | 531.623 | 534.972 | 24.24 |
| Float64 | A*X | 32 | tiled MR=8 | 709.435 | 711.867 | 18.16 |
| Float64 | Aᵀ*X | 32 | mul! (register-tiled) | 344.933 | 345.095 | 37.35 |
| Float64 | Aᵀ*X | 32 | tiled MR=1 | 611.999 | 612.210 | 21.05 |
| Float64 | Aᵀ*X | 32 | tiled MR=2 | 316.594 | 316.864 | 40.70 |
| Float64 | Aᵀ*X | 32 | tiled MR=4 | 241.984 | 242.535 | 53.25 |
| Float64 | Aᵀ*X | 32 | tiled MR=6 (default) | 345.626 | 346.018 | 37.28 |
| Float64 | Aᵀ*X | 32 | tiled MR=8 | 414.373 | 414.457 | 31.09 |
| Float64 | A*X | 128 | mul! (lookup) | 1482.531 | 1483.813 | 34.76 |
| Float64 | A*X | 128 | tiled MR=1 | 2898.520 | 2930.810 | 17.78 |
| Float64 | A*X | 128 | tiled MR=2 | 1994.201 | 2028.853 | 25.84 |
| Float64 | A*X | 128 | tiled MR=4 | 1609.185 | 1623.099 | 32.03 |
| Float64 | A*X | 128 | tiled MR=6 (default) | 1925.337 | 1943.069 | 26.77 |
| Float64 | A*X | 128 | tiled MR=8 | 2876.518 | 2888.655 | 17.92 |
| Float64 | Aᵀ*X | 128 | mul! (register-tiled) | 1484.700 | 1489.855 | 34.71 |
| Float64 | Aᵀ*X | 128 | tiled MR=1 | 2467.775 | 2477.655 | 20.89 |
| Float64 | Aᵀ*X | 128 | tiled MR=2 | 1351.735 | 1357.497 | 38.13 |
| Float64 | Aᵀ*X | 128 | tiled MR=4 | 1050.492 | 1052.292 | 49.06 |
| Float64 | Aᵀ*X | 128 | tiled MR=6 (default) | 1453.847 | 1460.152 | 35.45 |
| Float64 | Aᵀ*X | 128 | tiled MR=8 | 1735.809 | 1737.473 | 29.69 |

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
| `linalg_kernels_znver2_fork_20261004-021241_AX.s` | 12 |
| `linalg_kernels_znver2_fork_20261004-021241_AtX.s` | 24 |

## Peak RSS

1280 MiB
