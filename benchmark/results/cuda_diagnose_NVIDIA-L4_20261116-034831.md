# `cuda_diagnose.jl` on NVIDIA L4

## Environment

| Fact | Value |
| --- | --- |
| GPU | `NVIDIA L4` |
| GPU memory | `22.278 GiB` |
| compute capability | `8.9.0` |
| CUDA.jl | `6.5.1` |
| CUDA runtime | `13.4.0` |
| threads | `8` |
| Sys.CPU_NAME | `cascadelake` |
| VERSION | `1.12.7` |
| pkgdir | `/home/jupyter/repos/SnpArrays.jl` |
| bed | `/home/jupyter/repos/allxall-phenotypes/data/09_methods/eur_pm/eur_pm_50k_ldakthin.bed` |
| bed used | `/home/jupyter/repos/allxall-phenotypes/data/09_methods/eur_pm/eur_pm_50k_ldakthin.bed` |
| bed bytes | `8338687503` |
| m, n | `m = 50000, n = 667095` |
| sections | `shapes` |

Centered, scaled, mean-imputed throughout.

## shapes: tile shapes of the tiled Float32 kernels

Min and median of 3. rel. diff: from the default shape's output. rel. error: worst column against the Float64 tiled product on the same packed words, `randn` right-hand sides rounded to Float32.

`A*X`, shape (BM,BN,BK,TM,TK); default `AX_TILES[_tile_band(k)]`.

| k | shape | default | min ms | median ms | GFMA/s | rel. diff | rel. error |
|---:|---|---|---:|---:|---:|---:|---:|
| 48 | `(128, 16, 64, 8, 8)` | yes | 558.049 | 565.171 | 2868.97 | 0.0e+00 | 1.40e-05 |
| 48 | `(256, 16, 64, 8, 8)` |  | 522.839 | 523.535 | 3062.18 | 1.1e-05 | 1.16e-05 |
| 64 | `(128, 16, 64, 8, 8)` | yes | 612.233 | 614.787 | 3486.75 | 0.0e+00 | 1.42e-05 |
| 64 | `(256, 16, 64, 8, 8)` |  | 553.224 | 554.613 | 3858.66 | 1.1e-05 | 1.06e-05 |
| 96 | `(128, 16, 64, 8, 8)` | yes | 1172.354 | 1172.513 | 2731.30 | 0.0e+00 | 2.60e-05 |
| 96 | `(256, 16, 64, 8, 8)` |  | 1040.840 | 1041.926 | 3076.42 | 1.4e-05 | 1.50e-05 |
| 128 | `(128, 16, 64, 8, 8)` | yes | 1234.689 | 1236.032 | 3457.88 | 0.0e+00 | 2.11e-05 |
| 128 | `(256, 16, 64, 8, 8)` |  | 1063.988 | 1064.959 | 4012.65 | 1.4e-05 | 1.59e-05 |

`Aᵀ*X`, shape (BM,BN,BK,TN,TK); default `ATX_TILES[_tile_band(k)]`.

| k | shape | default | min ms | median ms | GFMA/s | rel. diff | rel. error |
|---:|---|---|---:|---:|---:|---:|---:|
| 32 | `(32, 128, 32, 4, 8)` | yes | 351.785 | 391.094 | 3034.10 | 0.0e+00 | 4.03e-06 |
| 32 | `(32, 128, 64, 8, 8)` |  | 567.487 | 572.628 | 1880.84 | 0.0e+00 | 4.03e-06 |
| 32 | `(64, 64, 64, 4, 8)` |  | 722.590 | 723.253 | 1477.12 | 0.0e+00 | 4.03e-06 |
| 32 | `(32, 256, 64, 8, 8)` |  | 542.485 | 547.429 | 1967.53 | 0.0e+00 | 4.03e-06 |
| 32 | `(64, 128, 32, 4, 8)` |  | 509.060 | 512.278 | 2096.71 | 0.0e+00 | 4.03e-06 |
| 128 | `(32, 128, 32, 4, 8)` |  | 1594.454 | 1603.683 | 2677.66 | 0.0e+00 | 4.04e-06 |
| 128 | `(32, 128, 64, 8, 8)` | yes | 1378.988 | 1382.506 | 3096.04 | 0.0e+00 | 4.04e-06 |
| 128 | `(64, 64, 64, 4, 8)` |  | 1904.218 | 1904.936 | 2242.08 | 0.0e+00 | 4.04e-06 |
| 128 | `(32, 256, 64, 8, 8)` |  | 1155.308 | 1156.770 | 3695.47 | 0.0e+00 | 4.04e-06 |
| 128 | `(64, 128, 32, 4, 8)` |  | 2049.228 | 2050.010 | 2083.42 | 0.0e+00 | 4.04e-06 |

## Peak memory

| device in use, sampled, pool cache | 8895 MiB |
| RSS | 14031 MiB |
