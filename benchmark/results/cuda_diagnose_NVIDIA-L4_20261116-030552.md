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
| sections | `stream,scale,tiles,accuracy` |

Centered, scaled, mean-imputed throughout.

## stream: what bounds a streamed sweep?

Float32, k = 32, width = 4096, 163 chunks. First and second sweep in wall seconds; GB/s is bed bytes over the second.

| Part | Product | Tensor | first s | second s | bed GB/s |
|---|---|---|---:|---:|---:|
| (a) host read | - | - | 1.470 | 1.073 | 7.77 |
| (b) read + upload | - | - | 1.646 | 1.078 | 7.73 |

(c) Splits per chunk (min of 3), times 163 chunks.

| Split | per chunk ms | per sweep s |
|---|---:|---:|
| `A*X` rhs (4096 rows) + tables | 0.031 | 0.005 |
| `Aᵀ*X` rhs (50000 rows) + tables | 0.039 | 0.006 |

(d) Streamed products.

| Part | Product | Tensor | first s | second s | bed GB/s |
|---|---|---|---:|---:|---:|
| (d) streamed | A*X | on | 4.516 | 1.124 | 7.42 |
| (d) streamed | Aᵀ*X | on | 1.933 | 1.217 | 6.85 |
| (d) streamed | A*(Aᵀ*X)/n | on | 1.384 | 1.164 | 7.16 |
| (d) streamed | A*(Aᵀ*X)/n mixed | on | 1.844 | 1.126 | 7.40 |
| (d) streamed | A*X | off | 2.416 | 1.183 | 7.05 |
| (d) streamed | Aᵀ*X | off | 2.199 | 1.216 | 6.86 |
| (d) streamed | A*(Aᵀ*X)/n | off | 1.285 | 1.300 | 6.42 |
| (d) streamed | A*(Aᵀ*X)/n mixed | off | 1.298 | 1.312 | 6.36 |

(e) The same kernels in memory (min of 3), and the remainder (d) second − (b) second − (e). The mixed row's (e) is the Float32 in-memory product.

| Product | Tensor | (e) min s | (e) median s | remainder s |
|---|---|---:|---:|---:|
| A*X | on | 0.156 | 0.165 | -0.111 |
| Aᵀ*X | on | 0.168 | 0.178 | -0.030 |
| A*(Aᵀ*X)/n | on | 0.340 | 0.347 | -0.255 |
| A*(Aᵀ*X)/n mixed | on | 0.340 | 0.347 | -0.292 |
| A*X | off | 0.319 | 0.321 | -0.215 |
| Aᵀ*X | off | 0.368 | 0.371 | -0.231 |
| A*(Aᵀ*X)/n | off | 0.682 | 0.694 | -0.460 |
| A*(Aᵀ*X)/n mixed | off | 0.682 | 0.694 | -0.448 |

## scale: k-independent cost, and growth with m

Float32, min and median of 3. The tensor kernels are called directly at every k, below the dispatch thresholds too.

| Variant | m | n | k | min ms | median ms | GFMA/s |
|---|---:|---:|---:|---:|---:|---:|
| tiled A*X | 50000 | 667095 | 2 | 152.664 | 154.238 | 436.97 |
| tiled A*X | 50000 | 667095 | 4 | 159.226 | 163.447 | 837.92 |
| tiled A*X | 50000 | 667095 | 8 | 166.502 | 166.755 | 1602.61 |
| tiled A*X | 50000 | 667095 | 12 | 292.797 | 294.140 | 1367.01 |
| tiled A*X | 50000 | 667095 | 16 | 302.329 | 304.607 | 1765.22 |
| tiled A*X | 50000 | 667095 | 24 | 312.358 | 316.299 | 2562.81 |
| tiled A*X | 50000 | 667095 | 32 | 319.701 | 321.325 | 3338.59 |
| tiled A*X | 50000 | 667095 | 48 | 553.993 | 559.496 | 2889.98 |
| tiled A*X | 50000 | 667095 | 64 | 600.334 | 614.032 | 3555.86 |
| tiled A*X | 50000 | 667095 | 96 | 1149.547 | 1154.650 | 2785.49 |
| tiled A*X | 50000 | 667095 | 128 | 1215.165 | 1216.297 | 3513.44 |
| tensor A*X | 50000 | 667095 | 2 | 138.653 | 145.883 | 481.13 |
| tensor A*X | 50000 | 667095 | 4 | 146.921 | 149.959 | 908.10 |
| tensor A*X | 50000 | 667095 | 8 | 151.815 | 151.856 | 1757.65 |
| tensor A*X | 50000 | 667095 | 12 | 156.305 | 159.220 | 2560.74 |
| tensor A*X | 50000 | 667095 | 16 | 157.728 | 157.834 | 3383.53 |
| tensor A*X | 50000 | 667095 | 24 | 163.771 | 164.661 | 4888.00 |
| tensor A*X | 50000 | 667095 | 32 | 172.024 | 172.373 | 6204.68 |
| tensor A*X | 50000 | 667095 | 48 | 248.961 | 249.273 | 6430.84 |
| tensor A*X | 50000 | 667095 | 64 | 262.651 | 263.132 | 8127.53 |
| tensor A*X | 50000 | 667095 | 96 | 518.748 | 519.414 | 6172.66 |
| tensor A*X | 50000 | 667095 | 128 | 543.075 | 544.904 | 7861.54 |
| tiled Aᵀ*X | 50000 | 667095 | 2 | 198.691 | 208.825 | 335.75 |
| tiled Aᵀ*X | 50000 | 667095 | 4 | 215.125 | 222.531 | 620.19 |
| tiled Aᵀ*X | 50000 | 667095 | 8 | 224.683 | 232.945 | 1187.62 |
| tiled Aᵀ*X | 50000 | 667095 | 12 | 343.535 | 346.726 | 1165.11 |
| tiled Aᵀ*X | 50000 | 667095 | 16 | 355.616 | 359.339 | 1500.71 |
| tiled Aᵀ*X | 50000 | 667095 | 24 | 370.188 | 383.486 | 2162.45 |
| tiled Aᵀ*X | 50000 | 667095 | 32 | 392.296 | 396.979 | 2720.78 |
| tiled Aᵀ*X | 50000 | 667095 | 48 | 626.587 | 632.749 | 2555.16 |
| tiled Aᵀ*X | 50000 | 667095 | 64 | 685.543 | 687.097 | 3113.89 |
| tiled Aᵀ*X | 50000 | 667095 | 96 | 1249.371 | 1251.440 | 2562.93 |
| tiled Aᵀ*X | 50000 | 667095 | 128 | 1379.673 | 1382.053 | 3094.51 |
| tensor Aᵀ*X | 50000 | 667095 | 2 | 149.822 | 160.129 | 445.26 |
| tensor Aᵀ*X | 50000 | 667095 | 4 | 161.690 | 171.471 | 825.16 |
| tensor Aᵀ*X | 50000 | 667095 | 8 | 165.353 | 167.509 | 1613.74 |
| tensor Aᵀ*X | 50000 | 667095 | 12 | 170.882 | 171.704 | 2342.30 |
| tensor Aᵀ*X | 50000 | 667095 | 16 | 177.704 | 179.196 | 3003.17 |
| tensor Aᵀ*X | 50000 | 667095 | 24 | 182.182 | 187.556 | 4394.04 |
| tensor Aᵀ*X | 50000 | 667095 | 32 | 194.898 | 197.493 | 5476.47 |
| tensor Aᵀ*X | 50000 | 667095 | 48 | 294.766 | 298.561 | 5431.53 |
| tensor Aᵀ*X | 50000 | 667095 | 64 | 307.795 | 312.130 | 6935.47 |
| tensor Aᵀ*X | 50000 | 667095 | 96 | 498.500 | 499.504 | 6423.39 |
| tensor Aᵀ*X | 50000 | 667095 | 128 | 521.744 | 523.961 | 8182.95 |

Least-squares `time = a + b·k`, seconds.

| Variant | k range | a | b | a / time(32) |
|---|---|---:|---:|---:|
| tiled A*X | 12..32 | 2.795e-01 | 1.301e-03 | 0.874 |
| tiled A*X | 48..128 | 9.423e-02 | 9.352e-03 | 0.295 |
| tensor A*X | 12..32 | 1.457e-01 | 7.974e-04 | 0.847 |
| tensor A*X | 48..128 | 3.955e-02 | 4.212e-03 | 0.230 |
| tiled Aᵀ*X | 12..32 | 3.159e-01 | 2.356e-03 | 0.805 |
| tiled Aᵀ*X | 48..128 | 1.081e-01 | 1.044e-02 | 0.276 |
| tensor Aᵀ*X | 12..32 | 1.579e-01 | 1.118e-03 | 0.810 |
| tensor Aᵀ*X | 48..128 | 1.349e-01 | 3.223e-03 | 0.692 |

Tiled-to-tensor time ratio (above 1: tensor faster).

| k | A*X | Aᵀ*X |
|---:|---:|---:|
| 2 | 1.101 | 1.326 |
| 4 | 1.084 | 1.330 |
| 8 | 1.097 | 1.359 |
| 12 | 1.873 | 2.010 |
| 16 | 1.917 | 2.001 |
| 24 | 1.907 | 2.032 |
| 32 | 1.858 | 2.013 |
| 48 | 2.225 | 2.126 |
| 64 | 2.286 | 2.227 |
| 96 | 2.216 | 2.506 |
| 128 | 2.238 | 2.644 |

Growth at k = 32; sub-operators share the value tables.

| Variant | m | n | k | min ms | median ms | GFMA/s |
|---|---:|---:|---:|---:|---:|---:|
| tiled A*X | 12496 | 667095 | 32 | 82.259 | 83.048 | 3242.84 |
| tensor A*X | 12496 | 667095 | 32 | 44.025 | 44.113 | 6059.14 |
| tiled Aᵀ*X | 12496 | 667095 | 32 | 93.922 | 94.070 | 2840.14 |
| tensor Aᵀ*X | 12496 | 667095 | 32 | 51.534 | 52.465 | 5176.26 |
| tiled A*X | 24992 | 667095 | 32 | 163.414 | 167.456 | 3264.75 |
| tensor A*X | 24992 | 667095 | 32 | 85.091 | 85.618 | 6269.80 |
| tiled Aᵀ*X | 24992 | 667095 | 32 | 187.601 | 190.535 | 2843.83 |
| tensor Aᵀ*X | 24992 | 667095 | 32 | 94.838 | 95.949 | 5625.45 |
| tiled A*X | 50000 | 667095 | 32 | 326.309 | 333.092 | 3270.99 |
| tensor A*X | 50000 | 667095 | 32 | 169.292 | 169.603 | 6304.81 |
| tiled Aᵀ*X | 50000 | 667095 | 32 | 381.122 | 383.797 | 2800.56 |
| tensor Aᵀ*X | 50000 | 667095 | 32 | 192.666 | 199.183 | 5539.92 |
| tiled A*X | 50000 | 166773 | 32 | 80.222 | 80.442 | 3326.22 |
| tensor A*X | 50000 | 166773 | 32 | 41.775 | 41.879 | 6387.46 |
| tiled Aᵀ*X | 50000 | 166773 | 32 | 95.256 | 95.856 | 2801.27 |
| tensor Aᵀ*X | 50000 | 166773 | 32 | 50.751 | 53.123 | 5257.71 |
| tiled A*X | 50000 | 333547 | 32 | 161.162 | 161.664 | 3311.42 |
| tensor A*X | 50000 | 333547 | 32 | 84.539 | 84.733 | 6312.74 |
| tiled Aᵀ*X | 50000 | 333547 | 32 | 194.554 | 195.176 | 2743.07 |
| tensor Aᵀ*X | 50000 | 333547 | 32 | 95.357 | 96.691 | 5596.61 |

## tiles: tile constants or kernel body?

Float32 `A*X`, min and median of 3. Default shape for k: `AX_TILES[_tile_band(k)]`.

| k | shape (BM,BN,BK,TM,TK) | default | min ms | median ms | GFMA/s | rel. diff |
|---:|---|---|---:|---:|---:|---:|
| 32 | `(256, 16, 32, 8, 8)` | yes | 327.205 | 327.761 | 3262.03 | 0.0e+00 |
| 32 | `(128, 16, 64, 8, 8)` |  | 541.279 | 541.849 | 1971.91 | 1.1e-05 |
| 32 | `(256, 16, 64, 8, 8)` |  | 508.973 | 509.827 | 2097.07 | 0.0e+00 |
| 32 | `(256, 32, 32, 8, 8)` |  | 382.219 | 382.985 | 2792.51 | 0.0e+00 |
| 32 | `(128, 32, 32, 8, 8)` |  | 463.657 | 463.797 | 2302.03 | 1.1e-05 |
| 128 | `(256, 16, 32, 8, 8)` |  | 1328.601 | 1334.109 | 3213.46 | 0.0e+00 |
| 128 | `(128, 16, 64, 8, 8)` | yes | 1251.929 | 1286.902 | 3410.26 | 0.0e+00 |
| 128 | `(256, 16, 64, 8, 8)` |  | 1074.103 | 1075.239 | 3974.86 | 1.4e-05 |
| 128 | `(256, 32, 32, 8, 8)` |  | 1426.939 | 1434.635 | 2992.00 | 0.0e+00 |
| 128 | `(128, 32, 32, 8, 8)` |  | 1695.684 | 1700.993 | 2517.81 | 0.0e+00 |

## accuracy: Float32 paths vs Float64

k = 32. Reference: Float64 tiled product on the same packed words, on the Float32-rounded right-hand sides. Worst column of `‖out − ref‖ / ‖ref‖`.

| Product | Path | RHS | rel. error |
|---|---|---|---:|
| A*X | tiled A*X | randn | 1.00e-05 |
| A*X | tensor A*X | randn | 1.27e-06 |
| Aᵀ*X | tiled Aᵀ*X | randn | 4.03e-06 |
| Aᵀ*X | tensor Aᵀ*X | randn | 8.49e-07 |
| A*X | tiled A*X | columns | 1.04e-05 |
| A*X | tensor A*X | columns | 1.27e-06 |
| Aᵀ*X | tiled Aᵀ*X | columns | 4.03e-06 |
| Aᵀ*X | tensor Aᵀ*X | columns | 8.40e-07 |
| A*X | tiled A*X | entries | 6.54e-06 |
| A*X | tensor A*X | entries | 1.24e-06 |
| Aᵀ*X | tiled Aᵀ*X | entries | 2.91e-06 |
| Aᵀ*X | tensor Aᵀ*X | entries | 8.22e-07 |
| A*(Aᵀ*X)/n | stream mixed vs Float64 | randn | 4.55e-07 |

`Aᵀ*X` by allele-frequency bin: worst row `‖out[i, :] − ref[i, :]‖ / ‖ref[i, :]‖`, rows with nonzero reference.

| Path | RHS | MAF bin | SNPs | rel. error |
|---|---|---|---:|---:|
| tiled Aᵀ*X | randn | MAF < 1% | 758 | 8.46e-06 |
| tiled Aᵀ*X | randn | 1% ≤ MAF ≤ 5% | 161786 | 1.01e-05 |
| tiled Aᵀ*X | randn | MAF > 5% | 504551 | 1.12e-05 |
| tensor Aᵀ*X | randn | MAF < 1% | 758 | 2.26e-06 |
| tensor Aᵀ*X | randn | 1% ≤ MAF ≤ 5% | 161786 | 2.46e-06 |
| tensor Aᵀ*X | randn | MAF > 5% | 504551 | 1.98e-06 |
| tiled Aᵀ*X | columns | MAF < 1% | 758 | 4.50e-05 |
| tiled Aᵀ*X | columns | 1% ≤ MAF ≤ 5% | 161786 | 1.36e-04 |
| tiled Aᵀ*X | columns | MAF > 5% | 504551 | 1.65e-04 |
| tensor Aᵀ*X | columns | MAF < 1% | 758 | 1.31e-05 |
| tensor Aᵀ*X | columns | 1% ≤ MAF ≤ 5% | 161786 | 2.93e-05 |
| tensor Aᵀ*X | columns | MAF > 5% | 504551 | 3.01e-05 |
| tiled Aᵀ*X | entries | MAF < 1% | 758 | 5.43e-06 |
| tiled Aᵀ*X | entries | 1% ≤ MAF ≤ 5% | 161786 | 7.45e-06 |
| tiled Aᵀ*X | entries | MAF > 5% | 504551 | 7.54e-06 |
| tensor Aᵀ*X | entries | MAF < 1% | 758 | 2.17e-06 |
| tensor Aᵀ*X | entries | 1% ≤ MAF ≤ 5% | 161786 | 2.79e-06 |
| tensor Aᵀ*X | entries | MAF > 5% | 504551 | 2.01e-06 |

## Peak memory

| device in use, sampled, pool cache | 13045 MiB |
| RSS | 11677 MiB |
