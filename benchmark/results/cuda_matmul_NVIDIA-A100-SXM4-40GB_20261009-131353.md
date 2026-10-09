# `cuda_matmul.jl` on NVIDIA A100-SXM4-40GB

## Environment

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
| bed | `/u/scratch/b/bhchau/cudaext_check/synthetic_v1_chr-21.bed` |
| m, n | `m = 1008000, n = 109673` |
| k | `128` |

## Timings

Centered, scaled, mean-imputed. GFMA/s is from the warm call.

| Type | Product | k | Variant | first s | warm s | GFMA/s |
|---|---|---:|---|---:|---:|---:|
| Float32 | A*X | 128 | mul! (tensor-core) | 9.048 | 0.634 | 22324.65 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 1.750 | 0.560 | 25262.79 |
| Float64 | A*X | 128 | mul! (tiled) | 6.045 | 4.205 | 3364.99 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 5.298 | 4.022 | 3518.41 |

## Upload

`CuSnpArray{T}(G)`: column statistics on the host, then the packed words to the device.

| Type | seconds | packed words | bed GB/s |
|---|---:|---:|---:|
| Float32 | 3.776 | 25.740 GiB | 7.32 |
| Float64 | 2.585 | 25.740 GiB | 10.69 |

## Checksums

| Type | Product | sum(abs2) |
| --- | --- | ---: |
| Float32 | `A*X` | 14763056693248 |
| Float32 | `Aᵀ*X` | 14856992325632 |
| Float64 | `A*X` | 14928534452221.605 |
| Float64 | `Aᵀ*X` | 14779120789704.586 |

## Peak RSS

29209 MiB
