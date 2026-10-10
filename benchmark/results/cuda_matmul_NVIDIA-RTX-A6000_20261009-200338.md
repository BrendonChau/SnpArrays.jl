# `cuda_matmul.jl` on NVIDIA RTX A6000

## Environment

| Fact | Value |
| --- | --- |
| GPU | `NVIDIA RTX A6000` |
| GPU memory | `47.529 GiB` |
| CUDA.jl | `6.4.2` |
| CUDA runtime | `12.4.0` |
| threads | `8` |
| Sys.CPU_NAME | `znver3` |
| VERSION | `1.12.7` |
| pkgdir | `/u/home/b/bhchau/projects/SnpArrays.jl` |
| bed | `/work/15115979.1.gpu_a6000.q/chr21/synthetic_v1_chr-21.bed` |
| m, n | `m = 1008000, n = 109673` |
| k | `128` |

## Timings

Centered, scaled, mean-imputed. GFMA/s is from the warm call.

| Type | Product | k | Variant | first s | warm s | GFMA/s |
|---|---|---:|---|---:|---:|---:|
| Float32 | A*X | 128 | mul! (tensor-core) | 7.296 | 0.960 | 14744.60 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 1.650 | 1.029 | 13749.13 |
| Float64 | A*X | 128 | mul! (tiled) | 45.661 | 45.210 | 312.99 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 47.348 | 46.679 | 303.15 |

## Upload

`CuSnpArray{T}(G)`: column statistics on the host, then the packed words to the device.

| Type | seconds | packed words | bed GB/s |
|---|---:|---:|---:|
| Float32 | 3.449 | 25.740 GiB | 8.01 |
| Float64 | 1.421 | 25.740 GiB | 19.45 |

## Checksums

| Type | Product | sum(abs2) |
| --- | --- | ---: |
| Float32 | `A*X` | 14763056693248 |
| Float32 | `Aᵀ*X` | 14856992325632 |
| Float64 | `A*X` | 14928534452221.605 |
| Float64 | `Aᵀ*X` | 14779120789704.586 |

## Peak RSS

28804 MiB
