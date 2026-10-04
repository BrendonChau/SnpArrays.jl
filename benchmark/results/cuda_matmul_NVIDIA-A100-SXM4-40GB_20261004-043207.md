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
| Float32 | A*X | 128 | mul! (tensor-core) | 0.634 | 0.633 | 22361.06 |
| Float32 | Aᵀ*X | 128 | mul! (tensor-core) | 0.561 | 0.560 | 25273.93 |
| Float64 | A*X | 128 | mul! (tiled) | 4.204 | 4.204 | 3365.75 |
| Float64 | Aᵀ*X | 128 | mul! (tiled) | 4.020 | 4.020 | 3519.84 |

## Upload

`CuSnpArray{T}(G)`: column statistics on the host, then the packed words to the device.

| Type | seconds | packed words | bed GB/s |
|---|---:|---:|---:|
| Float32 | 3.347 | 25.740 GiB | 8.26 |
| Float64 | 2.866 | 25.740 GiB | 9.64 |

## Checksums

| Type | Product | sum(abs2) |
| --- | --- | ---: |
| Float32 | `A*X` | 14763056693248 |
| Float32 | `Aᵀ*X` | 14856992325632 |
| Float64 | `A*X` | 14928534452221.605 |
| Float64 | `Aᵀ*X` | 14779120789704.586 |

## Peak RSS

33562 MiB
