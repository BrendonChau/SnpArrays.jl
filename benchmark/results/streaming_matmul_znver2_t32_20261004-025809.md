# `streaming_matmul.jl` on znver2

## Environment

| Fact | Value |
| --- | --- |
| threads | `32` |
| Sys.CPU_NAME | `znver2` |
| Sys.ARCH | `x86_64` |
| VERSION | `1.12.7` |
| pkgdir | `/u/home/b/bhchau/projects/SnpArrays.jl` |
| bed | `/u/scratch/b/bhchau/cudaext_check/synthetic_v1_chr-21.bed` |
| bed bytes | `27637596003` |
| m, n | `m = 1008000, n = 109673` |
| k | `128` |
| width | `4096` |
| chunks | `27` |
| prefetch | `true` |
| readers | `8` |
| VECTOR_BYTES | `32` |

## Timings

| Type | Product | k | seconds | GFMA/s |
|---|---|---:|---:|---:|
| Float32 | A*X | 128 | 25.934 | 545.63 |
| Float32 | Aᵀ*X | 128 | 56.650 | 249.79 |
| Float32 | A*(Aᵀ*X)/n | 128 | 94.071 | 300.85 |
| Float64 | A*X | 128 | 63.570 | 222.60 |
| Float64 | Aᵀ*X | 128 | 116.756 | 121.20 |
| Float64 | A*(Aᵀ*X)/n | 128 | 179.148 | 157.97 |
| Float32 stream, Float64 X | A*(Aᵀ*X)/n | 128 | 100.619 | 281.27 |

## Checksums

| Type | Product | sum(abs2) |
| --- | --- | ---: |
| Float32 | `A*X` | 14763062984704 |
| Float32 | `Aᵀ*X` | 14856998617088 |
| Float32 | `A*(Aᵀ*X)/n` | 471755292672 |
| Float64 | `A*X` | 14928534452221.605 |
| Float64 | `Aᵀ*X` | 14779120789704.586 |
| Float64 | `A*(Aᵀ*X)/n` | 449858781938.81378 |
| Float32 stream, Float64 X | `A*(Aᵀ*X)/n` | 449858786892.78571 |

## Peak RSS

11982 MiB
