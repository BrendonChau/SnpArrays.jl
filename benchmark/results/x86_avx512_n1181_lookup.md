# Lookup-table `A*X` kernel on x86 AVX-512

`mul!(out, A::SnpLinAlg, X::Matrix)` gained a second kernel that, per
block of four SNPs, builds a 256-row table `T[code, :] = Σ_l
values[code_l, j_l] * X[j_l, :]` and gathers one table row per sample
instead of four FMAs per SNP. It is selected when `m >= LOOKUP_MIN_ROWS`
(2048) and `k >= LOOKUP_MIN_RHS` (4). Tables and tile partials use the
element type `T`. The design follows product 2 of `kq_pass.c`.

## Environment

| Fact | Value |
| --- | --- |
| Node | n1181, 16 cores allocated |
| ISA | `avx512f dq bw vl vbmi vnni`, `VECTOR_BYTES[] = 64` |
| Julia | 1.12.7 |
| Shape | `EUR_subset` stacked 101x: m = 38279 samples, n = 54051 SNPs |
| Timing | minimum of 3 calls after one warm-up |
| Tunables | `LOOKUP_ROW_TILE = 512`, `LOOKUP_CHUNK_SNPS = 1024`, `LOOKUP_GROUP_BUDGET = 1 MiB` (not yet swept) |

## `A*X`, seconds

| T | k | threads | register-tiled (before) | lookup (after) | `kq_pass.c` product 2 (Float32 tables) |
| --- | --- | --- | --- | --- | --- |
| Float32 | 8 | 1 | 2.01 | 0.76 | 1.33 |
| Float64 | 8 | 1 | 2.18 | 0.71 | 1.33 |
| Float32 | 64 | 1 | 5.03 | 1.64 | 2.75 |
| Float64 | 64 | 1 | 8.50 | 3.29 | 2.75 |
| Float32 | 8 | 16 | 0.18 | 0.16 | 0.12 |
| Float64 | 8 | 16 | 0.19 | 0.14 | 0.12 |
| Float32 | 64 | 16 | 0.44 | 0.23 | 0.33 |
| Float64 | 64 | 16 | 0.60 | 0.41 | 0.33 |

`transpose(A)*X` is unchanged (0.79 s / 2.08 s / 4.04 s single-threaded at
`k = 8` Float32, 64 Float32, 64 Float64; 0.06 s / 0.15 s / 0.29 s at 16
threads).

## Notes

1. Single-threaded the lookup kernel is 2.6x to 3.1x faster than the
   register-tiled kernel and 1.7x to 1.9x faster than `kq_pass.c` at
   equal precision (Float32).
2. Scaling from 1 to 16 threads is 4.7x to 7x, below the 11x to 14x of the
   register-tiled kernel, so the gain at 16 threads is 1.1x to 1.9x. Each
   chunk of 1024 SNPs is two `@sync` phases (table build over blocks,
   gather over sample blocks), and the per-chunk transposed codes (m x 256
   bytes) and output tile flush are memory traffic the register-tiled
   kernel does not have. Sweeping `LOOKUP_CHUNK_SNPS` and
   `LOOKUP_GROUP_BUDGET`, and merging the table build into the gather
   tasks, are the untried levers.
3. One 16-thread run measured Float32 `k = 8` at 0.33 s; a repeat gave
   0.16 s. Treat 16-thread differences under 0.05 s as noise.
4. Correctness: `test/linalg_simd.jl` passes at 1 and 4 threads with the
   existing `64 eps(T)` tolerance, including chunk, tile, byte, and rhs
   sub-block remainders, all three models, and `impute=false` NaN
   propagation.
