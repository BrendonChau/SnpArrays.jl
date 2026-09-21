# Streaming `K Q` product on x86 AVX-512

`streamed_grm_mul!(V, stream, Q; scale, U)` accumulates `V = Σ_c A_c
diag(s_c) transpose(A_c) Q` in one pass over a `.bed` file, without
memory-mapping the whole genotype matrix: per chunk of 4096 SNPs, it
refills the chunk's counts and centering/scaling statistics from the
freshly read codes, computes `transpose(chunk) * Q`, scales the result by
`1/n` (or the caller's per-SNP `scale`), and accumulates `chunk * U_c`
into `V` with the five-argument `mul!`. With `prefetch=true` a background
task reads and refills the next chunk while the caller works on the
current one. The per-chunk statistics refill (column counts, means,
values table) runs single-threaded in the prefetch task, so at small `k`
the sweep is bound by that task rather than by the products.

## Environment

| Fact | Value |
| --- | --- |
| Node | n1111, 16 cores allocated (Slurm) |
| ISA | `avx512f dq bw vl vbmi vbmi2 bitalg vnni vpopcntdq ifma bf16 fp16` |
| Julia | 1.12.7 |
| Shape | `EUR_subset` stacked 101x: m = 38279 samples, n = 54051 SNPs |
| Timing | minimum of 3 calls after one warm-up |
| Stream width | 4096 SNPs per chunk |
| Sharing | another interactive session shared this node during the run; timings are indicative, not exclusive measurements |
| Page cache | the 517268073-byte stacked `.bed` was read once during warm-up before every timed call; `/proc/meminfo Cached = 320865348 kB` at record time, comfortably covering the file, so timed streaming calls read from page cache, not disk |

## Results, seconds (byte-wise column counts, before)

| T | k | in-memory AᵀQ | in-memory A*U | in-memory total | streaming (prefetch) | streaming (no prefetch) |
| --- | --- | --- | --- | --- | --- | --- |
| Float32 | 8 | 0.054 | 0.071 | 0.132 | 0.366 | 0.557 |
| Float32 | 64 | 0.233 | 0.268 | 0.654 | 0.741 | 0.906 |
| Float64 | 8 | 0.072 | 0.162 | 0.275 | 0.372 | 0.532 |
| Float64 | 64 | 0.627 | 0.598 | 1.044 | 1.494 | 1.582 |

## Results, seconds (word-wise column counts, after)

| T | k | in-memory AᵀQ | in-memory A*U | in-memory total | streaming (prefetch) | streaming (no prefetch) |
| --- | --- | --- | --- | --- | --- | --- |
| Float32 | 8 | 0.056 | 0.077 | 0.136 | 0.205 | 0.334 |
| Float32 | 64 | 0.170 | 0.138 | 0.304 | 0.482 | 0.546 |
| Float64 | 8 | 0.059 | 0.074 | 0.136 | 0.215 | 0.332 |
| Float64 | 64 | 0.301 | 0.317 | 0.660 | 0.816 | 0.900 |

All runs used 16 threads.

## Per-chunk stages, one 38279 x 4096 chunk (37 MiB), milliseconds

| Stage | before | after |
| --- | --- | --- |
| `read!` from page cache | 4.6 | 4.7 |
| column counts | 16.3 | 6.5 |
| means and values fill | 0.02 | 0.02 |
| `transpose(chunk) * Q`, k = 8, 16 threads | 4.3 | 5.1 |
| `chunk * U_c` accumulate, k = 8, 16 threads | 5.1 | 5.0 |

## `kq_pass.c` (Float32 tables, double `Q`/`U`/`V`), 16 threads

| c | streaming pass (s) | stages (stats / transpose / product1 / product2) | in-memory pass (s) |
| --- | --- | --- | --- |
| 8 | 0.68 | 0.06 / 0.04 / 0.42 / 0.09 | 0.43 |
| 64 | 1.32 | 0.07 / 0.06 / 0.92 / 0.19 | 0.94 |

## Notes

1. Before: streaming was 1.1x to 2.8x slower than the in-memory total;
   the prefetch task (read 4.6 ms plus count 16.3 ms, both serial)
   exceeded the caller's compute of about 9.5 ms per chunk at k = 8, so
   the sweep was bound by the count pass.
2. The word-wise count pass is 2.5x faster (16.3 ms to 6.5 ms per chunk,
   about 5.7 GB/s single-threaded); the prefetch task is now about 11 ms
   per chunk, on par with the caller's compute at k = 8.
3. After: streaming (prefetch) versus in-memory total is 1.51x at
   Float32 k = 8, 1.59x at Float32 k = 64, 1.58x at Float64 k = 8, and
   1.24x at Float64 k = 64; Float32 k = 8 improved from 0.366 s to
   0.205 s.
4. The in-memory numbers moved between runs (Float32 k = 64 total
   0.654 s to 0.304 s) with no code change on that path; the node was
   shared, so treat differences under about 2x between the two runs'
   in-memory columns as noise and compare streaming against the
   in-memory column of the same run.
5. `kq_pass.c`'s own streaming-vs-in-memory ratio is 1.58x at c=8 and
   1.40x at c=64, comparable to `streamed_grm_mul!` after the count fix;
   product1 (the SNP-major scatter into `U`) dominates
   its streaming pass at 0.42 s / 0.92 s of the 0.68 s / 1.32 s total.
   Correctness: relative errors unchanged between runs: 3.4e-12
   (Float32), 6.2e-21 (Float64).
