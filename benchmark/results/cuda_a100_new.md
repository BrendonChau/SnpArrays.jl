# Packed-word CUDA kernels on an A100

Timings for the packed-word `CuSnpArray` kernels (16 samples per `UInt32`,
per-column lookup table) against the CPU `SnpLinAlg` and CuBLAS on the
materialized dense Float32 matrix. The old byte-per-sample kernels no longer
exist; their k = 1 times come from `cuda_a100_baseline.md`.

## Environment

| Fact | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB |
| CPU threads | 16 (Slurm job 299170) |
| CUDA.jl | 6.1.0 |
| Julia | 1.12 |
| Shape | `data/EUR_subset.bed` (379 x 54051) stacked 100 times: m = 37900, n = 54051 |
| Model | `ADDITIVE_MODEL`, `T = Float32`, `center = scale = true`, `impute = false` |
| Timing | warm-up of at least 3 calls and 0.5 s, then minimum of 5; GPU via `CUDA.@elapsed` |
| Peak host memory | 10.05 GiB (`Sys.maxrss()`) |
| Log | `$SCRATCH/cudaext_check/bench_new.log` |

Plain FP32 SIMT kernels, no tensor cores. k = 1 uses vectors.

## Results

| implementation | direction | k | time (ms) | G elem/s |
|---|---|---|---|---|
| CuSnpArray (new) | A*X | 1 | 1.994 | 1027.489 |
| CuSnpArray (new) | A'*X | 1 | 2.330 | 879.350 |
| CuSnpArray (new) | A*X | 8 | 5.411 | 3028.797 |
| CuSnpArray (new) | A'*X | 8 | 6.447 | 2541.957 |
| CuSnpArray (new) | A*X | 32 | 12.041 | 5444.056 |
| CuSnpArray (new) | A'*X | 32 | 11.299 | 5801.763 |
| CuSnpArray (new) | A*X | 128 | 45.107 | 5813.090 |
| CuSnpArray (new) | A'*X | 128 | 41.313 | 6346.923 |
| CPU SnpLinAlg | A*X | 1 | 87.807 | 23.330 |
| CPU SnpLinAlg | A'*X | 1 | 49.571 | 41.325 |
| CPU SnpLinAlg | A*X | 8 | 190.603 | 85.981 |
| CPU SnpLinAlg | A'*X | 8 | 83.609 | 196.011 |
| CPU SnpLinAlg | A*X | 32 | 232.050 | 282.495 |
| CPU SnpLinAlg | A'*X | 32 | 241.696 | 271.221 |
| CPU SnpLinAlg | A*X | 128 | 676.539 | 387.579 |
| CPU SnpLinAlg | A'*X | 128 | 978.796 | 267.893 |
| CuBLAS dense | A*X | 1 | 7.236 | 283.119 |
| CuBLAS dense | A'*X | 1 | 5.780 | 354.388 |
| CuBLAS dense | A*X | 8 | 6.301 | 2601.034 |
| CuBLAS dense | A'*X | 8 | 9.834 | 1666.406 |
| CuBLAS dense | A*X | 32 | 7.860 | 8339.846 |
| CuBLAS dense | A'*X | 32 | 8.807 | 7442.932 |
| CuBLAS dense | A*X | 128 | 27.527 | 9525.579 |
| CuBLAS dense | A'*X | 128 | 30.172 | 8690.535 |

## Comparison

Time ratio to CuBLAS below 1 means the packed kernel is faster. Bandwidth
counts the 512.1 MB of packed words; GFLOP/s is 2mnk / time.

| direction | k | new (ms) | vs old | new / CuBLAS | throughput |
|---|---|---|---|---|---|
| A*X | 1 | 1.994 | 13.05x faster (26.029 ms) | 0.28 | 257 GB/s |
| A'*X | 1 | 2.330 | 5.96x faster (13.886 ms) | 0.40 | 220 GB/s |
| A*X | 8 | 5.411 | - | 0.86 | 6058 GFLOP/s |
| A'*X | 8 | 6.447 | - | 0.66 | 5085 GFLOP/s |
| A*X | 32 | 12.041 | - | 1.53 | 10889 GFLOP/s |
| A'*X | 32 | 11.299 | - | 1.28 | 11604 GFLOP/s |
| A*X | 128 | 45.107 | - | 1.64 | 11626 GFLOP/s |
| A'*X | 128 | 41.313 | - | 1.37 | 12694 GFLOP/s |

The k = 1 kernels run far below the 1.5 TB/s memory bandwidth: they are
bound by the per-sample decode (3 selects per genotype) rather than by the
packed-word loads. For k >= 32 the register-blocked tiles reach about
11-13 TFLOP/s, 55-65% of the FP32 peak; CuBLAS reaches 16-17 TFLOP/s on the
16x larger dense matrix. Closing that gap is the later tensor-core phase.

## Tile configurations

`(BM, BN, BK, TM, TK)` for `A*X`, `(BM, BN, BK, TN, TK)` for `A'*X`;
reduction splits give about eight blocks per SM.

| k band | A*X | A'*X |
|---|---|---|
| k <= 8 | (256, 16, 8, 4, 4) | (64, 64, 8, 2, 2) |
| 8 < k <= 32 | (256, 16, 32, 8, 8) | (32, 128, 32, 4, 8) |
| k > 32 | (128, 16, 64, 8, 8) | (32, 128, 64, 8, 8) |

## Command

```sh
KAIMON_BYPASS=1 JULIA_DEPOT_PATH="/u/scratch/b/bhchau/julia-cuda-depot:" \
    julia --project=/u/scratch/b/bhchau/cudaext_check/bench_env -t 16 \
    lib/SnpArrays/benchmark/cuda_linalg.jl
```
