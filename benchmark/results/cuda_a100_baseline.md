# Baseline CUDA kernels on an A100

Baseline ("before") timings for the current `CuSnpArray` kernels in
`ext/SnpArraysCUDAExt.jl`, which handle vectors only (k = 1), against the CPU
`SnpLinAlg` and CuBLAS on the materialized dense Float32 matrix.

## Environment

| Fact | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB |
| CPU threads | 16 (`Threads.nthreads()`, Slurm job 299170) |
| CUDA.jl | 5.11.3 |
| Julia | 1.12 (`Pkg` 1.12.1) |
| Shape | `data/EUR_subset.bed` (379 x 54051) stacked 100 times: m = 37900, n = 54051 |
| Model | `ADDITIVE_MODEL`, `T = Float32`, `center = scale = true`, `impute = false` |
| Timing | GPU: `CUDA.@elapsed` after one warm-up call, minimum of 5; CPU: `@elapsed` after one warm-up call, minimum of 5 |
| Throughput | G elem/s = m * n * k / time / 1e9 |
| Peak host memory | 10.09 GiB (`Sys.maxrss()`) |
| Wall time | 1 min 20 s (including compilation and dense materialization) |
| Log | `$SCRATCH/cudaext_check/baseline.log` |

`A*X` has `X` of size n x k; `A'*X` is `mul!(out, transpose(A), Y)` with `Y`
of size m x k. k = 1 uses vectors. The script makes no correctness
comparison.

## Results

| implementation | direction | k | time (ms) | G elem/s |
|---|---|---|---|---|
| CuSnpArray (current) | A*X | 1 | 26.029 | 78.702 |
| CuSnpArray (current) | A'*X | 1 | 13.886 | 147.520 |
| CPU SnpLinAlg | A*X | 1 | 73.642 | 27.818 |
| CPU SnpLinAlg | A'*X | 1 | 46.535 | 44.021 |
| CPU SnpLinAlg | A*X | 8 | 174.222 | 94.065 |
| CPU SnpLinAlg | A'*X | 8 | 72.439 | 226.235 |
| CPU SnpLinAlg | A*X | 32 | 226.477 | 289.447 |
| CPU SnpLinAlg | A'*X | 32 | 206.238 | 317.852 |
| CPU SnpLinAlg | A*X | 128 | 593.746 | 441.623 |
| CPU SnpLinAlg | A'*X | 128 | 859.564 | 305.052 |
| CuBLAS dense | A*X | 1 | 7.229 | 283.360 |
| CuBLAS dense | A'*X | 1 | 5.774 | 354.765 |
| CuBLAS dense | A*X | 8 | 6.295 | 2603.573 |
| CuBLAS dense | A'*X | 8 | 9.841 | 1665.366 |
| CuBLAS dense | A*X | 32 | 7.869 | 8330.078 |
| CuBLAS dense | A'*X | 32 | 8.803 | 7446.395 |
| CuBLAS dense | A*X | 128 | 27.496 | 9536.221 |
| CuBLAS dense | A'*X | 128 | 29.984 | 8745.145 |

## Command

```sh
KAIMON_BYPASS=1 JULIA_DEPOT_PATH="/u/scratch/b/bhchau/julia-cuda-depot:" \
    julia --project=/u/scratch/b/bhchau/cudaext_check/bench_env -t 16 \
    lib/SnpArrays/benchmark/cuda_linalg.jl
```

`bench_env` is a throwaway project with `SnpArrays` developed from
`lib/SnpArrays` plus `CUDA` and `Adapt`.
