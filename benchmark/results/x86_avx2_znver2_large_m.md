# Streamed transpose(A) * Y and A * U at large m on AMD Zen 2

The `benchmark/stream_large_m.jl` whole-file passes rerun on an AMD EPYC
node under a 64 GiB cgroup, which holds the whole 25.7 GiB file in page
cache. Every pass after the first mapped scan therefore reads from memory,
the opposite of the cold, 24 GiB-capped conditions in
`x86_avx512_large_m.md`, so the two records measure different regimes and
their ratios should not be compared. This run also adds the forward product
`A * U` (`run_forward_rounds`).

## Environment

| Fact | Value |
| --- | --- |
| Node | g15, Slurm job 299170, 16 CPUs allocated |
| CPU | AMD EPYC 7542, 32 cores, 1 socket, 4 NUMA nodes, 128 MiB L3 (`Sys.CPU_NAME = znver2`) |
| ISA | AVX2, no AVX-512; `VECTOR_BYTES = 32` |
| Julia | 1.12.7, `-t 16` |
| Shape | `genotypes/synthetic_v1_chr-21.bed`: m = 1008000 samples, n = 109673 SNPs, 27637596003 bytes |
| cgroup | v2, `memory.max = 68719476736` (64 GiB); page cache is charged |
| File system | NFS (`172.16.108.213:/u/home/b/bhchau`) |
| Stream | width 4096 SNPs (27 chunks), T = Float32, k = 8, `center = scale = impute = true`, prefetch on |
| Readers | default 8 at 16 threads; compared with 1 |
| Page cache | `fincore` is not installed on g15, so the fincore columns read -1 (printed as -0.00 GiB) |
| Logs | `logs/` is `$SCRATCH/SnpArraysBenchmarks/logs/` |

## Whole-file passes, first run

Source: `logs/stream_large_m_g15_20260929-203443.log`. Round 1's mapped scan
is the only cold pass: it raised `memory.current` from 15.24 to 42.69 GiB
as the file entered page cache.

| Round | Pass | Wall (s) |
| --- | --- | --- |
| 1 | mapped scan | 71.500 |
| 1 | mapped mul! | 6.778 |
| 1 | streamed readers=1 | 9.906 |
| 1 | streamed readers=8 | 9.321 |
| 2 | mapped scan | 1.120 |
| 2 | mapped mul! | 5.800 |
| 2 | streamed readers=1 | 8.636 |
| 2 | streamed readers=8 | 8.753 |

Medians: streamed(8)/mapped 1.442, streamed(8)/streamed(1) 0.977. Peak
`memory.current` 46.66 GiB.

## Whole-file passes, second run with A * U

Source: `logs/stream_large_m_g15_20260929-203933.log`. The file was still
resident from the first run, so every pass is warm.

| Product | Round | mapped (s) | streamed r1 (s) | streamed r8 (s) | r8/mapped | r8/r1 |
| --- | --- | --- | --- | --- | --- | --- |
| `transpose(A) * Y` | 1 | 5.849 | 8.615 | 8.190 | 1.400 | 0.951 |
| `transpose(A) * Y` | 2 | 4.123 | 7.648 | 7.378 | 1.790 | 0.965 |
| `A * U` | 1 | 7.901 | 9.692 | 9.533 | 1.207 | 0.984 |
| `A * U` | 2 | 7.642 | 8.985 | 8.716 | 1.140 | 0.970 |

Medians of r8/mapped: 1.595 for `transpose(A) * Y`, 1.173 for `A * U`.
Correctness: rel err 0.0 in all eight streamed passes, tol 0.0038299256.

## Per-chunk stages

First chunk, min/median/max of 5 calls, from the second run.

| Stage | Min (s) | Median (s) | Max (s) |
| --- | --- | --- | --- |
| read+refill readers=1 | 0.1987 | 0.2068 | 0.2144 |
| read+refill readers=8 | 0.1094 | 0.1106 | 0.1193 |
| count | 0.0368 | 0.0370 | 0.0371 |
| refill | 0.0335 | 0.0388 | 0.0409 |
| AtY | 0.1926 | 0.2160 | 0.2194 |
| AU | 0.3278 | 0.3317 | 0.3447 |

## Notes

1. With the file resident, the mapped operator wins: streaming is 1.4x to
   1.8x slower for `transpose(A) * Y` and 1.1x to 1.2x slower for `A * U`.
   The stream still pays for reading, counting, and refilling each chunk,
   which the mapped operator did once in its scan.
2. Eight readers halve the per-chunk read (0.207 s to 0.111 s) but gain
   only 2% to 5% per pass, because the passes are bound by the products,
   not the reads. The per-chunk `AtY` (0.216 s) and `AU` (0.332 s) are
   1.7x and 3.2x the AVX-512 figures.
3. `A * U` is the slower product on this node, so the fixed per-chunk
   overhead is a smaller share of its pass and its streaming penalty is
   lower.
4. A comparison in the cold regime on this node needs a cgroup smaller than
   the file, as in `x86_avx512_large_m.md`.
