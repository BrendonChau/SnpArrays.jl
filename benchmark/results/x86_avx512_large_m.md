# Streamed transpose(A) * Y at large m on x86 AVX-512

`SnpLinAlgStream` reads each chunk of a plain `.bed` file with `readers`
concurrent tasks (keyword `readers`, default `_default_readers() = max(1,
min(8, Threads.nthreads() ÷ 2))`; `readers = 1` is the single sequential
`read!`; compressed files always use one), and `_column_counts!` splits the
columns across tasks via `_task_axis_step` when `s.data` holds at least
`COUNT_TASK_MIN_BYTES = 1 << 20` bytes. The streamed whole-file `transpose(A)
* Y` at k = 8 is compared with the memory-mapped `SnpLinAlg` product under a
24 GiB cgroup that charges page cache, so every whole-file pass is cold and
only within-process ratios are meaningful.

## Environment

| Fact | Value |
| --- | --- |
| Node | n1197, 16 cores (Slurm job 280585) |
| CPU | Intel Xeon 6736P (`Sys.CPU_NAME = graniterapids`) |
| ISA | `avx512f dq ifma cd bw vl bf16 vbmi vbmi2 vnni bitalg vpopcntdq fp16`, `VECTOR_BYTES = 64` |
| Julia | 1.12.7, `-t 16` for timing runs |
| Shape | `genotypes/synthetic_v1_chr-21.bed`: m = 1008000 samples, n = 109673 SNPs, 27637596003 bytes |
| cgroup | v2, `memory.max = 25769803776` (24 GiB), `memory.peak` already at 24.00 GiB at run start; page cache is charged |
| File system | NFS (`172.16.108.235:/u/home/b/bhchau/projects`) |
| Stream | width 4096 SNPs (27 chunks, last 3177 wide), T = Float32, k = 8, `center = scale = impute = true`, prefetch on |
| Readers | default 8 at 16 threads; compared with 1 |
| Timing | one cold pass per line, two rounds, mapped/stream(1)/stream(8) alternating; no warm-up possible |
| Logs | `logs/` is `$SCRATCH/SnpArraysBenchmarks/logs/` |

## Whole-file passes

Source: `logs/stream_large_m_n1197_20260921-200851.log`.

| Round | Pass | Wall (s) | mem.current before -> after (GiB) | mem.peak (GiB) | fincore before -> after (GiB) |
| --- | --- | --- | --- | --- | --- |
| 1 | mapped scan | 54.772 | 18.27 -> 24.00 | 24.00 | 9.38 -> 17.54 |
| 1 | mapped mul! | 36.965 | 24.00 -> 24.00 | 24.00 | 17.44 -> 17.40 |
| 1 | streamed readers=1 | 29.446 | 24.00 -> 24.00 | 24.00 | 17.40 -> 15.74 |
| 1 | streamed readers=8 | 13.098 | 24.00 -> 24.00 | 24.00 | 15.74 -> 14.09 |
| 2 | mapped scan | 51.173 | 24.00 -> 23.99 | 24.00 | 14.09 -> 14.22 |
| 2 | mapped mul! | 46.095 | 24.00 -> 24.00 | 24.00 | 14.22 -> 14.21 |
| 2 | streamed readers=1 | 31.172 | 24.00 -> 24.00 | 24.00 | 14.20 -> 14.20 |
| 2 | streamed readers=8 | 12.041 | 24.00 -> 24.00 | 24.00 | 14.19 -> 14.21 |

Initial header line: mem.current 14.23 GiB, mem.peak 24.00 GiB, fincore
9.38 GiB.

Ratios: round 1 streamed(8)/mapped mul! 0.354, streamed(8)/streamed(1)
0.445; round 2 streamed(8)/mapped mul! 0.261, streamed(8)/streamed(1) 0.386.
Medians across the two rounds: P1 0.308, P3 0.416.

Correctness: rel err 0.0 for both reader counts in both rounds, tol
0.0038299256.

## Per-chunk stages

First chunk, 1008000 x 4096 = 1032192000 bytes; min/median/max of 5 calls.
Source: `logs/stream_large_m_n1197_20260921-200851.log`.

| Stage | Min (s) | Median (s) | Max (s) |
| --- | --- | --- | --- |
| read+refill readers=1 | 0.1788 | 0.2020 | 1.3619 |
| read+refill readers=8 | 0.0392 | 0.0422 | 0.0443 |
| count (`_column_counts!`) | 0.0141 | 0.0210 | 0.0218 |
| refill (`_refill_statistics!`) | 0.0152 | 0.0167 | 0.0195 |
| AtY (`mul!(piece, transpose(chunk), Y)`) | 0.1125 | 0.1250 | 0.1262 |
| AU (`mul!(V, chunk, piece)`) | 0.1016 | 0.1028 | 0.3540 |

The five calls re-read the same chunk, so calls after the first hit the page
cache; the readers=1 max (1.3619 s) is the cold first call, the min/median
are warm.

## Small-m regression

Source: `logs/streaming_101_n1197_20260921-201428.log`
(`benchmark/streaming.jl 101`, `EUR_subset` stacked 101x, m = 38279, n =
54051, page-cache warm as in `x86_avx512_streaming.md`).

| T | k | mem total (s) | stream(pf) (s) | ratio | recorded ratio (`x86_avx512_streaming.md`, after table) | ratio / recorded |
| --- | --- | --- | --- | --- | --- | --- |
| Float32 | 8 | 0.144 | 0.215 | 1.493 | 1.51 (0.205/0.136) | 0.99 |
| Float32 | 64 | 0.305 | 0.458 | 1.502 | 1.59 (0.482/0.304) | 0.95 |
| Float64 | 8 | 0.152 | 0.214 | 1.408 | 1.58 (0.215/0.136) | 0.89 |
| Float64 | 64 | 0.638 | 0.798 | 1.251 | 1.24 (0.816/0.660) | 1.01 |

Pass rule: no ratio exceeds the recorded ratio by more than 1.1x. Passed
(max ratio/recorded 1.01).

## Pass rules

| Rule | Condition | Observed | Result |
| --- | --- | --- | --- |
| P1 | median streamed(default)/mapped mul! wall time at k = 8 <= 1.20 | 0.308 | pass |
| P2 | chunk `count` stage <= 0.03 s | 0.0210 s (median) | pass |
| P3 | streamed(default)/streamed(readers=1), same round, <= 0.65 | 0.416 (median; 0.445, 0.386) | pass |
| P4 | every correctness check passes | rel err 0.0 in all four streamed passes; `test/runtests.jl` green at `-t 1`, `-t 4`, `-t 16` (`logs/runtests_t1_n1197_20260921-200230.log`, `logs/runtests_t4_n1197_20260921-200432.log`, `logs/runtests_t16_n1197_20260921-200625.log`, each 51 "Test Summary" lines, no failures) | pass |

Small-m regression: pass (max ratio/recorded 1.01).

## Notes

1. Reader count: `_default_readers()` halves the thread count and caps at 8
   because a blocking `unsafe_read` parks a default-pool thread, and the
   prefetch task overlaps the reads with the 16-thread products. At 8
   readers the streamed pass is 2.3x to 2.6x faster than at 1 (P3), and
   2.8x to 3.8x faster than the mapped `mul!`, whose 16 page-faulting
   threads pull less bandwidth than 8 sequential 128 MB reads. `readers`
   is a keyword, so 2 and 4 can be timed without a rebuild; they were not
   needed since P1 and P3 passed.
2. `_row_counts!` stays serial: it accumulates into `counts[:, row]`
   shared across columns, so a column split would race; the stream never
   calls it (`_refill_statistics!` reaches only the column counts).
3. Handle leak: abandoning an iteration midway leaks the open handle, as
   before; `readers > 1` multiplies that by `readers` on plain `.bed`
   files.
4. The mapped scan (54.8 s, 51.2 s) is a whole-file cold pass of its own;
   the mapped operator therefore costs two passes (scan plus product) per
   product where the stream costs one, on top of the P1 ratio.
5. `memory.current` sits at the 24 GiB cap throughout and `fincore` drifts
   between 14 and 18 GiB of the 25.7 GiB file, so no pass ran from a
   resident file; the mapped scan raised residency from 9.4 to 17.5 GiB,
   and the streamed passes lowered it.
6. Deferred: the transpose-kernel `row_step` tiling change on the compute
   side; per-chunk `AtY` is 0.125 s against a 0.042 s parallel read, and
   27 chunks give about 3.4 s of product inside a 12 s to 13 s pass, so
   the pass is bound by I/O, not by the kernel.
