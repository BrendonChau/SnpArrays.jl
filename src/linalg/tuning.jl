"""
    SIMD_FLOAT

Element types the SIMD kernels accept: `Float32` and `Float64`.
"""
const SIMD_FLOAT = Union{Float32, Float64}

"""
    VECTOR_BYTES

SIMD register width in bytes used by the register-tiled kernels: 64 on
x86_64 hosts whose ISA includes the AVX-512 tier, 32 otherwise (AVX2, or
NEON packed as two 128-bit registers). Set in `__init__` from
`_detect_vector_bytes()`; the first use of each `(T, W)` pair compiles
in-process (no precompile workload). On a Xeon 6736P, 64 measured 1.27x to
1.92x faster than 32 for every `k` at or above 8, and LLVM emitted zmm
registers rather than splitting the 512-bit vectors; see
`benchmark/results/x86_avx512_n1183.md`.
"""
const VECTOR_BYTES = Ref{Int}(32)

"""
    _detect_vector_bytes() -> Int

Return 64 on x86_64 hosts whose CPU ISA includes AVX-512, else 32. Relies
on `Base.BinaryPlatforms` internals (`arch_march_isa_mapping`,
`CPUID.cpu_isa()`) that exist in Julia 1.12 but are not part of the public
API, so the fallback to 32 is mandatory if they are unavailable or raise
an error.
"""
function _detect_vector_bytes()
    Sys.ARCH === :x86_64 || return 32
    try
        isa_map = Dict(Base.BinaryPlatforms.arch_march_isa_mapping["x86_64"])
        return isa_map["avx512"] <= Base.BinaryPlatforms.CPUID.cpu_isa() ?
               64 : 32
    catch
        return 32
    end
end

"""
    L1_OUT_BUDGET

Byte budget, per task, for the `A*x` output block (read-modified-written
once per SNP column) and the `transpose(A)*x` rhs block (re-read once per
column), so it stays resident in a 32-48 KB-class L1 data cache with room
left for the packed genotype bytes. The best swept alternative on a Xeon
6736P was 1.13x faster in one of four cells and no better in the rest.
"""
const L1_OUT_BUDGET = 16_384

"""
    L2_PANEL_BUDGET

Byte budget, per task, for the transposed rhs slab (`k_padded x
column_step * sizeof(T)`) of one inner block, so it fits a 256 KB-class
private L2 slice. Enlarging it to match the 2 MB private L2 of a Xeon
6736P measured neutral for `A*X` and worse for `transpose(A)*X`.
"""
const L2_PANEL_BUDGET = 262_144

"""
    PACKED_LINES_BUDGET

Byte budget for the packed cache lines touched by one `A*X` row tile, one
64-byte line per SNP column of the block, re-touched by every following
row tile, so they stay L1-resident. The `column_step` it caps sits on a
flat part of the measured response curve on a Xeon 6736P.
"""
const PACKED_LINES_BUDGET = 32_768

"""
    LOOKUP_ROW_TILE

Samples per inner tile of the lookup-table `A*X` kernel; the tile's
row-major partial sums (`LOOKUP_ROW_TILE` times the rhs slice width) stay
L1-resident while every 4-SNP block of a group is gathered. Initial value
from `kq_pass.c`.
"""
const LOOKUP_ROW_TILE = 512

"""
    LOOKUP_CHUNK_SNPS

SNPs per chunk of the lookup-table `A*X` kernel (a multiple of 4). One
chunk's 256-row tables for all `k` rhs columns are built once, shared by
every gather task, and cost `LOOKUP_CHUNK_SNPS / 4 * 256 * k_padded *
sizeof(T)` bytes. Initial value from `kq_pass.c`.
"""
const LOOKUP_CHUNK_SNPS = 1024

"""
    LOOKUP_GROUP_BUDGET

Bytes of lookup tables swept per inner sample tile, so a group of 4-SNP
blocks stays L2-resident across the tile. Initial value from `kq_pass.c`.
"""
const LOOKUP_GROUP_BUDGET = 1 << 20

"""
    LOOKUP_MIN_ROWS

Fewest samples for which `A*X` uses the lookup-table kernel; below it the
`256 * k` table build per 4-SNP block is not amortised.
"""
const LOOKUP_MIN_ROWS = 2048

"""
    LOOKUP_MIN_RHS

Fewest rhs columns for which `A*X` uses the lookup-table kernel, as
`(Float32, Float64)`. Set in `__init__` from `_detect_lookup_min_rhs()`.
"""
const LOOKUP_MIN_RHS = Ref{NTuple{2, Int}}((4, 4))

"""
    _detect_lookup_min_rhs() -> NTuple{2, Int}

Return the `(Float32, Float64)` rhs thresholds for the lookup-table `A*X`
kernel: `(32, typemax(Int))` on aarch64, else `(4, 4)`. On an Apple M1 at 4
threads (m = 8192, n = 49152) the lookup kernel was 1.5x (Float32) and 1.3x
(Float64) slower than the register-tiled kernel at k = 8, about 1.1x faster
for Float32 at k = 32 and 128, and never faster for Float64; `typemax(Int)`
turns the lookup kernel off.
"""
function _detect_lookup_min_rhs()
    return Sys.ARCH === :aarch64 ? (32, typemax(Int)) : (4, 4)
end

"""
    _lookup_min_rhs(::Type{T}) -> Int

Return the fewest rhs columns for which `A*X` with element type `T` uses the
lookup-table kernel, from `LOOKUP_MIN_RHS[]`.
"""
_lookup_min_rhs(::Type{Float32}) = LOOKUP_MIN_RHS[][1]
_lookup_min_rhs(::Type{Float64}) = LOOKUP_MIN_RHS[][2]

"""
    LOOKUP_SLICE_MAX

Most rhs columns in one slice of the lookup-table `A*X` kernel.
"""
const LOOKUP_SLICE_MAX = 32

"""
    _uses_lookup_kernel(::Type{T}, m::Int, k::Int) -> Bool

Return whether `A*X` with element type `T`, `m` samples and `k` rhs columns
runs the lookup-table kernel rather than the register-tiled kernel.
"""
function _uses_lookup_kernel(
    ::Type{T},
    m::Int,
    k::Int,
) where T <: SIMD_FLOAT
    return m >= LOOKUP_MIN_ROWS && k >= _lookup_min_rhs(T)
end

"""
    _vector_width(::Type{T}) -> Int

Return the number of `T` lanes in a `Vec{W,T}` register of width
`VECTOR_BYTES[]`.
"""
_vector_width(::Type{T}) where T = VECTOR_BYTES[] ÷ sizeof(T)

"""
    _rhs_width(::Type{T}, k::Int) -> Int

Return the `Vec` lane count for the rhs tile: the largest power of two that
is at most `_vector_width(T)` and does not exceed `k`, floored at 4. A `k`
narrower than the register width would otherwise leave lanes idle in every
tile.
"""
function _rhs_width(::Type{T}, k::Int) where T
    width = _vector_width(T)
    while width > 4 && k < width
        width >>= 1
    end
    return width
end

"""
    _lookup_slice_shape(::Type{T}, k::Int) -> (lanes, vectors)

Return the `Vec` lane count and the vectors per rhs slice of the lookup-table
`A*X` kernel for `k >= 1` rhs columns: the fewest equal slices of at most
`LOOKUP_SLICE_MAX` columns, each rounded up to whole vectors.
"""
function _lookup_slice_shape(::Type{T}, k::Int) where T
    lanes = _rhs_width(T, k)
    slices = cld(k, LOOKUP_SLICE_MAX)
    vectors = cld(cld(k, slices), lanes)
    return lanes, vectors
end

"""
    _register_tile_shape(::Type{T}) -> (MR, NR)

Return the register-tile shape `(MR, NR)` for element type `T`: `NR =
2 * _vector_width(T)` rhs lanes and `MR` accumulator rows. Register
budget: AVX2 Float32 gives `MR=6, NR=16` (12 accumulators + 2 rhs + 1
broadcast = 15 ymm registers); AVX2 Float64 gives `MR=6, NR=8` (15 ymm);
AVX-512 gives `MR=8` (19 zmm); NEON packs `W=8` Float32 lanes as two q
registers, using 30 of 32 registers. `transpose(A)*X` takes its tile height
from `ATX_TILE_COLUMNS[]` instead of `MR`.
"""
function _register_tile_shape(::Type{T}) where T
    mr = VECTOR_BYTES[] == 64 ? 8 : 6
    return mr, 2 * _vector_width(T)
end

"""
    ATX_TILE_COLUMNS

SNP columns per register tile of `transpose(A)*X`. Set in `__init__` from
`_detect_atx_tile_columns(VECTOR_BYTES[])`.
"""
const ATX_TILE_COLUMNS = Ref{Int}(6)

"""
    _detect_atx_tile_columns(vector_bytes::Int) -> Int

Return the `transpose(A)*X` tile height: 8 for 64-byte vectors, 2 on x86_64
with 32-byte vectors, else 6. The 16 ymm registers of AVX2 cannot hold the
6-column tile, which also keeps one broadcast per column live: on a Haswell
Xeon at 8 threads (m = 50000, n = 660966, k = 128) 6 columns spilled 10
vectors per sample, 2 spilled none and measured 1.33x faster.
"""
function _detect_atx_tile_columns(vector_bytes::Int)
    vector_bytes == 64 && return 8
    return Sys.ARCH === :x86_64 ? 2 : 6
end

"""
    _round_down_decode(x::Int) -> Int

Return `x` rounded down to a multiple of `DECODE_WIDTH` and floored at
`DECODE_WIDTH`.
"""
function _round_down_decode(x::Int)
    return max(DECODE_WIDTH, (x ÷ DECODE_WIDTH) * DECODE_WIDTH)
end

"""
    _panel_step(::Type{T}, k::Int) -> Int

Return the number of samples or SNPs whose transposed rhs slab, `k` capped at
256 and padded to a whole register tile, fits `L2_PANEL_BUDGET`, rounded down
to a multiple of `DECODE_WIDTH`. The caller clamps it to its own bounds.
"""
function _panel_step(::Type{T}, k::Int) where T <: AbstractFloat
    _, nr = _register_tile_shape(T)
    k_padded = cld(max(min(k, 256), 1), nr) * nr
    return _round_down_decode(L2_PANEL_BUDGET ÷ (k_padded * sizeof(T)))
end

"""
    _rhs_step(k::Int) -> Int

Return the rhs columns per task, `k` capped at 256 (the register tile handles
all rhs blocking below 256) and at least 1.
"""
_rhs_step(k::Int) = max(min(k, 256), 1)

"""
    _snparray_ax_row_step(::Type{T}, m::Int) -> row_step

Return the samples per task of `A*x` for `m` samples. `row_step` is a
multiple of `DECODE_WIDTH`, at least `DECODE_WIDTH`, and keeps the output
block within `L1_OUT_BUDGET` across the SNP column loop; the vector kernel
needs no inner column blocking, so each task sweeps every SNP. `m == 0` does
not error.
"""
function _snparray_ax_row_step(::Type{T}, m::Int) where T <: AbstractFloat
    return _round_down_decode(_task_axis_step(m, L1_OUT_BUDGET ÷ sizeof(T)))
end

"""
    _snparray_atx_steps(::Type{T}, n::Int) -> (row_step, column_step)

Return the task tile sizes of `transpose(A)*x` for `n` SNPs. `column_step`
is task-partitioned over SNPs and at least 1; `row_step` is a multiple of
`DECODE_WIDTH`, at least `DECODE_WIDTH`, and keeps the rhs slice within
`L1_OUT_BUDGET` across the column loop. `n == 0` does not error.
"""
function _snparray_atx_steps(::Type{T}, n::Int) where T <: AbstractFloat
    row_step = _round_down_decode(L1_OUT_BUDGET ÷ sizeof(T))
    column_step = max(_task_axis_step(n, 2048), 1)
    return row_step, column_step
end

"""
    _snparray_AX_steps(::Type{T}, m::Int, k::Int)
        -> (row_step, column_step, rhs_step)

Return the task tile sizes of `A*X` for `m` samples and `k` rhs columns.
`rhs_step` is `k` capped at 256; `column_step` sizes the transposed rhs slab
of one column block to `L2_PANEL_BUDGET` and the packed cache lines it
touches to `PACKED_LINES_BUDGET`; `row_step` sizes the packed genotype block,
re-read once per rhs tile, to `4 * L2_PANEL_BUDGET / column_step` bytes (the
`A*X` accumulators live in registers, so `row_step` is not bound by
`L1_OUT_BUDGET`). `row_step` is a multiple of `DECODE_WIDTH` and at least
`DECODE_WIDTH`; `column_step` and `rhs_step` are at least 1. `m == 0` does
not error. A matrix product with one rhs column still runs the register-tiled
kernel, so it uses this rule at `k = 1`.
"""
function _snparray_AX_steps(
    ::Type{T},
    m::Int,
    k::Int,
) where T <: AbstractFloat
    column_step = clamp(_panel_step(T, k), 64, PACKED_LINES_BUDGET ÷ 64)
    row_step = _round_down_decode(
        _task_axis_step(m, 4 * L2_PANEL_BUDGET ÷ column_step),
    )
    return row_step, max(column_step, 1), _rhs_step(k)
end

"""
    _snparray_AtX_steps(::Type{T}, n::Int, k::Int)
        -> (row_step, column_step, rhs_step)

Return the task tile sizes of `transpose(A)*X` for `n` SNPs and `k` rhs
columns. `rhs_step` is `k` capped at 256; `row_step` sizes the transposed rhs
slab over samples to `L2_PANEL_BUDGET`; `column_step` is task-partitioned
over SNPs. `row_step` is a multiple of `DECODE_WIDTH` and at least
`DECODE_WIDTH`; `column_step` and `rhs_step` are at least 1. `n == 0` does
not error.

Across 106 swept configurations on a Xeon 6736P the step functions of the
four products were within 1.15x of the best in every operation and element
type; see `benchmark/results/x86_avx512_n1183.md`.
"""
function _snparray_AtX_steps(
    ::Type{T},
    n::Int,
    k::Int,
) where T <: AbstractFloat
    row_step = clamp(_panel_step(T, k), 64, 4096)
    column_step = max(_task_axis_step(n, 2048), 1)
    return row_step, column_step, _rhs_step(k)
end
