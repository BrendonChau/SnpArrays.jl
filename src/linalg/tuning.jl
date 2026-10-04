# Cache budgets and the block sizes the schedulers cut each product into.

"""
    SIMD_FLOAT

Element types the SIMD kernels accept: `Float32` and `Float64`.
"""
const SIMD_FLOAT = Union{Float32, Float64}

"""
    VECTOR_BYTES

SIMD register width in bytes used by the register-tiled kernels: 64 on
x86_64 hosts that report the AVX-512 features, 32 otherwise.
"""
const VECTOR_BYTES = Ref{Int}(32)

"""
    _detect_vector_bytes() -> Int

Return 64 on x86_64 hosts whose CPU reports the AVX-512 F, VL, BW, DQ and
CD features, else 32.
"""
function _detect_vector_bytes()
    Sys.ARCH === :x86_64 || return 32
    # `Base.BinaryPlatforms.CPUID` is not public API; fall back to 32.
    try
        cpuid = Base.BinaryPlatforms.CPUID
        features = (
            cpuid.JL_X86_avx512f,
            cpuid.JL_X86_avx512vl,
            cpuid.JL_X86_avx512bw,
            cpuid.JL_X86_avx512dq,
            cpuid.JL_X86_avx512cd,
        )
        return all(cpuid.test_cpu_feature, features) ? 64 : 32
    catch
        return 32
    end
end

# Re-read once per SNP column, so it stays in a 32-48 KB-class L1 data cache
# with room left for the packed genotype bytes.
"""
    L1_OUT_BUDGET

Byte budget, per task, for the `A*x` output block and the `transpose(A)*x`
rhs block.
"""
const L1_OUT_BUDGET = 16_384

# The slab is `k_padded x column_step * sizeof(T)` bytes and fits a 256
# KB-class private L2 slice.
"""
    L2_PANEL_BUDGET

Byte budget, per task, for the transposed rhs slab of one inner block.
"""
const L2_PANEL_BUDGET = 262_144

# Every following row tile re-touches the lines, so they stay L1-resident.
"""
    PACKED_LINES_BUDGET

Byte budget for the packed cache lines touched by one `A*X` row tile, one
64-byte line per SNP column of the block.
"""
const PACKED_LINES_BUDGET = 32_768

# The tile's partial sums, `LOOKUP_ROW_TILE` times the rhs slice width, stay
# L1-resident while the blocks of a group are gathered.
"""
    LOOKUP_ROW_TILE

Samples per inner tile of the lookup-table `A*X` kernel.
"""
const LOOKUP_ROW_TILE = 512

# One chunk's tables cost `LOOKUP_CHUNK_SNPS / 4 * 256 * k_padded * sizeof(T)`
# bytes, built once and shared by every gather task.
"""
    LOOKUP_CHUNK_SNPS

SNPs per chunk of the lookup-table `A*X` kernel (a multiple of 4).
"""
const LOOKUP_CHUNK_SNPS = 1024

# A group of 4-SNP blocks stays L2-resident across the tile.
"""
    LOOKUP_GROUP_BUDGET

Bytes of lookup tables swept per inner sample tile.
"""
const LOOKUP_GROUP_BUDGET = 1 << 20

# Below it the `256 * k` table build per 4-SNP block is not amortised.
"""
    LOOKUP_MIN_ROWS

Fewest samples for which `A*X` uses the lookup-table kernel.
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
is at most `_vector_width(T)` and does not exceed `k`, floored at 4.
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

Return the register-tile shape `(MR, NR)` for element type `T`: `MR`
accumulator rows and `NR = 2 * _vector_width(T)` rhs lanes.
`transpose(A)*X` takes its tile height from `ATX_TILE_COLUMNS[]` instead of
`MR`.
"""
function _register_tile_shape(::Type{T}) where T
    # 2 * MR accumulators, 2 rhs, and 1 broadcast register stay in registers.
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

Return the `transpose(A)*X` tile height: 4 for 64-byte vectors, 2 on x86_64
with 32-byte vectors, else 6. With 64-byte vectors on a `znver4` host at 8
threads (m = 50000, n = 667095, k = 128), 4 and 6 columns measured within 3%
of each other, 1.7x faster than 8 and 2.0x faster than 2. With 32-byte
vectors, the 16 ymm registers of AVX2 cannot hold the
6-column tile, which also keeps one broadcast per column live: on a Haswell
Xeon at 8 threads (m = 50000, n = 660966, k = 128) 6 columns spilled 10
vectors per sample, 2 spilled none and measured 1.33x faster.
"""
function _detect_atx_tile_columns(vector_bytes::Int)
    vector_bytes == 64 && return 4
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
to a multiple of `DECODE_WIDTH`.
"""
function _panel_step(::Type{T}, k::Int) where T <: AbstractFloat
    _, nr = _register_tile_shape(T)
    k_padded = cld(max(min(k, 256), 1), nr) * nr
    return _round_down_decode(L2_PANEL_BUDGET ÷ (k_padded * sizeof(T)))
end

"""
    _rhs_step(k::Int) -> Int

Return the rhs columns per task: `k` capped at 256 and at least 1.
"""
_rhs_step(k::Int) = max(min(k, 256), 1)

"""
    _snparray_ax_row_step(::Type{T}, m::Int) -> row_step

Return the samples each task of `_snparray_ax_schedule!` covers for `m`
samples: a multiple of `DECODE_WIDTH`, at least `DECODE_WIDTH`. `m == 0`
does not error.
"""
function _snparray_ax_row_step(::Type{T}, m::Int) where T <: AbstractFloat
    return _round_down_decode(_task_axis_step(m, L1_OUT_BUDGET ÷ sizeof(T)))
end

"""
    _snparray_atx_steps(::Type{T}, n::Int) -> (row_step, column_step)

Return how `_snparray_atx_schedule!` cuts up `transpose(A)*x` with `n` SNPs:
`column_step` SNPs per task, swept in blocks of `row_step` samples.
`row_step` is a multiple of `DECODE_WIDTH` and at least `DECODE_WIDTH`;
`column_step` is at least 1, also when `n == 0`.
"""
function _snparray_atx_steps(::Type{T}, n::Int) where T <: AbstractFloat
    row_step = _round_down_decode(L1_OUT_BUDGET ÷ sizeof(T))
    column_step = max(_task_axis_step(n, 2048), 1)
    return row_step, column_step
end

"""
    _snparray_AX_steps(::Type{T}, m::Int, k::Int)
        -> (row_step, column_step, rhs_step)

Return how `_snparray_AX_schedule!` cuts up `A*X` with `m` samples and `k`
rhs columns: `row_step` samples and `rhs_step` rhs columns per task, swept in
blocks of `column_step` SNPs. `row_step` is a multiple of `DECODE_WIDTH` and
at least `DECODE_WIDTH`; the others are at least 1, also when `m == 0`.
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

Return how `_snparray_AtX_schedule!` cuts up `transpose(A)*X` with `n` SNPs
and `k` rhs columns: `column_step` SNPs and `rhs_step` rhs columns per task,
swept in blocks of `row_step` samples. `row_step` is a multiple of
`DECODE_WIDTH` and at least `DECODE_WIDTH`; the others are at least 1, also
when `n == 0`.
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
