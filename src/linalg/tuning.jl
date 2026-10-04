# Cache budgets and the block sizes the schedulers cut each product into.

"""
    SIMD_FLOAT

Element types the SIMD kernels accept: `Float32` and `Float64`.
"""
const SIMD_FLOAT = Union{Float32, Float64}

"""
    VECTOR_BYTES

SIMD register width in bytes used by the register-tiled kernels: 64 on
x86_64 hosts whose ISA includes the AVX-512 tier, 32 otherwise.
"""
const VECTOR_BYTES = Ref{Int}(32)

"""
    _detect_vector_bytes() -> Int

Return 64 on x86_64 hosts whose CPU ISA includes AVX-512, else 32.
"""
function _detect_vector_bytes()
    Sys.ARCH === :x86_64 || return 32
    # `Base.BinaryPlatforms.CPUID` is not public API; fall back to 32.
    try
        isa_map = Dict(Base.BinaryPlatforms.arch_march_isa_mapping["x86_64"])
        return isa_map["avx512"] <= Base.BinaryPlatforms.CPUID.cpu_isa() ?
               64 : 32
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

Fewest rhs columns for which `A*X` uses the lookup-table kernel.
"""
const LOOKUP_MIN_RHS = 4

"""
    _uses_lookup_kernel(m::Int, k::Int) -> Bool

Return whether `A*X` with `m` samples and `k` rhs columns runs the
lookup-table kernel rather than the register-tiled kernel.
"""
function _uses_lookup_kernel(m::Int, k::Int)
    return m >= LOOKUP_MIN_ROWS && k >= LOOKUP_MIN_RHS
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
    _register_tile_shape(::Type{T}) -> (MR, NR)

Return the register-tile shape `(MR, NR)` for element type `T`: `MR`
accumulator rows and `NR = 2 * _vector_width(T)` rhs lanes.
"""
function _register_tile_shape(::Type{T}) where T
    # 2 * MR accumulators, 2 rhs, and 1 broadcast register stay in registers.
    mr = VECTOR_BYTES[] == 64 ? 8 : 6
    return mr, 2 * _vector_width(T)
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
