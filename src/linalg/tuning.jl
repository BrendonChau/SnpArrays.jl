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
    _register_tile_shape(::Type{T}) -> (MR, NR)

Return the register-tile shape `(MR, NR)` for element type `T`: `NR =
2 * _vector_width(T)` rhs lanes and `MR` accumulator rows. Register
budget: AVX2 Float32 gives `MR=6, NR=16` (12 accumulators + 2 rhs + 1
broadcast = 15 ymm registers); AVX2 Float64 gives `MR=6, NR=8` (15 ymm);
AVX-512 gives `MR=8` (19 zmm); NEON packs `W=8` Float32 lanes as two q
registers, using 30 of 32 registers.
"""
function _register_tile_shape(::Type{T}) where T
    mr = VECTOR_BYTES[] == 64 ? 8 : 6
    return mr, 2 * _vector_width(T)
end

"""
    _tile_sizes(::Type{T}, m::Int, n::Int, k::Int, direction::Symbol;
        vector::Bool = (k == 1)) -> (row_step, column_step, rhs_step)

Return the task-scheduling tile sizes for a `SnpLinAlg` product with `m`
samples, `n` SNPs, and `k` right-hand-side columns. `direction` is
`:forward` for `A*X` or `:transpose` for `transpose(A)*X`. `row_step` is
always a multiple of `DECODE_WIDTH` and at least `DECODE_WIDTH`;
`column_step` and `rhs_step` are always at least 1. `m == 0` or `n == 0`
does not error. `vector` selects the rules for the `k = 1` vector
kernels (`A*x`, `transpose(A)*x`); a matrix product with one rhs column
still runs the register-tiled kernel, so it keeps the matrix rules by
passing `vector=false`.

For `:forward, vector` (`A*x`), `row_step` keeps the output block within
`L1_OUT_BUDGET` across the SNP column loop and `column_step` covers every
SNP (the vector kernel needs no inner column blocking).

For `:forward, !vector` (`A*X`), `rhs_step` is `k` capped at 256 (the
register tile handles all rhs blocking below 256); `column_step` sizes
the transposed rhs slab of one column block to `L2_PANEL_BUDGET` and the
packed cache lines it touches to `PACKED_LINES_BUDGET`;
`row_step` sizes the packed genotype block, re-read once per rhs tile, to
`4 * L2_PANEL_BUDGET / column_step` bytes (the `A*X` accumulators live in
registers, so `row_step` is not bound by `L1_OUT_BUDGET`).

For `:transpose, vector` (`transpose(A)*x`), `column_step` is
task-partitioned over SNPs and `row_step` keeps the rhs slice within
`L1_OUT_BUDGET` across the column loop.

For `:transpose, !vector` (`transpose(A)*X`), `rhs_step` is `k` capped at
256; `row_step` sizes the transposed rhs slab over samples to
`L2_PANEL_BUDGET`; `column_step` is task-partitioned over SNPs.

Across 106 swept configurations on a Xeon 6736P these values were within
1.15x of the best in every operation and element type; see
`benchmark/results/x86_avx512_n1183.md`.
"""
function _tile_sizes(
    ::Type{T},
    m::Int,
    n::Int,
    k::Int,
    direction::Symbol;
    vector::Bool = (k == 1),
) where T <: AbstractFloat
    direction in (:forward, :transpose) || throw(ArgumentError(
        "direction must be :forward or :transpose, got $direction",
    ))
    round_down16(x) = max(DECODE_WIDTH, (x ÷ DECODE_WIDTH) * DECODE_WIDTH)
    _, nr = _register_tile_shape(T)
    k_block = min(k, 256)
    k_padded = cld(max(k_block, 1), nr) * nr
    if direction == :forward
        if vector
            row_step = _task_axis_step(m, L1_OUT_BUDGET ÷ sizeof(T))
            column_step = max(n, 1)
            rhs_step = 1
        else
            rhs_step = k <= 256 ? k : 256
            column_step = clamp(
                round_down16(L2_PANEL_BUDGET ÷ (k_padded * sizeof(T))),
                64, PACKED_LINES_BUDGET ÷ 64,
            )
            row_step =
                _task_axis_step(m, 4 * L2_PANEL_BUDGET ÷ column_step)
        end
    else
        if vector
            column_step = _task_axis_step(n, 2048)
            row_step = L1_OUT_BUDGET ÷ sizeof(T)
            rhs_step = 1
        else
            rhs_step = k <= 256 ? k : 256
            row_step = clamp(
                round_down16(L2_PANEL_BUDGET ÷ (k_padded * sizeof(T))),
                64, 4096,
            )
            column_step = _task_axis_step(n, 2048)
        end
    end
    row_step = max(DECODE_WIDTH, (row_step ÷ DECODE_WIDTH) * DECODE_WIDTH)
    return row_step, max(column_step, 1), max(rhs_step, 1)
end
