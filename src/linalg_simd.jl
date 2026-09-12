"""
    VECTOR_BYTES

SIMD register width in bytes used by the register-tiled kernels: 64 on
x86_64 hosts whose ISA includes the AVX-512 tier, 32 otherwise (AVX2, or
NEON packed as two 128-bit registers). Set in `__init__` from
`_detect_vector_bytes()`; the first use of each `(T, W)` pair compiles
in-process (no precompile workload). On AVX-512 hosts, verify with
`@code_native` that LLVM emits zmm registers for the micro-kernel (LLVM's
prefer-vector-width default may split 512-bit vectors), and if it does
not, set `VECTOR_BYTES[] = 32`. This is a heuristic default awaiting one
benchmark run on an AVX2/AVX-512 x86 machine; no x86 machine was available
during development (an Apple M1 was used instead).
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
    DECODE_WIDTH

Number of genotypes produced by one vector decode (four packed bytes).
Separate from `VECTOR_BYTES`: `PACKED_EXPANSION`, `PACKED_SHIFTS`, the
`VecRange{4}` byte load, and every `row += 16` step stay fixed at 16
regardless of the SIMD register width.
"""
const DECODE_WIDTH = 16

"""
    L1_OUT_BUDGET

Byte budget, per task, for the `A*x` output block (read-modified-written
once per SNP column) and the `transpose(A)*x` rhs block (re-read once per
column), so it stays resident in a 32-48 KB-class L1 data cache with room
left for the packed genotype bytes. Heuristic default awaiting one
benchmark run on an AVX2/AVX-512 x86 machine.
"""
const L1_OUT_BUDGET = 16_384

"""
    L2_PANEL_BUDGET

Byte budget, per task, for the transposed rhs slab (`k_padded x
column_step * sizeof(T)`) of one inner block, so it fits a 256 KB-class
private L2 slice. Heuristic default awaiting one benchmark run on an
AVX2/AVX-512 x86 machine.
"""
const L2_PANEL_BUDGET = 262_144

"""
    PACKED_LINES_BUDGET

Byte budget for the packed cache lines touched by one `A*X` row tile, one
64-byte line per SNP column of the block, re-touched by every following
row tile, so they stay L1-resident. Heuristic default awaiting one
benchmark run on an AVX2/AVX-512 x86 machine.
"""
const PACKED_LINES_BUDGET = 32_768

"""
    TASKS_PER_THREAD

Target task count is `TASKS_PER_THREAD * Threads.nthreads()`, so uneven
per-task progress balances across threads. Heuristic default awaiting one
benchmark run on an AVX2/AVX-512 x86 machine.
"""
const TASKS_PER_THREAD = 4

"""
    TASK_AXIS_FLOOR

Minimum block size along a task-partitioned axis; below it, per-task
overhead and the output-tile read-modify-write dominate. Heuristic
default awaiting one benchmark run on an AVX2/AVX-512 x86 machine.
"""
const TASK_AXIS_FLOOR = 256

"""
    _vector_width(::Type{T}) -> Int

Return the number of `T` lanes in a `Vec{W,T}` register of width
`VECTOR_BYTES[]`.
"""
_vector_width(::Type{T}) where T = VECTOR_BYTES[] ÷ sizeof(T)

"""
    _micro_tile(::Type{T}) -> (MR, NR)

Return the register-tile shape `(MR, NR)` for element type `T`: `NR =
2 * _vector_width(T)` rhs lanes and `MR` accumulator rows. Register
budget: AVX2 Float32 gives `MR=6, NR=16` (12 accumulators + 2 rhs + 1
broadcast = 15 ymm registers); AVX2 Float64 gives `MR=6, NR=8` (15 ymm);
AVX-512 gives `MR=8` (19 zmm); NEON packs `W=8` Float32 lanes as two q
registers, using 30 of 32 registers.
"""
function _micro_tile(::Type{T}) where T
    mr = VECTOR_BYTES[] == 64 ? 8 : 6
    return mr, 2 * _vector_width(T)
end

"""
    _task_axis_step(length::Int, upper::Int) -> Int

Return the per-task block size along a task-partitioned axis of total
`length`, targeting `TASKS_PER_THREAD * Threads.nthreads()` tasks, rounded
up to a multiple of `DECODE_WIDTH`, and clamped between `TASK_AXIS_FLOOR`
and `upper` (`upper` is itself rounded down to a multiple of
`DECODE_WIDTH` and floored at `TASK_AXIS_FLOOR` before use).
"""
function _task_axis_step(length::Int, upper::Int)
    round_up16(x) = cld(x, DECODE_WIDTH) * DECODE_WIDTH
    round_down16(x) = (x ÷ DECODE_WIDTH) * DECODE_WIDTH
    bounded_upper = max(TASK_AXIS_FLOOR, round_down16(upper))
    tasks = TASKS_PER_THREAD * Threads.nthreads()
    step = length == 0 ? bounded_upper : round_up16(cld(length, tasks))
    return min(bounded_upper, max(TASK_AXIS_FLOOR, step))
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

These values are heuristic defaults awaiting one benchmark run on an
AVX2/AVX-512 x86 machine; no x86 machine was available during
development.
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
    _, nr = _micro_tile(T)
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

const SIMD_FLOAT = Union{Float32, Float64}
const PACKED_EXPANSION = Val((0, 0, 0, 0, 1, 1, 1, 1,
                              2, 2, 2, 2, 3, 3, 3, 3))
const PACKED_SHIFTS = Vec{16, UInt8}((0, 2, 4, 6, 0, 2, 4, 6,
                                      0, 2, 4, 6, 0, 2, 4, 6))

@inline function _decode_genotype_codes(
    packed::Matrix{UInt8},
    byte_index::Int,
    column::Int,
)
    bytes = @inbounds packed[VecRange{4}(byte_index), column]
    expanded = shufflevector(bytes, PACKED_EXPANSION)
    return (expanded >>> PACKED_SHIFTS) & UInt8(3)
end

"""
    _broadcast_lookup(values, column, ::Val{N})

Broadcast the four genotype-code values of SNP `column` into `N`-lane
vectors, once per column rather than once per 16-sample decode step.
"""
@inline function _broadcast_lookup(
    values::Matrix{T},
    column::Int,
    ::Val{N},
) where {N, T <: SIMD_FLOAT}
    return @inbounds (
        Vec{N, T}(values[1, column]),
        Vec{N, T}(values[2, column]),
        Vec{N, T}(values[3, column]),
        Vec{N, T}(values[4, column]),
    )
end

@inline function _select_genotypes(
    codes::Vec{N, UInt8},
    lookup::NTuple{4, Vec{N, T}},
) where {N, T <: SIMD_FLOAT}
    low_bit_is_zero = (codes & UInt8(1)) == UInt8(0)
    high_bit_is_zero = (codes & UInt8(2)) == UInt8(0)
    lower_values = vifelse(low_bit_is_zero, lookup[1], lookup[2])
    upper_values = vifelse(low_bit_is_zero, lookup[3], lookup[4])
    return vifelse(high_bit_is_zero, lower_values, upper_values)
end

@inline function _decode_genotypes(
    packed::Matrix{UInt8},
    lookup::NTuple{4, Vec{N, T}},
    byte_index::Int,
    column::Int,
) where {N, T <: SIMD_FLOAT}
    codes = _decode_genotype_codes(packed, byte_index, column)
    return _select_genotypes(codes, lookup)
end

"""
    _snparray_ax_kernel!(out, packed, rhs, values, row_first, row_last,
        column_first, column_last)

Accumulate `out[row_first:row_last] += A[row_first:row_last,
column_first:column_last] * rhs[column_first:column_last]` for a
`SnpLinAlg` matrix `A`, decoding genotypes with the 16-lane vector decode.
Requires `row_first ≡ 1 (mod 4)`.
"""
function _snparray_ax_kernel!(
    out::Vector{T},
    packed::Matrix{UInt8},
    rhs::Vector{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
) where T <: SIMD_FLOAT
    vectorized_row_last = row_last - 15
    @inbounds for column in column_first:column_last
        rhs_value = rhs[column]
        lookup = _broadcast_lookup(values, column, Val(16))
        row = row_first
        while row <= vectorized_row_last
            byte_index = ((row - 1) >>> 2) + 1
            genotypes = _decode_genotypes(packed, lookup, byte_index, column)
            out[VecRange{16}(row)] =
                muladd(genotypes, rhs_value, out[VecRange{16}(row)])
            row += 16
        end
        while row <= row_last
            out[row] += values[_packed_code(packed, row, column), column] *
                        rhs_value
            row += 1
        end
    end
    return out
end

"""
    _fill_blocks(value, ::Val{U})

Return a length-`U` tuple with every entry set to `value`.
"""
@inline function _fill_blocks(value::V, ::Val{U}) where {V, U}
    return ntuple(_ -> value, Val(U))
end

"""
    _multiply_add_blocks(accumulators, left, right)

Return `muladd(left, right[i], accumulators[i])` for each `i`, applying
the scalar `left` to every one of the `U` blocks.
"""
@inline function _multiply_add_blocks(
    accumulators::NTuple{U, A},
    left::L,
    right::NTuple{U, R},
) where {U, A, L, R}
    return ntuple(Val(U)) do offset
        muladd(left, right[offset], accumulators[offset])
    end
end

"""
    _pack_rhs_panel!(panel, rhs, column_first, column_last, rhs_column,
        valid, unroll, width)

Copy the `valid` rhs columns starting at `rhs_column` of SNP rows
`column_first:column_last` into `panel` so that the `U * W` lanes of one
SNP row are contiguous, zero-filling the lanes beyond `valid`.
"""
@inline function _pack_rhs_panel!(
    panel::Vector{T},
    rhs::Matrix{T},
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    valid::Int,
    ::Val{U},
    ::Val{W},
) where {T <: SIMD_FLOAT, U, W}
    lanes = U * W
    @inbounds for j in 1:(column_last - column_first + 1)
        base = (j - 1) * lanes
        for t in 1:valid
            panel[base + t] = rhs[column_first + j - 1, rhs_column + t - 1]
        end
        for t in (valid + 1):lanes
            panel[base + t] = zero(T)
        end
    end
    return panel
end

"""
    _load_panel_vectors(panel, offset, unroll, width)

Load the `U` consecutive `Vec{W,T}` vectors of `panel` that start at
element `offset + 1`.
"""
@inline function _load_panel_vectors(
    panel::Vector{T},
    offset::Int,
    ::Val{U},
    ::Val{W},
) where {T <: SIMD_FLOAT, U, W}
    return ntuple(Val(U)) do block
        @inbounds panel[VecRange{W}(offset + (block - 1) * W + 1)]
    end
end

"""
    _multiply_add_rows(accumulators, packed, values, rhs_vectors, row,
        column)

Fuse the genotypes of samples `row:row + MR - 1` at SNP `column` into the
`MR` rows of `accumulators`.
"""
@inline function _multiply_add_rows(
    accumulators::NTuple{MR, NTuple{U, Vec{W, T}}},
    packed::Matrix{UInt8},
    values::Matrix{T},
    rhs_vectors::NTuple{U, Vec{W, T}},
    row::Int,
    column::Int,
) where {T <: SIMD_FLOAT, MR, U, W}
    return ntuple(Val(MR)) do i
        genotype = Vec{W, T}(@inbounds values[
            _packed_code(packed, row + i - 1, column), column,
        ])
        _multiply_add_blocks(accumulators[i], genotype, rhs_vectors)
    end
end

"""
    _snparray_AX_micro_tile!(out, packed, values, panel, row,
        column_first, column_last, rhs_column, valid, rows, unroll, width)

Accumulate the `MR x (U * W)` register tile of `A*X` rooted at sample
`row` and rhs column `rhs_column` over the SNP columns
`column_first:column_last`, then add the `valid` leading lanes into `out`.
"""
@inline function _snparray_AX_micro_tile!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    values::Matrix{T},
    panel::Vector{T},
    row::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    valid::Int,
    ::Val{MR},
    unroll::Val{U},
    ::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    accumulators = ntuple(_ -> _fill_blocks(zero(Vec{W, T}), unroll), Val(MR))
    offset = 0
    for column in column_first:column_last
        rhs_vectors = _load_panel_vectors(panel, offset, unroll, Val(W))
        accumulators = _multiply_add_rows(
            accumulators, packed, values, rhs_vectors, row, column,
        )
        offset += U * W
    end
    @inbounds for i in 1:MR
        blocks = accumulators[i]
        for block in 1:U
            vector = blocks[block]
            for lane in 1:W
                position = (block - 1) * W + lane
                position <= valid || break
                out[row + i - 1, rhs_column + position - 1] += vector[lane]
            end
        end
    end
    return out
end

"""
    _snparray_AX_row_tiles!(out, packed, values, panel, row_first, row_last,
        column_first, column_last, rhs_column, valid, rows, unroll, width)

Sweep samples `row_first:row_last` with `MR`-row register tiles, finishing
the tail one sample at a time.
"""
@inline function _snparray_AX_row_tiles!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    values::Matrix{T},
    panel::Vector{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    valid::Int,
    rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    row = row_first
    while row + MR - 1 <= row_last
        _snparray_AX_micro_tile!(
            out, packed, values, panel, row, column_first, column_last,
            rhs_column, valid, rows, unroll, width,
        )
        row += MR
    end
    while row <= row_last
        _snparray_AX_micro_tile!(
            out, packed, values, panel, row, column_first, column_last,
            rhs_column, valid, Val(1), unroll, width,
        )
        row += 1
    end
    return out
end

"""
    _snparray_AX_task!(out, packed, rhs, values, panel, row_first, row_last,
        column_step, rhs_first, rhs_last, rows, width)

Run one `A*X` task with a compile-time tile height `MR`, looping over SNP
column blocks of width `column_step` and rhs tiles of width `2W`.
"""
function _snparray_AX_task!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    panel::Vector{T},
    row_first::Int,
    row_last::Int,
    column_step::Int,
    rhs_first::Int,
    rhs_last::Int,
    rows::Val{MR},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, W}
    n = size(packed, 2)
    for column_first in 1:column_step:n
        column_last = min(column_first + column_step - 1, n)
        rhs_column = rhs_first
        while rhs_column <= rhs_last
            valid = min(2W, rhs_last - rhs_column + 1)
            if valid <= W
                _pack_rhs_panel!(
                    panel, rhs, column_first, column_last, rhs_column, valid,
                    Val(1), width,
                )
                _snparray_AX_row_tiles!(
                    out, packed, values, panel, row_first, row_last,
                    column_first, column_last, rhs_column, valid, rows,
                    Val(1), width,
                )
            else
                _pack_rhs_panel!(
                    panel, rhs, column_first, column_last, rhs_column, valid,
                    Val(2), width,
                )
                _snparray_AX_row_tiles!(
                    out, packed, values, panel, row_first, row_last,
                    column_first, column_last, rhs_column, valid, rows,
                    Val(2), width,
                )
            end
            rhs_column += 2W
        end
    end
    return out
end

"""
    _snparray_AX_kernel!(out, packed, rhs, values, row_first, row_last,
        column_step, rhs_first, rhs_last, width)

Run one `A*X` task over samples `row_first:row_last` and rhs columns
`rhs_first:rhs_last`, blocking the SNP columns into `column_step`-wide
inner tiles and accumulating each `MR x 2W` register tile in registers.

The tile shape is `(MR, NR) = _micro_tile(T)`: AVX2 `Float32` gives
`MR = 6`, `NR = 16`, i.e. 12 accumulators + 2 rhs vectors + 1 broadcast
genotype = 15 of 16 ymm registers. The rhs values of one SNP column are
packed contiguously into a per-task panel buffer of `NR * column_step`
elements, zero-padded past the last rhs column of a partial tile, so each
column contributes `U` full vector loads. x86 performance is unverified
(development machine is an Apple M1); an x86 run should check
`@code_native` for accumulator spills.
"""
function _snparray_AX_kernel!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_step::Int,
    rhs_first::Int,
    rhs_last::Int,
    width::Val{W},
) where {T <: SIMD_FLOAT, W}
    size(packed, 2) == 0 && return out
    tile_rows, tile_lanes = _micro_tile(T)
    panel = zeros(T, tile_lanes * column_step)
    return _snparray_AX_task!(
        out, packed, rhs, values, panel, row_first, row_last, column_step,
        rhs_first, rhs_last, Val(tile_rows), width,
    )
end

"""
    _snparray_atx_kernel!(out, packed, rhs, values, row_first, row_last,
        column_first, column_last)

Accumulate `out[column_first:column_last] +=
transpose(A[row_first:row_last, column_first:column_last]) *
rhs[row_first:row_last]` for a `SnpLinAlg` matrix `A`, decoding genotypes
with the 16-lane vector decode and reducing each column's partial vector
with a SIMD tree reduction. Requires `row_first ≡ 1 (mod 4)`.
"""
function _snparray_atx_kernel!(
    out::Vector{T},
    packed::Matrix{UInt8},
    rhs::Vector{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
) where T <: SIMD_FLOAT
    vectorized_row_last = row_last - 15
    @inbounds for column in column_first:column_last
        total = out[column]
        vector_total = zero(Vec{16, T})
        lookup = _broadcast_lookup(values, column, Val(16))
        row = row_first
        while row <= vectorized_row_last
            byte_index = ((row - 1) >>> 2) + 1
            genotypes = _decode_genotypes(packed, lookup, byte_index, column)
            vector_total = muladd(genotypes, rhs[VecRange{16}(row)],
                                  vector_total)
            row += 16
        end
        total += sum(vector_total)
        while row <= row_last
            total +=
                values[_packed_code(packed, row, column), column] * rhs[row]
            row += 1
        end
        out[column] = total
    end
    return out
end

"""
    _load_packed_bytes(packed, byte_index, column, rows)

Load byte `byte_index` of the `MR` SNP columns starting at `column`, each
holding the genotype codes of four consecutive samples.
"""
@inline function _load_packed_bytes(
    packed::Matrix{UInt8},
    byte_index::Int,
    column::Int,
    ::Val{MR},
) where MR
    return ntuple(Val(MR)) do c
        @inbounds packed[byte_index, column + c - 1]
    end
end

"""
    _multiply_add_columns(accumulators, bytes, shift, values, column,
        rhs_vectors)

Fuse the genotypes held at bit position `shift` of `bytes` at the `MR` SNP
columns starting at `column` into the `MR` rows of `accumulators`.
"""
@inline function _multiply_add_columns(
    accumulators::NTuple{MR, NTuple{U, Vec{W, T}}},
    bytes::NTuple{MR, UInt8},
    shift::Int,
    values::Matrix{T},
    column::Int,
    rhs_vectors::NTuple{U, Vec{W, T}},
) where {T <: SIMD_FLOAT, MR, U, W}
    return ntuple(Val(MR)) do c
        code = Int((bytes[c] >> shift) & 0x03) + 1
        genotype = Vec{W, T}(@inbounds values[code, column + c - 1])
        _multiply_add_blocks(accumulators[c], genotype, rhs_vectors)
    end
end

"""
    _multiply_add_columns(accumulators, packed, values, rhs_vectors, row,
        column)

Fuse the genotypes of sample `row` at the `MR` SNP columns starting at
`column` into the `MR` rows of `accumulators`.
"""
@inline function _multiply_add_columns(
    accumulators::NTuple{MR, NTuple{U, Vec{W, T}}},
    packed::Matrix{UInt8},
    values::Matrix{T},
    rhs_vectors::NTuple{U, Vec{W, T}},
    row::Int,
    column::Int,
) where {T <: SIMD_FLOAT, MR, U, W}
    return ntuple(Val(MR)) do c
        genotype = Vec{W, T}(@inbounds values[
            _packed_code(packed, row, column + c - 1), column + c - 1,
        ])
        _multiply_add_blocks(accumulators[c], genotype, rhs_vectors)
    end
end

"""
    _snparray_AtX_micro_tile!(out, packed, values, panel, row_first,
        row_last, column, rhs_column, valid, rows, unroll, width)

Accumulate the `MR x (U * W)` register tile of `transpose(A)*X` rooted at
SNP `column` and rhs column `rhs_column` over the samples
`row_first:row_last`, then add the `valid` leading lanes into `out`.
"""
@inline function _snparray_AtX_micro_tile!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    values::Matrix{T},
    panel::Vector{T},
    row_first::Int,
    row_last::Int,
    column::Int,
    rhs_column::Int,
    valid::Int,
    rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    accumulators = ntuple(_ -> _fill_blocks(zero(Vec{W, T}), unroll), Val(MR))
    lanes = U * W
    byte_index = ((row_first - 1) >>> 2) + 1
    row = row_first
    offset = 0
    # `row_first ≡ 1 (mod 4)`, so one byte per column covers four samples.
    while row + 3 <= row_last
        bytes = _load_packed_bytes(packed, byte_index, column, rows)
        for s in 0:3
            rhs_vectors = _load_panel_vectors(
                panel, offset + s * lanes, unroll, width,
            )
            accumulators = _multiply_add_columns(
                accumulators, bytes, 2s, values, column, rhs_vectors,
            )
        end
        row += 4
        offset += 4 * lanes
        byte_index += 1
    end
    while row <= row_last
        rhs_vectors = _load_panel_vectors(panel, offset, unroll, width)
        accumulators = _multiply_add_columns(
            accumulators, packed, values, rhs_vectors, row, column,
        )
        row += 1
        offset += lanes
    end
    @inbounds for c in 1:MR
        blocks = accumulators[c]
        for block in 1:U
            vector = blocks[block]
            for lane in 1:W
                position = (block - 1) * W + lane
                position <= valid || break
                out[column + c - 1, rhs_column + position - 1] += vector[lane]
            end
        end
    end
    return out
end

"""
    _snparray_AtX_column_tiles!(out, packed, values, panel, row_first,
        row_last, column_first, column_last, rhs_column, valid, rows,
        unroll, width)

Sweep SNP columns `column_first:column_last` with `MR`-column register
tiles, finishing the tail one column at a time.
"""
@inline function _snparray_AtX_column_tiles!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    values::Matrix{T},
    panel::Vector{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    valid::Int,
    rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    column = column_first
    while column + MR - 1 <= column_last
        _snparray_AtX_micro_tile!(
            out, packed, values, panel, row_first, row_last, column,
            rhs_column, valid, rows, unroll, width,
        )
        column += MR
    end
    while column <= column_last
        _snparray_AtX_micro_tile!(
            out, packed, values, panel, row_first, row_last, column,
            rhs_column, valid, Val(1), unroll, width,
        )
        column += 1
    end
    return out
end

"""
    _snparray_AtX_task!(out, packed, rhs, values, panel, row_step,
        rows_filled, column_first, column_last, rhs_first, rhs_last, rows,
        width)

Run one `transpose(A)*X` task with a compile-time tile width `MR`, looping
over sample blocks of `row_step` samples and rhs tiles of width `2W`.
"""
function _snparray_AtX_task!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    panel::Vector{T},
    row_step::Int,
    rows_filled::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
    rows::Val{MR},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, W}
    for row_first in 1:row_step:rows_filled
        row_last = min(row_first + row_step - 1, rows_filled)
        rhs_column = rhs_first
        while rhs_column <= rhs_last
            valid = min(2W, rhs_last - rhs_column + 1)
            if valid <= W
                _pack_rhs_panel!(
                    panel, rhs, row_first, row_last, rhs_column, valid,
                    Val(1), width,
                )
                _snparray_AtX_column_tiles!(
                    out, packed, values, panel, row_first, row_last,
                    column_first, column_last, rhs_column, valid, rows,
                    Val(1), width,
                )
            else
                _pack_rhs_panel!(
                    panel, rhs, row_first, row_last, rhs_column, valid,
                    Val(2), width,
                )
                _snparray_AtX_column_tiles!(
                    out, packed, values, panel, row_first, row_last,
                    column_first, column_last, rhs_column, valid, rows,
                    Val(2), width,
                )
            end
            rhs_column += 2W
        end
    end
    return out
end

"""
    _snparray_AtX_kernel!(out, packed, rhs, values, row_step, rows_filled,
        column_first, column_last, rhs_first, rhs_last, width)

Run one `transpose(A)*X` task over SNP columns `column_first:column_last`
and rhs columns `rhs_first:rhs_last`, blocking the samples into
`row_step`-wide inner blocks and accumulating each `MR x 2W` register tile
in registers over a whole sample block.

The tile shape is `(MR, NR) = _micro_tile(T)`: AVX2 `Float32` gives
`MR = 6`, `NR = 16`, i.e. 12 accumulators + 2 rhs vectors + 1 broadcast
genotype = 15 of 16 ymm registers. The rhs values of one sample are packed
contiguously into a per-task panel buffer of `NR * row_step` elements,
zero-padded past the last rhs column of a partial tile, so each sample
contributes `U` full vector loads. The reduction reads one packed byte per
SNP column per four samples and shifts out the four codes. x86 performance
is unverified (development machine is an Apple M1); an x86 run should check
`@code_native` for accumulator spills.
"""
function _snparray_AtX_kernel!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_step::Int,
    rows_filled::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
    width::Val{W},
) where {T <: SIMD_FLOAT, W}
    rows_filled == 0 && return out
    tile_columns, tile_lanes = _micro_tile(T)
    panel = zeros(T, tile_lanes * row_step)
    return _snparray_AtX_task!(
        out, packed, rhs, values, panel, row_step, rows_filled, column_first,
        column_last, rhs_first, rhs_last, Val(tile_columns), width,
    )
end
