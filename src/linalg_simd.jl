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
    packed::StridedMatrix{UInt8},
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
    values::StridedMatrix{T},
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
    packed::StridedMatrix{UInt8},
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
    _pack_rhs_panel!(panel, panel_offset, rhs, column_first, column_last,
        rhs_column, valid, unroll, width)

Copy the `valid` rhs columns starting at `rhs_column` of SNP rows
`column_first:column_last` into `panel` at `panel_offset` so that the
`U * W` lanes of one SNP row are contiguous, zero-filling the lanes beyond
`valid`. `panel` is shared between concurrent tasks, each writing the slice
that starts at its own `panel_offset`.
"""
@inline function _pack_rhs_panel!(
    panel::Vector{T},
    panel_offset::Int,
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
        base = panel_offset + (j - 1) * lanes
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
    _snparray_AX_micro_tile!(out, packed, values, panel, panel_offset, row,
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
    panel_offset::Int,
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
    offset = panel_offset
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
    _snparray_AX_row_tiles!(out, packed, values, panel, panel_offset,
        row_first, row_last, column_first, column_last, rhs_column, valid,
        rows, unroll, width)

Sweep samples `row_first:row_last` with `MR`-row register tiles, finishing
the tail one sample at a time.
"""
@inline function _snparray_AX_row_tiles!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    values::Matrix{T},
    panel::Vector{T},
    panel_offset::Int,
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
            out, packed, values, panel, panel_offset, row, column_first,
            column_last, rhs_column, valid, rows, unroll, width,
        )
        row += MR
    end
    while row <= row_last
        _snparray_AX_micro_tile!(
            out, packed, values, panel, panel_offset, row, column_first,
            column_last, rhs_column, valid, Val(1), unroll, width,
        )
        row += 1
    end
    return out
end

"""
    _snparray_AX_task!(out, packed, rhs, values, panel, panel_offset,
        row_first, row_last, column_step, rhs_first, rhs_last, rows, width)

Run one `A*X` task with a compile-time tile height `MR`, looping over SNP
column blocks of width `column_step` and rhs tiles of width `2W`.
"""
function _snparray_AX_task!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    panel::Vector{T},
    panel_offset::Int,
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
                    panel, panel_offset, rhs, column_first, column_last,
                    rhs_column, valid, Val(1), width,
                )
                _snparray_AX_row_tiles!(
                    out, packed, values, panel, panel_offset, row_first,
                    row_last, column_first, column_last, rhs_column, valid,
                    rows, Val(1), width,
                )
            else
                _pack_rhs_panel!(
                    panel, panel_offset, rhs, column_first, column_last,
                    rhs_column, valid, Val(2), width,
                )
                _snparray_AX_row_tiles!(
                    out, packed, values, panel, panel_offset, row_first,
                    row_last, column_first, column_last, rhs_column, valid,
                    rows, Val(2), width,
                )
            end
            rhs_column += 2W
        end
    end
    return out
end

"""
    _snparray_AX_kernel!(out, packed, rhs, values, panel, panel_offset,
        row_first, row_last, column_step, rhs_first, rhs_last, width)

Run one `A*X` task over samples `row_first:row_last` and rhs columns
`rhs_first:rhs_last`, blocking the SNP columns into `column_step`-wide
inner tiles and accumulating each `MR x 2W` register tile in registers.

The tile shape is `(MR, NR) = _micro_tile(T)`: AVX2 `Float32` gives
`MR = 6`, `NR = 16`, i.e. 12 accumulators + 2 rhs vectors + 1 broadcast
genotype = 15 of 16 ymm registers. The rhs values of one SNP column are
packed contiguously into the `2W * column_step` elements of `panel` that
start at `panel_offset`, zero-padded past the last rhs column of a partial
tile, so each column contributes `U` full vector loads. On a Xeon 6736P the
hot loop compiles to 16 FMAs on zmm accumulators with no spills, 19 of 32
registers live.
"""
function _snparray_AX_kernel!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    panel::Vector{T},
    panel_offset::Int,
    row_first::Int,
    row_last::Int,
    column_step::Int,
    rhs_first::Int,
    rhs_last::Int,
    width::Val{W},
) where {T <: SIMD_FLOAT, W}
    size(packed, 2) == 0 && return out
    tile_rows, _ = _micro_tile(T)
    return _snparray_AX_task!(
        out, packed, rhs, values, panel, panel_offset, row_first, row_last,
        column_step, rhs_first, rhs_last, Val(tile_rows), width,
    )
end

"""
    _snparray_atx_kernel!(out, packed, rhs, values, row_first, row_last,
        column_first, column_last, out_offset)

Accumulate `out[column_first - out_offset:column_last - out_offset] +=
transpose(A[row_first:row_last, column_first:column_last]) *
rhs[row_first:row_last]` for a `SnpLinAlg` matrix `A`, decoding genotypes
with the 16-lane vector decode and reducing each column's partial vector
with a SIMD tree reduction. Requires `row_first ≡ 1 (mod 4)`.

`column_first` and `column_last` index `packed` and `values`, which are the
full genotype arrays; `out_offset` shifts them onto `out`, so a restricted
column range needs no view of the genotypes.
"""
function _snparray_atx_kernel!(
    out::Vector{T},
    packed::StridedMatrix{UInt8},
    rhs::Vector{T},
    values::StridedMatrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    out_offset::Int,
) where T <: SIMD_FLOAT
    vectorized_row_last = row_last - 15
    @inbounds for column in column_first:column_last
        total = out[column - out_offset]
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
        out[column - out_offset] = total
    end
    return out
end

"""
    _load_packed_bytes(packed, byte_index, column, rows)

Load byte `byte_index` of the `MR` SNP columns starting at `column`, each
holding the genotype codes of four consecutive samples.
"""
@inline function _load_packed_bytes(
    packed::StridedMatrix{UInt8},
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
    values::StridedMatrix{T},
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
    packed::StridedMatrix{UInt8},
    values::StridedMatrix{T},
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
    _snparray_AtX_micro_tile!(out, packed, values, panel, panel_offset,
        row_first, row_last, column, rhs_column, valid, out_offset, rows,
        unroll, width)

Accumulate the `MR x (U * W)` register tile of `transpose(A)*X` rooted at
SNP `column` and rhs column `rhs_column` over the samples
`row_first:row_last`, then add the `valid` leading lanes into `out` at row
`column - out_offset`.
"""
@inline function _snparray_AtX_micro_tile!(
    out::Matrix{T},
    packed::StridedMatrix{UInt8},
    values::StridedMatrix{T},
    panel::Vector{T},
    panel_offset::Int,
    row_first::Int,
    row_last::Int,
    column::Int,
    rhs_column::Int,
    valid::Int,
    out_offset::Int,
    rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    accumulators = ntuple(_ -> _fill_blocks(zero(Vec{W, T}), unroll), Val(MR))
    lanes = U * W
    byte_index = ((row_first - 1) >>> 2) + 1
    row = row_first
    offset = panel_offset
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
                out[column - out_offset + c - 1, rhs_column + position - 1] +=
                    vector[lane]
            end
        end
    end
    return out
end

"""
    _snparray_AtX_column_tiles!(out, packed, values, panel, panel_offset,
        row_first, row_last, column_first, column_last, rhs_column, valid,
        out_offset, rows, unroll, width)

Sweep SNP columns `column_first:column_last` with `MR`-column register
tiles, finishing the tail one column at a time.
"""
@inline function _snparray_AtX_column_tiles!(
    out::Matrix{T},
    packed::StridedMatrix{UInt8},
    values::StridedMatrix{T},
    panel::Vector{T},
    panel_offset::Int,
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    valid::Int,
    out_offset::Int,
    rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    column = column_first
    while column + MR - 1 <= column_last
        _snparray_AtX_micro_tile!(
            out, packed, values, panel, panel_offset, row_first, row_last,
            column, rhs_column, valid, out_offset, rows, unroll, width,
        )
        column += MR
    end
    while column <= column_last
        _snparray_AtX_micro_tile!(
            out, packed, values, panel, panel_offset, row_first, row_last,
            column, rhs_column, valid, out_offset, Val(1), unroll, width,
        )
        column += 1
    end
    return out
end

"""
    _snparray_AtX_task!(out, packed, rhs, values, panel, panel_offset,
        row_step, rows_filled, column_first, column_last, rhs_first,
        rhs_last, out_offset, rows, width)

Run one `transpose(A)*X` task with a compile-time tile width `MR`, looping
over sample blocks of `row_step` samples and rhs tiles of width `2W`.
"""
function _snparray_AtX_task!(
    out::Matrix{T},
    packed::StridedMatrix{UInt8},
    rhs::Matrix{T},
    values::StridedMatrix{T},
    panel::Vector{T},
    panel_offset::Int,
    row_step::Int,
    rows_filled::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
    out_offset::Int,
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
                    panel, panel_offset, rhs, row_first, row_last,
                    rhs_column, valid, Val(1), width,
                )
                _snparray_AtX_column_tiles!(
                    out, packed, values, panel, panel_offset, row_first,
                    row_last, column_first, column_last, rhs_column, valid,
                    out_offset, rows, Val(1), width,
                )
            else
                _pack_rhs_panel!(
                    panel, panel_offset, rhs, row_first, row_last,
                    rhs_column, valid, Val(2), width,
                )
                _snparray_AtX_column_tiles!(
                    out, packed, values, panel, panel_offset, row_first,
                    row_last, column_first, column_last, rhs_column, valid,
                    out_offset, rows, Val(2), width,
                )
            end
            rhs_column += 2W
        end
    end
    return out
end

"""
    _snparray_AtX_kernel!(out, packed, rhs, values, panel, panel_offset,
        row_step, rows_filled, column_first, column_last, rhs_first,
        rhs_last, out_offset, width)

Run one `transpose(A)*X` task over SNP columns `column_first:column_last`
and rhs columns `rhs_first:rhs_last`, blocking the samples into
`row_step`-wide inner blocks and accumulating each `MR x 2W` register tile
in registers over a whole sample block. `column_first` and `column_last`
index the full genotype arrays; `out_offset` shifts them onto `out`.

The tile shape is `(MR, NR) = _micro_tile(T)`: AVX2 `Float32` gives
`MR = 6`, `NR = 16`, i.e. 12 accumulators + 2 rhs vectors + 1 broadcast
genotype = 15 of 16 ymm registers. The rhs values of one sample are packed
contiguously into the `2W * row_step` elements of `panel` that start at
`panel_offset`, zero-padded past the last rhs column of a partial tile, so
each sample contributes `U` full vector loads. The reduction reads one
packed byte per SNP column per four samples and shifts out the four codes.
On a Xeon 6736P the hot loop compiles to 16 FMAs on zmm accumulators with
no spills.
"""
function _snparray_AtX_kernel!(
    out::Matrix{T},
    packed::StridedMatrix{UInt8},
    rhs::Matrix{T},
    values::StridedMatrix{T},
    panel::Vector{T},
    panel_offset::Int,
    row_step::Int,
    rows_filled::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
    out_offset::Int,
    width::Val{W},
) where {T <: SIMD_FLOAT, W}
    rows_filled == 0 && return out
    tile_columns, _ = _micro_tile(T)
    return _snparray_AtX_task!(
        out, packed, rhs, values, panel, panel_offset, row_step, rows_filled,
        column_first, column_last, rhs_first, rhs_last, out_offset,
        Val(tile_columns), width,
    )
end

"""
    LOOKUP_ROW_TILE

Samples per inner tile of the lookup-table `A*X` kernel; the tile's
row-major partial sums (`LOOKUP_ROW_TILE` times the rhs slice width) stay
L1-resident
while every 4-SNP block of a group is gathered. Initial value from
`kq_pass.c`.
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
    _lookup_gather4(x::UInt32, s::Int) -> UInt8

Return the byte holding the four 2-bit codes of sample `s` (0 to 3) from
`x`, whose bytes are the packed bytes of four consecutive SNP columns for
the same four samples.
"""
@inline function _lookup_gather4(x::UInt32, s::Int)
    y = (x >>> (2s)) & 0x03030303
    y = (y | (y >>> 6)) & 0x000f000f
    y = (y | (y >>> 12)) & 0x000000ff
    return y % UInt8
end

"""
    _lookup_transpose!(blk, row_span, packed, row_first, row_last,
        column_first, nblk)

Write, for samples `row_first:row_last` (`row_first ≡ 1 (mod 4)`) and the
`nblk` blocks of four SNP columns starting at `column_first`, one byte per
(block, sample) holding that sample's four codes, at
`blk[(b - 1) * row_span + row]`. Columns past `size(packed, 2)` read as
code 0.
"""
function _lookup_transpose!(
    blk::Vector{UInt8},
    row_span::Int,
    packed::Matrix{UInt8},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    nblk::Int,
)
    n = size(packed, 2)
    byte_first = ((row_first - 1) >>> 2) + 1
    byte_last = ((row_last - 1) >>> 2) + 1
    @inbounds for b in 1:nblk
        j0 = column_first + 4(b - 1)
        base = (b - 1) * row_span + row_first - 1
        for q in byte_first:byte_last
            x = UInt32(packed[q, j0])
            j0 + 1 <= n && (x |= UInt32(packed[q, j0 + 1]) << 8)
            j0 + 2 <= n && (x |= UInt32(packed[q, j0 + 2]) << 16)
            j0 + 3 <= n && (x |= UInt32(packed[q, j0 + 3]) << 24)
            position = base + 4(q - byte_first)
            blk[position + 1] = _lookup_gather4(x, 0)
            blk[position + 2] = _lookup_gather4(x, 1)
            blk[position + 3] = _lookup_gather4(x, 2)
            blk[position + 4] = _lookup_gather4(x, 3)
        end
    end
    return blk
end

"""
    _lookup_weights(values, column, n) -> NTuple{4, T}

Return the four genotype values of SNP `column`, or zeros past `n`.
"""
@inline function _lookup_weights(
    values::Matrix{T},
    column::Int,
    n::Int,
) where T <: SIMD_FLOAT
    column <= n || return (zero(T), zero(T), zero(T), zero(T))
    return @inbounds (values[1, column], values[2, column],
                      values[3, column], values[4, column])
end

"""
    _lookup_build_tables!(tables, stage, stage_offset, n, rhs, values,
        column_first, block_first, block_last, k_padded, slice_stride,
        ::Val{W}, ::Val{NV})

Fill, for blocks `block_first:block_last` of the chunk starting at SNP
`column_first` and every rhs slice `s` of width `S = NV * W`, the 256 table
rows `tables[s * slice_stride + ((b - 1) * 256 + code) * S + t] =
Σ_l values[code_l + 1, j_l] * rhs[j_l, s * S + t]` over the four SNPs `j_l`
of block `b`, zero past `size(rhs, 2)`. `stage` holds the four rhs rows,
`4 * k_padded` elements from `stage_offset`.
"""
function _lookup_build_tables!(
    tables::Vector{T},
    stage::Vector{T},
    stage_offset::Int,
    n::Int,
    rhs::Matrix{T},
    values::Matrix{T},
    column_first::Int,
    block_first::Int,
    block_last::Int,
    k_padded::Int,
    slice_stride::Int,
    ::Val{W},
    ::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice = NV * W
    k = size(rhs, 2)
    @inbounds for b in block_first:block_last
        j0 = column_first + 4(b - 1)
        for l in 0:3
            j = j0 + l
            offset = stage_offset + l * k_padded
            for t in 1:k
                stage[offset + t] = j <= n ? rhs[j, t] : zero(T)
            end
            for t in (k + 1):k_padded
                stage[offset + t] = zero(T)
            end
        end
        w0 = _lookup_weights(values, j0, n)
        w1 = _lookup_weights(values, j0 + 1, n)
        w2 = _lookup_weights(values, j0 + 2, n)
        w3 = _lookup_weights(values, j0 + 3, n)
        table_base = (b - 1) * 256 * slice
        for code in 0:255
            a0 = Vec{W, T}(w0[(code & 3) + 1])
            a1 = Vec{W, T}(w1[((code >>> 2) & 3) + 1])
            a2 = Vec{W, T}(w2[((code >>> 4) & 3) + 1])
            a3 = Vec{W, T}(w3[((code >>> 6) & 3) + 1])
            for s0 in 0:slice:(k_padded - 1)
                row_base = (s0 ÷ slice) * slice_stride + table_base +
                           code * slice - s0
                for v in (s0 + 1):W:(s0 + slice)
                    u0 = stage[VecRange{W}(stage_offset + v)]
                    u1 = stage[VecRange{W}(stage_offset + k_padded + v)]
                    u2 = stage[VecRange{W}(stage_offset + 2k_padded + v)]
                    u3 = stage[VecRange{W}(stage_offset + 3k_padded + v)]
                    tables[VecRange{W}(row_base + v)] = muladd(
                        a3, u3, muladd(a2, u2, muladd(a1, u1, a0 * u0)))
                end
            end
        end
    end
    return tables
end

"""
    _lookup_load(x, offset, ::Val{W}, ::Val{NV}) -> NTuple{NV, Vec{W, T}}

Load the `NV` consecutive `W`-vectors of `x` starting after `offset`.
"""
@inline function _lookup_load(
    x::Vector{T},
    offset::Int,
    ::Val{W},
    ::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    return ntuple(j -> @inbounds(x[VecRange{W}(offset + (j - 1) * W + 1)]),
                  Val(NV))
end

"""
    _lookup_store!(x, offset, vectors::NTuple{NV, Vec{W, T}})

Store `vectors` into consecutive `W`-vectors of `x` starting after `offset`.
"""
@inline function _lookup_store!(
    x::Vector{T},
    offset::Int,
    vectors::NTuple{NV, Vec{W, T}},
) where {T <: SIMD_FLOAT, W, NV}
    @inbounds for j in 1:NV
        x[VecRange{W}(offset + (j - 1) * W + 1)] = vectors[j]
    end
    return x
end

"""
    _lookup_add(accumulators::NTuple{NV, Vec{W, T}}, x, offset)

Return `accumulators` plus the `NV` consecutive `W`-vectors of `x` starting
after `offset`.
"""
@inline function _lookup_add(
    accumulators::NTuple{NV, Vec{W, T}},
    x::Vector{T},
    offset::Int,
) where {T <: SIMD_FLOAT, W, NV}
    return ntuple(
        j -> accumulators[j] +
             @inbounds(x[VecRange{W}(offset + (j - 1) * W + 1)]),
        Val(NV),
    )
end

"""
    _lookup_gather_tile!(tile, tile_offset, tables, blk, row_span, tile_first,
        tile_rows, nblk, group, slice_offset, ::Val{W}, ::Val{NV})

Set the row-major `tile_rows x NV * W` tile at `tile_offset` to the sum over
the `nblk` blocks of the table rows selected by `blk`, for the rhs slice
whose tables start after `slice_offset`, sweeping `group` blocks per pass.
"""
function _lookup_gather_tile!(
    tile::Vector{T},
    tile_offset::Int,
    tables::Vector{T},
    blk::Vector{UInt8},
    row_span::Int,
    tile_first::Int,
    tile_rows::Int,
    nblk::Int,
    group::Int,
    slice_offset::Int,
    width::Val{W},
    count::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice = NV * W
    @inbounds for i in 1:(tile_rows * slice)
        tile[tile_offset + i] = zero(T)
    end
    @inbounds for g0 in 1:group:nblk
        g1 = min(g0 + group - 1, nblk)
        for i in 1:tile_rows
            tile_row = tile_offset + (i - 1) * slice
            accumulators = _lookup_load(tile, tile_row, width, count)
            sample = tile_first + i - 1
            for b in g0:g1
                code = Int(blk[(b - 1) * row_span + sample])
                r = ((b - 1) * 256 + code) * slice + slice_offset
                accumulators = _lookup_add(accumulators, tables, r)
            end
            _lookup_store!(tile, tile_row, accumulators)
        end
    end
    return tile
end

"""
    _lookup_flush_tile!(out, tile, tile_offset, tile_first, tile_rows,
        rhs_column, valid, lanes)

Add the leading `valid` lanes of each row of the `tile_rows x lanes` tile
into `out[tile_first:tile_first + tile_rows - 1, rhs_column:rhs_column +
valid - 1]`.
"""
function _lookup_flush_tile!(
    out::Matrix{T},
    tile::Vector{T},
    tile_offset::Int,
    tile_first::Int,
    tile_rows::Int,
    rhs_column::Int,
    valid::Int,
    lanes::Int,
) where T <: SIMD_FLOAT
    @inbounds for t in 1:valid
        column = rhs_column + t - 1
        for i in 1:tile_rows
            out[tile_first + i - 1, column] +=
                tile[tile_offset + (i - 1) * lanes + t]
        end
    end
    return out
end

"""
    _snparray_AX_lookup_task!(out, packed, tables, tile, tile_offset, blk,
        row_span, row_first, row_last, column_first, nblk, slice_stride,
        group, ::Val{W}, ::Val{NV})

Run one gather task of the lookup-table `A*X` kernel: transpose the codes
of samples `row_first:row_last` for the chunk of `nblk` blocks starting at
SNP `column_first`, then for every `NV * W`-wide rhs slice (tables
`slice_stride` elements apart) and `LOOKUP_ROW_TILE`-row tile, gather the
table rows and add them into `out`.
"""
function _snparray_AX_lookup_task!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    tables::Vector{T},
    tile::Vector{T},
    tile_offset::Int,
    blk::Vector{UInt8},
    row_span::Int,
    row_first::Int,
    row_last::Int,
    column_first::Int,
    nblk::Int,
    slice_stride::Int,
    group::Int,
    width::Val{W},
    count::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice = NV * W
    k = size(out, 2)
    _lookup_transpose!(blk, row_span, packed, row_first, row_last,
                       column_first, nblk)
    for rhs_column in 1:slice:k
        valid = min(slice, k - rhs_column + 1)
        slice_offset = ((rhs_column - 1) ÷ slice) * slice_stride
        for tile_first in row_first:LOOKUP_ROW_TILE:row_last
            tile_rows = min(LOOKUP_ROW_TILE, row_last - tile_first + 1)
            _lookup_gather_tile!(
                tile, tile_offset, tables, blk, row_span, tile_first,
                tile_rows, nblk, group, slice_offset, width, count,
            )
            _lookup_flush_tile!(
                out, tile, tile_offset, tile_first, tile_rows, rhs_column,
                valid, slice,
            )
        end
    end
    return out
end
