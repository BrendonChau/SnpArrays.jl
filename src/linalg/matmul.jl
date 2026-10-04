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
    rhs::StridedMatrix{T},
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
            _unsafe_getindex(packed, row + i - 1, column) + 1, column,
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
    out::StridedMatrix{T},
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
    out::StridedMatrix{T},
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
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    rhs::StridedMatrix{T},
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
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    rhs::StridedMatrix{T},
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
            _unsafe_getindex(packed, row, column + c - 1) + 1, column + c - 1,
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
    out::StridedMatrix{T},
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
    out::StridedMatrix{T},
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
    out::StridedMatrix{T},
    packed::StridedMatrix{UInt8},
    rhs::StridedMatrix{T},
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
    out::StridedMatrix{T},
    packed::StridedMatrix{UInt8},
    rhs::StridedMatrix{T},
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
