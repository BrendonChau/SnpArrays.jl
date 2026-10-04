"""
    RegisterTileTask(out, packed, values, panel, panel_offset, out_offset)

The arrays one register-tiled `A*X` or `transpose(A)*X` task works on. The
task packs rhs values into the slice of `panel` after `panel_offset`, and
adds the tile for SNP column `j` into `out` row `j - out_offset`
(`out_offset = 0` for `A*X`).
"""
struct RegisterTileTask{T, O <: AbstractMatrix{T}, P <: AbstractMatrix{UInt8},
                        V <: AbstractMatrix{T}}
    out::O
    packed::P
    values::V
    panel::Vector{T}
    panel_offset::Int    # this task's slice of `panel` starts after it
    out_offset::Int      # SNP column minus `out` row; 0 for A*X
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
    _zero_accumulators(T, ::Val{MR}, ::Val{U}, ::Val{W})

Return `MR` rows of `U` zero `Vec{W,T}` accumulators.
"""
@inline function _zero_accumulators(
    ::Type{T},
    ::Val{MR},
    unroll::Val{U},
    ::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    return ntuple(_ -> _fill_blocks(zero(Vec{W, T}), unroll), Val(MR))
end

"""
    _add_register_tile!(out, accumulators, out_row, tile_columns)

Add the leading `length(tile_columns)` lanes of accumulator row `i` into
`out[out_row + i - 1, tile_columns]`.
"""
@inline function _add_register_tile!(
    out::AbstractMatrix{T},
    accumulators::NTuple{MR, NTuple{U, Vec{W, T}}},
    out_row::Int,
    tile_columns::UnitRange{Int},
) where {T <: SIMD_FLOAT, MR, U, W}
    valid = length(tile_columns)
    rhs_column = first(tile_columns)
    @inbounds for i in 1:MR
        blocks = accumulators[i]
        for block in 1:U
            vector = blocks[block]
            for lane in 1:W
                position = (block - 1) * W + lane
                position <= valid || break
                out[out_row + i - 1, rhs_column + position - 1] +=
                    vector[lane]
            end
        end
    end
    return out
end

"""
    _pack_rhs_panel!(task, rhs, rhs_rows, tile_columns, ::Val{U}, ::Val{W})

Copy `rhs[rhs_rows, tile_columns]` into `task.panel` after
`task.panel_offset` so that the `U * W` lanes of one rhs row are contiguous,
zero-filling the lanes beyond `length(tile_columns)`.
"""
@inline function _pack_rhs_panel!(
    task::RegisterTileTask{T},
    rhs::StridedMatrix{T},
    rhs_rows::UnitRange{Int},
    tile_columns::UnitRange{Int},
    ::Val{U},
    ::Val{W},
) where {T <: SIMD_FLOAT, U, W}
    panel = task.panel
    panel_offset = task.panel_offset
    row_first = first(rhs_rows)
    rhs_column = first(tile_columns)
    valid = length(tile_columns)
    lanes = U * W
    @inbounds for j in 1:length(rhs_rows)
        base = panel_offset + (j - 1) * lanes
        for t in 1:valid
            panel[base + t] = rhs[row_first + j - 1, rhs_column + t - 1]
        end
        for t in (valid + 1):lanes
            panel[base + t] = zero(T)
        end
    end
    return panel
end

"""
    _load_vectors(x, offset, ::Val{count}, ::Val{W})

Load the `count` consecutive `Vec{W, T}` vectors of `x` that start at
element `offset + 1`.
"""
@inline function _load_vectors(
    x::Vector{T},
    offset::Int,
    ::Val{count},
    ::Val{W},
) where {T <: SIMD_FLOAT, count, W}
    return ntuple(Val(count)) do block
        @inbounds x[VecRange{W}(offset + (block - 1) * W + 1)]
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
    _snparray_AX_register_tile!(task, row, columns, tile_columns, ::Val{MR},
        ::Val{U}, ::Val{W})

Accumulate the `MR x (U * W)` register tile of `A*X` rooted at sample `row`
over SNP `columns`, then add it into `task.out[:, tile_columns]`.
"""
@inline function _snparray_AX_register_tile!(
    task::RegisterTileTask{T},
    row::Int,
    columns::UnitRange{Int},
    tile_columns::UnitRange{Int},
    tile_rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    packed = task.packed
    values = task.values
    panel = task.panel
    accumulators = _zero_accumulators(T, tile_rows, unroll, width)
    offset = task.panel_offset
    for column in columns
        rhs_vectors = _load_vectors(panel, offset, unroll, width)
        accumulators = _multiply_add_rows(
            accumulators, packed, values, rhs_vectors, row, column,
        )
        offset += U * W
    end
    return _add_register_tile!(task.out, accumulators, row, tile_columns)
end

"""
    _snparray_AX_row_sweep!(task, rows, columns, tile_columns, ::Val{MR},
        ::Val{U}, ::Val{W})

Sweep samples `rows` with `MR`-row register tiles, finishing the tail one
sample at a time.
"""
@inline function _snparray_AX_row_sweep!(
    task::RegisterTileTask{T},
    rows::UnitRange{Int},
    columns::UnitRange{Int},
    tile_columns::UnitRange{Int},
    tile_rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    row = first(rows)
    row_last = last(rows)
    while row + MR - 1 <= row_last
        _snparray_AX_register_tile!(
            task, row, columns, tile_columns, tile_rows, unroll, width,
        )
        row += MR
    end
    while row <= row_last
        _snparray_AX_register_tile!(
            task, row, columns, tile_columns, Val(1), unroll, width,
        )
        row += 1
    end
    return task.out
end

"""
    _snparray_AX_rhs_tile!(task, rhs, rows, columns, tile_columns, ::Val{MR},
        ::Val{U}, ::Val{W})

Pack `rhs[columns, tile_columns]` into the task's panel, then sweep samples
`rows` over it.
"""
@inline function _snparray_AX_rhs_tile!(
    task::RegisterTileTask{T},
    rhs::StridedMatrix{T},
    rows::UnitRange{Int},
    columns::UnitRange{Int},
    tile_columns::UnitRange{Int},
    tile_rows::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    _pack_rhs_panel!(task, rhs, columns, tile_columns, unroll, width)
    return _snparray_AX_row_sweep!(
        task, rows, columns, tile_columns, tile_rows, unroll, width,
    )
end

"""
    _snparray_AX_task!(task, rhs, rows, column_step, rhs_columns, ::Val{MR},
        ::Val{W})

Run one `A*X` task over samples `rows` and rhs columns `rhs_columns`,
blocking the SNP columns into `column_step`-wide inner tiles and holding
each `MR x 2W` tile in registers. On a Xeon 6736P the hot loop compiles to
16 FMAs on zmm accumulators with no spills, 19 of 32 registers live.
"""
function _snparray_AX_task!(
    task::RegisterTileTask{T, <:StridedMatrix{T}, Matrix{UInt8}, Matrix{T}},
    rhs::StridedMatrix{T},
    rows::UnitRange{Int},
    column_step::Int,
    rhs_columns::UnitRange{Int},
    tile_rows::Val{MR},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, W}
    n = size(task.packed, 2)
    n == 0 && return task.out
    rhs_last = last(rhs_columns)
    for column_first in 1:column_step:n
        columns = column_first:min(column_first + column_step - 1, n)
        rhs_column = first(rhs_columns)
        while rhs_column <= rhs_last
            tile_columns = rhs_column:min(rhs_column + 2W - 1, rhs_last)
            if length(tile_columns) <= W
                _snparray_AX_rhs_tile!(
                    task, rhs, rows, columns, tile_columns, tile_rows,
                    Val(1), width,
                )
            else
                _snparray_AX_rhs_tile!(
                    task, rhs, rows, columns, tile_columns, tile_rows,
                    Val(2), width,
                )
            end
            rhs_column += 2W
        end
    end
    return task.out
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
    _snparray_AtX_register_tile!(task, rows, column, tile_columns, ::Val{MR},
        ::Val{U}, ::Val{W})

Accumulate the `MR x (U * W)` register tile of `transpose(A)*X` rooted at
SNP `column` over samples `rows`, then add it into `task.out` at row
`column - task.out_offset`, columns `tile_columns`.
"""
@inline function _snparray_AtX_register_tile!(
    task::RegisterTileTask{T},
    rows::UnitRange{Int},
    column::Int,
    tile_columns::UnitRange{Int},
    tile_width::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    packed = task.packed
    values = task.values
    panel = task.panel
    row_first = first(rows)
    row_last = last(rows)
    accumulators = _zero_accumulators(T, tile_width, unroll, width)
    lanes = U * W
    byte_index = ((row_first - 1) >>> 2) + 1
    row = row_first
    offset = task.panel_offset
    # `row_first ≡ 1 (mod 4)`, so one byte per column covers four samples.
    while row + 3 <= row_last
        bytes = _load_packed_bytes(packed, byte_index, column, tile_width)
        for s in 0:3
            rhs_vectors = _load_vectors(
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
        rhs_vectors = _load_vectors(panel, offset, unroll, width)
        accumulators = _multiply_add_columns(
            accumulators, packed, values, rhs_vectors, row, column,
        )
        row += 1
        offset += lanes
    end
    return _add_register_tile!(
        task.out, accumulators, column - task.out_offset, tile_columns,
    )
end

"""
    _snparray_AtX_column_sweep!(task, rows, columns, tile_columns,
        ::Val{MR}, ::Val{U}, ::Val{W})

Sweep SNP `columns` with `MR`-column register tiles, finishing the tail one
column at a time.
"""
@inline function _snparray_AtX_column_sweep!(
    task::RegisterTileTask{T},
    rows::UnitRange{Int},
    columns::UnitRange{Int},
    tile_columns::UnitRange{Int},
    tile_width::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    column = first(columns)
    column_last = last(columns)
    while column + MR - 1 <= column_last
        _snparray_AtX_register_tile!(
            task, rows, column, tile_columns, tile_width, unroll, width,
        )
        column += MR
    end
    while column <= column_last
        _snparray_AtX_register_tile!(
            task, rows, column, tile_columns, Val(1), unroll, width,
        )
        column += 1
    end
    return task.out
end

"""
    _snparray_AtX_rhs_tile!(task, rhs, rows, columns, tile_columns,
        ::Val{MR}, ::Val{U}, ::Val{W})

Pack `rhs[rows, tile_columns]` into the task's panel, then sweep SNP
`columns` over it.
"""
@inline function _snparray_AtX_rhs_tile!(
    task::RegisterTileTask{T},
    rhs::StridedMatrix{T},
    rows::UnitRange{Int},
    columns::UnitRange{Int},
    tile_columns::UnitRange{Int},
    tile_width::Val{MR},
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, U, W}
    _pack_rhs_panel!(task, rhs, rows, tile_columns, unroll, width)
    return _snparray_AtX_column_sweep!(
        task, rows, columns, tile_columns, tile_width, unroll, width,
    )
end

"""
    _snparray_AtX_task!(task, rhs, row_step, rows_filled, columns,
        rhs_columns, ::Val{MR}, ::Val{W})

Run one `transpose(A)*X` task over SNP `columns` (indices into the full
genotype arrays) and rhs columns `rhs_columns`, blocking the samples into
`row_step`-wide inner blocks and holding each `MR x 2W` tile in registers
over a whole sample block. On a Xeon 6736P the hot loop compiles to 16 FMAs
on zmm accumulators with no spills.
"""
function _snparray_AtX_task!(
    task::RegisterTileTask{
        T, <:StridedMatrix{T}, <:StridedMatrix{UInt8}, <:StridedMatrix{T},
    },
    rhs::StridedMatrix{T},
    row_step::Int,
    rows_filled::Int,
    columns::UnitRange{Int},
    rhs_columns::UnitRange{Int},
    tile_width::Val{MR},
    width::Val{W},
) where {T <: SIMD_FLOAT, MR, W}
    rows_filled == 0 && return task.out
    rhs_last = last(rhs_columns)
    for row_first in 1:row_step:rows_filled
        rows = row_first:min(row_first + row_step - 1, rows_filled)
        rhs_column = first(rhs_columns)
        while rhs_column <= rhs_last
            tile_columns = rhs_column:min(rhs_column + 2W - 1, rhs_last)
            if length(tile_columns) <= W
                _snparray_AtX_rhs_tile!(
                    task, rhs, rows, columns, tile_columns, tile_width,
                    Val(1), width,
                )
            else
                _snparray_AtX_rhs_tile!(
                    task, rhs, rows, columns, tile_columns, tile_width,
                    Val(2), width,
                )
            end
            rhs_column += 2W
        end
    end
    return task.out
end
