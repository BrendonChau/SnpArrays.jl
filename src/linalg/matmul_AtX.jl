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
