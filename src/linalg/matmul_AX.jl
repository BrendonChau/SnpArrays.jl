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
