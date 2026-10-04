# Generic fallbacks that the SIMD methods of matvec.jl and matmul.jl specialize.

function _snparray_ax_kernel!(
    out::AbstractVector,
    packed::AbstractMatrix{UInt8},
    rhs::AbstractVector,
    values::AbstractMatrix,
    rows::UnitRange{Int},
    columns::UnitRange{Int},
)
    @inbounds for column in columns
        rhs_value = rhs[column]
        for row in rows
            idx = _unsafe_getindex(packed, row, column) + 1
            out[row] += values[idx, column] * rhs_value
        end
    end
    return out
end

function _snparray_AX_task!(
    task::RegisterTileTask,
    rhs::AbstractMatrix,
    rows::UnitRange{Int},
    column_step::Int,
    rhs_columns::UnitRange{Int},
    ::Val,
)
    out = task.out
    packed = task.packed
    values = task.values
    @inbounds for rhs_column in rhs_columns
        for column in 1:size(packed, 2)
            rhs_value = rhs[column, rhs_column]
            for row in rows
                idx = _unsafe_getindex(packed, row, column) + 1
                out[row, rhs_column] += values[idx, column] * rhs_value
            end
        end
    end
    return out
end

function _snparray_atx_kernel!(
    out::AbstractVector,
    packed::AbstractMatrix{UInt8},
    rhs::AbstractVector,
    values::AbstractMatrix,
    rows::UnitRange{Int},
    columns::UnitRange{Int},
    out_offset::Int,
)
    @inbounds for column in columns
        total = out[column - out_offset]
        for row in rows
            idx = _unsafe_getindex(packed, row, column) + 1
            total += values[idx, column] * rhs[row]
        end
        out[column - out_offset] = total
    end
    return out
end

function _snparray_AtX_task!(
    task::RegisterTileTask,
    rhs::AbstractMatrix,
    row_step::Int,
    rows_filled::Int,
    columns::UnitRange{Int},
    rhs_columns::UnitRange{Int},
    ::Val,
)
    out = task.out
    packed = task.packed
    values = task.values
    out_offset = task.out_offset
    @inbounds for rhs_column in rhs_columns
        for column in columns
            total = out[column - out_offset, rhs_column]
            for row in 1:rows_filled
                idx = _unsafe_getindex(packed, row, column) + 1
                total += values[idx, column] * rhs[row, rhs_column]
            end
            out[column - out_offset, rhs_column] = total
        end
    end
    return out
end
