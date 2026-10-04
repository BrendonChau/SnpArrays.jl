# Generic fallbacks that the SIMD methods of matvec.jl and matmul.jl specialize.

function _snparray_ax_kernel!(
    out::AbstractVector,
    packed::AbstractMatrix{UInt8},
    rhs::AbstractVector,
    values::AbstractMatrix,
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
)
    @inbounds for column in column_first:column_last
        rhs_value = rhs[column]
        for row in row_first:row_last
            idx = _unsafe_getindex(packed, row, column) + 1
            out[row] += values[idx, column] * rhs_value
        end
    end
    return out
end

function _snparray_AX_task!(
    out::AbstractMatrix,
    packed::AbstractMatrix{UInt8},
    rhs::AbstractMatrix,
    values::AbstractMatrix,
    panel::Vector,
    panel_offset::Int,
    row_first::Int,
    row_last::Int,
    column_step::Int,
    rhs_first::Int,
    rhs_last::Int,
    ::Val,
)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in 1:size(packed, 2)
            rhs_value = rhs[column, rhs_column]
            for row in row_first:row_last
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
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    out_offset::Int,
)
    @inbounds for column in column_first:column_last
        total = out[column - out_offset]
        for row in row_first:row_last
            idx = _unsafe_getindex(packed, row, column) + 1
            total += values[idx, column] * rhs[row]
        end
        out[column - out_offset] = total
    end
    return out
end

function _snparray_AtX_task!(
    out::AbstractMatrix,
    packed::AbstractMatrix{UInt8},
    rhs::AbstractMatrix,
    values::AbstractMatrix,
    panel::Vector,
    panel_offset::Int,
    row_step::Int,
    rows_filled::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
    out_offset::Int,
    ::Val,
)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in column_first:column_last
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
