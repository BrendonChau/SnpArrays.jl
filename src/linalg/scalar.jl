function _snparray_ax_kernel!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last)
    return _snparray_ax_scalar!(out, packed, rhs, values, row_first, row_last,
                                column_first, column_last)
end

function _snparray_ax_scalar!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last)
    @inbounds for column in column_first:column_last
        rhs_value = rhs[column]
        for row in row_first:row_last
            idx = _unsafe_getindex(packed, row, column) + 1
            out[row] += values[idx, column] * rhs_value
        end
    end
    return out
end

function _snparray_AX_kernel!(out, packed, rhs, values, panel, panel_offset,
                              row_first, row_last, column_step, rhs_first,
                              rhs_last, ::Val)
    return _snparray_AX_scalar!(out, packed, rhs, values, row_first, row_last,
                                1, size(packed, 2), rhs_first, rhs_last)
end

function _snparray_AX_scalar!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last, rhs_first, rhs_last)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in column_first:column_last
            rhs_value = rhs[column, rhs_column]
            for row in row_first:row_last
                idx = _unsafe_getindex(packed, row, column) + 1
                out[row, rhs_column] += values[idx, column] * rhs_value
            end
        end
    end
    return out
end

function _snparray_atx_kernel!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, out_offset)
    return _snparray_atx_scalar!(out, packed, rhs, values, row_first, row_last,
                                 column_first, column_last, out_offset)
end

function _snparray_atx_scalar!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, out_offset)
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

function _snparray_AtX_kernel!(out, packed, rhs, values, panel, panel_offset,
                               row_step, rows_filled, column_first,
                               column_last, rhs_first, rhs_last, out_offset,
                               ::Val)
    return _snparray_AtX_scalar!(out, packed, rhs, values, 1, rows_filled,
                                 column_first, column_last, rhs_first,
                                 rhs_last, out_offset)
end

function _snparray_AtX_scalar!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, rhs_first, rhs_last,
                               out_offset)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in column_first:column_last
            total = out[column - out_offset, rhs_column]
            for row in row_first:row_last
                idx = _unsafe_getindex(packed, row, column) + 1
                total += values[idx, column] * rhs[row, rhs_column]
            end
            out[column - out_offset, rhs_column] = total
        end
    end
    return out
end
