"""
    RegisterTileTask(out, packed, values, panel, panel_offset, out_offset)

The arrays one register-tiled `A*X` or `transpose(A)*X` task works on.
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

Return `muladd(left, right[i], accumulators[i])` for each of the `U` blocks.
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
