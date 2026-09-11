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

@inline function _select_genotypes(
    codes::Vec{N, UInt8},
    values::Matrix{T},
    column::Int,
) where {N, T <: SIMD_FLOAT}
    @inbounds begin
        code0 = Vec{N, T}(values[1, column])
        code1 = Vec{N, T}(values[2, column])
        code2 = Vec{N, T}(values[3, column])
        code3 = Vec{N, T}(values[4, column])
    end
    return vifelse(codes == UInt8(0), code0,
                   vifelse(codes == UInt8(1), code1,
                            vifelse(codes == UInt8(2), code2, code3)))
end

@inline function _decode_genotypes(
    packed::Matrix{UInt8},
    values::Matrix{T},
    byte_index::Int,
    column::Int,
) where T <: SIMD_FLOAT
    codes = _decode_genotype_codes(packed, byte_index, column)
    return _select_genotypes(codes, values, column)
end

@inline function _accumulate_lanes(total::T, products::Vec{N, T}) where {N, T}
    @inbounds for lane in 1:N
        total += products[lane]
    end
    return total
end

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
        row = row_first
        while row <= vectorized_row_last
            byte_index = ((row - 1) >>> 2) + 1
            genotypes = _decode_genotypes(packed, values, byte_index, column)
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

@inline function _load_blocks(
    array::Matrix{T},
    row_index::R,
    first_column::Int,
    ::Val{U},
) where {T, R, U}
    return ntuple(Val(U)) do offset
        @inbounds array[row_index, first_column + offset - 1]
    end
end

@inline function _store_blocks!(
    array::Matrix{T},
    row_index::R,
    first_column::Int,
    blocks::NTuple{U, B},
) where {T, R, U, B}
    @inbounds for offset in 1:U
        array[row_index, first_column + offset - 1] = blocks[offset]
    end
    return array
end

@inline function _fill_blocks(value::V, ::Val{U}) where {V, U}
    return ntuple(_ -> value, Val(U))
end

@inline function _multiply_add_blocks(
    accumulators::NTuple{U, A},
    left::L,
    right::NTuple{U, R},
) where {U, A, L, R}
    return ntuple(Val(U)) do offset
        muladd(left, right[offset], accumulators[offset])
    end
end

@inline function _snparray_AX_vector_block!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    block_row::Int,
    ::Val{U},
    ::Val{W},
    lanes::Val{L},
    row_offset::Int,
) where {T <: SIMD_FLOAT, U, W, L}
    row = block_row + row_offset
    row_index = VecRange{W}(row)
    accumulators = _load_blocks(out, row_index, rhs_column, Val(U))
    @inbounds for column in column_first:column_last
        byte_index = ((block_row - 1) >>> 2) + 1
        codes = _decode_genotype_codes(packed, byte_index, column)
        genotypes = _select_genotypes(
            shufflevector(codes, lanes), values, column,
        )
        rhs_values = _load_blocks(rhs, column, rhs_column, Val(U))
        accumulators = _multiply_add_blocks(
            accumulators, genotypes, rhs_values,
        )
    end
    _store_blocks!(out, row_index, rhs_column, accumulators)
    return out
end

@inline function _snparray_AX_vector_blocks!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    block_row::Int,
    unroll::Val{U},
    width::Val{8},
) where {T <: SIMD_FLOAT, U}
    _snparray_AX_vector_block!(
        out, packed, rhs, values, column_first, column_last, rhs_column,
        block_row, unroll, width, Val((0, 1, 2, 3, 4, 5, 6, 7)), 0,
    )
    _snparray_AX_vector_block!(
        out, packed, rhs, values, column_first, column_last, rhs_column,
        block_row, unroll, width, Val((8, 9, 10, 11, 12, 13, 14, 15)), 8,
    )
    return out
end

@inline function _snparray_AX_vector_blocks!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    block_row::Int,
    unroll::Val{U},
    width::Val{4},
) where {T <: SIMD_FLOAT, U}
    _snparray_AX_vector_block!(
        out, packed, rhs, values, column_first, column_last, rhs_column,
        block_row, unroll, width, Val((0, 1, 2, 3)), 0,
    )
    _snparray_AX_vector_block!(
        out, packed, rhs, values, column_first, column_last, rhs_column,
        block_row, unroll, width, Val((4, 5, 6, 7)), 4,
    )
    _snparray_AX_vector_block!(
        out, packed, rhs, values, column_first, column_last, rhs_column,
        block_row, unroll, width, Val((8, 9, 10, 11)), 8,
    )
    _snparray_AX_vector_block!(
        out, packed, rhs, values, column_first, column_last, rhs_column,
        block_row, unroll, width, Val((12, 13, 14, 15)), 12,
    )
    return out
end

function _snparray_AX_register_blocks!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    rhs_last::Int,
    unroll::Val{U},
    width::Val{W},
) where {T <: SIMD_FLOAT, U, W}
    vectorized_row_last = row_last - 15
    while rhs_column + U - 1 <= rhs_last
        block_row = row_first
        while block_row <= vectorized_row_last
            _snparray_AX_vector_blocks!(
                out, packed, rhs, values, column_first, column_last,
                rhs_column, block_row, unroll, width,
            )
            block_row += 16
        end
        row = block_row
        while row <= row_last
            accumulators = _load_blocks(out, row, rhs_column, unroll)
            @inbounds for column in column_first:column_last
                genotype =
                    values[_packed_code(packed, row, column), column]
                rhs_values = _load_blocks(rhs, column, rhs_column, unroll)
                accumulators = _multiply_add_blocks(
                    accumulators, genotype, rhs_values,
                )
            end
            _store_blocks!(out, row, rhs_column, accumulators)
            row += 1
        end
        rhs_column += U
    end
    return rhs_column
end

function _snparray_AX_kernel!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
) where T <: SIMD_FLOAT
    simd_width = Val(32 ÷ sizeof(T))
    rhs_column = _snparray_AX_register_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_first, rhs_last, Val(8), simd_width,
    )
    rhs_column = _snparray_AX_register_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_column, rhs_last, Val(4), simd_width,
    )
    rhs_column = _snparray_AX_register_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_column, rhs_last, Val(2), simd_width,
    )
    _snparray_AX_register_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_column, rhs_last, Val(1), simd_width,
    )
    return out
end

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
        row = row_first
        while row <= vectorized_row_last
            byte_index = ((row - 1) >>> 2) + 1
            genotypes = _decode_genotypes(packed, values, byte_index, column)
            vector_total = muladd(genotypes, rhs[VecRange{16}(row)],
                                  vector_total)
            row += 16
        end
        total = _accumulate_lanes(total, vector_total)
        while row <= row_last
            total +=
                values[_packed_code(packed, row, column), column] * rhs[row]
            row += 1
        end
        out[column] = total
    end
    return out
end

@inline function _reduce_blocks(
    totals::NTuple{U, T},
    vectors::NTuple{U, Vec{W, T}},
) where {U, W, T}
    return ntuple(Val(U)) do offset
        _accumulate_lanes(totals[offset], vectors[offset])
    end
end

function _snparray_AtX_register_blocks!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    rhs_last::Int,
    unroll::Val{U},
) where {T <: SIMD_FLOAT, U}
    vectorized_row_last = row_last - 15
    while rhs_column + U - 1 <= rhs_last
        for column in column_first:column_last
            vectors = _fill_blocks(zero(Vec{16, T}), unroll)
            row = row_first
            while row <= vectorized_row_last
                byte_index = ((row - 1) >>> 2) + 1
                genotypes = _decode_genotypes(
                    packed, values, byte_index, column,
                )
                rhs_values = _load_blocks(
                    rhs, VecRange{16}(row), rhs_column, unroll,
                )
                vectors = _multiply_add_blocks(
                    vectors, genotypes, rhs_values,
                )
                row += 16
            end
            totals = _load_blocks(out, column, rhs_column, unroll)
            totals = _reduce_blocks(totals, vectors)
            while row <= row_last
                genotype = @inbounds values[
                    _packed_code(packed, row, column), column,
                ]
                rhs_values = _load_blocks(rhs, row, rhs_column, unroll)
                totals = _multiply_add_blocks(totals, genotype, rhs_values)
                row += 1
            end
            _store_blocks!(out, column, rhs_column, totals)
        end
        rhs_column += U
    end
    return rhs_column
end

@inline function _snparray_AtX_half_block(
    totals::NTuple{U, T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_first::Int,
    vectorized_row_last::Int,
    column::Int,
    rhs_column::Int,
    unroll::Val{U},
    lanes::Val{L},
    row_offset::Int,
) where {T <: SIMD_FLOAT, U, L}
    vectors = _fill_blocks(zero(Vec{8, T}), unroll)
    block_row = row_first
    while block_row <= vectorized_row_last
        row = block_row + row_offset
        byte_index = ((block_row - 1) >>> 2) + 1
        codes = _decode_genotype_codes(packed, byte_index, column)
        genotypes = _select_genotypes(
            shufflevector(codes, lanes), values, column,
        )
        rhs_values = _load_blocks(
            rhs, VecRange{8}(row), rhs_column, unroll,
        )
        vectors = _multiply_add_blocks(vectors, genotypes, rhs_values)
        block_row += 16
    end
    return _reduce_blocks(totals, vectors), block_row
end

function _snparray_AtX_chunked_blocks!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_column::Int,
    rhs_last::Int,
    unroll::Val{U},
) where {T <: SIMD_FLOAT, U}
    vectorized_row_last = row_last - 15
    while rhs_column + U - 1 <= rhs_last
        for column in column_first:column_last
            totals = _load_blocks(out, column, rhs_column, unroll)
            totals, _ = _snparray_AtX_half_block(
                totals, packed, rhs, values, row_first,
                vectorized_row_last, column, rhs_column, unroll,
                Val((0, 1, 2, 3, 4, 5, 6, 7)), 0,
            )
            totals, row = _snparray_AtX_half_block(
                totals, packed, rhs, values, row_first,
                vectorized_row_last, column, rhs_column, unroll,
                Val((8, 9, 10, 11, 12, 13, 14, 15)), 8,
            )
            while row <= row_last
                genotype = @inbounds values[
                    _packed_code(packed, row, column), column,
                ]
                rhs_values = _load_blocks(rhs, row, rhs_column, unroll)
                totals = _multiply_add_blocks(totals, genotype, rhs_values)
                row += 1
            end
            _store_blocks!(out, column, rhs_column, totals)
        end
        rhs_column += U
    end
    return rhs_column
end

function _snparray_AtX_kernel!(
    out::Matrix{Float32},
    packed::Matrix{UInt8},
    rhs::Matrix{Float32},
    values::Matrix{Float32},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
)
    rhs_column = _snparray_AtX_register_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_first, rhs_last, Val(4),
    )
    _snparray_AtX_register_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_column, rhs_last, Val(1),
    )
    return out
end

function _snparray_AtX_kernel!(
    out::Matrix{Float64},
    packed::Matrix{UInt8},
    rhs::Matrix{Float64},
    values::Matrix{Float64},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    column_last::Int,
    rhs_first::Int,
    rhs_last::Int,
)
    rhs_column = _snparray_AtX_chunked_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_first, rhs_last, Val(4),
    )
    rhs_column = _snparray_AtX_chunked_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_column, rhs_last, Val(2),
    )
    _snparray_AtX_chunked_blocks!(
        out, packed, rhs, values, row_first, row_last, column_first,
        column_last, rhs_column, rhs_last, Val(1),
    )
    return out
end
