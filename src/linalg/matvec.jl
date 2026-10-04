"""
    PACKED_EXPANSION

Shuffle that copies each of four packed bytes into four adjacent lanes of a
16-lane vector, one lane per sample.
"""
const PACKED_EXPANSION = Val((0, 0, 0, 0, 1, 1, 1, 1,
                              2, 2, 2, 2, 3, 3, 3, 3))

"""
    PACKED_SHIFTS

Per-lane right shifts that bring the 2-bit code of each sample to the low bits
of its lane after `PACKED_EXPANSION`.
"""
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
    _snparray_ax_kernel!(out, packed, rhs, values, rows, columns)

Accumulate `out[rows] += A[rows, columns] * rhs[columns]` for a `SnpLinAlg`
matrix `A`, decoding genotypes with the 16-lane vector decode. Requires
`first(rows) ≡ 1 (mod 4)`.
"""
function _snparray_ax_kernel!(
    out::Vector{T},
    packed::Matrix{UInt8},
    rhs::Vector{T},
    values::Matrix{T},
    rows::UnitRange{Int},
    columns::UnitRange{Int},
) where T <: SIMD_FLOAT
    row_first = first(rows)
    row_last = last(rows)
    vectorized_row_last = row_last - 15
    @inbounds for column in columns
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
            idx = _unsafe_getindex(packed, row, column) + 1
            out[row] += values[idx, column] * rhs_value
            row += 1
        end
    end
    return out
end

"""
    _snparray_atx_kernel!(out, packed, rhs, values, rows, columns,
        out_offset)

Accumulate `out[columns .- out_offset] += transpose(A[rows, columns]) *
rhs[rows]` for a `SnpLinAlg` matrix `A`, decoding genotypes with the 16-lane
vector decode and a SIMD tree reduction per column. `columns` index the full
genotype arrays `packed` and `values`; requires `first(rows) ≡ 1 (mod 4)`.
"""
function _snparray_atx_kernel!(
    out::Vector{T},
    packed::StridedMatrix{UInt8},
    rhs::Vector{T},
    values::StridedMatrix{T},
    rows::UnitRange{Int},
    columns::UnitRange{Int},
    out_offset::Int,
) where T <: SIMD_FLOAT
    row_first = first(rows)
    row_last = last(rows)
    vectorized_row_last = row_last - 15
    @inbounds for column in columns
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
            idx = _unsafe_getindex(packed, row, column) + 1
            total += values[idx, column] * rhs[row]
            row += 1
        end
        out[column - out_offset] = total
    end
    return out
end
