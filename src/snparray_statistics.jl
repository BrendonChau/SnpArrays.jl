"""
    mean(s; dims = :, model = ADDITIVE_MODEL)

Compute means of `s` along `dims`, ignoring missing genotypes. Returns a
scalar when `dims = :` and an array otherwise.

# Throws
- `ArgumentError`: `dims` is not `:`, `1`, or `2`
"""
Statistics.mean(
    s::AbstractSnpArray;
    dims::Union{Colon, Integer} = :,
    model::Union{Val{1}, Val{2}, Val{3}} = ADDITIVE_MODEL,
) =
    _mean(s, dims, model)

@inline _mean_numerator(count2::Int, count3::Int, ::typeof(ADDITIVE_MODEL)) =
    count2 + 2count3
@inline _mean_numerator(count2::Int, count3::Int, ::typeof(DOMINANT_MODEL)) =
    count2 + count3
@inline _mean_numerator(::Int, count3::Int, ::typeof(RECESSIVE_MODEL)) = count3

"""
    mean!(out, s; dims, model = ADDITIVE_MODEL)

Write means of `s` along `dims` to `out`, ignoring missing genotypes.

# Throws
- `DimensionMismatch`: `out` does not have one entry per retained index
- `ArgumentError`: `dims` is not `1` or `2`
"""
function Statistics.mean!(
    out::AbstractVecOrMat{T},
    s::AbstractSnpArray;
    dims::Integer,
    model::Union{Val{1}, Val{2}, Val{3}} = ADDITIVE_MODEL,
) where T <: AbstractFloat
    result_length = isone(dims) ? size(s, 2) :
                    dims == 2 ? size(s, 1) :
                    throw(ArgumentError("mean! only supports dims=1 or dims=2"))
    length(out) == result_length || throw(DimensionMismatch(
        "output has length $(length(out)); expected $result_length",
    ))
    counts = _counts(s, dims)
    group = 0
    @inbounds for output_index in eachindex(out)
        group += 1
        nonmissing = counts[1, group] + counts[3, group] + counts[4, group]
        numerator = _mean_numerator(counts[3, group], counts[4, group], model)
        out[output_index] = T(numerator) / T(nonmissing)
    end
    return out
end

function _mean(
    s::AbstractSnpArray,
    dims::Integer,
    model::Union{Val{1}, Val{2}, Val{3}},
)
    result = Matrix{Float64}(undef, isone(dims) ? (1, size(s, 2)) :
                            dims == 2 ? (size(s, 1), 1) :
                            throw(ArgumentError(
                                "mean only supports dims=1 or dims=2",
                            )))
    return mean!(result, s; dims=dims, model=model)
end

function _mean(
    s::AbstractSnpArray,
    ::Colon,
    model::Union{Val{1}, Val{2}, Val{3}},
)
    counts = _counts(s, 1)
    count0 = count2 = count3 = 0
    @inbounds for column in axes(counts, 2)
        count0 += counts[1, column]
        count2 += counts[3, column]
        count3 += counts[4, column]
    end
    return _mean_numerator(count2, count3, model) / (count0 + count2 + count3)
end

# TODO: need to implement `var` for different SNP models

"""
    var(s; dims = :, corrected = true, mean = nothing)

Compute additive-model variances of `s` along `dims`, ignoring missing
genotypes. Returns a scalar when `dims = :` and an array otherwise.

# Throws
- `DimensionMismatch`: a supplied `mean` has the wrong length
- `ArgumentError`: `dims` is not `:`, `1`, or `2`
"""
Statistics.var(
    s::AbstractSnpArray;
    corrected::Bool = true,
    mean::Union{Nothing, AbstractArray} = nothing,
    dims::Union{Colon, Integer} = :,
) = _var(s, corrected, mean, dims)

"""
    var!(out, s; dims, corrected = true, mean = nothing)

Write additive-model variances of `s` along `dims` to `out`, ignoring missing
genotypes.

# Throws
- `DimensionMismatch`: `out` or a supplied `mean` has the wrong length
- `ArgumentError`: `dims` is not `1` or `2`
"""
function var!(
    out::AbstractVecOrMat{T},
    s::AbstractSnpArray;
    dims::Integer,
    corrected::Bool = true,
    mean::Union{Nothing, AbstractArray} = nothing,
) where T <: AbstractFloat
    result_length = isone(dims) ? size(s, 2) :
                    dims == 2 ? size(s, 1) :
                    throw(ArgumentError("var! only supports dims=1 or dims=2"))
    length(out) == result_length || throw(DimensionMismatch(
        "output has length $(length(out)); expected $result_length",
    ))
    if mean !== nothing
        length(mean) == result_length || throw(DimensionMismatch(
            "mean has length $(length(mean)); expected $result_length",
        ))
    end
    counts = _counts(s, dims)
    group = 0
    mean_indices = mean === nothing ? eachindex(out) : eachindex(mean)
    @inbounds for (output_index, mean_index) in zip(eachindex(out), mean_indices)
        group += 1
        count0 = counts[1, group]
        count2 = counts[3, group]
        count3 = counts[4, group]
        nonmissing = count0 + count2 + count3
        group_mean = mean === nothing ?
                     T(count2 + 2count3) / T(nonmissing) : T(mean[mean_index])
        numerator = abs2(group_mean) * T(count0) +
                    abs2(one(T) - group_mean) * T(count2) +
                    abs2(T(2) - group_mean) * T(count3)
        denominator = nonmissing - Int(corrected)
        out[output_index] = numerator / T(denominator)
    end
    return out
end

function _var(
    s::AbstractSnpArray,
    corrected::Bool,
    mean::Union{Nothing, AbstractArray},
    dims::Integer,
)
    result = Matrix{Float64}(undef, isone(dims) ? (1, size(s, 2)) :
                            dims == 2 ? (size(s, 1), 1) :
                            throw(ArgumentError(
                                "var only supports dims=1 or dims=2",
                            )))
    return var!(result, s; dims=dims, corrected=corrected, mean=mean)
end

function _var(
    s::AbstractSnpArray,
    corrected::Bool,
    mean::Union{Nothing, AbstractArray},
    ::Colon,
)
    counts = _counts(s, 1)
    count0 = count2 = count3 = 0
    @inbounds for column in axes(counts, 2)
        count0 += counts[1, column]
        count2 += counts[3, column]
        count3 += counts[4, column]
    end
    nonmissing = count0 + count2 + count3
    if mean === nothing
        overall_mean = (count2 + 2count3) / nonmissing
    else
        length(mean) == 1 || throw(DimensionMismatch(
            "mean has length $(length(mean)); expected 1 when dims = :",
        ))
        overall_mean = first(mean)
    end
    numerator = abs2(overall_mean) * count0 +
                abs2(1 - overall_mean) * count2 +
                abs2(2 - overall_mean) * count3
    return numerator / (nonmissing - Int(corrected))
end

"""
    maf!(out, s)

Populate `out` with minor allele frequencies of SnpArray `s`.
"""
function maf!(out::AbstractVector{T}, s::AbstractSnpArray) where T <: AbstractFloat
    cc = _counts(s, 1)
    @inbounds for j in 1:size(s, 2)
        out[j] = (cc[3, j] + 2cc[4, j]) / 2(cc[1, j] + cc[3, j] + cc[4, j])
        (out[j] > 0.5) && (out[j] = 1 - out[j])
    end
    out
end
"""
    maf(s)

Calculate minor allele frequencies of SnpArray `s`.
"""
maf(s::AbstractSnpArray) = maf!(Vector{Float64}(undef, size(s, 2)), s)

"""
    minorallele!(out, s)

Populate `out` with minor allele indicators. `out[j] == true` means A2 is the minor
allele of `j`th column; `out[j] == false` means A1 is the minor allele.
"""
function minorallele!(out::AbstractVector{Bool}, s::AbstractSnpArray)
    cc = _counts(s, 1)
    @inbounds for j in 1:size(s, 2)
        out[j] = cc[1, j] > cc[4, j]
    end
    out
end

"""
    minorallele(s)

Calculate minor allele indicators. `out[j] == true` means A2 is the minor
allele of `j`th column; `out[j] == false` means A1 is the minor allele.
"""
minorallele(s::AbstractSnpArray) = minorallele!(BitVector(undef, size(s, 2)), s)

"""
    missingpos(s::SnpArray)

Return a `SparseMatrixCSC{Bool,Int32}` of the same size as `s` indicating the positions with missing data.
"""
function missingpos(s::SnpArray)
    m, n = size(s)
    counts = _counts(s, 1)
    colptr = Vector{Int32}(undef, n + 1)
    colptr[1] = 1
    @inbounds for column in 1:n
        colptr[column + 1] = colptr[column] + Int32(counts[2, column])
    end
    stored_values = Int(colptr[end] - 1)
    rowval = Vector{Int32}(undef, stored_values)
    nzval = fill(true, stored_values)
    full_bytes, trailing_genotypes = divrem(m, 4)
    position = 1
    @inbounds for column in 1:n
        for byte_index in 1:full_bytes
            byte = s.data[byte_index, column]
            mask = (byte & ~(byte >> 1)) & 0x55
            while !iszero(mask)
                bit = trailing_zeros(mask)
                rowval[position] = Int32(4byte_index - 3 + (bit >> 1))
                position += 1
                mask &= mask - one(UInt8)
            end
        end
        if !iszero(trailing_genotypes)
            byte = s.data[full_bytes + 1, column]
            mask = (byte & ~(byte >> 1)) & 0x55
            valid_mask = trailing_genotypes == 1 ? 0x01 :
                         trailing_genotypes == 2 ? 0x05 : 0x15
            mask &= valid_mask
            while !iszero(mask)
                bit = trailing_zeros(mask)
                rowval[position] = Int32(4full_bytes + 1 + (bit >> 1))
                position += 1
                mask &= mask - one(UInt8)
            end
        end
    end
    return SparseMatrixCSC(m, n, colptr, rowval, nzval)
end

function missingpos(s::AbstractSnpArray)
    m, n = size(s, 1), size(s, 2)
    colptr = Vector{Int32}(undef, n + 1)
    colptr[1] = 1
    rowval = Int32[]
    @inbounds for column in 1:n
        for row in 1:m
            isone(s[row, column]) && push!(rowval, row)
        end
        colptr[column + 1] = length(rowval) + 1
    end
    return SparseMatrixCSC(m, n, colptr, rowval, fill(true, length(rowval)))
end

"""
    missingrate!(out, s, dims)

Write missing-genotype rates along `dims` to `out`.

# Throws
- `DimensionMismatch`: `out` does not have one entry per retained index
- `ArgumentError`: `dims` is not `1` or `2`
"""
function missingrate!(
    out::AbstractVector{T},
    s::AbstractSnpArray,
    dims::Integer,
) where T <: AbstractFloat
    m, n = size(s, 1), size(s, 2)
    result_length = isone(dims) ? n :
                    dims == 2 ? m :
                    throw(ArgumentError(
                        "missingrate! only supports dims=1 or dims=2",
                    ))
    length(out) == result_length || throw(DimensionMismatch(
        "output has length $(length(out)); expected $result_length",
    ))
    group = 0
    if isone(dims)
        counts = _counts(s, 1)
        @inbounds for output_index in eachindex(out)
            group += 1
            out[output_index] = T(counts[2, group]) / T(m)
        end
    else
        counts = _counts(s, 2)
        @inbounds for output_index in eachindex(out)
            group += 1
            out[output_index] = T(counts[2, group]) / T(n)
        end
    end
    return out
end

"""
    missingrate(s, dims)

Calculate missing-genotype rates of `s` along `dims`.

# Throws
- `ArgumentError`: `dims` is not `1` or `2`
"""
function missingrate(s::AbstractSnpArray, dims::Integer)
    if isone(dims)
        return missingrate!(Vector{Float64}(undef, size(s, 2)), s, 1)
    elseif dims == 2
        return missingrate!(Vector{Float64}(undef, size(s, 1)), s, 2)
    else
        throw(ArgumentError(
            "missingrate(s::SnpArray, dims=k) only defined for k = 1 or 2",
        ))
    end
end
