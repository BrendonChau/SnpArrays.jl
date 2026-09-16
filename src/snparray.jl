"""
    SnpArray
    SnpArray(bednm, m)
    SnpArray(plknm)
    SnpArray(undef, m, n)

Raw .bed file as a shared, memory-mapped Matrix{UInt8}.  The number of rows, `m`
is stored separately because it is not uniquely determined by the size of the `data` field.
"""
struct SnpArray <: AbstractMatrix{UInt8}
    data::Matrix{UInt8}
    columncounts::Matrix{Int}
    rowcounts::Matrix{Int}
    m::Int
end

"""
    StackedSnpArray(s::Vector{SnpArray})

Stacked SnpArray for unified indexing
"""
struct StackedSnpArray <: AbstractMatrix{UInt8} # details in stackedsnparray.jl
    arrays::Vector{SnpArray}
    m::Int
    n::Int
    ns::Vector{Int}
    offsets::Vector{Int} # 0-based
end

const AbstractSnpArray = Union{SnpArray, SubArray{UInt8, 1, SnpArray}, SubArray{UInt8, 2, SnpArray}, 
    StackedSnpArray, SubArray{UInt8, 1, StackedSnpArray}, SubArray{UInt8, 2, StackedSnpArray}}

function SnpArray(bednm::AbstractString, m::Integer, args...; kwargs...)
    checkplinkfilename(bednm, "bed")
    data = makestream(bednm, args...; kwargs...) do io
        read(io, UInt16) == 0x1b6c || throw(ArgumentError("wrong magic number in file $bednm"))
        read(io, UInt8) == 0x01 || throw(ArgumentError(".bed file, $bednm, is not in correct orientation"))
        if endswith(bednm, ".bed")
            return Mmap.mmap(io)
        else
            return read(io)
        end
    end
    drows = (m + 3) >> 2   # the number of rows in the Matrix{UInt8}
    n, r = divrem(length(data), drows)
    iszero(r) || throw(ArgumentError("filesize of $bednm is not a multiple of $drows"))
    SnpArray(reshape(data, (drows, n)), zeros(Int, (4, n)), zeros(Int, (4, m)), m)
end

function SnpArray(
    bednm::AbstractString, 
    args...; 
    famnm::Union{AbstractString, Nothing}=nothing, 
    kwargs...)
    checkplinkfilename(bednm, "bed")
    # check user supplied fam filename
    if famnm === nothing
        famnm = replace(bednm, ".bed" => ".fam") 
        isfile(famnm) || throw(ArgumentError("fam file not found"))
    else # user supplied fam file name
        checkplinkfilename(famnm, "fam")
        isfile(famnm) || throw(ArgumentError("file $famnm not found"))
    end
    m = makestream(famnm) do stream
        countlines(stream)
    end
    SnpArray(bednm, m, args...; kwargs...)
end

function SnpArray(::UndefInitializer, m::Integer, n::Integer)
    SnpArray(Matrix{UInt8}(undef, ((m + 3) >> 2, n)), zeros(Int, (4, n)), zeros(Int, (4, m)), m)
end

function SnpArray(file::AbstractString, s::SnpArray)
    makestream(file, "w+") do io
        write(io, 0x1b6c)
        write(io, 0x01)
        write(io, s.data)
    end
    SnpArray(file, s.m, "r+")
end

function SnpArray(file::AbstractString, m::Integer, n::Integer)
    makestream(file, "w+") do io
        write(io, 0x1b6c)
        write(io, 0x01)
        write(io, fill(0x00, ((m + 3) >> 2, n)))
    end
    SnpArray(file, m, "r+")
end

"""
    counts(s; dims = :)

Count genotype codes of `s` along `dims`, returning a 4-row matrix whose rows
give the counts of codes `0`, `1` (missing), `2`, and `3` respectively.
`dims = 1` gives one output column per SNP, `dims = 2` one per individual, and
`dims = :` a single column totalling the whole array.
"""
StatsBase.counts(s::AbstractSnpArray; dims=:) = _counts(s, dims)

@inline function _packed_counts(byte::UInt8)
    low = byte & 0x55
    high = (byte >> 1) & 0x55
    count1 = count_ones(low & ~high)
    count2 = count_ones(~low & high)
    count3 = count_ones(low & high)
    return 4 - count1 - count2 - count3, count1, count2, count3
end

function _column_counts!(counts::Matrix{Int}, s::SnpArray)
    full_bytes, trailing_genotypes = divrem(s.m, 4)
    @inbounds for column in axes(s.data, 2)
        count0 = count1 = count2 = count3 = 0
        for byte_index in 1:full_bytes
            byte_counts = _packed_counts(s.data[byte_index, column])
            count0 += byte_counts[1]
            count1 += byte_counts[2]
            count2 += byte_counts[3]
            count3 += byte_counts[4]
        end
        if !iszero(trailing_genotypes)
            byte = s.data[full_bytes + 1, column]
            for offset in 0:(trailing_genotypes - 1)
                genotype = (byte >> (2offset)) & 0x03
                genotype == 0 && (count0 += 1)
                genotype == 1 && (count1 += 1)
                genotype == 2 && (count2 += 1)
                genotype == 3 && (count3 += 1)
            end
        end
        counts[1, column] = count0
        counts[2, column] = count1
        counts[3, column] = count2
        counts[4, column] = count3
    end
    return counts
end

function _row_counts!(counts::Matrix{Int}, s::SnpArray)
    full_bytes, trailing_genotypes = divrem(s.m, 4)
    @inbounds for column in axes(s.data, 2)
        for byte_index in 1:full_bytes
            byte = s.data[byte_index, column]
            row = 4byte_index - 3
            counts[(byte & 0x03) + 1, row] += 1
            counts[((byte >> 2) & 0x03) + 1, row + 1] += 1
            counts[((byte >> 4) & 0x03) + 1, row + 2] += 1
            counts[((byte >> 6) & 0x03) + 1, row + 3] += 1
        end
        if !iszero(trailing_genotypes)
            byte = s.data[full_bytes + 1, column]
            row = 4full_bytes + 1
            for offset in 0:(trailing_genotypes - 1)
                genotype = (byte >> (2offset)) & 0x03
                counts[genotype + 1, row + offset] += 1
            end
        end
    end
    return counts
end

function _counts(s::SnpArray, dims::Integer)
    if isone(dims)
        all(iszero, s.columncounts) && _column_counts!(s.columncounts, s)
        return s.columncounts
    elseif dims == 2
        all(iszero, s.rowcounts) && _row_counts!(s.rowcounts, s)
        return s.rowcounts
    else
        throw(ArgumentError("counts(s::SnpArray, dims=k) only defined for " *
                            "k = 1 or 2"))
    end
end

function _counts(s::AbstractSnpArray, dims::Integer)
    if isone(dims)
        result = zeros(Int, (4, size(s, 2)))
        @inbounds for column in axes(s, 2)
            for row in axes(s, 1)
                result[s[row, column] + 1, column] += 1
            end
        end
        return result
    elseif dims == 2
        result = zeros(Int, (4, size(s, 1)))
        @inbounds for column in axes(s, 2)
            for row in axes(s, 1)
                result[s[row, column] + 1, row] += 1
            end
        end
        return result
    else
        throw(ArgumentError("counts(s::SnpArray, dims=k) only defined for " *
                            "k = 1 or 2"))
    end
end

_counts(s::AbstractSnpArray, ::Colon) = sum(_counts(s, 1), dims=2)

function Base.getindex(s::SnpArray, i::Int)  # Linear indexing
    d, r = divrem(i - 1, s.m)
    s[r + 1, d + 1]
end

@inline function Base.getindex(s::SnpArray, i::Integer, j::Integer)
    @boundscheck checkbounds(s, i, j)
    ip3 = i + 3
    (s.data[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
end

function Base.setindex!(s::SnpArray, x::UInt8, i::Int)  # Linear indexing
    d, r = divrem(i - 1, s.m)
    Base.setindex!(s, x, r + 1, d + 1)
end

@inline function _has_cached_counts(s::SnpArray, row::Integer, column::Integer)
    column_cached = !iszero(
        s.columncounts[1, column] |
        s.columncounts[2, column] |
        s.columncounts[3, column] |
        s.columncounts[4, column],
    )
    row_cached = !iszero(
        s.rowcounts[1, row] |
        s.rowcounts[2, row] |
        s.rowcounts[3, row] |
        s.rowcounts[4, row],
    )
    return column_cached || row_cached
end

function _invalidate_counts!(s::SnpArray)
    @warn "Mutating this SnpArray invalidated its cached summary statistics"
    fill!(s.columncounts, 0)
    fill!(s.rowcounts, 0)
    return s
end

@inline function Base.setindex!(s::SnpArray, x::UInt8, i::Integer, j::Integer)
    @boundscheck checkbounds(s, i, j)
    ip3 = i + 3
    shft = (ip3 & 0x03) << 1
    byte_index = ip3 >> 2
    byte = s.data[byte_index, j]
    old_value = (byte >> shft) & 0x03
    x == old_value && return x
    _has_cached_counts(s, i, j) && _invalidate_counts!(s)
    mask = ~(0x03 << shft)
    s.data[byte_index, j] = (byte & mask) | (x << shft)
    return x
end

Base.eltype(s::SnpArray) = UInt8

Base.length(s::SnpArray) = s.m * size(s.data, 2)

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

Base.size(s::SnpArray) = s.m, size(s.data, 2)

Base.size(s::SnpArray, k::Integer) = 
k == 1 ? s.m : k == 2 ? size(s.data, 2) : k > 2 ? 1 : error("Dimension k out of range")

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

@inline function convert(::Type{T}, x::UInt8, ::Val{1}) where T <: AbstractFloat
    iszero(x) ? zero(T) : isone(x) ? T(NaN) : T(x - 1)
end

@inline function convert(::Type{T}, x::UInt8, ::Val{2}) where T <: AbstractFloat
    iszero(x) ? zero(T) : isone(x) ? T(NaN) : one(T)
end

@inline function convert(::Type{T}, x::UInt8, ::Val{3}) where T <: AbstractFloat
    (iszero(x) || x == 2) ? zero(T) : isone(x) ? T(NaN) : one(T)
end

"""
    Base.copyto!(v, s, model=ADDITIVE_MODEL, center=false, scale=false, impute=false)

Copy SnpArray `s` to numeric vector or matrix `v`.

# Arguments
- `model::Union{Val{1}, Val{2}, Val{3}}=ADDITIVE_MODEL`: `ADDITIVE_MODEL` (default), `DOMINANT_MODEL`, or `RECESSIVE_MODEL`.  
- `center::Bool=false`: center column by mean.
- `scale::Bool=false`: scale column by theoretical variance.
- `impute::Bool=false`: impute missing values by column mean.
"""
function Base.copyto!(
    v::AbstractVecOrMat{T},
    s::AbstractSnpArray;
    model::Union{Val{1}, Val{2}, Val{3}} = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = false
    ) where T <: AbstractFloat
    m, n = size(s, 1), size(s, 2)
    _check_copy_size(v, m, n)
    if !center && !scale && !impute
        @inbounds for j in 1:n
            for i in 1:m
                v[i, j] = SnpArrays.convert(T, s[i, j], model)
            end
        end
        return v
    end
    @inbounds for j in 1:n
        μj, mj = zero(T), 0
        for i in 1:m
            vij = SnpArrays.convert(T, s[i, j], model)
            v[i, j] = vij
            μj += isnan(vij) ? zero(T) : vij
            mj += isnan(vij) ? 0 : 1
        end
        μj /= mj
        σj = model == ADDITIVE_MODEL ? sqrt(μj * (1 - μj / 2)) : sqrt(μj * (1 - μj))
        @simd for i in 1:m
            impute && isnan(v[i, j]) && (v[i, j] = μj)
            center && (v[i, j] -= μj)
            scale && σj > 0 && (v[i, j] /= σj)
        end
    end
    return v
end

@inline function _check_copy_size(v::AbstractVector, rows::Int, columns::Int)
    columns == 1 && length(v) == rows || throw(DimensionMismatch(
        "vector output requires a one-column SnpArray of matching length",
    ))
    return nothing
end

@inline function _check_copy_size(v::AbstractMatrix, rows::Int, columns::Int)
    size(v) == (rows, columns) || throw(DimensionMismatch(
        "output has size $(size(v)); expected ($rows, $columns)",
    ))
    return nothing
end

@inline function _store_genotype!(
    v::AbstractVector{T},
    value::T,
    row::Int,
    ::Int,
) where T
    return setindex!(v, value, row)
end

@inline function _store_genotype!(
    v::AbstractMatrix{T},
    value::T,
    row::Int,
    column::Int,
) where T
    return setindex!(v, value, row, column)
end

@inline function _genotype_values(
    ::Type{T},
    model::Union{Val{1}, Val{2}, Val{3}},
) where T <: AbstractFloat
    return (
        SnpArrays.convert(T, 0x00, model),
        SnpArrays.convert(T, 0x01, model),
        SnpArrays.convert(T, 0x02, model),
        SnpArrays.convert(T, 0x03, model),
    )
end

@inline function _theoretical_std(
    mean::T,
    ::typeof(ADDITIVE_MODEL),
) where T <: AbstractFloat
    return sqrt(mean * (one(T) - mean / T(2)))
end

@inline function _theoretical_std(
    mean::T,
    ::Union{typeof(DOMINANT_MODEL), typeof(RECESSIVE_MODEL)},
) where T <: AbstractFloat
    return sqrt(mean * (one(T) - mean))
end

@inline function _transform_genotype(
    value::T,
    mean::T,
    scale_inverse::T,
    center::Bool,
    scale::Bool,
) where T <: AbstractFloat
    result = center ? value - mean : value
    return scale ? result * scale_inverse : result
end

@inline function _transformed_genotype_values(
    ::Type{T},
    mean::T,
    scale_inverse::T,
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    values = _genotype_values(T, model)
    missing_value = impute ? mean : values[2]
    return (
        _transform_genotype(values[1], mean, scale_inverse, center, scale),
        _transform_genotype(missing_value, mean, scale_inverse, center, scale),
        _transform_genotype(values[3], mean, scale_inverse, center, scale),
        _transform_genotype(values[4], mean, scale_inverse, center, scale),
    )
end

function _copy_packed_column!(
    out::AbstractVecOrMat{T},
    s::SnpArray,
    column::Int,
    values::NTuple{4, T},
    full_bytes::Int,
    trailing_genotypes::Int,
) where T <: AbstractFloat
    @inbounds for byte_index in 1:full_bytes
        byte = s.data[byte_index, column]
        row = 4byte_index - 3
        _store_genotype!(out, values[Int(byte & 0x03) + 1], row, column)
        _store_genotype!(out, values[Int((byte >> 2) & 0x03) + 1],
                         row + 1, column)
        _store_genotype!(out, values[Int((byte >> 4) & 0x03) + 1],
                         row + 2, column)
        _store_genotype!(out, values[Int((byte >> 6) & 0x03) + 1],
                         row + 3, column)
    end
    if !iszero(trailing_genotypes)
        byte = s.data[full_bytes + 1, column]
        row = 4full_bytes + 1
        @inbounds for offset in 0:(trailing_genotypes - 1)
            genotype = Int((byte >> (2offset)) & 0x03) + 1
            _store_genotype!(out, values[genotype], row + offset, column)
        end
    end
    return out
end

function Base.copyto!(
    out::AbstractVecOrMat{T},
    s::SnpArray;
    model::Union{Val{1}, Val{2}, Val{3}} = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = false,
) where T <: AbstractFloat
    rows, columns = size(s)
    _check_copy_size(out, rows, columns)
    full_bytes, trailing_genotypes = divrem(rows, 4)
    if !center && !scale && !impute
        values = _genotype_values(T, model)
        for column in 1:columns
            _copy_packed_column!(out, s, column, values, full_bytes,
                                 trailing_genotypes)
        end
        return out
    end

    counts = _counts(s, 1)
    @inbounds for column in 1:columns
        nonmissing = counts[1, column] + counts[3, column] + counts[4, column]
        numerator = _mean_numerator(counts[3, column], counts[4, column], model)
        column_mean = T(numerator) / T(nonmissing)
        column_std = _theoretical_std(column_mean, model)
        scale_inverse = column_std > zero(T) ? inv(column_std) : one(T)
        values = _transformed_genotype_values(
            T,
            column_mean,
            scale_inverse,
            model,
            center,
            scale,
            impute,
        )
        _copy_packed_column!(out, s, column, values, full_bytes,
                             trailing_genotypes)
    end
    return out
end


"""
    Base.convert(t, s, model=ADDITIVE_MODEL, center=false, scale=false, impute=false)

Convert a SnpArray `s` to a numeric vector or matrix of same shape as `s`.

# Arguments
- `t::Type{AbstractVecOrMat{T}}`: Vector or matrix type.
- `model::Union{Val{1}, Val{2}, Val{3}}=ADDITIVE_MODEL`: `ADDITIVE_MODEL` (default), `DOMINANT_MODEL`, or `RECESSIVE_MODEL`.  
- `center::Bool=false`: center column by mean.
- `scale::Bool=false`: scale column by theoretical variance.
- `impute::Bool=false`: impute missing values by column mean.
"""
function Base.convert(
    ::Type{T},
    s::AbstractSnpArray;
    kwargs...) where T <: Array
    T(s; kwargs...)
end
Array{T,N}(s::AbstractSnpArray; kwargs...) where {T,N} = 
    copyto!(Array{T,N}(undef, size(s)), s; kwargs...)


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
