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
        for i in 1:m
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
