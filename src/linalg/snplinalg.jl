struct SnpLinAlg{T} <: AbstractMatrix{T}
    s::SnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    center::Bool
    scale::Bool
    impute::Bool
    μ::Vector{T}
    σinv::Vector{T}
    values::Matrix{T}
    # Packing workspace for the register-tiled matrix products, one slice
    # per spawned task. Resized on the first product of a given shape and
    # reused after, so a repeated product allocates nothing here.
    panel::Vector{T}
    # Transposed genotype codes (one byte per sample and 4-SNP block) for
    # the lookup-table `A*X` kernel; same lifetime as `panel`.
    blk::Vector{UInt8}
    # Guards `panel` and `blk`, which every matrix product writes.
    lock::ReentrantLock
end

AbstractSnpLinAlg = Union{SnpLinAlg, SubArray{T, 1, SnpLinAlg{T}},
    SubArray{T, 2, SnpLinAlg{T}}} where T

"""
    SnpLinAlg{T}(s; model=ADDITIVE_MODEL, center=false, scale=false,
                 impute=true)

Wrap a `SnpArray` for direct linear algebra without materializing its genotypes.
Missing genotypes use the column mean before centering and scaling when
`impute=true`, and use `NaN` otherwise.

# Arguments
- `s`: a `SnpArray`
- `model`: `ADDITIVE_MODEL`, `DOMINANT_MODEL`, or `RECESSIVE_MODEL`
- `center`: center each column when `true`
- `scale`: scale each column to unit standard deviation when `true`
- `impute`: replace missing genotypes with the column mean when `true`
"""
function SnpLinAlg{T}(
    s::AbstractSnpArray;
    model = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = true,
) where T <: AbstractFloat
    model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL) ||
        throw(ArgumentError("unrecognized model $model"))
    means = Vector{T}(undef, size(s, 2))
    inverse_standard_deviations = Vector{T}(undef, size(s, 2))
    values = Matrix{T}(undef, 4, size(s, 2))
    _fill_statistics!(means, inverse_standard_deviations, values, s, model,
                      center, scale, impute)
    return SnpLinAlg{T}(s, model, center, scale, impute, means,
                        inverse_standard_deviations, values, T[], UInt8[],
                        ReentrantLock())
end

"""
    _fill_statistics!(means, inverse_standard_deviations, values, s, model,
        center, scale, impute) -> values

Recompute `means`, `inverse_standard_deviations`, and `values` in place
from the genotype counts of `s`.
"""
function _fill_statistics!(
    means::Vector{T},
    inverse_standard_deviations::Vector{T},
    values::Matrix{T},
    s::AbstractSnpArray,
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    mean!(means, s; dims=1, model=model)
    @inbounds @simd for column in eachindex(means)
        column_mean = means[column]
        variance = model == ADDITIVE_MODEL ?
                   column_mean * (one(T) - column_mean / T(2)) :
                   column_mean * (one(T) - column_mean)
        standard_deviation = sqrt(variance)
        inverse_standard_deviations[column] =
            standard_deviation > zero(T) ? inv(standard_deviation) : one(T)
    end
    _fill_genotype_values!(values, means, inverse_standard_deviations, model,
                           center, scale, impute)
    return values
end

"""
    _refill_statistics!(sla::SnpLinAlg) -> sla

Recompute `sla.μ`, `sla.σinv`, and `sla.values` from `sla.s`; the caller
must zero `sla.s.columncounts` first if the data changed.
"""
function _refill_statistics!(sla::SnpLinAlg{T}) where T <: AbstractFloat
    _fill_statistics!(sla.μ, sla.σinv, sla.values, sla.s, sla.model,
                      sla.center, sla.scale, sla.impute)
    return sla
end

function _fill_genotype_values!(
    values::AbstractMatrix{T},
    means::AbstractVector{T},
    inverse_standard_deviations::AbstractVector{T},
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    @inbounds for column in axes(values, 2)
        transformed = _transformed_genotype_values(
            T, means[column], inverse_standard_deviations[column], model,
            center, scale, impute,
        )
        for code in 1:4
            values[code, column] = transformed[code]
        end
    end
    return values
end

Base.size(sla::SnpLinAlg) = size(sla.s)
Base.size(sla::SnpLinAlg, dimension::Integer) = size(sla.s, dimension)
Base.eltype(::SnpLinAlg{T}) where T = T

@inline function Base.getindex(sla::SnpLinAlg, row::Int, column::Int)
    code = getindex(sla.s, row, column)
    return @inbounds sla.values[Int(code) + 1, column]
end

"""
    Base.copyto!(destination, source)

Copy a `SnpLinAlg` or one of its views to a floating-point vector or matrix.
"""
function Base.copyto!(
    destination::AbstractVecOrMat{T},
    source::AbstractSnpLinAlg,
) where T <: AbstractFloat
    size(destination) == size(source) || throw(DimensionMismatch(
        "destination has size $(size(destination)); expected $(size(source))",
    ))
    for index in eachindex(destination, source)
        @inbounds destination[index] = source[index]
    end
    return destination
end

"""
    Base.convert(T, source)

Convert a `SnpLinAlg` or one of its views to an array with the same shape.
"""
Base.convert(::Type{T}, source::AbstractSnpLinAlg) where T <: Array = T(source)
Array{T, N}(source::AbstractSnpLinAlg) where {T, N} =
    copyto!(Array{T, N}(undef, size(source)), source)
