"""
    MtlSnpArray{T}(s; model=ADDITIVE_MODEL, center=false, scale=false,
                   impute=true)

Copy a `SnpArray` to an Apple Metal GPU to perform linear algebra
operations. Constructors and `mul!` methods are defined by the
`SnpArraysMetalExt` extension, loaded by `using Metal`. Metal has no
`Float64`; use `T = Float32`.

Genotypes are stored as packed `UInt32` words, 16 samples per word, and
decoded through a per-column lookup table `values` (4 x n) that holds the
transformed value of each 2-bit PLINK code.

# Arguments
- s: a `SnpArray`.
- model: one of `ADDITIVE_MODEL`(default), `DOMINANT_MODEL`,
  `RECESSIVE_MODEL`.
- center: whether to center (default: false).
- scale: whether to scale to standard deviation 1 (default: false).
- impute: whether to impute missing values with the column mean (default:
  true). With `impute=false` a missing genotype counts as 0 before
  centering and scaling.
"""
struct MtlSnpArray{T, D <: AbstractMatrix{UInt32}, V <: AbstractVector{T},
                   W <: AbstractMatrix{T}} <: AbstractMatrix{UInt8}
    data::D
    m::Int
    model::Union{Val{1}, Val{2}, Val{3}}
    center::Bool
    scale::Bool
    impute::Bool
    μ::V
    σinv::V
    values::W
end

function MtlSnpArray{T}(data::D, m::Integer,
    model::Union{Val{1}, Val{2}, Val{3}}, center::Bool, scale::Bool,
    impute::Bool, μ::V, σinv::V, values::W
) where {T, D <: AbstractMatrix{UInt32}, V <: AbstractVector{T},
         W <: AbstractMatrix{T}}
    MtlSnpArray{T, D, V, W}(data, m, model, center, scale, impute, μ, σinv,
        values)
end

size(s::MtlSnpArray) = s.m, size(s.data, 2)
eltype(::MtlSnpArray{T}) where {T} = T
