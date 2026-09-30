"""
    CuSnpArray{T}(s; model=ADDITIVE_MODEL, center=false, scale=false, impute=true)

Copy a `SnpArray` to a CUDA GPU to perform linear algebera operations.
Constructors and `mul!` methods are defined by the `SnpArraysCUDAExt`
extension, loaded by `using CUDA`.

# Arguments
- s: a `SnpArray`.
- model: one of `ADDITIVE_MODEL`(default), `DOMINANT_MODEL`, `RECESSIVE_MODEL`.
- center: whether to center (default: false).
- scale: whether to scale to standard deviation 1 (default: false).
- impute: whether to impute missing value with column mean (default: true).
"""
struct CuSnpArray{T, M <: AbstractMatrix{UInt8}, V <: AbstractVector{T}} <:
       AbstractMatrix{UInt8}
    data::M
    m::Int
    model::Union{Val{1}, Val{2}, Val{3}}
    center::Bool
    scale::Bool
    impute::Bool
    μ::V
    σinv::V
    storagev1::V
    storagev2::V
end

function CuSnpArray{T}(data::M, m::Integer,
    model::Union{Val{1}, Val{2}, Val{3}}, center::Bool, scale::Bool,
    impute::Bool, μ::V, σinv::V, storagev1::V, storagev2::V
) where {T, M <: AbstractMatrix{UInt8}, V <: AbstractVector{T}}
    CuSnpArray{T, M, V}(data, m, model, center, scale, impute, μ, σinv,
        storagev1, storagev2)
end

size(s::CuSnpArray) = s.m, size(s.data, 2)
eltype(s::CuSnpArray) = eltype(s.μ)
