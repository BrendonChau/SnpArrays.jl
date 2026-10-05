"""
    CuSnpArray{T}(s; model=ADDITIVE_MODEL, center=false, scale=false,
                  impute=true)

Copy a `SnpArray` to a CUDA GPU to perform linear algebra operations.
Constructors and `mul!` methods are defined by the `SnpArraysCUDAExt`
extension, loaded by `using CUDA`.

Genotypes are stored as packed `UInt32` words, 16 samples per word, and
decoded through a per-column lookup table `values` (4 x n) that holds the
transformed value of each 2-bit PLINK code.

`Float32` matrix products can run on tensor cores; this is experimental and
off unless enabled with `SnpArrays.cuda_tensor_cores!(true)`.

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
struct CuSnpArray{T, D <: AbstractMatrix{UInt32}, V <: AbstractVector{T},
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

function CuSnpArray{T}(data::D, m::Integer,
    model::Union{Val{1}, Val{2}, Val{3}}, center::Bool, scale::Bool,
    impute::Bool, μ::V, σinv::V, values::W
) where {T, D <: AbstractMatrix{UInt32}, V <: AbstractVector{T},
         W <: AbstractMatrix{T}}
    CuSnpArray{T, D, V, W}(data, m, model, center, scale, impute, μ, σinv,
        values)
end

size(s::CuSnpArray) = s.m, size(s.data, 2)
eltype(::CuSnpArray{T}) where {T} = T

"""
Whether `Float32` matrix products on a `CuSnpArray` may use the experimental
tensor-core kernels; set with `cuda_tensor_cores!`.
"""
const CUDA_TENSOR_CORES = Ref(false)

"""
    cuda_tensor_cores!(enable::Bool) -> Bool

Enable or disable the experimental tensor-core kernels for `Float32` matrix
products on a `CuSnpArray`, and return `enable`. They are disabled by
default and take effect only on a GPU of compute capability 8.0 or newer;
otherwise the tiled kernels run.
"""
cuda_tensor_cores!(enable::Bool) = CUDA_TENSOR_CORES[] = enable

"""
    _packed_words(data::AbstractMatrix{UInt8}, m::Integer)

Repack PLINK bytes (4 samples per byte, column major) into a
`Matrix{UInt32}` of size `(cld(m, 16), size(data, 2))`, 16 samples per
word, with the bits of samples past `m` zeroed.
"""
function _packed_words(data::AbstractMatrix{UInt8}, m::Integer)
    n = size(data, 2)
    size(data, 1) == cld(m, 4) || throw(DimensionMismatch(
        "data has $(size(data, 1)) rows; expected $(cld(m, 4))",
    ))
    padded = zeros(UInt8, 4 * cld(m, 16), n)
    padded[1:size(data, 1), :] .= data
    remainder = mod(m, 4)
    if remainder != 0
        mask = UInt8((1 << (2 * remainder)) - 1)
        padded[cld(m, 4), :] .&= mask
    end
    words = reinterpret(UInt32, vec(padded))
    return collect(reshape(words, cld(m, 16), n))
end
