"""
    streamed_mul!(out, stream::SnpLinAlgStream, X; transpose = false,
                  slots = 3) -> out

Compute `out = A * X`, or `out = transpose(A) * X`, for the genotype matrix
`A` behind `stream` in one pass over its chunks.

# Keywords
- `transpose::Bool = false`: multiply by `transpose(A)`
- `slots::Integer = 3`: ignored

# Throws
- `DimensionMismatch`: `X` or `out` does not match `size(stream)`
"""
function streamed_mul!(
    out::AbstractMatrix{T},
    stream::SnpLinAlgStream{T},
    X::AbstractMatrix{T};
    transpose::Bool = false,
    slots::Integer = 3,
) where T <: AbstractFloat
    m, n = size(stream)
    k = size(X, 2)
    rows, inner = transpose ? (n, m) : (m, n)
    size(X, 1) == inner || throw(DimensionMismatch(
        "right-hand side has $(size(X, 1)) rows; expected $inner",
    ))
    size(out) == (rows, k) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((rows, k))",
    ))
    fill!(out, zero(T))
    for (cols, chunk) in stream
        if transpose
            mul!(view(out, cols, :), Base.transpose(chunk), X)
        else
            mul!(out, chunk, view(X, cols, :), one(T), one(T))
        end
    end
    return out
end

"""
    _check_streamed_grm_dims(V, Q, U, scale, m, n, k)

Throw a `DimensionMismatch` naming the offending argument unless `V` is
`m × k`, `Q` has `m` rows, `U` (when given) is `n × k`, and a vector
`scale` has length `n`.
"""
function _check_streamed_grm_dims(
    V::AbstractMatrix,
    Q::AbstractMatrix,
    U::Union{Nothing, AbstractMatrix},
    scale::Union{AbstractFloat, AbstractVector{<:AbstractFloat}},
    m::Int,
    n::Int,
    k::Int,
)
    size(V) == (m, k) || throw(DimensionMismatch(
        "V has size $(size(V)); expected $((m, k))",
    ))
    size(Q, 1) == m || throw(DimensionMismatch(
        "Q has $(size(Q, 1)) rows; expected $m",
    ))
    scale isa AbstractVector && length(scale) != n && throw(DimensionMismatch(
        "scale has length $(length(scale)); expected $n",
    ))
    if U !== nothing
        size(U) == (n, k) || throw(DimensionMismatch(
            "U has size $(size(U)); expected $((n, k))",
        ))
    end
    return nothing
end

"""
    _chunk_scale(scale, cols)

Return `scale` for the SNP columns `cols`: a scalar applies to every
column, and a per-SNP vector is restricted to `cols`.
"""
_chunk_scale(scale::AbstractFloat, cols::UnitRange{Int}) = scale
_chunk_scale(scale::AbstractVector{<:AbstractFloat}, cols::UnitRange{Int}) =
    view(scale, cols)

"""
    _streamed_grm_mul!(V, stream, Q_kernel, scale_kernel, U, V_scratch)

Run the per-chunk products in `Q_kernel`'s type and accumulate into the
already zeroed `V`, through `V_scratch` when given.
"""
function _streamed_grm_mul!(
    V::AbstractMatrix{TV},
    stream::SnpLinAlgStream{TS},
    Q_kernel::AbstractMatrix{TS},
    scale_kernel::Union{TS, AbstractVector{TS}},
    U::Union{Nothing, AbstractMatrix{TV}},
    V_scratch::Union{Nothing, Matrix{TS}},
) where {TV <: AbstractFloat, TS <: AbstractFloat}
    k = size(Q_kernel, 2)
    buffers = Dict{Int, Matrix{TS}}()
    for (cols, chunk) in stream
        width = length(cols)
        chunk_output = get!(buffers, width) do
            Matrix{TS}(undef, width, k)
        end
        mul!(chunk_output, transpose(chunk), Q_kernel)
        U !== nothing && copyto!(view(U, cols, :), chunk_output)
        chunk_output .*= _chunk_scale(scale_kernel, cols)
        if V_scratch === nothing
            mul!(V, chunk, chunk_output, 1, 1)
        else
            mul!(V_scratch, chunk, chunk_output)
            V .+= V_scratch
        end
    end
    return V
end

"""
    streamed_grm_mul!(V, stream::SnpLinAlgStream, Q; scale, U = nothing)
        -> V

Compute `V = A * (scale .* (transpose(A) * Q))` for the genotype matrix `A`
behind `stream` in one pass over its chunks.

`V`, `Q`, and `stream` share one element type, except that a `Float32`
stream accepts `Float64` `V` and `Q`: each chunk's product then runs in
single precision and accumulates into `V` in double precision.

# Keywords
- `scale = inv(size(stream, 2))`: a scalar, or a vector with one weight per
  SNP, with the element type of `V`
- `U = nothing`: a matrix that receives the unscaled `transpose(A) * Q`

# Throws
- `DimensionMismatch`: `V`, `Q`, `U`, or a vector `scale` does not match
  `size(stream)`
- `TypeError`: `scale` does not have the element type of `V`
"""
function streamed_grm_mul!(
    V::AbstractMatrix{T},
    stream::SnpLinAlgStream{T},
    Q::AbstractMatrix{T};
    scale::Union{T, AbstractVector{T}} = inv(T(size(stream, 2))),
    U::Union{Nothing, AbstractMatrix{T}} = nothing,
) where T <: AbstractFloat
    m, n = size(stream)
    k = size(Q, 2)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, k)
    fill!(V, zero(T))
    _streamed_grm_mul!(V, stream, Q, scale, U, nothing)
    return V
end

function streamed_grm_mul!(
    V::AbstractMatrix{Float64},
    stream::SnpLinAlgStream{Float32},
    Q::AbstractMatrix{Float64};
    scale::Union{Float64, AbstractVector{Float64}} = inv(size(stream, 2)),
    U::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)
    m, n = size(stream)
    k = size(Q, 2)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, k)
    fill!(V, zero(Float64))
    Q32 = Matrix{Float32}(Q)
    scale32 = Float32.(scale)
    V_scratch = Matrix{Float32}(undef, m, k)
    _streamed_grm_mul!(V, stream, Q32, scale32, U, V_scratch)
    return V
end
