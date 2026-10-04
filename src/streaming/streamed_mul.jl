"""
    streamed_mul!(out, stream::SnpLinAlgStream, X; transpose = false) -> out

Compute `out = A * X`, or `out = transpose(A) * X`, for the genotype matrix
`A` behind `stream` in one pass over its chunks.

# Keywords
- `transpose::Bool = false`: multiply by `transpose(A)`

# Throws
- `DimensionMismatch`: `X` or `out` does not match `size(stream)`
"""
function streamed_mul!(
    out::AbstractMatrix{T},
    stream::SnpLinAlgStream{T},
    X::AbstractMatrix{T};
    transpose::Bool = false,
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
    return transpose ? _streamed_mul_transposed!(out, stream, X) :
           _streamed_mul_forward!(out, stream, X)
end

"""
    streamed_grm_mul!(V, stream::SnpLinAlgStream, Q; scale) -> V
    streamed_grm_mul!(V, U, stream::SnpLinAlgStream, Q; scale) -> V

Compute `V = A * (scale .* (transpose(A) * Q))` for the genotype matrix `A`
behind `stream` in one pass over its chunks. With `U`, also store the
unscaled `transpose(A) * Q` in `U`.

`V`, `U`, `Q`, and `stream` share one element type, except that a `Float32`
stream accepts `Float64` `V`, `U`, and `Q`: each chunk's product then runs
in single precision and accumulates into `V` in double precision.

# Keywords
- `scale = inv(size(stream, 2))`: a scalar, or a vector with one weight per
  SNP, with the element type of `V`

# Throws
- `ArgumentError`: the element types of `V` and `stream` are not a supported
  pair
- `DimensionMismatch`: `V`, `U`, `Q`, or a vector `scale` does not match
  `size(stream)`
- `TypeError`: `scale` does not have the element type of `V`
"""
function streamed_grm_mul!(
    V::AbstractMatrix{TV},
    stream::SnpLinAlgStream{TS},
    Q::AbstractMatrix{TV};
    scale::Union{TV, AbstractVector{TV}} = inv(TV(size(stream, 2))),
) where {TV <: AbstractFloat, TS <: AbstractFloat}
    return _streamed_grm_mul!(V, nothing, stream, Q, scale)
end

function streamed_grm_mul!(
    V::AbstractMatrix{TV},
    U::AbstractMatrix{TV},
    stream::SnpLinAlgStream{TS},
    Q::AbstractMatrix{TV};
    scale::Union{TV, AbstractVector{TV}} = inv(TV(size(stream, 2))),
) where {TV <: AbstractFloat, TS <: AbstractFloat}
    return _streamed_grm_mul!(V, U, stream, Q, scale)
end

"""
    _streamed_grm_mul!(V, U, stream, Q, scale) -> V

Check the arguments, zero `V`, and add each chunk's
`A_c * (s_c .* (A_cᵀ * Q))` to it, in single precision for a `Float32` stream
with `Float64` `V` and `Q`.
"""
function _streamed_grm_mul!(
    V::AbstractMatrix{TV},
    U::Union{Nothing, AbstractMatrix{TV}},
    stream::SnpLinAlgStream{TS},
    Q::AbstractMatrix{TV},
    scale::Union{TV, AbstractVector{TV}},
) where {TV <: AbstractFloat, TS <: AbstractFloat}
    (TV === TS || (TV === Float64 && TS === Float32)) || throw(ArgumentError(
        "V has element type $TV but the stream has element type $TS; " *
        "expected equal types, or Float64 with a Float32 stream",
    ))
    m, n = size(stream)
    k = size(Q, 2)
    size(V) == (m, k) || throw(DimensionMismatch(
        "V has size $(size(V)); expected $((m, k))",
    ))
    size(Q, 1) == m || throw(DimensionMismatch(
        "Q has $(size(Q, 1)) rows; expected $m",
    ))
    U === nothing || size(U) == (n, k) || throw(DimensionMismatch(
        "U has size $(size(U)); expected $((n, k))",
    ))
    scale isa TV || length(scale) == n || throw(DimensionMismatch(
        "scale has length $(length(scale)); expected $n",
    ))

    mixed = TV !== TS
    Q_stream = mixed ? Matrix{TS}(Q) : Q
    weights = mixed ? TS.(scale) : scale
    product = mixed ? Matrix{TS}(undef, m, k) : nothing
    full = Matrix{TS}(undef, stream.width, k)

    fill!(V, zero(TV))
    for (cols, chunk) in stream
        # A file's short last chunk gets an exact-size matrix.
        AtQ = length(cols) == stream.width ? full :
              Matrix{TS}(undef, length(cols), k)
        mul!(AtQ, transpose(chunk), Q_stream)
        U === nothing || copyto!(view(U, cols, :), AtQ)
        AtQ .*= _chunk_scale(weights, cols)
        if product === nothing
            mul!(V, chunk, AtQ, one(TS), one(TS))
        else
            mul!(product, chunk, AtQ)
            V .+= product
        end
    end
    return V
end

"""
    _chunk_scale(scale, cols)

Return `scale` for the SNP columns `cols`: a scalar for every column, or the
matching slice of a per-SNP vector.
"""
_chunk_scale(scale::AbstractFloat, cols::UnitRange{Int}) = scale
_chunk_scale(scale::AbstractVector{<:AbstractFloat}, cols::UnitRange{Int}) =
    view(scale, cols)

"""
    _streamed_mul_forward!(out, stream, X) -> out

Set `out = A * X` by accumulating each chunk's `A_c * X[cols, :]`.
"""
function _streamed_mul_forward!(
    out::AbstractMatrix{T},
    stream::SnpLinAlgStream{T},
    X::AbstractMatrix{T},
) where T <: AbstractFloat
    fill!(out, zero(T))
    for (cols, chunk) in stream
        mul!(out, chunk, view(X, cols, :), one(T), one(T))
    end
    return out
end

"""
    _streamed_mul_transposed!(out, stream, X) -> out

Set `out = transpose(A) * X`, one chunk per block of rows of `out`.
"""
function _streamed_mul_transposed!(
    out::AbstractMatrix{T},
    stream::SnpLinAlgStream{T},
    X::AbstractMatrix{T},
) where T <: AbstractFloat
    for (cols, chunk) in stream
        mul!(view(out, cols, :), transpose(chunk), X)
    end
    return out
end
