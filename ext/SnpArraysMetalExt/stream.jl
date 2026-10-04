"""Device buffers for one streamed chunk of up to `width` SNPs."""
struct _DeviceChunk{T}
    words::MtlMatrix{UInt32}
    values::MtlMatrix{T}
    statistics::MtlVector{T}
end

_DeviceChunk{T}(word_rows::Int, width::Int) where {T} = _DeviceChunk{T}(
    MtlMatrix{UInt32}(undef, word_rows, width), MtlMatrix{T}(undef, 4, width),
    MtlVector{T}(undef, width))

"""
    _upload_chunk!(buf, chunk, m, width) -> buf

Copy `chunk`'s packed genotypes, repacked as 16-sample words, and its
lookup values into the first `width` columns of `buf`. Metal.jl copies
from host arrays synchronously, so the host chunk may be reused on return.
"""
function _upload_chunk!(buf::_DeviceChunk{T}, chunk, m::Int,
    width::Int) where {T}
    words = _packed_words(chunk.s.data, m)
    copyto!(buf.words, 1, words, 1, cld(m, 16) * width)
    copyto!(buf.values, 1, chunk.values, 1, 4 * width)
    return buf
end

"""
    _chunk_array(buf, stream, width) -> MtlSnpArray

`MtlSnpArray` over the first `width` columns of `buf`, with `stream`'s
model and flags; its `μ` and `σinv` are placeholders, as the kernels read
only the lookup values.
"""
function _chunk_array(buf::_DeviceChunk{T}, stream::SnpLinAlgStream{T},
    width::Int) where {T}
    statistics = view(buf.statistics, 1:width)
    return MtlSnpArray{T}(view(buf.words, :, 1:width), stream.m,
        stream.model, stream.center, stream.scale, stream.impute, statistics,
        statistics, view(buf.values, :, 1:width))
end

"""
    _foreach_chunk(f, stream::SnpLinAlgStream{T}) -> nothing

Call `f(cols, s)` for each chunk of `stream`, in order, with `s` an
`MtlSnpArray` over the chunk's copy in one reused device buffer.
"""
function _foreach_chunk(f::F, stream::SnpLinAlgStream{T}) where {F, T}
    m = size(stream, 1)
    buf = _DeviceChunk{T}(cld(m, 16), stream.width)
    for (cols, chunk) in stream
        _upload_chunk!(buf, chunk, m, length(cols))
        f(cols, _chunk_array(buf, stream, length(cols)))
    end
    return nothing
end

"""
    _check_stream_type(stream) -> nothing

Throw an `ArgumentError` unless `stream` holds `Float32`, the only
floating-point type on Metal.
"""
function _check_stream_type(::SnpLinAlgStream{T}) where {T}
    T <: Float32 || throw(ArgumentError(
        "the stream has element type $T; Metal supports only Float32, " *
        "so use SnpLinAlgStream{Float32}",
    ))
    return nothing
end

"""
    streamed_mul!(out::MtlMatrix{Float32}, stream::SnpLinAlgStream{Float32},
                  X::MtlMatrix{Float32}; transpose=false) -> out

`out = A * X` (or `transpose(A) * X`) on the GPU in one pass over
`stream`.
"""
function streamed_mul!(
    out::MtlMatrix{Float32}, stream::SnpLinAlgStream{Float32},
    X::MtlMatrix{Float32}; transpose::Bool=false,
)
    m, n = size(stream)
    k = size(X, 2)
    rows, inner = transpose ? (n, m) : (m, n)
    size(X, 1) == inner || throw(DimensionMismatch(
        "right-hand side has $(size(X, 1)) rows; expected $inner"))
    size(out) == (rows, k) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((rows, k))"))
    fill!(out, 0.0f0)
    partial = transpose ? nothing : similar(out)
    _foreach_chunk(stream) do cols, s
        if transpose
            piece = MtlMatrix{Float32}(undef, length(cols), k)
            mul!(piece, Base.transpose(s), X)
            view(out, cols, :) .= piece
        else
            mul!(partial, s, X[cols, :])
            out .+= partial
        end
    end
    return out
end

function streamed_mul!(
    out::MtlMatrix{Float32}, stream::SnpLinAlgStream, X::MtlMatrix{Float32};
    transpose::Bool=false,
)
    _check_stream_type(stream)
    return streamed_mul!(out, stream, X; transpose)
end

"""
    streamed_grm_mul!(V::MtlMatrix, stream::SnpLinAlgStream, Q::MtlMatrix;
                      scale) -> V
    streamed_grm_mul!(V::MtlMatrix, U::MtlMatrix, stream::SnpLinAlgStream,
                      Q::MtlMatrix; scale) -> V

GPU `V = A * (scale .* (transpose(A) * Q))` in one pass over `stream`;
with `U`, also store `transpose(A) * Q` in `U`. Matrices, `scale`, and the
stream are `Float32`.
"""
function streamed_grm_mul!(
    V::MtlMatrix{Float32}, stream::SnpLinAlgStream{TS}, Q::MtlMatrix{Float32};
    scale::Union{Float32, AbstractVector{Float32}}=inv(
        Float32(size(stream, 2))),
) where {TS <: AbstractFloat}
    return _device_grm_mul!(V, nothing, stream, Q, scale)
end

function streamed_grm_mul!(
    V::MtlMatrix{Float32}, U::MtlMatrix{Float32},
    stream::SnpLinAlgStream{TS}, Q::MtlMatrix{Float32};
    scale::Union{Float32, AbstractVector{Float32}}=inv(
        Float32(size(stream, 2))),
) where {TS <: AbstractFloat}
    return _device_grm_mul!(V, U, stream, Q, scale)
end

"""
    _device_grm_mul!(V, U, stream, Q, scale) -> V

Check the arguments, zero `V`, and add each chunk's
`A_c * (s_c .* (A_cᵀ * Q))` to it.
"""
function _device_grm_mul!(
    V::MtlMatrix{Float32}, U::Union{Nothing, MtlMatrix{Float32}},
    stream::SnpLinAlgStream, Q::MtlMatrix{Float32},
    scale::Union{Float32, AbstractVector{Float32}},
)
    _check_stream_type(stream)
    m, n = size(stream)
    k = size(Q, 2)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, k)
    weights = scale isa Number ? scale : MtlVector{Float32}(scale)
    product = MtlMatrix{Float32}(undef, m, k)
    fill!(V, 0.0f0)
    _foreach_chunk(stream) do cols, s
        piece = MtlMatrix{Float32}(undef, length(cols), k)
        mul!(piece, transpose(s), Q)
        U === nothing || (view(U, cols, :) .= piece)
        piece .*= weights isa Number ? weights : view(weights, cols)
        mul!(product, s, piece)
        V .+= product
    end
    return V
end
