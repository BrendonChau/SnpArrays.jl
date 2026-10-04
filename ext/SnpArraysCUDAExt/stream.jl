"""Device buffers for one streamed chunk of up to `width` SNPs."""
struct _DeviceChunk{T}
    words::CuMatrix{UInt32}
    values::CuMatrix{T}
    statistics::CuVector{T}
end

_DeviceChunk{T}(word_rows::Int, width::Int) where {T} = _DeviceChunk{T}(
    CuMatrix{UInt32}(undef, word_rows, width), CuMatrix{T}(undef, 4, width),
    CuVector{T}(undef, width))

"""
    _upload_chunk!(buf, chunk, m, width) -> buf

Copy `chunk`'s packed genotypes, repacked as 16-sample words, and its
lookup values into the first `width` columns of `buf`, on the calling
task's stream. When `m` is a multiple of 16 the `.bed` bytes already are
the words and are copied without repacking.
"""
function _upload_chunk!(buf::_DeviceChunk{T}, chunk, m::Int,
    width::Int) where {T}
    word_rows = cld(m, 16)
    data = chunk.s.data
    if m % 16 == 0
        GC.@preserve data begin
            words = unsafe_wrap(Array, Ptr{UInt32}(pointer(data)),
                word_rows * width)
            copyto!(buf.words, 1, words, 1, word_rows * width)
        end
    else
        words = _packed_words(data, m)
        copyto!(buf.words, 1, words, 1, word_rows * width)
    end
    copyto!(buf.values, 1, chunk.values, 1, 4 * width)
    return buf
end

"""
    _chunk_array(buf, stream, width) -> CuSnpArray

`CuSnpArray` over the first `width` columns of `buf`, with `stream`'s
model and flags; its `μ` and `σinv` are placeholders, as the kernels read
only the lookup values.
"""
function _chunk_array(buf::_DeviceChunk{T}, stream::SnpLinAlgStream{T},
    width::Int) where {T}
    statistics = view(buf.statistics, 1:width)
    return CuSnpArray{T}(view(buf.words, :, 1:width), stream.m,
        stream.model, stream.center, stream.scale, stream.impute, statistics,
        statistics, view(buf.values, :, 1:width))
end

"""
    _foreach_chunk(f, stream::SnpLinAlgStream{T}) -> nothing

Call `f(cols, s)` for each chunk of `stream`, in order, with `s` a
`CuSnpArray` over the chunk's copy in one reused device buffer.
"""
function _foreach_chunk(f::F, stream::SnpLinAlgStream{T}) where {F, T}
    m = size(stream, 1)
    for buffer in stream.full_buffers
        pin(buffer.s.data)
        pin(buffer.values)
    end
    buf = _DeviceChunk{T}(cld(m, 16), stream.width)
    for (cols, chunk) in stream
        _upload_chunk!(buf, chunk, m, length(cols))
        # The stream's prefetch reuses this pinned host buffer next.
        synchronize()
        f(cols, _chunk_array(buf, stream, length(cols)))
    end
    return nothing
end

"""
    streamed_mul!(out::CuMatrix{T}, stream::SnpLinAlgStream{T},
                  X::CuMatrix{T}; transpose=false) -> out

`out = A * X` (or `transpose(A) * X`) on the GPU in one pass over
`stream`.
"""
function streamed_mul!(
    out::CuMatrix{T}, stream::SnpLinAlgStream{T}, X::CuMatrix{T};
    transpose::Bool=false,
) where {T <: Union{Float32, Float64}}
    m, n = size(stream)
    k = size(X, 2)
    rows, inner = transpose ? (n, m) : (m, n)
    size(X, 1) == inner || throw(DimensionMismatch(
        "right-hand side has $(size(X, 1)) rows; expected $inner"))
    size(out) == (rows, k) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((rows, k))"))
    fill!(out, zero(T))
    partial = transpose ? nothing : similar(out)
    _foreach_chunk(stream) do cols, s
        if transpose
            piece = CuMatrix{T}(undef, length(cols), k)
            mul!(piece, Base.transpose(s), X)
            view(out, cols, :) .= piece
            unsafe_free!(piece)
        else
            rhs = X[cols, :]
            mul!(partial, s, rhs)
            out .+= partial
            unsafe_free!(rhs)
        end
    end
    partial === nothing || unsafe_free!(partial)
    return out
end

"""
    streamed_grm_mul!(V::CuMatrix, stream::SnpLinAlgStream, Q::CuMatrix;
                      scale) -> V
    streamed_grm_mul!(V::CuMatrix, U::CuMatrix, stream::SnpLinAlgStream,
                      Q::CuMatrix; scale) -> V

GPU `V = A * (scale .* (transpose(A) * Q))` in one pass over `stream`;
with `U`, also store the unscaled `transpose(A) * Q` in `U`. Element types
and `scale` follow the host method.
"""
function streamed_grm_mul!(
    V::CuMatrix{TV}, stream::SnpLinAlgStream{TS}, Q::CuMatrix{TV};
    scale::Union{TV, AbstractVector{TV}}=inv(TV(size(stream, 2))),
) where {TV <: Union{Float32, Float64}, TS <: Union{Float32, Float64}}
    return _device_grm_mul!(V, nothing, stream, Q, scale)
end

function streamed_grm_mul!(
    V::CuMatrix{TV}, U::CuMatrix{TV}, stream::SnpLinAlgStream{TS},
    Q::CuMatrix{TV};
    scale::Union{TV, AbstractVector{TV}}=inv(TV(size(stream, 2))),
) where {TV <: Union{Float32, Float64}, TS <: Union{Float32, Float64}}
    return _device_grm_mul!(V, U, stream, Q, scale)
end

"""
    _device_grm_mul!(V, U, stream, Q, scale) -> V

Check the arguments, zero `V`, and add each chunk's
`A_c * (s_c .* (A_cᵀ * Q))` to it, with the products in the stream's
element type.
"""
function _device_grm_mul!(
    V::CuMatrix{TV}, U::Union{Nothing, CuMatrix{TV}},
    stream::SnpLinAlgStream{TS}, Q::CuMatrix{TV},
    scale::Union{TV, AbstractVector{TV}},
) where {TV <: Union{Float32, Float64}, TS <: Union{Float32, Float64}}
    (TV === TS || (TV === Float64 && TS === Float32)) || throw(ArgumentError(
        "V has element type $TV but the stream has element type $TS; " *
        "expected equal types, or Float64 with a Float32 stream",
    ))
    m, n = size(stream)
    k = size(Q, 2)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, k)
    mixed = TV !== TS
    Q_stream = mixed ? CuMatrix{TS}(Q) : Q
    weights = scale isa Number ? TS(scale) : CuVector{TS}(scale)
    product = CuMatrix{TS}(undef, m, k)
    fill!(V, zero(TV))
    _foreach_chunk(stream) do cols, s
        piece = CuMatrix{TS}(undef, length(cols), k)
        mul!(piece, transpose(s), Q_stream)
        U === nothing || (view(U, cols, :) .= piece)
        piece .*= weights isa Number ? weights : view(weights, cols)
        mul!(product, s, piece)
        V .+= product
        unsafe_free!(piece)
    end
    unsafe_free!(product)
    weights isa Number || unsafe_free!(weights)
    mixed && unsafe_free!(Q_stream)
    return V
end
