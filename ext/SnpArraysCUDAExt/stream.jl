"""Device buffers for one streamed chunk of up to `width` SNPs."""
struct _StreamSlot{T}
    words::CuMatrix{UInt32}
    values::CuMatrix{T}
    statistics::CuVector{T}
end

_StreamSlot{T}(word_rows::Int, width::Int) where {T} = _StreamSlot{T}(
    CuMatrix{UInt32}(undef, word_rows, width), CuMatrix{T}(undef, 4, width),
    CuVector{T}(undef, width))

"""
    _upload_chunk!(slot, chunk, m, width) -> slot

Copy `chunk`'s packed genotypes, repacked as 16-sample words, and its
lookup values into the first `width` columns of `slot`, on the calling
task's stream. When `m` is a multiple of 16 the `.bed` bytes already are
the words and are copied without repacking.
"""
function _upload_chunk!(slot::_StreamSlot{T}, chunk, m::Int,
    width::Int) where {T}
    word_rows = cld(m, 16)
    data = chunk.s.data
    if m % 16 == 0
        GC.@preserve data begin
            words = unsafe_wrap(Array, Ptr{UInt32}(pointer(data)),
                word_rows * width)
            copyto!(slot.words, 1, words, 1, word_rows * width)
        end
    else
        words = _packed_words(data, m)
        copyto!(slot.words, 1, words, 1, word_rows * width)
    end
    copyto!(slot.values, 1, chunk.values, 1, 4 * width)
    return slot
end

"""
    _chunk_array(slot, stream, width) -> CuSnpArray

`CuSnpArray` over the first `width` columns of `slot`, with `stream`'s
model and flags; its `μ` and `σinv` are placeholders, as the kernels read
only the lookup values.
"""
function _chunk_array(slot::_StreamSlot{T}, stream::SnpLinAlgStream{T},
    width::Int) where {T}
    statistics = view(slot.statistics, 1:width)
    return CuSnpArray{T}(view(slot.words, :, 1:width), stream.m,
        stream.model, stream.center, stream.scale, stream.impute, statistics,
        statistics, view(slot.values, :, 1:width))
end

"""
    streamed_mul!(out::CuMatrix{T}, stream::SnpLinAlgStream{T},
                  X::CuMatrix{T}; transpose=false, slots=3) -> out

`out = A * X` (or `transpose(A) * X`) on the GPU in one pass over
`stream`. Three stages overlap: the stream's prefetch task reads the next
chunk from disk, a producer task copies the current one from pinned host
memory into one of `slots` device buffers on its own CUDA stream, and the
calling task runs the products on earlier chunks.
"""
function streamed_mul!(
    out::CuMatrix{T}, stream::SnpLinAlgStream{T}, X::CuMatrix{T};
    transpose::Bool=false, slots::Integer=3,
) where {T <: Union{Float32, Float64}}
    m, n = size(stream)
    k = size(X, 2)
    rows, inner = transpose ? (n, m) : (m, n)
    size(X, 1) == inner || throw(DimensionMismatch(
        "right-hand side has $(size(X, 1)) rows; expected $inner"))
    size(out) == (rows, k) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((rows, k))"))
    slots >= 2 || throw(ArgumentError("slots must be at least 2, got $slots"))
    for buffer in stream.full_buffers
        pin(buffer.s.data)
        pin(buffer.values)
    end
    word_rows = cld(m, 16)
    free = Channel{Tuple{_StreamSlot{T}, Union{Nothing, CuEvent}}}(slots)
    for _ in 1:slots
        put!(free, (_StreamSlot{T}(word_rows, stream.width), nothing))
    end
    ready = Channel{Union{Nothing, Tuple{UnitRange{Int}, _StreamSlot{T}}}}(
        slots)
    ctx = context()
    producer = Threads.@spawn begin
        context!(ctx)
        try
            for (cols, chunk) in stream
                slot, event = take!(free)
                event === nothing || synchronize(event)
                _upload_chunk!(slot, chunk, m, length(cols))
                synchronize()
                put!(ready, (cols, slot))
            end
        finally
            put!(ready, nothing)
        end
    end
    started = false
    partial = transpose || length(stream) <= 1 ? nothing : similar(out)
    try
        while (item = take!(ready)) !== nothing
            cols, slot = item
            s = _chunk_array(slot, stream, length(cols))
            if transpose
                piece = CuMatrix{T}(undef, length(cols), k)
                mul!(piece, Base.transpose(s), X)
                view(out, cols, :) .= piece
                unsafe_free!(piece)
            else
                rhs = X[cols, :]
                if started
                    mul!(partial, s, rhs)
                    out .+= partial
                else
                    mul!(out, s, rhs)
                end
                unsafe_free!(rhs)
            end
            started = true
            event = CuEvent()
            record(event)
            put!(free, (slot, event))
        end
    catch
        close(free)
        close(ready)
        rethrow()
    end
    wait(producer)
    partial === nothing || unsafe_free!(partial)
    started || fill!(out, zero(T))
    return out
end
