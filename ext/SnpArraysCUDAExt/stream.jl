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
    _foreach_chunk(f, stream::SnpLinAlgStream{T}, slots) -> nothing

Call `f(cols, s)` on the calling task for each chunk of `stream`, in
order, with `s` a `CuSnpArray` over the chunk's device copy. Three stages
overlap: the stream's prefetch task reads the next chunk from disk, a
producer task copies the current one from pinned host memory into one of
`slots` device buffers on its own CUDA stream, and `f` runs on earlier
chunks; a buffer is reused only after the work `f` queued on it is done.
"""
function _foreach_chunk(f::F, stream::SnpLinAlgStream{T},
    slots::Integer) where {F, T}
    slots >= 2 || throw(ArgumentError("slots must be at least 2, got $slots"))
    m = size(stream, 1)
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
    try
        while (item = take!(ready)) !== nothing
            cols, slot = item
            f(cols, _chunk_array(slot, stream, length(cols)))
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
    return nothing
end

"""
    streamed_mul!(out::CuMatrix{T}, stream::SnpLinAlgStream{T},
                  X::CuMatrix{T}; transpose=false, slots=3) -> out

`out = A * X` (or `transpose(A) * X`) on the GPU in one pass over
`stream`, with the chunks' reads, copies and products overlapped as in
`_foreach_chunk`.
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
    fill!(out, zero(T))
    partial = transpose ? nothing : similar(out)
    _foreach_chunk(stream, slots) do cols, s
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
    _grm_chunks!(V, stream, Q, scale, U, slots; V_scratch=nothing) -> V

Accumulate `V += A_c diag(s_c) transpose(A_c) Q` over `stream`'s chunks
on the GPU, writing the unscaled `transpose(A_c) Q` into `U[cols, :]` when
`U` is given. With `V_scratch`, each chunk's `A_c U_c` lands there first
and is added into `V`, whose element type may be wider than `Q`'s.
"""
function _grm_chunks!(V::CuMatrix, stream::SnpLinAlgStream{T},
    Q::CuMatrix{T}, scale::Union{Number, CuVector{T}},
    U::Union{Nothing, CuMatrix}, slots::Integer,
    V_scratch::CuMatrix{T}) where {T}
    k = size(Q, 2)
    fill!(V, zero(eltype(V)))
    _foreach_chunk(stream, slots) do cols, s
        piece = CuMatrix{T}(undef, length(cols), k)
        mul!(piece, transpose(s), Q)
        U === nothing || (view(U, cols, :) .= piece)
        piece .*= scale isa Number ? T(scale) : view(scale, cols)
        mul!(V_scratch, s, piece)
        V .+= V_scratch
        unsafe_free!(piece)
    end
    return V
end

"""
    streamed_grm_mul!(V::CuMatrix{T}, stream::SnpLinAlgStream{T},
                      Q::CuMatrix{T}; scale=inv(size(stream, 2)),
                      U=nothing, slots=3) -> V
    streamed_grm_mul!(V::CuMatrix{Float64}, stream::SnpLinAlgStream{Float32},
                      Q::CuMatrix{Float64}; scale, U, slots) -> V

GPU `V = Σ_c A_c diag(s_c) transpose(A_c) Q` in one pass over `stream`:
each chunk is read once and used for both of its products while on the
device, with reads, copies and products overlapped as in
`_foreach_chunk`. `scale` is a scalar or a length-`size(stream, 2)`
vector; `U`, when given, receives the unscaled `transpose(A) * Q`. With a
`Float32` stream and `Float64` `V` and `Q`, the products run in single
precision and accumulate into `V` in double.
"""
function streamed_grm_mul!(
    V::CuMatrix{T}, stream::SnpLinAlgStream{T}, Q::CuMatrix{T};
    scale::Union{Number, AbstractVector{<:Real}}=inv(size(stream, 2)),
    U::Union{Nothing, CuMatrix{T}}=nothing, slots::Integer=3,
) where {T <: Union{Float32, Float64}}
    m, n = size(stream)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, size(Q, 2))
    scale_device = scale isa Number ? scale : CuVector{T}(scale)
    V_scratch = similar(V)
    _grm_chunks!(V, stream, Q, scale_device, U, slots, V_scratch)
    unsafe_free!(V_scratch)
    return V
end

function streamed_grm_mul!(
    V::CuMatrix{Float64}, stream::SnpLinAlgStream{Float32},
    Q::CuMatrix{Float64};
    scale::Union{Number, AbstractVector{<:Real}}=inv(size(stream, 2)),
    U::Union{Nothing, CuMatrix{Float64}}=nothing, slots::Integer=3,
)
    m, n = size(stream)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, size(Q, 2))
    scale_device = scale isa Number ? scale : CuVector{Float32}(scale)
    Q32 = CuMatrix{Float32}(Q)
    V_scratch = CuMatrix{Float32}(undef, size(V))
    _grm_chunks!(V, stream, Q32, scale_device, U, slots, V_scratch)
    unsafe_free!(Q32)
    unsafe_free!(V_scratch)
    return V
end
