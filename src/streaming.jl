"""
    SnpLinAlgStream{T}(bedfiles; m=nothing, width=4096, prefetch=true,
                       readers=_default_readers(), model=ADDITIVE_MODEL,
                       center=false, scale=false, impute=true)

Iterate `.bed` files as consecutive column chunks of a `SnpLinAlg`, without
memory-mapping the whole genotype matrix.

# Arguments
- `bedfiles`: a `.bed` path, or paths concatenated column-wise
- `m`: sample count; defaults to the first file's `.fam` line count
- `width`: chunk width in SNPs
- `prefetch`: overlaps the next chunk's read with the current chunk's use
- `readers`: concurrent reader tasks per chunk of a plain `.bed` file;
  compressed files always use one
- `model`, `center`, `scale`, `impute`: forwarded to each chunk's `SnpLinAlg`

`for (cols, chunk) in stream` yields `(cols, chunk)` with
`cols::UnitRange{Int}` the global column range and `chunk::SnpLinAlg{T}`
of size `(m, length(cols))`; chunks never straddle a file. `chunk` is a
reused buffer valid only until the next chunk is requested. A fresh `for`
restarts from the first column.
"""
struct SnpLinAlgStream{T}
    files::Vector{String}
    m::Int
    ns::Vector{Int}
    offsets::Vector{Int}
    width::Int
    prefetch::Bool
    readers::Int
    model::Union{Val{1}, Val{2}, Val{3}}
    center::Bool
    scale::Bool
    impute::Bool
    full_buffers::NTuple{2, SnpLinAlg{T}}
    short_buffers::Dict{Int, NTuple{2, SnpLinAlg{T}}}
end

"""
    _fam_row_count(bednm) -> Int

Count the samples of the `.bed` file `bednm` from its sibling `.fam` file's
line count.
"""
function _fam_row_count(bednm::AbstractString)
    famnm = replace(bednm, ".bed" => ".fam")
    isfile(famnm) || throw(ArgumentError("fam file not found: $famnm"))
    return makestream(famnm) do io
        countlines(io)
    end
end

"""
    _bim_column_count(bednm) -> Int

Count the SNPs of the `.bed` file `bednm` from its sibling `.bim` file's
line count.
"""
function _bim_column_count(bednm::AbstractString)
    bimnm = replace(bednm, ".bed" => ".bim")
    return makestream(bimnm) do io
        countlines(io)
    end
end

"""
    _make_buffer(T, m, width, model, center, scale, impute) -> SnpLinAlg{T}

Build one reusable, zeroed `m × width` chunk buffer.
"""
function _make_buffer(
    ::Type{T},
    m::Int,
    width::Int,
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    s = SnpArray(undef, m, width)
    fill!(s.data, 0x00)
    fill!(s.columncounts, 0)
    return SnpLinAlg{T}(s; model=model, center=center, scale=scale,
                        impute=impute)
end

"""
    _make_buffer_pair(T, m, width, model, center, scale, impute)
        -> NTuple{2, SnpLinAlg{T}}

Build the two alternating chunk buffers of a buffer pair.
"""
_make_buffer_pair(
    ::Type{T}, m::Int, width::Int, model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool, scale::Bool, impute::Bool,
) where T <: AbstractFloat =
    (_make_buffer(T, m, width, model, center, scale, impute),
     _make_buffer(T, m, width, model, center, scale, impute))

"""
    _default_readers() -> Int

Return the default number of concurrent reader tasks per plain `.bed`
chunk: half the thread count, between 1 and 8, so blocking reads do not
park the threads the prefetch task overlaps with.
"""
_default_readers() = max(1, min(8, Threads.nthreads() ÷ 2))

function SnpLinAlgStream{T}(
    bedfiles::Union{AbstractString, AbstractVector{<:AbstractString}};
    m::Union{Integer, Nothing} = nothing,
    width::Integer = 4096,
    prefetch::Bool = true,
    readers::Integer = _default_readers(),
    model::Union{Val{1}, Val{2}, Val{3}} = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = true,
) where T <: AbstractFloat
    files = bedfiles isa AbstractString ? [String(bedfiles)] :
            Vector{String}(bedfiles)
    isempty(files) && throw(ArgumentError("bedfiles must be nonempty"))
    for path in files
        checkplinkfilename(path, "bed")
    end
    model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL) ||
        throw(ArgumentError("unrecognized model $model"))
    width >= 1 || throw(ArgumentError("width must be at least 1, got $width"))
    readers >= 1 ||
        throw(ArgumentError("readers must be at least 1, got $readers"))
    row_count = m === nothing ? _fam_row_count(first(files)) : Int(m)
    drows = (row_count + 3) >> 2
    ns = Vector{Int}(undef, length(files))
    for (index, path) in enumerate(files)
        n = _bim_column_count(path)
        if endswith(path, ".bed")
            filesize(path) - 3 == drows * n || throw(ArgumentError(
                "filesize of $path is not consistent with $row_count " *
                "rows and $n columns",
            ))
        end
        ns[index] = n
    end
    offsets = Vector{Int}(undef, length(files))
    total = 0
    for index in eachindex(ns)
        offsets[index] = total
        total += ns[index]
    end
    full_buffers = _make_buffer_pair(T, row_count, Int(width), model, center,
                                     scale, impute)
    return SnpLinAlgStream{T}(files, row_count, ns, offsets, Int(width),
                              prefetch, Int(readers), model, center, scale,
                              impute, full_buffers,
                              Dict{Int, NTuple{2, SnpLinAlg{T}}}())
end

Base.size(stream::SnpLinAlgStream) = (stream.m, sum(stream.ns))
Base.size(stream::SnpLinAlgStream, dimension::Integer) =
    size(stream)[dimension]
Base.length(stream::SnpLinAlgStream) =
    sum(cld(n, stream.width) for n in stream.ns)
Base.eltype(::Type{<:SnpLinAlgStream{T}}) where T =
    Tuple{UnitRange{Int}, SnpLinAlg{T}}

"""
    _buffer_pair(stream, width) -> NTuple{2, SnpLinAlg{T}}

Return `stream`'s buffer pair for chunk width `width`, creating and
caching one lazily the first time a non-default width is seen.
"""
function _buffer_pair(stream::SnpLinAlgStream{T}, width::Int) where T
    width == stream.width && return stream.full_buffers
    return get!(stream.short_buffers, width) do
        _make_buffer_pair(T, stream.m, width, stream.model, stream.center,
                          stream.scale, stream.impute)
    end
end

"""
    _check_bed_magic!(io, path) -> io

Verify `io`'s three-byte `.bed` header, closing `io` and throwing an
`ArgumentError` naming `path` if it is wrong.
"""
function _check_bed_magic!(io::IO, path::AbstractString)
    if read(io, UInt16) != 0x1b6c
        close(io)
        throw(ArgumentError("wrong magic number in file $path"))
    end
    if read(io, UInt8) != 0x01
        close(io)
        throw(ArgumentError(".bed file, $path, is not in correct orientation"))
    end
    return io
end

"""
    _open_handles(path, io, readers) -> Vector{IOStream}

Return `io` plus `readers - 1` further read handles on `path` when
`readers > 1` and `io` is a seekable `IOStream`; otherwise an empty vector,
which selects the serial chunk read.
"""
function _open_handles(path::AbstractString, io::IO, readers::Int)
    (readers > 1 && io isa IOStream) || return IOStream[]
    handles = Vector{IOStream}(undef, readers)
    handles[1] = io
    for index in 2:readers
        handles[index] = open(path, "r")
    end
    return handles
end

"""
    _finish_chunk!(buffer) -> buffer

Zero `buffer`'s cached counts and refill its statistics from the freshly
read codes in `buffer.s.data`.
"""
function _finish_chunk!(buffer::SnpLinAlg{T}) where T <: AbstractFloat
    fill!(buffer.s.columncounts, 0)
    fill!(buffer.s.rowcounts, 0)
    _refill_statistics!(buffer)
    return buffer
end

"""
    _read_chunk!(io, buffer) -> buffer

Read one chunk of packed genotypes into `buffer.s.data` with a single
sequential read and refill `buffer`'s statistics.
"""
function _read_chunk!(io::IO, buffer::SnpLinAlg{T}) where T <: AbstractFloat
    read!(io, buffer.s.data)
    return _finish_chunk!(buffer)
end

"""
    _read_block!(io, data, file_offset, byte_first, nbytes) -> data

Seek `io` to `file_offset` and read `nbytes` bytes into `data` starting at
linear index `byte_first`; throws an `ArgumentError` naming the offset if
fewer than `nbytes` bytes remain.
"""
function _read_block!(
    io::IOStream,
    data::Matrix{UInt8},
    file_offset::Int,
    byte_first::Int,
    nbytes::Int,
)
    seek(io, file_offset)
    remaining = filesize(io) - file_offset
    remaining >= nbytes || throw(ArgumentError(
        "short read at byte offset $file_offset: $remaining bytes " *
        "remain, $nbytes needed",
    ))
    GC.@preserve data unsafe_read(io, pointer(data, byte_first), UInt(nbytes))
    return data
end

"""
    _read_chunk_parallel!(handles, buffer, byte_offset, drows) -> buffer

Read the chunk starting at file byte `byte_offset` into `buffer.s.data`
using one task per contiguous block of `DECODE_WIDTH`-aligned columns, one
block per handle, then refill `buffer`'s statistics.
"""
function _read_chunk_parallel!(
    handles::Vector{IOStream},
    buffer::SnpLinAlg{T},
    byte_offset::Int,
    drows::Int,
) where T <: AbstractFloat
    data = buffer.s.data
    width = size(data, 2)
    step = cld(cld(width, length(handles)), DECODE_WIDTH) * DECODE_WIDTH
    @sync for (index, first) in enumerate(1:step:width)
        last = min(first + step - 1, width)
        nbytes = drows * (last - first + 1)
        skip = drows * (first - 1)
        io = handles[index]
        Threads.@spawn _read_block!(io, data, byte_offset + skip,
                                    skip + 1, nbytes)
    end
    return _finish_chunk!(buffer)
end

# Mutable per-iteration state: which file and column within it is next to
# be handed out, the file's open handle (and, for a plain `.bed` file with
# multiple readers, the parallel-read handles), which buffer of a pair is
# next to be filled, the byte offset of the prepared chunk, and (with
# prefetch) the in-flight read/refill task for the chunk after the one
# about to be delivered.
mutable struct SnpLinAlgStreamState
    file_index::Int
    file_col::Int
    io::IO
    which::Int
    next_cols::UnitRange{Int}
    next_which::Int
    next_width::Int
    next_task::Union{Nothing, Task}
    has_next::Bool
    handles::Vector{IOStream}
    byte_offset::Int
end

# Advance the file/column bookkeeping past one chunk, opening the next
# file (and closing the previous one, and its extra reader handles) at a
# file boundary; returns the chunk's global column range, width, io, and
# starting byte offset, or `nothing` once every file is exhausted.
function _advance!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState,
) where T
    if state.file_col >= stream.ns[state.file_index]
        for h in state.handles
            h === state.io || close(h)
        end
        close(state.io)
        state.file_index += 1
        state.file_index > length(stream.files) && return nothing
        path = stream.files[state.file_index]
        io = makestream(path)
        _check_bed_magic!(io, path)
        state.io = io
        state.file_col = 0
        state.handles = _open_handles(path, io, stream.readers)
    end
    file_index = state.file_index
    chunk_width = min(stream.width, stream.ns[file_index] - state.file_col)
    column_first = stream.offsets[file_index] + state.file_col + 1
    column_last = stream.offsets[file_index] + state.file_col + chunk_width
    cols = column_first:column_last
    drows = (stream.m + 3) >> 2
    byte_offset = 3 + drows * state.file_col
    state.file_col += chunk_width
    return (cols=cols, chunk_width=chunk_width, io=state.io,
           byte_offset=byte_offset)
end

# Prepare (and, with prefetch, start reading) the chunk that will be
# delivered by the next `_deliver!` call.
function _kick_off!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState,
) where T
    desc = _advance!(stream, state)
    if desc === nothing
        state.has_next = false
        return state
    end
    which = state.which
    state.which = which == 1 ? 2 : 1
    buffer = _buffer_pair(stream, desc.chunk_width)[which]
    io = desc.io
    handles = state.handles
    byte_offset = desc.byte_offset
    state.next_cols = desc.cols
    state.next_which = which
    state.next_width = desc.chunk_width
    state.byte_offset = byte_offset
    state.has_next = true
    if stream.prefetch
        state.next_task = Threads.@spawn _read_prepared!(
            stream, handles, byte_offset, io, buffer,
        )
    else
        state.next_task = nothing
    end
    return state
end

# Read the prepared chunk into `buffer`, in parallel when `handles` is
# nonempty, else serially from `io`.
function _read_prepared!(
    stream::SnpLinAlgStream{T},
    handles::Vector{IOStream},
    byte_offset::Int,
    io::IO,
    buffer::SnpLinAlg{T},
) where T
    isempty(handles) && return _read_chunk!(io, buffer)
    drows = (stream.m + 3) >> 2
    return _read_chunk_parallel!(handles, buffer, byte_offset, drows)
end

# Wait for (or perform) the read of the prepared chunk, then kick off the
# following one before returning the prepared chunk to the caller. This
# runs before `_kick_off!` advances the state, so `state.handles` and
# `state.byte_offset` still describe the prepared chunk.
function _deliver!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState,
) where T
    state.has_next || return nothing
    cols = state.next_cols
    which = state.next_which
    width = state.next_width
    task = state.next_task
    buffer = _buffer_pair(stream, width)[which]
    if task === nothing
        _read_prepared!(stream, state.handles, state.byte_offset, state.io,
                        buffer)
    else
        wait(task)
    end
    _kick_off!(stream, state)
    return (cols, buffer), state
end

function Base.iterate(stream::SnpLinAlgStream{T}) where T
    io = makestream(stream.files[1])
    _check_bed_magic!(io, stream.files[1])
    handles = _open_handles(stream.files[1], io, stream.readers)
    state = SnpLinAlgStreamState(1, 0, io, 1, 1:0, 1, 0, nothing, false,
                                 handles, 0)
    _kick_off!(stream, state)
    return _deliver!(stream, state)
end

Base.iterate(stream::SnpLinAlgStream, state::SnpLinAlgStreamState) =
    _deliver!(stream, state)

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
    scale::Union{Number, AbstractVector},
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
    _streamed_grm_mul!(V, stream, Q_kernel, scale_kernel, U, V_scratch)

Run the per-chunk products in `Q_kernel`'s type and accumulate into the
already zeroed `V`, through `V_scratch` when given.
"""
function _streamed_grm_mul!(
    V::AbstractMatrix{TV},
    stream::SnpLinAlgStream{TS},
    Q_kernel::AbstractMatrix{TS},
    scale_kernel::Union{Number, AbstractVector{TS}},
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
        chunk_scale = scale_kernel isa AbstractVector ?
            view(scale_kernel, cols) : scale_kernel
        chunk_output .*= chunk_scale
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
    streamed_grm_mul!(V, stream, Q; scale=inv(size(stream, 2)), U=nothing)
    streamed_grm_mul!(V::AbstractMatrix{Float64},
                      stream::SnpLinAlgStream{Float32},
                      Q::AbstractMatrix{Float64};
                      scale=inv(size(stream, 2)), U=nothing)

Accumulate `V = Σ_c A_c diag(s_c) transpose(A_c) Q` over the chunks `A_c`
of `stream`, where `s_c` is `scale` restricted to chunk `c`'s columns and
`scale` may be a scalar or a length-`size(stream, 2)` vector. When `U` is
an `AbstractMatrix`, also write the unscaled `transpose(A) * Q` into it.
With a `Float32` stream and `Float64` `V` and `Q`, each chunk's product
runs in single precision and is accumulated into `V` in double precision.
"""
function streamed_grm_mul!(
    V::AbstractMatrix{T},
    stream::SnpLinAlgStream{T},
    Q::AbstractMatrix{T};
    scale::Union{Number, AbstractVector{T}} = inv(size(stream, 2)),
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
    scale::Union{Number, AbstractVector{<:Real}} = inv(size(stream, 2)),
    U::Union{Nothing, AbstractMatrix{Float64}} = nothing,
)
    m, n = size(stream)
    k = size(Q, 2)
    _check_streamed_grm_dims(V, Q, U, scale, m, n, k)
    fill!(V, zero(Float64))
    Q32 = Matrix{Float32}(Q)
    scale32 = scale isa AbstractVector ? Vector{Float32}(scale) :
              Float32(scale)
    V_scratch = Matrix{Float32}(undef, m, k)
    _streamed_grm_mul!(V, stream, Q32, scale32, U, V_scratch)
    return V
end
