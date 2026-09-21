"""
    SnpLinAlgStream{T}(bedfiles; m=nothing, width=4096, prefetch=true,
                       model=ADDITIVE_MODEL, center=false, scale=false,
                       impute=true)

Iterate `.bed` files as consecutive column chunks of a `SnpLinAlg`, without
memory-mapping the whole genotype matrix.

# Arguments
- `bedfiles`: a `.bed` path, or a vector of `.bed` paths concatenated
  column-wise (each file's row count must match)
- `m`: sample count; defaults to the line count of the first file's `.fam`
- `width`: chunk width in SNPs
- `prefetch`: overlap the next chunk's `.bed` read with the caller's use of
  the current chunk
- `model`, `center`, `scale`, `impute`: forwarded to each chunk's
  `SnpLinAlg`

# Iteration
`for (cols, chunk) in stream` yields, in column order and never straddling
a file boundary, a global column range `cols::UnitRange{Int}` and a
`chunk::SnpLinAlg{T}` of size `(m, length(cols))`. `chunk` wraps one of two
reused buffers per chunk width, so it is only valid until the next chunk is
requested: the caller must finish with `chunk` before advancing the `for`
loop. Iterating a stream with no state (`iterate(stream)`) reopens every
file from the start, so the same stream can be swept repeatedly.

# Prefetch
With `prefetch=true`, while the caller works on the chunk just returned, a
background task both `read!`s and refills the statistics of the following
chunk into the other buffer of the pair; the next call to `iterate` waits
on that task. This is only correct because the caller is done with the
current chunk before requesting the next one, which a `for` loop
guarantees.
"""
struct SnpLinAlgStream{T}
    files::Vector{String}
    m::Int
    ns::Vector{Int}
    offsets::Vector{Int}
    width::Int
    prefetch::Bool
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

Build one reusable `m × width` chunk buffer, zeroed and ready for a first
`_read_chunk!`.
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

Build the two alternating chunk buffers of one buffer pair.
"""
_make_buffer_pair(
    ::Type{T}, m::Int, width::Int, model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool, scale::Bool, impute::Bool,
) where T <: AbstractFloat =
    (_make_buffer(T, m, width, model, center, scale, impute),
     _make_buffer(T, m, width, model, center, scale, impute))

function SnpLinAlgStream{T}(
    bedfiles::Union{AbstractString, AbstractVector{<:AbstractString}};
    m::Union{Integer, Nothing} = nothing,
    width::Integer = 4096,
    prefetch::Bool = true,
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
                              prefetch, model, center, scale, impute,
                              full_buffers,
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

Return `stream`'s buffer pair for chunk width `width`, creating and caching
one lazily the first time a non-default width (a final short chunk) is
seen.
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
`ArgumentError` naming `path` if the magic number or orientation byte is
wrong.
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
    _read_chunk!(io, buffer) -> buffer

Read one chunk of packed genotypes into `buffer.s.data` and refill
`buffer`'s means, inverse standard deviations, and genotype values from
the freshly read codes.
"""
function _read_chunk!(io::IO, buffer::SnpLinAlg{T}) where T <: AbstractFloat
    read!(io, buffer.s.data)
    fill!(buffer.s.columncounts, 0)
    fill!(buffer.s.rowcounts, 0)
    _refill_statistics!(buffer)
    return buffer
end

# Mutable per-iteration state: which file and column within it is next to
# be handed out, the file's open handle, which buffer of a pair is next to
# be filled, and (with prefetch) the in-flight read/refill task for the
# chunk after the one about to be delivered.
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
end

# Advance the file/column bookkeeping past one chunk, opening the next
# file (and closing the previous one) at a file boundary; returns the
# chunk's global column range, width, and io, or `nothing` once every
# file is exhausted.
function _advance!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState,
) where T
    if state.file_col >= stream.ns[state.file_index]
        close(state.io)
        state.file_index += 1
        state.file_index > length(stream.files) && return nothing
        io = makestream(stream.files[state.file_index])
        _check_bed_magic!(io, stream.files[state.file_index])
        state.io = io
        state.file_col = 0
    end
    file_index = state.file_index
    chunk_width = min(stream.width, stream.ns[file_index] - state.file_col)
    column_first = stream.offsets[file_index] + state.file_col + 1
    column_last = stream.offsets[file_index] + state.file_col + chunk_width
    cols = column_first:column_last
    state.file_col += chunk_width
    return (cols=cols, chunk_width=chunk_width, io=state.io)
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
    state.next_cols = desc.cols
    state.next_which = which
    state.next_width = desc.chunk_width
    state.has_next = true
    if stream.prefetch
        state.next_task = Threads.@spawn _read_chunk!(io, buffer)
    else
        state.next_task = nothing
    end
    return state
end

# Wait for (or perform) the read of the prepared chunk, then kick off the
# following one before returning the prepared chunk to the caller.
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
    task === nothing ? _read_chunk!(state.io, buffer) : wait(task)
    _kick_off!(stream, state)
    return (cols, buffer), state
end

function Base.iterate(stream::SnpLinAlgStream{T}) where T
    io = makestream(stream.files[1])
    _check_bed_magic!(io, stream.files[1])
    state = SnpLinAlgStreamState(1, 0, io, 1, 1:0, 1, 0, nothing, false)
    _kick_off!(stream, state)
    return _deliver!(stream, state)
end

Base.iterate(stream::SnpLinAlgStream, state::SnpLinAlgStreamState) =
    _deliver!(stream, state)

"""
    streamed_grm_mul!(V, stream, Q; scale=inv(size(stream, 2)), U=nothing)

Accumulate `V = Σ_c A_c diag(s_c) transpose(A_c) Q` over the chunks `A_c`
of `stream`, where `s_c` is `scale` restricted to chunk `c`'s columns
(`scale` may be a scalar or a length-`size(stream, 2)` vector). When `U` is
an `AbstractMatrix`, also write the unscaled `transpose(A) * Q` into it.
Matrix `Q` and `V` only; reshape a vector right-hand side yourself.
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
    fill!(V, zero(T))
    buffers = Dict{Int, Matrix{T}}()
    for (cols, chunk) in stream
        width = length(cols)
        chunk_output = get!(buffers, width) do
            Matrix{T}(undef, width, k)
        end
        mul!(chunk_output, transpose(chunk), Q)
        U !== nothing && copyto!(view(U, cols, :), chunk_output)
        chunk_scale = scale isa AbstractVector ? view(scale, cols) : scale
        chunk_output .*= chunk_scale
        mul!(V, chunk, chunk_output, 1, 1)
    end
    return V
end
