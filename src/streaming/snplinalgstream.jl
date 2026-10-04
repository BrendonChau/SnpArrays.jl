"""
    ScheduledChunk{T}

The chunk the next `iterate` call returns: its columns, its buffer, the file
byte where it starts, and its prefetch task, `nothing` for a read on demand.
"""
struct ScheduledChunk{T}
    cols::UnitRange{Int}
    buffer::SnpLinAlg{T}
    byte_offset::Int
    task::Union{Nothing, Task}
end

"""
    SnpLinAlgStreamState{T}

Iteration state of a `SnpLinAlgStream{T}`: the read position in the files
and the chunk scheduled for the next `iterate` call.
"""
mutable struct SnpLinAlgStreamState{T}
    file_index::Int
    # Columns of the current file already scheduled.
    file_col::Int
    # Abstract: plain and compressed files give different stream types.
    io::IO
    # Reader handles of the current file, the first being `io`; empty
    # selects the serial read.
    handles::Vector{IOStream}
    # Buffer of the pair, 1 or 2, that the next scheduled chunk fills.
    buffer_index::Int
    scheduled::Union{Nothing, ScheduledChunk{T}}
end

"""
    SnpLinAlgStream{T}(bedfiles; kwargs...)

Iterate `.bed` files as consecutive column chunks, each a `SnpLinAlg{T}`,
without holding the whole genotype matrix in memory.

Iteration yields `(cols, chunk)`: `cols::UnitRange{Int}` is the chunk's
column range in the concatenated files, and `chunk` is an
`m × length(cols)` `SnpLinAlg{T}`. A chunk never spans two files. `chunk`
is a reused buffer, valid only until the next chunk is requested. Only one
iteration of a stream may be in progress at a time.

# Arguments
- `bedfiles`: a `.bed` path, or a vector of paths concatenated column-wise

# Keywords
- `m = nothing`: sample count; `nothing` reads it from the first file's
  `.fam` line count
- `width = 4096`: chunk width in SNPs
- `prefetch = true`: read the next chunk while the current one is in use
- `readers = _default_reader_count()`: reader tasks per chunk of a plain
  `.bed` file; a compressed file always uses one
- `model`, `center`, `scale`, `impute`: as in `SnpLinAlg`

# Examples
```julia
stream = SnpLinAlgStream{Float64}("genotypes.bed"; width = 1024)
for (cols, chunk) in stream
    mul!(view(out, cols, :), transpose(chunk), X)
end
```
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
    # State of the sweep in progress; `nothing` between sweeps.
    active::Base.RefValue{Union{Nothing, SnpLinAlgStreamState{T}}}
end

"""
    _bytes_per_column(m) -> Int

Return the number of bytes one SNP column of `m` samples takes in a `.bed`
file.
"""
_bytes_per_column(m::Int) = (m + 3) >> 2

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
    _make_chunk_buffer(T, m, width, model, center, scale, impute)
        -> SnpLinAlg{T}

Build one reusable, zeroed `m × width` chunk buffer.
"""
function _make_chunk_buffer(
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
    _make_chunk_buffer_pair(T, m, width, model, center, scale, impute)
        -> NTuple{2, SnpLinAlg{T}}

Build the two alternating chunk buffers of a buffer pair.
"""
_make_chunk_buffer_pair(
    ::Type{T}, m::Int, width::Int, model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool, scale::Bool, impute::Bool,
) where T <: AbstractFloat =
    (_make_chunk_buffer(T, m, width, model, center, scale, impute),
     _make_chunk_buffer(T, m, width, model, center, scale, impute))

"""
    _default_reader_count() -> Int

Return the default number of reader tasks per plain `.bed` chunk: half the
thread count, clamped to 1 through 8.
"""
_default_reader_count() = max(1, min(8, Threads.nthreads() ÷ 2))

function SnpLinAlgStream{T}(
    bedfiles::Union{AbstractString, AbstractVector{<:AbstractString}};
    m::Union{Integer, Nothing} = nothing,
    width::Integer = 4096,
    prefetch::Bool = true,
    readers::Integer = _default_reader_count(),
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
    drows = _bytes_per_column(row_count)
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
    full_buffers = _make_chunk_buffer_pair(T, row_count, Int(width), model,
                                           center, scale, impute)
    active = Ref{Union{Nothing, SnpLinAlgStreamState{T}}}(nothing)
    return SnpLinAlgStream{T}(files, row_count, ns, offsets, Int(width),
                              prefetch, Int(readers), model, center, scale,
                              impute, full_buffers,
                              Dict{Int, NTuple{2, SnpLinAlg{T}}}(), active)
end

Base.size(stream::SnpLinAlgStream) = (stream.m, sum(stream.ns))
Base.size(stream::SnpLinAlgStream, dimension::Integer) =
    size(stream)[dimension]
Base.length(stream::SnpLinAlgStream) =
    sum(cld(n, stream.width) for n in stream.ns)
Base.eltype(::Type{<:SnpLinAlgStream{T}}) where T =
    Tuple{UnitRange{Int}, SnpLinAlg{T}}

"""
    _chunk_buffer_pair(stream, width) -> NTuple{2, SnpLinAlg{T}}

Return `stream`'s buffer pair for chunk width `width`, creating and
caching one lazily the first time a non-default width is seen.
"""
function _chunk_buffer_pair(stream::SnpLinAlgStream{T}, width::Int) where T
    width == stream.width && return stream.full_buffers
    return get!(stream.short_buffers, width) do
        _make_chunk_buffer_pair(T, stream.m, width, stream.model,
                                stream.center, stream.scale, stream.impute)
    end
end
