"""
    SnpLinAlgStream{T}(bedfiles; m=nothing, width=4096, prefetch=true,
                       readers=_default_reader_count(), model=ADDITIVE_MODEL,
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

Return the default number of concurrent reader tasks per plain `.bed`
chunk: half the thread count, between 1 and 8, so blocking reads do not
park the threads the prefetch task overlaps with.
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
    full_buffers = _make_chunk_buffer_pair(T, row_count, Int(width), model,
                                           center, scale, impute)
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
