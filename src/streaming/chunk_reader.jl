"""
    _check_bed_magic!(io, path) -> io

Close `io` and throw an `ArgumentError` naming `path` unless `io` starts
with the three-byte `.bed` header.
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
    _open_reader_handles(path, io, readers) -> Vector{IOStream}

Return `readers` read handles on `path`, the first being `io`, or an empty
vector when `readers == 1` or `io` is not an `IOStream`.
"""
function _open_reader_handles(path::AbstractString, io::IO, readers::Int)
    (readers > 1 && io isa IOStream) || return IOStream[]
    handles = Vector{IOStream}(undef, readers)
    handles[1] = io
    for index in 2:readers
        handles[index] = open(path, "r")
    end
    return handles
end

"""
    _refresh_chunk_statistics!(buffer) -> buffer

Recompute `buffer`'s counts, means, and genotype values from the codes in
`buffer.s.data`.
"""
function _refresh_chunk_statistics!(
    buffer::SnpLinAlg{T},
) where T <: AbstractFloat
    fill!(buffer.s.columncounts, 0)
    fill!(buffer.s.rowcounts, 0)
    _refill_statistics!(buffer)
    return buffer
end

"""
    _read_chunk_serial!(io, buffer) -> buffer

Read one chunk of packed genotypes from `io` into `buffer` and refresh its
statistics.
"""
function _read_chunk_serial!(
    io::IO,
    buffer::SnpLinAlg{T},
) where T <: AbstractFloat
    read!(io, buffer.s.data)
    return _refresh_chunk_statistics!(buffer)
end

"""
    _read_column_block!(io, data, file_offset, byte_first, nbytes) -> data

Read `nbytes` bytes at byte `file_offset` of `io` into `data` from linear
index `byte_first`, throwing an `ArgumentError` when fewer remain.
"""
function _read_column_block!(
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

Read the chunk at file byte `byte_offset` into `buffer`, one block of
columns per handle, and refresh its statistics. `drows` is the number of
bytes per SNP column.
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
        Threads.@spawn _read_column_block!(io, data, byte_offset + skip,
                                           skip + 1, nbytes)
    end
    return _refresh_chunk_statistics!(buffer)
end
