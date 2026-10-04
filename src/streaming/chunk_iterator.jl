"""
    SnpLinAlgStreamState

Iteration state of a `SnpLinAlgStream`: the read position in the files and
the chunk scheduled for the next `iterate` call.
"""
mutable struct SnpLinAlgStreamState
    file_index::Int
    # Columns of the current file already scheduled.
    file_col::Int
    io::IO
    # Buffer of the pair, 1 or 2, that the next scheduled chunk fills.
    which::Int
    # The scheduled chunk: its columns, buffer, width, and prefetch task.
    next_cols::UnitRange{Int}
    next_which::Int
    next_width::Int
    next_task::Union{Nothing, Task}
    has_next::Bool
    # Reader handles of the current file; empty selects the serial read.
    handles::Vector{IOStream}
    # File byte where the scheduled chunk starts.
    byte_offset::Int
end

"""
    _next_chunk_location!(stream, state)

Advance `state` past one chunk, opening the next file at a file boundary,
and return the chunk's `cols`, `chunk_width`, `io`, and `byte_offset`, or
`nothing` when every file is exhausted.
"""
function _next_chunk_location!(
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
        state.handles = _open_reader_handles(path, io, stream.readers)
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

"""
    _schedule_next_chunk!(stream, state) -> state

Record the next chunk in `state` and, with `stream.prefetch`, start reading
it.
"""
function _schedule_next_chunk!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState,
) where T
    desc = _next_chunk_location!(stream, state)
    if desc === nothing
        state.has_next = false
        return state
    end
    which = state.which
    state.which = which == 1 ? 2 : 1
    buffer = _chunk_buffer_pair(stream, desc.chunk_width)[which]
    io = desc.io
    handles = state.handles
    byte_offset = desc.byte_offset
    state.next_cols = desc.cols
    state.next_which = which
    state.next_width = desc.chunk_width
    state.byte_offset = byte_offset
    state.has_next = true
    if stream.prefetch
        state.next_task = Threads.@spawn _read_scheduled_chunk!(
            stream, handles, byte_offset, io, buffer,
        )
    else
        state.next_task = nothing
    end
    return state
end

"""
    _read_scheduled_chunk!(stream, handles, byte_offset, io, buffer)
        -> buffer

Read the scheduled chunk into `buffer`, in parallel when `handles` is
nonempty and serially from `io` otherwise.
"""
function _read_scheduled_chunk!(
    stream::SnpLinAlgStream{T},
    handles::Vector{IOStream},
    byte_offset::Int,
    io::IO,
    buffer::SnpLinAlg{T},
) where T
    isempty(handles) && return _read_chunk_serial!(io, buffer)
    drows = (stream.m + 3) >> 2
    return _read_chunk_parallel!(handles, buffer, byte_offset, drows)
end

"""
    _take_chunk!(stream, state)

Finish reading the scheduled chunk, schedule the following one, and return
`((cols, chunk), state)`, or `nothing` when no chunk remains.
"""
function _take_chunk!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState,
) where T
    state.has_next || return nothing
    cols = state.next_cols
    which = state.next_which
    width = state.next_width
    task = state.next_task
    buffer = _chunk_buffer_pair(stream, width)[which]
    # `state` describes this chunk until the next one is scheduled below.
    if task === nothing
        _read_scheduled_chunk!(stream, state.handles, state.byte_offset,
                               state.io, buffer)
    else
        wait(task)
    end
    _schedule_next_chunk!(stream, state)
    return (cols, buffer), state
end

function Base.iterate(stream::SnpLinAlgStream{T}) where T
    io = makestream(stream.files[1])
    _check_bed_magic!(io, stream.files[1])
    handles = _open_reader_handles(stream.files[1], io, stream.readers)
    state = SnpLinAlgStreamState(1, 0, io, 1, 1:0, 1, 0, nothing, false,
                                 handles, 0)
    _schedule_next_chunk!(stream, state)
    return _take_chunk!(stream, state)
end

Base.iterate(stream::SnpLinAlgStream, state::SnpLinAlgStreamState) =
    _take_chunk!(stream, state)
