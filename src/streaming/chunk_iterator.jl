"""
    _close_files!(state) -> state

Close every file handle `state` holds on its current file.
"""
function _close_files!(state::SnpLinAlgStreamState)
    # `handles[1]` is `state.io`, which is closed once, below.
    for handle in state.handles
        handle === state.io || close(handle)
    end
    close(state.io)
    return state
end

"""
    _finish_active_sweep!(stream) -> stream

Wait for the read an abandoned sweep of `stream` left running and close its
files.
"""
function _finish_active_sweep!(stream::SnpLinAlgStream)
    state = stream.active[]
    state === nothing && return stream
    # Cleared first, so a failed leftover read throws once, not every sweep.
    stream.active[] = nothing
    scheduled = state.scheduled
    task = scheduled === nothing ? nothing : scheduled.task
    task === nothing || wait(task)
    _close_files!(state)
    return stream
end

"""
    _next_chunk_location!(stream, state)

Advance `state` past one chunk, opening the next file at a file boundary,
and return the chunk's `(cols, byte_offset)`, or `nothing` when every file
is exhausted.
"""
function _next_chunk_location!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState{T},
) where T
    if state.file_col >= stream.ns[state.file_index]
        _close_files!(state)
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
    cols = column_first:(column_first + chunk_width - 1)
    byte_offset = 3 + _bytes_per_column(stream.m) * state.file_col
    state.file_col += chunk_width
    return cols, byte_offset
end

"""
    _schedule_next_chunk!(stream, state) -> state

Record the next chunk in `state.scheduled` and, with `stream.prefetch`,
start reading it; at exhaustion record `nothing` and end the sweep.
"""
function _schedule_next_chunk!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState{T},
) where T
    location = _next_chunk_location!(stream, state)
    if location === nothing
        state.scheduled = nothing
        stream.active[] = nothing
        return state
    end
    cols, byte_offset = location
    buffer = _chunk_buffer_pair(stream, length(cols))[state.buffer_index]
    state.buffer_index = state.buffer_index == 1 ? 2 : 1
    io = state.io
    handles = state.handles
    if stream.prefetch
        task = Threads.@spawn _read_scheduled_chunk!(
            stream, handles, byte_offset, io, buffer,
        )
        state.scheduled = ScheduledChunk{T}(cols, buffer, byte_offset, task)
    else
        state.scheduled = ScheduledChunk{T}(cols, buffer, byte_offset,
                                            nothing)
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
    drows = _bytes_per_column(stream.m)
    return _read_chunk_parallel!(handles, buffer, byte_offset, drows)
end

"""
    _take_chunk!(stream, state)

Finish reading the scheduled chunk, schedule the following one, and return
`((cols, chunk), state)`, or `nothing` when no chunk remains.
"""
function _take_chunk!(
    stream::SnpLinAlgStream{T},
    state::SnpLinAlgStreamState{T},
) where T
    scheduled = state.scheduled
    scheduled === nothing && return nothing
    task = scheduled.task
    # `state.io` and `state.handles` belong to this chunk's file until the
    # next chunk is scheduled below.
    if task === nothing
        _read_scheduled_chunk!(stream, state.handles, scheduled.byte_offset,
                               state.io, scheduled.buffer)
    else
        wait(task)
    end
    _schedule_next_chunk!(stream, state)
    return (scheduled.cols, scheduled.buffer), state
end

function Base.iterate(stream::SnpLinAlgStream{T}) where T
    _finish_active_sweep!(stream)
    io = makestream(stream.files[1])
    _check_bed_magic!(io, stream.files[1])
    handles = _open_reader_handles(stream.files[1], io, stream.readers)
    state = SnpLinAlgStreamState{T}(1, 0, io, handles, 1, nothing)
    stream.active[] = state
    _schedule_next_chunk!(stream, state)
    return _take_chunk!(stream, state)
end

Base.iterate(
    stream::SnpLinAlgStream{T}, state::SnpLinAlgStreamState{T},
) where T = _take_chunk!(stream, state)
