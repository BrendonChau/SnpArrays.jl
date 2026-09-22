"""
    SnpArray
    SnpArray(bednm, m)
    SnpArray(plknm)
    SnpArray(undef, m, n)

Raw .bed file as a shared, memory-mapped Matrix{UInt8}.  The number of rows, `m`
is stored separately because it is not uniquely determined by the size of the `data` field.
"""
struct SnpArray <: AbstractMatrix{UInt8}
    data::Matrix{UInt8}
    columncounts::Matrix{Int}
    rowcounts::Matrix{Int}
    m::Int
end

"""
    StackedSnpArray(s::Vector{SnpArray})

Stacked SnpArray for unified indexing
"""
struct StackedSnpArray <: AbstractMatrix{UInt8} # details in stackedsnparray.jl
    arrays::Vector{SnpArray}
    m::Int
    n::Int
    ns::Vector{Int}
    offsets::Vector{Int} # 0-based
end

const AbstractSnpArray = Union{SnpArray, SubArray{UInt8, 1, SnpArray}, SubArray{UInt8, 2, SnpArray}, 
    StackedSnpArray, SubArray{UInt8, 1, StackedSnpArray}, SubArray{UInt8, 2, StackedSnpArray}}

function SnpArray(bednm::AbstractString, m::Integer, args...; kwargs...)
    checkplinkfilename(bednm, "bed")
    data = makestream(bednm, args...; kwargs...) do io
        read(io, UInt16) == 0x1b6c || throw(ArgumentError("wrong magic number in file $bednm"))
        read(io, UInt8) == 0x01 || throw(ArgumentError(".bed file, $bednm, is not in correct orientation"))
        if endswith(bednm, ".bed")
            return Mmap.mmap(io)
        else
            return read(io)
        end
    end
    drows = (m + 3) >> 2   # the number of rows in the Matrix{UInt8}
    n, r = divrem(length(data), drows)
    iszero(r) || throw(ArgumentError("filesize of $bednm is not a multiple of $drows"))
    SnpArray(reshape(data, (drows, n)), zeros(Int, (4, n)), zeros(Int, (4, m)), m)
end

function SnpArray(
    bednm::AbstractString, 
    args...; 
    famnm::Union{AbstractString, Nothing}=nothing, 
    kwargs...)
    checkplinkfilename(bednm, "bed")
    # check user supplied fam filename
    if famnm === nothing
        famnm = replace(bednm, ".bed" => ".fam") 
        isfile(famnm) || throw(ArgumentError("fam file not found"))
    else # user supplied fam file name
        checkplinkfilename(famnm, "fam")
        isfile(famnm) || throw(ArgumentError("file $famnm not found"))
    end
    m = makestream(famnm) do stream
        countlines(stream)
    end
    SnpArray(bednm, m, args...; kwargs...)
end

function SnpArray(::UndefInitializer, m::Integer, n::Integer)
    SnpArray(Matrix{UInt8}(undef, ((m + 3) >> 2, n)), zeros(Int, (4, n)), zeros(Int, (4, m)), m)
end

function SnpArray(file::AbstractString, s::SnpArray)
    makestream(file, "w+") do io
        write(io, 0x1b6c)
        write(io, 0x01)
        write(io, s.data)
    end
    SnpArray(file, s.m, "r+")
end

function SnpArray(file::AbstractString, m::Integer, n::Integer)
    makestream(file, "w+") do io
        write(io, 0x1b6c)
        write(io, 0x01)
        write(io, fill(0x00, ((m + 3) >> 2, n)))
    end
    SnpArray(file, m, "r+")
end

StatsBase.counts(s::AbstractSnpArray; dims=:) = _counts(s, dims)

@inline function _packed_counts(byte::UInt8)
    low = byte & 0x55
    high = (byte >> 1) & 0x55
    count1 = count_ones(low & ~high)
    count2 = count_ones(~low & high)
    count3 = count_ones(low & high)
    return 4 - count1 - count2 - count3, count1, count2, count3
end

# mask selecting the low bit of each 2-bit genotype code within a 64-bit word
const _WORD_LOW_MASK = 0x5555555555555555

"""
    COUNT_TASK_MIN_BYTES

Packed-byte count of `s.data` below which `_column_counts!` runs serially;
smaller arrays do not repay the task spawn.
"""
const COUNT_TASK_MIN_BYTES = 1 << 20

"""
    _column_counts_range!(counts::Matrix{Int}, s::SnpArray,
        columns::UnitRange{Int})

Count the four packed genotype codes per column of `s` in `columns` into
`counts`, processing eight fully-valid bytes at a time as one 64-bit word.
"""
function _column_counts_range!(counts::Matrix{Int}, s::SnpArray,
    columns::UnitRange{Int})
    full_bytes, trailing_genotypes = divrem(s.m, 4)
    full_words = full_bytes >> 3
    @inbounds for column in columns
        count1 = count2 = count3 = 0
        for word_index in 1:full_words
            base = 8 * (word_index - 1)
            word = UInt64(s.data[base + 1, column]) |
                UInt64(s.data[base + 2, column]) << 8 |
                UInt64(s.data[base + 3, column]) << 16 |
                UInt64(s.data[base + 4, column]) << 24 |
                UInt64(s.data[base + 5, column]) << 32 |
                UInt64(s.data[base + 6, column]) << 40 |
                UInt64(s.data[base + 7, column]) << 48 |
                UInt64(s.data[base + 8, column]) << 56
            low = word & _WORD_LOW_MASK
            high = (word >> 1) & _WORD_LOW_MASK
            count1 += count_ones(low & ~high)
            count2 += count_ones(~low & high)
            count3 += count_ones(low & high)
        end
        for byte_index in (8full_words + 1):full_bytes
            byte_counts = _packed_counts(s.data[byte_index, column])
            count1 += byte_counts[2]
            count2 += byte_counts[3]
            count3 += byte_counts[4]
        end
        count0 = 4full_bytes - count1 - count2 - count3
        if !iszero(trailing_genotypes)
            byte = s.data[full_bytes + 1, column]
            for offset in 0:(trailing_genotypes - 1)
                genotype = (byte >> (2offset)) & 0x03
                genotype == 0 && (count0 += 1)
                genotype == 1 && (count1 += 1)
                genotype == 2 && (count2 += 1)
                genotype == 3 && (count3 += 1)
            end
        end
        counts[1, column] = count0
        counts[2, column] = count1
        counts[3, column] = count2
        counts[4, column] = count3
    end
    return counts
end

"""
    _column_counts!(counts::Matrix{Int}, s::SnpArray)

Count the four packed genotype codes per column of `s` into `counts`,
splitting the columns across tasks when `s.data` holds at least
`COUNT_TASK_MIN_BYTES` bytes and more than one thread is available;
results are identical to the serial pass.
"""
function _column_counts!(counts::Matrix{Int}, s::SnpArray)
    n = size(s.data, 2)
    if Threads.nthreads() == 1 || length(s.data) < COUNT_TASK_MIN_BYTES
        return _column_counts_range!(counts, s, 1:n)
    end
    step = _task_axis_step(n, n)
    @sync for first in 1:step:n
        last = min(first + step - 1, n)
        Threads.@spawn _column_counts_range!(counts, s, first:last)
    end
    return counts
end

function _row_counts!(counts::Matrix{Int}, s::SnpArray)
    full_bytes, trailing_genotypes = divrem(s.m, 4)
    @inbounds for column in axes(s.data, 2)
        for byte_index in 1:full_bytes
            byte = s.data[byte_index, column]
            row = 4byte_index - 3
            counts[(byte & 0x03) + 1, row] += 1
            counts[((byte >> 2) & 0x03) + 1, row + 1] += 1
            counts[((byte >> 4) & 0x03) + 1, row + 2] += 1
            counts[((byte >> 6) & 0x03) + 1, row + 3] += 1
        end
        if !iszero(trailing_genotypes)
            byte = s.data[full_bytes + 1, column]
            row = 4full_bytes + 1
            for offset in 0:(trailing_genotypes - 1)
                genotype = (byte >> (2offset)) & 0x03
                counts[genotype + 1, row + offset] += 1
            end
        end
    end
    return counts
end

function _counts(s::SnpArray, dims::Integer)
    if isone(dims)
        all(iszero, s.columncounts) && _column_counts!(s.columncounts, s)
        return s.columncounts
    elseif dims == 2
        all(iszero, s.rowcounts) && _row_counts!(s.rowcounts, s)
        return s.rowcounts
    else
        throw(ArgumentError("counts(s::SnpArray, dims=k) only defined for " *
                            "k = 1 or 2"))
    end
end

function _counts(s::AbstractSnpArray, dims::Integer)
    if isone(dims)
        result = zeros(Int, (4, size(s, 2)))
        @inbounds for column in axes(s, 2)
            for row in axes(s, 1)
                result[s[row, column] + 1, column] += 1
            end
        end
        return result
    elseif dims == 2
        result = zeros(Int, (4, size(s, 1)))
        @inbounds for column in axes(s, 2)
            for row in axes(s, 1)
                result[s[row, column] + 1, row] += 1
            end
        end
        return result
    else
        throw(ArgumentError("counts(s::SnpArray, dims=k) only defined for " *
                            "k = 1 or 2"))
    end
end

_counts(s::AbstractSnpArray, ::Colon) = sum(_counts(s, 1), dims=2)

function Base.getindex(s::SnpArray, i::Int)  # Linear indexing
    d, r = divrem(i - 1, s.m)
    s[r + 1, d + 1]
end

@inline function Base.getindex(s::SnpArray, i::Integer, j::Integer)
    @boundscheck checkbounds(s, i, j)
    ip3 = i + 3
    (s.data[ip3 >> 2, j] >> ((ip3 & 0x03) << 1)) & 0x03
end

function Base.setindex!(s::SnpArray, x::UInt8, i::Int)  # Linear indexing
    d, r = divrem(i - 1, s.m)
    Base.setindex!(s, x, r + 1, d + 1)
end

@inline function _has_cached_counts(s::SnpArray, row::Integer, column::Integer)
    column_cached = !iszero(
        s.columncounts[1, column] |
        s.columncounts[2, column] |
        s.columncounts[3, column] |
        s.columncounts[4, column],
    )
    row_cached = !iszero(
        s.rowcounts[1, row] |
        s.rowcounts[2, row] |
        s.rowcounts[3, row] |
        s.rowcounts[4, row],
    )
    return column_cached || row_cached
end

function _invalidate_counts!(s::SnpArray)
    @warn "Mutating this SnpArray invalidated its cached summary statistics"
    fill!(s.columncounts, 0)
    fill!(s.rowcounts, 0)
    return s
end

@inline function Base.setindex!(s::SnpArray, x::UInt8, i::Integer, j::Integer)
    @boundscheck checkbounds(s, i, j)
    ip3 = i + 3
    shft = (ip3 & 0x03) << 1
    byte_index = ip3 >> 2
    byte = s.data[byte_index, j]
    old_value = (byte >> shft) & 0x03
    x == old_value && return x
    _has_cached_counts(s, i, j) && _invalidate_counts!(s)
    mask = ~(0x03 << shft)
    s.data[byte_index, j] = (byte & mask) | (x << shft)
    return x
end

Base.eltype(s::SnpArray) = UInt8

Base.length(s::SnpArray) = s.m * size(s.data, 2)

Base.size(s::SnpArray) = s.m, size(s.data, 2)

Base.size(s::SnpArray, k::Integer) = 
k == 1 ? s.m : k == 2 ? size(s.data, 2) : k > 2 ? 1 : error("Dimension k out of range")
