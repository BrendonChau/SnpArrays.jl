"""Threads per block for the `A*x` and chunk-sum kernels."""
const THREADS_PER_BLOCK = 256

"""Warps per block for the `transpose(A)*x` kernel."""
const WARPS_PER_BLOCK = 8

"""SNP columns per warp in the `transpose(A)*x` kernel."""
const COLUMNS_PER_WARP = 4

"""
    _genotype_lookup(values, column)

Load the four lookup values of SNP `column` into registers.
"""
@inline function _genotype_lookup(values, column)
    v0 = @inbounds values[1, column]
    v1 = @inbounds values[2, column]
    v2 = @inbounds values[3, column]
    v3 = @inbounds values[4, column]
    return (v0, v1, v2, v3)
end

"""
    _genotype_value(lookup, code)

Select the lookup value of a 2-bit genotype `code` without branching.
"""
@inline function _genotype_value(lookup, code)
    low = ifelse((code & UInt8(0x01)) == UInt8(0), lookup[1], lookup[2])
    high = ifelse((code & UInt8(0x01)) == UInt8(0), lookup[3], lookup[4])
    return ifelse((code & UInt8(0x02)) == UInt8(0), low, high)
end

"""
    _decode_word(lookup, word::UInt32)

Decode the 16 samples of a packed word into their lookup values.
"""
@inline function _decode_word(lookup, word::UInt32)
    return ntuple(Val(16)) do s
        shift = Int32(2) * (Int32(s) - Int32(1))
        code = UInt8((word >> shift) & UInt32(0x03))
        _genotype_value(lookup, code)
    end
end

"""
    _accumulate16(genotypes, xv, acc)

Return `acc .+ genotypes .* xv` for 16-tuples, unrolled.
"""
@generated function _accumulate16(
    genotypes::NTuple{16, T}, xv::T, acc::NTuple{16, T},
) where {T}
    terms = [:(acc[$i] + genotypes[$i] * xv) for i in 1:16]
    return quote
        Base.@_inline_meta
        $(Expr(:tuple, terms...))
    end
end

"""
    _ax_direct_kernel!(partials, words, x, values, m, n, chunk_columns)

`A*x` kernel: each thread owns one packed word row (16 samples) and reduces
over the SNP columns of chunk `blockIdx().y` into `partials[:, chunk]`.
"""
function _ax_direct_kernel!(
    partials::AbstractMatrix{T}, words::AbstractMatrix{UInt32},
    x::AbstractVector{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
    chunk_columns::Int32,
) where {T}
    total_word_rows = cld(m, Int32(16))
    word_row = (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x) +
               Int32(threadIdx().x)
    if word_row > total_word_rows
        return
    end
    chunk = Int32(blockIdx().y)
    column_first = (chunk - Int32(1)) * chunk_columns + Int32(1)
    column_last = min(chunk * chunk_columns, n)

    acc = ntuple(_ -> zero(T), Val(16))
    column = column_first
    while column <= column_last
        word = @inbounds words[word_row, column]
        lookup = _genotype_lookup(values, column)
        xv = @inbounds x[column]
        acc = _accumulate16(_decode_word(lookup, word), xv, acc)
        column += Int32(1)
    end

    # Constant tuple indices keep `acc` in registers.
    base = Int32(16) * (word_row - Int32(1))
    Base.Cartesian.@nexprs 16 s -> begin
        if base + Int32(s) <= m
            @inbounds partials[base + Int32(s), chunk] = acc[s]
        end
    end
    return
end

"""
    _sum_chunks_kernel!(out, partials, rows, chunks)

Sum each row of the `rows x chunks` matrix `partials` into `out`.
"""
function _sum_chunks_kernel!(
    out::AbstractVector{T}, partials::AbstractMatrix{T}, rows::Int32,
    chunks::Int32,
) where {T}
    row = (Int32(blockIdx().x) - Int32(1)) * Int32(blockDim().x) +
          Int32(threadIdx().x)
    if row > rows
        return
    end
    total = zero(T)
    @inbounds for chunk in Int32(1):chunks
        total += partials[row, chunk]
    end
    @inbounds out[row] = total
    return
end

"""
    _word_dot(total, lookup, word::UInt32, xs::NTuple{16})

Return `total` plus the dot product of the decoded `word` with `xs`.
"""
@inline function _word_dot(total::T, lookup, word::UInt32,
    xs::NTuple{16, T}) where {T}
    genotypes = _decode_word(lookup, word)
    Base.Cartesian.@nexprs 16 s -> begin
        total += genotypes[s] * xs[s]
    end
    return total
end

"""
    _x_segment(x, word_row, m, masked) -> NTuple{16}

The 16 entries of `x` under word row `word_row`; `masked` zeroes samples
past `m`.
"""
@inline function _x_segment(x::AbstractVector{T}, word_row::Int32, m::Int32,
    masked::Bool) where {T}
    base = Int32(16) * (word_row - Int32(1))
    return Base.Cartesian.@ntuple 16 s -> ifelse(
        masked & (base + Int32(s) > m), zero(T),
        @inbounds x[min(base + Int32(s), m)])
end

"""
    _warp_sum(total::T)

Shuffle-reduce `total` across the warp; the sum lands in lane 1.
"""
@inline function _warp_sum(total::T) where {T}
    offset = Int32(16)
    while offset > Int32(0)
        total += shfl_down_sync(0xffffffff, total, offset)
        offset ÷= Int32(2)
    end
    return total
end

"""
    _atx_warp_kernel!(out, words, x, values, m, n)

`transpose(A)*x` kernel: each warp owns `COLUMNS_PER_WARP` SNP columns,
lanes stride over packed word rows sharing each `x` load across the
columns, and a shuffle reduction leaves the sums in lane 1.
"""
function _atx_warp_kernel!(
    out::AbstractVector{T}, words::AbstractMatrix{UInt32},
    x::AbstractVector{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
) where {T}
    total_word_rows = cld(m, Int32(16))
    thread = Int32(threadIdx().x) - Int32(1)
    lane = thread % Int32(32)
    warp = thread ÷ Int32(32)
    first_column = ((Int32(blockIdx().x) - Int32(1)) *
        Int32(WARPS_PER_BLOCK) + warp) * Int32(COLUMNS_PER_WARP) + Int32(1)
    if first_column > n
        return
    end
    # The literal 4 is COLUMNS_PER_WARP. Columns past n alias column n;
    # their sums are never stored.
    Base.Cartesian.@nexprs 4 c -> begin
        column_c = min(first_column + Int32(c - 1), n)
        lookup_c = _genotype_lookup(values, column_c)
        total_c = zero(T)
    end
    word_row = lane + Int32(1)
    while word_row < total_word_rows
        xs = _x_segment(x, word_row, m, false)
        Base.Cartesian.@nexprs 4 c -> begin
            total_c = _word_dot(total_c, lookup_c,
                @inbounds(words[word_row, column_c]), xs)
        end
        word_row += Int32(32)
    end
    if word_row == total_word_rows
        xs = _x_segment(x, word_row, m, true)
        Base.Cartesian.@nexprs 4 c -> begin
            total_c = _word_dot(total_c, lookup_c,
                @inbounds(words[word_row, column_c]), xs)
        end
    end

    Base.Cartesian.@nexprs 4 c -> begin
        total_c = _warp_sum(total_c)
        if lane == Int32(0) && first_column + Int32(c - 1) <= n
            @inbounds out[first_column + Int32(c - 1)] = total_c
        end
    end
    return
end


"""
    _accumulate_tile(genotype_tile, x_tile, offset, lane, lanes, rhs_lane,
                     rhs_lanes, acc, Val(TM), Val(TK))

Add the rank-1 update of reduction index `offset` to a thread's
`TM x TK` register block `acc`.
"""
@generated function _accumulate_tile(
    genotype_tile, x_tile, offset, lane, lanes, rhs_lane, rhs_lanes,
    acc::NTuple{L, T}, ::Val{TM}, ::Val{TK},
) where {L, T, TM, TK}
    loads = Expr[]
    for t in 1:TM
        push!(loads, :($(Symbol(:g, t)) =
            genotype_tile[lane + lanes * Int32($(t - 1)), offset]))
    end
    for u in 1:TK
        push!(loads, :($(Symbol(:x, u)) =
            x_tile[offset, rhs_lane + rhs_lanes * Int32($(u - 1))]))
    end
    terms = Vector{Expr}(undef, L)
    for idx in 1:L
        t = mod(idx - 1, TM) + 1
        u = div(idx - 1, TM) + 1
        terms[idx] = :(muladd($(Symbol(:g, t)), $(Symbol(:x, u)), acc[$idx]))
    end
    return quote
        Base.@_inline_meta
        @inbounds begin
            $(loads...)
        end
        $(Expr(:tuple, terms...))
    end
end

"""
    _accumulate_steps(genotype_tile, x_tile, lane, lanes, rhs_lane, rhs_lanes,
                      acc, Val(STEPS), Val(TM), Val(TK))

Apply `_accumulate_tile` for reduction offsets `1:STEPS`, unrolled in
groups of about 256 multiply-adds per thread.
"""
@generated function _accumulate_steps(
    genotype_tile, x_tile, lane, lanes, rhs_lane, rhs_lanes, acc,
    ::Val{STEPS}, ::Val{TM}, ::Val{TK},
) where {STEPS, TM, TK}
    group = clamp(256 ÷ (TM * TK), 1, STEPS)
    while STEPS % group != 0
        group -= 1
    end
    steps = [:(acc = _accumulate_tile(genotype_tile, x_tile,
        base + Int32($step),
        lane, lanes, rhs_lane, rhs_lanes, acc, Val(TM), Val(TK)))
        for step in 1:group]
    return quote
        Base.@_inline_meta
        base = Int32(0)
        while base < Int32($STEPS)
            $(steps...)
            base += Int32($group)
        end
        return acc
    end
end

"""
    _store_tile!(out, acc, first, lane, lanes, rhs_first, rhs_lane,
                 rhs_lanes, rows, k, split, Val(TM), Val(TK))

Write a thread's `TM x TK` register block into `out[:, :, split]`, skipping
entries past `rows x k`.
"""
@generated function _store_tile!(
    out, acc::NTuple{L, T}, first, lane, lanes, rhs_first, rhs_lane,
    rhs_lanes, rows, k, split, ::Val{TM}, ::Val{TK},
) where {L, T, TM, TK}
    statements = Vector{Expr}(undef, L)
    for idx in 1:L
        t = mod(idx - 1, TM) + 1
        u = div(idx - 1, TM) + 1
        statements[idx] = quote
            row = first - Int32(1) + lane + lanes * Int32($(t - 1))
            rhs_column = rhs_first - Int32(1) + rhs_lane +
                rhs_lanes * Int32($(u - 1))
            if row <= rows && rhs_column <= k
                @inbounds out[row, rhs_column, split] = acc[$idx]
            end
        end
    end
    return quote
        Base.@_inline_meta
        $(statements...)
        nothing
    end
end

"""
    _aX_tiled_kernel!(out, words, rhs, values, m, n, k, split_columns,
                      Val(BM), Val(BN), Val(BK), Val(TM), Val(TK))

Register-blocked `A*X` kernel. A block covers `BM` samples x `BK`
right-hand-side columns over the SNP columns of split `blockIdx().z`, in
steps of `BN`; each thread owns a `TM x TK` block of
`out[:, :, blockIdx().z]`.
"""
function _aX_tiled_kernel!(
    out::AbstractArray{T, 3}, words::AbstractMatrix{UInt32},
    rhs::AbstractMatrix{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
    k::Int32, split_columns::Int32,
    ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TM}, ::Val{TK},
) where {T, BM, BN, BK, TM, TK}
    bm = Int32(BM)
    bn = Int32(BN)
    bk = Int32(BK)
    threads = Int32(blockDim().x)
    rows_lanes = bm ÷ Int32(TM)
    thread = Int32(threadIdx().x)
    row_lane = mod(thread - Int32(1), rows_lanes) + Int32(1)
    rhs_lane = div(thread - Int32(1), rows_lanes) + Int32(1)

    sample_first = (Int32(blockIdx().x) - Int32(1)) * bm + Int32(1)
    rhs_column_first = (Int32(blockIdx().y) - Int32(1)) * bk + Int32(1)
    split = Int32(blockIdx().z)
    column_stop = min(split * split_columns, n)

    word_rows_per_tile = bm ÷ Int32(16)
    total_word_rows = cld(m, Int32(16))
    group_word_row_first = (Int32(blockIdx().x) - Int32(1)) *
        word_rows_per_tile

    genotype_tile = CuStaticSharedArray(T, (BM + 1, BN))
    x_tile = CuStaticSharedArray(T, (BN, BK))

    acc = ntuple(_ -> zero(T), Val(TM * TK))

    column_first = (split - Int32(1)) * split_columns + Int32(1)
    while column_first <= column_stop
        wi = thread
        while wi <= word_rows_per_tile * bn
            col_offset = mod(wi - Int32(1), bn) + Int32(1)
            word_row = div(wi - Int32(1), bn) + Int32(1)
            column = column_first + col_offset - Int32(1)
            global_word_row = group_word_row_first + word_row
            genotypes = ntuple(_ -> zero(T), Val(16))
            if column <= column_stop && global_word_row <= total_word_rows
                word = @inbounds words[global_word_row, column]
                genotypes = _decode_word(_genotype_lookup(values, column), word)
            end
            base = Int32(16) * (word_row - Int32(1))
            Base.Cartesian.@nexprs 16 s -> begin
                @inbounds genotype_tile[base + Int32(s), col_offset] =
                    genotypes[s]
            end
            wi += threads
        end

        xi = thread
        while xi <= bn * bk
            col_offset = mod(xi - Int32(1), bn) + Int32(1)
            rhs_offset = div(xi - Int32(1), bn) + Int32(1)
            column = column_first + col_offset - Int32(1)
            rhs_column = rhs_column_first + rhs_offset - Int32(1)
            value = (column <= column_stop && rhs_column <= k) ?
                (@inbounds rhs[column, rhs_column]) : zero(T)
            @inbounds x_tile[col_offset, rhs_offset] = value
            xi += threads
        end

        sync_threads()
        acc = _accumulate_steps(
            genotype_tile, x_tile, row_lane, rows_lanes, rhs_lane,
            bk ÷ Int32(TK), acc, Val(BN), Val(TM), Val(TK),
        )
        sync_threads()
        column_first += bn
    end

    _store_tile!(
        out, acc, sample_first, row_lane, rows_lanes, rhs_column_first,
        rhs_lane, bk ÷ Int32(TK), m, k, split, Val(TM), Val(TK),
    )
    return
end

"""
    _atX_tiled_kernel!(out, words, rhs, values, m, n, k, split_word_rows,
                       Val(BM), Val(BN), Val(BK), Val(TN), Val(TK))

Register-blocked `transpose(A)*X` kernel. A block covers `BN` SNP columns x
`BK` right-hand-side columns over the word rows of split `blockIdx().z`, in
steps of `BM` samples; each thread owns a `TN x TK` block of
`out[:, :, blockIdx().z]`.
"""
function _atX_tiled_kernel!(
    out::AbstractArray{T, 3}, words::AbstractMatrix{UInt32},
    rhs::AbstractMatrix{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
    k::Int32, split_word_rows::Int32,
    ::Val{BM}, ::Val{BN}, ::Val{BK}, ::Val{TN}, ::Val{TK},
) where {T, BM, BN, BK, TN, TK}
    bm = Int32(BM)
    bn = Int32(BN)
    bk = Int32(BK)
    threads = Int32(blockDim().x)
    col_lanes = bn ÷ Int32(TN)
    thread = Int32(threadIdx().x)
    col_lane = mod(thread - Int32(1), col_lanes) + Int32(1)
    rhs_lane = div(thread - Int32(1), col_lanes) + Int32(1)

    column_first = (Int32(blockIdx().x) - Int32(1)) * bn + Int32(1)
    rhs_column_first = (Int32(blockIdx().y) - Int32(1)) * bk + Int32(1)
    split = Int32(blockIdx().z)

    word_rows_per_tile = bm ÷ Int32(16)
    total_word_rows = cld(m, Int32(16))
    word_row_stop = min(split * split_word_rows, total_word_rows)
    sample_stop = min(Int32(16) * word_row_stop, m)

    genotype_tile = CuStaticSharedArray(T, (BN + 1, BM))
    x_tile = CuStaticSharedArray(T, (BM, BK))

    acc = ntuple(_ -> zero(T), Val(TN * TK))

    sample_first = Int32(16) * (split - Int32(1)) * split_word_rows +
        Int32(1)
    while sample_first <= sample_stop
        group_sample_word_first = (sample_first - Int32(1)) ÷ Int32(16)
        wi = thread
        while wi <= word_rows_per_tile * bn
            col_offset = mod(wi - Int32(1), bn) + Int32(1)
            word_row = div(wi - Int32(1), bn) + Int32(1)
            column = column_first + col_offset - Int32(1)
            global_word_row = group_sample_word_first + word_row
            genotypes = ntuple(_ -> zero(T), Val(16))
            if column <= n && global_word_row <= word_row_stop
                word = @inbounds words[global_word_row, column]
                genotypes = _decode_word(_genotype_lookup(values, column), word)
            end
            base = Int32(16) * (word_row - Int32(1))
            Base.Cartesian.@nexprs 16 s -> begin
                @inbounds genotype_tile[col_offset, base + Int32(s)] =
                    genotypes[s]
            end
            wi += threads
        end

        xi = thread
        while xi <= bm * bk
            sample_offset = mod(xi - Int32(1), bm) + Int32(1)
            rhs_offset = div(xi - Int32(1), bm) + Int32(1)
            sample = sample_first + sample_offset - Int32(1)
            rhs_column = rhs_column_first + rhs_offset - Int32(1)
            value = (sample <= sample_stop && rhs_column <= k) ?
                (@inbounds rhs[sample, rhs_column]) : zero(T)
            @inbounds x_tile[sample_offset, rhs_offset] = value
            xi += threads
        end

        sync_threads()
        acc = _accumulate_steps(
            genotype_tile, x_tile, col_lane, col_lanes, rhs_lane,
            bk ÷ Int32(TK), acc, Val(BM), Val(TN), Val(TK),
        )
        sync_threads()
        sample_first += bm
    end

    _store_tile!(
        out, acc, column_first, col_lane, col_lanes, rhs_column_first,
        rhs_lane, bk ÷ Int32(TK), n, k, split, Val(TN), Val(TK),
    )
    return
end
