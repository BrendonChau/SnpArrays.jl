"""
    _lookup_codes(w0, w1, w2, w3) -> NTuple{4, UInt32}

Bit-transpose the words of four consecutive SNPs of one word row: byte `b`
of result `r + 1` holds the code byte of sample `4b + r`, SNP `l` in bits
`2l:2l+1`.
"""
@inline function _lookup_codes(w0::UInt32, w1::UInt32, w2::UInt32,
    w3::UInt32)
    lo = 0x33333333
    u0 = (w0 & lo) | ((w1 & lo) << 2)
    v0 = ((w0 >> 2) & lo) | (w1 & ~lo)
    u1 = (w2 & lo) | ((w3 & lo) << 2)
    v1 = ((w2 >> 2) & lo) | (w3 & ~lo)
    nib = 0x0f0f0f0f
    return ((u0 & nib) | ((u1 & nib) << 4),
        (v0 & nib) | ((v1 & nib) << 4),
        ((u0 >> 4) & nib) | (u1 & ~nib),
        ((v0 >> 4) & nib) | (v1 & ~nib))
end

"""
    _vadd(a::NTuple{C, VecElement{T}}, b::NTuple{C, VecElement{T}})

Elementwise sum of two vector tuples.
"""
@inline function _vadd(a::NTuple{C, VecElement{T}},
    b::NTuple{C, VecElement{T}}) where {C, T}
    return ntuple(i -> VecElement(a[i].value + b[i].value), Val(C))
end

"""
    _lookup_halves!(halves, rhs_t, values, n, k, block, block_stop,
                    column_first, Val(C), Val(CL), Val(G))

Fill the 16-entry pair tables of SNPs `(1, 2)` and `(3, 4)` of each of the
`G` 4-SNP blocks starting at `block`, for the rhs columns of this slice.
"""
@inline function _lookup_halves!(
    halves, rhs_t::AbstractMatrix{T}, values::AbstractMatrix{T}, n::Int32,
    k::Int32, block::Int32, block_stop::Int32, column_first::Int32,
    ::Val{C}, ::Val{CL}, ::Val{G},
) where {T, C, CL, G}
    u = Int32(threadIdx().x) - Int32(1)
    while u < Int32(CL * 32 * G)
        cl = u % Int32(CL)
        idx = (u ÷ Int32(CL)) % Int32(32)
        g = u ÷ Int32(CL * 32)
        b = block + g
        pair = idx ÷ Int32(16)
        code = idx % Int32(16)
        ja = Int32(4) * (b - Int32(1)) + Int32(2) * pair + Int32(1)
        jb = ja + Int32(1)
        live = b <= block_stop
        wa = live & (ja <= n) ?
            (@inbounds values[(code & Int32(3)) + Int32(1), ja]) : zero(T)
        wb = live & (jb <= n) ?
            (@inbounds values[(code >> Int32(2)) + Int32(1), jb]) : zero(T)
        entry = ntuple(Val(C)) do c
            column = column_first + cl * Int32(C) + Int32(c)
            xa = (wa != zero(T)) & (column <= k) ?
                (@inbounds rhs_t[column, ja]) : zero(T)
            xb = (wb != zero(T)) & (column <= k) ?
                (@inbounds rhs_t[column, jb]) : zero(T)
            VecElement(muladd(wb, xb, wa * xa))
        end
        @inbounds halves[cl + Int32(1), idx + Int32(1), g + Int32(1)] = entry
        u += Int32(blockDim().x)
    end
    return nothing
end

"""
    _lookup_entries!(tables, halves, buffer, Val(CL), Val(G))

Fill the 256-row tables of buffer `buffer` from the pair tables: row `code`
is the sum of the `(1, 2)` entry `code & 15` and the `(3, 4)` entry
`code >> 4`.
"""
@inline function _lookup_entries!(
    tables, halves, buffer::Int32, ::Val{CL}, ::Val{G},
) where {CL, G}
    u = Int32(threadIdx().x) - Int32(1)
    while u < Int32(CL * 64 * G)
        cl = u % Int32(CL) + Int32(1)
        lo = (u ÷ Int32(CL)) % Int32(16)
        quarter = (u ÷ Int32(CL * 16)) % Int32(4)
        g = u ÷ Int32(CL * 64) + Int32(1)
        low = @inbounds halves[cl, lo + Int32(1), g]
        Base.Cartesian.@nexprs 4 h -> begin
            hi = Int32(4) * quarter + Int32(h - 1)
            @inbounds tables[cl, Int32(16) * hi + lo + Int32(1), g, buffer] =
                _vadd(low, halves[cl, Int32(17) + hi, g])
        end
        u += Int32(blockDim().x)
    end
    return nothing
end

"""
    _lookup_words(words, word_row, live_row, b, block_stop, n)

The four words of 4-SNP block `b` in `word_row`, zero past `block_stop`,
`n`, or the last word row.
"""
@inline function _lookup_words(
    words::AbstractMatrix{UInt32}, word_row::Int32, live_row::Bool, b::Int32,
    block_stop::Int32, n::Int32,
)
    j = Int32(4) * (b - Int32(1))
    live = live_row & (b <= block_stop)
    return Base.Cartesian.@ntuple 4 l -> live & (j + Int32(l) <= n) ?
        (@inbounds words[word_row, j + Int32(l)]) : UInt32(0)
end

"""
    _lookup_gather(acc,tables, cl, codes, g, buffer)

Add to each of the 16 accumulators the table row selected by its sample's
code byte.
"""
@inline function _lookup_gather(
    acc::NTuple{16, V}, tables, cl::Int32, codes::NTuple{4, UInt32},
    g::Int32, buffer::Int32,
) where {V}
    return Base.Cartesian.@ntuple 16 s -> begin
        code = Int32((codes[(s - 1) % 4 + 1] >> (8 * ((s - 1) ÷ 4))) &
            0x000000ff)
        _vadd(acc[s], @inbounds tables[cl, code + Int32(1), g, buffer])
    end
end

"""
    _aX_lookup_kernel!(out, words, rhs_t, values, m, n, k, split_blocks,
                       Val(C), Val(CL), Val(G))

Lookup-table `A*X` kernel. A block covers the `C * CL` rhs columns of slice
`blockIdx().x`, `16 * blockDim().x ÷ CL` samples, and the 4-SNP blocks of
split `blockIdx().z`, `G` per stage; it builds each stage's 256-row tables
in shared memory while gathering the previous stage, and each thread
accumulates 16 samples x `C` columns into `out[:, :, blockIdx().z]`.
"""
function _aX_lookup_kernel!(
    out::AbstractArray{T, 3}, words::AbstractMatrix{UInt32},
    rhs_t::AbstractMatrix{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
    k::Int32, split_blocks::Int32, ::Val{C}, ::Val{CL}, ::Val{G},
) where {T, C, CL, G}
    V = NTuple{C, VecElement{T}}
    tables = CuDynamicSharedArray(V, (CL, 256, G, 2))
    halves = CuDynamicSharedArray(V, (CL, 32, G), sizeof(V) * CL * 512 * G)
    thread = Int32(threadIdx().x) - Int32(1)
    cl = thread % Int32(CL) + Int32(1)
    word_row = (Int32(blockIdx().y) - Int32(1)) *
        (Int32(blockDim().x) ÷ Int32(CL)) + thread ÷ Int32(CL) + Int32(1)
    live_row = word_row <= cld(m, Int32(16))
    column_first = (Int32(blockIdx().x) - Int32(1)) * Int32(C * CL)
    split = Int32(blockIdx().z)
    block_first = (split - Int32(1)) * split_blocks + Int32(1)
    block_stop = min(split * split_blocks, cld(n, Int32(4)))
    stages = cld(block_stop - block_first + Int32(1), Int32(G))

    _lookup_halves!(halves, rhs_t, values, n, k, block_first, block_stop,
        column_first, Val(C), Val(CL), Val(G))
    sync_threads()
    _lookup_entries!(tables, halves, Int32(1), Val(CL), Val(G))
    sync_threads()

    zero_v = ntuple(_ -> VecElement(zero(T)), Val(C))
    acc = ntuple(_ -> zero_v, Val(16))
    # Word loads run one 4-SNP block ahead of the gather.
    w = _lookup_words(words, word_row, live_row, block_first, block_stop, n)
    stage = Int32(1)
    while stage <= stages
        buffer = (stage - Int32(1)) % Int32(2) + Int32(1)
        stage_first = block_first + (stage - Int32(1)) * Int32(G)
        if stage < stages
            _lookup_halves!(halves, rhs_t, values, n, k,
                stage_first + Int32(G), block_stop, column_first,
                Val(C), Val(CL), Val(G))
        end
        sync_threads()
        if stage < stages
            _lookup_entries!(tables, halves, Int32(3) - buffer,
                Val(CL), Val(G))
        end
        g = Int32(1)
        while g <= Int32(G)
            b = stage_first + g - Int32(1)
            w_next = _lookup_words(words, word_row, live_row, b + Int32(1),
                block_stop, n)
            acc = _lookup_gather(acc, tables, cl, _lookup_codes(w...), g,
                buffer)
            w = w_next
            g += Int32(1)
        end
        sync_threads()
        stage += Int32(1)
    end

    base = Int32(16) * (word_row - Int32(1))
    column = column_first + (cl - Int32(1)) * Int32(C)
    Base.Cartesian.@nexprs 16 s -> begin
        if base + Int32(s) <= m
            _lookup_store!(out, acc[s], base + Int32(s), column, k, split)
        end
    end
    return
end

"""
    _lookup_store!(out, vector, row, column, k, split)

Store `vector` into `out[row, column .+ (1:C), split]`, skipping columns
past `k`.
"""
@generated function _lookup_store!(
    out, vector::NTuple{C, VecElement{T}}, row::Int32, column::Int32,
    k::Int32, split::Int32,
) where {C, T}
    stores = [quote
        if column + Int32($c) <= k
            @inbounds out[row, column + Int32($c), split] = vector[$c].value
        end
    end for c in 1:C]
    return quote
        Base.@_inline_meta
        $(stores...)
        nothing
    end
end

"""
`(C, CL, G, threads, per_sm)` of the lookup-table `A*X` kernel for `k == 1`,
`k <= 8`, `k <= 32`, `k > 32`, for Float32; tuned on an A100.
"""
const LOOKUP_CONFIGS = ((1, 1, 16, 256, 2), (4, 2, 8, 512, 2),
    (4, 4, 4, 512, 3), (4, 4, 4, 512, 1))

_lookup_config(k::Integer) = LOOKUP_CONFIGS[k == 1 ? 1 : k <= 8 ? 2 :
    k <= 32 ? 3 : 4]

"""
    _lookup_mul!(out::CuMatrix{T}, s::CuSnpArray{T}, X::CuMatrix{T},
                 config) -> out

`out = s * X` with the lookup-table kernel; `config` is
`(C, CL, G, threads, per_sm)` with `per_sm` the resident blocks per SM
used to size the SNP splits.
"""
function _lookup_mul!(
    out::CuMatrix{T}, s::CuSnpArray{T}, X::CuMatrix{T},
    config::NTuple{5, Int},
) where {T}
    (C, CL, G, threads, per_sm) = config
    # Fewer blocks per stage and threads keep Float64 tables within shared
    # memory and accumulators within the register file.
    G = max(1, G * 4 ÷ sizeof(T))
    threads = C > 1 ? threads * 4 ÷ sizeof(T) : threads
    m, n = size(s)
    k = size(X, 2)
    slices = cld(k, C * CL)
    tiles = cld(cld(m, 16), threads ÷ CL)
    blocks = cld(n, 4)
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    splits = clamp(cld(2 * sms * per_sm, slices * tiles), 1,
        max(1, blocks ÷ (4 * G)))
    split_blocks = G * cld(cld(blocks, splits), G)
    splits = cld(blocks, split_blocks)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        CuArray{T, 3}(undef, m, k, splits)
    rhs_t = k == 1 ? reshape(X, 1, n) : permutedims(X)
    args = (partials, s.data, rhs_t, s.values, Int32(m), Int32(n), Int32(k),
        Int32(split_blocks), Val(C), Val(CL), Val(G))
    kernel = @cuda launch=false _aX_lookup_kernel!(args...)
    shmem = C * sizeof(T) * CL * 544 * G
    attributes(kernel.fun)[FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
        shmem
    kernel(args...; threads, blocks=(slices, tiles, splits), shmem)
    k == 1 || unsafe_free!(rhs_t)
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end
