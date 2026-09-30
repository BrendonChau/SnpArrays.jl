"""WMMA shape and accumulator type of the tensor-core kernels."""
const WMMA_CONFIG = WMMA.Config{16, 16, 16, Float32}

"""SNP columns per shared-memory stage of the tensor-core `A*X` kernel."""
const WMMA_BN = 32

"""
Samples per shared-memory stage of the tensor-core `transpose(A)*X`
kernel.
"""
const WMMA_T_BS = 32

"""Fewest rhs columns for which Float32 `A*X` uses the tensor-core kernel."""
const WMMA_MIN_RHS = 16

"""
Fewest rhs columns for which Float32 `transpose(A)*X` uses the tensor-core
kernel.
"""
const WMMA_T_MIN_RHS = 2

"""Target resident blocks per SM when splitting the reduction."""
const WMMA_BLOCKS_PER_SM = 16

"""
    _wmma_rhs_width(k) -> Int

Rhs columns per block for `k` columns: the power of two from 32 to 256
that covers `k`, capped at 256.
"""
_wmma_rhs_width(k::Integer) = clamp(nextpow(2, k), 32, 256)

"""
    _wmma_block_rows(BK) -> Int

Output rows per block (samples for `A*X`, SNPs for `transpose(A)*X`) at
rhs width `BK`: 256 up to `BK = 64`, then `16384 ÷ BK`. Each warp's tile
stays 64 x 32 or smaller, and a decoded genotype tile serves all `BK` rhs
columns of its block.
"""
_wmma_block_rows(BK::Integer) = BK <= 64 ? 256 : 16384 ÷ BK

"""
    _half_pointer(p, offset)

Float16 pointer `offset` halves past `p`.
"""
@inline function _half_pointer(p::Core.LLVMPtr{T, A},
    offset::Int32) where {T, A}
    return reinterpret(Core.LLVMPtr{Float16, A}, p) + 2 * Int(offset)
end

"""
    _split_tables(values::CuMatrix{Float32}) -> CuMatrix{UInt64}

`2 x n` packed Float16 tables: row 1 holds the high Float16 parts of the
four code values of each column (code `c` in bits `16c:16c+15`), row 2 the
low parts `Float16(v - high)`.
"""
function _split_tables(values::CuMatrix{Float32})
    high = Float16.(values)
    low = Float16.(values .- Float32.(high))
    bits(x) = UInt64(reinterpret(UInt16, x))
    pack(a, b, c, d) = bits(a) | (bits(b) << 16) | (bits(c) << 32) |
        (bits(d) << 48)
    th = pack.(view(high, 1, :), view(high, 2, :), view(high, 3, :),
        view(high, 4, :))
    tl = pack.(view(low, 1, :), view(low, 2, :), view(low, 3, :),
        view(low, 4, :))
    return vcat(transpose(th), transpose(tl))
end

"""
    _split_rhs(X, rows_padded, k_padded) -> (high, low, scale_inv)

`X` zero padded to `rows_padded x k_padded`, scaled per column by a power
of two that puts its largest magnitude in `[2^14, 2^15)`, and split into
high and low Float16 parts, returned 8 per `UInt128`; `scale_inv` undoes
the scaling.
"""
function _split_rhs(X::CuMatrix{Float32}, rows_padded::Int, k_padded::Int)
    rows, k = size(X)
    y = CuArray{Float32}(undef, rows_padded, k_padded)
    fill!(y, 0.0f0)
    view(y, 1:rows, 1:k) .= X
    scale = vec(maximum(abs, y; dims=1))
    scale .= ifelse.(scale .> 0,
        exp2.(14 .- Float32.(exponent.(max.(scale, floatmin(Float32))))),
        1.0f0)
    y .*= transpose(scale)
    high = Float16.(y)
    low = Float16.(y .- Float32.(high))
    unsafe_free!(y)
    scale .= inv.(scale)
    return reinterpret(UInt128, high), reinterpret(UInt128, low), scale
end

"""
    _decode_split(w::UInt32, th::UInt64, tl::UInt64) -> NTuple{4, UInt128}

The high and low Float16 lookup values of the 16 samples of packed word
`w` from its column's packed tables: (high 1-8, high 9-16, low 1-8,
low 9-16).
"""
@inline function _decode_split(w::UInt32, th::UInt64, tl::UInt64)
    parts = ntuple(Val(4)) do q
        a = UInt64(0)
        b = UInt64(0)
        for i in 0:3
            c = Int((w >> (2 * (4 * (q - 1) + i))) & UInt32(3))
            a |= ((th >> (16 * c)) & UInt64(0xffff)) << (16 * i)
            b |= ((tl >> (16 * c)) & UInt64(0xffff)) << (16 * i)
        end
        (a, b)
    end
    return (UInt128(parts[1][1]) | (UInt128(parts[2][1]) << 64),
        UInt128(parts[3][1]) | (UInt128(parts[4][1]) << 64),
        UInt128(parts[1][2]) | (UInt128(parts[2][2]) << 64),
        UInt128(parts[3][2]) | (UInt128(parts[4][2]) << 64))
end

"""
    _fragment_add(acc, stage)

Elementwise Float32 sum of two tuples of accumulator fragments on CUDA
cores, which round to nearest; tensor-core accumulation truncates, so each
stage starts from zero and is added here.
"""
@inline function _fragment_add(acc::NTuple{L, F},
    stage::NTuple{L, F}) where {L, F}
    return ntuple(Val(L)) do f
        F(ntuple(i -> acc[f].x[i] + stage[f].x[i], Val(8)))
    end
end

"""
    _wmma_step(acc, gh, gl, bh, bl, a_offset, b_offset, Val(LAYOUT),
               Val(LDA), Val(LDB), Val(STRIDE), Val(FN))

Accumulate one 16-deep step into a warp's `4 x FN` fragments as
`Vh Xh + Vh Xl + Vl Xh`; the four genotype fragments are `STRIDE` halves
apart in the `LAYOUT` tiles `gh`, `gl`.
"""
@generated function _wmma_step(acc, gh, gl, bh, bl, a_offset::Int32,
    b_offset::Int32, ::Val{LAYOUT}, ::Val{LDA}, ::Val{LDB}, ::Val{STRIDE},
    ::Val{FN}) where {LAYOUT, LDA, LDB, STRIDE, FN}
    body = Expr[]
    for i in 1:4
        offset = :(a_offset + Int32($(STRIDE * (i - 1))))
        push!(body, :($(Symbol(:ah, i)) = WMMA.load_a(
            _half_pointer(pointer(gh), $offset), $LDA, $LAYOUT,
            WMMA_CONFIG)))
        push!(body, :($(Symbol(:al, i)) = WMMA.load_a(
            _half_pointer(pointer(gl), $offset), $LDA, $LAYOUT,
            WMMA_CONFIG)))
    end
    results = Symbol[]
    for j in 1:FN
        offset = :(b_offset + Int32($(16 * (j - 1) * LDB)))
        h = Symbol(:h, j)
        l = Symbol(:l, j)
        push!(body, :($h = WMMA.load_b(_half_pointer(pointer(bh), $offset),
            $LDB, WMMA.ColMajor, WMMA_CONFIG)))
        push!(body, :($l = WMMA.load_b(_half_pointer(pointer(bl), $offset),
            $LDB, WMMA.ColMajor, WMMA_CONFIG)))
        for i in 1:4
            f = 4 * (j - 1) + i
            ah = Symbol(:ah, i)
            al = Symbol(:al, i)
            c = Symbol(:c, f)
            push!(body, :($c = WMMA.mma($al, $h, WMMA.mma($ah, $l,
                WMMA.mma($ah, $h, acc[$f], WMMA_CONFIG), WMMA_CONFIG),
                WMMA_CONFIG)))
            push!(results, c)
        end
    end
    return quote
        Base.@_inline_meta
        $(body...)
        return ($(results...),)
    end
end

"""
    _wmma_store!(out, scratch, frag, scale_inv, warp, lane, row0, col0,
                 rows, k, split)

Write one 16 x 16 fragment at zero-based origin `(row0, col0)` of
`out[:, :, split]`, times `scale_inv[col]`.
"""
@inline function _wmma_store!(out, scratch, frag, scale_inv, warp::Int32,
    lane::Int32, row0::Int32, col0::Int32, rows::Int32, k::Int32,
    split::Int32)
    WMMA.store_d(pointer(scratch, Int32(256) * warp + Int32(1)), frag, 16,
        WMMA.ColMajor, WMMA_CONFIG)
    e = lane
    while e < Int32(256)
        rr = e % Int32(16)
        cc = e ÷ Int32(16)
        row = row0 + rr + Int32(1)
        col = col0 + cc + Int32(1)
        if row <= rows && col <= k
            @inbounds out[row, col, split] = scratch[rr + Int32(1),
                cc + Int32(1), warp + Int32(1)] * scale_inv[col]
        end
        e += Int32(32)
    end
    return nothing
end

"""
    _wmma_store_all!(out, scratch, acc, scale_inv, warp, lane, row0, col0,
                     rows, k, split)

Write a warp's `4 x FN` fragments, unrolled.
"""
@generated function _wmma_store_all!(out, scratch, acc::NTuple{L},
    scale_inv, warp::Int32, lane::Int32, row0::Int32, col0::Int32,
    rows::Int32, k::Int32, split::Int32) where {L}
    stores = [:(_wmma_store!(out, scratch, acc[$f], scale_inv, warp, lane,
        row0 + Int32($(16 * ((f - 1) % 4))),
        col0 + Int32($(16 * ((f - 1) ÷ 4))), rows, k, split)) for f in 1:L]
    return quote
        Base.@_inline_meta
        $(stores...)
        return nothing
    end
end

"""
    _load_words(words, tid, word_row0, word_stop, wr_count, column0,
                column_stop, Val(UNITS)) -> NTuple{cld(UNITS, 256), UInt32}

Load thread `tid`'s packed words of a `UNITS`-word tile, `wr_count` word
rows deep, starting at word row `word_row0 + 1` and column `column0 + 1`;
words past `word_stop`, `column_stop` or the tile load as 0.
"""
@inline function _load_words(words, tid::Int32, word_row0::Int32,
    word_stop::Int32, wr_count::Int32, column0::Int32, column_stop::Int32,
    ::Val{UNITS}) where {UNITS}
    return ntuple(Val(cld(UNITS, 256))) do r
        u = tid + Int32(256 * (r - 1))
        column = column0 + u ÷ wr_count + Int32(1)
        word_row = word_row0 + u % wr_count + Int32(1)
        live = (column <= column_stop) & (word_row <= word_stop) &
            (u < Int32(UNITS))
        live ? (@inbounds words[word_row, column]) : UInt32(0)
    end
end

"""
    _store_words!(gh, gl, loaded, tables, tid, wr_count, column0,
                  column_stop, Val(UNITS))

Decode the words from `_load_words` with their columns' packed tables into
the `[sample, SNP]` Float16 tiles `gh`, `gl`.
"""
@generated function _store_words!(gh, gl, loaded::NTuple{R, UInt32}, tables,
    tid::Int32, wr_count::Int32, column0::Int32, column_stop::Int32,
    ::Val{UNITS}) where {R, UNITS}
    stores = map(1:R) do r
        store = :(_store_word!(gh, gl, loaded[$r], tables,
            tid + Int32($(256 * (r - 1))), wr_count, column0, column_stop))
        256 * r <= UNITS ? store :
            :(tid + Int32($(256 * (r - 1))) < Int32($UNITS) && $store)
    end
    return quote
        Base.@_inline_meta
        $(stores...)
        return nothing
    end
end

"""
    _store_word!(gh, gl, w, tables, u, wr_count, column0, column_stop)

Decode word `w`, unit `u` of the tile, into `gh`, `gl`; columns past
`column_stop` decode to 0.
"""
@inline function _store_word!(gh, gl, w::UInt32, tables, u::Int32,
    wr_count::Int32, column0::Int32, column_stop::Int32)
    wr = u % wr_count
    j = u ÷ wr_count
    column = column0 + j + Int32(1)
    live = column <= column_stop
    th = live ? (@inbounds tables[1, column]) : UInt64(0)
    tl = live ? (@inbounds tables[2, column]) : UInt64(0)
    d = _decode_split(w, th, tl)
    @inbounds gh[Int32(2) * wr + Int32(1), j + Int32(1)] = d[1]
    @inbounds gh[Int32(2) * wr + Int32(2), j + Int32(1)] = d[2]
    @inbounds gl[Int32(2) * wr + Int32(1), j + Int32(1)] = d[3]
    @inbounds gl[Int32(2) * wr + Int32(2), j + Int32(1)] = d[4]
    return nothing
end

"""
    _cp_async!(dst, src)

Start an asynchronous 16-byte global-to-shared copy (`cp.async.cg`,
sm_80+), bypassing registers and L1.
"""
@inline function _cp_async!(dst::Core.LLVMPtr{T, 3},
    src::Core.LLVMPtr{T, 1}) where {T}
    ccall("llvm.nvvm.cp.async.cg.shared.global.16", llvmcall, Cvoid,
        (Core.LLVMPtr{UInt8, 3}, Core.LLVMPtr{UInt8, 1}),
        reinterpret(Core.LLVMPtr{UInt8, 3}, dst),
        reinterpret(Core.LLVMPtr{UInt8, 1}, src))
    return nothing
end

"""Commit this thread's outstanding `cp.async` copies as a group."""
@inline _cp_async_commit() =
    ccall("llvm.nvvm.cp.async.commit.group", llvmcall, Cvoid, ())

"""Wait until all of this thread's `cp.async` copies have landed."""
@inline _cp_async_wait() =
    ccall("llvm.nvvm.cp.async.wait.all", llvmcall, Cvoid, ())

"""
    _copy_rhs_async!(bh, bl, yh, yl, tid, source0, rhs0, buffer, Val(BK))

Start copying thread `tid`'s 128-bit words of the `4 x BK` split rhs tile
(32 rows) starting at row word `source0 + 1` and column `rhs0 + 1` into
shared buffer `buffer` (0 or 1) of `bh`, `bl`, and commit the copies.
"""
@inline function _copy_rhs_async!(bh, bl, yh, yl, tid::Int32,
    source0::Int32, rhs0::Int32, buffer::Int32, ::Val{BK}) where {BK}
    u = tid
    while u < Int32(4 * BK)
        row = u % Int32(4)
        col = u ÷ Int32(4)
        dst = Int(row + Int32(size(bh, 1)) * (col + Int32(BK) * buffer)) + 1
        src = Int(source0 + row) + size(yh, 1) * Int(rhs0 + col) + 1
        _cp_async!(pointer(bh, dst), pointer(yh, src))
        _cp_async!(pointer(bl, dst), pointer(yl, src))
        u += Int32(256)
    end
    _cp_async_commit()
    return nothing
end

"""
    _aX_wmma_kernel!(out, words, tables, yh, yl, scale_inv, m, n, k,
                     split_columns, Val(BK), Val(BM))

Tensor-core `A*X` kernel on the centered lookup values. A block of 8 warps
covers `BM` samples x `BK` rhs columns over the SNP columns of split
`blockIdx().z`, `WMMA_BN` per stage; warps tile it `BM ÷ 64` down.
"""
function _aX_wmma_kernel!(
    out::AbstractArray{Float32, 3}, words::AbstractMatrix{UInt32},
    tables::AbstractMatrix{UInt64}, yh::AbstractMatrix{UInt128},
    yl::AbstractMatrix{UInt128}, scale_inv::AbstractVector{Float32},
    m::Int32, n::Int32, k::Int32, split_columns::Int32, ::Val{BK},
    ::Val{BM},
) where {BK, BM}
    WM = BM ÷ 64
    WN = 8 ÷ WM
    FN = BK ÷ (16 * WN)
    UNITS = 2 * BM
    LDG = BM + 8
    LDB = WMMA_BN + 8
    G8 = LDG ÷ 8
    B8 = LDB ÷ 8
    gh = CuDynamicSharedArray(UInt128, (G8, WMMA_BN))
    gl = CuDynamicSharedArray(UInt128, (G8, WMMA_BN), 16 * G8 * WMMA_BN)
    bh = CuDynamicSharedArray(UInt128, (B8, BK, 2), 32 * G8 * WMMA_BN)
    bl = CuDynamicSharedArray(UInt128, (B8, BK, 2),
        32 * G8 * WMMA_BN + 32 * B8 * BK)
    scratch = CuDynamicSharedArray(Float32, (16, 16, 8),
        32 * G8 * WMMA_BN + 64 * B8 * BK)

    tid = Int32(threadIdx().x) - Int32(1)
    warp = tid ÷ Int32(32)
    lane = tid % Int32(32)
    wm = warp % Int32(WM)
    wn = warp ÷ Int32(WM)
    sample0 = (Int32(blockIdx().x) - Int32(1)) * Int32(BM)
    rhs0 = (Int32(blockIdx().y) - Int32(1)) * Int32(BK)
    split = Int32(blockIdx().z)
    column_stop = min(split * split_columns, n)

    acc = ntuple(_ -> WMMA.fill_c(0.0f0, WMMA_CONFIG), Val(4 * FN))
    column0 = (split - Int32(1)) * split_columns
    word_row0 = sample0 ÷ Int32(16)
    word_stop = cld(m, Int32(16))
    wr_count = Int32(BM ÷ 16)
    loaded = _load_words(words, tid, word_row0, word_stop, wr_count,
        column0, column_stop, Val(UNITS))
    buffer = Int32(0)
    _copy_rhs_async!(bh, bl, yh, yl, tid, column0 ÷ Int32(8), rhs0, buffer,
        Val(BK))
    while column0 < column_stop
        _store_words!(gh, gl, loaded, tables, tid, wr_count, column0,
            column_stop, Val(UNITS))
        _cp_async_wait()
        sync_threads()
        next = column0 + Int32(WMMA_BN)
        if next < column_stop
            loaded = _load_words(words, tid, word_row0, word_stop,
                wr_count, next, column_stop, Val(UNITS))
            _copy_rhs_async!(bh, bl, yh, yl, tid, next ÷ Int32(8), rhs0,
                Int32(1) - buffer, Val(BK))
        end
        stage = ntuple(_ -> WMMA.fill_c(0.0f0, WMMA_CONFIG), Val(4 * FN))
        kk = Int32(0)
        while kk < Int32(WMMA_BN)
            stage = _wmma_step(stage, gh, gl, bh, bl,
                wm * Int32(64) + kk * Int32(LDG),
                kk + wn * Int32(16 * FN * LDB) + buffer * Int32(BK * LDB),
                Val(WMMA.ColMajor), Val(LDG), Val(LDB), Val(16), Val(FN))
            kk += Int32(16)
        end
        acc = _fragment_add(acc, stage)
        sync_threads()
        buffer = Int32(1) - buffer
        column0 = next
    end
    _wmma_store_all!(out, scratch, acc, scale_inv, warp, lane,
        sample0 + wm * Int32(64), rhs0 + wn * Int32(16 * FN), m, k, split)
    return
end

"""
    _atX_wmma_kernel!(out, words, tables, xh, xl, scale_inv, m, n, k,
                      split_word_rows, Val(BK), Val(BR))

Tensor-core `transpose(A)*X` kernel on the centered lookup values. A block
of 8 warps covers `BR` SNP columns x `BK` rhs columns over the word rows of
split `blockIdx().z`, `WMMA_T_BS` samples per stage; the `[sample, SNP]`
tiles load row major as `Vᵀ`.
"""
function _atX_wmma_kernel!(
    out::AbstractArray{Float32, 3}, words::AbstractMatrix{UInt32},
    tables::AbstractMatrix{UInt64}, xh::AbstractMatrix{UInt128},
    xl::AbstractMatrix{UInt128}, scale_inv::AbstractVector{Float32},
    m::Int32, n::Int32, k::Int32, split_word_rows::Int32, ::Val{BK},
    ::Val{BR},
) where {BK, BR}
    WM = BR ÷ 64
    WN = 8 ÷ WM
    FN = BK ÷ (16 * WN)
    LDS = WMMA_T_BS + 8
    L8 = LDS ÷ 8
    WR = WMMA_T_BS ÷ 16
    UNITS = WR * BR
    gh = CuDynamicSharedArray(UInt128, (L8, BR))
    gl = CuDynamicSharedArray(UInt128, (L8, BR), 16 * L8 * BR)
    bh = CuDynamicSharedArray(UInt128, (L8, BK, 2), 32 * L8 * BR)
    bl = CuDynamicSharedArray(UInt128, (L8, BK, 2),
        32 * L8 * BR + 32 * L8 * BK)
    scratch = CuDynamicSharedArray(Float32, (16, 16, 8),
        32 * L8 * BR + 64 * L8 * BK)

    tid = Int32(threadIdx().x) - Int32(1)
    warp = tid ÷ Int32(32)
    lane = tid % Int32(32)
    wm = warp % Int32(WM)
    wn = warp ÷ Int32(WM)
    snp0 = (Int32(blockIdx().x) - Int32(1)) * Int32(BR)
    rhs0 = (Int32(blockIdx().y) - Int32(1)) * Int32(BK)
    split = Int32(blockIdx().z)
    word_stop = min(split * split_word_rows, cld(m, Int32(16)))

    acc = ntuple(_ -> WMMA.fill_c(0.0f0, WMMA_CONFIG), Val(4 * FN))
    wr0 = (split - Int32(1)) * split_word_rows
    loaded = _load_words(words, tid, wr0, word_stop, Int32(WR), snp0,
        n, Val(UNITS))
    buffer = Int32(0)
    _copy_rhs_async!(bh, bl, xh, xl, tid, Int32(2) * wr0, rhs0, buffer,
        Val(BK))
    while wr0 < word_stop
        _store_words!(gh, gl, loaded, tables, tid, Int32(WR), snp0, n,
            Val(UNITS))
        _cp_async_wait()
        sync_threads()
        next = wr0 + Int32(WR)
        if next < word_stop
            loaded = _load_words(words, tid, next, word_stop,
                Int32(WR), snp0, n, Val(UNITS))
            _copy_rhs_async!(bh, bl, xh, xl, tid, Int32(2) * next, rhs0,
                Int32(1) - buffer, Val(BK))
        end
        stage = ntuple(_ -> WMMA.fill_c(0.0f0, WMMA_CONFIG), Val(4 * FN))
        kk = Int32(0)
        while kk < Int32(WMMA_T_BS)
            stage = _wmma_step(stage, gh, gl, bh, bl,
                wm * Int32(64 * LDS) + kk,
                kk + wn * Int32(16 * FN * LDS) + buffer * Int32(BK * LDS),
                Val(WMMA.RowMajor), Val(LDS), Val(LDS), Val(16 * LDS),
                Val(FN))
            kk += Int32(16)
        end
        acc = _fragment_add(acc, stage)
        sync_threads()
        buffer = Int32(1) - buffer
        wr0 = next
    end
    _wmma_store_all!(out, scratch, acc, scale_inv, warp, lane,
        snp0 + wm * Int32(64), rhs0 + wn * Int32(16 * FN), n, k, split)
    return
end

"""
    _uses_wmma(s::CuSnpArray, k) -> Bool

Return whether `s * X` with `k` rhs columns runs the tensor-core kernel
(Float32 and `k >= WMMA_MIN_RHS`).
"""
_uses_wmma(::CuSnpArray, ::Integer) = false
_uses_wmma(::CuSnpArray{Float32}, k::Integer) = k >= WMMA_MIN_RHS

"""
    _uses_wmma_t(s::CuSnpArray, k) -> Bool

Return whether `transpose(s) * X` with `k` rhs columns runs the
tensor-core kernel (Float32 and `k >= WMMA_T_MIN_RHS`).
"""
_uses_wmma_t(::CuSnpArray, ::Integer) = false
_uses_wmma_t(::CuSnpArray{Float32}, k::Integer) = k >= WMMA_T_MIN_RHS

"""
    _launch_wmma!(kernel_function, args, grid, shmem)

Compile `kernel_function` for `args`, allow `shmem` bytes of dynamic shared
memory, and launch it with 256 threads on `grid`.
"""
function _launch_wmma!(kernel_function::F, args::Tuple, grid::NTuple{3, Int},
    shmem::Int) where {F}
    kernel = @cuda launch=false kernel_function(args...)
    attributes(kernel.fun)[FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] =
        shmem
    kernel(args...; threads=256, blocks=grid, shmem)
    return nothing
end

"""
    _wmma_mul!(out::CuMatrix{Float32}, s::CuSnpArray{Float32},
               X::CuMatrix{Float32}) -> out

`out = s * X` on tensor cores. The centered lookup values `V` and `X` are
each split into high and low Float16 parts, `V X ≈ Vh Xh + Vh Xl + Vl Xh`,
and each shared-memory stage is summed into the Float32 accumulators on
CUDA cores.
"""
function _wmma_mul!(
    out::CuMatrix{Float32}, s::CuSnpArray{Float32}, X::CuMatrix{Float32},
)
    m, n = size(s)
    k = size(X, 2)
    BK = _wmma_rhs_width(k)
    tables = _split_tables(s.values)
    yh, yl, scale_inv = _split_rhs(X, WMMA_BN * cld(n, WMMA_BN),
        BK * cld(k, BK))
    BM = _wmma_block_rows(BK)
    row_blocks = cld(m, BM)
    rhs_blocks = cld(k, BK)
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    splits = clamp(cld(WMMA_BLOCKS_PER_SM * sms, row_blocks * rhs_blocks), 1,
        max(1, n ÷ (8 * WMMA_BN)))
    split_columns = WMMA_BN * cld(cld(n, splits), WMMA_BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        CuArray{Float32, 3}(undef, m, k, splits)
    G8 = (BM + 8) ÷ 8
    B8 = (WMMA_BN + 8) ÷ 8
    shmem = 32 * G8 * WMMA_BN + 64 * B8 * BK + 4 * 16 * 16 * 8
    _launch_wmma!(_aX_wmma_kernel!, (partials, s.data, tables, yh, yl,
        scale_inv, Int32(m), Int32(n), Int32(k), Int32(split_columns),
        Val(BK), Val(BM)), (row_blocks, rhs_blocks, splits), shmem)
    splits == 1 || _reduce_splits!(out, partials, splits)
    for a in (tables, yh, yl, scale_inv)
        unsafe_free!(a)
    end
    return out
end

"""
    _wmma_t_mul!(out::CuMatrix{Float32}, s::CuSnpArray{Float32},
                 X::CuMatrix{Float32}) -> out

`out = transpose(s) * X` on tensor cores, split and staged as in
`_wmma_mul!`.
"""
function _wmma_t_mul!(
    out::CuMatrix{Float32}, s::CuSnpArray{Float32}, X::CuMatrix{Float32},
)
    m, n = size(s)
    k = size(X, 2)
    BK = _wmma_rhs_width(k)
    tables = _split_tables(s.values)
    xh, xl, scale_inv = _split_rhs(X, WMMA_T_BS * cld(m, WMMA_T_BS),
        BK * cld(k, BK))
    BR = _wmma_block_rows(BK)
    row_blocks = cld(n, BR)
    rhs_blocks = cld(k, BK)
    word_rows = cld(m, 16)
    step = WMMA_T_BS ÷ 16
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    splits = clamp(cld(WMMA_BLOCKS_PER_SM * sms, row_blocks * rhs_blocks), 1,
        max(1, word_rows ÷ (8 * step)))
    split_word_rows = step * cld(cld(word_rows, splits), step)
    splits = cld(word_rows, split_word_rows)
    partials = splits == 1 ? reshape(out, n, k, 1) :
        CuArray{Float32, 3}(undef, n, k, splits)
    L8 = (WMMA_T_BS + 8) ÷ 8
    shmem = 32 * L8 * BR + 64 * L8 * BK + 4 * 16 * 16 * 8
    _launch_wmma!(_atX_wmma_kernel!, (partials, s.data, tables, xh, xl,
        scale_inv, Int32(m), Int32(n), Int32(k), Int32(split_word_rows),
        Val(BK), Val(BR)), (row_blocks, rhs_blocks, splits), shmem)
    splits == 1 || _reduce_splits!(out, partials, splits)
    for a in (tables, xh, xl, scale_inv)
        unsafe_free!(a)
    end
    return out
end
