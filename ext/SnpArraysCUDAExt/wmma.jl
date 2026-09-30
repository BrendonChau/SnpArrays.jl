"""WMMA shape and accumulator type of the tensor-core `A*X` kernel."""
const WMMA_CONFIG = WMMA.Config{16, 16, 16, Float32}

"""Samples per block of the tensor-core `A*X` kernel (4 warps x 32)."""
const WMMA_BM = 128

"""Rhs columns per block of the tensor-core `A*X` kernel (2 warps x 16)."""
const WMMA_BK = 32

"""SNP columns per shared-memory stage of the tensor-core `A*X` kernel."""
const WMMA_BN = 64

"""Fewest rhs columns for which Float32 `A*X` uses the tensor-core kernel."""
const WMMA_MIN_RHS = 17

"""
    _raw_count_table(model) -> (NTuple{4, Float16}, Int)

Raw genotype count of codes 0x00..0x03 under `model` (missing 0x01 counts
0) and the count of code 0x03.
"""
function _raw_count_table(model::Union{Val{1}, Val{2}, Val{3}})
    model == ADDITIVE_MODEL && return (Float16.((0, 0, 1, 2)), 2)
    model == DOMINANT_MODEL && return (Float16.((0, 0, 1, 1)), 1)
    return (Float16.((0, 0, 0, 1)), 1)
end

"""
    _wmma_step(acc, a1, a2, bh, bl, offset, Val(LDB))

Accumulate the high and low rhs fragments of one 16-SNP step into the
warp's two accumulators.
"""
@inline function _wmma_step(acc, a1, a2, bh, bl, offset::Int32,
    ::Val{LDB}) where {LDB}
    h = WMMA.load_b(pointer(bh, offset), LDB, WMMA.ColMajor, WMMA_CONFIG)
    l = WMMA.load_b(pointer(bl, offset), LDB, WMMA.ColMajor, WMMA_CONFIG)
    c1 = WMMA.mma(a1, l, WMMA.mma(a1, h, acc[1], WMMA_CONFIG), WMMA_CONFIG)
    c2 = WMMA.mma(a2, l, WMMA.mma(a2, h, acc[2], WMMA_CONFIG), WMMA_CONFIG)
    return (c1, c2)
end

"""
    _wmma_store!(out, scratch, frag, scale_inv, r, warp, lane, row0, col0,
                 m, k, split)

Write one 16 x 16 accumulator fragment at zero-based origin `(row0, col0)`
of `out[:, :, split]`, unscaled, adding `r` in the first split.
"""
@inline function _wmma_store!(out, scratch, frag, scale_inv, r, warp::Int32,
    lane::Int32, row0::Int32, col0::Int32, m::Int32, k::Int32, split::Int32)
    WMMA.store_d(pointer(scratch, Int32(256) * warp + Int32(1)), frag, 16,
        WMMA.ColMajor, WMMA_CONFIG)
    e = lane
    while e < Int32(256)
        rr = e % Int32(16)
        cc = e ÷ Int32(16)
        row = row0 + rr + Int32(1)
        col = col0 + cc + Int32(1)
        if row <= m && col <= k
            v = @inbounds scratch[rr + Int32(1), cc + Int32(1),
                warp + Int32(1)] * scale_inv[col]
            if split == Int32(1)
                v += @inbounds r[col]
            end
            @inbounds out[row, col, split] = v
        end
        e += Int32(32)
    end
    return nothing
end

"""
    _aX_wmma_kernel!(out, words, counts, yh, yl, scale_inv, r, m, n, k,
                     split_columns)

Tensor-core `A*X` kernel. A block of 8 warps covers `WMMA_BM` samples x
`WMMA_BK` rhs columns over the SNP columns of split `blockIdx().z`,
decoding raw genotype counts to Float16 in shared memory `WMMA_BN` SNPs at
a time; `out[:, :, split]` is `G * (yh + yl) .* scale_inv`, plus `r` in the
first split.
"""
function _aX_wmma_kernel!(
    out::AbstractArray{Float32, 3}, words::AbstractMatrix{UInt32},
    counts::NTuple{4, Float16}, yh::AbstractMatrix{Float16},
    yl::AbstractMatrix{Float16}, scale_inv::AbstractVector{Float32},
    r::AbstractVector{Float32}, m::Int32, n::Int32, k::Int32,
    split_columns::Int32,
)
    BN = WMMA_BN
    LDG = WMMA_BM + 8
    LDB = BN + 8
    gs = CuStaticSharedArray(Float16, (LDG, BN))
    bh = CuStaticSharedArray(Float16, (LDB, WMMA_BK))
    bl = CuStaticSharedArray(Float16, (LDB, WMMA_BK))
    scratch = CuStaticSharedArray(Float32, (16, 16, 8))

    tid = Int32(threadIdx().x) - Int32(1)
    warp = tid ÷ Int32(32)
    lane = tid % Int32(32)
    wm = warp % Int32(4)
    wn = warp ÷ Int32(4)
    sample0 = (Int32(blockIdx().x) - Int32(1)) * Int32(WMMA_BM)
    rhs0 = (Int32(blockIdx().y) - Int32(1)) * Int32(WMMA_BK)
    split = Int32(blockIdx().z)
    column_stop = min(split * split_columns, n)
    word_row0 = sample0 ÷ Int32(16)
    word_rows = cld(m, Int32(16))

    acc = (WMMA.fill_c(0.0f0, WMMA_CONFIG), WMMA.fill_c(0.0f0, WMMA_CONFIG))
    column0 = (split - Int32(1)) * split_columns
    while column0 < column_stop
        u = tid
        while u < Int32((WMMA_BM ÷ 16) * BN)
            wr = u % Int32(WMMA_BM ÷ 16)
            j = u ÷ Int32(WMMA_BM ÷ 16)
            column = column0 + j + Int32(1)
            word_row = word_row0 + wr + Int32(1)
            w = (column <= column_stop) & (word_row <= word_rows) ?
                (@inbounds words[word_row, column]) : UInt32(0)
            base = Int32(16) * wr
            Base.Cartesian.@nexprs 16 s -> begin
                code = (w >> (Int32(2) * Int32(s - 1))) & UInt32(3)
                odd = code & UInt32(1) != UInt32(0)
                low = ifelse(odd, counts[2], counts[1])
                high = ifelse(odd, counts[4], counts[3])
                @inbounds gs[base + Int32(s), j + Int32(1)] =
                    ifelse(code & UInt32(2) == UInt32(0), low, high)
            end
            u += Int32(256)
        end
        u = tid
        while u < Int32(BN * WMMA_BK)
            jj = u % Int32(BN)
            tt = u ÷ Int32(BN)
            column = column0 + jj + Int32(1)
            live = column <= column_stop
            @inbounds bh[jj + Int32(1), tt + Int32(1)] =
                live ? yh[column, rhs0 + tt + Int32(1)] : Float16(0)
            @inbounds bl[jj + Int32(1), tt + Int32(1)] =
                live ? yl[column, rhs0 + tt + Int32(1)] : Float16(0)
            u += Int32(256)
        end
        sync_threads()
        kk = Int32(0)
        while kk < Int32(BN)
            a1 = WMMA.load_a(pointer(gs, wm * Int32(32) + kk * Int32(LDG) +
                Int32(1)), LDG, WMMA.ColMajor, WMMA_CONFIG)
            a2 = WMMA.load_a(pointer(gs, wm * Int32(32) + Int32(16) +
                kk * Int32(LDG) + Int32(1)), LDG, WMMA.ColMajor, WMMA_CONFIG)
            acc = _wmma_step(acc, a1, a2, bh, bl,
                kk + wn * Int32(16 * LDB) + Int32(1), Val(LDB))
            kk += Int32(16)
        end
        sync_threads()
        column0 += Int32(BN)
    end

    row0 = sample0 + wm * Int32(32)
    col0 = rhs0 + wn * Int32(16)
    _wmma_store!(out, scratch, acc[1], scale_inv, r, warp, lane, row0, col0,
        m, k, split)
    _wmma_store!(out, scratch, acc[2], scale_inv, r, warp, lane,
        row0 + Int32(16), col0, m, k, split)
    return
end

"""
    _uses_wmma(s::CuSnpArray, k) -> Bool

Return whether Float32 `s * X` with `k` rhs columns runs the tensor-core
kernel: `k >= WMMA_MIN_RHS` and no missing genotype differs from code 0x00
(true unless `impute=true` and a column has missing genotypes).
"""
_uses_wmma(::CuSnpArray, ::Integer) = false

function _uses_wmma(s::CuSnpArray{Float32}, k::Integer)
    k >= WMMA_MIN_RHS || return false
    v = s.values
    return !any(view(v, 2, :) .!= view(v, 1, :))
end

"""
    _wmma_mul!(out::CuMatrix{Float32}, s::CuSnpArray{Float32},
               X::CuMatrix{Float32}) -> out

`out = s * X` on tensor cores. Each lookup value is `α + β g` in the raw
count `g`, so `s * X = 1 (αᵀ X) + G (β .* X)`; `β .* X` is scaled per
column by a power of two and split into high and low Float16 parts, and the
Float32 accumulators recover about 22 significant bits.
"""
function _wmma_mul!(
    out::CuMatrix{Float32}, s::CuSnpArray{Float32}, X::CuMatrix{Float32},
)
    m, n = size(s)
    k = size(X, 2)
    counts, top = _raw_count_table(s.model)
    v = s.values
    α = v[1, :]
    β = (v[4, :] .- v[1, :]) ./ Float32(top)
    k_padded = WMMA_BK * cld(k, WMMA_BK)
    y = CuArray{Float32}(undef, n, k_padded)
    fill!(view(y, :, (k + 1):k_padded), 0.0f0)
    view(y, :, 1:k) .= β .* X
    scale = vec(maximum(abs, y; dims=1))
    scale .= ifelse.(scale .> 0,
        exp2.(14 .- Float32.(exponent.(max.(scale, floatmin(Float32))))),
        1.0f0)
    y .*= transpose(scale)
    yh = Float16.(y)
    yl = Float16.(y .- Float32.(yh))
    scale .= inv.(scale)
    r = vec(sum(α .* X; dims=1))

    row_blocks = cld(m, WMMA_BM)
    rhs_blocks = cld(k, WMMA_BK)
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    splits = clamp(cld(4 * sms, row_blocks * rhs_blocks), 1,
        max(1, n ÷ (8 * WMMA_BN)))
    split_columns = WMMA_BN * cld(cld(n, splits), WMMA_BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        CuArray{Float32, 3}(undef, m, k, splits)
    @cuda threads=256 blocks=(row_blocks, rhs_blocks, splits) (
        _aX_wmma_kernel!(partials, s.data, counts, yh, yl, scale, r,
            Int32(m), Int32(n), Int32(k), Int32(split_columns))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    for a in (α, β, y, scale, yh, yl, r)
        unsafe_free!(a)
    end
    return out
end
