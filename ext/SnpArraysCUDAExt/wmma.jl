"""WMMA shape and accumulator type of the tensor-core `A*X` kernel."""
const WMMA_CONFIG = WMMA.Config{16, 16, 16, Float32}

"""Samples per block of the tensor-core `A*X` kernel (4 warps x 64)."""
const WMMA_BM = 256

"""SNP columns per shared-memory stage of the tensor-core `A*X` kernel."""
const WMMA_BN = 32

"""Fewest rhs columns for which Float32 `A*X` uses the tensor-core kernel."""
const WMMA_MIN_RHS = 16

"""Target resident blocks per SM when splitting the SNP reduction."""
const WMMA_BLOCKS_PER_SM = 16

"""
    _wmma_rhs_width(k) -> Int

Rhs columns per block for `k` columns: 32 up to `k = 32`, else 64.
"""
_wmma_rhs_width(k::Integer) = k <= 32 ? 32 : 64

"""
    _plane_weights(model) -> (UInt64, UInt64)

Float16 bit patterns `c1`, `c2` such that the raw count under `model` of a
sample with high bit `hi` and low bit `lo` is `c1 * hi + c2 * (hi & lo)`
(0x3c00 is 1.0 and 0x3c00 + 0x0400 is 2.0).
"""
function _plane_weights(model::Union{Val{1}, Val{2}, Val{3}})
    model == ADDITIVE_MODEL && return (UInt64(0x3c00), UInt64(0x0400))
    model == DOMINANT_MODEL && return (UInt64(0x3c00), UInt64(0))
    return (UInt64(0), UInt64(0x3c00))
end

"""
    _half_pointer(p, offset)

Float16 pointer `offset` halves past `p`.
"""
@inline function _half_pointer(p::Core.LLVMPtr{T, A},
    offset::Int32) where {T, A}
    return reinterpret(Core.LLVMPtr{Float16, A}, p) + 2 * Int(offset)
end

"""
    _spread4(plane::UInt32, q::Int32) -> UInt64

Move the plane bits of samples `4q .. 4q + 3` (bits `8q + 2i`) to bits
`16i`; the multiplier's four shifts land every other product bit on a
distinct position, so no carries occur.
"""
@inline function _spread4(plane::UInt32, q::Int32)
    t = UInt64((plane >> (Int32(8) * q)) & UInt32(0x55))
    return (t * UInt64(0x0000_0400_1000_4001)) &
        UInt64(0x0001_0001_0001_0001)
end

"""
    _decode_halves(w::UInt32, c1::UInt64, c2::UInt64) -> NTuple{2, UInt128}

The raw counts of the 16 samples of packed word `w` as Float16 bit
patterns, 8 per `UInt128`.
"""
@inline function _decode_halves(w::UInt32, c1::UInt64, c2::UInt64)
    lo = w & UInt32(0x55555555)
    hi = (w >> 1) & UInt32(0x55555555)
    both = hi & lo
    l1 = _spread4(hi, Int32(0)) * c1 + _spread4(both, Int32(0)) * c2
    l2 = _spread4(hi, Int32(1)) * c1 + _spread4(both, Int32(1)) * c2
    l3 = _spread4(hi, Int32(2)) * c1 + _spread4(both, Int32(2)) * c2
    l4 = _spread4(hi, Int32(3)) * c1 + _spread4(both, Int32(3)) * c2
    return (UInt128(l1) | (UInt128(l2) << 64),
        UInt128(l3) | (UInt128(l4) << 64))
end

"""
    _wmma_step(acc, gs, bh, bl, a_offset, b_offset, Val(LDG), Val(LDB),
               Val(FN))

Accumulate one 16-SNP step into a warp's `4 x FN` fragments: 4 genotype
fragments times `FN` high and low rhs fragments.
"""
@generated function _wmma_step(acc, gs, bh, bl, a_offset::Int32,
    b_offset::Int32, ::Val{LDG}, ::Val{LDB}, ::Val{FN}) where {LDG, LDB, FN}
    body = Expr[]
    for i in 1:4
        push!(body, :($(Symbol(:a, i)) = WMMA.load_a(
            _half_pointer(pointer(gs), a_offset + Int32($(16 * (i - 1)))),
            $LDG, WMMA.ColMajor, WMMA_CONFIG)))
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
            a = Symbol(:a, i)
            c = Symbol(:c, f)
            push!(body, :($c = WMMA.mma($a, $l,
                WMMA.mma($a, $h, acc[$f], WMMA_CONFIG), WMMA_CONFIG)))
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
    _wmma_store_all!(out, scratch, acc, scale_inv, r, warp, lane, row0,
                     col0, m, k, split)

Write a warp's `4 x FN` accumulator fragments, unrolled.
"""
@generated function _wmma_store_all!(out, scratch, acc::NTuple{L},
    scale_inv, r, warp::Int32, lane::Int32, row0::Int32, col0::Int32,
    m::Int32, k::Int32, split::Int32) where {L}
    stores = [:(_wmma_store!(out, scratch, acc[$f], scale_inv, r, warp, lane,
        row0 + Int32($(16 * ((f - 1) % 4))),
        col0 + Int32($(16 * ((f - 1) ÷ 4))), m, k, split)) for f in 1:L]
    return quote
        Base.@_inline_meta
        $(stores...)
        return nothing
    end
end

"""
    _aX_wmma_kernel!(out, words, c1, c2, yh, yl, scale_inv, r, m, n, k,
                     split_columns, Val(BK))

Tensor-core `A*X` kernel. A block of 8 warps covers `WMMA_BM` samples x
`BK` rhs columns over the SNP columns of split `blockIdx().z`, decoding raw
counts to Float16 in shared memory `WMMA_BN` SNPs at a time; `yh` and `yl`
hold the zero-padded high and low Float16 parts of the scaled rhs, 8 per
`UInt128`. `out[:, :, split]` is `G * (yh + yl) .* scale_inv`, plus `r` in
the first split.
"""
function _aX_wmma_kernel!(
    out::AbstractArray{Float32, 3}, words::AbstractMatrix{UInt32},
    c1::UInt64, c2::UInt64, yh::AbstractMatrix{UInt128},
    yl::AbstractMatrix{UInt128}, scale_inv::AbstractVector{Float32},
    r::AbstractVector{Float32}, m::Int32, n::Int32, k::Int32,
    split_columns::Int32, ::Val{BK},
) where {BK}
    FN = BK ÷ 32
    LDG = WMMA_BM + 8
    LDB = WMMA_BN + 8
    gs = CuStaticSharedArray(UInt128, (LDG ÷ 8, WMMA_BN))
    bh = CuStaticSharedArray(UInt128, (LDB ÷ 8, BK))
    bl = CuStaticSharedArray(UInt128, (LDB ÷ 8, BK))
    scratch = CuStaticSharedArray(Float32, (16, 16, 8))

    tid = Int32(threadIdx().x) - Int32(1)
    warp = tid ÷ Int32(32)
    lane = tid % Int32(32)
    wm = warp % Int32(4)
    wn = warp ÷ Int32(4)
    sample0 = (Int32(blockIdx().x) - Int32(1)) * Int32(WMMA_BM)
    rhs0 = (Int32(blockIdx().y) - Int32(1)) * Int32(BK)
    split = Int32(blockIdx().z)
    column_stop = min(split * split_columns, n)
    word_row0 = sample0 ÷ Int32(16)
    word_rows = cld(m, Int32(16))

    acc = ntuple(_ -> WMMA.fill_c(0.0f0, WMMA_CONFIG), Val(4 * FN))
    column0 = (split - Int32(1)) * split_columns
    while column0 < column_stop
        u = tid
        while u < Int32((WMMA_BM ÷ 16) * WMMA_BN)
            wr = u % Int32(WMMA_BM ÷ 16)
            j = u ÷ Int32(WMMA_BM ÷ 16)
            column = column0 + j + Int32(1)
            word_row = word_row0 + wr + Int32(1)
            w = (column <= column_stop) & (word_row <= word_rows) ?
                (@inbounds words[word_row, column]) : UInt32(0)
            halves = _decode_halves(w, c1, c2)
            @inbounds gs[Int32(2) * wr + Int32(1), j + Int32(1)] = halves[1]
            @inbounds gs[Int32(2) * wr + Int32(2), j + Int32(1)] = halves[2]
            u += Int32(256)
        end
        u = tid
        while u < Int32((WMMA_BN ÷ 8) * BK)
            jj = u % Int32(WMMA_BN ÷ 8)
            tt = u ÷ Int32(WMMA_BN ÷ 8)
            source = column0 ÷ Int32(8) + jj + Int32(1)
            @inbounds bh[jj + Int32(1), tt + Int32(1)] =
                yh[source, rhs0 + tt + Int32(1)]
            @inbounds bl[jj + Int32(1), tt + Int32(1)] =
                yl[source, rhs0 + tt + Int32(1)]
            u += Int32(256)
        end
        sync_threads()
        kk = Int32(0)
        while kk < Int32(WMMA_BN)
            acc = _wmma_step(acc, gs, bh, bl,
                wm * Int32(64) + kk * Int32(LDG),
                kk + wn * Int32(16 * FN * LDB), Val(LDG), Val(LDB), Val(FN))
            kk += Int32(16)
        end
        sync_threads()
        column0 += Int32(WMMA_BN)
    end
    _wmma_store_all!(out, scratch, acc, scale_inv, r, warp, lane,
        sample0 + wm * Int32(64), rhs0 + wn * Int32(16 * FN), m, k, split)
    return
end

"""
    _uses_wmma(s::CuSnpArray, k) -> Bool

Return whether `s * X` with `k` rhs columns runs the tensor-core kernel:
Float32, `k >= WMMA_MIN_RHS`, and no missing genotype differs from code
0x00 (true unless `impute=true` and a column has missing genotypes).
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
    BK = _wmma_rhs_width(k)
    c1, c2 = _plane_weights(s.model)
    top = s.model == ADDITIVE_MODEL ? 2 : 1
    v = s.values
    α = v[1, :]
    β = (v[4, :] .- v[1, :]) ./ Float32(top)
    k_padded = BK * cld(k, BK)
    n_padded = WMMA_BN * cld(n, WMMA_BN)
    y = CuArray{Float32}(undef, n_padded, k_padded)
    fill!(y, 0.0f0)
    view(y, 1:n, 1:k) .= β .* X
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
    rhs_blocks = cld(k, BK)
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    splits = clamp(cld(WMMA_BLOCKS_PER_SM * sms, row_blocks * rhs_blocks), 1,
        max(1, n ÷ (8 * WMMA_BN)))
    split_columns = WMMA_BN * cld(cld(n, splits), WMMA_BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        CuArray{Float32, 3}(undef, m, k, splits)
    @cuda threads=256 blocks=(row_blocks, rhs_blocks, splits) (
        _aX_wmma_kernel!(partials, s.data, c1, c2,
            reinterpret(UInt128, yh), reinterpret(UInt128, yl), scale, r,
            Int32(m), Int32(n), Int32(k), Int32(split_columns), Val(BK))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    for a in (α, β, y, scale, yh, yl, r)
        unsafe_free!(a)
    end
    return out
end
