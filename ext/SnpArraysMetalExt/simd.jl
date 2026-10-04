"""
`A*X` simdgroup tile `(BN, WM, WN, TM, TN)` for `k > 32`: `BM = 8 * TM *
WM` samples x `BK = 8 * TN * WN` rhs columns per threadgroup.
"""
const AX_SIMD_TILE = (16, 4, 2, 4, 4)

"""
`transpose(A)*X` simdgroup tile `(BM, WM, WN, TM, TN)` for `k > 8`:
`BN = 8 * TM * WM` SNP columns x `BK = 8 * TN * WN` rhs columns per
threadgroup.
"""
const ATX_SIMD_TILE = (32, 4, 2, 4, 4)

"""Fewest rhs columns for which `A*X` uses the simdgroup kernel."""
const SIMD_MIN_RHS = 33

"""
Fewest rhs columns for which `transpose(A)*X` uses the simdgroup kernel.
"""
const SIMD_T_MIN_RHS = 9

"""
    _simd_afrags(tile, row_origin, ks, Val(TM))

The `TM` 8x8 fragments of `tile` stacked down from zero-based
`(row_origin, ks)`.
"""
@inline function _simd_afrags(tile, row_origin::Int32, ks::Int32,
    ::Val{TM}) where {TM}
    return ntuple(Val(TM)) do ti
        simdgroup_load(tile, (Int(row_origin) + (ti - 1) * 8 + 1,
            Int(ks) + 1))
    end
end

"""
    _simd_bfrags(tile, column_origin, ks, Val(TN))

The `TN` 8x8 fragments of `tile` side by side from zero-based
`(ks, column_origin)`.
"""
@inline function _simd_bfrags(tile, column_origin::Int32, ks::Int32,
    ::Val{TN}) where {TN}
    return ntuple(Val(TN)) do tj
        simdgroup_load(tile, (Int(ks) + 1,
            Int(column_origin) + (tj - 1) * 8 + 1))
    end
end

"""
    _simd_mma(acc, afrags, bfrags, Val(TM), Val(TN))

Return the `TM x TN` accumulator fragments `acc` plus `afrags * bfrags`.
"""
@inline function _simd_mma(acc, afrags, bfrags, ::Val{TM},
    ::Val{TN}) where {TM, TN}
    return ntuple(Val(TM * TN)) do idx
        ti = (idx - 1) % TM + 1
        tj = (idx - 1) ÷ TM + 1
        simdgroup_multiply_accumulate(afrags[ti], bfrags[tj], acc[idx])
    end
end

"""
    _simd_store_tile!(out, scratch, acc, row_first, column_first,
                      column_offset, scratch_first, lane, rows, columns,
                      Val(TM), Val(TN))

Write a simdgroup's `TM x TN` accumulator fragments at zero-based
`(row_first, column_first)` of the `rows x columns` block of `out` that
starts after column `column_offset`. Edge fragments go through `scratch`
with bounds checks.
"""
@inline function _simd_store_tile!(
    out, scratch, acc, row_first::Int32, column_first::Int32,
    column_offset::Int32, scratch_first::Int32, lane::Int32, rows::Int32,
    columns::Int32, ::Val{TM}, ::Val{TN},
) where {TM, TN}
    ntuple(Val(TM * TN)) do idx
        ti = (idx - 1) % TM
        tj = (idx - 1) ÷ TM
        row = row_first + Int32(8) * Int32(ti)
        column = column_first + Int32(8) * Int32(tj)
        if row + Int32(8) <= rows && column + Int32(8) <= columns
            simdgroup_store(acc[idx], out,
                (Int(row) + 1, Int(column_offset + column) + 1))
        else
            simdgroup_store(acc[idx], scratch, (1, Int(scratch_first) + 1))
            simdgroup_barrier(MemoryFlagThreadGroup)
            e = lane
            while e < Int32(64)
                r = e % Int32(8)
                c = e ÷ Int32(8)
                if row + r < rows && column + c < columns
                    @inbounds out[row + r + Int32(1),
                        column_offset + column + c + Int32(1)] =
                        scratch[r + Int32(1), scratch_first + c + Int32(1)]
                end
                e += Int32(32)
            end
            simdgroup_barrier(MemoryFlagThreadGroup)
        end
        nothing
    end
    return
end

"""
    _aX_simd_kernel!(out, words, rhs, values, m, n, k, split_columns,
                     Val(BN), Val(WM), Val(WN), Val(TM), Val(TN))

Simdgroup-matrix `A*X` kernel. A threadgroup holds a `WM x WN` grid of
simdgroups over `BM` samples x `BK` rhs columns and reduces over the SNP
columns of split `_group().z` in steps of `BN`, writing the
`m x k` block `split` of `out` (`m x (k * splits)`).
"""
function _aX_simd_kernel!(
    out::AbstractMatrix{T}, words::AbstractMatrix{UInt32},
    rhs::AbstractMatrix{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
    k::Int32, split_columns::Int32,
    ::Val{BN}, ::Val{WM}, ::Val{WN}, ::Val{TM}, ::Val{TN},
) where {T, BN, WM, WN, TM, TN}
    BM = 8 * TM * WM
    BK = 8 * TN * WN
    bm = Int32(BM)
    bn = Int32(BN)
    bk = Int32(BK)
    threads = Int32(WM * WN * 32)
    thread = Int32(thread_index_in_threadgroup())
    simdgroup = Int32(simdgroup_index_in_threadgroup()) - Int32(1)
    lane = Int32(thread_index_in_simdgroup()) - Int32(1)
    simd_row_first = (simdgroup % Int32(WM)) * Int32(8 * TM)
    simd_column_first = (simdgroup ÷ Int32(WM)) * Int32(8 * TN)

    group = _group()
    sample_first = (Int32(group.x) - Int32(1)) * bm
    rhs_column_first = (Int32(group.y) - Int32(1)) * bk
    split = Int32(group.z)
    column_stop = min(split * split_columns, n)

    word_rows_per_tile = bm ÷ Int32(16)
    total_word_rows = cld(m, Int32(16))
    group_word_row_first = (Int32(group.x) - Int32(1)) * word_rows_per_tile

    # The 4-element padding staggers rows across threadgroup memory banks.
    genotype_tile = MtlThreadGroupArray(T, (BM + 4, BN))
    x_tile = MtlThreadGroupArray(T, (BN + 4, BK))
    scratch = MtlThreadGroupArray(T, (8 + 4, 8 * WM * WN))

    acc = ntuple(_ -> ntuple(_ -> VecElement{T}(zero(T)), Val(64)),
        Val(TM * TN))

    column_first = (split - Int32(1)) * split_columns
    while column_first < column_stop
        wi = thread
        while wi <= word_rows_per_tile * bn
            col_offset = mod(wi - Int32(1), bn) + Int32(1)
            word_row = div(wi - Int32(1), bn) + Int32(1)
            column = column_first + col_offset
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
            column = column_first + col_offset
            rhs_column = rhs_column_first + rhs_offset
            value = (column <= column_stop && rhs_column <= k) ?
                (@inbounds rhs[column, rhs_column]) : zero(T)
            @inbounds x_tile[col_offset, rhs_offset] = value
            xi += threads
        end

        threadgroup_barrier(MemoryFlagThreadGroup)
        ks = Int32(0)
        while ks < bn
            afrags = _simd_afrags(genotype_tile, simd_row_first, ks, Val(TM))
            bfrags = _simd_bfrags(x_tile, simd_column_first, ks, Val(TN))
            acc = _simd_mma(acc, afrags, bfrags, Val(TM), Val(TN))
            ks += Int32(8)
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        column_first += bn
    end

    _simd_store_tile!(
        out, scratch, acc, sample_first + simd_row_first,
        rhs_column_first + simd_column_first, (split - Int32(1)) * k,
        simdgroup * Int32(8), lane, m, k, Val(TM), Val(TN),
    )
    return
end

"""
    _atX_simd_kernel!(out, words, rhs, values, m, n, k, split_samples,
                      Val(BM), Val(WM), Val(WN), Val(TM), Val(TN))

Simdgroup-matrix `transpose(A)*X` kernel. A threadgroup holds a `WM x WN`
grid of simdgroups over `BN` SNP columns x `BK` rhs columns and reduces
over the samples of split `_group().z` in steps of `BM`, writing the
`n x k` block `split` of `out` (`n x (k * splits)`).
"""
function _atX_simd_kernel!(
    out::AbstractMatrix{T}, words::AbstractMatrix{UInt32},
    rhs::AbstractMatrix{T}, values::AbstractMatrix{T}, m::Int32, n::Int32,
    k::Int32, split_samples::Int32,
    ::Val{BM}, ::Val{WM}, ::Val{WN}, ::Val{TM}, ::Val{TN},
) where {T, BM, WM, WN, TM, TN}
    BN = 8 * TM * WM
    BK = 8 * TN * WN
    bm = Int32(BM)
    bn = Int32(BN)
    bk = Int32(BK)
    threads = Int32(WM * WN * 32)
    thread = Int32(thread_index_in_threadgroup())
    simdgroup = Int32(simdgroup_index_in_threadgroup()) - Int32(1)
    lane = Int32(thread_index_in_simdgroup()) - Int32(1)
    simd_row_first = (simdgroup % Int32(WM)) * Int32(8 * TM)
    simd_column_first = (simdgroup ÷ Int32(WM)) * Int32(8 * TN)

    group = _group()
    column_first = (Int32(group.x) - Int32(1)) * bn
    rhs_column_first = (Int32(group.y) - Int32(1)) * bk
    split = Int32(group.z)
    sample_stop = min(split * split_samples, m)

    word_rows_per_tile = bm ÷ Int32(16)
    word_row_stop = cld(sample_stop, Int32(16))

    genotype_tile = MtlThreadGroupArray(T, (BN + 4, BM))
    x_tile = MtlThreadGroupArray(T, (BM + 4, BK))
    scratch = MtlThreadGroupArray(T, (8 + 4, 8 * WM * WN))

    acc = ntuple(_ -> ntuple(_ -> VecElement{T}(zero(T)), Val(64)),
        Val(TM * TN))

    sample_first = (split - Int32(1)) * split_samples
    while sample_first < sample_stop
        group_sample_word_first = sample_first ÷ Int32(16)
        wi = thread
        while wi <= word_rows_per_tile * bn
            col_offset = mod(wi - Int32(1), bn) + Int32(1)
            word_row = div(wi - Int32(1), bn) + Int32(1)
            column = column_first + col_offset
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
            sample = sample_first + sample_offset
            rhs_column = rhs_column_first + rhs_offset
            value = (sample <= sample_stop && rhs_column <= k) ?
                (@inbounds rhs[sample, rhs_column]) : zero(T)
            @inbounds x_tile[sample_offset, rhs_offset] = value
            xi += threads
        end

        threadgroup_barrier(MemoryFlagThreadGroup)
        ks = Int32(0)
        while ks < bm
            afrags = _simd_afrags(genotype_tile, simd_row_first, ks, Val(TM))
            bfrags = _simd_bfrags(x_tile, simd_column_first, ks, Val(TN))
            acc = _simd_mma(acc, afrags, bfrags, Val(TM), Val(TN))
            ks += Int32(8)
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        sample_first += bm
    end

    _simd_store_tile!(
        out, scratch, acc, column_first + simd_row_first,
        rhs_column_first + simd_column_first, (split - Int32(1)) * k,
        simdgroup * Int32(8), lane, n, k, Val(TM), Val(TN),
    )
    return
end

"""
    _uses_simd(s::MtlSnpArray, k) -> Bool

Return whether `s * X` with `k` rhs columns runs the simdgroup-matrix kernel
(Float32 and `k >= SIMD_MIN_RHS`).
"""
_uses_simd(::MtlSnpArray, ::Integer) = false
_uses_simd(::MtlSnpArray{Float32}, k::Integer) = k >= SIMD_MIN_RHS

"""
    _uses_simd_t(s::MtlSnpArray, k) -> Bool

Return whether `transpose(s) * X` with `k` rhs columns runs the
simdgroup-matrix kernel (Float32 and `k >= SIMD_T_MIN_RHS`).
"""
_uses_simd_t(::MtlSnpArray, ::Integer) = false
_uses_simd_t(::MtlSnpArray{Float32}, k::Integer) = k >= SIMD_T_MIN_RHS

"""
    _simd_mul!(out::MtlMatrix{Float32}, s::MtlSnpArray{Float32},
               X::MtlMatrix{Float32}; tile = AX_SIMD_TILE,
               split_groups = SPLIT_GROUPS) -> out

`out = s * X` with `_aX_simd_kernel!` on `tile = (BN, WM, WN, TM, TN)`.
"""
function _simd_mul!(
    out::MtlMatrix{Float32}, s::MtlSnpArray{Float32}, X::MtlMatrix{Float32};
    tile::NTuple{5, Int} = AX_SIMD_TILE, split_groups::Integer = SPLIT_GROUPS,
)
    m, n = size(s)
    k = size(X, 2)
    (BN, WM, WN, TM, TN) = tile
    BM = 8 * TM * WM
    BK = 8 * TN * WN
    row_blocks = cld(m, BM)
    rhs_blocks = cld(k, BK)
    splits = _splits(row_blocks * rhs_blocks, n, BN; split_groups)
    split_columns = BN * cld(cld(n, splits), BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        MtlArray{Float32, 3}(undef, m, k, splits)
    @metal threads=WM * WN * 32 groups=(row_blocks, rhs_blocks, splits) (
        _aX_simd_kernel!(reshape(partials, m, k * splits), s.data, X,
            s.values, Int32(m), Int32(n), Int32(k), Int32(split_columns),
            Val(BN), Val(WM), Val(WN), Val(TM), Val(TN))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end

"""
    _simd_t_mul!(out::MtlMatrix{Float32}, s::MtlSnpArray{Float32},
                 X::MtlMatrix{Float32}; tile = ATX_SIMD_TILE,
                 split_groups = SPLIT_GROUPS) -> out

`out = transpose(s) * X` with `_atX_simd_kernel!` on
`tile = (BM, WM, WN, TM, TN)`.
"""
function _simd_t_mul!(
    out::MtlMatrix{Float32}, s::MtlSnpArray{Float32}, X::MtlMatrix{Float32};
    tile::NTuple{5, Int} = ATX_SIMD_TILE, split_groups::Integer = SPLIT_GROUPS,
)
    m, n = size(s)
    k = size(X, 2)
    (BM, WM, WN, TM, TN) = tile
    BN = 8 * TM * WM
    BK = 8 * TN * WN
    column_blocks = cld(n, BN)
    rhs_blocks = cld(k, BK)
    splits = _splits(column_blocks * rhs_blocks, m, BM; split_groups)
    split_samples = BM * cld(cld(m, splits), BM)
    splits = cld(m, split_samples)
    partials = splits == 1 ? reshape(out, n, k, 1) :
        MtlArray{Float32, 3}(undef, n, k, splits)
    @metal threads=WM * WN * 32 groups=(column_blocks, rhs_blocks, splits) (
        _atX_simd_kernel!(reshape(partials, n, k * splits), s.data, X,
            s.values, Int32(m), Int32(n), Int32(k), Int32(split_samples),
            Val(BM), Val(WM), Val(WN), Val(TM), Val(TN))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end
