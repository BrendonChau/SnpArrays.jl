"""
    _raw_genotypes(model) -> NTuple{4, Int}

Raw genotype of PLINK codes 0x00..0x03 under `model`, with the missing
code 0x01 mapped to 0.
"""
function _raw_genotypes(model::Union{Val{1}, Val{2}, Val{3}})
    model == ADDITIVE_MODEL && return (0, 0, 1, 2)
    model == DOMINANT_MODEL && return (0, 0, 1, 1)
    return (0, 0, 0, 1)
end

"""
    _lookup_values(μ, σinv, model, center, scale, impute) -> Matrix

The 4 x n table of transformed values of each genotype code per column.
"""
function _lookup_values(
    μ::Vector{T}, σinv::Vector{T}, model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool, scale::Bool, impute::Bool,
) where {T <: AbstractFloat}
    raw = _raw_genotypes(model)
    values = Matrix{T}(undef, 4, length(μ))
    for j in eachindex(μ)
        shift = center ? μ[j] : zero(T)
        factor = scale ? σinv[j] : one(T)
        for code in 1:4
            g = code == 2 && impute ? μ[j] : T(raw[code])
            values[code, j] = (g - shift) * factor
        end
    end
    return values
end

function MtlSnpArray{T}(s::SnpArray;
    model = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = false,
) where {T <: AbstractFloat}
    T <: Float32 || throw(ArgumentError(
        "MtlSnpArray{$T} is not supported; use MtlSnpArray{Float32}",
    ))
    model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL) ||
        throw(ArgumentError("unrecognized model $model"))
    n = size(s, 2)
    μ = Vector{T}(undef, n)
    μ[:] = mean(s; dims=1, model=model)
    σinv = Vector{T}(undef, n)
    for j in 1:n
        σ = model == ADDITIVE_MODEL ? sqrt(μ[j] * (1 - μ[j] / 2)) :
            sqrt(μ[j] * (1 - μ[j]))
        σinv[j] = σ > 0 ? inv(σ) : one(T)
    end
    values = _lookup_values(μ, σinv, model, center, scale, impute)
    return MtlSnpArray{T}(
        MtlArray(_packed_words(s.data, s.m)), s.m, model, center, scale,
        impute, MtlArray(μ), MtlArray(σinv), MtlArray(values),
    )
end

"""
    _check_mul(out, rows, rhs, columns)

Throw a `DimensionMismatch` unless `length(out) == rows` and
`length(rhs) == columns`.
"""
function _check_mul(
    out::AbstractVector, rows::Integer, rhs::AbstractVector, columns::Integer,
)
    length(out) == rows || throw(DimensionMismatch(
        "output has length $(length(out)); expected $rows",
    ))
    length(rhs) == columns || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $columns",
    ))
    return nothing
end

"""
    _chunks(threads, length_; chunk_threads = CHUNK_THREADS,
            max_chunks = MAX_CHUNKS) -> Int

Number of reduction chunks so that a grid of `threads` threads per chunk
covers about `chunk_threads` threads, at most `max_chunks` and at most
`length_`.
"""
function _chunks(
    threads::Integer, length_::Integer;
    chunk_threads::Integer = CHUNK_THREADS, max_chunks::Integer = MAX_CHUNKS,
)
    return clamp(cld(chunk_threads, threads), 1, min(max_chunks,
        max(1, length_)))
end

"""
    LinearAlgebra.mul!(out::MtlVector{T}, s::MtlSnpArray{T}, v::MtlVector{T})

In-place matrix-vector multiplication on a Metal GPU. Asynchronous:
synchronize before reading `out` on the host.
"""
function mul!(
    out::MtlVector{T}, s::MtlSnpArray{T}, v::MtlVector{T},
) where {T <: AbstractFloat}
    m, n = size(s)
    _check_mul(out, m, v, n)
    m == 0 && return out
    if n == 0
        fill!(out, zero(T))
        return out
    end
    return _direct_mul!(out, s, v)
end

"""
    _direct_mul!(out::MtlVector{T}, s::MtlSnpArray{T}, v::MtlVector{T};
                 threads = THREADS_PER_GROUP, chunk_threads = CHUNK_THREADS,
                 max_chunks = MAX_CHUNKS)

`out = s * v` with `_ax_direct_kernel!`.
"""
function _direct_mul!(
    out::MtlVector{T}, s::MtlSnpArray{T}, v::MtlVector{T};
    threads::Integer = THREADS_PER_GROUP,
    chunk_threads::Integer = CHUNK_THREADS, max_chunks::Integer = MAX_CHUNKS,
) where {T <: AbstractFloat}
    m, n = size(s)
    word_rows = cld(m, 16)
    chunks = _chunks(word_rows, n; chunk_threads, max_chunks)
    chunk_columns = cld(n, chunks)
    chunks = cld(n, chunk_columns)
    partials = chunks == 1 ? reshape(out, m, 1) :
        MtlArray{T, 2}(undef, m, chunks)
    @metal threads=threads groups=(cld(word_rows, threads), chunks) (
        _ax_direct_kernel!(partials, s.data, v, s.values, Int32(m),
            Int32(n), Int32(chunk_columns))
    )
    chunks == 1 || _sum_chunks!(out, partials, chunks)
    return out
end

"""
    _sum_chunks!(out::MtlVector{T}, partials::MtlMatrix{T}, chunks)

Sum the `chunks` columns of `partials` into `out` with
`_sum_chunks_kernel!`.
"""
function _sum_chunks!(
    out::MtlVector{T}, partials::MtlMatrix{T}, chunks::Integer,
) where {T}
    rows = length(out)
    @metal threads=THREADS_PER_GROUP groups=cld(rows, THREADS_PER_GROUP) (
        _sum_chunks_kernel!(out, partials, Int32(rows), Int32(chunks))
    )
    return out
end

"""
    LinearAlgebra.mul!(out::MtlVector{T}, st::Union{Transpose{T, <:MtlSnpArray{T}}, Adjoint{T, <:MtlSnpArray{T}}}, v::MtlVector{T})

In-place matrix-vector multiplication on a Metal GPU, with transposed
MtlSnpArray. Asynchronous: synchronize before reading `out` on the host.
"""
function mul!(
    out::MtlVector{T},
    st::Union{Transpose{T, <:MtlSnpArray{T}}, Adjoint{T, <:MtlSnpArray{T}}},
    v::MtlVector{T},
) where {T <: AbstractFloat}
    s = parent(st)
    m, n = size(s)
    _check_mul(out, n, v, m)
    n == 0 && return out
    if m == 0
        fill!(out, zero(T))
        return out
    end
    return _direct_t_mul!(out, s, v)
end

"""
    _direct_t_mul!(out::MtlVector{T}, s::MtlSnpArray{T}, v::MtlVector{T};
                   threads = THREADS_PER_GROUP, chunk_threads = CHUNK_THREADS,
                   max_chunks = MAX_CHUNKS)

`out = transpose(s) * v` with `_atx_direct_kernel!`.
"""
function _direct_t_mul!(
    out::MtlVector{T}, s::MtlSnpArray{T}, v::MtlVector{T};
    threads::Integer = THREADS_PER_GROUP,
    chunk_threads::Integer = CHUNK_THREADS, max_chunks::Integer = MAX_CHUNKS,
) where {T <: AbstractFloat}
    m, n = size(s)
    word_rows = cld(m, 16)
    chunks = _chunks(n, word_rows; chunk_threads, max_chunks)
    chunk_word_rows = cld(word_rows, chunks)
    chunks = cld(word_rows, chunk_word_rows)
    partials = chunks == 1 ? reshape(out, n, 1) :
        MtlArray{T, 2}(undef, n, chunks)
    @metal threads=threads groups=(cld(n, threads), chunks) (
        _atx_direct_kernel!(partials, s.data, v, s.values, Int32(m),
            Int32(n), Int32(chunk_word_rows))
    )
    chunks == 1 || _sum_chunks!(out, partials, chunks)
    return out
end

"""`A*X` tiles `(BM, BN, BK, TM, TK)` for `k <= 8` and `k > 8`."""
const AX_TILES = ((256, 16, 8, 8, 1), (256, 16, 32, 8, 8))

"""`transpose(A)*X` tiles `(BM, BN, BK, TN, TK)` by the same `k` bands."""
const ATX_TILES = ((128, 32, 8, 1, 1), (64, 64, 32, 4, 4))

_tile_band(k::Integer) = k <= 8 ? 1 : 2

"""Target number of threadgroups when splitting the reduction."""
const SPLIT_GROUPS = 128

"""
    _splits(groups, length_, step; split_groups = SPLIT_GROUPS) -> Int

Number of reduction splits giving about `split_groups` threadgroups from
`groups` output tiles, keeping at least `8 * step` reduction entries per
split.
"""
function _splits(
    groups::Integer, length_::Integer, step::Integer;
    split_groups::Integer = SPLIT_GROUPS,
)
    return clamp(cld(split_groups, groups), 1, max(1, length_ ÷ (8 * step)))
end

"""
    _check_matmul(out, rows, rhs, columns)

Throw a `DimensionMismatch` unless `rhs` has `columns` rows and `out` is
`rows x size(rhs, 2)`.
"""
function _check_matmul(
    out::AbstractMatrix, rows::Integer, rhs::AbstractMatrix, columns::Integer,
)
    size(rhs, 1) == columns || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $columns",
    ))
    size(out) == (rows, size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((rows, size(rhs, 2)))",
    ))
    return nothing
end

"""
    _reduce_splits!(out, partials, splits)

Sum the `splits` slices of `partials` into `out`.
"""
function _reduce_splits!(
    out::MtlMatrix{T}, partials::MtlArray{T, 3}, splits::Integer,
) where {T}
    len = length(out)
    return _sum_chunks!(reshape(out, len), reshape(partials, len, splits),
        splits)
end

"""
    LinearAlgebra.mul!(out::MtlMatrix{T}, s::MtlSnpArray{T}, X::MtlMatrix{T})

In-place `s * X` on a Metal GPU for any number of columns of `X`.
Asynchronous.
"""
function mul!(
    out::MtlMatrix{T}, s::MtlSnpArray{T}, X::MtlMatrix{T},
) where {T <: AbstractFloat}
    m, n = size(s)
    _check_matmul(out, m, X, n)
    k = size(X, 2)
    (m == 0 || k == 0) && return out
    if n == 0
        fill!(out, zero(T))
        return out
    end
    _uses_simd(s, k) && return _simd_mul!(out, s, X)
    return _tiled_mul!(out, s, X)
end

"""
    _tiled_mul!(out::MtlMatrix{T}, s::MtlSnpArray{T}, X::MtlMatrix{T};
                tile = AX_TILES[_tile_band(size(X, 2))],
                split_groups = SPLIT_GROUPS)

`out = s * X` with `_aX_tiled_kernel!` on `tile = (BM, BN, BK, TM, TK)`.
"""
function _tiled_mul!(
    out::MtlMatrix{T}, s::MtlSnpArray{T}, X::MtlMatrix{T};
    tile::NTuple{5, Int} = AX_TILES[_tile_band(size(X, 2))],
    split_groups::Integer = SPLIT_GROUPS,
) where {T <: AbstractFloat}
    m, n = size(s)
    k = size(X, 2)
    k == 1 && return (_direct_mul!(vec(out), s, vec(X)); out)
    (BM, BN, BK, TM, TK) = tile
    row_blocks = cld(m, BM)
    rhs_blocks = cld(k, BK)
    splits = _splits(row_blocks * rhs_blocks, n, BN; split_groups)
    split_columns = BN * cld(cld(n, splits), BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        MtlArray{T, 3}(undef, m, k, splits)
    threads = (BM ÷ TM) * (BK ÷ TK)
    @metal threads=threads groups=(row_blocks, rhs_blocks, splits) (
        _aX_tiled_kernel!(partials, s.data, X, s.values, Int32(m), Int32(n),
            Int32(k), Int32(split_columns),
            Val(BM), Val(BN), Val(BK), Val(TM), Val(TK))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end

"""
    LinearAlgebra.mul!(out::MtlMatrix{T}, st::Union{Transpose{T, <:MtlSnpArray{T}}, Adjoint{T, <:MtlSnpArray{T}}}, X::MtlMatrix{T})

In-place `transpose(s) * X` on a Metal GPU for any number of columns of
`X`. Asynchronous.
"""
function mul!(
    out::MtlMatrix{T},
    st::Union{Transpose{T, <:MtlSnpArray{T}}, Adjoint{T, <:MtlSnpArray{T}}},
    X::MtlMatrix{T},
) where {T <: AbstractFloat}
    s = parent(st)
    m, n = size(s)
    _check_matmul(out, n, X, m)
    k = size(X, 2)
    (n == 0 || k == 0) && return out
    if m == 0
        fill!(out, zero(T))
        return out
    end
    k == 1 && return (mul!(vec(out), st, vec(X)); out)
    _uses_simd_t(s, k) && return _simd_t_mul!(out, s, X)
    return _tiled_t_mul!(out, s, X)
end

"""
    _tiled_t_mul!(out::MtlMatrix{T}, s::MtlSnpArray{T}, X::MtlMatrix{T};
                  tile = ATX_TILES[_tile_band(size(X, 2))],
                  split_groups = SPLIT_GROUPS)

`out = transpose(s) * X` with `_atX_tiled_kernel!` on
`tile = (BM, BN, BK, TN, TK)`.
"""
function _tiled_t_mul!(
    out::MtlMatrix{T}, s::MtlSnpArray{T}, X::MtlMatrix{T};
    tile::NTuple{5, Int} = ATX_TILES[_tile_band(size(X, 2))],
    split_groups::Integer = SPLIT_GROUPS,
) where {T <: AbstractFloat}
    m, n = size(s)
    k = size(X, 2)
    (BM, BN, BK, TN, TK) = tile
    column_blocks = cld(n, BN)
    rhs_blocks = cld(k, BK)
    word_rows = cld(m, 16)
    splits = _splits(column_blocks * rhs_blocks, word_rows, BM ÷ 16;
        split_groups)
    split_word_rows = (BM ÷ 16) * cld(cld(word_rows, splits), BM ÷ 16)
    splits = cld(word_rows, split_word_rows)
    partials = splits == 1 ? reshape(out, n, k, 1) :
        MtlArray{T, 3}(undef, n, k, splits)
    threads = (BN ÷ TN) * (BK ÷ TK)
    @metal threads=threads groups=(column_blocks, rhs_blocks, splits) (
        _atX_tiled_kernel!(partials, s.data, X, s.values, Int32(m), Int32(n),
            Int32(k), Int32(split_word_rows),
            Val(BM), Val(BN), Val(BK), Val(TN), Val(TK))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end

"""
    LinearAlgebra.mul!(out::MtlMatrix{T}, st::Transpose{T, <:MtlSnpArray{T}}, X::MtlMatrix{T}, cols::UnitRange{Int})

Compute `transpose(s)[cols, :] * X`, the product restricted to SNP columns
`cols`, into `out` of size `length(cols) x size(X, 2)`. Asynchronous.
"""
function mul!(
    out::MtlMatrix{T}, st::Transpose{T, <:MtlSnpArray{T}}, X::MtlMatrix{T},
    cols::UnitRange{Int},
) where {T <: AbstractFloat}
    s = parent(st)
    size(out, 1) == length(cols) || throw(DimensionMismatch(
        "output has $(size(out, 1)) rows; expected $(length(cols))",
    ))
    isempty(cols) || (first(cols) >= 1 && last(cols) <= size(s, 2)) ||
        throw(ArgumentError(
            "cols = $cols must be a subset of 1:$(size(s, 2))",
        ))
    block = MtlSnpArray{T}(
        view(s.data, :, cols), s.m, s.model, s.center, s.scale, s.impute,
        view(s.μ, cols), view(s.σinv, cols), view(s.values, :, cols),
    )
    return mul!(out, transpose(block), X)
end

function *(s::MtlSnpArray{T}, X::MtlMatrix{T}) where {T <: AbstractFloat}
    return mul!(MtlMatrix{T}(undef, size(s, 1), size(X, 2)), s, X)
end

function *(
    st::Union{Transpose{T, <:MtlSnpArray{T}}, Adjoint{T, <:MtlSnpArray{T}}},
    X::MtlMatrix{T},
) where {T <: AbstractFloat}
    return mul!(MtlMatrix{T}(undef, size(st, 1), size(X, 2)), st, X)
end
