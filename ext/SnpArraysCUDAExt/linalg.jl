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

function CuSnpArray{T}(s::SnpArray;
    model = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = false,
) where {T <: AbstractFloat}
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
    return CuSnpArray{T}(
        _upload_packed_words(s.data, s.m), s.m, model, center, scale,
        impute, CuArray(μ), CuArray(σinv), CuArray(values),
    )
end

"""
    _upload_packed_words(data::AbstractMatrix{UInt8}, m;
                         chunk_bytes=2^28) -> CuMatrix{UInt32}

`_packed_words(data, m)` built on the device in column chunks of about
`chunk_bytes`, so host memory stays bounded for genotype files larger than
RAM. Each chunk's columns are copied by all threads into one of two
page-locked staging buffers, zero padded and masked in place, and sent to
the device asynchronously while the other buffer fills.
"""
function _upload_packed_words(data::Matrix{UInt8}, m::Integer;
    chunk_bytes::Integer=2^28)
    word_rows = cld(m, 16)
    n = size(data, 2)
    words = CuMatrix{UInt32}(undef, word_rows, n)
    step = max(1, min(n, chunk_bytes ÷ (4 * word_rows)))
    # The driver allocates and frees the staging memory: `pin` on a
    # garbage-collected vector of this size uploaded stale words.
    memory = ntuple(_ -> alloc(HostMemory, 4 * word_rows * step), 2)
    try
        staging = map(host -> unsafe_wrap(Array, convert(Ptr{UInt32}, host),
            word_rows * step), memory)
        events = Union{Nothing, CuEvent}[nothing, nothing]
        for (index, first_column) in enumerate(1:step:n)
            slot = isodd(index) ? 1 : 2
            events[slot] === nothing || synchronize(events[slot])
            columns = first_column:min(first_column + step - 1, n)
            buffer = staging[slot]
            _stage_columns!(buffer, data, columns, m)
            copyto!(words, (first_column - 1) * word_rows + 1, buffer, 1,
                word_rows * length(columns))
            event = CuEvent()
            record(event)
            events[slot] = event
        end
        synchronize()
    finally
        foreach(free, memory)
    end
    return words
end

"""
    _stage_columns!(buffer::Vector{UInt32}, data::Matrix{UInt8}, columns, m)
        -> buffer

Copy the PLINK bytes of `columns` of `data` into consecutive
`cld(m, 16)`-word columns of `buffer` in parallel, zeroing the padding
bytes and the bits of samples past `m`.
"""
function _stage_columns!(buffer::Vector{UInt32}, data::Matrix{UInt8},
    columns::UnitRange{Int}, m::Integer)
    column_bytes = size(data, 1)
    padded_bytes = 4 * cld(m, 16)
    remainder = mod(m, 4)
    mask = UInt8((1 << (2 * remainder)) - 1)
    GC.@preserve buffer data begin
        base = Ptr{UInt8}(pointer(buffer))
        Threads.@threads for j in eachindex(columns)
            destination = base + (j - 1) * padded_bytes
            source = pointer(data, (columns[j] - 1) * column_bytes + 1)
            unsafe_copyto!(destination, source, column_bytes)
            for offset in column_bytes:(padded_bytes - 1)
                unsafe_store!(destination + offset, 0x00)
            end
            if remainder != 0
                last = destination + column_bytes - 1
                unsafe_store!(last, unsafe_load(last) & mask)
            end
        end
    end
    return buffer
end

function _upload_packed_words(data::AbstractMatrix{UInt8}, m::Integer;
    chunk_bytes::Integer=2^28)
    return _upload_packed_words(Matrix{UInt8}(data), m; chunk_bytes)
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
    _ax_chunks(row_blocks, n) -> Int

Number of SNP-column chunks so that the `A*x` grid covers about four waves
of resident blocks, with at least 64 columns per chunk.
"""
function _ax_chunks(row_blocks::Integer, n::Integer)
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    blocks_per_sm = 2048 ÷ THREADS_PER_BLOCK
    target = cld(4 * sms * blocks_per_sm, row_blocks)
    return clamp(target, 1, max(1, n ÷ 64))
end

"""
    LinearAlgebra.mul!(out::CuVector{T}, s::CuSnpArray{T}, v::CuVector{T})

In-place matrix-vector multiplication on a GPU. Asynchronous: synchronize
before reading `out` on the host.
"""
function mul!(
    out::CuVector{T}, s::CuSnpArray{T}, v::CuVector{T},
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
    _direct_mul!(out::CuVector{T}, s::CuSnpArray{T}, v::CuVector{T})

`out = s * v` with `_ax_direct_kernel!`.
"""
function _direct_mul!(
    out::CuVector{T}, s::CuSnpArray{T}, v::CuVector{T},
) where {T <: AbstractFloat}
    m, n = size(s)
    row_blocks = cld(cld(m, 16), THREADS_PER_BLOCK)
    chunks = _ax_chunks(row_blocks, n)
    chunk_columns = cld(n, chunks)
    chunks = cld(n, chunk_columns)
    # The partials come from CUDA.jl's stream-ordered pool and are freed in
    # stream order, so no per-call device allocation or synchronization.
    partials = chunks == 1 ? reshape(out, m, 1) :
        CuArray{T, 2}(undef, m, chunks)
    @cuda threads=THREADS_PER_BLOCK blocks=(row_blocks, chunks) (
        _ax_direct_kernel!(partials, s.data, v, s.values, Int32(m),
            Int32(n), Int32(chunk_columns))
    )
    if chunks != 1
        @cuda threads=THREADS_PER_BLOCK blocks=cld(m, THREADS_PER_BLOCK) (
            _sum_chunks_kernel!(out, partials, Int32(m), Int32(chunks))
        )
        unsafe_free!(partials)
    end
    return out
end

"""
    LinearAlgebra.mul!(out::CuVector{T}, st::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}}, v::CuVector{T})

In-place matrix-vector multiplication on a GPU, with transposed
CuSnpArray. Asynchronous: synchronize before reading `out` on the host.
"""
function mul!(
    out::CuVector{T},
    st::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}},
    v::CuVector{T},
) where {T <: AbstractFloat}
    s = parent(st)
    m, n = size(s)
    _check_mul(out, n, v, m)
    n == 0 && return out
    if m == 0
        fill!(out, zero(T))
        return out
    end
    blocks = cld(n, WARPS_PER_BLOCK * COLUMNS_PER_WARP)
    @cuda threads=32 * WARPS_PER_BLOCK blocks=blocks (
        _atx_warp_kernel!(out, s.data, v, s.values, Int32(m), Int32(n))
    )
    return out
end

"""`A*X` tiles `(BM, BN, BK, TM, TK)` for `k <= 8`, `k <= 32`, `k > 32`."""
const AX_TILES = ((256, 16, 8, 4, 4), (256, 16, 32, 8, 8),
    (128, 16, 64, 8, 8))

"""`transpose(A)*X` tiles `(BM, BN, BK, TN, TK)` by the same `k` bands."""
const ATX_TILES = ((64, 64, 8, 2, 2), (32, 128, 32, 4, 8),
    (32, 128, 64, 8, 8))

_tile_band(k::Integer) = k <= 8 ? 1 : k <= 32 ? 2 : 3

"""
    _splits(blocks, length_, step) -> Int

Number of reduction splits giving about eight blocks per SM from `blocks`
output tiles, keeping at least `8 * step` reduction entries per split.
"""
function _splits(blocks::Integer, length_::Integer, step::Integer)
    sms = attribute(device(), DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    target = cld(2 * 4 * sms, blocks)
    return clamp(target, 1, max(1, length_ ÷ (8 * step)))
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

Sum the `splits` slices of `partials` into `out` and free `partials`.
"""
function _reduce_splits!(
    out::CuMatrix{T}, partials::CuArray{T, 3}, splits::Integer,
) where {T}
    len = length(out)
    @cuda threads=THREADS_PER_BLOCK blocks=cld(len, THREADS_PER_BLOCK) (
        _sum_chunks_kernel!(reshape(out, len), reshape(partials, len, splits),
            Int32(len), Int32(splits))
    )
    unsafe_free!(partials)
    return out
end

"""
    LinearAlgebra.mul!(out::CuMatrix{T}, s::CuSnpArray{T}, X::CuMatrix{T})

In-place `s * X` on a GPU for any number of columns of `X`. Asynchronous.
"""
function mul!(
    out::CuMatrix{T}, s::CuSnpArray{T}, X::CuMatrix{T},
) where {T <: AbstractFloat}
    m, n = size(s)
    _check_matmul(out, m, X, n)
    k = size(X, 2)
    (m == 0 || k == 0) && return out
    if n == 0
        fill!(out, zero(T))
        return out
    end
    _uses_wmma(s, k) && return _wmma_mul!(out, s, X)
    return _tiled_mul!(out, s, X)
end

"""
    _tiled_mul!(out::CuMatrix{T}, s::CuSnpArray{T}, X::CuMatrix{T})

`out = s * X` with `_aX_tiled_kernel!`.
"""
function _tiled_mul!(
    out::CuMatrix{T}, s::CuSnpArray{T}, X::CuMatrix{T},
) where {T <: AbstractFloat}
    m, n = size(s)
    k = size(X, 2)
    k == 1 && return (_direct_mul!(vec(out), s, vec(X)); out)
    (BM, BN, BK, TM, TK) = AX_TILES[_tile_band(k)]
    row_blocks = cld(m, BM)
    rhs_blocks = cld(k, BK)
    splits = _splits(row_blocks * rhs_blocks, n, BN)
    split_columns = BN * cld(cld(n, splits), BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        CuArray{T, 3}(undef, m, k, splits)
    threads = (BM ÷ TM) * (BK ÷ TK)
    @cuda threads=threads blocks=(row_blocks, rhs_blocks, splits) (
        _aX_tiled_kernel!(partials, s.data, X, s.values, Int32(m), Int32(n),
            Int32(k), Int32(split_columns),
            Val(BM), Val(BN), Val(BK), Val(TM), Val(TK))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end

"""
    LinearAlgebra.mul!(out::CuMatrix{T}, st::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}}, X::CuMatrix{T})

In-place `transpose(s) * X` on a GPU for any number of columns of `X`.
Asynchronous.
"""
function mul!(
    out::CuMatrix{T},
    st::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}},
    X::CuMatrix{T},
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
    _uses_wmma_t(s, k) && return _wmma_t_mul!(out, s, X)
    (BM, BN, BK, TN, TK) = ATX_TILES[_tile_band(k)]
    # Halve the sample step until the tiles fit in 48 KiB of static
    # shared memory (needed for Float64).
    while ((BN + 1) * BM + BM * BK) * sizeof(T) > 48 * 1024 && BM > 16
        BM ÷= 2
    end
    column_blocks = cld(n, BN)
    rhs_blocks = cld(k, BK)
    word_rows = cld(m, 16)
    splits = _splits(column_blocks * rhs_blocks, word_rows, BM ÷ 16)
    split_word_rows = (BM ÷ 16) * cld(cld(word_rows, splits), BM ÷ 16)
    splits = cld(word_rows, split_word_rows)
    partials = splits == 1 ? reshape(out, n, k, 1) :
        CuArray{T, 3}(undef, n, k, splits)
    threads = (BN ÷ TN) * (BK ÷ TK)
    @cuda threads=threads blocks=(column_blocks, rhs_blocks, splits) (
        _atX_tiled_kernel!(partials, s.data, X, s.values, Int32(m), Int32(n),
            Int32(k), Int32(split_word_rows),
            Val(BM), Val(BN), Val(BK), Val(TN), Val(TK))
    )
    splits == 1 || _reduce_splits!(out, partials, splits)
    return out
end

"""
    LinearAlgebra.mul!(out::CuMatrix{T}, st::Transpose{T, <:CuSnpArray{T}}, X::CuMatrix{T}, cols::UnitRange{Int})

Compute `transpose(s)[cols, :] * X`, the product restricted to SNP columns
`cols`, into `out` of size `length(cols) x size(X, 2)`. Asynchronous.
"""
function mul!(
    out::CuMatrix{T}, st::Transpose{T, <:CuSnpArray{T}}, X::CuMatrix{T},
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
    block = CuSnpArray{T}(
        view(s.data, :, cols), s.m, s.model, s.center, s.scale, s.impute,
        view(s.μ, cols), view(s.σinv, cols), view(s.values, :, cols),
    )
    return mul!(out, transpose(block), X)
end

function *(s::CuSnpArray{T}, X::CuMatrix{T}) where {T <: AbstractFloat}
    return mul!(CuMatrix{T}(undef, size(s, 1), size(X, 2)), s, X)
end

function *(
    st::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}},
    X::CuMatrix{T},
) where {T <: AbstractFloat}
    return mul!(CuMatrix{T}(undef, size(st, 1), size(X, 2)), st, X)
end
