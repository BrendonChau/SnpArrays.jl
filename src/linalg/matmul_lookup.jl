"""
    LOOKUP_ROW_TILE

Samples per inner tile of the lookup-table `A*X` kernel; the tile's
row-major partial sums (`LOOKUP_ROW_TILE` times the rhs slice width) stay
L1-resident
while every 4-SNP block of a group is gathered. Initial value from
`kq_pass.c`.
"""
const LOOKUP_ROW_TILE = 512

"""
    LOOKUP_CHUNK_SNPS

SNPs per chunk of the lookup-table `A*X` kernel (a multiple of 4). One
chunk's 256-row tables for all `k` rhs columns are built once, shared by
every gather task, and cost `LOOKUP_CHUNK_SNPS / 4 * 256 * k_padded *
sizeof(T)` bytes. Initial value from `kq_pass.c`.
"""
const LOOKUP_CHUNK_SNPS = 1024

"""
    LOOKUP_GROUP_BUDGET

Bytes of lookup tables swept per inner sample tile, so a group of 4-SNP
blocks stays L2-resident across the tile. Initial value from `kq_pass.c`.
"""
const LOOKUP_GROUP_BUDGET = 1 << 20

"""
    LOOKUP_MIN_ROWS

Fewest samples for which `A*X` uses the lookup-table kernel; below it the
`256 * k` table build per 4-SNP block is not amortised.
"""
const LOOKUP_MIN_ROWS = 2048

"""
    LOOKUP_MIN_RHS

Fewest rhs columns for which `A*X` uses the lookup-table kernel.
"""
const LOOKUP_MIN_RHS = 4

"""
    _uses_lookup_kernel(m::Int, k::Int) -> Bool

Return whether `A*X` with `m` samples and `k` rhs columns runs the
lookup-table kernel rather than the register-tiled kernel.
"""
function _uses_lookup_kernel(m::Int, k::Int)
    return m >= LOOKUP_MIN_ROWS && k >= LOOKUP_MIN_RHS
end

"""
    _lookup_gather4(x::UInt32, s::Int) -> UInt8

Return the byte holding the four 2-bit codes of sample `s` (0 to 3) from
`x`, whose bytes are the packed bytes of four consecutive SNP columns for
the same four samples.
"""
@inline function _lookup_gather4(x::UInt32, s::Int)
    y = (x >>> (2s)) & 0x03030303
    y = (y | (y >>> 6)) & 0x000f000f
    y = (y | (y >>> 12)) & 0x000000ff
    return y % UInt8
end

"""
    _lookup_transpose!(blk, row_span, packed, row_first, row_last,
        column_first, nblk)

Write, for samples `row_first:row_last` (`row_first ≡ 1 (mod 4)`) and the
`nblk` blocks of four SNP columns starting at `column_first`, one byte per
(block, sample) holding that sample's four codes, at
`blk[(b - 1) * row_span + row]`. Columns past `size(packed, 2)` read as
code 0.
"""
function _lookup_transpose!(
    blk::Vector{UInt8},
    row_span::Int,
    packed::Matrix{UInt8},
    row_first::Int,
    row_last::Int,
    column_first::Int,
    nblk::Int,
)
    n = size(packed, 2)
    byte_first = ((row_first - 1) >>> 2) + 1
    byte_last = ((row_last - 1) >>> 2) + 1
    @inbounds for b in 1:nblk
        j0 = column_first + 4(b - 1)
        base = (b - 1) * row_span + row_first - 1
        for q in byte_first:byte_last
            x = UInt32(packed[q, j0])
            j0 + 1 <= n && (x |= UInt32(packed[q, j0 + 1]) << 8)
            j0 + 2 <= n && (x |= UInt32(packed[q, j0 + 2]) << 16)
            j0 + 3 <= n && (x |= UInt32(packed[q, j0 + 3]) << 24)
            position = base + 4(q - byte_first)
            blk[position + 1] = _lookup_gather4(x, 0)
            blk[position + 2] = _lookup_gather4(x, 1)
            blk[position + 3] = _lookup_gather4(x, 2)
            blk[position + 4] = _lookup_gather4(x, 3)
        end
    end
    return blk
end

"""
    _lookup_weights(values, column, n) -> NTuple{4, T}

Return the four genotype values of SNP `column`, or zeros past `n`.
"""
@inline function _lookup_weights(
    values::Matrix{T},
    column::Int,
    n::Int,
) where T <: SIMD_FLOAT
    column <= n || return (zero(T), zero(T), zero(T), zero(T))
    return @inbounds (values[1, column], values[2, column],
                      values[3, column], values[4, column])
end

"""
    _lookup_build_tables!(tables, stage, stage_offset, n, rhs, values,
        column_first, block_first, block_last, k_padded, slice_stride,
        ::Val{W}, ::Val{NV})

Fill, for blocks `block_first:block_last` of the chunk starting at SNP
`column_first` and every rhs slice `s` of width `S = NV * W`, the 256 table
rows `tables[s * slice_stride + ((b - 1) * 256 + code) * S + t] =
Σ_l values[code_l + 1, j_l] * rhs[j_l, s * S + t]` over the four SNPs `j_l`
of block `b`, zero past `size(rhs, 2)`. `stage` holds the four rhs rows,
`4 * k_padded` elements from `stage_offset`.
"""
function _lookup_build_tables!(
    tables::Vector{T},
    stage::Vector{T},
    stage_offset::Int,
    n::Int,
    rhs::StridedMatrix{T},
    values::Matrix{T},
    column_first::Int,
    block_first::Int,
    block_last::Int,
    k_padded::Int,
    slice_stride::Int,
    ::Val{W},
    ::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice = NV * W
    k = size(rhs, 2)
    @inbounds for b in block_first:block_last
        j0 = column_first + 4(b - 1)
        for l in 0:3
            j = j0 + l
            offset = stage_offset + l * k_padded
            for t in 1:k
                stage[offset + t] = j <= n ? rhs[j, t] : zero(T)
            end
            for t in (k + 1):k_padded
                stage[offset + t] = zero(T)
            end
        end
        w0 = _lookup_weights(values, j0, n)
        w1 = _lookup_weights(values, j0 + 1, n)
        w2 = _lookup_weights(values, j0 + 2, n)
        w3 = _lookup_weights(values, j0 + 3, n)
        table_base = (b - 1) * 256 * slice
        for code in 0:255
            a0 = Vec{W, T}(w0[(code & 3) + 1])
            a1 = Vec{W, T}(w1[((code >>> 2) & 3) + 1])
            a2 = Vec{W, T}(w2[((code >>> 4) & 3) + 1])
            a3 = Vec{W, T}(w3[((code >>> 6) & 3) + 1])
            for s0 in 0:slice:(k_padded - 1)
                row_base = (s0 ÷ slice) * slice_stride + table_base +
                           code * slice - s0
                for v in (s0 + 1):W:(s0 + slice)
                    u0 = stage[VecRange{W}(stage_offset + v)]
                    u1 = stage[VecRange{W}(stage_offset + k_padded + v)]
                    u2 = stage[VecRange{W}(stage_offset + 2k_padded + v)]
                    u3 = stage[VecRange{W}(stage_offset + 3k_padded + v)]
                    tables[VecRange{W}(row_base + v)] = muladd(
                        a3, u3, muladd(a2, u2, muladd(a1, u1, a0 * u0)))
                end
            end
        end
    end
    return tables
end

"""
    _lookup_load(x, offset, ::Val{W}, ::Val{NV}) -> NTuple{NV, Vec{W, T}}

Load the `NV` consecutive `W`-vectors of `x` starting after `offset`.
"""
@inline function _lookup_load(
    x::Vector{T},
    offset::Int,
    ::Val{W},
    ::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    return ntuple(j -> @inbounds(x[VecRange{W}(offset + (j - 1) * W + 1)]),
                  Val(NV))
end

"""
    _lookup_store!(x, offset, vectors::NTuple{NV, Vec{W, T}})

Store `vectors` into consecutive `W`-vectors of `x` starting after `offset`.
"""
@inline function _lookup_store!(
    x::Vector{T},
    offset::Int,
    vectors::NTuple{NV, Vec{W, T}},
) where {T <: SIMD_FLOAT, W, NV}
    @inbounds for j in 1:NV
        x[VecRange{W}(offset + (j - 1) * W + 1)] = vectors[j]
    end
    return x
end

"""
    _lookup_add(accumulators::NTuple{NV, Vec{W, T}}, x, offset)

Return `accumulators` plus the `NV` consecutive `W`-vectors of `x` starting
after `offset`.
"""
@inline function _lookup_add(
    accumulators::NTuple{NV, Vec{W, T}},
    x::Vector{T},
    offset::Int,
) where {T <: SIMD_FLOAT, W, NV}
    return ntuple(
        j -> accumulators[j] +
             @inbounds(x[VecRange{W}(offset + (j - 1) * W + 1)]),
        Val(NV),
    )
end

"""
    _lookup_gather_tile!(tile, tile_offset, tables, blk, row_span, tile_first,
        tile_rows, nblk, group, slice_offset, ::Val{W}, ::Val{NV})

Set the row-major `tile_rows x NV * W` tile at `tile_offset` to the sum over
the `nblk` blocks of the table rows selected by `blk`, for the rhs slice
whose tables start after `slice_offset`, sweeping `group` blocks per pass.
"""
function _lookup_gather_tile!(
    tile::Vector{T},
    tile_offset::Int,
    tables::Vector{T},
    blk::Vector{UInt8},
    row_span::Int,
    tile_first::Int,
    tile_rows::Int,
    nblk::Int,
    group::Int,
    slice_offset::Int,
    width::Val{W},
    count::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice = NV * W
    @inbounds for i in 1:(tile_rows * slice)
        tile[tile_offset + i] = zero(T)
    end
    @inbounds for g0 in 1:group:nblk
        g1 = min(g0 + group - 1, nblk)
        for i in 1:tile_rows
            tile_row = tile_offset + (i - 1) * slice
            accumulators = _lookup_load(tile, tile_row, width, count)
            sample = tile_first + i - 1
            for b in g0:g1
                code = Int(blk[(b - 1) * row_span + sample])
                r = ((b - 1) * 256 + code) * slice + slice_offset
                accumulators = _lookup_add(accumulators, tables, r)
            end
            _lookup_store!(tile, tile_row, accumulators)
        end
    end
    return tile
end

"""
    _lookup_flush_tile!(out, tile, tile_offset, tile_first, tile_rows,
        rhs_column, valid, lanes)

Add the leading `valid` lanes of each row of the `tile_rows x lanes` tile
into `out[tile_first:tile_first + tile_rows - 1, rhs_column:rhs_column +
valid - 1]`.
"""
function _lookup_flush_tile!(
    out::StridedMatrix{T},
    tile::Vector{T},
    tile_offset::Int,
    tile_first::Int,
    tile_rows::Int,
    rhs_column::Int,
    valid::Int,
    lanes::Int,
) where T <: SIMD_FLOAT
    @inbounds for t in 1:valid
        column = rhs_column + t - 1
        for i in 1:tile_rows
            out[tile_first + i - 1, column] +=
                tile[tile_offset + (i - 1) * lanes + t]
        end
    end
    return out
end

"""
    _snparray_AX_lookup_task!(out, packed, tables, tile, tile_offset, blk,
        row_span, row_first, row_last, column_first, nblk, slice_stride,
        group, ::Val{W}, ::Val{NV})

Run one gather task of the lookup-table `A*X` kernel: transpose the codes
of samples `row_first:row_last` for the chunk of `nblk` blocks starting at
SNP `column_first`, then for every `NV * W`-wide rhs slice (tables
`slice_stride` elements apart) and `LOOKUP_ROW_TILE`-row tile, gather the
table rows and add them into `out`.
"""
function _snparray_AX_lookup_task!(
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    tables::Vector{T},
    tile::Vector{T},
    tile_offset::Int,
    blk::Vector{UInt8},
    row_span::Int,
    row_first::Int,
    row_last::Int,
    column_first::Int,
    nblk::Int,
    slice_stride::Int,
    group::Int,
    width::Val{W},
    count::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice = NV * W
    k = size(out, 2)
    _lookup_transpose!(blk, row_span, packed, row_first, row_last,
                       column_first, nblk)
    for rhs_column in 1:slice:k
        valid = min(slice, k - rhs_column + 1)
        slice_offset = ((rhs_column - 1) ÷ slice) * slice_stride
        for tile_first in row_first:LOOKUP_ROW_TILE:row_last
            tile_rows = min(LOOKUP_ROW_TILE, row_last - tile_first + 1)
            _lookup_gather_tile!(
                tile, tile_offset, tables, blk, row_span, tile_first,
                tile_rows, nblk, group, slice_offset, width, count,
            )
            _lookup_flush_tile!(
                out, tile, tile_offset, tile_first, tile_rows, rhs_column,
                valid, slice,
            )
        end
    end
    return out
end
