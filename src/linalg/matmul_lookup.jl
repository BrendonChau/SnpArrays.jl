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
    _lookup_transpose!(blk, row_span, packed, rows, column_first, nblk)

Write, for samples `rows` (`first(rows) ≡ 1 (mod 4)`) and the `nblk` blocks
of four SNP columns starting at `column_first`, one byte per (block, sample)
holding that sample's four codes, at `blk[(b - 1) * row_span + row]`.
Columns past `size(packed, 2)` read as code 0.
"""
function _lookup_transpose!(
    blk::Vector{UInt8},
    row_span::Int,
    packed::Matrix{UInt8},
    rows::UnitRange{Int},
    column_first::Int,
    nblk::Int,
)
    n = size(packed, 2)
    row_first = first(rows)
    row_last = last(rows)
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
    _lookup_build_tables!(workspace, stage_offset, n, rhs, values,
        column_first, blocks, layout, ::Val{W}, ::Val{NV})

Fill, for `blocks` of the chunk starting at SNP `column_first` and every rhs
slice `s` of width `S = NV * W`, the 256 table rows `workspace[s *
slice_stride + ((b - 1) * 256 + code) * S + t] = Σ_l values[code_l + 1, j_l]
* rhs[j_l, s * S + t]` over the four SNPs `j_l` of block `b`, zero past
`size(rhs, 2)` or SNP `n`. The four rhs rows are staged in `workspace` from
`stage_offset`.
"""
function _lookup_build_tables!(
    workspace::Vector{T},
    stage_offset::Int,
    n::Int,
    rhs::StridedMatrix{T},
    values::Matrix{T},
    column_first::Int,
    blocks::UnitRange{Int},
    layout::LookupLayout,
    ::Val{W},
    ::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    tables = workspace
    stage = workspace
    k_padded = layout.k_padded
    slice_stride = layout.slice_stride
    slice = NV * W
    k = size(rhs, 2)
    @inbounds for b in blocks
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
    _lookup_gather_tile!(workspace, tile_offset, blk, tile_rows, nblk,
        slice_offset, layout, ::Val{W}, ::Val{NV})

Set the row-major `length(tile_rows) x NV * W` tile at `tile_offset` to the
sum over the `nblk` blocks of the table rows selected by `blk` for samples
`tile_rows`, for the rhs slice whose tables start after `slice_offset`,
sweeping `layout.group` blocks per pass.
"""
function _lookup_gather_tile!(
    workspace::Vector{T},
    tile_offset::Int,
    blk::Vector{UInt8},
    tile_rows::UnitRange{Int},
    nblk::Int,
    slice_offset::Int,
    layout::LookupLayout,
    width::Val{W},
    count::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    tile = workspace
    tables = workspace
    row_span = layout.row_span
    group = layout.group
    tile_first = first(tile_rows)
    nrows = length(tile_rows)
    slice = NV * W
    @inbounds for i in 1:(nrows * slice)
        tile[tile_offset + i] = zero(T)
    end
    @inbounds for g0 in 1:group:nblk
        g1 = min(g0 + group - 1, nblk)
        for i in 1:nrows
            tile_row = tile_offset + (i - 1) * slice
            accumulators = _load_vectors(tile, tile_row, count, width)
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
    _lookup_flush_tile!(out, workspace, tile_offset, tile_rows, tile_columns,
        lanes)

Add the leading `length(tile_columns)` lanes of each row of the
`length(tile_rows) x lanes` tile at `tile_offset` into
`out[tile_rows, tile_columns]`.
"""
function _lookup_flush_tile!(
    out::StridedMatrix{T},
    workspace::Vector{T},
    tile_offset::Int,
    tile_rows::UnitRange{Int},
    tile_columns::UnitRange{Int},
    lanes::Int,
) where T <: SIMD_FLOAT
    tile = workspace
    tile_first = first(tile_rows)
    nrows = length(tile_rows)
    rhs_column = first(tile_columns)
    @inbounds for t in 1:length(tile_columns)
        column = rhs_column + t - 1
        for i in 1:nrows
            out[tile_first + i - 1, column] +=
                tile[tile_offset + (i - 1) * lanes + t]
        end
    end
    return out
end

"""
    _snparray_AX_lookup_gather_task!(out, packed, workspace, tile_offset, blk,
        rows, column_first, nblk, layout, ::Val{W}, ::Val{NV})

Run one gather task of the lookup-table `A*X` kernel: transpose the codes
of samples `rows` for the chunk of `nblk` blocks starting at SNP
`column_first`, then for every `NV * W`-wide rhs slice and
`LOOKUP_ROW_TILE`-row tile, gather the table rows and add them into `out`.
"""
function _snparray_AX_lookup_gather_task!(
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    workspace::Vector{T},
    tile_offset::Int,
    blk::Vector{UInt8},
    rows::UnitRange{Int},
    column_first::Int,
    nblk::Int,
    layout::LookupLayout,
    width::Val{W},
    count::Val{NV},
) where {T <: SIMD_FLOAT, W, NV}
    slice_stride = layout.slice_stride
    row_last = last(rows)
    slice = NV * W
    k = size(out, 2)
    _lookup_transpose!(blk, layout.row_span, packed, rows, column_first, nblk)
    for rhs_column in 1:slice:k
        tile_columns = rhs_column:min(rhs_column + slice - 1, k)
        slice_offset = ((rhs_column - 1) ÷ slice) * slice_stride
        for tile_first in first(rows):LOOKUP_ROW_TILE:row_last
            tile_rows =
                tile_first:min(tile_first + LOOKUP_ROW_TILE - 1, row_last)
            _lookup_gather_tile!(
                workspace, tile_offset, blk, tile_rows, nblk, slice_offset,
                layout, width, count,
            )
            _lookup_flush_tile!(
                out, workspace, tile_offset, tile_rows, tile_columns, slice,
            )
        end
    end
    return out
end
