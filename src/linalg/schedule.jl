function _snparray_ax_tile!(out, packed, rhs, values, rows_filled)
    n = size(packed, 2)
    row_step, column_step, _ =
        _tile_sizes(eltype(out), rows_filled, n, 1, :forward; vector=true)
    @assert row_step % DECODE_WIDTH == 0 "row_step must be a multiple of DECODE_WIDTH"
    @sync begin
        for row_first in 1:row_step:rows_filled
            row_last = min(row_first + row_step - 1, rows_filled)
            @assert (row_first - 1) % 4 == 0 "row_first must be ≡ 1 (mod 4)"
            Threads.@spawn begin
                for column_first in 1:column_step:n
                    column_last = min(column_first + column_step - 1, n)
                    _snparray_ax_kernel!(
                        out, packed, rhs, values, $row_first, $row_last,
                        column_first, column_last,
                    )
                end
            end
        end
    end
    return out
end

"""
    _resize_workspace!(workspace, count) -> workspace

Resize the shared panel workspace to exactly `count` elements. Repeated
products at one shape reuse the buffer; a change of shape reallocates.
"""
function _resize_workspace!(workspace::Vector, count::Int)
    length(workspace) == count || resize!(workspace, count)
    return workspace
end

function _snparray_AX_tile!(out, packed, rhs, values, rows_filled, workspace,
                            blk)
    n = size(packed, 2)
    k = size(out, 2)
    T = eltype(out)
    if _uses_lookup_kernel(rows_filled, k) && T <: SIMD_FLOAT &&
       out isa StridedMatrix{T} && rhs isa StridedMatrix{T} &&
       packed isa Matrix{UInt8}
        return _snparray_AX_lookup_tile!(out, packed, rhs, values,
                                         rows_filled, workspace, blk)
    end
    row_step, column_step, rhs_step =
        _tile_sizes(T, rows_filled, n, k, :forward; vector=false)
    lanes = _rhs_width(T, k)
    width = Val(lanes)
    @assert row_step % DECODE_WIDTH == 0 "row_step must be a multiple of DECODE_WIDTH"
    # One panel slice per spawned task, so concurrent tasks never overlap.
    panel_length = 2lanes * column_step
    tasks = cld(rows_filled, row_step) * cld(k, rhs_step)
    _resize_workspace!(workspace, tasks * panel_length)
    task_index = 0
    @sync begin
        for rhs_first in 1:rhs_step:k
            rhs_last = min(rhs_first + rhs_step - 1, k)
            for row_first in 1:row_step:rows_filled
                row_last = min(row_first + row_step - 1, rows_filled)
                @assert (row_first - 1) % 4 == 0 "row_first must be ≡ 1 (mod 4)"
                panel_offset = task_index * panel_length
                task_index += 1
                Threads.@spawn _snparray_AX_kernel!(
                    out, packed, rhs, values, workspace, $panel_offset,
                    $row_first, $row_last, $column_step, $rhs_first,
                    $rhs_last, $width,
                )
            end
        end
    end
    return out
end

"""
    _snparray_AX_lookup_tile!(out, packed, rhs, values, rows_filled,
        workspace, blk)

Accumulate `out += A * rhs` with the lookup-table kernel: per chunk of
`LOOKUP_CHUNK_SNPS` SNPs, build the 256-row tables of every 4-SNP block
for all rhs columns (tasks over blocks), then gather them per sample
(tasks over sample blocks). Tables are laid out `[slice][block][code][lane]`
over rhs slices of `NV` vectors of width `W = _vector_width(T)`, with
`NV = 2` for `k ≤ 2W` and `NV = max(2, 32 ÷ W)` otherwise; `k` is padded
to a whole slice. `workspace` holds the tables, one rhs staging
slice per build task, and one output tile per gather task; `blk` holds the
transposed codes of one chunk.
"""
function _snparray_AX_lookup_tile!(out, packed, rhs, values, rows_filled,
                                   workspace, blk)
    n = size(packed, 2)
    k = size(out, 2)
    T = eltype(out)
    lanes = _vector_width(T)
    vectors = k <= 2lanes ? 2 : max(2, 32 ÷ lanes)
    slice = vectors * lanes
    k_padded = cld(k, slice) * slice
    nblk_max = min(LOOKUP_CHUNK_SNPS ÷ 4, cld(n, 4))
    row_span = 4 * cld(rows_filled, 4)
    slice_stride = nblk_max * 256 * slice
    tables_len = nblk_max * 256 * k_padded
    stage_len = 4k_padded
    tile_len = LOOKUP_ROW_TILE * slice
    row_step = _task_axis_step(rows_filled, rows_filled)
    row_tasks = cld(rows_filled, row_step)
    block_step = max(1, cld(nblk_max, TASKS_PER_THREAD * Threads.nthreads()))
    block_tasks = cld(nblk_max, block_step)
    stage_base = tables_len
    tile_base = stage_base + block_tasks * stage_len
    _resize_workspace!(workspace, tile_base + row_tasks * tile_len)
    _resize_workspace!(blk, nblk_max * row_span)
    group = clamp(LOOKUP_GROUP_BUDGET ÷ (256 * slice * sizeof(T)), 1,
                  nblk_max)
    @assert row_step % 4 == 0 "row_step must be a multiple of 4"
    _snparray_AX_lookup_chunks!(
        out, packed, rhs, values, rows_filled, workspace, blk, k_padded,
        slice_stride, stage_len, tile_len, row_span, row_step, block_step,
        stage_base, tile_base, group, Val(lanes), Val(vectors),
    )
    return out
end

"""
    _snparray_AX_lookup_chunks!(out, packed, rhs, values, rows_filled,
        workspace, blk, k_padded, slice_stride, stage_len, tile_len,
        row_span, row_step, block_step, stage_base, tile_base, group,
        ::Val{W}, ::Val{NV})

Run the build and gather phases of `_snparray_AX_lookup_tile!` for every
SNP chunk with rhs slices of `NV` vectors of width `W`.
"""
function _snparray_AX_lookup_chunks!(
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    rhs::StridedMatrix{T},
    values::Matrix{T},
    rows_filled::Int,
    workspace::Vector{T},
    blk::Vector{UInt8},
    k_padded::Int,
    slice_stride::Int,
    stage_len::Int,
    tile_len::Int,
    row_span::Int,
    row_step::Int,
    block_step::Int,
    stage_base::Int,
    tile_base::Int,
    group::Int,
    width::Val{W},
    count::Val{NV},
) where {T <: AbstractFloat, W, NV}
    n = size(packed, 2)
    for column_first in 1:LOOKUP_CHUNK_SNPS:n
        nblk = cld(min(LOOKUP_CHUNK_SNPS, n - column_first + 1), 4)
        @sync begin
            task_index = 0
            for block_first in 1:block_step:nblk
                block_last = min(block_first + block_step - 1, nblk)
                stage_offset = stage_base + task_index * stage_len
                task_index += 1
                Threads.@spawn _lookup_build_tables!(
                    workspace, workspace, $stage_offset, n, rhs, values,
                    $column_first, $block_first, $block_last, k_padded,
                    slice_stride, width, count,
                )
            end
        end
        @sync begin
            task_index = 0
            for row_first in 1:row_step:rows_filled
                row_last = min(row_first + row_step - 1, rows_filled)
                tile_offset = tile_base + task_index * tile_len
                task_index += 1
                Threads.@spawn _snparray_AX_lookup_task!(
                    out, packed, workspace, workspace, $tile_offset, blk,
                    row_span, $row_first, $row_last, $column_first, $nblk,
                    slice_stride, group, width, count,
                )
            end
        end
    end
    return out
end

function _snparray_atx_tile!(out, packed, rhs, values, rows_filled, cols)
    n = length(cols)
    out_offset = first(cols) - 1
    row_step, column_step, _ =
        _tile_sizes(eltype(out), rows_filled, n, 1, :transpose; vector=true)
    @assert row_step % DECODE_WIDTH == 0 "row_step must be a multiple of DECODE_WIDTH"
    @sync begin
        for column_first in first(cols):column_step:last(cols)
            column_last = min(column_first + column_step - 1, last(cols))
            Threads.@spawn begin
                for row_first in 1:row_step:rows_filled
                    row_last = min(row_first + row_step - 1, rows_filled)
                    @assert (row_first - 1) % 4 == 0 "row_first must be ≡ 1 (mod 4)"
                    _snparray_atx_kernel!(
                        out, packed, rhs, values, row_first, row_last,
                        $column_first, $column_last, $out_offset,
                    )
                end
            end
        end
    end
    return out
end

function _snparray_AtX_tile!(out, packed, rhs, values, rows_filled, cols,
                             workspace)
    n = length(cols)
    k = size(out, 2)
    T = eltype(out)
    out_offset = first(cols) - 1
    row_step, column_step, rhs_step =
        _tile_sizes(T, rows_filled, n, k, :transpose; vector=false)
    lanes = _rhs_width(T, k)
    width = Val(lanes)
    @assert row_step % DECODE_WIDTH == 0 "row_step must be a multiple of DECODE_WIDTH"
    panel_length = 2lanes * row_step
    tasks = cld(n, column_step) * cld(k, rhs_step)
    _resize_workspace!(workspace, tasks * panel_length)
    task_index = 0
    @sync begin
        for rhs_first in 1:rhs_step:k
            rhs_last = min(rhs_first + rhs_step - 1, k)
            for column_first in first(cols):column_step:last(cols)
                column_last = min(column_first + column_step - 1, last(cols))
                panel_offset = task_index * panel_length
                task_index += 1
                Threads.@spawn _snparray_AtX_kernel!(
                    out, packed, rhs, values, workspace, $panel_offset,
                    $row_step, rows_filled, $column_first, $column_last,
                    $rhs_first, $rhs_last, $out_offset, $width,
                )
            end
        end
    end
    return out
end
