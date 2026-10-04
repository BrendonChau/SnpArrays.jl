"""
    _snparray_ax_schedule!(out, packed, rhs, values, rows_filled) -> out

Accumulate `out += A * rhs` by spawning one task per block of samples, each
sweeping every SNP column with `_snparray_ax_kernel!`.
"""
function _snparray_ax_schedule!(
    out::AbstractVector{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractVector{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
) where T <: AbstractFloat
    n = size(packed, 2)
    row_step = _snparray_ax_row_step(T, rows_filled)
    @assert(
        row_step % DECODE_WIDTH == 0,
        "row_step must be a multiple of DECODE_WIDTH",
    )
    @sync begin
        for row_first in 1:row_step:rows_filled
            row_last = min(row_first + row_step - 1, rows_filled)
            @assert (row_first - 1) % 4 == 0 "row_first must be ≡ 1 (mod 4)"
            Threads.@spawn _snparray_ax_kernel!(
                out, packed, rhs, values, $(row_first:row_last), 1:n,
            )
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

"""
    _supports_lookup(out, packed, rhs) -> Bool

Return whether the argument types are the dense `Float32` or `Float64` arrays
that the lookup-table `A*X` kernel accepts.
"""
function _supports_lookup(
    ::StridedMatrix{T},
    ::Matrix{UInt8},
    ::StridedMatrix{T},
) where T <: SIMD_FLOAT
    return true
end

_supports_lookup(out, packed, rhs) = false

"""
    _snparray_AX_schedule!(out, packed, rhs, values, rows_filled, workspace,
        blk) -> out

Accumulate `out += A * rhs` with the lookup-table kernel when the shape and
array types allow it, else by spawning one register-tiled task per row block
and rhs tile.
"""
function _snparray_AX_schedule!(
    out::AbstractMatrix{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractMatrix{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
    workspace::Vector{T},
    blk::Vector{UInt8},
) where T <: AbstractFloat
    k = size(out, 2)
    if _uses_lookup_kernel(rows_filled, k) &&
       _supports_lookup(out, packed, rhs)
        return _snparray_AX_lookup_schedule!(out, packed, rhs, values,
                                             rows_filled, workspace, blk)
    end
    row_step, column_step, rhs_step =
        _snparray_AX_steps(T, rows_filled, k)
    lanes = _rhs_width(T, k)
    @assert(
        row_step % DECODE_WIDTH == 0,
        "row_step must be a multiple of DECODE_WIDTH",
    )
    # One panel slice per spawned task, so concurrent tasks never overlap.
    panel_length = 2lanes * column_step
    tasks = cld(rows_filled, row_step) * cld(k, rhs_step)
    _resize_workspace!(workspace, tasks * panel_length)
    tile_rows, _ = _register_tile_shape(T)
    return _snparray_AX_spawn_tasks!(
        out, packed, rhs, values, rows_filled, workspace, row_step,
        column_step, rhs_step, panel_length, Val(tile_rows), Val(lanes),
    )
end

"""
    _snparray_AX_spawn_tasks!(out, packed, rhs, values, rows_filled,
        workspace, row_step, column_step, rhs_step, panel_length, ::Val{MR},
        ::Val{W}) -> out

Spawn one `_snparray_AX_task!` per row block and rhs tile, each with its own
`panel_length` slice of `workspace`. The tile shape enters as `Val`s here so
that each product makes one dynamic call, and the spawn loop and every task
call inside it dispatch statically.
"""
function _snparray_AX_spawn_tasks!(
    out::AbstractMatrix{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractMatrix{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
    workspace::Vector{T},
    row_step::Int,
    column_step::Int,
    rhs_step::Int,
    panel_length::Int,
    tile_rows::Val{MR},
    width::Val{W},
) where {T <: AbstractFloat, MR, W}
    k = size(out, 2)
    task_index = 0
    @sync begin
        for rhs_first in 1:rhs_step:k
            rhs_last = min(rhs_first + rhs_step - 1, k)
            for row_first in 1:row_step:rows_filled
                row_last = min(row_first + row_step - 1, rows_filled)
                @assert (row_first - 1) % 4 == 0 "row_first must be ≡ 1 (mod 4)"
                tile_task = RegisterTileTask(
                    out, packed, values, workspace,
                    task_index * panel_length, 0,
                )
                task_index += 1
                Threads.@spawn _snparray_AX_task!(
                    $tile_task, rhs, $(row_first:row_last), $column_step,
                    $(rhs_first:rhs_last), tile_rows, width,
                )
            end
        end
    end
    return out
end

"""
    _snparray_AX_lookup_schedule!(out, packed, rhs, values, rows_filled,
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
function _snparray_AX_lookup_schedule!(
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    rhs::StridedMatrix{T},
    values::Matrix{T},
    rows_filled::Int,
    workspace::Vector{T},
    blk::Vector{UInt8},
) where T <: SIMD_FLOAT
    n = size(packed, 2)
    k = size(out, 2)
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

Run the build and gather phases of `_snparray_AX_lookup_schedule!` for every
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
                Threads.@spawn _snparray_AX_lookup_gather_task!(
                    out, packed, workspace, workspace, $tile_offset, blk,
                    row_span, $row_first, $row_last, $column_first, $nblk,
                    slice_stride, group, width, count,
                )
            end
        end
    end
    return out
end

"""
    _snparray_atx_schedule!(out, packed, rhs, values, rows_filled, cols) -> out

Accumulate `out += transpose(A[:, cols]) * rhs` by spawning one task per block
of SNP columns, each sweeping every sample with `_snparray_atx_kernel!`.
"""
function _snparray_atx_schedule!(
    out::AbstractVector{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractVector{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
    cols::UnitRange{Int},
) where T <: AbstractFloat
    n = length(cols)
    out_offset = first(cols) - 1
    row_step, column_step = _snparray_atx_steps(T, n)
    @assert(
        row_step % DECODE_WIDTH == 0,
        "row_step must be a multiple of DECODE_WIDTH",
    )
    @sync begin
        for column_first in first(cols):column_step:last(cols)
            column_last = min(column_first + column_step - 1, last(cols))
            Threads.@spawn begin
                for row_first in 1:row_step:rows_filled
                    row_last = min(row_first + row_step - 1, rows_filled)
                    @assert(
                        (row_first - 1) % 4 == 0,
                        "row_first must be ≡ 1 (mod 4)",
                    )
                    _snparray_atx_kernel!(
                        out, packed, rhs, values, row_first:row_last,
                        $(column_first:column_last), $out_offset,
                    )
                end
            end
        end
    end
    return out
end

"""
    _snparray_AtX_schedule!(out, packed, rhs, values, rows_filled, cols,
        workspace) -> out

Accumulate `out += transpose(A[:, cols]) * rhs` by spawning one register-tiled
task per block of SNP columns and rhs tile.
"""
function _snparray_AtX_schedule!(
    out::AbstractMatrix{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractMatrix{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
    cols::UnitRange{Int},
    workspace::Vector{T},
) where T <: AbstractFloat
    n = length(cols)
    k = size(out, 2)
    row_step, column_step, rhs_step =
        _snparray_AtX_steps(T, n, k)
    lanes = _rhs_width(T, k)
    @assert(
        row_step % DECODE_WIDTH == 0,
        "row_step must be a multiple of DECODE_WIDTH",
    )
    panel_length = 2lanes * row_step
    tasks = cld(n, column_step) * cld(k, rhs_step)
    _resize_workspace!(workspace, tasks * panel_length)
    tile_columns, _ = _register_tile_shape(T)
    return _snparray_AtX_spawn_tasks!(
        out, packed, rhs, values, rows_filled, cols, workspace, row_step,
        column_step, rhs_step, panel_length, Val(tile_columns), Val(lanes),
    )
end

"""
    _snparray_AtX_spawn_tasks!(out, packed, rhs, values, rows_filled, cols,
        workspace, row_step, column_step, rhs_step, panel_length, ::Val{MR},
        ::Val{W}) -> out

Spawn one `_snparray_AtX_task!` per block of SNP columns in `cols` and rhs
tile, each with its own `panel_length` slice of `workspace`.
"""
function _snparray_AtX_spawn_tasks!(
    out::AbstractMatrix{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractMatrix{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
    cols::UnitRange{Int},
    workspace::Vector{T},
    row_step::Int,
    column_step::Int,
    rhs_step::Int,
    panel_length::Int,
    tile_width::Val{MR},
    width::Val{W},
) where {T <: AbstractFloat, MR, W}
    k = size(out, 2)
    out_offset = first(cols) - 1
    task_index = 0
    @sync begin
        for rhs_first in 1:rhs_step:k
            rhs_last = min(rhs_first + rhs_step - 1, k)
            for column_first in first(cols):column_step:last(cols)
                column_last = min(column_first + column_step - 1, last(cols))
                tile_task = RegisterTileTask(
                    out, packed, values, workspace,
                    task_index * panel_length, out_offset,
                )
                task_index += 1
                Threads.@spawn _snparray_AtX_task!(
                    $tile_task, rhs, $row_step, rows_filled,
                    $(column_first:column_last), $(rhs_first:rhs_last),
                    tile_width, width,
                )
            end
        end
    end
    return out
end
