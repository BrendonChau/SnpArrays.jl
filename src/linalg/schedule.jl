"""
    _snparray_ax_schedule!(out, packed, rhs, values, rows_filled) -> out

Accumulate `out += A * rhs` for a vector `rhs` over the first `rows_filled`
samples, with one `_snparray_ax_kernel!` task per `_snparray_ax_row_step`
samples.
"""
function _snparray_ax_schedule!(
    out::AbstractVector{T},
    packed::AbstractMatrix{UInt8},
    rhs::AbstractVector{T},
    values::AbstractMatrix{T},
    rows_filled::Int,
) where T <: AbstractFloat
    n = size(packed, 2)
    # One task per block of samples; each sweeps every SNP column.
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

Resize the shared panel workspace to exactly `count` elements.
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

Accumulate `out += A * rhs` for a matrix `rhs` over the first `rows_filled`
samples, with `_snparray_AX_lookup_schedule!` when `_uses_lookup_kernel`
holds and else with tasks sized by `_snparray_AX_steps`.
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
    # One register-tiled task per block along the task axis and rhs tile.
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
    # `Val`s enter here: one dynamic call per product, static dispatch below.
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
`panel_length` slice of `workspace`.
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
    _snparray_atx_schedule!(out, packed, rhs, values, rows_filled, cols) -> out

Accumulate `out += transpose(A[:, cols]) * rhs` for a vector `rhs` over the
first `rows_filled` samples, with `_snparray_atx_kernel!` tasks sized by
`_snparray_atx_steps`.
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
    # One task per block of SNP columns; each sweeps the samples in blocks.
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

Accumulate `out += transpose(A[:, cols]) * rhs` for a matrix `rhs` over the
first `rows_filled` samples, with tasks sized by `_snparray_AtX_steps`.
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
    # One register-tiled task per block along the task axis and rhs tile.
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
