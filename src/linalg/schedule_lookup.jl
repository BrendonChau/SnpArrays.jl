"""
    LookupLayout

Layout numbers of one lookup-table `A*X` product. The shared workspace holds
`[tables | one staging slice per build task | one output tile per gather
task]`; the tables are laid out `[slice][block][code][lane]`.
"""
struct LookupLayout
    k_padded::Int        # rhs columns padded to a whole slice
    slice_stride::Int    # workspace elements between two slices' tables
    row_span::Int        # samples per block in `blk`, a multiple of 4
    group::Int           # 4-SNP blocks swept per pass over a row tile
    stage_base::Int      # workspace offset of the first rhs staging slice
    stage_len::Int       # elements per staging slice
    tile_base::Int       # workspace offset of the first output tile
    tile_len::Int        # elements per output tile
    row_step::Int        # samples per gather task
    block_step::Int      # 4-SNP blocks per build task
end

"""
    _snparray_AX_lookup_schedule!(out, packed, rhs, values, rows_filled,
        workspace, blk)

Accumulate `out += A * rhs` with the lookup-table kernel: per chunk of
`LOOKUP_CHUNK_SNPS` SNPs, build the 256-row tables of every 4-SNP block
for all rhs columns (tasks over blocks), then gather them per sample
(tasks over sample blocks). Rhs slices hold `NV` vectors of width
`W = _vector_width(T)`, with `NV = 2` for `k ≤ 2W` and
`NV = max(2, 32 ÷ W)` otherwise; `workspace` is laid out as in
`LookupLayout`, and `blk` holds the transposed codes of one chunk.
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
    layout = LookupLayout(
        k_padded, slice_stride, row_span, group, stage_base, stage_len,
        tile_base, tile_len, row_step, block_step,
    )
    _snparray_AX_lookup_chunks!(
        out, packed, rhs, values, rows_filled, workspace, blk, layout,
        Val(lanes), Val(vectors),
    )
    return out
end

"""
    _snparray_AX_lookup_chunks!(out, packed, rhs, values, rows_filled,
        workspace, blk, layout, ::Val{W}, ::Val{NV})

Run the build and gather phases of `_snparray_AX_lookup_schedule!` for every
SNP chunk with rhs slices of `NV` vectors of width `W`. The `Val`s enter
here so that each product makes one dynamic call and every spawned task
call dispatches statically.
"""
function _snparray_AX_lookup_chunks!(
    out::StridedMatrix{T},
    packed::Matrix{UInt8},
    rhs::StridedMatrix{T},
    values::Matrix{T},
    rows_filled::Int,
    workspace::Vector{T},
    blk::Vector{UInt8},
    layout::LookupLayout,
    width::Val{W},
    count::Val{NV},
) where {T <: AbstractFloat, W, NV}
    stage_base = layout.stage_base
    stage_len = layout.stage_len
    tile_base = layout.tile_base
    tile_len = layout.tile_len
    row_step = layout.row_step
    block_step = layout.block_step
    # Each task closure captures this one pointer, not the 80-byte layout.
    shared = Ref(layout)
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
                    workspace, $stage_offset, n, rhs, values,
                    $column_first, $(block_first:block_last), shared[],
                    width, count,
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
                    out, packed, workspace, $tile_offset, blk,
                    $(row_first:row_last), $column_first, $nblk, shared[],
                    width, count,
                )
            end
        end
    end
    return out
end
