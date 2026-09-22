struct SnpLinAlg{T} <: AbstractMatrix{T}
    s::SnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    center::Bool
    scale::Bool
    impute::Bool
    μ::Vector{T}
    σinv::Vector{T}
    values::Matrix{T}
    # Packing workspace for the register-tiled matrix products, one slice
    # per spawned task. Resized on the first product of a given shape and
    # reused after, so a repeated product allocates nothing here. Not
    # reentrant: concurrent `mul!` calls on one `SnpLinAlg` would share it.
    panel::Vector{T}
    # Transposed genotype codes (one byte per sample and 4-SNP block) for
    # the lookup-table `A*X` kernel; same lifetime and caveat as `panel`.
    blk::Vector{UInt8}
end

AbstractSnpLinAlg = Union{SnpLinAlg, SubArray{T, 1, SnpLinAlg{T}},
    SubArray{T, 2, SnpLinAlg{T}}} where T

"""
    SnpLinAlg{T}(s; model=ADDITIVE_MODEL, center=false, scale=false,
                 impute=true)

Wrap a `SnpArray` for direct linear algebra without materializing its genotypes.
Missing genotypes use the column mean before centering and scaling when
`impute=true`, and use `NaN` otherwise.

# Arguments
- `s`: a `SnpArray`
- `model`: `ADDITIVE_MODEL`, `DOMINANT_MODEL`, or `RECESSIVE_MODEL`
- `center`: center each column when `true`
- `scale`: scale each column to unit standard deviation when `true`
- `impute`: replace missing genotypes with the column mean when `true`
"""
function SnpLinAlg{T}(
    s::AbstractSnpArray;
    model = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false,
    impute::Bool = true,
) where T <: AbstractFloat
    model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL) ||
        throw(ArgumentError("unrecognized model $model"))
    means = Vector{T}(undef, size(s, 2))
    inverse_standard_deviations = Vector{T}(undef, size(s, 2))
    values = Matrix{T}(undef, 4, size(s, 2))
    _fill_statistics!(means, inverse_standard_deviations, values, s, model,
                      center, scale, impute)
    return SnpLinAlg{T}(s, model, center, scale, impute, means,
                        inverse_standard_deviations, values, T[], UInt8[])
end

"""
    _fill_statistics!(means, inverse_standard_deviations, values, s, model,
        center, scale, impute) -> values

Recompute `means`, `inverse_standard_deviations`, and `values` in place
from the genotype counts of `s`.
"""
function _fill_statistics!(
    means::Vector{T},
    inverse_standard_deviations::Vector{T},
    values::Matrix{T},
    s::AbstractSnpArray,
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    mean!(means, s; dims=1, model=model)
    @inbounds @simd for column in eachindex(means)
        column_mean = means[column]
        variance = model == ADDITIVE_MODEL ?
                   column_mean * (one(T) - column_mean / T(2)) :
                   column_mean * (one(T) - column_mean)
        standard_deviation = sqrt(variance)
        inverse_standard_deviations[column] =
            standard_deviation > zero(T) ? inv(standard_deviation) : one(T)
    end
    _fill_genotype_values!(values, means, inverse_standard_deviations, model,
                           center, scale, impute)
    return values
end

"""
    _refill_statistics!(sla::SnpLinAlg) -> sla

Recompute `sla.μ`, `sla.σinv`, and `sla.values` from `sla.s`; the caller
must zero `sla.s.columncounts` first if the data changed.
"""
function _refill_statistics!(sla::SnpLinAlg{T}) where T <: AbstractFloat
    _fill_statistics!(sla.μ, sla.σinv, sla.values, sla.s, sla.model,
                      sla.center, sla.scale, sla.impute)
    return sla
end

function _fill_genotype_values!(
    values::AbstractMatrix{T},
    means::AbstractVector{T},
    inverse_standard_deviations::AbstractVector{T},
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    @inbounds for column in axes(values, 2)
        transformed = _transformed_genotype_values(
            T, means[column], inverse_standard_deviations[column], model,
            center, scale, impute,
        )
        for code in 1:4
            values[code, column] = transformed[code]
        end
    end
    return values
end

Base.size(sla::SnpLinAlg) = size(sla.s)
Base.size(sla::SnpLinAlg, dimension::Integer) = size(sla.s, dimension)
Base.eltype(::SnpLinAlg{T}) where T = T

@inline function Base.getindex(sla::SnpLinAlg, row::Int, column::Int)
    code = getindex(sla.s, row, column)
    return @inbounds sla.values[Int(code) + 1, column]
end

"""
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs)
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs, α, β)

Multiply `sla` by a vector or matrix and overwrite `out`. A matrix `rhs`
with at least `LOOKUP_MIN_RHS` columns on at least `LOOKUP_MIN_ROWS`
samples uses the lookup-table kernel (`_snparray_AX_lookup_tile!`); other
shapes use the register-tiled kernel. The five-argument form computes
`out = sla * rhs + β * out` and requires `α == 1` and `β ∈ (0, 1)`.
"""
function mul!(
    out::AbstractVector{T},
    sla::SnpLinAlg{T},
    rhs::AbstractVector{T},
) where T <: AbstractFloat
    return mul!(out, sla, rhs, 1, 0)
end

function mul!(
    out::AbstractVector{T},
    sla::SnpLinAlg{T},
    rhs::AbstractVector{T},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    α == 1 || throw(ArgumentError("α must be 1, got $α"))
    β == 0 || β == 1 || throw(ArgumentError("β must be 0 or 1, got $β"))
    length(out) == size(sla, 1) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 1))",
    ))
    length(rhs) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 2))",
    ))
    β == 0 && fill!(out, zero(T))
    _snparray_ax_tile!(out, sla.s.data, rhs, sla.values, sla.s.m)
    return out
end

function mul!(
    out::AbstractMatrix{T},
    sla::SnpLinAlg{T},
    rhs::AbstractMatrix{T},
) where T <: AbstractFloat
    return mul!(out, sla, rhs, 1, 0)
end

function mul!(
    out::AbstractMatrix{T},
    sla::SnpLinAlg{T},
    rhs::AbstractMatrix{T},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    α == 1 || throw(ArgumentError("α must be 1, got $α"))
    β == 0 || β == 1 || throw(ArgumentError("β must be 0 or 1, got $β"))
    size(out) == (size(sla, 1), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 1), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 2))",
    ))
    β == 0 && fill!(out, zero(T))
    _snparray_AX_tile!(out, sla.s.data, rhs, sla.values, sla.s.m, sla.panel,
                       sla.blk)
    return out
end

"""
    LinearAlgebra.mul!(out, adjoint_sla, rhs)
    LinearAlgebra.mul!(out, adjoint_sla, rhs, cols)

Multiply the transpose or adjoint of a `SnpLinAlg` by a vector or matrix and
overwrite `out`. With a `cols::UnitRange{Int}` argument, compute
`transpose(sla[:, cols]) * rhs` (i.e. restrict the SNP columns of `sla`
without materializing a new `SnpLinAlg`); `out` then has `length(cols)`
rows.
"""
function mul!(
    out::AbstractVector{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractVector{T},
) where T <: AbstractFloat
    sla = transposed.parent
    return mul!(out, transposed, rhs, 1:size(sla, 2))
end

function mul!(
    out::AbstractVector{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractVector{T},
    cols::UnitRange{Int},
) where T <: AbstractFloat
    sla = transposed.parent
    cols ⊆ 1:size(sla, 2) || throw(ArgumentError(
        "cols $cols is not a subset of 1:$(size(sla, 2))",
    ))
    length(out) == length(cols) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(length(cols))",
    ))
    length(rhs) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 1))",
    ))
    fill!(out, zero(T))
    # The dense genotype arrays with `cols` as an index offset, never a
    # view: indexing a `SubArray` per element costs about 2x in the
    # register-tiled kernel.
    _snparray_atx_tile!(out, sla.s.data, rhs, sla.values, sla.s.m, cols)
    return out
end

function mul!(
    out::AbstractMatrix{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{T},
) where T <: AbstractFloat
    sla = transposed.parent
    return mul!(out, transposed, rhs, 1:size(sla, 2))
end

function mul!(
    out::AbstractMatrix{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{T},
    cols::UnitRange{Int},
) where T <: AbstractFloat
    sla = transposed.parent
    cols ⊆ 1:size(sla, 2) || throw(ArgumentError(
        "cols $cols is not a subset of 1:$(size(sla, 2))",
    ))
    size(out) == (length(cols), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((length(cols), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 1))",
    ))
    fill!(out, zero(T))
    # The dense genotype arrays with `cols` as an index offset, never a
    # view: indexing a `SubArray` per element costs about 2x in the
    # register-tiled kernel.
    _snparray_AtX_tile!(out, sla.s.data, rhs, sla.values, sla.s.m, cols,
                        sla.panel)
    return out
end

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
       out isa Matrix{T} && rhs isa Matrix{T} && packed isa Matrix{UInt8}
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
(tasks over sample blocks). `workspace` holds the tables, one rhs staging
slice per build task, and one output tile per gather task; `blk` holds the
transposed codes of one chunk.
"""
function _snparray_AX_lookup_tile!(out, packed, rhs, values, rows_filled,
                                   workspace, blk)
    n = size(packed, 2)
    k = size(out, 2)
    T = eltype(out)
    lanes = _vector_width(T)
    width = Val(lanes)
    k_padded = cld(k, 2lanes) * 2lanes
    nblk_max = min(LOOKUP_CHUNK_SNPS ÷ 4, cld(n, 4))
    row_span = 4 * cld(rows_filled, 4)
    tables_len = nblk_max * 256 * k_padded
    stage_len = 4k_padded
    tile_len = LOOKUP_ROW_TILE * 2lanes
    row_step = _task_axis_step(rows_filled, rows_filled)
    row_tasks = cld(rows_filled, row_step)
    block_step = max(1, cld(nblk_max, TASKS_PER_THREAD * Threads.nthreads()))
    block_tasks = cld(nblk_max, block_step)
    stage_base = tables_len
    tile_base = stage_base + block_tasks * stage_len
    _resize_workspace!(workspace, tile_base + row_tasks * tile_len)
    _resize_workspace!(blk, nblk_max * row_span)
    group = clamp(LOOKUP_GROUP_BUDGET ÷ (256 * 2lanes * sizeof(T)), 1,
                  nblk_max)
    @assert row_step % 4 == 0 "row_step must be a multiple of 4"
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
                    $column_first, $block_first, $block_last, k_padded, width,
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
                    k_padded, group, width,
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

@inline function _packed_code(packed, row::Int, column::Int)
    byte = @inbounds packed[((row - 1) >>> 2) + 1, column]
    return Int((byte >> (2((row - 1) & 3))) & 0x03) + 1
end

function _snparray_ax_kernel!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last)
    return _snparray_ax_scalar!(out, packed, rhs, values, row_first, row_last,
                                column_first, column_last)
end

function _snparray_ax_scalar!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last)
    @inbounds for column in column_first:column_last
        rhs_value = rhs[column]
        for row in row_first:row_last
            out[row] += values[_packed_code(packed, row, column), column] *
                        rhs_value
        end
    end
    return out
end

function _snparray_AX_kernel!(out, packed, rhs, values, panel, panel_offset,
                              row_first, row_last, column_step, rhs_first,
                              rhs_last, ::Val)
    return _snparray_AX_scalar!(out, packed, rhs, values, row_first, row_last,
                                1, size(packed, 2), rhs_first, rhs_last)
end

function _snparray_AX_scalar!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last, rhs_first, rhs_last)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in column_first:column_last
            rhs_value = rhs[column, rhs_column]
            for row in row_first:row_last
                out[row, rhs_column] +=
                    values[_packed_code(packed, row, column), column] * rhs_value
            end
        end
    end
    return out
end

function _snparray_atx_kernel!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, out_offset)
    return _snparray_atx_scalar!(out, packed, rhs, values, row_first, row_last,
                                 column_first, column_last, out_offset)
end

function _snparray_atx_scalar!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, out_offset)
    @inbounds for column in column_first:column_last
        total = out[column - out_offset]
        for row in row_first:row_last
            total += values[_packed_code(packed, row, column), column] * rhs[row]
        end
        out[column - out_offset] = total
    end
    return out
end

function _snparray_AtX_kernel!(out, packed, rhs, values, panel, panel_offset,
                               row_step, rows_filled, column_first,
                               column_last, rhs_first, rhs_last, out_offset,
                               ::Val)
    return _snparray_AtX_scalar!(out, packed, rhs, values, 1, rows_filled,
                                 column_first, column_last, rhs_first,
                                 rhs_last, out_offset)
end

function _snparray_AtX_scalar!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, rhs_first, rhs_last,
                               out_offset)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in column_first:column_last
            total = out[column - out_offset, rhs_column]
            for row in row_first:row_last
                total += values[_packed_code(packed, row, column), column] *
                         rhs[row, rhs_column]
            end
            out[column - out_offset, rhs_column] = total
        end
    end
    return out
end

"""
    Base.copyto!(destination, source)

Copy a `SnpLinAlg` or one of its views to a floating-point vector or matrix.
"""
function Base.copyto!(
    destination::AbstractVecOrMat{T},
    source::AbstractSnpLinAlg,
) where T <: AbstractFloat
    size(destination) == size(source) || throw(DimensionMismatch(
        "destination has size $(size(destination)); expected $(size(source))",
    ))
    for index in eachindex(destination, source)
        @inbounds destination[index] = source[index]
    end
    return destination
end

"""
    Base.convert(T, source)

Convert a `SnpLinAlg` or one of its views to an array with the same shape.
"""
Base.convert(::Type{T}, source::AbstractSnpLinAlg) where T <: Array = T(source)
Array{T, N}(source::AbstractSnpLinAlg) where {T, N} =
    copyto!(Array{T, N}(undef, size(source)), source)
