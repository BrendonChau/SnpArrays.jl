"""
    _begin_scaled_product!(out, α, β) -> Bool

Scale `out` so that accumulating a product into it and multiplying by `α`
gives `α * product + β * out`, and return whether `α` is nonzero; when `α`
is zero, `out` holds `β * out`.
"""
function _begin_scaled_product!(
    out::AbstractVecOrMat{T},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    if iszero(β)
        fill!(out, zero(T))
    elseif iszero(α)
        rmul!(out, T(β))
    elseif β != α
        rmul!(out, T(β / α))
    end
    return !iszero(α)
end

"""
    _finish_scaled_product!(out, α) -> out

Multiply `out` by `α`.
"""
function _finish_scaled_product!(
    out::AbstractVecOrMat{T},
    α::Number,
) where T <: AbstractFloat
    isone(α) || rmul!(out, T(α))
    return out
end

"""
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs)
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs, α, β)

Multiply `sla` by a vector or matrix and overwrite `out`. The five-argument
form computes `out = α * sla * rhs + β * out`.
Concurrent matrix products on one `SnpLinAlg`, forward or transposed, run
one at a time.
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
    length(out) == size(sla, 1) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 1))",
    ))
    length(rhs) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 2))",
    ))
    _begin_scaled_product!(out, α, β) || return out
    _snparray_ax_schedule!(out, sla.s.data, rhs, sla.values, sla.s.m)
    return _finish_scaled_product!(out, α)
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
    size(out) == (size(sla, 1), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 1), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 2))",
    ))
    _begin_scaled_product!(out, α, β) || return out
    @lock sla.lock _snparray_AX_schedule!(out, sla.s.data, rhs, sla.values,
                                          sla.s.m, sla.panel, sla.blk)
    return _finish_scaled_product!(out, α)
end

"""
    LinearAlgebra.mul!(out, adjoint_sla, rhs)
    LinearAlgebra.mul!(out, adjoint_sla, rhs, cols)
    LinearAlgebra.mul!(out, adjoint_sla, rhs, α, β)

Multiply the transpose or adjoint of a `SnpLinAlg` by a vector or matrix and
overwrite `out`. With a `cols::UnitRange{Int}` argument, compute
`transpose(sla[:, cols]) * rhs`; `out` then has `length(cols)` rows. The
five-argument form computes
`out = α * transpose(sla) * rhs + β * out`.
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
    _snparray_atx_schedule!(out, sla.s.data, rhs, sla.values, sla.s.m, cols)
    return out
end

function mul!(
    out::AbstractVector{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractVector{T},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    sla = transposed.parent
    length(out) == size(sla, 2) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 2))",
    ))
    length(rhs) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 1))",
    ))
    _begin_scaled_product!(out, α, β) || return out
    _snparray_atx_schedule!(out, sla.s.data, rhs, sla.values, sla.s.m,
                            1:size(sla, 2))
    return _finish_scaled_product!(out, α)
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
    @lock sla.lock _snparray_AtX_schedule!(out, sla.s.data, rhs, sla.values,
                                           sla.s.m, cols, sla.panel)
    return out
end

function mul!(
    out::AbstractMatrix{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{T},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    sla = transposed.parent
    size(out) == (size(sla, 2), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 2), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 1))",
    ))
    _begin_scaled_product!(out, α, β) || return out
    @lock sla.lock _snparray_AtX_schedule!(out, sla.s.data, rhs, sla.values,
                                           sla.s.m, 1:size(sla, 2), sla.panel)
    return _finish_scaled_product!(out, α)
end
