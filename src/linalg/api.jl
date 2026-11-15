"""
    _begin_scaled_product!(out, α, β) -> Bool

Prepare `out` for `out = α * product + β * out`, where a kernel then
accumulates the unscaled product into `out`: replace `out` by
`(β / α) * out`, which `_finish_scaled_product!` multiplies by `α`. Return
`false` when `α` is zero; `out` then already holds `β * out` and the caller
skips the product.
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

Multiply `out`, prepared by `_begin_scaled_product!` and holding the
accumulated product, by `α`.
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

Multiply `sla` by a vector or matrix and overwrite `out`. A matrix `rhs`
on at least `LOOKUP_MIN_ROWS` samples with at least the `LOOKUP_MIN_RHS`
columns for its element type on this host uses the lookup-table kernel
(`_snparray_AX_lookup_schedule!`); other shapes use the register-tiled
kernel. The five-argument form computes
`out = α * sla * rhs + β * out`.
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

"""
    _supports_eltypes(::Type{TO}, ::Type{T}, ::Type{TR}) -> Bool

Return whether a matrix product with output element type `TO`, operator
element type `T`, and rhs element type `TR` is supported: all equal, or a
`Float32` operator with each of `TO` and `TR` either `Float32` or `Float64`.
"""
_supports_eltypes(::Type{T}, ::Type{T}, ::Type{T}) where T = true
_supports_eltypes(::Type{Float32}, ::Type{Float32}, ::Type{Float32}) = true
function _supports_eltypes(
    ::Type{<:SIMD_FLOAT},
    ::Type{Float32},
    ::Type{<:SIMD_FLOAT},
)
    return true
end
_supports_eltypes(::Type, ::Type, ::Type) = false

"""
    _check_eltypes(out, T, rhs)

Throw an `ArgumentError` unless `_supports_eltypes` accepts the element types
of `out`, the operator element type `T`, and the element type of `rhs`.
"""
function _check_eltypes(
    out::AbstractMatrix{TO},
    ::Type{T},
    rhs::AbstractMatrix{TR},
) where {TO, T, TR}
    _supports_eltypes(TO, T, TR) || throw(ArgumentError(
        "output has element type $TO, the operator $T, and the " *
        "right-hand side $TR; expected equal types, or a Float32 operator " *
        "with Float32 or Float64 output and right-hand side",
    ))
    return nothing
end

"""
    _use_vector_kernels(out, ::Type{T}, rhs) -> Bool

Return whether a matrix product can run on the vector kernels through
`vec(out)` and `vec(rhs)`: both are dense one-column `Matrix{T}` sharing the
`Float32` or `Float64` operator element type `T`, so `vec` aliases them as the
`Vector{T}` the SIMD kernels take.
"""
function _use_vector_kernels(
    out::Matrix{T},
    ::Type{T},
    rhs::Matrix{T},
) where T <: SIMD_FLOAT
    return size(rhs, 2) == 1
end

_use_vector_kernels(out, ::Type, rhs) = false

"""
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs::AbstractMatrix)
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs::AbstractMatrix, α, β)

Matrix right-hand-side form of the method above. `out`, `sla`, and `rhs`
share one element type, or `sla` is a `SnpLinAlg{Float32}` and each of `out`
and `rhs` is a `Float32` or `Float64` matrix. A `Float64` `rhs` is rounded to
`Float32`; with a `Float64` `out`, partial sums over at most 1024 SNPs are
formed in `Float32` and summed in `Float64`. Other element types throw an
`ArgumentError`. A one-column dense `Matrix` `out` and `rhs` of the
operator's `Float32` or `Float64` element type run the vector kernel.
"""
function mul!(
    out::AbstractMatrix{<:AbstractFloat},
    sla::SnpLinAlg{T},
    rhs::AbstractMatrix{<:AbstractFloat},
) where T <: AbstractFloat
    return mul!(out, sla, rhs, 1, 0)
end

function mul!(
    out::AbstractMatrix{<:AbstractFloat},
    sla::SnpLinAlg{T},
    rhs::AbstractMatrix{<:AbstractFloat},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    _check_eltypes(out, T, rhs)
    size(out) == (size(sla, 1), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 1), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 2))",
    ))
    if _use_vector_kernels(out, T, rhs)
        mul!(vec(out), sla, vec(rhs), α, β)
        return out
    end
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
`transpose(sla[:, cols]) * rhs` (i.e. restrict the SNP columns of `sla`
without materializing a new `SnpLinAlg`); `out` then has `length(cols)`
rows. The five-argument form computes
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

"""
    LinearAlgebra.mul!(out, adjoint_sla, rhs::AbstractMatrix)
    LinearAlgebra.mul!(out, adjoint_sla, rhs::AbstractMatrix, cols)
    LinearAlgebra.mul!(out, adjoint_sla, rhs::AbstractMatrix, α, β)

Matrix right-hand-side forms of the methods above. `out`, the operator, and
`rhs` share one element type, or the operator is a `SnpLinAlg{Float32}` and
each of `out` and `rhs` is a `Float32` or `Float64` matrix. A `Float64` `rhs`
is rounded to `Float32`; with a `Float64` `out`, partial sums over at most
4096 samples are formed in `Float32` and summed in `Float64`. Other element
types throw an `ArgumentError`. A one-column dense `Matrix` `out` and `rhs`
of the operator's `Float32` or `Float64` element type run the vector kernel.
"""
function mul!(
    out::AbstractMatrix{<:AbstractFloat},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{<:AbstractFloat},
) where T <: AbstractFloat
    sla = transposed.parent
    return mul!(out, transposed, rhs, 1:size(sla, 2))
end

function mul!(
    out::AbstractMatrix{TO},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{<:AbstractFloat},
    cols::UnitRange{Int},
) where {TO <: AbstractFloat, T <: AbstractFloat}
    _check_eltypes(out, T, rhs)
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
    if _use_vector_kernels(out, T, rhs)
        mul!(vec(out), transposed, vec(rhs), cols)
        return out
    end
    fill!(out, zero(TO))
    # The dense genotype arrays with `cols` as an index offset, never a
    # view: indexing a `SubArray` per element costs about 2x in the
    # register-tiled kernel.
    @lock sla.lock _snparray_AtX_schedule!(out, sla.s.data, rhs, sla.values,
                                           sla.s.m, cols, sla.panel)
    return out
end

function mul!(
    out::AbstractMatrix{<:AbstractFloat},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{<:AbstractFloat},
    α::Number,
    β::Number,
) where T <: AbstractFloat
    _check_eltypes(out, T, rhs)
    sla = transposed.parent
    size(out) == (size(sla, 2), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 2), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 1))",
    ))
    if _use_vector_kernels(out, T, rhs)
        mul!(vec(out), transposed, vec(rhs), α, β)
        return out
    end
    _begin_scaled_product!(out, α, β) || return out
    @lock sla.lock _snparray_AtX_schedule!(out, sla.s.data, rhs, sla.values,
                                           sla.s.m, 1:size(sla, 2), sla.panel)
    return _finish_scaled_product!(out, α)
end
