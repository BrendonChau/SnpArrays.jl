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
A matrix `rhs` on at least `LOOKUP_MIN_ROWS` samples with at least the
`LOOKUP_MIN_RHS` columns for its element type on this host uses the
lookup-table kernel.
A `SnpLinAlg{Float32}` also takes a `Float32` or `Float64` vector `out` and
`rhs`, each a `Vector` or `view(M, :, j)` of a `Matrix`: a `Float64` `rhs`
is rounded to `Float32`, and with a `Float64` `out`, `Float32` partial sums
over runs of at most 256 SNPs are added in `Float64`.
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
    SimdVector{T}

`Vector{T}` or a unit-stride contiguous vector view of a `Matrix{T}`, such
as `view(M, :, j)`: the vectors the SIMD vector kernels index with
`VecRange`.
"""
const SimdVector{T} = Union{
    Vector{T},
    Base.FastContiguousSubArray{T, 1, Matrix{T}},
}

"""
    ColumnRangeView{T}

Contiguous column-range view `view(M, :, j:k)` of a `Matrix{T}`.
"""
const ColumnRangeView{T} = SubArray{T, 2, Matrix{T},
    Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}

"""
    _column_vector(x) -> SimdVector

Alias the one column of `x` as a `SimdVector` without copying.
"""
_column_vector(x::Matrix) = vec(x)

function _column_vector(x::ColumnRangeView)
    return view(parent(x), :, first(parentindices(x)[2]))
end

"""
    _use_vector_kernels(out, ::Type{T}, rhs) -> Bool

Return whether a matrix product can run on the vector kernels through
`_column_vector(out)` and `_column_vector(rhs)`: each is a dense `Matrix` or
a `ColumnRangeView` with one column, of `Float32` or `Float64` element types
that `_supports_eltypes` accepts with the operator element type `T`.
"""
function _use_vector_kernels(
    out::Union{Matrix{TO}, ColumnRangeView{TO}},
    ::Type{T},
    rhs::Union{Matrix{TR}, ColumnRangeView{TR}},
) where {TO <: SIMD_FLOAT, T <: SIMD_FLOAT, TR <: SIMD_FLOAT}
    return _supports_eltypes(TO, T, TR) && size(rhs, 2) == 1
end

_use_vector_kernels(out, ::Type, rhs) = false

"""
    _mixed_ax!(out, sla, rhs, α, β) -> out

Mixed-precision `out = α * sla * rhs + β * out` on the SIMD vector kernels.
"""
function _mixed_ax!(
    out::SimdVector{<:SIMD_FLOAT},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{<:SIMD_FLOAT},
    α::Number,
    β::Number,
)
    length(out) == size(sla, 1) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 1))",
    ))
    length(rhs) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 2))",
    ))
    _begin_scaled_product!(out, α, β) || return out
    _mixed_ax_accumulate!(out, sla, rhs)
    return _finish_scaled_product!(out, α)
end

function _mixed_ax_accumulate!(
    out::SimdVector{Float32},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{Float64},
)
    return _snparray_ax_schedule!(out, sla.s.data, rhs, sla.values, sla.s.m)
end

# The Float32 run sums live in `sla.panel`, the matrix products' workspace.
function _mixed_ax_accumulate!(
    out::SimdVector{Float64},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{<:SIMD_FLOAT},
)
    @lock sla.lock begin
        scratch = fill!(_resize_workspace!(sla.panel, sla.s.m), 0.0f0)
        _snparray_ax_mixed_schedule!(out, scratch, sla.s.data, rhs,
                                     sla.values, sla.s.m)
    end
    return out
end

"""
    _mixed_atx!(out, sla, rhs, cols) -> out
    _mixed_atx!(out, sla, rhs, α, β) -> out

Mixed-precision transposed products on the SIMD vector kernels.
"""
function _mixed_atx!(
    out::SimdVector{<:SIMD_FLOAT},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{<:SIMD_FLOAT},
    cols::UnitRange{Int},
)
    cols ⊆ 1:size(sla, 2) || throw(ArgumentError(
        "cols $cols is not a subset of 1:$(size(sla, 2))",
    ))
    length(out) == length(cols) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(length(cols))",
    ))
    length(rhs) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 1))",
    ))
    fill!(out, zero(eltype(out)))
    return _mixed_atx_accumulate!(out, sla, rhs, cols)
end

function _mixed_atx!(
    out::SimdVector{<:SIMD_FLOAT},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{<:SIMD_FLOAT},
    α::Number,
    β::Number,
)
    length(out) == size(sla, 2) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 2))",
    ))
    length(rhs) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 1))",
    ))
    _begin_scaled_product!(out, α, β) || return out
    _mixed_atx_accumulate!(out, sla, rhs, 1:size(sla, 2))
    return _finish_scaled_product!(out, α)
end

function _mixed_atx_accumulate!(
    out::SimdVector{Float64},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{Float32},
    cols::UnitRange{Int},
)
    return _snparray_atx_schedule!(out, sla.s.data, rhs, sla.values, sla.s.m,
                                   cols)
end

# `rhs` is rounded once into `sla.panel`, the matrix products' workspace.
function _mixed_atx_accumulate!(
    out::SimdVector{<:SIMD_FLOAT},
    sla::SnpLinAlg{Float32},
    rhs::SimdVector{Float64},
    cols::UnitRange{Int},
)
    @lock sla.lock begin
        narrowed = copyto!(_resize_workspace!(sla.panel, length(rhs)), rhs)
        _snparray_atx_schedule!(out, sla.s.data, narrowed, sla.values,
                                sla.s.m, cols)
    end
    return out
end

const MIXED_TRANSPOSED = Union{Transpose{Float32, SnpLinAlg{Float32}},
                               Adjoint{Float32, SnpLinAlg{Float32}}}

# Supported mixed pairs (out, rhs): Float64 out with either rhs, or Float32
# out with Float64 rhs. All-Float32 calls keep the same-type methods.
for (TO, TR) in ((Float64, SIMD_FLOAT), (Float32, Float64))
    @eval begin
        function mul!(out::SimdVector{$TO}, sla::SnpLinAlg{Float32},
                      rhs::SimdVector{<:$TR})
            return _mixed_ax!(out, sla, rhs, 1, 0)
        end
        function mul!(out::SimdVector{$TO}, sla::SnpLinAlg{Float32},
                      rhs::SimdVector{<:$TR}, α::Number, β::Number)
            return _mixed_ax!(out, sla, rhs, α, β)
        end
        function mul!(out::SimdVector{$TO}, transposed::MIXED_TRANSPOSED,
                      rhs::SimdVector{<:$TR})
            sla = transposed.parent
            return _mixed_atx!(out, sla, rhs, 1:size(sla, 2))
        end
        function mul!(out::SimdVector{$TO}, transposed::MIXED_TRANSPOSED,
                      rhs::SimdVector{<:$TR}, cols::UnitRange{Int})
            return _mixed_atx!(out, transposed.parent, rhs, cols)
        end
        function mul!(out::SimdVector{$TO}, transposed::MIXED_TRANSPOSED,
                      rhs::SimdVector{<:$TR}, α::Number, β::Number)
            return _mixed_atx!(out, transposed.parent, rhs, α, β)
        end
    end
end

"""
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs::AbstractMatrix)
    LinearAlgebra.mul!(out, sla::SnpLinAlg, rhs::AbstractMatrix, α, β)

Matrix right-hand-side form of the method above. `out`, `sla`, and `rhs`
share one element type, or `sla` is a `SnpLinAlg{Float32}` and each of `out`
and `rhs` is a `Float32` or `Float64` matrix. A `Float64` `rhs` is rounded to
`Float32`; with a `Float64` `out`, partial sums over at most 1024 SNPs are
formed in `Float32` and summed in `Float64`. Other element types throw an
`ArgumentError`. One-column `out` and `rhs` of supported `Float32` or
`Float64` element types, each a dense `Matrix` or a `view(M, :, j:j)` of
one, run the vector kernel, with its run bound of 256 SNPs.
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
        mul!(_column_vector(out), sla, _column_vector(rhs), α, β)
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
`transpose(sla[:, cols]) * rhs`; `out` then has `length(cols)` rows. The
five-argument form computes
`out = α * transpose(sla) * rhs + β * out`.
For a `SnpLinAlg{Float32}`, `out` and `rhs` may be mixed as in the forward
form: a `Float64` `rhs` is rounded to `Float32` once per call, and with a
`Float64` `out`, `Float32` partial sums over at most 4096 samples are added
in `Float64`.
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
types throw an `ArgumentError`. One-column `out` and `rhs` of supported
`Float32` or `Float64` element types, each a dense `Matrix` or a
`view(M, :, j:j)` of one, run the vector kernel (same 4096-sample bound).
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
        mul!(_column_vector(out), transposed, _column_vector(rhs), cols)
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
        mul!(_column_vector(out), transposed, _column_vector(rhs), α, β)
        return out
    end
    _begin_scaled_product!(out, α, β) || return out
    @lock sla.lock _snparray_AtX_schedule!(out, sla.s.data, rhs, sla.values,
                                           sla.s.m, 1:size(sla, 2), sla.panel)
    return _finish_scaled_product!(out, α)
end
