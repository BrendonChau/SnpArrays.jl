"""
    SnpLinAlg{T}

Wraps a `SnpArray` with the parameters and precomputed genotype values for
linear algebra.

# Fields
- `s`: the underlying genotype array.
- `model`: genetic model used to convert genotypes to numbers.
- `center`: whether to center columns.
- `scale`: whether to scale columns to standard deviation 1.
- `impute`: whether to impute missing genotypes with the column mean.
- `μ`: column means.
- `σinv`: inverse column standard deviations.
- `values`: a `4 × n` lookup table where entry `(code + 1, j)` is genotype code
  `code` in column `j` after model conversion, imputation, centering and
  scaling.
"""
struct SnpLinAlg{T} <: AbstractMatrix{T}
    s::SnpArray
    model::Union{Val{1}, Val{2}, Val{3}}
    center::Bool
    scale::Bool
    impute::Bool
    μ::Vector{T}
    σinv::Vector{T}
    values::Matrix{T}
end

"""
    AbstractSnpLinAlg

Union of `SnpLinAlg{T}` with its 1-D and 2-D `SubArray` views, so that
methods dispatch identically on a `SnpLinAlg` or a view of it.
"""
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
    means = Vector{T}(dropdims(mean(s; dims=1, model=model); dims=1))
    inverse_standard_deviations = Vector{T}(undef, size(s, 2))
    @inbounds @simd for column in eachindex(means)
        column_mean = means[column]
        variance = model == ADDITIVE_MODEL ?
                   column_mean * (one(T) - column_mean / T(2)) :
                   column_mean * (one(T) - column_mean)
        standard_deviation = sqrt(variance)
        inverse_standard_deviations[column] =
            standard_deviation > zero(T) ? inv(standard_deviation) : one(T)
    end
    values = Matrix{T}(undef, 4, size(s, 2))
    _fill_genotype_values!(values, means, inverse_standard_deviations, model,
                           center, scale, impute)
    return SnpLinAlg{T}(s, model, center, scale, impute, means,
                        inverse_standard_deviations, values)
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

Multiply `sla` by a vector or matrix and overwrite `out`.
"""
function mul!(
    out::AbstractVector{T},
    sla::SnpLinAlg{T},
    rhs::AbstractVector{T},
) where T <: AbstractFloat
    length(out) == size(sla, 1) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 1))",
    ))
    length(rhs) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 2))",
    ))
    fill!(out, zero(T))
    _snparray_ax_tile!(out, sla.s.data, rhs, sla.values, sla.s.m)
    return out
end

function mul!(
    out::AbstractMatrix{T},
    sla::SnpLinAlg{T},
    rhs::AbstractMatrix{T},
) where T <: AbstractFloat
    size(out) == (size(sla, 1), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 1), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 2) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 2))",
    ))
    fill!(out, zero(T))
    _snparray_AX_tile!(out, sla.s.data, rhs, sla.values, sla.s.m)
    return out
end

"""
    LinearAlgebra.mul!(out, adjoint_sla, rhs)

Multiply the transpose or adjoint of a `SnpLinAlg` by a vector or matrix and
overwrite `out`.
"""
function mul!(
    out::AbstractVector{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractVector{T},
) where T <: AbstractFloat
    sla = transposed.parent
    length(out) == size(sla, 2) || throw(DimensionMismatch(
        "output has length $(length(out)); expected $(size(sla, 2))",
    ))
    length(rhs) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has length $(length(rhs)); expected $(size(sla, 1))",
    ))
    fill!(out, zero(T))
    _snparray_atx_tile!(out, sla.s.data, rhs, sla.values, sla.s.m)
    return out
end

function mul!(
    out::AbstractMatrix{T},
    transposed::Union{Transpose{T, SnpLinAlg{T}},
                      Adjoint{T, SnpLinAlg{T}}},
    rhs::AbstractMatrix{T},
) where T <: AbstractFloat
    sla = transposed.parent
    size(out) == (size(sla, 2), size(rhs, 2)) || throw(DimensionMismatch(
        "output has size $(size(out)); expected $((size(sla, 2), size(rhs, 2)))",
    ))
    size(rhs, 1) == size(sla, 1) || throw(DimensionMismatch(
        "right-hand side has $(size(rhs, 1)) rows; expected $(size(sla, 1))",
    ))
    fill!(out, zero(T))
    _snparray_AtX_tile!(out, sla.s.data, rhs, sla.values, sla.s.m)
    return out
end

function _snparray_ax_tile!(out, packed, rhs, values, rows_filled)
    row_step = 4096
    column_step = 1024
    @sync begin
        for row_first in 1:row_step:rows_filled
            row_last = min(row_first + row_step - 1, rows_filled)
            Threads.@spawn begin
                for column_first in 1:column_step:size(packed, 2)
                    column_last = min(column_first + column_step - 1,
                                      size(packed, 2))
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

function _snparray_AX_tile!(out, packed, rhs, values, rows_filled)
    row_step = 1024
    column_step = 256
    rhs_step = 256
    @sync begin
        for rhs_first in 1:rhs_step:size(out, 2)
            rhs_last = min(rhs_first + rhs_step - 1, size(out, 2))
            for row_first in 1:row_step:rows_filled
                row_last = min(row_first + row_step - 1, rows_filled)
                Threads.@spawn begin
                    for column_first in 1:column_step:size(packed, 2)
                        column_last = min(column_first + column_step - 1,
                                          size(packed, 2))
                        _snparray_AX_kernel!(
                            out, packed, rhs, values, $row_first, $row_last,
                            column_first, column_last, $rhs_first, $rhs_last,
                        )
                    end
                end
            end
        end
    end
    return out
end

function _snparray_atx_tile!(out, packed, rhs, values, rows_filled)
    row_step = 8192
    column_step = 2048
    @sync begin
        for column_first in 1:column_step:size(packed, 2)
            column_last = min(column_first + column_step - 1,
                              size(packed, 2))
            Threads.@spawn begin
                for row_first in 1:row_step:rows_filled
                    row_last = min(row_first + row_step - 1, rows_filled)
                    _snparray_atx_kernel!(
                        out, packed, rhs, values, row_first, row_last,
                        $column_first, $column_last,
                    )
                end
            end
        end
    end
    return out
end

function _snparray_AtX_tile!(out, packed, rhs, values, rows_filled)
    row_step = 8192
    column_step = 2048
    rhs_step = 2048
    @sync begin
        for rhs_first in 1:rhs_step:size(out, 2)
            rhs_last = min(rhs_first + rhs_step - 1, size(out, 2))
            for column_first in 1:column_step:size(packed, 2)
                column_last = min(column_first + column_step - 1,
                                  size(packed, 2))
                Threads.@spawn begin
                    for row_first in 1:row_step:rows_filled
                        row_last = min(row_first + row_step - 1, rows_filled)
                        _snparray_AtX_kernel!(
                            out, packed, rhs, values, row_first, row_last,
                            $column_first, $column_last,
                            $rhs_first, $rhs_last,
                        )
                    end
                end
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

function _snparray_AX_kernel!(out, packed, rhs, values, row_first, row_last,
                              column_first, column_last, rhs_first, rhs_last)
    return _snparray_AX_scalar!(out, packed, rhs, values, row_first, row_last,
                                column_first, column_last, rhs_first, rhs_last)
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
                               column_first, column_last)
    return _snparray_atx_scalar!(out, packed, rhs, values, row_first, row_last,
                                 column_first, column_last)
end

function _snparray_atx_scalar!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last)
    @inbounds for column in column_first:column_last
        total = out[column]
        for row in row_first:row_last
            total += values[_packed_code(packed, row, column), column] * rhs[row]
        end
        out[column] = total
    end
    return out
end

function _snparray_AtX_kernel!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, rhs_first, rhs_last)
    return _snparray_AtX_scalar!(out, packed, rhs, values, row_first, row_last,
                                 column_first, column_last, rhs_first, rhs_last)
end

function _snparray_AtX_scalar!(out, packed, rhs, values, row_first, row_last,
                               column_first, column_last, rhs_first, rhs_last)
    @inbounds for rhs_column in rhs_first:rhs_last
        for column in column_first:column_last
            total = out[column, rhs_column]
            for row in row_first:row_last
                total += values[_packed_code(packed, row, column), column] *
                         rhs[row, rhs_column]
            end
            out[column, rhs_column] = total
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
