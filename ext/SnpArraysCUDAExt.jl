module SnpArraysCUDAExt

import LinearAlgebra: mul!

using Adapt: adapt
using CUDA
using LinearAlgebra: Adjoint, Transpose, dot
using SnpArrays: CuSnpArray, SnpArray, ADDITIVE_MODEL, DOMINANT_MODEL,
    RECESSIVE_MODEL, mean

function CuSnpArray{T}(s::SnpArray; 
    model = ADDITIVE_MODEL,
    center::Bool = false,
    scale::Bool = false, impute::Bool = false) where T <: AbstractFloat

    data = adapt(CuMatrix{UInt8}, s.data)
    if model == ADDITIVE_MODEL
        if center || scale
            μ = Vector{T}(undef, size(s, 2))
            μ[:] = mean(s; dims=1, model=ADDITIVE_MODEL)
        else
            μ = T[]
        end
        if scale
            σinv = Vector{T}(undef, size(s, 2))
            @inbounds @simd for j in 1:size(s, 2)
                σinv[j] = sqrt(μ[j] * (1 - μ[j] / 2))
                σinv[j] = σinv[j] > 0 ? inv(σinv[j]) : one(T)
            end
        else
            σinv = T[]
        end
    elseif model == DOMINANT_MODEL
        if center || scale
            μ = Vector{T}(undef, size(s, 2))
            μ[:] = mean(s; dims=1, model=DOMINANT_MODEL)
        else
            μ = T[]
        end
        if scale
            σinv = Vector{T}(undef, size(s, 2))
            @inbounds @simd for j in 1:size(s, 2)
                σinv[j] = sqrt(μ[j] * (1 - μ[j]))
                σinv[j] = σinv[j] > 0 ? inv(σinv[j]) : one(T)
            end
        else
            σinv = T[]
        end
    elseif model == RECESSIVE_MODEL
        if center || scale
            μ = Vector{T}(undef, size(s, 2))
            μ[:] = mean(s; dims=1, model=RECESSIVE_MODEL)
        else
            μ = T[]
        end
        if scale
            σinv = Vector{T}(undef, size(s, 2))
            @inbounds @simd for j in 1:size(s, 2)
                σinv[j] = sqrt(μ[j] * (1 - μ[j]))
                σinv[j] = σinv[j] > 0 ? inv(σinv[j]) : one(T)
            end
        else
            σinv = T[]
        end
    else
        throw(ArgumentError("unrecognized model $model"))
    end
    μ = adapt(CuVector{T}, μ)
    σinv = adapt(CuVector{T}, σinv)
    storagev1 = CuVector{T}(undef, size(s, 1))
    storagev2 = CuVector{T}(undef, size(s, 2))
    CuSnpArray{T}(data, s.m, model, center, scale, impute, μ, σinv, storagev1, storagev2)
end

for (_ftn!, expr) in [
        (:_snparray_cuda_ax_additive!, :(((Aij >= 2) + (Aij >= 3)) * v[j])),
        (:_snparray_cuda_ax_dominant!, :((Aij >= 2) * v[j])),
        (:_snparray_cuda_ax_recessive!, :((Aij >= 3) * v[j])),
        (:_snparray_cuda_ax_additive_meanimpute!, :(((Aij >= 2) + (Aij >= 3) + (Aij == 1) * μ[j]) * v[j])),
        (:_snparray_cuda_ax_dominant_meanimpute!, :(((Aij >= 2) + (Aij == 1) * μ[j]) * v[j])),
        (:_snparray_cuda_ax_recessive_meanimpute!, :(((Aij >= 3) + (Aij == 1) * μ[j]) * v[j]))
    ]
    @eval begin
        function $_ftn!(out, s, v)
            packedstride = size(s, 1)
            index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            stride_x = blockDim().x * gridDim().x
            for i = index_x:stride_x:length(out)
                outi = zero(eltype(out))
                for j = 1:length(v)
                    k = 2 * ((i-1) & 3)
                    block = s[(j-1) * packedstride + ((i-1) >> 2) + 1]
                    Aij = (block >> k) & 3
                    @inbounds outi += $expr
                end
                out[i] = outi
            end
        end
    end
end

for (_ftn!, expr) in [
        (:_snparray_cuda_atx_additive!, :((((Aij >= 2) + (Aij >= 3))) * v[j])),
        (:_snparray_cuda_atx_dominant!, :((Aij >= 2) * v[j])),
        (:_snparray_cuda_atx_recessive!, :((Aij >= 3) * v[j])),
        (:_snparray_cuda_atx_additive_meanimpute!, :(((Aij >= 2) + (Aij >= 3) + (Aij == 1) * μ[i]) * v[j])),
        (:_snparray_cuda_atx_dominant_meanimpute!, :(((Aij >= 2) + (Aij == 1) * μ[i]) * v[j])),
        (:_snparray_cuda_atx_recessive_meanimpute!, :(((Aij >= 3) + (Aij == 1) * μ[i]) * v[j]))
    ]
    @eval begin
        function $_ftn!(out, s, v)
            packedstride = size(s, 1)
            index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            stride_x = blockDim().x * gridDim().x
            for i = index_x:stride_x:length(out)
                outi = zero(eltype(out))
                for j = 1:length(v)
                    k = 2 * ((j-1) & 3)
                    block = s[(i-1) * packedstride + ((j-1) >> 2) + 1]
                    Aij = (block >> k) & 3
                    @inbounds outi += $expr
                end
                out[i] = outi
            end
        end
    end
end

"""
    LinearAlgebra.mul!(out::CuVector{T}, s::CuSnpArray{T}, v::CuVector{T})

In-place matrix-vector multiplication on a GPU.
"""
function mul!(
    out::CuVector{T}, 
    s::CuSnpArray{T}, 
    v::CuVector{T}) where T <: AbstractFloat
    @assert length(out) == size(s, 1) && length(v) == size(s, 2)
    fill!(out, zero(T))
    if s.scale
        s.storagev2 .= s.σinv .* v
        w = s.storagev2
    else
        w = v
    end
    numblocks = ceil(Int, length(out)/256)
    CUDA.@sync begin
        if s.model == ADDITIVE_MODEL
            @cuda threads=256 blocks=numblocks _snparray_cuda_ax_additive!(out, s.data, w)
        elseif s.model == DOMINANT_MODEL
            @cuda threads=256 blocks=numblocks _snparray_cuda_ax_dominant!(out, s.data, w) 
        else
            @cuda threads=256 blocks=numblocks _snparray_cuda_ax_recessive!(out, s.data, w) 
        end  
    end         

    if s.center
        return out .-= dot(s.μ, w)
    else
        return out
    end
end

"""
    LinearAlgebra.mul!(out::CuVector{T}, s::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}}, v::CuVector{T})

In-place matrix-vector multiplication on a GPU, with transposed CuSnpArray.
"""
function mul!(
    out::CuVector{T}, 
    st::Union{Transpose{T, <:CuSnpArray{T}}, Adjoint{T, <:CuSnpArray{T}}},
    v::CuVector{T}) where T <: AbstractFloat
    @assert length(out) == size(st, 1) && length(v) == size(st, 2)
    fill!(out, zero(T))
    s = st.parent
    numblocks = ceil(Int, length(out)/256)
    CUDA.@sync begin
        if !s.impute
            if s.model == ADDITIVE_MODEL
                @cuda threads=256 blocks=numblocks _snparray_cuda_atx_additive!(out, s.data, v)
            elseif s.model == DOMINANT_MODEL
                @cuda threads=256 blocks=numblocks _snparray_cuda_atx_dominant!(out, s.data, v) 
            else
                @cuda threads=256 blocks=numblocks _snparray_cuda_atx_recessive!(out, s.data, v) 
            end
        else
            if s.model == ADDITIVE_MODEL
                @cuda threads=256 blocks=numblocks _snparray_cuda_atx_additive_meanimpute!(out, s.data, v, s.μ)
            elseif s.model == DOMINANT_MODEL
                @cuda threads=256 blocks=numblocks _snparray_cuda_atx_dominant_meanimpute!(out, s.data, v, s.μ) 
            else
                @cuda threads=256 blocks=numblocks _snparray_cuda_atx_recessive_meanimpute!(out, s.data, v, s.μ) 
            end
        end
    end

    if s.center
        out .-= sum(v) .* s.μ
    end
    if s.scale
        return out .*= s.σinv
    else
        return out
    end
end

end # module
