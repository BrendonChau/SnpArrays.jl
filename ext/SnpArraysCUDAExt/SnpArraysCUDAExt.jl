module SnpArraysCUDAExt

import Base: *
import LinearAlgebra: mul!

using CUDACore: @cuda, CuArray, CuMatrix, CuStaticSharedArray, CuVector,
    DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, attribute, blockDim, blockIdx,
    device, shfl_down_sync, sync_threads, threadIdx, unsafe_free!
using LinearAlgebra: Adjoint, Transpose
using SnpArrays: ADDITIVE_MODEL, CuSnpArray, DOMINANT_MODEL, RECESSIVE_MODEL,
    SnpArray, _packed_words, mean

include("kernels.jl")
include("linalg.jl")

end # module
