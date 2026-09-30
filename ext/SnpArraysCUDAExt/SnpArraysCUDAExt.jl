module SnpArraysCUDAExt

import Base: *
import LinearAlgebra: mul!

using CUDACore: @cuda, CuArray, CuDynamicSharedArray, CuMatrix,
    CuStaticSharedArray, CuVector, DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, attribute, attributes,
    blockDim, blockIdx, device, shfl_down_sync, sync_threads, threadIdx,
    unsafe_free!
using LinearAlgebra: Adjoint, Transpose
using SnpArrays: ADDITIVE_MODEL, CuSnpArray, DOMINANT_MODEL, RECESSIVE_MODEL,
    SnpArray, _packed_words, mean

include("kernels.jl")
include("lookup.jl")
include("linalg.jl")

end # module
