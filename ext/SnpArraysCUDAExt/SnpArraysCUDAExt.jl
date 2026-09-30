module SnpArraysCUDAExt

import Base: *
import LinearAlgebra: mul!

using CUDACore: @cuda, CuArray, CuDynamicSharedArray, CuEvent, CuMatrix,
    CuStaticSharedArray, CuVector, DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, attribute, attributes,
    WMMA, blockDim, blockIdx, context, context!, device, gridDim, pin,
    record, shfl_down_sync, sync_threads, synchronize, threadIdx,
    unsafe_free!
using LinearAlgebra: Adjoint, Transpose
using SnpArrays: ADDITIVE_MODEL, CuSnpArray, DOMINANT_MODEL, RECESSIVE_MODEL,
    SnpArray, SnpLinAlgStream, _packed_words, mean
import SnpArrays: streamed_mul!

include("kernels.jl")
include("lookup.jl")
include("wmma.jl")
include("linalg.jl")
include("stream.jl")

end # module
