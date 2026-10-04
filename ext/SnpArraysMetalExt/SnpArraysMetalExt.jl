module SnpArraysMetalExt

import Base: *
import LinearAlgebra: mul!

using Metal: @metal, MemoryFlagThreadGroup, MtlArray, MtlMatrix,
    MtlThreadGroupArray, MtlVector, simdgroup_barrier,
    simdgroup_index_in_threadgroup, simdgroup_load,
    simdgroup_multiply_accumulate, simdgroup_store, thread_index_in_simdgroup,
    thread_index_in_threadgroup, thread_position_in_threadgroup,
    threadgroup_barrier, threadgroup_position_in_grid, threads_per_threadgroup
using LinearAlgebra: Adjoint, Transpose
using SnpArrays: ADDITIVE_MODEL, DOMINANT_MODEL, MtlSnpArray, RECESSIVE_MODEL,
    SnpArray, _packed_words, mean

include("kernels.jl")
include("simd.jl")
include("linalg.jl")

end # module
