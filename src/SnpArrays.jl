__precompile__()

module SnpArrays

using CodecZlib, CodecXz, CodecBzip2, CodecZstd,  TranscodingStreams
using Glob, LinearAlgebra, Missings, Mmap, Printf, SIMD
using SparseArrays, Statistics, StatsBase, Random
import Base: IndexStyle, convert, copyto!, eltype, getindex, setindex!, length, size
import DataFrames: DataFrame, rename!, eachrow
import DelimitedFiles: readdlm, writedlm
import CSV # for CSV.read, to avoid clash with Base.read
import LinearAlgebra: copytri!, mul!
import Statistics: mean, mean!, std, var
import StatsBase: counts
import SpecialFunctions: gamma_inc
import Tables: table
export AbstractSnpArray, AbstractSnpBitMatrix, AbstractSnpLinAlg
export SnpArray, SnpBitMatrix, SnpLinAlg, SnpLinAlgStream, SnpData, StackedSnpArray
export streamed_grm_mul!, streamed_mul!
export simulate!
export compress_plink, decompress_plink, split_plink, merge_plink, write_plink 
export counts, grm, grm_admixture, maf, mean, mean!, minorallele
export missingpos, missingrate, missingrate!, std, var, var!, vcf2plink
export kinship_pruning
export ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL
export CuSnpArray
import VariantCallFormat: findgenokey, VCF, header

const ADDITIVE_MODEL = Val(1)
const DOMINANT_MODEL = Val(2)
const RECESSIVE_MODEL = Val(3)

include("codec.jl")
include("task_partition.jl")
include("snparray.jl")
include("stackedsnparray.jl")
include("snparray_statistics.jl")
include("snparray_conversion.jl")
include("filter.jl")
include("cat.jl")
include("snpdata.jl")
include("grm.jl")
include("kinship_pruning.jl")
include("linalg/snplinalg.jl")
include("linalg/tuning.jl")
include("linalg/api.jl")
include("linalg/schedule.jl")
include("linalg/schedule_lookup.jl")
include("linalg/matvec.jl")
include("linalg/matmul.jl")
include("linalg/matmul_AX.jl")
include("linalg/matmul_AtX.jl")
include("linalg/matmul_lookup.jl")
include("linalg/scalar.jl")
include("streaming/snplinalgstream.jl")
include("streaming/chunk_reader.jl")
include("streaming/chunk_iterator.jl")
include("streaming/streamed_mul.jl")
include("linalg/bitmatrix.jl")
include("reorder.jl")
include("vcf2plink.jl")
include("admixture.jl")
include("simulation.jl")
include("cuda_types.jl")

datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)

function __init__()
    VECTOR_BYTES[] = _detect_vector_bytes()
end

end # module
