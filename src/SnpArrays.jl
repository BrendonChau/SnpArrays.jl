__precompile__()

module SnpArrays

using CodecZlib, CodecXz, CodecBzip2, CodecZstd,  TranscodingStreams
using Adapt, Glob, LinearAlgebra, LoopVectorization, Missings, Mmap, Printf
using Requires, SparseArrays, Statistics, StatsBase, Random
import Base: IndexStyle, convert, copyto!, eltype, getindex, setindex!, length, size, wait
import DataFrames: DataFrame, rename!, eachrow
import DelimitedFiles: readdlm, writedlm
import CSV # for CSV.read, to avoid clash with Base.read
import LinearAlgebra: copytri!, mul!
import Statistics: mean, mean!, std, var
import StatsBase: counts
import SpecialFunctions: gamma_inc
import VectorizationBase: gesp
import Tables: table
export AbstractSnpArray, AbstractSnpBitMatrix, AbstractSnpLinAlg
export SnpArray, SnpBitMatrix, SnpLinAlg, SnpData, StackedSnpArray
export simulate!
export compress_plink, decompress_plink, split_plink, merge_plink, write_plink 
export counts, grm, grm_admixture, maf, mean, mean!, minorallele
export missingpos, missingrate, missingrate!, std, var, var!, vcf2plink
export kinship_pruning
export ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL
export CuSnpArray
import VariantCallFormat: findgenokey, VCF, header

"""
    ADDITIVE_MODEL

Value for the `model` keyword selecting additive genotype coding:
A1A1 as 0, A1A2 as 1, A2A2 as 2.
"""
const ADDITIVE_MODEL = Val(1)

"""
    DOMINANT_MODEL

Value for the `model` keyword selecting dominant genotype coding:
A1A1 as 0, A1A2 as 1, A2A2 as 1.
"""
const DOMINANT_MODEL = Val(2)

"""
    RECESSIVE_MODEL

Value for the `model` keyword selecting recessive genotype coding:
A1A1 as 0, A1A2 as 0, A2A2 as 1.
"""
const RECESSIVE_MODEL = Val(3)

include("codec.jl")
include("snparray.jl")
include("stackedsnparray.jl")
include("filter.jl")
include("cat.jl")
include("snpdata.jl")
include("grm.jl")
include("kinship_pruning.jl")
include("linalg_direct.jl")
include("linalg_bitmatrix.jl")
include("reorder.jl")
include("vcf2plink.jl")
include("admixture.jl")
include("simulation.jl")

datadir(parts...) = joinpath(@__DIR__, "..", "data", parts...)

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end # module
