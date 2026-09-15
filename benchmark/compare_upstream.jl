# Benchmark SnpLinAlg mul! against whichever SnpArrays the active project
# provides, so this fork and the registered package can be compared.
#
# Set up the comparison environment once:
#
#     julia --project=/path/to/scratch -e 'using Pkg; Pkg.add("SnpArrays")'
#
# then run the same script against each, writing results to distinct
# prefixes:
#
#     JULIA_NUM_THREADS=12 julia --project=lib/SnpArrays \
#       lib/SnpArrays/benchmark/compare_upstream.jl 20 /tmp/fork
#     JULIA_NUM_THREADS=12 julia --project=/path/to/scratch \
#       lib/SnpArrays/benchmark/compare_upstream.jl 20 /tmp/upstream
#
# Both runs read the genotypes from this repository regardless of which
# SnpArrays is loaded, and each writes `<prefix>_verify.bin`. Compare those
# files before comparing timings: a speed difference is only meaningful if
# the two packages agree.
using LinearAlgebra
using Printf
using Random
using SnpArrays

const R = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const PREFIX = length(ARGS) >= 2 ? ARGS[2] : "compare"
const BED = joinpath(@__DIR__, "..", "data", "EUR_subset.bed")

function stack_snparray(eur::SnpArray, R::Int)
    m, n = size(eur)
    stacked = SnpArray(undef, R * m, n)
    for r in 1:R
        stacked[(r - 1) * m + 1:r * m, :] .= view(eur, :, :)
    end
    return stacked
end

# One timed call: minimum of `repeats` elapsed times, after one warm-up call.
function min_elapsed(f!::F, args...; repeats::Int = 5) where F
    f!(args...)
    best = Inf
    for _ in 1:repeats
        best = min(best, @elapsed f!(args...))
    end
    return best
end

function report(name::AbstractString, seconds::Float64, flops::Float64)
    @printf("%s   %10.3f ms   %8.2f GFMA/s\n", name, seconds * 1000,
            flops / seconds / 1e9)
    return nothing
end

"""
    write_verification(path, eur)

Write the four products on the unstacked fixture as `Float64`, so two
packages can be checked for agreement across processes.
"""
function write_verification(path::AbstractString, eur::SnpArray)
    m, n = size(eur)
    k = 8
    operator = SnpLinAlg{Float64}(eur; center = true, scale = true,
                                  impute = true)
    x = randn(Xoshiro(11), Float64, n)
    y = randn(Xoshiro(12), Float64, m)
    X = randn(Xoshiro(13), Float64, n, k)
    Y = randn(Xoshiro(14), Float64, m, k)
    open(path, "w") do io
        write(io, mul!(Vector{Float64}(undef, m), operator, x))
        write(io, mul!(Vector{Float64}(undef, n), transpose(operator), y))
        write(io, mul!(Matrix{Float64}(undef, m, k), operator, X))
        write(io, mul!(Matrix{Float64}(undef, n, k), transpose(operator), Y))
    end
    return nothing
end

function run_benchmarks(stacked::SnpArray, rows::Int, n::Int)
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(stacked; center = true, scale = true,
                                impute = true)

        x = randn(Xoshiro(1), T, n)
        out = Vector{T}(undef, rows)
        seconds = min_elapsed(mul!, out, operator, x)
        report(@sprintf("%-8s A*x    k=1  ", T), seconds, Float64(rows) * n)

        y = randn(Xoshiro(2), T, rows)
        out_t = Vector{T}(undef, n)
        seconds = min_elapsed(mul!, out_t, transpose(operator), y)
        report(@sprintf("%-8s Aᵀ*y   k=1  ", T), seconds, Float64(rows) * n)

        for k in (8, 32, 128)
            X = randn(Xoshiro(3), T, n, k)
            out_matrix = Matrix{T}(undef, rows, k)
            seconds = min_elapsed(mul!, out_matrix, operator, X)
            report(@sprintf("%-8s A*X    k=%-4d", T, k), seconds,
                   Float64(rows) * n * k)
        end

        for k in (8, 32, 128)
            Y = randn(Xoshiro(4), T, rows, k)
            out_tmatrix = Matrix{T}(undef, n, k)
            seconds = min_elapsed(mul!, out_tmatrix, transpose(operator), Y)
            report(@sprintf("%-8s Aᵀ*Y   k=%-4d", T, k), seconds,
                   Float64(rows) * n * k)
        end
    end
    return nothing
end

function main()
    eur = SnpArray(BED)
    m, n = size(eur)
    println("pkgdir = ", pkgdir(SnpArrays))
    println("nthreads = ", Threads.nthreads())
    println("CPU_NAME = ", Sys.CPU_NAME)
    println("R = ", R, "   m = ", R * m, "   n = ", n)

    write_verification(PREFIX * "_verify.bin", eur)

    stacked = stack_snparray(eur, R)
    run_benchmarks(stacked, R * m, n)
end

main()
