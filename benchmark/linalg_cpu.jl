# CPU benchmark for SnpLinAlg mul! kernels.
using SnpArrays
using LinearAlgebra
using Random
using Printf
using Statistics

const R = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20

function stack_snparray(eur::SnpArray, R::Int)
    m, n = size(eur)
    stacked = SnpArray(undef, R * m, n)
    for r in 1:R
        stacked[(r - 1) * m + 1 : r * m, :] .= view(eur, :, :)
    end
    return stacked
end

# One timed call: minimum of 5 elapsed times, after one warm-up call.
function min_elapsed(f!::F, args...) where F
    f!(args...)
    best = Inf
    for _ in 1:5
        t = @elapsed f!(args...)
        best = min(best, t)
    end
    return best
end

function report(name::AbstractString, seconds::Float64, flops::Float64)
    ms = seconds * 1000
    gfma = flops / seconds / 1e9
    @printf("%s   %10.3f ms   %8.2f GFMA/s\n", name, ms, gfma)
end

function print_header(R::Int, m::Int, n::Int)
    println("Threads.nthreads() = ", Threads.nthreads())
    println("Sys.CPU_NAME = ", Sys.CPU_NAME)
    println("Sys.ARCH = ", Sys.ARCH)
    println("VERSION = ", VERSION)
    println("R = ", R)
    println("samples (m) = ", m)
    println("SNPs (n) = ", n)

    if isdefined(SnpArrays, :VECTOR_BYTES)
        println("VECTOR_BYTES = ", SnpArrays.VECTOR_BYTES[])
    else
        println("VECTOR_BYTES = n/a")
    end

    for T in (Float32, Float64)
        if isdefined(SnpArrays, :_micro_tile)
            println("_micro_tile(", T, ") = ", SnpArrays._micro_tile(T))
        else
            println("_micro_tile(", T, ") = n/a")
        end
    end

    stacked_m = R * m
    for T in (Float32, Float64)
        for k in (1, 8, 32, 128)
            if isdefined(SnpArrays, :_tile_sizes)
                fwd = SnpArrays._tile_sizes(T, stacked_m, n, k, :forward)
                trs = SnpArrays._tile_sizes(T, stacked_m, n, k, :transpose)
                println("_tile_sizes(", T, ", k=", k, ", :forward) = ", fwd)
                println("_tile_sizes(", T, ", k=", k, ", :transpose) = ", trs)
            else
                println("_tile_sizes(", T, ", k=", k, ") = n/a")
            end
        end
    end
end

function run_benchmarks(stacked::SnpArray, R::Int, m::Int, n::Int)
    stacked_m = R * m
    for T in (Float32, Float64)
        A = SnpLinAlg{T}(stacked; center=true, scale=true, impute=true)

        x = randn(MersenneTwister(1), T, n)
        out = Vector{T}(undef, stacked_m)
        seconds = min_elapsed(mul!, out, A, x)
        report(@sprintf("%-8s A*x    k=1  ", T), seconds, Float64(stacked_m) * n)

        y = randn(MersenneTwister(1), T, stacked_m)
        out_t = Vector{T}(undef, n)
        seconds = min_elapsed(mul!, out_t, transpose(A), y)
        report(@sprintf("%-8s Aᵀ*y   k=1  ", T), seconds, Float64(stacked_m) * n)

        for k in (8, 32, 128)
            X = randn(MersenneTwister(1), T, n, k)
            Out = Matrix{T}(undef, stacked_m, k)
            seconds = min_elapsed(mul!, Out, A, X)
            report(@sprintf("%-8s A*X    k=%-4d", T, k),
                   seconds, Float64(stacked_m) * n * k)
        end

        for k in (8, 32, 128)
            Y = randn(MersenneTwister(1), T, stacked_m, k)
            Out_t = Matrix{T}(undef, n, k)
            seconds = min_elapsed(mul!, Out_t, transpose(A), Y)
            report(@sprintf("%-8s Aᵀ*Y   k=%-4d", T, k),
                   seconds, Float64(stacked_m) * n * k)
        end
    end
end

function main()
    eur = SnpArray(SnpArrays.datadir("EUR_subset.bed"))
    m, n = size(eur)

    print_header(R, m, n)

    println(stderr, "stacking genotypes (R=", R, ")...")
    t_stack = @elapsed stacked = stack_snparray(eur, R)
    println(stderr, "stacking took ", round(t_stack; digits=2), " s")

    run_benchmarks(stacked, R, m, n)
end

main()
