# Time SnpLinAlg A*X and Aᵀ*X with K right-hand sides on a PLINK bed file and
# append one CSV row per type and product to `csv`.
#
#     julia -t N --project=. benchmark/matmul.jl \
#       [bed] [k] [repeats] [csv] [label]
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

const BED = length(ARGS) >= 1 ? ARGS[1] :
    "/u/scratch/b/bhchau/cudaext_check/synthetic_v1_chr-21.bed"
const K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const REPEATS = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
const CSV = length(ARGS) >= 4 ? ARGS[4] :
    joinpath(@__DIR__, "results", "matmul.csv")
const LABEL = length(ARGS) >= 5 ? ARGS[5] : ""
const HEADER = "timestamp,label,cpu,threads,bed,m,n,k,type,product,repeats," *
               "min_ms,median_ms,gfma_per_s,sum_abs2"

"""
    time_call(f!, args...; repeats = 5) -> (minimum, median)

Time `f!(args...)` in seconds, after one warm-up call.
"""
function time_call(f!::F, args...; repeats::Int = 5) where F
    f!(args...)
    times = [@elapsed f!(args...) for _ in 1:repeats]
    sort!(times)
    return times[1], times[(repeats + 1) ÷ 2]
end

"""
    report(io, stamp, ::Type{T}, product, dims, times, out)

Print one result line and append its CSV row to `io`, then flush `io`.
"""
function report(
    io::IO,
    stamp::AbstractString,
    ::Type{T},
    product::AbstractString,
    dims::NTuple{2, Int},
    times::NTuple{2, Float64},
    out::AbstractMatrix{T},
) where T
    m, n = dims
    gfma = m * n * K / times[1] / 1e9
    checksum = @sprintf("%.17g", sum(abs2, out))
    @printf("%-8s %-5s min %10.3f ms  median %10.3f ms  %8.2f GFMA/s  %s\n",
            T, product, 1000 * times[1], 1000 * times[2], gfma, checksum)
    println(io, join((stamp, LABEL, Sys.CPU_NAME, Threads.nthreads(),
                      basename(BED), m, n, K, T, product, REPEATS,
                      @sprintf("%.6f", 1000 * times[1]),
                      @sprintf("%.6f", 1000 * times[2]),
                      @sprintf("%.4f", gfma), checksum), ","))
    flush(io)
    return nothing
end

"""
    bench_type(io, stamp, G, ::Type{T})

Time A*X and Aᵀ*X for the standardized `SnpLinAlg{T}` of `G`.
"""
function bench_type(
    io::IO,
    stamp::AbstractString,
    G::SnpArray,
    ::Type{T},
) where T
    m, n = size(G)
    operator = SnpLinAlg{T}(G; center = true, scale = true, impute = true)
    X = randn(Xoshiro(1), T, n, K)
    Y = randn(Xoshiro(2), T, m, K)
    AX = Matrix{T}(undef, m, K)
    AtX = Matrix{T}(undef, n, K)
    times = time_call(mul!, AX, operator, X; repeats = REPEATS)
    report(io, stamp, T, "A*X", (m, n), times, AX)
    times = time_call(mul!, AtX, transpose(operator), Y; repeats = REPEATS)
    report(io, stamp, T, "Aᵀ*X", (m, n), times, AtX)
    return nothing
end

"""
    main()

Benchmark both element types on `BED` and append the rows to `CSV`.
"""
function main()
    G = SnpArray(BED)
    m, n = size(G)
    println("bed: $BED\nm = $m, n = $n, k = $K, repeats = $REPEATS")
    println("threads = $(Threads.nthreads()), cpu = $(Sys.CPU_NAME), " *
            "julia = $VERSION")
    stamp = Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")
    mkpath(dirname(CSV))
    new_file = !isfile(CSV) || filesize(CSV) == 0
    io = open(CSV, "a")
    if new_file
        println(io, HEADER)
        flush(io)
    end
    for T in (Float32, Float64)
        bench_type(io, stamp, G, T)
    end
    close(io)
    println("csv: $CSV")
    return nothing
end

main()
