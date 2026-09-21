# Benchmark the streaming K*Q product against the in-memory SnpLinAlg pair.
using SnpArrays
using LinearAlgebra
using Random
using Printf

const R = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 101

function stack_snparray(eur::SnpArray, R::Int)
    m, n = size(eur)
    stacked = SnpArray(undef, R * m, n)
    for r in 1:R
        stacked[(r - 1) * m + 1 : r * m, :] .= view(eur, :, :)
    end
    return stacked
end

# One timed call: minimum of 3 elapsed times, after one warm-up call.
function min_elapsed(f!::F, args...) where F
    f!(args...)
    best = Inf
    for _ in 1:3
        t = @elapsed f!(args...)
        best = min(best, t)
    end
    return best
end

function print_header(m::Int, n::Int, path::AbstractString)
    println("Threads.nthreads() = ", Threads.nthreads())
    println("Sys.CPU_NAME = ", Sys.CPU_NAME)
    println("VERSION = ", VERSION)
    println("samples (m) = ", m)
    println("SNPs (n) = ", n)
    println("file size = ", filesize(path), " bytes")
    if isdefined(SnpArrays, :VECTOR_BYTES)
        println("VECTOR_BYTES = ", SnpArrays.VECTOR_BYTES[])
    else
        println("VECTOR_BYTES = n/a")
    end
end

# Write `stacked` as a standalone .bed/.bim pair at `path` (without the
# .bim extension), reusing an existing .bed if its size already matches.
function write_stacked_bed(stacked::SnpArray, path::AbstractString)
    m, n = size(stacked)
    expected_bytes = 3 + length(stacked.data)
    if isfile(path) && filesize(path) == expected_bytes
        return path
    end
    open(path, "w") do io
        write(io, 0x1b6c)
        write(io, 0x01)
        write(io, stacked.data)
    end
    bim_path = replace(path, ".bed" => ".bim")
    open(bim_path, "w") do io
        for j in 1:n
            println(io, "1\tsnp$(j)\t0\t$(j)\tA\tG")
        end
    end
    return path
end

function report_page_cache(path::AbstractString)
    cached_kb = 0
    open("/proc/meminfo") do io
        for line in eachline(io)
            if startswith(line, "Cached:")
                cached_kb = parse(Int, split(line)[2])
            end
        end
    end
    println("file size = ", filesize(path), " bytes; /proc/meminfo Cached = ",
            cached_kb, " kB (warm-up reads populate the page cache; ",
            "timed calls below are expected to hit it)")
end

# The in-memory pair AᵀQ then A*U, timed together as `mem total`.
function inmemory_pair!(V::Matrix{T}, U::Matrix{T}, sla::SnpLinAlg{T},
                        Q::Matrix{T}) where T
    mul!(U, transpose(sla), Q)
    mul!(V, sla, U)
    return V
end

function run_benchmarks(stacked::SnpArray, path::AbstractString, m::Int, n::Int)
    println()
    @printf("%-8s %4s | %10s %10s %10s | %10s %10s %10s\n",
            "T", "k", "mem AᵀQ", "mem A*U", "mem total", "stream(pf)",
            "stream(no pf)", "stream(mixed)")
    for T in (Float32, Float64)
        sla = SnpLinAlg{T}(stacked; center=true, scale=true, impute=true)
        for k in (8, 64)
            Q = randn(Xoshiro(1), T, m, k)
            U = Matrix{T}(undef, n, k)
            V = Matrix{T}(undef, m, k)

            t_atq = min_elapsed(mul!, U, transpose(sla), Q)
            t_au = min_elapsed(mul!, V, sla, U)

            t_total = min_elapsed(inmemory_pair!, V, U, sla, Q)

            stream = SnpLinAlgStream{T}(path; m=m, width=4096, center=true,
                                        scale=true, impute=true)
            V2 = Matrix{T}(undef, m, k)
            t_stream = min_elapsed(streamed_grm_mul!, V2, stream, Q)

            err = norm(V2 - V ./ n) / norm(V)
            tol = 32 * sqrt(T(max(m, n))) * eps(T)
            println("correctness check T=", T, " k=", k, ": rel err = ", err,
                    " (tol = ", tol, ")")
            err < tol || error("streamed_grm_mul! correctness check failed: " *
                               "rel err $err >= tol $tol for T=$T, k=$k")

            stream_noprefetch = SnpLinAlgStream{T}(path; m=m, width=4096,
                                                   prefetch=false, center=true,
                                                   scale=true, impute=true)
            V3 = Matrix{T}(undef, m, k)
            t_stream_noprefetch = min_elapsed(streamed_grm_mul!, V3,
                                              stream_noprefetch, Q)

            mixed_str = lpad("-", 10)
            if T == Float32
                Q64 = Matrix{Float64}(Q)
                V64 = Matrix{Float64}(undef, m, k)
                t_mixed = min_elapsed(streamed_grm_mul!, V64, stream, Q64)
                err_mixed = norm(V64 - Matrix{Float64}(V) ./ n) /
                            norm(Matrix{Float64}(V) ./ n)
                tol32 = 32 * sqrt(Float32(max(m, n))) * eps(Float32)
                println("correctness check T=", T, " k=", k,
                        " (mixed): rel err = ", err_mixed, " (tol = ", tol32,
                        ")")
                err_mixed < tol32 || error("streamed_grm_mul! mixed-" *
                                           "precision correctness check " *
                                           "failed: rel err $err_mixed >= " *
                                           "tol $tol32 for k=$k")
                mixed_str = @sprintf("%10.3f", t_mixed)
            end
            @printf("%-8s %4d | %10.3f %10.3f %10.3f | %10.3f %10.3f %s\n",
                    T, k, t_atq, t_au, t_total, t_stream,
                    t_stream_noprefetch, mixed_str)
        end
    end
end

function main()
    eur = SnpArray(SnpArrays.datadir("EUR_subset.bed"))
    m, n = size(eur)

    println(stderr, "stacking genotypes (R=", R, ")...")
    t_stack = @elapsed stacked = stack_snparray(eur, R)
    println(stderr, "stacking took ", round(t_stack; digits=2), " s")

    stacked_m = R * m
    path = joinpath(ENV["TMPDIR"], "eur_stacked_$(R).bed")
    write_stacked_bed(stacked, path)

    print_header(stacked_m, n, path)
    report_page_cache(path)

    run_benchmarks(stacked, path, stacked_m, n)
end

main()
