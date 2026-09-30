# Benchmark the streaming AᵀY product against the mapped SnpLinAlg on a
# large-m .bed file, with per-chunk stage timings and cgroup/page-cache
# figures.
using SnpArrays
using LinearAlgebra
using Random
using Printf
using Statistics: median

# Path, sample count, and SNP count of the benchmarked `.bed` file, passed
# through the timing helpers instead of closing over globals.
struct BenchTarget
    path::String
    m::Int
    n::Int
end

"""
    cgroup_dir() -> String

Return this job's cgroup v2 directory under the Slurm step's scope.
"""
cgroup_dir() = "/sys/fs/cgroup/system.slice/slurmstepd.scope/job_" *
               ENV["SLURM_JOB_ID"]

"""
    cgroup_bytes() -> (current, peak)

Read `memory.current` and `memory.peak` from this job's cgroup directory,
returning `(-1, -1)` when the cgroup files are not present.
"""
function cgroup_bytes()
    dir = cgroup_dir()
    current_path = joinpath(dir, "memory.current")
    peak_path = joinpath(dir, "memory.peak")
    (isfile(current_path) && isfile(peak_path)) || return (-1, -1)
    current = parse(Int, strip(read(current_path, String)))
    peak = parse(Int, strip(read(peak_path, String)))
    return (current, peak)
end

"""
    fincore_bytes(path) -> Int

Return the resident page-cache bytes of `path`, from the `fincore` tool,
returning `-1` when `fincore` is not on the `PATH`.
"""
function fincore_bytes(path::AbstractString)
    Sys.which("fincore") === nothing && return -1
    output = read(`fincore --bytes --output RES --noheadings $path`, String)
    return parse(Int, strip(output))
end

"""
    timed(label, target, f, args...) -> Float64

Run `f(args...)`, print one line with the elapsed time and the cgroup and
page-cache figures before and after, and return the elapsed time.
"""
function timed(label::AbstractString, target::BenchTarget, f::F,
              args...) where F
    (cur0, peak0) = cgroup_bytes()
    fc0 = fincore_bytes(target.path)
    t = @elapsed f(args...)
    (cur1, peak1) = cgroup_bytes()
    fc1 = fincore_bytes(target.path)
    gib = 1024.0^3
    @printf(
        "%-28s %10.3f s | mem.current %6.2f -> %6.2f GiB | mem.peak %6.2f GiB | fincore %6.2f -> %6.2f GiB\n",
        label, t, cur0 / gib, cur1 / gib, peak1 / gib, fc0 / gib, fc1 / gib,
    )
    return t
end

"""
    print_header(target, k, width, passes, ::Type{T})

Print the run configuration: thread and CPU identity, problem sizes, the
target file size, the Slurm job id, and the initial cgroup/fincore figures.
"""
function print_header(target::BenchTarget, k::Int, width::Int, passes::Int,
                      ::Type{T}) where T
    println("Threads.nthreads() = ", Threads.nthreads())
    println("Sys.CPU_NAME = ", Sys.CPU_NAME)
    println("VERSION = ", VERSION)
    println("VECTOR_BYTES = ", SnpArrays.VECTOR_BYTES[])
    println("default readers = ", SnpArrays._default_readers())
    println("m = ", target.m)
    println("n = ", target.n)
    println("k = ", k)
    println("width = ", width)
    println("passes = ", passes)
    println("T = ", T)
    println("file size = ", filesize(target.path), " bytes")
    slurm_job_id = get(ENV, "SLURM_JOB_ID", "")
    println("SLURM_JOB_ID = ", slurm_job_id)
    dir = cgroup_dir()
    memory_max_path = joinpath(dir, "memory.max")
    memory_max = isfile(memory_max_path) ?
                 strip(read(memory_max_path, String)) : "n/a"
    println("cgroup memory.max = ", memory_max)
    (cur0, peak0) = cgroup_bytes()
    fc0 = fincore_bytes(target.path)
    gib = 1024.0^3
    @printf("initial mem.current %.2f GiB, mem.peak %.2f GiB, fincore %.2f GiB\n", cur0 / gib, peak0 / gib, fc0 / gib)
    return nothing
end

"""
    sweep!(U, stream, Y)

Fill `U` with `transpose(A) * Y` for the streamed genotype matrix `A`,
one chunk at a time, through a per-width scratch buffer.
"""
function sweep!(U::Matrix{T}, stream::SnpLinAlgStream{T},
                Y::Matrix{T}) where T
    k = size(Y, 2)
    buffers = Dict{Int, Matrix{T}}()
    for (cols, chunk) in stream
        width = length(cols)
        piece = get!(buffers, width) do
            Matrix{T}(undef, width, k)
        end
        mul!(piece, transpose(chunk), Y)
        copyto!(view(U, cols, :), piece)
    end
    return U
end

"""
    scan!(sla_ref, target, ::Type{T})

Build the mapped `SnpLinAlg{T}` for `target`, computing its column
statistics in the constructor (the "mapped scan"), and store it in
`sla_ref` for the caller to retrieve outside the timed call.
"""
function scan!(sla_ref::Ref{Any}, target::BenchTarget, ::Type{T}) where T
    s = SnpArray(target.path)
    sla_ref[] = SnpLinAlg{T}(s; center=true, scale=true, impute=true)
    return sla_ref
end

"""
    run_rounds(target, k, width, passes, ::Type{T})
        -> (t_scan, t_map, t_s1, t_sd)

Run `passes` rounds of the mapped scan and mul!, and the readers=1 and
readers=default streamed sweeps, checking each streamed result against the
mapped result. Return the per-round elapsed times.
"""
function run_rounds(target::BenchTarget, k::Int, width::Int, passes::Int,
                    ::Type{T}) where T
    m = target.m
    n = target.n
    Y = randn(Xoshiro(1), T, m, k)
    U_map = Matrix{T}(undef, n, k)
    U_s1 = Matrix{T}(undef, n, k)
    U_sd = Matrix{T}(undef, n, k)
    stream1 = SnpLinAlgStream{T}(target.path; m=m, width=width, readers=1,
                                 center=true, scale=true, impute=true)
    default_readers = SnpArrays._default_readers()
    stream_default = SnpLinAlgStream{T}(target.path; m=m, width=width,
                                        readers=default_readers, center=true,
                                        scale=true, impute=true)
    t_scan = Vector{Float64}(undef, passes)
    t_map = Vector{Float64}(undef, passes)
    t_s1 = Vector{Float64}(undef, passes)
    t_sd = Vector{Float64}(undef, passes)
    sla_ref = Ref{Any}()
    tol = 32 * sqrt(T(max(m, n))) * eps(T)
    for round in 1:passes
        t_scan[round] = timed("mapped scan (round $round)", target, scan!,
                              sla_ref, target, T)
        sla = sla_ref[]::SnpLinAlg{T}
        t_map[round] = timed("mapped mul! (round $round)", target, mul!,
                             U_map, transpose(sla), Y)
        t_s1[round] = timed("streamed readers=1 (round $round)", target,
                            sweep!, U_s1, stream1, Y)
        err1 = norm(U_s1 - U_map) / norm(U_map)
        println("correctness readers=1 (round $round): rel err = ", err1,
                " (tol = ", tol, ")")
        err1 < tol || error("streamed readers=1 correctness check failed: " *
                            "rel err $err1 >= tol $tol (round $round)")

        t_sd[round] = timed(
            "streamed readers=$default_readers (round $round)", target,
            sweep!, U_sd, stream_default, Y,
        )
        errd = norm(U_sd - U_map) / norm(U_map)
        println("correctness readers=$default_readers (round $round): " *
                "rel err = ", errd, " (tol = ", tol, ")")
        errd < tol || error(
            "streamed readers=$default_readers correctness check failed: " *
            "rel err $errd >= tol $tol (round $round)",
        )

        ratio_map = t_sd[round] / t_map[round]
        ratio_s1 = t_sd[round] / t_s1[round]
        @printf("ratio streamed(default)/mapped mul! = %.3f\n", ratio_map)
        @printf("ratio streamed(default)/streamed(readers=1) = %.3f\n",
                ratio_s1)
    end
    sla_ref[] = nothing
    return (t_scan, t_map, t_s1, t_sd)
end

"""
    print_summary(t_map, t_s1, t_sd, passes)

Print the per-round timing table and the P1/P3 median ratios.
"""
function print_summary(t_map::Vector{Float64}, t_s1::Vector{Float64},
                       t_sd::Vector{Float64}, t_scan::Vector{Float64},
                       passes::Int)
    println()
    @printf("%-6s %12s %12s %12s %12s %10s %10s\n", "round", "mapped scan",
            "mapped mul!", "streamed r1", "streamed rd", "ratio P1",
            "ratio P3")
    ratios_p1 = Vector{Float64}(undef, passes)
    ratios_p3 = Vector{Float64}(undef, passes)
    for round in 1:passes
        ratios_p1[round] = t_sd[round] / t_map[round]
        ratios_p3[round] = t_sd[round] / t_s1[round]
        @printf("%-6d %12.3f %12.3f %12.3f %12.3f %10.3f %10.3f\n", round,
                t_scan[round], t_map[round], t_s1[round], t_sd[round],
                ratios_p1[round], ratios_p3[round])
    end
    @printf("P1 median ratio streamed(default)/mapped = %.3f\n",
            median(ratios_p1))
    @printf("P3 median ratio default/readers=1 = %.3f\n", median(ratios_p3))
    return nothing
end

"""
    forward_sweep!(V, stream, U)

Fill `V` with `A * U` for the streamed genotype matrix `A`, accumulating
each chunk's product with the matching rows of `U` copied into a per-width
scratch buffer.
"""
function forward_sweep!(V::Matrix{T}, stream::SnpLinAlgStream{T},
                        U::Matrix{T}) where T
    k = size(U, 2)
    buffers = Dict{Int, Matrix{T}}()
    fill!(V, zero(T))
    for (cols, chunk) in stream
        piece = get!(buffers, length(cols)) do
            Matrix{T}(undef, length(cols), k)
        end
        copyto!(piece, view(U, cols, :))
        mul!(V, chunk, piece, one(T), one(T))
    end
    return V
end

"""
    run_forward_rounds(target, k, width, passes, ::Type{T})

Run `passes` rounds of the mapped `A * U` and the readers=1 and
readers=default streamed forward sweeps, checking each streamed result
against the mapped result, and print the per-round times and ratios.
"""
function run_forward_rounds(target::BenchTarget, k::Int, width::Int,
                            passes::Int, ::Type{T}) where T
    println()
    m = target.m
    n = target.n
    U = randn(Xoshiro(2), T, n, k)
    V_map = Matrix{T}(undef, m, k)
    V_s1 = Matrix{T}(undef, m, k)
    V_sd = Matrix{T}(undef, m, k)
    default_readers = SnpArrays._default_readers()
    stream1 = SnpLinAlgStream{T}(target.path; m=m, width=width, readers=1,
                                 center=true, scale=true, impute=true)
    stream_default = SnpLinAlgStream{T}(target.path; m=m, width=width,
                                        readers=default_readers, center=true,
                                        scale=true, impute=true)
    sla = SnpLinAlg{T}(SnpArray(target.path); center=true, scale=true,
                       impute=true)
    tol = 32 * sqrt(T(max(m, n))) * eps(T)
    t_map = Vector{Float64}(undef, passes)
    t_s1 = Vector{Float64}(undef, passes)
    t_sd = Vector{Float64}(undef, passes)
    for round in 1:passes
        t_map[round] = timed("mapped A*U (round $round)", target, mul!,
                             V_map, sla, U)
        for (label, stream, V, t) in
            (("readers=1", stream1, V_s1, t_s1),
             ("readers=$default_readers", stream_default, V_sd, t_sd))
            t[round] = timed("streamed A*U $label (round $round)", target,
                             forward_sweep!, V, stream, U)
            err = norm(V - V_map) / norm(V_map)
            println("correctness A*U $label (round $round): rel err = ", err,
                    " (tol = ", tol, ")")
            err < tol || error("streamed A*U $label correctness check " *
                               "failed: rel err $err >= tol $tol")
        end
    end
    @printf("%-6s %12s %12s %12s %10s %10s\n", "round", "mapped A*U",
            "streamed r1", "streamed rd", "rd/mapped", "rd/r1")
    for round in 1:passes
        @printf("%-6d %12.3f %12.3f %12.3f %10.3f %10.3f\n", round,
                t_map[round], t_s1[round], t_sd[round],
                t_sd[round] / t_map[round], t_sd[round] / t_s1[round])
    end
    @printf("A*U median ratio streamed(default)/mapped = %.3f\n",
            median(t_sd ./ t_map))
    return nothing
end

"""
    refill_once!(buffer)

Zero `buffer`'s cached counts and refill its statistics, timing only the
refill stage.
"""
function refill_once!(buffer::SnpLinAlg)
    fill!(buffer.s.columncounts, 0)
    fill!(buffer.s.rowcounts, 0)
    SnpArrays._refill_statistics!(buffer)
    return buffer
end

"""
    min5(f, args...) -> (min, median, max)

Run `f(args...)` 5 times, timing each call with `@elapsed`, and return the
minimum, median, and maximum elapsed time.
"""
function min5(f::F, args...) where F
    times = [@elapsed(f(args...)) for _ in 1:5]
    return (minimum(times), median(times), maximum(times))
end

"""
    read_chunk_seek!(io, buffer)

Seek `io` to the first chunk's byte offset, then read and refill it, so
repeated timed calls always re-read the same chunk.
"""
function read_chunk_seek!(io::IO, buffer::SnpLinAlg)
    seek(io, 3)
    return SnpArrays._read_chunk!(io, buffer)
end

"""
    run_chunk_stages(target, k, width, ::Type{T})

Time the individual stages of preparing and using the first chunk: the
serial and parallel read (each including the statistics refill), the
count and refill stages alone, and the AᵀY and AU products.
"""
function run_chunk_stages(target::BenchTarget, k::Int, width::Int,
                          ::Type{T}) where T
    println()
    m = target.m
    drows = (m + 3) >> 2
    readers = SnpArrays._default_readers()
    Y = randn(Xoshiro(1), T, m, k)
    buffer = SnpArrays._make_buffer(T, m, width, SnpArrays.ADDITIVE_MODEL,
                                    true, true, true)
    counts = zeros(Int, 4, width)
    piece = Matrix{T}(undef, width, k)
    V = Matrix{T}(undef, m, k)

    io = open(target.path, "r")
    (min1, med1, max1) = min5(read_chunk_seek!, io, buffer)
    @printf("%-24s min %8.4f  median %8.4f  max %8.4f s\n",
            "read+refill readers=1", min1, med1, max1)

    handles = [open(target.path, "r") for _ in 1:readers]
    (minD, medD, maxD) = min5(SnpArrays._read_chunk_parallel!, handles,
                              buffer, 3, drows)
    @printf("%-24s min %8.4f  median %8.4f  max %8.4f s\n",
            "read+refill readers=D", minD, medD, maxD)

    (minc, medc, maxc) = min5(SnpArrays._column_counts!, counts, buffer.s)
    @printf("%-24s min %8.4f  median %8.4f  max %8.4f s\n", "count", minc,
            medc, maxc)
    println("P2 chunk count stage median = ", medc, " s")

    (minr, medr, maxr) = min5(refill_once!, buffer)
    @printf("%-24s min %8.4f  median %8.4f  max %8.4f s\n", "refill", minr,
            medr, maxr)

    (mina, meda, maxa) = min5(mul!, piece, transpose(buffer), Y)
    @printf("%-24s min %8.4f  median %8.4f  max %8.4f s\n", "AtY", mina,
            meda, maxa)

    (minu, medu, maxu) = min5(mul!, V, buffer, piece)
    @printf("%-24s min %8.4f  median %8.4f  max %8.4f s\n", "AU", minu,
            medu, maxu)

    for handle in handles
        close(handle)
    end
    close(io)
    return nothing
end

function main()
    length(ARGS) >= 1 ||
        throw(ArgumentError("usage: stream_large_m.jl bed_path [k] " *
                            "[width] [passes] [T]"))
    path = ARGS[1]
    k = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
    width = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4096
    passes = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 2
    T = length(ARGS) >= 5 ?
        (ARGS[5] == "Float64" ? Float64 : Float32) : Float32

    fam_path = replace(path, ".bed" => ".fam")
    bim_path = replace(path, ".bed" => ".bim")
    m = open(countlines, fam_path)
    n = open(countlines, bim_path)
    target = BenchTarget(path, m, n)

    print_header(target, k, width, passes, T)

    (t_scan, t_map, t_s1, t_sd) = run_rounds(target, k, width, passes, T)
    print_summary(t_map, t_s1, t_sd, t_scan, passes)

    GC.gc()

    run_forward_rounds(target, k, width, passes, T)

    GC.gc()

    run_chunk_stages(target, k, width, T)

    return nothing
end

main()
