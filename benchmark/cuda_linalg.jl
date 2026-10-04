# Baseline GPU benchmark: CuSnpArray kernels, CPU SnpLinAlg, and CuBLAS on
# the materialized dense matrix. New implementations add rows by calling
# `record_row!` with a timing closure.
#
#   julia --project=<env with CUDA, Adapt, SnpArrays> -t 16 \
#       lib/SnpArrays/benchmark/cuda_linalg.jl [REPEATS]

using Adapt
using CUDA
using LinearAlgebra
using Printf
using Random
using SnpArrays

const REPEATS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
const REPETITIONS = 5
const KS = (1, 8, 32, 128)

function min_elapsed(f::Function, repetitions::Integer)
    # Warm up for at least 3 calls and 0.5 s so the GPU clock ramps up.
    start = time()
    calls = 0
    while calls < 3 || time() - start < 0.5
        f()
        calls += 1
    end
    return minimum(f() for _ in 1:repetitions)
end

cpu_time(f::Function) = min_elapsed(() -> @elapsed(f()), REPETITIONS)
gpu_time(f::Function) = min_elapsed(() -> CUDA.@elapsed(f()), REPETITIONS)

const ROWS = NamedTuple[]

"""
    record_row!(impl, direction, k, m, n, seconds)

Append one result row and print it.
"""
function record_row!(
    impl::AbstractString, direction::AbstractString, k::Integer,
    m::Integer, n::Integer, seconds::Real,
)
    throughput = m * n * k / seconds / 1e9
    push!(ROWS, (; impl, direction, k, ms=1000seconds, throughput))
    @printf(
        "%-28s %-6s %4d %12.3f %12.3f\n", impl, direction, k,
        1000seconds, throughput,
    )
    return nothing
end

"""
    operand(rng, ::Type{T}, dims, k, alloc)

Random operand: a vector when `k == 1`, else a matrix with `k` columns.
`alloc` maps a host array to its final container.
"""
function operand(
    rng::AbstractRNG, ::Type{T}, len::Integer, k::Integer, alloc::Function,
) where {T}
    return alloc(k == 1 ? rand(rng, T, len) : rand(rng, T, len, k))
end

function empty_like(::Type{A}, len::Integer, k::Integer) where {A}
    return k == 1 ? A(undef, len) : A(undef, len, k)
end

"""
    bench_impl!(label, A, m, n, alloc, timer, rng; ks)

For each `k`, time `mul!(out, A, X)` and `mul!(out, transpose(A), Y)`.
`alloc` moves host arrays to the implementation's array type, `timer`
is `cpu_time` or `gpu_time`.
"""
function bench_impl!(
    label::AbstractString, A, m::Integer, n::Integer, alloc::Function,
    timer::Function, rng::AbstractRNG; ks=KS,
)
    for k in ks
        X = operand(rng, Float32, n, k, alloc)
        out = alloc(k == 1 ? zeros(Float32, m) : zeros(Float32, m, k))
        record_row!(
            label, "A*X", k, m, n, timer(() -> mul!(out, A, X)),
        )
        Y = operand(rng, Float32, m, k, alloc)
        outT = alloc(k == 1 ? zeros(Float32, n) : zeros(Float32, n, k))
        record_row!(
            label, "A'*X", k, m, n,
            timer(() -> mul!(outT, transpose(A), Y)),
        )
        X = Y = out = outT = nothing
    end
    return nothing
end

function markdown_table(rows::AbstractVector)
    lines = String[
        "| implementation | direction | k | time (ms) | G elem/s |",
        "|---|---|---|---|---|",
    ]
    for r in rows
        push!(lines, @sprintf(
            "| %s | %s | %d | %.3f | %.3f |", r.impl, r.direction, r.k,
            r.ms, r.throughput,
        ))
    end
    return join(lines, "\n")
end

function main()
    root = pkgdir(SnpArrays)
    snp = SnpArray(joinpath(root, "data", "EUR_subset.bed"))
    m0, n = size(snp)
    stacked = SnpArray(undef, REPEATS * m0, n)
    for repeat_index in 1:REPEATS
        rows = ((repeat_index - 1) * m0 + 1):(repeat_index * m0)
        stacked[rows, :] .= view(snp, :, :)
    end
    m = REPEATS * m0
    rng = Random.Xoshiro(1)

    println("GPU: ", CUDA.name(CUDA.device()))
    println("CUDA.jl: ", pkgversion(CUDA))
    println("Threads.nthreads() = ", Threads.nthreads())
    @printf("Stacked size: m = %d, n = %d\n", m, n)
    @printf(
        "%-28s %-6s %4s %12s %12s\n", "implementation", "dir", "k",
        "ms", "G elem/s",
    )

    # Packed-word CuSnpArray kernels. The byte-per-sample
    # kernels they replace ran 26.029 ms (A*X) and 13.886 ms (A'*X) on an
    # A100 at this shape.
    cu_snp = CuSnpArray{Float32}(
        stacked; model=ADDITIVE_MODEL, center=true, scale=true,
        impute=false,
    )
    bench_impl!(
        "CuSnpArray (new)", cu_snp, m, n, x -> CuArray(x), gpu_time, rng,
    )
    # The tiled kernels, which Float32 `A*X` leaves for tensor cores at
    # k >= 16.
    ext = Base.get_extension(SnpArrays, :SnpArraysCUDAExt)
    for k in KS
        X = operand(rng, Float32, n, k, CuArray)
        out = k == 1 ? CuArray(zeros(Float32, m)) :
            CuArray(zeros(Float32, m, k))
        record_row!(
            "CuSnpArray tiled", "A*X", k, m, n,
            gpu_time(() -> ext._tiled_mul!(out, cu_snp, X)),
        )
    end
    cu_snp = nothing
    GC.gc(); CUDA.reclaim()

    # CPU SnpLinAlg.
    host = SnpLinAlg{Float32}(
        stacked; model=ADDITIVE_MODEL, center=true, scale=true,
    )
    bench_impl!(
        "CPU SnpLinAlg", host, m, n, identity, cpu_time, rng,
    )
    host = nothing

    # CuBLAS on the materialized dense matrix.
    dense = CuArray(convert(
        Matrix{Float32}, stacked; model=ADDITIVE_MODEL, center=true,
        scale=true,
    ))
    GC.gc()
    bench_impl!(
        "CuBLAS dense", dense, m, n, x -> CuArray(x), gpu_time, rng,
    )
    dense = nothing

    println()
    println("Markdown table:")
    println(markdown_table(ROWS))
    @printf("Peak host memory (maxrss): %.2f GiB\n", Sys.maxrss() / 2^30)
    return nothing
end

main()
