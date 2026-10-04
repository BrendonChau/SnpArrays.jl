# Benchmark the CuSnpArray mul! kernels on simulated genotypes against the
# decode kernels and CuBLAS on the materialized dense matrix.
#
#     julia -t N --project=<env with CUDA and SnpArrays> \
#       benchmark/cuda_kernels.jl [m] [n] [prefix] [results_dir]
#
# The report is also written to a timestamped markdown file in `results_dir`.
# Defaults: m = 8192, n = 49152, prefix = "linalg_kernels", results_dir =
# `results/` next to this script. `<prefix>.bed` is reused when present and
# simulated otherwise, so a run on the prefix of a `linalg_kernels.jl` run
# has `sum(abs2)` rows comparable with that report.
using CUDA
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

const M = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8192
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 49152
const PREFIX = length(ARGS) >= 3 ? ARGS[3] : "linalg_kernels"
const RESULTS_DIR = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "results")
const RHS_COUNTS = (8, 32, 128)
const EXT = Base.get_extension(SnpArrays, :SnpArraysCUDAExt)

"""
    emit(io, line)

Print `line` to `stdout` and to `io`, then flush `io`.
"""
function emit(io::IO, line::AbstractString)
    println(line)
    println(io, line)
    flush(io)
    return nothing
end

"""
    gpu_name() -> String

Return the active device's name with runs of other characters as `-`.
"""
gpu_name() = replace(CUDA.name(CUDA.device()), r"[^A-Za-z0-9]+" => "-")

"""
    report_path(dir) -> String

Create `dir` and return the timestamped report path in it; an existing file
at that path is an error.
"""
function report_path(dir::AbstractString)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    path = joinpath(dir, "cuda_kernels_$(gpu_name())_$(stamp).md")
    ispath(path) && error("$(path) exists; wait a second or remove it")
    mkpath(dir)
    return path
end

"""
    write_bed(path, G)

Write `G` as `path`, plus the `.fam` file `SnpArray(path)` counts samples in.
"""
function write_bed(path::AbstractString, G::SnpArray)
    io = open(splitext(path)[1] * ".fam", "w")
    for i in 1:size(G, 1)
        println(io, "$(i)\t$(i)\t0\t0\t0\t-9")
    end
    close(io)
    io = open(path, "w")
    write(io, 0x1b6c)
    write(io, 0x01)
    write(io, G.data)
    close(io)
    return path
end

"""
    load_genotypes(io, m, n, prefix) -> SnpArray

Read `<prefix>.bed`, simulating it first if absent (MAFs uniform on
[0.05, 0.5], 1% missing per SNP). An existing file is never overwritten; one
of the wrong size is an error.
"""
function load_genotypes(io::IO, m::Int, n::Int, prefix::AbstractString)
    path = prefix * ".bed"
    if isfile(path)
        filesize(path) == 3 + cld(m, 4) * n || error(
            "$(path) has $(filesize(path)) bytes, not the $(3 + cld(m, 4) * n)",
            " of an $(m) x $(n) genotype matrix; remove it or change m, n",
        )
        emit(io, "Reusing `$(path)`.")
    else
        rng = Xoshiro(1)
        mafs = 0.05 .+ 0.45 .* rand(rng, n)
        G = SnpArrays.simulate(rng, m, n, mafs)
        SnpArrays.simulate_missing!(rng, G, fill(0.01, n))
        write_bed(path, G)
        emit(io, "Simulated `$(path)`.")
    end
    G = SnpArray(path)
    size(G) == (m, n) || error("$(path) holds a $(size(G)) matrix, not ",
                               "$((m, n)); check its .fam file")
    return G
end

"""
    print_fact(io, name, value)

Print one row of the `| Fact | Value |` table.
"""
function print_fact(io::IO, name::AbstractString, value)
    emit(io, "| $(name) | `$(value)` |")
    return nothing
end

"""
    print_header(io, m, n)

Print the environment facts as a markdown table.
"""
function print_header(io::IO, m::Int, n::Int)
    device = CUDA.device()
    emit(io, "")
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "GPU", CUDA.name(device))
    print_fact(io, "GPU memory",
               Base.format_bytes(CUDA.totalmem(device)))
    print_fact(io, "CUDA.jl", pkgversion(CUDA))
    print_fact(io, "CUDA runtime", CUDA.runtime_version())
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    return nothing
end

"""
    time_call(f!, args...; repeats = 5) -> (minimum, median)

Time `f!(args...)` on the device in seconds, after one warm-up call.
"""
function time_call(f!::F, args...; repeats::Int = 5) where F
    f!(args...)
    CUDA.synchronize()
    times = [Float64(CUDA.@elapsed f!(args...)) for _ in 1:repeats]
    sort!(times)
    return times[1], times[(repeats + 1) ÷ 2]
end

"""
    bench_row!(io, ::Type{T}, product, k, variant, f!, out, A, rhs, expected)
        -> Array{T}

Time `f!(out, A, rhs)`, print its row with its relative difference from
`expected` (zero when `expected` is `nothing`), and return `out` on the host.
"""
function bench_row!(
    io::IO,
    ::Type{T},
    product::AbstractString,
    k::Int,
    variant::AbstractString,
    f!::F,
    out::CuArray{T},
    A::AbstractMatrix,
    rhs::CuArray{T},
    expected::Union{Nothing, Array{T}},
) where {T <: AbstractFloat, F}
    times = time_call(f!, out, A, rhs)
    result = Array(out)
    difference = expected === nothing ? 0.0 :
                 Float64(norm(result .- expected) / norm(expected))
    fmas = Float64(size(A, 1)) * size(A, 2) * k
    emit(io, @sprintf(
        "| %s | %s | %d | %s | %.3f | %.3f | %.2f | %.1e |",
        T, product, k, variant, 1000 * times[1], 1000 * times[2],
        fmas / times[1] / 1e9, difference,
    ))
    return result
end

"""
    decode_mul!(out, A, X) -> out

Set `out = A * X` with the decode kernels.
"""
function decode_mul!(out::CuArray{T}, A::CuSnpArray{T},
                     X::CuArray{T}) where T <: AbstractFloat
    return EXT._decode_mul!(out, A, X)
end

"""
    tf32_mul!(out, A, X) -> out

Set `out = A * X` with CuBLAS in its TF32 tensor-core math mode.
"""
function tf32_mul!(out::CuMatrix{Float32}, A::AbstractMatrix{Float32},
                   X::CuMatrix{Float32})
    CUDA.math_mode!(CUDA.FAST_MATH; precision = :TensorFloat32)
    mul!(out, A, X)
    CUDA.math_mode!(CUDA.DEFAULT_MATH; precision = :TensorFloat32)
    return out
end

"""
    free_operator!(A)

Free the device arrays of `A`.
"""
function free_operator!(A::CuSnpArray)
    for array in (A.data, A.μ, A.σinv, A.values)
        CUDA.unsafe_free!(array)
    end
    return nothing
end

"""
    run_type(io, ::Type{T}, G)

Print the table rows of every product for element type `T`.
"""
function run_type(io::IO, ::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    A = CuSnpArray{T}(G; center = true, scale = true, impute = true)
    dense = CuArray(convert(Matrix{T}, G; center = true, scale = true,
                            impute = true))
    x = CuArray(randn(Xoshiro(1), T, n))
    out = CuVector{T}(undef, m)
    bench_row!(io, T, "A*x", 1, "mul!", mul!, out, A, x, nothing)
    y = CuArray(randn(Xoshiro(2), T, m))
    out_t = CuVector{T}(undef, n)
    bench_row!(io, T, "Aᵀ*y", 1, "mul!", mul!, out_t, transpose(A), y,
               nothing)
    for k in RHS_COUNTS
        X = CuArray(randn(Xoshiro(3), T, n, k))
        Y = CuArray(randn(Xoshiro(4), T, m, k))
        AX = CuMatrix{T}(undef, m, k)
        AtX = CuMatrix{T}(undef, n, k)
        tensor = EXT._uses_wmma(A, k)
        expected = bench_row!(
            io, T, "A*X", k, tensor ? "mul! (tensor-core)" : "mul! (decode)",
            mul!, AX, A, X, nothing,
        )
        tensor && bench_row!(io, T, "A*X", k, "decode", decode_mul!, AX, A,
                             X, expected)
        bench_row!(io, T, "A*X", k, "CuBLAS dense", mul!, AX, dense, X,
                   expected)
        T <: Float32 && bench_row!(io, T, "A*X", k, "CuBLAS dense TF32",
                                   tf32_mul!, AX, dense, X, expected)
        tensor = EXT._uses_wmma_t(A, k)
        expected = bench_row!(
            io, T, "Aᵀ*X", k, tensor ? "mul! (tensor-core)" : "mul! (tiled)",
            mul!, AtX, transpose(A), Y, nothing,
        )
        bench_row!(io, T, "Aᵀ*X", k, "CuBLAS dense", mul!, AtX,
                   transpose(dense), Y, expected)
        T <: Float32 && bench_row!(io, T, "Aᵀ*X", k, "CuBLAS dense TF32",
                                   tf32_mul!, AtX, transpose(dense), Y,
                                   expected)
    end
    CUDA.unsafe_free!(dense)
    free_operator!(A)
    return nothing
end

"""
    print_agreement(io, G)

Print `sum(abs2, ...)` of the four `Float64` products at `k = 8`, on the
right-hand sides of `linalg_kernels.jl`.
"""
function print_agreement(io::IO, G::SnpArray)
    m, n = size(G)
    k = 8
    A = CuSnpArray{Float64}(G; center = true, scale = true, impute = true)
    x = CuArray(randn(Xoshiro(11), Float64, n))
    y = CuArray(randn(Xoshiro(12), Float64, m))
    X = CuArray(randn(Xoshiro(13), Float64, n, k))
    Y = CuArray(randn(Xoshiro(14), Float64, m, k))
    emit(io, "")
    emit(io, "## Cross-package check")
    emit(io, "")
    emit(io, "Float64, k = 8.")
    emit(io, "")
    emit(io, "| Product | sum(abs2) |")
    emit(io, "| --- | ---: |")
    products = (
        mul!(CuVector{Float64}(undef, m), A, x),
        mul!(CuVector{Float64}(undef, n), transpose(A), y),
        mul!(CuMatrix{Float64}(undef, m, k), A, X),
        mul!(CuMatrix{Float64}(undef, n, k), transpose(A), Y),
    )
    for (name, product) in zip(("A*x", "Aᵀ*y", "A*X", "Aᵀ*X"), products)
        emit(io, @sprintf("| `%s` | %.17g |", name,
                          sum(abs2, Array(product))))
    end
    free_operator!(A)
    return nothing
end

function main()
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `cuda_kernels.jl` on $(CUDA.name(CUDA.device()))")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    G = load_genotypes(io, M, N, PREFIX)
    print_header(io, M, N)
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "Centered, scaled, mean-imputed. The last column is the " *
             "relative difference from the `mul!` row of the same product.")
    emit(io, "")
    emit(io, "| Type | Product | k | Variant | min ms | median ms | " *
             "GFMA/s | rel. diff |")
    emit(io, "|---|---|---:|---|---:|---:|---:|---:|")
    for T in (Float32, Float64)
        run_type(io, T, G)
    end
    print_agreement(io, G)
    emit(io, "")
    emit(io, "## Peak RSS")
    emit(io, "")
    emit(io, @sprintf("%.0f MiB", Sys.maxrss() / 2^20))
    close(io)
    println(path)
    return nothing
end

main()
