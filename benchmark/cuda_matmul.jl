# Time the CuSnpArray matrix products A*X and Aᵀ*X with k right-hand sides on
# a PLINK bed file, and the upload of its genotypes to the device.
#
#     julia -t N --project=<env with CUDA and SnpArrays> \
#       benchmark/cuda_matmul.jl [bed] [k] [results_dir]
#
# Each product runs twice: the first call includes kernel compilation, the
# second is the warm time. The right-hand sides match `matmul.jl`, so the
# checksums are comparable with its report.
# The report is also written to a timestamped markdown file in `results_dir`.
# Defaults: bed = the synthetic chromosome 21 genotypes below, k = 128,
# results_dir = `results/` next to this script.
using CUDA
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

const BED = length(ARGS) >= 1 ? ARGS[1] :
    "/u/scratch/b/bhchau/cudaext_check/synthetic_v1_chr-21.bed"
const K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const RESULTS_DIR = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "results")
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
    path = joinpath(dir, "cuda_matmul_$(gpu_name())_$(stamp).md")
    ispath(path) && error("$(path) exists; wait a second or remove it")
    mkpath(dir)
    return path
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
    print_fact(io, "bed", BED)
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    print_fact(io, "k", K)
    return nothing
end

"""
    print_row(io, ::Type{T}, product, variant, out, A, rhs, fmas)

Run `mul!(out, A, rhs)` twice and print its first and warm times in seconds.
"""
function print_row(
    io::IO,
    ::Type{T},
    product::AbstractString,
    variant::AbstractString,
    out::CuMatrix{T},
    A::AbstractMatrix,
    rhs::CuMatrix{T},
    fmas::Float64,
) where T <: AbstractFloat
    first_call = Float64(CUDA.@elapsed mul!(out, A, rhs))
    warm = Float64(CUDA.@elapsed mul!(out, A, rhs))
    emit(io, @sprintf(
        "| %s | %s | %d | %s | %.3f | %.3f | %.2f |",
        T, product, K, variant, first_call, warm, fmas / warm / 1e9,
    ))
    return nothing
end

"""
    checksum_row(::Type{T}, product, out) -> String

Return the checksum table row holding `sum(abs2, out)`.
"""
function checksum_row(
    ::Type{T},
    product::AbstractString,
    out::CuMatrix{T},
) where T
    return @sprintf("| %s | `%s` | %.17g |", T, product,
                    sum(abs2, Array(out)))
end

"""
    run_type(io, ::Type{T}, G) -> (String, Vector{String})

Upload `G`, print the timing rows of both products for element type `T`,
and return the upload table row and the checksum rows.
"""
function run_type(io::IO, ::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    started = time_ns()
    A = CuSnpArray{T}(G; center = true, scale = true, impute = true)
    CUDA.synchronize()
    seconds = (time_ns() - started) / 1e9
    upload = @sprintf("| %s | %.3f | %s | %.2f |", T, seconds,
                      Base.format_bytes(sizeof(A.data)),
                      sizeof(G.data) / seconds / 1e9)
    X = CuArray(randn(Xoshiro(1), T, n, K))
    Y = CuArray(randn(Xoshiro(2), T, m, K))
    AX = CuMatrix{T}(undef, m, K)
    AtX = CuMatrix{T}(undef, n, K)
    fmas = Float64(m) * n * K
    print_row(io, T, "A*X",
              EXT._uses_wmma(A, K) ? "mul! (tensor-core)" : "mul! (tiled)",
              AX, A, X, fmas)
    print_row(io, T, "Aᵀ*X",
              EXT._uses_wmma_t(A, K) ? "mul! (tensor-core)" : "mul! (tiled)",
              AtX, transpose(A), Y, fmas)
    checksums = [checksum_row(T, "A*X", AX), checksum_row(T, "Aᵀ*X", AtX)]
    for array in (A.data, A.μ, A.σinv, A.values, X, Y, AX, AtX)
        CUDA.unsafe_free!(array)
    end
    CUDA.reclaim()
    return upload, checksums
end

function main()
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `cuda_matmul.jl` on $(CUDA.name(CUDA.device()))")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    G = SnpArray(BED)
    print_header(io, size(G, 1), size(G, 2))
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "Centered, scaled, mean-imputed. GFMA/s is from the warm call.")
    emit(io, "")
    emit(io, "| Type | Product | k | Variant | first s | warm s | GFMA/s |")
    emit(io, "|---|---|---:|---|---:|---:|---:|")
    uploads = String[]
    checksums = String[]
    for T in (Float32, Float64)
        upload, rows = run_type(io, T, G)
        push!(uploads, upload)
        append!(checksums, rows)
    end
    emit(io, "")
    emit(io, "## Upload")
    emit(io, "")
    emit(io, "`CuSnpArray{T}(G)`: column statistics on the host, then the " *
             "packed words to the device.")
    emit(io, "")
    emit(io, "| Type | seconds | packed words | bed GB/s |")
    emit(io, "|---|---:|---:|---:|")
    for row in uploads
        emit(io, row)
    end
    emit(io, "")
    emit(io, "## Checksums")
    emit(io, "")
    emit(io, "| Type | Product | sum(abs2) |")
    emit(io, "| --- | --- | ---: |")
    for row in checksums
        emit(io, row)
    end
    emit(io, "")
    emit(io, "## Peak RSS")
    emit(io, "")
    emit(io, @sprintf("%.0f MiB", Sys.maxrss() / 2^20))
    close(io)
    println(path)
    return nothing
end

main()
