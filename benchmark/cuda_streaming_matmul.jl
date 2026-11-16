# Time the GPU streamed SnpLinAlgStream products A*X, Aᵀ*X, and A*(Aᵀ*X)/n
# with k right-hand sides on a PLINK bed file.
#
#     julia -t N --project=<env with CUDA and SnpArrays> \
#       benchmark/cuda_streaming_matmul.jl [bed] [k] [width] [results_dir]
#
# Each product is one sweep over the file, with the right-hand sides and the
# output on the device. Each runs twice: the first call includes kernel
# compilation and a cold read of the file, the second is the warm time. The
# last row is the mixed-precision product: a Float32 stream with Float64
# right-hand sides and output. The right-hand sides match
# `streaming_matmul.jl`, so the checksums are comparable with its report.
# The report is also written to a timestamped markdown file in `results_dir`.
# Defaults: bed = the synthetic chromosome 21 genotypes below, k = 128,
# width = 4096 SNPs per chunk, results_dir = `results/` next to this script.
using CUDA
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

const BED = length(ARGS) >= 1 ? ARGS[1] :
    "/u/scratch/b/bhchau/cudaext_check/synthetic_v1_chr-21.bed"
const K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const WIDTH = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4096
const RESULTS_DIR = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "results")
# The tensor-core kernels are opt-in; this benchmark measures them.
SnpArrays.cuda_tensor_cores!(true)

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
    path = joinpath(dir, "cuda_streaming_matmul_$(gpu_name())_$(stamp).md")
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
    print_header(io, stream)

Print the environment facts as a markdown table.
"""
function print_header(io::IO, stream::SnpLinAlgStream)
    m, n = size(stream)
    device = CUDA.device()
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "GPU", CUDA.name(device))
    print_fact(io, "GPU memory", Base.format_bytes(CUDA.totalmem(device)))
    print_fact(io, "CUDA.jl", pkgversion(CUDA))
    print_fact(io, "CUDA runtime", CUDA.runtime_version())
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "bed", BED)
    print_fact(io, "bed bytes", filesize(BED))
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    print_fact(io, "k", K)
    print_fact(io, "width", stream.width)
    print_fact(io, "chunks", length(stream))
    print_fact(io, "prefetch", stream.prefetch)
    print_fact(io, "readers", stream.readers)
    return nothing
end

"""
    print_row(io, type, product, first_call, warm, fmas)

Print one row of the timing table; GFMA/s is from the warm call.
"""
function print_row(
    io::IO,
    type::AbstractString,
    product::AbstractString,
    first_call::Float64,
    warm::Float64,
    fmas::Float64,
)
    emit(io, @sprintf(
        "| %s | %s | %d | %.3f | %.3f | %.2f |",
        type, product, K, first_call, warm, fmas / warm / 1e9,
    ))
    return nothing
end

"""
    time_mul!(out, stream, X, transpose) -> (Float64, Float64)

Run `streamed_mul!(out, stream, X; transpose)` twice and return its first
and warm times in seconds.
"""
function time_mul!(
    out::CuMatrix, stream::SnpLinAlgStream, X::CuMatrix, transpose::Bool,
)
    first_call = CUDA.@elapsed streamed_mul!(out, stream, X; transpose)
    warm = CUDA.@elapsed streamed_mul!(out, stream, X; transpose)
    return Float64(first_call), Float64(warm)
end

"""
    time_grm_mul!(V, stream, Q) -> (Float64, Float64)

Run `streamed_grm_mul!(V, stream, Q)` twice and return its first and warm
times in seconds.
"""
function time_grm_mul!(V::CuMatrix, stream::SnpLinAlgStream, Q::CuMatrix)
    first_call = CUDA.@elapsed streamed_grm_mul!(V, stream, Q)
    warm = CUDA.@elapsed streamed_grm_mul!(V, stream, Q)
    return Float64(first_call), Float64(warm)
end

"""
    checksum_row(type, product, out) -> String

Return the checksum table row holding `sum(abs2, out)`.
"""
function checksum_row(
    type::AbstractString,
    product::AbstractString,
    out::CuMatrix{<:AbstractFloat},
)
    return @sprintf("| %s | `%s` | %.17g |", type, product,
                    sum(abs2, Array(out)))
end

"""
    run_type(io, stream::SnpLinAlgStream{T}) -> Vector{String}

Print the timing rows of the three streamed products and return their
checksum rows.
"""
function run_type(io::IO, stream::SnpLinAlgStream{T}) where T <: AbstractFloat
    m, n = size(stream)
    X = CuArray(randn(Xoshiro(1), T, n, K))
    Y = CuArray(randn(Xoshiro(2), T, m, K))
    AX = CuMatrix{T}(undef, m, K)
    AtX = CuMatrix{T}(undef, n, K)
    V = CuMatrix{T}(undef, m, K)
    fmas = Float64(m) * n * K
    type = string(T)
    first_call, warm = time_mul!(AX, stream, X, false)
    print_row(io, type, "A*X", first_call, warm, fmas)
    first_call, warm = time_mul!(AtX, stream, Y, true)
    print_row(io, type, "Aᵀ*X", first_call, warm, fmas)
    first_call, warm = time_grm_mul!(V, stream, Y)
    print_row(io, type, "A*(Aᵀ*X)/n", first_call, warm, 2 * fmas)
    checksums = [checksum_row(type, "A*X", AX),
                 checksum_row(type, "Aᵀ*X", AtX),
                 checksum_row(type, "A*(Aᵀ*X)/n", V)]
    for array in (X, Y, AX, AtX, V)
        CUDA.unsafe_free!(array)
    end
    CUDA.reclaim()
    return checksums
end

"""
    run_mixed(io, stream::SnpLinAlgStream{Float32}) -> Vector{String}

Print the timing row of the mixed-precision A*(Aᵀ*X)/n, with `Float64`
right-hand sides and output, and return its checksum row.
"""
function run_mixed(io::IO, stream::SnpLinAlgStream{Float32})
    m, n = size(stream)
    Y = CuArray(randn(Xoshiro(2), Float64, m, K))
    V = CuMatrix{Float64}(undef, m, K)
    type = "Float32 stream, Float64 X"
    first_call, warm = time_grm_mul!(V, stream, Y)
    print_row(io, type, "A*(Aᵀ*X)/n", first_call, warm,
              2 * Float64(m) * n * K)
    checksums = [checksum_row(type, "A*(Aᵀ*X)/n", V)]
    CUDA.unsafe_free!(Y)
    CUDA.unsafe_free!(V)
    CUDA.reclaim()
    return checksums
end

"""
    make_stream(::Type{T}) -> SnpLinAlgStream{T}

Open `BED` as a centered, scaled, imputed stream of `WIDTH`-SNP chunks.
"""
function make_stream(::Type{T}) where T <: AbstractFloat
    return SnpLinAlgStream{T}(BED; width = WIDTH, center = true, scale = true,
                              impute = true)
end

function main()
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `cuda_streaming_matmul.jl` on $(CUDA.name(CUDA.device()))")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    print_header(io, make_stream(Float32))
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "Centered, scaled, mean-imputed. GFMA/s is from the warm call.")
    emit(io, "")
    emit(io, "| Type | Product | k | first s | warm s | GFMA/s |")
    emit(io, "|---|---|---:|---:|---:|---:|")
    checksums = String[]
    for T in (Float32, Float64)
        append!(checksums, run_type(io, make_stream(T)))
    end
    append!(checksums, run_mixed(io, make_stream(Float32)))
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
