# Time the streamed SnpLinAlgStream products A*X, Aᵀ*X, and A*(Aᵀ*X)/n with
# k right-hand sides on a PLINK bed file.
#
#     julia -t N --project=. benchmark/streaming_matmul.jl \
#       [bed] [k] [width] [results_dir]
#
# Each product is one sweep over the file, run once with no warm-up, so its
# time includes compilation. The right-hand sides match `matmul.jl`, so the
# A*X and Aᵀ*X checksums are comparable with its report.
# The report is also written to a timestamped markdown file in `results_dir`.
# Defaults: bed = the synthetic chromosome 21 genotypes below, k = 128,
# width = 4096 SNPs per chunk, results_dir = `results/` next to this script.
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
    report_path(dir) -> String

Create `dir` and return the timestamped report path in it; an existing file
at that path is an error.
"""
function report_path(dir::AbstractString)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    path = joinpath(dir, "streaming_matmul_$(Sys.CPU_NAME)_$(stamp).md")
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
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "Sys.ARCH", Sys.ARCH)
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
    isdefined(SnpArrays, :VECTOR_BYTES) &&
        print_fact(io, "VECTOR_BYTES", SnpArrays.VECTOR_BYTES[])
    return nothing
end

"""
    print_row(io, ::Type{T}, product, seconds, fmas)

Print one row of the timing table.
"""
function print_row(
    io::IO,
    ::Type{T},
    product::AbstractString,
    seconds::Float64,
    fmas::Float64,
) where T
    emit(io, @sprintf(
        "| %s | %s | %d | %.3f | %.2f |",
        T, product, K, seconds, fmas / seconds / 1e9,
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
    out::Matrix{T},
) where T
    return @sprintf("| %s | `%s` | %.17g |", T, product, sum(abs2, out))
end

"""
    run_type(io, stream::SnpLinAlgStream{T}) -> Vector{String}

Print the timing rows of the three streamed products and return their
checksum rows.
"""
function run_type(io::IO, stream::SnpLinAlgStream{T}) where T <: AbstractFloat
    m, n = size(stream)
    X = randn(Xoshiro(1), T, n, K)
    Y = randn(Xoshiro(2), T, m, K)
    AX = Matrix{T}(undef, m, K)
    AtX = Matrix{T}(undef, n, K)
    V = Matrix{T}(undef, m, K)
    fmas = Float64(m) * n * K
    print_row(io, T, "A*X", @elapsed(streamed_mul!(AX, stream, X)), fmas)
    print_row(io, T, "Aᵀ*X",
              @elapsed(streamed_mul!(AtX, stream, Y; transpose = true)), fmas)
    print_row(io, T, "A*(Aᵀ*X)/n", @elapsed(streamed_grm_mul!(V, stream, Y)),
              2 * fmas)
    return [checksum_row(T, "A*X", AX), checksum_row(T, "Aᵀ*X", AtX),
            checksum_row(T, "A*(Aᵀ*X)/n", V)]
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
    emit(io, "# `streaming_matmul.jl` on $(Sys.CPU_NAME)")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    print_header(io, make_stream(Float32))
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "| Type | Product | k | seconds | GFMA/s |")
    emit(io, "|---|---|---:|---:|---:|")
    checksums = String[]
    for T in (Float32, Float64)
        append!(checksums, run_type(io, make_stream(T)))
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
