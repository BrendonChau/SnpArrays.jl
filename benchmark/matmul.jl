# Time the SnpLinAlg matrix products A*X and Aᵀ*X with k right-hand sides on a
# PLINK bed file.
#
#     julia -t N --project=. benchmark/matmul.jl [bed] [k] [results_dir]
#
# Each product runs once, with no warm-up, so its time includes compilation.
# The report is also written to a timestamped markdown file in `results_dir`.
# Defaults: bed = the synthetic chromosome 21 genotypes below, k = 128,
# results_dir = `results/` next to this script.
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

const BED = length(ARGS) >= 1 ? ARGS[1] :
    "/u/scratch/b/bhchau/cudaext_check/synthetic_v1_chr-21.bed"
const K = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const RESULTS_DIR = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "results")

"""
    has_fork_internals() -> Bool

Return whether the loaded SnpArrays has the register-tiled schedulers.
"""
has_fork_internals() = isdefined(SnpArrays, :_snparray_AX_spawn_tasks!)

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
    kind = has_fork_internals() ? "fork" : "upstream"
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    threads = Threads.nthreads()
    path = joinpath(
        dir, "matmul_$(Sys.CPU_NAME)_$(kind)_t$(threads)_$(stamp).md",
    )
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
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "SnpArrays", has_fork_internals() ? "fork" : "upstream")
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "Sys.ARCH", Sys.ARCH)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "bed", BED)
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    print_fact(io, "k", K)
    isdefined(SnpArrays, :VECTOR_BYTES) &&
        print_fact(io, "VECTOR_BYTES", SnpArrays.VECTOR_BYTES[])
    return nothing
end

"""
    print_row(io, ::Type{T}, product, variant, seconds, fmas)

Print one row of the timing table.
"""
function print_row(
    io::IO,
    ::Type{T},
    product::AbstractString,
    variant::AbstractString,
    seconds::Float64,
    fmas::Float64,
) where T
    emit(io, @sprintf(
        "| %s | %s | %d | %s | %.3f | %.2f |",
        T, product, K, variant, seconds, fmas / seconds / 1e9,
    ))
    return nothing
end

function matrix_variant(m::Int, k::Int)
    isdefined(SnpArrays, :_uses_lookup_kernel) || return "mul!"
    return SnpArrays._uses_lookup_kernel(m, k) ? "mul! (lookup)" :
           "mul! (register-tiled)"
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
    run_type(io, ::Type{T}, G) -> Vector{String}

Print the timing rows of both products for element type `T` and return their
checksum rows.
"""
function run_type(io::IO, ::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    operator = SnpLinAlg{T}(G; center = true, scale = true, impute = true)
    X = randn(Xoshiro(1), T, n, K)
    Y = randn(Xoshiro(2), T, m, K)
    AX = Matrix{T}(undef, m, K)
    AtX = Matrix{T}(undef, n, K)
    fmas = Float64(m) * n * K
    print_row(io, T, "A*X", matrix_variant(m, K),
              @elapsed(mul!(AX, operator, X)), fmas)
    print_row(io, T, "Aᵀ*X",
              has_fork_internals() ? "mul! (register-tiled)" : "mul!",
              @elapsed(mul!(AtX, transpose(operator), Y)), fmas)
    return [checksum_row(T, "A*X", AX), checksum_row(T, "Aᵀ*X", AtX)]
end

function main()
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `matmul.jl` on $(Sys.CPU_NAME)")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    G = SnpArray(BED)
    print_header(io, size(G, 1), size(G, 2))
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "| Type | Product | k | Variant | seconds | GFMA/s |")
    emit(io, "|---|---|---:|---|---:|---:|")
    checksums = String[]
    for T in (Float32, Float64)
        append!(checksums, run_type(io, T, G))
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
