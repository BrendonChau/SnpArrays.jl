# Compare the SnpLinAlg products A*X and Aᵀ*Y with decoding blocks of variants
# to dense matrices and calling BLAS gemm, on a PLINK bed file.
#
#     julia -t N --project=. benchmark/decode_blas.jl [bed] [ks] [block]
#         [results_dir]
#
# Each timed product runs twice and the second, warm time is reported, so
# compilation is excluded. The report is also written to a timestamped markdown
# file in `results_dir`. An empty string selects the default of an argument.
# Defaults: bed = `SnpArrays.datadir("mouse.bed")`, ks = 4,16,32,128 (comma
# separated right-hand-side counts), block = 64 variants per decoded block,
# results_dir = `results/` next to this script.
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

"""
    argument(position, default) -> String

Return command-line argument `position`, or `default` when it is absent or
empty.
"""
function argument(position::Int, default::AbstractString)
    length(ARGS) >= position || return String(default)
    return isempty(ARGS[position]) ? String(default) : ARGS[position]
end

const BED = argument(1, SnpArrays.datadir("mouse.bed"))
const KS = parse.(Int, split(argument(2, "4,16,32,128"), ','))
const BLOCK = parse(Int, argument(3, "64"))
const RESULTS_DIR = argument(4, joinpath(@__DIR__, "results"))

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
    threads = Threads.nthreads()
    path = joinpath(
        dir, "decode_blas_$(Sys.CPU_NAME)_t$(threads)_$(stamp).md",
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
    libraries = join(basename.(getfield.(BLAS.get_config().loaded_libs,
                                         :libname)), ", ")
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "BLAS.vendor", BLAS.vendor())
    print_fact(io, "BLAS.get_config", libraries)
    print_fact(io, "BLAS threads", BLAS.get_num_threads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "Sys.ARCH", Sys.ARCH)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "bed", BED)
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    print_fact(io, "ks", join(KS, ","))
    print_fact(io, "block", BLOCK)
    isdefined(SnpArrays, :VECTOR_BYTES) &&
        print_fact(io, "VECTOR_BYTES", SnpArrays.VECTOR_BYTES[])
    isdefined(SnpArrays, :ATX_TILE_COLUMNS) &&
        print_fact(io, "ATX_TILE_COLUMNS", SnpArrays.ATX_TILE_COLUMNS[])
    return nothing
end

"""
    fill_table!(table, values, column) -> table

Fill the 4×256 `table` so that column `b + 1` holds the values of the four
2-bit genotype codes packed in byte `b`, for variant `column` of `values`.
"""
function fill_table!(
    table::AbstractMatrix{T},
    values::Matrix{T},
    column::Int,
) where T
    for byte in 0:255, slot in 0:3
        code = (byte >> (2 * slot)) & 0x03
        table[slot + 1, byte + 1] = values[code + 1, column]
    end
    return table
end

"""
    decode_columns!(D, table, data, values, m, offset, locals)

Decode variants `offset .+ locals` of the packed genotypes `data` into
columns `locals` of `D`, through `table` as per-variant workspace.
"""
function decode_columns!(
    D::Matrix{T},
    table::AbstractMatrix{T},
    data::Matrix{UInt8},
    values::Matrix{T},
    m::Int,
    offset::Int,
    locals::UnitRange{Int},
) where T
    full_bytes = m >> 2
    remainder = m - 4 * full_bytes
    for column in locals
        variant = offset + column
        fill_table!(table, values, variant)
        @inbounds for byte_index in 1:full_bytes
            entry = Int(data[byte_index, variant]) + 1
            row = 4 * (byte_index - 1)
            for slot in 1:4
                D[row + slot, column] = table[slot, entry]
            end
        end
        # The last byte holds only `remainder` samples when 4 ∤ m.
        if remainder > 0
            entry = Int(data[full_bytes + 1, variant]) + 1
            for slot in 1:remainder
                D[4 * full_bytes + slot, column] = table[slot, entry]
            end
        end
    end
    return nothing
end

"""
    decode_block!(D, tables, operator, block)

Decode variants `block` of `operator` into the leading columns of `D`, one
task and one slice of `tables` per contiguous chunk of columns.
"""
function decode_block!(
    D::Matrix{T},
    tables::Array{T, 3},
    operator::SnpLinAlg{T},
    block::UnitRange{Int},
) where T
    data = operator.s.data
    m = operator.s.m
    width = length(block)
    size(D, 1) == m && size(D, 2) >= width ||
        throw(DimensionMismatch("D is too small for the block"))
    size(data, 1) == (m + 3) >> 2 && last(block) <= size(data, 2) ||
        throw(DimensionMismatch("block exceeds the packed genotypes"))
    tasks = min(size(tables, 3), width)
    offset = first(block) - 1
    @sync for task in 1:tasks
        locals = (((task - 1) * width) ÷ tasks + 1):((task * width) ÷ tasks)
        Threads.@spawn decode_columns!(
            D, view(tables, :, :, task), data, operator.values, m, offset,
            locals,
        )
    end
    return nothing
end

"""
    decode_blas_AX!(AX, D, tables, operator, X) -> (decode, gemm)

Overwrite `AX` with `operator * X` by decoding blocks of `size(D, 2)` variants
and accumulating with gemm; return the seconds spent in each phase.
"""
function decode_blas_AX!(
    AX::Matrix{T},
    D::Matrix{T},
    tables::Array{T, 3},
    operator::SnpLinAlg{T},
    X::Matrix{T},
) where T
    n = size(operator, 2)
    decode_ns = UInt64(0)
    gemm_ns = UInt64(0)
    fill!(AX, zero(T))
    for start in 1:size(D, 2):n
        block = start:min(start + size(D, 2) - 1, n)
        t0 = time_ns()
        decode_block!(D, tables, operator, block)
        t1 = time_ns()
        mul!(AX, view(D, :, 1:length(block)), view(X, block, :), one(T),
             one(T))
        t2 = time_ns()
        decode_ns += t1 - t0
        gemm_ns += t2 - t1
    end
    return decode_ns / 1e9, gemm_ns / 1e9
end

"""
    decode_blas_AtY!(AtY, D, tables, operator, Y) -> (decode, gemm)

Overwrite `AtY` with `transpose(operator) * Y` by decoding blocks of
`size(D, 2)` variants and calling gemm; return the seconds spent in each phase.
"""
function decode_blas_AtY!(
    AtY::Matrix{T},
    D::Matrix{T},
    tables::Array{T, 3},
    operator::SnpLinAlg{T},
    Y::Matrix{T},
) where T
    n = size(operator, 2)
    decode_ns = UInt64(0)
    gemm_ns = UInt64(0)
    for start in 1:size(D, 2):n
        block = start:min(start + size(D, 2) - 1, n)
        t0 = time_ns()
        decode_block!(D, tables, operator, block)
        t1 = time_ns()
        mul!(view(AtY, block, :),
             transpose(view(D, :, 1:length(block))), Y)
        t2 = time_ns()
        decode_ns += t1 - t0
        gemm_ns += t2 - t1
    end
    return decode_ns / 1e9, gemm_ns / 1e9
end

"""
    print_row(io, ::Type{T}, product, k, path, seconds, fmas, phases)

Print one row of the timing table; `phases` is the decode and gemm seconds,
or `nothing` for a path without phases.
"""
function print_row(
    io::IO,
    ::Type{T},
    product::AbstractString,
    k::Int,
    path::AbstractString,
    seconds::Float64,
    fmas::Float64,
    phases::Union{Nothing, Tuple{Float64, Float64}},
) where T
    tail = phases === nothing ? "| | |" :
           @sprintf("| %.4f | %.4f |", phases[1], phases[2])
    emit(io, @sprintf(
        "| %s | %s | %d | %s | %.4f | %.2f %s",
        T, product, k, path, seconds, fmas / seconds / 1e9, tail,
    ))
    return nothing
end

"""
    checksum_row(::Type{T}, product, k, direct, decoded) -> String

Return the checksum table row holding `sum(abs2, out)` of both paths and
their maximum absolute difference.
"""
function checksum_row(
    ::Type{T},
    product::AbstractString,
    k::Int,
    direct::Matrix{T},
    decoded::Matrix{T},
) where T
    difference = zero(T)
    for index in eachindex(direct, decoded)
        difference = max(difference, abs(direct[index] - decoded[index]))
    end
    return @sprintf(
        "| %s | `%s` | %d | %.17g | %.17g | %.3g |",
        T, product, k, sum(abs2, direct), sum(abs2, decoded), difference,
    )
end

"""
    run_type(io, ::Type{T}, G) -> Vector{String}

Print the timing rows of both products and both paths for element type `T`
at every k and return their checksum rows.
"""
function run_type(io::IO, ::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    operator = SnpLinAlg{T}(G; center = true, scale = true, impute = true)
    D = zeros(T, m, BLOCK)
    tables = zeros(T, 4, 256, Threads.nthreads())
    checksums = String[]
    for k in KS
        X = randn(Xoshiro(1), T, n, k)
        Y = randn(Xoshiro(2), T, m, k)
        AX = zeros(T, m, k)
        AtY = zeros(T, n, k)
        AX_decoded = zeros(T, m, k)
        AtY_decoded = zeros(T, n, k)
        fmas = Float64(m) * n * k

        mul!(AX, operator, X)
        seconds = @elapsed mul!(AX, operator, X)
        print_row(io, T, "A*X", k, "SnpLinAlg", seconds, fmas, nothing)
        decode_blas_AX!(AX_decoded, D, tables, operator, X)
        t0 = time_ns()
        phases = decode_blas_AX!(AX_decoded, D, tables, operator, X)
        seconds = (time_ns() - t0) / 1e9
        print_row(io, T, "A*X", k, "decode + gemm", seconds, fmas, phases)

        mul!(AtY, transpose(operator), Y)
        seconds = @elapsed mul!(AtY, transpose(operator), Y)
        print_row(io, T, "Aᵀ*Y", k, "SnpLinAlg", seconds, fmas, nothing)
        decode_blas_AtY!(AtY_decoded, D, tables, operator, Y)
        t0 = time_ns()
        phases = decode_blas_AtY!(AtY_decoded, D, tables, operator, Y)
        seconds = (time_ns() - t0) / 1e9
        print_row(io, T, "Aᵀ*Y", k, "decode + gemm", seconds, fmas, phases)

        push!(checksums, checksum_row(T, "A*X", k, AX, AX_decoded))
        push!(checksums, checksum_row(T, "Aᵀ*Y", k, AtY, AtY_decoded))
    end
    return checksums
end

function main()
    BLOCK > 0 || throw(ArgumentError("block must be positive"))
    all(>(0), KS) || throw(ArgumentError("every k must be positive"))
    BLAS.set_num_threads(Threads.nthreads())
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `decode_blas.jl` on $(Sys.CPU_NAME)")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    G = SnpArray(BED)
    print_header(io, size(G, 1), size(G, 2))
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "Second (warm) run of each product.")
    emit(io, "")
    emit(io, "| Type | Product | k | Path | seconds | GFMA/s | " *
             "decode seconds | gemm seconds |")
    emit(io, "|---|---|---:|---|---:|---:|---:|---:|")
    checksums = String[]
    for T in (Float32, Float64)
        append!(checksums, run_type(io, T, G))
    end
    emit(io, "")
    emit(io, "## Checksums")
    emit(io, "")
    emit(io, "| Type | Product | k | sum(abs2) SnpLinAlg | " *
             "sum(abs2) decode + gemm | max abs difference |")
    emit(io, "| --- | --- | ---: | ---: | ---: | ---: |")
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
