# Benchmark the MtlSnpArray mul! kernels on simulated genotypes against the
# alternative kernel and MPS on the materialized dense matrix.
#
#     julia --project=<env with Metal and SnpArrays> \
#       benchmark/metal_kernels.jl [m] [n] [prefix] [results_dir] [sweep]
#
# The report is also written to a timestamped markdown file in `results_dir`.
# Defaults: m = 4096, n = 32768, prefix = "metal_kernels", results_dir =
# `results/` next to this script. The defaults are smaller than those of
# `cuda_kernels.jl` so the dense baseline, its host copy and the genotypes
# stay well under 4 GB on a laptop with unified memory. `<prefix>.bed` is
# reused when present and simulated otherwise. A fifth argument `sweep` adds
# the tuning sweep of the kernel constants.
using Dates
using LinearAlgebra
using Metal
using Printf
using Random
using SnpArrays

const M = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4096
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 32768
const PREFIX = length(ARGS) >= 3 ? ARGS[3] : "metal_kernels"
const RESULTS_DIR = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "results")
const SWEEP = length(ARGS) >= 5 && ARGS[5] == "sweep"
const RHS_COUNTS = (8, 32, 128)
const SWEEP_RHS_COUNTS = (2, 4, 8, 9, 12, 16, 24, 32, 48, 64, 96, 128)
const SPLIT_TARGETS = (1, 16, 32, 64, 128, 256, 512, 1024)
const EXT = Base.get_extension(SnpArrays, :SnpArraysMetalExt)

# Apple GPU limits of the kernels' launches.
const MAX_THREADS = 1024
const MAX_SHARED_BYTES = 32768

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
gpu_name() = replace(String(Metal.device().name), r"[^A-Za-z0-9]+" => "-")

"""
    report_path(dir) -> String

Create `dir` and return the timestamped report path in it; an existing file
at that path is an error.
"""
function report_path(dir::AbstractString)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    path = joinpath(dir, "metal_kernels_$(gpu_name())_$(stamp).md")
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
    emit(io, "")
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "GPU", String(Metal.device().name))
    print_fact(io, "RAM", Base.format_bytes(Sys.total_memory()))
    print_fact(io, "Metal.jl", pkgversion(Metal))
    print_fact(io, "macOS", Metal.macos_version())
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    return nothing
end

"""
    time_call(f!, args...; repeats = 7) -> (minimum, median)

Wall time of `Metal.@sync f!(args...)` in seconds, after one warm-up call.
"""
function time_call(f!::F, args...; repeats::Int = 7) where F
    Metal.@sync f!(args...)
    times = Vector{Float64}(undef, repeats)
    for r in 1:repeats
        start = time_ns()
        Metal.@sync f!(args...)
        times[r] = (time_ns() - start) / 1e9
    end
    sort!(times)
    return times[1], times[(repeats + 1) ÷ 2]
end

"""
    relative_difference(result, expected) -> Float64

`norm(result - expected) / norm(expected)`, or zero when `expected` is
`nothing`.
"""
function relative_difference(result::Array{T},
                             expected::Union{Nothing, Array{T}}) where T
    expected === nothing && return 0.0
    return Float64(norm(result .- expected) / norm(expected))
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
    out::MtlArray{T},
    A::AbstractMatrix,
    rhs::MtlArray{T},
    expected::Union{Nothing, Array{T}},
) where {T <: AbstractFloat, F}
    times = time_call(f!, out, A, rhs)
    result = Array(out)
    fmas = Float64(size(A, 1)) * size(A, 2) * k
    emit(io, @sprintf(
        "| %s | %s | %d | %s | %.3f | %.3f | %.2f | %.1e |",
        T, product, k, variant, 1000 * times[1], 1000 * times[2],
        fmas / times[1] / 1e9, relative_difference(result, expected),
    ))
    return result
end

"""
    sweep_row!(io, label, f!, out, A, rhs, expected)

Time `f!(out, A, rhs)` and print `label`, the minimum and median in ms and
the relative difference from `expected`.
"""
function sweep_row!(io::IO, label::AbstractString, f!::F, out::MtlArray{T},
                    A::AbstractMatrix, rhs::MtlArray{T},
                    expected::Array{T}) where {T, F}
    times = time_call(f!, out, A, rhs)
    emit(io, @sprintf("| %s | %.3f | %.3f | %.1e |", label, 1000 * times[1],
                      1000 * times[2],
                      relative_difference(Array(out), expected)))
    return nothing
end

"""
    tiled_mul!(out, A, X) -> out

Set `out = A * X` with the tiled kernel.
"""
function tiled_mul!(out::MtlMatrix{T}, A::MtlSnpArray{T},
                    X::MtlMatrix{T}) where T <: AbstractFloat
    return EXT._tiled_mul!(out, A, X)
end

"""
    tiled_t_mul!(out, At, X) -> out

Set `out = transpose(A) * X` with the tiled kernel.
"""
function tiled_t_mul!(out::MtlMatrix{T}, At::Transpose{T, <:MtlSnpArray{T}},
                      X::MtlMatrix{T}) where T <: AbstractFloat
    return EXT._tiled_t_mul!(out, parent(At), X)
end

"""
    simd_mul!(out, A, X) -> out

Set `out = A * X` with the simdgroup-matrix kernel.
"""
function simd_mul!(out::MtlMatrix{Float32}, A::MtlSnpArray{Float32},
                   X::MtlMatrix{Float32})
    return EXT._simd_mul!(out, A, X)
end

"""
    simd_t_mul!(out, At, X) -> out

Set `out = transpose(A) * X` with the simdgroup-matrix kernel.
"""
function simd_t_mul!(out::MtlMatrix{Float32},
                     At::Transpose{Float32, <:MtlSnpArray{Float32}},
                     X::MtlMatrix{Float32})
    return EXT._simd_t_mul!(out, parent(At), X)
end

"""
    free_operator!(A)

Free the device arrays of `A`.
"""
function free_operator!(A::MtlSnpArray)
    for array in (A.data, A.μ, A.σinv, A.values)
        Metal.unsafe_free!(array)
    end
    return nothing
end

"""
    dense_matrix(G) -> MtlMatrix{Float32}

The centered, scaled, mean-imputed `G` on the device; the host copy is
released before returning.
"""
function dense_matrix(G::SnpArray)
    host = convert(Matrix{Float32}, G; center = true, scale = true,
                   impute = true)
    dense = MtlArray(host)
    host = nothing
    GC.gc()
    return dense
end

"""
    run_type(io, ::Type{T}, G)

Print the table rows of every product for element type `T`.
"""
function run_type(io::IO, ::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    A = MtlSnpArray{T}(G; center = true, scale = true, impute = true)
    dense = dense_matrix(G)
    x = MtlArray(randn(Xoshiro(1), T, n))
    out = MtlVector{T}(undef, m)
    # Raise the GPU clock before the first timed row.
    for _ in 1:50
        Metal.@sync mul!(out, A, x)
    end
    expected = bench_row!(io, T, "A*x", 1, "mul!", mul!, out, A, x, nothing)
    bench_row!(io, T, "A*x", 1, "MPS dense", mul!, out, dense, x, expected)
    y = MtlArray(randn(Xoshiro(2), T, m))
    out_t = MtlVector{T}(undef, n)
    expected = bench_row!(io, T, "Aᵀ*y", 1, "mul!", mul!, out_t,
                          transpose(A), y, nothing)
    bench_row!(io, T, "Aᵀ*y", 1, "MPS dense", mul!, out_t, transpose(dense),
               y, expected)
    for k in RHS_COUNTS
        X = MtlArray(randn(Xoshiro(3), T, n, k))
        Y = MtlArray(randn(Xoshiro(4), T, m, k))
        AX = MtlMatrix{T}(undef, m, k)
        AtX = MtlMatrix{T}(undef, n, k)
        simd = EXT._uses_simd(A, k)
        expected = bench_row!(
            io, T, "A*X", k, simd ? "mul! (simdgroup)" : "mul! (tiled)",
            mul!, AX, A, X, nothing,
        )
        bench_row!(io, T, "A*X", k, simd ? "tiled" : "simdgroup",
                   simd ? tiled_mul! : simd_mul!, AX, A, X, expected)
        bench_row!(io, T, "A*X", k, "MPS dense", mul!, AX, dense, X, expected)
        simd = EXT._uses_simd_t(A, k)
        expected = bench_row!(
            io, T, "Aᵀ*X", k, simd ? "mul! (simdgroup)" : "mul! (tiled)",
            mul!, AtX, transpose(A), Y, nothing,
        )
        bench_row!(io, T, "Aᵀ*X", k, simd ? "tiled" : "simdgroup",
                   simd ? tiled_t_mul! : simd_t_mul!, AtX, transpose(A), Y,
                   expected)
        bench_row!(io, T, "Aᵀ*X", k, "MPS dense", mul!, AtX,
                   transpose(dense), Y, expected)
    end
    Metal.unsafe_free!(dense)
    free_operator!(A)
    return nothing
end

"""
    valid_tiled(tile, transposed) -> Bool

Whether the tiled kernel accepts `tile`: `(BM, BN, BK, TM, TK)` for `A*X` or
`(BM, BN, BK, TN, TK)` for `transpose(A)*X`.
"""
function valid_tiled(tile::NTuple{5, Int}, transposed::Bool)
    (BM, BN, BK, TB, TK) = tile
    lanes = transposed ? BN : BM
    lanes % TB == 0 && BK % TK == 0 && BM % 16 == 0 || return false
    threads = (lanes ÷ TB) * (BK ÷ TK)
    shared = transposed ? (BN + 1) * BM + BM * BK : (BM + 1) * BN + BN * BK
    return 32 <= threads <= MAX_THREADS && 4 * shared <= MAX_SHARED_BYTES
end

"""
    valid_simd(tile, transposed) -> Bool

Whether the simdgroup kernel accepts `tile`: `(BN, WM, WN, TM, TN)` for
`A*X` or `(BM, WM, WN, TM, TN)` for `transpose(A)*X`.
"""
function valid_simd(tile::NTuple{5, Int}, transposed::Bool)
    (step, WM, WN, TM, TN) = tile
    rows = 8 * TM * WM
    BK = 8 * TN * WN
    step % (transposed ? 16 : 8) == 0 || return false
    transposed || rows % 16 == 0 || return false
    shared = (rows + 4) * step + (step + 4) * BK + 12 * 8 * WM * WN
    return WM * WN * 32 <= MAX_THREADS && 4 * shared <= MAX_SHARED_BYTES
end

"""
    sweep_header(io, title, label)

Print a sweep table's heading and column header.
"""
function sweep_header(io::IO, title::AbstractString, label::AbstractString)
    emit(io, "")
    emit(io, "### $(title)")
    emit(io, "")
    emit(io, "| $(label) | min ms | median ms | rel. diff |")
    emit(io, "|---|---:|---:|---:|")
    return nothing
end

"""
    sweep_kernels(io, A, At)

Time the tiled and simdgroup kernels of both products at every `k` in
`SWEEP_RHS_COUNTS`.
"""
function sweep_kernels(io::IO, A::MtlSnpArray{Float32},
                       At::Transpose{Float32, <:MtlSnpArray{Float32}})
    m, n = size(A)
    sweep_header(io, "Kernel by k", "Product, k, kernel")
    for k in SWEEP_RHS_COUNTS
        X = MtlArray(randn(Xoshiro(3), Float32, n, k))
        Y = MtlArray(randn(Xoshiro(4), Float32, m, k))
        AX = MtlMatrix{Float32}(undef, m, k)
        AtX = MtlMatrix{Float32}(undef, n, k)
        expected = Array(mul!(AX, A, X))
        sweep_row!(io, "A*X, $(k), tiled", tiled_mul!, AX, A, X, expected)
        sweep_row!(io, "A*X, $(k), simdgroup", simd_mul!, AX, A, X, expected)
        expected = Array(mul!(AtX, At, Y))
        sweep_row!(io, "Aᵀ*X, $(k), tiled", tiled_t_mul!, AtX, At, Y,
                   expected)
        sweep_row!(io, "Aᵀ*X, $(k), simdgroup", simd_t_mul!, AtX, At, Y,
                   expected)
    end
    return nothing
end

"""
    sweep_splits(io, A, At)

Time each kernel at every split target in `SPLIT_TARGETS`.
"""
function sweep_splits(io::IO, A::MtlSnpArray{Float32},
                      At::Transpose{Float32, <:MtlSnpArray{Float32}})
    m, n = size(A)
    sweep_header(io, "Split target", "Kernel, k, split target")
    kernels = (
        ("A*X tiled", false, EXT._tiled_mul!, (8, 32)),
        ("A*X simdgroup", false, EXT._simd_mul!, (64, 128)),
        ("Aᵀ*X tiled", true, EXT._tiled_t_mul!, (8, 32)),
        ("Aᵀ*X simdgroup", true, EXT._simd_t_mul!, (64, 128)),
    )
    for (name, transposed, kernel!, ks) in kernels
        for k in ks
            rhs = MtlArray(randn(Xoshiro(3), Float32, transposed ? m : n, k))
            out = MtlMatrix{Float32}(undef, transposed ? n : m, k)
            expected = Array(mul!(out, transposed ? At : A, rhs))
            for target in SPLIT_TARGETS
                f! = (o, B, r) -> kernel!(o, B, r; split_groups = target)
                sweep_row!(io, "$(name), $(k), $(target)", f!, out, A, rhs,
                           expected)
            end
        end
    end
    return nothing
end

"""Candidate `A*X` tiled tiles `(BM, BN, BK, TM, TK)`."""
const AX_TILE_CANDIDATES = (
    (256, 16, 8, 8, 1), (128, 16, 8, 4, 1), (512, 16, 8, 8, 1),
    (256, 32, 8, 8, 1), (256, 16, 8, 4, 2), (256, 16, 8, 8, 2),
    (256, 16, 32, 8, 4), (256, 16, 32, 8, 8), (128, 16, 32, 4, 4),
    (128, 32, 32, 8, 4), (256, 16, 16, 8, 4), (512, 16, 32, 16, 4),
)

"""Candidate `transpose(A)*X` tiled tiles `(BM, BN, BK, TN, TK)`."""
const ATX_TILE_CANDIDATES = (
    (128, 32, 8, 1, 1), (128, 64, 8, 2, 1), (64, 64, 8, 2, 2),
    (256, 32, 8, 1, 1), (128, 32, 8, 1, 2), (64, 32, 8, 1, 1),
    (64, 64, 32, 2, 4), (64, 64, 32, 4, 4), (128, 64, 32, 2, 4),
    (64, 128, 32, 4, 4), (32, 64, 32, 2, 4), (64, 32, 32, 1, 4),
)

"""Candidate `A*X` simdgroup tiles `(BN, WM, WN, TM, TN)`."""
const AX_SIMD_CANDIDATES = (
    (16, 4, 2, 4, 4), (8, 4, 2, 4, 4), (32, 4, 2, 4, 4), (16, 2, 2, 4, 4),
    (16, 4, 1, 4, 4), (16, 2, 4, 4, 4), (16, 4, 2, 2, 4), (16, 4, 2, 4, 2),
    (32, 2, 2, 4, 4),
)

"""Candidate `transpose(A)*X` simdgroup tiles `(BM, WM, WN, TM, TN)`."""
const ATX_SIMD_CANDIDATES = (
    (32, 2, 4, 4, 4), (16, 2, 4, 4, 4), (32, 4, 2, 4, 4), (32, 2, 2, 4, 4),
    (64, 2, 2, 4, 4), (32, 2, 4, 2, 4), (32, 2, 4, 4, 2), (16, 4, 4, 4, 4),
)

"""
    sweep_tiles(io, A, At)

Time every valid candidate tile of each kernel at a `k` of its band.
"""
function sweep_tiles(io::IO, A::MtlSnpArray{Float32},
                     At::Transpose{Float32, <:MtlSnpArray{Float32}})
    m, n = size(A)
    sweep_header(io, "Tile shape", "Kernel, k, tile")
    kernels = (
        ("A*X tiled", false, EXT._tiled_mul!, AX_TILE_CANDIDATES,
         valid_tiled, (8, 32)),
        ("Aᵀ*X tiled", true, EXT._tiled_t_mul!, ATX_TILE_CANDIDATES,
         valid_tiled, (8, 32)),
        ("A*X simdgroup", false, EXT._simd_mul!, AX_SIMD_CANDIDATES,
         valid_simd, (64, 128)),
        ("Aᵀ*X simdgroup", true, EXT._simd_t_mul!, ATX_SIMD_CANDIDATES,
         valid_simd, (64, 128)),
    )
    for (name, transposed, kernel!, tiles, valid, ks) in kernels
        for k in ks
            rhs = MtlArray(randn(Xoshiro(3), Float32, transposed ? m : n, k))
            out = MtlMatrix{Float32}(undef, transposed ? n : m, k)
            expected = Array(mul!(out, transposed ? At : A, rhs))
            for tile in tiles
                valid(tile, transposed) || continue
                f! = (o, B, r) -> kernel!(o, B, r; tile)
                sweep_row!(io, "$(name), $(k), $(tile)", f!, out, A, rhs,
                           expected)
            end
        end
    end
    return nothing
end

"""
    sweep_vectors(io, A, At)

Time `A*x` and `transpose(A)*y` over the chunk rule and the threadgroup
size.
"""
function sweep_vectors(io::IO, A::MtlSnpArray{Float32},
                       At::Transpose{Float32, <:MtlSnpArray{Float32}})
    m, n = size(A)
    sweep_header(io, "Vector chunk rule",
                 "Product, threads, chunk threads, max chunks")
    x = MtlArray(randn(Xoshiro(1), Float32, n))
    y = MtlArray(randn(Xoshiro(2), Float32, m))
    out = MtlVector{Float32}(undef, m)
    out_t = MtlVector{Float32}(undef, n)
    expected = Array(mul!(out, A, x))
    expected_t = Array(mul!(out_t, At, y))
    settings = [(256, c, mc) for c in (1, 8192, 32768, 131072)
                for mc in (8, 32, 128)]
    append!(settings, [(t, 32768, 32) for t in (64, 128, 512, 1024)])
    for (threads, chunk_threads, max_chunks) in settings
        label = "$(threads), $(chunk_threads), $(max_chunks)"
        f! = (o, B, r) -> EXT._direct_mul!(o, B, r; threads, chunk_threads,
                                           max_chunks)
        sweep_row!(io, "A*x, $(label)", f!, out, A, x, expected)
        f_t! = (o, B, r) -> EXT._direct_t_mul!(o, parent(B), r; threads,
                                               chunk_threads, max_chunks)
        sweep_row!(io, "Aᵀ*y, $(label)", f_t!, out_t, At, y, expected_t)
    end
    return nothing
end

"""
    run_sweep(io, G)

Print the tuning sweep tables for `Float32`.
"""
function run_sweep(io::IO, G::SnpArray)
    A = MtlSnpArray{Float32}(G; center = true, scale = true, impute = true)
    At = transpose(A)
    emit(io, "")
    emit(io, "## Tuning sweep")
    emit(io, "")
    emit(io, "Float32. The last column is the relative difference from " *
             "`mul!` with the default constants.")
    sweep_kernels(io, A, At)
    sweep_splits(io, A, At)
    sweep_tiles(io, A, At)
    sweep_vectors(io, A, At)
    free_operator!(A)
    return nothing
end

function main()
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `metal_kernels.jl` on $(Metal.device().name)")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    G = load_genotypes(io, M, N, PREFIX)
    print_header(io, M, N)
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "Centered, scaled, mean-imputed. Times are wall times of " *
             "`Metal.@sync` calls. The last column is the relative " *
             "difference from the `mul!` row of the same product.")
    emit(io, "")
    emit(io, "| Type | Product | k | Variant | min ms | median ms | " *
             "GFMA/s | rel. diff |")
    emit(io, "|---|---|---:|---|---:|---:|---:|---:|")
    run_type(io, Float32, G)
    SWEEP && run_sweep(io, G)
    emit(io, "")
    emit(io, "## Peak RSS")
    emit(io, "")
    emit(io, @sprintf("%.0f MiB", Sys.maxrss() / 2^20))
    close(io)
    println(path)
    return nothing
end

main()
