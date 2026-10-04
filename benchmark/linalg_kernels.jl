# Benchmark the SnpLinAlg mul! kernels, the register-tile height MR, and the
# lookup kernel, so AVX2 and AVX-512 nodes can be compared with each other and
# with the registered SnpArrays (LoopVectorization kernels).
#
#     julia -t N --project=. benchmark/linalg_kernels.jl [m] [n] [prefix]
#
# Defaults: m = 8192, n = 49152, prefix = "linalg_kernels". Run the fork
# first; it simulates `<prefix>.bed` and `<prefix>.fam`. Then run the registered
# package on the same files from a second process:
#
#     julia -t 12 --project=. benchmark/linalg_kernels.jl 8192 49152 /tmp/lk
#     julia -t 12 --project=/path/to/scratch \
#       benchmark/linalg_kernels.jl 8192 49152 /tmp/lk
#
# The upstream run skips the fork-only rows and the generated-code step. Compare
# the `sum(abs2)` lines of the two runs before comparing timings.
using InteractiveUtils
using LinearAlgebra
using Printf
using Random
using SnpArrays

const M = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8192
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 49152
const PREFIX = length(ARGS) >= 3 ? ARGS[3] : "linalg_kernels"
const TILE_ROWS = (1, 2, 4, 6, 8)
const RHS_COUNTS = (8, 32, 128)

"""
    has_fork_internals() -> Bool

Return whether the loaded SnpArrays has the register-tiled schedulers.
"""
has_fork_internals() = isdefined(SnpArrays, :_snparray_AX_spawn_tasks!)

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
    simulate_bed(path, m, n)

Simulate `m x n` genotypes (MAFs uniform on [0.05, 0.5], 1% missing per SNP),
time it, and write them to `path`.
"""
function simulate_bed(path::AbstractString, m::Int, n::Int)
    rng = Xoshiro(1)
    mafs = 0.05 .+ 0.45 .* rand(rng, n)
    rates = fill(0.01, n)
    warmup = SnpArrays.simulate(Xoshiro(0), 8, 8, fill(0.3, 8))
    SnpArrays.simulate_missing!(Xoshiro(0), warmup, fill(0.01, 8))
    started = time_ns()
    G = SnpArrays.simulate(rng, m, n, mafs)
    SnpArrays.simulate_missing!(rng, G, rates)
    seconds = (time_ns() - started) / 1e9
    @printf("simulated %d x %d genotypes in %.3f s (%.3e genotypes/s)\n",
            m, n, seconds, Float64(m) * n / seconds)
    write_bed(path, G)
    return nothing
end

"""
    load_genotypes(m, n, prefix) -> SnpArray

Read `<prefix>.bed`, simulating it first if absent and `simulate` exists.
An existing file is never overwritten; one of the wrong size is an error.
"""
function load_genotypes(m::Int, n::Int, prefix::AbstractString)
    path = prefix * ".bed"
    if isfile(path)
        filesize(path) == 3 + cld(m, 4) * n || error(
            "$(path) has $(filesize(path)) bytes, not the $(3 + cld(m, 4) * n)",
            " of an $(m) x $(n) genotype matrix; remove it or change m, n",
        )
        println("reusing ", path)
    elseif isdefined(SnpArrays, :simulate)
        simulate_bed(path, m, n)
    else
        error("$(path) is missing and this SnpArrays has no `simulate`; run ",
              "the fork first to create it")
    end
    G = SnpArray(path)
    size(G) == (m, n) || error("$(path) holds a $(size(G)) matrix, not ",
                               "$((m, n)); check its .fam file")
    return G
end

function print_header(m::Int, n::Int)
    println("threads = ", Threads.nthreads())
    println("Sys.CPU_NAME = ", Sys.CPU_NAME)
    println("Sys.ARCH = ", Sys.ARCH)
    println("VERSION = ", VERSION)
    println("pkgdir = ", pkgdir(SnpArrays))
    println("m = ", m, "   n = ", n)
    isdefined(SnpArrays, :VECTOR_BYTES) &&
        println("VECTOR_BYTES = ", SnpArrays.VECTOR_BYTES[])
    if isdefined(SnpArrays, :_register_tile_shape)
        for T in (Float32, Float64)
            println("_register_tile_shape(", T, ") = ",
                    SnpArrays._register_tile_shape(T))
        end
    end
    return nothing
end

"""
    time_call(f!, args...; repeats = 5) -> (minimum, median)

Time `f!(args...)` in seconds, after one warm-up call.
"""
function time_call(f!::F, args...; repeats::Int = 5) where F
    f!(args...)
    times = [@elapsed f!(args...) for _ in 1:repeats]
    sort!(times)
    return times[1], times[(repeats + 1) ÷ 2]
end

function print_row(
    ::Type{T},
    product::AbstractString,
    k::Int,
    variant::AbstractString,
    times::NTuple{2, Float64},
    fmas::Float64,
) where T
    @printf("| %s | %s | %d | %s | %.3f | %.3f | %.2f |\n", T, product, k,
            variant, 1000 * times[1], 1000 * times[2], fmas / times[1] / 1e9)
    return nothing
end

"""
    ax_register_tile!(out, operator, rhs, workspace, ::Val{MR}) -> out

Run the register-tiled branch of `_snparray_AX_schedule!` with `MR` rows.
"""
function ax_register_tile!(
    out::Matrix{T},
    operator::SnpLinAlg{T},
    rhs::Matrix{T},
    workspace::Vector{T},
    tile_rows::Val{MR},
) where {T <: AbstractFloat, MR}
    rows_filled = operator.s.m
    k = size(out, 2)
    fill!(out, zero(T))
    row_step, column_step, rhs_step =
        SnpArrays._snparray_AX_steps(T, rows_filled, k)
    lanes = SnpArrays._rhs_width(T, k)
    panel_length = 2lanes * column_step
    tasks = cld(rows_filled, row_step) * cld(k, rhs_step)
    SnpArrays._resize_workspace!(workspace, tasks * panel_length)
    return SnpArrays._snparray_AX_spawn_tasks!(
        out, operator.s.data, rhs, operator.values, rows_filled, workspace,
        row_step, column_step, rhs_step, panel_length, tile_rows, Val(lanes),
    )
end

"""
    AtX_register_tile!(out, operator, rhs, workspace, ::Val{MR}) -> out

Run the body of `_snparray_AtX_schedule!` over every SNP with `MR` columns.
"""
function AtX_register_tile!(
    out::Matrix{T},
    operator::SnpLinAlg{T},
    rhs::Matrix{T},
    workspace::Vector{T},
    tile_width::Val{MR},
) where {T <: AbstractFloat, MR}
    n = size(operator, 2)
    k = size(out, 2)
    fill!(out, zero(T))
    row_step, column_step, rhs_step = SnpArrays._snparray_AtX_steps(T, n, k)
    lanes = SnpArrays._rhs_width(T, k)
    panel_length = 2lanes * row_step
    tasks = cld(n, column_step) * cld(k, rhs_step)
    SnpArrays._resize_workspace!(workspace, tasks * panel_length)
    return SnpArrays._snparray_AtX_spawn_tasks!(
        out, operator.s.data, rhs, operator.values, operator.s.m, 1:n,
        workspace, row_step, column_step, rhs_step, panel_length, tile_width,
        Val(lanes),
    )
end

"""
    sweep_tiled!(driver!, out, expected, operator, rhs, workspace, product, k)

Check the direct driver against `expected` for each `MR` in `TILE_ROWS`, time
it, and print its table rows.
"""
function sweep_tiled!(
    driver!::F,
    out::Matrix{T},
    expected::Matrix{T},
    operator::SnpLinAlg{T},
    rhs::Matrix{T},
    workspace::Vector{T},
    product::AbstractString,
    k::Int,
) where {F, T}
    m, n = size(operator)
    for rows in TILE_ROWS
        tile = Val(rows)
        driver!(out, operator, rhs, workspace, tile)
        relative = norm(vec(out) .- vec(expected)) / norm(vec(expected))
        relative <= 32 * sqrt(T(max(m, n))) * eps(T) || error(
            "$(product) MR=$(rows) k=$(k) $(T) disagrees with mul!: ",
            "relative error $(relative)",
        )
        times = time_call(driver!, out, operator, rhs, workspace, tile)
        default = rows == SnpArrays._register_tile_shape(T)[1] ?
                  " (default)" : ""
        print_row(T, product, k, "tiled MR=$(rows)$(default)", times,
                  Float64(m) * n * k)
    end
    return nothing
end

function matrix_variant(m::Int, k::Int)
    isdefined(SnpArrays, :_uses_lookup_kernel) || return "mul!"
    return SnpArrays._uses_lookup_kernel(m, k) ? "mul! (lookup)" :
           "mul! (register-tiled)"
end

"""
    run_type(::Type{T}, G)

Print the table rows of every product for element type `T`.
"""
function run_type(::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    operator = SnpLinAlg{T}(G; center = true, scale = true, impute = true)
    x = randn(Xoshiro(1), T, n)
    out = Vector{T}(undef, m)
    print_row(T, "A*x", 1, "mul!", time_call(mul!, out, operator, x),
              Float64(m) * n)
    y = randn(Xoshiro(2), T, m)
    out_t = Vector{T}(undef, n)
    print_row(T, "Aᵀ*y", 1, "mul!",
              time_call(mul!, out_t, transpose(operator), y), Float64(m) * n)
    workspace = T[]
    fork = has_fork_internals()
    for k in RHS_COUNTS
        X = randn(Xoshiro(3), T, n, k)
        Y = randn(Xoshiro(4), T, m, k)
        AX = Matrix{T}(undef, m, k)
        AtX = Matrix{T}(undef, n, k)
        fmas = Float64(m) * n * k
        print_row(T, "A*X", k, matrix_variant(m, k),
                  time_call(mul!, AX, operator, X), fmas)
        expected_AX = copy(AX)
        fork && sweep_tiled!(ax_register_tile!, AX, expected_AX, operator, X,
                             workspace, "A*X", k)
        print_row(T, "Aᵀ*X", k, fork ? "mul! (register-tiled)" : "mul!",
                  time_call(mul!, AtX, transpose(operator), Y), fmas)
        expected_AtX = copy(AtX)
        fork && sweep_tiled!(AtX_register_tile!, AtX, expected_AtX, operator,
                             Y, workspace, "Aᵀ*X", k)
    end
    return nothing
end

"""
    print_agreement(G)

Print `sum(abs2, ...)` of the four `Float64` products at `k = 8`, for
comparison between a fork run and an upstream run on the same `.bed`.
"""
function print_agreement(G::SnpArray)
    m, n = size(G)
    k = 8
    operator = SnpLinAlg{Float64}(G; center = true, scale = true,
                                  impute = true)
    x = randn(Xoshiro(11), Float64, n)
    y = randn(Xoshiro(12), Float64, m)
    X = randn(Xoshiro(13), Float64, n, k)
    Y = randn(Xoshiro(14), Float64, m, k)
    println("\nCross-package check, Float64, k = 8")
    products = (
        mul!(Vector{Float64}(undef, m), operator, x),
        mul!(Vector{Float64}(undef, n), transpose(operator), y),
        mul!(Matrix{Float64}(undef, m, k), operator, X),
        mul!(Matrix{Float64}(undef, n, k), transpose(operator), Y),
    )
    for (name, product) in zip(("A*x ", "Aᵀ*y", "A*X ", "Aᵀ*X"), products)
        @printf("sum(abs2, %s) = %.17g\n", name, sum(abs2, product))
    end
    return nothing
end

"""
    write_native(path, f, types)

Write the Intel-syntax native code of `f` for `types` to `path`, and print the
number of FMA instruction lines in it.
"""
function write_native(path::AbstractString, f::Function, types::Type{<:Tuple})
    io = open(path, "w")
    code_native(io, f, types; syntax = :intel, debuginfo = :none)
    close(io)
    fmas = count(Base.Fix1(occursin, r"vfmadd|fmla"), readlines(path))
    println(path, ": ", fmas, " lines with vfmadd or fmla")
    return nothing
end

"""
    write_generated_code(prefix, G)

Write the `Float32` register-tile code at the host default `(MR, U = 2, W)`.
"""
function write_generated_code(prefix::AbstractString, G::SnpArray)
    operator = SnpLinAlg{Float32}(G; center = true, scale = true,
                                  impute = true)
    task = SnpArrays.RegisterTileTask(
        Matrix{Float32}(undef, 1, 1), operator.s.data, operator.values,
        Float32[], 0, 0,
    )
    tile = typeof(Val(SnpArrays._register_tile_shape(Float32)[1]))
    width = typeof(Val(SnpArrays._vector_width(Float32)))
    println("\nGenerated code for ", tile, ", Val{2}, ", width)
    write_native(prefix * "_AX.s", SnpArrays._snparray_AX_register_tile!,
                 Tuple{typeof(task), Int, UnitRange{Int}, UnitRange{Int},
                       tile, Val{2}, width})
    write_native(prefix * "_AtX.s", SnpArrays._snparray_AtX_register_tile!,
                 Tuple{typeof(task), UnitRange{Int}, Int, UnitRange{Int},
                       tile, Val{2}, width})
    return nothing
end

function main()
    G = load_genotypes(M, N, PREFIX)
    print_header(M, N)
    println("\n| Type | Product | k | Variant | min ms | median ms | GFMA/s |")
    println("|---|---|---:|---|---:|---:|---:|")
    for T in (Float32, Float64)
        run_type(T, G)
    end
    print_agreement(G)
    has_fork_internals() && write_generated_code(PREFIX, G)
    @printf("\npeak RSS = %.0f MiB\n", Sys.maxrss() / 2^20)
    return nothing
end

main()
