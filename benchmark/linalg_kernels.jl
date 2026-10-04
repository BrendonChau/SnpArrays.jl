# Benchmark the SnpLinAlg mul! kernels, the register-tile height MR, and the
# lookup kernel, so AVX2 and AVX-512 nodes can be compared with each other and
# with the registered SnpArrays (LoopVectorization kernels).
#
#     julia -t N --project=. benchmark/linalg_kernels.jl \
#       [m] [n] [prefix] [results_dir]
#
# The report is also written to a timestamped markdown file in `results_dir`,
# with the register-tile assembly in two `.s` files beside it.
# Defaults: m = 8192, n = 49152, prefix = "linalg_kernels", results_dir =
# `results/` next to this script. Run the fork first; it simulates
# `<prefix>.bed` and `<prefix>.fam`. Then run the registered package on the
# same files from a second process:
#
#     julia -t 12 --project=. benchmark/linalg_kernels.jl 8192 49152 /tmp/lk
#     julia -t 12 --project=/path/to/scratch \
#       benchmark/linalg_kernels.jl 8192 49152 /tmp/lk
#
# The upstream run skips the fork-only rows and the generated-code step. Compare
# the `sum(abs2)` rows of the two runs before comparing timings.
using Dates
using InteractiveUtils
using LinearAlgebra
using Printf
using Random
using SnpArrays

const M = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8192
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 49152
const PREFIX = length(ARGS) >= 3 ? ARGS[3] : "linalg_kernels"
const RESULTS_DIR = length(ARGS) >= 4 ? ARGS[4] : joinpath(@__DIR__, "results")
const TILE_ROWS = (1, 2, 4, 6, 8)
const RHS_COUNTS = (8, 32, 128)

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
    path = joinpath(dir, "linalg_kernels_$(Sys.CPU_NAME)_$(kind)_$(stamp).md")
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
    simulate_bed(io, path, m, n)

Simulate `m x n` genotypes (MAFs uniform on [0.05, 0.5], 1% missing per SNP),
time it, and write them to `path`.
"""
function simulate_bed(io::IO, path::AbstractString, m::Int, n::Int)
    rng = Xoshiro(1)
    mafs = 0.05 .+ 0.45 .* rand(rng, n)
    rates = fill(0.01, n)
    warmup = SnpArrays.simulate(Xoshiro(0), 8, 8, fill(0.3, 8))
    SnpArrays.simulate_missing!(Xoshiro(0), warmup, fill(0.01, 8))
    started = time_ns()
    G = SnpArrays.simulate(rng, m, n, mafs)
    SnpArrays.simulate_missing!(rng, G, rates)
    seconds = (time_ns() - started) / 1e9
    emit(io, @sprintf(
        "Simulated %d x %d genotypes in %.3f s (%.3e genotypes/s).",
        m, n, seconds, Float64(m) * n / seconds,
    ))
    write_bed(path, G)
    return nothing
end

"""
    load_genotypes(io, m, n, prefix) -> SnpArray

Read `<prefix>.bed`, simulating it first if absent and `simulate` exists.
An existing file is never overwritten; one of the wrong size is an error.
"""
function load_genotypes(io::IO, m::Int, n::Int, prefix::AbstractString)
    path = prefix * ".bed"
    if isfile(path)
        filesize(path) == 3 + cld(m, 4) * n || error(
            "$(path) has $(filesize(path)) bytes, not the $(3 + cld(m, 4) * n)",
            " of an $(m) x $(n) genotype matrix; remove it or change m, n",
        )
        emit(io, "Reusing `$(path)`.")
    elseif isdefined(SnpArrays, :simulate)
        simulate_bed(io, path, m, n)
    else
        error("$(path) is missing and this SnpArrays has no `simulate`; run ",
              "the fork first to create it")
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
    print_fact(io, "SnpArrays", has_fork_internals() ? "fork" : "upstream")
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "Sys.ARCH", Sys.ARCH)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    isdefined(SnpArrays, :VECTOR_BYTES) &&
        print_fact(io, "VECTOR_BYTES", SnpArrays.VECTOR_BYTES[])
    if isdefined(SnpArrays, :_register_tile_shape)
        for T in (Float32, Float64)
            print_fact(io, "_register_tile_shape($(T))",
                       SnpArrays._register_tile_shape(T))
        end
    end
    return nothing
end

"""
    print_steps(io, ::Type{T}, product, k, steps, tile)

Print one row of the tile-size table; `steps` is `(row_step, column_step,
rhs_step)` and `tile` the register tile.
"""
function print_steps(
    io::IO,
    ::Type{T},
    product::AbstractString,
    k::Int,
    steps::NTuple{3, Int},
    tile::AbstractString,
) where T
    emit(io, "| $(T) | $(product) | $(k) | $(steps[1]) | $(steps[2]) | " *
             "$(steps[3]) | $(tile) |")
    return nothing
end

"""
    print_tile_sizes(io, m, n)

Print the cache tile sizes (samples, SNPs, and rhs columns per block) and the
`MR x 2W` register tile of every product timed below.
"""
function print_tile_sizes(io::IO, m::Int, n::Int)
    emit(io, "")
    emit(io, "## Tile sizes")
    emit(io, "")
    emit(io, "| Type | Product | k | row_step | column_step | rhs_step | " *
             "register tile |")
    emit(io, "|---|---|---:|---:|---:|---:|---|")
    for T in (Float32, Float64)
        rows = SnpArrays._snparray_ax_row_step(T, m)
        print_steps(io, T, "A*x", 1, (rows, n, 1), "none")
        rows, columns = SnpArrays._snparray_atx_steps(T, n)
        print_steps(io, T, "Aᵀ*y", 1, (rows, columns, 1), "none")
        tile_rows = SnpArrays._register_tile_shape(T)[1]
        for k in RHS_COUNTS
            tile = "$(tile_rows) x $(2 * SnpArrays._rhs_width(T, k))"
            print_steps(io, T, "A*X", k, SnpArrays._snparray_AX_steps(T, m, k),
                        tile)
            print_steps(io, T, "Aᵀ*X", k,
                        SnpArrays._snparray_AtX_steps(T, n, k), tile)
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

"""
    print_row(io, ::Type{T}, product, k, variant, times, fmas)

Print one row of the timing table; `times` is `(minimum, median)` in seconds.
"""
function print_row(
    io::IO,
    ::Type{T},
    product::AbstractString,
    k::Int,
    variant::AbstractString,
    times::NTuple{2, Float64},
    fmas::Float64,
) where T
    emit(io, @sprintf(
        "| %s | %s | %d | %s | %.3f | %.3f | %.2f |",
        T, product, k, variant, 1000 * times[1], 1000 * times[2],
        fmas / times[1] / 1e9,
    ))
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
    sweep_tiled!(io, driver!, out, expected, operator, rhs, workspace, product,
                 k)

Check the direct driver against `expected` for each `MR` in `TILE_ROWS`, time
it, and print its table rows.
"""
function sweep_tiled!(
    io::IO,
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
        print_row(io, T, product, k, "tiled MR=$(rows)$(default)", times,
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
    run_type(io, ::Type{T}, G)

Print the table rows of every product for element type `T`.
"""
function run_type(io::IO, ::Type{T}, G::SnpArray) where T <: AbstractFloat
    m, n = size(G)
    operator = SnpLinAlg{T}(G; center = true, scale = true, impute = true)
    x = randn(Xoshiro(1), T, n)
    out = Vector{T}(undef, m)
    print_row(io, T, "A*x", 1, "mul!", time_call(mul!, out, operator, x),
              Float64(m) * n)
    y = randn(Xoshiro(2), T, m)
    out_t = Vector{T}(undef, n)
    print_row(io, T, "Aᵀ*y", 1, "mul!",
              time_call(mul!, out_t, transpose(operator), y), Float64(m) * n)
    workspace = T[]
    fork = has_fork_internals()
    for k in RHS_COUNTS
        X = randn(Xoshiro(3), T, n, k)
        Y = randn(Xoshiro(4), T, m, k)
        AX = Matrix{T}(undef, m, k)
        AtX = Matrix{T}(undef, n, k)
        fmas = Float64(m) * n * k
        print_row(io, T, "A*X", k, matrix_variant(m, k),
                  time_call(mul!, AX, operator, X), fmas)
        expected_AX = copy(AX)
        fork && sweep_tiled!(io, ax_register_tile!, AX, expected_AX, operator,
                             X, workspace, "A*X", k)
        print_row(io, T, "Aᵀ*X", k, fork ? "mul! (register-tiled)" : "mul!",
                  time_call(mul!, AtX, transpose(operator), Y), fmas)
        expected_AtX = copy(AtX)
        fork && sweep_tiled!(io, AtX_register_tile!, AtX, expected_AtX,
                             operator, Y, workspace, "Aᵀ*X", k)
    end
    return nothing
end

"""
    print_agreement(io, G)

Print `sum(abs2, ...)` of the four `Float64` products at `k = 8`, for
comparison between a fork run and an upstream run on the same `.bed`.
"""
function print_agreement(io::IO, G::SnpArray)
    m, n = size(G)
    k = 8
    operator = SnpLinAlg{Float64}(G; center = true, scale = true,
                                  impute = true)
    x = randn(Xoshiro(11), Float64, n)
    y = randn(Xoshiro(12), Float64, m)
    X = randn(Xoshiro(13), Float64, n, k)
    Y = randn(Xoshiro(14), Float64, m, k)
    emit(io, "")
    emit(io, "## Cross-package check")
    emit(io, "")
    emit(io, "Float64, k = 8.")
    emit(io, "")
    emit(io, "| Product | sum(abs2) |")
    emit(io, "| --- | ---: |")
    products = (
        mul!(Vector{Float64}(undef, m), operator, x),
        mul!(Vector{Float64}(undef, n), transpose(operator), y),
        mul!(Matrix{Float64}(undef, m, k), operator, X),
        mul!(Matrix{Float64}(undef, n, k), transpose(operator), Y),
    )
    for (name, product) in zip(("A*x", "Aᵀ*y", "A*X", "Aᵀ*X"), products)
        emit(io, @sprintf("| `%s` | %.17g |", name, sum(abs2, product)))
    end
    return nothing
end

"""
    write_native(io, path, f, types)

Write the Intel-syntax native code of `f` for `types` to `path`, and print the
number of FMA instruction lines in it.
"""
function write_native(
    io::IO,
    path::AbstractString,
    f::Function,
    types::Type{<:Tuple},
)
    asm = open(path, "w")
    code_native(asm, f, types; syntax = :intel, debuginfo = :none)
    close(asm)
    fmas = count(Base.Fix1(occursin, r"vfmadd|fmla"), readlines(path))
    emit(io, "| `$(basename(path))` | $(fmas) |")
    return nothing
end

"""
    write_generated_code(io, stem, G)

Write the `Float32` register-tile code at the host default `(MR, U = 2, W)`
to `<stem>_AX.s` and `<stem>_AtX.s`.
"""
function write_generated_code(io::IO, stem::AbstractString, G::SnpArray)
    operator = SnpLinAlg{Float32}(G; center = true, scale = true,
                                  impute = true)
    task = SnpArrays.RegisterTileTask(
        Matrix{Float32}(undef, 1, 1), operator.s.data, operator.values,
        Float32[], 0, 0,
    )
    tile = typeof(Val(SnpArrays._register_tile_shape(Float32)[1]))
    width = typeof(Val(SnpArrays._vector_width(Float32)))
    emit(io, "")
    emit(io, "## Generated code")
    emit(io, "")
    emit(io, "Generated code for `$(tile)`, `Val{2}`, `$(width)`.")
    emit(io, "")
    emit(io, "| File | Lines with vfmadd or fmla |")
    emit(io, "| --- | ---: |")
    write_native(io, stem * "_AX.s", SnpArrays._snparray_AX_register_tile!,
                 Tuple{typeof(task), Int, UnitRange{Int}, UnitRange{Int},
                       tile, Val{2}, width})
    write_native(io, stem * "_AtX.s", SnpArrays._snparray_AtX_register_tile!,
                 Tuple{typeof(task), UnitRange{Int}, Int, UnitRange{Int},
                       tile, Val{2}, width})
    return nothing
end

function main()
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `linalg_kernels.jl` on $(Sys.CPU_NAME)")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    G = load_genotypes(io, M, N, PREFIX)
    print_header(io, M, N)
    has_fork_internals() && print_tile_sizes(io, M, N)
    emit(io, "")
    emit(io, "## Timings")
    emit(io, "")
    emit(io, "| Type | Product | k | Variant | min ms | median ms | " *
             "GFMA/s |")
    emit(io, "|---|---|---:|---|---:|---:|---:|")
    for T in (Float32, Float64)
        run_type(io, T, G)
    end
    print_agreement(io, G)
    has_fork_internals() && write_generated_code(io, splitext(path)[1], G)
    emit(io, "")
    emit(io, "## Peak RSS")
    emit(io, "")
    emit(io, @sprintf("%.0f MiB", Sys.maxrss() / 2^20))
    close(io)
    println(path)
    return nothing
end

main()
