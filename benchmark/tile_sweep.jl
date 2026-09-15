# Tile-size sweep for the SnpLinAlg SIMD kernels.
using LinearAlgebra
using Printf
using Random
using SnpArrays

const R = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
const DECODE_WIDTH = SnpArrays.DECODE_WIDTH

function stack_snparray(eur::SnpArray, R::Int)
    m, n = size(eur)
    stacked = SnpArray(undef, R * m, n)
    for r in 1:R
        stacked[(r - 1) * m + 1:r * m, :] .= view(eur, :, :)
    end
    return stacked
end

# One timed call: minimum of `repeats` elapsed times, after one warm-up call.
function min_elapsed(f!::F, args...; repeats::Int = 3) where F
    f!(args...)
    best = Inf
    for _ in 1:repeats
        t = @elapsed f!(args...)
        best = min(best, t)
    end
    return best
end

"""
    ax_tile!(out, packed, rhs, values, rows_filled, row_step, column_step)

Mirror `SnpArrays._snparray_ax_tile!` with the tile sizes supplied instead of
taken from `_tile_sizes`.
"""
function ax_tile!(
    out::Vector{T},
    packed::Matrix{UInt8},
    rhs::Vector{T},
    values::Matrix{T},
    rows_filled::Int,
    row_step::Int,
    column_step::Int,
) where T <: AbstractFloat
    n = size(packed, 2)
    @assert row_step % DECODE_WIDTH == 0
    fill!(out, zero(T))
    @sync begin
        for row_first in 1:row_step:rows_filled
            row_last = min(row_first + row_step - 1, rows_filled)
            @assert (row_first - 1) % 4 == 0
            Threads.@spawn begin
                for column_first in 1:column_step:n
                    column_last = min(column_first + column_step - 1, n)
                    SnpArrays._snparray_ax_kernel!(
                        out, packed, rhs, values, $row_first, $row_last,
                        column_first, column_last,
                    )
                end
            end
        end
    end
    return out
end

"""
    AX_tile!(out, packed, rhs, values, rows_filled, row_step, column_step,
        rhs_step)

Mirror `SnpArrays._snparray_AX_tile!` with the tile sizes supplied.
"""
function AX_tile!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    rows_filled::Int,
    row_step::Int,
    column_step::Int,
    rhs_step::Int,
) where T <: AbstractFloat
    k = size(out, 2)
    lanes = SnpArrays._rhs_width(T, k)
    width = Val(lanes)
    @assert row_step % DECODE_WIDTH == 0
    fill!(out, zero(T))
    panel_length = 2lanes * column_step
    tasks = cld(rows_filled, row_step) * cld(k, rhs_step)
    panel = zeros(T, tasks * panel_length)
    task_index = 0
    @sync begin
        for rhs_first in 1:rhs_step:k
            rhs_last = min(rhs_first + rhs_step - 1, k)
            for row_first in 1:row_step:rows_filled
                row_last = min(row_first + row_step - 1, rows_filled)
                @assert (row_first - 1) % 4 == 0
                panel_offset = task_index * panel_length
                task_index += 1
                Threads.@spawn SnpArrays._snparray_AX_kernel!(
                    out, packed, rhs, values, panel, $panel_offset,
                    $row_first, $row_last, $column_step, $rhs_first,
                    $rhs_last, $width,
                )
            end
        end
    end
    return out
end

"""
    atx_tile!(out, packed, rhs, values, rows_filled, row_step, column_step)

Mirror `SnpArrays._snparray_atx_tile!` with the tile sizes supplied.
"""
function atx_tile!(
    out::Vector{T},
    packed::Matrix{UInt8},
    rhs::Vector{T},
    values::Matrix{T},
    rows_filled::Int,
    row_step::Int,
    column_step::Int,
) where T <: AbstractFloat
    n = size(packed, 2)
    @assert row_step % DECODE_WIDTH == 0
    fill!(out, zero(T))
    @sync begin
        for column_first in 1:column_step:n
            column_last = min(column_first + column_step - 1, n)
            Threads.@spawn begin
                for row_first in 1:row_step:rows_filled
                    row_last = min(row_first + row_step - 1, rows_filled)
                    @assert (row_first - 1) % 4 == 0
                    SnpArrays._snparray_atx_kernel!(
                        out, packed, rhs, values, row_first, row_last,
                        $column_first, $column_last, 0,
                    )
                end
            end
        end
    end
    return out
end

"""
    AtX_tile!(out, packed, rhs, values, rows_filled, row_step, column_step,
        rhs_step)

Mirror `SnpArrays._snparray_AtX_tile!` with the tile sizes supplied.
"""
function AtX_tile!(
    out::Matrix{T},
    packed::Matrix{UInt8},
    rhs::Matrix{T},
    values::Matrix{T},
    rows_filled::Int,
    row_step::Int,
    column_step::Int,
    rhs_step::Int,
) where T <: AbstractFloat
    n = size(packed, 2)
    k = size(out, 2)
    lanes = SnpArrays._rhs_width(T, k)
    width = Val(lanes)
    @assert row_step % DECODE_WIDTH == 0
    fill!(out, zero(T))
    panel_length = 2lanes * row_step
    tasks = cld(n, column_step) * cld(k, rhs_step)
    panel = zeros(T, tasks * panel_length)
    task_index = 0
    @sync begin
        for rhs_first in 1:rhs_step:k
            rhs_last = min(rhs_first + rhs_step - 1, k)
            for column_first in 1:column_step:n
                column_last = min(column_first + column_step - 1, n)
                panel_offset = task_index * panel_length
                task_index += 1
                Threads.@spawn SnpArrays._snparray_AtX_kernel!(
                    out, packed, rhs, values, panel, $panel_offset,
                    $row_step, rows_filled, $column_first, $column_last,
                    $rhs_first, $rhs_last, 0, $width,
                )
            end
        end
    end
    return out
end

relative_error(a::AbstractArray, b::AbstractArray) =
    norm(vec(a) .- vec(b)) / norm(vec(b))

"""
    verify(eur)

Check each explicit-step driver against `mul!` on an unstacked fixture, and
error out before any timing if a driver was mis-transcribed.
"""
function verify(eur::SnpArray)
    m, n = size(eur)
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(eur; center = true, scale = true,
                                impute = true)
        packed = operator.s.data
        values = operator.values
        rows_filled = operator.s.m
        tolerance = 32 * sqrt(T(max(m, n))) * eps(T)
        k = 8

        x = randn(Xoshiro(1), T, n)
        expected = mul!(Vector{T}(undef, m), operator, x)
        actual = ax_tile!(Vector{T}(undef, m), packed, x, values,
                          rows_filled, 256, n)
        relative_error(actual, expected) <= tolerance ||
            error("ax_tile! disagrees with mul! for $T")

        y = randn(Xoshiro(2), T, m)
        expected_t = mul!(Vector{T}(undef, n), transpose(operator), y)
        actual_t = atx_tile!(Vector{T}(undef, n), packed, y, values,
                             rows_filled, 1024, 1136)
        relative_error(actual_t, expected_t) <= tolerance ||
            error("atx_tile! disagrees with mul! for $T")

        X = randn(Xoshiro(3), T, n, k)
        expected_matrix = mul!(Matrix{T}(undef, m, k), operator, X)
        actual_matrix = AX_tile!(Matrix{T}(undef, m, k), packed, X, values,
                                 rows_filled, 256, 512, k)
        relative_error(actual_matrix, expected_matrix) <= tolerance ||
            error("AX_tile! disagrees with mul! for $T")

        Y = randn(Xoshiro(4), T, m, k)
        expected_tmatrix =
            mul!(Matrix{T}(undef, n, k), transpose(operator), Y)
        actual_tmatrix = AtX_tile!(Matrix{T}(undef, n, k), packed, Y, values,
                                   rows_filled, 1024, 1136, k)
        relative_error(actual_tmatrix, expected_tmatrix) <= tolerance ||
            error("AtX_tile! disagrees with mul! for $T")
    end
    println("verification ok")
    return nothing
end

function print_header(R::Int, m::Int, n::Int)
    println("Threads.nthreads() = ", Threads.nthreads())
    println("Sys.CPU_NAME = ", Sys.CPU_NAME)
    println("VERSION = ", VERSION)
    println("R = ", R)
    println("samples (m) = ", m)
    println("SNPs (n) = ", n)
    println("VECTOR_BYTES = ", SnpArrays.VECTOR_BYTES[])
    for T in (Float32, Float64)
        println("_micro_tile(", T, ") = ", SnpArrays._micro_tile(T))
    end
    return nothing
end

"""
    panel_bytes(::Type{T}, tasks, lanes_step)

Total per-call panel-buffer bytes: one buffer of `_micro_tile(T)[2] *
lanes_step` elements per spawned task.
"""
function panel_bytes(::Type{T}, tasks::Int, lanes_step::Int) where T
    _, tile_lanes = SnpArrays._micro_tile(T)
    return tasks * tile_lanes * lanes_step * sizeof(T)
end

function report_line(
    label::AbstractString,
    steps::NTuple{3, Int},
    seconds::Float64,
    flops::Float64,
    bytes::Int,
)
    @printf("%-34s (%6d,%6d,%4d)  %10.3f ms  %8.2f GFMA/s  panel %8.2f MB\n",
            label, steps[1], steps[2], steps[3], seconds * 1000,
            flops / seconds / 1e9, bytes / 2^20)
    return nothing
end

function report_best(
    label::AbstractString,
    steps::NTuple{3, Int},
    seconds::Float64,
    baseline::Float64,
)
    @printf("BEST %-29s (%6d,%6d,%4d)  %10.3f ms  %6.2fx over default\n",
            label, steps[1], steps[2], steps[3], seconds * 1000,
            baseline / seconds)
    return nothing
end

"""
    sweep_s1(stacked, m, n)

Time the production `mul!` path at both `VECTOR_BYTES` settings, the only
sweep that exercises `_tile_sizes` itself rather than explicit steps.
`_micro_tile` and `_vector_width` read the `Ref` at run time, so the kernels
and the drivers stay consistent within a block.
"""
function sweep_s1(stacked::SnpArray, m::Int, n::Int)
    println("\n===== S1: VECTOR_BYTES =====")
    for bytes in (64, 32)
        SnpArrays.VECTOR_BYTES[] = bytes
        for T in (Float32, Float64)
            operator = SnpLinAlg{T}(stacked; center = true, scale = true,
                                    impute = true)
            for k in (1, 8, 32)
                flops = Float64(m) * n * k
                if k == 1
                    x = randn(Xoshiro(1), T, n)
                    out = Vector{T}(undef, m)
                    seconds = min_elapsed(mul!, out, operator, x)
                    report_line(@sprintf("vb%-3d %-8s A*x   k=%-4d", bytes,
                                         T, k), (0, 0, k), seconds, flops, 0)
                    y = randn(Xoshiro(2), T, m)
                    out_t = Vector{T}(undef, n)
                    seconds = min_elapsed(mul!, out_t, transpose(operator), y)
                    report_line(@sprintf("vb%-3d %-8s Aᵀ*y  k=%-4d", bytes,
                                         T, k), (0, 0, k), seconds, flops, 0)
                else
                    X = randn(Xoshiro(3), T, n, k)
                    out = Matrix{T}(undef, m, k)
                    seconds = min_elapsed(mul!, out, operator, X)
                    report_line(@sprintf("vb%-3d %-8s A*X   k=%-4d", bytes,
                                         T, k), (0, 0, k), seconds, flops, 0)
                    Y = randn(Xoshiro(4), T, m, k)
                    out_t = Matrix{T}(undef, n, k)
                    seconds = min_elapsed(mul!, out_t, transpose(operator), Y)
                    report_line(@sprintf("vb%-3d %-8s Aᵀ*Y  k=%-4d", bytes,
                                         T, k), (0, 0, k), seconds, flops, 0)
                end
            end
        end
    end
    SnpArrays.VECTOR_BYTES[] = 64
    return nothing
end

function grid_with_default(candidates::Vector{Int}, default::Int)
    return sort(unique(vcat(candidates, default)))
end

function sweep_s2(stacked::SnpArray, m::Int, n::Int)
    println("\n===== S2: A*x row_step =====")
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(stacked; center = true, scale = true,
                                impute = true)
        packed = operator.s.data
        values = operator.values
        rows_filled = operator.s.m
        x = randn(Xoshiro(1), T, n)
        out = Vector{T}(undef, m)
        flops = Float64(m) * n
        default_step = SnpArrays._tile_sizes(T, rows_filled, n, 1, :forward;
                                             vector = true)[1]
        best_time = Inf
        best_steps = (0, 0, 0)
        baseline = Inf
        for row_step in grid_with_default(
                [256, 512, 1024, 2048, 4096, 8192], default_step)
            row_step % DECODE_WIDTH == 0 || continue
            steps = (row_step, n, 1)
            seconds = min_elapsed(ax_tile!, out, packed, x, values,
                                  rows_filled, row_step, n)
            report_line(@sprintf("%-8s A*x", T), steps, seconds, flops, 0)
            row_step == default_step && (baseline = seconds)
            seconds < best_time && ((best_time, best_steps) = (seconds, steps))
        end
        report_best(@sprintf("%-8s A*x", T), best_steps, best_time, baseline)
    end
    return nothing
end

function sweep_s3(stacked::SnpArray, m::Int, n::Int)
    println("\n===== S3: A*X k=32, column_step x row_step =====")
    k = 32
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(stacked; center = true, scale = true,
                                impute = true)
        packed = operator.s.data
        values = operator.values
        rows_filled = operator.s.m
        X = randn(Xoshiro(3), T, n, k)
        out = Matrix{T}(undef, m, k)
        flops = Float64(m) * n * k
        default = SnpArrays._tile_sizes(T, rows_filled, n, k, :forward;
                                        vector = false)
        best_time = Inf
        best_steps = (0, 0, 0)
        baseline = Inf
        for column_step in grid_with_default(
                    [128, 256, 512, 1024, 2048], default[2]),
            row_step in grid_with_default([256, 1024, 4096], default[1])

            row_step % DECODE_WIDTH == 0 || continue
            steps = (row_step, column_step, k)
            tasks = cld(rows_filled, row_step)
            bytes = panel_bytes(T, tasks, column_step)
            seconds = min_elapsed(AX_tile!, out, packed, X, values,
                                  rows_filled, row_step, column_step, k)
            report_line(@sprintf("%-8s A*X", T), steps, seconds, flops, bytes)
            steps == (default[1], default[2], k) && (baseline = seconds)
            seconds < best_time && ((best_time, best_steps) = (seconds, steps))
        end
        report_best(@sprintf("%-8s A*X", T), best_steps, best_time, baseline)
    end
    return nothing
end

function sweep_s4(stacked::SnpArray, m::Int, n::Int)
    println("\n===== S4: Aᵀ*x column_step x row_step =====")
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(stacked; center = true, scale = true,
                                impute = true)
        packed = operator.s.data
        values = operator.values
        rows_filled = operator.s.m
        y = randn(Xoshiro(2), T, m)
        out = Vector{T}(undef, n)
        flops = Float64(m) * n
        default = SnpArrays._tile_sizes(T, rows_filled, n, 1, :transpose;
                                        vector = true)
        best_time = Inf
        best_steps = (0, 0, 0)
        baseline = Inf
        for column_step in grid_with_default(
                    [512, 1136, 2048, 4096, 8192], default[2]),
            row_step in grid_with_default([1024, 4096, 16384], default[1])

            row_step % DECODE_WIDTH == 0 || continue
            steps = (row_step, column_step, 1)
            seconds = min_elapsed(atx_tile!, out, packed, y, values,
                                  rows_filled, row_step, column_step)
            report_line(@sprintf("%-8s Aᵀ*x", T), steps, seconds, flops, 0)
            steps == (default[1], default[2], 1) && (baseline = seconds)
            seconds < best_time && ((best_time, best_steps) = (seconds, steps))
        end
        report_best(@sprintf("%-8s Aᵀ*x", T), best_steps, best_time, baseline)
    end
    return nothing
end

function sweep_s5(stacked::SnpArray, m::Int, n::Int)
    println("\n===== S5: Aᵀ*X k=32, row_step x column_step =====")
    k = 32
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(stacked; center = true, scale = true,
                                impute = true)
        packed = operator.s.data
        values = operator.values
        rows_filled = operator.s.m
        Y = randn(Xoshiro(4), T, m, k)
        out = Matrix{T}(undef, n, k)
        flops = Float64(m) * n * k
        default = SnpArrays._tile_sizes(T, rows_filled, n, k, :transpose;
                                        vector = false)
        best_time = Inf
        best_steps = (0, 0, 0)
        baseline = Inf
        for row_step in grid_with_default(
                    [256, 1024, 2048, 4096, 8192], default[1]),
            column_step in grid_with_default([1136, 2048, 4096], default[2])

            row_step % DECODE_WIDTH == 0 || continue
            steps = (row_step, column_step, k)
            tasks = cld(n, column_step)
            bytes = panel_bytes(T, tasks, row_step)
            seconds = min_elapsed(AtX_tile!, out, packed, Y, values,
                                  rows_filled, row_step, column_step, k)
            report_line(@sprintf("%-8s Aᵀ*X", T), steps, seconds, flops, bytes)
            steps == (default[1], default[2], k) && (baseline = seconds)
            seconds < best_time && ((best_time, best_steps) = (seconds, steps))
        end
        report_best(@sprintf("%-8s Aᵀ*X", T), best_steps, best_time, baseline)
    end
    return nothing
end

function main()
    eur = SnpArray(SnpArrays.datadir("EUR_subset.bed"))
    m, n = size(eur)
    verify(eur)

    println(stderr, "stacking genotypes (R=", R, ")...")
    t_stack = @elapsed stacked = stack_snparray(eur, R)
    println(stderr, "stacking took ", round(t_stack; digits = 2), " s")

    rows = R * m
    print_header(R, rows, n)
    sweep_s1(stacked, rows, n)
    sweep_s2(stacked, rows, n)
    sweep_s3(stacked, rows, n)
    sweep_s4(stacked, rows, n)
    sweep_s5(stacked, rows, n)
end

main()
