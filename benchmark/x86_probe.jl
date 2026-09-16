# x86 probe: compare master vs register-tiled-kernels A*X / Aᵀ*Y performance
# and the lookup-table-vs-tiling decomposition, using only SnpArrays + stdlib.
using SnpArrays
using LinearAlgebra: mul!, transpose
using Random
using Printf
using Statistics: mean

"""
    min_elapsed_ns(f!, args...; warmup::Int = 1, runs::Int = 5) -> Int

Call `f!(args...)` `warmup` times, then `runs` times, timing each call with
`time_ns()`, and return the minimum elapsed nanoseconds over the timed runs.
"""
function min_elapsed_ns(f!::F, args...; warmup::Int = 1, runs::Int = 5) where F
    for _ in 1:warmup
        f!(args...)
    end
    best = typemax(Int)
    for _ in 1:runs
        start = time_ns()
        f!(args...)
        elapsed = Int(time_ns() - start)
        best = min(best, elapsed)
    end
    return best
end

"""
    print_machine_report()

Print Julia version, architecture, CPU identification, thread counts, the
detected ISA tier, and (on the branch checkout) the register-tiling
constants.
"""
function print_machine_report()
    println("== machine report ==")
    println("VERSION = ", VERSION)
    println("Sys.ARCH = ", Sys.ARCH)
    println("Sys.CPU_NAME = ", Sys.CPU_NAME)
    println("Sys.CPU_THREADS = ", Sys.CPU_THREADS)
    println("Threads.nthreads() = ", Threads.nthreads())

    isa_tier = if Sys.ARCH !== :x86_64
        "n/a (not x86_64)"
    else
        try
            isa_map = Base.BinaryPlatforms.arch_march_isa_mapping["x86_64"]
            cpu_isa = Base.BinaryPlatforms.CPUID.cpu_isa()
            matches = [(march, isa) for (march, isa) in isa_map if isa <= cpu_isa]
            isempty(matches) ? "unknown" :
                string(matches[argmax(last.(matches))][1])
        catch
            "unavailable"
        end
    end
    println("isa_tier = ", isa_tier)

    if isdefined(SnpArrays, :VECTOR_BYTES)
        println("SnpArrays.VECTOR_BYTES[] = ", SnpArrays.VECTOR_BYTES[])
        println("SnpArrays._micro_tile(Float64) = ",
                SnpArrays._micro_tile(Float64))
        println("SnpArrays._micro_tile(Float32) = ",
                SnpArrays._micro_tile(Float32))
    else
        println("SnpArrays.VECTOR_BYTES[] = n/a (not on branch)")
    end
end

"""
    print_mul_comparison(m, n, k)

Build a random `m × n` `SnpArray`, wrap it in a `SnpLinAlg{Float64}`, and
time `mul!(C, sla, X)` and `mul!(D, transpose(sla), Y)`, printing elapsed
milliseconds and checksums.
"""
function print_mul_comparison(m::Int, n::Int, k::Int)
    println("== end-to-end mul! comparison ==")
    s = SnpArray(undef, m, n)
    Random.rand!(Random.Xoshiro(1), s.data)
    sla = SnpLinAlg{Float64}(
        s; model=ADDITIVE_MODEL, center=false, scale=false, impute=true,
    )

    X = randn(Random.Xoshiro(2), Float64, n, k)
    C = Matrix{Float64}(undef, m, k)
    ns_forward = min_elapsed_ns(mul!, C, sla, X)
    @printf("A*X ms = %.6f\n", ns_forward / 1e6)
    @printf("sum(C) = %.17g\n", sum(C))

    Y = randn(Random.Xoshiro(3), Float64, m, k)
    D = Matrix{Float64}(undef, n, k)
    ns_transpose = min_elapsed_ns(mul!, D, transpose(sla), Y)
    @printf("Aᵀ*Y ms = %.6f\n", ns_transpose / 1e6)
    @printf("sum(D) = %.17g\n", sum(D))
end

"""
    fixed_lookup_values(mu) -> Matrix{Float64}

Return the `4 × n` centered-additive-mean-impute lookup table for column
means `mu`, with `σ⁻¹ = 1`: row 1 is the code-0 value, row 2 is the
imputed (missing, code-1) value, and rows 3-4 are the code-2, code-3
values.
"""
function fixed_lookup_values(mu::Vector{Float64})
    n = length(mu)
    values = Matrix{Float64}(undef, 4, n)
    for j in 1:n
        values[1, j] = -mu[j]
        values[2, j] = 0.0
        values[3, j] = 1.0 - mu[j]
        values[4, j] = 2.0 - mu[j]
    end
    return values
end

# Master's `@turbo` loop nest is only defined (and LoopVectorization only
# loaded) when the master-only symbol is present, so nothing on this branch
# ever references the package that is not a dependency there.
if isdefined(SnpArrays, :_snparray_AX_additive_meanimpute!)
    @eval using LoopVectorization

    @eval begin
        """
            turbo_lookup_ax!(out, packed, V, values, srows, scols, Vcols)

        Master's exact `@turbo` loop nest for `A*X`, selecting the genotype
        value from `values` by bit-select instead of the additive-mean-impute
        formula.
        """
        function turbo_lookup_ax!(
            out::Matrix{Float64},
            packed::Matrix{UInt8},
            V::Matrix{Float64},
            values::Matrix{Float64},
            srows::Int,
            scols::Int,
            Vcols::Int,
        )
            k = srows >> 2
            LoopVectorization.@turbo for c in 1:Vcols
                for j in 1:scols
                    for l in 1:k
                        for p in 1:4
                            block = packed[l, j]
                            Aij = (block >> (2 * (p - 1))) & 3
                            lo = (Aij & 1) == 0
                            hi = (Aij & 2) == 0
                            g = ifelse(
                                hi,
                                ifelse(lo, values[1, j], values[2, j]),
                                ifelse(lo, values[3, j], values[4, j]),
                            )
                            out[4 * (l - 1) + p, c] += g * V[j, c]
                        end
                    end
                end
            end
            return out
        end
    end
end

"""
    timed_meanimpute!(out, s, V, srows, scols, Vcols, mu, muimpute, sigmainv)

Zero `out`, then run master's `_snparray_AX_additive_meanimpute!` kernel;
zeroing inside the timed call keeps repeated timed runs from compounding
an `out += ...` accumulation into Inf/NaN.
"""
function timed_meanimpute!(
    out::Matrix{Float64},
    s::Matrix{UInt8},
    V::Matrix{Float64},
    srows::Int,
    scols::Int,
    Vcols::Int,
    mu::Vector{Float64},
    muimpute::Vector{Float64},
    sigmainv::Vector{Float64},
)
    fill!(out, 0.0)
    return SnpArrays._snparray_AX_additive_meanimpute!(
        out, s, V, srows, scols, Vcols, mu, muimpute, sigmainv,
    )
end

"""
    timed_turbo_lookup_ax!(out, packed, V, values, srows, scols, Vcols)

Zero `out`, then run `turbo_lookup_ax!`; zeroing inside the timed call
keeps repeated timed runs from compounding an `out += ...` accumulation
into Inf/NaN.
"""
function timed_turbo_lookup_ax!(
    out::Matrix{Float64},
    packed::Matrix{UInt8},
    V::Matrix{Float64},
    values::Matrix{Float64},
    srows::Int,
    scols::Int,
    Vcols::Int,
)
    fill!(out, 0.0)
    return turbo_lookup_ax!(out, packed, V, values, srows, scols, Vcols)
end

"""
    scalar_lookup_ax!(out, packed, V, values, m, n, k)

Plain scalar triple loop over the lookup table, used to verify the branch's
`_snparray_AX_kernel!` against a ground truth that does not depend on the
`@turbo`/SIMD paths under test.
"""
function scalar_lookup_ax!(
    out::Matrix{Float64},
    packed::Matrix{UInt8},
    V::Matrix{Float64},
    values::Matrix{Float64},
    m::Int,
    n::Int,
    k::Int,
)
    for c in 1:k
        for j in 1:n
            rhs_value = V[j, c]
            for row in 1:m
                code = SnpArrays._packed_code(packed, row, j)
                out[row, c] += values[code, j] * rhs_value
            end
        end
    end
    return out
end

"""
    timed_branch_kernel!(out, packed, V, values, row_first, row_last,
        column_step, rhs_first, rhs_last, width)

Zero `out`, then run the branch's `_snparray_AX_kernel!`; zeroing inside
the timed call keeps repeated timed runs from compounding an
`out += ...` accumulation into Inf/NaN.
"""
function timed_branch_kernel!(
    out::Matrix{Float64},
    packed::Matrix{UInt8},
    V::Matrix{Float64},
    values::Matrix{Float64},
    row_first::Int,
    row_last::Int,
    column_step::Int,
    rhs_first::Int,
    rhs_last::Int,
    width::Val,
)
    fill!(out, 0.0)
    return SnpArrays._snparray_AX_kernel!(
        out, packed, V, values, row_first, row_last, column_step, rhs_first,
        rhs_last, width,
    )
end

"""
    print_decomposition(m, n, k)

Time the lookup-table-only kernel and the register-tiled kernel on
identical data, on whichever branch applies, printing a clear skip note
for the section that does not.
"""
function print_decomposition(m::Int, n::Int, k::Int)
    println("== lookup-vs-tiling decomposition ==")
    println("decomposition shape: m = ", m, ", n = ", n, ", k = ", k)

    s = SnpArray(undef, m, n)
    Random.rand!(Random.Xoshiro(4), s.data)
    mu = Vector{Float64}(
        dropdims(mean(s; dims=1, model=ADDITIVE_MODEL); dims=1),
    )
    values = fixed_lookup_values(mu)
    packed = s.data
    V = randn(Random.Xoshiro(5), Float64, n, k)

    if isdefined(SnpArrays, :_snparray_AX_additive_meanimpute!)
        sigmainv = ones(n)
        muimpute = mu
        # `srows` is the sample count `m`, not `size(packed, 1)` (the packed
        # byte-row count `m ÷ 4`); master's kernel takes the packed byte
        # matrix `s.data`, not the `SnpArray` `s`.
        srows, scols = m, n

        out_meanimpute = zeros(Float64, m, k)
        out_turbo = zeros(Float64, m, k)
        timed_meanimpute!(
            out_meanimpute, packed, V, srows, scols, k, mu, muimpute,
            sigmainv,
        )
        timed_turbo_lookup_ax!(out_turbo, packed, V, values, srows, scols, k)
        max_abs_diff = maximum(abs.(out_meanimpute .- out_turbo))
        @printf("master kernels agree, max abs diff = %.3e\n", max_abs_diff)
        max_abs_diff < 1e-8 || throw(ErrorException(
            "master lookup-table kernel disagrees with meanimpute kernel",
        ))

        ns_meanimpute = min_elapsed_ns(
            timed_meanimpute!, out_meanimpute, packed, V, srows, scols, k,
            mu, muimpute, sigmainv,
        )
        @printf("meanimpute_kernel ms = %.6f\n", ns_meanimpute / 1e6)

        ns_turbo = min_elapsed_ns(
            timed_turbo_lookup_ax!, out_turbo, packed, V, values, srows,
            scols, k,
        )
        @printf("lookup_turbo_kernel ms = %.6f\n", ns_turbo / 1e6)
    else
        println("master-only kernels skipped: not on master checkout")
    end

    if isdefined(SnpArrays, :_snparray_AX_kernel!) &&
       isdefined(SnpArrays, :_vector_width)
        W = SnpArrays._vector_width(Float64)
        out_branch = zeros(Float64, m, k)
        out_scalar = zeros(Float64, m, k)
        timed_branch_kernel!(
            out_branch, packed, V, values, 1, m, n, 1, k, Val(W),
        )
        scalar_lookup_ax!(out_scalar, packed, V, values, m, n, k)
        max_abs_diff = maximum(abs.(out_branch .- out_scalar))
        @printf("branch kernel verification max abs diff = %.3e\n",
                max_abs_diff)

        ns_branch = min_elapsed_ns(
            timed_branch_kernel!, out_branch, packed, V, values, 1, m, n, 1,
            k, Val(W),
        )
        @printf("tiled_kernel ms = %.6f\n", ns_branch / 1e6)
    else
        println("branch-only kernel skipped: not on register-tiled-kernels " *
                "checkout")
    end
end

function main()
    m = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8192
    n = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 4096
    k = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64

    version = if isdefined(SnpArrays, :VECTOR_BYTES)
        "register-tiled-kernels"
    elseif isdefined(SnpArrays, :_snparray_AX_additive_meanimpute!)
        "master"
    else
        "unknown"
    end
    println("detected version = ", version)

    print_machine_report()
    print_mul_comparison(m, n, k)
    print_decomposition(m ÷ 2, n ÷ 2, k)
end

main()
