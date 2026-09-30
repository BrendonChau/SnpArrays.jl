# Capture golden CuSnpArray outputs (impute = false) from the byte-per-sample
# CUDA kernels that predate the packed-word rewrite. Run once on a GPU:
#
#   julia --project=<env with SnpArrays, CUDA, Adapt> capture.jl
#
# Files written next to this script:
#   codes.txt  m x n PLINK 2-bit codes of the fixture (0x00..0x03)
#   x.txt      length-n right-hand side for A*x (Float32 values)
#   y.txt      length-m right-hand side for transpose(A)*y (Float32 values)
#   combos.txt one line per output column: model center scale T
#   ax.txt     m x 24 outputs of A*x, as Float64
#   atx.txt    n x 24 outputs of transpose(A)*y, as Float64

using Adapt
using CUDA
using DelimitedFiles
using Random
using SnpArrays

"""
    golden_fixture() -> (codes, x, y)

Deterministic 37 x 23 code matrix with missing codes (0x01) in several
columns, a missing-free column 1 and a monomorphic column 2.
"""
function golden_fixture()
    rng = Random.Xoshiro(20260929)
    m, n = 37, 23
    codes = rand(rng, UInt8[0x00, 0x02, 0x03], m, n)
    for column in 3:2:n
        rows = rand(rng, 1:m, 1 + mod(column, 4))
        codes[rows, column] .= 0x01
    end
    codes[:, 1] .= rand(rng, UInt8[0x00, 0x02, 0x03], m)
    codes[:, 2] .= 0x03
    x = randn(rng, Float32, n)
    y = randn(rng, Float32, m)
    return codes, x, y
end

"""
    fixture_snparray(codes::AbstractMatrix{UInt8}) -> SnpArray

In-memory `SnpArray` holding `codes`.
"""
function fixture_snparray(codes::AbstractMatrix{UInt8})
    s = SnpArray(undef, size(codes)...)
    for j in axes(codes, 2), i in axes(codes, 1)
        s[i, j] = codes[i, j]
    end
    return s
end

function main()
    codes, x, y = golden_fixture()
    s = fixture_snparray(codes)
    combos = String[]
    ax = Vector{Vector{Float64}}()
    atx = Vector{Vector{Float64}}()
    for model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL),
            center in (false, true), scale in (false, true),
            T in (Float32, Float64)
        A = CuSnpArray{T}(s; model, center, scale, impute=false)
        push!(ax, Float64.(Array(A * CuArray(T.(x)))))
        push!(atx, Float64.(Array(transpose(A) * CuArray(T.(y)))))
        push!(combos, "$(typeof(model).parameters[1]) $center $scale $T")
    end
    dir = @__DIR__
    writedlm(joinpath(dir, "codes.txt"), Int.(codes))
    writedlm(joinpath(dir, "x.txt"), Float64.(x))
    writedlm(joinpath(dir, "y.txt"), Float64.(y))
    writedlm(joinpath(dir, "combos.txt"), combos)
    writedlm(joinpath(dir, "ax.txt"), reduce(hcat, ax))
    writedlm(joinpath(dir, "atx.txt"), reduce(hcat, atx))
    return nothing
end

main()
