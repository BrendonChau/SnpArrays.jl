using Metal
using DelimitedFiles

"""
    golden_snparray(codes::AbstractMatrix{UInt8}) -> SnpArray

In-memory `SnpArray` holding the golden fixture codes.
"""
function golden_snparray(codes::AbstractMatrix{UInt8})
    s = SnpArray(undef, size(codes)...)
    for j in axes(codes, 2), i in axes(codes, 1)
        s[i, j] = codes[i, j]
    end
    return s
end

relerr(a, b) = norm(a - b) / max(norm(b), eps())

@testset "MtlSnpArray golden outputs (impute=false)" begin
    dir = joinpath(@__DIR__, "data", "cuda_golden")
    s = golden_snparray(UInt8.(readdlm(joinpath(dir, "codes.txt"), Int)))
    x = vec(readdlm(joinpath(dir, "x.txt"), Float64))
    y = vec(readdlm(joinpath(dir, "y.txt"), Float64))
    combos = readdlm(joinpath(dir, "combos.txt"), ' ', String)
    ax = readdlm(joinpath(dir, "ax.txt"), Float64)
    atx = readdlm(joinpath(dir, "atx.txt"), Float64)
    models = Dict("1" => ADDITIVE_MODEL, "2" => DOMINANT_MODEL,
        "3" => RECESSIVE_MODEL)
    @test size(combos, 1) == 24
    for row in axes(combos, 1)
        combos[row, 4] == "Float32" || continue
        model = models[combos[row, 1]]
        center = parse(Bool, combos[row, 2])
        scale = parse(Bool, combos[row, 3])
        T = Float32
        rtol = 1e-5
        A = MtlSnpArray{T}(s; model, center, scale)
        @test A isa MtlSnpArray{T}
        @test size(A) == size(s)
        @test eltype(A) == T
        out = MtlVector{T}(undef, size(s, 1))
        mul!(out, A, MtlArray(T.(x)))
        @test relerr(Float64.(Array(out)), ax[:, row]) < rtol
        outt = MtlVector{T}(undef, size(s, 2))
        mul!(outt, adjoint(A), MtlArray(T.(y)))
        @test relerr(Float64.(Array(outt)), atx[:, row]) < rtol
        @test relerr(Float64.(Array(transpose(A) * MtlArray(T.(y)))),
            atx[:, row]) < rtol
    end
end

@testset "MtlSnpArray mean imputation" begin
    dir = joinpath(@__DIR__, "data", "cuda_golden")
    fixture = golden_snparray(
        UInt8.(readdlm(joinpath(dir, "codes.txt"), Int)))
    rng = Random.Xoshiro(7)
    for s in (fixture, EUR), T in (Float32,),
            model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL),
            center in (false, true), scale in (false, true)
        rtol = 1e-5
        dense = convert(Matrix{T}, s; model, center, scale, impute=true)
        A = MtlSnpArray{T}(s; model, center, scale, impute=true)
        x = randn(rng, T, size(s, 2))
        y = randn(rng, T, size(s, 1))
        @test relerr(Array(A * MtlArray(x)), dense * x) < rtol
        @test relerr(Array(transpose(A) * MtlArray(y)),
            transpose(dense) * y) < rtol
    end
end

@testset "MtlSnpArray dimension checks" begin
    A = MtlSnpArray{Float32}(EUR)
    @test_throws ArgumentError MtlSnpArray{Float64}(EUR)
    m, n = size(EUR)
    @test_throws DimensionMismatch mul!(MtlVector{Float32}(undef, m + 1), A,
        Metal.zeros(Float32, n))
    @test_throws DimensionMismatch mul!(MtlVector{Float32}(undef, m), A,
        Metal.zeros(Float32, n - 1))
    @test_throws DimensionMismatch mul!(MtlVector{Float32}(undef, n),
        transpose(A), Metal.zeros(Float32, m + 1))
end

"""
    missing_as_zero(s, codes, T, model, center, scale) -> Matrix{T}

Dense `impute=false` matrix: a missing genotype counts as 0 before
centering by the column mean and scaling by the inverse standard deviation.
"""
function missing_as_zero(
    s::SnpArray, codes::AbstractMatrix{UInt8}, ::Type{T}, model,
    center::Bool, scale::Bool,
) where {T}
    μ = T.(vec(mean(s; dims=1, model)))
    σ = model == ADDITIVE_MODEL ? sqrt.(μ .* (1 .- μ ./ 2)) :
        sqrt.(μ .* (1 .- μ))
    σinv = ifelse.(σ .> 0, inv.(σ), one(T))
    raw = convert(Matrix{T}, s; model, impute=true)
    raw[codes .== 0x01] .= zero(T)
    shift = center ? transpose(μ) : zeros(T, 1, size(s, 2))
    factor = scale ? transpose(σinv) : ones(T, 1, size(s, 2))
    return (raw .- shift) .* factor
end

@testset "MtlSnpArray matrix multiplication" begin
    rng = Random.Xoshiro(11)
    # m = 1000 is not a multiple of 16; n = 777 is not a multiple of any
    # tile width.
    codes = rand(rng, UInt8(0):UInt8(3), 1000, 777)
    s = golden_snparray(codes)
    m, n = size(s)
    for T in (Float32,),
            model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL),
            center in (false, true), scale in (false, true),
            impute in (false, true)
        rtol = 1e-5
        dense = impute ?
            convert(Matrix{T}, s; model, center, scale, impute=true) :
            missing_as_zero(s, codes, T, model, center, scale)
        A = MtlSnpArray{T}(s; model, center, scale, impute)
        for k in (1, 3, 8, 33, 128)
            X = randn(rng, T, n, k)
            Y = randn(rng, T, m, k)
            out = MtlMatrix{T}(undef, m, k)
            mul!(out, A, MtlArray(X))
            @test relerr(Array(out), dense * X) < rtol
            @test relerr(Array(A * MtlArray(X)), dense * X) < rtol
            outt = MtlMatrix{T}(undef, n, k)
            mul!(outt, transpose(A), MtlArray(Y))
            @test relerr(Array(outt), transpose(dense) * Y) < rtol
            @test relerr(Array(adjoint(A) * MtlArray(Y)),
                transpose(dense) * Y) < rtol
        end
    end
end

@testset "MtlSnpArray column block" begin
    s = mouse
    dense = convert(Matrix{Float32}, s; center=true, scale=true,
        impute=true)
    A = MtlSnpArray{Float32}(s; center=true, scale=true, impute=true)
    Y = randn(Random.Xoshiro(3), Float32, size(s, 1), 8)
    for cols in (1:1, 1:100, 37:1100, (size(s, 2) - 5):size(s, 2))
        out = MtlMatrix{Float32}(undef, length(cols), 8)
        mul!(out, transpose(A), MtlArray(Y), cols)
        @test relerr(Array(out), transpose(dense[:, cols]) * Y) < 1e-5
    end
    @test_throws DimensionMismatch mul!(MtlMatrix{Float32}(undef, 3, 8),
        transpose(A), MtlArray(Y), 1:4)
    @test_throws ArgumentError mul!(MtlMatrix{Float32}(undef, 2, 8),
        transpose(A), MtlArray(Y), 0:1)
end

@testset "MtlSnpArray matrix dimension checks" begin
    A = MtlSnpArray{Float32}(EUR)
    m, n = size(EUR)
    @test_throws DimensionMismatch mul!(MtlMatrix{Float32}(undef, m, 4), A,
        Metal.zeros(Float32, n + 1, 4))
    @test_throws DimensionMismatch mul!(MtlMatrix{Float32}(undef, m, 3), A,
        Metal.zeros(Float32, n, 4))
    @test_throws DimensionMismatch mul!(MtlMatrix{Float32}(undef, n, 4),
        transpose(A), Metal.zeros(Float32, m - 1, 4))
    @test_throws DimensionMismatch mul!(MtlMatrix{Float32}(undef, n + 1, 4),
        transpose(A), Metal.zeros(Float32, m, 4))
end

@testset "MtlSnpArray tiled and simdgroup kernels agree" begin
    ext = Base.get_extension(SnpArrays, :SnpArraysMetalExt)
    rng = Random.Xoshiro(13)
    # m not a multiple of 16, n not a multiple of 4, and n large enough
    # for several SNP splits.
    for (m, n) in ((37, 5), (1001, 2049), (4099, 3001)), T in (Float32,)
        codes = rand(rng, UInt8(0):UInt8(3), m, n)
        s = golden_snparray(codes)
        rtol = 1e-5
        dense = missing_as_zero(s, codes, T, ADDITIVE_MODEL, true, true)
        A = MtlSnpArray{T}(s; center=true, scale=true)
        for k in (1, 2, 8, 9, 40)
            X = randn(rng, T, n, k)
            out = MtlMatrix{T}(undef, m, k)
            @test relerr(Array(mul!(out, A, MtlArray(X))), dense * X) < rtol
            ext._tiled_mul!(out, A, MtlArray(X))
            @test relerr(Array(out), dense * X) < rtol
            ext._simd_mul!(out, A, MtlArray(X))
            @test relerr(Array(out), dense * X) < rtol
            Y = randn(rng, T, m, k)
            outt = MtlMatrix{T}(undef, n, k)
            ext._tiled_t_mul!(outt, A, MtlArray(Y))
            @test relerr(Array(outt), transpose(dense) * Y) < rtol
            ext._simd_t_mul!(outt, A, MtlArray(Y))
            @test relerr(Array(outt), transpose(dense) * Y) < rtol
        end
    end
end

@testset "streamed_mul! on MtlMatrix" begin
    rng = Random.Xoshiro(17)
    dir = mktempdir()
    # m = 64 fills whole 16-sample words, m = 379 does not; two files test
    # chunks that stop at file boundaries.
    beds = map(enumerate(((64, 300), (64, 211)))) do (index, (m, n))
        path = joinpath(dir, "stream_$index.bed")
        s = SnpArray(path, m, n)
        s .= rand(rng, (0x00, 0x01, 0x02, 0x03), m, n)
        write(replace(path, ".bed" => ".fam"), repeat("f s 0 0 1 -9\n", m))
        write(replace(path, ".bed" => ".bim"), repeat("1 r 0 1 A G\n", n))
        path
    end
    cases = ((beds, 64), (SnpArrays.datadir("EUR_subset.bed"), 379))
    for (files, m) in cases
        stream = SnpLinAlgStream{Float32}(files; width=128, center=true,
            scale=true)
        chunks = [copy(chunk.s.data) for (cols, chunk) in stream]
        s = SnpArray(undef, m, size(stream, 2))
        s.data .= reduce(hcat, chunks)
        sla = SnpLinAlg{Float32}(s; center=true, scale=true)
        rtol = 1e-5
        for k in (1, 8, 40)
            X = randn(rng, Float32, size(stream, 2), k)
            Y = randn(rng, Float32, m, k)
            out = MtlMatrix{Float32}(undef, m, k)
            @test relerr(Array(streamed_mul!(out, stream, MtlArray(X))),
                sla * X) < rtol
            out = MtlMatrix{Float32}(undef, size(stream, 2), k)
            @test relerr(Array(streamed_mul!(out, stream, MtlArray(Y);
                transpose=true)), transpose(sla) * Y) < rtol
        end
    end
    stream = SnpLinAlgStream{Float32}(beds; width=128)
    @test_throws DimensionMismatch streamed_mul!(Metal.zeros(Float32, 64, 2),
        stream, Metal.zeros(Float32, 510, 2))
    @test_throws DimensionMismatch streamed_mul!(Metal.zeros(Float32, 510, 2),
        stream, Metal.zeros(Float32, 64, 2); transpose=true)
    @test_throws ArgumentError streamed_mul!(Metal.zeros(Float32, 64, 2),
        SnpLinAlgStream{Float64}(beds; width=128),
        Metal.zeros(Float32, 511, 2))
end

@testset "streamed_grm_mul! on MtlMatrix" begin
    rng = Random.Xoshiro(19)
    bed = SnpArrays.datadir("EUR_subset.bed")
    m, n = size(EUR)
    stream = SnpLinAlgStream{Float32}(bed; width=500, center=true,
        scale=true)
    rtol = 1e-5
    for k in (1, 8, 40)
        Q = randn(rng, Float32, m, k)
        scale = rand(rng, Float32, n)
        V = zeros(Float32, m, k)
        U = zeros(Float32, n, k)
        streamed_grm_mul!(V, U, stream, Q; scale)
        Vd = MtlMatrix{Float32}(undef, m, k)
        Ud = MtlMatrix{Float32}(undef, n, k)
        streamed_grm_mul!(Vd, Ud, stream, MtlArray(Q); scale)
        @test relerr(Array(Vd), V) < rtol
        @test relerr(Array(Ud), U) < rtol
        streamed_grm_mul!(V, stream, Q)
        @test relerr(Array(streamed_grm_mul!(Vd, stream, MtlArray(Q))),
            V) < rtol
    end
    @test_throws ArgumentError streamed_grm_mul!(
        Metal.zeros(Float32, m, 2), SnpLinAlgStream{Float64}(bed),
        Metal.zeros(Float32, m, 2))
end
