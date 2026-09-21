using LinearAlgebra, Random, SnpArrays, SparseArrays, Test

"""
    write_bed(path, s::SnpArray)

Write `s` as a PLINK `.bed` file at `path` without memory-mapping it.
"""
function write_bed(path::AbstractString, s::SnpArray)
    open(path, "w") do io
        write(io, 0x1b6c)
        write(io, 0x01)
        write(io, s.data)
    end
    return path
end

# Standalone-safe: build fresh in-memory arrays rather than relying on the
# `EUR`/`mouse` constants that `runtests.jl` defines before this file.
eur_full = SnpArray(SnpArrays.datadir("EUR_subset.bed"))
mouse_full = SnpArray(SnpArrays.datadir("mouse.bed"))

@testset "SnpLinAlgStream chunk equivalence" begin
    datasets = (
        (path = SnpArrays.datadir("EUR_subset.bed"), s = eur_full,
         kwargs = (center = true, scale = true, impute = true)),
        (path = SnpArrays.datadir("mouse.bed"), s = mouse_full,
         kwargs = (impute = true, center = true, scale = true)),
    )
    for dataset in datasets
        s = dataset.s
        m, n = size(s)
        full_counts = counts(s; dims = 1)
        for T in (Float32, Float64)
            # Float32 restricted to a subset of widths to keep runtime
            # reasonable; Float64 exercises the full width set.
            widths = T == Float32 ? (1000, n + 1, 37) :
                     (1000, 4096, n, n + 1, 37)
            for width in widths
                sla = SnpLinAlg{T}(s; dataset.kwargs...)
                stream = SnpLinAlgStream{T}(dataset.path; width = width,
                                            dataset.kwargs...)
                @test size(stream) == (m, n)
                @test length(stream) == cld(n, width)
                tolerance = 32 * sqrt(T(max(m, n))) * eps(T)
                Q = randn(Xoshiro(1), T, m, 4)
                assembled = Matrix{T}(undef, n, 4)
                next_col = 1
                nchunks = 0
                for (cols, chunk) in stream
                    nchunks += 1
                    @test first(cols) == next_col
                    next_col = last(cols) + 1
                    @test size(chunk) == (m, length(cols))
                    @test chunk.values == view(sla.values, :, cols)
                    @test counts(chunk.s; dims = 1) ==
                          view(full_counts, :, cols)
                    @test chunk.μ == view(sla.μ, cols)
                    piece = Matrix{T}(undef, length(cols), 4)
                    mul!(piece, transpose(chunk), Q)
                    copyto!(view(assembled, cols, :), piece)
                end
                @test next_col == n + 1
                @test nchunks == length(stream)
                expected = transpose(sla) * Q
                @test isapprox(norm(assembled - expected) / norm(expected),
                              zero(T); atol = tolerance, rtol = tolerance)
            end
        end
    end
end

@testset "SnpLinAlgStream multi-file and compressed" begin
    m, n = size(eur_full)
    n1 = n ÷ 2
    n2 = n - n1
    tolerance = 32 * sqrt(Float64(max(m, n))) * eps(Float64)
    Q = randn(Xoshiro(1), Float64, m, 4)
    sla = SnpLinAlg{Float64}(eur_full; center = true, scale = true)
    expected = transpose(sla) * Q

    mktempdir(ENV["TMPDIR"]) do dir
        s1 = SnpArray(undef, m, n1)
        s1.data .= view(eur_full.data, :, 1:n1)
        s2 = SnpArray(undef, m, n2)
        s2.data .= view(eur_full.data, :, (n1 + 1):n)
        a = joinpath(dir, "a.bed")
        b = joinpath(dir, "b.bed")
        write_bed(a, s1)
        write_bed(b, s2)
        open(joinpath(dir, "a.bim"), "w") do io
            for i in 1:n1
                println(io, "1 snp$i 0 $i A C")
            end
        end
        open(joinpath(dir, "b.bim"), "w") do io
            for i in 1:n2
                println(io, "1 snp$(n1 + i) 0 $(n1 + i) A C")
            end
        end

        stream = SnpLinAlgStream{Float64}([a, b]; m = m, width = 4096,
                                          center = true, scale = true)
        @test size(stream) == (m, n)
        assembled = Matrix{Float64}(undef, n, 4)
        for (cols, chunk) in stream
            n1 in cols && @test last(cols) == n1
            piece = Matrix{Float64}(undef, length(cols), 4)
            mul!(piece, transpose(chunk), Q)
            copyto!(view(assembled, cols, :), piece)
        end
        @test isapprox(norm(assembled - expected) / norm(expected), 0.0;
                      atol = tolerance, rtol = tolerance)

        eur_prefix = joinpath(dir, "eur")
        for suffix in (".bed", ".bim", ".fam")
            cp(SnpArrays.datadir("EUR_subset" * suffix), eur_prefix * suffix)
        end
        compress_plink(eur_prefix, "gz")
        gz_stream = SnpLinAlgStream{Float64}(eur_prefix * ".bed.gz";
                                             width = 4096, center = true,
                                             scale = true)
        @test size(gz_stream) == (m, n)
        assembled_gz = Matrix{Float64}(undef, n, 4)
        for (cols, chunk) in gz_stream
            piece = Matrix{Float64}(undef, length(cols), 4)
            mul!(piece, transpose(chunk), Q)
            copyto!(view(assembled_gz, cols, :), piece)
        end
        @test isapprox(norm(assembled_gz - expected) / norm(expected), 0.0;
                      atol = tolerance, rtol = tolerance)
    end
end

@testset "streamed_grm_mul!" begin
    m, n = size(eur_full)
    reps = cld(2048, m)
    M = reps * m

    mktempdir(ENV["TMPDIR"]) do dir
        stacked = SnpArray(undef, M, n)
        for r in 1:reps
            stacked[((r - 1) * m + 1):(r * m), :] .= view(eur_full, :, :)
        end
        path = joinpath(dir, "stacked.bed")
        write_bed(path, stacked)
        open(joinpath(dir, "stacked.bim"), "w") do io
            for i in 1:n
                println(io, "1 snp$i 0 $i A C")
            end
        end

        for T in (Float32, Float64)
            sla = SnpLinAlg{T}(stacked; center = true, scale = true,
                               impute = true)
            stream = SnpLinAlgStream{T}(path; m = M, width = 4096,
                                        center = true, scale = true,
                                        impute = true)
            tolerance = 32 * sqrt(T(max(M, n))) * eps(T)
            for k in (1, 4, 32)
                Q = randn(Xoshiro(k), T, M, k)
                V = Matrix{T}(undef, M, k)
                U = Matrix{T}(undef, n, k)
                @test streamed_grm_mul!(V, stream, Q; U = U) === V
                Uref = transpose(sla) * Q
                Vref = (sla * Uref) ./ n
                @test isapprox(norm(U - Uref) / norm(Uref), zero(T);
                              atol = tolerance, rtol = tolerance)
                @test isapprox(norm(V - Vref) / norm(Vref), zero(T);
                              atol = tolerance, rtol = tolerance)

                scale_vec = rand(Xoshiro(5), T, n)
                V2 = Matrix{T}(undef, M, k)
                streamed_grm_mul!(V2, stream, Q; scale = scale_vec, U = U)
                Vref2 = sla * (Uref .* scale_vec)
                @test isapprox(norm(V2 - Vref2) / norm(Vref2), zero(T);
                              atol = tolerance, rtol = tolerance)

                @test_throws DimensionMismatch streamed_grm_mul!(
                    Matrix{T}(undef, M + 1, k), stream, Q)
                @test_throws DimensionMismatch streamed_grm_mul!(
                    V, stream, Matrix{T}(undef, M + 1, k))
                @test_throws DimensionMismatch streamed_grm_mul!(
                    V, stream, Q; U = Matrix{T}(undef, n + 1, k))
                @test_throws DimensionMismatch streamed_grm_mul!(
                    V, stream, Q; scale = Vector{T}(undef, n + 1))
            end
        end
    end
end

@testset "five-argument forward mul!" begin
    m, n = size(eur_full)
    for T in (Float32, Float64)
        sla = SnpLinAlg{T}(eur_full; center = true, scale = true,
                           impute = true)
        tolerance = 32 * sqrt(T(max(m, n))) * eps(T)

        rhs = randn(Xoshiro(2), T, n, 5)
        out = randn(Xoshiro(3), T, m, 5)
        expected = out + sla * rhs
        result = copy(out)
        @test mul!(result, sla, rhs, 1, 1) === result
        @test isapprox(norm(result - expected) / norm(expected), zero(T);
                      atol = tolerance, rtol = tolerance)

        result0 = copy(out)
        expected0 = sla * rhs
        @test mul!(result0, sla, rhs, 1, 0) === result0
        @test isapprox(norm(result0 - expected0) / norm(expected0), zero(T);
                      atol = tolerance, rtol = tolerance)

        rhs_vector = randn(Xoshiro(2), T, n)
        out_vector = randn(Xoshiro(3), T, m)
        expected_vector = out_vector + sla * rhs_vector
        result_vector = copy(out_vector)
        @test mul!(result_vector, sla, rhs_vector, 1, 1) === result_vector
        @test isapprox(norm(result_vector - expected_vector) /
                      norm(expected_vector), zero(T); atol = tolerance,
                      rtol = tolerance)

        result_vector0 = copy(out_vector)
        expected_vector0 = sla * rhs_vector
        @test mul!(result_vector0, sla, rhs_vector, 1, 0) === result_vector0
        @test isapprox(norm(result_vector0 - expected_vector0) /
                      norm(expected_vector0), zero(T); atol = tolerance,
                      rtol = tolerance)

        @test_throws ArgumentError mul!(result, sla, rhs, 2, 1)
        @test_throws ArgumentError mul!(result, sla, rhs, 1, 0.5)
        @test_throws ArgumentError mul!(result_vector, sla, rhs_vector, 2, 1)
        @test_throws ArgumentError mul!(result_vector, sla, rhs_vector, 1,
                                        0.5)
    end
end

@testset "prefetch and repeated sweeps" begin
    mouse_path = SnpArrays.datadir("mouse.bed")
    m, n = size(mouse_full)
    stream_no_prefetch = SnpLinAlgStream{Float64}(mouse_path; width = 1000,
                                                  prefetch = false,
                                                  center = true, scale = true,
                                                  impute = true)
    stream_prefetch = SnpLinAlgStream{Float64}(mouse_path; width = 1000,
                                               prefetch = true, center = true,
                                               scale = true, impute = true)
    Q = randn(Xoshiro(1), Float64, m, 3)

    values_no_prefetch = Matrix{Float64}[]
    for (cols, chunk) in stream_no_prefetch
        push!(values_no_prefetch, copy(chunk.values))
    end
    values_prefetch = Matrix{Float64}[]
    for (cols, chunk) in stream_prefetch
        push!(values_prefetch, copy(chunk.values))
    end
    @test length(values_no_prefetch) == length(values_prefetch)
    for (a, b) in zip(values_no_prefetch, values_prefetch)
        @test a == b
    end

    V_no = Matrix{Float64}(undef, m, 3)
    V_yes = Matrix{Float64}(undef, m, 3)
    streamed_grm_mul!(V_no, stream_no_prefetch, Q)
    streamed_grm_mul!(V_yes, stream_prefetch, Q)
    @test V_no == V_yes

    first_cols = UnitRange{Int}[]
    first_values = Matrix{Float64}[]
    for (cols, chunk) in stream_no_prefetch
        push!(first_cols, cols)
        push!(first_values, copy(chunk.values))
    end
    second_cols = UnitRange{Int}[]
    second_values = Matrix{Float64}[]
    for (cols, chunk) in stream_no_prefetch
        push!(second_cols, cols)
        push!(second_values, copy(chunk.values))
    end
    @test first_cols == second_cols
    for (a, b) in zip(first_values, second_values)
        @test a == b
    end
end

@testset "impute=false NaN propagation" begin
    mktempdir(ENV["TMPDIR"]) do dir
        packed = SnpArray(undef, 2048, 5)
        fill!(packed.data, 0x00)
        packed.data[1, 2] = 0x04
        path = joinpath(dir, "nan.bed")
        write_bed(path, packed)
        open(joinpath(dir, "nan.bim"), "w") do io
            for i in 1:5
                println(io, "1 snp$i 0 $i A C")
            end
        end
        for T in (Float32, Float64)
            stream = SnpLinAlgStream{T}(path; m = 2048, width = 5,
                                        impute = false)
            for (cols, chunk) in stream
                product = chunk * ones(T, 5, 4)
                @test all(isnan, product[2, :])
                @test all(iszero, product[[1; 3:2048], :])
            end
        end
    end
end
