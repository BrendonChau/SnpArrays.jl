function simd_test_fixture(rows::Int, columns::Int)
    dense = Matrix{UInt8}(undef, rows, columns)
    packed = SnpArray(undef, rows, columns)
    fill!(packed.data, 0x55)
    for column in 1:columns, row in 1:rows
        code = UInt8(mod(row + 2column, 4))
        if rows == 1 && code == 0x01
            code = 0x03
        end
        dense[row, column] = code
        byte_index = ((row - 1) >>> 2) + 1
        shift = 2((row - 1) & 3)
        mask = ~(UInt8(3) << shift)
        packed.data[byte_index, column] &= mask
        packed.data[byte_index, column] |= code << shift
    end
    return packed, dense
end

function simd_test_reference(
    dense::AbstractMatrix{UInt8},
    ::Type{T},
    model::Union{Val{1}, Val{2}, Val{3}},
    center::Bool,
    scale::Bool,
    impute::Bool,
) where T <: AbstractFloat
    code0 = zero(T)
    code2 = model == RECESSIVE_MODEL ? zero(T) : one(T)
    code3 = model == ADDITIVE_MODEL ? T(2) : one(T)
    result = Matrix{T}(undef, size(dense))
    for column in axes(dense, 2)
        total = zero(T)
        observed = 0
        for row in axes(dense, 1)
            code = dense[row, column]
            if code != 0x01
                total += code == 0x00 ? code0 : code == 0x02 ? code2 : code3
                observed += 1
            end
        end
        column_mean = total / T(observed)
        variance = model == ADDITIVE_MODEL ?
                   column_mean * (one(T) - column_mean / T(2)) :
                   column_mean * (one(T) - column_mean)
        standard_deviation = sqrt(variance)
        multiplier = scale && standard_deviation > zero(T) ?
                     inv(standard_deviation) : one(T)
        offset = center ? column_mean : zero(T)
        for row in axes(dense, 1)
            code = dense[row, column]
            value = code == 0x00 ? code0 :
                    code == 0x01 ? (impute ? column_mean : T(NaN)) :
                    code == 0x02 ? code2 : code3
            result[row, column] = (value - offset) * multiplier
        end
    end
    return result
end

function simd_test_rhs(::Type{T}, rows::Int, columns::Int) where T
    result = Matrix{T}(undef, rows, columns)
    for column in 1:columns, row in 1:rows
        result[row, column] = T(mod(3row + 5column, 11) - 5) / T(4)
    end
    return result
end

function test_simd_products(
    operator::SnpLinAlg{T},
    reference::Matrix{T},
    width::Int;
    tolerance::T = T(64) * eps(T),
) where T <: AbstractFloat
    rows, columns = size(reference)
    forward_vector = vec(simd_test_rhs(T, columns, 1))
    transpose_vector = vec(simd_test_rhs(T, rows, 1))
    forward_matrix = simd_test_rhs(T, columns, width)
    transpose_matrix = simd_test_rhs(T, rows, width)

    expected = reference * forward_vector
    @test isapprox(operator * forward_vector, expected; nans=true,
                   atol=tolerance, rtol=tolerance)
    destination = fill(T(19), rows)
    @test mul!(destination, operator, forward_vector) === destination
    @test isapprox(destination, expected; nans=true, atol=tolerance,
                   rtol=tolerance)

    expected = reference * forward_matrix
    @test isapprox(operator * forward_matrix, expected; nans=true,
                   atol=tolerance, rtol=tolerance)
    matrix_destination = fill(T(19), rows, width)
    @test mul!(matrix_destination, operator, forward_matrix) ===
          matrix_destination
    @test isapprox(matrix_destination, expected; nans=true,
                   atol=tolerance, rtol=tolerance)

    for transposed in (transpose(operator), adjoint(operator))
        expected = transpose(reference) * transpose_vector
        @test isapprox(transposed * transpose_vector, expected; nans=true,
                       atol=tolerance, rtol=tolerance)
        destination = fill(T(19), columns)
        @test mul!(destination, transposed, transpose_vector) === destination
        @test isapprox(destination, expected; nans=true, atol=tolerance,
                       rtol=tolerance)

        expected = transpose(reference) * transpose_matrix
        @test isapprox(transposed * transpose_matrix, expected; nans=true,
                       atol=tolerance, rtol=tolerance)
        matrix_destination = fill(T(19), columns, width)
        @test mul!(matrix_destination, transposed, transpose_matrix) ===
              matrix_destination
        @test isapprox(matrix_destination, expected; nans=true,
                       atol=tolerance, rtol=tolerance)
    end
    return nothing
end

@testset "SnpLinAlg SIMD transformations and products" begin
    packed, dense = simd_test_fixture(17, 9)
    for T in (Float32, Float64),
        model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL),
        impute in (false, true), center in (false, true), scale in (false, true)
        operator = SnpLinAlg{T}(packed; model=model, impute=impute,
                                center=center, scale=scale)
        reference = simd_test_reference(dense, T, model, center, scale, impute)
        tolerance = T(64) * eps(T)
        @test isapprox(Matrix(operator), reference; nans=true,
                       atol=tolerance, rtol=tolerance)
        materialized = fill(T(19), size(reference))
        @test copyto!(materialized, operator) === materialized
        @test isapprox(materialized, reference; nans=true,
                       atol=tolerance, rtol=tolerance)
        for column in axes(reference, 2), row in axes(reference, 1)
            @test isequal(operator[row, column], reference[row, column])
        end
        test_simd_products(operator, reference, 8; tolerance=tolerance)
    end

    for width in (1, 8, 32, 256)
        operator = SnpLinAlg{Float64}(packed; model=ADDITIVE_MODEL,
                                      impute=true, center=true, scale=true)
        reference = simd_test_reference(dense, Float64, ADDITIVE_MODEL,
                                        true, true, true)
        test_simd_products(operator, reference, width)
    end
end

@testset "SnpLinAlg SIMD row and padding edges" begin
    for rows in (1:5..., 15:17...)
        packed, dense = simd_test_fixture(rows, 7)
        for impute in (false, true)
            operator = SnpLinAlg{Float32}(packed; model=DOMINANT_MODEL,
                                          impute=impute, center=true,
                                          scale=true)
            reference = simd_test_reference(dense, Float32, DOMINANT_MODEL,
                                            true, true, impute)
            @test isapprox(Matrix(operator), reference; nans=true,
                           atol=64eps(Float32), rtol=64eps(Float32))
            test_simd_products(operator, reference, 1)
        end
    end
end

@testset "SnpLinAlg scalar and strided fallbacks" begin
    packed, dense = simd_test_fixture(17, 9)
    operator = SnpLinAlg{Float16}(packed; model=RECESSIVE_MODEL, impute=true,
                                  center=true, scale=true)
    reference = simd_test_reference(dense, Float16, RECESSIVE_MODEL,
                                    true, true, true)
    test_simd_products(operator, reference, 8; tolerance=128eps(Float16))

    operator32 = SnpLinAlg{Float32}(packed; model=ADDITIVE_MODEL, impute=true,
                                    center=true, scale=true)
    reference32 = simd_test_reference(dense, Float32, ADDITIVE_MODEL,
                                      true, true, true)
    forward_vector_parent = zeros(Float32, 2size(operator32, 2))
    forward_vector = view(forward_vector_parent, 1:2:length(forward_vector_parent))
    forward_vector .= vec(simd_test_rhs(Float32, size(operator32, 2), 1))
    @test isapprox(operator32 * forward_vector, reference32 * forward_vector;
                   nans=true, atol=64eps(Float32), rtol=64eps(Float32))
    destination_parent = fill(19.0f0, 2size(operator32, 1))
    destination = view(destination_parent, 1:2:length(destination_parent))
    @test mul!(destination, operator32, forward_vector) === destination
    @test isapprox(destination, reference32 * forward_vector; nans=true,
                   atol=64eps(Float32), rtol=64eps(Float32))

    forward_matrix_parent = zeros(Float32, 2size(operator32, 2), 16)
    forward_matrix = view(forward_matrix_parent,
                          1:2:size(forward_matrix_parent, 1), 1:2:16)
    forward_matrix .= simd_test_rhs(Float32, size(operator32, 2), 8)
    @test isapprox(operator32 * forward_matrix, reference32 * forward_matrix;
                   nans=true, atol=64eps(Float32), rtol=64eps(Float32))
    matrix_destination_parent = fill(19.0f0, 2size(operator32, 1), 16)
    matrix_destination = view(matrix_destination_parent,
                              1:2:size(matrix_destination_parent, 1), 1:2:16)
    @test mul!(matrix_destination, operator32, forward_matrix) ===
          matrix_destination
    @test isapprox(matrix_destination, reference32 * forward_matrix; nans=true,
                   atol=64eps(Float32), rtol=64eps(Float32))

    transpose_vector_parent = zeros(Float32, 2size(operator32, 1))
    transpose_vector = view(transpose_vector_parent,
                            1:2:length(transpose_vector_parent))
    transpose_vector .= vec(simd_test_rhs(Float32, size(operator32, 1), 1))
    @test isapprox(adjoint(operator32) * transpose_vector,
                   transpose(reference32) * transpose_vector; nans=true,
                   atol=64eps(Float32), rtol=64eps(Float32))
    transpose_destination_parent = fill(19.0f0, 2size(operator32, 2))
    transpose_destination = view(transpose_destination_parent,
                                 1:2:length(transpose_destination_parent))
    @test mul!(transpose_destination, transpose(operator32),
               transpose_vector) === transpose_destination
    @test isapprox(transpose_destination,
                   transpose(reference32) * transpose_vector; nans=true,
                   atol=64eps(Float32), rtol=64eps(Float32))

    transpose_matrix_parent = zeros(Float32, 2size(operator32, 1), 16)
    transpose_matrix = view(transpose_matrix_parent,
                            1:2:size(transpose_matrix_parent, 1), 1:2:16)
    transpose_matrix .= simd_test_rhs(Float32, size(operator32, 1), 8)
    @test isapprox(transpose(operator32) * transpose_matrix,
                   transpose(reference32) * transpose_matrix; nans=true,
                   atol=64eps(Float32), rtol=64eps(Float32))
    transpose_matrix_destination_parent =
        fill(19.0f0, 2size(operator32, 2), 16)
    transpose_matrix_destination = view(
        transpose_matrix_destination_parent,
        1:2:size(transpose_matrix_destination_parent, 1), 1:2:16,
    )
    @test mul!(transpose_matrix_destination, adjoint(operator32),
               transpose_matrix) === transpose_matrix_destination
    @test isapprox(transpose_matrix_destination,
                   transpose(reference32) * transpose_matrix; nans=true,
                   atol=64eps(Float32), rtol=64eps(Float32))
end

@testset "SnpLinAlg IEEE nonfinite propagation" begin
    packed = SnpArray(undef, 4, 2)
    packed.data[:, :] .= reshape(UInt8[0xe1, 0x00], 1, 2)
    operator = SnpLinAlg{Float64}(packed; impute=false)
    forward_vector = [0.0, 1.0]
    @test isnan((operator * forward_vector)[1])
    @test isnan((operator * reshape(forward_vector, 2, 1))[1, 1])
    transpose_vector = [0.0, 1.0, 1.0, 1.0]
    @test isnan((transpose(operator) * transpose_vector)[1])
    @test isnan((adjoint(operator) * reshape(transpose_vector, 4, 1))[1, 1])

    observed, dense = simd_test_fixture(17, 5)
    observed.data .= observed.data .& 0xaa
    dense .= UInt8.(dense .& 0x02)
    finite_operator = SnpLinAlg{Float64}(observed; impute=true)
    finite_reference = simd_test_reference(dense, Float64, ADDITIVE_MODEL,
                                           false, false, true)
    forward = simd_test_rhs(Float64, 5, 2)
    forward[1, 1] = Inf
    forward[2, 2] = NaN
    @test isapprox(finite_operator * forward, finite_reference * forward;
                   nans=true)
    backward = simd_test_rhs(Float64, 17, 2)
    backward[1, 1] = Inf
    backward[2, 2] = NaN
    @test isapprox(transpose(finite_operator) * backward,
                   transpose(finite_reference) * backward; nans=true)
end

function filled_genotype_fixture(rows::Int, columns::Int, code::UInt8)
    fixture = SnpArray(undef, rows, columns)
    fill!(fixture.data, 0x55)
    for column in 1:columns, row in 1:rows
        byte_index = ((row - 1) >>> 2) + 1
        shift = 2((row - 1) & 3)
        fixture.data[byte_index, column] &= ~(UInt8(3) << shift)
        fixture.data[byte_index, column] |= code << shift
    end
    return fixture
end

@testset "SnpLinAlg scheduler tile boundaries" begin
    forward_vector_fixture = filled_genotype_fixture(4097, 1025, 0x03)
    forward_vector_operator = SnpLinAlg{Float32}(forward_vector_fixture)
    @test forward_vector_operator * ones(Float32, 1025) ==
          fill(Float32(2050), 4097)

    forward_matrix_fixture = filled_genotype_fixture(1025, 257, 0x03)
    forward_matrix_operator = SnpLinAlg{Float32}(forward_matrix_fixture)
    @test forward_matrix_operator * ones(Float32, 257, 256) ==
          fill(Float32(514), 1025, 256)

    transpose_fixture = filled_genotype_fixture(8193, 2049, 0x03)
    transpose_operator = SnpLinAlg{Float32}(transpose_fixture)
    expected = fill(Float32(16386), 2049)
    @test transpose(transpose_operator) * ones(Float32, 8193) == expected
    @test vec(adjoint(transpose_operator) * ones(Float32, 8193, 1)) == expected

    rhs_fixture = filled_genotype_fixture(17, 9, 0x03)
    rhs_operator = SnpLinAlg{Float32}(rhs_fixture)
    @test rhs_operator * ones(Float32, 9, 257) ==
          fill(Float32(18), 17, 257)
    @test transpose(rhs_operator) * ones(Float32, 17, 2049) ==
          fill(Float32(34), 9, 2049)
end
