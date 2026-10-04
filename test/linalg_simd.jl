function simd_test_fixture(
    rows::Int,
    columns::Int;
    allow_missing::Bool = true,
)
    dense = Matrix{UInt8}(undef, rows, columns)
    packed = SnpArray(undef, rows, columns)
    fill!(packed.data, 0x55)
    for column in 1:columns, row in 1:rows
        code = UInt8(mod(row + 2column, 4))
        if (rows == 1 || !allow_missing) && code == 0x01
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

@testset "SnpLinAlg scheduler tile boundaries" begin
    probe_m, probe_n = 4097, 2049

    for T in (Float32, Float64)
        # A*x / A*X: samples axis (row_step), small SNP count.
        for k in (1, 8)
            step = SnpArrays._tile_sizes(T, probe_m, 37, k, :forward)[1]
            for m in (step - 1, step + 1, 2step + 1)
                packed, dense = simd_test_fixture(m, 37; allow_missing=false)
                reference = simd_test_reference(
                    dense, T, ADDITIVE_MODEL, false, false, true,
                )
                operator = SnpLinAlg{T}(packed)
                rhs = simd_test_rhs(T, 37, k)
                expected = reference * rhs
                actual = SnpArrays._tile_sizes(T, m, 37, k, :forward)[1]
                @test actual <= step
                m > step && @test m > actual
                result = k == 1 ? operator * vec(rhs) : operator * rhs
                @test isapprox(result, k == 1 ? vec(expected) : expected;
                               atol=64eps(T), rtol=64eps(T), nans=true)
            end
        end

        # A*X inner column axis (column_step), samples = 257.
        let column_step =
                SnpArrays._tile_sizes(T, 257, probe_n, 8, :forward)[2]
            n = column_step + 1
            actual = SnpArrays._tile_sizes(T, 257, n, 8, :forward)[2]
            @test actual <= column_step
            @test n > actual
            packed, dense = simd_test_fixture(257, n; allow_missing=false)
            reference = simd_test_reference(
                dense, T, ADDITIVE_MODEL, false, false, true,
            )
            operator = SnpLinAlg{T}(packed)
            rhs = simd_test_rhs(T, n, 8)
            @test isapprox(operator * rhs, reference * rhs;
                           atol=64eps(T), rtol=64eps(T), nans=true)
        end

        # transpose(A)*x / transpose(A)*X: SNP axis (column_step), m = 37.
        for k in (1, 8)
            step = SnpArrays._tile_sizes(T, 37, probe_n, k, :transpose)[2]
            for n in (step - 1, step + 1, 2step + 1)
                packed, dense = simd_test_fixture(37, n; allow_missing=false)
                reference = simd_test_reference(
                    dense, T, ADDITIVE_MODEL, false, false, true,
                )
                operator = SnpLinAlg{T}(packed)
                rhs = simd_test_rhs(T, 37, k)
                expected = transpose(reference) * rhs
                actual = SnpArrays._tile_sizes(T, 37, n, k, :transpose)[2]
                @test actual <= step
                n > step && @test n > actual
                result = k == 1 ?
                    transpose(operator) * vec(rhs) : transpose(operator) * rhs
                @test isapprox(result, k == 1 ? vec(expected) : expected;
                               atol=64eps(T), rtol=64eps(T), nans=true)
            end
        end

        # transpose(A)*x / transpose(A)*X: sample-block axis (row_step),
        # SNPs = 37.
        for k in (1, 8)
            step = SnpArrays._tile_sizes(T, probe_m, 37, k, :transpose)[1]
            for m in (step - 1, step + 1, 2step + 1)
                packed, dense = simd_test_fixture(m, 37; allow_missing=false)
                reference = simd_test_reference(
                    dense, T, ADDITIVE_MODEL, false, false, true,
                )
                operator = SnpLinAlg{T}(packed)
                rhs = simd_test_rhs(T, m, k)
                expected = transpose(reference) * rhs
                actual = SnpArrays._tile_sizes(T, m, 37, k, :transpose)[1]
                @test actual <= step
                m > step && @test m > actual
                result = k == 1 ?
                    transpose(operator) * vec(rhs) : transpose(operator) * rhs
                @test isapprox(result, k == 1 ? vec(expected) : expected;
                               atol=64eps(T), rtol=64eps(T), nans=true)
            end
        end

        # rhs axis: k = 257 exercises the outer rhs_step = 256 loop.
        packed, dense = simd_test_fixture(17, 9; allow_missing=false)
        reference = simd_test_reference(
            dense, T, ADDITIVE_MODEL, false, false, true,
        )
        operator = SnpLinAlg{T}(packed)
        forward_rhs = simd_test_rhs(T, 9, 257)
        @test isapprox(operator * forward_rhs, reference * forward_rhs;
                       atol=64eps(T), rtol=64eps(T), nans=true)
        transpose_rhs = simd_test_rhs(T, 17, 257)
        @test isapprox(transpose(operator) * transpose_rhs,
                       transpose(reference) * transpose_rhs;
                       atol=64eps(T), rtol=64eps(T), nans=true)

        # A one-column matrix rhs keeps the k > 1 (matrix) rules, since it
        # still runs the register-tiled kernel rather than the vector one.
        @test SnpArrays._tile_sizes(T, 4097, 54051, 1, :forward;
                                     vector=false)[2] ==
              SnpArrays._tile_sizes(T, 4097, 54051, 2, :forward)[2]
        @test SnpArrays._tile_sizes(T, 4097, 54051, 1, :transpose;
                                     vector=false)[1] ==
              SnpArrays._tile_sizes(T, 4097, 54051, 2, :transpose)[1]

        let step = SnpArrays._tile_sizes(T, 4097, 37, 1, :forward;
                                          vector=false)[1]
            packed, dense = simd_test_fixture(step + 1, 37;
                                               allow_missing=false)
            reference = simd_test_reference(
                dense, T, ADDITIVE_MODEL, false, false, true,
            )
            operator = SnpLinAlg{T}(packed)
            rhs = simd_test_rhs(T, 37, 1)
            @test isapprox(operator * rhs, reference * rhs;
                           atol=64eps(T), rtol=64eps(T), nans=true)
        end

        let step = SnpArrays._tile_sizes(T, 4097, 37, 1, :transpose;
                                          vector=false)[1]
            packed, dense = simd_test_fixture(step + 1, 37;
                                               allow_missing=false)
            reference = simd_test_reference(
                dense, T, ADDITIVE_MODEL, false, false, true,
            )
            operator = SnpLinAlg{T}(packed)
            rhs = simd_test_rhs(T, step + 1, 1)
            @test isapprox(transpose(operator) * rhs,
                           transpose(reference) * rhs;
                           atol=64eps(T), rtol=64eps(T), nans=true)
        end

        # every _tile_sizes result satisfies the basic invariants
        for direction in (:forward, :transpose),
            k in (1, 8, 128, 257),
            m in (0, 1, 17, 4097),
            n in (0, 1, 9, 2049)

            row_step, column_step, rhs_step =
                SnpArrays._tile_sizes(T, m, n, k, direction)
            @test row_step % 16 == 0
            @test column_step >= 1
            @test rhs_step >= 1
        end
    end
end

@testset "SnpLinAlg transpose over a column range" begin
    rows, columns = 53, 97
    packed, dense = simd_test_fixture(rows, columns)
    ranges = (1:columns, 1:1, columns:columns, 2:2, 5:17, 9:21, 33:96,
              1:13, columns - 12:columns, 40:47, 40:48, 40:49)

    for T in (Float32, Float64, Float16)
        operator = SnpLinAlg{T}(packed; model=ADDITIVE_MODEL, impute=true,
                                center=true, scale=true)
        reference = simd_test_reference(dense, T, ADDITIVE_MODEL, true, true,
                                        true)
        tolerance = T == Float16 ? 128eps(Float16) : T(64) * eps(T)
        for cols in ranges, transposed in (transpose(operator),
                                           adjoint(operator))
            restricted = transpose(view(reference, :, cols))

            vector_rhs = vec(simd_test_rhs(T, rows, 1))
            expected = restricted * vector_rhs
            destination = fill(T(19), length(cols))
            @test mul!(destination, transposed, vector_rhs, cols) ===
                  destination
            @test isapprox(destination, expected; nans=true, atol=tolerance,
                           rtol=tolerance)

            for width in (1, 8, 33)
                matrix_rhs = simd_test_rhs(T, rows, width)
                expected_matrix = restricted * matrix_rhs
                matrix_destination = fill(T(19), length(cols), width)
                @test mul!(matrix_destination, transposed, matrix_rhs,
                           cols) === matrix_destination
                @test isapprox(matrix_destination, expected_matrix;
                               nans=true, atol=tolerance, rtol=tolerance)
            end
        end
    end

    # A restricted range must agree with the full product over that range.
    operator = SnpLinAlg{Float64}(packed; impute=true, center=true,
                                  scale=true)
    matrix_rhs = simd_test_rhs(Float64, rows, 8)
    full = mul!(Matrix{Float64}(undef, columns, 8), transpose(operator),
                matrix_rhs)
    for cols in ranges
        block = mul!(Matrix{Float64}(undef, length(cols), 8),
                     transpose(operator), matrix_rhs, cols)
        @test block ≈ view(full, cols, :)
    end

    @test_throws ArgumentError mul!(zeros(Float64, 3), transpose(operator),
                                    zeros(Float64, rows), 0:2)
    @test_throws ArgumentError mul!(zeros(Float64, 3), transpose(operator),
                                    zeros(Float64, rows),
                                    (columns - 1):(columns + 1))
    @test_throws DimensionMismatch mul!(zeros(Float64, 4),
                                        transpose(operator),
                                        zeros(Float64, rows), 1:3)
    @test_throws DimensionMismatch mul!(zeros(Float64, 3, 8),
                                        transpose(operator),
                                        zeros(Float64, rows + 1, 8), 1:3)
end

@testset "SnpLinAlg rhs width selection" begin
    for T in (Float32, Float64)
        full = SnpArrays._vector_width(T)
        previous = 0
        for k in 1:(4full + 3)
            width = SnpArrays._rhs_width(T, k)
            @test ispow2(width)
            @test 4 <= width <= full
            @test width <= max(k, 4)
            @test width >= previous
            previous = width
        end
        # A rhs at least as wide as the register uses the full register.
        @test SnpArrays._rhs_width(T, full) == full
        @test SnpArrays._rhs_width(T, 4full) == full
    end
end

@testset "SnpLinAlg micro-kernel edge cases" begin
    for T in (Float32, Float64)
        mr, nr = SnpArrays._micro_tile(T)
        w = SnpArrays._vector_width(T)
        tolerance = T(64) * eps(T)

        # 1. sample-count remainders against MR, 16, and 16q + r.
        column_count = 2mr + 3
        sample_counts = sort(unique(vcat(
            collect(1:(mr - 1)), mr, collect((mr + 1):15), 16,
            [16q + r for q in (1, 3) for r in 1:15],
        )))
        for m in sample_counts
            packed, dense = simd_test_fixture(m, column_count;
                                              allow_missing=false)
            reference = simd_test_reference(dense, T, ADDITIVE_MODEL, false,
                                            false, true)
            operator = SnpLinAlg{T}(packed)

            k = nr + 3
            forward_rhs = simd_test_rhs(T, column_count, k)
            @test isapprox(operator * forward_rhs, reference * forward_rhs;
                           atol=tolerance, rtol=tolerance, nans=true)
            transpose_rhs = simd_test_rhs(T, m, k)
            @test isapprox(transpose(operator) * transpose_rhs,
                           transpose(reference) * transpose_rhs;
                           atol=tolerance, rtol=tolerance, nans=true)

            forward_vector = vec(simd_test_rhs(T, column_count, 1))
            @test isapprox(operator * forward_vector,
                           vec(reference * forward_vector);
                           atol=tolerance, rtol=tolerance, nans=true)
        end

        # 2. rhs-width remainders for a fixed, tile-remainder-bearing shape.
        m = 16 * 3 + 5
        n = 2mr + 3
        packed, dense = simd_test_fixture(m, n; allow_missing=false)
        reference = simd_test_reference(dense, T, ADDITIVE_MODEL, false,
                                        false, true)
        operator = SnpLinAlg{T}(packed)
        widths = sort(unique(vcat(
            [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 128, 130],
            [w - 1, w, w + 1, nr - 1, nr, nr + 1],
        )))
        for k in widths
            forward_rhs = simd_test_rhs(T, n, k)
            @test isapprox(operator * forward_rhs, reference * forward_rhs;
                           atol=tolerance, rtol=tolerance, nans=true)
            transpose_rhs = simd_test_rhs(T, m, k)
            @test isapprox(transpose(operator) * transpose_rhs,
                           transpose(reference) * transpose_rhs;
                           atol=tolerance, rtol=tolerance, nans=true)
        end

        # 3. SNP-count remainders.
        m3 = 37
        k3 = nr + 1
        for n3 in (1, mr - 1, mr, mr + 1, 2mr + 1, 16, 17)
            packed3, dense3 = simd_test_fixture(m3, n3; allow_missing=false)
            reference3 = simd_test_reference(dense3, T, ADDITIVE_MODEL,
                                             false, false, true)
            operator3 = SnpLinAlg{T}(packed3)
            forward_rhs = simd_test_rhs(T, n3, k3)
            @test isapprox(operator3 * forward_rhs, reference3 * forward_rhs;
                           atol=tolerance, rtol=tolerance, nans=true)
            transpose_rhs = simd_test_rhs(T, m3, k3)
            @test isapprox(transpose(operator3) * transpose_rhs,
                           transpose(reference3) * transpose_rhs;
                           atol=tolerance, rtol=tolerance, nans=true)
        end

        # 4. mul! with prefilled destination and adjoint.
        test_simd_products(operator, reference, nr + 1; tolerance=tolerance)
    end
end

@testset "SnpLinAlg lookup-table A*X" begin
    min_rows = SnpArrays.LOOKUP_MIN_ROWS
    min_rhs = SnpArrays.LOOKUP_MIN_RHS
    tile = SnpArrays.LOOKUP_ROW_TILE
    chunk = SnpArrays.LOOKUP_CHUNK_SNPS

    @test SnpArrays._uses_lookup_kernel(min_rows, min_rhs)
    @test !SnpArrays._uses_lookup_kernel(min_rows - 1, min_rhs)
    @test !SnpArrays._uses_lookup_kernel(min_rows, min_rhs - 1)

    # 1. 4x4 two-bit transpose: sample s of x = r0 | r1<<8 | r2<<16 | r3<<24
    #    gets code (r_l >> 2s) & 3 in bits 2l:2l+1.
    for x in (0x00000000, 0xffffffff, 0x1b6ce4b1, 0x93c0a55f), s in 0:3
        expected = UInt8(0)
        for l in 0:3
            byte = (x >>> (8l)) & 0xff
            expected |= UInt8((byte >>> (2s)) & 3) << (2l)
        end
        @test SnpArrays._lookup_gather4(x, s) == expected
    end

    for T in (Float32, Float64)
        nr = 2 * SnpArrays._vector_width(T)
        tolerance = T(64) * eps(T)

        # 2. Sample, SNP, and rhs remainders against the tile, chunk, byte,
        #    4-SNP block, and rhs sub-block boundaries.
        for m in (min_rows, min_rows + 1, 2tile + 3),
            n in (3, 4, 5, chunk - 1, chunk + 2),
            k in (min_rhs, nr - 1, nr, nr + 1, 2nr + 3)
            packed, dense = simd_test_fixture(m, n)
            reference = simd_test_reference(dense, T, ADDITIVE_MODEL, true,
                                            true, true)
            operator = SnpLinAlg{T}(packed; center=true, scale=true)
            rhs = simd_test_rhs(T, n, k)
            @test isapprox(operator * rhs, reference * rhs; atol=tolerance,
                           rtol=tolerance, nans=true)
        end

        # 2b. Rhs widths on both sides of the 2W slice and the 32-wide slice,
        #     including partial last slices.
        packed, dense = simd_test_fixture(min_rows + 1, 37)
        reference = simd_test_reference(dense, T, ADDITIVE_MODEL, true, true,
                                        true)
        operator = SnpLinAlg{T}(packed; center=true, scale=true)
        for k in (5, 16, 17, 40, 64)
            rhs = simd_test_rhs(T, 37, k)
            @test isapprox(operator * rhs, reference * rhs; atol=tolerance,
                           rtol=tolerance, nans=true)
        end

        # 3. Every model and transformation at one lookup-path shape.
        packed, dense = simd_test_fixture(min_rows + 5, 9)
        for model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL),
            impute in (false, true), center in (false, true),
            scale in (false, true)
            operator = SnpLinAlg{T}(packed; model=model, impute=impute,
                                    center=center, scale=scale)
            reference = simd_test_reference(dense, T, model, center, scale,
                                            impute)
            rhs = simd_test_rhs(T, 9, nr + 1)
            @test isapprox(operator * rhs, reference * rhs; atol=tolerance,
                           rtol=tolerance, nans=true)
        end

        # 4. mul! overwrites a prefilled destination and, after a warm-up,
        #    a repeated product reuses the workspaces.
        operator = SnpLinAlg{T}(packed; center=true, scale=true)
        reference = simd_test_reference(dense, T, ADDITIVE_MODEL, true, true,
                                        true)
        rhs = simd_test_rhs(T, 9, nr + 1)
        destination = fill(T(19), min_rows + 5, nr + 1)
        @test mul!(destination, operator, rhs) === destination
        @test isapprox(destination, reference * rhs; atol=tolerance,
                       rtol=tolerance, nans=true)
        panel_length = length(operator.panel)
        blk_length = length(operator.blk)
        mul!(destination, operator, rhs)
        @test length(operator.panel) == panel_length
        @test length(operator.blk) == blk_length
    end

    # 5. A missing genotype with impute=false yields NaN in its row only.
    packed = SnpArray(undef, min_rows, 5)
    fill!(packed.data, 0x00)
    packed.data[1, 2] = 0x04
    operator = SnpLinAlg{Float64}(packed; impute=false)
    product = operator * ones(5, min_rhs)
    @test all(isnan, product[2, :])
    @test all(iszero, product[[1; 3:min_rows], :])
end

@testset "concurrent products on one SnpLinAlg" begin
    min_rows = SnpArrays.LOOKUP_MIN_ROWS
    min_rhs = SnpArrays.LOOKUP_MIN_RHS
    # A register-tiled shape and a lookup-table shape.
    for (rows, widths) in ((203, (2, 3)), (min_rows + 5, (min_rhs, 2min_rhs)))
        packed, _ = simd_test_fixture(rows, 61)
        operator = SnpLinAlg{Float64}(packed; center=true, scale=true)
        forward_rhs = map(k -> simd_test_rhs(Float64, 61, k), widths)
        transpose_rhs = map(k -> simd_test_rhs(Float64, rows, k), widths)
        forward = map(rhs -> operator * rhs, forward_rhs)
        transposed = map(rhs -> transpose(operator) * rhs, transpose_rhs)
        for _ in 1:20
            tasks = (
                Threads.@spawn(operator * forward_rhs[1]),
                Threads.@spawn(operator * forward_rhs[2]),
                Threads.@spawn(transpose(operator) * transpose_rhs[1]),
                Threads.@spawn(transpose(operator) * transpose_rhs[2]),
            )
            results = map(fetch, tasks)
            @test results[1] == forward[1]
            @test results[2] == forward[2]
            @test results[3] == transposed[1]
            @test results[4] == transposed[2]
        end
    end
end
