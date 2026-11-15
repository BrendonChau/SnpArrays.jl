# Task step of the product the operator runs for `k` rhs columns: the vector
# kernels at `k == 1` through `*` with a vector, the register-tiled ones else.
function simd_test_forward_row_step(::Type{T}, m::Int, k::Int) where T
    return k == 1 ? SnpArrays._snparray_ax_row_step(T, m) :
                    SnpArrays._snparray_AX_steps(T, m, k)[1]
end

function simd_test_transpose_row_step(::Type{T}, n::Int, k::Int) where T
    return k == 1 ? SnpArrays._snparray_atx_steps(T, n)[1] :
                    SnpArrays._snparray_AtX_steps(T, n, k)[1]
end

function simd_test_transpose_column_step(::Type{T}, n::Int, k::Int) where T
    return k == 1 ? SnpArrays._snparray_atx_steps(T, n)[2] :
                    SnpArrays._snparray_AtX_steps(T, n, k)[2]
end

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

@testset "SnpLinAlg from a view of a SnpArray" begin
    packed, dense = simd_test_fixture(17, 9)
    columns = 3:6
    operator = SnpLinAlg{Float64}(view(packed, :, columns); center=true,
                                  scale=true)
    reference = simd_test_reference(dense[:, columns], Float64,
                                    ADDITIVE_MODEL, true, true, true)
    @test size(operator) == (17, length(columns))
    @test isapprox(Matrix(operator), reference; atol=64eps(Float64),
                   rtol=64eps(Float64))
    @test operator.s.data !== packed.data
    @test_throws MethodError SnpLinAlg{Float64}(view(packed, 1:8, :))
    @test_throws MethodError SnpLinAlg{Float64}(view(packed, :, [1, 3]))
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
            step = simd_test_forward_row_step(T, probe_m, k)
            for m in (step - 1, step + 1, 2step + 1)
                packed, dense = simd_test_fixture(m, 37; allow_missing=false)
                reference = simd_test_reference(
                    dense, T, ADDITIVE_MODEL, false, false, true,
                )
                operator = SnpLinAlg{T}(packed)
                rhs = simd_test_rhs(T, 37, k)
                expected = reference * rhs
                actual = simd_test_forward_row_step(T, m, k)
                @test actual <= step
                m > step && @test m > actual
                result = k == 1 ? operator * vec(rhs) : operator * rhs
                @test isapprox(result, k == 1 ? vec(expected) : expected;
                               atol=64eps(T), rtol=64eps(T), nans=true)
            end
        end

        # A*X inner column axis (column_step), samples = 257.
        let column_step = SnpArrays._snparray_AX_steps(T, 257, 8)[2]
            n = column_step + 1
            actual = SnpArrays._snparray_AX_steps(T, 257, 8)[2]
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
            step = simd_test_transpose_column_step(T, probe_n, k)
            for n in (step - 1, step + 1, 2step + 1)
                packed, dense = simd_test_fixture(37, n; allow_missing=false)
                reference = simd_test_reference(
                    dense, T, ADDITIVE_MODEL, false, false, true,
                )
                operator = SnpLinAlg{T}(packed)
                rhs = simd_test_rhs(T, 37, k)
                expected = transpose(reference) * rhs
                actual = simd_test_transpose_column_step(T, n, k)
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
            step = simd_test_transpose_row_step(T, 37, k)
            for m in (step - 1, step + 1, 2step + 1)
                packed, dense = simd_test_fixture(m, 37; allow_missing=false)
                reference = simd_test_reference(
                    dense, T, ADDITIVE_MODEL, false, false, true,
                )
                operator = SnpLinAlg{T}(packed)
                rhs = simd_test_rhs(T, m, k)
                expected = transpose(reference) * rhs
                actual = simd_test_transpose_row_step(T, 37, k)
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
        @test SnpArrays._snparray_AX_steps(T, 4097, 1)[2] ==
              SnpArrays._snparray_AX_steps(T, 4097, 2)[2]
        @test SnpArrays._snparray_AtX_steps(T, 54051, 1)[1] ==
              SnpArrays._snparray_AtX_steps(T, 54051, 2)[1]

        let step = SnpArrays._snparray_AX_steps(T, 4097, 1)[1]
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

        let step = SnpArrays._snparray_AtX_steps(T, 37, 1)[1]
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

        # every step function result satisfies the basic invariants
        for direction in (:forward, :transpose),
            k in (1, 8, 128, 257),
            m in (0, 1, 17, 4097),
            n in (0, 1, 9, 2049)

            if k == 1 && direction == :forward
                @test SnpArrays._snparray_ax_row_step(T, m) % 16 == 0
            elseif k == 1
                row_step, column_step = SnpArrays._snparray_atx_steps(T, n)
                @test row_step % 16 == 0
                @test column_step >= 1
            else
                row_step, column_step, rhs_step = direction == :forward ?
                    SnpArrays._snparray_AX_steps(T, m, k) :
                    SnpArrays._snparray_AtX_steps(T, n, k)
                @test row_step % 16 == 0
                @test column_step >= 1
                @test rhs_step >= 1
            end
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
        mr, nr = SnpArrays._register_tile_shape(T)
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
    SnpArrays.LOOKUP_MIN_RHS[] = (4, 4)
    min_rows = SnpArrays.LOOKUP_MIN_ROWS
    min_rhs = 4
    tile = SnpArrays.LOOKUP_ROW_TILE
    chunk = SnpArrays.LOOKUP_CHUNK_SNPS

    for T in (Float32, Float64)
        @test SnpArrays._uses_lookup_kernel(T, min_rows, min_rhs)
        @test !SnpArrays._uses_lookup_kernel(T, min_rows - 1, min_rhs)
        @test !SnpArrays._uses_lookup_kernel(T, min_rows, min_rhs - 1)
    end

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

    # Slice shape: fewest equal slices of at most LOOKUP_SLICE_MAX columns,
    # padding under one vector per slice.
    for T in (Float32, Float64)
        shape_ok = true
        for k in 1:200
            lanes, vectors = SnpArrays._lookup_slice_shape(T, k)
            slice = vectors * lanes
            slices = cld(k, slice)
            shape_ok &= lanes == SnpArrays._rhs_width(T, k) &&
                        vectors >= 1 &&
                        slice <= SnpArrays.LOOKUP_SLICE_MAX &&
                        slices == cld(k, SnpArrays.LOOKUP_SLICE_MAX) &&
                        slices * slice - k < slices * lanes
        end
        @test shape_ok
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

        # 2b. Every rhs width from the lookup minimum to 70, covering every
        #     slice shape: one-vector and odd-vector slices and partial last
        #     slices.
        packed, dense = simd_test_fixture(min_rows + 1, 37)
        reference = simd_test_reference(dense, T, ADDITIVE_MODEL, true, true,
                                        true)
        operator = SnpLinAlg{T}(packed; center=true, scale=true)
        for k in min_rhs:70
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
    SnpArrays.LOOKUP_MIN_RHS[] = SnpArrays._detect_lookup_min_rhs()
end

@testset "lookup-table rhs threshold by host and element type" begin
    expected = Sys.ARCH === :aarch64 ? (32, typemax(Int)) : (4, 4)
    @test SnpArrays._detect_lookup_min_rhs() == expected
    saved = SnpArrays.LOOKUP_MIN_RHS[]
    SnpArrays.LOOKUP_MIN_RHS[] = (32, typemax(Int))
    min_rows = SnpArrays.LOOKUP_MIN_ROWS
    @test SnpArrays._uses_lookup_kernel(Float32, min_rows, 32)
    @test !SnpArrays._uses_lookup_kernel(Float32, min_rows, 31)
    @test !SnpArrays._uses_lookup_kernel(Float64, min_rows, 4096)
    SnpArrays.LOOKUP_MIN_RHS[] = saved
end

@testset "transpose tile height by host" begin
    @test SnpArrays._detect_atx_tile_columns(64) == 4
    @test SnpArrays._detect_atx_tile_columns(32) ==
          (Sys.ARCH === :x86_64 ? 2 : 6)
    saved = SnpArrays.ATX_TILE_COLUMNS[]
    packed, dense = simd_test_fixture(37, 23; allow_missing=false)
    for T in (Float32, Float64)
        reference = simd_test_reference(
            dense, T, ADDITIVE_MODEL, false, false, true,
        )
        operator = SnpLinAlg{T}(packed)
        rhs = simd_test_rhs(T, 37, 20)
        expected = transpose(reference) * rhs
        for tile_columns in (1, 2, 4, 6, 8)
            SnpArrays.ATX_TILE_COLUMNS[] = tile_columns
            @test isapprox(transpose(operator) * rhs, expected;
                           atol=64eps(T), rtol=64eps(T))
        end
    end
    SnpArrays.ATX_TILE_COLUMNS[] = saved
end

@testset "SnpLinAlg sums of squares from genotype counts" begin
    for allow_missing in (true, false)
        packed, dense = simd_test_fixture(37, 23; allow_missing=allow_missing)
        for T in (Float32, Float64),
            model in (ADDITIVE_MODEL, DOMINANT_MODEL, RECESSIVE_MODEL),
            impute in (false, true), center in (false, true),
            scale in (false, true)
            operator = SnpLinAlg{T}(packed; model=model, impute=impute,
                                    center=center, scale=scale)
            reference = simd_test_reference(dense, T, model, center, scale,
                                            impute)
            tolerance = T(64) * eps(T)
            expected = sum(abs2, reference; dims=1)
            observed = sum(abs2, operator; dims=1)
            @test size(observed) == (1, 23)
            @test isapprox(observed, expected; nans=true, atol=tolerance,
                           rtol=tolerance)
            @test isapprox(sum(abs2, operator), sum(expected); nans=true,
                           atol=tolerance, rtol=tolerance)
            @test isapprox(sum(abs2, operator; dims=2),
                           sum(abs2, reference; dims=2); nans=true,
                           atol=tolerance, rtol=tolerance)
        end
    end
end

@testset "concurrent products on one SnpLinAlg" begin
    SnpArrays.LOOKUP_MIN_RHS[] = (4, 4)
    min_rows = SnpArrays.LOOKUP_MIN_ROWS
    min_rhs = 4
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
    SnpArrays.LOOKUP_MIN_RHS[] = SnpArrays._detect_lookup_min_rhs()
end

@testset "SnpLinAlg{Float32} with Float64 output or rhs" begin
    SnpArrays.LOOKUP_MIN_RHS[] = (4, 4)
    min_rows = SnpArrays.LOOKUP_MIN_ROWS
    chunk = SnpArrays.LOOKUP_CHUNK_SNPS
    nr = 2 * SnpArrays._vector_width(Float32)
    combinations = ((Float32, Float32), (Float32, Float64),
                    (Float64, Float32), (Float64, Float64))

    # 1. One accumulation run per output entry (at most 64 SNPs for A*X, at
    #    most 64 samples for Aᵀ*X, one lookup chunk), so widening the run
    #    into Float64 reproduces the Float32 product of Float32.(rhs) exactly.
    for (m, n, k) in ((37, 23, 5), (37, 23, nr + 1), (min_rows + 5, 37, nr + 1))
        packed, _ = simd_test_fixture(m, n)
        operator = SnpLinAlg{Float32}(packed; center=true, scale=true)
        @test SnpArrays._uses_lookup_kernel(Float32, m, k) == (m >= min_rows)
        for TR in (Float32, Float64)
            rhs = simd_test_rhs(TR, n, k)
            expected = mul!(zeros(Float32, m, k), operator, Float32.(rhs))
            @test mul!(zeros(Float32, m, k), operator, rhs) == expected
            @test mul!(zeros(Float64, m, k), operator, rhs) ==
                  Float64.(expected)
            if m <= 64
                rhs = simd_test_rhs(TR, m, k)
                expected = mul!(zeros(Float32, n, k), transpose(operator),
                                Float32.(rhs))
                @test mul!(zeros(Float64, n, k), transpose(operator), rhs) ==
                      Float64.(expected)
            end
        end
    end

    # 2. Against the dense Float64 product: register-tiled shapes one past
    #    the Float32 SNP and sample blocks, and a lookup shape past a chunk.
    k = nr + 3
    forward_step = SnpArrays._snparray_AX_steps(Float32, 37, k)[2]
    transpose_step = SnpArrays._snparray_AtX_steps(Float32, 23, k)[1]
    tolerance = 64eps(Float32)
    α, β = 2.5, -0.75
    for (m, n) in ((37, forward_step + 1), (transpose_step + 1, 23),
                   (min_rows + 5, chunk + 2))
        packed, dense = simd_test_fixture(m, n)
        reference = simd_test_reference(dense, Float64, ADDITIVE_MODEL, true,
                                        true, true)
        operator = SnpLinAlg{Float32}(packed; center=true, scale=true)
        cols = 3:(n - 2)
        for (TO, TR) in combinations
            rhs = simd_test_rhs(TR, n, k)
            expected = reference * Float64.(rhs)
            @test isapprox(mul!(zeros(TO, m, k), operator, rhs), expected;
                           atol=tolerance, rtol=tolerance)
            prior = simd_test_rhs(TO, m, k)
            @test isapprox(mul!(copy(prior), operator, rhs, α, β),
                           α .* expected .+ β .* prior;
                           atol=tolerance, rtol=tolerance)

            rhs = simd_test_rhs(TR, m, k)
            expected = transpose(reference) * Float64.(rhs)
            prior = simd_test_rhs(TO, n, k)
            for wrapped in (transpose(operator), adjoint(operator))
                @test isapprox(mul!(zeros(TO, n, k), wrapped, rhs), expected;
                               atol=tolerance, rtol=tolerance)
                @test isapprox(mul!(copy(prior), wrapped, rhs, α, β),
                               α .* expected .+ β .* prior;
                               atol=tolerance, rtol=tolerance)
                @test isapprox(
                    mul!(zeros(TO, length(cols), k), wrapped, rhs, cols),
                    transpose(reference[:, cols]) * Float64.(rhs);
                    atol=tolerance, rtol=tolerance,
                )
            end
        end
    end

    # 3. Other mixed element types are rejected.
    packed, _ = simd_test_fixture(37, 23)
    operator = SnpLinAlg{Float64}(packed)
    @test_throws ArgumentError mul!(zeros(Float32, 37, 2), operator,
                                    zeros(23, 2))
    @test_throws ArgumentError mul!(zeros(37, 2), operator,
                                    zeros(Float32, 23, 2))
    @test_throws ArgumentError mul!(zeros(Float32, 23, 2),
                                    transpose(operator), zeros(37, 2))
    @test_throws ArgumentError mul!(zeros(Float32, 3, 2), adjoint(operator),
                                    zeros(37, 2), 1:3)
    @test_throws ArgumentError mul!(zeros(23, 2), transpose(operator),
                                    zeros(Float32, 37, 2), 1.0, 0.0)
    SnpArrays.LOOKUP_MIN_RHS[] = SnpArrays._detect_lookup_min_rhs()
end

@testset "SnpLinAlg one-column matrix products" begin
    # A one-column same-type product runs the vector kernel, so it equals the
    # vector product exactly; other inputs keep the matrix path.
    packed, dense = simd_test_fixture(37, 23)
    m, n = size(dense)
    cols = 3:(n - 2)
    α, β = 2.5, -0.75
    for T in (Float32, Float64),
        impute in (false, true), center in (false, true), scale in (false, true)
        operator = SnpLinAlg{T}(packed; impute=impute, center=center,
                                scale=scale)
        rhs = simd_test_rhs(T, n, 1)
        prior = simd_test_rhs(T, m, 1)
        @test isequal(mul!(zeros(T, m, 1), operator, rhs),
                      reshape(operator * vec(rhs), m, 1))
        @test isequal(mul!(copy(prior), operator, rhs, α, β),
                      reshape(mul!(vec(copy(prior)), operator, vec(rhs), α, β),
                              m, 1))
        rhs = simd_test_rhs(T, m, 1)
        prior = simd_test_rhs(T, n, 1)
        for wrapped in (transpose(operator), adjoint(operator))
            destination = zeros(T, n, 1)
            @test mul!(destination, wrapped, rhs) === destination
            @test isequal(destination, reshape(wrapped * vec(rhs), n, 1))
            @test isequal(mul!(copy(prior), wrapped, rhs, α, β),
                          reshape(mul!(vec(copy(prior)), wrapped, vec(rhs),
                                       α, β), n, 1))
            @test isequal(mul!(zeros(T, length(cols), 1), wrapped, rhs, cols),
                          reshape(mul!(zeros(T, length(cols)), wrapped,
                                       vec(rhs), cols), length(cols), 1))
        end
    end

    reference = simd_test_reference(dense, Float64, ADDITIVE_MODEL, true,
                                    true, true)
    tolerance = 64eps(Float32)
    operator32 = SnpLinAlg{Float32}(packed; center=true, scale=true)
    rhs = simd_test_rhs(Float64, n, 1)
    @test isapprox(mul!(zeros(m, 1), operator32, rhs), reference * rhs;
                   atol=tolerance, rtol=tolerance)
    rhs = simd_test_rhs(Float64, m, 1)
    @test isapprox(mul!(zeros(n, 1), transpose(operator32), rhs),
                   transpose(reference) * rhs; atol=tolerance, rtol=tolerance)

    operator = SnpLinAlg{Float64}(packed; center=true, scale=true)
    tolerance = 64eps(Float64)
    rhs_parent = zeros(n, 2)
    rhs_parent[:, 2] .= vec(simd_test_rhs(Float64, n, 1))
    rhs = view(rhs_parent, :, 2:2)
    destination_parent = fill(19.0, 2m, 1)
    destination = view(destination_parent, 1:2:2m, :)
    @test mul!(destination, operator, rhs) === destination
    @test isapprox(destination, reference * rhs; atol=tolerance,
                   rtol=tolerance)
    rhs_parent = zeros(2m, 1)
    rhs = view(rhs_parent, 2:2:2m, :)
    rhs .= simd_test_rhs(Float64, m, 1)
    destination = fill(19.0, length(cols), 1)
    @test mul!(destination, adjoint(operator), rhs, cols) === destination
    @test isapprox(destination, transpose(reference[:, cols]) * rhs;
                   atol=tolerance, rtol=tolerance)

    @test_throws DimensionMismatch mul!(zeros(m + 1, 1), operator,
                                        zeros(n, 1))
    @test_throws DimensionMismatch mul!(zeros(m, 1), operator,
                                        zeros(n + 1, 1), 1.0, 0.0)
    @test_throws DimensionMismatch mul!(zeros(n, 1), transpose(operator),
                                        zeros(m + 1, 1))
    @test_throws DimensionMismatch mul!(zeros(n + 1, 1), adjoint(operator),
                                        zeros(m, 1), 1.0, 0.0)
    @test_throws DimensionMismatch mul!(zeros(length(cols) + 1, 1),
                                        transpose(operator), zeros(m, 1), cols)
    @test_throws ArgumentError mul!(zeros(n + 1, 1), transpose(operator),
                                    zeros(m, 1), 1:(n + 1))
end

# A one-column operand: a dense `Matrix` (j == 0), or column `j` of a
# three-column buffer whose other columns hold `sentinel`.
function simd_test_column(values::Matrix{T}, j::Int, sentinel::T) where T
    j == 0 && return copy(values), values
    buffer = fill(sentinel, size(values, 1), 3)
    buffer[:, j] .= vec(values)
    return view(buffer, :, j:j), buffer
end

function simd_test_others_untouched(buffer::Matrix, j::Int, sentinel)
    j == 0 && return true
    return all(isequal(sentinel), view(buffer, :, setdiff(1:3, j)))
end

@testset "SnpLinAlg products on contiguous column views" begin
    m, n = 70, 23
    cols = 3:17
    α, β = 2.5, -0.75
    packed, _ = simd_test_fixture(m, n)
    for T in (Float32, Float64)
        operator = SnpLinAlg{T}(packed; center=true, scale=true)
        sentinel = T(19)
        x = simd_test_rhs(T, n, 1)
        y = simd_test_rhs(T, m, 1)
        prior_m = simd_test_rhs(T, m, 1)
        prior_n = simd_test_rhs(T, n, 1)
        forward = mul!(zeros(T, m), operator, vec(x))
        forward5 = mul!(vec(copy(prior_m)), operator, vec(x), α, β)
        for jo in 0:2, jr in 0:2
            rhs, _ = simd_test_column(x, jr, sentinel)
            @test SnpArrays._use_vector_kernels(zeros(T, m, 1), T, rhs)
            out, buffer = simd_test_column(zeros(T, m, 1), jo, sentinel)
            @test mul!(out, operator, rhs) === out
            @test isequal(vec(out), forward)
            @test simd_test_others_untouched(buffer, jo, sentinel)
            out, buffer = simd_test_column(prior_m, jo, sentinel)
            mul!(out, operator, rhs, α, β)
            @test isequal(vec(out), forward5)
            @test simd_test_others_untouched(buffer, jo, sentinel)

            rhs, _ = simd_test_column(y, jr, sentinel)
            for wrapped in (transpose(operator), adjoint(operator))
                out, buffer = simd_test_column(zeros(T, n, 1), jo, sentinel)
                @test mul!(out, wrapped, rhs) === out
                @test isequal(vec(out), mul!(zeros(T, n), wrapped, vec(y)))
                @test simd_test_others_untouched(buffer, jo, sentinel)
                out, buffer = simd_test_column(prior_n, jo, sentinel)
                mul!(out, wrapped, rhs, α, β)
                @test isequal(vec(out),
                              mul!(vec(copy(prior_n)), wrapped, vec(y), α, β))
                @test simd_test_others_untouched(buffer, jo, sentinel)
                out, buffer = simd_test_column(zeros(T, length(cols), 1), jo,
                                               sentinel)
                mul!(out, wrapped, rhs, cols)
                @test isequal(vec(out), mul!(zeros(T, length(cols)), wrapped,
                                             vec(y), cols))
                @test simd_test_others_untouched(buffer, jo, sentinel)
            end
        end

        # Vector form on `view(buffer, :, 2)` for both operands.
        rhs_buffer = fill(sentinel, n, 3)
        rhs_buffer[:, 2] .= vec(x)
        out_buffer = fill(sentinel, m, 3)
        mul!(view(out_buffer, :, 2), operator, view(rhs_buffer, :, 2))
        @test isequal(out_buffer[:, 2], forward)
        @test simd_test_others_untouched(out_buffer, 2, sentinel)

        # The reduced operands reach the SIMD methods in matvec.jl.
        column = SnpArrays._column_vector(view(out_buffer, :, 2:2))
        @test column isa SubArray{T, 1, Matrix{T},
                                  Tuple{Base.Slice{Base.OneTo{Int}}, Int}, true}
        @test parent(column) === out_buffer
        @test column isa SnpArrays.SimdVector{T}
        ax_arguments = (column, packed.data, column, operator.values, 1:1, 1:1)
        atx_arguments = (ax_arguments..., 0)
        for (kernel, arguments) in (
            (SnpArrays._snparray_ax_kernel!, ax_arguments),
            (SnpArrays._snparray_atx_kernel!, atx_arguments),
        )
            method = which(kernel, typeof(arguments))
            @test endswith(String(method.file), "matvec.jl")
        end
    end
    @test SnpArrays._use_vector_kernels(zeros(Float64, 4, 1), Float32,
                                        zeros(Float64, 4, 1))
    @test !SnpArrays._use_vector_kernels(zeros(Float32, 4, 1), Float64,
                                         zeros(Float64, 4, 1))
    @test !SnpArrays._use_vector_kernels(view(zeros(8, 1), 1:2:8, :), Float64,
                                         zeros(4, 1))
end

@testset "SnpLinAlg{Float32} vector products with Float64 operands" begin
    combinations = ((Float32, Float32), (Float32, Float64),
                    (Float64, Float32), (Float64, Float64))
    α, β = 2.5, -0.75
    tolerance = 64eps(Float32)

    # 1. One accumulation run per output entry (at most MIXED_AX_RUN SNPs;
    #    a multiple of 16 samples, all inside one Aᵀ*y SIMD run), so the
    #    widened result reproduces the Float32 product of Float32.(rhs).
    m, n = 64, 37
    @test n <= SnpArrays.MIXED_AX_RUN
    @test m <= SnpArrays._snparray_atx_steps(Float32, n)[1]
    packed, _ = simd_test_fixture(m, n)
    operator = SnpLinAlg{Float32}(packed; center=true, scale=true)
    cols = 3:(n - 2)
    for TR in (Float32, Float64)
        x = vec(simd_test_rhs(TR, n, 1))
        y = vec(simd_test_rhs(TR, m, 1))
        expected = mul!(zeros(Float32, m), operator, Float32.(x))
        for TO in (Float32, Float64)
            @test isequal(mul!(zeros(TO, m), operator, x), TO.(expected))
        end
        for wrapped in (transpose(operator), adjoint(operator))
            expected = mul!(zeros(Float32, n), wrapped, Float32.(y))
            expected_cols = mul!(zeros(Float32, length(cols)), wrapped,
                                 Float32.(y), cols)
            for TO in (Float32, Float64)
                @test isequal(mul!(zeros(TO, n), wrapped, y), TO.(expected))
                @test isequal(mul!(zeros(TO, length(cols)), wrapped, y, cols),
                              TO.(expected_cols))
            end
        end
    end

    # 2. Against the dense Float64 product, on the small shape and on one
    #    past two A*x runs and one Aᵀ*y SIMD run, spanning several tasks.
    m_large = SnpArrays._snparray_atx_steps(Float32, 1)[1] + 37
    n_large = 2SnpArrays.MIXED_AX_RUN + 3
    @test SnpArrays._snparray_ax_row_step(Float32, m_large) < m_large
    for (m, n) in ((64, 37), (m_large, n_large))
        packed, dense = simd_test_fixture(m, n)
        reference = simd_test_reference(dense, Float64, ADDITIVE_MODEL, true,
                                        true, true)
        operator = SnpLinAlg{Float32}(packed; center=true, scale=true)
        cols = 3:(n - 2)
        for (TO, TR) in combinations
            x = vec(simd_test_rhs(TR, n, 1))
            expected = reference * Float64.(x)
            @test isapprox(mul!(zeros(TO, m), operator, x), expected;
                           atol=tolerance, rtol=tolerance)
            prior = vec(simd_test_rhs(TO, m, 1))
            @test isapprox(mul!(copy(prior), operator, x, α, β),
                           α .* expected .+ β .* prior;
                           atol=tolerance, rtol=tolerance)
            y = vec(simd_test_rhs(TR, m, 1))
            expected = transpose(reference) * Float64.(y)
            prior = vec(simd_test_rhs(TO, n, 1))
            for wrapped in (transpose(operator), adjoint(operator))
                @test isapprox(mul!(zeros(TO, n), wrapped, y), expected;
                               atol=tolerance, rtol=tolerance)
                @test isapprox(mul!(copy(prior), wrapped, y, α, β),
                               α .* expected .+ β .* prior;
                               atol=tolerance, rtol=tolerance)
                @test isapprox(mul!(zeros(TO, length(cols)), wrapped, y, cols),
                               transpose(reference[:, cols]) * Float64.(y);
                               atol=tolerance, rtol=tolerance)
            end
        end
    end

    # 3. One-column mixed matrix products equal the mixed vector products.
    m, n = 70, 23
    packed, _ = simd_test_fixture(m, n)
    operator = SnpLinAlg{Float32}(packed; center=true, scale=true)
    cols = 3:17
    for (TO, TR) in combinations[2:end]
        sentinel_o, sentinel_r = TO(19), TR(19)
        x = simd_test_rhs(TR, n, 1)
        y = simd_test_rhs(TR, m, 1)
        prior_m = simd_test_rhs(TO, m, 1)
        prior_n = simd_test_rhs(TO, n, 1)
        forward = mul!(zeros(TO, m), operator, vec(x))
        forward5 = mul!(vec(copy(prior_m)), operator, vec(x), α, β)
        for jo in 0:2, jr in 0:2
            rhs, _ = simd_test_column(x, jr, sentinel_r)
            out, buffer = simd_test_column(zeros(TO, m, 1), jo, sentinel_o)
            @test SnpArrays._use_vector_kernels(out, Float32, rhs)
            method = which(mul!, Tuple{
                typeof(SnpArrays._column_vector(out)), SnpLinAlg{Float32},
                typeof(SnpArrays._column_vector(rhs)), Float64, Float64,
            })
            @test Base.unwrap_unionall(method.sig).parameters[3] ===
                  SnpLinAlg{Float32}
            @test mul!(out, operator, rhs) === out
            @test isequal(vec(out), forward)
            @test simd_test_others_untouched(buffer, jo, sentinel_o)
            out, buffer = simd_test_column(prior_m, jo, sentinel_o)
            mul!(out, operator, rhs, α, β)
            @test isequal(vec(out), forward5)
            @test simd_test_others_untouched(buffer, jo, sentinel_o)

            rhs, _ = simd_test_column(y, jr, sentinel_r)
            for wrapped in (transpose(operator), adjoint(operator))
                out, buffer = simd_test_column(zeros(TO, n, 1), jo, sentinel_o)
                @test mul!(out, wrapped, rhs) === out
                @test isequal(vec(out), mul!(zeros(TO, n), wrapped, vec(y)))
                @test simd_test_others_untouched(buffer, jo, sentinel_o)
                out, buffer = simd_test_column(prior_n, jo, sentinel_o)
                mul!(out, wrapped, rhs, α, β)
                @test isequal(vec(out),
                              mul!(vec(copy(prior_n)), wrapped, vec(y), α, β))
                @test simd_test_others_untouched(buffer, jo, sentinel_o)
                out, buffer = simd_test_column(zeros(TO, length(cols), 1), jo,
                                               sentinel_o)
                mul!(out, wrapped, rhs, cols)
                @test isequal(vec(out), mul!(zeros(TO, length(cols)), wrapped,
                                             vec(y), cols))
                @test simd_test_others_untouched(buffer, jo, sentinel_o)
            end
        end
    end

    # 4. Other mixed vector calls keep the generic LinearAlgebra fallback
    #    they used before the mixed methods existed.
    packed, dense = simd_test_fixture(37, 23)
    reference = simd_test_reference(dense, Float64, ADDITIVE_MODEL, false,
                                    false, true)
    operator = SnpLinAlg{Float64}(packed; impute=true)
    x = vec(simd_test_rhs(Float64, 23, 1))
    y = vec(simd_test_rhs(Float64, 37, 1))
    @test isapprox(mul!(zeros(Float32, 37), operator, x), reference * x;
                   atol=tolerance, rtol=tolerance)
    @test isapprox(mul!(zeros(Float32, 23), transpose(operator), y),
                   transpose(reference) * y; atol=tolerance, rtol=tolerance)
end
