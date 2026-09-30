using DelimitedFiles

"""
    unpack_words(words, m)

Inverse of `SnpArrays._packed_words`: the m x n matrix of 2-bit codes.
"""
function unpack_words(words::AbstractMatrix{UInt32}, m::Integer)
    codes = Matrix{UInt8}(undef, m, size(words, 2))
    for j in axes(codes, 2), i in 1:m
        word = words[cld(i, 16), j]
        codes[i, j] = UInt8((word >> (2 * mod(i - 1, 16))) & 0x03)
    end
    return codes
end

@testset "cuda packed words (host)" begin
    golden = joinpath(@__DIR__, "data", "cuda_golden", "codes.txt")
    fixture_codes = UInt8.(readdlm(golden, Int))
    for m in (1, 3, 4, 15, 16, 17, 37, 64)
        codes = m == 37 ? fixture_codes :
            rand(Random.Xoshiro(m), UInt8(0):UInt8(3), m, 5)
        s = SnpArray(undef, size(codes)...)
        for j in axes(codes, 2), i in axes(codes, 1)
            s[i, j] = codes[i, j]
        end
        # Garbage in the unused bits of the last byte must be masked.
        if mod(m, 4) != 0
            s.data[end, :] .|= ~UInt8((1 << (2 * mod(m, 4))) - 1)
        end
        words = SnpArrays._packed_words(s.data, m)
        @test size(words) == (cld(m, 16), size(codes, 2))
        @test unpack_words(words, m) == codes
        used = mod(m, 16)
        if used != 0
            pad_mask = ~UInt32((UInt64(1) << (2 * used)) - 1)
            @test all(iszero, words[end, :] .& pad_mask)
        end
    end
    @test_throws DimensionMismatch SnpArrays._packed_words(
        zeros(UInt8, 3, 2), 16)
end
