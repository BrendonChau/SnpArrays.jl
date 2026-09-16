import SnpArrays
import BenchmarkTools

using BenchmarkTools: @benchmark
using LinearAlgebra: BLAS, mul!, transpose
using Random: Xoshiro, rand!
using SHA: sha256
using SnpArrays: SnpArray, SnpLinAlg
using Statistics: median, quantile

const RELEASE_COMMIT = "007ab2970ecd3144542dcc0666a200d4977eae29"
const FIXTURE_SEED = UInt64(20260911)
const SYNTHETIC_ROWS = 2047
const SYNTHETIC_COLUMNS = 4097

function source_root()::String
    return dirname(dirname(pathof(SnpArrays)))
end

function source_commit()::String
    return readchomp(`git -C $(source_root()) rev-parse HEAD`)
end

function hash_hex(bytes)::String
    return bytes2hex(sha256(bytes))
end

function mix_bits(value::UInt64)::UInt64
    value = (value ⊻ (value >> 30)) * 0xbf58476d1ce4e5b9
    value = (value ⊻ (value >> 27)) * 0x94d049bb133111eb
    return value ⊻ (value >> 31)
end

function synthetic_path()::String
    root = get(ENV, "SNP_BENCH_FIXTURES", joinpath(@__DIR__, "fixtures"))
    return joinpath(root, "synthetic_2047x4097.u8")
end

function synthetic_fixture()::SnpArray
    rows = SYNTHETIC_ROWS
    columns = SYNTHETIC_COLUMNS
    data_rows = cld(rows, 4)
    path = synthetic_path()
    expected = data_rows * columns
    if isfile(path)
        packed = read(path)
        length(packed) == expected ||
            throw(ArgumentError("bad synthetic fixture"))
    else
        mkpath(dirname(path))
        packed = Matrix{UInt8}(undef, data_rows, columns)
        rand!(Xoshiro(FIXTURE_SEED), packed)
        for column in axes(packed, 2)
            packed[end, column] &= 0x03
        end
        write(path, packed)
        packed = vec(packed)
    end
    s = SnpArray(undef, rows, columns)
    copyto!(vec(s.data), packed)
    return s
end

function fixture(name::String)::SnpArray
    name == "synthetic" || throw(ArgumentError("bad dataset"))
    return synthetic_fixture()
end

function selected_datasets()::Vector{String}
    datasets = split(get(ENV, "SNP_BENCH_DATASETS", "synthetic"), ',')
    all(==("synthetic"), datasets) ||
        throw(ArgumentError("bad benchmark dataset"))
    return datasets
end

function selected_widths()::Vector{Int}
    widths = parse.(Int, split(get(ENV, "SNP_BENCH_WIDTHS",
                                   "8,17,128,256"), ','))
    all(width -> width in (8, 17, 128, 256), widths) ||
        throw(ArgumentError("bad benchmark width"))
    return widths
end

function selected_types()::Vector{DataType}
    names = split(get(ENV, "SNP_BENCH_TYPES", "Float32,Float64"), ',')
    all(name -> name in ("Float32", "Float64"), names) ||
        throw(ArgumentError("bad benchmark element type"))
    return [name == "Float32" ? Float32 : Float64 for name in names]
end

function selected_orientations()::Vector{String}
    orientations = split(get(ENV, "SNP_BENCH_ORIENTATIONS", "AX,AtX"), ',')
    all(name -> name in ("AX", "AtX"), orientations) ||
        throw(ArgumentError("bad benchmark orientation"))
    return orientations
end

function right_hand_side(::Type{T}, rows::Int, width::Int) where T
    input = zeros(T, rows, width)
    for column in axes(input, 2), row in axes(input, 1)
        key = UInt64(row) + UInt64(column) * 0xd6e8feb86659fd93
        input[row, column] = T(Int(mix_bits(key) % 2049) - 1024) / T(512)
    end
    return input
end

function clean(value)::String
    return replace(string(value), ',' => ';', '\n' => ' ')
end

function emit(parts...)::Nothing
    println(join(clean.(parts), ','))
    flush(stdout)
    return nothing
end

function errors(out::AbstractMatrix{T}, expected::AbstractMatrix) where T
    absolute = maximum(abs, out - expected)
    relative = absolute / max(maximum(abs, expected), eps(T))
    valid = isapprox(out, expected; atol=64eps(T), rtol=64eps(T))
    return absolute, relative, valid
end

function bigfloat_spots(operator, input::AbstractMatrix)
    rows = unique((1, cld(size(operator, 1), 2), size(operator, 1)))
    columns = unique((1, size(input, 2)))
    spots = Vector{Tuple{CartesianIndex{2}, BigFloat}}()
    setprecision(BigFloat, 256) do
        for column in columns, row in rows
            total = BigFloat(0)
            for reduction in axes(input, 1)
                total += BigFloat(operator[row, reduction]) *
                         BigFloat(input[reduction, column])
            end
            push!(spots, (CartesianIndex(row, column), total))
        end
    end
    return spots
end

function bigfloat_error(out::AbstractMatrix, spots)::Float64
    return setprecision(BigFloat, 256) do
        Float64(maximum(abs(BigFloat(out[index]) - value)
                        for (index, value) in spots))
    end
end

function measure!(backend::Symbol, out::AbstractMatrix, operator,
                  input::AbstractMatrix, round::Int,
                  dataset::String, orientation::String, samples::Int,
                  mode::String, expected::AbstractMatrix,
                  spots, timed!, checked!)::Nothing
    checked!()
    all(isfinite, out) || throw(ArgumentError("nonfinite product"))
    absolute, relative, valid = errors(out, expected)
    spot = bigfloat_error(out, spots)
    timed!()
    timed!()
    trial = @benchmark $timed!() evals=1 samples=samples seconds=3600
    length(trial) == samples || throw(ArgumentError("incomplete trial"))
    fields = (round, dataset, eltype(out), orientation, size(input, 2),
              backend, mode, samples, median(trial.times),
              quantile(trial.times, 0.1), quantile(trial.times, 0.9),
              trial.memory, trial.allocs, absolute, relative, spot, valid)
    emit("RESULT", fields...)
    for (sample, elapsed) in enumerate(trial.times)
        emit("SAMPLE", round, dataset, eltype(out), orientation,
             size(input, 2), backend, mode, sample, elapsed)
    end
    return nothing
end

function benchmark_release!(s::SnpArray, dataset::String, ::Type{T},
                            widths::Vector{Int}, round::Int,
                            samples::Int, orientations::Vector{String},
                            mode::String) where T
    a = SnpLinAlg{T}(s; center=true, scale=true, impute=true)
    emit("STAT", mode, dataset, T, hash_hex(reinterpret(UInt8, a.μ)),
         hash_hex(reinterpret(UInt8, a.σinv)))
    for orientation in orientations
        operator = orientation == "AX" ? a : transpose(a)
        dense_operator = Float64.(Matrix(operator))
        for width in widths
            input = right_hand_side(T, size(operator, 2), width)
            emit("INPUT", dataset, T, orientation, width,
                 hash_hex(reinterpret(UInt8, vec(input))))
            out = zeros(T, size(operator, 1), width)
            checked! = () -> mul!(out, operator, input)
            expected = dense_operator * Float64.(input)
            spots = bigfloat_spots(operator, input)
            measure!(Symbol(mode), out, operator, input, round, dataset,
                     orientation, samples, mode, expected, spots,
                     checked!, checked!)
        end
    end
    return nothing
end

function benchmark_suite(mode::String, round::Int, samples::Int)::Nothing
    Threads.nthreads() == 1 || throw(ArgumentError("use JULIA_NUM_THREADS=1"))
    BLAS.set_num_threads(1)
    BLAS.get_num_threads() == 1 || throw(ArgumentError("use one BLAS thread"))
    if mode == "release"
        source_commit() == RELEASE_COMMIT ||
            throw(ArgumentError("wrong release source commit"))
    end
    datasets = selected_datasets()
    widths = selected_widths()
    types = selected_types()
    orientations = selected_orientations()
    emit("META", "mode", mode)
    emit("META", "round", round)
    emit("META", "round_order",
         get(ENV, "SNP_BENCH_ENVIRONMENT_ORDER", "unknown"))
    emit("META", "commit", source_commit())
    emit("META", "julia", VERSION)
    emit("META", "cpu", Sys.CPU_NAME)
    emit("META", "blas", BLAS.get_config())
    emit("META", "source", pathof(SnpArrays))
    emit("META", "snparrays_version", Base.pkgversion(SnpArrays))
    emit("META", "benchmarktools_version", Base.pkgversion(BenchmarkTools))
    emit("META", "module_sha256", hash_hex(read(pathof(SnpArrays))))
    emit("META", "linalg_direct_sha256",
         hash_hex(read(joinpath(source_root(), "src", "linalg_direct.jl"))))
    for dataset in datasets
        s = fixture(dataset)
        emit("DATA", dataset, size(s, 1), size(s, 2), hash_hex(vec(s.data)))
        for T in types
            benchmark_release!(s, dataset, T, widths, round, samples,
                               orientations, mode)
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 3 || throw(ArgumentError("expected mode round samples"))
    mode = ARGS[1]
    mode in ("current", "release") || throw(ArgumentError("bad mode"))
    benchmark_suite(mode, parse(Int, ARGS[2]), parse(Int, ARGS[3]))
end
