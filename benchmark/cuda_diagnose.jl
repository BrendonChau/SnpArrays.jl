# Break the GPU product times of a PLINK bed file into their parts: what
# bounds a streamed sweep, the k-independent cost of each kernel and its
# growth with m and n, tile constants against the kernel body, and the
# accuracy of the Float32 paths against Float64.
#
#     julia -t N --project=<env with CUDA and SnpArrays> \
#       benchmark/cuda_diagnose.jl <bed> [results_dir] [sections] \
#       [m_cap] [n_cap]
#
# `sections` is a comma list from `stream,scale,tiles,accuracy` (the
# default, and the run order) and the opt-in `shapes`, run last. `m_cap`
# (a multiple of 16) and `n_cap` keep the first samples and SNPs; a capped
# copy of the files is then written to a temporary directory for the
# stream. Centered, scaled, mean-imputed. The report is also written, one
# flushed row at a time, to a timestamped markdown file in `results_dir`
# (default `results/` next to this script).
using CUDA
using Dates
using LinearAlgebra
using Printf
using Random
using SnpArrays

length(ARGS) >= 1 || throw(ArgumentError(
    "usage: cuda_diagnose.jl <bed> [results_dir] [sections] [m_cap] [n_cap]"))
const BED = ARGS[1]
const RESULTS_DIR = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results")
const ALL_SECTIONS = ("stream", "scale", "tiles", "accuracy")
const SECTIONS = length(ARGS) >= 3 ? String.(split(ARGS[3], ",")) :
    collect(String, ALL_SECTIONS)
const M_CAP = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : typemax(Int)
const N_CAP = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : typemax(Int)
const K = 32
const WIDTH = 4096
const REPEATS = 3
const SCALE_KS = (2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)
const TILE_SHAPES = ((256, 16, 32, 8, 8), (128, 16, 64, 8, 8),
    (256, 16, 64, 8, 8), (256, 32, 32, 8, 8), (128, 32, 32, 8, 8))
const EXT = Base.get_extension(SnpArrays, :SnpArraysCUDAExt)
"""Largest device memory in use seen by `note_memory!`, in bytes."""
const PEAK_DEVICE = Ref(0)
# The tensor-core kernels are opt-in; this benchmark measures them.
SnpArrays.cuda_tensor_cores!(true)

"""
    emit(io, line)

Print `line` to `stdout` and to `io`, then flush both.
"""
function emit(io::IO, line::AbstractString)
    println(line)
    println(io, line)
    flush(io)
    flush(stdout)
    return nothing
end

"""
    gpu_name() -> String

Return the active device's name with runs of other characters as `-`.
"""
gpu_name() = replace(CUDA.name(CUDA.device()), r"[^A-Za-z0-9]+" => "-")

"""
    report_path(dir) -> String

Create `dir` and return the timestamped report path in it; an existing file
at that path is an error.
"""
function report_path(dir::AbstractString)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    path = joinpath(dir, "cuda_diagnose_$(gpu_name())_$(stamp).md")
    ispath(path) && error("$(path) exists; wait a second or remove it")
    mkpath(dir)
    return path
end

"""
    print_fact(io, name, value)

Print one row of the `| Fact | Value |` table.
"""
function print_fact(io::IO, name::AbstractString, value)
    emit(io, "| $(name) | `$(value)` |")
    return nothing
end

"""
    print_header(io, path, G)

Print the environment facts as a markdown table.
"""
function print_header(io::IO, path::AbstractString, G::SnpArray)
    m, n = size(G)
    device = CUDA.device()
    emit(io, "| Fact | Value |")
    emit(io, "| --- | --- |")
    print_fact(io, "GPU", CUDA.name(device))
    print_fact(io, "GPU memory", Base.format_bytes(CUDA.totalmem(device)))
    print_fact(io, "compute capability", CUDA.capability(device))
    print_fact(io, "CUDA.jl", pkgversion(CUDA))
    print_fact(io, "CUDA runtime", CUDA.runtime_version())
    print_fact(io, "threads", Threads.nthreads())
    print_fact(io, "Sys.CPU_NAME", Sys.CPU_NAME)
    print_fact(io, "VERSION", VERSION)
    print_fact(io, "pkgdir", pkgdir(SnpArrays))
    print_fact(io, "bed", BED)
    print_fact(io, "bed used", path)
    print_fact(io, "bed bytes", filesize(path))
    print_fact(io, "m, n", "m = $(m), n = $(n)")
    print_fact(io, "sections", join(SECTIONS, ","))
    return nothing
end

"""
    note_memory!()

Raise `PEAK_DEVICE` to the device memory now in use, pool cache included.
"""
function note_memory!()
    used = CUDA.total_memory() - CUDA.free_memory()
    PEAK_DEVICE[] = max(PEAK_DEVICE[], Int(used))
    return nothing
end

"""
    time_call(f!, args...; repeats = REPEATS) -> (minimum, median)

Time `f!(args...)` on the device in seconds, after one warm-up call.
"""
function time_call(f!::F, args...; repeats::Int = REPEATS) where F
    f!(args...)
    CUDA.synchronize()
    times = [Float64(CUDA.@elapsed f!(args...)) for _ in 1:repeats]
    note_memory!()
    sort!(times)
    return times[1], times[(repeats + 1) ÷ 2]
end

"""
    release!()

Collect garbage and return the pool's cached device memory to the driver.
"""
function release!()
    GC.gc()
    CUDA.reclaim()
    return nothing
end

"""
    free_operator!(A)

Free the device arrays of `A`.
"""
function free_operator!(A::CuSnpArray)
    for array in (A.data, A.μ, A.σinv, A.values)
        CUDA.unsafe_free!(array)
    end
    return nothing
end

"""
    free_tables!(A)

Free the device arrays of `A` except its packed words, which it shares.
"""
function free_tables!(A::CuSnpArray)
    for array in (A.μ, A.σinv, A.values)
        CUDA.unsafe_free!(array)
    end
    return nothing
end

"""
    operator(::Type{T}, G) -> CuSnpArray{T}

Upload `G` centered, scaled and mean-imputed.
"""
function operator(::Type{T}, G::SnpArray) where T <: AbstractFloat
    A = CuSnpArray{T}(G; center = true, scale = true, impute = true)
    note_memory!()
    return A
end

"""
    make_stream(::Type{T}, path) -> SnpLinAlgStream{T}

Open `path` as a centered, scaled, imputed stream of `WIDTH`-SNP chunks.
"""
function make_stream(::Type{T}, path::AbstractString) where T <: AbstractFloat
    return SnpLinAlgStream{T}(path; width = WIDTH, center = true,
                              scale = true, impute = true)
end

"""
    capped_genotypes(io) -> (path, G)

Open `BED`, or, under a cap smaller than its size, write its first `M_CAP`
samples and `N_CAP` SNPs as a new bed, bim and fam in a temporary directory
and open that.
"""
function capped_genotypes(io::IO)
    full = SnpArray(BED)
    m, n = size(full)
    mc = min(M_CAP, m)
    nc = min(N_CAP, n)
    (mc, nc) == (m, n) && return BED, full
    mc % 16 == 0 || throw(ArgumentError(
        "m_cap must be a multiple of 16 below m = $(m); got $(mc)"))
    nc >= 1 || throw(ArgumentError("n_cap must be at least 1; got $(nc)"))
    path = joinpath(mktempdir(), "capped.bed")
    G = SnpArray(undef, mc, nc)
    G.data .= view(full.data, 1:(mc ÷ 4), 1:nc)
    fam = open(splitext(path)[1] * ".fam", "w")
    for i in 1:mc
        println(fam, "$(i)\t$(i)\t0\t0\t0\t-9")
    end
    close(fam)
    bim = open(splitext(path)[1] * ".bim", "w")
    for j in 1:nc
        println(bim, "1\tsnp$(j)\t0\t$(j)\tA\tG")
    end
    close(bim)
    emit(io, "Capped to m = $(mc), n = $(nc) in `$(path)`.")
    return path, SnpArray(path, G)
end

# ---------------------------------------------------------------- stream

"""
    read_sweep(stream) -> nothing

Iterate `stream` on the host without using the chunks.
"""
function read_sweep(stream::SnpLinAlgStream)
    for (cols, chunk) in stream
    end
    return nothing
end

"""
    do_nothing(cols, s) -> nothing

Chunk function of `upload_sweep`.
"""
do_nothing(cols::UnitRange{Int}, s::CuSnpArray) = nothing

"""
    upload_sweep(stream) -> nothing

Read and upload every chunk of `stream` through `_foreach_chunk`.
"""
function upload_sweep(stream::SnpLinAlgStream)
    EXT._foreach_chunk(do_nothing, stream)
    return nothing
end

"""
    first_and_second(f, stream) -> (Float64, Float64)

Wall seconds of two consecutive calls `f(stream)`, synchronized.
"""
function first_and_second(f::F, stream::SnpLinAlgStream) where F
    times = zeros(2)
    for i in 1:2
        CUDA.synchronize()
        start = time_ns()
        f(stream)
        CUDA.synchronize()
        times[i] = (time_ns() - start) / 1e9
    end
    note_memory!()
    return times[1], times[2]
end

"""
    first_and_second(f!, out, stream, rhs) -> (Float64, Float64)

Wall seconds of two consecutive calls `f!(out, stream, rhs)`, synchronized.
"""
function first_and_second(f!::F, out::CuMatrix, stream::SnpLinAlgStream,
                          rhs::CuMatrix) where F
    times = zeros(2)
    for i in 1:2
        CUDA.synchronize()
        start = time_ns()
        f!(out, stream, rhs)
        CUDA.synchronize()
        times[i] = (time_ns() - start) / 1e9
    end
    note_memory!()
    return times[1], times[2]
end

"""
    streamed_ax!(out, stream, X) -> out

Set `out = A * X` in one streamed sweep.
"""
function streamed_ax!(out::CuMatrix{T}, stream::SnpLinAlgStream{T},
                      X::CuMatrix{T}) where T <: AbstractFloat
    return streamed_mul!(out, stream, X; transpose = false)
end

"""
    streamed_atx!(out, stream, Y) -> out

Set `out = Aᵀ * Y` in one streamed sweep.
"""
function streamed_atx!(out::CuMatrix{T}, stream::SnpLinAlgStream{T},
                       Y::CuMatrix{T}) where T <: AbstractFloat
    return streamed_mul!(out, stream, Y; transpose = true)
end

"""
    streamed_grm!(V, stream, Q) -> V

Set `V = A * (Aᵀ * Q) / n` in one streamed sweep.
"""
function streamed_grm!(V::CuMatrix, stream::SnpLinAlgStream,
                       Q::CuMatrix)
    return streamed_grm_mul!(V, stream, Q)
end

"""
    grm_in_memory!(V, U, A, Q) -> V

Set `U = Aᵀ * Q / n` and `V = A * U` on the in-memory operator.
"""
function grm_in_memory!(V::CuMatrix{T}, U::CuMatrix{T}, A::CuSnpArray{T},
                        Q::CuMatrix{T}) where T <: AbstractFloat
    mul!(U, transpose(A), Q)
    U .*= inv(T(size(A, 2)))
    mul!(V, A, U)
    return V
end

"""
    split_once!(X, values, rows_padded) -> nothing

Run `_split_rhs` on `X` padded to `rows_padded` rows and `_split_tables` on
`values`, as the tensor-core products do per call, and free the results.
"""
function split_once!(X::CuMatrix{Float32}, values::CuMatrix{Float32},
                     rows_padded::Int)
    k = size(X, 2)
    BK = EXT._wmma_rhs_width(k)
    high, low, scale_inv = EXT._split_rhs(X, rows_padded, BK * cld(k, BK))
    tables = EXT._split_tables(values)
    for array in (high, low, scale_inv, tables)
        CUDA.unsafe_free!(array)
    end
    return nothing
end

"""
    stream_row(io, part, label, tensor, times, bytes)

Print one row of the stream table; GB/s is from the second time.
"""
function stream_row(io::IO, part::AbstractString, label::AbstractString,
                    tensor::AbstractString, times::NTuple{2, Float64},
                    bytes::Int)
    emit(io, @sprintf("| %s | %s | %s | %.3f | %.3f | %.2f |", part, label,
                      tensor, times[1], times[2], bytes / times[2] / 1e9))
    return nothing
end

"""
    run_stream(io, path, G)

Print what bounds a streamed `Float32` sweep at `k = K`: host read, read
plus upload, the per-chunk splits, the streamed products, and the same
kernels in memory.
"""
function run_stream(io::IO, path::AbstractString, G::SnpArray)
    bytes = Int(filesize(path))
    stream = make_stream(Float32, path)
    m, n = size(stream)
    chunks = length(stream)
    emit(io, "")
    emit(io, "## stream: what bounds a streamed sweep?")
    emit(io, "")
    emit(io, "Float32, k = $(K), width = $(WIDTH), $(chunks) chunks. " *
             "First and second sweep in wall seconds; GB/s is bed bytes " *
             "over the second.")
    emit(io, "")
    emit(io, "| Part | Product | Tensor | first s | second s | bed GB/s |")
    emit(io, "|---|---|---|---:|---:|---:|")
    stream_row(io, "(a) host read", "-", "-",
               first_and_second(read_sweep, stream), bytes)
    upload = first_and_second(upload_sweep, stream)
    stream_row(io, "(b) read + upload", "-", "-", upload, bytes)

    Xc = CuArray(randn(Xoshiro(1), Float32, WIDTH, K))
    Yc = CuArray(randn(Xoshiro(2), Float32, m, K))
    values = CuArray(randn(Xoshiro(3), Float32, 4, WIDTH))
    ax_split = time_call(split_once!, Xc, values,
                         EXT.WMMA_BN * cld(WIDTH, EXT.WMMA_BN))
    atx_split = time_call(split_once!, Yc, values,
                          EXT.WMMA_T_BS * cld(m, EXT.WMMA_T_BS))
    for array in (Xc, Yc, values)
        CUDA.unsafe_free!(array)
    end
    emit(io, "")
    emit(io, "(c) Splits per chunk (min of $(REPEATS)), times $(chunks) " *
             "chunks.")
    emit(io, "")
    emit(io, "| Split | per chunk ms | per sweep s |")
    emit(io, "|---|---:|---:|")
    emit(io, @sprintf("| `A*X` rhs (%d rows) + tables | %.3f | %.3f |",
                      WIDTH, 1000 * ax_split[1], chunks * ax_split[1]))
    emit(io, @sprintf("| `Aᵀ*X` rhs (%d rows) + tables | %.3f | %.3f |",
                      m, 1000 * atx_split[1], chunks * atx_split[1]))

    X = CuArray(randn(Xoshiro(4), Float32, n, K))
    Y = CuArray(randn(Xoshiro(5), Float32, m, K))
    Y64 = CuArray(randn(Xoshiro(5), Float64, m, K))
    AX = CuMatrix{Float32}(undef, m, K)
    AtX = CuMatrix{Float32}(undef, n, K)
    V = CuMatrix{Float32}(undef, m, K)
    V64 = CuMatrix{Float64}(undef, m, K)
    names = ("A*X", "Aᵀ*X", "A*(Aᵀ*X)/n", "A*(Aᵀ*X)/n mixed")
    streamed = zeros(length(names), 2)
    emit(io, "")
    emit(io, "(d) Streamed products.")
    emit(io, "")
    emit(io, "| Part | Product | Tensor | first s | second s | bed GB/s |")
    emit(io, "|---|---|---|---:|---:|---:|")
    for (column, tensor) in enumerate((true, false))
        SnpArrays.cuda_tensor_cores!(tensor)
        label = tensor ? "on" : "off"
        times = (first_and_second(streamed_ax!, AX, stream, X),
                 first_and_second(streamed_atx!, AtX, stream, Y),
                 first_and_second(streamed_grm!, V, stream, Y),
                 first_and_second(streamed_grm!, V64, stream, Y64))
        for (row, name) in enumerate(names)
            stream_row(io, "(d) streamed", name, label, times[row], bytes)
            streamed[row, column] = times[row][2]
        end
    end
    CUDA.unsafe_free!(V64)
    CUDA.unsafe_free!(Y64)
    release!()

    A = operator(Float32, G)
    U = CuMatrix{Float32}(undef, n, K)
    emit(io, "")
    emit(io, "(e) The same kernels in memory (min of $(REPEATS)), and the " *
             "remainder (d) second − (b) second − (e). The mixed row's (e) " *
             "is the Float32 in-memory product.")
    emit(io, "")
    emit(io, "| Product | Tensor | (e) min s | (e) median s | remainder s |")
    emit(io, "|---|---|---:|---:|---:|")
    for (column, tensor) in enumerate((true, false))
        SnpArrays.cuda_tensor_cores!(tensor)
        label = tensor ? "on" : "off"
        grm = time_call(grm_in_memory!, V, U, A, Y)
        times = (time_call(mul!, AX, A, X),
                 time_call(mul!, AtX, transpose(A), Y), grm, grm)
        for (row, name) in enumerate(names)
            emit(io, @sprintf("| %s | %s | %.3f | %.3f | %.3f |", name, label,
                              times[row][1], times[row][2],
                              streamed[row, column] - upload[2] -
                              times[row][1]))
        end
    end
    SnpArrays.cuda_tensor_cores!(true)
    for array in (X, Y, AX, AtX, V, U)
        CUDA.unsafe_free!(array)
    end
    free_operator!(A)
    release!()
    return nothing
end

# ----------------------------------------------------------------- scale

"""
    tiled_ax!(out, A, X) -> out

Set `out = A * X` with the tiled kernel.
"""
function tiled_ax!(out::CuMatrix{Float32}, A::CuSnpArray{Float32},
                   X::CuMatrix{Float32})
    return EXT._tiled_mul!(out, A, X)
end

"""
    wmma_ax!(out, A, X) -> out

Set `out = A * X` with the tensor-core kernel.
"""
function wmma_ax!(out::CuMatrix{Float32}, A::CuSnpArray{Float32},
                  X::CuMatrix{Float32})
    return EXT._wmma_mul!(out, A, X)
end

"""
    tiled_atx!(out, A, Y) -> out

Set `out = Aᵀ * Y` through `mul!`; tiled while tensor cores are off.
"""
function tiled_atx!(out::CuMatrix{Float32}, A::CuSnpArray{Float32},
                    Y::CuMatrix{Float32})
    return mul!(out, transpose(A), Y)
end

"""
    wmma_atx!(out, A, Y) -> out

Set `out = Aᵀ * Y` with the tensor-core kernel.
"""
function wmma_atx!(out::CuMatrix{Float32}, A::CuSnpArray{Float32},
                   Y::CuMatrix{Float32})
    return EXT._wmma_t_mul!(out, A, Y)
end

"""The four `Float32` variants: name, function, transposed, tensor-core."""
const VARIANTS = (
    (name = "tiled A*X", f! = tiled_ax!, transposed = false, tensor = false),
    (name = "tensor A*X", f! = wmma_ax!, transposed = false, tensor = true),
    (name = "tiled Aᵀ*X", f! = tiled_atx!, transposed = true, tensor = false),
    (name = "tensor Aᵀ*X", f! = wmma_atx!, transposed = true, tensor = true),
)

"""
    wmma_supported() -> Bool

Return whether the device has compute capability 8.0, which the
tensor-core kernels' `cp.async` requires.
"""
wmma_supported() = CUDA.capability(CUDA.device()) >= v"8.0"

"""
    time_variant(f!, A, X, Y, transposed, k) -> (minimum, median)

Time `f!` on the first `k` columns of `X` (or of `Y` when `transposed`).
"""
function time_variant(f!::F, A::CuSnpArray{Float32}, X::CuMatrix{Float32},
                      Y::CuMatrix{Float32}, transposed::Bool,
                      k::Int) where F
    m, n = size(A)
    rhs = transposed ? Y[1:m, 1:k] : X[1:n, 1:k]
    out = CuMatrix{Float32}(undef, transposed ? n : m, k)
    times = time_call(f!, out, A, rhs)
    CUDA.unsafe_free!(rhs)
    CUDA.unsafe_free!(out)
    return times
end

"""
    fit_line(ks, times) -> (a, b)

Least-squares `time = a + b * k` over the finite `times`.
"""
function fit_line(ks::Vector{Float64}, times::Vector{Float64})
    keep = isfinite.(times)
    x = ks[keep]
    y = times[keep]
    length(x) >= 2 || return NaN, NaN
    x̄ = sum(x) / length(x)
    ȳ = sum(y) / length(y)
    b = sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
    return ȳ - b * x̄, b
end

"""
    upload_sub_words(G, m2, n2) -> CuMatrix{UInt32}

Packed words of the first `m2` samples and `n2` SNPs of `G`, uploaded with
`_upload_packed_words` in column blocks of about 256 MiB of host bytes.
"""
function upload_sub_words(G::SnpArray, m2::Int, n2::Int)
    rows = cld(m2, 4)
    word_rows = cld(m2, 16)
    words = CuMatrix{UInt32}(undef, word_rows, n2)
    step = max(1, 2^28 ÷ rows)
    for first_column in 1:step:n2
        cols = first_column:min(first_column + step - 1, n2)
        piece = EXT._upload_packed_words(G.data[1:rows, cols], m2)
        copyto!(words, (first_column - 1) * word_rows + 1, piece, 1,
                length(piece))
        CUDA.unsafe_free!(piece)
    end
    return words
end

"""
    sub_operator(G, A, m2, n2) -> CuSnpArray{Float32}

Operator on the first `m2` samples and `n2` SNPs with new packed words and
`A`'s value tables (copies of their first `n2` columns when `n2 < n`).
"""
function sub_operator(G::SnpArray, A::CuSnpArray{Float32}, m2::Int, n2::Int)
    words = upload_sub_words(G, m2, n2)
    full = n2 == size(A, 2)
    μ = full ? A.μ : A.μ[1:n2]
    σinv = full ? A.σinv : A.σinv[1:n2]
    values = full ? A.values : A.values[:, 1:n2]
    S = CuSnpArray{Float32}(words, m2, A.model, A.center, A.scale, A.impute,
                            μ, σinv, values)
    note_memory!()
    return S
end

"""
    free_sub!(S, A)

Free what `sub_operator` allocated for `S`.
"""
function free_sub!(S::CuSnpArray{Float32}, A::CuSnpArray{Float32})
    CUDA.unsafe_free!(S.data)
    S.values === A.values || free_tables!(S)
    return nothing
end

"""
    scale_cell(io, variant, A, X, Y, k) -> Float64

Print one row of a scale table and return its minimum time, `NaN` when the
variant is skipped.
"""
function scale_cell(io::IO, variant::NamedTuple, A::CuSnpArray{Float32},
                    X::CuMatrix{Float32}, Y::CuMatrix{Float32}, k::Int)
    m, n = size(A)
    if variant.tensor && !wmma_supported()
        emit(io, "| $(variant.name) | $(m) | $(n) | $(k) | skipped: " *
                 "compute capability below 8.0 | | |")
        return NaN
    end
    times = time_variant(variant.f!, A, X, Y, variant.transposed, k)
    fmas = Float64(m) * n * k
    emit(io, @sprintf("| %s | %d | %d | %d | %.3f | %.3f | %.2f |",
                      variant.name, m, n, k, 1000 * times[1],
                      1000 * times[2], fmas / times[1] / 1e9))
    return times[1]
end

"""
    run_scale(io, G)

Print the `k` sweep with its fitted lines, the growth with `m` and `n` at
`k = K`, and the tiled-to-tensor ratios.
"""
function run_scale(io::IO, G::SnpArray)
    SnpArrays.cuda_tensor_cores!(false)
    A = operator(Float32, G)
    m, n = size(A)
    kmax = maximum(SCALE_KS)
    X = CuArray(randn(Xoshiro(21), Float32, n, kmax))
    Y = CuArray(randn(Xoshiro(22), Float32, m, kmax))
    emit(io, "")
    emit(io, "## scale: k-independent cost, and growth with m")
    emit(io, "")
    emit(io, "Float32, min and median of $(REPEATS). The tensor kernels " *
             "are called directly at every k, below the dispatch " *
             "thresholds too.")
    emit(io, "")
    emit(io, "| Variant | m | n | k | min ms | median ms | GFMA/s |")
    emit(io, "|---|---:|---:|---:|---:|---:|---:|")
    times = fill(NaN, length(SCALE_KS), length(VARIANTS))
    for (j, variant) in enumerate(VARIANTS)
        for (i, k) in enumerate(SCALE_KS)
            times[i, j] = scale_cell(io, variant, A, X, Y, k)
        end
    end
    ks = Float64[k for k in SCALE_KS]
    i32 = findfirst(==(32), SCALE_KS)
    emit(io, "")
    emit(io, "Least-squares `time = a + b·k`, seconds.")
    emit(io, "")
    emit(io, "| Variant | k range | a | b | a / time(32) |")
    emit(io, "|---|---|---:|---:|---:|")
    for (j, variant) in enumerate(VARIANTS)
        for (low, high) in ((12, 32), (48, 128))
            band = (ks .>= low) .& (ks .<= high)
            a, b = fit_line(ks[band], times[band, j])
            emit(io, @sprintf("| %s | %d..%d | %.3e | %.3e | %.3f |",
                              variant.name, low, high, a, b,
                              a / times[i32, j]))
        end
    end
    emit(io, "")
    emit(io, "Tiled-to-tensor time ratio (above 1: tensor faster).")
    emit(io, "")
    emit(io, "| k | A*X | Aᵀ*X |")
    emit(io, "|---:|---:|---:|")
    for (i, k) in enumerate(SCALE_KS)
        emit(io, @sprintf("| %d | %.3f | %.3f |", k,
                          times[i, 1] / times[i, 2],
                          times[i, 3] / times[i, 4]))
    end
    emit(io, "")
    emit(io, "Growth at k = $(K); sub-operators share the value tables.")
    emit(io, "")
    emit(io, "| Variant | m | n | k | min ms | median ms | GFMA/s |")
    emit(io, "|---|---:|---:|---:|---:|---:|---:|")
    sizes = [(16 * fld(m, 64), n), (16 * fld(m, 32), n), (m, n),
             (m, fld(n, 4)), (m, fld(n, 2))]
    for (m2, n2) in sizes
        (m2 >= 16 && n2 >= 1) || continue
        S = (m2, n2) == (m, n) ? A : sub_operator(G, A, m2, n2)
        for variant in VARIANTS
            scale_cell(io, variant, S, X, Y, K)
        end
        S === A || free_sub!(S, A)
        release!()
    end
    CUDA.unsafe_free!(X)
    CUDA.unsafe_free!(Y)
    free_operator!(A)
    SnpArrays.cuda_tensor_cores!(true)
    release!()
    return nothing
end

# ----------------------------------------------------------------- tiles

"""
    shape_problem(::Type{T}, shape) -> Union{Nothing, String}

Why `_aX_tiled_kernel!` cannot run `shape = (BM, BN, BK, TM, TK)` for
element type `T`, or `nothing` when it can.
"""
function shape_problem(::Type{T}, shape::NTuple{5, Int}) where T
    (BM, BN, BK, TM, TK) = shape
    BM % 16 == 0 || return "BM not a multiple of 16"
    BM % TM == 0 || return "BM not a multiple of TM"
    BK % TK == 0 || return "BK not a multiple of TK"
    threads = (BM ÷ TM) * (BK ÷ TK)
    threads <= 1024 || return "$(threads) threads per block"
    shared = ((BM + 1) * BN + BN * BK) * sizeof(T)
    shared <= 48 * 1024 || return "$(shared) bytes of static shared memory"
    return nothing
end

"""
    tiled_mul_shape!(out, A, X, shape) -> out

Set `out = A * X` with `_tiled_mul!`'s launch at tile shape
`(BM, BN, BK, TM, TK)`.
"""
function tiled_mul_shape!(out::CuMatrix{T}, A::CuSnpArray{T}, X::CuMatrix{T},
                          shape::NTuple{5, Int}) where T <: AbstractFloat
    m, n = size(A)
    k = size(X, 2)
    (BM, BN, BK, TM, TK) = shape
    row_blocks = cld(m, BM)
    rhs_blocks = cld(k, BK)
    splits = EXT._splits(row_blocks * rhs_blocks, n, BN)
    split_columns = BN * cld(cld(n, splits), BN)
    splits = cld(n, split_columns)
    partials = splits == 1 ? reshape(out, m, k, 1) :
        CuArray{T, 3}(undef, m, k, splits)
    threads = (BM ÷ TM) * (BK ÷ TK)
    @cuda threads=threads blocks=(row_blocks, rhs_blocks, splits) (
        EXT._aX_tiled_kernel!(partials, A.data, X, A.values, Int32(m),
            Int32(n), Int32(k), Int32(split_columns),
            Val(BM), Val(BN), Val(BK), Val(TM), Val(TK))
    )
    splits == 1 || EXT._reduce_splits!(out, partials, splits)
    return out
end

"""
    run_tiles(io, G)

Print tiled `Float32` `A*X` at `k = 32` and `128` for each tile shape, with
its relative difference from `_tiled_mul!`.
"""
function run_tiles(io::IO, G::SnpArray)
    A = operator(Float32, G)
    m, n = size(A)
    emit(io, "")
    emit(io, "## tiles: tile constants or kernel body?")
    emit(io, "")
    emit(io, "Float32 `A*X`, min and median of $(REPEATS). Default shape " *
             "for k: `AX_TILES[_tile_band(k)]`.")
    emit(io, "")
    emit(io, "| k | shape (BM,BN,BK,TM,TK) | default | min ms | median ms " *
             "| GFMA/s | rel. diff |")
    emit(io, "|---:|---|---|---:|---:|---:|---:|")
    for k in (32, 128)
        X = CuArray(randn(Xoshiro(31), Float32, n, k))
        out = CuMatrix{Float32}(undef, m, k)
        expected = Array(EXT._tiled_mul!(out, A, X))
        default = EXT.AX_TILES[EXT._tile_band(k)]
        for shape in TILE_SHAPES
            mark = shape == default ? "yes" : ""
            problem = shape_problem(Float32, shape)
            if problem !== nothing
                emit(io, "| $(k) | `$(shape)` | $(mark) | skipped: " *
                         "$(problem) | | | |")
                continue
            end
            times = time_call(tiled_mul_shape!, out, A, X, shape)
            difference = norm(Array(out) .- expected) / norm(expected)
            emit(io, @sprintf("| %d | `%s` | %s | %.3f | %.3f | %.2f | %.1e |",
                              k, shape, mark, 1000 * times[1],
                              1000 * times[2],
                              Float64(m) * n * k / times[1] / 1e9,
                              difference))
        end
        CUDA.unsafe_free!(X)
        CUDA.unsafe_free!(out)
    end
    free_operator!(A)
    release!()
    return nothing
end

# -------------------------------------------------------------- accuracy

"""
    shared_float64(A, G) -> CuSnpArray{Float64}

`Float64` operator on `A`'s packed words with tables computed as
`CuSnpArray{Float64}(G; center, scale, impute)` does.
"""
function shared_float64(A::CuSnpArray{Float32}, G::SnpArray)
    n = size(G, 2)
    μ = vec(Float64.(SnpArrays.mean(G; dims = 1, model = A.model)))
    σinv = zeros(n)
    for j in 1:n
        σ = sqrt(μ[j] * (1 - μ[j] / 2))
        σinv[j] = σ > 0 ? inv(σ) : 1.0
    end
    values = EXT._lookup_values(μ, σinv, A.model, true, true, true)
    return CuSnpArray{Float64}(A.data, A.m, A.model, true, true, true,
                               CuArray(μ), CuArray(σinv), CuArray(values))
end

"""
    make_rhs(rng, rows, k, kind) -> Matrix{Float64}

`randn` right-hand sides, with columns (`:columns`) or entries
(`:entries`) scaled over `1e-6` to `1e6`, or unscaled (`:randn`); values
are rounded to `Float32`.
"""
function make_rhs(rng::AbstractRNG, rows::Int, k::Int, kind::Symbol)
    X = randn(rng, rows, k)
    if kind === :columns
        X .*= transpose(exp10.(range(-6.0, 6.0; length = k)))
    elseif kind === :entries
        X .*= exp10.(12.0 .* rand(rng, rows, k) .- 6.0)
    end
    return Float64.(Float32.(X))
end

"""
    worst_column_error(out, ref) -> Float64

Largest `‖out[:, j] - ref[:, j]‖ / ‖ref[:, j]‖` over columns `j`.
"""
function worst_column_error(out::Matrix{Float64}, ref::Matrix{Float64})
    worst = 0.0
    for j in axes(ref, 2)
        worst = max(worst, norm(view(out, :, j) .- view(ref, :, j)) /
                           norm(view(ref, :, j)))
    end
    return worst
end

"""
    worst_row_errors(out, ref, mafs) -> NTuple{3, Tuple{Int, Float64}}

SNP count and largest row relative error in the MAF bins below 1%, 1 to
5%, and above 5%, over rows with nonzero `ref`.
"""
function worst_row_errors(out::Matrix{Float64}, ref::Matrix{Float64},
                          mafs::Vector{Float64})
    counts = zeros(Int, 3)
    worst = zeros(3)
    for i in axes(ref, 1)
        scale = norm(view(ref, i, :))
        scale > 0 || continue
        bin = mafs[i] < 0.01 ? 1 : mafs[i] <= 0.05 ? 2 : 3
        counts[bin] += 1
        worst[bin] = max(worst[bin],
                         norm(view(out, i, :) .- view(ref, i, :)) / scale)
    end
    return ((counts[1], worst[1]), (counts[2], worst[2]),
            (counts[3], worst[3]))
end

"""
    float32_product(f!, A, rhs, rows) -> Matrix{Float64}

Run `f!(out, A, rhs)` into a new `rows x k` `Float32` output and return it
on the host as `Float64`.
"""
function float32_product(f!::F, A::CuSnpArray{Float32}, rhs::CuMatrix{Float32},
                         rows::Int) where F
    out = CuMatrix{Float32}(undef, rows, size(rhs, 2))
    f!(out, A, rhs)
    result = Float64.(Array(out))
    note_memory!()
    CUDA.unsafe_free!(out)
    return result
end

"""
    run_accuracy(io, path, G)

Print the relative errors of the `Float32` paths against the `Float64`
tiled product on shared packed words, and of the streamed mixed product
against the `Float64` streamed one.
"""
function run_accuracy(io::IO, path::AbstractString, G::SnpArray)
    SnpArrays.cuda_tensor_cores!(false)
    A = operator(Float32, G)
    A64 = shared_float64(A, G)
    m, n = size(A)
    mafs = maf(G)
    rng = Xoshiro(41)
    emit(io, "")
    emit(io, "## accuracy: Float32 paths vs Float64")
    emit(io, "")
    emit(io, "k = $(K). Reference: Float64 tiled product on the same " *
             "packed words, on the Float32-rounded right-hand sides. " *
             "Worst column of `‖out − ref‖ / ‖ref‖`.")
    emit(io, "")
    emit(io, "| Product | Path | RHS | rel. error |")
    emit(io, "|---|---|---|---:|")
    bins = ("MAF < 1%", "1% ≤ MAF ≤ 5%", "MAF > 5%")
    maf_rows = String[]
    for kind in (:randn, :columns, :entries)
        X = make_rhs(rng, n, K, kind)
        ref = Array(EXT._tiled_mul!(CuMatrix{Float64}(undef, m, K), A64,
                                    CuArray(X)))
        X32 = CuArray(Float32.(X))
        for variant in VARIANTS[1:2]
            if variant.tensor && !wmma_supported()
                emit(io, "| A*X | $(variant.name) | $(kind) | skipped |")
                continue
            end
            out = float32_product(variant.f!, A, X32, m)
            emit(io, @sprintf("| A*X | %s | %s | %.2e |", variant.name, kind,
                              worst_column_error(out, ref)))
        end
        CUDA.unsafe_free!(X32)
        Y = make_rhs(rng, m, K, kind)
        ref = Array(mul!(CuMatrix{Float64}(undef, n, K), transpose(A64),
                         CuArray(Y)))
        Y32 = CuArray(Float32.(Y))
        for variant in VARIANTS[3:4]
            if variant.tensor && !wmma_supported()
                emit(io, "| Aᵀ*X | $(variant.name) | $(kind) | skipped |")
                continue
            end
            out = float32_product(variant.f!, A, Y32, n)
            emit(io, @sprintf("| Aᵀ*X | %s | %s | %.2e |", variant.name,
                              kind, worst_column_error(out, ref)))
            for (bin, (count, worst)) in zip(bins,
                                             worst_row_errors(out, ref, mafs))
                push!(maf_rows, @sprintf("| %s | %s | %s | %d | %.2e |",
                                         variant.name, kind, bin, count,
                                         worst))
            end
        end
        CUDA.unsafe_free!(Y32)
        release!()
    end
    free_tables!(A64)
    free_operator!(A)
    SnpArrays.cuda_tensor_cores!(true)
    release!()

    stream32 = make_stream(Float32, path)
    stream64 = make_stream(Float64, path)
    Q = CuArray(make_rhs(rng, m, K, :randn))
    V = CuMatrix{Float64}(undef, m, K)
    ref = Array(streamed_grm_mul!(V, stream64, Q))
    mixed = Array(streamed_grm_mul!(V, stream32, Q))
    note_memory!()
    emit(io, @sprintf("| A*(Aᵀ*X)/n | stream mixed vs Float64 | randn | %.2e |",
                      worst_column_error(mixed, ref)))
    CUDA.unsafe_free!(Q)
    CUDA.unsafe_free!(V)
    emit(io, "")
    emit(io, "`Aᵀ*X` by allele-frequency bin: worst row " *
             "`‖out[i, :] − ref[i, :]‖ / ‖ref[i, :]‖`, rows with " *
             "nonzero reference.")
    emit(io, "")
    emit(io, "| Path | RHS | MAF bin | SNPs | rel. error |")
    emit(io, "|---|---|---|---:|---:|")
    for row in maf_rows
        emit(io, row)
    end
    release!()
    return nothing
end

# ---------------------------------------------------------------- shapes

"""`A*X` shapes and `k` for `run_shapes`, the `k > 32` band."""
const AX_SWEEP_SHAPES = ((128, 16, 64, 8, 8), (256, 16, 64, 8, 8))
const AX_SWEEP_KS = (48, 64, 96, 128)
"""`Aᵀ*X` shapes `(BM, BN, BK, TN, TK)` and `k` for `run_shapes`."""
const ATX_SWEEP_SHAPES = ((32, 128, 32, 4, 8), (32, 128, 64, 8, 8),
    (64, 64, 64, 4, 8), (32, 256, 64, 8, 8), (64, 128, 32, 4, 8))
const ATX_SWEEP_KS = (32, 128)

"""
    atx_shape_problem(::Type{T}, shape) -> Union{Nothing, String}

Why `_atX_tiled_kernel!` cannot run `shape = (BM, BN, BK, TN, TK)` for
element type `T` without halving `BM`, or `nothing` when it can.
"""
function atx_shape_problem(::Type{T}, shape::NTuple{5, Int}) where T
    (BM, BN, BK, TN, TK) = shape
    BM % 16 == 0 || return "BM not a multiple of 16"
    BN % TN == 0 || return "BN not a multiple of TN"
    BK % TK == 0 || return "BK not a multiple of TK"
    threads = (BN ÷ TN) * (BK ÷ TK)
    threads <= 1024 || return "$(threads) threads per block"
    shared = ((BN + 1) * BM + BM * BK) * sizeof(T)
    shared <= 48 * 1024 || return "$(shared) bytes of static shared memory"
    return nothing
end

"""
    tiled_atx_shape!(out, A, Y, shape) -> out

Set `out = Aᵀ * Y` with `mul!`'s tiled `Aᵀ*X` launch at tile shape
`(BM, BN, BK, TN, TK)`.
"""
function tiled_atx_shape!(out::CuMatrix{T}, A::CuSnpArray{T}, Y::CuMatrix{T},
                          shape::NTuple{5, Int}) where T <: AbstractFloat
    m, n = size(A)
    k = size(Y, 2)
    (BM, BN, BK, TN, TK) = shape
    column_blocks = cld(n, BN)
    rhs_blocks = cld(k, BK)
    word_rows = cld(m, 16)
    splits = EXT._splits(column_blocks * rhs_blocks, word_rows, BM ÷ 16)
    split_word_rows = (BM ÷ 16) * cld(cld(word_rows, splits), BM ÷ 16)
    splits = cld(word_rows, split_word_rows)
    partials = splits == 1 ? reshape(out, n, k, 1) :
        CuArray{T, 3}(undef, n, k, splits)
    threads = (BN ÷ TN) * (BK ÷ TK)
    @cuda threads=threads blocks=(column_blocks, rhs_blocks, splits) (
        EXT._atX_tiled_kernel!(partials, A.data, Y, A.values, Int32(m),
            Int32(n), Int32(k), Int32(split_word_rows),
            Val(BM), Val(BN), Val(BK), Val(TN), Val(TK))
    )
    splits == 1 || EXT._reduce_splits!(out, partials, splits)
    return out
end

"""
    shape_rows(io, f!, problem, A, rhs, ref, k, rows, shapes, default)

Emit one row per shape in `shapes` for `f!(out, A, rhs, shape)` into a
`rows x k` output: times, GFMA/s, relative difference from `default`'s
output, and worst column error against `ref`. A shape for which
`problem(Float32, shape)` returns a reason gets a skip row.
"""
function shape_rows(io::IO, f!::F, problem::P, A::CuSnpArray{Float32},
                    rhs::CuMatrix{Float32}, ref::Matrix{Float64}, k::Int,
                    rows::Int, shapes, default::NTuple{5, Int}) where {F, P}
    m, n = size(A)
    out = CuMatrix{Float32}(undef, rows, k)
    expected = Float64.(Array(f!(out, A, rhs, default)))
    note_memory!()
    for shape in shapes
        mark = shape == default ? "yes" : ""
        reason = problem(Float32, shape)
        if reason !== nothing
            emit(io, "| $(k) | `$(shape)` | $(mark) | skipped: $(reason) " *
                     "| | | | |")
            continue
        end
        times = time_call(f!, out, A, rhs, shape)
        result = Float64.(Array(out))
        difference = norm(result .- expected) / norm(expected)
        gfma = Float64(m) * n * k / times[1] / 1e9
        worst = worst_column_error(result, ref)
        emit(io, @sprintf("| %d | `%s` | %s | %.3f | %.3f | %.2f | %.1e | %.2e |",
                          k, shape, mark, 1000 * times[1], 1000 * times[2],
                          gfma, difference, worst))
    end
    CUDA.unsafe_free!(out)
    return nothing
end

"""
    run_shapes(io, G)

Print tiled `Float32` `A*X` and `Aᵀ*X` at each sweep shape and `k`, with
the relative difference from the default shape and the worst column error
against the `Float64` tiled product on shared packed words.
"""
function run_shapes(io::IO, G::SnpArray)
    prior = SnpArrays.CUDA_TENSOR_CORES[]
    SnpArrays.cuda_tensor_cores!(false)
    A = operator(Float32, G)
    A64 = shared_float64(A, G)
    m, n = size(A)
    rng = Xoshiro(51)
    header = "| k | shape | default | min ms | median ms | GFMA/s " *
             "| rel. diff | rel. error |"
    rule = "|---:|---|---|---:|---:|---:|---:|---:|"
    emit(io, "")
    emit(io, "## shapes: tile shapes of the tiled Float32 kernels")
    emit(io, "")
    emit(io, "Min and median of $(REPEATS). rel. diff: from the default " *
             "shape's output. rel. error: worst column against the " *
             "Float64 tiled product on the same packed words, `randn` " *
             "right-hand sides rounded to Float32.")
    emit(io, "")
    emit(io, "`A*X`, shape (BM,BN,BK,TM,TK); default " *
             "`AX_TILES[_tile_band(k)]`.")
    emit(io, "")
    emit(io, header)
    emit(io, rule)
    for k in AX_SWEEP_KS
        X = make_rhs(rng, n, k, :randn)
        X64 = CuArray(X)
        out64 = CuMatrix{Float64}(undef, m, k)
        ref = Array(EXT._tiled_mul!(out64, A64, X64))
        note_memory!()
        CUDA.unsafe_free!(X64)
        CUDA.unsafe_free!(out64)
        X32 = CuArray(Float32.(X))
        shape_rows(io, tiled_mul_shape!, shape_problem, A, X32, ref, k, m,
                   AX_SWEEP_SHAPES, EXT.AX_TILES[EXT._tile_band(k)])
        CUDA.unsafe_free!(X32)
        X = ref = nothing
        release!()
    end
    emit(io, "")
    emit(io, "`Aᵀ*X`, shape (BM,BN,BK,TN,TK); default " *
             "`ATX_TILES[_tile_band(k)]`.")
    emit(io, "")
    emit(io, header)
    emit(io, rule)
    for k in ATX_SWEEP_KS
        Y = make_rhs(rng, m, k, :randn)
        Y64 = CuArray(Y)
        out64 = CuMatrix{Float64}(undef, n, k)
        ref = Array(mul!(out64, transpose(A64), Y64))
        note_memory!()
        CUDA.unsafe_free!(Y64)
        CUDA.unsafe_free!(out64)
        Y32 = CuArray(Float32.(Y))
        shape_rows(io, tiled_atx_shape!, atx_shape_problem, A, Y32, ref, k, n,
                   ATX_SWEEP_SHAPES, EXT.ATX_TILES[EXT._tile_band(k)])
        CUDA.unsafe_free!(Y32)
        Y = ref = nothing
        release!()
    end
    free_tables!(A64)
    free_operator!(A)
    SnpArrays.cuda_tensor_cores!(prior)
    release!()
    return nothing
end

function main()
    for section in SECTIONS
        section in (ALL_SECTIONS..., "shapes") || throw(ArgumentError(
            "unknown section $(section); expected some of $((ALL_SECTIONS..., "shapes"))"))
    end
    M_CAP % 16 == 0 || M_CAP == typemax(Int) || throw(ArgumentError(
        "m_cap must be a multiple of 16; got $(M_CAP)"))
    path = report_path(RESULTS_DIR)
    io = open(path, "w")
    emit(io, "# `cuda_diagnose.jl` on $(CUDA.name(CUDA.device()))")
    emit(io, "")
    emit(io, "## Environment")
    emit(io, "")
    bed, G = capped_genotypes(io)
    print_header(io, bed, G)
    emit(io, "")
    emit(io, "Centered, scaled, mean-imputed throughout.")
    "stream" in SECTIONS && run_stream(io, bed, G)
    "scale" in SECTIONS && run_scale(io, G)
    "tiles" in SECTIONS && run_tiles(io, G)
    "accuracy" in SECTIONS && run_accuracy(io, bed, G)
    "shapes" in SECTIONS && run_shapes(io, G)
    emit(io, "")
    emit(io, "## Peak memory")
    emit(io, "")
    emit(io, @sprintf("| device in use, sampled, pool cache | %.0f MiB |",
                      PEAK_DEVICE[] / 2^20))
    emit(io, @sprintf("| RSS | %.0f MiB |", Sys.maxrss() / 2^20))
    close(io)
    println(path)
    return nothing
end

main()
