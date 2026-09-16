using Statistics: median, quantile

const MODES = ("current", "release")
const BACKEND_MODES = (("current", "current"),
                       ("release", "release"))

function parse_logs(directory::String, rounds::AbstractUnitRange{Int})
    samples = Dict{NTuple{6, String}, Vector{Tuple{Int, Int, Float64}}}()
    results = Dict{NTuple{6, String}, Vector{Vector{String}}}()
    metadata = Dict{String, Set{String}}()
    data = Dict{String, Set{String}}()
    inputs = Dict{NTuple{4, String}, Set{String}}()
    for mode in MODES, round in rounds
        path = joinpath(directory, "$mode-$round.log")
        isfile(path) || throw(ArgumentError("missing log $path"))
        for line in eachline(path)
            fields = split(line, ',')
            isempty(fields) && continue
            if fields[1] == "SAMPLE"
                key = Tuple(fields[3:8])
                value = (parse(Int, fields[2]), parse(Int, fields[9]),
                         parse(Float64, fields[10]))
                push!(get!(samples, key, Tuple{Int, Int, Float64}[]), value)
            elseif fields[1] == "RESULT"
                key = Tuple(fields[3:8])
                push!(get!(results, key, Vector{String}[]), fields)
            elseif fields[1] == "META"
                push!(get!(metadata, fields[2], Set{String}()), fields[3])
            elseif fields[1] == "STAT"
                key = "STAT," * join(fields[2:4], ',')
                push!(get!(metadata, key, Set{String}()),
                      join(fields[5:end], ','))
            elseif fields[1] == "DATA"
                push!(get!(data, fields[2], Set{String}()),
                      join(fields[3:end], ','))
            elseif fields[1] == "INPUT"
                key = Tuple(fields[2:5])
                push!(get!(inputs, key, Set{String}()), fields[6])
            end
        end
    end
    return samples, results, metadata, data, inputs
end

function validate!(samples, results, metadata, data, inputs,
                   rounds::AbstractUnitRange{Int}, samples_per_round::Int,
                   datasets::Vector{String}, types::Vector{String},
                   orientations::Vector{String}, widths::Vector{String})::Nothing
    expected_keys = Set((dataset, type, orientation, width, backend, mode)
                        for dataset in datasets, type in types,
                        orientation in orientations, width in widths,
                        (backend, mode) in BACKEND_MODES)
    Set(keys(samples)) == expected_keys ||
        throw(ArgumentError("wrong benchmark cells"))
    Set(keys(results)) == expected_keys ||
        throw(ArgumentError("wrong result cells"))
    all(length(values) == 1 for values in values(data)) ||
        throw(ArgumentError("fixture hashes differ"))
    all(length(values) == 1 for values in values(inputs)) ||
        throw(ArgumentError("RHS hashes differ"))
    stat_keys = Set(key for key in keys(metadata) if startswith(key, "STAT,"))
    expected_stats = Set("STAT,$mode,$dataset,$type" for mode in MODES,
                         dataset in datasets, type in types)
    stat_keys == expected_stats ||
        throw(ArgumentError("missing native statistic hashes"))
    all(length(metadata[key]) == 1 for key in stat_keys) ||
        throw(ArgumentError("native statistic hashes differ"))
    expected_samples = length(rounds) * samples_per_round
    for (key, entries) in samples
        length(entries) == expected_samples ||
            throw(ArgumentError("wrong samples $key"))
        seen = Set((round, sample) for (round, sample, _) in entries)
        seen == Set((round, sample) for round in rounds
                   for sample in 1:samples_per_round) ||
            throw(ArgumentError("missing samples $key"))
        length(results[key]) == length(rounds) ||
            throw(ArgumentError("wrong results $key"))
        all(parse(Int, row[9]) == samples_per_round for row in results[key]) ||
            throw(ArgumentError("wrong result samples $key"))
    end
    return nothing
end

function ordered_keys(samples)
    return sort!(collect(keys(samples)))
end

function summary_row(key, samples, results)
    times = [entry[3] for entry in samples[key]]
    rows = results[key]
    return (median(times), quantile(times, 0.1), quantile(times, 0.9),
            maximum(parse(Int, row[13]) for row in rows),
            maximum(parse(Int, row[14]) for row in rows),
            maximum(parse(Float64, row[15]) for row in rows),
            maximum(parse(Float64, row[16]) for row in rows),
            maximum(row[17] == "NA" ? NaN : parse(Float64, row[17])
                    for row in rows),
            count(row -> row[18] == "false", rows))
end

function baseline_key(key, backend::String, mode::String)
    return (key[1], key[2], key[3], key[4], backend, mode)
end

function write_samples(path::String, samples)::Nothing
    open(path, "w") do io
        println(io,
                "dataset,type,orientation,width,backend,mode,round,sample,ns")
        for key in ordered_keys(samples)
            for entry in sort!(copy(samples[key]))
                println(io, join(key, ','), ',', join(entry, ','))
            end
        end
    end
    return nothing
end

function write_summary(path::String, samples, results)
    open(path, "w") do io
        println(io, "dataset,type,orientation,width,backend,mode,samples,",
                "median_ms,p10_ms,p90_ms,bytes,allocations,max_abs_error,",
                "max_rel_error,max_bigfloat_spot_error,invalid_rounds,",
                "ratio_to_release")
        for key in ordered_keys(samples)
            metric = summary_row(key, samples, results)
            release = summary_row(baseline_key(key, "release", "release"),
                                  samples, results)[1]
            values = (length(samples[key]), metric[1] / 1.0e6,
                      metric[2] / 1.0e6, metric[3] / 1.0e6, metric[4],
                      metric[5], metric[6], metric[7], metric[8], metric[9],
                      release / metric[1])
            println(io, join(key, ','), ',', join(values, ','))
        end
    end
    return nothing
end

function write_metadata(path::String, metadata, data, inputs)::Nothing
    open(path, "w") do io
        for key in sort!(collect(keys(metadata)))
            for value in sort!(collect(metadata[key]))
                prefix = startswith(key, "STAT,") ? "" : "META,"
                println(io, prefix, key, ',', value)
            end
        end
        for key in sort!(collect(keys(data)))
            println(io, "DATA,", key, ',', only(data[key]))
        end
        for key in sort!(collect(keys(inputs)))
            println(io, "INPUT,", join(key, ','), ',', only(inputs[key]))
        end
    end
    return nothing
end

function write_report(path::String, samples, results)::Nothing
    open(path, "w") do io
        println(io, "# Packed SNP kernel measurements")
        println(io)
        println(io, "| Dataset | Type | Product | Width | Backend | Mode | ",
                "Median ms | p10–p90 ms | Max relative error | 64ε | ",
                "Release ratio |")
        println(io, "|---|---|---|---:|---|---|---:|---:|---:|---|---:|")
        for key in ordered_keys(samples)
            metric = summary_row(key, samples, results)
            release = summary_row(baseline_key(key, "release", "release"),
                                  samples, results)[1]
            println(io, "| ", key[1], " | ", key[2], " | ", key[3], " | ",
                    key[4], " | ", key[5], " | ", key[6], " | ",
                    round(metric[1] / 1.0e6; digits=3), " | ",
                    round(metric[2] / 1.0e6; digits=3), "–",
                    round(metric[3] / 1.0e6; digits=3), " | ",
                    metric[7], " | ", metric[9] == 0 ? "pass" : "fail", " | ",
                    round(release / metric[1]; digits=3), " |")
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 8 || throw(ArgumentError("expected benchmark settings"))
    round_count = parse(Int, ARGS[3])
    samples_per_round = parse(Int, ARGS[4])
    round_count > 0 || throw(ArgumentError("rounds must be positive"))
    samples_per_round > 0 || throw(ArgumentError("samples must be positive"))
    rounds = 1:round_count
    datasets = String.(split(ARGS[5], ','))
    types = String.(split(ARGS[6], ','))
    orientations = String.(split(ARGS[7], ','))
    widths = String.(split(ARGS[8], ','))
    samples, results, metadata, data, inputs = parse_logs(ARGS[1], rounds)
    validate!(samples, results, metadata, data, inputs, rounds,
              samples_per_round, datasets, types, orientations, widths)
    write_samples(joinpath(ARGS[2], "kernel_samples.csv"), samples)
    write_summary(joinpath(ARGS[2], "kernel_summary.csv"), samples, results)
    write_metadata(joinpath(ARGS[2], "kernel_metadata.txt"), metadata, data,
                   inputs)
    write_report(joinpath(ARGS[2], "kernel_results.md"), samples, results)
end
