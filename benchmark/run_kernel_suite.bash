#!/usr/bin/env bash
set -euo pipefail

repo=$(realpath "$(dirname "$0")/..")
parallelism=${SNP_BENCH_JOBS:-4}
round_count=${SNP_BENCH_ROUNDS:-4}
sample_count=${SNP_BENCH_SAMPLES:-25}
datasets_csv=${SNP_BENCH_DATASETS:-synthetic}
types_csv=${SNP_BENCH_TYPES:-Float32,Float64}
orientations_csv=${SNP_BENCH_ORIENTATIONS:-AX,AtX}
widths_csv=${SNP_BENCH_WIDTHS:-8,17,128,256}
release_commit=007ab2970ecd3144542dcc0666a200d4977eae29
release_url=https://github.com/OpenMendel/SnpArrays.jl.git

positive_integer() {
    local name=$1 value=$2
    [[ $value =~ ^[1-9][0-9]*$ ]] || {
        echo "$name must be a positive integer" >&2
        exit 1
    }
}

validate_list() {
    local name=$1 allowed=$2
    shift 2
    local value seen=,
    (( $# > 0 )) || {
        echo "$name must not be empty" >&2
        exit 1
    }
    for value in "$@"; do
        [[ -n $value && ,$allowed, == *,$value,* ]] || {
            echo "$name contains an unsupported value: $value" >&2
            exit 1
        }
        [[ $seen != *,$value,* ]] || {
            echo "$name contains a duplicate value: $value" >&2
            exit 1
        }
        seen+="$value,"
    done
}

validate_csv() {
    local name=$1 value=$2
    [[ -n $value && $value != ,* && $value != *, && $value != *,,* ]] || {
        echo "$name must be a nonempty comma-separated list" >&2
        exit 1
    }
}

positive_integer SNP_BENCH_JOBS "$parallelism"
positive_integer SNP_BENCH_ROUNDS "$round_count"
positive_integer SNP_BENCH_SAMPLES "$sample_count"
validate_csv SNP_BENCH_DATASETS "$datasets_csv"
validate_csv SNP_BENCH_TYPES "$types_csv"
validate_csv SNP_BENCH_ORIENTATIONS "$orientations_csv"
validate_csv SNP_BENCH_WIDTHS "$widths_csv"
IFS=',' read -r -a datasets <<< "$datasets_csv"
IFS=',' read -r -a types <<< "$types_csv"
IFS=',' read -r -a orientations <<< "$orientations_csv"
IFS=',' read -r -a widths <<< "$widths_csv"
validate_list SNP_BENCH_DATASETS synthetic "${datasets[@]}"
validate_list SNP_BENCH_TYPES Float32,Float64 "${types[@]}"
validate_list SNP_BENCH_ORIENTATIONS AX,AtX "${orientations[@]}"
validate_list SNP_BENCH_WIDTHS 8,17,128,256 "${widths[@]}"

[[ -z $(git -C "$repo" status --porcelain=v1 --untracked-files=all -- \
    src Project.toml) ]] || {
    echo 'src/ or Project.toml differs from the current commit' >&2
    exit 1
}

run_root=${SNP_BENCH_RUN_ROOT:-$(mktemp -d /tmp/snparrays-kernel.XXXXXX)}
logs=${SNP_BENCH_LOGS:-$run_root/logs}
fixtures=${SNP_BENCH_FIXTURES:-$run_root/fixtures}
output=${SNP_BENCH_OUTPUT:-$repo/benchmark}
reference=$run_root/SnpArrays-v0.3.23
mkdir -p "$logs" "$fixtures" "$output"

reference_matches() {
    [[ -d $reference/.git ]] &&
        [[ $(git -C "$reference" remote get-url origin) == $release_url ]] &&
        [[ -z $(git -C "$reference" status --porcelain=v1 \
            --untracked-files=all) ]] &&
        [[ $(git -C "$reference" rev-parse HEAD) == $release_commit ]]
}

if [[ -e $reference ]]; then
    reference_matches || {
        echo "existing release checkout cannot be reused: $reference" >&2
        exit 1
    }
else
    git clone --depth 1 --single-branch --branch v0.3.23 \
        "$release_url" "$reference"
    reference_matches || {
        echo 'cloned release checkout failed provenance validation' >&2
        exit 1
    }
fi

JULIA_LOAD_PATH="@:@stdlib" julia \
    --startup-file=no --project="$repo/benchmark" -e \
    'using Pkg; Pkg.develop(path=ARGS[1]); Pkg.instantiate()' \
    "$repo" |& tee "$logs/current_setup.log"

JULIA_LOAD_PATH="@:@stdlib" julia \
    --startup-file=no --project="$repo/benchmark/reference_env" -e \
    'using Pkg; Pkg.develop(path=ARGS[1]); Pkg.instantiate()' \
    "$reference" |& tee "$logs/reference_setup.log"

SNP_BENCH_FIXTURES=$fixtures JULIA_LOAD_PATH="@:@stdlib" julia \
    --startup-file=no --project="$repo/benchmark" -e \
    'suite = only(ARGS); include(suite); synthetic_fixture()' \
    "$repo/benchmark/kernel_suite.jl" |& \
    tee "$logs/fixture_setup.log"

run_cell() {
    local mode=$1 round=$2 type=$3 orientation=$4 width=$5 order=$6
    local project log
    if [[ $mode == current ]]; then
        project=$repo/benchmark
    else
        project=$repo/benchmark/reference_env
    fi
    log=$logs/fragments/$mode-$round/$type-$orientation-$width.log
    echo "START,$mode,$round,$type,$orientation,$width"
    if JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
        SNP_BENCH_FIXTURES=$fixtures SNP_BENCH_DATASETS=$datasets_csv \
        SNP_BENCH_ENVIRONMENT_ORDER=$order \
        SNP_BENCH_TYPES=$type SNP_BENCH_ORIENTATIONS=$orientation \
        SNP_BENCH_WIDTHS=$width JULIA_LOAD_PATH="@:@stdlib" \
        julia --startup-file=no --project="$project" \
        "$repo/benchmark/kernel_suite.jl" \
        "$mode" "$round" "$sample_count" |& tee "$log" >/dev/null; then
        echo "DONE,$mode,$round,$type,$orientation,$width"
    else
        echo "FAILED,$mode,$round,$type,$orientation,$width" >&2
        return 1
    fi
}

run_mode() {
    local mode=$1 round=$2 order=$3
    local type orientation width pid cell_log failed=0
    local -a cell_logs=() pids=()
    mkdir -p "$logs/fragments/$mode-$round"
    for type in "${types[@]}"; do
        for orientation in "${orientations[@]}"; do
            for width in "${widths[@]}"; do
                cell_log=$logs/fragments/$mode-$round/\
$type-$orientation-$width.log
                cell_logs+=("$cell_log")
                run_cell "$mode" "$round" "$type" "$orientation" \
                    "$width" "$order" &
                pids+=($!)
                if (( ${#pids[@]} == parallelism )); then
                    for pid in "${pids[@]}"; do
                        wait "$pid" || failed=1
                    done
                    pids=()
                fi
            done
        done
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 )) || return 1
    cat "${cell_logs[@]}" > "$logs/$mode-$round.log"
}

for (( round=1; round<=round_count; round++ )); do
    if (( round % 2 )); then
        modes=(current release)
    else
        modes=(release current)
    fi
    order=$(IFS=';'; echo "${modes[*]}")
    for mode in "${modes[@]}"; do
        run_mode "$mode" "$round" "$order"
    done
done

JULIA_LOAD_PATH="@:@stdlib" julia \
    --startup-file=no --project="$repo/benchmark" \
    "$repo/benchmark/kernel_report.jl" \
    "$logs" "$output" "$round_count" "$sample_count" \
    "$datasets_csv" "$types_csv" "$orientations_csv" "$widths_csv"
