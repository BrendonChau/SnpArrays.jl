# Task-partitioning constants shared by the SIMD kernels and column counts.

"""
    DECODE_WIDTH

Number of genotypes produced by one vector decode (four packed bytes).
Separate from `VECTOR_BYTES`: `PACKED_EXPANSION`, `PACKED_SHIFTS`, the
`VecRange{4}` byte load, and every `row += 16` step stay fixed at 16
regardless of the SIMD register width.
"""
const DECODE_WIDTH = 16

"""
    TASKS_PER_THREAD

Target task count is `TASKS_PER_THREAD * Threads.nthreads()`, so uneven
per-task progress balances across threads.
"""
const TASKS_PER_THREAD = 4

"""
    TASK_AXIS_FLOOR

Minimum block size along a task-partitioned axis; below it, per-task
overhead and the output-tile read-modify-write dominate.
"""
const TASK_AXIS_FLOOR = 256

"""
    _task_axis_step(length::Int, upper::Int) -> Int

Return the per-task block size along a task-partitioned axis of total
`length`, targeting `TASKS_PER_THREAD * Threads.nthreads()` tasks, rounded
up to a multiple of `DECODE_WIDTH`, and clamped between `TASK_AXIS_FLOOR`
and `upper` (`upper` is itself rounded down to a multiple of
`DECODE_WIDTH` and floored at `TASK_AXIS_FLOOR` before use).
"""
function _task_axis_step(length::Int, upper::Int)
    round_up16(x) = cld(x, DECODE_WIDTH) * DECODE_WIDTH
    round_down16(x) = (x ÷ DECODE_WIDTH) * DECODE_WIDTH
    bounded_upper = max(TASK_AXIS_FLOOR, round_down16(upper))
    tasks = TASKS_PER_THREAD * Threads.nthreads()
    step = length == 0 ? bounded_upper : round_up16(cld(length, tasks))
    return min(bounded_upper, max(TASK_AXIS_FLOOR, step))
end
