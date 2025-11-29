#!/usr/bin/env python3
"""
NumPy benchmark script with auto-calibration

Improved benchmarking approach:
- Auto-calibrates to run enough ops to hit minimum measurement time (100ms)
- Runs operations in batches to reduce measurement overhead
- Provides ops/sec for easier interpretation
- Uses multiple samples for statistical robustness
"""

import json
import sys
import time
import traceback
from typing import Any, Dict

import numpy as np

# Benchmark configuration (can be overridden from stdin)
MIN_SAMPLE_TIME_MS = 100  # Minimum time per sample (reduces noise)
TARGET_SAMPLES = 5  # Number of samples to collect for statistics


def setup_arrays(setup: Dict[str, Any], operation: str = None) -> Dict[str, np.ndarray]:
    """Create arrays based on setup specification"""
    import io
    arrays = {}

    for key, spec in setup.items():
        shape = spec["shape"]
        dtype = spec.get("dtype", "float64")
        fill_type = spec.get("fill", "zeros")

        # Handle scalar values (n, axis, new_shape, shape, fill_value)
        if key in ["n", "axis", "new_shape", "shape", "fill_value"]:
            if len(shape) == 1:
                arrays[key] = shape[0]
            else:
                arrays[key] = tuple(shape)
            continue

        # Check 'value' first to avoid default fill creating zeros
        if "value" in spec:
            arrays[key] = np.full(shape, spec["value"], dtype=dtype)
        elif fill_type == "zeros":
            arrays[key] = np.zeros(shape, dtype=dtype)
        elif fill_type == "ones":
            arrays[key] = np.ones(shape, dtype=dtype)
        elif fill_type == "random":
            arrays[key] = np.random.randn(*shape).astype(dtype)
        elif fill_type == "arange":
            arrays[key] = np.arange(np.prod(shape), dtype=dtype).reshape(shape)

    # Pre-serialize data for parsing benchmarks
    if operation == "parseNpy" and "a" in arrays:
        buffer = io.BytesIO()
        np.save(buffer, arrays["a"])
        arrays["_npyBytes"] = buffer.getvalue()
    elif operation == "serializeNpzSync" and "a" in arrays:
        # Create dict of arrays for NPZ serialization
        npz_arrays = {k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}
        arrays["_npzArrays"] = npz_arrays
    elif operation == "parseNpzSync" and "a" in arrays:
        # Create and pre-serialize NPZ
        npz_arrays = {k: v for k, v in arrays.items() if isinstance(v, np.ndarray)}
        buffer = io.BytesIO()
        np.savez(buffer, **npz_arrays)
        arrays["_npzBytes"] = buffer.getvalue()

    return arrays


def execute_operation(operation: str, arrays: Dict[str, np.ndarray]) -> Any:
    """Execute the benchmark operation"""
    # Array creation
    if operation == "zeros":
        return np.zeros(arrays["shape"])
    elif operation == "ones":
        return np.ones(arrays["shape"])
    elif operation == "empty":
        return np.empty(arrays["shape"])
    elif operation == "full":
        return np.full(arrays["shape"], arrays["fill_value"])
    elif operation == "arange":
        return np.arange(arrays["n"])
    elif operation == "linspace":
        return np.linspace(0, 100, arrays["n"])
    elif operation == "logspace":
        return np.logspace(0, 3, arrays["n"])
    elif operation == "geomspace":
        return np.geomspace(1, 1000, arrays["n"])
    elif operation == "eye":
        return np.eye(arrays["n"])
    elif operation == "identity":
        return np.identity(arrays["n"])
    elif operation == "copy":
        return np.copy(arrays["a"])
    elif operation == "zeros_like":
        return np.zeros_like(arrays["a"])
    elif operation == "ones_like":
        return np.ones_like(arrays["a"])
    elif operation == "empty_like":
        return np.empty_like(arrays["a"])
    elif operation == "full_like":
        return np.full_like(arrays["a"], 7)

    # Arithmetic
    elif operation == "add":
        return arrays["a"] + arrays["b"]
    elif operation == "subtract":
        return arrays["a"] - arrays["b"]
    elif operation == "multiply":
        return arrays["a"] * arrays["b"]
    elif operation == "divide":
        return arrays["a"] / arrays["b"]
    elif operation == "mod":
        return np.mod(arrays["a"], arrays["b"])
    elif operation == "floor_divide":
        return np.floor_divide(arrays["a"], arrays["b"])
    elif operation == "reciprocal":
        return np.reciprocal(arrays["a"])
    elif operation == "positive":
        return np.positive(arrays["a"])

    # Mathematical operations
    elif operation == "sqrt":
        return np.sqrt(arrays["a"])
    elif operation == "power":
        return np.power(arrays["a"], arrays["b"])
    elif operation == "absolute":
        return np.absolute(arrays["a"])
    elif operation == "negative":
        return np.negative(arrays["a"])
    elif operation == "sign":
        return np.sign(arrays["a"])

    # Linear algebra
    elif operation == "dot":
        return np.dot(arrays["a"], arrays["b"])
    elif operation == "inner":
        return np.inner(arrays["a"], arrays["b"])
    elif operation == "outer":
        return np.outer(arrays["a"], arrays["b"])
    elif operation == "tensordot":
        axes = arrays.get("axes", 2)
        return np.tensordot(arrays["a"], arrays["b"], axes=axes)
    elif operation == "matmul":
        return arrays["a"] @ arrays["b"]
    elif operation == "trace":
        return np.trace(arrays["a"])
    elif operation == "transpose":
        return arrays["a"].T

    # Reductions
    elif operation == "sum":
        axis = arrays.get("axis")
        return arrays["a"].sum(axis=axis)
    elif operation == "mean":
        axis = arrays.get("axis")
        return arrays["a"].mean(axis=axis)
    elif operation == "max":
        axis = arrays.get("axis")
        return arrays["a"].max(axis=axis)
    elif operation == "min":
        axis = arrays.get("axis")
        return arrays["a"].min(axis=axis)
    elif operation == "prod":
        axis = arrays.get("axis")
        return arrays["a"].prod(axis=axis)
    elif operation == "argmin":
        axis = arrays.get("axis")
        return arrays["a"].argmin(axis=axis)
    elif operation == "argmax":
        axis = arrays.get("axis")
        return arrays["a"].argmax(axis=axis)
    elif operation == "var":
        axis = arrays.get("axis")
        return arrays["a"].var(axis=axis)
    elif operation == "std":
        axis = arrays.get("axis")
        return arrays["a"].std(axis=axis)
    elif operation == "all":
        axis = arrays.get("axis")
        return arrays["a"].all(axis=axis)
    elif operation == "any":
        axis = arrays.get("axis")
        return arrays["a"].any(axis=axis)

    # Reshape
    elif operation == "reshape":
        return arrays["a"].reshape(arrays["new_shape"])
    elif operation == "flatten":
        return arrays["a"].flatten()
    elif operation == "ravel":
        return arrays["a"].ravel()
    elif operation == "squeeze":
        return arrays["a"].squeeze()

    # Slicing
    elif operation == "slice":
        return arrays["a"][:100, :100]

    # IO operations (NPY/NPZ)
    elif operation == "serializeNpy":
        import io
        buffer = io.BytesIO()
        np.save(buffer, arrays["a"])
        return buffer.getvalue()
    elif operation == "parseNpy":
        import io
        # arrays["_npyBytes"] should be pre-serialized
        buffer = io.BytesIO(arrays["_npyBytes"])
        return np.load(buffer)
    elif operation == "serializeNpzSync":
        import io
        buffer = io.BytesIO()
        np.savez(buffer, **arrays["_npzArrays"])
        return buffer.getvalue()
    elif operation == "parseNpzSync":
        import io
        # arrays["_npzBytes"] should be pre-serialized
        buffer = io.BytesIO(arrays["_npzBytes"])
        return np.load(buffer)

    else:
        raise ValueError(f"Unknown operation: {operation}")


def calibrate_ops_per_sample(
    operation: str,
    arrays: Dict[str, np.ndarray],
    target_time_ms: float = MIN_SAMPLE_TIME_MS,
) -> int:
    """
    Auto-calibrate: Determine how many operations to run per sample
    to achieve the target minimum sample time
    """
    ops_per_sample = 1
    calibration_runs = 0
    max_calibration_runs = 10

    while calibration_runs < max_calibration_runs:
        start = time.perf_counter()

        # Run operations in batch
        for _ in range(ops_per_sample):
            result = execute_operation(operation, arrays)
            _ = result  # Prevent optimization

        elapsed_ms = (time.perf_counter() - start) * 1000

        # If we hit the target time, we're done
        if elapsed_ms >= target_time_ms:
            break

        # If operation is very fast, increase ops exponentially
        if elapsed_ms < target_time_ms / 10:
            ops_per_sample *= 10
        elif elapsed_ms < target_time_ms / 2:
            ops_per_sample *= 2
        else:
            # Close enough, calculate exact number needed
            target_ops = int(np.ceil((ops_per_sample * target_time_ms) / elapsed_ms))
            ops_per_sample = max(target_ops, ops_per_sample + 1)
            break

        calibration_runs += 1

    # Cap at reasonable maximum to prevent too-long samples
    max_ops_per_sample = 100000
    return min(ops_per_sample, max_ops_per_sample)


def run_benchmark(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single benchmark with auto-calibration and return timing results"""
    name = spec["name"]
    operation = spec["operation"]
    setup = spec["setup"]
    warmup = spec["warmup"]

    # Setup arrays (pass operation for IO benchmarks that need pre-serialized data)
    arrays = setup_arrays(setup, operation)

    # Warmup phase - run several times to stabilize JIT/caching
    for _ in range(warmup):
        execute_operation(operation, arrays)

    # Calibration phase - determine ops per sample
    ops_per_sample = calibrate_ops_per_sample(operation, arrays)

    # Benchmark phase - collect samples
    sample_times = []
    total_ops = 0

    for _ in range(TARGET_SAMPLES):
        start = time.perf_counter()

        # Run batch of operations
        for _ in range(ops_per_sample):
            result = execute_operation(operation, arrays)
            _ = result  # Prevent optimization

        elapsed_ms = (time.perf_counter() - start) * 1000
        time_per_op = elapsed_ms / ops_per_sample
        sample_times.append(time_per_op)
        total_ops += ops_per_sample

    # Calculate statistics
    times_array = np.array(sample_times)
    mean_ms = float(np.mean(times_array))
    median_ms = float(np.median(times_array))
    min_ms = float(np.min(times_array))
    max_ms = float(np.max(times_array))
    std_ms = float(np.std(times_array))
    ops_per_sec = 1000.0 / mean_ms

    return {
        "name": name,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "std_ms": std_ms,
        "ops_per_sec": ops_per_sec,
        "total_ops": total_ops,
        "total_samples": TARGET_SAMPLES,
    }


def main():
    """Main entry point - read specs and config from stdin, output results to stdout"""
    global MIN_SAMPLE_TIME_MS, TARGET_SAMPLES

    try:
        # Read benchmark specifications and config from stdin
        input_data = json.loads(sys.stdin.read())

        # Support both old format (just specs) and new format (specs + config)
        if isinstance(input_data, dict) and "specs" in input_data:
            specs = input_data["specs"]
            config = input_data.get("config", {})
            MIN_SAMPLE_TIME_MS = config.get("minSampleTimeMs", MIN_SAMPLE_TIME_MS)
            TARGET_SAMPLES = config.get("targetSamples", TARGET_SAMPLES)
        else:
            # Old format - just specs array
            specs = input_data

        results = []

        # Print environment info to stderr
        print(f"Python {sys.version.split()[0]}", file=sys.stderr)
        print(f"NumPy {np.__version__}", file=sys.stderr)
        print(
            f"Running {len(specs)} benchmarks with auto-calibration...", file=sys.stderr
        )
        print(
            f"Target: {MIN_SAMPLE_TIME_MS}ms per sample, {TARGET_SAMPLES} samples per benchmark\n",
            file=sys.stderr,
        )

        for i, spec in enumerate(specs, 1):
            result = run_benchmark(spec)
            results.append(result)

            # Print progress to stderr (matching TypeScript format)
            name_padded = spec["name"].ljust(40)
            mean_padded = f"{result['mean_ms']:.3f}".rjust(8)
            ops_formatted = f"{int(result['ops_per_sec']):,}".rjust(12)
            print(
                f"  [{i}/{len(specs)}] {name_padded} {mean_padded}ms  {ops_formatted} ops/sec",
                file=sys.stderr,
            )

        # Output results as JSON to stdout
        print(json.dumps(results, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
