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

        # Handle scalar values (n, axis, new_shape, shape, fill_value, target_shape, dims, kth)
        if key in ["n", "axis", "new_shape", "shape", "fill_value", "target_shape", "dims", "kth"]:
            if len(shape) == 1:
                arrays[key] = shape[0]
            else:
                arrays[key] = tuple(shape)
            continue

        # Handle indices array
        if key == "indices":
            arrays[key] = shape
            continue

        # Handle string values (like einsum subscripts)
        if key == "subscripts":
            arrays[key] = spec.get("value")
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
        elif fill_type == "invertible":
            # Create an invertible matrix: arange + n*I (diagonally dominant)
            n = shape[0]
            arange_matrix = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            identity = np.eye(n, dtype=dtype)
            arrays[key] = arange_matrix + identity * (n * n)
        elif fill_type == "complex":
            # Create complex array with [1+1j, 2+2j, 3+3j, ...]
            size = int(np.prod(shape))
            arrays[key] = np.array([complex(i+1, i+1) for i in range(size)], dtype=np.complex128).reshape(shape)

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
    elif operation == "cbrt":
        return np.cbrt(arrays["a"])
    elif operation == "fabs":
        return np.fabs(arrays["a"])
    elif operation == "divmod":
        q, r = np.divmod(arrays["a"], arrays["b"])
        return q  # Just return quotient for benchmarking
    elif operation == "gcd":
        return np.gcd(arrays["a"], arrays["b"])
    elif operation == "lcm":
        return np.lcm(arrays["a"], arrays["b"])
    elif operation == "float_power":
        return np.float_power(arrays["a"], arrays["b"])

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

    # Trigonometric
    elif operation == "sin":
        return np.sin(arrays["a"])
    elif operation == "cos":
        return np.cos(arrays["a"])
    elif operation == "tan":
        return np.tan(arrays["a"])
    elif operation == "arctan2":
        return np.arctan2(arrays["a"], arrays["b"])
    elif operation == "hypot":
        return np.hypot(arrays["a"], arrays["b"])

    # Hyperbolic
    elif operation == "sinh":
        return np.sinh(arrays["a"])
    elif operation == "cosh":
        return np.cosh(arrays["a"])
    elif operation == "tanh":
        return np.tanh(arrays["a"])

    # Exponential
    elif operation == "exp":
        return np.exp(arrays["a"])
    elif operation == "exp2":
        return np.exp2(arrays["a"])
    elif operation == "expm1":
        return np.expm1(arrays["a"])
    elif operation == "log":
        return np.log(arrays["a"])
    elif operation == "log2":
        return np.log2(arrays["a"])
    elif operation == "log10":
        return np.log10(arrays["a"])
    elif operation == "log1p":
        return np.log1p(arrays["a"])
    elif operation == "logaddexp":
        return np.logaddexp(arrays["a"], arrays["b"])
    elif operation == "logaddexp2":
        return np.logaddexp2(arrays["a"], arrays["b"])

    # Gradient
    elif operation == "diff":
        return np.diff(arrays["a"])
    elif operation == "ediff1d":
        return np.ediff1d(arrays["a"])
    elif operation == "gradient":
        return np.gradient(arrays["a"])
    elif operation == "cross":
        return np.cross(arrays["a"], arrays["b"])

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
    elif operation == "diagonal":
        return np.diagonal(arrays["a"])
    elif operation == "kron":
        return np.kron(arrays["a"], arrays["b"])
    elif operation == "einsum":
        return np.einsum(arrays["subscripts"], arrays["a"], arrays["b"])
    elif operation == "deg2rad":
        return np.deg2rad(arrays["a"])
    elif operation == "rad2deg":
        return np.rad2deg(arrays["a"])

    # numpy.linalg module operations
    elif operation == "linalg_det":
        return np.linalg.det(arrays["a"])
    elif operation == "linalg_inv":
        return np.linalg.inv(arrays["a"])
    elif operation == "linalg_solve":
        return np.linalg.solve(arrays["a"], arrays["b"])
    elif operation == "linalg_qr":
        return np.linalg.qr(arrays["a"])
    elif operation == "linalg_cholesky":
        # Create positive definite matrix: A^T * A + n*I
        a = arrays["a"]
        n = a.shape[0]
        posdef = a.T @ a + np.eye(n) * n
        return np.linalg.cholesky(posdef)
    elif operation == "linalg_svd":
        return np.linalg.svd(arrays["a"])
    elif operation == "linalg_eig":
        return np.linalg.eig(arrays["a"])
    elif operation == "linalg_eigh":
        # Create symmetric matrix: (A + A^T) / 2
        a = arrays["a"]
        sym = (a + a.T) / 2
        return np.linalg.eigh(sym)
    elif operation == "linalg_norm":
        return np.linalg.norm(arrays["a"])
    elif operation == "linalg_matrix_rank":
        return np.linalg.matrix_rank(arrays["a"])
    elif operation == "linalg_pinv":
        return np.linalg.pinv(arrays["a"])
    elif operation == "linalg_cond":
        return np.linalg.cond(arrays["a"])
    elif operation == "linalg_matrix_power":
        return np.linalg.matrix_power(arrays["a"], 3)
    elif operation == "linalg_lstsq":
        return np.linalg.lstsq(arrays["a"], arrays["b"], rcond=None)
    elif operation == "linalg_cross":
        return np.cross(arrays["a"], arrays["b"])

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

    # New reduction functions
    elif operation == "cumsum":
        return arrays["a"].cumsum()
    elif operation == "cumprod":
        return arrays["a"].cumprod()
    elif operation == "ptp":
        return np.ptp(arrays["a"])
    elif operation == "median":
        return np.median(arrays["a"])
    elif operation == "percentile":
        return np.percentile(arrays["a"], 50)
    elif operation == "quantile":
        return np.quantile(arrays["a"], 0.5)
    elif operation == "average":
        return np.average(arrays["a"])
    elif operation == "nansum":
        return np.nansum(arrays["a"])
    elif operation == "nanmean":
        return np.nanmean(arrays["a"])
    elif operation == "nanmin":
        return np.nanmin(arrays["a"])
    elif operation == "nanmax":
        return np.nanmax(arrays["a"])
    elif operation == "nanquantile":
        return np.nanquantile(arrays["a"], 0.5)
    elif operation == "nanpercentile":
        return np.nanpercentile(arrays["a"], 50)

    # Array creation - extra
    elif operation == "asarray_chkfinite":
        return np.asarray_chkfinite(arrays["a"])
    elif operation == "require":
        return np.require(arrays["a"], requirements='C')

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

    # Array manipulation
    elif operation == "swapaxes":
        return np.swapaxes(arrays["a"], 0, 1)
    elif operation == "concatenate":
        return np.concatenate([arrays["a"], arrays["b"]], axis=0)
    elif operation == "stack":
        return np.stack([arrays["a"], arrays["b"]], axis=0)
    elif operation == "vstack":
        return np.vstack([arrays["a"], arrays["b"]])
    elif operation == "hstack":
        return np.hstack([arrays["a"], arrays["b"]])
    elif operation == "tile":
        return np.tile(arrays["a"], [2, 2])
    elif operation == "repeat":
        return np.repeat(arrays["a"], 2)
    elif operation == "concat":
        return np.concatenate([arrays["a"], arrays["b"]], axis=0)
    elif operation == "unstack":
        return [arr for arr in arrays["a"]]
    elif operation == "block":
        return np.block([arrays["a"], arrays["b"]])
    elif operation == "item":
        return arrays["a"].item(0)
    elif operation == "tolist":
        return arrays["a"].tolist()

    # Advanced
    elif operation == "broadcast_to":
        return np.broadcast_to(arrays["a"], arrays["target_shape"])
    elif operation == "take":
        return np.take(arrays["a"], arrays["indices"])

    # New creation functions
    elif operation == "diag":
        return np.diag(arrays["a"])
    elif operation == "tri":
        return np.tri(arrays["shape"][0], arrays["shape"][1])
    elif operation == "tril":
        return np.tril(arrays["a"])
    elif operation == "triu":
        return np.triu(arrays["a"])

    # New manipulation functions
    elif operation == "flip":
        return np.flip(arrays["a"])
    elif operation == "rot90":
        return np.rot90(arrays["a"])
    elif operation == "roll":
        return np.roll(arrays["a"], 10)
    elif operation == "pad":
        return np.pad(arrays["a"], 2)

    # Indexing functions
    elif operation == "take_along_axis":
        return np.take_along_axis(arrays["a"], arrays["b"].astype(np.intp), axis=0)
    elif operation == "compress":
        return np.compress(arrays["b"].astype(bool), arrays["a"], axis=0)
    elif operation == "diag_indices":
        return np.diag_indices(arrays["n"])
    elif operation == "tril_indices":
        return np.tril_indices(arrays["n"])
    elif operation == "triu_indices":
        return np.triu_indices(arrays["n"])
    elif operation == "indices":
        return np.indices(tuple(arrays["shape"]))
    elif operation == "ravel_multi_index":
        return np.ravel_multi_index((arrays["a"].astype(np.intp).ravel(), arrays["b"].astype(np.intp).ravel()), tuple(arrays["dims"]))
    elif operation == "unravel_index":
        return np.unravel_index(arrays["a"].astype(np.intp).ravel(), tuple(arrays["dims"]))

    # Bitwise operations
    elif operation == "bitwise_and":
        return np.bitwise_and(arrays["a"], arrays["b"])
    elif operation == "bitwise_or":
        return np.bitwise_or(arrays["a"], arrays["b"])
    elif operation == "bitwise_xor":
        return np.bitwise_xor(arrays["a"], arrays["b"])
    elif operation == "bitwise_not":
        return np.bitwise_not(arrays["a"])
    elif operation == "invert":
        return np.invert(arrays["a"])
    elif operation == "left_shift":
        return np.left_shift(arrays["a"], arrays["b"])
    elif operation == "right_shift":
        return np.right_shift(arrays["a"], arrays["b"])
    elif operation == "packbits":
        return np.packbits(arrays["a"].astype(np.uint8))
    elif operation == "unpackbits":
        return np.unpackbits(arrays["a"].astype(np.uint8))

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

    # Sorting operations
    elif operation == "sort":
        return np.sort(arrays["a"])
    elif operation == "argsort":
        return np.argsort(arrays["a"])
    elif operation == "partition":
        kth = arrays.get("kth", 0)
        return np.partition(arrays["a"], kth)
    elif operation == "argpartition":
        kth = arrays.get("kth", 0)
        return np.argpartition(arrays["a"], kth)
    elif operation == "lexsort":
        return np.lexsort((arrays["a"].ravel(), arrays["b"].ravel()))
    elif operation == "sort_complex":
        return np.sort_complex(arrays["a"])

    # Searching operations
    elif operation == "nonzero":
        return np.nonzero(arrays["a"])
    elif operation == "argwhere":
        return np.argwhere(arrays["a"])
    elif operation == "flatnonzero":
        return np.flatnonzero(arrays["a"])
    elif operation == "where":
        return np.where(arrays["a"], arrays["b"], arrays["c"])
    elif operation == "searchsorted":
        return np.searchsorted(arrays["a"].ravel(), arrays["b"].ravel())
    elif operation == "extract":
        return np.extract(arrays["condition"], arrays["a"])
    elif operation == "count_nonzero":
        return np.count_nonzero(arrays["a"])

    # Statistics operations
    elif operation == "bincount":
        return np.bincount(arrays["a"].flatten().astype(int))
    elif operation == "digitize":
        return np.digitize(arrays["a"].flatten(), arrays["b"].flatten())
    elif operation == "histogram":
        hist, _ = np.histogram(arrays["a"].flatten(), bins=10)
        return hist
    elif operation == "histogram2d":
        hist, _, _ = np.histogram2d(arrays["a"].flatten(), arrays["b"].flatten(), bins=10)
        return hist
    elif operation == "correlate":
        return np.correlate(arrays["a"].flatten(), arrays["b"].flatten(), mode="full")
    elif operation == "convolve":
        return np.convolve(arrays["a"].flatten(), arrays["b"].flatten(), mode="full")
    elif operation == "cov":
        return np.cov(arrays["a"])
    elif operation == "corrcoef":
        return np.corrcoef(arrays["a"])

    # Logic operations
    elif operation == "logical_and":
        if "b" in arrays:
            return np.logical_and(arrays["a"], arrays["b"])
        else:
            return np.logical_and(arrays["a"], arrays["scalar"])
    elif operation == "logical_or":
        if "b" in arrays:
            return np.logical_or(arrays["a"], arrays["b"])
        else:
            return np.logical_or(arrays["a"], arrays["scalar"])
    elif operation == "logical_not":
        return np.logical_not(arrays["a"])
    elif operation == "logical_xor":
        if "b" in arrays:
            return np.logical_xor(arrays["a"], arrays["b"])
        else:
            return np.logical_xor(arrays["a"], arrays["scalar"])
    elif operation == "isfinite":
        return np.isfinite(arrays["a"])
    elif operation == "isinf":
        return np.isinf(arrays["a"])
    elif operation == "isnan":
        return np.isnan(arrays["a"])
    elif operation == "isneginf":
        return np.isneginf(arrays["a"])
    elif operation == "isposinf":
        return np.isposinf(arrays["a"])
    elif operation == "isreal":
        return np.isreal(arrays["a"])
    elif operation == "signbit":
        return np.signbit(arrays["a"])
    elif operation == "copysign":
        if "b" in arrays:
            return np.copysign(arrays["a"], arrays["b"])
        else:
            return np.copysign(arrays["a"], arrays["scalar"])

    # Random operations
    elif operation == "random_random":
        return np.random.random(arrays["shape"])
    elif operation == "random_rand":
        shape = arrays["shape"]
        return np.random.rand(*shape)
    elif operation == "random_randn":
        shape = arrays["shape"]
        return np.random.randn(*shape)
    elif operation == "random_randint":
        return np.random.randint(0, 100, arrays["shape"])
    elif operation == "random_uniform":
        return np.random.uniform(0, 1, arrays["shape"])
    elif operation == "random_normal":
        return np.random.normal(0, 1, arrays["shape"])
    elif operation == "random_standard_normal":
        return np.random.standard_normal(arrays["shape"])
    elif operation == "random_exponential":
        return np.random.exponential(1, arrays["shape"])
    elif operation == "random_poisson":
        return np.random.poisson(5, arrays["shape"])
    elif operation == "random_binomial":
        return np.random.binomial(10, 0.5, arrays["shape"])
    elif operation == "random_choice":
        return np.random.choice(arrays["n"], 100)
    elif operation == "random_permutation":
        return np.random.permutation(arrays["n"])

    # Complex operations
    elif operation == "complex_zeros":
        return np.zeros(arrays["shape"], dtype=np.complex128)
    elif operation == "complex_ones":
        return np.ones(arrays["shape"], dtype=np.complex128)
    elif operation == "complex_add":
        return arrays["a"] + arrays["b"]
    elif operation == "complex_multiply":
        return arrays["a"] * arrays["b"]
    elif operation == "complex_divide":
        return arrays["a"] / arrays["b"]
    elif operation == "complex_real":
        return np.real(arrays["a"])
    elif operation == "complex_imag":
        return np.imag(arrays["a"])
    elif operation == "complex_conj":
        return np.conj(arrays["a"])
    elif operation == "complex_angle":
        return np.angle(arrays["a"])
    elif operation == "complex_abs":
        return np.abs(arrays["a"])
    elif operation == "complex_sqrt":
        return np.sqrt(arrays["a"])
    elif operation == "complex_sum":
        return np.sum(arrays["a"])
    elif operation == "complex_mean":
        return np.mean(arrays["a"])
    elif operation == "complex_prod":
        return np.prod(arrays["a"])

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
