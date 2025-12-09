#!/usr/bin/env python3
"""
Validation script for benchmark correctness
Runs operations with NumPy and returns results for comparison
"""

import json
import sys
import numpy as np


def setup_arrays(setup_config):
    """Create NumPy arrays from setup configuration"""
    arrays = {}

    for key, config in setup_config.items():
        shape = config["shape"]
        dtype = config.get("dtype", "float64")
        fill = config.get("fill", "zeros")
        value = config.get("value")

        # Handle scalar values
        if key in ["n", "axis", "new_shape", "shape", "fill_value", "target_shape", "dims", "kth"]:
            arrays[key] = shape[0]
            if key in ["new_shape", "shape", "target_shape", "dims"]:
                arrays[key] = shape
            continue

        # Handle indices array
        if key == "indices":
            arrays[key] = shape
            continue

        # Handle string values (like einsum subscripts)
        if key == "subscripts":
            arrays[key] = value
            continue

        # Map dtype names
        np_dtype = dtype
        if dtype == "bool":
            np_dtype = "bool_"

        # Create arrays
        if value is not None:
            arrays[key] = np.full(shape, value, dtype=np_dtype)
        elif fill == "zeros":
            arrays[key] = np.zeros(shape, dtype=np_dtype)
        elif fill == "ones":
            arrays[key] = np.ones(shape, dtype=np_dtype)
        elif fill in ["random", "arange"]:
            size = np.prod(shape)
            flat = np.arange(0, size, 1, dtype=np_dtype)
            arrays[key] = flat.reshape(shape)
        elif fill == "invertible":
            # Create an invertible matrix: arange + n*I (diagonally dominant)
            n = shape[0]
            size = np.prod(shape)
            flat = np.arange(0, size, 1, dtype=np_dtype)
            arange_matrix = flat.reshape(shape)
            identity = np.eye(n, dtype=np_dtype)
            arrays[key] = arange_matrix + identity * (n * n)

    return arrays


def run_operation(spec):
    """Run a single operation and return result"""
    arrays = setup_arrays(spec["setup"])
    operation = spec["operation"]

    # Execute operation
    if operation == "zeros":
        result = np.zeros(arrays["shape"])
    elif operation == "ones":
        result = np.ones(arrays["shape"])
    elif operation == "arange":
        result = np.arange(0, arrays["n"])
    elif operation == "linspace":
        result = np.linspace(0, 100, arrays["n"])
    elif operation == "logspace":
        result = np.logspace(0, 3, arrays["n"])
    elif operation == "geomspace":
        result = np.geomspace(1, 1000, arrays["n"])
    elif operation == "eye":
        result = np.eye(arrays["n"])
    elif operation == "identity":
        result = np.identity(arrays["n"])
    elif operation == "empty":
        result = np.empty(arrays["shape"])
        # For empty, just return zeros (we can't compare uninitialized data)
        result = np.zeros(arrays["shape"])
    elif operation == "full":
        result = np.full(arrays["shape"], arrays["fill_value"])
    elif operation == "copy":
        result = arrays["a"].copy()
    elif operation == "zeros_like":
        result = np.zeros_like(arrays["a"])

    # Arithmetic
    elif operation == "add":
        result = arrays["a"] + (
            arrays.get("b") if "b" in arrays else arrays.get("scalar")
        )
    elif operation == "multiply":
        result = arrays["a"] * (
            arrays.get("b") if "b" in arrays else arrays.get("scalar")
        )
    elif operation == "mod":
        divisor = arrays.get("b") if "b" in arrays else arrays.get("scalar")
        result = np.mod(arrays["a"], divisor)
    elif operation == "floor_divide":
        divisor = arrays.get("b") if "b" in arrays else arrays.get("scalar")
        result = np.floor_divide(arrays["a"], divisor)
    elif operation == "reciprocal":
        result = np.reciprocal(arrays["a"])
    elif operation == "cbrt":
        result = np.cbrt(arrays["a"])
    elif operation == "fabs":
        result = np.fabs(arrays["a"])
    elif operation == "divmod":
        divisor = arrays.get("b") if "b" in arrays else arrays.get("scalar")
        q, r = np.divmod(arrays["a"], divisor)
        result = q  # Just return quotient for validation

    # Math
    elif operation == "sqrt":
        result = np.sqrt(arrays["a"])
    elif operation == "power":
        result = np.power(arrays["a"], 2)
    elif operation == "absolute":
        result = np.absolute(arrays["a"])
    elif operation == "negative":
        result = np.negative(arrays["a"])
    elif operation == "sign":
        result = np.sign(arrays["a"])

    # Trigonometric
    elif operation == "sin":
        result = np.sin(arrays["a"])
    elif operation == "cos":
        result = np.cos(arrays["a"])
    elif operation == "tan":
        result = np.tan(arrays["a"])
    elif operation == "arctan2":
        result = np.arctan2(arrays["a"], arrays["b"])
    elif operation == "hypot":
        result = np.hypot(arrays["a"], arrays["b"])

    # Hyperbolic
    elif operation == "sinh":
        result = np.sinh(arrays["a"])
    elif operation == "cosh":
        result = np.cosh(arrays["a"])
    elif operation == "tanh":
        result = np.tanh(arrays["a"])

    # Exponential
    elif operation == "exp":
        result = np.exp(arrays["a"])
    elif operation == "exp2":
        result = np.exp2(arrays["a"])
    elif operation == "expm1":
        result = np.expm1(arrays["a"])
    elif operation == "log":
        result = np.log(arrays["a"])
    elif operation == "log2":
        result = np.log2(arrays["a"])
    elif operation == "log10":
        result = np.log10(arrays["a"])
    elif operation == "log1p":
        result = np.log1p(arrays["a"])
    elif operation == "logaddexp":
        result = np.logaddexp(arrays["a"], arrays["b"])
    elif operation == "logaddexp2":
        result = np.logaddexp2(arrays["a"], arrays["b"])

    # Gradient
    elif operation == "diff":
        result = np.diff(arrays["a"])
    elif operation == "ediff1d":
        result = np.ediff1d(arrays["a"])
    elif operation == "gradient":
        # np.gradient can return list of arrays for multi-dimensional input
        # For simplicity, when input is 1D, it returns single array
        result = np.gradient(arrays["a"])
    elif operation == "cross":
        result = np.cross(arrays["a"], arrays["b"])

    # Linalg
    elif operation == "dot":
        result = np.dot(arrays["a"], arrays["b"])
    elif operation == "inner":
        result = np.inner(arrays["a"], arrays["b"])
    elif operation == "outer":
        result = np.outer(arrays["a"], arrays["b"])
    elif operation == "matmul":
        result = arrays["a"] @ arrays["b"]
    elif operation == "trace":
        result = np.trace(arrays["a"])
    elif operation == "transpose":
        result = arrays["a"].T
    elif operation == "diagonal":
        result = np.diagonal(arrays["a"])
    elif operation == "kron":
        result = np.kron(arrays["a"], arrays["b"])
    elif operation == "einsum":
        result = np.einsum(arrays["subscripts"], arrays["a"], arrays["b"])
    elif operation == "deg2rad":
        result = np.deg2rad(arrays["a"])
    elif operation == "rad2deg":
        result = np.rad2deg(arrays["a"])

    # numpy.linalg module operations
    elif operation == "linalg_det":
        result = np.linalg.det(arrays["a"])
    elif operation == "linalg_inv":
        result = np.linalg.inv(arrays["a"])
    elif operation == "linalg_solve":
        result = np.linalg.solve(arrays["a"], arrays["b"])
    elif operation == "linalg_qr":
        q, r = np.linalg.qr(arrays["a"])
        result = q  # Return just Q for validation
    elif operation == "linalg_cholesky":
        # Create positive definite matrix: A^T * A + n*I
        a = arrays["a"]
        n = a.shape[0]
        posdef = a.T @ a + np.eye(n) * n
        result = np.linalg.cholesky(posdef)
    elif operation == "linalg_svd":
        u, s, vt = np.linalg.svd(arrays["a"])
        result = s  # Return just singular values for validation
    elif operation == "linalg_eig":
        w, v = np.linalg.eig(arrays["a"])
        result = w  # Return just eigenvalues for validation
    elif operation == "linalg_eigh":
        # Create symmetric matrix: (A + A^T) / 2
        a = arrays["a"]
        sym = (a + a.T) / 2
        w, v = np.linalg.eigh(sym)
        result = w  # Return just eigenvalues for validation
    elif operation == "linalg_norm":
        result = np.linalg.norm(arrays["a"])
    elif operation == "linalg_matrix_rank":
        result = np.linalg.matrix_rank(arrays["a"])
    elif operation == "linalg_pinv":
        result = np.linalg.pinv(arrays["a"])
    elif operation == "linalg_cond":
        result = np.linalg.cond(arrays["a"])
    elif operation == "linalg_matrix_power":
        result = np.linalg.matrix_power(arrays["a"], 3)
    elif operation == "linalg_lstsq":
        x, residuals, rank, s = np.linalg.lstsq(arrays["a"], arrays["b"], rcond=None)
        result = x  # Return just solution for validation
    elif operation == "linalg_cross":
        result = np.cross(arrays["a"], arrays["b"])

    # Reductions
    elif operation == "sum":
        result = arrays["a"].sum(axis=arrays.get("axis"))
    elif operation == "mean":
        result = arrays["a"].mean()
    elif operation == "max":
        result = arrays["a"].max()
    elif operation == "min":
        result = arrays["a"].min()
    elif operation == "prod":
        result = arrays["a"].prod()
    elif operation == "argmin":
        result = arrays["a"].argmin()
    elif operation == "argmax":
        result = arrays["a"].argmax()
    elif operation == "var":
        result = arrays["a"].var()
    elif operation == "std":
        result = arrays["a"].std()
    elif operation == "all":
        result = arrays["a"].all()
    elif operation == "any":
        result = arrays["a"].any()

    # New reduction functions
    elif operation == "cumsum":
        result = arrays["a"].cumsum()
    elif operation == "cumprod":
        result = arrays["a"].cumprod()
    elif operation == "ptp":
        result = np.ptp(arrays["a"])
    elif operation == "median":
        result = np.median(arrays["a"])
    elif operation == "percentile":
        result = np.percentile(arrays["a"], 50)
    elif operation == "quantile":
        result = np.quantile(arrays["a"], 0.5)
    elif operation == "average":
        result = np.average(arrays["a"])
    elif operation == "nansum":
        result = np.nansum(arrays["a"])
    elif operation == "nanmean":
        result = np.nanmean(arrays["a"])
    elif operation == "nanmin":
        result = np.nanmin(arrays["a"])
    elif operation == "nanmax":
        result = np.nanmax(arrays["a"])

    # Reshape
    elif operation == "reshape":
        result = arrays["a"].reshape(*arrays["new_shape"])
    elif operation == "flatten":
        result = arrays["a"].flatten()
    elif operation == "ravel":
        result = arrays["a"].ravel()

    # Array manipulation
    elif operation == "swapaxes":
        result = np.swapaxes(arrays["a"], 0, 1)
    elif operation == "concatenate":
        result = np.concatenate([arrays["a"], arrays["b"]], axis=0)
    elif operation == "stack":
        result = np.stack([arrays["a"], arrays["b"]], axis=0)
    elif operation == "vstack":
        result = np.vstack([arrays["a"], arrays["b"]])
    elif operation == "hstack":
        result = np.hstack([arrays["a"], arrays["b"]])
    elif operation == "tile":
        result = np.tile(arrays["a"], [2, 2])
    elif operation == "repeat":
        result = np.repeat(arrays["a"], 2)

    # Advanced
    elif operation == "broadcast_to":
        result = np.broadcast_to(arrays["a"], arrays["target_shape"])
    elif operation == "take":
        result = np.take(arrays["a"], arrays["indices"])

    # New creation functions
    elif operation == "diag":
        result = np.diag(arrays["a"])
    elif operation == "tri":
        result = np.tri(arrays["shape"][0], arrays["shape"][1])
    elif operation == "tril":
        result = np.tril(arrays["a"])
    elif operation == "triu":
        result = np.triu(arrays["a"])

    # New manipulation functions
    elif operation == "flip":
        result = np.flip(arrays["a"])
    elif operation == "rot90":
        result = np.rot90(arrays["a"])
    elif operation == "roll":
        result = np.roll(arrays["a"], 10)
    elif operation == "pad":
        result = np.pad(arrays["a"], 2)

    # Indexing functions
    elif operation == "take_along_axis":
        result = np.take_along_axis(arrays["a"], arrays["b"].astype(np.intp), axis=0)
    elif operation == "compress":
        result = np.compress(arrays["b"].astype(bool), arrays["a"], axis=0)
    elif operation == "diag_indices":
        idx = np.diag_indices(arrays["n"])
        # Stack them for comparison
        result = np.stack([idx[0], idx[1]], axis=0)
    elif operation == "tril_indices":
        idx = np.tril_indices(arrays["n"])
        result = np.stack([idx[0], idx[1]], axis=0)
    elif operation == "triu_indices":
        idx = np.triu_indices(arrays["n"])
        result = np.stack([idx[0], idx[1]], axis=0)
    elif operation == "indices":
        result = np.indices(tuple(arrays["shape"]))
    elif operation == "ravel_multi_index":
        result = np.ravel_multi_index((arrays["a"].astype(np.intp).ravel(), arrays["b"].astype(np.intp).ravel()), tuple(arrays["dims"]))
    elif operation == "unravel_index":
        idx = np.unravel_index(arrays["a"].astype(np.intp).ravel(), tuple(arrays["dims"]))
        result = np.stack([np.array(i) for i in idx], axis=0)

    # Bitwise operations
    elif operation == "bitwise_and":
        result = np.bitwise_and(arrays["a"], arrays["b"])
    elif operation == "bitwise_or":
        result = np.bitwise_or(arrays["a"], arrays["b"])
    elif operation == "bitwise_xor":
        result = np.bitwise_xor(arrays["a"], arrays["b"])
    elif operation == "bitwise_not":
        result = np.bitwise_not(arrays["a"])
    elif operation == "invert":
        result = np.invert(arrays["a"])
    elif operation == "left_shift":
        shift = arrays.get("b") if "b" in arrays else 2
        result = np.left_shift(arrays["a"], shift)
    elif operation == "right_shift":
        shift = arrays.get("b") if "b" in arrays else 2
        result = np.right_shift(arrays["a"], shift)
    elif operation == "packbits":
        result = np.packbits(arrays["a"].astype(np.uint8))
    elif operation == "unpackbits":
        result = np.unpackbits(arrays["a"].astype(np.uint8))

    # Sorting operations
    elif operation == "sort":
        result = np.sort(arrays["a"])
    elif operation == "argsort":
        result = np.argsort(arrays["a"])
    elif operation == "partition":
        kth = arrays.get("kth", 0)
        result = np.partition(arrays["a"], kth)
    elif operation == "argpartition":
        kth = arrays.get("kth", 0)
        result = np.argpartition(arrays["a"], kth)
    elif operation == "lexsort":
        result = np.lexsort((arrays["a"].ravel(), arrays["b"].ravel()))
    elif operation == "sort_complex":
        result = np.sort_complex(arrays["a"]).real  # Return real part for comparison

    # Searching operations
    elif operation == "nonzero":
        idx = np.nonzero(arrays["a"])
        result = np.stack(idx, axis=0)
    elif operation == "flatnonzero":
        result = np.flatnonzero(arrays["a"])
    elif operation == "where":
        result = np.where(arrays["a"], arrays["b"], arrays["c"])
    elif operation == "searchsorted":
        result = np.searchsorted(arrays["a"].ravel(), arrays["b"].ravel())
    elif operation == "extract":
        result = np.extract(arrays["condition"], arrays["a"])
    elif operation == "count_nonzero":
        result = np.count_nonzero(arrays["a"])

    # Random operations - return placeholder with correct shape
    # (actual values will differ between NumPy and numpy-ts)
    elif operation == "random_random":
        result = np.random.random(tuple(arrays["shape"]))
    elif operation == "random_rand":
        result = np.random.rand(*arrays["shape"])
    elif operation == "random_randn":
        result = np.random.randn(*arrays["shape"])
    elif operation == "random_randint":
        result = np.random.randint(0, 100, tuple(arrays["shape"]))
    elif operation == "random_uniform":
        result = np.random.uniform(0, 1, tuple(arrays["shape"]))
    elif operation == "random_normal":
        result = np.random.normal(0, 1, tuple(arrays["shape"]))
    elif operation == "random_standard_normal":
        result = np.random.standard_normal(tuple(arrays["shape"]))
    elif operation == "random_exponential":
        result = np.random.exponential(1, tuple(arrays["shape"]))
    elif operation == "random_poisson":
        result = np.random.poisson(5, tuple(arrays["shape"]))
    elif operation == "random_binomial":
        result = np.random.binomial(10, 0.5, tuple(arrays["shape"]))
    elif operation == "random_choice":
        result = np.random.choice(arrays["n"], 100)
    elif operation == "random_permutation":
        result = np.random.permutation(arrays["n"])

    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Convert result to JSON-serializable format
    if isinstance(result, np.ndarray):
        # Handle complex arrays by converting to real
        if np.iscomplexobj(result):
            result = result.real
        return {"shape": list(result.shape), "data": result.tolist()}
    elif isinstance(result, (np.integer, np.floating)):
        return float(result)
    elif isinstance(result, np.bool_):
        return bool(result)
    elif isinstance(result, (complex, np.complexfloating)):
        return float(result.real)
    else:
        return result


import math

def serialize_value(val):
    """Recursively serialize values, handling Infinity, NaN, and complex numbers"""
    if isinstance(val, dict):
        return {k: serialize_value(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple)):
        return [serialize_value(v) for v in val]
    elif isinstance(val, complex):
        # Return real part for comparison (complex eigenvalues differ)
        return serialize_value(val.real)
    elif isinstance(val, (np.complexfloating, np.complex64, np.complex128)):
        return serialize_value(float(val.real))
    elif isinstance(val, float):
        if math.isnan(val):
            return "__NaN__"
        elif math.isinf(val):
            return "__Infinity__" if val > 0 else "__-Infinity__"
        return val
    elif isinstance(val, (np.floating, np.integer)):
        return serialize_value(float(val))
    return val


def main():
    # Read specs from stdin
    input_data = json.loads(sys.stdin.read())
    specs = input_data["specs"]

    results = []

    for spec in specs:
        try:
            result = run_operation(spec)
            # Serialize to handle Infinity/NaN
            result = serialize_value(result)
            results.append(result)
        except Exception as e:
            print(f"Error running {spec['name']}: {e}", file=sys.stderr)
            results.append(None)

    # Output results as JSON
    print(json.dumps(results))


if __name__ == "__main__":
    main()
