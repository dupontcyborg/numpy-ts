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
        elif fill == "complex":
            # Create complex array with [1+1j, 2+2j, 3+3j, ...]
            size = np.prod(shape)
            arrays[key] = np.array([complex(i+1, i+1) for i in range(size)], dtype=np.complex128).reshape(shape)
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
    elif operation == "gcd":
        result = np.gcd(arrays["a"], arrays["b"])
    elif operation == "lcm":
        result = np.lcm(arrays["a"], arrays["b"])
    elif operation == "float_power":
        result = np.float_power(arrays["a"], arrays["b"])

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
    elif operation == "linalg_slogdet":
        sign, logdet = np.linalg.slogdet(arrays["a"])
        # Return both as a structured result
        result = np.array([sign, logdet])
    elif operation == "linalg_svdvals":
        result = np.linalg.svdvals(arrays["a"])
    elif operation == "linalg_multi_dot":
        matrices = [arrays["a"], arrays["b"]]
        if "c" in arrays:
            matrices.append(arrays["c"])
        result = np.linalg.multi_dot(matrices)
    elif operation == "vdot":
        result = np.vdot(arrays["a"], arrays["b"])
    elif operation == "vecdot":
        # vecdot computes dot product along the last axis
        # NumPy 2.0+ has np.linalg.vecdot, for older versions use einsum
        if hasattr(np.linalg, 'vecdot'):
            result = np.linalg.vecdot(arrays["a"], arrays["b"])
        else:
            # Fallback: use einsum to compute dot product along last axis
            result = np.einsum('...i,...i->...', arrays["a"], arrays["b"])
    elif operation == "matrix_transpose":
        # NumPy 2.0+ has .mT property, for older versions use swapaxes
        if hasattr(arrays["a"], 'mT'):
            result = arrays["a"].mT
        else:
            result = np.swapaxes(arrays["a"], -2, -1)
    elif operation == "matvec":
        # Matrix-vector multiplication: (..., M, K) @ (..., K) -> (..., M)
        # NumPy 2.0+ has np.linalg.matvec
        if hasattr(np.linalg, 'matvec'):
            result = np.linalg.matvec(arrays["a"], arrays["b"])
        else:
            result = np.matmul(arrays["a"], arrays["b"])
    elif operation == "vecmat":
        # Vector-matrix multiplication: (..., K) @ (..., K, N) -> (..., N)
        # NumPy 2.0+ has np.linalg.vecmat
        if hasattr(np.linalg, 'vecmat'):
            result = np.linalg.vecmat(arrays["a"], arrays["b"])
        else:
            result = np.matmul(arrays["a"], arrays["b"])

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
    elif operation == "nanquantile":
        result = np.nanquantile(arrays["a"], 0.5)
    elif operation == "nanpercentile":
        result = np.nanpercentile(arrays["a"], 50)

    # Array creation - extra
    elif operation == "asarray_chkfinite":
        result = np.asarray_chkfinite(arrays["a"])
    elif operation == "require":
        # np.require with 'C' requirement returns a C-contiguous array
        result = np.require(arrays["a"], requirements='C')

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
    elif operation == "concat":
        # NumPy's concat is np.concatenate
        result = np.concatenate([arrays["a"], arrays["b"]], axis=0)
    elif operation == "unstack":
        # Unstack splits along the specified axis (default 0)
        # Return list of arrays, but for validation we'll return the stacked result
        # since comparing lists of arrays is complex
        result = [arr for arr in arrays["a"]]
        # Return as a list - validation will handle this specially
        result = result
    elif operation == "block":
        # np.block assembles from nested lists
        result = np.block([arrays["a"], arrays["b"]])
    elif operation == "item":
        # Extract scalar at given index
        result = arrays["a"].item(0)
    elif operation == "tolist":
        # Convert array to nested Python list
        result = arrays["a"].tolist()

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
    elif operation == "bitwise_count":
        result = np.bitwise_count(arrays["a"])

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
    elif operation == "argwhere":
        result = np.argwhere(arrays["a"])
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

    # Statistics operations
    elif operation == "bincount":
        result = np.bincount(arrays["a"].flatten().astype(int))
    elif operation == "digitize":
        result = np.digitize(arrays["a"].flatten(), arrays["b"].flatten())
    elif operation == "histogram":
        result, _ = np.histogram(arrays["a"].flatten(), bins=10)
    elif operation == "histogram2d":
        result, _, _ = np.histogram2d(arrays["a"].flatten(), arrays["b"].flatten(), bins=10)
    elif operation == "correlate":
        result = np.correlate(arrays["a"].flatten(), arrays["b"].flatten(), mode="full")
    elif operation == "convolve":
        result = np.convolve(arrays["a"].flatten(), arrays["b"].flatten(), mode="full")
    elif operation == "cov":
        result = np.cov(arrays["a"])
    elif operation == "corrcoef":
        result = np.corrcoef(arrays["a"])
    elif operation == "histogram_bin_edges":
        result = np.histogram_bin_edges(arrays["a"].flatten(), bins=10)
    elif operation == "trapezoid":
        result = np.trapezoid(arrays["a"].flatten())

    # Set operations
    elif operation == "trim_zeros":
        result = np.trim_zeros(arrays["a"].flatten())
    elif operation == "unique_values":
        result = np.unique(arrays["a"].flatten())
    elif operation == "unique_counts":
        values, counts = np.unique(arrays["a"].flatten(), return_counts=True)
        result = {"values": values.tolist(), "counts": counts.tolist()}

    # Logic operations
    elif operation == "logical_and":
        if "b" in arrays:
            result = np.logical_and(arrays["a"], arrays["b"]).astype(np.uint8)
        else:
            result = np.logical_and(arrays["a"], arrays["scalar"]).astype(np.uint8)
    elif operation == "logical_or":
        if "b" in arrays:
            result = np.logical_or(arrays["a"], arrays["b"]).astype(np.uint8)
        else:
            result = np.logical_or(arrays["a"], arrays["scalar"]).astype(np.uint8)
    elif operation == "logical_not":
        result = np.logical_not(arrays["a"]).astype(np.uint8)
    elif operation == "logical_xor":
        if "b" in arrays:
            result = np.logical_xor(arrays["a"], arrays["b"]).astype(np.uint8)
        else:
            result = np.logical_xor(arrays["a"], arrays["scalar"]).astype(np.uint8)
    elif operation == "isfinite":
        result = np.isfinite(arrays["a"]).astype(np.uint8)
    elif operation == "isinf":
        result = np.isinf(arrays["a"]).astype(np.uint8)
    elif operation == "isnan":
        result = np.isnan(arrays["a"]).astype(np.uint8)
    elif operation == "isneginf":
        result = np.isneginf(arrays["a"]).astype(np.uint8)
    elif operation == "isposinf":
        result = np.isposinf(arrays["a"]).astype(np.uint8)
    elif operation == "isreal":
        result = np.isreal(arrays["a"]).astype(np.uint8)
    elif operation == "signbit":
        result = np.signbit(arrays["a"]).astype(np.uint8)
    elif operation == "copysign":
        if "b" in arrays:
            result = np.copysign(arrays["a"], arrays["b"])
        else:
            result = np.copysign(arrays["a"], arrays["scalar"])

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

    # Complex operations
    elif operation == "complex_zeros":
        result = np.zeros(arrays["shape"], dtype=np.complex128)
    elif operation == "complex_ones":
        result = np.ones(arrays["shape"], dtype=np.complex128)
    elif operation == "complex_add":
        result = arrays["a"] + arrays["b"]
    elif operation == "complex_multiply":
        result = arrays["a"] * arrays["b"]
    elif operation == "complex_divide":
        result = arrays["a"] / arrays["b"]
    elif operation == "complex_real":
        result = np.real(arrays["a"])
    elif operation == "complex_imag":
        result = np.imag(arrays["a"])
    elif operation == "complex_conj":
        result = np.conj(arrays["a"])
    elif operation == "complex_angle":
        result = np.angle(arrays["a"])
    elif operation == "complex_abs":
        result = np.abs(arrays["a"])
    elif operation == "complex_sqrt":
        result = np.sqrt(arrays["a"])
    elif operation == "complex_sum":
        result = np.sum(arrays["a"])
    elif operation == "complex_mean":
        result = np.mean(arrays["a"])
    elif operation == "complex_prod":
        result = np.prod(arrays["a"])

    # Other Math operations
    elif operation == "clip":
        result = np.clip(arrays["a"], 10, 100)
    elif operation == "maximum":
        result = np.maximum(arrays["a"], arrays["b"])
    elif operation == "minimum":
        result = np.minimum(arrays["a"], arrays["b"])
    elif operation == "fmax":
        result = np.fmax(arrays["a"], arrays["b"])
    elif operation == "fmin":
        result = np.fmin(arrays["a"], arrays["b"])
    elif operation == "nan_to_num":
        result = np.nan_to_num(arrays["a"])
    elif operation == "interp":
        result = np.interp(arrays["x"], arrays["xp"], arrays["fp"])
    elif operation == "unwrap":
        result = np.unwrap(arrays["a"])
    elif operation == "sinc":
        result = np.sinc(arrays["a"])
    elif operation == "i0":
        result = np.i0(arrays["a"])

    # Polynomial operations
    elif operation == "poly":
        result = np.poly(arrays["a"])
    elif operation == "polyadd":
        result = np.polyadd(arrays["a"], arrays["b"])
    elif operation == "polyder":
        result = np.polyder(arrays["a"])
    elif operation == "polydiv":
        q, r = np.polydiv(arrays["a"], arrays["b"])
        result = q  # Return just quotient for validation
    elif operation == "polyfit":
        result = np.polyfit(arrays["a"], arrays["b"], 2)
    elif operation == "polyint":
        result = np.polyint(arrays["a"])
    elif operation == "polymul":
        result = np.polymul(arrays["a"], arrays["b"])
    elif operation == "polysub":
        result = np.polysub(arrays["a"], arrays["b"])
    elif operation == "polyval":
        result = np.polyval(arrays["a"], arrays["b"])
    elif operation == "roots":
        result = np.roots(arrays["a"])

    # Type checking operations (return booleans/strings, not arrays)
    elif operation == "can_cast":
        result = np.can_cast("int32", "float64")
    elif operation == "result_type":
        result = str(np.result_type("int32", "float64"))
    elif operation == "min_scalar_type":
        result = str(np.min_scalar_type(1000))
    elif operation == "issubdtype":
        result = np.issubdtype(np.int32, np.integer)

    # FFT operations
    elif operation == "fft":
        result = np.fft.fft(arrays["a"])
    elif operation == "ifft":
        result = np.fft.ifft(arrays["a"])
    elif operation == "fft2":
        result = np.fft.fft2(arrays["a"])
    elif operation == "ifft2":
        result = np.fft.ifft2(arrays["a"])
    elif operation == "fftn":
        result = np.fft.fftn(arrays["a"])
    elif operation == "ifftn":
        result = np.fft.ifftn(arrays["a"])
    elif operation == "rfft":
        result = np.fft.rfft(arrays["a"])
    elif operation == "irfft":
        result = np.fft.irfft(arrays["a"])
    elif operation == "rfft2":
        result = np.fft.rfft2(arrays["a"])
    elif operation == "irfft2":
        result = np.fft.irfft2(arrays["a"])
    elif operation == "rfftn":
        result = np.fft.rfftn(arrays["a"])
    elif operation == "irfftn":
        result = np.fft.irfftn(arrays["a"])
    elif operation == "hfft":
        result = np.fft.hfft(arrays["a"])
    elif operation == "ihfft":
        result = np.fft.ihfft(arrays["a"])
    elif operation == "fftfreq":
        result = np.fft.fftfreq(arrays["n"])
    elif operation == "rfftfreq":
        result = np.fft.rfftfreq(arrays["n"])
    elif operation == "fftshift":
        result = np.fft.fftshift(arrays["a"])
    elif operation == "ifftshift":
        result = np.fft.ifftshift(arrays["a"])

    else:
        raise ValueError(f"Unknown operation: {operation}")

    # Convert result to JSON-serializable format
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], np.ndarray):
        # Handle list of arrays (e.g., from unstack)
        return [{"shape": list(arr.shape), "data": arr.tolist()} for arr in result]
    elif isinstance(result, np.ndarray):
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
