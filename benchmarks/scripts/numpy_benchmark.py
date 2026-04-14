#!/usr/bin/env python3
"""
NumPy benchmark script with auto-calibration

Improved benchmarking approach:
- Auto-calibrates to run enough ops to hit minimum measurement time (100ms)
- Runs operations in batches to reduce measurement overhead
- Provides ops/sec for easier interpretation
- Uses multiple samples for statistical robustness
- Operation dispatch uses a dict lookup instead of a long if/elif chain,
  turning ~400 comparisons into an O(1) access.
"""

import gc
import io
import json
import sys
import time
import traceback
from typing import Any, Callable, Dict

import numpy as np

# Require NumPy 2.0+
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))
if NUMPY_VERSION < (2, 0):
    print(f"Error: NumPy 2.0+ is required for benchmarks. Found NumPy {np.__version__}", file=sys.stderr)
    print("Please upgrade: pip install --upgrade 'numpy>=2.0'", file=sys.stderr)
    sys.exit(1)

# Benchmark configuration (can be overridden from stdin)
MIN_SAMPLE_TIME_MS = 100  # Minimum time per sample (reduces noise)
TARGET_SAMPLES = 5  # Number of samples to collect for statistics


def setup_arrays(setup: Dict[str, Any], operation: str = None) -> Dict[str, np.ndarray]:
    """Create arrays based on setup specification"""
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
            # Store dtype for operations that need it (e.g., randint)
            if key == "shape" and dtype != "float64":
                arrays["dtype"] = dtype
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
        elif fill_type == "shuffled":
            # Deterministic Fisher-Yates shuffle of arange using LCG (seed=42)
            # Use arange with target dtype first (wraps for narrow types), then shuffle
            size = int(np.prod(shape))
            wrapped = np.arange(size, dtype=dtype)
            vals = wrapped.tolist()
            seed = 42
            for i in range(size - 1, 0, -1):
                seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                j = seed % (i + 1)
                vals[i], vals[j] = vals[j], vals[i]
            arrays[key] = np.array(vals, dtype=dtype).reshape(shape)
        elif fill_type == "invertible":
            # Create an invertible matrix: arange + n*I (diagonally dominant)
            n = shape[0]
            arange_matrix = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            identity = np.eye(n, dtype=dtype)
            arrays[key] = arange_matrix + identity * (n * n)
        elif fill_type in ("complex", "complex_small"):
            # 'complex': [1+1j, 2+2j, ...], 'complex_small': modular values to avoid overflow
            size = int(np.prod(shape))
            cdtype = np.complex64 if dtype == "complex64" else np.complex128
            if fill_type == "complex_small":
                arrays[key] = np.array([complex((i % 10) + 1, (i % 10) + 1) for i in range(size)], dtype=cdtype).reshape(shape)
            else:
                arrays[key] = np.array([complex(i+1, i+1) for i in range(size)], dtype=cdtype).reshape(shape)

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

    if operation == "linalg_cholesky" and "a" in arrays:
        a = arrays["a"]
        n = a.shape[0]
        ata = a.T @ a
        arrays["_posdef"] = ata + np.eye(n, dtype=a.dtype) * np.trace(ata)

    return arrays


# ---------------------------------------------------------------------------
# Operation dispatch table
#
# Each entry is `name -> fn(arrays_dict) -> result`.  Using a dict turns
# dispatch from ~400 elif comparisons into an O(1) lookup.
# ---------------------------------------------------------------------------

def _divmod(a):
    q, _ = np.divmod(a["a"], a["b"])
    return q


def _frexp(a):
    m, _ = np.frexp(a["a"])
    return m


def _modf(a):
    f, _ = np.modf(a["a"])
    return f


def _linalg_eigh(a):
    arr = a["a"]
    sym = (arr + arr.T) / 2
    return np.linalg.eigh(sym)


def _linalg_multi_dot(a):
    matrices = [a["a"], a["b"]]
    if "c" in a:
        matrices.append(a["c"])
    return np.linalg.multi_dot(matrices)


def _vecdot(a):
    if hasattr(np.linalg, 'vecdot'):
        return np.linalg.vecdot(a["a"], a["b"])
    return np.einsum('...i,...i->...', a["a"], a["b"])


def _matrix_transpose(a):
    arr = a["a"]
    if hasattr(arr, 'mT'):
        return arr.mT
    return np.swapaxes(arr, -2, -1)


def _matvec(a):
    if hasattr(np.linalg, 'matvec'):
        return np.linalg.matvec(a["a"], a["b"])
    return np.matmul(a["a"], a["b"])


def _vecmat(a):
    if hasattr(np.linalg, 'vecmat'):
        return np.linalg.vecmat(a["a"], a["b"])
    return np.matmul(a["a"], a["b"])


def _serialize_npy(a):
    buffer = io.BytesIO()
    np.save(buffer, a["a"])
    return buffer.getvalue()


def _parse_npy(a):
    return np.load(io.BytesIO(a["_npyBytes"]))


def _serialize_npz(a):
    buffer = io.BytesIO()
    np.savez(buffer, **a["_npzArrays"])
    return buffer.getvalue()


def _parse_npz(a):
    return np.load(io.BytesIO(a["_npzBytes"]))


def _unique_counts(a):
    values, counts = np.unique(a["a"].flatten(), return_counts=True)
    return (values, counts)


def _logical_and(a):
    return np.logical_and(a["a"], a["b"]) if "b" in a else np.logical_and(a["a"], a["scalar"])


def _logical_or(a):
    return np.logical_or(a["a"], a["b"]) if "b" in a else np.logical_or(a["a"], a["scalar"])


def _logical_xor(a):
    return np.logical_xor(a["a"], a["b"]) if "b" in a else np.logical_xor(a["a"], a["scalar"])


def _copysign(a):
    return np.copysign(a["a"], a["b"]) if "b" in a else np.copysign(a["a"], a["scalar"])


def _random_dirichlet(a):
    shape = a["shape"]
    alpha = np.ones(10)
    if isinstance(shape, (list, tuple)):
        size = shape[0] if len(shape) == 1 else shape
    else:
        size = shape
    return np.random.dirichlet(alpha, size)


def _gen(method, *args):
    """Build a Generator-method callable that lazily creates an rng each call."""
    def fn(a):
        rng = np.random.default_rng(42)
        return getattr(rng, method)(*args, tuple(a["shape"]))
    return fn


def _gen_integers(a):
    rng = np.random.default_rng(42)
    return rng.integers(0, 100, tuple(a["shape"]))


def _gen_permutation(a):
    rng = np.random.default_rng(42)
    return rng.permutation(a["n"])


def _ravel_multi_index(a):
    return np.ravel_multi_index(
        (a["a"].astype(np.intp).ravel(), a["b"].astype(np.intp).ravel()),
        tuple(a["dims"]),
    )


def _unravel_index(a):
    return np.unravel_index(a["a"].astype(np.intp).ravel(), tuple(a["dims"]))


def _histogram(a):
    hist, _ = np.histogram(a["a"].flatten(), bins=10)
    return hist


def _histogram2d(a):
    hist, _, _ = np.histogram2d(a["a"].flatten(), a["b"].flatten(), bins=10)
    return hist


OPERATIONS: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    # Array creation
    "zeros": lambda a: np.zeros(a["shape"]),
    "ones": lambda a: np.ones(a["shape"]),
    "empty": lambda a: np.empty(a["shape"]),
    "full": lambda a: np.full(a["shape"], a["fill_value"]),
    "arange": lambda a: np.arange(a["n"]),
    "linspace": lambda a: np.linspace(0, 100, a["n"]),
    "logspace": lambda a: np.logspace(0, 3, a["n"]),
    "geomspace": lambda a: np.geomspace(1, 1000, a["n"]),
    "eye": lambda a: np.eye(a["n"]),
    "identity": lambda a: np.identity(a["n"]),
    "copy": lambda a: np.copy(a["a"]),
    "zeros_like": lambda a: np.zeros_like(a["a"]),
    "ones_like": lambda a: np.ones_like(a["a"]),
    "empty_like": lambda a: np.empty_like(a["a"]),
    "full_like": lambda a: np.full_like(a["a"], 7),

    # Arithmetic
    "add": lambda a: a["a"] + a["b"],
    "subtract": lambda a: a["a"] - a["b"],
    "multiply": lambda a: a["a"] * a["b"],
    "divide": lambda a: a["a"] / a["b"],
    "mod": lambda a: np.mod(a["a"], a["b"]),
    "floor_divide": lambda a: np.floor_divide(a["a"], a["b"]),
    "reciprocal": lambda a: np.reciprocal(a["a"]),
    "positive": lambda a: np.positive(a["a"]),
    "cbrt": lambda a: np.cbrt(a["a"]),
    "fabs": lambda a: np.fabs(a["a"]),
    "divmod": _divmod,
    "gcd": lambda a: np.gcd(a["a"], a["b"]),
    "lcm": lambda a: np.lcm(a["a"], a["b"]),
    "float_power": lambda a: np.float_power(a["a"], a["b"]),
    "square": lambda a: np.square(a["a"]),
    "remainder": lambda a: np.remainder(a["a"], a["b"]),
    "heaviside": lambda a: np.heaviside(a["a"], a["b"]),
    "fmod": lambda a: np.fmod(a["a"], a["b"]),
    "frexp": _frexp,
    "ldexp": lambda a: np.ldexp(a["a"], np.asarray(a["b"], dtype=np.int32)),
    "modf": _modf,

    # Math
    "sqrt": lambda a: np.sqrt(a["a"]),
    "power": lambda a: np.power(a["a"], a["b"]),
    "absolute": lambda a: np.absolute(a["a"]),
    "negative": lambda a: np.negative(a["a"]),
    "sign": lambda a: np.sign(a["a"]),

    # Trig
    "sin": lambda a: np.sin(a["a"]),
    "cos": lambda a: np.cos(a["a"]),
    "tan": lambda a: np.tan(a["a"]),
    "arcsin": lambda a: np.arcsin(a["a"]),
    "arccos": lambda a: np.arccos(a["a"]),
    "arctan": lambda a: np.arctan(a["a"]),
    "arctan2": lambda a: np.arctan2(a["a"], a["b"]),
    "hypot": lambda a: np.hypot(a["a"], a["b"]),

    # Hyperbolic
    "sinh": lambda a: np.sinh(a["a"]),
    "cosh": lambda a: np.cosh(a["a"]),
    "tanh": lambda a: np.tanh(a["a"]),
    "arcsinh": lambda a: np.arcsinh(a["a"]),
    "arccosh": lambda a: np.arccosh(a["a"]),
    "arctanh": lambda a: np.arctanh(a["a"]),

    # Exponential
    "exp": lambda a: np.exp(a["a"]),
    "exp2": lambda a: np.exp2(a["a"]),
    "expm1": lambda a: np.expm1(a["a"]),
    "log": lambda a: np.log(a["a"]),
    "log2": lambda a: np.log2(a["a"]),
    "log10": lambda a: np.log10(a["a"]),
    "log1p": lambda a: np.log1p(a["a"]),
    "logaddexp": lambda a: np.logaddexp(a["a"], a["b"]),
    "logaddexp2": lambda a: np.logaddexp2(a["a"], a["b"]),

    # Gradient
    "diff": lambda a: np.diff(a["a"]),
    "ediff1d": lambda a: np.ediff1d(a["a"]),
    "gradient": lambda a: np.gradient(a["a"]),
    "cross": lambda a: np.cross(a["a"], a["b"]),

    # Linear algebra
    "dot": lambda a: np.dot(a["a"], a["b"]),
    "inner": lambda a: np.inner(a["a"], a["b"]),
    "outer": lambda a: np.outer(a["a"], a["b"]),
    "tensordot": lambda a: np.tensordot(a["a"], a["b"], axes=a.get("axes", 2)),
    "matmul": lambda a: a["a"] @ a["b"],
    "trace": lambda a: np.trace(a["a"]),
    "transpose": lambda a: a["a"].T,
    "diagonal": lambda a: np.diagonal(a["a"]),
    "kron": lambda a: np.kron(a["a"], a["b"]),
    "einsum": lambda a: np.einsum(a["subscripts"], a["a"], a["b"]),
    "deg2rad": lambda a: np.deg2rad(a["a"]),
    "rad2deg": lambda a: np.rad2deg(a["a"]),

    # numpy.linalg
    "linalg_det": lambda a: np.linalg.det(a["a"]),
    "linalg_inv": lambda a: np.linalg.inv(a["a"]),
    "linalg_solve": lambda a: np.linalg.solve(a["a"], a["b"]),
    "linalg_qr": lambda a: np.linalg.qr(a["a"]),
    "linalg_cholesky": lambda a: np.linalg.cholesky(a["_posdef"]),
    "linalg_svd": lambda a: np.linalg.svd(a["a"]),
    "linalg_eig": lambda a: np.linalg.eig(a["a"]),
    "linalg_eigh": _linalg_eigh,
    "linalg_norm": lambda a: np.linalg.norm(a["a"]),
    "linalg_matrix_rank": lambda a: np.linalg.matrix_rank(a["a"]),
    "linalg_pinv": lambda a: np.linalg.pinv(a["a"]),
    "linalg_cond": lambda a: np.linalg.cond(a["a"]),
    "linalg_matrix_power": lambda a: np.linalg.matrix_power(a["a"], 3),
    "linalg_lstsq": lambda a: np.linalg.lstsq(a["a"], a["b"], rcond=None),
    "linalg_cross": lambda a: np.cross(a["a"], a["b"]),
    "linalg_slogdet": lambda a: np.linalg.slogdet(a["a"]),
    "linalg_svdvals": lambda a: np.linalg.svdvals(a["a"]),
    "linalg_multi_dot": _linalg_multi_dot,
    "vdot": lambda a: np.vdot(a["a"], a["b"]),
    "vecdot": _vecdot,
    "matrix_transpose": _matrix_transpose,
    "matvec": _matvec,
    "vecmat": _vecmat,

    # Reductions
    "sum": lambda a: a["a"].sum(axis=a.get("axis")),
    "mean": lambda a: a["a"].mean(axis=a.get("axis")),
    "max": lambda a: a["a"].max(axis=a.get("axis")),
    "min": lambda a: a["a"].min(axis=a.get("axis")),
    "prod": lambda a: a["a"].prod(axis=a.get("axis")),
    "argmin": lambda a: a["a"].argmin(axis=a.get("axis")),
    "argmax": lambda a: a["a"].argmax(axis=a.get("axis")),
    "var": lambda a: a["a"].var(axis=a.get("axis")),
    "std": lambda a: a["a"].std(axis=a.get("axis")),
    "all": lambda a: a["a"].all(axis=a.get("axis")),
    "any": lambda a: a["a"].any(axis=a.get("axis")),
    "cumsum": lambda a: a["a"].cumsum(),
    "cumprod": lambda a: a["a"].cumprod(),
    "ptp": lambda a: np.ptp(a["a"]),
    "median": lambda a: np.median(a["a"]),
    "percentile": lambda a: np.percentile(a["a"], 50),
    "quantile": lambda a: np.quantile(a["a"], 0.5),
    "average": lambda a: np.average(a["a"]),
    "nansum": lambda a: np.nansum(a["a"]),
    "nanmean": lambda a: np.nanmean(a["a"]),
    "nanmin": lambda a: np.nanmin(a["a"]),
    "nanmax": lambda a: np.nanmax(a["a"]),
    "nanquantile": lambda a: np.nanquantile(a["a"], 0.5),
    "nanpercentile": lambda a: np.nanpercentile(a["a"], 50),

    # Array creation - extra
    "asarray_chkfinite": lambda a: np.asarray_chkfinite(a["a"]),
    "require": lambda a: np.require(a["a"], requirements='C'),

    # Reshape
    "reshape": lambda a: a["a"].reshape(a["new_shape"]),
    "flatten": lambda a: a["a"].flatten(),
    "ravel": lambda a: a["a"].ravel(),
    "squeeze": lambda a: a["a"].squeeze(),

    # Slicing
    "slice": lambda a: a["a"][:100, :100],

    # Manipulation
    "swapaxes": lambda a: np.swapaxes(a["a"], 0, 1),
    "concatenate": lambda a: np.concatenate([a["a"], a["b"]], axis=0),
    "stack": lambda a: np.stack([a["a"], a["b"]], axis=0),
    "vstack": lambda a: np.vstack([a["a"], a["b"]]),
    "hstack": lambda a: np.hstack([a["a"], a["b"]]),
    "tile": lambda a: np.tile(a["a"], [2, 2]),
    "repeat": lambda a: np.repeat(a["a"], 2),
    "concat": lambda a: np.concatenate([a["a"], a["b"]], axis=0),
    "unstack": lambda a: [arr for arr in a["a"]],
    "block": lambda a: np.block([a["a"], a["b"]]),
    "item": lambda a: a["a"].item(0),
    "tolist": lambda a: a["a"].tolist(),
    "broadcast_to": lambda a: np.broadcast_to(a["a"], a["target_shape"]),
    "take": lambda a: np.take(a["a"], a["indices"]),

    # Creation - misc
    "diag": lambda a: np.diag(a["a"]),
    "tri": lambda a: np.tri(a["shape"][0], a["shape"][1]),
    "tril": lambda a: np.tril(a["a"]),
    "triu": lambda a: np.triu(a["a"]),

    # Manipulation - misc
    "flip": lambda a: np.flip(a["a"]),
    "rot90": lambda a: np.rot90(a["a"]),
    "roll": lambda a: np.roll(a["a"], 10),
    "pad": lambda a: np.pad(a["a"], 2),

    # Indexing
    "take_along_axis": lambda a: np.take_along_axis(a["a"], a["b"].astype(np.intp), axis=0),
    "compress": lambda a: np.compress(a["b"].astype(bool), a["a"], axis=0),
    "diag_indices": lambda a: np.diag_indices(a["n"]),
    "tril_indices": lambda a: np.tril_indices(a["n"]),
    "triu_indices": lambda a: np.triu_indices(a["n"]),
    "indices": lambda a: np.indices(tuple(a["shape"])),
    "ravel_multi_index": _ravel_multi_index,
    "unravel_index": _unravel_index,

    # Bitwise
    "bitwise_and": lambda a: np.bitwise_and(a["a"], a["b"]),
    "bitwise_or": lambda a: np.bitwise_or(a["a"], a["b"]),
    "bitwise_xor": lambda a: np.bitwise_xor(a["a"], a["b"]),
    "bitwise_not": lambda a: np.bitwise_not(a["a"]),
    "invert": lambda a: np.invert(a["a"]),
    "left_shift": lambda a: np.left_shift(a["a"], a["b"]),
    "right_shift": lambda a: np.right_shift(a["a"], a["b"]),
    "packbits": lambda a: np.packbits(a["a"].astype(np.uint8)),
    "unpackbits": lambda a: np.unpackbits(a["a"].astype(np.uint8)),
    "bitwise_count": lambda a: np.bitwise_count(a["a"]),

    # IO
    "serializeNpy": _serialize_npy,
    "parseNpy": _parse_npy,
    "serializeNpzSync": _serialize_npz,
    "parseNpzSync": _parse_npz,

    # Sorting
    "sort": lambda a: np.sort(a["a"]),
    "argsort": lambda a: np.argsort(a["a"]),
    "partition": lambda a: np.partition(a["a"], a.get("kth", 0)),
    "argpartition": lambda a: np.argpartition(a["a"], a.get("kth", 0)),
    "lexsort": lambda a: np.lexsort((a["a"].ravel(), a["b"].ravel())),
    "sort_complex": lambda a: np.sort_complex(a["a"]),

    # Searching
    "nonzero": lambda a: np.nonzero(a["a"]),
    "argwhere": lambda a: np.argwhere(a["a"]),
    "flatnonzero": lambda a: np.flatnonzero(a["a"]),
    "where": lambda a: np.where(a["a"], a["b"], a["c"]),
    "searchsorted": lambda a: np.searchsorted(a["a"].ravel(), a["b"].ravel()),
    "extract": lambda a: np.extract(a["condition"], a["a"]),
    "count_nonzero": lambda a: np.count_nonzero(a["a"]),

    # Statistics
    "bincount": lambda a: np.bincount(a["a"].flatten().astype(int)),
    "digitize": lambda a: np.digitize(a["a"].flatten(), a["b"].flatten()),
    "histogram": _histogram,
    "histogram2d": _histogram2d,
    "correlate": lambda a: np.correlate(a["a"].flatten(), a["b"].flatten(), mode="full"),
    "convolve": lambda a: np.convolve(a["a"].flatten(), a["b"].flatten(), mode="full"),
    "cov": lambda a: np.cov(a["a"]),
    "corrcoef": lambda a: np.corrcoef(a["a"]),
    "histogram_bin_edges": lambda a: np.histogram_bin_edges(a["a"].flatten(), bins=10),
    "trapezoid": lambda a: np.trapezoid(a["a"].flatten()),

    # Set
    "trim_zeros": lambda a: np.trim_zeros(a["a"].flatten()),
    "unique_values": lambda a: np.unique(a["a"].flatten()),
    "unique_counts": _unique_counts,

    # Logic
    "logical_and": _logical_and,
    "logical_or": _logical_or,
    "logical_not": lambda a: np.logical_not(a["a"]),
    "logical_xor": _logical_xor,
    "isfinite": lambda a: np.isfinite(a["a"]),
    "isinf": lambda a: np.isinf(a["a"]),
    "isnan": lambda a: np.isnan(a["a"]),
    "isneginf": lambda a: np.isneginf(a["a"]),
    "isposinf": lambda a: np.isposinf(a["a"]),
    "isreal": lambda a: np.isreal(a["a"]),
    "signbit": lambda a: np.signbit(a["a"]),
    "copysign": _copysign,

    # Random (legacy module)
    "random_random": lambda a: np.random.random(a["shape"]),
    "random_rand": lambda a: np.random.rand(*a["shape"]),
    "random_randn": lambda a: np.random.randn(*a["shape"]),
    "random_randint": lambda a: np.random.randint(0, 100, a["shape"], dtype=a.get("dtype", "int64")),
    "random_uniform": lambda a: np.random.uniform(0, 1, a["shape"]),
    "random_normal": lambda a: np.random.normal(0, 1, a["shape"]),
    "random_standard_normal": lambda a: np.random.standard_normal(a["shape"]),
    "random_exponential": lambda a: np.random.exponential(1, a["shape"]),
    "random_poisson": lambda a: np.random.poisson(5, a["shape"]),
    "random_binomial": lambda a: np.random.binomial(10, 0.5, a["shape"]),
    "random_choice": lambda a: np.random.choice(a["n"], 100),
    "random_permutation": lambda a: np.random.permutation(a["n"]),
    "random_gamma": lambda a: np.random.gamma(2.0, 2.0, a["shape"]),
    "random_beta": lambda a: np.random.beta(2.0, 5.0, a["shape"]),
    "random_chisquare": lambda a: np.random.chisquare(2.0, a["shape"]),
    "random_laplace": lambda a: np.random.laplace(0.0, 1.0, a["shape"]),
    "random_geometric": lambda a: np.random.geometric(0.5, a["shape"]),
    "random_dirichlet": _random_dirichlet,
    "random_standard_exponential": lambda a: np.random.standard_exponential(a["shape"]),
    "random_logistic": lambda a: np.random.logistic(0.0, 1.0, a["shape"]),
    "random_lognormal": lambda a: np.random.lognormal(0.0, 1.0, a["shape"]),
    "random_gumbel": lambda a: np.random.gumbel(0.0, 1.0, a["shape"]),
    "random_pareto": lambda a: np.random.pareto(3.0, a["shape"]),
    "random_power": lambda a: np.random.power(3.0, a["shape"]),
    "random_rayleigh": lambda a: np.random.rayleigh(1.0, a["shape"]),
    "random_weibull": lambda a: np.random.weibull(3.0, a["shape"]),
    "random_triangular": lambda a: np.random.triangular(0.0, 0.5, 1.0, a["shape"]),
    "random_standard_cauchy": lambda a: np.random.standard_cauchy(a["shape"]),
    "random_standard_t": lambda a: np.random.standard_t(5.0, a["shape"]),
    "random_wald": lambda a: np.random.wald(1.0, 1.0, a["shape"]),
    "random_vonmises": lambda a: np.random.vonmises(0.0, 1.0, a["shape"]),
    "random_zipf": lambda a: np.random.zipf(2.0, a["shape"]),

    # Generator (PCG64) random
    "gen_random": _gen("random"),
    "gen_uniform": _gen("uniform", 0.0, 1.0),
    "gen_standard_normal": _gen("standard_normal"),
    "gen_normal": _gen("normal", 0.0, 1.0),
    "gen_exponential": _gen("exponential", 1.0),
    "gen_integers": _gen_integers,
    "gen_permutation": _gen_permutation,

    # Complex
    "complex_zeros": lambda a: np.zeros(a["shape"], dtype=np.complex128),
    "complex_ones": lambda a: np.ones(a["shape"], dtype=np.complex128),
    "complex_add": lambda a: a["a"] + a["b"],
    "complex_multiply": lambda a: a["a"] * a["b"],
    "complex_divide": lambda a: a["a"] / a["b"],
    "complex_real": lambda a: np.real(a["a"]),
    "complex_imag": lambda a: np.imag(a["a"]),
    "complex_conj": lambda a: np.conj(a["a"]),
    "complex_angle": lambda a: np.angle(a["a"]),
    "complex_abs": lambda a: np.abs(a["a"]),
    "complex_sqrt": lambda a: np.sqrt(a["a"]),
    "complex_sum": lambda a: np.sum(a["a"]),
    "complex_mean": lambda a: np.mean(a["a"]),
    "complex_prod": lambda a: np.prod(a["a"]),

    # Other math
    "clip": lambda a: np.clip(a["a"], 10, 100),
    "maximum": lambda a: np.maximum(a["a"], a["b"]),
    "minimum": lambda a: np.minimum(a["a"], a["b"]),
    "fmax": lambda a: np.fmax(a["a"], a["b"]),
    "fmin": lambda a: np.fmin(a["a"], a["b"]),
    "nan_to_num": lambda a: np.nan_to_num(a["a"]),
    "interp": lambda a: np.interp(a["x"], a["xp"], a["fp"]),
    "unwrap": lambda a: np.unwrap(a["a"]),
    "sinc": lambda a: np.sinc(a["a"]),
    "i0": lambda a: np.i0(a["a"]),

    # Polynomial
    "poly": lambda a: np.poly(a["a"]),
    "polyadd": lambda a: np.polyadd(a["a"], a["b"]),
    "polyder": lambda a: np.polyder(a["a"]),
    "polydiv": lambda a: np.polydiv(a["a"], a["b"]),
    "polyfit": lambda a: np.polyfit(a["a"], a["b"], 2),
    "polyint": lambda a: np.polyint(a["a"]),
    "polymul": lambda a: np.polymul(a["a"], a["b"]),
    "polysub": lambda a: np.polysub(a["a"], a["b"]),
    "polyval": lambda a: np.polyval(a["a"], a["b"]),
    "roots": lambda a: np.roots(a["a"]),

    # Type checking
    "can_cast": lambda a: np.can_cast("int32", "float64"),
    "result_type": lambda a: np.result_type("int32", "float64"),
    "min_scalar_type": lambda a: np.min_scalar_type(1000),
    "issubdtype": lambda a: np.issubdtype(np.int32, np.integer),

    # FFT
    "fft": lambda a: np.fft.fft(a["a"]),
    "ifft": lambda a: np.fft.ifft(a["a"]),
    "fft2": lambda a: np.fft.fft2(a["a"]),
    "ifft2": lambda a: np.fft.ifft2(a["a"]),
    "fftn": lambda a: np.fft.fftn(a["a"]),
    "ifftn": lambda a: np.fft.ifftn(a["a"]),
    "rfft": lambda a: np.fft.rfft(a["a"]),
    "irfft": lambda a: np.fft.irfft(a["a"]),
    "rfft2": lambda a: np.fft.rfft2(a["a"]),
    "irfft2": lambda a: np.fft.irfft2(a["a"]),
    "rfftn": lambda a: np.fft.rfftn(a["a"]),
    "irfftn": lambda a: np.fft.irfftn(a["a"]),
    "hfft": lambda a: np.fft.hfft(a["a"]),
    "ihfft": lambda a: np.fft.ihfft(a["a"]),
    "fftfreq": lambda a: np.fft.fftfreq(a["n"]),
    "rfftfreq": lambda a: np.fft.rfftfreq(a["n"]),
    "fftshift": lambda a: np.fft.fftshift(a["a"]),
    "ifftshift": lambda a: np.fft.ifftshift(a["a"]),
}


def execute_operation(operation: str, arrays: Dict[str, np.ndarray]) -> Any:
    """Execute a named operation via O(1) dict lookup."""
    fn = OPERATIONS.get(operation)
    if fn is None:
        raise ValueError(f"Unknown operation: {operation}")
    return fn(arrays)


def calibrate_ops_per_sample(
    operation: str,
    arrays: Dict[str, np.ndarray],
    target_time_ms: float = MIN_SAMPLE_TIME_MS,
) -> int:
    """
    Auto-calibrate: Determine how many operations to run per sample
    to achieve the target minimum sample time.
    """
    ops_per_sample = 1
    calibration_runs = 0
    max_calibration_runs = 10

    while calibration_runs < max_calibration_runs:
        start = time.perf_counter()
        for _ in range(ops_per_sample):
            execute_operation(operation, arrays)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if elapsed_ms >= target_time_ms:
            break

        if elapsed_ms < target_time_ms / 10:
            ops_per_sample *= 10
        elif elapsed_ms < target_time_ms / 2:
            ops_per_sample *= 2
        else:
            target_ops = int(np.ceil((ops_per_sample * target_time_ms) / elapsed_ms))
            ops_per_sample = max(target_ops, ops_per_sample + 1)
            break

        calibration_runs += 1

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

    # Warmup phase - run several times to stabilize caches
    for _ in range(warmup):
        execute_operation(operation, arrays)

    # Calibration phase - determine ops per sample
    ops_per_sample = calibrate_ops_per_sample(operation, arrays)

    # Benchmark phase - collect samples
    sample_times = []
    total_ops = 0

    for _ in range(TARGET_SAMPLES):
        start = time.perf_counter()
        for _ in range(ops_per_sample):
            execute_operation(operation, arrays)
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
            gc.collect()  # Free unreferenced objects between specs
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
