#!/usr/bin/env python3
"""
Audit NumPy API to ensure complete coverage in numpy-ts.

This script:
1. Extracts all public NumPy functions (top-level and ndarray methods)
2. Categorizes them
3. Exports to JSON
4. Compares against API-REFERENCE.md to find gaps
"""

import numpy as np
import inspect
import json
import re
from pathlib import Path

def get_numpy_top_level_functions():
    """Get all public top-level NumPy functions."""
    functions = {}

    # Get all public members
    for name in dir(np):
        if name.startswith('_'):
            continue

        # Skip internal/test modules
        if name in ['test', 'core', 'lib', 'f2py', 'distutils', 'testing', 'ma', 'matlib', 'rec', 'char', 'ctypeslib']:
            continue

        try:
            obj = getattr(np, name)
            # Include functions and ufuncs (universal functions like add, multiply)
            if callable(obj) and not inspect.isclass(obj) and not inspect.ismodule(obj):
                # Get signature if possible
                try:
                    sig = str(inspect.signature(obj))
                except:
                    sig = "()"

                functions[name] = {
                    'type': 'function',
                    'signature': sig,
                    'module': 'numpy'
                }
        except:
            pass

    return functions

def get_numpy_submodule_functions():
    """Get functions from key NumPy submodules."""
    submodules = {
        'linalg': np.linalg,
        'fft': np.fft,
        'random': np.random,
    }

    functions = {}

    for mod_name, mod in submodules.items():
        for name in dir(mod):
            if name.startswith('_'):
                continue

            # Skip test modules
            if name == 'test':
                continue

            try:
                obj = getattr(mod, name)
                if callable(obj) and not inspect.isclass(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except:
                        sig = "()"

                    full_name = f"{mod_name}.{name}"
                    functions[full_name] = {
                        'type': 'function',
                        'signature': sig,
                        'module': f'numpy.{mod_name}'
                    }
            except:
                pass

    return functions

def get_ndarray_methods():
    """Get all public methods of ndarray."""
    methods = {}

    for name in dir(np.ndarray):
        if name.startswith('_'):
            continue

        try:
            obj = getattr(np.ndarray, name)
            if callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                except:
                    sig = "()"

                methods[name] = {
                    'type': 'method',
                    'signature': sig,
                    'class': 'ndarray'
                }
        except:
            pass

    return methods

def categorize_functions(functions):
    """Categorize NumPy functions by type."""
    categories = {
        'Array Creation': [
            'array', 'zeros', 'ones', 'empty', 'full', 'arange', 'linspace', 'logspace', 'geomspace',
            'eye', 'identity', 'zeros_like', 'ones_like', 'empty_like', 'full_like',
            'diag', 'diagflat', 'tri', 'tril', 'triu', 'vander', 'meshgrid', 'fromfunction',
            'frombuffer', 'fromfile', 'fromiter', 'fromstring', 'asarray', 'asanyarray',
            'ascontiguousarray', 'asfortranarray', 'copy',
            'asarray_chkfinite', 'astype', 'require'
        ],
        'Array Manipulation': [
            'reshape', 'ravel', 'transpose', 'swapaxes', 'moveaxis', 'rollaxis',
            'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack', 'row_stack',
            'split', 'array_split', 'hsplit', 'vsplit', 'dsplit',
            'tile', 'repeat', 'delete', 'insert', 'append', 'resize',
            'squeeze', 'expand_dims', 'atleast_1d', 'atleast_2d', 'atleast_3d',
            'flip', 'fliplr', 'flipud', 'roll', 'rot90', 'pad',
            'byteswap', 'fill', 'item', 'tobytes', 'tofile', 'tolist',
            'flatten', 'block', 'concat', 'unstack', 'view'
        ],
        'Arithmetic': [
            'add', 'subtract', 'multiply', 'divide', 'power', 'mod', 'remainder', 'divmod',
            'floor_divide', 'negative', 'positive', 'absolute', 'fabs', 'sign', 'reciprocal',
            'sqrt', 'square', 'cbrt', 'heaviside',
            'abs',  # alias for absolute
            'float_power', 'fmod', 'gcd', 'lcm', 'modf', 'frexp', 'ldexp',
            'pow', 'true_divide'  # aliases
        ],
        'Trigonometric': [
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'arctan2',
            'hypot', 'degrees', 'radians', 'deg2rad', 'rad2deg',
            'acos', 'asin', 'atan', 'atan2'  # aliases
        ],
        'Hyperbolic': [
            'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
            'acosh', 'asinh', 'atanh'  # aliases
        ],
        'Exponential': [
            'exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'logaddexp', 'logaddexp2'
        ],
        'Rounding': [
            'around', 'round', 'floor', 'ceil', 'trunc', 'rint', 'fix'
        ],
        'Other Math': [
            'clip', 'maximum', 'minimum', 'fmax', 'fmin', 'nan_to_num',
            'real', 'imag', 'conj', 'conjugate', 'angle',
            'interp', 'unwrap', 'sinc', 'i0'  # Interpolation and special functions
        ],
        'Reductions': [
            'sum', 'prod', 'mean', 'std', 'var', 'min', 'max',
            'argmin', 'argmax', 'all', 'any', 'nanargmin', 'nanargmax',
            'median', 'percentile', 'quantile', 'average',
            'cumsum', 'cumprod', 'nancumsum', 'nancumprod',
            'ptp', 'nanmin', 'nanmax', 'nanmean', 'nanstd', 'nanvar', 'nanmedian',
            'nansum', 'nanprod',
            'amax', 'amin',  # aliases for max, min
            'round_',  # alias for around/round
            'cumulative_prod', 'cumulative_sum',  # aliases for cumprod, cumsum
            'nanpercentile', 'nanquantile'
        ],
        'Linear Algebra': [
            'dot', 'matmul', 'inner', 'outer', 'tensordot', 'einsum', 'einsum_path', 'kron',
            'trace', 'diagonal',
            'vdot', 'vecdot', 'vecmat', 'matvec', 'matrix_transpose', 'permute_dims'
        ],
        'Linear Algebra (linalg)': [
            'linalg.norm', 'linalg.det', 'linalg.matrix_rank', 'linalg.matrix_power',
            'linalg.inv', 'linalg.pinv', 'linalg.solve', 'linalg.lstsq',
            'linalg.eig', 'linalg.eigh', 'linalg.eigvals', 'linalg.eigvalsh',
            'linalg.svd', 'linalg.qr', 'linalg.cholesky', 'linalg.cond',
            'linalg.matrix_norm', 'linalg.vector_norm', 'linalg.cross',
            'linalg.diagonal', 'linalg.matmul', 'linalg.matrix_transpose', 'linalg.multi_dot',
            'linalg.outer', 'linalg.slogdet', 'linalg.svdvals', 'linalg.tensordot',
            'linalg.tensorinv', 'linalg.tensorsolve', 'linalg.trace', 'linalg.vecdot'
        ],
        'Logic': [
            'logical_and', 'logical_or', 'logical_not', 'logical_xor',
            'isfinite', 'isinf', 'isnan', 'isnat',
            'signbit', 'copysign', 'nextafter', 'spacing',
            'iscomplex', 'iscomplexobj', 'isfortran', 'isreal', 'isrealobj', 'isscalar',
            'isdtype', 'isneginf', 'isposinf', 'iterable', 'promote_types', 'real_if_close'
        ],
        'Comparison': [
            'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
            'isclose', 'allclose', 'array_equal', 'array_equiv'
        ],
        'Sorting': [
            'sort', 'argsort', 'lexsort', 'partition', 'argpartition',
            'sort_complex'
        ],
        'Searching': [
            'argmax', 'argmin', 'nonzero', 'where', 'searchsorted',
            'extract', 'count_nonzero', 'flatnonzero', 'argwhere'
        ],
        'Set Operations': [
            'unique', 'in1d', 'isin', 'intersect1d', 'union1d', 'setdiff1d', 'setxor1d',
            'unique_all', 'unique_counts', 'unique_inverse', 'unique_values', 'trim_zeros'
        ],
        'Bit Operations': [
            'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not',
            'left_shift', 'right_shift', 'invert', 'packbits', 'unpackbits',
            'bitwise_count', 'bitwise_invert', 'bitwise_left_shift', 'bitwise_right_shift'
        ],
        'Statistics': [
            'histogram', 'histogram2d', 'histogramdd', 'bincount', 'digitize',
            'corrcoef', 'correlate', 'cov', 'convolve', 'histogram_bin_edges',
            'trapezoid'
        ],
        'Gradient': [
            'gradient', 'diff', 'ediff1d', 'cross'
        ],
        'Indexing': [
            'take', 'take_along_axis', 'put', 'put_along_axis', 'putmask',
            'choose', 'compress', 'select', 'place',
            'diag_indices', 'diag_indices_from', 'tril_indices', 'tril_indices_from',
            'triu_indices', 'triu_indices_from', 'mask_indices',
            'indices', 'ix_', 'ravel_multi_index', 'unravel_index',
            'fill_diagonal'
        ],
        'Broadcasting': [
            'broadcast_to', 'broadcast_arrays', 'broadcast_shapes'
        ],
        'I/O': [
            'load', 'save', 'savez', 'savez_compressed',
            'loadtxt', 'savetxt', 'genfromtxt', 'fromregex'
        ],
        'FFT': [
            'fft.fft', 'fft.ifft', 'fft.fft2', 'fft.ifft2', 'fft.fftn', 'fft.ifftn',
            'fft.rfft', 'fft.irfft', 'fft.rfft2', 'fft.irfft2', 'fft.rfftn', 'fft.irfftn',
            'fft.hfft', 'fft.ihfft',
            'fft.fftfreq', 'fft.rfftfreq', 'fft.fftshift', 'fft.ifftshift'
        ],
        'Random': [
            'random.rand', 'random.randn', 'random.randint', 'random.random',
            'random.uniform', 'random.normal', 'random.standard_normal',
            'random.exponential', 'random.poisson', 'random.binomial',
            'random.shuffle', 'random.permutation', 'random.choice',
            'random.seed', 'random.get_state', 'random.set_state',
            'random.default_rng',
            # Additional distributions
            'random.beta', 'random.chisquare', 'random.dirichlet', 'random.f', 'random.gamma',
            'random.geometric', 'random.gumbel', 'random.hypergeometric', 'random.laplace',
            'random.logistic', 'random.lognormal', 'random.logseries', 'random.multinomial',
            'random.multivariate_normal', 'random.negative_binomial', 'random.noncentral_chisquare',
            'random.noncentral_f', 'random.pareto', 'random.power', 'random.rayleigh',
            'random.standard_cauchy', 'random.standard_exponential', 'random.standard_gamma',
            'random.standard_t', 'random.triangular', 'random.vonmises', 'random.wald',
            'random.weibull', 'random.zipf',
            # Aliases and utilities
            'random.bytes', 'random.get_bit_generator', 'random.set_bit_generator',
            'random.random_integers', 'random.random_sample', 'random.ranf', 'random.sample'
        ],
        'Printing/Formatting': [
            'array2string', 'array_repr', 'array_str',
            'base_repr', 'binary_repr',
            'format_float_positional', 'format_float_scientific',
            'set_printoptions', 'get_printoptions', 'printoptions'
        ],
        'Type Checking': [
            'can_cast', 'common_type', 'result_type', 'min_scalar_type',
            'issubdtype', 'typename', 'mintypecode'
            # Note: issctype, issubclass_, issubsctype, sctype2char are in np.core (deprecated)
        ],
        'Utilities': [
            'apply_along_axis', 'apply_over_axes', 'copyto',
            'may_share_memory', 'shares_memory',
            'ndim', 'shape', 'size',
            'geterr', 'seterr'
        ],
        'Polynomials': [
            'poly', 'polyadd', 'polyder', 'polydiv', 'polyfit', 'polyint',
            'polymul', 'polysub', 'polyval', 'roots'
        ],
        'Unplanned': [
            # Methods we won't implement
            'dump', 'dumps', 'getfield', 'setfield', 'setflags', 'to_device',
            # Datetime/business days (no datetime dtype support)
            'busday_count', 'busday_offset', 'is_busday', 'datetime_as_string', 'datetime_data',
            # Signal processing (out of scope)
            'bartlett', 'blackman', 'hamming', 'hanning', 'kaiser',
            # Deprecated/legacy/internal
            'asmatrix', 'bmat', 'matrix', 'msort', 'getbuffer', 'newaxis',
            'issctype', 'issubclass_', 'issubsctype', 'sctype2char',  # np.core (deprecated)
            'trapz',  # Removed in NumPy 2.0 (use trapezoid instead)
            # Internal/meta
            'get_include', 'get_array_wrap',
            'show_config', 'show_runtime', 'info',  # NumPy introspection, not applicable to JS
            'geterrcall', 'seterrcall', 'seterrobj', 'geterrobj',  # FP error callbacks, not applicable to JS
            'byte_bounds',  # Memory introspection, not applicable to JS
            # Advanced/niche (DLPack, ufuncs, iterators)
            'from_dlpack', 'frompyfunc', 'vectorize', 'piecewise',
            'nested_iters', 'nditer', 'broadcast', 'ndindex', 'ndenumerate', 'flatiter',
            # Buffer management
            'getbufsize', 'setbufsize'
        ]
    }

    # Categorize
    categorized = {}
    uncategorized = []

    for func_name in functions:
        found = False
        for cat, func_list in categories.items():
            if func_name in func_list:
                if cat not in categorized:
                    categorized[cat] = []
                categorized[cat].append(func_name)
                found = True
                break

        if not found:
            uncategorized.append(func_name)

    return categorized, uncategorized

def parse_api_reference(api_ref_path):
    """Parse API-REFERENCE.md to get implemented functions."""
    with open(api_ref_path, 'r') as f:
        content = f.read()

    implemented = []
    unimplemented = []

    # Find all checkbox items
    checkbox_pattern = r'^- \[(x| )\] `([^`]+)`'

    for match in re.finditer(checkbox_pattern, content, re.MULTILINE):
        checked = match.group(1) == 'x'
        func_name = match.group(2)

        # Clean up function name (remove parameters)
        func_name = func_name.split('(')[0]

        if checked:
            implemented.append(func_name)
        else:
            unimplemented.append(func_name)

    return implemented, unimplemented

def main():
    print("Auditing NumPy API coverage...\n")

    # Get all NumPy functions
    print("1. Extracting NumPy top-level functions...")
    top_level = get_numpy_top_level_functions()
    print(f"   Found {len(top_level)} top-level functions")

    print("2. Extracting NumPy submodule functions...")
    submodule = get_numpy_submodule_functions()
    print(f"   Found {len(submodule)} submodule functions")

    print("3. Extracting ndarray methods...")
    methods = get_ndarray_methods()
    print(f"   Found {len(methods)} ndarray methods")

    # Combine all
    all_functions = {**top_level, **submodule}

    print(f"\n4. Total NumPy functions: {len(all_functions)}")
    print(f"   Total ndarray methods: {len(methods)}")

    # Categorize (include both functions and methods)
    print("\n5. Categorizing functions and methods...")
    all_apis = {**all_functions, **methods}
    categorized, uncategorized = categorize_functions(all_apis)

    print(f"   Categorized: {sum(len(v) for v in categorized.values())}")
    print(f"   Uncategorized: {len(uncategorized)}")

    # Parse API-REFERENCE.md
    api_ref_path = Path(__file__).parent.parent / 'docs' / 'API-REFERENCE.md'
    if api_ref_path.exists():
        print("\n6. Parsing API-REFERENCE.md...")
        implemented, unimplemented = parse_api_reference(api_ref_path)
        print(f"   Implemented: {len(implemented)}")
        print(f"   Unimplemented: {len(unimplemented)}")
        print(f"   Total in API-REFERENCE: {len(implemented) + len(unimplemented)}")

    # Export to JSON
    output = {
        'numpy_functions': all_functions,
        'ndarray_methods': methods,
        'categorized': categorized,
        'uncategorized': uncategorized,
        'stats': {
            'total_numpy_functions': len(all_functions),
            'total_ndarray_methods': len(methods),
            'categorized_count': sum(len(v) for v in categorized.values()),
            'uncategorized_count': len(uncategorized)
        }
    }

    if api_ref_path.exists():
        output['api_reference'] = {
            'implemented': implemented,
            'unimplemented': unimplemented,
            'total': len(implemented) + len(unimplemented)
        }

    output_path = Path(__file__).parent.parent / 'scripts' / 'numpy-api-audit.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n7. Exported to {output_path}")

    # Print summary by category
    print("\n" + "="*60)
    print("CATEGORY SUMMARY")
    print("="*60)
    for cat in sorted(categorized.keys()):
        funcs = categorized[cat]
        print(f"{cat:30s}: {len(funcs):3d} functions")

    if uncategorized:
        print(f"\nUncategorized: {len(uncategorized)} functions")
        print("Examples:", ', '.join(uncategorized[:10]))

    print("\n" + "="*60)
    print("Done!")

if __name__ == '__main__':
    main()
