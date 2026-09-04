/**
 * Benchmark specifications
 * Defines all benchmarks to run
 */

import type { DType } from '../../src/common/dtype';
import type { BenchmarkCase, BenchmarkMode, BenchmarkSetup, SizeScale } from './types';

// ========================================
// Dtype sweep configuration
// ========================================
// Exported so tests/unit/benchmark-coverage.test.ts can assert that every
// benchmarked op either gets its category's full dtype sweep or appears in
// exactly one skip list below. At module scope, a dropped sweep becomes a
// test failure instead of a silent loss of coverage.

export type DtypeFamily = 'float' | 'int' | 'uint' | 'complex';

// Which dtype families each category supports
export const CATEGORY_DTYPE_SUPPORT: Record<string, DtypeFamily[]> = {
  creation: ['float', 'int', 'uint', 'complex'],
  arithmetic: ['float', 'int', 'uint', 'complex'],
  math: ['float', 'int', 'uint', 'complex'],
  linalg: ['float', 'int', 'uint', 'complex'],
  reductions: ['float', 'int', 'uint', 'complex'],
  statistics: ['float'],
  manipulation: ['float', 'int', 'uint'],
  sorting: ['float', 'int', 'uint', 'complex'],
  sets: ['float', 'int', 'uint'],
  logic: ['float', 'int', 'uint'],
  gradient: ['float', 'int', 'uint'],
  trig: ['float', 'int', 'uint', 'complex'],
  indexing: ['float', 'int', 'uint'],
  polynomials: ['float'],
  bitwise: ['int', 'uint'],
  fft: ['float', 'complex'],
  io: ['float', 'int', 'uint'],
  // Skipped entirely (no variants): random, utilities
};

// Variant dtypes for each family, keyed by minimum mode required
export const FAMILY_VARIANTS: Record<DtypeFamily, { dtype: string; minMode: BenchmarkMode }[]> = {
  float: [
    { dtype: 'float32', minMode: 'standard' },
    { dtype: 'float16', minMode: 'full' },
  ],
  int: [
    { dtype: 'int64', minMode: 'full' },
    { dtype: 'int32', minMode: 'full' },
    { dtype: 'int16', minMode: 'full' },
    { dtype: 'int8', minMode: 'full' },
  ],
  uint: [
    { dtype: 'uint64', minMode: 'full' },
    { dtype: 'uint32', minMode: 'full' },
    { dtype: 'uint16', minMode: 'full' },
    { dtype: 'uint8', minMode: 'full' },
  ],
  complex: [
    { dtype: 'complex128', minMode: 'full' },
    { dtype: 'complex64', minMode: 'full' },
  ],
};

// Mode ordering for comparison
export const MODE_RANK: Record<BenchmarkMode, number> = { quick: 0, standard: 1, full: 2 };

// Setup keys that represent data arrays (get dtype changed)
export const DATA_ARRAY_KEYS = new Set(['a', 'b', 'c', 'n', 'shape']);

// Integer-only operations that live in a float-bearing category.
//
// `gcd`/`lcm` are defined only for integers, so their base specs pin int32.
// Their category (arithmetic) does include float, so the int-only-category rule
// below does not apply and the pin silently blocked the whole sweep — they ran
// at int32 and nothing else. Declaring them here makes int32/uint32 a sweepable
// base and restricts the sweep to the int and uint families.
export const INT_ONLY_OPERATIONS = new Set(['gcd', 'lcm']);

// Operations whose base dtype is a semantic pin rather than a category default:
// index/count arrays that are meaningless at other dtypes. Declared separately
// from SKIP_DTYPE_OPERATIONS (which is about numerical instability) so the
// benchmark-coverage test can tell an intentional single-dtype op from a spec
// that silently lost its sweep.
export const PINNED_INDEX_DTYPE_OPERATIONS = new Set([
  'ix_', // takes index arrays, not data
  'bincount', // bin counts are int32 by definition
  'ravel_multi_index', // multi-index input is int32
]);

// Operations to skip for ALL auto dtype variants
export const SKIP_DTYPE_OPERATIONS = new Set([
  // No array input to vary (shape/count parameters only)
  'broadcast_shapes',
  'fromiter',
  'mask_indices',
  // 'linalg_cholesky', // positive-definiteness lost in float32
  'linalg_eigh', // eigenvalue decomposition numerically sensitive
  'interp', // special setup, not dtype-variant-friendly
  'unravel_index', // index dtype pinned manually (int32, int64, float64)
  'packbits', // always uint8
  'unpackbits', // always uint8
]);

// Operations to skip for float16 dtype variants (precision too low for numerical algorithms)
export const SKIP_FLOAT16_OPERATIONS = new Set([
  // Still diverge at benchmark scale, though not on a handful of elements:
  // NumPy accumulates in f16 while we accumulate wider, so the running total
  // drifts once there are enough terms. At [100x100] nancumsum differs from
  // element 66 onward (2212 vs 2210); histogramdd lands 2 of 100 bins
  // differently; nancumprod complex diverges from element 39.
  'nancumsum',
  'nancumprod',
  'histogramdd',
  // Accumulating contraction, like dot/einsum
  'tensordot',
  'linalg_cholesky',
  'linalg_eigh',
  'linalg_svd',
  'linalg_svdvals',
  'linalg_pinv',
  'linalg_lstsq',
  'linalg_qr',
  'linalg_cond',
  'linalg_matrix_rank',
  'linalg_norm',
  'linalg_det',
  'linalg_slogdet',
  'linalg_inv',
  'linalg_solve',
  'linalg_matrix_power',
  'linalg_multi_dot',
  'einsum',
  'correlate',
  'convolve',
  // NumPy's linalg explicitly rejects float16 for polynomial ops (uses eigvals/lstsq internally)
  'polyfit',
  'polyval',
  'roots',
  // dot/inner/vdot: NumPy accumulates in f16 (overflows to inf), our WASM uses f32 (finite)
  'dot',
  'inner',
  'vdot',
]);

// Operations to skip for ALL int dtype variants (blocks both int and uint families)
export const SKIP_INT_OPERATIONS = new Set([
  // Vandermonde powers: the dtype promotion to int64 is correct, but the values
  // diverge once the powers overflow it — 319 of 1024 entries differ at the
  // benchmark's N=32, where the largest column is 31^31. Small inputs agree,
  // which is why this looked fixed.
  'vander',
  // Real-float-only spacing/step semantics
  'nextafter',
  'spacing',
  // Real-float-only ops. These previously sat in SKIP_DTYPE_OPERATIONS, which
  // also suppressed their float32/float16 variants — they are float-family
  // ops, not un-sweepable ops.
  'fabs',
  'cbrt',
  'float_power',
  'heaviside',
  'fmod',
  'frexp',
  'ldexp',
  'modf',
  // Float-only linalg decompositions/solvers — numerically require float
  'linalg_det',
  'linalg_slogdet',
  'linalg_inv',
  'linalg_solve',
  'linalg_cholesky',
  'linalg_eigh',
  'linalg_svd',
  'linalg_svdvals',
  'linalg_pinv',
  'linalg_lstsq',
  'linalg_qr',
  'linalg_cond',
  'linalg_matrix_rank',
  'linalg_norm',
  // Truly incompatible with BigInt (int64) — would throw at runtime
  'asarray_chkfinite', // NaN/Inf check doesn't work with BigInt
]);

// Operations to skip for uint dtype variants specifically.
// NumPy raises TypeError for these on unsigned integer arrays.
export const SKIP_UINT_OPERATIONS = new Set([
  'sign', // np.sign raises TypeError for uint types
]);

// Operations to skip for 64-bit int variants (int64/uint64) only.
// numpy-ts throws on these paths today (BigInt conversion inside the set
// helpers), so the benchmark cannot be validated against NumPy.
// Operations whose 64-bit integer *broadcast* path throws, while their
// same-shape path is fine. Mirrors the existing "complex broadcasting is buggy"
// rule below: skip only the mixed-shape variants so the same-shape int64/uint64
// coverage is kept.
//
//   np.gcd(int64[4], int64[4])  -> ok
//   np.gcd(int64[4], int64[1])  -> TypeError: Cannot convert 1 to a BigInt
//   np.gcd(int32[4], int32[1])  -> ok  (32-bit broadcast is fine)
// Empty: gcd/lcm were the only entries. Their 64-bit scalar path built a
// BigInt64Array and then stored a Number into it, so `gcd(int64[...], 6)` threw
// while the same-shape path was fine (finding 4.1b). Kept as an empty set rather
// than deleted — the mechanism is the right shape for the next op that needs it.
export const SKIP_INT64_BROADCAST_OPERATIONS = new Set<string>([]);

// Empty: `union1d` was fixed with the 64-bit set-op work, and `tensordot`
// accumulated its contraction in a JS number before storing to a BigInt array.
export const SKIP_INT64_OPERATIONS = new Set<string>([]);

// Operations to skip for NARROW int types (int8/int16) only.
// Operations where int8/int16 variants produce different results than NumPy
// due to overflow affecting ordering/convolution logic.
export const SKIP_NARROW_INT_OPERATIONS = new Set([
  // Products overflow differently at int8/int16
  'nancumprod',
  'nanprod',
  'vander',
  'correlate',
  'convolve',
  'unwrap',
  'searchsorted', // overflow affects sort order
  'argpartition', // overflow affects element ordering
]);

// Operations to skip for COMPLEX dtype variants
export const SKIP_COMPLEX_OPERATIONS = new Set([
  // NumPy raises TypeError: ufunc not supported for complex inputs
  'mod',
  'floor_divide',
  'divmod',
  // complex running products still diverge at scale — see the float16 note above
  'nancumprod',
  // Rounding, ordering and real-only predicates reject complex
  'ceil',
  'fix',
  'floor',
  'isinf',
  'logaddexp2',
  'nanargmax',
  'nanargmin',
  'nanmedian',
  'nanstd',
  'nanvar',
  'nextafter',
  'rint',
  'round',
  'spacing',
  'trunc',
  'cbrt',
  'float_power',
  // Linalg: real-only decompositions/solvers
  'linalg_det',
  'linalg_slogdet',
  'linalg_inv',
  'linalg_solve',
  'linalg_cholesky',
  'linalg_eigh',
  'linalg_svd',
  'linalg_svdvals',
  'linalg_pinv',
  'linalg_lstsq',
  'linalg_qr',
  'linalg_cond',
  'linalg_matrix_rank',
  'linalg_norm',
  'linalg_matrix_power',
  'linalg_multi_dot',
  'einsum',
  'trace',
  'transpose',
  'matrix_transpose',
  'diagonal',
  // Comparison/ordering: complex numbers are not orderable
  'max',
  'min',
  'argmax',
  'argmin',
  'maximum',
  'minimum',
  'fmax',
  'fmin',
  'clip',
  'median',
  'percentile',
  'quantile',
  'nanmax',
  'nanmin',
  'nanpercentile',
  'nanquantile',
  'ptp',
  'partition',
  'argpartition',
  'searchsorted',
  'lexsort',
  // Real-only math
  'sign',
  'signbit',
  'copysign',
  'reciprocal',
  'logaddexp',
  'hypot',
  'heaviside',
  'fmod',
  'frexp',
  'ldexp',
  'modf',
  'fabs',
  'cbrt',
  'remainder',
  'square', // overflow risk with complex_small fill
  // Boolean-result reductions
  'all',
  'any',
  'count_nonzero',
  'isfinite',
  'isnan',
  'isneginf',
  'isposinf',
  'isreal',
  // Creation ops with incompatible fill patterns
  'arange',
  'linspace',
  'logspace',
  'geomspace',
  'eye',
  'identity',
  // Manipulation that's trivial for complex (no compute difference)
  'copy',
  'flatten',
  'ravel',
  'reshape',
  'broadcast_to',
  'concatenate',
  'concat',
  'stack',
  'hstack',
  'vstack',
  'block',
  'unstack',
  'swapaxes',
  'flip',
  'rot90',
  'roll',
  'tile',
  'repeat',
  'pad',
  'take',
  'extract',
  'compress',
  'where',
  'require',
  'item',
  'tolist',
  'indices',
  'diag',
  'tri',
  'tril',
  'triu',
  'trim_zeros',
  'nan_to_num',
  'flatnonzero',
  'nonzero',
  'argwhere',
  // Overflow-guaranteed (cumulative product of complex overflows float64 after ~50 elements)
  'cumprod',
  // Real-only functions
  'arctan2', // atan2 not defined for complex
  'i0', // Bessel function, real-only
  'sinc', // real-only
  'unwrap', // phase unwrapping, real-only
  'asarray_chkfinite', // NaN/Inf check, real-only
  // Misc incompatible
  'diff',
  'gradient',
  'cross',
  'unique_values',
  'unique_counts',
  // FFT: real-input ops and utilities (complex variants don't make sense)
  'rfft',
  'irfft',
  'rfft2',
  'irfft2',
  'rfftn',
  'irfftn',
  'hfft',
  'ihfft',
  'fftfreq',
  'rfftfreq',
  'fftshift',
  'ifftshift',
]);

export function getBenchmarkSpecs(
  mode: BenchmarkMode = 'standard',
  sizeScale: SizeScale = 'default',
): BenchmarkCase[] {
  // Array sizes: 1D and 2D base sizes, plus special sizes for expensive ops.
  // sizeScale selects small/default/large arrays; mode only affects warmup/iterations.
  const sizeScaleConfig = {
    small: {
      d1: 100,
      d2: [32, 32] as [number, number],
      io: [100, 100] as [number, number],
      kron: [6, 6] as [number, number],
      einsum: [20, 20] as [number, number],
      linalg: [20, 20] as [number, number],
      linalgSlow: [20, 20] as [number, number],
    },
    default: {
      d1: 1000,
      d2: [100, 100] as [number, number],
      io: [500, 500] as [number, number],
      kron: [10, 10] as [number, number],
      einsum: [50, 50] as [number, number],
      linalg: [50, 50] as [number, number],
      linalgSlow: [50, 50] as [number, number],
    },
    large: {
      d1: 10000,
      d2: [1000, 1000] as [number, number],
      io: [1000, 1000] as [number, number],
      kron: [32, 32] as [number, number],
      einsum: [1000, 1000] as [number, number],
      linalg: [500, 500] as [number, number],
      linalgSlow: [200, 200] as [number, number],
    },
  };

  const scaledSizes = sizeScaleConfig[sizeScale];

  const warmupConfig = {
    quick: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 3, // Less warmup for faster feedback
    },
    standard: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 10, // More warmup for stable results
    },
    full: {
      iterations: 1, // Not used with auto-calibration, kept for compatibility
      warmup: 20, // More warmup for stable full-mode results
    },
  };

  const config = warmupConfig[mode] ?? warmupConfig.standard;

  const { iterations, warmup } = config;
  const specs: BenchmarkCase[] = [];

  // ========================================
  // Array Creation Benchmarks
  // ========================================

  specs.push({
    name: `zeros [${scaledSizes.d1}]`,
    category: 'creation',
    operation: 'zeros',
    setup: {
      shape: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(scaledSizes.d2)) {
    specs.push({
      name: `zeros [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'zeros',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `ones [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'ones',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });
  }

  specs.push({
    name: `arange(${scaledSizes.d1})`,
    category: 'creation',
    operation: 'arange',
    setup: {
      n: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `linspace(0, 100, ${scaledSizes.d1})`,
    category: 'creation',
    operation: 'linspace',
    setup: {
      n: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `logspace(0, 3, ${scaledSizes.d1})`,
    category: 'creation',
    operation: 'logspace',
    setup: {
      n: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `geomspace(1, 1000, ${scaledSizes.d1})`,
    category: 'creation',
    operation: 'geomspace',
    setup: {
      n: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(scaledSizes.d2)) {
    const eyeSize = scaledSizes.d2[0]!;
    specs.push({
      name: `eye(${eyeSize})`,
      category: 'creation',
      operation: 'eye',
      setup: {
        n: { shape: [eyeSize] },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `identity(${eyeSize})`,
      category: 'creation',
      operation: 'identity',
      setup: {
        n: { shape: [eyeSize] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `empty [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'empty',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `full [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'full',
      setup: {
        shape: { shape: scaledSizes.d2 },
        fill_value: { shape: [7] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `copy [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'copy',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `zeros_like [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'zeros_like',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `asarray_chkfinite [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'asarray_chkfinite',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `require [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'require',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Arithmetic Benchmarks
  // ========================================

  if (Array.isArray(scaledSizes.d2)) {
    // --- add ---
    specs.push({
      name: `add [${scaledSizes.d2.join('x')}] + scalar`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
        b: { shape: [1], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `add [${scaledSizes.d2.join('x')}] + [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'add',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // --- subtract ---
    specs.push({
      name: `subtract [${scaledSizes.d2.join('x')}] - scalar`,
      category: 'arithmetic',
      operation: 'subtract',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `subtract [${scaledSizes.d2.join('x')}] - [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'subtract',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // --- multiply ---
    specs.push({
      name: `multiply [${scaledSizes.d2.join('x')}] * scalar`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `multiply [${scaledSizes.d2.join('x')}] * [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'multiply',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // --- divide ---
    specs.push({
      name: `divide [${scaledSizes.d2.join('x')}] / scalar`,
      category: 'arithmetic',
      operation: 'divide',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 3 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `divide [${scaledSizes.d2.join('x')}] / [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'divide',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // --- mod ---
    specs.push({
      name: `mod [${scaledSizes.d2.join('x')}] % scalar`,
      category: 'arithmetic',
      operation: 'mod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 7 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mod [${scaledSizes.d2.join('x')}] % [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'mod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- floor_divide ---
    specs.push({
      name: `floor_divide [${scaledSizes.d2.join('x')}] // scalar`,
      category: 'arithmetic',
      operation: 'floor_divide',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 3 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `floor_divide [${scaledSizes.d2.join('x')}] // [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'floor_divide',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- remainder ---
    specs.push({
      name: `remainder [${scaledSizes.d2.join('x')}] % scalar`,
      category: 'arithmetic',
      operation: 'remainder',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 7 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `remainder [${scaledSizes.d2.join('x')}] % [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'remainder',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- float_power ---
    specs.push({
      name: `float_power [${scaledSizes.d2.join('x')}] ** scalar`,
      category: 'arithmetic',
      operation: 'float_power',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `float_power [${scaledSizes.d2.join('x')}] ** [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'float_power',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- divmod ---
    specs.push({
      name: `divmod [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'divmod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 7 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `divmod [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'divmod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- gcd ---
    specs.push({
      name: `gcd [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'gcd',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1, dtype: 'int32' },
        b: { shape: [1], value: 6, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `gcd [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'gcd',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1, dtype: 'int32' },
        b: { shape: scaledSizes.d2, fill: 'arange', value: 1, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    // --- lcm ---
    specs.push({
      name: `lcm [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'lcm',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1, dtype: 'int32' },
        b: { shape: [1], value: 6, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `lcm [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'lcm',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1, dtype: 'int32' },
        b: { shape: scaledSizes.d2, fill: 'arange', value: 1, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    // --- absolute ---
    specs.push({
      name: `absolute [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'absolute',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // --- negative ---
    specs.push({
      name: `negative [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'negative',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- positive ---
    specs.push({
      name: `positive [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'positive',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- sign ---
    specs.push({
      name: `sign [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'sign',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- square ---
    specs.push({
      name: `square [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'square',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- reciprocal ---
    specs.push({
      name: `reciprocal [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'reciprocal',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- cbrt ---
    specs.push({
      name: `cbrt [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'cbrt',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    // --- fabs ---
    specs.push({
      name: `fabs [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'fabs',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: -100 },
      },
      iterations,
      warmup,
    });

    // --- heaviside ---
    specs.push({
      name: `heaviside [${scaledSizes.d2.join('x')}] scalar`,
      category: 'math',
      operation: 'heaviside',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 0.5 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `heaviside [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'heaviside',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- fmod ---
    specs.push({
      name: `fmod [${scaledSizes.d2.join('x')}] % scalar`,
      category: 'arithmetic',
      operation: 'fmod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 7 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `fmod [${scaledSizes.d2.join('x')}] % [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'fmod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- frexp ---
    specs.push({
      name: `frexp [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'frexp',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    // --- ldexp ---
    specs.push({
      name: `ldexp [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'ldexp',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: [1], value: 3 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ldexp [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'ldexp',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- modf ---
    specs.push({
      name: `modf [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'modf',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    // --- clip ---
    specs.push({
      name: `clip [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'clip',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- maximum ---
    specs.push({
      name: `maximum [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'maximum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 50 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `maximum [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'maximum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- minimum ---
    specs.push({
      name: `minimum [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'minimum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 50 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `minimum [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'minimum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- fmax ---
    specs.push({
      name: `fmax [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'fmax',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 50 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `fmax [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'fmax',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- fmin ---
    specs.push({
      name: `fmin [${scaledSizes.d2.join('x')}] scalar`,
      category: 'arithmetic',
      operation: 'fmin',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [1], value: 50 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `fmin [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'fmin',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // --- nan_to_num ---
    specs.push({
      name: `nan_to_num [${scaledSizes.d2.join('x')}]`,
      category: 'arithmetic',
      operation: 'nan_to_num',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- interp ---
    specs.push({
      name: `interp [${scaledSizes.d1}]`,
      category: 'math',
      operation: 'interp',
      setup: {
        x: { shape: [scaledSizes.d1], fill: 'arange' },
        xp: { shape: [100], fill: 'arange' },
        fp: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- unwrap ---
    specs.push({
      name: `unwrap [${scaledSizes.d1}]`,
      category: 'trig',
      operation: 'unwrap',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- sinc ---
    specs.push({
      name: `sinc [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'sinc',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // --- i0 ---
    specs.push({
      name: `i0 [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'i0',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Mathematical Operations Benchmarks
  // ========================================

  if (Array.isArray(scaledSizes.d2)) {
    specs.push({
      name: `sqrt [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'sqrt',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `power [${scaledSizes.d2.join('x')}] ** 2`,
      category: 'math',
      operation: 'power',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: [1], value: 2 },
      },
      iterations,
      warmup,
    });

    // absolute, negative, sign are already in arithmetic — not duplicated here.

    // Trigonometric & hyperbolic functions
    specs.push({
      name: `sin [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'sin',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `cos [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'cos',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tan [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'tan',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arcsin [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arcsin',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arccos [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arccos',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'random' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arctan [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arctan',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arctan2 [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arctan2',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `hypot [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'hypot',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
        b: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    // Hyperbolic functions
    // Use scaled values to avoid overflow (sinh/cosh overflow around 710)
    specs.push({
      name: `sinh [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'sinh',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cosh [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'cosh',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tanh [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'tanh',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arcsinh [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arcsinh',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arccosh [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arccosh',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `arctanh [${scaledSizes.d2.join('x')}]`,
      category: 'trig',
      operation: 'arctanh',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'random' },
      },
      iterations,
      warmup,
    });

    // Exponential functions
    specs.push({
      name: `exp [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'exp',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `exp2 [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'exp2',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `log [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'log',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `log2 [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'log2',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `log10 [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'log10',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logaddexp [${scaledSizes.d2.join('x')}]`,
      category: 'math',
      operation: 'logaddexp',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Gradient functions
    specs.push({
      name: `diff [${scaledSizes.d2.join('x')}]`,
      category: 'gradient',
      operation: 'diff',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `gradient [${scaledSizes.d1}]`,
      category: 'gradient',
      operation: 'gradient',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cross [${scaledSizes.d1}x3]`,
      category: 'linalg',
      operation: 'cross',
      setup: {
        a: { shape: [scaledSizes.d1, 3], fill: 'arange' },
        b: { shape: [scaledSizes.d1, 3], fill: 'ones' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Linear Algebra Benchmarks
  // ========================================

  if (Array.isArray(scaledSizes.d2)) {
    const [m, n] = scaledSizes.d2;

    // Dot product benchmarks
    specs.push({
      name: `dot 1D · 1D [${scaledSizes.d1}]`,
      category: 'linalg',
      operation: 'dot',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `dot 2D · 1D [${m}x${n}] · [${n}]`,
      category: 'linalg',
      operation: 'dot',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `dot 2D · 2D [${m}x${n}] · [${n}x${m}]`,
      category: 'linalg',
      operation: 'dot',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // Matrix multiplication
    specs.push({
      name: `matmul [${m}x${n}] @ [${n}x${m}]`,
      category: 'linalg',
      operation: 'matmul',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [n!, m!], fill: 'arange', dtype: 'float64' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // Trace
    specs.push({
      name: `trace [${m}x${n}]`,
      category: 'linalg',
      operation: 'trace',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Inner product
    specs.push({
      name: `inner 1D · 1D [${scaledSizes.d1}]`,
      category: 'linalg',
      operation: 'inner',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `inner 2D · 2D [${m}x${n}]`,
      category: 'linalg',
      operation: 'inner',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Outer product — output is NxN, so use 2D dimension to keep output manageable
    const outerN = scaledSizes.d2[0]!;
    specs.push({
      name: `outer [${outerN}] x [${outerN}]`,
      category: 'linalg',
      operation: 'outer',
      setup: {
        a: { shape: [outerN], fill: 'arange' },
        b: { shape: [outerN], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Transpose
    specs.push({
      name: `transpose [${m}x${n}]`,
      category: 'linalg',
      operation: 'transpose',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });
  }

  // ========================================
  // Reduction Benchmarks
  // ========================================

  if (Array.isArray(scaledSizes.d2)) {
    specs.push({
      name: `sum [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `sum [${scaledSizes.d2.join('x')}] axis=0`,
      category: 'reductions',
      operation: 'sum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        axis: { shape: [0] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `mean [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'mean',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `max [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'max',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `min [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'min',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `prod [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'prod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argmin [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'argmin',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argmax [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'argmax',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `var [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'var',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `std [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'std',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `all [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'all',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `any [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'any',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'zeros' },
      },
      iterations,
      warmup,
    });

    // New reduction functions
    specs.push({
      name: `cumsum [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'cumsum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cumprod [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'cumprod',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ptp [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'ptp',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `median [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'median',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `percentile [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'percentile',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `quantile [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'quantile',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `average [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'average',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // NaN-aware reduction functions
    specs.push({
      name: `nansum [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'nansum',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanmean [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'nanmean',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanmin [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'nanmin',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanmax [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'nanmax',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanquantile [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'nanquantile',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `nanpercentile [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'nanpercentile',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // Reductions axis=0 variants (standard/full only)
  if (mode !== 'quick' && Array.isArray(scaledSizes.d2)) {
    for (const op of [
      'mean',
      'max',
      'min',
      'argmax',
      'argmin',
      'var',
      'std',
      'prod',
      'median',
    ] as const) {
      specs.push({
        name: `${op} [${scaledSizes.d2.join('x')}] axis=0`,
        category: 'reductions',
        operation: op,
        setup: {
          a: { shape: scaledSizes.d2, fill: 'arange' },
          axis: { shape: [0] },
        },
        iterations,
        warmup,
      });
    }
  }

  // ========================================
  // Reshape Benchmarks
  // ========================================

  if (Array.isArray(scaledSizes.d2)) {
    const [m, n] = scaledSizes.d2;

    specs.push({
      name: `reshape [${m}x${n}] -> [${n}x${m}] (contiguous)`,
      category: 'manipulation',
      operation: 'reshape',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        new_shape: { shape: [n!, m!] },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `flatten [${m}x${n}]`,
      category: 'manipulation',
      operation: 'flatten',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ravel [${m}x${n}]`,
      category: 'manipulation',
      operation: 'ravel',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Array Manipulation Benchmarks
    // ========================================

    specs.push({
      name: `swapaxes [${m}x${n}]`,
      category: 'manipulation',
      operation: 'swapaxes',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `concatenate [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'concatenate',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `stack [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'stack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vstack [${m}x${n}]`,
      category: 'manipulation',
      operation: 'vstack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `hstack [${m}x${n}]`,
      category: 'manipulation',
      operation: 'hstack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tile [${m}x${n}] x [2,2]`,
      category: 'manipulation',
      operation: 'tile',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `repeat [${m}x${n}] x 2`,
      category: 'manipulation',
      operation: 'repeat',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `broadcast_to [${n}] -> [${m}x${n}]`,
      category: 'manipulation',
      operation: 'broadcast_to',
      setup: {
        a: { shape: [n!], fill: 'arange' },
        target_shape: { shape: [m!, n!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `take [${m}x${n}] 100 indices`,
      category: 'manipulation',
      operation: 'take',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        indices: { shape: Array.from({ length: 100 }, (_, i) => i % (m! * n!)) },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `concat [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'concat',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unstack [${m}x${n}]`,
      category: 'manipulation',
      operation: 'unstack',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `block [${m}x${n}] + [${m}x${n}]`,
      category: 'manipulation',
      operation: 'block',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
        b: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `item [${m}x${n}]`,
      category: 'manipulation',
      operation: 'item',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tolist [${m}x${n}]`,
      category: 'manipulation',
      operation: 'tolist',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // IO Benchmarks (NPY/NPZ parsing and serialization)
  // ========================================

  if (Array.isArray(scaledSizes.d2)) {
    const [m, n] = scaledSizes.d2;
    // NPY serialization
    specs.push({
      name: `serializeNpy [${m}x${n}]`,
      category: 'io',
      operation: 'serializeNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // NPY parsing (uses pre-serialized bytes)
    specs.push({
      name: `parseNpy [${m}x${n}]`,
      category: 'io',
      operation: 'parseNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // NPZ serialization (sync, no compression)
    specs.push({
      name: `serializeNpzSync {a, b} [${m}x${n}]`,
      category: 'io',
      operation: 'serializeNpzSync',
      setup: {
        a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
        b: { shape: [m!, n!], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    // NPZ parsing (sync, no compression)
    // Skip for large arrays: NumPy's np.load() returns a lazy NpzFile (~7μs regardless)
    // Once we add compression/decompression, I'll re-add this
    if (sizeScale !== 'large') {
      specs.push({
        name: `parseNpzSync {a, b} [${m}x${n}]`,
        category: 'io',
        operation: 'parseNpzSync',
        setup: {
          a: { shape: [m!, n!], fill: 'arange', dtype: 'float64' },
          b: { shape: [m!, n!], fill: 'ones', dtype: 'float64' },
        },
        iterations,
        includeInQuick: true,
        warmup,
      });
    }
  }

  // Larger IO benchmarks for non-quick mode
  if (mode !== 'quick' && Array.isArray(scaledSizes.io)) {
    const [m, n] = scaledSizes.io;

    specs.push({
      name: `serializeNpy [${m}x${n}]`,
      category: 'io',
      operation: 'serializeNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations: Math.floor(iterations / 2),
      warmup: Math.floor(warmup / 2),
    });

    specs.push({
      name: `parseNpy [${m}x${n}]`,
      category: 'io',
      operation: 'parseNpy',
      setup: {
        a: { shape: [m!, n!], fill: 'arange' },
      },
      iterations: Math.floor(iterations / 2),
      warmup: Math.floor(warmup / 2),
    });
  }

  // New functions benchmarks
  // Trigonometric conversions
  specs.push({
    name: `deg2rad [${scaledSizes.d1}]`,
    category: 'trig',
    operation: 'deg2rad',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange', value: 0 },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rad2deg [${scaledSizes.d1}]`,
    category: 'trig',
    operation: 'rad2deg',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange', value: 0 },
    },
    iterations,
    warmup,
  });

  if (Array.isArray(scaledSizes.d2)) {
    // Linear algebra operations
    specs.push({
      name: `diagonal [${scaledSizes.d2.join('x')}]`,
      category: 'linalg',
      operation: 'diagonal',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Kron produces large outputs, so use smaller inputs
    const kronSize = scaledSizes.kron;
    specs.push({
      name: `kron [${kronSize.join('x')}] ⊗ [${kronSize.join('x')}]`,
      category: 'linalg',
      operation: 'kron',
      setup: {
        a: { shape: kronSize, fill: 'arange' },
        b: { shape: kronSize, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // New array creation benchmarks
    specs.push({
      name: `diag [${scaledSizes.d2[0]}]`,
      category: 'creation',
      operation: 'diag',
      setup: {
        a: { shape: [scaledSizes.d2[0]!], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tri [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'tri',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tril [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'tril',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `triu [${scaledSizes.d2.join('x')}]`,
      category: 'creation',
      operation: 'triu',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // New array manipulation benchmarks
    specs.push({
      name: `flip [${scaledSizes.d2.join('x')}]`,
      category: 'manipulation',
      operation: 'flip',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `rot90 [${scaledSizes.d2.join('x')}]`,
      category: 'manipulation',
      operation: 'rot90',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `roll [${scaledSizes.d2.join('x')}] shift=10`,
      category: 'manipulation',
      operation: 'roll',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `pad [${scaledSizes.d2.join('x')}] width=2`,
      category: 'manipulation',
      operation: 'pad',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // einsum - matrix multiplication
    const einsumSize = scaledSizes.einsum;
    specs.push({
      name: `einsum_path matmul [${einsumSize.join('x')}]`,
      category: 'linalg',
      operation: 'einsum_path',
      setup: {
        subscripts: { shape: [], value: 'ij,jk->ik' },
        a: { shape: einsumSize, fill: 'arange' },
        b: { shape: einsumSize, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `einsum matmul [${einsumSize.join('x')}]`,
      category: 'linalg',
      operation: 'einsum',
      setup: {
        subscripts: { shape: [], value: 'ij,jk->ik' },
        a: { shape: einsumSize, fill: 'arange' },
        b: { shape: einsumSize, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // numpy.linalg Module Benchmarks
    // ========================================

    // O(n³) operations use scaled linalg sizes
    // These benchmarks use special 'invertible' fill mode handled in runner
    const linalgSize = scaledSizes.linalg;
    const linalgN = linalgSize[0]!;

    specs.push({
      name: `linalg.det [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_det',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `linalg.inv [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_inv',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `linalg.solve [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_solve',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
        b: { shape: [linalgN], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.qr [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_qr',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.cholesky [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_cholesky',
      setup: {
        // Setup will create a positive definite matrix in runner
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    const linalgSlowSize = scaledSizes.linalgSlow;
    const linalgSlowN = linalgSlowSize[0]!;
    specs.push({
      name: `linalg.svd [${linalgSlowN}x${linalgSlowN}]`,
      category: 'linalg',
      operation: 'linalg_svd',
      setup: {
        a: { shape: linalgSlowSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `linalg.eigh [${linalgSlowN}x${linalgSlowN}]`,
      category: 'linalg',
      operation: 'linalg_eigh',
      setup: {
        a: { shape: linalgSlowSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.norm [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_norm',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.matrix_rank [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_matrix_rank',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.pinv [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_pinv',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.cond [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_cond',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.matrix_power [${linalgN}x${linalgN}] n=3`,
      category: 'linalg',
      operation: 'linalg_matrix_power',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.lstsq [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_lstsq',
      setup: {
        a: { shape: linalgSize, fill: 'invertible', dtype: 'float64' },
        b: { shape: [linalgN], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.cross [3]`,
      category: 'linalg',
      operation: 'linalg_cross',
      setup: {
        a: { shape: [3], fill: 'arange', dtype: 'float64' },
        b: { shape: [3], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // New linalg functions
    specs.push({
      name: `linalg.slogdet [${linalgN}x${linalgN}]`,
      category: 'linalg',
      operation: 'linalg_slogdet',
      setup: {
        a: { shape: linalgSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `linalg.svdvals [${linalgSlowN}x${linalgSlowN}]`,
      category: 'linalg',
      operation: 'linalg_svdvals',
      setup: {
        a: { shape: linalgSlowSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    const mdN = linalgSlowN;
    specs.push({
      name: `linalg.multi_dot [${mdN}x${mdN}] x3`,
      category: 'linalg',
      operation: 'linalg_multi_dot',
      setup: {
        a: { shape: linalgSlowSize, fill: 'arange', dtype: 'float64' },
        b: { shape: linalgSlowSize, fill: 'arange', dtype: 'float64' },
        c: { shape: linalgSlowSize, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vdot [${scaledSizes.d1}]`,
      category: 'linalg',
      operation: 'vdot',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange', dtype: 'float64' },
        b: { shape: [scaledSizes.d1], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vecdot [${scaledSizes.d2.join('x')}]`,
      category: 'linalg',
      operation: 'vecdot',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'float64' },
        b: { shape: scaledSizes.d2, fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `matrix_transpose [${scaledSizes.d2.join('x')}]`,
      category: 'linalg',
      operation: 'matrix_transpose',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    const mvN = scaledSizes.d2[0]!;
    specs.push({
      name: `matvec [${mvN}x${mvN}] · [${mvN}]`,
      category: 'linalg',
      operation: 'matvec',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'float64' },
        b: { shape: [mvN], fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `vecmat [${mvN}] · [${mvN}x${mvN}]`,
      category: 'linalg',
      operation: 'vecmat',
      setup: {
        a: { shape: [mvN], fill: 'arange', dtype: 'float64' },
        b: { shape: scaledSizes.d2, fill: 'ones', dtype: 'float64' },
      },
      iterations,
      warmup,
    });

    // Indexing benchmarks
    specs.push({
      name: `take_along_axis [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'take_along_axis',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        // indices array with same shape
        b: { shape: scaledSizes.d2, fill: 'zeros', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `compress [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'compress',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: [scaledSizes.d2[0]!], fill: 'ones', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `diag_indices n=${scaledSizes.d2[0]}`,
      category: 'indexing',
      operation: 'diag_indices',
      setup: {
        n: { shape: [scaledSizes.d2[0]!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `tril_indices n=${scaledSizes.d2[0]}`,
      category: 'indexing',
      operation: 'tril_indices',
      setup: {
        n: { shape: [scaledSizes.d2[0]!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `triu_indices n=${scaledSizes.d2[0]}`,
      category: 'indexing',
      operation: 'triu_indices',
      setup: {
        n: { shape: [scaledSizes.d2[0]!] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `indices [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'indices',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `ravel_multi_index [${scaledSizes.d1}]`,
      category: 'indexing',
      operation: 'ravel_multi_index',
      setup: {
        // 1D index arrays
        a: { shape: [scaledSizes.d1], fill: 'zeros', dtype: 'int32' },
        b: { shape: [scaledSizes.d1], fill: 'zeros', dtype: 'int32' },
        dims: { shape: [100, 100] },
      },
      iterations,
      warmup,
    });

    for (const idxDtype of ['int32', 'int64', 'uint32', 'uint64'] as const) {
      specs.push({
        name: `unravel_index [${scaledSizes.d1}]${idxDtype === 'int32' ? '' : ` ${idxDtype}`}`,
        category: 'indexing',
        operation: 'unravel_index',
        setup: {
          a: { shape: [scaledSizes.d1], fill: 'arange', dtype: idxDtype },
          dims: { shape: [100, 100] },
        },
        iterations,
        warmup,
      });
    }

    // ========================================
    // Bitwise Operations Benchmarks
    // ========================================

    specs.push({
      name: `bitwise_and [${scaledSizes.d2.join('x')}] & [${scaledSizes.d2.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_and',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
        b: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `bitwise_or [${scaledSizes.d2.join('x')}] | [${scaledSizes.d2.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_or',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
        b: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_xor [${scaledSizes.d2.join('x')}] ^ [${scaledSizes.d2.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_xor',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
        b: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_not [${scaledSizes.d2.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_not',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `invert [${scaledSizes.d2.join('x')}]`,
      category: 'bitwise',
      operation: 'invert',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `left_shift [${scaledSizes.d2.join('x')}] << 2`,
      category: 'bitwise',
      operation: 'left_shift',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
        b: { shape: [1], value: 2, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `right_shift [${scaledSizes.d2.join('x')}] >> 2`,
      category: 'bitwise',
      operation: 'right_shift',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'int32' },
        b: { shape: [1], value: 2, dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `packbits [${scaledSizes.d1}]`,
      category: 'bitwise',
      operation: 'packbits',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange', dtype: 'uint8' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unpackbits [${Math.ceil(scaledSizes.d1 / 8)}]`,
      category: 'bitwise',
      operation: 'unpackbits',
      setup: {
        a: { shape: [Math.ceil(scaledSizes.d1 / 8)], fill: 'arange', dtype: 'uint8' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `bitwise_count [${scaledSizes.d2.join('x')}]`,
      category: 'bitwise',
      operation: 'bitwise_count',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange', dtype: 'uint32' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Sorting Benchmarks (shuffled = realistic random data)
    // ========================================

    specs.push({
      name: `sort [${scaledSizes.d2.join('x')}]`,
      category: 'sorting',
      operation: 'sort',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'shuffled' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `sort [${scaledSizes.d2.join('x')}] sorted`,
      category: 'sorting',
      operation: 'sort',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argsort [${scaledSizes.d2.join('x')}]`,
      category: 'sorting',
      operation: 'argsort',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'shuffled' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `partition [${scaledSizes.d2.join('x')}] kth=${Math.floor(scaledSizes.d2[0]! / 2)}`,
      category: 'sorting',
      operation: 'partition',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'shuffled' },
        kth: { shape: [Math.floor(scaledSizes.d2[0]! / 2)] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argpartition [${scaledSizes.d2.join('x')}] kth=${Math.floor(scaledSizes.d2[0]! / 2)}`,
      category: 'sorting',
      operation: 'argpartition',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'shuffled' },
        kth: { shape: [Math.floor(scaledSizes.d2[0]! / 2)] },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `lexsort [${scaledSizes.d1}] x 2 keys`,
      category: 'sorting',
      operation: 'lexsort',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'shuffled' },
        b: { shape: [scaledSizes.d1], fill: 'shuffled' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Searching Benchmarks
    // ========================================

    specs.push({
      name: `nonzero [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'nonzero',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `argwhere [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'argwhere',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `flatnonzero [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'flatnonzero',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `where [${scaledSizes.d2.join('x')}] with x,y`,
      category: 'logic',
      operation: 'where',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'ones', dtype: 'int32' },
        b: { shape: scaledSizes.d2, fill: 'arange' },
        c: { shape: scaledSizes.d2, fill: 'zeros' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `searchsorted [${scaledSizes.d1}]`,
      category: 'sorting',
      operation: 'searchsorted',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `extract [${scaledSizes.d2.join('x')}]`,
      category: 'indexing',
      operation: 'extract',
      setup: {
        condition: { shape: scaledSizes.d2, fill: 'ones', dtype: 'int32' },
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `count_nonzero [${scaledSizes.d2.join('x')}]`,
      category: 'reductions',
      operation: 'count_nonzero',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Statistics Benchmarks
    // ========================================

    specs.push({
      name: `bincount [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'bincount',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange', dtype: 'int32' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `digitize [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'digitize',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `histogram [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'histogram',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `histogram2d [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'histogram2d',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `correlate [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'correlate',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [100], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `convolve [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'convolve',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
        b: { shape: [100], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `cov [${scaledSizes.d2.join('x')}]`,
      category: 'statistics',
      operation: 'cov',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `corrcoef [${scaledSizes.d2.join('x')}]`,
      category: 'statistics',
      operation: 'corrcoef',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `histogram_bin_edges [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'histogram_bin_edges',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `trapezoid [${scaledSizes.d1}]`,
      category: 'statistics',
      operation: 'trapezoid',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // Set operation benchmarks
    specs.push({
      name: `trim_zeros [${scaledSizes.d1}]`,
      category: 'sets',
      operation: 'trim_zeros',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `unique_values [${scaledSizes.d1}]`,
      category: 'sets',
      operation: 'unique_values',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `unique_counts [${scaledSizes.d1}]`,
      category: 'sets',
      operation: 'unique_counts',
      setup: {
        a: { shape: [scaledSizes.d1], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Logic Benchmarks
    // ========================================

    specs.push({
      name: `logical_and [${scaledSizes.d2.join('x')}] & scalar`,
      category: 'logic',
      operation: 'logical_and',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        scalar: { shape: [1], value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_and [${scaledSizes.d2.join('x')}] & [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'logical_and',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        b: { shape: scaledSizes.d2, fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_or [${scaledSizes.d2.join('x')}] | scalar`,
      category: 'logic',
      operation: 'logical_or',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        scalar: { shape: [1], value: 0 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_not [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'logical_not',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `logical_xor [${scaledSizes.d2.join('x')}] ^ scalar`,
      category: 'logic',
      operation: 'logical_xor',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        scalar: { shape: [1], value: 1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isfinite [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'isfinite',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isnan [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'isnan',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `signbit [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'signbit',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `copysign [${scaledSizes.d2.join('x')}] scalar`,
      category: 'logic',
      operation: 'copysign',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
        scalar: { shape: [1], value: -1 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isneginf [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'isneginf',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isposinf [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'isposinf',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `isreal [${scaledSizes.d2.join('x')}]`,
      category: 'logic',
      operation: 'isreal',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Random Benchmarks
    // ========================================

    specs.push({
      name: `random.random [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_random',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.rand [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_rand',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.randn [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_randn',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    for (const dt of [
      'int64',
      'int32',
      'int16',
      'int8',
      'uint64',
      'uint32',
      'uint16',
      'uint8',
    ] as const) {
      specs.push({
        name: `random.randint [${scaledSizes.d2.join('x')}] ${dt}`,
        category: 'random',
        operation: 'random_randint',
        setup: {
          shape: { shape: scaledSizes.d2, dtype: dt },
        },
        iterations,
        warmup,
      });
    }

    specs.push({
      name: `random.uniform [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_uniform',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `random.normal [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_normal',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `random.standard_normal [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_standard_normal',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.exponential [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_exponential',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.poisson [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_poisson',
      setup: {
        shape: { shape: scaledSizes.d2, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.binomial [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_binomial',
      setup: {
        shape: { shape: scaledSizes.d2, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.choice [${scaledSizes.d1}]`,
      category: 'random',
      operation: 'random_choice',
      setup: {
        n: { shape: [scaledSizes.d1], dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.permutation [${scaledSizes.d1}]`,
      category: 'random',
      operation: 'random_permutation',
      setup: {
        n: { shape: [scaledSizes.d1], dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // New random distributions
    specs.push({
      name: `random.gamma [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_gamma',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.beta [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_beta',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.chisquare [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_chisquare',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.laplace [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_laplace',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.geometric [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'random_geometric',
      setup: {
        shape: { shape: scaledSizes.d2, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `random.dirichlet [${scaledSizes.d1}]`,
      category: 'random',
      operation: 'random_dirichlet',
      setup: {
        shape: { shape: [scaledSizes.d1] },
      },
      iterations,
      warmup,
    });

    for (const op of [
      { name: 'standard_exponential', operation: 'random_standard_exponential' },
      { name: 'logistic', operation: 'random_logistic' },
      { name: 'lognormal', operation: 'random_lognormal' },
      { name: 'gumbel', operation: 'random_gumbel' },
      { name: 'pareto', operation: 'random_pareto' },
      { name: 'power', operation: 'random_power' },
      { name: 'rayleigh', operation: 'random_rayleigh' },
      { name: 'weibull', operation: 'random_weibull' },
      { name: 'triangular', operation: 'random_triangular' },
      { name: 'standard_cauchy', operation: 'random_standard_cauchy' },
      { name: 'standard_t', operation: 'random_standard_t' },
      { name: 'wald', operation: 'random_wald' },
      { name: 'vonmises', operation: 'random_vonmises' },
      { name: 'zipf', operation: 'random_zipf' },
    ] as const) {
      specs.push({
        name: `random.${op.name} [${scaledSizes.d2.join('x')}]`,
        category: 'random',
        operation: op.operation,
        setup: {
          shape: { shape: scaledSizes.d2 },
        },
        iterations,
        warmup,
      });
    }

    // ========================================
    // Generator (PCG64) Random Benchmarks
    // ========================================

    specs.push({
      name: `Generator.random [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'gen_random',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `Generator.uniform [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'gen_uniform',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `Generator.standard_normal [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'gen_standard_normal',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `Generator.normal [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'gen_normal',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `Generator.exponential [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'gen_exponential',
      setup: {
        shape: { shape: scaledSizes.d2 },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `Generator.integers [${scaledSizes.d2.join('x')}]`,
      category: 'random',
      operation: 'gen_integers',
      setup: {
        shape: { shape: scaledSizes.d2, dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `Generator.permutation [${scaledSizes.d1}]`,
      category: 'random',
      operation: 'gen_permutation',
      setup: {
        n: { shape: [scaledSizes.d1], dtype: 'int64' },
      },
      iterations,
      warmup,
    });

    // ========================================
    // Complex Number Benchmarks
    // (categorized with their natural category)
    // ========================================

    // complex_zeros and complex_ones removed — auto-generated as zeros/ones complex128

    specs.push({
      name: `real [${scaledSizes.d2.join('x')}] complex128`,
      category: 'math',
      operation: 'complex_real',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `real [${scaledSizes.d2.join('x')}] complex64`,
      category: 'math',
      operation: 'complex_real',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex_small', dtype: 'complex64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `imag [${scaledSizes.d2.join('x')}] complex128`,
      category: 'math',
      operation: 'complex_imag',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `imag [${scaledSizes.d2.join('x')}] complex64`,
      category: 'math',
      operation: 'complex_imag',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex_small', dtype: 'complex64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `conj [${scaledSizes.d2.join('x')}] complex128`,
      category: 'math',
      operation: 'complex_conj',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `conj [${scaledSizes.d2.join('x')}] complex64`,
      category: 'math',
      operation: 'complex_conj',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex_small', dtype: 'complex64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `angle [${scaledSizes.d2.join('x')}] complex128`,
      category: 'math',
      operation: 'complex_angle',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `angle [${scaledSizes.d2.join('x')}] complex64`,
      category: 'math',
      operation: 'complex_angle',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex_small', dtype: 'complex64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `abs [${scaledSizes.d2.join('x')}] complex128`,
      category: 'math',
      operation: 'complex_abs',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `abs [${scaledSizes.d2.join('x')}] complex64`,
      category: 'math',
      operation: 'complex_abs',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex_small', dtype: 'complex64' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sqrt [${scaledSizes.d2.join('x')}] complex128`,
      category: 'math',
      operation: 'complex_sqrt',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `sqrt [${scaledSizes.d2.join('x')}] complex64`,
      category: 'math',
      operation: 'complex_sqrt',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex_small', dtype: 'complex64' },
      },
      iterations,
      warmup,
    });

    // complex_sum, complex_mean, complex_prod removed — auto-generated as sum/mean/prod complex128

    // ========================================
    // Polynomial Benchmarks
    // ========================================

    specs.push({
      name: `poly [10 roots]`,
      category: 'polynomials',
      operation: 'poly',
      setup: {
        a: { shape: [10], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyadd [10] + [10]`,
      category: 'polynomials',
      operation: 'polyadd',
      setup: {
        a: { shape: [10], fill: 'arange' },
        b: { shape: [10], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyder [10]`,
      category: 'polynomials',
      operation: 'polyder',
      setup: {
        a: { shape: [10], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polydiv [10] / [5]`,
      category: 'polynomials',
      operation: 'polydiv',
      setup: {
        a: { shape: [10], fill: 'ones' },
        b: { shape: [5], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyfit [100] deg=5`,
      category: 'polynomials',
      operation: 'polyfit',
      setup: {
        a: { shape: [100], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyint [10]`,
      category: 'polynomials',
      operation: 'polyint',
      setup: {
        a: { shape: [10], fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polymul [10] * [10]`,
      category: 'polynomials',
      operation: 'polymul',
      setup: {
        a: { shape: [10], fill: 'ones' },
        b: { shape: [10], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polysub [10] - [10]`,
      category: 'polynomials',
      operation: 'polysub',
      setup: {
        a: { shape: [10], fill: 'arange' },
        b: { shape: [10], fill: 'ones' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `polyval [10] at [100 points]`,
      category: 'polynomials',
      operation: 'polyval',
      setup: {
        a: { shape: [10], fill: 'arange' },
        b: { shape: [100], fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `roots [5]`,
      category: 'polynomials',
      operation: 'roots',
      setup: {
        a: { shape: [5], fill: 'ones' },
      },
      iterations,
      warmup,
    });
  }

  // ========================================
  // Other Math Benchmarks
  // ========================================

  // ========================================
  // FFT Benchmarks
  // ========================================

  // 1D FFT operations
  specs.push({
    name: `fft [${scaledSizes.d1}]`,
    category: 'fft',
    operation: 'fft',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ifft [${scaledSizes.d1}]`,
    category: 'fft',
    operation: 'ifft',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rfft [${scaledSizes.d1}]`,
    category: 'fft',
    operation: 'rfft',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `irfft [${scaledSizes.d1 / 2 + 1}]`,
    category: 'fft',
    operation: 'irfft',
    setup: {
      a: { shape: [Math.floor(scaledSizes.d1 / 2) + 1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `hfft [${scaledSizes.d1 / 2 + 1}]`,
    category: 'fft',
    operation: 'hfft',
    setup: {
      a: { shape: [Math.floor(scaledSizes.d1 / 2) + 1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ihfft [${scaledSizes.d1}]`,
    category: 'fft',
    operation: 'ihfft',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  // 2D FFT operations
  if (Array.isArray(scaledSizes.d2)) {
    specs.push({
      name: `fft2 [${scaledSizes.d2.join('x')}]`,
      category: 'fft',
      operation: 'fft2',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      includeInQuick: true,
      warmup,
    });

    specs.push({
      name: `ifft2 [${scaledSizes.d2.join('x')}]`,
      category: 'fft',
      operation: 'ifft2',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'complex' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `rfft2 [${scaledSizes.d2.join('x')}]`,
      category: 'fft',
      operation: 'rfft2',
      setup: {
        a: { shape: scaledSizes.d2, fill: 'arange' },
      },
      iterations,
      warmup,
    });

    specs.push({
      name: `irfft2 [${scaledSizes.d2[0]}x${Math.floor(scaledSizes.d2[1] / 2) + 1}]`,
      category: 'fft',
      operation: 'irfft2',
      setup: {
        a: { shape: [scaledSizes.d2[0], Math.floor(scaledSizes.d2[1] / 2) + 1], fill: 'complex' },
      },
      iterations,
      warmup,
    });
  }

  // N-D FFT operations (use 3D)
  const fft3dSize = [32, 32, 32];
  specs.push({
    name: `fftn [${fft3dSize.join('x')}]`,
    category: 'fft',
    operation: 'fftn',
    setup: {
      a: { shape: fft3dSize, fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ifftn [${fft3dSize.join('x')}]`,
    category: 'fft',
    operation: 'ifftn',
    setup: {
      a: { shape: fft3dSize, fill: 'complex' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rfftn [${fft3dSize.join('x')}]`,
    category: 'fft',
    operation: 'rfftn',
    setup: {
      a: { shape: fft3dSize, fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `irfftn [${fft3dSize[0]}x${fft3dSize[1]}x${Math.floor(fft3dSize[2] / 2) + 1}]`,
    category: 'fft',
    operation: 'irfftn',
    setup: {
      a: { shape: [fft3dSize[0], fft3dSize[1], Math.floor(fft3dSize[2] / 2) + 1], fill: 'complex' },
    },
    iterations,
    warmup,
  });

  // FFT utility functions
  specs.push({
    name: `fftfreq(${scaledSizes.d1})`,
    category: 'fft',
    operation: 'fftfreq',
    setup: {
      n: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rfftfreq(${scaledSizes.d1})`,
    category: 'fft',
    operation: 'rfftfreq',
    setup: {
      n: { shape: [scaledSizes.d1] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `fftshift [${scaledSizes.d1}]`,
    category: 'fft',
    operation: 'fftshift',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ifftshift [${scaledSizes.d1}]`,
    category: 'fft',
    operation: 'ifftshift',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'arange' },
    },
    iterations,
    warmup,
  });

  // ========================================
  // Auto-generate dtype variants
  // ========================================
  // For each base spec, generate variants with alternative dtypes.
  // Variants are inserted inline right after their base spec.
  // Categories declare which dtype families they support: float, int, complex.
  //
  // Standard mode: float64 (base) + float32
  // Full mode:     float64 (base) + float32 + complex128 + int64 + int32 + int16 + int8

  // ========================================
  // Previously-unbenchmarked public operations
  // ========================================
  // Added so tests/unit/benchmark-coverage.test.ts can require that every
  // benchmarkable public function has a spec.

  specs.push({
    name: `allclose [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'allclose',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `angle [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'angle',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `append [${scaledSizes.d1}] [${scaledSizes.d1}]`,
    category: 'manipulation',
    operation: 'append',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      b: { shape: [scaledSizes.d1], fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `apply_along_axis sum [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'apply_along_axis',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `apply_over_axes sum [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'apply_over_axes',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `array_equal [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'array_equal',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `array_equiv [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'array_equiv',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `array_split [${scaledSizes.d1}] into 3`,
    category: 'manipulation',
    operation: 'array_split',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `atleast_1d [${scaledSizes.d1}]`,
    category: 'manipulation',
    operation: 'atleast_1d',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `atleast_2d [${scaledSizes.d1}]`,
    category: 'manipulation',
    operation: 'atleast_2d',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `atleast_3d [${scaledSizes.d1}]`,
    category: 'manipulation',
    operation: 'atleast_3d',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `broadcast_arrays [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'manipulation',
    operation: 'broadcast_arrays',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `broadcast_shapes`,
    category: 'manipulation',
    operation: 'broadcast_shapes',
    setup: { shape: { shape: scaledSizes.d2 } },
    iterations,
    warmup,
  });

  specs.push({
    name: `ceil [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'ceil',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `choose [${scaledSizes.d1}] from 3`,
    category: 'indexing',
    operation: 'choose',
    setup: {
      a: { shape: [scaledSizes.d1], dtype: 'int32', value: 1 },
      b: { shape: [scaledSizes.d1], fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `column_stack [${scaledSizes.d1}] x2`,
    category: 'manipulation',
    operation: 'column_stack',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      b: { shape: [scaledSizes.d1], fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `diag_indices_from [${scaledSizes.linalg.join('x')}]`,
    category: 'indexing',
    operation: 'diag_indices_from',
    setup: { a: { shape: scaledSizes.linalg, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `diagflat [64]`,
    category: 'manipulation',
    operation: 'diagflat',
    setup: { a: { shape: [64], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `dsplit [16x16x4] into 2`,
    category: 'manipulation',
    operation: 'dsplit',
    setup: { a: { shape: [16, 16, 4], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `dstack [${scaledSizes.d2.join('x')}] x2`,
    category: 'manipulation',
    operation: 'dstack',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ediff1d [${scaledSizes.d1}]`,
    category: 'manipulation',
    operation: 'ediff1d',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `empty_like [${scaledSizes.d2.join('x')}]`,
    category: 'creation',
    operation: 'empty_like',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `equal [${scaledSizes.d2.join('x')}] == [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'equal',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `expand_dims [${scaledSizes.d2.join('x')}]`,
    category: 'manipulation',
    operation: 'expand_dims',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `expm1 [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'expm1',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `fill_diagonal [${scaledSizes.linalg.join('x')}]`,
    category: 'indexing',
    operation: 'fill_diagonal',
    setup: { a: { shape: scaledSizes.linalg, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `fix [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'fix',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `fliplr [${scaledSizes.d2.join('x')}]`,
    category: 'manipulation',
    operation: 'fliplr',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `flipud [${scaledSizes.d2.join('x')}]`,
    category: 'manipulation',
    operation: 'flipud',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `floor [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'floor',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `fromiter [${scaledSizes.d1}]`,
    category: 'creation',
    operation: 'fromiter',
    setup: { n: { shape: [scaledSizes.d1] } },
    iterations,
    warmup,
  });

  specs.push({
    name: `full_like [${scaledSizes.d2.join('x')}]`,
    category: 'creation',
    operation: 'full_like',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `greater [${scaledSizes.d2.join('x')}] > [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'greater',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `greater_equal [${scaledSizes.d2.join('x')}] >= [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'greater_equal',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `histogramdd [${scaledSizes.d1}x2]`,
    category: 'statistics',
    operation: 'histogramdd',
    setup: { a: { shape: [scaledSizes.d1, 2], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `hsplit [${scaledSizes.d2.join('x')}] into 2`,
    category: 'manipulation',
    operation: 'hsplit',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `imag [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'imag',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `insert [${scaledSizes.d1}]`,
    category: 'manipulation',
    operation: 'insert',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `intersect1d [${scaledSizes.d1}] [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'intersect1d',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'shuffled' },
      b: { shape: [scaledSizes.d1], fill: 'shuffled' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `isclose [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'isclose',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `iscomplex [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'iscomplex',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `isin [${scaledSizes.d1}] [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'isin',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'shuffled' },
      b: { shape: [scaledSizes.d1], fill: 'shuffled' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `isinf [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'isinf',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `ix_ [64] [64]`,
    category: 'indexing',
    operation: 'ix_',
    setup: {
      a: { shape: [64], fill: 'arange', dtype: 'int32' },
      b: { shape: [64], fill: 'arange', dtype: 'int32' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `less [${scaledSizes.d2.join('x')}] < [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'less',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `less_equal [${scaledSizes.d2.join('x')}] <= [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'less_equal',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `log1p [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'log1p',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `logaddexp2 [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'logaddexp2',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `mask_indices n=${scaledSizes.linalg[0]}`,
    category: 'indexing',
    operation: 'mask_indices',
    setup: { n: { shape: [scaledSizes.linalg[0]] } },
    iterations,
    warmup,
  });

  specs.push({
    name: `meshgrid [256] x2`,
    category: 'manipulation',
    operation: 'meshgrid',
    setup: { a: { shape: [256], fill: 'random' }, b: { shape: [256], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `moveaxis [${scaledSizes.d2.join('x')}]`,
    category: 'manipulation',
    operation: 'moveaxis',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nanargmax [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nanargmax',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nanargmin [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nanargmin',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nancumprod [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nancumprod',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nancumsum [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nancumsum',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nanmedian [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nanmedian',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nanprod [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nanprod',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nanstd [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nanstd',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nanvar [${scaledSizes.d2.join('x')}]`,
    category: 'reductions',
    operation: 'nanvar',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `nextafter [${scaledSizes.d2.join('x')}] [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'nextafter',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `not_equal [${scaledSizes.d2.join('x')}] != [${scaledSizes.d2.join('x')}]`,
    category: 'logic',
    operation: 'not_equal',
    setup: {
      a: { shape: scaledSizes.d2, fill: 'random' },
      b: { shape: scaledSizes.d2, fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `ones_like [${scaledSizes.d2.join('x')}]`,
    category: 'creation',
    operation: 'ones_like',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `place [${scaledSizes.d1}]`,
    category: 'indexing',
    operation: 'place',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      b: { shape: [scaledSizes.d1], dtype: 'bool', fill: 'ones' },
      c: { shape: [1], fill: 'zeros' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `put [${scaledSizes.d1}]`,
    category: 'indexing',
    operation: 'put',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      c: { shape: [3], value: 9 },
      indices: { shape: [0, 2, 4] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `put_along_axis [${scaledSizes.d1}]`,
    category: 'indexing',
    operation: 'put_along_axis',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      c: { shape: [3], value: 9 },
      indices: { shape: [0, 2, 4] },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `putmask [${scaledSizes.d1}]`,
    category: 'indexing',
    operation: 'putmask',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      b: { shape: [scaledSizes.d1], dtype: 'bool', fill: 'ones' },
      c: { shape: [1], fill: 'zeros' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `real [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'real',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `real_if_close [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'real_if_close',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `resize [${scaledSizes.d1}] -> [${scaledSizes.linalg.join('x')}]`,
    category: 'manipulation',
    operation: 'resize',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      new_shape: { shape: scaledSizes.linalg },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `rint [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'rint',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `rollaxis [${scaledSizes.d2.join('x')}]`,
    category: 'manipulation',
    operation: 'rollaxis',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `round [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'round',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `select [${scaledSizes.d1}]`,
    category: 'indexing',
    operation: 'select',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'random' },
      b: { shape: [scaledSizes.d1], dtype: 'bool', fill: 'ones' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `setdiff1d [${scaledSizes.d1}] [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'setdiff1d',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'shuffled' },
      b: { shape: [scaledSizes.d1], fill: 'shuffled' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `setxor1d [${scaledSizes.d1}] [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'setxor1d',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'shuffled' },
      b: { shape: [scaledSizes.d1], fill: 'shuffled' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `sort_complex [${scaledSizes.d1}]`,
    category: 'sorting',
    operation: 'sort_complex',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `spacing [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'spacing',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `split [${scaledSizes.d1}] into 2`,
    category: 'manipulation',
    operation: 'split',
    setup: { a: { shape: [scaledSizes.d1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `squeeze [1x${scaledSizes.d1}x1]`,
    category: 'manipulation',
    operation: 'squeeze',
    setup: { a: { shape: [1, scaledSizes.d1, 1], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `tensordot [8x32x32] [32x32x8]`,
    category: 'linalg',
    operation: 'tensordot',
    setup: {
      a: { shape: [8, 32, 32], fill: 'random' },
      b: { shape: [32, 32, 8], fill: 'random' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `tril_indices_from [${scaledSizes.linalg.join('x')}]`,
    category: 'indexing',
    operation: 'tril_indices_from',
    setup: { a: { shape: scaledSizes.linalg, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `triu_indices_from [${scaledSizes.linalg.join('x')}]`,
    category: 'indexing',
    operation: 'triu_indices_from',
    setup: { a: { shape: scaledSizes.linalg, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `trunc [${scaledSizes.d2.join('x')}]`,
    category: 'math',
    operation: 'trunc',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `union1d [${scaledSizes.d1}] [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'union1d',
    setup: {
      a: { shape: [scaledSizes.d1], fill: 'shuffled' },
      b: { shape: [scaledSizes.d1], fill: 'shuffled' },
    },
    iterations,
    warmup,
  });

  specs.push({
    name: `unique [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'unique',
    setup: { a: { shape: [scaledSizes.d1], fill: 'shuffled' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `unique_all [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'unique_all',
    setup: { a: { shape: [scaledSizes.d1], fill: 'shuffled' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `unique_inverse [${scaledSizes.d1}]`,
    category: 'sets',
    operation: 'unique_inverse',
    setup: { a: { shape: [scaledSizes.d1], fill: 'shuffled' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `vander [32]`,
    category: 'manipulation',
    operation: 'vander',
    setup: { a: { shape: [32], fill: 'random' } },
    iterations,
    warmup,
  });

  specs.push({
    name: `vsplit [${scaledSizes.d2.join('x')}] into 2`,
    category: 'manipulation',
    operation: 'vsplit',
    setup: { a: { shape: scaledSizes.d2, fill: 'random' } },
    iterations,
    warmup,
  });

  if (mode !== 'quick') {
    const expanded: BenchmarkCase[] = [];

    for (const spec of specs) {
      expanded.push(spec);

      // Skip operations that are numerically unstable at lower precision
      if (SKIP_DTYPE_OPERATIONS.has(spec.operation)) continue;

      // Skip handwritten complex specs — they already target complex128
      if (spec.operation.startsWith('complex_')) continue;

      const dataEntries = Object.entries(spec.setup).filter(([key]) => DATA_ARRAY_KEYS.has(key));
      if (dataEntries.length === 0) continue;

      // Category not listed = skip entirely (random, utilities, etc.)
      const families = CATEGORY_DTYPE_SUPPORT[spec.category];
      if (!families) continue;

      // Only vary entries whose dtype is the category's natural base.
      //
      // For float-bearing categories that is float64 (or unset); a pinned
      // non-default dtype there is a semantic pin (e.g. index arrays at int32)
      // and is left alone.
      //
      // Integer-only categories (bitwise) have no float64 base at all — their
      // natural base IS int32/uint32, so those must stay sweepable. Treating
      // them as semantic pins silently collapsed every bitwise op to a single
      // dtype, hiding e.g. bitwise_count's int8 path entirely.
      const intOnlyCategory = !families.includes('float') && !families.includes('complex');
      const intOnlyOperation = INT_ONLY_OPERATIONS.has(spec.operation);
      const intBaseSweepable = intOnlyCategory || intOnlyOperation;
      const isSweepableBase = (dtype: DType | undefined): boolean =>
        !dtype ||
        dtype === 'float64' ||
        (intBaseSweepable && (dtype === 'int32' || dtype === 'uint32'));
      const variableEntries = dataEntries.filter(([, entry]) => isSweepableBase(entry.dtype));
      if (variableEntries.length === 0) continue;

      // Collect all applicable variant dtypes
      for (const family of families) {
        // Integer-only operations take no float or complex variants
        if (intOnlyOperation && family !== 'int' && family !== 'uint') continue;
        // Skip int/uint variants for operations prone to overflow
        if ((family === 'int' || family === 'uint') && SKIP_INT_OPERATIONS.has(spec.operation))
          continue;
        // Skip uint variants for operations NumPy raises TypeError on for unsigned types
        if (family === 'uint' && SKIP_UINT_OPERATIONS.has(spec.operation)) continue;
        // Skip complex variants for unsupported operations
        if (family === 'complex' && SKIP_COMPLEX_OPERATIONS.has(spec.operation)) continue;
        // Skip complex variants for broadcasting specs (complex broadcasting is buggy)
        if (family === 'complex') {
          const shapes = dataEntries.map(([, e]) => JSON.stringify(e.shape));
          if (new Set(shapes).size > 1) continue;
        }

        for (const variant of FAMILY_VARIANTS[family]!) {
          if (MODE_RANK[mode]! < MODE_RANK[variant.minMode]!) continue;

          // Skip variant if it matches the base spec's dtype (would be a duplicate)
          const baseDtype = dataEntries[0]?.[1]?.dtype;
          if (baseDtype && variant.dtype === baseDtype) continue;

          // Skip float16 for numerically sensitive operations
          if (variant.dtype === 'float16' && SKIP_FLOAT16_OPERATIONS.has(spec.operation)) continue;

          // Skip narrow int/uint types for accumulation ops where overflow wrapping differs
          if (
            (family === 'int' || family === 'uint') &&
            (variant.dtype === 'int8' ||
              variant.dtype === 'int16' ||
              variant.dtype === 'uint8' ||
              variant.dtype === 'uint16') &&
            SKIP_NARROW_INT_OPERATIONS.has(spec.operation)
          )
            continue;

          // Skip 64-bit int variants where numpy-ts currently throws
          if (
            (variant.dtype === 'int64' || variant.dtype === 'uint64') &&
            SKIP_INT64_OPERATIONS.has(spec.operation)
          )
            continue;

          // Same, but only for the broadcasting specs of ops whose 64-bit
          // same-shape path works (see SKIP_INT64_BROADCAST_OPERATIONS)
          if (
            (variant.dtype === 'int64' || variant.dtype === 'uint64') &&
            SKIP_INT64_BROADCAST_OPERATIONS.has(spec.operation) &&
            new Set(dataEntries.map(([, e]) => JSON.stringify(e.shape))).size > 1
          )
            continue;

          // Clone setup: vary dtype only on entries without a pinned non-default dtype
          const variantSetup: BenchmarkSetup = {};
          let skipVariant = false;
          const variableKeys = new Set(variableEntries.map(([k]) => k));
          for (const [key, entry] of Object.entries(spec.setup)) {
            if (DATA_ARRAY_KEYS.has(key) && variableKeys.has(key)) {
              const cloned = { ...entry, dtype: variant.dtype as DType };
              if (family === 'complex') {
                // Complex needs fill: 'complex' for all data arrays (arange/ones/zeros with
                // complex dtype have bugs). Skip specs that use value with complex since
                // np.full with complex is broken.
                if (entry.value !== undefined) {
                  skipVariant = true;
                  break;
                }
                cloned.fill = 'complex_small';
              }
              if (family === 'uint' && typeof entry.value === 'number' && entry.value < 0) {
                // NumPy 2.0 raises OverflowError for np.full(shape, negative, dtype=uint*).
                // Skip this variant rather than produce a test that will always mismatch.
                skipVariant = true;
                break;
              }
              variantSetup[key] = cloned;
            } else {
              variantSetup[key] = { ...entry };
            }
          }
          if (skipVariant) continue;

          expanded.push({
            ...spec,
            name: `${spec.name} ${variant.dtype}`,
            setup: variantSetup,
            includeInQuick: undefined, // Never include variants in quick mode
          });
        }
      }
    }

    specs.length = 0;
    specs.push(...expanded);
  }

  // Append dtype suffix to specs with explicit non-default dtype so the
  // visualization layer shows the correct badge instead of defaulting to float64.
  //
  // An unset dtype means float64, so it counts towards the set like any other.
  // Dropping the unset entries instead would let a single pinned auxiliary array
  // name the whole spec: a bool mask beside float64 operands labelled place,
  // putmask and select as the only bool benchmarks in the suite, and the docs
  // then reported bool as the slowest dtype off those three.
  for (const spec of specs) {
    // Already has a dtype suffix from auto-generation? Skip.
    if (
      /\s+(float64|float32|float16|complex128|complex64|int64|int32|int16|int8|uint64|uint32|uint16|uint8|bool)$/.test(
        spec.name,
      )
    )
      continue;
    const dataEntries = Object.entries(spec.setup).filter(([key]) => DATA_ARRAY_KEYS.has(key));
    if (dataEntries.length === 0) continue;
    const dtypes = new Set(dataEntries.map(([, e]) => e.dtype ?? 'float64'));
    if (dtypes.size === 1) {
      const dtype = Array.from(dtypes)[0]!;
      if (dtype !== 'float64') {
        spec.name = `${spec.name} ${dtype}`;
      }
    }
  }

  return specs;
}

export function filterByCategory(specs: BenchmarkCase[], category: string): BenchmarkCase[] {
  return specs.filter((spec) => spec.category === category);
}

export function getCategories(specs: BenchmarkCase[]): string[] {
  const categories = new Set(specs.map((spec) => spec.category));
  return Array.from(categories).sort();
}
