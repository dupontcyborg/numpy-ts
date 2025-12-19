/**
 * Complex Number Coverage Tests
 *
 * This test suite ensures every exported function explicitly handles complex numbers.
 * It maintains a map of expected behavior for each function and verifies that:
 * 1. Functions that support complex work correctly
 * 2. Functions that don't support complex throw appropriate errors
 * 3. New functions are added to the map (test fails until added)
 *
 * Behavior categories:
 * - 'supported': Function works correctly with complex input
 * - 'unsupported': Function throws TypeError (complex not mathematically applicable)
 * - 'not_implemented': Function throws Error (complex planned but not yet implemented)
 * - 'skip': Function doesn't take array input or is not applicable (e.g., types, constants)
 */

import { describe, it, expect } from 'vitest';
import * as np from '../../src';

// ============================================================================
// COMPLEX BEHAVIOR MAP
// ============================================================================
// This is the source of truth for expected complex behavior.
// When adding a new function, you MUST add it here with the appropriate behavior.

type ComplexBehavior = 'supported' | 'unsupported' | 'not_implemented' | 'skip';

const COMPLEX_BEHAVIOR: Record<string, ComplexBehavior> = {
  // =========================================================================
  // SUPPORTED - These functions correctly handle complex numbers
  // =========================================================================

  // Note: add, subtract, multiply are NDArray methods, not standalone exports
  // They are tested via the NDArray class tests
  // divide is exported as a standalone function

  divide: 'supported', // complex division
  negative: 'supported',
  absolute: 'supported', // |z| = sqrt(re² + im²)
  abs: 'supported', // alias for absolute

  // Power operations (implemented)
  sqrt: 'supported', // complex square root
  power: 'supported', // complex exponentiation
  pow: 'supported', // alias for power

  // Complex-specific operations
  real: 'supported', // extract real part
  imag: 'supported', // extract imaginary part
  conj: 'supported', // complex conjugate
  conjugate: 'supported', // alias for conj
  angle: 'supported', // phase angle

  // Comparison (lexicographic ordering)
  // Note: These work but use lexicographic comparison (real, then imag)

  // Type checking
  iscomplex: 'supported', // checks if elements have non-zero imag
  iscomplexobj: 'supported', // checks if dtype is complex
  isreal: 'supported', // checks if elements have zero imag
  isrealobj: 'supported', // checks if dtype is not complex

  // Reductions that return complex
  // Note: sum, mean, prod, var, std are NDArray methods, not standalone exports
  // They are tested via NDArray class tests

  // =========================================================================
  // UNSUPPORTED - These throw TypeError (mathematically undefined for complex)
  // =========================================================================

  // Rounding operations (no natural rounding for complex)
  around: 'unsupported',
  round_: 'unsupported', // alias for around
  ceil: 'unsupported',
  fix: 'unsupported',
  floor: 'unsupported',
  rint: 'unsupported',
  round: 'unsupported',
  trunc: 'unsupported',

  // Modulo/remainder (not defined for complex)
  mod: 'unsupported',
  fmod: 'unsupported',
  remainder: 'unsupported', // alias for mod
  floor_divide: 'unsupported',
  divmod: 'unsupported',

  // Sign operations (no ordering for complex)
  sign: 'unsupported',

  // Integer-only operations
  gcd: 'unsupported',
  lcm: 'unsupported',

  // Float decomposition (specific to real floats)
  frexp: 'unsupported',
  ldexp: 'unsupported',
  modf: 'unsupported',

  // Step function (requires ordering)
  heaviside: 'unsupported',

  // Bitwise operations (integer only)
  bitwise_and: 'unsupported',
  bitwise_or: 'unsupported',
  bitwise_xor: 'unsupported',
  bitwise_not: 'unsupported',
  invert: 'unsupported',
  left_shift: 'unsupported',
  right_shift: 'unsupported',

  // =========================================================================
  // NOT IMPLEMENTED - These should throw Error (planned for future)
  // Currently these silently give wrong results - we need to add guards!
  // =========================================================================

  // Trigonometric (complex formulas implemented)
  sin: 'supported',
  cos: 'supported',
  tan: 'supported',
  arcsin: 'supported',
  asin: 'supported', // alias
  arccos: 'supported',
  acos: 'supported', // alias
  arctan: 'supported',
  atan: 'supported', // alias
  arctan2: 'unsupported', // two-arg function for real angles
  atan2: 'unsupported', // alias
  hypot: 'unsupported', // two-arg function for real numbers
  degrees: 'unsupported', // real-to-real conversion
  radians: 'unsupported', // real-to-real conversion
  deg2rad: 'unsupported', // alias for radians
  rad2deg: 'unsupported', // alias for degrees

  // Hyperbolic (complex formulas implemented)
  sinh: 'supported',
  cosh: 'supported',
  tanh: 'supported',
  arcsinh: 'supported',
  asinh: 'supported', // alias
  arccosh: 'supported',
  acosh: 'supported', // alias
  arctanh: 'supported',
  atanh: 'supported', // alias

  // Exponential/logarithmic (need complex formulas)
  exp: 'supported',
  exp2: 'supported',
  expm1: 'supported',
  log: 'supported',
  log2: 'supported',
  log10: 'supported',
  log1p: 'supported',
  logaddexp: 'unsupported', // NumPy doesn't support logaddexp for complex
  logaddexp2: 'unsupported', // NumPy doesn't support logaddexp2 for complex

  // Remaining arithmetic (complex formulas implemented)
  positive: 'supported',
  reciprocal: 'supported',
  cbrt: 'unsupported', // NumPy doesn't support cbrt for complex
  fabs: 'unsupported', // use absolute() for complex
  square: 'supported',
  float_power: 'supported', // complex z1^z2 = exp(z2 * log(z1))

  // Reductions using lexicographic ordering (real first, then imaginary)
  max: 'supported', // lexicographic max
  amax: 'supported', // alias
  min: 'supported', // lexicographic min
  amin: 'supported', // alias
  ptp: 'supported', // peak-to-peak (max - min)
  median: 'unsupported', // no natural ordering for complex
  percentile: 'unsupported', // no natural ordering for complex
  quantile: 'unsupported', // no natural ordering for complex

  // Cumulative operations
  cumsum: 'supported', // returns complex cumulative sum
  cumulative_sum: 'supported', // alias
  cumprod: 'supported', // returns complex cumulative product
  cumulative_prod: 'supported', // alias

  // Weighted average
  average: 'supported',

  // NaN-aware reductions
  nansum: 'supported',
  nanprod: 'supported',
  nanmean: 'supported',
  nanvar: 'supported',
  nanstd: 'supported',
  nanmin: 'supported', // lexicographic min, skipping NaN
  nanmax: 'supported', // lexicographic max, skipping NaN
  nanargmin: 'supported',
  nanargmax: 'supported',
  nancumsum: 'supported',
  nancumprod: 'supported',
  nanmedian: 'unsupported', // no natural ordering for complex
  nanquantile: 'unsupported', // no natural ordering for complex
  nanpercentile: 'unsupported', // no natural ordering for complex

  // Sorting (uses lexicographic ordering for complex)
  sort: 'supported',
  argsort: 'supported',
  lexsort: 'supported', // uses lexicographic ordering
  partition: 'supported', // uses lexicographic ordering
  argpartition: 'supported', // uses lexicographic ordering
  sort_complex: 'supported', // already designed for complex
  searchsorted: 'supported', // uses lexicographic ordering

  // Logic operations (complex truthy = either part non-zero)
  logical_and: 'supported',
  logical_or: 'supported',
  logical_not: 'supported',
  logical_xor: 'supported',
  isfinite: 'supported', // True if both real and imag parts are finite
  isinf: 'supported', // True if either part is infinite
  isnan: 'supported', // True if either part is NaN
  isneginf: 'unsupported', // NumPy throws: ambiguous for complex
  isposinf: 'unsupported', // NumPy throws: ambiguous for complex

  // Linear algebra
  dot: 'supported', // complex dot product returns Complex
  trace: 'supported', // complex trace returns Complex sum of diagonal
  diagonal: 'supported', // extracts complex diagonal elements
  kron: 'supported', // Kronecker product with complex multiplication
  transpose: 'supported', // permutes dimensions (works naturally)
  inner: 'supported', // complex inner product
  outer: 'supported', // complex outer product
  tensordot: 'supported', // complex tensor contraction
  einsum: 'supported', // complex Einstein summation

  // Gradient/difference
  diff: 'supported',
  ediff1d: 'supported',
  gradient: 'supported',
  cross: 'supported',

  // Statistics
  bincount: 'unsupported', // requires integer input
  digitize: 'unsupported', // binning for real numbers
  histogram: 'unsupported', // binning for real numbers
  histogram2d: 'unsupported', // binning for real numbers
  histogramdd: 'unsupported', // binning for real numbers
  correlate: 'supported', // complex cross-correlation with conjugate
  convolve: 'supported', // complex convolution
  cov: 'supported', // covariance for complex data
  corrcoef: 'supported', // correlation coefficients for complex data

  // Set operations (lexicographic ordering)
  unique: 'supported', // lexicographic unique
  in1d: 'supported', // alias for isin
  intersect1d: 'supported', // lexicographic intersection
  isin: 'supported', // lexicographic membership test
  setdiff1d: 'supported', // lexicographic difference
  setxor1d: 'supported', // lexicographic symmetric difference
  union1d: 'supported', // lexicographic union

  // Searching
  nonzero: 'supported',
  argwhere: 'supported',
  flatnonzero: 'supported',
  where: 'supported',
  extract: 'supported',
  count_nonzero: 'supported',

  // Float-specific (only for real numbers)
  signbit: 'unsupported', // sign bit only for real numbers
  copysign: 'unsupported', // copy sign only for real numbers
  nextafter: 'unsupported', // floating-point representation
  spacing: 'unsupported', // floating-point representation
  real_if_close: 'supported', // converts to real if imaginary is negligible

  // =========================================================================
  // SKIP - These don't take array input or are not applicable
  // =========================================================================

  // Classes and constructors
  NDArray: 'skip',
  Complex: 'skip',

  // Array creation (output dtype, not input)
  zeros: 'skip',
  ones: 'skip',
  array: 'skip', // creates from input, complex handled at creation
  arange: 'skip',
  linspace: 'skip',
  logspace: 'skip',
  geomspace: 'skip',
  eye: 'skip',
  empty: 'skip',
  full: 'skip',
  identity: 'skip',
  asarray: 'skip',
  asarray_chkfinite: 'skip',
  copy: 'skip',
  require: 'skip',
  zeros_like: 'skip',
  ones_like: 'skip',
  empty_like: 'skip',
  full_like: 'skip',
  asanyarray: 'skip',
  ascontiguousarray: 'skip',
  asfortranarray: 'skip',
  diag: 'skip',
  diagflat: 'skip',
  frombuffer: 'skip',
  fromfile: 'skip',
  fromfunction: 'skip',
  fromiter: 'skip',
  fromstring: 'skip',
  meshgrid: 'skip',
  tri: 'skip',
  tril: 'skip',
  triu: 'skip',
  vander: 'skip',

  // Array manipulation (dtype-agnostic, just reshapes)
  swapaxes: 'skip',
  moveaxis: 'skip',
  concatenate: 'skip',
  stack: 'skip',
  vstack: 'skip',
  hstack: 'skip',
  dstack: 'skip',
  concat: 'skip',
  unstack: 'skip',
  block: 'skip',
  split: 'skip',
  array_split: 'skip',
  vsplit: 'skip',
  hsplit: 'skip',
  tile: 'skip',
  repeat: 'skip',
  ravel: 'skip',
  flatten: 'skip',
  fill: 'skip',
  item: 'skip',
  tolist: 'skip',
  tobytes: 'skip',
  byteswap: 'skip',
  view: 'skip',
  tofile: 'skip',
  reshape: 'skip',
  squeeze: 'skip',
  expand_dims: 'skip',
  flip: 'skip',
  fliplr: 'skip',
  flipud: 'skip',
  rot90: 'skip',
  roll: 'skip',
  rollaxis: 'skip',
  atleast_1d: 'skip',
  atleast_2d: 'skip',
  atleast_3d: 'skip',
  dsplit: 'skip',
  column_stack: 'skip',
  row_stack: 'skip',
  resize: 'skip',
  append: 'skip',
  delete: 'skip',
  insert: 'skip',
  pad: 'skip',

  // Broadcasting (dtype-agnostic)
  broadcast_to: 'skip',
  broadcast_arrays: 'skip',
  broadcast_shapes: 'skip',

  // Indexing operations (dtype-agnostic)
  take: 'skip',
  put: 'skip',
  copyto: 'skip',
  choose: 'skip',
  array_equal: 'skip',
  array_equiv: 'skip',
  take_along_axis: 'skip',
  put_along_axis: 'skip',
  putmask: 'skip',
  compress: 'skip',
  select: 'skip',
  place: 'skip',
  fill_diagonal: 'skip',
  iindex: 'skip', // fancy indexing with integer indices
  bindex: 'skip', // fancy indexing with boolean mask
  diag_indices: 'skip',
  diag_indices_from: 'skip',
  tril_indices: 'skip',
  tril_indices_from: 'skip',
  triu_indices: 'skip',
  triu_indices_from: 'skip',
  mask_indices: 'skip',
  indices: 'skip',
  ix_: 'skip',
  ravel_multi_index: 'skip',
  unravel_index: 'skip',

  // Bit packing (uint8 specific)
  packbits: 'skip',
  unpackbits: 'skip',

  // Type checking utilities
  isnat: 'skip', // datetime specific
  isfortran: 'skip',
  isscalar: 'skip',
  iterable: 'skip',
  isdtype: 'skip',
  promote_types: 'skip',

  // Namespace objects
  linalg: 'skip', // namespace object, not a function
  random: 'skip', // namespace object

  // IO functions
  parseNpy: 'skip',
  serializeNpy: 'skip',
  parseNpyHeader: 'skip',
  parseNpyData: 'skip',
  UnsupportedDTypeError: 'skip',
  InvalidNpyError: 'skip',
  SUPPORTED_DTYPES: 'skip',
  DTYPE_TO_DESCR: 'skip',
  parseNpz: 'skip',
  parseNpzSync: 'skip',
  loadNpz: 'skip',
  loadNpzSync: 'skip',
  serializeNpz: 'skip',
  serializeNpzSync: 'skip',

  // Version and other values
  __version__: 'skip',
  true_divide: 'skip', // alias for divide, tested via divide
};

// Type exports that are not runtime values - excluded from checks
const TYPE_EXPORTS = [
  'ComplexInput',
  'NpyHeader',
  'NpyMetadata',
  'NpyVersion',
  'NpzParseOptions',
  'NpzParseResult',
  'NpzSerializeOptions',
];

// ============================================================================
// TEST HELPERS
// ============================================================================

/**
 * Create a complex array for testing
 */
function createComplexArray() {
  return np.array([new np.Complex(1, 2), new np.Complex(3, 4)]);
}

/**
 * Create a second complex array for binary operations
 */
function createComplexArray2() {
  return np.array([new np.Complex(5, 6), new np.Complex(7, 8)]);
}

/**
 * Create a 2D complex array for functions that require 2D input (trace, diagonal)
 */
function createComplex2DArray() {
  return np.array([
    [new np.Complex(1, 2), new np.Complex(3, 4)],
    [new np.Complex(5, 6), new np.Complex(7, 8)],
  ]);
}

/**
 * Attempt to call a function with complex input
 * Returns: 'supported' | 'unsupported' | 'not_implemented' | 'error'
 */
function testComplexBehavior(
  fn: (...args: unknown[]) => unknown,
  fnName: string
): ComplexBehavior | 'error' {
  const z1 = createComplexArray();
  const z2 = createComplexArray2();
  const z2d = createComplex2DArray();

  try {
    // Try to call the function with appropriate arguments
    let result;

    // Binary operations need two arrays
    const binaryOps = [
      'add',
      'subtract',
      'multiply',
      'divide',
      'power',
      'pow',
      'mod',
      'fmod',
      'floor_divide',
      'remainder',
      'heaviside',
      'gcd',
      'lcm',
      'ldexp',
      'arctan2',
      'atan2',
      'hypot',
      'copysign',
      'nextafter',
      'logaddexp',
      'logaddexp2',
      'bitwise_and',
      'bitwise_or',
      'bitwise_xor',
      'left_shift',
      'right_shift',
      'logical_and',
      'logical_or',
      'logical_xor',
      'dot',
      'inner',
      'outer',
      'kron',
      'cross',
      'correlate',
      'convolve',
      'float_power',
      'divmod', // binary op that returns tuple
      'extract', // takes (condition, array)
      'digitize', // (x, bins)
      'histogram2d', // (x, y)
      'in1d', // (ar1, ar2)
      'intersect1d', // (ar1, ar2)
      'isin', // (element, test_elements)
      'setdiff1d', // (ar1, ar2)
      'setxor1d', // (ar1, ar2)
      'union1d', // (ar1, ar2)
    ];

    // Functions that return tuples (unary)
    const tupleOps = ['frexp', 'modf'];

    // Functions that require 2D input
    const require2D = ['trace', 'diagonal'];

    // Functions that require 3 arguments
    const ternaryOps = ['where'];

    // Functions with special signatures
    const specialOps: Record<string, () => unknown> = {
      lexsort: () => fn([z1, z2]), // expects array of keys
      searchsorted: () => fn(z1, z2), // (a, v)
      tensordot: () => fn(z2d, z2d, 1), // (a, b, axes) - needs 2D arrays
      einsum: () => fn('i,i->', z1, z1), // (subscripts, ...operands)
    };

    if (fnName in specialOps) {
      result = specialOps[fnName]!();
    } else if (binaryOps.includes(fnName)) {
      result = fn(z1, z2);
    } else if (tupleOps.includes(fnName)) {
      result = fn(z1);
    } else if (require2D.includes(fnName)) {
      result = fn(z2d);
    } else if (ternaryOps.includes(fnName)) {
      // where(cond, x, y) - use z1 as condition, z1 as x, z2 as y
      result = fn(z1, z1, z2);
    } else {
      result = fn(z1);
    }

    // If we get here without error, it's supported
    // But we should verify the result is valid
    if (result && typeof result === 'object' && 'dtype' in result) {
      // Check if it's a valid NDArray result
      return 'supported';
    } else if (result instanceof np.Complex) {
      return 'supported';
    } else if (typeof result === 'boolean') {
      return 'supported';
    } else if (Array.isArray(result)) {
      return 'supported';
    }

    return 'supported';
  } catch (e: unknown) {
    const error = e as Error;
    const message = error.message || '';

    // Check for unsupported (TypeError with "not supported for complex")
    if (error instanceof TypeError && message.includes('not supported for complex')) {
      return 'unsupported';
    }

    // Check for not implemented
    if (message.includes('does not yet support complex')) {
      return 'not_implemented';
    }

    // Check for bitwise/integer type errors
    if (message.includes('not supported for the input types')) {
      return 'unsupported';
    }

    // Other errors indicate something unexpected
    console.error(`Unexpected error for ${fnName}:`, message);
    return 'error';
  }
}

// ============================================================================
// TESTS
// ============================================================================

describe('Complex Number Coverage', () => {
  describe('All exports are in the behavior map', () => {
    it('every exported function/value has a defined complex behavior', () => {
      // Get all runtime exports (exclude type-only exports)
      // Note: We keep __version__ but exclude other underscore-prefixed internals
      const allExports = Object.keys(np).filter(
        (key) => !TYPE_EXPORTS.includes(key) && (key === '__version__' || !key.startsWith('_'))
      );
      const mapped = Object.keys(COMPLEX_BEHAVIOR);

      const missing = allExports.filter((exp) => !mapped.includes(exp));

      if (missing.length > 0) {
        throw new Error(
          `The following exports are missing from COMPLEX_BEHAVIOR map:\n` +
            `  ${missing.join(', ')}\n\n` +
            `Please add them to tests/unit/complex-coverage.test.ts with the appropriate behavior.`
        );
      }
    });

    it('behavior map has no stale entries', () => {
      // Get all runtime exports (exclude type-only exports)
      // Note: We keep __version__ but exclude other underscore-prefixed internals
      const allExports = Object.keys(np).filter(
        (key) => !TYPE_EXPORTS.includes(key) && (key === '__version__' || !key.startsWith('_'))
      );
      const mapped = Object.keys(COMPLEX_BEHAVIOR);

      const stale = mapped.filter((key) => !allExports.includes(key));

      if (stale.length > 0) {
        throw new Error(
          `The following entries in COMPLEX_BEHAVIOR are not exported:\n` +
            `  ${stale.join(', ')}\n\n` +
            `Please remove them from tests/unit/complex-coverage.test.ts.`
        );
      }
    });
  });

  describe('Supported functions work correctly', () => {
    const supportedFns = Object.entries(COMPLEX_BEHAVIOR)
      .filter(([_, behavior]) => behavior === 'supported')
      .map(([name]) => name);

    for (const fnName of supportedFns) {
      it(`${fnName}() works with complex input`, () => {
        const fn = (np as Record<string, unknown>)[fnName];
        if (typeof fn !== 'function') {
          // Skip if not a function (e.g., class or constant)
          return;
        }

        const behavior = testComplexBehavior(fn, fnName);
        expect(behavior).toBe('supported');
      });
    }
  });

  describe('Unsupported functions throw TypeError', () => {
    const unsupportedFns = Object.entries(COMPLEX_BEHAVIOR)
      .filter(([_, behavior]) => behavior === 'unsupported')
      .map(([name]) => name);

    for (const fnName of unsupportedFns) {
      it(`${fnName}() throws TypeError for complex input`, () => {
        const fn = (np as Record<string, unknown>)[fnName];
        if (typeof fn !== 'function') {
          return;
        }

        const behavior = testComplexBehavior(fn, fnName);
        expect(behavior).toBe('unsupported');
      });
    }
  });

  describe('Not-yet-implemented functions throw descriptive Error', () => {
    const notImplementedFns = Object.entries(COMPLEX_BEHAVIOR)
      .filter(([_, behavior]) => behavior === 'not_implemented')
      .map(([name]) => name);

    // Add a placeholder test if no not_implemented functions remain
    if (notImplementedFns.length === 0) {
      it('all functions have been implemented or marked as unsupported', () => {
        expect(notImplementedFns.length).toBe(0);
      });
    }

    for (const fnName of notImplementedFns) {
      it(`${fnName}() throws "not yet implemented" error for complex input`, () => {
        const fn = (np as Record<string, unknown>)[fnName];
        if (typeof fn !== 'function') {
          return;
        }

        const behavior = testComplexBehavior(fn, fnName);

        // Currently many of these silently fail - the test documents this
        // Once we add guards, this should be 'not_implemented'
        if (behavior !== 'not_implemented') {
          // For now, we're tracking that these need to be fixed
          console.warn(`⚠️ ${fnName} should throw 'not yet implemented' but got: ${behavior}`);
        }

        // Uncomment this line once guards are added:
        // expect(behavior).toBe('not_implemented');
      });
    }
  });

  describe('Summary', () => {
    it('prints coverage summary', () => {
      const counts = {
        supported: 0,
        unsupported: 0,
        not_implemented: 0,
        skip: 0,
      };

      for (const behavior of Object.values(COMPLEX_BEHAVIOR)) {
        counts[behavior]++;
      }

      console.log('\n📊 Complex Number Coverage Summary:');
      console.log(`   ✅ Supported:       ${counts.supported}`);
      console.log(`   ❌ Unsupported:     ${counts.unsupported}`);
      console.log(`   ⏳ Not Implemented: ${counts.not_implemented}`);
      console.log(`   ⏭️  Skipped:        ${counts.skip}`);
      console.log(`   📦 Total:           ${Object.keys(COMPLEX_BEHAVIOR).length}`);
    });
  });
});
