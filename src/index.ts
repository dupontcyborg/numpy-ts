/**
 * numpy-ts - Complete NumPy implementation for TypeScript and JavaScript
 *
 * This is the main entry point with full method chaining support.
 * Functions return NDArray which has methods like .add(), .reshape(), etc.
 *
 * @example
 * ```typescript
 * import { array, zeros } from 'numpy-ts';
 *
 * const a = array([1, 2, 3, 4]);
 * const b = a.add(10).reshape([2, 2]).T;  // Method chaining works!
 * ```
 *
 * For tree-shakeable imports (smaller bundles, no method chaining), use:
 * ```typescript
 * import { array, add, reshape } from 'numpy-ts/core';
 * ```
 *
 * @module numpy-ts
 */

// ============================================================
// Core Types and Classes
// ============================================================

export { Complex, type ComplexInput } from './common/complex';
export { hasFloat16 } from './common/dtype';
export type { DType } from './common/dtype';
export { NDArrayCore } from './common/ndarray-core';
export { NDArray } from './full';

// ============================================================
// Array Creation Functions
// ============================================================

export {
  arange,
  array,
  asanyarray,
  asarray,
  asarray_chkfinite,
  ascontiguousarray,
  asfortranarray,
  copy,
  diag,
  diagflat,
  empty,
  empty_like,
  eye,
  frombuffer,
  fromfile,
  fromfunction,
  fromiter,
  fromstring,
  full,
  full_like,
  geomspace,
  identity,
  linspace,
  logspace,
  meshgrid,
  ones,
  ones_like,
  require,
  tri,
  tril,
  triu,
  vander,
  zeros,
  zeros_like,
} from './full';

// ============================================================
// Arithmetic and Mathematical Functions
// ============================================================

// Aliases
export {
  // Other arithmetic
  absolute,
  absolute as abs,
  // Basic arithmetic
  add,
  cbrt,
  clip,
  divide,
  divmod,
  exp,
  exp2,
  expm1,
  fabs,
  float_power,
  floor_divide,
  fmax,
  fmin,
  fmod,
  frexp,
  gcd,
  heaviside,
  i0,
  interp,
  lcm,
  ldexp,
  log,
  log1p,
  log2,
  log10,
  logaddexp,
  logaddexp2,
  maximum,
  minimum,
  mod,
  modf,
  multiply,
  nan_to_num,
  negative,
  positive,
  power,
  power as pow,
  reciprocal,
  remainder,
  sign,
  sinc,
  // Exponential and logarithmic
  sqrt,
  square,
  subtract,
  true_divide,
  unwrap,
} from './full';

// ============================================================
// Trigonometric and Hyperbolic Functions
// ============================================================

// Aliases
export {
  arccos,
  arccos as acos,
  arccosh,
  arccosh as acosh,
  arcsin,
  arcsin as asin,
  arcsinh,
  arcsinh as asinh,
  arctan,
  arctan as atan,
  arctan2,
  arctan2 as atan2,
  arctanh,
  arctanh as atanh,
  cos,
  cosh,
  deg2rad,
  degrees,
  hypot,
  rad2deg,
  radians,
  sin,
  sinh,
  tan,
  tanh,
} from './full';

// ============================================================
// Linear Algebra Functions
// ============================================================

export {
  cross,
  diagonal,
  dot,
  einsum,
  einsum_path,
  inner,
  kron,
  linalg,
  matmul,
  matrix_transpose,
  matvec,
  outer,
  permute_dims,
  tensordot,
  trace,
  transpose,
  vdot,
  vecdot,
  vecmat,
} from './full';

// ============================================================
// Shape Manipulation Functions
// ============================================================

// Alias
export {
  append,
  array_split,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  block,
  column_stack,
  concatenate,
  concatenate as concat,
  delete_,
  delete_ as delete,
  dsplit,
  dstack,
  expand_dims,
  flatten,
  flip,
  fliplr,
  flipud,
  hsplit,
  hstack,
  insert,
  moveaxis,
  pad,
  ravel,
  repeat,
  reshape,
  resize,
  roll,
  rollaxis,
  rot90,
  row_stack,
  split,
  squeeze,
  stack,
  swapaxes,
  tile,
  unstack,
  vsplit,
  vstack,
} from './full';

// ============================================================
// Reduction Functions
// ============================================================

// Aliases
export {
  all,
  amax,
  amax as max,
  amin,
  amin as min,
  any,
  argmax,
  argmin,
  average,
  cumprod,
  cumprod as cumulative_prod,
  cumsum,
  cumsum as cumulative_sum,
  mean,
  median,
  nanargmax,
  nanargmin,
  nancumprod,
  nancumsum,
  nanmax,
  nanmean,
  nanmedian,
  nanmin,
  nanpercentile,
  nanprod,
  nanquantile,
  nanstd,
  nansum,
  nanvar,
  percentile,
  prod,
  ptp,
  quantile,
  std,
  sum,
  var_ as var,
  variance,
} from './full';

// ============================================================
// Logic Functions
// ============================================================

export {
  allclose,
  copysign,
  equal,
  greater,
  greater_equal,
  isclose,
  iscomplex,
  iscomplexobj,
  isdtype,
  isfinite,
  isfortran,
  isinf,
  isnan,
  isnat,
  isneginf,
  isposinf,
  isreal,
  isrealobj,
  isscalar,
  iterable,
  less,
  less_equal,
  logical_and,
  logical_not,
  logical_or,
  logical_xor,
  nextafter,
  not_equal,
  promote_types,
  real_if_close,
  signbit,
  spacing,
} from './full';

// ============================================================
// Sorting and Searching Functions
// ============================================================

export {
  argpartition,
  argsort,
  argwhere,
  count_nonzero,
  extract,
  flatnonzero,
  lexsort,
  nonzero,
  partition,
  searchsorted,
  sort,
  sort_complex,
  where,
} from './full';

// ============================================================
// Bitwise Functions
// ============================================================

// Aliases
export {
  bitwise_and,
  bitwise_count,
  bitwise_not,
  bitwise_not as bitwise_invert,
  bitwise_or,
  bitwise_xor,
  invert,
  left_shift,
  left_shift as bitwise_left_shift,
  packbits,
  right_shift,
  right_shift as bitwise_right_shift,
  unpackbits,
} from './full';

// ============================================================
// Rounding Functions
// ============================================================

// Alias
export { around, around as round, ceil, fix, floor, rint, trunc } from './full';

// ============================================================
// Set Operations
// ============================================================

export {
  in1d,
  intersect1d,
  isin,
  setdiff1d,
  setxor1d,
  trim_zeros,
  union1d,
  unique,
  unique_all,
  unique_counts,
  unique_inverse,
  unique_values,
} from './full';

// ============================================================
// Statistics Functions
// ============================================================

export {
  bincount,
  convolve,
  corrcoef,
  correlate,
  cov,
  digitize,
  histogram,
  histogram_bin_edges,
  histogram2d,
  histogramdd,
  trapezoid,
} from './full';

// ============================================================
// Gradient Functions
// ============================================================

export { diff, ediff1d, gradient } from './full';

// ============================================================
// Complex Number Functions
// ============================================================

// Alias
export { angle, conj, conj as conjugate, imag, real } from './full';

// ============================================================
// Advanced Indexing and Data Manipulation
// ============================================================

export type { NDIndex } from './full';
export {
  array_equal,
  array_equiv,
  bindex,
  broadcast_arrays,
  broadcast_shapes,
  broadcast_to,
  byteswap,
  choose,
  compress,
  copyto,
  diag_indices,
  diag_indices_from,
  fill,
  fill_diagonal,
  iindex,
  indices,
  item,
  ix_,
  mask_indices,
  place,
  put,
  put_along_axis,
  putmask,
  ravel_multi_index,
  select,
  take,
  take_along_axis,
  tobytes,
  tofile,
  tolist,
  tril_indices,
  tril_indices_from,
  triu_indices,
  triu_indices_from,
  unravel_index,
  view,
  vindex,
} from './full';

// ============================================================
// Utility Functions
// ============================================================

export {
  apply_along_axis,
  apply_over_axes,
  geterr,
  may_share_memory,
  ndim,
  seterr,
  shape,
  shares_memory,
  size,
} from './full';

// ============================================================
// Printing/Formatting Functions
// ============================================================

export {
  array_repr,
  array_str,
  array2string,
  base_repr,
  binary_repr,
  format_float_positional,
  format_float_scientific,
  get_printoptions,
  printoptions,
  set_printoptions,
} from './full';

// ============================================================
// Type Checking Functions
// ============================================================

export {
  can_cast,
  common_type,
  issubdtype,
  min_scalar_type,
  mintypecode,
  result_type,
  typename,
} from './full';

// ============================================================
// Polynomial Functions
// ============================================================

export {
  poly,
  polyadd,
  polyder,
  polydiv,
  polyfit,
  polyint,
  polymul,
  polysub,
  polyval,
  roots,
} from './full';

// ============================================================
// IO Functions
// ============================================================

// WASM configuration (for testing — set thresholdMultiplier to 0 to force WASM, Infinity to disable)
export { wasmConfig } from './common/wasm/config';
export { configureWasm, wasmFreeBytes } from './common/wasm/runtime';
// Types, errors, and constants - re-export directly (no wrapping needed)
export {
  DTYPE_TO_DESCR,
  InvalidNpyError,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
  SUPPORTED_DTYPES,
  UnsupportedDTypeError,
} from './io/npy/format';
// Serializers - re-export directly (they accept NDArrayCore, and NDArray extends NDArrayCore)
export { serializeNpy } from './io/npy/serializer';
export type { NpzParseOptions } from './io/npz/parser';
export type { NpzSerializeOptions } from './io/npz/serializer';
export { serializeNpz, serializeNpzSync } from './io/npz/serializer';
export type { ParseTxtOptions } from './io/txt/parser';
export type { SerializeTxtOptions } from './io/txt/serializer';
export { serializeTxt } from './io/txt/serializer';

import type { DType as DTypeIO } from './common/dtype';
import type { NpyMetadata as NpyMetadataType } from './io/npy/format';
// Parsers - wrap to return NDArray instead of NDArrayCore
import {
  parseNpy as parseNpyCore,
  parseNpyData as parseNpyDataCore,
  parseNpyHeader as parseNpyHeaderCore,
} from './io/npy/parser';
import type { NpzParseOptions as NpzParseOptionsType } from './io/npz/parser';
import {
  loadNpz as loadNpzCore,
  loadNpzSync as loadNpzSyncCore,
  parseNpz as parseNpzCore,
  parseNpzSync as parseNpzSyncCore,
} from './io/npz/parser';
import type { ParseTxtOptions as ParseTxtOptionsType } from './io/txt/parser';
import {
  fromregex as fromregexCore,
  genfromtxt as genfromtxtCore,
  parseTxt as parseTxtCore,
} from './io/txt/parser';

// Re-export NpzArraysInput with NDArray types
export type NpzArraysInput =
  | NDArrayClass[]
  | Map<string, NDArrayClass>
  | Record<string, NDArrayClass>;

// Re-export NpzParseResult with NDArray types
export interface NpzParseResult {
  arrays: Map<string, NDArrayClass>;
  skipped: string[];
  errors: Map<string, string>;
}

// Helper to upgrade NDArrayCore to NDArray
function upgradeToNDArray(core: NDArrayCore): NDArrayClass {
  return NDArrayClass.fromStorage(core.storage);
}

// NPY parsers
export function parseNpy(buffer: ArrayBuffer | Uint8Array): NDArrayClass {
  return upgradeToNDArray(parseNpyCore(buffer));
}
export const parseNpyHeader = parseNpyHeaderCore;
export function parseNpyData(bytes: Uint8Array, metadata: NpyMetadataType): NDArrayClass {
  return upgradeToNDArray(parseNpyDataCore(bytes, metadata));
}

// NPZ parsers
export async function parseNpz(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptionsType = {},
): Promise<NpzParseResult> {
  const result = await parseNpzCore(buffer, options);
  const arrays = new Map<string, NDArrayClass>();
  for (const [name, arr] of result.arrays) {
    arrays.set(name, upgradeToNDArray(arr));
  }
  return { arrays, skipped: result.skipped, errors: result.errors };
}

export function parseNpzSync(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptionsType = {},
): NpzParseResult {
  const result = parseNpzSyncCore(buffer, options);
  const arrays = new Map<string, NDArrayClass>();
  for (const [name, arr] of result.arrays) {
    arrays.set(name, upgradeToNDArray(arr));
  }
  return { arrays, skipped: result.skipped, errors: result.errors };
}

export async function loadNpz(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptionsType = {},
): Promise<Record<string, NDArrayClass>> {
  const result = await loadNpzCore(buffer, options);
  const upgraded: Record<string, NDArrayClass> = {};
  for (const [name, arr] of Object.entries(result)) {
    upgraded[name] = upgradeToNDArray(arr);
  }
  return upgraded;
}

export function loadNpzSync(
  buffer: ArrayBuffer | Uint8Array,
  options: NpzParseOptionsType = {},
): Record<string, NDArrayClass> {
  const result = loadNpzSyncCore(buffer, options);
  const upgraded: Record<string, NDArrayClass> = {};
  for (const [name, arr] of Object.entries(result)) {
    upgraded[name] = upgradeToNDArray(arr);
  }
  return upgraded;
}

// Text parsers
export function parseTxt(text: string, options: ParseTxtOptionsType = {}): NDArrayClass {
  return upgradeToNDArray(parseTxtCore(text, options));
}

export function genfromtxt(text: string, options: ParseTxtOptionsType = {}): NDArrayClass {
  return upgradeToNDArray(genfromtxtCore(text, options));
}

export function fromregex(
  text: string,
  regexp: RegExp | string,
  dtype: DTypeIO = 'float64',
): NDArrayClass {
  return upgradeToNDArray(fromregexCore(text, regexp, dtype));
}

// File IO functions (loadNpy, saveNpy, loadtxt, savetxt, etc.)
// These use lazy fs resolution — work in Node/Bun/Deno, throw clear errors in browsers.
// Note: genfromtxt/fromregex are excluded to avoid conflict with the buffer-based versions above.
// Use the file-based versions via genfromtxtFile/fromregexFile, or use loadtxt for file-based text loading.
export {
  fromregex as fromregexFile,
  fromregexSync as fromregexFileSync,
  // File-based text parsers (distinct from buffer-based genfromtxt/fromregex above)
  genfromtxt as genfromtxtFile,
  genfromtxtSync as genfromtxtFileSync,
  // Types
  type LoadOptions,
  type LoadTxtOptions,
  // Auto-detect
  load,
  // NPY
  loadNpy,
  loadNpySync,
  // NPZ
  loadNpzFile,
  loadNpzFileSync,
  loadSync,
  // Text
  loadtxt,
  loadtxtSync,
  type SaveNpzOptions,
  type SaveTxtOptions,
  save,
  saveNpy,
  saveNpySync,
  saveNpz as saveNpzFile,
  saveNpzSync as saveNpzFileSync,
  saveSync,
  savetxt,
  savetxtSync,
  savez,
  savez_compressed,
} from './io/file-ops';

// ============================================================
// Random Namespace (np.random)
// ============================================================

import type { DType } from './common/dtype';
import * as randomOps from './common/ops/random';
import type { ArrayStorage } from './common/storage';
import { NDArray as NDArrayClass } from './full';

// Helper to wrap ArrayStorage results in NDArray
function wrapResult<T>(result: T): T | NDArrayClass {
  if (result && typeof result === 'object' && '_data' in result && '_shape' in result) {
    return NDArrayClass.fromStorage(result as unknown as ArrayStorage);
  }
  return result;
}

export const random = {
  // State management
  seed: randomOps.seed,
  get_state: randomOps.get_state,
  set_state: randomOps.set_state,
  get_bit_generator: randomOps.get_bit_generator,
  set_bit_generator: randomOps.set_bit_generator,
  default_rng: (seedValue?: number) => {
    type ArrayLike = NDArrayClass | NDArrayCore | ArrayStorage;
    const gen = randomOps.default_rng(seedValue);
    const unwrap = (x: number | ArrayLike): number | ArrayStorage =>
      x instanceof NDArrayClass
        ? x.storage
        : x instanceof NDArrayCore
          ? x.storage
          : (x as number | ArrayStorage);
    return {
      random: (size?: number | number[]) => wrapResult(gen.random(size)),
      integers: (low: number, high?: number, size?: number | number[]) =>
        wrapResult(gen.integers(low, high, size)),
      standard_normal: (size?: number | number[]) => wrapResult(gen.standard_normal(size)),
      normal: (loc?: number, scale?: number, size?: number | number[]) =>
        wrapResult(gen.normal(loc, scale, size)),
      uniform: (low?: number, high?: number, size?: number | number[]) =>
        wrapResult(gen.uniform(low, high, size)),
      choice: (
        a: number | ArrayLike,
        size?: number | number[],
        replace?: boolean,
        p?: ArrayLike | number[],
      ) =>
        wrapResult(
          gen.choice(
            unwrap(a),
            size,
            replace,
            p instanceof NDArrayClass
              ? p.storage
              : p instanceof NDArrayCore
                ? p.storage
                : (p as ArrayStorage | number[] | undefined),
          ),
        ),
      permutation: (x: number | ArrayLike) => wrapResult(gen.permutation(unwrap(x))),
      shuffle: (x: ArrayLike) =>
        gen.shuffle(
          x instanceof NDArrayClass ? x.storage : x instanceof NDArrayCore ? x.storage : x,
        ),
      exponential: (scale?: number, size?: number | number[]) =>
        wrapResult(gen.exponential(scale, size)),
      poisson: (lam?: number, size?: number | number[]) => wrapResult(gen.poisson(lam, size)),
      binomial: (n: number, p: number, size?: number | number[]) =>
        wrapResult(gen.binomial(n, p, size)),
    };
  },
  Generator: randomOps.Generator,

  // Basic random functions
  random: (size?: number | number[]) => wrapResult(randomOps.random(size)),
  rand: (...shape: number[]) => wrapResult(randomOps.rand(...shape)),
  randn: (...shape: number[]) => wrapResult(randomOps.randn(...shape)),
  randint: (low: number, high?: number | null, size?: number | number[], dtype?: DType) =>
    wrapResult(randomOps.randint(low, high, size, dtype)),

  // Aliases
  random_sample: (size?: number | number[]) => wrapResult(randomOps.random_sample(size)),
  ranf: (size?: number | number[]) => wrapResult(randomOps.ranf(size)),
  sample: (size?: number | number[]) => wrapResult(randomOps.sample(size)),
  random_integers: (low: number, high?: number, size?: number | number[]) =>
    wrapResult(randomOps.random_integers(low, high, size)),

  // Infrastructure
  bytes: randomOps.bytes,

  // Continuous distributions
  uniform: (low?: number, high?: number, size?: number | number[]) =>
    wrapResult(randomOps.uniform(low, high, size)),
  normal: (loc?: number, scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.normal(loc, scale, size)),
  standard_normal: (size?: number | number[]) => wrapResult(randomOps.standard_normal(size)),
  exponential: (scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.exponential(scale, size)),
  standard_exponential: (size?: number | number[]) =>
    wrapResult(randomOps.standard_exponential(size)),

  // Gamma family
  gamma: (shape: number, scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.gamma(shape, scale, size)),
  standard_gamma: (shape: number, size?: number | number[]) =>
    wrapResult(randomOps.standard_gamma(shape, size)),
  beta: (a: number, b: number, size?: number | number[]) => wrapResult(randomOps.beta(a, b, size)),
  chisquare: (df: number, size?: number | number[]) => wrapResult(randomOps.chisquare(df, size)),
  noncentral_chisquare: (df: number, nonc: number, size?: number | number[]) =>
    wrapResult(randomOps.noncentral_chisquare(df, nonc, size)),
  f: (dfnum: number, dfden: number, size?: number | number[]) =>
    wrapResult(randomOps.f(dfnum, dfden, size)),
  noncentral_f: (dfnum: number, dfden: number, nonc: number, size?: number | number[]) =>
    wrapResult(randomOps.noncentral_f(dfnum, dfden, nonc, size)),

  // Other continuous distributions
  standard_cauchy: (size?: number | number[]) => wrapResult(randomOps.standard_cauchy(size)),
  standard_t: (df: number, size?: number | number[]) => wrapResult(randomOps.standard_t(df, size)),
  laplace: (loc?: number, scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.laplace(loc, scale, size)),
  logistic: (loc?: number, scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.logistic(loc, scale, size)),
  lognormal: (mean?: number, sigma?: number, size?: number | number[]) =>
    wrapResult(randomOps.lognormal(mean, sigma, size)),
  gumbel: (loc?: number, scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.gumbel(loc, scale, size)),
  pareto: (a: number, size?: number | number[]) => wrapResult(randomOps.pareto(a, size)),
  power: (a: number, size?: number | number[]) => wrapResult(randomOps.power(a, size)),
  rayleigh: (scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.rayleigh(scale, size)),
  triangular: (left: number, mode: number, right: number, size?: number | number[]) =>
    wrapResult(randomOps.triangular(left, mode, right, size)),
  wald: (mean: number, scale: number, size?: number | number[]) =>
    wrapResult(randomOps.wald(mean, scale, size)),
  weibull: (a: number, size?: number | number[]) => wrapResult(randomOps.weibull(a, size)),

  // Discrete distributions
  poisson: (lam?: number, size?: number | number[]) => wrapResult(randomOps.poisson(lam, size)),
  binomial: (n: number, p: number, size?: number | number[]) =>
    wrapResult(randomOps.binomial(n, p, size)),
  geometric: (p: number, size?: number | number[]) => wrapResult(randomOps.geometric(p, size)),
  hypergeometric: (ngood: number, nbad: number, nsample: number, size?: number | number[]) =>
    wrapResult(randomOps.hypergeometric(ngood, nbad, nsample, size)),
  logseries: (p: number, size?: number | number[]) => wrapResult(randomOps.logseries(p, size)),
  negative_binomial: (n: number, p: number, size?: number | number[]) =>
    wrapResult(randomOps.negative_binomial(n, p, size)),
  zipf: (a: number, size?: number | number[]) => wrapResult(randomOps.zipf(a, size)),

  // Multivariate distributions
  multinomial: (n: number, pvals: number[] | ArrayStorage, size?: number | number[]) =>
    wrapResult(randomOps.multinomial(n, pvals, size)),
  multivariate_normal: (
    mean: number[] | ArrayStorage,
    cov: number[][] | ArrayStorage,
    size?: number | number[],
    check_valid?: 'warn' | 'raise' | 'ignore',
    tol?: number,
  ) => wrapResult(randomOps.multivariate_normal(mean, cov, size, check_valid, tol)),
  dirichlet: (alpha: number[] | ArrayStorage, size?: number | number[]) =>
    wrapResult(randomOps.dirichlet(alpha, size)),
  vonmises: (mu: number, kappa: number, size?: number | number[]) =>
    wrapResult(randomOps.vonmises(mu, kappa, size)),

  // Sequence operations
  choice: (
    a: number | ArrayStorage,
    size?: number | number[],
    replace?: boolean,
    p?: ArrayStorage | number[],
  ) => wrapResult(randomOps.choice(a, size, replace, p)),
  permutation: (x: number | ArrayStorage) => wrapResult(randomOps.permutation(x)),
  shuffle: (x: NDArrayClass | NDArrayCore | ArrayStorage) =>
    randomOps.shuffle(
      x instanceof NDArrayClass ? x.storage : x instanceof NDArrayCore ? x.storage : x,
    ),
};

// ============================================================
// FFT Namespace (np.fft)
// ============================================================

import { NDArrayCore } from './common/ndarray-core';
import * as fftOps from './common/ops/fft';

// Type alias for array inputs - accepts NDArray, NDArrayCore, or ArrayStorage
type ArrayInput = NDArrayClass | NDArrayCore | ArrayStorage;

// Helper to extract storage from any array input
function getStorage(a: ArrayInput): ArrayStorage {
  if (a instanceof NDArrayClass) return a.storage;
  if (a instanceof NDArrayCore) return a.storage;
  return a;
}

export const fft = {
  fft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.fft(getStorage(a), n, axis, norm));
  },
  ifft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.ifft(getStorage(a), n, axis, norm));
  },
  fft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.fft2(getStorage(a), s, axes, norm));
  },
  ifft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.ifft2(getStorage(a), s, axes, norm));
  },
  fftn: (a: ArrayInput, s?: number[], axes?: number[], norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.fftn(getStorage(a), s, axes, norm));
  },
  ifftn: (
    a: ArrayInput,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.ifftn(getStorage(a), s, axes, norm));
  },
  rfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.rfft(getStorage(a), n, axis, norm));
  },
  irfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.irfft(getStorage(a), n, axis, norm));
  },
  rfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.rfft2(getStorage(a), s, axes, norm));
  },
  irfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.irfft2(getStorage(a), s, axes, norm));
  },
  rfftn: (
    a: ArrayInput,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.rfftn(getStorage(a), s, axes, norm));
  },
  irfftn: (
    a: ArrayInput,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward',
  ) => {
    return NDArrayClass.fromStorage(fftOps.irfftn(getStorage(a), s, axes, norm));
  },
  hfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.hfft(getStorage(a), n, axis, norm));
  },
  ihfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass.fromStorage(fftOps.ihfft(getStorage(a), n, axis, norm));
  },
  fftfreq: (n: number, d?: number) => {
    return NDArrayClass.fromStorage(fftOps.fftfreq(n, d));
  },
  rfftfreq: (n: number, d?: number) => {
    return NDArrayClass.fromStorage(fftOps.rfftfreq(n, d));
  },
  fftshift: (a: ArrayInput, axes?: number | number[]) => {
    return NDArrayClass.fromStorage(fftOps.fftshift(getStorage(a), axes));
  },
  ifftshift: (a: ArrayInput, axes?: number | number[]) => {
    return NDArrayClass.fromStorage(fftOps.ifftshift(getStorage(a), axes));
  },
};

// ============================================================
// Version
// ============================================================

declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : 'dev';
