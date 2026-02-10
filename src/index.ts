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
export { NDArray } from './full';
export { NDArrayCore } from './common/ndarray-core';

// ============================================================
// Array Creation Functions
// ============================================================

export {
  zeros,
  ones,
  empty,
  full,
  array,
  arange,
  linspace,
  logspace,
  geomspace,
  eye,
  identity,
  asarray,
  zeros_like,
  ones_like,
  empty_like,
  full_like,
  copy,
  asanyarray,
  asarray_chkfinite,
  ascontiguousarray,
  asfortranarray,
  require,
  diag,
  diagflat,
  frombuffer,
  fromfile,
  fromfunction,
  fromiter,
  fromstring,
  meshgrid,
  tri,
  tril,
  triu,
  vander,
} from './full';

// ============================================================
// Arithmetic and Mathematical Functions
// ============================================================

export {
  // Basic arithmetic
  add,
  subtract,
  multiply,
  divide,
  true_divide,
  floor_divide,
  mod,
  // Exponential and logarithmic
  sqrt,
  power,
  exp,
  exp2,
  expm1,
  log,
  log2,
  log10,
  log1p,
  logaddexp,
  logaddexp2,
  // Other arithmetic
  absolute,
  negative,
  sign,
  positive,
  reciprocal,
  cbrt,
  fabs,
  divmod,
  square,
  remainder,
  heaviside,
  float_power,
  fmod,
  frexp,
  gcd,
  lcm,
  ldexp,
  modf,
  clip,
  maximum,
  minimum,
  fmax,
  fmin,
  nan_to_num,
  interp,
  unwrap,
  sinc,
  i0,
} from './full';

// Aliases
export { power as pow, absolute as abs } from './full';

// ============================================================
// Trigonometric and Hyperbolic Functions
// ============================================================

export {
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  arctan2,
  hypot,
  degrees,
  radians,
  deg2rad,
  rad2deg,
  sinh,
  cosh,
  tanh,
  arcsinh,
  arccosh,
  arctanh,
} from './full';

// Aliases
export {
  arcsin as asin,
  arccos as acos,
  arctan as atan,
  arctan2 as atan2,
  arcsinh as asinh,
  arccosh as acosh,
  arctanh as atanh,
} from './full';

// ============================================================
// Linear Algebra Functions
// ============================================================

export {
  dot,
  trace,
  diagonal,
  kron,
  transpose,
  inner,
  outer,
  tensordot,
  einsum,
  einsum_path,
  vdot,
  vecdot,
  matrix_transpose,
  permute_dims,
  matvec,
  vecmat,
  cross,
  linalg,
} from './full';

// ============================================================
// Shape Manipulation Functions
// ============================================================

export {
  reshape,
  flatten,
  ravel,
  squeeze,
  expand_dims,
  swapaxes,
  moveaxis,
  rollaxis,
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  column_stack,
  row_stack,
  block,
  split,
  array_split,
  vsplit,
  hsplit,
  dsplit,
  unstack,
  tile,
  repeat,
  flip,
  fliplr,
  flipud,
  rot90,
  roll,
  resize,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  append,
  delete_,
  delete_ as delete,
  insert,
  pad,
} from './full';

// Alias
export { concatenate as concat } from './full';

// ============================================================
// Reduction Functions
// ============================================================

export {
  sum,
  mean,
  prod,
  amax,
  amin,
  ptp,
  argmin,
  argmax,
  std,
  variance,
  median,
  percentile,
  quantile,
  average,
  all,
  any,
  cumsum,
  cumprod,
  nansum,
  nanprod,
  nanmean,
  nanvar,
  nanstd,
  nanmin,
  nanmax,
  nanargmin,
  nanargmax,
  nancumsum,
  nancumprod,
  nanmedian,
  nanquantile,
  nanpercentile,
} from './full';

// Aliases
export {
  amax as max,
  amin as min,
  cumsum as cumulative_sum,
  cumprod as cumulative_prod,
} from './full';

// ============================================================
// Logic Functions
// ============================================================

export {
  logical_and,
  logical_or,
  logical_not,
  logical_xor,
  isfinite,
  isinf,
  isnan,
  isnat,
  isneginf,
  isposinf,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
  real_if_close,
  isfortran,
  isscalar,
  iterable,
  isdtype,
  promote_types,
  copysign,
  signbit,
  nextafter,
  spacing,
} from './full';

// ============================================================
// Sorting and Searching Functions
// ============================================================

export {
  sort,
  argsort,
  lexsort,
  partition,
  argpartition,
  sort_complex,
  nonzero,
  argwhere,
  flatnonzero,
  where,
  searchsorted,
  extract,
  count_nonzero,
} from './full';

// ============================================================
// Bitwise Functions
// ============================================================

export {
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  bitwise_not,
  invert,
  left_shift,
  right_shift,
  packbits,
  unpackbits,
  bitwise_count,
} from './full';

// Aliases
export {
  bitwise_not as bitwise_invert,
  left_shift as bitwise_left_shift,
  right_shift as bitwise_right_shift,
} from './full';

// ============================================================
// Rounding Functions
// ============================================================

export { around, round_, ceil, fix, floor, rint, trunc } from './full';

// Alias
export { around as round } from './full';

// ============================================================
// Set Operations
// ============================================================

export {
  unique,
  in1d,
  intersect1d,
  isin,
  setdiff1d,
  setxor1d,
  union1d,
  trim_zeros,
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
  digitize,
  histogram,
  histogram2d,
  histogramdd,
  correlate,
  convolve,
  cov,
  corrcoef,
  histogram_bin_edges,
  trapezoid,
} from './full';

// ============================================================
// Gradient Functions
// ============================================================

export { diff, ediff1d, gradient } from './full';

// ============================================================
// Complex Number Functions
// ============================================================

export { real, imag, conj, angle } from './full';

// Alias
export { conj as conjugate } from './full';

// ============================================================
// Advanced Indexing and Data Manipulation
// ============================================================

export {
  broadcast_to,
  broadcast_arrays,
  broadcast_shapes,
  take,
  put,
  iindex,
  bindex,
  copyto,
  choose,
  array_equal,
  array_equiv,
  take_along_axis,
  put_along_axis,
  putmask,
  compress,
  select,
  place,
  fill_diagonal,
  diag_indices,
  diag_indices_from,
  tril_indices,
  tril_indices_from,
  triu_indices,
  triu_indices_from,
  mask_indices,
  indices,
  ix_,
  ravel_multi_index,
  unravel_index,
  fill,
  item,
  tolist,
  tobytes,
  byteswap,
  view,
  tofile,
} from './full';

// ============================================================
// Utility Functions
// ============================================================

export {
  apply_along_axis,
  apply_over_axes,
  may_share_memory,
  shares_memory,
  ndim,
  shape,
  size,
  geterr,
  seterr,
} from './full';

// ============================================================
// Printing/Formatting Functions
// ============================================================

export {
  array2string,
  array_repr,
  array_str,
  base_repr,
  binary_repr,
  format_float_positional,
  format_float_scientific,
  get_printoptions,
  set_printoptions,
  printoptions,
} from './full';

// ============================================================
// Type Checking Functions
// ============================================================

export {
  can_cast,
  common_type,
  result_type,
  min_scalar_type,
  issubdtype,
  typename,
  mintypecode,
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

export {
  parseNpy,
  serializeNpy,
  parseNpyHeader,
  parseNpyData,
  UnsupportedDTypeError,
  InvalidNpyError,
  SUPPORTED_DTYPES,
  DTYPE_TO_DESCR,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
  parseNpz,
  parseNpzSync,
  loadNpz,
  loadNpzSync,
  serializeNpz,
  serializeNpzSync,
  type NpzParseOptions,
  type NpzParseResult,
  type NpzSerializeOptions,
} from './io';

// ============================================================
// Random Namespace (np.random)
// ============================================================

import * as randomOps from './common/ops/random';
import { ArrayStorage } from './common/storage';
import { NDArray as NDArrayClass } from './full';
import { DType } from './common/dtype';

// Helper to wrap ArrayStorage results in NDArray
function wrapResult<T>(result: T): T | NDArrayClass {
  if (result && typeof result === 'object' && '_data' in result && '_shape' in result) {
    return NDArrayClass._fromStorage(result as unknown as ArrayStorage);
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
  default_rng: randomOps.default_rng,
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
    tol?: number
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
    p?: ArrayStorage | number[]
  ) => wrapResult(randomOps.choice(a, size, replace, p)),
  permutation: (x: number | ArrayStorage) => wrapResult(randomOps.permutation(x)),
  shuffle: randomOps.shuffle,
};

// ============================================================
// FFT Namespace (np.fft)
// ============================================================

import * as fftOps from './common/ops/fft';
import { NDArrayCore } from './common/ndarray-core';

// Type alias for array inputs - accepts NDArray, NDArrayCore, or ArrayStorage
type ArrayInput = NDArrayClass | NDArrayCore | ArrayStorage;

// Helper to extract storage from any array input
function getStorage(a: ArrayInput): ArrayStorage {
  if (a instanceof NDArrayClass) return a['_storage'];
  if (a instanceof NDArrayCore) return a['_storage'];
  return a;
}

export const fft = {
  fft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.fft(getStorage(a), n, axis, norm));
  },
  ifft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.ifft(getStorage(a), n, axis, norm));
  },
  fft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.fft2(getStorage(a), s, axes, norm));
  },
  ifft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.ifft2(getStorage(a), s, axes, norm));
  },
  fftn: (a: ArrayInput, s?: number[], axes?: number[], norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.fftn(getStorage(a), s, axes, norm));
  },
  ifftn: (
    a: ArrayInput,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.ifftn(getStorage(a), s, axes, norm));
  },
  rfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.rfft(getStorage(a), n, axis, norm));
  },
  irfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.irfft(getStorage(a), n, axis, norm));
  },
  rfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.rfft2(getStorage(a), s, axes, norm));
  },
  irfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.irfft2(getStorage(a), s, axes, norm));
  },
  rfftn: (
    a: ArrayInput,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.rfftn(getStorage(a), s, axes, norm));
  },
  irfftn: (
    a: ArrayInput,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    return NDArrayClass._fromStorage(fftOps.irfftn(getStorage(a), s, axes, norm));
  },
  hfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.hfft(getStorage(a), n, axis, norm));
  },
  ihfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') => {
    return NDArrayClass._fromStorage(fftOps.ihfft(getStorage(a), n, axis, norm));
  },
  fftfreq: (n: number, d?: number) => {
    return NDArrayClass._fromStorage(fftOps.fftfreq(n, d));
  },
  rfftfreq: (n: number, d?: number) => {
    return NDArrayClass._fromStorage(fftOps.rfftfreq(n, d));
  },
  fftshift: (a: ArrayInput, axes?: number | number[]) => {
    return NDArrayClass._fromStorage(fftOps.fftshift(getStorage(a), axes));
  },
  ifftshift: (a: ArrayInput, axes?: number | number[]) => {
    return NDArrayClass._fromStorage(fftOps.ifftshift(getStorage(a), axes));
  },
};

// ============================================================
// Version
// ============================================================

declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : '0.12.0';
