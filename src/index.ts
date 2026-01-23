/**
 * numpy-ts - Complete NumPy implementation for TypeScript and JavaScript
 *
 * This module is designed for optimal tree-shaking. Each function is
 * re-exported from modular files that only import the dependencies they need.
 *
 * For even better tree-shaking, you can import from submodules:
 *   import { zeros, ones } from 'numpy-ts/creation';
 *   import { sin, cos } from 'numpy-ts/functions/trig';
 *
 * @module numpy-ts
 */

// ============================================================
// Core Types and Classes
// ============================================================

// Complex number class
export { Complex, type ComplexInput } from './core/complex';

// NDArray class with full method support (includes all ops for method chaining)
// Note: Importing NDArray will include all ops due to its method definitions.
// For tree-shakeable array creation, use standalone functions instead.
export { NDArray } from './core/ndarray';

// NDArrayCore for advanced use (minimal class without ops)
export { NDArrayCore } from './core/ndarray-core';

// ============================================================
// Array Creation Functions (from modular creation module)
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
} from './creation';

// Additional creation functions from ndarray (these need the full ndarray module)
export {
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
} from './core/ndarray';

// ============================================================
// Arithmetic and Mathematical Functions
// ============================================================

export {
  sqrt,
  power,
  pow,
  exp,
  exp2,
  expm1,
  log,
  log2,
  log10,
  log1p,
  logaddexp,
  logaddexp2,
  absolute,
  abs,
  negative,
  sign,
  mod,
  divide,
  true_divide,
  floor_divide,
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
} from './functions/arithmetic';

// ============================================================
// Trigonometric and Hyperbolic Functions
// ============================================================

export {
  sin,
  cos,
  tan,
  arcsin,
  asin,
  arccos,
  acos,
  arctan,
  atan,
  arctan2,
  atan2,
  hypot,
  degrees,
  radians,
  deg2rad,
  rad2deg,
  sinh,
  cosh,
  tanh,
  arcsinh,
  asinh,
  arccosh,
  acosh,
  arctanh,
  atanh,
} from './functions/trig';

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
} from './functions/linalg';

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
  concat,
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
} from './functions/shape';

// Additional shape functions from ndarray
export { append, delete_ as delete, insert, pad } from './core/ndarray';

// ============================================================
// Reduction Functions
// ============================================================

export {
  sum,
  mean,
  prod,
  max,
  amax,
  min,
  amin,
  ptp,
  argmin,
  argmax,
  std,
  median,
  percentile,
  quantile,
  average,
  all,
  any,
  cumsum,
  cumulative_sum,
  cumprod,
  cumulative_prod,
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
} from './functions/reduction';

// Variance is exported from reduction but needs aliasing
import { variance } from './functions/reduction';
export { variance };

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
} from './functions/logic';

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
} from './functions/sorting';

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
  bitwise_invert,
  bitwise_left_shift,
  bitwise_right_shift,
} from './functions/bitwise';

// ============================================================
// Rounding Functions
// ============================================================

export { around, round_, round, ceil, fix, floor, rint, trunc } from './functions/rounding';

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
} from './functions/sets';

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
} from './functions/statistics';

// ============================================================
// Gradient Functions
// ============================================================

export { diff, ediff1d, gradient } from './functions/gradient';

// ============================================================
// Complex Number Functions
// ============================================================

export { real, imag, conj, conjugate, angle } from './functions/complex';

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
} from './core/ndarray';

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
} from './core/ndarray';

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
} from './core/ndarray';

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
} from './core/ndarray';

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
} from './core/ndarray';

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

import * as randomOps from './ops/random';
import { ArrayStorage } from './core/storage';
import { NDArray as NDArrayClass } from './core/ndarray';
import { DType } from './core/dtype';

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

import * as fftOps from './ops/fft';

export const fft = {
  fft: (
    a: NDArrayClass | ArrayStorage,
    n?: number,
    axis?: number,
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.fft(storage, n, axis, norm));
  },
  ifft: (
    a: NDArrayClass | ArrayStorage,
    n?: number,
    axis?: number,
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.ifft(storage, n, axis, norm));
  },
  fft2: (
    a: NDArrayClass | ArrayStorage,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.fft2(storage, s, axes, norm));
  },
  ifft2: (
    a: NDArrayClass | ArrayStorage,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.ifft2(storage, s, axes, norm));
  },
  fftn: (
    a: NDArrayClass | ArrayStorage,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.fftn(storage, s, axes, norm));
  },
  ifftn: (
    a: NDArrayClass | ArrayStorage,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.ifftn(storage, s, axes, norm));
  },
  rfft: (
    a: NDArrayClass | ArrayStorage,
    n?: number,
    axis?: number,
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.rfft(storage, n, axis, norm));
  },
  irfft: (
    a: NDArrayClass | ArrayStorage,
    n?: number,
    axis?: number,
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.irfft(storage, n, axis, norm));
  },
  rfft2: (
    a: NDArrayClass | ArrayStorage,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.rfft2(storage, s, axes, norm));
  },
  irfft2: (
    a: NDArrayClass | ArrayStorage,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.irfft2(storage, s, axes, norm));
  },
  rfftn: (
    a: NDArrayClass | ArrayStorage,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.rfftn(storage, s, axes, norm));
  },
  irfftn: (
    a: NDArrayClass | ArrayStorage,
    s?: number[],
    axes?: number[],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.irfftn(storage, s, axes, norm));
  },
  hfft: (
    a: NDArrayClass | ArrayStorage,
    n?: number,
    axis?: number,
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.hfft(storage, n, axis, norm));
  },
  ihfft: (
    a: NDArrayClass | ArrayStorage,
    n?: number,
    axis?: number,
    norm?: 'backward' | 'ortho' | 'forward'
  ) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.ihfft(storage, n, axis, norm));
  },
  fftfreq: (n: number, d?: number) => {
    return NDArrayClass._fromStorage(fftOps.fftfreq(n, d));
  },
  rfftfreq: (n: number, d?: number) => {
    return NDArrayClass._fromStorage(fftOps.rfftfreq(n, d));
  },
  fftshift: (a: NDArrayClass | ArrayStorage, axes?: number | number[]) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.fftshift(storage, axes));
  },
  ifftshift: (a: NDArrayClass | ArrayStorage, axes?: number | number[]) => {
    const storage = a instanceof NDArrayClass ? a['_storage'] : a;
    return NDArrayClass._fromStorage(fftOps.ifftshift(storage, axes));
  },
};

// ============================================================
// Version
// ============================================================

declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : '0.12.0';
