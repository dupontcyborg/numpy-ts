/**
 * Core Module - Tree-shakeable standalone functions
 *
 * This module re-exports all standalone functions from the various
 * function modules. All functions return NDArrayCore for optimal tree-shaking.
 *
 * For optimal tree-shaking, you can also import from specific submodules:
 *   import { sin, cos } from 'numpy-ts/core/trig';
 *   import { dot, inv } from 'numpy-ts/core/linalg';
 */

// Types and core classes
export type { DType, TypedArray } from './types';
export { NDArrayCore, Complex, ArrayStorage } from './types';

// Array creation functions
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
  ascontiguousarray,
  asfortranarray,
  asarray_chkfinite,
  require,
  diag,
  diagflat,
  tri,
  tril,
  triu,
  vander,
  frombuffer,
  fromfunction,
  fromiter,
  fromstring,
  fromfile,
  meshgrid,
} from './creation';

// Arithmetic and mathematical functions
export {
  add,
  subtract,
  multiply,
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
} from './arithmetic';

// Trigonometric functions
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
} from './trig';

// Linear algebra
export {
  dot,
  matmul,
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
} from './linalg';

// Shape manipulation
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
} from './shape';

// Reduction functions
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
  variance,
  var_,
  variance as var,
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
} from './reduction';

// Logic functions
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
  greater,
  greater_equal,
  less,
  less_equal,
  equal,
  not_equal,
  isclose,
  allclose,
} from './logic';

// Sorting and searching
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
} from './sorting';

// Bitwise functions
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
} from './bitwise';

// Rounding functions
export { around, round, ceil, fix, floor, rint, trunc } from './rounding';

// Set operations
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
} from './sets';

// Statistics functions
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
} from './statistics';

// Gradient functions
export { diff, ediff1d, gradient } from './gradient';

// Complex number functions
export { real, imag, conj, conjugate, angle } from './complex';

// Advanced indexing and data manipulation
export {
  broadcast_to,
  broadcast_arrays,
  broadcast_shapes,
  take,
  put,
  take_along_axis,
  put_along_axis,
  choose,
  compress,
  select,
  place,
  putmask,
  copyto,
  indices,
  ix_,
  ravel_multi_index,
  unravel_index,
  diag_indices,
  diag_indices_from,
  fill_diagonal,
  tril_indices,
  tril_indices_from,
  triu_indices,
  triu_indices_from,
  mask_indices,
  array_equal,
  array_equiv,
  apply_along_axis,
  apply_over_axes,
  may_share_memory,
  shares_memory,
  geterr,
  seterr,
  iindex,
  bindex,
  vindex,
} from './advanced';
export type { NDIndex } from './advanced';

// Formatting and printing functions
export {
  set_printoptions,
  get_printoptions,
  printoptions,
  format_float_positional,
  format_float_scientific,
  base_repr,
  binary_repr,
  array2string,
  array_repr,
  array_str,
} from './formatting';

// Utility functions
export { ndim, shape, size, item, tolist, tobytes, byteswap, view, tofile, fill } from './utility';

// Shape manipulation (extra)
export { append, delete_, delete_ as delete, insert, pad } from './shape-extra';

// Type checking functions
export {
  can_cast,
  common_type,
  result_type,
  min_scalar_type,
  issubdtype,
  typename,
  mintypecode,
} from './typechecking';

// Polynomial functions
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
} from './polynomial';

// IO Functions - NPY format
export { parseNpy, parseNpyHeader, parseNpyData } from '../io/npy/parser';
export { serializeNpy } from '../io/npy/serializer';
export {
  UnsupportedDTypeError,
  InvalidNpyError,
  SUPPORTED_DTYPES,
  DTYPE_TO_DESCR,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
} from '../io/npy/format';

// IO Functions - NPZ format
export {
  parseNpz,
  parseNpzSync,
  loadNpz,
  loadNpzSync,
  type NpzParseOptions,
  type NpzParseResult,
} from '../io/npz/parser';
export {
  serializeNpz,
  serializeNpzSync,
  type NpzArraysInput,
  type NpzSerializeOptions,
} from '../io/npz/serializer';

// IO Functions - Text format
export { parseTxt, genfromtxt, fromregex, type ParseTxtOptions } from '../io/txt/parser';
export { serializeTxt, type SerializeTxtOptions } from '../io/txt/serializer';

// Random Namespace (core.random)

import * as randomOps from '../common/ops/random';
import { ArrayStorage } from '../common/storage';
import { NDArrayCore as NDArrayCoreClass } from '../common/ndarray-core';
import { DType } from '../common/dtype';

// Helper to wrap ArrayStorage results in NDArrayCore
function wrapResult<T>(result: T): T | NDArrayCoreClass {
  if (result && typeof result === 'object' && '_data' in result && '_shape' in result) {
    return NDArrayCoreClass.fromStorage(result as unknown as ArrayStorage);
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
    type ArrayLike = NDArrayCoreClass | ArrayStorage;
    const gen = randomOps.default_rng(seedValue);
    const unwrap = (x: number | ArrayLike): number | ArrayStorage =>
      x instanceof NDArrayCoreClass ? x.storage : (x as number | ArrayStorage);
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
        p?: ArrayLike | number[]
      ) =>
        wrapResult(
          gen.choice(
            unwrap(a),
            size,
            replace,
            p instanceof NDArrayCoreClass ? p.storage : (p as ArrayStorage | number[] | undefined)
          )
        ),
      permutation: (x: number | ArrayLike) => wrapResult(gen.permutation(unwrap(x))),
      shuffle: (x: ArrayLike) => gen.shuffle(x instanceof NDArrayCoreClass ? x.storage : x),
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
  shuffle: (x: NDArrayCoreClass | ArrayStorage) =>
    randomOps.shuffle(x instanceof NDArrayCoreClass ? x.storage : x),
};

// FFT Namespace (core.fft)

import * as fftOps from '../common/ops/fft';

// Type alias for array inputs
type ArrayInput = NDArrayCoreClass | ArrayStorage;

// Helper to extract storage from any array input
function getStorage(a: ArrayInput): ArrayStorage {
  if (a instanceof NDArrayCoreClass) return a.storage;
  return a;
}

export const fft = {
  fft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.fft(getStorage(a), n, axis, norm)),
  ifft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.ifft(getStorage(a), n, axis, norm)),
  fft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => NDArrayCoreClass.fromStorage(fftOps.fft2(getStorage(a), s, axes, norm)),
  ifft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => NDArrayCoreClass.fromStorage(fftOps.ifft2(getStorage(a), s, axes, norm)),
  fftn: (a: ArrayInput, s?: number[], axes?: number[], norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.fftn(getStorage(a), s, axes, norm)),
  ifftn: (a: ArrayInput, s?: number[], axes?: number[], norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.ifftn(getStorage(a), s, axes, norm)),
  rfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.rfft(getStorage(a), n, axis, norm)),
  irfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.irfft(getStorage(a), n, axis, norm)),
  rfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => NDArrayCoreClass.fromStorage(fftOps.rfft2(getStorage(a), s, axes, norm)),
  irfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward'
  ) => NDArrayCoreClass.fromStorage(fftOps.irfft2(getStorage(a), s, axes, norm)),
  rfftn: (a: ArrayInput, s?: number[], axes?: number[], norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.rfftn(getStorage(a), s, axes, norm)),
  irfftn: (a: ArrayInput, s?: number[], axes?: number[], norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.irfftn(getStorage(a), s, axes, norm)),
  hfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.hfft(getStorage(a), n, axis, norm)),
  ihfft: (a: ArrayInput, n?: number, axis?: number, norm?: 'backward' | 'ortho' | 'forward') =>
    NDArrayCoreClass.fromStorage(fftOps.ihfft(getStorage(a), n, axis, norm)),
  fftfreq: (n: number, d?: number) => NDArrayCoreClass.fromStorage(fftOps.fftfreq(n, d)),
  rfftfreq: (n: number, d?: number) => NDArrayCoreClass.fromStorage(fftOps.rfftfreq(n, d)),
  fftshift: (a: ArrayInput, axes?: number | number[]) =>
    NDArrayCoreClass.fromStorage(fftOps.fftshift(getStorage(a), axes)),
  ifftshift: (a: ArrayInput, axes?: number | number[]) =>
    NDArrayCoreClass.fromStorage(fftOps.ifftshift(getStorage(a), axes)),
};
