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

// Runtime capabilities
export { hasFloat16 } from '../common/dtype';
// WASM configuration
export { wasmConfig } from '../common/wasm/config';
export { configureWasm, wasmFreeBytes } from '../common/wasm/runtime';
export {
  DTYPE_TO_DESCR,
  InvalidNpyError,
  type NpyHeader,
  type NpyMetadata,
  type NpyVersion,
  SUPPORTED_DTYPES,
  UnsupportedDTypeError,
} from '../io/npy/format';
// IO Functions - NPY format
export { parseNpy, parseNpyData, parseNpyHeader } from '../io/npy/parser';
export { serializeNpy } from '../io/npy/serializer';
// IO Functions - NPZ format
export {
  loadNpz,
  loadNpzSync,
  type NpzParseOptions,
  type NpzParseResult,
  parseNpz,
  parseNpzSync,
} from '../io/npz/parser';
export {
  type NpzArraysInput,
  type NpzSerializeOptions,
  serializeNpz,
  serializeNpzSync,
} from '../io/npz/serializer';
// IO Functions - Text format
export { fromregex, genfromtxt, type ParseTxtOptions, parseTxt } from '../io/txt/parser';
export { type SerializeTxtOptions, serializeTxt } from '../io/txt/serializer';
export type { NDIndex } from './advanced';
// Advanced indexing and data manipulation
export {
  apply_along_axis,
  apply_over_axes,
  array_equal,
  array_equiv,
  bindex,
  broadcast_arrays,
  broadcast_shapes,
  broadcast_to,
  choose,
  compress,
  copyto,
  diag_indices,
  diag_indices_from,
  fill_diagonal,
  geterr,
  iindex,
  indices,
  ix_,
  mask_indices,
  may_share_memory,
  place,
  put,
  put_along_axis,
  putmask,
  ravel_multi_index,
  select,
  seterr,
  shares_memory,
  take,
  take_along_axis,
  tril_indices,
  tril_indices_from,
  triu_indices,
  triu_indices_from,
  unravel_index,
  vindex,
} from './advanced';
// Arithmetic and mathematical functions
export {
  abs,
  absolute,
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
  pow,
  power,
  reciprocal,
  remainder,
  sign,
  sinc,
  sqrt,
  square,
  subtract,
  true_divide,
  unwrap,
} from './arithmetic';
// Bitwise functions
export {
  bitwise_and,
  bitwise_count,
  bitwise_invert,
  bitwise_left_shift,
  bitwise_not,
  bitwise_or,
  bitwise_right_shift,
  bitwise_xor,
  invert,
  left_shift,
  packbits,
  right_shift,
  unpackbits,
} from './bitwise';
// Complex number functions
export { angle, conj, conjugate, imag, real } from './complex';
// Array creation functions
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
} from './creation';
// Formatting and printing functions
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
} from './formatting';
// Gradient functions
export { diff, ediff1d, gradient } from './gradient';
// Linear algebra
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
} from './linalg';
// Logic functions
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
} from './logic';
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
// Reduction functions
export {
  all,
  amax,
  amin,
  any,
  argmax,
  argmin,
  average,
  cumprod,
  cumsum,
  cumulative_prod,
  cumulative_sum,
  max,
  mean,
  median,
  min,
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
  var_,
  variance,
  variance as var,
} from './reduction';
// Rounding functions
export { around, ceil, fix, floor, rint, round, trunc } from './rounding';
// Set operations
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
} from './sets';
// Shape manipulation
export {
  array_split,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  block,
  column_stack,
  concat,
  concatenate,
  dsplit,
  dstack,
  expand_dims,
  flatten,
  flip,
  fliplr,
  flipud,
  hsplit,
  hstack,
  moveaxis,
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
} from './shape';
// Shape manipulation (extra)
export { append, delete_, delete_ as delete, insert, pad } from './shape-extra';
// Sorting and searching
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
} from './sorting';
// Statistics functions
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
} from './statistics';
// Trigonometric functions
export {
  acos,
  acosh,
  arccos,
  arccosh,
  arcsin,
  arcsinh,
  arctan,
  arctan2,
  arctanh,
  asin,
  asinh,
  atan,
  atan2,
  atanh,
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
} from './trig';
// Type checking functions
export {
  can_cast,
  common_type,
  issubdtype,
  min_scalar_type,
  mintypecode,
  result_type,
  typename,
} from './typechecking';
// Types and core classes
export type { DType, TypedArray } from './types';
export { ArrayStorage, Complex, NDArrayCore } from './types';
// Utility functions
export { byteswap, fill, item, ndim, shape, size, tobytes, tofile, tolist, view } from './utility';

// Random Namespace (core.random)

import { DType } from '../common/dtype';
import { NDArrayCore as NDArrayCoreClass } from '../common/ndarray-core';
import * as randomOps from '../common/ops/random';
import { ArrayStorage } from '../common/storage';

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
        p?: ArrayLike | number[],
      ) =>
        wrapResult(
          gen.choice(
            unwrap(a),
            size,
            replace,
            p instanceof NDArrayCoreClass ? p.storage : (p as ArrayStorage | number[] | undefined),
          ),
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
    norm?: 'backward' | 'ortho' | 'forward',
  ) => NDArrayCoreClass.fromStorage(fftOps.fft2(getStorage(a), s, axes, norm)),
  ifft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward',
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
    norm?: 'backward' | 'ortho' | 'forward',
  ) => NDArrayCoreClass.fromStorage(fftOps.rfft2(getStorage(a), s, axes, norm)),
  irfft2: (
    a: ArrayInput,
    s?: [number, number],
    axes?: [number, number],
    norm?: 'backward' | 'ortho' | 'forward',
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
