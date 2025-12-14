/**
 * numpy-ts - Complete NumPy implementation for TypeScript and JavaScript
 *
 * @module numpy-ts
 */

// Complex number class
export { Complex, type ComplexInput } from './core/complex';

// Core array functions
export {
  NDArray,
  zeros,
  ones,
  array,
  arange,
  linspace,
  logspace,
  geomspace,
  eye,
  empty,
  full,
  identity,
  asarray,
  copy,
  zeros_like,
  ones_like,
  empty_like,
  full_like,
  // New array creation functions
  asanyarray,
  ascontiguousarray,
  asfortranarray,
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
  // Math functions
  sqrt,
  power,
  pow, // alias for power
  // Exponential functions
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
  abs, // alias for absolute
  negative,
  sign,
  mod,
  divide,
  true_divide, // alias for divide
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
  dot,
  trace,
  diagonal,
  kron,
  transpose,
  inner,
  outer,
  tensordot,
  einsum,
  // Linear algebra module (numpy.linalg)
  linalg,
  // Trigonometric functions
  sin,
  cos,
  tan,
  arcsin,
  asin, // alias for arcsin
  arccos,
  acos, // alias for arccos
  arctan,
  atan, // alias for arctan
  arctan2,
  atan2, // alias for arctan2
  hypot,
  degrees,
  radians,
  deg2rad,
  rad2deg,
  // Hyperbolic functions
  sinh,
  cosh,
  tanh,
  arcsinh,
  asinh, // alias for arcsinh
  arccosh,
  acosh, // alias for arccosh
  arctanh,
  atanh, // alias for arctanh
  // Array manipulation
  swapaxes,
  moveaxis,
  concatenate,
  stack,
  vstack,
  hstack,
  dstack,
  split,
  array_split,
  vsplit,
  hsplit,
  tile,
  repeat,
  // New array manipulation functions
  ravel,
  reshape,
  squeeze,
  expand_dims,
  flip,
  fliplr,
  flipud,
  rot90,
  roll,
  rollaxis,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  dsplit,
  column_stack,
  row_stack,
  resize,
  append,
  delete_ as delete,
  insert,
  pad,
  // Advanced
  broadcast_to,
  broadcast_arrays,
  broadcast_shapes,
  take,
  put,
  copyto,
  choose,
  array_equal,
  array_equiv,
  // Indexing functions
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
  // Reduction functions
  cumsum,
  cumulative_sum, // alias for cumsum
  cumprod,
  cumulative_prod, // alias for cumprod
  max,
  amax, // alias for max
  min,
  amin, // alias for min
  ptp,
  median,
  percentile,
  quantile,
  average,
  // NaN-aware reduction functions
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
  // Bitwise functions
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  bitwise_not,
  invert,
  left_shift,
  right_shift,
  packbits,
  unpackbits,
  // Logic functions
  logical_and,
  logical_or,
  logical_not,
  logical_xor,
  isfinite,
  isinf,
  isnan,
  isnat,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
  // Complex number functions
  real,
  imag,
  conj,
  conjugate,
  angle,
  isneginf,
  isposinf,
  isfortran,
  real_if_close,
  isscalar,
  iterable,
  isdtype,
  promote_types,
  copysign,
  signbit,
  nextafter,
  spacing,
  // Sorting functions
  sort,
  argsort,
  lexsort,
  partition,
  argpartition,
  sort_complex,
  // Searching functions
  nonzero,
  argwhere,
  flatnonzero,
  where,
  searchsorted,
  extract,
  count_nonzero,
  // Rounding functions
  around,
  round_, // alias for around
  ceil,
  fix,
  floor,
  rint,
  round,
  trunc,
  // Set operations
  unique,
  in1d,
  intersect1d,
  isin,
  setdiff1d,
  setxor1d,
  union1d,
  // Gradient functions
  diff,
  ediff1d,
  gradient,
  cross,
  // Statistics functions
  bincount,
  digitize,
  histogram,
  histogram2d,
  histogramdd,
  correlate,
  convolve,
  cov,
  corrcoef,
} from './core/ndarray';

// IO functions (environment-agnostic parsing/serialization)
// These work with bytes (ArrayBuffer/Uint8Array), not files
export {
  // NPY format
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
  // NPZ format
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

// Random functions (np.random namespace)
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
  seed: randomOps.seed,
  random: (size?: number | number[]) => wrapResult(randomOps.random(size)),
  rand: (...shape: number[]) => wrapResult(randomOps.rand(...shape)),
  randn: (...shape: number[]) => wrapResult(randomOps.randn(...shape)),
  randint: (low: number, high?: number | null, size?: number | number[], dtype?: DType) =>
    wrapResult(randomOps.randint(low, high, size, dtype)),
  uniform: (low?: number, high?: number, size?: number | number[]) =>
    wrapResult(randomOps.uniform(low, high, size)),
  normal: (loc?: number, scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.normal(loc, scale, size)),
  standard_normal: (size?: number | number[]) => wrapResult(randomOps.standard_normal(size)),
  exponential: (scale?: number, size?: number | number[]) =>
    wrapResult(randomOps.exponential(scale, size)),
  poisson: (lam?: number, size?: number | number[]) => wrapResult(randomOps.poisson(lam, size)),
  binomial: (n: number, p: number, size?: number | number[]) =>
    wrapResult(randomOps.binomial(n, p, size)),
  choice: (
    a: number | ArrayStorage,
    size?: number | number[],
    replace?: boolean,
    p?: ArrayStorage | number[]
  ) => wrapResult(randomOps.choice(a, size, replace, p)),
  permutation: (x: number | ArrayStorage) => wrapResult(randomOps.permutation(x)),
  shuffle: randomOps.shuffle,
  get_state: randomOps.get_state,
  set_state: randomOps.set_state,
  default_rng: randomOps.default_rng,
  Generator: randomOps.Generator,
};

// Version (replaced at build time from package.json)
// In development/tests, use package.json directly; in production, use the replaced value
declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : '0.10.0'; // Fallback for development/tests
