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
  // Other math functions
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
  // New linear algebra functions
  vdot,
  vecdot,
  matrix_transpose,
  permute_dims,
  matvec,
  vecmat,
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
  concat,
  unstack,
  block,
  split,
  array_split,
  vsplit,
  hsplit,
  tile,
  repeat,
  // New array manipulation functions
  ravel,
  flatten,
  fill,
  item,
  tolist,
  tobytes,
  byteswap,
  view,
  tofile,
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
  iindex,
  bindex,
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
  nanquantile,
  nanpercentile,
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
  bitwise_count,
  bitwise_invert,
  bitwise_left_shift,
  bitwise_right_shift,
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
  trim_zeros,
  unique_all,
  unique_counts,
  unique_inverse,
  unique_values,
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
  histogram_bin_edges,
  trapezoid,
  // Utility functions
  apply_along_axis,
  apply_over_axes,
  may_share_memory,
  shares_memory,
  ndim,
  shape,
  size,
  geterr,
  seterr,
  // Printing/Formatting functions
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
  // Type checking functions
  can_cast,
  common_type,
  result_type,
  min_scalar_type,
  issubdtype,
  typename,
  mintypecode,
  // Polynomial functions
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
// FFT functions (np.fft namespace)
import * as fftOps from './ops/fft';
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

// FFT namespace (np.fft)
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

// Masked array module (np.ma namespace)
import * as maOps from './ma';

export const ma = maOps;

// Also export MaskedArray class directly for convenience
export { MaskedArray } from './ma/MaskedArray';

// Version (replaced at build time from package.json)
// In development/tests, use package.json directly; in production, use the replaced value
declare const __VERSION_PLACEHOLDER__: string;
export const __version__ =
  typeof __VERSION_PLACEHOLDER__ !== 'undefined' ? __VERSION_PLACEHOLDER__ : '0.12.0'; // Fallback for development/tests
