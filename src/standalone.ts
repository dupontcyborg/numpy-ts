/**
 * numpy-ts/standalone - Tree-shakeable standalone functions
 *
 * This module provides standalone functions that return NDArrayCore (minimal class).
 * Use this entry point when bundle size is critical and you don't need method chaining.
 *
 * Functions here return NDArrayCore, which has properties (shape, dtype, data) but
 * no operation methods (.add(), .sin(), etc.). Use standalone functions instead:
 *
 * @example
 * ```typescript
 * import { array, add, reshape } from 'numpy-ts/standalone';
 *
 * const a = array([1, 2, 3, 4]);
 * const b = add(a, 10);              // Standalone function
 * const c = reshape(b, [2, 2]);      // Standalone function
 * ```
 *
 * For method chaining, use the main entry point instead:
 * ```typescript
 * import { array } from 'numpy-ts';
 *
 * const a = array([1, 2, 3, 4]);
 * const c = a.add(10).reshape([2, 2]);  // Method chaining
 * ```
 *
 * @module numpy-ts/standalone
 */

// ============================================================
// Core Types and Classes
// ============================================================

export { Complex, type ComplexInput } from './core/complex';
export { NDArrayCore } from './core/ndarray-core';

// ============================================================
// Array Creation Functions (Basic)
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

// ============================================================
// Array Creation Functions (Extended)
// ============================================================

export {
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
} from './standalone/creation';

// ============================================================
// Arithmetic and Mathematical Functions
// ============================================================

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
} from './standalone/arithmetic';

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
} from './standalone/trig';

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
} from './standalone/linalg';

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
} from './standalone/shape';

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
  variance,
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
} from './standalone/reduction';

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
} from './standalone/logic';

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
} from './standalone/sorting';

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
} from './standalone/bitwise';

// ============================================================
// Rounding Functions
// ============================================================

export { around, round_, round, ceil, fix, floor, rint, trunc } from './standalone/rounding';

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
} from './standalone/sets';

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
} from './standalone/statistics';

// ============================================================
// Gradient Functions
// ============================================================

export { diff, ediff1d, gradient } from './standalone/gradient';

// ============================================================
// Complex Number Functions
// ============================================================

export { real, imag, conj, conjugate, angle } from './standalone/complex';

// ============================================================
// Advanced Indexing and Data Manipulation
// ============================================================

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
} from './standalone/advanced';

// ============================================================
// Formatting and Printing Functions
// ============================================================

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
} from './standalone/formatting';

// ============================================================
// Utility Functions
// ============================================================

export {
  ndim,
  shape,
  size,
  item,
  tolist,
  tobytes,
  byteswap,
  view,
  tofile,
  fill,
} from './standalone/utility';

// ============================================================
// Shape Manipulation (Extra)
// ============================================================

export { append, delete_ as delete, insert, pad } from './standalone/shape-extra';

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
} from './standalone/typechecking';

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
} from './standalone/polynomial';

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
