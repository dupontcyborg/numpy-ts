/**
 * Functions Module - Tree-shakeable standalone functions
 *
 * This module re-exports all standalone functions from the various
 * function modules. Import from here or from the main 'numpy-ts' entry.
 *
 * For optimal tree-shaking, you can also import from specific submodules:
 *   import { sin, cos } from 'numpy-ts/functions/trig';
 *   import { dot, inv } from 'numpy-ts/functions/linalg';
 */

// Types
export type { DType, TypedArray } from './types';
export { NDArrayCore, Complex, ArrayStorage } from './types';

// Arithmetic and mathematical functions
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
export { around, round_, round, ceil, fix, floor, rint, trunc } from './rounding';

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
