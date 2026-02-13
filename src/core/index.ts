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
} from './advanced';

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
