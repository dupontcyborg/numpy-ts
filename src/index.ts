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
export type { BigIntDType, ComplexDType, DType, Scalar } from './common/dtype';
export { hasFloat16, isComplexDType, isFloatDType, isIntegerDType } from './common/dtype';
// Compile-time dtype result-type helpers (types-only; generated from the runtime SSOT).
export type {
  Abs,
  Angle,
  BoolArith,
  ComplexComponent,
  Divide,
  FftReal,
  FftResult,
  MathBinary,
  MathResult,
  MatMul,
  Power,
  Promote,
  ReductionAccum,
  StdVar,
  TrueDivide,
} from './common/dtype-promotion';
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

type SizeArg = number | readonly number[];

// Wraps an ArrayStorage result as an NDArray and leaves scalars alone.
function wrapResult(result: unknown): unknown {
  if (result && typeof result === 'object' && '_data' in result && '_shape' in result) {
    return NDArrayClass.fromStorage(result as unknown as ArrayStorage);
  }
  return result;
}

// A sampling function returns an array of dtype R when size is given and a plain number otherwise.
// Req are the leading required parameters and Opt the optional ones between them and size, each
// typed with undefined so it can be skipped to reach size.
interface Sampler<Req extends unknown[], Opt extends unknown[], R extends DType = 'float64'> {
  (...args: [...Req, ...Opt, size: SizeArg]): NDArrayClass<R>;
  (...args: [...Req, ...Partial<Opt>, size?: undefined]): number;
}

function sampler<Req extends unknown[], Opt extends unknown[], R extends DType = 'float64'>(
  fn: (...args: never[]) => unknown,
): Sampler<Req, Opt, R> {
  return ((...args: unknown[]) =>
    wrapResult((fn as (...a: unknown[]) => unknown)(...args))) as Sampler<Req, Opt, R>;
}

// Shape-from-arguments generators: no arguments gives a number, any dimensions give an array.
interface DimSampler {
  (): number;
  (...shape: [number, ...number[]]): NDArrayClass<'float64'>;
}

interface RandInt {
  <D extends DType = 'int64'>(
    low: number,
    high: number | null | undefined,
    size: SizeArg,
    dtype?: D,
  ): NDArrayClass<D>;
  (low: number, high?: number | null, size?: undefined, dtype?: DType): number;
}

// Unlike the samplers, choice draws from an array, so the element dtype is not known statically.
interface Choice {
  (
    a: number | RngArray,
    size: SizeArg,
    replace?: boolean,
    p?: RngArray | readonly number[],
  ): NDArrayClass;
  (
    a: number | RngArray,
    size?: undefined,
    replace?: boolean,
    p?: RngArray | readonly number[],
  ): number;
}

type RngArray = NDArrayCore | ArrayStorage;

function toRandomStorage<T>(x: NDArrayCore | T): ArrayStorage | T {
  return x instanceof NDArrayCore ? x.storage : (x as T);
}

function choiceFn(
  pick: (
    a: number | ArrayStorage,
    size?: number | number[],
    replace?: boolean,
    p?: ArrayStorage | number[],
  ) => unknown,
): Choice {
  return ((
    a: number | RngArray,
    size?: SizeArg,
    replace?: boolean,
    p?: RngArray | readonly number[],
  ) =>
    wrapResult(
      pick(
        toRandomStorage(a) as number | ArrayStorage,
        size as number | number[] | undefined,
        replace,
        toRandomStorage(p) as ArrayStorage | number[] | undefined,
      ),
    )) as Choice;
}

export const random = {
  // State management
  seed: randomOps.seed,
  get_state: randomOps.get_state,
  set_state: randomOps.set_state,
  get_bit_generator: randomOps.get_bit_generator,
  set_bit_generator: randomOps.set_bit_generator,
  default_rng: (seedValue?: number) => {
    const gen = randomOps.default_rng(seedValue);
    return {
      random: sampler<[], []>(gen.random.bind(gen)),
      integers: sampler<[low: number], [high: number | undefined], 'int64'>(gen.integers.bind(gen)),
      standard_normal: sampler<[], []>(gen.standard_normal.bind(gen)),
      normal: sampler<[], [loc: number | undefined, scale: number | undefined]>(
        gen.normal.bind(gen),
      ),
      uniform: sampler<[], [low: number | undefined, high: number | undefined]>(
        gen.uniform.bind(gen),
      ),
      choice: choiceFn(gen.choice.bind(gen)),
      permutation: (x: number | RngArray): NDArrayClass =>
        NDArrayClass.fromStorage(gen.permutation(toRandomStorage(x) as number | ArrayStorage)),
      shuffle: (x: RngArray) => gen.shuffle(toRandomStorage(x) as ArrayStorage),
      exponential: sampler<[], [scale: number | undefined]>(gen.exponential.bind(gen)),
      poisson: sampler<[], [lam: number | undefined], 'int64'>(gen.poisson.bind(gen)),
      binomial: sampler<[n: number, p: number], [], 'int64'>(gen.binomial.bind(gen)),
    };
  },
  Generator: randomOps.Generator,

  // Basic random functions
  random: sampler<[], []>(randomOps.random),
  rand: ((...shape: number[]) => wrapResult(randomOps.rand(...shape))) as DimSampler,
  randn: ((...shape: number[]) => wrapResult(randomOps.randn(...shape))) as DimSampler,
  randint: ((low: number, high?: number | null, size?: SizeArg, dtype?: DType) =>
    wrapResult(
      randomOps.randint(low, high, size as number | number[] | undefined, dtype),
    )) as RandInt,

  // Aliases
  random_sample: sampler<[], []>(randomOps.random_sample),
  ranf: sampler<[], []>(randomOps.ranf),
  sample: sampler<[], []>(randomOps.sample),
  random_integers: sampler<[low: number], [high: number | undefined], 'int64'>(
    randomOps.random_integers,
  ),

  // Infrastructure
  bytes: randomOps.bytes,

  // Continuous distributions
  uniform: sampler<[], [low: number | undefined, high: number | undefined]>(randomOps.uniform),
  normal: sampler<[], [loc: number | undefined, scale: number | undefined]>(randomOps.normal),
  standard_normal: sampler<[], []>(randomOps.standard_normal),
  exponential: sampler<[], [scale: number | undefined]>(randomOps.exponential),
  standard_exponential: sampler<[], []>(randomOps.standard_exponential),

  // Gamma family
  gamma: sampler<[shape: number], [scale: number | undefined]>(randomOps.gamma),
  standard_gamma: sampler<[shape: number], []>(randomOps.standard_gamma),
  beta: sampler<[a: number, b: number], []>(randomOps.beta),
  chisquare: sampler<[df: number], []>(randomOps.chisquare),
  noncentral_chisquare: sampler<[df: number, nonc: number], []>(randomOps.noncentral_chisquare),
  f: sampler<[dfnum: number, dfden: number], []>(randomOps.f),
  noncentral_f: sampler<[dfnum: number, dfden: number, nonc: number], []>(randomOps.noncentral_f),

  // Other continuous distributions
  standard_cauchy: sampler<[], []>(randomOps.standard_cauchy),
  standard_t: sampler<[df: number], []>(randomOps.standard_t),
  laplace: sampler<[], [loc: number | undefined, scale: number | undefined]>(randomOps.laplace),
  logistic: sampler<[], [loc: number | undefined, scale: number | undefined]>(randomOps.logistic),
  lognormal: sampler<[], [mean: number | undefined, sigma: number | undefined]>(
    randomOps.lognormal,
  ),
  gumbel: sampler<[], [loc: number | undefined, scale: number | undefined]>(randomOps.gumbel),
  pareto: sampler<[a: number], []>(randomOps.pareto),
  power: sampler<[a: number], []>(randomOps.power),
  rayleigh: sampler<[], [scale: number | undefined]>(randomOps.rayleigh),
  triangular: sampler<[left: number, mode: number, right: number], []>(randomOps.triangular),
  wald: sampler<[mean: number, scale: number], []>(randomOps.wald),
  weibull: sampler<[a: number], []>(randomOps.weibull),

  // Discrete distributions
  poisson: sampler<[], [lam: number | undefined], 'int64'>(randomOps.poisson),
  binomial: sampler<[n: number, p: number], [], 'int64'>(randomOps.binomial),
  geometric: sampler<[p: number], [], 'int64'>(randomOps.geometric),
  hypergeometric: sampler<[ngood: number, nbad: number, nsample: number], [], 'int64'>(
    randomOps.hypergeometric,
  ),
  logseries: sampler<[p: number], [], 'int64'>(randomOps.logseries),
  negative_binomial: sampler<[n: number, p: number], [], 'int64'>(randomOps.negative_binomial),
  zipf: sampler<[a: number], [], 'int64'>(randomOps.zipf),

  // Multivariate distributions
  multinomial: (n: number, pvals: number[] | RngArray, size?: SizeArg): NDArrayClass<'int64'> =>
    wrapResult(
      randomOps.multinomial(
        n,
        toRandomStorage(pvals) as number[] | ArrayStorage,
        size as number | number[] | undefined,
      ),
    ) as NDArrayClass<'int64'>,
  multivariate_normal: (
    mean: number[] | RngArray,
    cov: number[][] | RngArray,
    size?: SizeArg,
    check_valid?: 'warn' | 'raise' | 'ignore',
    tol?: number,
  ): NDArrayClass<'float64'> =>
    wrapResult(
      randomOps.multivariate_normal(
        toRandomStorage(mean) as number[] | ArrayStorage,
        toRandomStorage(cov) as number[][] | ArrayStorage,
        size as number | number[] | undefined,
        check_valid,
        tol,
      ),
    ) as NDArrayClass<'float64'>,
  dirichlet: (alpha: number[] | RngArray, size?: SizeArg): NDArrayClass<'float64'> =>
    wrapResult(
      randomOps.dirichlet(
        toRandomStorage(alpha) as number[] | ArrayStorage,
        size as number | number[] | undefined,
      ),
    ) as NDArrayClass<'float64'>,
  vonmises: sampler<[mu: number, kappa: number], []>(randomOps.vonmises),

  // Sequence operations
  choice: choiceFn(randomOps.choice),
  permutation: (x: number | RngArray): NDArrayClass =>
    NDArrayClass.fromStorage(randomOps.permutation(toRandomStorage(x) as number | ArrayStorage)),
  shuffle: (x: RngArray) => randomOps.shuffle(toRandomStorage(x) as ArrayStorage),
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
