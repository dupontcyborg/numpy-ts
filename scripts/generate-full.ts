/**
 * Code Generator for full/ module
 *
 * This script generates:
 *   - full/index.ts  — wrapped core functions returning NDArray
 *   - full/ndarray.ts — NDArray class with ~163 methods
 *
 * Run with: npx ts-node scripts/generate-full.ts
 */

import { execSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { type FunctionDeclaration, Project } from 'ts-morph';
import { MESHGRID_FUNCTION, METHOD_DEFS, type MethodDef } from './ndarray-methods';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const CORE_DIR = path.join(__dirname, '../src/core');
const FULL_DIR = path.join(__dirname, '../src/full');
const INDEX_OUTPUT_FILE = path.join(FULL_DIR, 'index.ts');
const NDARRAY_OUTPUT_FILE = path.join(FULL_DIR, 'ndarray.ts');

// Files to skip (not function modules)
const SKIP_FILES = new Set(['index.ts', 'types.ts']);

// Functions that should NOT be wrapped (they don't return NDArrayCore)
const NO_WRAP_FUNCTIONS = new Set([
  // Functions returning primitives
  'ndim',
  'size',
  'item',
  'tolist',
  'tobytes',
  'tofile',
  'fill',
  'byteswap',
  'view',
  'can_cast',
  'common_type',
  'result_type',
  'min_scalar_type',
  'issubdtype',
  'typename',
  'mintypecode',
  'iscomplexobj',
  'isrealobj',
  'isfortran',
  'isscalar',
  'iterable',
  'isdtype',
  'promote_types',
  'count_nonzero',
  // Functions returning other types
  'set_printoptions',
  'get_printoptions',
  'printoptions',
  'format_float_positional',
  'format_float_scientific',
  'base_repr',
  'binary_repr',
  'array2string',
  'array_repr',
  'array_str',
  'geterr',
  'seterr',
  'linalg', // namespace object
  'einsum_path', // returns tuple
  'may_share_memory',
  'shares_memory', // return boolean
  'array_equal',
  'array_equiv', // return boolean
  'allclose', // returns boolean
  // Functions returning complex objects with NDArrayCore fields
  'unique',
  'unique_all',
  'unique_counts',
  'unique_inverse',
  'histogram',
  'histogram2d',
  'histogramdd',
  'histogram_bin_edges',
  // Functions returning union of single and array (NDArrayCore | NDArrayCore[])
  'gradient',
  'atleast_1d',
  'atleast_2d',
  'atleast_3d',
  'searchsorted',
  // Functions returning NDArrayCore[] with overloads
  'split',
  'dsplit',
  'hsplit',
  'vsplit',
  'array_split',
  // Functions with optional array parameters returning NDArrayCore[] | undefined
  'broadcast_arrays',
  'meshgrid',
  'einsum',
  // Functions with rest parameters or union returns needing special handling
  'ix_',
  'where',
  // meshgrid is re-exported from ./ndarray (needs NDArray-aware implementation)
  'meshgrid',
]);

// Map function names when calling core (for reserved keywords)
const CORE_NAME_MAP: Record<string, string> = {
  delete_: 'delete',
};

// Custom wrapper overrides for functions that need hand-written logic.
// Key = function name, value = { returnType, body }.
// The body can reference: core, up, NDArray, NDArrayCore, Complex,
// ArrayStorage, promoteDTypes, getTypedArrayConstructor, isComplexDType, DType.
// `generic: 'unary'` injects `<D extends DType>` and retypes the first NDArrayCore
// param to `NDArrayCore<D>`; `generic: 'binary'` injects `<A, B>` and retypes the
// first two (so the return type can be expressed in terms of the operand dtypes).
const CUSTOM_WRAPPERS: Record<
  string,
  { returnType: string; body: string; generic?: 'unary' | 'binary' }
> = {
  // cross returns NDArrayCore | number | Complex from core, but full/ should
  // always return NDArray by wrapping scalar results into 0-d arrays (NumPy behavior)
  cross: {
    generic: 'binary',
    returnType: 'NDArray<Promote<A, B>>',
    body: `return ((): NDArray => {
  const r = core.cross(a, b, axisa, axisb, axisc, axis);
  const dtype = promoteDTypes(a.dtype as DType, b.dtype as DType);
  if (r instanceof Complex) {
    const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
    const Ctor = getTypedArrayConstructor(baseDtype)!;
    const data = new Ctor(2);
    data[0] = r.re;
    data[1] = r.im;
    return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
  }
  if (typeof r === 'number' || typeof r === 'bigint') {
    if (isComplexDType(dtype)) {
      const baseDtype = dtype === 'complex64' ? 'float32' : 'float64';
      const Ctor = getTypedArrayConstructor(baseDtype)!;
      const data = new Ctor(2);
      data[0] = Number(r);
      data[1] = 0;
      return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
    }
    const Ctor = getTypedArrayConstructor(dtype)!;
    const data = new Ctor(1);
    data[0] = r as never;
    return NDArray.fromStorage(ArrayStorage.fromData(data, [], dtype));
  }
  return up(r);
  })() as unknown as NDArray<Promote<A, B>>;`,
  },
  // apply_over_axes: callback receives NDArrayCore from core, but user expects NDArray
  apply_over_axes: {
    returnType: 'NDArray',
    body: `const wrappedFunc = (arr: NDArrayCore, axis: number): NDArrayCore => {
    return func(up(arr), axis);
  };
  return up(core.apply_over_axes(wrappedFunc, a, axes));`,
  },
  // apply_along_axis: callback receives NDArrayCore from core, but user expects NDArray.
  // Forwards extra args (NumPy: apply_along_axis(func1d, axis, arr, *args)).
  apply_along_axis: {
    returnType: 'NDArray',
    body: `const wrappedFunc1d = (arr: NDArrayCore, ...passed: unknown[]): NDArrayCore | number => {
    return func1d(up(arr), ...passed);
  };
  return up(core.apply_along_axis(wrappedFunc1d, axis, arr, ...args));`,
  },
  // average can return either a single value or a tuple [avg, sum_of_weights]
  // when returned=true. Map the inner NDArrayCore values to NDArray.
  average: {
    generic: 'unary',
    // avg → TrueDivide<D>; with returned=true, [avg, sum_of_weights]. Typed for
    // the common case (no weights / equal-or-lower-kind weights), like average().
    returnType:
      'NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>> | [NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>, NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>]',
    body: `const r = core.average(a, axis, weights, keepdims, returned);
  if (Array.isArray(r)) {
    const [avg, sw] = r;
    const upAvg = avg instanceof NDArrayCore ? up(avg) : avg;
    const upSw = sw instanceof NDArrayCore ? up(sw) : sw;
    return [upAvg, upSw] as unknown as [
      NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>,
      NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>,
    ];
  }
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as
    | NDArray<TrueDivide<D>>
    | Scalar<TrueDivide<D>>;`,
  },
};

// Functions returning arrays (should be wrapped)
const _RETURNS_ARRAY_FUNCTIONS = new Set([
  // Most core functions return arrays
]);

interface FunctionInfo {
  name: string;
  jsDoc: string;
  params: string;
  paramNames: string[];
  invokeParams: string; // Store the parameter names as they would be for a function invocation
  returnType: string;
  typeParams: string; // e.g. "<D extends DType = 'float64'>" — empty if none
  isExported: boolean;
}

function extractFunctionInfo(func: FunctionDeclaration): FunctionInfo | null {
  const name = func.getName();
  if (!name) return null;

  const jsDocNodes = func.getJsDocs();
  const jsDoc = jsDocNodes.map((doc) => doc.getFullText()).join('\n');

  // Extract parameter names directly using ts-morph (reliable, handles complex types)
  const paramNames: string[] = [];

  const params = func
    .getParameters()
    .map((p) => {
      const pName = p.getName();
      paramNames.push(pName); // Store the parameter name

      const typeNode = p.getTypeNode();
      const type = typeNode ? typeNode.getText() : 'any';
      const isOptional = p.isOptional() || p.hasInitializer();
      const isRest = p.isRestParameter();
      const initializer = p.getInitializer()?.getText();

      if (isRest) {
        return `...${pName}: ${type}`;
      } else if (initializer) {
        return `${pName}: ${type} = ${initializer}`;
      } else if (isOptional) {
        return `${pName}?: ${type}`;
      } else {
        return `${pName}: ${type}`;
      }
    })
    .join(', ');

  const invokeParams = func
    .getParameters()
    .map((p) => {
      const pName = p.getName();

      const isRest = p.isRestParameter();

      if (isRest) {
        return `...${pName}`;
      } else {
        return `${pName}`;
      }
    })
    .join(', ');

  const returnTypeNode = func.getReturnTypeNode();
  const returnType = returnTypeNode ? returnTypeNode.getText() : 'any';

  // Copy generic type parameters verbatim (e.g. creation functions parameterized
  // by dtype: `<D extends DType = 'float64'>`). Empty string when there are none.
  const tps = func.getTypeParameters().map((tp) => tp.getText());
  const typeParams = tps.length ? `<${tps.join(', ')}>` : '';

  return {
    name,
    jsDoc,
    params,
    paramNames,
    invokeParams,
    returnType,
    typeParams,
    isExported: func.isExported(),
  };
}

function shouldWrapFunction(info: FunctionInfo): boolean {
  if (NO_WRAP_FUNCTIONS.has(info.name)) {
    return false;
  }

  // Check if return type includes NDArrayCore
  if (info.returnType.includes('NDArrayCore')) {
    return true;
  }

  // Check if it returns an array of NDArrayCore
  if (info.returnType.includes('NDArrayCore[]')) {
    return true;
  }

  return false;
}

function transformReturnType(returnType: string): string {
  return returnType
    .replace(/\[NDArrayCore,\s*NDArrayCore\]/g, '[NDArray, NDArray]')
    .replace(/NDArrayCore\[\]/g, 'NDArray[]')
    .replace(/NDArrayCore/g, 'NDArray');
}

// Creation functions get a dtype type-parameter injected so the dtype literal
// flows into the result type (e.g. `zeros([3], 'int32')` → `NDArray<'int32'>`).
// The core functions stay untyped (NDArrayCore); genericity lives only here.
// `dflt` is the type-param default used when `dtype` is omitted: 'float64' for
// functions whose runtime default is DEFAULT_DTYPE, else DType (inferred).
const CREATION_GENERICS: Record<string, { dflt: string }> = {
  zeros: { dflt: "'float64'" },
  ones: { dflt: "'float64'" },
  empty: { dflt: "'float64'" },
  eye: { dflt: "'float64'" },
  identity: { dflt: "'float64'" },
  linspace: { dflt: "'float64'" },
  logspace: { dflt: "'float64'" },
  geomspace: { dflt: "'float64'" },
  tri: { dflt: "'float64'" },
  fromfunction: { dflt: "'float64'" },
  array: { dflt: 'DType' },
  asarray: { dflt: 'DType' },
  arange: { dflt: 'DType' },
  full: { dflt: 'DType' },
  frombuffer: { dflt: 'DType' },
  fromiter: { dflt: 'DType' },
  fromstring: { dflt: 'DType' },
  zeros_like: { dflt: 'DType' },
  ones_like: { dflt: 'DType' },
  empty_like: { dflt: 'DType' },
  full_like: { dflt: 'DType' },
};

// Free-function elementwise ops (np.add, np.sqrt, …) → dtype-tracked wrappers,
// mirroring the NDArray method taxonomy. Keyed by core function name.
const FUNCTION_RESULT_RULES: Record<
  string,
  keyof typeof UNARY_RESULT_TYPE | keyof typeof BINARY_RESULT | 'bool'
> = {
  // binary
  add: 'promote',
  subtract: 'promote',
  multiply: 'promote',
  bitwise_and: 'promote',
  bitwise_or: 'promote',
  bitwise_xor: 'promote',
  left_shift: 'promote',
  right_shift: 'promote',
  matmul: 'promote',
  outer: 'promote',
  power: 'power',
  mod: 'power',
  floor_divide: 'power',
  remainder: 'power',
  divide: 'divide',
  true_divide: 'divide',
  arctan2: 'mathbinary',
  hypot: 'mathbinary',
  copysign: 'mathbinary',
  nextafter: 'mathbinary',
  heaviside: 'mathbinary',
  logaddexp: 'mathbinary',
  logaddexp2: 'mathbinary',
  greater: 'bool',
  greater_equal: 'bool',
  less: 'bool',
  less_equal: 'bool',
  equal: 'bool',
  not_equal: 'bool',
  logical_and: 'bool',
  logical_or: 'bool',
  logical_xor: 'bool',
  // unary
  sqrt: 'math',
  exp: 'math',
  exp2: 'math',
  expm1: 'math',
  log: 'math',
  log2: 'math',
  log10: 'math',
  log1p: 'math',
  sin: 'math',
  cos: 'math',
  tan: 'math',
  arcsin: 'math',
  arccos: 'math',
  arctan: 'math',
  sinh: 'math',
  cosh: 'math',
  tanh: 'math',
  arcsinh: 'math',
  arccosh: 'math',
  arctanh: 'math',
  cbrt: 'math',
  degrees: 'math',
  radians: 'math',
  ceil: 'math',
  fix: 'math',
  floor: 'math',
  rint: 'math',
  trunc: 'math',
  fabs: 'math',
  spacing: 'math',
  absolute: 'abs',
  square: 'boolarith',
  negative: 'preserve',
  sign: 'preserve',
  positive: 'preserve',
  reciprocal: 'preserve',
  bitwise_not: 'preserve',
  invert: 'preserve',
  conj: 'preserve',
  conjugate: 'preserve',
  logical_not: 'bool',
  isfinite: 'bool',
  isinf: 'bool',
  isnan: 'bool',
  isnat: 'bool',
  signbit: 'bool',

  // ---- P1: free-function unary buckets ----
  // Shape / preserve → NDArray<D>
  reshape: 'preserve',
  ravel: 'preserve',
  flatten: 'preserve',
  transpose: 'preserve',
  squeeze: 'preserve',
  expand_dims: 'preserve',
  moveaxis: 'preserve',
  swapaxes: 'preserve',
  flip: 'preserve',
  fliplr: 'preserve',
  flipud: 'preserve',
  roll: 'preserve',
  rollaxis: 'preserve',
  rot90: 'preserve',
  tile: 'preserve',
  repeat: 'preserve',
  take: 'preserve',
  take_along_axis: 'preserve',
  resize: 'preserve',
  diagonal: 'preserve',
  broadcast_to: 'preserve',
  diff: 'preserve',
  ediff1d: 'preserve',
  pad: 'preserve',
  copy: 'preserve',
  trim_zeros: 'preserve',
  permute_dims: 'preserve',
  matrix_transpose: 'preserve',
  diag: 'preserve',
  diagflat: 'preserve',
  around: 'preserve',
  round: 'preserve',
  nan_to_num: 'preserve',
  unique_values: 'preserve',
  // setdiff1d returns elements of ar1 only → preserves ar1's dtype (NOT promote,
  // unlike union1d/intersect1d/setxor1d which compare across both operands).
  setdiff1d: 'preserve',
  // Accumulator → NDArray<ReductionAccum<D>>
  cumsum: 'reduction',
  cumprod: 'reduction',
  nancumsum: 'reduction',
  nancumprod: 'reduction',
  // Index → NDArray<'int64'>
  argsort: 'index',
  argpartition: 'index',
  argwhere: 'index',
  flatnonzero: 'index',
  lexsort: 'index',
  ravel_multi_index: 'index',
  // Unary math → NDArray<MathResult<D>>
  deg2rad: 'math',
  rad2deg: 'math',
  // NOTE: sinc, i0, unwrap, interp deferred — they return float64 for integer
  // inputs (not MathResult's float16/float32), so they need a dedicated helper.
  // Complex component / angle
  real: 'component',
  imag: 'component',
  angle: 'angle',
  // Predicate → NDArray<'bool'>
  iscomplex: 'bool',
  isreal: 'bool',
  isposinf: 'bool',
  isneginf: 'bool',
  isin: 'bool',
  in1d: 'bool',
  isclose: 'bool',

  // ---- P1/P2: set ops + binary ufuncs (promote of operands) ----
  union1d: 'promote',
  intersect1d: 'promote',
  setxor1d: 'promote',
  maximum: 'promote',
  minimum: 'promote',
  fmax: 'promote',
  fmin: 'promote',
  fmod: 'promote',
  gcd: 'promote',
  lcm: 'promote',
  convolve: 'promote',
  correlate: 'promote',
  kron: 'promote',
  matvec: 'promote',
  vecmat: 'promote',

  // ---- P6: previously-deferred rules, now validated ----
  // sinc/i0/unwrap: true-divide rule (float/complex preserved, ints/bool → float64).
  sinc: 'truediv',
  i0: 'truediv',
  unwrap: 'truediv',
  // vander: promote against the default int (float32 → float64, uint64 → float64).
  vander: 'accumint',
  // ldexp(x1, x2) = x1 * 2**x2 → preserves x1's (float) dtype.
  ldexp: 'preserve',
  // float_power: promote, then minimum precision float64 / complex128.
  float_power: 'floatpower',
  // corrcoef/cov: at-least-float64 (complex → complex128).
  corrcoef: 'atleastf64',
  cov: 'atleastf64',
};

// P3: free-function reductions/contractions → `NDArray<R> | Scalar<R>`. Keyed by
// the same result-dtype rules as the reduction methods (see REDUCTION_RESULT).
const FUNCTION_REDUCTION_RULES: Record<string, keyof typeof REDUCTION_RESULT> = {
  sum: 'accum',
  nansum: 'accum',
  prod: 'accum',
  nanprod: 'accum',
  mean: 'truedivide',
  nanmean: 'truedivide',
  median: 'truedivide',
  nanmedian: 'truedivide',
  percentile: 'truedivide',
  nanpercentile: 'truedivide',
  quantile: 'truedivide',
  nanquantile: 'truedivide',
  std: 'stdvar',
  variance: 'stdvar',
  nanvar: 'stdvar',
  nanstd: 'stdvar',
  min: 'preserve',
  max: 'preserve',
  nanmin: 'preserve',
  nanmax: 'preserve',
  ptp: 'preserve',
  all: 'bool',
  any: 'bool',
  argmin: 'index',
  argmax: 'index',
  nanargmin: 'index',
  nanargmax: 'index',
  // trace sums the diagonal → accumulator widening (int → int64), unary shape.
  trace: 'accum',
};

// Result type of the scalar-operand overload for each binary op kind (in terms of A).
const BINARY_NUMBER_RESULT: Record<string, string> = {
  promote: 'NDArray<A>',
  power: 'NDArray<A>',
  divide: 'NDArray<TrueDivide<A>>',
  mathbinary: 'NDArray<MathResult<A>>',
  floatpower: "NDArray<Promote<A, 'float64'>>",
};

// Split a parameter string at top-level commas (respecting <> () [] {} nesting).
function splitTopLevel(params: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = '';
  for (const ch of params) {
    if ('<([{'.includes(ch)) depth++;
    else if ('>)]}'.includes(ch)) depth--;
    if (ch === ',' && depth === 0) {
      out.push(cur.trim());
      cur = '';
    } else {
      cur += ch;
    }
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

// The declared name of a single param segment (before `?`, `:` or `=`).
function paramName(seg: string): string {
  return seg.trim().split(/[?:=]/)[0]!.trim();
}

// Retype the first `NDArrayCore` in a single param segment to `NDArrayCore<T>`.
function retypeArrayParam(seg: string, typeArg: string): string {
  return seg.replace(/NDArrayCore\b/, `NDArrayCore<${typeArg}>`);
}

// Convert a param segment to overload-signature form. Default values are illegal
// in overload declarations, so a defaulted param becomes optional (`x = 5` → `x?`).
function toOverloadParam(seg: string): string {
  const eq = seg.indexOf('=');
  if (eq === -1) return seg;
  const left = seg.slice(0, eq).trim();
  const colon = left.indexOf(':');
  const name = left.slice(0, colon).trim();
  const type = left.slice(colon + 1).trim();
  return name.endsWith('?') ? `${name}: ${type}` : `${name}?: ${type}`;
}

/**
 * Emit a dtype-tracked wrapper for an elementwise op. The leading param(s) are
 * the array operand(s); any trailing params pass through verbatim. Returns null
 * (→ fall back to the untyped wrapper) when the first param isn't an NDArrayCore,
 * or a binary kind lacks a second array operand.
 *
 * Unary kind  → `fn<D>(arr: NDArrayCore<D>, …rest): <UNARY_RESULT_TYPE[kind]>`.
 * Binary kind → array⊗array overload + scalar overload (only when the 2nd operand
 * accepts `number`), over a single untyped implementation.
 */
function generateTypedOpWrapper(info: FunctionInfo, kind: string): string | null {
  const core = `core.${CORE_NAME_MAP[info.name] || info.name}`;
  const segs = splitTopLevel(info.params);
  if (segs.length === 0 || !/:\s*NDArrayCore\b/.test(segs[0]!)) return null;

  const isBinaryKind = kind in BINARY_RESULT;
  const p1IsArray = segs.length >= 2 && /:\s*NDArrayCore\b/.test(segs[1]!);
  const treatBinary = isBinaryKind || (kind === 'bool' && p1IsArray);

  if (treatBinary) {
    if (!p1IsArray) return null;
    const alias = kind === 'bool' ? null : (BINARY_RESULT[kind]?.alias ?? null);
    const arrRet = alias ? `NDArray<${alias}<A, B>>` : "NDArray<'bool'>";
    const numRet = kind === 'bool' ? "NDArray<'bool'>" : BINARY_NUMBER_RESULT[kind];
    const acceptsNumber = /\bnumber\b/.test(segs[1]!);
    const p1name = paramName(segs[1]!);
    const rest = segs.slice(2).map(toOverloadParam);
    // The array⊗array overload never accepts `number` — otherwise it would
    // shadow the scalar overload (B couldn't be inferred from a number).
    const arrParams = [retypeArrayParam(segs[0]!, 'A'), `${p1name}: NDArrayCore<B>`, ...rest].join(
      ', ',
    );
    const overloads = [
      `export function ${info.name}<A extends DType, B extends DType>(${arrParams}): ${arrRet};`,
    ];
    if (acceptsNumber) {
      const numParams = [retypeArrayParam(segs[0]!, 'A'), `${p1name}: number`, ...rest].join(', ');
      overloads.push(`export function ${info.name}<A extends DType>(${numParams}): ${numRet};`);
    }
    const implCast = kind === 'bool' ? " as NDArray<'bool'>" : '';
    return `${info.jsDoc}
${overloads.join('\n')}
export function ${info.name}(${info.params}): NDArray {
  return up(${core}(${info.invokeParams}))${implCast};
}`;
  }

  // Unary: retype the first array param to <D>, keep the rest verbatim.
  const rt = UNARY_RESULT_TYPE[kind];
  if (!rt) return null;
  const retyped = [retypeArrayParam(segs[0]!, 'D'), ...segs.slice(1)].join(', ');
  return `${info.jsDoc}
export function ${info.name}<D extends DType>(${retyped}): ${rt} {
  return up(${core}(${info.invokeParams})) as ${rt};
}`;
}

// Reduction/contraction shape: retype the leading array param to <D>, keep the
// rest, and return `NDArray<R> | Scalar<R>` (the runtime yields either an array
// or a scalar depending on `axis`). Mirrors the `reduction` method pattern.
const REDUCTION_RESULT: Record<string, string> = {
  accum: 'NDArray<ReductionAccum<D>> | Scalar<ReductionAccum<D>>',
  truedivide: 'NDArray<TrueDivide<D>> | Scalar<TrueDivide<D>>',
  stdvar: 'NDArray<StdVar<D>> | Scalar<StdVar<D>>',
  preserve: 'NDArray<D> | Scalar<D>',
  index: "NDArray<'int64'> | number",
  bool: "NDArray<'bool'> | boolean",
};

function generateReductionWrapper(info: FunctionInfo, kind: string): string | null {
  const rt = REDUCTION_RESULT[kind];
  if (!rt) return null;
  const segs = splitTopLevel(info.params);
  if (segs.length === 0 || !/:\s*NDArrayCore\b/.test(segs[0]!)) return null;
  const core = `core.${CORE_NAME_MAP[info.name] || info.name}`;
  const retyped = [retypeArrayParam(segs[0]!, 'D'), ...segs.slice(1)].join(', ');
  return `${info.jsDoc}
export function ${info.name}<D extends DType>(${retyped}): ${rt} {
  const r = ${core}(${info.invokeParams});
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as ${rt};
}`;
}

// P4: binary contraction (dot/inner/tensordot/vecdot) → `Promote<A,B>` over both
// array operands, `NDArray<R> | Scalar<R>` since the result collapses to a scalar
// for low-rank inputs. Both operands are arrays (no scalar overload needed).
const FUNCTION_CONTRACTION_RULES = new Set(['dot', 'inner', 'tensordot', 'vecdot']);

function generateContractionWrapper(info: FunctionInfo): string | null {
  const segs = splitTopLevel(info.params);
  if (
    segs.length < 2 ||
    !/:\s*NDArrayCore\b/.test(segs[0]!) ||
    !/:\s*NDArrayCore\b/.test(segs[1]!)
  ) {
    return null;
  }
  const core = `core.${CORE_NAME_MAP[info.name] || info.name}`;
  const rt = 'NDArray<Promote<A, B>> | Scalar<Promote<A, B>>';
  const retyped = [
    retypeArrayParam(segs[0]!, 'A'),
    retypeArrayParam(segs[1]!, 'B'),
    ...segs.slice(2),
  ].join(', ');
  return `${info.jsDoc}
export function ${info.name}<A extends DType, B extends DType>(${retyped}): ${rt} {
  const r = ${core}(${info.invokeParams});
  return (r instanceof NDArrayCore ? up(r) : r) as unknown as ${rt};
}`;
}

// P5: tuple-returning ufuncs. Unary → per-slot rule over <D>; binary (divmod) →
// array⊗array overload + scalar overload over an untyped `[NDArray, NDArray]` impl.
const FUNCTION_TUPLE_RULES: Record<
  string,
  | { arity: 'unary'; slots: [string, string] }
  | { arity: 'binary'; arrSlots: [string, string]; numSlots: [string, string] }
> = {
  frexp: { arity: 'unary', slots: ['NDArray<MathResult<D>>', "NDArray<'int32'>"] },
  modf: { arity: 'unary', slots: ['NDArray<MathResult<D>>', 'NDArray<MathResult<D>>'] },
  divmod: {
    arity: 'binary',
    arrSlots: ['NDArray<Power<A, B>>', 'NDArray<Power<A, B>>'],
    numSlots: ['NDArray<A>', 'NDArray<A>'],
  },
};

function generateTupleWrapper(info: FunctionInfo): string | null {
  const cfg = FUNCTION_TUPLE_RULES[info.name];
  if (!cfg) return null;
  const segs = splitTopLevel(info.params);
  if (segs.length === 0 || !/:\s*NDArrayCore\b/.test(segs[0]!)) return null;
  const core = `core.${CORE_NAME_MAP[info.name] || info.name}`;

  if (cfg.arity === 'unary') {
    const rt = `[${cfg.slots[0]}, ${cfg.slots[1]}]`;
    const retyped = [retypeArrayParam(segs[0]!, 'D'), ...segs.slice(1)].join(', ');
    return `${info.jsDoc}
export function ${info.name}<D extends DType>(${retyped}): ${rt} {
  const r = ${core}(${info.invokeParams});
  return [up(r[0]), up(r[1])] as unknown as ${rt};
}`;
  }

  // Binary tuple (divmod): typed overloads over a single untyped implementation.
  if (segs.length < 2 || !/:\s*NDArrayCore\b/.test(segs[1]!)) return null;
  const p1name = paramName(segs[1]!);
  const acceptsNumber = /\bnumber\b/.test(segs[1]!);
  const arrRet = `[${cfg.arrSlots[0]}, ${cfg.arrSlots[1]}]`;
  const numRet = `[${cfg.numSlots[0]}, ${cfg.numSlots[1]}]`;
  const overloads = [
    `export function ${info.name}<A extends DType, B extends DType>(${retypeArrayParam(segs[0]!, 'A')}, ${p1name}: NDArrayCore<B>): ${arrRet};`,
  ];
  if (acceptsNumber) {
    overloads.push(
      `export function ${info.name}<A extends DType>(${retypeArrayParam(segs[0]!, 'A')}, ${p1name}: number): ${numRet};`,
    );
  }
  return `${info.jsDoc}
${overloads.join('\n')}
export function ${info.name}(${info.params}): [NDArray, NDArray] {
  const r = ${core}(${info.invokeParams});
  return [up(r[0]), up(r[1])] as [NDArray, NDArray];
}`;
}

// P5: index families that always return int64 index arrays, regardless of input
// dtype → `NDArray<'int64'>[]` (fixed). `unstack` preserves element dtype instead.
const INDEX_ARRAY_FAMILIES = new Set([
  'nonzero',
  'unravel_index',
  'diag_indices',
  'diag_indices_from',
  'tril_indices',
  'tril_indices_from',
  'triu_indices',
  'triu_indices_from',
  'mask_indices',
]);

function generateIndexArrayWrapper(info: FunctionInfo): string {
  const core = `core.${CORE_NAME_MAP[info.name] || info.name}`;
  return `${info.jsDoc}
export function ${info.name}(${info.params}): NDArray<'int64'>[] {
  return ${core}(${info.invokeParams}).map(up) as NDArray<'int64'>[];
}`;
}

// P6: polynomial ops whose operands are `NDArrayCore | number[]`. Type params
// default to 'float64' so bare `number[]` callers keep working (numpy-ts infers
// float64 from JS number arrays). Unary key off p; binary promote/divide over A,B.
// (poly/roots/polyfit stay untyped: data-dependent, or keyed off a non-first arg.)
const POLY_RULES: Record<
  string,
  { arity: 'unary'; ret: string } | { arity: 'binary'; ret: string; tuple?: boolean }
> = {
  polyint: { arity: 'unary', ret: "NDArray<Promote<D, 'float64'>>" },
  polyder: { arity: 'unary', ret: "NDArray<Promote<D, 'int64'>>" },
  polyadd: { arity: 'binary', ret: 'NDArray<Promote<A, B>>' },
  polysub: { arity: 'binary', ret: 'NDArray<Promote<A, B>>' },
  polymul: { arity: 'binary', ret: 'NDArray<Promote<A, B>>' },
  polydiv: { arity: 'binary', ret: '[NDArray<Divide<A, B>>, NDArray<Divide<A, B>>]', tuple: true },
};

function generatePolyWrapper(info: FunctionInfo): string | null {
  const cfg = POLY_RULES[info.name];
  if (!cfg) return null;
  const segs = splitTopLevel(info.params);
  if (segs.length === 0 || !/:\s*NDArrayCore\b/.test(segs[0]!)) return null;
  const core = `core.${CORE_NAME_MAP[info.name] || info.name}`;

  if (cfg.arity === 'unary') {
    const retyped = [retypeArrayParam(segs[0]!, 'D'), ...segs.slice(1)].join(', ');
    return `${info.jsDoc}
export function ${info.name}<D extends DType = 'float64'>(${retyped}): ${cfg.ret} {
  return up(${core}(${info.invokeParams})) as unknown as ${cfg.ret};
}`;
  }

  if (segs.length < 2 || !/:\s*NDArrayCore\b/.test(segs[1]!)) return null;
  const retyped = [
    retypeArrayParam(segs[0]!, 'A'),
    retypeArrayParam(segs[1]!, 'B'),
    ...segs.slice(2),
  ].join(', ');
  const body = cfg.tuple
    ? `const r = ${core}(${info.invokeParams});\n  return [up(r[0]), up(r[1])] as unknown as ${cfg.ret};`
    : `return up(${core}(${info.invokeParams})) as unknown as ${cfg.ret};`;
  return `${info.jsDoc}
export function ${info.name}<A extends DType = 'float64', B extends DType = 'float64'>(${retyped}): ${cfg.ret} {
  ${body}
}`;
}

function generateWrapperFunction(info: FunctionInfo, _moduleName: string): string {
  // Check for custom wrapper override
  if (CUSTOM_WRAPPERS[info.name]) {
    const custom = CUSTOM_WRAPPERS[info.name];
    let tp = '';
    let params = info.params;
    if (custom.generic === 'unary') {
      tp = '<D extends DType>';
      params = info.params.replace(/NDArrayCore\b(?!<)/, 'NDArrayCore<D>');
    } else if (custom.generic === 'binary') {
      // Lookahead skips an already-typed `NDArrayCore<…>` so the 2nd match is the
      // 2nd operand (not a re-match of the first).
      tp = '<A extends DType, B extends DType>';
      params = info.params
        .replace(/NDArrayCore\b(?!<)/, 'NDArrayCore<A>')
        .replace(/NDArrayCore\b(?!<)/, 'NDArrayCore<B>');
    }
    return `${info.jsDoc}
export function ${info.name}${tp}(${params}): ${custom.returnType} {
  ${custom.body}
}`;
  }

  // Creation functions: inject `<D extends DType = …>`, retype the `dtype`
  // parameter to `dtype?: D`, and return `NDArray<D>`.
  const gen = CREATION_GENERICS[info.name];
  if (gen && info.returnType === 'NDArrayCore') {
    // Retype the `dtype` parameter's DType → D while preserving optionality and
    // any runtime default (keeping default value preserves function arity).
    const params = info.params.replace(
      /\bdtype(\?)?\s*:\s*DType(?:\s*=\s*([^,)]+))?/,
      (_m, opt: string | undefined, def: string | undefined) =>
        def ? `dtype: D = ${def.trim()} as D` : opt ? 'dtype?: D' : 'dtype: D',
    );
    return `${info.jsDoc}
export function ${info.name}<D extends DType = ${gen.dflt}>(${params}): NDArray<D> {
  return up(core.${CORE_NAME_MAP[info.name] || info.name}(${info.invokeParams})) as NDArray<D>;
}`;
  }

  // Binary contraction (dot/inner/tensordot/vecdot) → Promote<A,B> union.
  if (FUNCTION_CONTRACTION_RULES.has(info.name)) {
    const typed = generateContractionWrapper(info);
    if (typed) return typed;
    console.warn(`⚠ contraction rule didn't match ${info.name} — left untyped`);
  }

  // Tuple-returning ufuncs (divmod/frexp/modf) → per-slot dtype-tracked tuple.
  if (FUNCTION_TUPLE_RULES[info.name]) {
    const typed = generateTupleWrapper(info);
    if (typed) return typed;
    console.warn(`⚠ tuple rule didn't match ${info.name} — left untyped`);
  }

  // Polynomial ops (operands may be number[]) → promote/divide over A,B or p.
  if (POLY_RULES[info.name]) {
    const typed = generatePolyWrapper(info);
    if (typed) return typed;
    console.warn(`⚠ poly rule didn't match ${info.name} — left untyped`);
  }

  // Index families → NDArray<'int64'>[]; unstack preserves element dtype.
  if (INDEX_ARRAY_FAMILIES.has(info.name)) {
    return generateIndexArrayWrapper(info);
  }
  if (info.name === 'unstack' && /:\s*NDArrayCore\b/.test(info.params)) {
    const params = info.params.replace(/NDArrayCore\b/, 'NDArrayCore<D>');
    return `${info.jsDoc}
export function unstack<D extends DType>(${params}): NDArray<D>[] {
  return core.unstack(${info.invokeParams}).map(up) as NDArray<D>[];
}`;
  }

  // Reduction/contraction → dtype-tracked union wrapper (NDArray<R> | Scalar<R>).
  const redRule = FUNCTION_REDUCTION_RULES[info.name];
  if (redRule) {
    const typed = generateReductionWrapper(info, redRule);
    if (typed) return typed;
    console.warn(`⚠ reduction rule '${redRule}' didn't match ${info.name} — left untyped`);
  }

  // Elementwise op → dtype-tracked wrapper (falls back to untyped if the core
  // signature shape isn't the expected unary/binary elementwise form).
  const opRule = FUNCTION_RESULT_RULES[info.name];
  if (opRule) {
    const typed = generateTypedOpWrapper(info, opRule);
    if (typed) return typed;
    console.warn(`⚠ op rule '${opRule}' didn't match ${info.name} — left untyped`);
  }

  const transformedReturnType = transformReturnType(info.returnType);

  const rt = info.returnType;

  // Map function name for core call (e.g., delete_ -> delete for reserved keywords)
  const coreFuncName = CORE_NAME_MAP[info.name] || info.name;

  let body: string;

  // Detect return type patterns
  const isTupleOfArrays = /^\[NDArrayCore,\s*NDArrayCore\]$/.test(rt.replace(/\s/g, ''));
  const isArrayOfArrays = rt.includes('NDArrayCore[]') && !rt.includes('|');
  const isOptionalArray =
    /NDArrayCore\[\]\s*\|\s*undefined/.test(rt) || /undefined\s*\|\s*NDArrayCore\[\]/.test(rt);
  const isUnionWithPrimitive =
    rt.includes('NDArrayCore') &&
    (rt.includes('number') ||
      rt.includes('bigint') ||
      rt.includes('Complex') ||
      rt.includes('boolean'));
  const isOptionalSingle =
    /NDArrayCore\s*\|\s*undefined/.test(rt) || /undefined\s*\|\s*NDArrayCore/.test(rt);
  const isPureSingle = rt === 'NDArrayCore';

  if (isTupleOfArrays) {
    // [NDArrayCore, NDArrayCore] → [up(r[0]), up(r[1])] as [NDArray, NDArray]
    body = `const r = core.${coreFuncName}(${info.invokeParams}); return [up(r[0]), up(r[1])] as [NDArray, NDArray];`;
  } else if (isOptionalArray) {
    // NDArrayCore[] | undefined → result?.map(up)
    body = `const r = core.${coreFuncName}(${info.invokeParams}); return r?.map(up);`;
  } else if (isArrayOfArrays) {
    // NDArrayCore[] → result.map(up)
    body = `return core.${coreFuncName}(${info.invokeParams}).map(up);`;
  } else if (isUnionWithPrimitive) {
    // number | NDArrayCore | ... → conditional wrap
    body = `const r = core.${coreFuncName}(${info.invokeParams}); return r instanceof NDArrayCore ? up(r) : r;`;
  } else if (isOptionalSingle) {
    // NDArrayCore | undefined → result ? up(result) : undefined
    body = `const r = core.${coreFuncName}(${info.invokeParams}); return r ? up(r) : undefined;`;
  } else if (isPureSingle) {
    // Pure NDArrayCore → up(result). When the function is generic (dtype-tracked
    // creation fns), `up` widens to NDArray, so assert back to the tracked type.
    const cast = info.typeParams ? ` as ${transformedReturnType}` : '';
    body = `return up(core.${coreFuncName}(${info.invokeParams}))${cast};`;
  } else {
    // Default: try simple wrap (will fail for complex cases)
    body = `return up(core.${coreFuncName}(${info.invokeParams}));`;
  }

  return `${info.jsDoc}
export function ${info.name}${info.typeParams}(${info.params}): ${transformedReturnType} {
  ${body}
}`;
}

function _generateReexport(info: FunctionInfo): string {
  return `export { ${info.name} } from '../core/${getModuleName(info.name)}';`;
}

function getModuleName(_funcName: string): string {
  // This would need to be tracked properly, for now we'll use 'index'
  return 'index';
}

// ============================================================
// Index file generation (existing logic)
// ============================================================

function generateIndexFile(
  allFunctions: Map<string, FunctionInfo>,
  wrappedFunctions: Set<string>,
  reexportedFunctions: Set<string>,
  coreImportPath: string = '../core',
  outputFile: string = INDEX_OUTPUT_FILE,
): void {
  const output: string[] = [];

  output.push(`// AUTO-GENERATED - DO NOT EDIT
// Run \`pnpm run generate\` to regenerate this file from core/ modules

/**
 * Full Module - NDArray with method chaining
 *
 * This module wraps core/ functions to return NDArray (with methods)
 * instead of NDArrayCore (minimal). Use this for method chaining.
 *
 * @example
 * \`\`\`typescript
 * import { array } from 'numpy-ts/full';
 *
 * const result = array([1, 2, 3]).add(10).reshape([3, 1]);
 * \`\`\`
 */

import * as core from '${coreImportPath}';
import { NDArray } from './ndarray';
import { NDArrayCore } from '../common/ndarray-core';
import { ArrayStorage } from '../common/storage';
import { Complex } from '../common/complex';
import type { ArrayLike, DType, TypedArray } from '../core/types';
import type {
  Abs,
  Angle,
  BoolArith,
  ComplexComponent,
  Divide,
  FloatPower,
  MathBinary,
  MathResult,
  Power,
  Promote,
  ReductionAccum,
  StdVar,
  TrueDivide,
} from '../common/dtype-promotion';
import type { NestedNDArrays } from '../core/shape';
import type { PadValueArg, PadWidthArg } from '../core/shape-extra';
import type { ReductionOpts } from '../core/reduction';
import {
  type Scalar,
  DEFAULT_DTYPE,
  getTypedArrayConstructor,
  promoteDTypes,
  isComplexDType,
} from '../common/dtype';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
// If already an NDArray, return as-is to preserve identity
// Also preserves base reference for views
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

// Re-export types
export type { DType, TypedArray } from '../core/types';
export type { NDIndex } from '../core/advanced';
export { NDArray, meshgrid } from './ndarray';
export { NDArrayCore } from '../common/ndarray-core';
export { Complex } from '../common/complex';
`);

  // Generate wrapped functions
  output.push('\n// ============================================================');
  output.push('// Wrapped Functions (return NDArray)');
  output.push('// ============================================================\n');

  for (const [name, info] of allFunctions) {
    if (wrappedFunctions.has(name)) {
      output.push(generateWrapperFunction(info, ''));
      output.push('');
    }
  }

  // Re-export non-wrapped functions
  output.push('\n// ============================================================');
  output.push('// Re-exported Functions (no wrapping needed)');
  output.push('// ============================================================\n');

  // Names already exported in the header (from ./ndarray or elsewhere)
  const HEADER_EXPORTS = new Set(['NDArray', 'NDArrayCore', 'Complex', 'meshgrid']);

  const reexports: string[] = [];
  for (const name of reexportedFunctions) {
    if (!HEADER_EXPORTS.has(name)) {
      reexports.push(name);
    }
  }

  if (reexports.length > 0) {
    output.push(`export {`);
    output.push(`  ${reexports.join(',\n  ')},`);
    output.push(`} from '${coreImportPath}';`);
  }

  // Static exports not derived from core/ modules
  output.push('');
  output.push('// WASM configuration');
  output.push(`export { wasmConfig } from '../common/wasm/config';`);
  output.push(`export { configureWasm, wasmFreeBytes } from '../common/wasm/runtime';`);
  output.push('');
  output.push('// Runtime capabilities');
  output.push(`export { hasFloat16 } from '../common/dtype';`);

  // Write output
  const outputContent = output.join('\n');
  fs.writeFileSync(outputFile, outputContent);
  execSync(`npx biome check --write "${path.relative(path.join(__dirname, '..'), outputFile)}"`, {
    cwd: path.join(__dirname, '..'),
    stdio: 'pipe',
  });

  console.log(`\nGenerated ${outputFile}`);
  console.log(`  - ${wrappedFunctions.size} wrapped functions`);
  console.log(`  - ${reexportedFunctions.size} re-exported functions`);
}

// ============================================================
// NDArray file generation (new)
// ============================================================

function formatJSDoc(doc: string | undefined): string {
  if (!doc) return '';
  const lines = doc.split('\n');
  if (lines.length === 1) {
    return `  /**\n   * ${lines[0]}\n   */\n`;
  }
  return `  /**\n${lines.map((l) => `   * ${l}`).join('\n')}\n   */\n`;
}

function extractParamNames(params: string): string {
  if (!params) return '';
  // Parse param names from a TypeScript parameter string
  // e.g. "axis?: number, keepdims: boolean = false" → "axis, keepdims"
  return params
    .split(',')
    .map((p) => {
      const name = p.trim().split(/[?:=]/)[0]!.trim();
      // Handle rest params
      return name.startsWith('...') ? name : name;
    })
    .join(', ');
}

// Unary/passthrough `result` kind → the `NDArray<…>` return type (in terms of `D`).
// A kind absent from this map falls back to bare `NDArray` (dtype not tracked).
const UNARY_RESULT_TYPE: Record<string, string> = {
  math: 'NDArray<MathResult<D>>',
  abs: 'NDArray<Abs<D>>',
  boolarith: 'NDArray<BoolArith<D>>',
  reduction: 'NDArray<ReductionAccum<D>>',
  preserve: 'NDArray<D>',
  bool: "NDArray<'bool'>",
  index: "NDArray<'int64'>",
  component: 'NDArray<ComplexComponent<D>>',
  angle: 'NDArray<Angle<D>>',
  // sinc/i0/unwrap: floats & complex preserved, ints/bool → float64 (= true-divide rule).
  truediv: 'NDArray<TrueDivide<D>>',
  // vander/polyder: multiply against the default int → promote(D, int64).
  accumint: "NDArray<Promote<D, 'int64'>>",
  // corrcoef/cov/polyint: at-least-float64 (complex → complex128) = promote(D, float64).
  atleastf64: "NDArray<Promote<D, 'float64'>>",
  complexfft: 'NDArray<FftResult<D>>',
  realfft: 'NDArray<FftReal<D>>',
};

// Binary `result` kind → { alias, number } where `alias<D,B>` types the
// array⊗array overload and `number` types the scalar-operand overload.
const BINARY_RESULT: Record<string, { alias: string; number: string }> = {
  promote: { alias: 'Promote', number: 'NDArray<D>' },
  power: { alias: 'Power', number: 'NDArray<D>' },
  divide: { alias: 'Divide', number: 'NDArray<TrueDivide<D>>' },
  mathbinary: { alias: 'MathBinary', number: 'NDArray<MathResult<D>>' },
  floatpower: { alias: 'FloatPower', number: "NDArray<Promote<D, 'float64'>>" },
};

function generateMethodCode(def: MethodDef): string {
  const doc = formatJSDoc(def.doc);
  const coreName = def.coreName || def.name;

  switch (def.pattern) {
    case 'manual':
      return `${doc}  ${def.manualCode!}`;

    case 'unary': {
      const rt = def.result ? UNARY_RESULT_TYPE[def.result] : undefined;
      if (rt) {
        // Cast the untyped `up(...)` result to the dtype-tracked return type.
        return `${doc}  ${def.name}(): ${rt} {\n    return up(core.${coreName}(this)) as ${rt};\n  }`;
      }
      return `${doc}  ${def.name}(): NDArray {\n    return up(core.${coreName}(this));\n  }`;
    }

    case 'binary': {
      const params = def.params || 'other: NDArray | number';
      const argName = def.coreArgs || extractParamNames(params);
      // The operand name (first param) — used to build typed array⊗array overloads.
      const operand = extractParamNames(params).split(',')[0]!.trim();
      const impl = `${def.name}(${params}): NDArray {\n    return up(core.${coreName}(this, ${argName}));\n  }`;

      // `result: 'bool'` — comparisons / logical ops always yield a bool array.
      if (def.result === 'bool') {
        return `${doc}  ${def.name}(${params}): NDArray<'bool'> {\n    return up(core.${coreName}(this, ${argName})) as NDArray<'bool'>;\n  }`;
      }

      // promote / power / divide / mathbinary — dtype-tracked binary result.
      // Emits a generic array⊗array overload, plus a scalar overload when the
      // signature accepts `number`, over a single untyped implementation.
      const rule = def.result ? BINARY_RESULT[def.result] : undefined;
      if (rule) {
        const overloads = [
          `${def.name}<B extends DType>(${operand}: NDArray<B>): NDArray<${rule.alias}<D, B>>;`,
        ];
        if (params.includes('number')) {
          overloads.push(`  ${def.name}(${operand}: number): ${rule.number};`);
        }
        return `${doc}  ${overloads.join('\n')}\n  ${impl}`;
      }

      return `${doc}  ${impl}`;
    }

    case 'reduction': {
      const params = def.params || 'axis?: number, keepdims: boolean = false';
      const returnType = def.returnType || 'NDArray | number | Complex';
      const argNames = def.coreArgs || extractParamNames(params);
      // Cast when the return type is dtype-tracked (contains a type argument);
      // the plain union case (no `<`) assigns directly as before.
      const tail = returnType.includes('<')
        ? `return (r instanceof NDArrayCore ? up(r) : r) as unknown as ${returnType};`
        : `return r instanceof NDArrayCore ? up(r) : r;`;
      return `${doc}  ${def.name}(${params}): ${returnType} {\n    const r = core.${coreName}(this, ${argNames});\n    ${tail}\n  }`;
    }

    case 'passthrough': {
      const params = def.params || '';
      const argNames = def.coreArgs || extractParamNames(params);
      const coreCall = argNames ? `core.${coreName}(this, ${argNames})` : `core.${coreName}(this)`;
      const rt = def.result ? UNARY_RESULT_TYPE[def.result] : undefined;
      if (rt) {
        return `${doc}  ${def.name}(${params}): ${rt} {\n    return up(${coreCall}) as ${rt};\n  }`;
      }
      return `${doc}  ${def.name}(${params}): NDArray {\n    return up(${coreCall});\n  }`;
    }

    case 'array_return': {
      if (def.result === 'index') {
        return `${doc}  ${def.name}(): NDArray<'int64'>[] {\n    return core.${coreName}(this).map(up) as NDArray<'int64'>[];\n  }`;
      }
      return `${doc}  ${def.name}(): NDArray[] {\n    return core.${coreName}(this).map(up);\n  }`;
    }

    case 'tuple_return': {
      const params = def.params || '';
      const argNames = def.coreArgs || extractParamNames(params);
      const coreCall = argNames ? `core.${coreName}(this, ${argNames})` : `core.${coreName}(this)`;
      return `${doc}  ${def.name}(${params}): [NDArray, NDArray] {\n    const r = ${coreCall};\n    return [up(r[0]), up(r[1])] as [NDArray, NDArray];\n  }`;
    }

    default:
      throw new Error(`Unknown pattern: ${def.pattern}`);
  }
}

function generateNDArrayFile(
  coreImportPath: string = '../core',
  outputFile: string = NDARRAY_OUTPUT_FILE,
): void {
  const output: string[] = [];

  // File header
  output.push(`// AUTO-GENERATED - DO NOT EDIT
// Run \`pnpm run generate\` to regenerate this file from scripts/ndarray-methods.ts

import {
  type DType,
  type Scalar,
  type TypedArray,
  getTypedArrayConstructor,
  getDTypeSize,
  isBigIntDType,
  isComplexDType,
} from '../common/dtype';
import type {
  Abs,
  BoolArith,
  Divide,
  MathBinary,
  MathResult,
  Power,
  Promote,
  ReductionAccum,
  StdVar,
  TrueDivide,
} from '../common/dtype-promotion';
import { Complex } from '../common/complex';
import { ArrayStorage } from '../common/storage';
import { NDArrayCore } from '../common/ndarray-core';
import * as core from '${coreImportPath}';

// Helper to upgrade NDArrayCore to NDArray (zero-copy via shared storage)
const up = (x: NDArrayCore): NDArray => {
  if (x instanceof NDArray) return x;
  const base = x.base ? up(x.base) : undefined;
  return NDArray.fromStorage(x.storage, base);
};

export class NDArray<D extends DType = DType> extends NDArrayCore<D> {`);

  // Group methods by pattern for organized output
  const manualMethods = METHOD_DEFS.filter((d) => d.pattern === 'manual');
  const unaryMethods = METHOD_DEFS.filter((d) => d.pattern === 'unary');
  const binaryMethods = METHOD_DEFS.filter((d) => d.pattern === 'binary');
  const reductionMethods = METHOD_DEFS.filter((d) => d.pattern === 'reduction');
  const passthroughMethods = METHOD_DEFS.filter((d) => d.pattern === 'passthrough');
  const arrayReturnMethods = METHOD_DEFS.filter((d) => d.pattern === 'array_return');
  const tupleReturnMethods = METHOD_DEFS.filter((d) => d.pattern === 'tuple_return');

  // --- Manual methods ---
  output.push('\n  // ========================================');
  output.push('  // Manual methods');
  output.push('  // ========================================\n');

  for (const def of manualMethods) {
    output.push(generateMethodCode(def));
    output.push('');
  }

  // --- Unary operations ---
  output.push('  // ========================================');
  output.push('  // Unary operations');
  output.push('  // ========================================\n');

  for (const def of unaryMethods) {
    output.push(generateMethodCode(def));
    output.push('');
  }

  // --- Binary operations ---
  output.push('  // ========================================');
  output.push('  // Binary operations');
  output.push('  // ========================================\n');

  for (const def of binaryMethods) {
    output.push(generateMethodCode(def));
    output.push('');
  }

  // --- Reductions ---
  output.push('  // ========================================');
  output.push('  // Reduction operations');
  output.push('  // ========================================\n');

  for (const def of reductionMethods) {
    output.push(generateMethodCode(def));
    output.push('');
  }

  // --- Passthrough ---
  output.push('  // ========================================');
  output.push('  // Passthrough operations');
  output.push('  // ========================================\n');

  for (const def of passthroughMethods) {
    output.push(generateMethodCode(def));
    output.push('');
  }

  // --- Array return ---
  if (arrayReturnMethods.length > 0) {
    output.push('  // ========================================');
    output.push('  // Array return operations');
    output.push('  // ========================================\n');

    for (const def of arrayReturnMethods) {
      output.push(generateMethodCode(def));
      output.push('');
    }
  }

  // --- Tuple return ---
  if (tupleReturnMethods.length > 0) {
    output.push('  // ========================================');
    output.push('  // Tuple return operations');
    output.push('  // ========================================\n');

    for (const def of tupleReturnMethods) {
      output.push(generateMethodCode(def));
      output.push('');
    }
  }

  // Close class
  output.push('}');

  // Meshgrid function (after class)
  output.push('');
  output.push('/**');
  output.push(' * Return coordinate matrices from coordinate vectors');
  output.push(' * @param arrays - 1D coordinate arrays');
  output.push(" * @param indexing - 'xy' (Cartesian, default) or 'ij' (matrix indexing)");
  output.push(' * @returns Array of coordinate grids');
  output.push(' */');
  output.push(MESHGRID_FUNCTION);
  output.push('');

  // Write output
  const outputContent = output.join('\n');
  fs.writeFileSync(outputFile, outputContent);
  execSync(`npx biome check --write "${path.relative(path.join(__dirname, '..'), outputFile)}"`, {
    cwd: path.join(__dirname, '..'),
    stdio: 'pipe',
  });

  const totalMethods = METHOD_DEFS.length;
  const autoMethods = totalMethods - manualMethods.length;
  console.log(`\nGenerated ${outputFile}`);
  console.log(`  - ${manualMethods.length} manual methods`);
  console.log(`  - ${autoMethods} auto-generated methods`);
  console.log(`  - ${totalMethods} total methods`);
}

// ============================================================
// Main
// ============================================================

async function main() {
  console.log('Generating full/ module files...\n');

  const project = new Project({
    tsConfigFilePath: path.join(__dirname, '../tsconfig.json'),
  });

  // Collect all function info from core modules
  const allFunctions: Map<string, FunctionInfo> = new Map();
  const wrappedFunctions: Set<string> = new Set();
  const reexportedFunctions: Set<string> = new Set();

  const coreFiles = fs.readdirSync(CORE_DIR).filter((f) => f.endsWith('.ts') && !SKIP_FILES.has(f));

  console.log(`Processing ${coreFiles.length} core modules...`);

  // Track const exports separately
  const constExports: Set<string> = new Set();

  for (const file of coreFiles) {
    const filePath = path.join(CORE_DIR, file);
    const sourceFile = project.addSourceFileAtPath(filePath);

    const functions = sourceFile
      .getFunctions()
      .filter((f) => f.isExported())
      .map(extractFunctionInfo)
      .filter((f): f is FunctionInfo => f !== null);

    for (const func of functions) {
      if (shouldWrapFunction(func)) {
        wrappedFunctions.add(func.name);
      } else {
        reexportedFunctions.add(func.name);
      }
      allFunctions.set(func.name, func);
    }

    // Also collect exported variable declarations (const exports, aliases)
    const varStatements = sourceFile.getVariableStatements().filter((v) => v.isExported());

    for (const varStatement of varStatements) {
      for (const decl of varStatement.getDeclarations()) {
        const name = decl.getName();
        constExports.add(name);
        reexportedFunctions.add(name);
      }
    }

    console.log(
      `  ${file}: ${functions.length} functions (${functions.filter((f) => shouldWrapFunction(f)).length} wrapped), ${varStatements.length} const exports`,
    );
  }

  // Generate full/ files
  generateIndexFile(
    allFunctions,
    wrappedFunctions,
    reexportedFunctions,
    '../core',
    INDEX_OUTPUT_FILE,
  );
  generateNDArrayFile('../core', NDARRAY_OUTPUT_FILE);
}

main().catch(console.error);
