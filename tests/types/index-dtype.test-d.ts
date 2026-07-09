/**
 * Compile-time assertions for the free-function API (src/full/index.ts):
 * dtype-literal inference through creation functions and elementwise ops.
 *
 * Checked by `pnpm run typecheck:types`.
 */

import type { DType } from '../../src/common/dtype';
import * as np from '../../src/full/index';
import type { NDArray } from '../../src/full/ndarray';

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Expect<T extends true> = T;

// ── creation functions: the dtype literal enters the type system ────────────
type _zerosI32 = Expect<Equal<ReturnType<typeof np.zeros<'int32'>>, NDArray<'int32'>>>;
type _onesLit = Expect<Equal<ReturnType<typeof np.ones<'uint8'>>, NDArray<'uint8'>>>;
const fl = np.full([2], 0, 'int16'); // NDArray<'int16'>
type _fullLit = Expect<Equal<typeof fl, NDArray<'int16'>>>;
type _eyeLit = Expect<Equal<ReturnType<typeof np.eye<'complex64'>>, NDArray<'complex64'>>>;
// `DType` referenced so the import is used in a representative default position.
declare const _wide: NDArray<DType>;

// Actual call-site inference (not just explicit type args).
const zi = np.zeros([3], 'int32'); // NDArray<'int32'>
type _ziInfer = Expect<Equal<typeof zi, NDArray<'int32'>>>;
const zf = np.zeros([3]); // NDArray<'float64'>
type _zfInfer = Expect<Equal<typeof zf, NDArray<'float64'>>>;

// ── elementwise ops track dtype through the free-function API ───────────────
declare const i32: NDArray<'int32'>;
declare const f32: NDArray<'float32'>;
declare const i8: NDArray<'int8'>;

type _addPromote = Expect<Equal<ReturnType<typeof np.add<'int32', 'float32'>>, NDArray<'float64'>>>;
type _sqrtMath = Expect<Equal<ReturnType<typeof np.sqrt<'int8'>>, NDArray<'float16'>>>;
type _greaterBool = Expect<Equal<ReturnType<typeof np.greater<'int32', 'int32'>>, NDArray<'bool'>>>;
type _divideFloat = Expect<
  Equal<ReturnType<typeof np.divide<'int32', 'int32'>>, NDArray<'float64'>>
>;

// Call-site inference through ops.
const s = np.add(i32, f32); // NDArray<'float64'>
type _sInfer = Expect<Equal<typeof s, NDArray<'float64'>>>;
const q = np.sqrt(i8); // NDArray<'float16'>
type _qInfer = Expect<Equal<typeof q, NDArray<'float16'>>>;
// Scalar operand keeps the array dtype (documented weak-promotion caveat).
const sc = np.add(i32, 2); // NDArray<'int32'>
type _scInfer = Expect<Equal<typeof sc, NDArray<'int32'>>>;

// ── P1–P3: free-function coverage across the newly-typed buckets ────────────
// preserve → NDArray<D> (shape ops, set diff, rounding …)
const rs = np.reshape(i32, [3]);
type _reshape = Expect<Equal<typeof rs, NDArray<'int32'>>>;
type _setdiff = Expect<Equal<ReturnType<typeof np.setdiff1d<'int16'>>, NDArray<'int16'>>>;
// accumulator → ReductionAccum<D>
const cs = np.cumsum(i8);
type _cumsum = Expect<Equal<typeof cs, NDArray<'int64'>>>;
// index → NDArray<'int64'>
const as_ = np.argsort(f32);
type _argsort = Expect<Equal<typeof as_, NDArray<'int64'>>>;
// unary math → MathResult<D>
type _deg2rad = Expect<Equal<ReturnType<typeof np.deg2rad<'int8'>>, NDArray<'float16'>>>;
// complex component / angle
type _real = Expect<Equal<ReturnType<typeof np.real<'complex64'>>, NDArray<'float32'>>>;
type _imag = Expect<Equal<ReturnType<typeof np.imag<'complex64'>>, NDArray<'float32'>>>;
type _angle = Expect<Equal<ReturnType<typeof np.angle<'int8'>>, NDArray<'float16'>>>;
// predicate → NDArray<'bool'>
const isn = np.isin(i32, f32);
type _isin = Expect<Equal<typeof isn, NDArray<'bool'>>>;
// binary ufuncs / set ops → Promote<A, B>
const mx = np.maximum(i32, f32);
type _maximum = Expect<Equal<typeof mx, NDArray<'float64'>>>;
type _kron = Expect<Equal<ReturnType<typeof np.kron<'int8', 'int16'>>, NDArray<'int16'>>>;
type _union = Expect<Equal<ReturnType<typeof np.union1d<'int32', 'float32'>>, NDArray<'float64'>>>;
// reductions → NDArray<R> | Scalar<R>
const sm = np.sum(i8);
type _sum = Expect<Equal<typeof sm, NDArray<'int64'> | bigint>>;
const mn = np.mean(i32);
type _mean = Expect<Equal<typeof mn, NDArray<'float64'> | number>>;
const am = np.argmax(f32);
type _argmax = Expect<Equal<typeof am, NDArray<'int64'> | number>>;
const al = np.all(i32);
type _all = Expect<Equal<typeof al, NDArray<'bool'> | boolean>>;

// ── P4: contractions → Promote<A,B> union; trace → ReductionAccum union ──────
type _dot = Expect<
  Equal<ReturnType<typeof np.dot<'int32', 'float32'>>, NDArray<'float64'> | number>
>;
type _vecdot = Expect<
  Equal<ReturnType<typeof np.vecdot<'int8', 'int16'>>, NDArray<'int16'> | number>
>;
type _trace = Expect<Equal<ReturnType<typeof np.trace<'int32'>>, NDArray<'int64'> | bigint>>;

// ── P5: tuples + index families ─────────────────────────────────────────────
type _frexp = Expect<
  Equal<ReturnType<typeof np.frexp<'int8'>>, [NDArray<'float16'>, NDArray<'int32'>]>
>;
type _modf = Expect<
  Equal<ReturnType<typeof np.modf<'float32'>>, [NDArray<'float32'>, NDArray<'float32'>]>
>;
type _divmodF = Expect<
  Equal<ReturnType<typeof np.divmod<'int8', 'int16'>>, [NDArray<'int16'>, NDArray<'int16'>]>
>;
const nz = np.nonzero(i32);
type _nonzero = Expect<Equal<typeof nz, NDArray<'int64'>[]>>;
type _unstack = Expect<Equal<ReturnType<typeof np.unstack<'int16'>>, NDArray<'int16'>[]>>;

// ── P6: previously-deferred rules ───────────────────────────────────────────
type _sinc = Expect<Equal<ReturnType<typeof np.sinc<'int8'>>, NDArray<'float64'>>>;
type _sincF16 = Expect<Equal<ReturnType<typeof np.sinc<'float16'>>, NDArray<'float16'>>>;
type _vander = Expect<Equal<ReturnType<typeof np.vander<'float32'>>, NDArray<'float64'>>>;
type _ldexp = Expect<Equal<ReturnType<typeof np.ldexp<'float32'>>, NDArray<'float32'>>>;
type _floatPower = Expect<
  Equal<ReturnType<typeof np.float_power<'int32', 'float32'>>, NDArray<'float64'>>
>;
type _floatPowerC = Expect<
  Equal<ReturnType<typeof np.float_power<'complex64', 'float32'>>, NDArray<'complex128'>>
>;
type _corrcoef = Expect<Equal<ReturnType<typeof np.corrcoef<'float32'>>, NDArray<'float64'>>>;
type _polyadd = Expect<
  Equal<ReturnType<typeof np.polyadd<'float32', 'float32'>>, NDArray<'float32'>>
>;
type _polyder = Expect<Equal<ReturnType<typeof np.polyder<'int32'>>, NDArray<'int64'>>>;
type _cross = Expect<Equal<ReturnType<typeof np.cross<'int8', 'int16'>>, NDArray<'int16'>>>;

export type _IndexAssertionsHold = [
  _zerosI32,
  _onesLit,
  _fullLit,
  _eyeLit,
  _ziInfer,
  _zfInfer,
  _addPromote,
  _sqrtMath,
  _greaterBool,
  _divideFloat,
  _sInfer,
  _qInfer,
  _scInfer,
  _reshape,
  _setdiff,
  _cumsum,
  _argsort,
  _deg2rad,
  _real,
  _imag,
  _angle,
  _isin,
  _maximum,
  _kron,
  _union,
  _sum,
  _mean,
  _argmax,
  _all,
  _dot,
  _vecdot,
  _trace,
  _frexp,
  _modf,
  _divmodF,
  _nonzero,
  _unstack,
  _sinc,
  _sincF16,
  _vander,
  _ldexp,
  _floatPower,
  _floatPowerC,
  _corrcoef,
  _polyadd,
  _polyder,
  _cross,
];
