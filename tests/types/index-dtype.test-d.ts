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
];
