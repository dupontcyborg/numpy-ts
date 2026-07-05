/**
 * Compile-time assertions for dtype tracking on the NDArray class.
 *
 * Checked by `pnpm run typecheck:types`. These verify the wiring of the `D`
 * type parameter: element-access scalar split, astype narrowing, and
 * promotion / bool results on arithmetic & comparison methods.
 */

import type { Complex } from '../../src/common/complex';
import type { NDArray } from '../../src/full/ndarray';

type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Expect<T extends true> = T;

declare const i32: NDArray<'int32'>;
declare const i8: NDArray<'int8'>;
declare const f32: NDArray<'float32'>;
declare const i64: NDArray<'int64'>;
declare const c128: NDArray<'complex128'>;
declare const wide: NDArray; // default D = DType

// ── element access: the bigint / complex / number split ─────────────────────
type _getI32 = Expect<Equal<ReturnType<typeof i32.get>, number>>;
type _getI64 = Expect<Equal<ReturnType<typeof i64.get>, bigint>>;
type _itemI64 = Expect<Equal<ReturnType<typeof i64.item>, bigint>>;
type _igetC128 = Expect<Equal<ReturnType<typeof c128.iget>, Complex>>;
// Default (untyped) array preserves the historical union.
type _getWide = Expect<Equal<ReturnType<typeof wide.get>, number | bigint | Complex>>;

// ── dtype property is the literal ───────────────────────────────────────────
type _dtypeI32 = Expect<Equal<typeof i32.dtype, 'int32'>>;

// ── astype narrows the parameter ────────────────────────────────────────────
type _astype = Expect<Equal<ReturnType<typeof i32.astype<'int8'>>, NDArray<'int8'>>>;
// result is a full NDArray (methods available), not a bare core.
type _astypeChain = Expect<Equal<ReturnType<typeof i32.astype<'float64'>>['dtype'], 'float64'>>;

// ── arithmetic promotion (array ⊗ array) ────────────────────────────────────
type _addPromote = Expect<Equal<ReturnType<typeof i32.add<'float32'>>, NDArray<'float64'>>>;
type _mulPromote = Expect<Equal<ReturnType<typeof i32.multiply<'int32'>>, NDArray<'int32'>>>;

// ── comparisons / logical always yield bool ─────────────────────────────────
type _greaterBool = Expect<Equal<ReturnType<typeof i32.greater>, NDArray<'bool'>>>;
type _equalBool = Expect<Equal<ReturnType<typeof f32.equal>, NDArray<'bool'>>>;

// ── power / mod / floor_divide use the Power rule (bool → int8) ──────────────
type _powArr = Expect<Equal<ReturnType<typeof i32.power<'int8'>>, NDArray<'int32'>>>;
type _modArr = Expect<Equal<ReturnType<typeof i32.mod<'int16'>>, NDArray<'int32'>>>;

// ── true division always floats ─────────────────────────────────────────────
type _divArr = Expect<Equal<ReturnType<typeof i32.divide<'int32'>>, NDArray<'float64'>>>;
type _divNum = Expect<Equal<ReturnType<typeof i32.divide>, NDArray<'float64'>>>; // number overload
type _divF32 = Expect<Equal<ReturnType<typeof f32.divide<'float32'>>, NDArray<'float32'>>>;

// ── unary math promotes ints to float ───────────────────────────────────────
type _sqrtI32 = Expect<Equal<ReturnType<typeof i32.sqrt>, NDArray<'float64'>>>;
type _sqrtI8 = Expect<Equal<ReturnType<typeof i8.sqrt>, NDArray<'float16'>>>;
type _sinF32 = Expect<Equal<ReturnType<typeof f32.sin>, NDArray<'float32'>>>;

// ── shape / sign-preserving unary keeps D ───────────────────────────────────
type _negKeeps = Expect<Equal<ReturnType<typeof i32.negative>, NDArray<'int32'>>>;
type _reshapeKeeps = Expect<Equal<ReturnType<typeof i32.transpose>, NDArray<'int32'>>>;
type _sortKeeps = Expect<Equal<ReturnType<typeof i32.sort>, NDArray<'int32'>>>;

// ── absolute: complex → real component, else preserved ──────────────────────
type _absC128 = Expect<Equal<ReturnType<typeof c128.absolute>, NDArray<'float64'>>>;
type _absI32 = Expect<Equal<ReturnType<typeof i32.absolute>, NDArray<'int32'>>>;

// ── predicates → bool ───────────────────────────────────────────────────────
type _isnanBool = Expect<Equal<ReturnType<typeof f32.isnan>, NDArray<'bool'>>>;

// ── index-returning ops → int64 ─────────────────────────────────────────────
type _argsortI64 = Expect<Equal<ReturnType<typeof f32.argsort>, NDArray<'int64'>>>;

// ── reductions: sum widens ints, full-reduction scalar reflects the dtype ────
type _sumI32 = Expect<Equal<ReturnType<typeof i32.sum>, NDArray<'int64'> | bigint>>;
type _cumsumI8 = Expect<Equal<ReturnType<typeof i8.cumsum>, NDArray<'int64'>>>;
type _meanI32 = Expect<Equal<ReturnType<typeof i32.mean>, NDArray<'float64'> | number>>;
type _stdC128 = Expect<Equal<ReturnType<typeof c128.std>, NDArray<'float64'> | number>>;

export type _NDArrayAssertionsHold = [
  _getI32,
  _getI64,
  _itemI64,
  _igetC128,
  _getWide,
  _dtypeI32,
  _astype,
  _astypeChain,
  _addPromote,
  _mulPromote,
  _greaterBool,
  _equalBool,
  _powArr,
  _modArr,
  _divArr,
  _divNum,
  _divF32,
  _sqrtI32,
  _sqrtI8,
  _sinF32,
  _negKeeps,
  _reshapeKeeps,
  _sortKeeps,
  _absC128,
  _absI32,
  _isnanBool,
  _argsortI64,
  _sumI32,
  _cumsumI8,
  _meanI32,
  _stdC128,
];
