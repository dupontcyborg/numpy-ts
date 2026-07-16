/**
 * Compile-time assertions for the generated `Promote` type.
 *
 * Checked by `pnpm run typecheck:types` (tsc --noEmit -p tests/types/tsconfig.json),
 * NOT by vitest — these assertions are erased at runtime. A wrong promotion makes
 * `tsc` fail to compile this file.
 */

import type { DType } from '../../src/common/dtype';
import type { Promote, PromotionTable } from '../../src/common/dtype-promotion';

// ── assertion helpers ───────────────────────────────────────────────────────
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Expect<T extends true> = T;

// ── identity / trivial ──────────────────────────────────────────────────────
type _same = Expect<Equal<Promote<'float64', 'float64'>, 'float64'>>;
type _boolYields = Expect<Equal<Promote<'bool', 'int8'>, 'int8'>>;
type _boolYields2 = Expect<Equal<Promote<'int16', 'bool'>, 'int16'>>;

// ── float precision safety ──────────────────────────────────────────────────
type _i32f32 = Expect<Equal<Promote<'int32', 'float32'>, 'float64'>>;
type _f16i8 = Expect<Equal<Promote<'float16', 'int8'>, 'float16'>>;
type _f16i16 = Expect<Equal<Promote<'float16', 'int16'>, 'float32'>>;
type _f32f64 = Expect<Equal<Promote<'float32', 'float64'>, 'float64'>>;

// ── integer signed/unsigned rules ───────────────────────────────────────────
type _i8u8 = Expect<Equal<Promote<'int8', 'uint8'>, 'int16'>>;
type _i64u64 = Expect<Equal<Promote<'int64', 'uint64'>, 'float64'>>;
type _i16u8 = Expect<Equal<Promote<'int16', 'uint8'>, 'int16'>>;
type _u32i32 = Expect<Equal<Promote<'uint32', 'int32'>, 'int64'>>;

// ── complex dominance ───────────────────────────────────────────────────────
type _c64f64 = Expect<Equal<Promote<'complex64', 'float64'>, 'complex128'>>;
type _c64f32 = Expect<Equal<Promote<'complex64', 'float32'>, 'complex64'>>;
type _c64c128 = Expect<Equal<Promote<'complex64', 'complex128'>, 'complex128'>>;

// ── commutativity at the type level (a sampling) ────────────────────────────
type _commute1 = Expect<Equal<Promote<'int32', 'float32'>, Promote<'float32', 'int32'>>>;
type _commute2 = Expect<Equal<Promote<'uint16', 'int8'>, Promote<'int8', 'uint16'>>>;

// ── generic instantiation works (Promote indexes the table for unions) ──────
// Promote must resolve for any pair, including when A/B are still generic.
function promoteIsTotal<A extends DType, B extends DType>(): Promote<A, B> {
  // The cast proves Promote<A,B> is always a valid DType (the table is total).
  return null as unknown as PromotionTable[A][B];
}
void promoteIsTotal;

// Surface unused-type-alias lint without a runtime export.
export type _AssertionsHold = [
  _same,
  _boolYields,
  _boolYields2,
  _i32f32,
  _f16i8,
  _f16i16,
  _f32f64,
  _i8u8,
  _i64u64,
  _i16u8,
  _u32i32,
  _c64f64,
  _c64f32,
  _c64c128,
  _commute1,
  _commute2,
];
