/**
 * AUTO-GENERATED — DO NOT EDIT BY HAND.
 *
 * Compile-time mirror of the runtime result-dtype functions (src/common/dtype.ts),
 * which are the single source of truth. Regenerate with:
 *
 *   npx tsx scripts/generate-dtype-promotion.ts
 *
 * This module is types-only: it contributes zero bytes to the runtime bundle.
 */

import type { DType } from './dtype';

/**
 * Full dtype promotion matrix. `PromotionTable[A][B]` is the dtype NumPy
 * promotes `A` and `B` to.
 */
export interface PromotionTable {
  float64: {
    float64: 'float64';
    float32: 'float64';
    float16: 'float64';
    complex128: 'complex128';
    complex64: 'complex128';
    int64: 'float64';
    int32: 'float64';
    int16: 'float64';
    int8: 'float64';
    uint64: 'float64';
    uint32: 'float64';
    uint16: 'float64';
    uint8: 'float64';
    bool: 'float64';
  };
  float32: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float32';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'float64';
    int32: 'float64';
    int16: 'float32';
    int8: 'float32';
    uint64: 'float64';
    uint32: 'float64';
    uint16: 'float32';
    uint8: 'float32';
    bool: 'float32';
  };
  float16: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float16';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'float64';
    int32: 'float64';
    int16: 'float32';
    int8: 'float16';
    uint64: 'float64';
    uint32: 'float64';
    uint16: 'float32';
    uint8: 'float16';
    bool: 'float16';
  };
  complex128: {
    float64: 'complex128';
    float32: 'complex128';
    float16: 'complex128';
    complex128: 'complex128';
    complex64: 'complex128';
    int64: 'complex128';
    int32: 'complex128';
    int16: 'complex128';
    int8: 'complex128';
    uint64: 'complex128';
    uint32: 'complex128';
    uint16: 'complex128';
    uint8: 'complex128';
    bool: 'complex128';
  };
  complex64: {
    float64: 'complex128';
    float32: 'complex64';
    float16: 'complex64';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'complex128';
    int32: 'complex128';
    int16: 'complex64';
    int8: 'complex64';
    uint64: 'complex128';
    uint32: 'complex128';
    uint16: 'complex64';
    uint8: 'complex64';
    bool: 'complex64';
  };
  int64: {
    float64: 'float64';
    float32: 'float64';
    float16: 'float64';
    complex128: 'complex128';
    complex64: 'complex128';
    int64: 'int64';
    int32: 'int64';
    int16: 'int64';
    int8: 'int64';
    uint64: 'float64';
    uint32: 'int64';
    uint16: 'int64';
    uint8: 'int64';
    bool: 'int64';
  };
  int32: {
    float64: 'float64';
    float32: 'float64';
    float16: 'float64';
    complex128: 'complex128';
    complex64: 'complex128';
    int64: 'int64';
    int32: 'int32';
    int16: 'int32';
    int8: 'int32';
    uint64: 'float64';
    uint32: 'int64';
    uint16: 'int32';
    uint8: 'int32';
    bool: 'int32';
  };
  int16: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float32';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'int64';
    int32: 'int32';
    int16: 'int16';
    int8: 'int16';
    uint64: 'float64';
    uint32: 'int64';
    uint16: 'int32';
    uint8: 'int16';
    bool: 'int16';
  };
  int8: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float16';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'int64';
    int32: 'int32';
    int16: 'int16';
    int8: 'int8';
    uint64: 'float64';
    uint32: 'int64';
    uint16: 'int32';
    uint8: 'int16';
    bool: 'int8';
  };
  uint64: {
    float64: 'float64';
    float32: 'float64';
    float16: 'float64';
    complex128: 'complex128';
    complex64: 'complex128';
    int64: 'float64';
    int32: 'float64';
    int16: 'float64';
    int8: 'float64';
    uint64: 'uint64';
    uint32: 'uint64';
    uint16: 'uint64';
    uint8: 'uint64';
    bool: 'uint64';
  };
  uint32: {
    float64: 'float64';
    float32: 'float64';
    float16: 'float64';
    complex128: 'complex128';
    complex64: 'complex128';
    int64: 'int64';
    int32: 'int64';
    int16: 'int64';
    int8: 'int64';
    uint64: 'uint64';
    uint32: 'uint32';
    uint16: 'uint32';
    uint8: 'uint32';
    bool: 'uint32';
  };
  uint16: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float32';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'int64';
    int32: 'int32';
    int16: 'int32';
    int8: 'int32';
    uint64: 'uint64';
    uint32: 'uint32';
    uint16: 'uint16';
    uint8: 'uint16';
    bool: 'uint16';
  };
  uint8: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float16';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'int64';
    int32: 'int32';
    int16: 'int16';
    int8: 'int16';
    uint64: 'uint64';
    uint32: 'uint32';
    uint16: 'uint16';
    uint8: 'uint8';
    bool: 'uint8';
  };
  bool: {
    float64: 'float64';
    float32: 'float32';
    float16: 'float16';
    complex128: 'complex128';
    complex64: 'complex64';
    int64: 'int64';
    int32: 'int32';
    int16: 'int16';
    int8: 'int8';
    uint64: 'uint64';
    uint32: 'uint32';
    uint16: 'uint16';
    uint8: 'uint8';
    bool: 'bool';
  };
}

/**
 * Result dtype of a binary op between arrays of dtype `A` and `B`.
 * @example type T = Promote<'int32', 'float32'>; // 'float64'
 */
export type Promote<A extends DType, B extends DType> = PromotionTable[A][B];

/** Result-dtype table for unary float math (sin, sqrt, exp, …). */
export interface MathResultTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'complex128';
  complex64: 'complex64';
  int64: 'float64';
  int32: 'float64';
  int16: 'float32';
  int8: 'float16';
  uint64: 'float64';
  uint32: 'float64';
  uint16: 'float32';
  uint8: 'float16';
  bool: 'float16';
}
export type MathResult<D extends DType> = MathResultTable[D];

/** Result-dtype table for reduction accumulation (sum, prod, cumsum, …). */
export interface ReductionAccumTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'complex128';
  complex64: 'complex64';
  int64: 'int64';
  int32: 'int64';
  int16: 'int64';
  int8: 'int64';
  uint64: 'uint64';
  uint32: 'uint64';
  uint16: 'uint64';
  uint8: 'uint64';
  bool: 'int64';
}
export type ReductionAccum<D extends DType> = ReductionAccumTable[D];

/** Result-dtype table for true division / mean / median. */
export interface TrueDivideTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'complex128';
  complex64: 'complex64';
  int64: 'float64';
  int32: 'float64';
  int16: 'float64';
  int8: 'float64';
  uint64: 'float64';
  uint32: 'float64';
  uint16: 'float64';
  uint8: 'float64';
  bool: 'float64';
}
export type TrueDivide<D extends DType> = TrueDivideTable[D];

/** Result-dtype table for std / var (spread statistics). */
export interface StdVarTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'float64';
  complex64: 'float32';
  int64: 'float64';
  int32: 'float64';
  int16: 'float64';
  int8: 'float64';
  uint64: 'float64';
  uint32: 'float64';
  uint16: 'float64';
  uint8: 'float64';
  bool: 'float64';
}
export type StdVar<D extends DType> = StdVarTable[D];

/** Result-dtype table for absolute value. */
export interface AbsTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'float64';
  complex64: 'float32';
  int64: 'int64';
  int32: 'int32';
  int16: 'int16';
  int8: 'int8';
  uint64: 'uint64';
  uint32: 'uint32';
  uint16: 'uint16';
  uint8: 'uint8';
  bool: 'bool';
}
export type Abs<D extends DType> = AbsTable[D];

/** Result-dtype table for angle (always real float). */
export interface AngleTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'float64';
  complex64: 'float32';
  int64: 'float64';
  int32: 'float64';
  int16: 'float32';
  int8: 'float16';
  uint64: 'float64';
  uint32: 'float64';
  uint16: 'float32';
  uint8: 'float16';
  bool: 'float64';
}
export type Angle<D extends DType> = AngleTable[D];

/** Result-dtype table for bool → int8 arithmetic promotion. */
export interface BoolArithTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'complex128';
  complex64: 'complex64';
  int64: 'int64';
  int32: 'int32';
  int16: 'int16';
  int8: 'int8';
  uint64: 'uint64';
  uint32: 'uint32';
  uint16: 'uint16';
  uint8: 'uint8';
  bool: 'int8';
}
export type BoolArith<D extends DType> = BoolArithTable[D];

/** Result-dtype table for complex-output FFT. */
export interface FftResultTable {
  float64: 'complex128';
  float32: 'complex64';
  float16: 'complex64';
  complex128: 'complex128';
  complex64: 'complex64';
  int64: 'complex128';
  int32: 'complex128';
  int16: 'complex128';
  int8: 'complex128';
  uint64: 'complex128';
  uint32: 'complex128';
  uint16: 'complex128';
  uint8: 'complex128';
  bool: 'complex128';
}
export type FftResult<D extends DType> = FftResultTable[D];

/** Result-dtype table for real-output FFT (hfft). */
export interface FftRealTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float32';
  complex128: 'float64';
  complex64: 'float32';
  int64: 'float64';
  int32: 'float64';
  int16: 'float64';
  int8: 'float64';
  uint64: 'float64';
  uint32: 'float64';
  uint16: 'float64';
  uint8: 'float64';
  bool: 'float64';
}
export type FftReal<D extends DType> = FftRealTable[D];

/** Result-dtype table for complex → real component (real, imag). */
export interface ComplexComponentTable {
  float64: 'float64';
  float32: 'float32';
  float16: 'float16';
  complex128: 'float64';
  complex64: 'float32';
  int64: 'int64';
  int32: 'int32';
  int16: 'int16';
  int8: 'int8';
  uint64: 'uint64';
  uint32: 'uint32';
  uint16: 'uint16';
  uint8: 'uint8';
  bool: 'bool';
}
export type ComplexComponent<D extends DType> = ComplexComponentTable[D];

/**
 * Result dtype for power / mod / floor_divide / remainder: promote, then apply
 * the bool → int8 arithmetic rule.
 */
export type Power<A extends DType, B extends DType> = BoolArith<Promote<A, B>>;

/** Result dtype for true division of two arrays: promote, then true-divide. */
export type Divide<A extends DType, B extends DType> = TrueDivide<Promote<A, B>>;

/** Result dtype for matmul / dot / inner / outer: NumPy promotion. */
export type MatMul<A extends DType, B extends DType> = Promote<A, B>;

/**
 * Result dtype for float-returning binary math (arctan2, hypot, copysign,
 * logaddexp, …): promote, then apply the unary float-math rule.
 */
export type MathBinary<A extends DType, B extends DType> = MathResult<Promote<A, B>>;

/**
 * Result dtype for float_power: promote, then bump to a minimum precision of
 * float64 (integers/float16/float32 → float64; complex → complex128). Modelled
 * as promotion against float64 (float64 preserves floats, upgrades ints/complex).
 */
export type FloatPower<A extends DType, B extends DType> = Promote<Promote<A, B>, 'float64'>;
