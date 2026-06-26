/**
 * AUTO-GENERATED — DO NOT EDIT BY HAND.
 *
 * Compile-time mirror of the runtime `promoteDTypes` function (src/common/dtype.ts),
 * which is the single source of truth. Regenerate with:
 *
 *   npx tsx scripts/generate-dtype-promotion.ts
 *
 * This module is types-only: it contributes zero bytes to the runtime bundle.
 */

import type { DType } from './dtype';

/**
 * Full dtype promotion matrix as a type. `PromotionTable[A][B]` is the dtype
 * that NumPy promotes `A` and `B` to.
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
 * Result dtype of a binary operation between arrays of dtype `A` and `B`,
 * following NumPy's type-promotion rules.
 *
 * @example
 * type T = Promote<'int32', 'float32'>; // 'float64'
 */
export type Promote<A extends DType, B extends DType> = PromotionTable[A][B];
