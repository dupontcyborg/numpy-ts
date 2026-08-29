/**
 * WASM-accelerated element-wise comparisons: eq, ne, lt, le, gt, ge.
 * Two same-dtype, same-shape, contiguous arrays in; a bool array out.
 * Returns null when WASM can't handle it, so callers fall back to JS.
 */

import { type DType, effectiveDType, isComplexDType } from '../dtype';
import { ArrayStorage } from '../storage';
import {
  eq_f32,
  eq_f64,
  eq_i8,
  eq_i16,
  eq_i32,
  eq_i64,
  eq_u8,
  eq_u16,
  eq_u32,
  eq_u64,
  ge_f32,
  ge_f64,
  ge_i8,
  ge_i16,
  ge_i32,
  ge_i64,
  ge_u8,
  ge_u16,
  ge_u32,
  ge_u64,
  gt_f32,
  gt_f64,
  gt_i8,
  gt_i16,
  gt_i32,
  gt_i64,
  gt_u8,
  gt_u16,
  gt_u32,
  gt_u64,
  le_f32,
  le_f64,
  le_i8,
  le_i16,
  le_i32,
  le_i64,
  le_u8,
  le_u16,
  le_u32,
  le_u64,
  lt_f32,
  lt_f64,
  lt_i8,
  lt_i16,
  lt_i32,
  lt_i64,
  lt_u8,
  lt_u16,
  lt_u32,
  lt_u64,
  ne_f32,
  ne_f64,
  ne_i8,
  ne_i16,
  ne_i32,
  ne_i64,
  ne_u8,
  ne_u16,
  ne_u32,
  ne_u64,
} from './bins/compare.wasm';
import { wasmConfig } from './config';
import { resetScratchAllocator, resolveInputPtr, wasmMalloc } from './runtime';

const BASE_THRESHOLD = 32;

type CmpFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;

const bpeMap: Partial<Record<DType, number>> = {
  float64: 8,
  float32: 4,
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
  bool: 1,
};

/** bool inputs share the uint8 kernels — same one-byte layout. */
const table: Record<string, Partial<Record<DType, CmpFn>>> = {
  eq: {
    float64: eq_f64,
    float32: eq_f32,
    int64: eq_i64,
    uint64: eq_u64,
    int32: eq_i32,
    uint32: eq_u32,
    int16: eq_i16,
    uint16: eq_u16,
    int8: eq_i8,
    uint8: eq_u8,
    bool: eq_u8,
  },
  ne: {
    float64: ne_f64,
    float32: ne_f32,
    int64: ne_i64,
    uint64: ne_u64,
    int32: ne_i32,
    uint32: ne_u32,
    int16: ne_i16,
    uint16: ne_u16,
    int8: ne_i8,
    uint8: ne_u8,
    bool: ne_u8,
  },
  lt: {
    float64: lt_f64,
    float32: lt_f32,
    int64: lt_i64,
    uint64: lt_u64,
    int32: lt_i32,
    uint32: lt_u32,
    int16: lt_i16,
    uint16: lt_u16,
    int8: lt_i8,
    uint8: lt_u8,
    bool: lt_u8,
  },
  le: {
    float64: le_f64,
    float32: le_f32,
    int64: le_i64,
    uint64: le_u64,
    int32: le_i32,
    uint32: le_u32,
    int16: le_i16,
    uint16: le_u16,
    int8: le_i8,
    uint8: le_u8,
    bool: le_u8,
  },
  gt: {
    float64: gt_f64,
    float32: gt_f32,
    int64: gt_i64,
    uint64: gt_u64,
    int32: gt_i32,
    uint32: gt_u32,
    int16: gt_i16,
    uint16: gt_u16,
    int8: gt_i8,
    uint8: gt_u8,
    bool: gt_u8,
  },
  ge: {
    float64: ge_f64,
    float32: ge_f32,
    int64: ge_i64,
    uint64: ge_u64,
    int32: ge_i32,
    uint32: ge_u32,
    int16: ge_i16,
    uint16: ge_u16,
    int8: ge_i8,
    uint8: ge_u8,
    bool: ge_u8,
  },
};

/**
 * Element-wise comparison of two arrays, returning a bool array. float16 inputs always
 * take the JS per-element loop; when the engine lacks Float16Array, effectiveDType has
 * already turned the dtype into float32, so the plain f32 kernel path handles it instead.
 */
export function wasmCompare(kind: string, a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  if (a.dtype !== b.dtype) return null;
  if (isComplexDType(a.dtype)) return null;
  if (a.size !== b.size) return null;
  if (a.shape.length !== b.shape.length) return null;
  for (let i = 0; i < a.shape.length; i++) if (a.shape[i] !== b.shape[i]) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // float16 falls back to the JS loop: widening both inputs to f32 scratch first
  // would be exact, but the extra passes cost more than the wider SIMD lanes save.
  if (a.dtype === 'float16') return null;

  const dtype = effectiveDType(a.dtype);
  const kernel = table[kind]?.[dtype];
  const bpe = bpeMap[dtype];
  if (!kernel || !bpe) return null;

  const outRegion = wasmMalloc(size);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);

  kernel(aPtr, bPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'bool',
    outRegion,
    size,
    Uint8Array as unknown as new (
      buf: ArrayBuffer,
      off: number,
      len: number,
    ) => Uint8Array,
  );
}
