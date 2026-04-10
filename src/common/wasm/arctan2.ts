/**
 * WASM-accelerated element-wise arctan2.
 *
 * Binary: out[i] = atan2(a[i], b[i])  (same-shape contiguous arrays)
 * Returns null if WASM can't handle this case.
 * Integer types use type-appropriate output:
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 */

import {
  arctan2_f64,
  arctan2_f32,
  arctan2_i64_f64,
  arctan2_u64_f64,
  arctan2_i32_f64,
  arctan2_u32_f64,
  arctan2_i16_f32,
  arctan2_u16_f32,
  arctan2_i8_f32,
  arctan2_u8_f32,
} from './bins/arctan2.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  f16InputToScratchF32,
  f32OutputToF16Region,
} from './runtime';
import { ArrayStorage } from '../storage';
import {
  effectiveDType,
  promoteDTypes,
  isComplexDType,
  hasFloat16,
  type DType,
  type TypedArray,
} from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: arctan2_f64,
  float32: arctan2_f32,
  float16: arctan2_f32,
};

// Large int → f64 output (i32/u32/i64/u64 need f64 precision)
const largeIntKernels: Partial<Record<DType, BinaryFn>> = {
  int64: arctan2_i64_f64,
  uint64: arctan2_u64_f64,
  int32: arctan2_i32_f64,
  uint32: arctan2_u32_f64,
};

// Small int → f32 output (i8/u8/i16/u16 → f32, then optionally downcast to f16)
const smallIntKernels: Partial<Record<DType, BinaryFn>> = {
  int16: arctan2_i16_f32,
  uint16: arctan2_u16_f32,
  int8: arctan2_i8_f32,
  uint8: arctan2_u8_f32,
};

const bpeMap: Partial<Record<DType, number>> = {
  int64: 8,
  uint64: 8,
  int32: 4,
  uint32: 4,
  int16: 2,
  uint16: 2,
  int8: 1,
  uint8: 1,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  float16: Float32Array,
};

/**
 * WASM-accelerated element-wise arctan2 of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmArctan2(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  // WASM kernels expect same-dtype inputs — bail on mixed dtypes
  if (a.dtype !== b.dtype) return null;

  // WASM kernel does not broadcast — sizes must match
  if (a.size !== b.size) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = effectiveDType(promoteDTypes(a.dtype, b.dtype));
  if (isComplexDType(dtype)) return null;

  const floatKernel = binaryKernels[dtype];

  // Float path (f64, f32)
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

    const outRegion = wasmMalloc(size * bpe);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();

    let aPtr: number;
    let bPtr: number;
    if (dtype === 'float16') {
      aPtr = f16InputToScratchF32(a, size);
      bPtr = f16InputToScratchF32(b, size);
    } else {
      aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
      bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, bpe);
    }

    floatKernel(aPtr, bPtr, outRegion.ptr, size);

    if (dtype === 'float16') {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        dtype,
        f16Region,
        size,
        Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      );
    }

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  const inBpe = bpeMap[dtype];
  if (!inBpe) return null;

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntKernels[dtype];
  if (smallKernel) {
    const outBytes = size * 4; // f32
    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;
    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);
    const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inBpe);

    smallKernel(aPtr, bPtr, outRegion.ptr, size);

    // i8/u8 → downcast f32 to f16 (matching NumPy's bool/int8/uint8 → float16)
    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8' || dtype === 'bool')) {
      const f16Region = f32OutputToF16Region(outRegion, size);
      outRegion.release();
      if (!f16Region) return null;
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'float16',
        f16Region,
        size,
        Float16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
      );
    }

    // i16/u16 → f32 output
    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float32',
      outRegion,
      size,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  // Large int path: i32/u32/i64/u64 → f64 output
  const largeKernel = largeIntKernels[dtype];
  if (!largeKernel) return null;

  const outRegion = wasmMalloc(size * 8);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inBpe);

  largeKernel(aPtr, bPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
