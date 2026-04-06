/**
 * WASM-accelerated element-wise arctan2.
 *
 * Binary: out[i] = atan2(a[i], b[i])  (same-shape contiguous arrays)
 * Returns null if WASM can't handle this case.
 */

import {
  arctan2_f64,
  arctan2_f32,
  arctan2_i64_f64,
  arctan2_u64_f64,
  arctan2_i32_f64,
  arctan2_u32_f64,
  arctan2_i16_f64,
  arctan2_u16_f64,
  arctan2_i8_f64,
  arctan2_u8_f64,
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

// Integer-to-f64 binary kernels: both inputs same integer type, output f64
const intBinaryKernels: Partial<Record<DType, BinaryFn>> = {
  int64: arctan2_i64_f64,
  uint64: arctan2_u64_f64,
  int32: arctan2_i32_f64,
  uint32: arctan2_u32_f64,
  int16: arctan2_i16_f64,
  uint16: arctan2_u16_f64,
  int8: arctan2_i8_f64,
  uint8: arctan2_u8_f64,
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

  // Integer path: Zig kernel reads native int type, converts to f64 internally.
  // Both inputs must be the same promoted dtype for the integer kernel.
  const intKernel = intBinaryKernels[dtype];
  const inBpe = bpeMap[dtype];
  if (!intKernel || !inBpe) return null;

  const outRegion = wasmMalloc(size * 8);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);
  const bPtr = resolveInputPtr(b.data, b.isWasmBacked, b.wasmPtr, b.offset, size, inBpe);

  intKernel(aPtr, bPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
