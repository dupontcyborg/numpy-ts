/**
 * WASM-accelerated element-wise square root.
 *
 * Unary: out[i] = sqrt(a[i])
 * Returns null if WASM can't handle this case.
 * Float types output same type; integer types use type-appropriate output:
 *   i8/u8 → f32 (then downcast to f16 if available)
 *   i16/u16 → f32
 *   i32/u32/i64/u64 → f64
 */

import {
  sqrt_f64,
  sqrt_f32,
  sqrt_i64,
  sqrt_i32,
  sqrt_i16_f32,
  sqrt_i8_f32,
} from './bins/sqrt.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr, f32ToF16InPlace } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, hasFloat16, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

// Float kernels: input & output same type
const floatKernels: Partial<Record<DType, UnaryFn>> = {
  float64: sqrt_f64,
  float32: sqrt_f32,
};

// Large int → f64 output (i32/u32/i64/u64 need f64 precision)
const largeIntKernels: Partial<Record<DType, UnaryFn>> = {
  int64: sqrt_i64,
  uint64: sqrt_i64,
  int32: sqrt_i32,
  uint32: sqrt_i32,
};

// Small int → f32 output (i8/u8/i16/u16 → f32, then optionally downcast to f16)
const smallIntKernels: Partial<Record<DType, UnaryFn>> = {
  int16: sqrt_i16_f32,
  uint16: sqrt_i16_f32,
  int8: sqrt_i8_f32,
  uint8: sqrt_i8_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const inputCtorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

/**
 * WASM-accelerated element-wise square root.
 * Returns null if WASM can't handle (complex, non-contiguous, too small).
 */
export function wasmSqrt(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (isComplexDType(dtype)) return null;

  // Float path: input & output same type
  const floatKernel = floatKernels[dtype];
  if (floatKernel) {
    const Ctor = inputCtorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

    floatKernel(aPtr, outRegion.ptr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
    );
  }

  // Small int path: i8/u8/i16/u16 → f32 output, optionally downcast to f16
  const smallKernel = smallIntKernels[dtype];
  if (smallKernel) {
    const InputCtor = inputCtorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 4; // f32

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    smallKernel(aPtr, outRegion.ptr, size);

    // i8/u8 → downcast f32 to f16 (matching NumPy's bool/int8/uint8 → float16)
    if (hasFloat16 && (dtype === 'int8' || dtype === 'uint8' || dtype === 'bool')) {
      f32ToF16InPlace(outRegion, size);
      return ArrayStorage.fromWasmRegion(
        Array.from(a.shape),
        'float16',
        outRegion,
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
      Float32Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
  }

  // Large int path: i32/u32/i64/u64 → f64 output
  const largeKernel = largeIntKernels[dtype];
  if (largeKernel) {
    const InputCtor = inputCtorMap[dtype]!;
    const inputBpe = (InputCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * 8;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inputBpe);

    largeKernel(aPtr, outRegion.ptr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      'float64',
      outRegion,
      size,
      Float64Array as unknown as new (
        buffer: ArrayBuffer,
        byteOffset: number,
        length: number
      ) => TypedArray
    );
  }

  return null;
}
